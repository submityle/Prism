//! Deterministic multi-surface, multi-view lighting resolve reference.
//!
//! This module is a CPU contract for a future frame-graph implementation. It
//! owns the data ordering, validation, BRDF reference math, energy-safe lobe
//! accumulation, and per-view temporal history rules. GPU code can consume the
//! same inputs and compare its output against this implementation without
//! inheriting a second policy for view or surface identity.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::f32::consts::PI;
use core::fmt;

/// Three-channel linear scene radiance or irradiance.
pub type Rgb = [f32; 3];

const EPSILON: f32 = 1.0e-6;
const MIN_ROUGHNESS: f32 = 0.045;
const MAX_SURFACES_PER_BATCH: usize = 1_048_576;
/// Largest representable resolve value that remains safe for an fp16 target.
pub const MAX_RESOLVE_RADIANCE: f32 = 65_504.0;

/// Stable identity of a surface in a view-local output.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SurfaceId(pub u32);

/// Stable identity of a camera/view resolve domain.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ViewId(pub u32);

/// Version tuple used to invalidate one view's temporal resolve history.
///
/// Any field change invalidates the history for that view. Keeping the tuple
/// in this contract makes camera, scene, lighting, and output-layout changes
/// explicit instead of relying on a single frame counter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResolveEpoch {
    pub geometry: u64,
    pub material: u64,
    pub lighting: u64,
    pub camera: u64,
    pub layout: u64,
}

/// Pixel extent of one view-local resolve target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewExtent {
    pub width: u32,
    pub height: u32,
}

impl ViewExtent {
    /// Creates a non-empty view extent.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    fn validate(self) -> Result<(), ResolveError> {
        if self.width == 0 || self.height == 0 {
            return Err(ResolveError::InvalidExtent);
        }
        Ok(())
    }
}

/// One surface and its lighting inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceInput {
    pub id: SurfaceId,
    pub pixel: [u32; 2],
    pub normal: Rgb,
    pub view_direction: Rgb,
    pub base_color: Rgb,
    pub metallic: f32,
    pub roughness: f32,
    pub diffuse_irradiance: Rgb,
    pub specular_radiance: Rgb,
    pub light_direction: Rgb,
    pub emission: Rgb,
    pub visibility: f32,
}

impl SurfaceInput {
    /// Validates a surface without changing the caller's values.
    pub fn validate(self) -> Result<(), ResolveError> {
        validate_rgb("normal", self.normal)?;
        validate_rgb("view_direction", self.view_direction)?;
        validate_rgb("base_color", self.base_color)?;
        validate_rgb("diffuse_irradiance", self.diffuse_irradiance)?;
        validate_rgb("specular_radiance", self.specular_radiance)?;
        validate_rgb("light_direction", self.light_direction)?;
        validate_rgb("emission", self.emission)?;
        validate_non_negative_rgb("base_color", self.base_color)?;
        validate_non_negative_rgb("diffuse_irradiance", self.diffuse_irradiance)?;
        validate_non_negative_rgb("specular_radiance", self.specular_radiance)?;
        validate_non_negative_rgb("emission", self.emission)?;
        validate_direction("normal", self.normal)?;
        validate_direction("view_direction", self.view_direction)?;
        validate_direction("light_direction", self.light_direction)?;
        validate_unit_interval("metallic", self.metallic)?;
        validate_unit_interval("visibility", self.visibility)?;
        if !self.roughness.is_finite() || !(0.0..=1.0).contains(&self.roughness) {
            return Err(ResolveError::InvalidRange { field: "roughness" });
        }
        Ok(())
    }
}

/// A contiguous group of surfaces written for one view.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceBatch {
    pub view_id: ViewId,
    pub surfaces: Vec<SurfaceInput>,
}

impl SurfaceBatch {
    /// Creates a batch and validates its hard capacity bound.
    pub fn new(view_id: ViewId, surfaces: Vec<SurfaceInput>) -> Result<Self, ResolveError> {
        if surfaces.len() > MAX_SURFACES_PER_BATCH {
            return Err(ResolveError::CapacityExceeded {
                field: "surfaces",
                limit: MAX_SURFACES_PER_BATCH,
            });
        }
        Ok(Self { view_id, surfaces })
    }
}

/// All batches belonging to one view for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewResolveInput {
    pub id: ViewId,
    pub extent: ViewExtent,
    pub epoch: ResolveEpoch,
    pub batches: Vec<SurfaceBatch>,
}

/// A complete deterministic resolve request.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolveInput {
    pub frame_index: u64,
    pub views: Vec<ViewResolveInput>,
}

/// Per-surface BRDF result before temporal view accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrdfEvaluation {
    pub diffuse: Rgb,
    pub specular: Rgb,
    pub emission: Rgb,
    pub total: Rgb,
    pub diffuse_energy_weight: f32,
    pub specular_energy_weight: f32,
}

/// Surface-local output retained for a view resolve consumer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedSurface {
    pub id: SurfaceId,
    pub radiance: Rgb,
    pub brdf: BrdfEvaluation,
}

/// Whether temporal history contributed to a view's output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryStatus {
    Fresh,
    Reused {
        previous_frame: u64,
        sample_count: u32,
    },
    Invalidated {
        reason: HistoryInvalidation,
    },
}

/// Explicit reason for rejecting a view's previous history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryInvalidation {
    EpochChanged,
    FrameDiscontinuity,
    SampleCountOverflow,
}

/// Resolve output local to one view.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewResolveOutput {
    pub id: ViewId,
    pub extent: ViewExtent,
    pub epoch: ResolveEpoch,
    pub surfaces: Vec<ResolvedSurface>,
    pub current_radiance: Rgb,
    pub accumulated_radiance: Rgb,
    pub sample_count: u32,
    pub history: HistoryStatus,
}

/// Complete multi-view resolve output. Views and surfaces are ID-sorted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolveOutput {
    pub frame_index: u64,
    pub views: Vec<ViewResolveOutput>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewHistory {
    epoch: ResolveEpoch,
    last_frame: u64,
    accumulated: Rgb,
    sample_count: u32,
}

/// Persistent history store, partitioned by [`ViewId`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolveHistory {
    views: BTreeMap<ViewId, ViewHistory>,
}

impl ResolveHistory {
    /// Creates an empty history store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes all stored view history.
    pub fn clear(&mut self) {
        self.views.clear();
    }

    /// Invalidates one view without affecting any other view's history.
    pub fn invalidate_view(&mut self, view_id: ViewId) -> bool {
        self.views.remove(&view_id).is_some()
    }

    /// Returns the epoch currently stored for one view, if any.
    #[must_use]
    pub fn epoch(&self, view_id: ViewId) -> Option<ResolveEpoch> {
        self.views.get(&view_id).map(|history| history.epoch)
    }

    /// Number of view histories currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Resolves all views and updates each view's history atomically in input
    /// order after validation and deterministic ID sorting.
    pub fn resolve(&mut self, input: ResolveInput) -> Result<ResolveOutput, ResolveError> {
        validate_input(&input)?;

        let mut views = input.views;
        views.sort_by_key(|view| view.id);

        let mut output_views = Vec::with_capacity(views.len());
        let mut next_history = self.views.clone();
        for view in views {
            let mut surfaces = flatten_surfaces(&view)?;
            surfaces.sort_by_key(|surface| surface.id);

            let mut resolved = Vec::with_capacity(surfaces.len());
            let mut current = EnergyAccumulator::default();
            for surface in surfaces {
                let brdf = evaluate_brdf(surface)?;
                current.add(brdf.total)?;
                resolved.push(ResolvedSurface {
                    id: surface.id,
                    radiance: brdf.total,
                    brdf,
                });
            }

            let current_radiance = current.finish();
            let (accumulated_radiance, sample_count, history_status) = update_view_history(
                &mut next_history,
                view.id,
                view.epoch,
                input.frame_index,
                current_radiance,
            );
            output_views.push(ViewResolveOutput {
                id: view.id,
                extent: view.extent,
                epoch: view.epoch,
                surfaces: resolved,
                current_radiance,
                accumulated_radiance,
                sample_count,
                history: history_status,
            });
        }

        self.views = next_history;
        Ok(ResolveOutput {
            frame_index: input.frame_index,
            views: output_views,
        })
    }
}

fn update_view_history(
    histories: &mut BTreeMap<ViewId, ViewHistory>,
    view_id: ViewId,
    epoch: ResolveEpoch,
    frame: u64,
    current: Rgb,
) -> (Rgb, u32, HistoryStatus) {
    let previous = histories.get(&view_id).copied();
    let Some(previous) = previous else {
        histories.insert(
            view_id,
            ViewHistory {
                epoch,
                last_frame: frame,
                accumulated: current,
                sample_count: 1,
            },
        );
        return (current, 1, HistoryStatus::Fresh);
    };

    let invalidation = if previous.epoch != epoch {
        Some(HistoryInvalidation::EpochChanged)
    } else if previous.last_frame.checked_add(1) != Some(frame) {
        Some(HistoryInvalidation::FrameDiscontinuity)
    } else {
        None
    };

    if let Some(reason) = invalidation {
        histories.insert(
            view_id,
            ViewHistory {
                epoch,
                last_frame: frame,
                accumulated: current,
                sample_count: 1,
            },
        );
        return (current, 1, HistoryStatus::Invalidated { reason });
    }

    let Some(next_count) = previous.sample_count.checked_add(1) else {
        histories.insert(
            view_id,
            ViewHistory {
                epoch,
                last_frame: frame,
                accumulated: current,
                sample_count: 1,
            },
        );
        return (
            current,
            1,
            HistoryStatus::Invalidated {
                reason: HistoryInvalidation::SampleCountOverflow,
            },
        );
    };

    let old_weight = (next_count - 1) as f32 / next_count as f32;
    let new_weight = 1.0 / next_count as f32;
    let accumulated = add_rgb(
        scale_rgb(previous.accumulated, old_weight),
        scale_rgb(current, new_weight),
    );
    histories.insert(
        view_id,
        ViewHistory {
            epoch,
            last_frame: frame,
            accumulated,
            sample_count: next_count,
        },
    );
    (
        accumulated,
        next_count,
        HistoryStatus::Reused {
            previous_frame: previous.last_frame,
            sample_count: next_count,
        },
    )
}

fn flatten_surfaces(view: &ViewResolveInput) -> Result<Vec<SurfaceInput>, ResolveError> {
    let mut surfaces = Vec::new();
    for batch in &view.batches {
        if batch.view_id != view.id {
            return Err(ResolveError::BatchViewMismatch {
                batch_view: batch.view_id,
                view: view.id,
            });
        }
        surfaces
            .try_reserve(batch.surfaces.len())
            .map_err(|_| ResolveError::AllocationFailed)?;
        surfaces.extend_from_slice(&batch.surfaces);
    }
    Ok(surfaces)
}

fn validate_input(input: &ResolveInput) -> Result<(), ResolveError> {
    let mut view_ids = BTreeSet::new();
    for view in &input.views {
        view.extent.validate()?;
        if !view_ids.insert(view.id) {
            return Err(ResolveError::DuplicateView { id: view.id });
        }
        let mut surface_ids = BTreeSet::new();
        for batch in &view.batches {
            if batch.view_id != view.id {
                return Err(ResolveError::BatchViewMismatch {
                    batch_view: batch.view_id,
                    view: view.id,
                });
            }
            if batch.surfaces.len() > MAX_SURFACES_PER_BATCH {
                return Err(ResolveError::CapacityExceeded {
                    field: "surfaces",
                    limit: MAX_SURFACES_PER_BATCH,
                });
            }
            for surface in &batch.surfaces {
                surface.validate()?;
                if surface.pixel[0] >= view.extent.width || surface.pixel[1] >= view.extent.height {
                    return Err(ResolveError::PixelOutOfBounds {
                        id: surface.id,
                        pixel: surface.pixel,
                        extent: view.extent,
                    });
                }
                if !surface_ids.insert(surface.id) {
                    return Err(ResolveError::DuplicateSurface { id: surface.id });
                }
            }
        }
    }
    Ok(())
}

/// Evaluates the reference Lambert + GGX response for one surface.
pub fn evaluate_brdf(surface: SurfaceInput) -> Result<BrdfEvaluation, ResolveError> {
    surface.validate()?;
    let n = normalize(surface.normal);
    let v = normalize(surface.view_direction);
    let l = normalize(surface.light_direction);
    let no_v = dot(n, v).max(0.0);
    let no_l = dot(n, l).max(0.0);
    let f0 = lerp_rgb([0.04, 0.04, 0.04], surface.base_color, surface.metallic);
    let max_f0 = max_component(f0);
    let diffuse_energy_weight = (1.0 - max_f0).max(0.0) * (1.0 - surface.metallic);
    let diffuse = scale_rgb(
        mul_rgb(surface.diffuse_irradiance, surface.base_color),
        diffuse_energy_weight / PI,
    );

    let h_sum = add_rgb(v, l);
    let specular = if no_v <= EPSILON || no_l <= EPSILON || length_squared(h_sum) <= EPSILON {
        [0.0; 3]
    } else {
        let h = normalize(h_sum);
        let no_h = dot(n, h).max(0.0);
        let vo_h = dot(v, h).max(0.0);
        let alpha = surface.roughness.max(MIN_ROUGHNESS);
        let alpha_squared = alpha * alpha;
        let d_denominator = no_h * no_h * (alpha_squared - 1.0) + 1.0;
        let d = alpha_squared / (PI * d_denominator * d_denominator).max(EPSILON);
        let g = smith_ggx(no_v, alpha) * smith_ggx(no_l, alpha);
        let fresnel = schlick_rgb(f0, vo_h);
        let scale = surface.visibility * no_l * d * g / (4.0 * no_v * no_l).max(EPSILON);
        scale_rgb(mul_rgb(surface.specular_radiance, fresnel), scale)
    };
    let specular_energy_weight = max_component(schlick_rgb(f0, 1.0)).clamp(0.0, 1.0);
    let emission = scale_rgb(surface.emission, surface.visibility);
    let raw_total = add_rgb(add_rgb(diffuse, specular), emission);
    let raw_peak = max_component(raw_total);
    let energy_scale = if raw_peak > MAX_RESOLVE_RADIANCE {
        MAX_RESOLVE_RADIANCE / raw_peak
    } else {
        1.0
    };
    let diffuse = scale_rgb(diffuse, energy_scale);
    let specular = scale_rgb(specular, energy_scale);
    let emission = scale_rgb(emission, energy_scale);
    let total = add_rgb(add_rgb(diffuse, specular), emission);
    if !rgb_is_finite_non_negative(total) {
        return Err(ResolveError::NonFinite {
            field: "brdf.total",
        });
    }
    Ok(BrdfEvaluation {
        diffuse,
        specular,
        emission,
        total,
        diffuse_energy_weight,
        specular_energy_weight,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct EnergyAccumulator {
    sum: Rgb,
    compensation: Rgb,
}

impl EnergyAccumulator {
    fn add(&mut self, value: Rgb) -> Result<(), ResolveError> {
        if !rgb_is_finite_non_negative(value) {
            return Err(ResolveError::NonFinite {
                field: "accumulated_radiance",
            });
        }
        for index in 0..3 {
            let corrected = value[index] - self.compensation[index];
            let next = (self.sum[index] + corrected).min(MAX_RESOLVE_RADIANCE);
            self.compensation[index] = (next - self.sum[index]) - corrected;
            self.sum[index] = next;
        }
        if !rgb_is_finite_non_negative(self.sum) {
            return Err(ResolveError::NonFinite {
                field: "accumulated_radiance",
            });
        }
        Ok(())
    }

    fn finish(self) -> Rgb {
        self.sum
    }
}

/// Input or contract validation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResolveError {
    NonFinite {
        field: &'static str,
    },
    Negative {
        field: &'static str,
    },
    InvalidRange {
        field: &'static str,
    },
    ZeroDirection {
        field: &'static str,
    },
    DuplicateView {
        id: ViewId,
    },
    DuplicateSurface {
        id: SurfaceId,
    },
    BatchViewMismatch {
        batch_view: ViewId,
        view: ViewId,
    },
    CapacityExceeded {
        field: &'static str,
        limit: usize,
    },
    InvalidExtent,
    PixelOutOfBounds {
        id: SurfaceId,
        pixel: [u32; 2],
        extent: ViewExtent,
    },
    AllocationFailed,
}

impl fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { field } => write!(formatter, "{field} contains a non-finite value"),
            Self::Negative { field } => write!(formatter, "{field} contains a negative value"),
            Self::InvalidRange { field } => write!(formatter, "{field} is outside its valid range"),
            Self::ZeroDirection { field } => write!(formatter, "{field} has zero length"),
            Self::DuplicateView { id } => write!(formatter, "duplicate view id {}", id.0),
            Self::DuplicateSurface { id } => write!(formatter, "duplicate surface id {}", id.0),
            Self::BatchViewMismatch { batch_view, view } => {
                write!(
                    formatter,
                    "batch view {} does not match view {}",
                    batch_view.0, view.0
                )
            }
            Self::CapacityExceeded { field, limit } => {
                write!(formatter, "{field} exceeds capacity {limit}")
            }
            Self::InvalidExtent => write!(formatter, "view extent must be non-zero"),
            Self::PixelOutOfBounds { id, pixel, extent } => write!(
                formatter,
                "surface {} pixel [{}, {}] is outside {}x{} view extent",
                id.0, pixel[0], pixel[1], extent.width, extent.height
            ),
            Self::AllocationFailed => {
                write!(formatter, "allocation failed while building resolve output")
            }
        }
    }
}

fn validate_rgb(field: &'static str, value: Rgb) -> Result<(), ResolveError> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(ResolveError::NonFinite { field });
    }
    Ok(())
}

fn validate_non_negative_rgb(field: &'static str, value: Rgb) -> Result<(), ResolveError> {
    if value.iter().any(|component| *component < 0.0) {
        return Err(ResolveError::Negative { field });
    }
    Ok(())
}

fn validate_direction(field: &'static str, value: Rgb) -> Result<(), ResolveError> {
    if length_squared(value) <= EPSILON {
        return Err(ResolveError::ZeroDirection { field });
    }
    Ok(())
}

fn validate_unit_interval(field: &'static str, value: f32) -> Result<(), ResolveError> {
    if !value.is_finite() {
        return Err(ResolveError::NonFinite { field });
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(ResolveError::InvalidRange { field });
    }
    Ok(())
}

fn rgb_is_finite_non_negative(value: Rgb) -> bool {
    value
        .iter()
        .all(|component| component.is_finite() && *component >= 0.0)
}

fn normalize(value: Rgb) -> Rgb {
    let inverse_length = 1.0 / length_squared(value).sqrt();
    scale_rgb(value, inverse_length)
}

fn length_squared(value: Rgb) -> f32 {
    dot(value, value)
}

fn dot(a: Rgb, b: Rgb) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add_rgb(a: Rgb, b: Rgb) -> Rgb {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul_rgb(a: Rgb, b: Rgb) -> Rgb {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn scale_rgb(value: Rgb, scale: f32) -> Rgb {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn lerp_rgb(a: Rgb, b: Rgb, factor: f32) -> Rgb {
    add_rgb(scale_rgb(a, 1.0 - factor), scale_rgb(b, factor))
}

fn max_component(value: Rgb) -> f32 {
    value[0].max(value[1]).max(value[2])
}

fn smith_ggx(no_x: f32, alpha: f32) -> f32 {
    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    no_x / (no_x * (1.0 - k) + k).max(EPSILON)
}

fn schlick_rgb(f0: Rgb, vo_h: f32) -> Rgb {
    let factor = (1.0 - vo_h.clamp(0.0, 1.0)).powi(5);
    add_rgb(
        f0,
        scale_rgb([1.0 - f0[0], 1.0 - f0[1], 1.0 - f0[2]], factor),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(id: u32, light: f32) -> SurfaceInput {
        SurfaceInput {
            id: SurfaceId(id),
            pixel: [id, 0],
            normal: [0.0, 0.0, 1.0],
            view_direction: [0.0, 0.0, 1.0],
            base_color: [0.8, 0.6, 0.4],
            metallic: 0.0,
            roughness: 0.5,
            diffuse_irradiance: [light; 3],
            specular_radiance: [light; 3],
            light_direction: [0.0, 0.0, 1.0],
            emission: [0.0; 3],
            visibility: 1.0,
        }
    }

    fn view(id: u32, epoch: ResolveEpoch, surfaces: Vec<SurfaceInput>) -> ViewResolveInput {
        ViewResolveInput {
            id: ViewId(id),
            extent: ViewExtent::new(32, 1),
            epoch,
            batches: vec![SurfaceBatch::new(ViewId(id), surfaces).expect("valid batch")],
        }
    }

    #[test]
    fn brdf_is_finite_non_negative_and_energy_weighted() {
        let result = evaluate_brdf(surface(1, 2.0)).expect("valid BRDF input");
        assert!(rgb_is_finite_non_negative(result.total));
        assert!(result.diffuse_energy_weight <= 1.0);
        assert!(result.specular_energy_weight <= 1.0);
        assert!(result.total[0] >= result.diffuse[0]);
    }

    #[test]
    fn lambert_reference_matches_unit_incident_radiance() {
        let mut input = surface(1, 1.0);
        input.base_color = [1.0; 3];
        input.specular_radiance = [0.0; 3];
        let result = evaluate_brdf(input).expect("valid Lambert input");
        let expected = 1.0 / PI * 0.96;
        assert!((result.diffuse[0] - expected).abs() <= 1.0e-6);
        assert!(result
            .diffuse
            .iter()
            .all(|value| (*value - expected).abs() <= 1.0e-6));
    }

    #[test]
    fn roughness_zero_is_clamped_to_a_stable_ggx_lobe() {
        let mut input = surface(1, 1.0);
        input.roughness = 0.0;
        let result = evaluate_brdf(input).expect("roughness zero is a valid boundary");
        assert!(rgb_is_finite_non_negative(result.specular));
    }

    #[test]
    fn multiple_views_and_batches_are_sorted_deterministically() {
        let epoch = ResolveEpoch::default();
        let mut history = ResolveHistory::new();
        let output = history
            .resolve(ResolveInput {
                frame_index: 1,
                views: vec![
                    view(9, epoch, vec![surface(4, 1.0)]),
                    ViewResolveInput {
                        id: ViewId(2),
                        extent: ViewExtent::new(32, 1),
                        epoch,
                        batches: vec![
                            SurfaceBatch::new(ViewId(2), vec![surface(8, 1.0)]).unwrap(),
                            SurfaceBatch::new(ViewId(2), vec![surface(3, 1.0)]).unwrap(),
                        ],
                    },
                ],
            })
            .expect("valid multi-view input");
        assert_eq!(
            output.views.iter().map(|view| view.id).collect::<Vec<_>>(),
            vec![ViewId(2), ViewId(9)]
        );
        assert_eq!(
            output.views[0]
                .surfaces
                .iter()
                .map(|surface| surface.id)
                .collect::<Vec<_>>(),
            vec![SurfaceId(3), SurfaceId(8)]
        );

        let mut replay = ResolveHistory::new();
        let replay_output = replay
            .resolve(ResolveInput {
                frame_index: 1,
                views: vec![
                    ViewResolveInput {
                        id: ViewId(2),
                        extent: ViewExtent::new(32, 1),
                        epoch,
                        batches: vec![
                            SurfaceBatch::new(ViewId(2), vec![surface(3, 1.0)]).unwrap(),
                            SurfaceBatch::new(ViewId(2), vec![surface(8, 1.0)]).unwrap(),
                        ],
                    },
                    view(9, epoch, vec![surface(4, 1.0)]),
                ],
            })
            .unwrap();
        assert_eq!(output, replay_output);
    }

    #[test]
    fn history_is_reused_per_view_and_epoch_change_invalidates_only_that_view() {
        let mut history = ResolveHistory::new();
        let epoch = ResolveEpoch::default();
        history
            .resolve(ResolveInput {
                frame_index: 10,
                views: vec![
                    view(1, epoch, vec![surface(1, 1.0)]),
                    view(2, epoch, vec![surface(2, 2.0)]),
                ],
            })
            .unwrap();
        let mut changed = epoch;
        changed.lighting = 1;
        let output = history
            .resolve(ResolveInput {
                frame_index: 11,
                views: vec![
                    view(1, epoch, vec![surface(1, 1.0)]),
                    view(2, changed, vec![surface(2, 2.0)]),
                ],
            })
            .unwrap();
        assert!(matches!(
            output.views[0].history,
            HistoryStatus::Reused { .. }
        ));
        assert_eq!(
            output.views[1].history,
            HistoryStatus::Invalidated {
                reason: HistoryInvalidation::EpochChanged
            }
        );
        assert_eq!(output.views[0].sample_count, 2);
        assert_eq!(output.views[1].sample_count, 1);
    }

    #[test]
    fn frame_gap_invalidates_history() {
        let mut history = ResolveHistory::new();
        let epoch = ResolveEpoch::default();
        history
            .resolve(ResolveInput {
                frame_index: 2,
                views: vec![view(1, epoch, vec![surface(1, 1.0)])],
            })
            .unwrap();
        let output = history
            .resolve(ResolveInput {
                frame_index: 4,
                views: vec![view(1, epoch, vec![surface(1, 1.0)])],
            })
            .unwrap();
        assert_eq!(
            output.views[0].history,
            HistoryStatus::Invalidated {
                reason: HistoryInvalidation::FrameDiscontinuity
            }
        );
    }

    #[test]
    fn duplicate_ids_and_invalid_values_are_rejected() {
        let epoch = ResolveEpoch::default();
        let mut duplicate = surface(1, 1.0);
        duplicate.id = SurfaceId(1);
        let error = ResolveHistory::new().resolve(ResolveInput {
            frame_index: 0,
            views: vec![view(1, epoch, vec![surface(1, 1.0), duplicate])],
        });
        assert_eq!(
            error,
            Err(ResolveError::DuplicateSurface { id: SurfaceId(1) })
        );

        let mut invalid = surface(2, 1.0);
        invalid.base_color[0] = -0.1;
        let error = ResolveHistory::new().resolve(ResolveInput {
            frame_index: 0,
            views: vec![view(1, epoch, vec![invalid])],
        });
        assert_eq!(
            error,
            Err(ResolveError::Negative {
                field: "base_color"
            })
        );
    }

    #[test]
    fn invalid_request_does_not_mutate_existing_history() {
        let mut history = ResolveHistory::new();
        let epoch = ResolveEpoch::default();
        history
            .resolve(ResolveInput {
                frame_index: 1,
                views: vec![view(1, epoch, vec![surface(1, 1.0)])],
            })
            .unwrap();
        let before = history.clone();
        let mut invalid = surface(1, 1.0);
        invalid.visibility = 2.0;
        assert!(history
            .resolve(ResolveInput {
                frame_index: 2,
                views: vec![view(1, epoch, vec![invalid])],
            })
            .is_err());
        assert_eq!(history, before);
    }

    #[test]
    fn view_extent_and_pixel_bounds_are_enforced() {
        let epoch = ResolveEpoch::default();
        let mut invalid = surface(1, 1.0);
        invalid.pixel = [4, 0];
        let error = ResolveHistory::new().resolve(ResolveInput {
            frame_index: 0,
            views: vec![ViewResolveInput {
                id: ViewId(1),
                extent: ViewExtent::new(4, 1),
                epoch,
                batches: vec![SurfaceBatch::new(ViewId(1), vec![invalid]).unwrap()],
            }],
        });
        assert_eq!(
            error,
            Err(ResolveError::PixelOutOfBounds {
                id: SurfaceId(1),
                pixel: [4, 0],
                extent: ViewExtent::new(4, 1),
            })
        );

        let error = ResolveHistory::new().resolve(ResolveInput {
            frame_index: 0,
            views: vec![ViewResolveInput {
                id: ViewId(1),
                extent: ViewExtent::new(0, 1),
                epoch,
                batches: Vec::new(),
            }],
        });
        assert_eq!(error, Err(ResolveError::InvalidExtent));
    }

    #[test]
    fn energy_safe_resolve_is_bounded_for_extreme_inputs() {
        let mut input = surface(1, 1.0e30);
        input.emission = [1.0e30; 3];
        let result = evaluate_brdf(input).expect("finite extreme input remains resolvable");
        assert!(result
            .total
            .iter()
            .all(|value| *value <= MAX_RESOLVE_RADIANCE));

        let epoch = ResolveEpoch::default();
        let mut history = ResolveHistory::new();
        let output = history
            .resolve(ResolveInput {
                frame_index: 0,
                views: vec![view(1, epoch, vec![input, surface(2, 1.0e30)])],
            })
            .unwrap();
        assert!(output.views[0]
            .current_radiance
            .iter()
            .all(|value| *value <= MAX_RESOLVE_RADIANCE));
    }
}
