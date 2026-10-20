//! Deterministic viewport, picking, and gizmo interaction contracts.
//!
//! The crate also includes an executable wgpu ID-buffer backend. It owns only
//! target/readback resources and leaves object-ID drawing to the renderer.

#![forbid(unsafe_code)]

extern crate alloc;

mod gpu_picking;

pub use gpu_picking::{
    EncodedPickId, GpuPickingBackend, GpuPickingConfig, GpuPickingError, GpuPickingFrame,
    PendingPickReadback, ID_BUFFER_BYTES_PER_TEXEL, ID_BUFFER_FORMAT, READBACK_ROW_ALIGNMENT,
};

use alloc::{collections::BTreeMap, string::String};
use core::fmt;

/// Monotonic identity for an editor viewport.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ViewportId(u64);

impl ViewportId {
    /// Creates an identity from a persisted value.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Camera projection used by a viewport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    /// Perspective projection with a vertical field of view in millidegrees.
    Perspective {
        /// Vertical field of view in millidegrees.
        fov_millidegrees: u32,
    },
    /// Orthographic projection with a scale in milli-units.
    Orthographic {
        /// Visible height scale in milli-units.
        scale_milliunits: u32,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Self::Perspective {
            fov_millidegrees: 60_000,
        }
    }
}

/// Integer camera position and orientation representation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CameraState {
    /// Position in world milli-units.
    pub position_milliunits: [i64; 3],
    /// Yaw, pitch, and roll in millidegrees.
    pub rotation_millidegrees: [i32; 3],
    /// Projection parameters.
    pub projection: Projection,
}

/// Stable render-side options exposed by the viewport model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewportSettings {
    /// Whether the grid overlay is requested.
    pub grid: bool,
    /// Whether the selection outline is requested.
    pub selection_outline: bool,
    /// Integer render scale in percent.
    pub render_scale_percent: u16,
}

impl Default for ViewportSettings {
    fn default() -> Self {
        Self {
            grid: true,
            selection_outline: true,
            render_scale_percent: 100,
        }
    }
}

/// One materialized viewport and its editor-visible state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Viewport {
    id: ViewportId,
    title: String,
    camera: CameraState,
    settings: ViewportSettings,
    generation: u64,
}

impl Viewport {
    /// Returns the stable viewport identity.
    #[must_use]
    pub const fn id(&self) -> ViewportId {
        self.id
    }

    /// Returns the user-facing title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Returns the camera state.
    #[must_use]
    pub const fn camera(&self) -> CameraState {
        self.camera
    }

    /// Returns the render settings.
    #[must_use]
    pub const fn settings(&self) -> ViewportSettings {
        self.settings
    }

    /// Returns the scene generation used to reject old async results.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Failure from viewport lifecycle or state updates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ViewportError {
    /// The requested viewport does not exist.
    Unknown(ViewportId),
    /// The monotonic identity space is exhausted.
    IdExhausted,
    /// A title cannot be empty or whitespace-only.
    EmptyTitle,
    /// A render scale is outside the supported range.
    InvalidRenderScale(u16),
}

impl fmt::Display for ViewportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(formatter, "unknown viewport {}", id.raw()),
            Self::IdExhausted => formatter.write_str("viewport ID space exhausted"),
            Self::EmptyTitle => formatter.write_str("viewport title must not be empty"),
            Self::InvalidRenderScale(value) => {
                write!(formatter, "invalid viewport render scale: {value}%")
            }
        }
    }
}

impl core::error::Error for ViewportError {}

/// Deterministic registry of open viewports.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ViewportRegistry {
    viewports: BTreeMap<ViewportId, Viewport>,
    next_id: u64,
}

impl ViewportRegistry {
    /// Creates an empty registry whose first identity is one.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            viewports: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Opens a viewport with default camera and overlay settings.
    pub fn open(&mut self, title: impl Into<String>) -> Result<Viewport, ViewportError> {
        let title = title.into();
        if title.trim().is_empty() {
            return Err(ViewportError::EmptyTitle);
        }
        let id = ViewportId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ViewportError::IdExhausted)?;
        let viewport = Viewport {
            id,
            title,
            camera: CameraState::default(),
            settings: ViewportSettings::default(),
            generation: 0,
        };
        self.viewports.insert(id, viewport.clone());
        Ok(viewport)
    }

    /// Closes a viewport and returns its final state.
    pub fn close(&mut self, id: ViewportId) -> Result<Viewport, ViewportError> {
        self.viewports.remove(&id).ok_or(ViewportError::Unknown(id))
    }

    /// Returns a viewport by identity.
    #[must_use]
    pub fn get(&self, id: ViewportId) -> Option<&Viewport> {
        self.viewports.get(&id)
    }

    /// Returns all viewports in stable identity order.
    pub fn iter(&self) -> impl Iterator<Item = &Viewport> {
        self.viewports.values()
    }

    /// Updates the camera and advances the scene generation.
    pub fn set_camera(&mut self, id: ViewportId, camera: CameraState) -> Result<(), ViewportError> {
        let viewport = self
            .viewports
            .get_mut(&id)
            .ok_or(ViewportError::Unknown(id))?;
        viewport.camera = camera;
        viewport.generation = viewport.generation.saturating_add(1);
        Ok(())
    }

    /// Updates render settings and advances the scene generation.
    pub fn set_settings(
        &mut self,
        id: ViewportId,
        settings: ViewportSettings,
    ) -> Result<(), ViewportError> {
        if !(25..=200).contains(&settings.render_scale_percent) {
            return Err(ViewportError::InvalidRenderScale(
                settings.render_scale_percent,
            ));
        }
        let viewport = self
            .viewports
            .get_mut(&id)
            .ok_or(ViewportError::Unknown(id))?;
        viewport.settings = settings;
        viewport.generation = viewport.generation.saturating_add(1);
        Ok(())
    }
}

/// Pixel coordinate supplied to an asynchronous pick request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelPoint {
    /// Horizontal pixel coordinate.
    pub x: u32,
    /// Vertical pixel coordinate.
    pub y: u32,
}

/// Correlation identity for a pending pick request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PickRequestId(u64);

impl PickRequestId {
    /// Returns the numeric request identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Renderer-neutral request for an ID-buffer or ray-pick operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PickRequest {
    /// Request correlation identity.
    pub id: PickRequestId,
    /// Viewport that produced the request.
    pub viewport: ViewportId,
    /// Scene generation captured by the request.
    pub generation: u64,
    /// Pixel to resolve.
    pub pixel: PixelPoint,
}

/// Result returned by a picking adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PickResult {
    /// Correlation identity copied from the request.
    pub id: PickRequestId,
    /// Viewport that produced the result.
    pub viewport: ViewportId,
    /// Scene generation used by the renderer.
    pub generation: u64,
    /// Stable editor selection identity, if anything was hit.
    pub entity: Option<u128>,
}

/// Why a pick result was not accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickRejectReason {
    /// The request was not submitted by this coordinator.
    UnknownRequest,
    /// The result refers to a viewport generation that is no longer current.
    StaleGeneration,
    /// The result belongs to another viewport.
    WrongViewport,
}

/// Accepted or rejected outcome of resolving a pick result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickOutcome {
    /// The result is current and may update editor selection.
    Accepted {
        /// Stable editor selection identity, if anything was hit.
        entity: Option<u128>,
    },
    /// The result is safe to discard.
    Rejected(PickRejectReason),
}

/// Bounded correlation state for asynchronous picks.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PickCoordinator {
    next_id: u64,
    pending: BTreeMap<PickRequestId, PickRequest>,
}

impl PickCoordinator {
    /// Creates an empty coordinator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            pending: BTreeMap::new(),
        }
    }

    /// Submits a request and returns its stable correlation record.
    pub fn submit(
        &mut self,
        viewport: &Viewport,
        pixel: PixelPoint,
    ) -> Result<PickRequest, ViewportError> {
        let id = PickRequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ViewportError::IdExhausted)?;
        let request = PickRequest {
            id,
            viewport: viewport.id,
            generation: viewport.generation,
            pixel,
        };
        self.pending.insert(id, request);
        Ok(request)
    }

    /// Resolves a renderer result against the current viewport state.
    pub fn resolve(&mut self, current: &Viewport, result: PickResult) -> PickOutcome {
        let Some(request) = self.pending.remove(&result.id) else {
            return PickOutcome::Rejected(PickRejectReason::UnknownRequest);
        };
        if request.viewport != current.id || result.viewport != current.id {
            return PickOutcome::Rejected(PickRejectReason::WrongViewport);
        }
        if request.generation != current.generation || result.generation != current.generation {
            return PickOutcome::Rejected(PickRejectReason::StaleGeneration);
        }
        PickOutcome::Accepted {
            entity: result.entity,
        }
    }

    /// Returns the number of requests awaiting a renderer result.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

/// Transform operation controlled by a gizmo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GizmoMode {
    /// Translate the selected objects.
    Translate,
    /// Rotate the selected objects.
    Rotate,
    /// Scale the selected objects.
    Scale,
}

/// Deterministic snapping options for a gizmo interaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapSettings {
    /// Translation increment in milli-units; zero disables translation snap.
    pub translation_milliunits: u32,
    /// Rotation increment in millidegrees; zero disables rotation snap.
    pub rotation_millidegrees: u32,
    /// Scale increment in milli-units; zero disables scale snap.
    pub scale_milliunits: u32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            translation_milliunits: 100,
            rotation_millidegrees: 15_000,
            scale_milliunits: 100,
        }
    }
}

/// Cumulative transform delta emitted by a gizmo drag.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransformDelta {
    /// Translation delta in world milli-units.
    pub translation_milliunits: [i64; 3],
    /// Rotation delta in millidegrees.
    pub rotation_millidegrees: [i32; 3],
    /// Multiplicative scale delta in milli-units, where 1000 is identity.
    pub scale_milliunits: [i64; 3],
}

impl TransformDelta {
    fn snap(self, mode: GizmoMode, settings: SnapSettings) -> Self {
        let mut delta = self;
        match mode {
            GizmoMode::Translate => {
                for value in &mut delta.translation_milliunits {
                    *value = snap_i64(*value, settings.translation_milliunits);
                }
            }
            GizmoMode::Rotate => {
                for value in &mut delta.rotation_millidegrees {
                    *value = snap_i32(*value, settings.rotation_millidegrees);
                }
            }
            GizmoMode::Scale => {
                for value in &mut delta.scale_milliunits {
                    *value = snap_i64(*value, settings.scale_milliunits);
                }
            }
        }
        delta
    }
}

fn snap_i64(value: i64, step: u32) -> i64 {
    let step = i64::from(step);
    if step == 0 {
        return value;
    }
    let quotient = value.div_euclid(step);
    let remainder = value.rem_euclid(step);
    if remainder.saturating_mul(2) >= step {
        (quotient + 1).saturating_mul(step)
    } else {
        quotient.saturating_mul(step)
    }
}

fn snap_i32(value: i32, step: u32) -> i32 {
    snap_i64(i64::from(value), step).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// A command-oriented gizmo interaction. Intermediate updates are coalesced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GizmoSession {
    mode: GizmoMode,
    snap: SnapSettings,
    accumulated: TransformDelta,
    active: bool,
}

impl GizmoSession {
    /// Starts a new drag.
    #[must_use]
    pub fn begin(mode: GizmoMode, snap: SnapSettings) -> Self {
        Self {
            mode,
            snap,
            accumulated: TransformDelta {
                scale_milliunits: [1000; 3],
                ..TransformDelta::default()
            },
            active: true,
        }
    }

    /// Adds an input delta and returns the snapped cumulative delta.
    pub fn update(&mut self, delta: TransformDelta) -> Option<TransformDelta> {
        if !self.active {
            return None;
        }
        self.accumulated.translation_milliunits = add_i64(
            self.accumulated.translation_milliunits,
            delta.translation_milliunits,
        );
        self.accumulated.rotation_millidegrees = add_i32(
            self.accumulated.rotation_millidegrees,
            delta.rotation_millidegrees,
        );
        self.accumulated.scale_milliunits = add_i64(
            self.accumulated.scale_milliunits,
            delta
                .scale_milliunits
                .map(|value| value.saturating_sub(1000)),
        );
        Some(self.accumulated.snap(self.mode, self.snap))
    }

    /// Finishes the drag and returns the one undoable cumulative delta.
    pub fn finish(&mut self) -> Option<TransformDelta> {
        if !self.active {
            return None;
        }
        self.active = false;
        Some(self.accumulated.snap(self.mode, self.snap))
    }

    /// Cancels the drag without producing an edit.
    pub fn cancel(&mut self) -> bool {
        if !self.active {
            return false;
        }
        self.active = false;
        true
    }

    /// Returns whether the drag is still active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

fn add_i64(left: [i64; 3], right: [i64; 3]) -> [i64; 3] {
    [
        left[0].saturating_add(right[0]),
        left[1].saturating_add(right[1]),
        left[2].saturating_add(right[2]),
    ]
}

fn add_i32(left: [i32; 3], right: [i32; 3]) -> [i32; 3] {
    [
        left[0].saturating_add(right[0]),
        left[1].saturating_add(right[1]),
        left[2].saturating_add(right[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_ids_are_stable_and_generation_changes_are_monotonic() {
        let mut registry = ViewportRegistry::new();
        let first = registry.open("Scene").expect("first");
        let second = registry.open("Game").expect("second");
        assert_eq!(first.id().raw(), 1);
        assert_eq!(second.id().raw(), 2);
        registry
            .set_camera(first.id(), CameraState::default())
            .expect("camera");
        assert_eq!(registry.get(first.id()).expect("viewport").generation(), 1);
        registry.close(first.id()).expect("close");
        let reopened = registry.open("Scene").expect("reopen");
        assert_eq!(reopened.id().raw(), 3);
    }

    #[test]
    fn pick_results_reject_unknown_viewport_and_stale_generation() {
        let mut registry = ViewportRegistry::new();
        let viewport = registry.open("Scene").expect("viewport");
        let mut coordinator = PickCoordinator::new();
        let request = coordinator
            .submit(&viewport, PixelPoint { x: 10, y: 20 })
            .expect("submit");
        registry
            .set_camera(viewport.id(), CameraState::default())
            .expect("camera");
        let current = registry.get(viewport.id()).expect("current");
        assert_eq!(
            coordinator.resolve(
                current,
                PickResult {
                    id: request.id,
                    viewport: current.id(),
                    generation: request.generation,
                    entity: Some(7),
                },
            ),
            PickOutcome::Rejected(PickRejectReason::StaleGeneration)
        );
        assert_eq!(coordinator.pending_len(), 0);
    }

    #[test]
    fn gizmo_updates_are_snapped_and_finish_once() {
        let mut session = GizmoSession::begin(
            GizmoMode::Translate,
            SnapSettings {
                translation_milliunits: 100,
                ..SnapSettings::default()
            },
        );
        let update = session
            .update(TransformDelta {
                translation_milliunits: [149, -151, 0],
                scale_milliunits: [1000; 3],
                ..TransformDelta::default()
            })
            .expect("update");
        assert_eq!(update.translation_milliunits, [100, -200, 0]);
        assert_eq!(
            session.finish().expect("finish").translation_milliunits,
            [100, -200, 0]
        );
        assert!(session.finish().is_none());
    }
}
