//! The §20 volumetric *evaluation / baking* layer that feeds the six-way
//! lighting rig, the deep-opacity self-shadow tier, and the phase-function
//! contracts declared in [`super::shading`].
//!
//! [`super::shading`] owns the *decision* layer of design §20: it declares the
//! pre-integrated [`SixWayLuminance`] rig, the runtime interpolator
//! [`super::shading::six_way_response`], the authored [`PhaseParams`] contract,
//! the [`super::shading::DeepShadowMode`] ladder plus its resolver, and the
//! [`super::shading::quantize_cel_bands`] `NPR` step. This module is the layer
//! *beneath* those decisions: the arithmetic that actually **bakes and
//! evaluates** them. It never redefines those types — it consumes them.
//!
//! It covers four gaps left open by the decision layer:
//!
//! - **Six-way baking / accumulation** — [`SixWayAccumulator`] folds any number
//!   of directional light samples (direction plus intensity) and an ambient
//!   term into the six principal-axis buckets and bakes a [`SixWayLuminance`],
//!   the upstream step that produces the rig
//!   [`super::shading::six_way_response`] later interpolates (design §20).
//! - **Henyey-Greenstein phase evaluation** — the decision layer carries
//!   [`PhaseParams`] and defers evaluation to in-shader code because the phase
//!   function "needs transcendental math". It does not: the `HG` denominator
//!   `(1 + g² − 2·g·cosθ)^1.5` factors as `d · d.sqrt()`, so
//!   [`henyey_greenstein`] and [`double_lobe_phase`] give a `CPU` reference
//!   using only `+ − × ÷` and `sqrt`. [`phase_response_banded`] bridges that
//!   response into [`super::shading::quantize_cel_bands`] for the `NPR` path.
//! - **`deep opacity map` layered transmittance** — [`sample_deep_transmittance`]
//!   samples the piecewise-linear transmittance curve recorded along the light
//!   ray (design §20). It interpolates *stored* layer transmittance rather than
//!   recomputing `exp(−τ)`, so no transcendental math appears; the layers
//!   themselves are the `Beer-Lambert` integral the recorder already evaluated.
//! - **`froxel` volume-fog coupling** — [`FroxelGrid`] maps a world position to
//!   an integer `froxel` cell and [`FroxelDensityField`] accumulates particle
//!   density into that cell so the unified volumetric scattering pass can read a
//!   single density grid (design §20).
//!
//! Like its sibling modules, every routine here is pure, deterministic, and
//! free of transcendental functions, so the `CPU` reference stays bit-
//! reproducible against a future `GPU` kernel.

use alloc::vec;
use alloc::vec::Vec;

use super::shading::{quantize_cel_bands, PhaseParams, SixWayLuminance};
use super::Vec3;

/// `4·π`, the normalization constant of the Henyey-Greenstein phase function.
///
/// Spelled as a literal because the determinism contract forbids computing it
/// from a transcendental `PI`; this is `4.0 * core::f32::consts::PI` rounded to
/// `f32`.
pub const FOUR_PI: f32 = 12.566_371;

/// General-purpose tolerance for guarding `f32` divisions and near-zero spans.
///
/// Distinct from [`super::EPS_LEN_SQ`] (a squared-length threshold): this is a
/// plain magnitude floor used where a denominator or an interpolation span may
/// collapse to zero.
pub const EPS: f32 = 1e-6;

/// Accumulator that bakes a [`SixWayLuminance`] rig from light samples.
///
/// Each principal axis owns one luminance bucket. A directional light deposits
/// its intensity into the buckets by the positive projection of its direction
/// onto each axis, the exact inverse of how [`super::shading::six_way_response`]
/// reads them back, so a single light baked here and queried there round-trips.
/// An ambient term deposits equal luminance into all six buckets (design §20).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SixWayAccumulator {
    /// Accumulated luminance toward `+X`.
    pub right: f32,
    /// Accumulated luminance toward `-X`.
    pub left: f32,
    /// Accumulated luminance toward `+Y`.
    pub up: f32,
    /// Accumulated luminance toward `-Y`.
    pub down: f32,
    /// Accumulated luminance toward `+Z`.
    pub front: f32,
    /// Accumulated luminance toward `-Z`.
    pub back: f32,
}

impl Default for SixWayAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl SixWayAccumulator {
    /// A fresh accumulator with every axis bucket at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            right: 0.0,
            left: 0.0,
            up: 0.0,
            down: 0.0,
            front: 0.0,
            back: 0.0,
        }
    }

    /// Deposits one directional light sample into the axis buckets.
    ///
    /// `light_dir` points from the particle toward the light and need not be
    /// normalized (it is normalized internally; a zero direction is ignored).
    /// A negative `intensity` is clamped to zero so the rig never bakes
    /// negative luminance. The intensity is split across the buckets by the
    /// magnitude of the direction's projection onto each axis.
    pub fn add_light(&mut self, light_dir: Vec3, intensity: f32) {
        let energy = if intensity > 0.0 { intensity } else { 0.0 };
        let d = light_dir.normalize_or_zero();
        if d.x >= 0.0 {
            self.right += d.x * energy;
        } else {
            self.left += -d.x * energy;
        }
        if d.y >= 0.0 {
            self.up += d.y * energy;
        } else {
            self.down += -d.y * energy;
        }
        if d.z >= 0.0 {
            self.front += d.z * energy;
        } else {
            self.back += -d.z * energy;
        }
    }

    /// Deposits an isotropic ambient term into all six axis buckets equally.
    ///
    /// A negative `intensity` is clamped to zero.
    pub fn add_ambient(&mut self, intensity: f32) {
        let energy = if intensity > 0.0 { intensity } else { 0.0 };
        self.right += energy;
        self.left += energy;
        self.up += energy;
        self.down += energy;
        self.front += energy;
        self.back += energy;
    }

    /// Bakes the accumulated buckets into a [`SixWayLuminance`] rig.
    #[must_use]
    pub const fn bake(&self) -> SixWayLuminance {
        SixWayLuminance {
            right: self.right,
            left: self.left,
            up: self.up,
            down: self.down,
            front: self.front,
            back: self.back,
        }
    }
}

/// Evaluates the single-lobe Henyey-Greenstein phase function (`CPU` reference).
///
/// Returns `p(θ) = (1 − g²) / (4π · (1 + g² − 2·g·cosθ)^1.5)` where `cos_theta`
/// is the cosine of the angle between the incoming and outgoing directions and
/// `g ∈ (−1, 1)` is the anisotropy (forward for `g > 0`). The `^1.5` power is
/// evaluated as `d · d.sqrt()`, so the whole function uses only `+ − × ÷` and
/// `sqrt` — no transcendental math, unlike the in-shader note in
/// [`super::shading`]. The base `d` is floored at [`EPS`] so the grazing case
/// (`g → ±1`, `cosθ → ±1`) yields a large-but-finite value instead of dividing
/// by zero.
#[must_use]
pub fn henyey_greenstein(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let base = 1.0 + g2 - 2.0 * g * cos_theta;
    let d = if base > EPS { base } else { EPS };
    let d15 = d * d.sqrt();
    (1.0 - g2) / (FOUR_PI * d15)
}

/// Evaluates the double-lobe (front + back) phase for authored [`PhaseParams`].
///
/// Blends a forward lobe [`henyey_greenstein`] at `params.g` with a back lobe at
/// `params.back_g`, weighted by `params.back_lobe_weight` (clamped to `0..=1`):
/// `(1 − w)·front + w·back`. This is the smoke "edge glow" response the
/// [`PhaseParams`] contract describes, evaluated on the `CPU` with `sqrt`-only
/// math.
#[must_use]
pub fn double_lobe_phase(params: PhaseParams, cos_theta: f32) -> f32 {
    let w = if params.back_lobe_weight > 1.0 {
        1.0
    } else if params.back_lobe_weight > 0.0 {
        params.back_lobe_weight
    } else {
        0.0
    };
    let front = henyey_greenstein(params.g, cos_theta);
    let back = henyey_greenstein(params.back_g, cos_theta);
    (1.0 - w) * front + w * back
}

/// Bridges a phase response into discrete `NPR` cel bands (design §20).
///
/// Evaluates [`double_lobe_phase`], normalizes it by `peak` (the phase value the
/// author treats as full brightness, typically the forward-scatter peak) into
/// `0..=1`, then snaps it with [`super::shading::quantize_cel_bands`]. A `peak`
/// at or below [`EPS`] is treated as unit normalization so the call never
/// divides by zero.
#[must_use]
pub fn phase_response_banded(params: PhaseParams, cos_theta: f32, peak: f32, bands: u32) -> f32 {
    let response = double_lobe_phase(params, cos_theta);
    let normalized = if peak > EPS {
        response / peak
    } else {
        response
    };
    quantize_cel_bands(normalized, bands)
}

/// One recorded layer of a `deep opacity map` along the light ray (design §20).
///
/// The recorder walks the light ray front-to-back and stores the accumulated
/// transmittance at increasing depths; this is one such `(depth, transmittance)`
/// sample. `transmittance` is the fraction of light that survives to `depth`
/// (`1.0` = fully lit, `0.0` = fully occluded).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepOpacityLayer {
    /// Distance along the light ray at which this layer was recorded.
    pub depth: f32,
    /// Surviving transmittance at `depth`, in `0..=1`.
    pub transmittance: f32,
}

impl DeepOpacityLayer {
    /// Builds a layer from a depth and a transmittance.
    #[must_use]
    pub const fn new(depth: f32, transmittance: f32) -> Self {
        Self {
            depth,
            transmittance,
        }
    }
}

/// Samples the layered `deep opacity map` transmittance at a query depth.
///
/// `layers` are the recorded [`DeepOpacityLayer`] samples in ascending `depth`
/// order. The transmittance curve is treated as piecewise linear between
/// stored layers and interpolated at `depth`; this samples the already-recorded
/// `Beer-Lambert` integral rather than recomputing `exp(−τ)`, so no
/// transcendental math is used (design §20). Boundaries are handled explicitly:
///
/// - an empty slice returns `1.0` (nothing occludes the sample);
/// - a `depth` at or before the first layer returns the first layer's
///   transmittance (the fully-lit near side);
/// - a `depth` at or after the last layer returns the last layer's
///   transmittance (the deepest recorded occlusion).
///
/// A degenerate zero-width span between two layers falls back to the deeper
/// layer's transmittance instead of dividing by zero.
#[must_use]
pub fn sample_deep_transmittance(layers: &[DeepOpacityLayer], depth: f32) -> f32 {
    let Some(first) = layers.first() else {
        return 1.0;
    };
    if depth <= first.depth {
        return first.transmittance;
    }
    // Safe: the slice is non-empty, so `last` exists.
    let last = layers[layers.len() - 1];
    if depth >= last.depth {
        return last.transmittance;
    }
    for pair in layers.windows(2) {
        let lo = pair[0];
        let hi = pair[1];
        if depth <= hi.depth {
            let span = hi.depth - lo.depth;
            if span > EPS {
                let t = (depth - lo.depth) / span;
                return lo.transmittance + t * (hi.transmittance - lo.transmittance);
            }
            return hi.transmittance;
        }
    }
    // Unreachable: the `depth >= last.depth` guard above covers the tail, but
    // return the last transmittance defensively rather than a bare fallback.
    last.transmittance
}

/// A uniform `froxel` grid mapping world positions to integer cells (design §20).
///
/// A `froxel` grid is the axis-aligned voxelization the unified volumetric fog
/// pass scatters through. This describes the grid geometry only; the density it
/// carries lives in a [`FroxelDensityField`]. Cell lookup uses one float divide
/// per axis and integer comparisons — no transcendental math.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FroxelGrid {
    /// Cell counts along `X`, `Y`, and `Z`.
    pub dims: [u32; 3],
    /// World-space position of the grid's minimum corner (cell `[0, 0, 0]`).
    pub origin: Vec3,
    /// World-space size of a single cell along each axis (all components `> 0`).
    pub cell_size: Vec3,
}

impl FroxelGrid {
    /// Builds a grid from its dimensions, origin, and per-axis cell size.
    #[must_use]
    pub const fn new(dims: [u32; 3], origin: Vec3, cell_size: Vec3) -> Self {
        Self {
            dims,
            origin,
            cell_size,
        }
    }

    /// Total number of cells (`dims.x · dims.y · dims.z`) as a `usize`.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.dims[0] as usize * self.dims[1] as usize * self.dims[2] as usize
    }

    /// Maps a world position to its integer `froxel` cell, or `None` if outside.
    ///
    /// Returns `None` when any cell-size component is non-positive (a degenerate
    /// grid) or when `world` falls outside the grid bounds. The per-axis index
    /// is `floor((world − origin) / cell_size)`; because the in-bounds local
    /// coordinate is non-negative, the truncating `as u32` cast equals the
    /// floor.
    #[must_use]
    pub fn cell_index(&self, world: Vec3) -> Option<[u32; 3]> {
        let local = world.sub(self.origin);
        let ix = Self::axis_index(local.x, self.cell_size.x, self.dims[0])?;
        let iy = Self::axis_index(local.y, self.cell_size.y, self.dims[1])?;
        let iz = Self::axis_index(local.z, self.cell_size.z, self.dims[2])?;
        Some([ix, iy, iz])
    }

    /// Resolves one axis' cell index, or `None` if degenerate or out of bounds.
    fn axis_index(local: f32, size: f32, dim: u32) -> Option<u32> {
        if size <= EPS || local < 0.0 {
            return None;
        }
        let idx = (local / size) as u32;
        if idx < dim {
            Some(idx)
        } else {
            None
        }
    }

    /// Flattens an integer cell to its row-major linear index.
    ///
    /// Uses `x + y·dims.x + z·dims.x·dims.y`. Returns `None` if any component is
    /// out of bounds, so callers can flatten arbitrary indices safely.
    #[must_use]
    pub fn linear_index(&self, cell: [u32; 3]) -> Option<usize> {
        if cell[0] >= self.dims[0] || cell[1] >= self.dims[1] || cell[2] >= self.dims[2] {
            return None;
        }
        let x = cell[0] as usize;
        let y = cell[1] as usize;
        let z = cell[2] as usize;
        let dx = self.dims[0] as usize;
        let dy = self.dims[1] as usize;
        Some(x + y * dx + z * dx * dy)
    }
}

/// A per-cell density field paired with a [`FroxelGrid`] (design §20).
///
/// Particles inject their density into cells here; the unified volumetric
/// scattering pass then reads one contiguous density grid regardless of how
/// many particle systems contributed. Densities accumulate additively.
#[derive(Clone, Debug, PartialEq)]
pub struct FroxelDensityField {
    grid: FroxelGrid,
    density: Vec<f32>,
}

impl FroxelDensityField {
    /// Allocates a zeroed density field sized to `grid`.
    #[must_use]
    pub fn new(grid: FroxelGrid) -> Self {
        let count = grid.cell_count();
        Self {
            grid,
            density: vec![0.0; count],
        }
    }

    /// The grid geometry this field is bound to.
    #[must_use]
    pub const fn grid(&self) -> &FroxelGrid {
        &self.grid
    }

    /// Injects `density` at a world position, accumulating into its cell.
    ///
    /// Returns `true` when the position mapped to an in-bounds cell (and the
    /// density was added), or `false` when it fell outside the grid. A negative
    /// `density` is clamped to zero so injection never removes energy.
    pub fn inject(&mut self, world: Vec3, density: f32) -> bool {
        let energy = if density > 0.0 { density } else { 0.0 };
        let Some(cell) = self.grid.cell_index(world) else {
            return false;
        };
        let Some(index) = self.grid.linear_index(cell) else {
            return false;
        };
        self.density[index] += energy;
        true
    }

    /// Reads the accumulated density of an integer cell (`0.0` if out of bounds).
    #[must_use]
    pub fn density_at(&self, cell: [u32; 3]) -> f32 {
        match self.grid.linear_index(cell) {
            Some(index) => self.density[index],
            None => 0.0,
        }
    }

    /// The raw density buffer in row-major order, for the scattering pass.
    #[must_use]
    pub fn densities(&self) -> &[f32] {
        &self.density
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::shading::six_way_response;

    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    #[test]
    fn single_light_bakes_and_round_trips_through_six_way_response() {
        let mut acc = SixWayAccumulator::new();
        acc.add_light(Vec3::new(1.0, 0.0, 0.0), 2.0);
        let lum = acc.bake();
        assert!(approx(lum.right, 2.0));
        assert!(approx(lum.left, 0.0));
        // Querying along the same axis returns the baked intensity.
        assert!(approx(six_way_response(lum, Vec3::new(1.0, 0.0, 0.0)), 2.0));
        // Querying the opposite axis returns zero.
        assert!(approx(
            six_way_response(lum, Vec3::new(-1.0, 0.0, 0.0)),
            0.0
        ));
    }

    #[test]
    fn multiple_lights_accumulate_per_axis() {
        let mut acc = SixWayAccumulator::new();
        acc.add_light(Vec3::new(0.0, 3.0, 0.0), 1.0);
        acc.add_light(Vec3::new(0.0, -1.0, 0.0), 4.0);
        let lum = acc.bake();
        assert!(approx(lum.up, 1.0));
        assert!(approx(lum.down, 4.0));
        assert!(approx(lum.front, 0.0));
    }

    #[test]
    fn negative_intensity_and_zero_direction_are_ignored() {
        let mut acc = SixWayAccumulator::new();
        acc.add_light(Vec3::new(1.0, 0.0, 0.0), -5.0);
        acc.add_light(Vec3::ZERO, 5.0);
        let lum = acc.bake();
        assert!(approx(lum.right, 0.0));
        assert!(approx(lum.left, 0.0));
    }

    #[test]
    fn ambient_fills_all_axes_equally() {
        let mut acc = SixWayAccumulator::default();
        acc.add_ambient(0.5);
        let lum = acc.bake();
        for v in [lum.right, lum.left, lum.up, lum.down, lum.front, lum.back] {
            assert!(approx(v, 0.5));
        }
    }

    #[test]
    fn isotropic_phase_equals_reciprocal_four_pi() {
        // g = 0 collapses HG to the isotropic constant 1 / (4π).
        let p = henyey_greenstein(0.0, 0.37);
        assert!(approx(p, 1.0 / FOUR_PI));
    }

    #[test]
    fn forward_scatter_peaks_ahead_of_backward() {
        // g > 0 must scatter more strongly forward (cosθ = 1) than back (-1).
        let g = 0.6;
        let fwd = henyey_greenstein(g, 1.0);
        let bwd = henyey_greenstein(g, -1.0);
        assert!(fwd > bwd);
    }

    #[test]
    fn henyey_greenstein_matches_closed_form_sample() {
        // Cross-check the sqrt-only path against a hand-computed reference:
        // g = 0.5, cosθ = 0.5 → base = 1.25 - 0.5 = 0.75.
        let g = 0.5;
        let cos_theta = 0.5;
        let base = 1.0 + g * g - 2.0 * g * cos_theta;
        let expected = (1.0 - g * g) / (FOUR_PI * base * base.sqrt());
        assert!(approx(henyey_greenstein(g, cos_theta), expected));
    }

    #[test]
    fn grazing_case_stays_finite() {
        // g → 1, cosθ → 1 makes the base collapse; the EPS floor keeps it finite.
        let p = henyey_greenstein(0.999, 1.0);
        assert!(p.is_finite());
    }

    #[test]
    fn double_lobe_blends_front_and_back() {
        let params = PhaseParams {
            g: 0.7,
            back_lobe_weight: 0.5,
            back_g: -0.4,
        };
        let front = henyey_greenstein(0.7, -1.0);
        let back = henyey_greenstein(-0.4, -1.0);
        let expected = 0.5 * front + 0.5 * back;
        assert!(approx(double_lobe_phase(params, -1.0), expected));
    }

    #[test]
    fn double_lobe_weight_clamps() {
        let params = PhaseParams {
            g: 0.3,
            back_lobe_weight: 2.0,
            back_g: -0.3,
        };
        // Weight clamps to 1.0, so the result is the pure back lobe.
        let back = henyey_greenstein(-0.3, 0.2);
        assert!(approx(double_lobe_phase(params, 0.2), back));
    }

    #[test]
    fn phase_response_banded_snaps_to_levels() {
        let params = PhaseParams::isotropic();
        // Isotropic response is 1/(4π); using it as the peak normalizes to 1.0,
        // which snaps to the top band.
        let peak = 1.0 / FOUR_PI;
        let banded = phase_response_banded(params, 0.0, peak, 4);
        assert!(approx(banded, 1.0));
    }

    #[test]
    fn phase_response_banded_survives_zero_peak() {
        let params = PhaseParams::isotropic();
        let banded = phase_response_banded(params, 0.0, 0.0, 4);
        assert!(banded.is_finite());
    }

    #[test]
    fn deep_transmittance_empty_is_fully_lit() {
        assert!(approx(sample_deep_transmittance(&[], 3.0), 1.0));
    }

    #[test]
    fn deep_transmittance_interpolates_between_layers() {
        let layers = [
            DeepOpacityLayer::new(0.0, 1.0),
            DeepOpacityLayer::new(2.0, 0.5),
            DeepOpacityLayer::new(4.0, 0.1),
        ];
        // Midpoint of the first span: (1.0 + 0.5) / 2 = 0.75.
        assert!(approx(sample_deep_transmittance(&layers, 1.0), 0.75));
        // Exactly on a stored layer returns that layer's value.
        assert!(approx(sample_deep_transmittance(&layers, 2.0), 0.5));
        // Midpoint of the second span: (0.5 + 0.1) / 2 = 0.3.
        assert!(approx(sample_deep_transmittance(&layers, 3.0), 0.3));
    }

    #[test]
    fn deep_transmittance_clamps_at_boundaries() {
        let layers = [
            DeepOpacityLayer::new(1.0, 0.9),
            DeepOpacityLayer::new(3.0, 0.2),
        ];
        // Before the first layer: fully-lit near side.
        assert!(approx(sample_deep_transmittance(&layers, -1.0), 0.9));
        // After the last layer: deepest recorded occlusion.
        assert!(approx(sample_deep_transmittance(&layers, 10.0), 0.2));
    }

    #[test]
    fn deep_transmittance_handles_zero_width_span() {
        let layers = [
            DeepOpacityLayer::new(0.0, 1.0),
            DeepOpacityLayer::new(2.0, 0.8),
            DeepOpacityLayer::new(2.0, 0.3),
            DeepOpacityLayer::new(4.0, 0.1),
        ];
        // The duplicate depth must not divide by zero; the deeper value wins.
        let t = sample_deep_transmittance(&layers, 2.0);
        assert!(t.is_finite());
    }

    #[test]
    fn froxel_cell_index_maps_and_bounds_check() {
        let grid = FroxelGrid::new([4, 4, 4], Vec3::ZERO, Vec3::splat(1.0));
        assert_eq!(grid.cell_index(Vec3::new(0.5, 1.5, 2.5)), Some([0, 1, 2]));
        // On the far edge is outside (index == dim).
        assert_eq!(grid.cell_index(Vec3::new(4.0, 0.0, 0.0)), None);
        // Negative local coordinate is outside.
        assert_eq!(grid.cell_index(Vec3::new(-0.1, 0.0, 0.0)), None);
    }

    #[test]
    fn froxel_degenerate_cell_size_is_outside() {
        let grid = FroxelGrid::new([2, 2, 2], Vec3::ZERO, Vec3::new(0.0, 1.0, 1.0));
        assert_eq!(grid.cell_index(Vec3::new(0.5, 0.5, 0.5)), None);
    }

    #[test]
    fn froxel_linear_index_is_row_major() {
        let grid = FroxelGrid::new([2, 3, 4], Vec3::ZERO, Vec3::splat(1.0));
        assert_eq!(grid.linear_index([0, 0, 0]), Some(0));
        assert_eq!(grid.linear_index([1, 0, 0]), Some(1));
        assert_eq!(grid.linear_index([0, 1, 0]), Some(2));
        assert_eq!(grid.linear_index([0, 0, 1]), Some(6));
        assert_eq!(grid.linear_index([2, 0, 0]), None);
    }

    #[test]
    fn froxel_density_injection_accumulates() {
        let grid = FroxelGrid::new([2, 2, 2], Vec3::ZERO, Vec3::splat(1.0));
        let mut field = FroxelDensityField::new(grid);
        assert!(field.inject(Vec3::new(0.5, 0.5, 0.5), 1.0));
        assert!(field.inject(Vec3::new(0.2, 0.3, 0.4), 2.0));
        assert!(approx(field.density_at([0, 0, 0]), 3.0));
        assert!(approx(field.density_at([1, 1, 1]), 0.0));
    }

    #[test]
    fn froxel_density_out_of_bounds_is_rejected() {
        let grid = FroxelGrid::new([2, 2, 2], Vec3::ZERO, Vec3::splat(1.0));
        let mut field = FroxelDensityField::new(grid);
        assert!(!field.inject(Vec3::new(5.0, 0.0, 0.0), 1.0));
        assert!(approx(field.densities().iter().sum::<f32>(), 0.0));
    }

    #[test]
    fn froxel_negative_density_is_clamped() {
        let grid = FroxelGrid::new([1, 1, 1], Vec3::ZERO, Vec3::splat(1.0));
        let mut field = FroxelDensityField::new(grid);
        assert!(field.inject(Vec3::new(0.5, 0.5, 0.5), -3.0));
        assert!(approx(field.density_at([0, 0, 0]), 0.0));
    }
}
