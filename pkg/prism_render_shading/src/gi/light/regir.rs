//! ReGIR world-space light-grid reservoirs — CPU golden reference.
//!
//! Reservoir-based Grid Importance Resampling (NVIDIA ReGIR, Boksansky et al.
//! 2021) precomputes, for every cell of a world-space grid, a small reservoir
//! of lights selected by weighted reservoir sampling proportional to each
//! light's contribution at the *cell centre*.  At shading time a surface point
//! looks up its cell and samples one already-resampled light, turning a
//! thousands-of-lights many-light problem into a tiny per-cell lookup.
//!
//! This module is the backend-neutral reference for that grid:
//!
//! * [`GridConfig`] describes the uniform grid — an `origin`, a per-axis
//!   `cell_size`, and integer `dims` — and maps world positions to cells with a
//!   quantisation that round-trips through [`GridConfig::cell_center`].
//! * [`GridLight`] is the lightweight per-light proxy the resampler needs: a
//!   world `position` and a scalar `power`.
//! * [`cell_target`] is the streaming RIS target `p_hat` — `power / dist^2`
//!   evaluated at a cell centre, with the same minimum-distance clamp used by
//!   the light tree.
//! * [`fill_cell_reservoir`] folds a list of candidate lights into a
//!   [`Reservoir`] (reusing the screen-probe [`crate::gi::screen_probe::restir`]
//!   WRS container) using an explicit, caller-supplied sequence of uniforms, and
//!   finalises the unbiased contribution weight.
//! * [`query`] reads the selected light and its unbiased weight back out.
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; math via `bevy_math`.  No transcendental
//!   functions are needed here, so none are called.
//! * Candidate lights are proposed from a *uniform* source distribution over
//!   the supplied candidate list, so a candidate's resampling weight is
//!   `p_hat / source_pdf = p_hat * candidate_count`.  The constant factor
//!   cancels in selection and is undone by [`Reservoir::finalize_weight`],
//!   yielding the standard RIS estimator `W = (1/p_hat) * mean(w_i)`.
//! * The grid is defensively clamped: `cell_size` is forced strictly positive
//!   per axis and `dims` to at least `1` per axis, so a degenerate
//!   configuration collapses to a single finite cell instead of dividing by
//!   zero or indexing out of range.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no global state, and no `unsafe`.

use bevy_math::{UVec3, Vec3};

use crate::gi::screen_probe::restir::Reservoir;

/// Smallest squared distance [`cell_target`] is allowed to divide by.
const MIN_DIST2_FLOOR: f32 = f32::MIN_POSITIVE;

/// A lightweight per-light proxy used to resample a grid cell.
///
/// ReGIR only needs each light's position and scalar power to evaluate the
/// resampling target at a cell centre; richer shading data is fetched later for
/// the single surviving light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLight {
    /// World-space position of the light (its representative point).
    pub position: Vec3,
    /// Scalar radiant power of the light (clamped non-negative when used).
    pub power: f32,
}

impl GridLight {
    /// Builds a grid light from a position and power.
    #[inline]
    pub fn new(position: Vec3, power: f32) -> Self {
        Self { position, power }
    }
}

/// Uniform world-space grid configuration for the ReGIR reservoir cache.
///
/// Cells tile the region starting at `origin` with per-axis `cell_size`, laid
/// out in `x`-fastest, `z`-slowest row-major order.  All accessors clamp the
/// stored `cell_size` and `dims` into their valid ranges, so a [`GridConfig`]
/// value is always safe to query even if constructed with degenerate fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridConfig {
    /// World-space position of the `(0, 0, 0)` cell's minimum corner.
    pub origin: Vec3,
    /// Per-axis edge length of a cell (forced strictly positive on use).
    pub cell_size: Vec3,
    /// Per-axis number of cells (forced to at least `1` on use).
    pub dims: UVec3,
}

impl GridConfig {
    /// Builds a grid configuration from its raw fields.
    #[inline]
    pub fn new(origin: Vec3, cell_size: Vec3, dims: UVec3) -> Self {
        Self {
            origin,
            cell_size,
            dims,
        }
    }

    /// The per-axis cell size, with each component forced strictly positive.
    #[inline]
    pub fn clamped_cell_size(&self) -> Vec3 {
        Vec3::new(
            positive_or_floor(self.cell_size.x),
            positive_or_floor(self.cell_size.y),
            positive_or_floor(self.cell_size.z),
        )
    }

    /// The per-axis cell count, with each component forced to at least `1`.
    #[inline]
    pub fn clamped_dims(&self) -> UVec3 {
        UVec3::new(
            self.dims.x.max(1),
            self.dims.y.max(1),
            self.dims.z.max(1),
        )
    }

    /// Total number of cells in the grid (always at least `1`).
    #[inline]
    pub fn cell_count(&self) -> usize {
        let d = self.clamped_dims();
        d.x as usize * d.y as usize * d.z as usize
    }

    /// Quantises a world position to its integer grid coordinate.
    ///
    /// Computes `floor((world - origin) / cell_size)` per axis and clamps the
    /// result into `[0, dims - 1]`, so a position anywhere (even outside the
    /// grid) maps to a valid in-range cell.
    #[inline]
    pub fn cell_coord(&self, world: Vec3) -> UVec3 {
        let size = self.clamped_cell_size();
        let dims = self.clamped_dims();
        let local = (world - self.origin) / size;
        UVec3::new(
            clamp_axis(local.x, dims.x),
            clamp_axis(local.y, dims.y),
            clamp_axis(local.z, dims.z),
        )
    }

    /// Flattens an in-range grid coordinate to a linear cell index.
    ///
    /// Uses `x`-fastest row-major order: `x + dims.x * (y + dims.y * z)`.  The
    /// coordinate is clamped into range first, so the returned index is always
    /// in `[0, cell_count)`.
    #[inline]
    pub fn flatten(&self, coord: UVec3) -> usize {
        let d = self.clamped_dims();
        let x = coord.x.min(d.x - 1) as usize;
        let y = coord.y.min(d.y - 1) as usize;
        let z = coord.z.min(d.z - 1) as usize;
        x + d.x as usize * (y + d.y as usize * z)
    }

    /// Maps a world position directly to its linear cell index.
    #[inline]
    pub fn cell_index(&self, world: Vec3) -> usize {
        self.flatten(self.cell_coord(world))
    }

    /// Returns the world-space centre of the cell at `coord`.
    ///
    /// This is the inverse of [`cell_coord`](Self::cell_coord) up to
    /// quantisation: feeding a cell centre back through `cell_coord` recovers
    /// the same coordinate.
    #[inline]
    pub fn cell_center(&self, coord: UVec3) -> Vec3 {
        let size = self.clamped_cell_size();
        let dims = self.clamped_dims();
        let cx = coord.x.min(dims.x - 1) as f32;
        let cy = coord.y.min(dims.y - 1) as f32;
        let cz = coord.z.min(dims.z - 1) as f32;
        self.origin + Vec3::new((cx + 0.5) * size.x, (cy + 0.5) * size.y, (cz + 0.5) * size.z)
    }
}

/// Streaming RIS target `p_hat` for a light evaluated at a cell centre.
///
/// Returns `power / dist^2`, where `dist^2` is the squared distance from the
/// light to `cell_center`, floored by `max(min_dist, 0)^2` (and an absolute
/// tiny floor) so a light sitting on the cell centre yields a finite value.
/// The result is always finite and non-negative; a zero-power light yields `0`.
#[inline]
pub fn cell_target(light: &GridLight, cell_center: Vec3, min_dist: f32) -> f32 {
    let power = light.power.max(0.0);
    if power <= 0.0 {
        return 0.0;
    }
    let md = min_dist.max(0.0);
    let dist2 = (light.position - cell_center)
        .length_squared()
        .max(md * md)
        .max(MIN_DIST2_FLOOR);
    let t = power / dist2;
    if t.is_finite() { t } else { 0.0 }
}

/// Fills a cell's reservoir by streaming RIS over a candidate light list.
///
/// `candidates` are indices into `lights`; `us` supplies one uniform in
/// `[0, 1]` per candidate (missing entries default to `0.5`).  Each candidate's
/// resampling weight is `p_hat * candidate_count` (uniform proposal), folded in
/// with [`Reservoir::update`].  The reservoir is then finalised against the
/// selected light's own `p_hat`, so [`query`] returns an unbiased contribution
/// weight.  The payload stored in the reservoir is the light index (`u32`).
///
/// An empty candidate list, or candidates that all evaluate to zero target,
/// yield an empty reservoir whose [`query`] returns `None`.
pub fn fill_cell_reservoir(
    cell_center: Vec3,
    lights: &[GridLight],
    candidates: &[u32],
    us: &[f32],
    min_dist: f32,
) -> Reservoir<u32> {
    let mut reservoir = Reservoir::<u32>::new();
    let count = candidates.len();
    if count == 0 {
        return reservoir;
    }
    let source_pdf = 1.0 / count as f32;

    for (k, &ci) in candidates.iter().enumerate() {
        let Some(light) = lights.get(ci as usize) else {
            continue;
        };
        let p_hat = cell_target(light, cell_center, min_dist);
        // Resampling weight w_i = p_hat / source_pdf; a zero target is dropped
        // by the reservoir's own non-positive-weight guard.
        let weight = p_hat / source_pdf;
        let u = us.get(k).copied().unwrap_or(0.5);
        reservoir.update(ci, weight, u);
    }

    if let Some(selected) = reservoir.sample() {
        // Re-evaluate the surviving light's target to finalise `W`.
        if let Some(light) = lights.get(selected as usize) {
            let p_hat_sel = cell_target(light, cell_center, min_dist);
            reservoir.finalize_weight(p_hat_sel);
        }
    }

    reservoir
}

/// The result of a grid lookup: the surviving light and its unbiased weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridQuery {
    /// Index of the selected light in the `lights` slice.
    pub light_index: u32,
    /// Unbiased RIS contribution weight `W` of the selected light.
    pub weight: f32,
}

/// Reads the selected light and its finalised weight out of a cell reservoir.
///
/// Returns `None` when the reservoir never captured a valid candidate (an empty
/// cell).  The weight is the value stored by
/// [`Reservoir::finalize_weight`] during [`fill_cell_reservoir`].
#[inline]
pub fn query(reservoir: &Reservoir<u32>) -> Option<GridQuery> {
    reservoir.sample().map(|light_index| GridQuery {
        light_index,
        weight: reservoir.contribution_weight(),
    })
}

/// Returns `x` if it is finite and strictly positive, else a tiny positive
/// floor so divisions by a cell size never blow up.
#[inline]
fn positive_or_floor(x: f32) -> f32 {
    if x.is_finite() && x > f32::MIN_POSITIVE {
        x
    } else {
        f32::MIN_POSITIVE
    }
}

/// Floors a per-axis grid coordinate and clamps it to `[0, dim - 1]`.
#[inline]
fn clamp_axis(local: f32, dim: u32) -> u32 {
    if !local.is_finite() || local <= 0.0 {
        return 0;
    }
    let max = dim.saturating_sub(1);
    let floored = local.floor();
    if floored >= max as f32 {
        max
    } else {
        floored as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_grid() -> GridConfig {
        GridConfig::new(Vec3::ZERO, Vec3::ONE, UVec3::new(4, 4, 4))
    }

    #[test]
    fn cell_center_roundtrips_through_cell_coord() {
        let grid = unit_grid();
        let dims = grid.clamped_dims();
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let coord = UVec3::new(x, y, z);
                    let center = grid.cell_center(coord);
                    assert_eq!(grid.cell_coord(center), coord);
                    assert_eq!(grid.cell_index(center), grid.flatten(coord));
                }
            }
        }
    }

    #[test]
    fn flatten_is_x_fastest_and_in_range() {
        let grid = GridConfig::new(Vec3::ZERO, Vec3::ONE, UVec3::new(2, 3, 4));
        assert_eq!(grid.cell_count(), 2 * 3 * 4);
        assert_eq!(grid.flatten(UVec3::new(0, 0, 0)), 0);
        assert_eq!(grid.flatten(UVec3::new(1, 0, 0)), 1);
        assert_eq!(grid.flatten(UVec3::new(0, 1, 0)), 2);
        assert_eq!(grid.flatten(UVec3::new(0, 0, 1)), 6);
        assert_eq!(grid.flatten(UVec3::new(1, 2, 3)), 1 + 2 * (2 + 3 * 3));
    }

    #[test]
    fn cell_coord_clamps_out_of_range_positions() {
        let grid = unit_grid();
        // Far negative -> cell 0; far positive -> last cell on each axis.
        assert_eq!(grid.cell_coord(Vec3::splat(-100.0)), UVec3::ZERO);
        assert_eq!(grid.cell_coord(Vec3::splat(100.0)), UVec3::new(3, 3, 3));
    }

    #[test]
    fn cell_coord_handles_origin_offset() {
        let grid = GridConfig::new(Vec3::new(10.0, 10.0, 10.0), Vec3::splat(2.0), UVec3::splat(5));
        // (10,10,10) is the min corner of cell 0; (13,10,10) lands in cell x=1.
        assert_eq!(grid.cell_coord(Vec3::new(10.5, 10.5, 10.5)), UVec3::ZERO);
        assert_eq!(grid.cell_coord(Vec3::new(13.0, 10.5, 10.5)), UVec3::new(1, 0, 0));
    }

    #[test]
    fn degenerate_grid_is_clamped() {
        // Zero dims and non-positive cell sizes must not panic or divide by 0.
        let grid = GridConfig::new(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), UVec3::ZERO);
        assert_eq!(grid.cell_count(), 1);
        assert_eq!(grid.cell_coord(Vec3::splat(5.0)), UVec3::ZERO);
        assert_eq!(grid.cell_index(Vec3::splat(5.0)), 0);
        let c = grid.cell_center(UVec3::ZERO);
        assert!(c.is_finite());
    }

    #[test]
    fn cell_target_falls_off_with_inverse_square_distance() {
        let light = GridLight::new(Vec3::ZERO, 4.0);
        let near = cell_target(&light, Vec3::new(0.0, 0.0, 1.0), 0.01);
        let far = cell_target(&light, Vec3::new(0.0, 0.0, 2.0), 0.01);
        assert!((near - 4.0).abs() < 1e-5, "near={near}");
        assert!((far - 1.0).abs() < 1e-5, "far={far}");
    }

    #[test]
    fn cell_target_clamps_and_rejects_zero_power() {
        let light = GridLight::new(Vec3::ZERO, 1.0);
        // Coincident point: clamped by min_dist so the result stays finite.
        let at = cell_target(&light, Vec3::ZERO, 0.5);
        assert!(at.is_finite());
        assert!((at - 4.0).abs() < 1e-3, "at={at}");
        // Zero power contributes nothing.
        let dark = GridLight::new(Vec3::ZERO, 0.0);
        assert_eq!(cell_target(&dark, Vec3::new(0.0, 0.0, 1.0), 0.1), 0.0);
    }

    #[test]
    fn single_candidate_finalizes_to_unit_weight() {
        // With one candidate, W = (w_sum / m) / p_hat = (p_hat / 1) / p_hat = 1.
        let lights = [GridLight::new(Vec3::new(0.0, 0.0, 2.0), 3.0)];
        let r = fill_cell_reservoir(Vec3::ZERO, &lights, &[0], &[0.5], 0.1);
        let q = query(&r).unwrap();
        assert_eq!(q.light_index, 0);
        assert!((q.weight - 1.0).abs() < 1e-5, "weight={}", q.weight);
    }

    #[test]
    fn fill_weight_equals_ris_estimator() {
        // Two candidates: W of the selected light must equal
        // sum(p_hat_i) / p_hat_selected (the RIS unbiased weight).
        let lights = [
            GridLight::new(Vec3::new(0.0, 0.0, 1.0), 1.0),
            GridLight::new(Vec3::new(0.0, 0.0, 2.0), 1.0),
        ];
        let center = Vec3::ZERO;
        let candidates = [0u32, 1];
        // u = 0 forces the first valid candidate to stay selected.
        let r = fill_cell_reservoir(center, &lights, &candidates, &[0.0, 1.0], 0.01);
        let q = query(&r).unwrap();
        let p0 = cell_target(&lights[0], center, 0.01);
        let p1 = cell_target(&lights[1], center, 0.01);
        let p_sel = cell_target(&lights[q.light_index as usize], center, 0.01);
        let expected = (p0 + p1) / p_sel;
        assert!((q.weight - expected).abs() < 1e-4, "w={} exp={expected}", q.weight);
    }

    #[test]
    fn empty_candidates_yield_no_sample() {
        let lights = [GridLight::new(Vec3::ZERO, 1.0)];
        let r = fill_cell_reservoir(Vec3::new(0.0, 0.0, 1.0), &lights, &[], &[], 0.1);
        assert!(query(&r).is_none());
    }

    #[test]
    fn all_zero_power_candidates_yield_no_sample() {
        let lights = [
            GridLight::new(Vec3::new(0.0, 0.0, 1.0), 0.0),
            GridLight::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
        ];
        let r = fill_cell_reservoir(Vec3::ZERO, &lights, &[0, 1], &[0.2, 0.8], 0.1);
        assert!(query(&r).is_none());
    }

    #[test]
    fn strong_near_light_is_preferred() {
        // A bright near light should win over a dim far one for most uniforms.
        let lights = [
            GridLight::new(Vec3::new(0.0, 0.0, 1.0), 10.0),
            GridLight::new(Vec3::new(0.0, 0.0, 8.0), 0.1),
        ];
        let center = Vec3::ZERO;
        let mut near_wins = 0;
        for k in 0..100 {
            let u1 = k as f32 / 100.0;
            let r = fill_cell_reservoir(center, &lights, &[0, 1], &[0.0, u1], 0.1);
            if query(&r).unwrap().light_index == 0 {
                near_wins += 1;
            }
        }
        assert!(near_wins > 90, "near_wins={near_wins}");
    }

    #[test]
    fn results_are_deterministic() {
        let lights = [
            GridLight::new(Vec3::new(0.0, 0.0, 1.0), 2.0),
            GridLight::new(Vec3::new(1.0, 0.0, 3.0), 4.0),
            GridLight::new(Vec3::new(-1.0, 1.0, 2.0), 1.0),
        ];
        let center = Vec3::new(0.0, 0.0, 0.0);
        let candidates = [0u32, 1, 2];
        let us = [0.3f32, 0.7, 0.1];
        let a = fill_cell_reservoir(center, &lights, &candidates, &us, 0.1);
        let b = fill_cell_reservoir(center, &lights, &candidates, &us, 0.1);
        assert_eq!(a, b);
        assert_eq!(query(&a), query(&b));
    }
}
