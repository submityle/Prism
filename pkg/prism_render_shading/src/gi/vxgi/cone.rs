//! Cone marching through a [`VoxelGrid`] — CPU golden.
//!
//! The second half of Crassin's *Voxel Cone Tracing* (2011) integrates the
//! voxelised scene along a **cone** rather than a ray.  Marching outward from
//! the apex, the cone's cross-section widens linearly with distance; at march
//! distance `t` its diameter is
//!
//! ```text
//! diameter(t) = 2 * tan(aperture / 2) * t
//! ```
//!
//! A cone of that diameter is approximated by a single pre-filtered voxel read
//! at the mip level whose voxel size matches the diameter:
//!
//! ```text
//! lod(t) = clamp(log2(diameter(t) / voxel_size), 0, max_level)
//! ```
//!
//! Samples are composited **front to back** with the standard emission /
//! absorption operator, so nearer occluders correctly shadow farther ones:
//!
//! ```text
//! C     += (1 - a_acc) * a_s * C_s
//! a_acc += (1 - a_acc) * a_s
//! ```
//!
//! Once `a_acc` reaches the opaque threshold the march terminates early — no
//! light behind a fully opaque front can reach the apex.  This module is the
//! backend-neutral, GPU-free reference for that integration.
//!
//! # Conventions
//! * **Direction.** `direction` is normalised internally; a degenerate
//!   (near-zero) direction falls back to `+Y` so no `NaN` enters the march.
//! * **Diameter / LOD.** [`cone_diameter`] and [`lod_from_diameter`] are the
//!   exact relations above; both are monotonically non-decreasing in distance,
//!   so a cone always reads from equal-or-coarser mips as it travels.
//! * **Stepping.** The step length is `step_scale * diameter(t)`, floored to a
//!   fraction of a voxel so a near-apex cone cannot stall; `step_scale` around
//!   `1` makes consecutive samples just touch.
//! * **Accumulation.** [`trace_cone`] returns the accumulated radiance, the
//!   accumulated opacity in `[0, 1]`, and the distance at which the march
//!   stopped.  Opacity is monotonically non-decreasing and never exceeds `1`.
//! * **Self-occlusion bias.** Marching starts at `start_offset` (clamped to at
//!   least one voxel) so the originating surface's own voxel does not occlude
//!   the gather.
//! * **Determinism / safety.** Pure functions, no RNG / I/O / GPU / `unsafe`;
//!   every divisor is guarded and no output can be `NaN`.

use bevy_math::{ops, Vec3};

use super::voxel::VoxelGrid;

/// Result of a single cone march.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeResult {
    /// Front-to-back accumulated radiance (non-negative RGB).
    pub radiance: Vec3,
    /// Accumulated opacity in `[0, 1]`; `1` means the cone became fully opaque.
    pub opacity: f32,
    /// March distance at which accumulation stopped (world units).
    pub distance: f32,
}

impl ConeResult {
    /// A cone that accumulated nothing (empty space).
    pub const EMPTY: Self = Self {
        radiance: Vec3::ZERO,
        opacity: 0.0,
        distance: 0.0,
    };
}

/// Tuning parameters for [`trace_cone`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeConfig {
    /// Full cone aperture in radians (apex angle).  Clamped to a small positive
    /// floor and below `pi` so `tan(aperture / 2)` stays finite and positive.
    pub aperture: f32,
    /// Maximum world-space march distance.  Clamped non-negative.
    pub max_distance: f32,
    /// Step length as a multiple of the cone diameter at each sample.  Clamped
    /// to a small positive floor; `~1` makes successive samples just touch.
    pub step_scale: f32,
    /// Distance from the apex at which marching begins, in world units.
    /// Clamped to at least one voxel to skip the originating surface.
    pub start_offset: f32,
    /// Marching-step budget.  Clamped to at least `1`.
    pub max_steps: u32,
    /// Accumulated-opacity threshold for early termination, in `[0, 1]`.
    pub alpha_termination: f32,
}

impl Default for ConeConfig {
    /// A balanced diffuse-ish default: a ~60-degree cone, an eight-unit trace,
    /// touching steps, a one-voxel start bias, a 64-step budget, and a 0.99
    /// opaque cutoff.
    fn default() -> Self {
        Self {
            aperture: core::f32::consts::FRAC_PI_3,
            max_distance: 8.0,
            step_scale: 1.0,
            start_offset: 0.0,
            max_steps: 64,
            alpha_termination: 0.99,
        }
    }
}

/// Smallest aperture (radians) tolerated; keeps `tan(aperture / 2)` positive.
const MIN_APERTURE: f32 = 1.0e-3;

/// Largest aperture (radians) tolerated; just under `pi` so the half-angle
/// stays below `pi/2` and `tan` stays finite.
const MAX_APERTURE: f32 = core::f32::consts::PI - 1.0e-3;

/// The cone diameter at march distance `dist` for a given full `aperture`.
///
/// `diameter = 2 * tan(aperture / 2) * dist`.  The aperture is clamped to a
/// valid open interval and the distance to non-negative, so the result is
/// always finite and non-negative.
#[inline]
pub fn cone_diameter(aperture: f32, dist: f32) -> f32 {
    let aperture = aperture.clamp(MIN_APERTURE, MAX_APERTURE);
    let dist = dist.max(0.0);
    2.0 * ops::tan(0.5 * aperture) * dist
}

/// Maps a cone `diameter` to a fractional mip LOD given the finest `voxel_size`
/// and the coarsest `max_level`.
///
/// `lod = clamp(log2(diameter / voxel_size), 0, max_level)`.  A diameter at or
/// below one voxel yields LOD `0`; the result is monotonically non-decreasing
/// in `diameter`.
#[inline]
pub fn lod_from_diameter(diameter: f32, voxel_size: f32, max_level: f32) -> f32 {
    let voxel_size = voxel_size.max(f32::MIN_POSITIVE);
    let ratio = (diameter / voxel_size).max(1.0);
    ops::log2(ratio).clamp(0.0, max_level.max(0.0))
}

/// Normalises `v`, falling back to `+Y` for a degenerate (near-zero) input.
#[inline]
fn normalize_or_up(v: Vec3) -> Vec3 {
    let len = v.length();
    if len > f32::MIN_POSITIVE {
        v / len
    } else {
        Vec3::Y
    }
}

/// Marches a cone from `origin` along `direction` through `grid`.
///
/// Composites pre-filtered voxel samples front-to-back with the emission /
/// absorption operator, selecting a mip LOD from the cone diameter at each
/// step, and terminates early once the accumulated opacity crosses
/// `alpha_termination`.
pub fn trace_cone(grid: &VoxelGrid, origin: Vec3, direction: Vec3, config: ConeConfig) -> ConeResult {
    let dir = normalize_or_up(direction);
    let aperture = config.aperture.clamp(MIN_APERTURE, MAX_APERTURE);
    let max_distance = config.max_distance.max(0.0);
    let step_scale = config.step_scale.max(1.0e-3);
    let max_steps = config.max_steps.max(1);
    let alpha_cutoff = config.alpha_termination.clamp(0.0, 1.0);

    let voxel_size = grid.voxel_size();
    let max_level = grid.max_level() as f32;
    // Floor the start and step so a near-zero diameter near the apex cannot
    // produce a zero advance and stall the march.
    let min_step = 0.5 * voxel_size;
    let mut dist = config.start_offset.max(voxel_size);

    let mut radiance = Vec3::ZERO;
    let mut alpha_acc = 0.0_f32;
    let mut steps = 0;

    while steps < max_steps && dist <= max_distance && alpha_acc < alpha_cutoff {
        let diameter = cone_diameter(aperture, dist).max(voxel_size);
        let lod = lod_from_diameter(diameter, voxel_size, max_level);
        let sample_pos = origin + dir * dist;
        let sample = grid.sample_world_lod(sample_pos, lod);

        let a_s = sample.opacity.clamp(0.0, 1.0);
        let transmit = 1.0 - alpha_acc;
        radiance += transmit * a_s * sample.radiance;
        alpha_acc += transmit * a_s;

        let step = (diameter * step_scale).max(min_step);
        dist += step;
        steps += 1;
    }

    ConeResult {
        radiance: radiance.max(Vec3::ZERO),
        opacity: alpha_acc.clamp(0.0, 1.0),
        distance: dist.min(max_distance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::vxgi::voxel::VoxelGrid;
    use bevy_math::UVec3;

    fn filled_grid(res: u32, opacity: f32, radiance: Vec3) -> VoxelGrid {
        let mut grid = VoxelGrid::new(UVec3::splat(res), Vec3::ZERO, Vec3::splat(res as f32));
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    grid.set_voxel(UVec3::new(x, y, z), opacity, radiance);
                }
            }
        }
        grid.build_mips();
        grid
    }

    #[test]
    fn diameter_grows_with_distance() {
        let a = core::f32::consts::FRAC_PI_3;
        let d0 = cone_diameter(a, 1.0);
        let d1 = cone_diameter(a, 2.0);
        let d2 = cone_diameter(a, 4.0);
        assert!(d0 < d1 && d1 < d2, "{d0} {d1} {d2}");
        // 2 * tan(30deg) * 1 ~= 1.1547
        assert!((d0 - 1.154_700_5).abs() < 1.0e-3, "{d0}");
    }

    #[test]
    fn lod_increases_with_distance() {
        let a = core::f32::consts::FRAC_PI_3;
        let vs = 1.0;
        let max_level = 6.0;
        let near = lod_from_diameter(cone_diameter(a, 1.0), vs, max_level);
        let mid = lod_from_diameter(cone_diameter(a, 8.0), vs, max_level);
        let far = lod_from_diameter(cone_diameter(a, 64.0), vs, max_level);
        assert!(near <= mid && mid <= far, "{near} {mid} {far}");
        assert!(far <= max_level + 1.0e-6);
    }

    #[test]
    fn lod_floored_at_zero_for_small_diameters() {
        let lod = lod_from_diameter(0.1, 1.0, 6.0);
        assert_eq!(lod, 0.0);
    }

    #[test]
    fn empty_grid_accumulates_nothing() {
        let grid = VoxelGrid::new(UVec3::splat(8), Vec3::ZERO, Vec3::splat(8.0));
        // No mips built, all zero.
        let r = trace_cone(&grid, Vec3::splat(0.5), Vec3::X, ConeConfig::default());
        assert_eq!(r.radiance, Vec3::ZERO);
        assert!(r.opacity.abs() < 1.0e-6, "opacity {}", r.opacity);
    }

    #[test]
    fn opacity_monotone_and_saturates() {
        let grid = filled_grid(8, 0.6, Vec3::splat(1.0));
        let cfg = ConeConfig {
            aperture: core::f32::consts::FRAC_PI_6,
            max_distance: 8.0,
            step_scale: 0.5,
            start_offset: 0.0,
            max_steps: 128,
            alpha_termination: 0.99,
        };
        let r = trace_cone(&grid, Vec3::splat(0.5), Vec3::new(1.0, 0.3, 0.2), cfg);
        // A dense, emissive medium drives accumulated opacity to the cutoff.
        assert!(r.opacity > 0.95, "opacity {}", r.opacity);
        assert!(r.opacity <= 1.0 + 1.0e-6, "opacity {}", r.opacity);
        assert!(r.radiance.x >= 0.0, "radiance {:?}", r.radiance);
    }

    #[test]
    fn opacity_never_exceeds_one_step_by_step() {
        // Reconstruct the accumulation by hand to assert monotonicity.
        let grid = filled_grid(8, 1.0, Vec3::splat(2.0));
        let mut alpha = 0.0_f32;
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let origin = Vec3::new(0.5, 4.0, 4.0);
        let vs = grid.voxel_size();
        let mut dist = vs;
        let mut prev = -1.0;
        for _ in 0..64 {
            let s = grid.sample_world_lod(origin + dir * dist, 0.0);
            alpha += (1.0 - alpha) * s.opacity.clamp(0.0, 1.0);
            assert!(alpha >= prev - 1.0e-6, "not monotone: {prev} -> {alpha}");
            assert!(alpha <= 1.0 + 1.0e-6, "exceeded one: {alpha}");
            prev = alpha;
            dist += vs;
        }
    }

    #[test]
    fn radiance_non_negative_and_finite() {
        let grid = filled_grid(8, 0.4, Vec3::new(0.5, 1.5, 3.0));
        let r = trace_cone(&grid, Vec3::splat(0.5), Vec3::new(0.2, 1.0, 0.1), ConeConfig::default());
        assert!(r.radiance.x >= 0.0 && r.radiance.y >= 0.0 && r.radiance.z >= 0.0);
        assert!(r.radiance.is_finite());
    }

    #[test]
    fn degenerate_direction_falls_back() {
        let grid = filled_grid(4, 0.5, Vec3::splat(1.0));
        let r = trace_cone(&grid, Vec3::splat(2.0), Vec3::ZERO, ConeConfig::default());
        // Should not panic or produce NaN; falls back to +Y.
        assert!(r.radiance.is_finite());
        assert!(r.opacity.is_finite());
    }

    #[test]
    fn degenerate_aperture_is_safe() {
        let grid = filled_grid(4, 0.5, Vec3::splat(1.0));
        let cfg = ConeConfig {
            aperture: 0.0,
            ..ConeConfig::default()
        };
        let r = trace_cone(&grid, Vec3::splat(0.5), Vec3::X, cfg);
        assert!(r.radiance.is_finite() && r.opacity.is_finite());
    }
}
