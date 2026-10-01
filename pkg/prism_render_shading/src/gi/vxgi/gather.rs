//! Diffuse and specular indirect gathering built on the cone marcher — CPU
//! golden.
//!
//! Given a shading point and its surface frame, Crassin-style voxel cone
//! tracing reconstructs two kinds of indirect light:
//!
//! * **Diffuse.** A small fan of wide cones covering the cosine hemisphere
//!   about the surface normal.  Each cone is marched independently and the
//!   results are combined with cosine weights, approximating the hemispherical
//!   irradiance integral `∫ L(ω) (n·ω) dω`.
//! * **Specular.** A single cone fired along the mirror-reflection direction
//!   whose aperture widens with surface roughness — a sharp mirror reads a thin
//!   cone, a rough surface a fat one.
//!
//! This module is the backend-neutral, GPU-free reference for those gathers.
//!
//! # Conventions
//! * **Frame.** Local cone directions have `+Z` aligned with the surface
//!   normal; [`orthonormal_basis`] lifts them into world space with a
//!   singularity-free Duff et al. (2017) basis.  A degenerate normal falls back
//!   to `+Y`.
//! * **Diffuse cone set.** One cone along the normal plus five evenly spread
//!   side cones tilted off the normal; all lie in the upper hemisphere so a
//!   source *behind* the surface contributes nothing.  Each cone is weighted by
//!   `max(n·ω, 0)` and the sum is normalised by the total weight, so the gather
//!   is an energy-preserving cosine average, never negative.
//! * **Specular aperture.** [`specular_aperture`] maps roughness in `[0, 1]`
//!   monotonically onto `[MIN_SPEC_APERTURE, MAX_SPEC_APERTURE]`; a mirror
//!   (`roughness = 0`) traces the thinnest cone.
//! * **Reflection.** The reflected direction is the mirror of the (outgoing)
//!   `view` vector about the normal; if it points below the surface it is
//!   clamped back into the hemisphere so the cone never marches into the
//!   geometry.
//! * **Determinism / safety.** Pure functions, no RNG / I/O / GPU / `unsafe`;
//!   every divisor is guarded and no output can be `NaN`.

use bevy_math::{ops, Vec3};

use super::cone::{trace_cone, ConeConfig};
use super::voxel::VoxelGrid;

/// Thinnest specular cone aperture (radians), used by a perfect mirror.
pub const MIN_SPEC_APERTURE: f32 = 0.02;

/// Widest specular cone aperture (radians), used by the roughest surface.
pub const MAX_SPEC_APERTURE: f32 = core::f32::consts::FRAC_PI_3;

/// Half-angle (radians) of each diffuse cone's tilt off the normal for the five
/// side cones.  ~45 degrees spreads the fan across the hemisphere.
const DIFFUSE_SIDE_TILT: f32 = core::f32::consts::FRAC_PI_4;

/// Number of side cones ringing the central (normal-aligned) diffuse cone.
const DIFFUSE_SIDE_CONES: usize = 5;

/// Parameters shared by the diffuse and specular gathers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatherParams {
    /// Full aperture of each diffuse hemisphere cone (radians).
    pub diffuse_aperture: f32,
    /// Maximum world-space march distance for every cone.
    pub max_distance: f32,
    /// Step length as a multiple of the cone diameter.
    pub step_scale: f32,
    /// Distance from the surface at which cones begin, world units.
    pub start_offset: f32,
    /// Marching-step budget per cone.
    pub max_steps: u32,
    /// Accumulated-opacity cutoff for early cone termination.
    pub alpha_termination: f32,
}

impl Default for GatherParams {
    fn default() -> Self {
        Self {
            diffuse_aperture: core::f32::consts::FRAC_PI_3,
            max_distance: 8.0,
            step_scale: 1.0,
            start_offset: 0.0,
            max_steps: 64,
            alpha_termination: 0.99,
        }
    }
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

/// Builds a right-handed orthonormal basis `(tangent, bitangent)` for a unit
/// `normal` using Duff et al. (2017), stable across the whole sphere.
#[inline]
fn orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    let sign = if normal.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = Vec3::new(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);
    (tangent, bitangent)
}

/// Maps surface `roughness` to a specular cone aperture (radians).
///
/// Linear and monotonically non-decreasing on `[0, 1]`: `0` yields
/// [`MIN_SPEC_APERTURE`] (mirror), `1` yields [`MAX_SPEC_APERTURE`].
#[inline]
pub fn specular_aperture(roughness: f32) -> f32 {
    let r = roughness.clamp(0.0, 1.0);
    MIN_SPEC_APERTURE + (MAX_SPEC_APERTURE - MIN_SPEC_APERTURE) * r
}

/// Mirror-reflects the outgoing `view` vector about `normal`, clamping the
/// result back into the upper hemisphere so a grazing reflection never dives
/// into the surface.
#[inline]
pub fn reflection_direction(view: Vec3, normal: Vec3) -> Vec3 {
    let n = normalize_or_up(normal);
    let v = normalize_or_up(view);
    let r = 2.0 * v.dot(n) * n - v;
    let r = normalize_or_up(r);
    // Keep the cone out of the geometry: if it points below the horizon, lift
    // it to graze along the surface.
    if r.dot(n) < 0.0 {
        normalize_or_up(r - n * r.dot(n))
    } else {
        r
    }
}

/// The six local diffuse cone directions (`+Z` = normal) with their cosine
/// weights `max(z, 0)`.
fn diffuse_local_cones() -> [(Vec3, f32); DIFFUSE_SIDE_CONES + 1] {
    let mut cones = [(Vec3::Z, 1.0); DIFFUSE_SIDE_CONES + 1];
    let sin_t = ops::sin(DIFFUSE_SIDE_TILT);
    let cos_t = ops::cos(DIFFUSE_SIDE_TILT);
    let step = core::f32::consts::TAU / DIFFUSE_SIDE_CONES as f32;
    for i in 0..DIFFUSE_SIDE_CONES {
        let az = step * i as f32;
        let dir = Vec3::new(sin_t * ops::cos(az), sin_t * ops::sin(az), cos_t);
        // Cosine weight for a direction whose z-component is cos_t.
        cones[i + 1] = (dir, cos_t.max(0.0));
    }
    cones
}

/// Gathers hemispherical indirect **diffuse** radiance at `position`.
///
/// Fires the fixed diffuse cone fan into the hemisphere about `normal`,
/// cosine-weights each cone's accumulated radiance, and normalises by the total
/// weight.  The result is non-negative and never contains `NaN`.
pub fn gather_diffuse(grid: &VoxelGrid, position: Vec3, normal: Vec3, params: GatherParams) -> Vec3 {
    let n = normalize_or_up(normal);
    let (tangent, bitangent) = orthonormal_basis(n);
    let aperture = params.diffuse_aperture;

    let cfg = ConeConfig {
        aperture,
        max_distance: params.max_distance,
        step_scale: params.step_scale,
        start_offset: params.start_offset,
        max_steps: params.max_steps,
        alpha_termination: params.alpha_termination,
    };

    let mut sum = Vec3::ZERO;
    let mut weight_sum = 0.0;
    for (local, cos_weight) in diffuse_local_cones() {
        let world_dir = tangent * local.x + bitangent * local.y + n * local.z;
        let world_dir = normalize_or_up(world_dir);
        // Only cones in the upper hemisphere carry weight.
        let w = world_dir.dot(n).max(0.0) * cos_weight;
        if w <= 0.0 {
            continue;
        }
        let result = trace_cone(grid, position, world_dir, cfg);
        sum += result.radiance * w;
        weight_sum += w;
    }

    if weight_sum > f32::MIN_POSITIVE {
        (sum / weight_sum).max(Vec3::ZERO)
    } else {
        Vec3::ZERO
    }
}

/// Gathers indirect **specular** radiance at `position` along the reflection of
/// `view` about `normal`.
///
/// The single reflection cone's aperture widens with `roughness`
/// ([`specular_aperture`]).  The returned radiance is non-negative and finite.
pub fn gather_specular(
    grid: &VoxelGrid,
    position: Vec3,
    normal: Vec3,
    view: Vec3,
    roughness: f32,
    params: GatherParams,
) -> Vec3 {
    let dir = reflection_direction(view, normal);
    let cfg = ConeConfig {
        aperture: specular_aperture(roughness),
        max_distance: params.max_distance,
        step_scale: params.step_scale,
        start_offset: params.start_offset,
        max_steps: params.max_steps,
        alpha_termination: params.alpha_termination,
    };
    trace_cone(grid, position, dir, cfg).radiance.max(Vec3::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::vxgi::voxel::VoxelGrid;
    use bevy_math::UVec3;

    /// A grid with a single bright, opaque voxel at `coord`.
    fn emitter_grid(res: u32, coord: UVec3, radiance: Vec3) -> VoxelGrid {
        let mut grid = VoxelGrid::new(UVec3::splat(res), Vec3::ZERO, Vec3::splat(res as f32));
        grid.set_voxel(coord, 1.0, radiance);
        grid.build_mips();
        grid
    }

    #[test]
    fn specular_aperture_monotone_in_roughness() {
        let a0 = specular_aperture(0.0);
        let a1 = specular_aperture(0.25);
        let a2 = specular_aperture(0.5);
        let a3 = specular_aperture(1.0);
        assert!(a0 < a1 && a1 < a2 && a2 < a3, "{a0} {a1} {a2} {a3}");
        assert!((a0 - MIN_SPEC_APERTURE).abs() < 1.0e-6);
        assert!((a3 - MAX_SPEC_APERTURE).abs() < 1.0e-6);
    }

    #[test]
    fn specular_aperture_clamps_out_of_range() {
        assert_eq!(specular_aperture(-5.0), MIN_SPEC_APERTURE);
        assert_eq!(specular_aperture(5.0), MAX_SPEC_APERTURE);
    }

    #[test]
    fn diffuse_cones_are_all_upper_hemisphere() {
        for (dir, w) in diffuse_local_cones() {
            assert!(dir.z >= 0.0, "cone dips below hemisphere: {dir:?}");
            assert!(w >= 0.0, "negative weight {w}");
        }
    }

    #[test]
    fn diffuse_energy_non_negative() {
        let grid = emitter_grid(8, UVec3::new(4, 7, 4), Vec3::splat(5.0));
        let pos = Vec3::new(4.5, 1.0, 4.5);
        let out = gather_diffuse(&grid, pos, Vec3::Y, GatherParams::default());
        assert!(out.x >= 0.0 && out.y >= 0.0 && out.z >= 0.0, "{out:?}");
        assert!(out.is_finite());
    }

    #[test]
    fn diffuse_receives_light_from_above() {
        // Emitter directly above a floor point whose normal points up.
        let grid = emitter_grid(8, UVec3::new(4, 7, 4), Vec3::splat(8.0));
        let pos = Vec3::new(4.5, 0.5, 4.5);
        let up = gather_diffuse(&grid, pos, Vec3::Y, GatherParams::default());
        assert!(up.length() > 0.0, "expected light from above, got {up:?}");
    }

    #[test]
    fn emitter_behind_surface_does_not_contribute() {
        // Emitter above, but the surface normal points DOWN, so the whole
        // hemisphere faces away from the emitter.
        let grid = emitter_grid(8, UVec3::new(4, 7, 4), Vec3::splat(8.0));
        let pos = Vec3::new(4.5, 0.5, 4.5);
        let facing_up = gather_diffuse(&grid, pos, Vec3::Y, GatherParams::default());
        let facing_down = gather_diffuse(&grid, pos, Vec3::NEG_Y, GatherParams::default());
        assert!(
            facing_down.length() < facing_up.length(),
            "down {facing_down:?} should be dimmer than up {facing_up:?}"
        );
    }

    #[test]
    fn reflection_direction_mirrors_about_normal() {
        // View straight down onto an up-facing surface reflects straight up.
        let r = reflection_direction(Vec3::Y, Vec3::Y);
        assert!((r - Vec3::Y).length() < 1.0e-5, "{r:?}");
    }

    #[test]
    fn reflection_stays_in_hemisphere() {
        // A grazing view whose mirror would dip below is lifted to the horizon.
        let view = Vec3::new(1.0, 0.05, 0.0).normalize();
        let r = reflection_direction(view, Vec3::Y);
        assert!(r.dot(Vec3::Y) >= -1.0e-6, "reflection below surface: {r:?}");
        assert!(r.is_finite());
    }

    #[test]
    fn specular_energy_non_negative() {
        let grid = emitter_grid(8, UVec3::new(4, 4, 7), Vec3::splat(6.0));
        let pos = Vec3::new(4.5, 4.5, 0.5);
        let view = Vec3::new(0.0, 0.0, -1.0);
        let out = gather_specular(&grid, pos, Vec3::Z, view, 0.3, GatherParams::default());
        assert!(out.x >= 0.0 && out.y >= 0.0 && out.z >= 0.0, "{out:?}");
        assert!(out.is_finite());
    }

    #[test]
    fn degenerate_normal_is_safe() {
        let grid = emitter_grid(4, UVec3::new(2, 2, 2), Vec3::splat(1.0));
        let out = gather_diffuse(&grid, Vec3::splat(1.0), Vec3::ZERO, GatherParams::default());
        assert!(out.is_finite());
    }
}
