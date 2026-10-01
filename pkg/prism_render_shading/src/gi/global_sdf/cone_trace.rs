//! Sphere marching and cone-tapered AO / GI gather over a global brick field —
//! CPU golden.
//!
//! Once object fields are merged into a [`BrickGrid`](super::brick_grid::BrickGrid)
//! the global distance field is traced for two things, exactly as Lumen does:
//!
//! * **Sphere marching** — a standard SDF ray march where the sampled distance
//!   at each step is a safe advance along the ray, converging on the first
//!   surface it reaches ([`sphere_march`]).
//! * **Cone tracing** — a handful of cones spread over the cosine hemisphere.
//!   Along a cone the sampled distance divided by the cone radius at that step
//!   gives the fraction of the cone still unobstructed; the minimum along the
//!   cone is that cone's visibility.  Averaging the per-cone visibilities with
//!   cosine weights yields hemispherical ambient occlusion ([`cone_trace_ao`]),
//!   and accumulating incoming radiance front-to-back along the same cones
//!   yields a one-bounce GI gather ([`cone_gather_gi`]).
//!
//! This module is the backend-neutral, GPU-free reference for those passes.
//!
//! # Conventions
//! * **AO return value.** [`cone_trace_ao`] returns *visibility* in `[0, 1]`:
//!   `1.0` is fully open (no occlusion), `0.0` fully occluded.  In genuinely
//!   open space it returns `1.0`.
//! * **Cone visibility.** Along a cone the running visibility is
//!   `min_t clamp(distance(t) / (radius_scale * t), 0, 1)`, the Lumen/UE
//!   soft-occlusion approximation where `radius_scale * t` is the cone radius at
//!   march distance `t`.  Equivalently the per-step occupancy is
//!   `clamp(1 - distance / cone_radius, 0, 1)` and the cone's occlusion is the
//!   maximum occupancy encountered; the closest approach dominates.
//! * **GI gather.** [`cone_gather_gi`] marches each cone front-to-back,
//!   converting per-step occupancy into an opacity and compositing the supplied
//!   `radiance` samples through the remaining transmittance.  Any transmittance
//!   left when a cone exits the trace distance picks up the far/sky radiance.
//!   The per-cone result is cosine-weighted into a hemispherical irradiance.
//! * **Hemisphere sampling.** Cone directions are a deterministic Fibonacci
//!   hemisphere lifted into the surface frame built from `normal`; the sampling
//!   is fixed and seedless, so results reproduce bit-for-bit.
//! * **Self-occlusion bias.** Tracing starts at `origin + normal * bias` so the
//!   surface a point sits on does not occlude itself; steps are floored to a
//!   minimum so a near-zero distance in open space cannot stall the march.
//! * **Determinism / safety.** Pure functions, no RNG/IO/GPU/global state, no
//!   `unsafe`; every divisor is guarded and no output can be `NaN`.

use bevy_math::{ops, Vec3};

use super::brick_grid::BrickGrid;

/// Golden-ratio conjugate used to spread Fibonacci-hemisphere azimuths.
const GOLDEN_ANGLE_FRAC: f32 = 0.618_033_99;

/// Convergence threshold (world units) for a sphere-march surface hit.
const HIT_EPS: f32 = 1.0e-3;

/// A single sphere-march intersection against a [`BrickGrid`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchHit {
    /// Distance travelled from the ray origin to the surface, world units.
    pub distance: f32,
    /// World-space position of the intersection (`origin + dir * distance`).
    pub position: Vec3,
    /// Number of marching iterations taken before convergence (diagnostic).
    pub steps: u32,
}

/// Tuning parameters shared by [`cone_trace_ao`] and [`cone_gather_gi`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConeConfig {
    /// Number of cones traced over the hemisphere.  Clamped to at least `1`.
    pub cone_count: u32,
    /// Maximum world-space distance marched along each cone.  Beyond this the
    /// cone is treated as unobstructed / open to the far field.
    pub max_distance: f32,
    /// Cone radius growth per unit distance: the cone radius at march distance
    /// `t` is `radius_scale * t`.  Larger values widen the cones.
    pub radius_scale: f32,
    /// Marching-step budget per cone.  Clamped to at least `1`.
    pub max_steps: u32,
    /// World-space offset along the normal where tracing begins, preventing the
    /// surface from occluding itself.  Clamped non-negative.
    pub bias: f32,
}

impl Default for ConeConfig {
    /// A balanced default: 9 cones, a one-unit trace, moderately wide cones, a
    /// 48-step budget, and a small self-occlusion bias.
    fn default() -> Self {
        Self {
            cone_count: 9,
            max_distance: 1.0,
            radius_scale: 0.5,
            max_steps: 48,
            bias: 0.02,
        }
    }
}

/// Normalises `v`, falling back to `+Y` for a degenerate (near-zero) input so
/// frames and cone directions never contain `NaN`.
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

/// The `i`-th of `n` Fibonacci-hemisphere directions in the local frame where
/// `+Z` is the normal, returned with its cosine weight (`local.z`).
#[inline]
fn hemisphere_dir(i: u32, n: f32) -> (Vec3, f32) {
    let fi = i as f32 + 0.5;
    // cos(theta) walks from near 1 (around the normal) toward 0 (horizon).
    let cos_theta = (1.0 - fi / n).clamp(0.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = core::f32::consts::TAU * (fi * GOLDEN_ANGLE_FRAC);
    let (sin_phi, cos_phi) = (ops::sin(phi), ops::cos(phi));
    let dir = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    (dir, cos_theta)
}

/// Sphere-traces a ray against the field and returns the first surface hit.
///
/// At each step the sampled distance is a safe radius to advance along `dir`.
/// Convergence is declared when the sampled distance drops below [`HIT_EPS`]
/// (including a ray that starts inside the surface, where the distance is
/// already negative).  Each advance is floored to a minimum step so a near-zero
/// distance in open space cannot stall the march, and the loop stops once the
/// travelled distance exceeds `max_dist` or `max_steps` is reached.
///
/// Returns `None` when the ray misses within the budget or when `dir` is too
/// short to normalise.
pub fn sphere_march(
    grid: &BrickGrid,
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    max_steps: u32,
) -> Option<MarchHit> {
    let len = dir.length();
    if len <= f32::MIN_POSITIVE {
        return None;
    }
    let dir = dir / len;
    let max_dist = max_dist.max(0.0);
    let min_step = (max_dist * 1.0e-3).max(1.0e-4);
    let max_steps = max_steps.max(1);

    let mut t = 0.0f32;
    for step in 0..max_steps {
        let position = origin + dir * t;
        let d = grid.sample_distance(position);
        if d < HIT_EPS {
            return Some(MarchHit {
                distance: t,
                position,
                steps: step,
            });
        }
        t += d.max(min_step);
        if t > max_dist {
            break;
        }
    }
    None
}

/// Visibility of one cone: marches the field and tracks the minimum
/// `distance / cone_radius` ratio (its closest approach to an occluder).
///
/// Returns `1.0` for a fully open cone and `0.0` when the cone penetrates
/// geometry (negative sampled distance) or grazes it tightly.
#[inline]
fn trace_cone_visibility(grid: &BrickGrid, start: Vec3, dir: Vec3, cfg: &ConeConfig) -> f32 {
    let max_distance = cfg.max_distance.max(0.0);
    let radius_scale = cfg.radius_scale.max(f32::MIN_POSITIVE);
    let max_steps = cfg.max_steps.max(1);
    let mut t = (max_distance * 1.0e-3).max(1.0e-4);
    let min_step = (max_distance / max_steps as f32).max(1.0e-4);
    let mut visibility = 1.0f32;

    for _ in 0..max_steps {
        if t > max_distance {
            break;
        }
        let d = grid.sample_distance(start + dir * t);
        if d <= 0.0 {
            return 0.0;
        }
        let cone_radius = radius_scale * t;
        let ratio = (d / cone_radius).clamp(0.0, 1.0);
        if ratio < visibility {
            visibility = ratio;
        }
        if visibility <= 0.0 {
            break;
        }
        t += d.max(min_step);
    }
    visibility.clamp(0.0, 1.0)
}

/// Cone-traces hemispherical ambient occlusion at a surface point.
///
/// `origin` is the shading position and `normal` its surface normal
/// (normalised internally, with a `+Y` fallback).  Returns hemispherical
/// *visibility* in `[0, 1]`, where `1.0` means fully unoccluded.  See the
/// [module docs](self) for the cone-visibility and weighting conventions.
///
/// Fully deterministic and defensively clamped: with no nearby geometry it
/// returns `1.0`, every divisor is guarded, and the result is never `NaN`.
pub fn cone_trace_ao(grid: &BrickGrid, origin: Vec3, normal: Vec3, cfg: &ConeConfig) -> f32 {
    let normal = normalize_or_up(normal);
    let cone_count = cfg.cone_count.max(1);
    let bias = cfg.bias.max(0.0);
    let start = origin + normal * bias;
    let (tangent, bitangent) = orthonormal_basis(normal);
    let n = cone_count as f32;

    let mut weighted_visibility = 0.0f32;
    let mut total_weight = 0.0f32;

    for i in 0..cone_count {
        let (local, weight) = hemisphere_dir(i, n);
        if weight <= 0.0 {
            continue;
        }
        let dir = normalize_or_up(tangent * local.x + bitangent * local.y + normal * local.z);
        let vis = trace_cone_visibility(grid, start, dir, cfg);
        weighted_visibility += vis * weight;
        total_weight += weight;
    }

    if total_weight > f32::MIN_POSITIVE {
        (weighted_visibility / total_weight).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Gathers one-bounce incoming irradiance by cone tracing the global field.
///
/// For each hemisphere cone the field is marched front-to-back.  At each step
/// the cone occupancy `clamp(1 - distance / cone_radius, 0, 1)` is treated as
/// the opacity of a thin occluding slab; the slab emits `radiance(position)`
/// scaled by that opacity and the transmittance still remaining, and the
/// transmittance is attenuated accordingly.  When a cone exits `max_distance`
/// with transmittance left, that remainder is lit by `radiance` sampled far
/// along the cone (the sky / far field).  Per-cone results are cosine-weighted
/// into a single hemispherical irradiance.
///
/// `radiance` is any deterministic world-space radiance lookup (an environment
/// map, a lit far field, a previous-frame cache).  The gather itself adds no
/// randomness; with an all-open field the result is the cosine-weighted average
/// of the far-field radiance.  Every component is finite and non-negative.
pub fn cone_gather_gi(
    grid: &BrickGrid,
    origin: Vec3,
    normal: Vec3,
    cfg: &ConeConfig,
    radiance: impl Fn(Vec3) -> Vec3,
) -> Vec3 {
    let normal = normalize_or_up(normal);
    let cone_count = cfg.cone_count.max(1);
    let bias = cfg.bias.max(0.0);
    let start = origin + normal * bias;
    let (tangent, bitangent) = orthonormal_basis(normal);
    let n = cone_count as f32;

    let max_distance = cfg.max_distance.max(0.0);
    let radius_scale = cfg.radius_scale.max(f32::MIN_POSITIVE);
    let max_steps = cfg.max_steps.max(1);
    let min_step = (max_distance / max_steps as f32).max(1.0e-4);

    let mut accum = Vec3::ZERO;
    let mut total_weight = 0.0f32;

    for i in 0..cone_count {
        let (local, weight) = hemisphere_dir(i, n);
        if weight <= 0.0 {
            continue;
        }
        let dir = normalize_or_up(tangent * local.x + bitangent * local.y + normal * local.z);

        let mut t = (max_distance * 1.0e-3).max(1.0e-4);
        let mut transmittance = 1.0f32;
        let mut cone_radiance = Vec3::ZERO;

        for _ in 0..max_steps {
            if t > max_distance || transmittance <= 1.0e-3 {
                break;
            }
            let position = start + dir * t;
            let d = grid.sample_distance(position);
            let cone_radius = radius_scale * t;
            // Occupancy of the cone's cross-section blocked here.
            let occupancy = (1.0 - d / cone_radius).clamp(0.0, 1.0);
            if occupancy > 0.0 {
                let sample = sanitize_radiance(radiance(position));
                cone_radiance += sample * (occupancy * transmittance);
                transmittance *= 1.0 - occupancy;
            }
            // Advance: inside geometry take the floor step, else sphere-step.
            t += d.max(min_step);
        }

        // Whatever light leaks past the cone comes from the far field.
        if transmittance > 0.0 {
            let far = start + dir * max_distance;
            cone_radiance += sanitize_radiance(radiance(far)) * transmittance;
        }

        accum += cone_radiance * weight;
        total_weight += weight;
    }

    if total_weight > f32::MIN_POSITIVE {
        accum / total_weight
    } else {
        Vec3::ZERO
    }
}

/// Clamps a radiance sample to finite, non-negative components so a stray
/// `NaN`/`inf` or negative lookup cannot poison the accumulation.
#[inline]
fn sanitize_radiance(c: Vec3) -> Vec3 {
    let fix = |x: f32| if x.is_finite() { x.max(0.0) } else { 0.0 };
    Vec3::new(fix(c.x), fix(c.y), fix(c.z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::brick_grid::DEFAULT_BRICK_DIM;
    use super::super::merge::{bake_merged, SdfObject, SdfPrimitive};
    use bevy_math::IVec3;

    /// A single unit sphere at the origin baked into a dense brick grid.
    fn sphere_grid() -> BrickGrid {
        let objs = [SdfObject::new(SdfPrimitive::Sphere { radius: 1.0 }, Vec3::ZERO)];
        bake_merged(
            &objs,
            Vec3::ZERO,
            0.1,
            DEFAULT_BRICK_DIM,
            IVec3::splat(-3),
            IVec3::splat(2),
            0.0,
        )
    }

    /// A large occluder sphere sitting directly above the origin, with plenty
    /// of baked open space below it.
    fn ceiling_grid() -> BrickGrid {
        let objs = [SdfObject::new(SdfPrimitive::Sphere { radius: 1.5 }, Vec3::new(0.0, 2.0, 0.0))];
        bake_merged(
            &objs,
            Vec3::ZERO,
            0.1,
            DEFAULT_BRICK_DIM,
            IVec3::new(-4, -3, -4),
            IVec3::new(4, 5, 4),
            0.0,
        )
    }

    #[test]
    fn sphere_march_hits_known_surface() {
        let grid = sphere_grid();
        // March from +X toward the origin; the unit sphere surface is at x = 1.
        // Start at x = 2.3 (inside the baked brick range) so the distance to
        // the surface is 1.3.
        let hit = sphere_march(&grid, Vec3::new(2.3, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), 10.0, 128)
            .expect("ray should hit the sphere");
        assert!((hit.distance - 1.3).abs() < 0.05, "distance {}", hit.distance);
        assert!((hit.position.x - 1.0).abs() < 0.05, "x {}", hit.position.x);
    }

    #[test]
    fn sphere_march_misses_empty_direction() {
        let grid = sphere_grid();
        // March away from the sphere: nothing to hit.
        let hit = sphere_march(&grid, Vec3::new(2.3, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 10.0, 128);
        assert!(hit.is_none());
    }

    #[test]
    fn sphere_march_degenerate_dir_is_none() {
        let grid = sphere_grid();
        assert!(sphere_march(&grid, Vec3::ZERO, Vec3::ZERO, 10.0, 64).is_none());
    }

    #[test]
    fn ao_is_open_in_free_space() {
        let grid = sphere_grid();
        // Far below the sphere, hemisphere facing down: essentially nothing
        // overhead within the trace distance.
        let cfg = ConeConfig {
            max_distance: 1.0,
            ..ConeConfig::default()
        };
        let vis = cone_trace_ao(&grid, Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.0, -1.0, 0.0), &cfg);
        assert!(vis > 0.95, "expected open, got {vis}");
    }

    #[test]
    fn ao_darkens_under_occluder() {
        let grid = ceiling_grid();
        let cfg = ConeConfig {
            max_distance: 3.0,
            radius_scale: 0.6,
            ..ConeConfig::default()
        };
        // Just below the ceiling sphere, hemisphere facing up into it.
        let vis = cone_trace_ao(&grid, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), &cfg);
        assert!(vis < 0.5, "expected occluded, got {vis}");
    }

    #[test]
    fn gi_gather_open_returns_sky() {
        let grid = sphere_grid();
        let cfg = ConeConfig {
            max_distance: 1.0,
            ..ConeConfig::default()
        };
        let sky = Vec3::new(0.2, 0.4, 0.8);
        // Open region facing away from the sphere; every cone should reach the
        // constant sky, so the gather returns (approximately) that colour.
        let gi = cone_gather_gi(
            &grid,
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            &cfg,
            |_p| sky,
        );
        assert!((gi - sky).length() < 0.05, "got {gi:?}");
    }

    #[test]
    fn gi_gather_blocked_is_darker_than_open() {
        let grid = ceiling_grid();
        let cfg = ConeConfig {
            max_distance: 3.0,
            radius_scale: 0.6,
            ..ConeConfig::default()
        };
        let sky = Vec3::splat(1.0);
        // The occluder is black (radiance 0 where distance is small); facing up
        // into it must gather less than facing down into open sky.
        let radiance = |p: Vec3| {
            if p.y > 0.8 {
                Vec3::ZERO
            } else {
                sky
            }
        };
        let up = cone_gather_gi(&grid, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), &cfg, radiance);
        let down = cone_gather_gi(&grid, Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), &cfg, radiance);
        assert!(up.length() < down.length(), "up {up:?} down {down:?}");
    }

    #[test]
    fn gi_gather_sanitizes_bad_radiance() {
        let grid = sphere_grid();
        let cfg = ConeConfig::default();
        let gi = cone_gather_gi(
            &grid,
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            &cfg,
            |_p| Vec3::new(f32::NAN, -5.0, f32::INFINITY),
        );
        assert!(gi.is_finite());
        assert!(gi.x >= 0.0 && gi.y >= 0.0 && gi.z >= 0.0);
    }

    #[test]
    fn tracing_is_deterministic() {
        let grid = ceiling_grid();
        let cfg = ConeConfig::default();
        let a = cone_trace_ao(&grid, Vec3::ZERO, Vec3::Y, &cfg);
        let b = cone_trace_ao(&grid, Vec3::ZERO, Vec3::Y, &cfg);
        assert_eq!(a, b);
    }
}
