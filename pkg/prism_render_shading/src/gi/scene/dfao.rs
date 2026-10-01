//! Distance-field ambient occlusion by cone tracing an [`MeshSdf`] — CPU
//! golden.
//!
//! Unreal Engine's *Distance Field Ambient Occlusion* approximates how much of
//! the hemisphere above a surface is blocked by nearby geometry without
//! shooting individual rays.  Instead it traces a handful of *cones* over the
//! cosine-weighted hemisphere and, at each step along a cone, reads the mesh
//! distance field.  The ratio of the sampled distance to the cone's radius at
//! that step is the fraction of the cone that is still unobstructed; the
//! minimum of that ratio along the cone is the cone's visibility.  Averaging
//! the per-cone visibilities with cosine weights yields a single hemispherical
//! visibility factor.  This module is the backend-neutral reference for that
//! computation, built on [`super::mesh_sdf::MeshSdf`].
//!
//! # Conventions
//! * **Return value.** [`cone_trace_ao`] returns *visibility* in `[0, 1]`:
//!   `1.0` is fully open (no occlusion), `0.0` is fully occluded.  Multiply
//!   albedo/irradiance by it directly.
//! * **Cone visibility.** Along a cone the running visibility is
//!   `min_t clamp(distance(t) / (radius_scale * t), 0, 1)` — the UE DFAO
//!   approximation, where `radius_scale * t` is the cone radius at march
//!   distance `t`.  The closest approach (smallest ratio) dominates, so a
//!   single near occluder darkens the whole cone.
//! * **Hemisphere sampling.** Cone directions are a deterministic Fibonacci
//!   hemisphere lifted into the surface frame built from `normal`, and each
//!   cone is weighted by its cosine with the normal (`local.z`).  The weighted
//!   average is the hemispherical visibility.  The sampling is fixed and
//!   seedless, so results are reproducible bit-for-bit.
//! * **Self-occlusion bias.** Tracing starts at `origin + normal * bias` so the
//!   surface the point sits on does not occlude itself.  Steps are floored to a
//!   minimum so a near-zero distance in open space cannot stall the march.
//! * **Shaping.** `intensity` scales the occlusion amount (`0` disables AO,
//!   returning `1`), and `power` applies a final exponent for artistic
//!   contrast.  Both are applied after the raw visibility is computed and the
//!   result is clamped to `[0, 1]`.
//! * **Determinism / safety.** Pure function, no RNG/IO/GPU/global state, no
//!   `unsafe`; every divisor is guarded and the output can never be `NaN`.

use bevy_math::{ops, Vec3};

use super::mesh_sdf::MeshSdf;

/// Golden-ratio conjugate used to spread Fibonacci-hemisphere azimuths.
const GOLDEN_ANGLE_FRAC: f32 = 0.618_033_99;

/// Marching-step budget per cone.  Enough to resolve a near occluder within the
/// trace distance while keeping the reference cheap and bounded.
const MAX_STEPS_PER_CONE: u32 = 64;

/// Tuning parameters for [`cone_trace_ao`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AoConfig {
    /// Number of cones traced over the hemisphere.  Clamped to at least `1`
    /// (a single cone traces straight along the normal).
    pub cone_count: u32,
    /// Maximum world-space distance marched along each cone.  Beyond this the
    /// cone is considered unobstructed.
    pub max_distance: f32,
    /// Cone radius growth per unit distance: the cone radius at march distance
    /// `t` is `radius_scale * t`.  Larger values widen the cones and gather
    /// occlusion from farther off-axis geometry.
    pub radius_scale: f32,
    /// Scales the final occlusion amount.  `0.0` disables AO (returns `1.0`);
    /// `1.0` applies it at full strength.  Clamped to be non-negative.
    pub intensity: f32,
    /// Exponent applied to the final visibility for contrast shaping.  Clamped
    /// to be at least a tiny positive value so it never inverts or divides.
    pub power: f32,
    /// World-space offset along the normal where tracing begins, preventing the
    /// surface from occluding itself.  Clamped to be non-negative.
    pub bias: f32,
}

impl Default for AoConfig {
    /// A balanced default: 9 cones, a one-unit trace, moderately wide cones and
    /// no artistic shaping.  These are reasonable starting values, not tuned to
    /// any particular scene scale.
    fn default() -> Self {
        Self {
            cone_count: 9,
            max_distance: 1.0,
            radius_scale: 0.5,
            intensity: 1.0,
            power: 1.0,
            bias: 0.02,
        }
    }
}

/// Normalises `v`, falling back to `+Y` for a degenerate (near-zero) input so
/// the surface frame and cone directions never contain `NaN`.
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
/// `normal` using Duff et al. (2017) "Building an Orthonormal Basis, Revisited".
///
/// The branchless construction is numerically stable across the whole sphere,
/// including near the poles where a naive `cross` with a fixed axis degenerates.
#[inline]
fn orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    let sign = if normal.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = Vec3::new(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);
    (tangent, bitangent)
}

/// Visibility of one cone: marches the field and tracks the minimum
/// `distance / cone_radius` ratio (the closest the cone comes to an occluder).
///
/// Returns `1.0` for a fully open cone and `0.0` when the cone penetrates
/// geometry (negative sampled distance) or grazes it tightly.
#[inline]
fn trace_cone(sdf: &MeshSdf, start: Vec3, dir: Vec3, cfg: &AoConfig) -> f32 {
    let max_distance = cfg.max_distance.max(0.0);
    let radius_scale = cfg.radius_scale.max(f32::MIN_POSITIVE);
    // Begin a touch off the surface so t = 0 does not divide by a zero radius.
    let mut t = (max_distance * 1.0e-3).max(1.0e-4);
    let min_step = (max_distance / MAX_STEPS_PER_CONE as f32).max(1.0e-4);
    let mut visibility = 1.0f32;

    for _ in 0..MAX_STEPS_PER_CONE {
        if t > max_distance {
            break;
        }
        let d = sdf.sample_distance(start + dir * t);
        if d <= 0.0 {
            // Inside geometry: the cone is fully blocked here.
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
        // Sphere-trace style advance, floored so open space still terminates.
        t += d.max(min_step);
    }
    visibility.clamp(0.0, 1.0)
}

/// Cone-traces distance-field ambient occlusion at a surface point.
///
/// `origin` is the shading position, `normal` its surface normal (normalised
/// internally, with a `+Y` fallback for a degenerate input), and `cfg` the
/// tuning parameters.  Returns hemispherical *visibility* in `[0, 1]`, where
/// `1.0` means fully unoccluded.  See the module docs for the exact cone
/// visibility and weighting conventions.
///
/// The computation is fully deterministic and defensively clamped: with no
/// nearby geometry it returns `1.0`, every divisor is guarded, and the result
/// is never `NaN`.
pub fn cone_trace_ao(sdf: &MeshSdf, origin: Vec3, normal: Vec3, cfg: &AoConfig) -> f32 {
    let normal = normalize_or_up(normal);
    let cone_count = cfg.cone_count.max(1);
    let bias = cfg.bias.max(0.0);
    let start = origin + normal * bias;
    let (tangent, bitangent) = orthonormal_basis(normal);

    let mut weighted_visibility = 0.0f32;
    let mut total_weight = 0.0f32;
    let n = cone_count as f32;

    for i in 0..cone_count {
        let fi = i as f32 + 0.5;
        // Fibonacci hemisphere: cos(theta) walks from near 1 (around the
        // normal) down toward 0 (the horizon); azimuth spins by the golden
        // angle for an even, low-discrepancy spread.
        let cos_theta = (1.0 - fi / n).clamp(0.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = core::f32::consts::TAU * (fi * GOLDEN_ANGLE_FRAC);
        let (sin_phi, cos_phi) = (ops::sin(phi), ops::cos(phi));

        let local = Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        let dir = tangent * local.x + bitangent * local.y + normal * local.z;
        let dir = normalize_or_up(dir);

        // Cosine weight == local.z == dot(normal, dir); zero-weight cones (on
        // the horizon) contribute nothing.
        let weight = cos_theta;
        if weight <= 0.0 {
            continue;
        }
        let vis = trace_cone(sdf, start, dir, cfg);
        weighted_visibility += vis * weight;
        total_weight += weight;
    }

    let visibility = if total_weight > f32::MIN_POSITIVE {
        weighted_visibility / total_weight
    } else {
        1.0
    };

    // Shape: scale the occlusion amount by intensity, then apply the exponent.
    let intensity = cfg.intensity.max(0.0);
    let occlusion = (1.0 - visibility) * intensity;
    let shaped = (1.0 - occlusion).clamp(0.0, 1.0);
    let power = cfg.power.max(f32::MIN_POSITIVE);
    ops::powf(shaped, power).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::UVec3;

    /// A field whose only geometry is far above the probe region, with enough
    /// padding that the origin sits inside the lattice in genuinely open space
    /// (every sampled distance there is large), so the hemisphere reads open.
    fn distant_sphere() -> MeshSdf {
        MeshSdf::from_sphere(UVec3::splat(32), Vec3::new(0.0, 100.0, 0.0), 1.0, 120.0)
    }

    /// A big occluder sitting directly above the origin.
    fn ceiling_sphere() -> MeshSdf {
        MeshSdf::from_sphere(UVec3::splat(48), Vec3::new(0.0, 2.0, 0.0), 1.5, 3.0)
    }

    #[test]
    fn open_space_is_unoccluded() {
        let sdf = distant_sphere();
        let cfg = AoConfig {
            max_distance: 5.0,
            ..AoConfig::default()
        };
        let ao = cone_trace_ao(&sdf, Vec3::ZERO, Vec3::Y, &cfg);
        assert!(ao > 0.98, "open hemisphere should be ~1, got {ao}");
    }

    #[test]
    fn facing_occluder_reduces_visibility() {
        let open = {
            let sdf = distant_sphere();
            let cfg = AoConfig {
                max_distance: 5.0,
                ..AoConfig::default()
            };
            cone_trace_ao(&sdf, Vec3::ZERO, Vec3::Y, &cfg)
        };
        let blocked = {
            let sdf = ceiling_sphere();
            let cfg = AoConfig {
                max_distance: 5.0,
                ..AoConfig::default()
            };
            cone_trace_ao(&sdf, Vec3::ZERO, Vec3::Y, &cfg)
        };
        assert!(
            blocked < open - 0.2,
            "occluder should darken AO: blocked {blocked} vs open {open}"
        );
        assert!((0.0..=1.0).contains(&blocked));
    }

    #[test]
    fn ao_is_always_in_unit_range() {
        let sdf = ceiling_sphere();
        let cfg = AoConfig {
            max_distance: 5.0,
            ..AoConfig::default()
        };
        for p in [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.3),
            Vec3::new(0.0, 0.4, 0.0),
            Vec3::new(-0.7, 0.1, 0.2),
        ] {
            for n in [Vec3::Y, Vec3::X, Vec3::new(0.3, 0.9, 0.1)] {
                let ao = cone_trace_ao(&sdf, p, n, &cfg);
                assert!((0.0..=1.0).contains(&ao), "{p:?} {n:?} -> {ao}");
                assert!(ao.is_finite());
            }
        }
    }

    #[test]
    fn zero_intensity_disables_occlusion() {
        let sdf = ceiling_sphere();
        let cfg = AoConfig {
            max_distance: 5.0,
            intensity: 0.0,
            ..AoConfig::default()
        };
        let ao = cone_trace_ao(&sdf, Vec3::ZERO, Vec3::Y, &cfg);
        assert!((ao - 1.0).abs() < 1.0e-6, "intensity 0 -> no AO, got {ao}");
    }

    #[test]
    fn higher_intensity_darkens() {
        let sdf = ceiling_sphere();
        let low = cone_trace_ao(
            &sdf,
            Vec3::ZERO,
            Vec3::Y,
            &AoConfig {
                max_distance: 5.0,
                intensity: 0.5,
                ..AoConfig::default()
            },
        );
        let high = cone_trace_ao(
            &sdf,
            Vec3::ZERO,
            Vec3::Y,
            &AoConfig {
                max_distance: 5.0,
                intensity: 1.0,
                ..AoConfig::default()
            },
        );
        assert!(high <= low + 1.0e-6, "more intensity is not brighter");
        assert!(high < low, "more intensity should darken: {high} vs {low}");
    }

    #[test]
    fn ao_is_deterministic() {
        let sdf = ceiling_sphere();
        let cfg = AoConfig {
            max_distance: 5.0,
            ..AoConfig::default()
        };
        let a = cone_trace_ao(&sdf, Vec3::new(0.1, 0.0, -0.2), Vec3::Y, &cfg);
        let b = cone_trace_ao(&sdf, Vec3::new(0.1, 0.0, -0.2), Vec3::Y, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn degenerate_config_is_safe() {
        let sdf = ceiling_sphere();
        // Zero cones, zero distance, zero radius, negative power: all guarded.
        let cfg = AoConfig {
            cone_count: 0,
            max_distance: 0.0,
            radius_scale: 0.0,
            intensity: 1.0,
            power: -2.0,
            bias: -1.0,
        };
        let ao = cone_trace_ao(&sdf, Vec3::ZERO, Vec3::ZERO, &cfg);
        assert!(ao.is_finite());
        assert!((0.0..=1.0).contains(&ao));
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        for n in [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::X,
            Vec3::new(0.3, 0.9, 0.1).normalize(),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let (t, b) = orthonormal_basis(n);
            assert!((t.length() - 1.0).abs() < 1.0e-4, "t unit for {n:?}");
            assert!((b.length() - 1.0).abs() < 1.0e-4, "b unit for {n:?}");
            assert!(t.dot(n).abs() < 1.0e-4, "t _|_ n for {n:?}");
            assert!(b.dot(n).abs() < 1.0e-4, "b _|_ n for {n:?}");
            assert!(t.dot(b).abs() < 1.0e-4, "t _|_ b for {n:?}");
        }
    }
}
