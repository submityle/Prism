//! Hard-shadow / visibility ray marching and light-penumbra parameters —
//! CPU golden.
//!
//! Where [`super::march`] estimates a *soft* shadow factor from how closely a
//! ray grazes the scene, this module supplies the companion queries that do
//! not need penumbra estimation:
//!
//! * [`hard_shadow`] — a binary occlusion test.  March the ray from the
//!   receiver towards the light; return `0.0` the instant the signed distance
//!   drops below the surface epsilon (occluded) and `1.0` if the ray clears
//!   `t_max` without a hit (lit).  This is the `k = ∞` limit of the soft
//!   marcher.
//! * [`nearest_hit`] — first-hit sphere marching that returns the travel
//!   distance to the surface along `dir` (or `None` within `t_max`), the
//!   primitive query a contact-shadow or ambient-occlusion pass is built on.
//! * [`soft_shadow_k`] — convert a light's angular radius into the penumbra
//!   sharpness `k` the soft marcher consumes: `k ≈ 1 / tan(angular_radius)`,
//!   the inverse tangent of the half-angle the light subtends.  A larger disc
//!   (bigger angular radius) gives a smaller `k` and therefore a wider, softer
//!   penumbra; a vanishingly small light gives a very large `k` and a crisp
//!   edge.
//!
//! # Conventions
//! * **Scene closure.** As in [`super::march`], the scene is any
//!   `Fn(Vec3) -> f32` signed-distance evaluator; the marchers are generic
//!   over it so no allocation or trait object is forced on the caller.
//! * **Safety.** Directions are renormalised (degenerate → the ray cannot be
//!   occluded, so [`hard_shadow`] reports lit and [`nearest_hit`] reports
//!   `None`); intervals are sanitised like the soft marcher; a bounded step
//!   count plus a minimum step guarantee termination; non-finite samples abort
//!   the march rather than looping.  Every result is finite.
//! * **Transcendental math.** [`soft_shadow_k`] needs a tangent and routes it
//!   through [`bevy_math::ops`] (`ops::tan`); the angular radius is clamped to
//!   a sensible open interval so the tangent is finite and non-zero and the
//!   reciprocal never divides by zero.  No `f32::tan`-style free function is
//!   used.
//! * **No qualified `Vec`/`alloc`.** Nothing is stored in a heap collection.
//!
//! # References
//! * Inigo Quilez, "soft shadows in raymarched SDFs" (2010),
//!   <https://iquilezles.org/articles/rmshadows/>.
//! * Inigo Quilez, "raymarching distance fields" (2008),
//!   <https://iquilezles.org/articles/raymarchingdf/>.

use bevy_math::{ops, Vec3};

/// Hard cap on marching iterations; backstop against a non-advancing scene.
const MAX_STEPS: u32 = 256;

/// Minimum advance per step so a grazing ray cannot stall the loop.
const MIN_STEP: f32 = 1.0e-4;

/// Surface-hit epsilon: a sample closer than this counts as a hit.
const HIT_EPS: f32 = 1.0e-4;

/// Largest permissible half-angle (just under a right angle) so the penumbra
/// tangent in [`soft_shadow_k`] stays finite.
const MAX_ANGULAR_RADIUS: f32 = 1.553_343_f32; // ≈ 89° in radians.

/// Smallest permissible half-angle so the reciprocal tangent in
/// [`soft_shadow_k`] cannot overflow to infinity.
const MIN_ANGULAR_RADIUS: f32 = 1.0e-4_f32;

/// Sanitise the ray interval `[t_min, t_max]` identically to the soft marcher:
/// a finite positive bias and a strictly greater upper bound.
#[inline]
fn sanitise_interval(t_min: f32, t_max: f32) -> (f32, f32) {
    let lo = if t_min.is_finite() && t_min > MIN_STEP {
        t_min
    } else {
        MIN_STEP
    };
    let hi = if t_max.is_finite() && t_max > lo {
        t_max
    } else {
        lo + 1.0
    };
    (lo, hi)
}

/// Normalise `dir`, returning `None` for a degenerate direction.
#[inline]
fn safe_dir(dir: Vec3) -> Option<Vec3> {
    let n = dir.normalize_or_zero();
    if n.length_squared() > 0.5 {
        Some(n)
    } else {
        None
    }
}

/// Binary occlusion test along `dir` between `t_min` and `t_max`.
///
/// Returns `0.0` as soon as the scene signed distance drops below the surface
/// epsilon (the receiver is occluded from the light) and `1.0` if the ray
/// reaches `t_max` unobstructed.  A degenerate direction reports `1.0` (an
/// unoccludable ray); a non-finite sample aborts the march and reports the
/// current `lit` state so the result is always finite.
#[inline]
pub fn hard_shadow<F>(origin: Vec3, dir: Vec3, t_min: f32, t_max: f32, scene_sdf: F) -> f32
where
    F: Fn(Vec3) -> f32,
{
    match nearest_hit(origin, dir, t_min, t_max, scene_sdf) {
        Some(_) => 0.0,
        None => 1.0,
    }
}

/// First-hit sphere march: the distance travelled along the normalised `dir`
/// until the scene surface is reached, or `None` if the ray clears `t_max`.
///
/// Steps by the sampled signed distance (clamped to a minimum) and stops when
/// the distance falls below the surface epsilon, returning the travel `t` at
/// that sample.  A degenerate direction or a non-finite sample yields `None`.
#[inline]
pub fn nearest_hit<F>(origin: Vec3, dir: Vec3, t_min: f32, t_max: f32, scene_sdf: F) -> Option<f32>
where
    F: Fn(Vec3) -> f32,
{
    let dir = safe_dir(dir)?;
    let (t_min, t_max) = sanitise_interval(t_min, t_max);

    let mut t = t_min;
    for _ in 0..MAX_STEPS {
        let h = scene_sdf(origin + dir * t);
        if !h.is_finite() {
            return None;
        }
        if h < HIT_EPS {
            return Some(t);
        }
        t += h.max(MIN_STEP);
        if t >= t_max {
            return None;
        }
    }
    None
}

/// Convert a light's angular radius (half-angle subtended at the receiver, in
/// radians) into the soft-shadow sharpness `k = 1 / tan(angular_radius)`.
///
/// The angular radius is clamped to `[MIN_ANGULAR_RADIUS, MAX_ANGULAR_RADIUS]`
/// so the tangent is finite and strictly positive and the reciprocal never
/// divides by zero or overflows.  A larger light (bigger angle) yields a
/// smaller `k` (softer penumbra); a near-point light yields a large `k`.
#[inline]
#[must_use]
pub fn soft_shadow_k(light_angular_radius: f32) -> f32 {
    let angle = if light_angular_radius.is_finite() {
        light_angular_radius.clamp(MIN_ANGULAR_RADIUS, MAX_ANGULAR_RADIUS)
    } else {
        MIN_ANGULAR_RADIUS
    };
    let t = ops::tan(angle);
    // `angle >= MIN_ANGULAR_RADIUS > 0` so `t` is strictly positive and finite.
    1.0 / t.max(f32::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::super::primitives::sphere;
    use super::*;

    fn sphere_scene(center: Vec3, radius: f32) -> impl Fn(Vec3) -> f32 {
        move |p: Vec3| sphere(p - center, radius)
    }

    #[test]
    fn hard_shadow_blocks_and_clears() {
        let blocked = sphere_scene(Vec3::new(0.0, 5.0, 0.0), 1.0);
        assert_eq!(hard_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, &blocked), 0.0);
        let clear = sphere_scene(Vec3::new(100.0, 0.0, 0.0), 1.0);
        assert_eq!(hard_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, &clear), 1.0);
    }

    #[test]
    fn nearest_hit_returns_front_face_distance() {
        // Sphere centre at y=5, radius 1 => front face at y=4.
        let scene = sphere_scene(Vec3::new(0.0, 5.0, 0.0), 1.0);
        let t = nearest_hit(Vec3::ZERO, Vec3::Y, 0.01, 20.0, &scene).expect("hit");
        assert!((t - 4.0).abs() < 1.0e-2, "expected ~4, got {t}");
    }

    #[test]
    fn nearest_hit_misses_return_none() {
        let scene = sphere_scene(Vec3::new(100.0, 0.0, 0.0), 1.0);
        assert!(nearest_hit(Vec3::ZERO, Vec3::Y, 0.01, 20.0, &scene).is_none());
    }

    #[test]
    fn degenerate_direction_is_lit_and_misses() {
        let scene = sphere_scene(Vec3::new(0.0, 5.0, 0.0), 1.0);
        assert_eq!(hard_shadow(Vec3::ZERO, Vec3::ZERO, 0.01, 20.0, &scene), 1.0);
        assert!(nearest_hit(Vec3::ZERO, Vec3::ZERO, 0.01, 20.0, &scene).is_none());
    }

    #[test]
    fn non_finite_sample_aborts_cleanly() {
        assert_eq!(hard_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, |_| f32::NAN), 1.0);
        assert!(nearest_hit(Vec3::ZERO, Vec3::Y, 0.01, 20.0, |_| f32::NAN).is_none());
    }

    #[test]
    fn k_decreases_with_light_size() {
        let small = soft_shadow_k(0.01);
        let large = soft_shadow_k(0.3);
        assert!(small > large, "smaller light should be sharper: {small} vs {large}");
        assert!(small.is_finite() && large.is_finite());
    }

    #[test]
    fn k_matches_inverse_tangent() {
        let angle = 0.1_f32;
        let expected = 1.0 / ops::tan(angle);
        assert!((soft_shadow_k(angle) - expected).abs() < 1.0e-3);
    }

    #[test]
    fn k_clamps_degenerate_inputs() {
        assert!(soft_shadow_k(0.0).is_finite());
        assert!(soft_shadow_k(-1.0).is_finite());
        assert!(soft_shadow_k(f32::NAN).is_finite());
        assert!(soft_shadow_k(100.0).is_finite());
        // A near-zero light must still yield a large but finite sharpness.
        assert!(soft_shadow_k(0.0) > 1.0);
    }
}
