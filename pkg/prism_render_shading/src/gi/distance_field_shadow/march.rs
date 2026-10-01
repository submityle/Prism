//! Distance-field soft-shadow ray marching — CPU golden.
//!
//! Given a scene expressed as a single signed-distance function `d(p)`, the
//! visibility of a surface point towards a (small) light along a direction
//! `dir` can be estimated by marching a ray from the receiver towards the
//! light.  At each step the ray has travelled a distance `t` from the origin
//! and the nearest surface is `h = d(origin + t·dir)` away.  The cone of
//! "almost occluded" directions subtended at the receiver has half-angle
//! `≈ atan(h / t)`, so the ratio
//!
//! ```text
//! penumbra = k · h / t
//! ```
//!
//! (with `k` the inverse tangent of the light's angular radius — large `k`
//! for a sharp point light, small `k` for a broad area light) measures how
//! close the ray grazes the geometry in units of the light's apparent size.
//! Taking the running minimum of this ratio over the whole march yields a soft
//! shadow factor in `[0, 1]`: `1` is fully lit, `0` fully shadowed, and
//! intermediate values trace out the penumbra.  This is Inigo Quilez's
//! classic soft-shadow trick, and this module is its backend-neutral,
//! GPU-free reference.
//!
//! Two estimators are provided:
//!
//! * [`soft_shadow`] — the original 2010 form `res = min(res, k·h/t)`.  Simple
//!   and robust, but can show faint banding where consecutive samples disagree
//!   about the closest silhouette.
//! * [`soft_shadow_improved`] — Quilez's 2010 refinement that removes the
//!   banding by accounting for the previous step: with the previous nearest
//!   distance `prev`, `y = h² / (2·prev)` corrects for the sample spacing and
//!   `d = sqrt(h² − y²)` is the true perpendicular clearance, giving
//!   `res = min(res, k·d / max(0, t − y))`.
//!
//! Both march with step length equal to the sampled distance (sphere
//! marching), clamped to a minimum step so a near-surface graze cannot stall
//! the loop, and both terminate early on a confirmed hit (`h < eps`).
//!
//! # Conventions
//! * **Scene closure.** The scene is any `Fn(Vec3) -> f32` returning the
//!   signed distance to the nearest surface at a world point; the marcher is
//!   generic over it so a caller can pass a primitive-slice evaluator, a baked
//!   grid sampler, or a closed-form test field without allocation.
//! * **Parameters.** `t_min` offsets the start off the surface to avoid self
//!   shadowing; `t_max` bounds the ray (e.g. the distance to the light).  Both
//!   are sanitised: a non-finite or negative `t_min` becomes a tiny positive
//!   bias, and `t_max` is forced to exceed `t_min`.
//! * **Safety.** `dir` is renormalised (a degenerate direction reports fully
//!   lit); `k` is clamped non-negative; `t` is guarded against division by
//!   zero; a bounded iteration cap and a minimum step guarantee termination.
//!   Every returned factor is finite and clamped to `[0, 1]`.
//! * **Transcendental math.** None is required here — only the inherent
//!   `f32::sqrt` method and `Vec3` algebra from `bevy_math`.
//! * **No qualified `Vec`/`alloc`.** Nothing is stored in a heap collection.
//!
//! # References
//! * Inigo Quilez, "soft shadows in raymarched SDFs" (2010),
//!   <https://iquilezles.org/articles/rmshadows/>.

use bevy_math::Vec3;

/// Hard cap on marching iterations, a backstop against a pathological scene
/// closure that never advances past the minimum step.
const MAX_STEPS: u32 = 256;

/// Smallest advance per step, as a safety floor so a ray grazing a surface
/// cannot stall; expressed in world units.
const MIN_STEP: f32 = 1.0e-4;

/// Surface-hit epsilon: once the nearest distance drops below this the ray is
/// considered fully occluded.
const HIT_EPS: f32 = 1.0e-4;

/// Sanitise the ray interval `[t_min, t_max]`, returning a finite pair with
/// `t_min` a small positive bias and `t_max` strictly greater.
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

/// Normalise `dir`, returning `None` for a degenerate direction so the caller
/// can report "fully lit" (an unoccludable ray).
#[inline]
fn safe_dir(dir: Vec3) -> Option<Vec3> {
    let n = dir.normalize_or_zero();
    if n.length_squared() > 0.5 {
        Some(n)
    } else {
        None
    }
}

/// Original Quilez soft-shadow estimator: `res = min(res, k·h/t)`.
///
/// Marches from `origin + t_min·dir` to `t_max` along the normalised `dir`,
/// sampling the scene signed distance `scene_sdf`.  Returns `0.0` on a
/// confirmed hit (`h < eps`) and otherwise the running-minimum penumbra ratio
/// clamped to `[0, 1]` when the ray reaches `t_max`.  `k` controls penumbra
/// sharpness (larger = sharper) and is clamped non-negative; a degenerate
/// direction reports `1.0` (fully lit).
#[inline]
pub fn soft_shadow<F>(origin: Vec3, dir: Vec3, t_min: f32, t_max: f32, k: f32, scene_sdf: F) -> f32
where
    F: Fn(Vec3) -> f32,
{
    let Some(dir) = safe_dir(dir) else {
        return 1.0;
    };
    let (t_min, t_max) = sanitise_interval(t_min, t_max);
    let k = k.max(0.0);

    let mut res = 1.0_f32;
    let mut t = t_min;
    for _ in 0..MAX_STEPS {
        let h = scene_sdf(origin + dir * t);
        if !h.is_finite() {
            // Degenerate sample: stop and report what we have so far.
            break;
        }
        if h < HIT_EPS {
            return 0.0;
        }
        // t >= t_min >= MIN_STEP > 0, so the divisor is always safe.
        res = res.min(k * h / t);
        t += h.max(MIN_STEP);
        if t >= t_max {
            break;
        }
    }
    res.clamp(0.0, 1.0)
}

/// Improved Quilez soft-shadow estimator that removes banding.
///
/// Identical marching to [`soft_shadow`] but the penumbra ratio accounts for
/// the previous nearest distance `prev`: `y = h²/(2·prev)` estimates how far
/// back along the ray the true closest approach lies, `d = sqrt(max(0, h²−y²))`
/// is the corrected perpendicular clearance, and the ratio uses the corrected
/// travel `max(ε, t − y)`.  Returns `0.0` on a hit and the clamped
/// running-minimum otherwise.
#[inline]
pub fn soft_shadow_improved<F>(
    origin: Vec3,
    dir: Vec3,
    t_min: f32,
    t_max: f32,
    k: f32,
    scene_sdf: F,
) -> f32
where
    F: Fn(Vec3) -> f32,
{
    let Some(dir) = safe_dir(dir) else {
        return 1.0;
    };
    let (t_min, t_max) = sanitise_interval(t_min, t_max);
    let k = k.max(0.0);

    let mut res = 1.0_f32;
    let mut t = t_min;
    // A large sentinel so the first step reduces to the plain estimator.
    let mut prev = 1.0e20_f32;
    for _ in 0..MAX_STEPS {
        let h = scene_sdf(origin + dir * t);
        if !h.is_finite() {
            break;
        }
        if h < HIT_EPS {
            return 0.0;
        }
        // Correct for the finite sample spacing (guard the divisor).
        let y = h * h / (2.0 * prev.max(MIN_STEP));
        let d = (h * h - y * y).max(0.0).sqrt();
        let travel = (t - y).max(MIN_STEP);
        res = res.min(k * d / travel);
        prev = h;
        t += h.max(MIN_STEP);
        if t >= t_max {
            break;
        }
    }
    res.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::super::primitives::sphere;
    use super::*;

    /// A unit sphere at `center`, as a scene closure.
    fn sphere_scene(center: Vec3, radius: f32) -> impl Fn(Vec3) -> f32 {
        move |p: Vec3| sphere(p - center, radius)
    }

    #[test]
    fn unobstructed_ray_is_fully_lit() {
        // Sphere far off to the side never enters the ray's path upward.
        let scene = sphere_scene(Vec3::new(100.0, 0.0, 0.0), 1.0);
        let s = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 10.0, 8.0, &scene);
        assert!((s - 1.0).abs() < 1.0e-4, "expected lit, got {s}");
    }

    #[test]
    fn blocker_on_the_ray_fully_shadows() {
        // Big sphere directly overhead: the ray marches straight into it.
        let scene = sphere_scene(Vec3::new(0.0, 5.0, 0.0), 1.0);
        let s = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 8.0, &scene);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn grazing_ray_is_partially_lit() {
        // Sphere offset so the ray grazes its silhouette: expect a penumbra.
        let scene = sphere_scene(Vec3::new(1.2, 5.0, 0.0), 1.0);
        let s = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 8.0, &scene);
        assert!(s > 0.0 && s < 1.0, "expected penumbra, got {s}");
    }

    #[test]
    fn sharper_k_darkens_penumbra() {
        let scene = sphere_scene(Vec3::new(1.2, 5.0, 0.0), 1.0);
        let soft = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 2.0, &scene);
        let sharp = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 32.0, &scene);
        // Larger k => the same clearance maps to a larger (lighter) ratio.
        assert!(sharp >= soft);
    }

    #[test]
    fn improved_agrees_with_plain_on_clear_cases() {
        let lit = sphere_scene(Vec3::new(100.0, 0.0, 0.0), 1.0);
        assert!((soft_shadow_improved(Vec3::ZERO, Vec3::Y, 0.01, 10.0, 8.0, &lit) - 1.0).abs() < 1.0e-4);
        let blocked = sphere_scene(Vec3::new(0.0, 5.0, 0.0), 1.0);
        assert_eq!(soft_shadow_improved(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 8.0, &blocked), 0.0);
    }

    #[test]
    fn improved_also_produces_penumbra() {
        let scene = sphere_scene(Vec3::new(1.2, 5.0, 0.0), 1.0);
        let s = soft_shadow_improved(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 8.0, &scene);
        assert!(s > 0.0 && s < 1.0, "expected penumbra, got {s}");
    }

    #[test]
    fn degenerate_direction_is_lit() {
        let scene = sphere_scene(Vec3::new(0.0, 5.0, 0.0), 1.0);
        assert_eq!(soft_shadow(Vec3::ZERO, Vec3::ZERO, 0.01, 20.0, 8.0, &scene), 1.0);
        assert_eq!(
            soft_shadow_improved(Vec3::ZERO, Vec3::ZERO, 0.01, 20.0, 8.0, &scene),
            1.0
        );
    }

    #[test]
    fn non_finite_scene_stays_finite() {
        let s = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, 8.0, |_| f32::NAN);
        assert!(s.is_finite());
    }

    #[test]
    fn bad_interval_is_sanitised() {
        let scene = sphere_scene(Vec3::new(100.0, 0.0, 0.0), 1.0);
        // t_max < t_min and a negative t_min: must not loop or NaN.
        let s = soft_shadow(Vec3::ZERO, Vec3::Y, -5.0, -10.0, 8.0, &scene);
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }

    #[test]
    fn result_always_in_unit_range() {
        let scene = sphere_scene(Vec3::new(0.5, 3.0, 0.0), 1.0);
        for k in [0.0_f32, 1.0, 10.0, 1000.0] {
            let s = soft_shadow(Vec3::ZERO, Vec3::Y, 0.01, 20.0, k, &scene);
            assert!((0.0..=1.0).contains(&s), "k={k} -> {s}");
        }
    }
}
