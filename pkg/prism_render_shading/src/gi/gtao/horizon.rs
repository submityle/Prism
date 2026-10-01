//! Single-slice horizon search for ground-truth ambient occlusion (GTAO).
//!
//! GTAO (Jimenez et al., SIGGRAPH 2016) estimates ambient occlusion by cutting
//! the view-space hemisphere above a shading point into azimuthal *slices* and,
//! inside each slice, marching outward along the on-screen projection of the
//! slice direction to find how high nearby geometry raises the *horizon*.  This
//! module is the backend-neutral CPU reference for that per-slice horizon
//! search: given the view-space position of the shading point, the unit view
//! direction, and the view-space positions of the depth-reconstructed samples
//! on each side of the slice, it returns the two horizon angles the visibility
//! integral in [`super::integral`] consumes.
//!
//! The horizon on one side is the direction, among the sampled occluders, that
//! rises *closest to the view ray* — i.e. the one with the largest cosine
//! against the view vector.  A larger cosine means a smaller horizon angle,
//! which blocks more of the sky, so maximising the cosine is the same as
//! finding the most occluding sample.  Two refinements temper the raw maximum:
//!
//! * **Distance falloff.**  Occluders farther than a radius contribute nothing;
//!   closer ones contribute proportionally more.  The candidate cosine is
//!   attenuated toward the unoccluded baseline by a smooth distance weight.
//! * **Thickness heuristic.**  A strict running maximum treats every occluder
//!   as infinitely thick, so a thin object keeps shadowing long after the ray
//!   has marched past it.  When a later (farther) sample would *lower* the
//!   horizon, the running maximum is relaxed toward it by a `thickness` factor,
//!   letting the horizon "see past" thin surfaces instead of clamping to the
//!   first silhouette.
//!
//! # Conventions
//! * All positions and directions are view-space `Vec3`s (camera at the
//!   origin); the view direction points from the shading point toward the
//!   camera and is renormalised defensively on entry.
//! * Samples are supplied in marching order (nearest first) so the thickness
//!   heuristic sees occluders in the order the GPU twin would.
//! * Returned horizon angles are *magnitudes* in `[0, PI/2]`, measured from the
//!   view direction, one per side.  `PI/2` is the fully unoccluded horizon
//!   (grazing); `0` is a fully occluding sample sitting on the view ray.  The
//!   visibility integral assigns the signs (negative side vs. positive side).
//! * Transcendental math goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method.  Every path clamps defensively and never emits `NaN`.
//! * Deterministic pure functions: no RNG, no I/O, no GPU, no allocation.

use bevy_math::{ops, Vec3};
use core::f32::consts::FRAC_PI_2;

/// Smallest squared length treated as a non-degenerate separation.
///
/// Samples closer than this to the shading point carry no reliable direction
/// and are skipped so the search never divides by zero.
const MIN_SEPARATION_SQ: f32 = 1.0e-12;

/// Tunable parameters for the per-slice horizon search.
///
/// The defaults are a reasonable mid-range configuration: a unit falloff radius
/// and a moderate thickness relaxation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizonParams {
    /// Distance (view-space units) at which an occluder stops contributing.
    ///
    /// Samples at zero distance contribute fully; samples at or beyond the
    /// radius contribute nothing.  A non-finite or non-positive radius disables
    /// falloff (every in-front sample contributes fully), which keeps a
    /// mis-configured radius from silently erasing all occlusion.
    pub falloff_radius: f32,
    /// Thin-surface recovery factor in `[0, 1]`.
    ///
    /// `0` keeps a strict running maximum (occluders are treated as infinitely
    /// thick); `1` lets the horizon fully follow a falling sample (occluders
    /// are treated as infinitely thin).  Intermediate values blend the two.
    pub thickness: f32,
}

impl Default for HorizonParams {
    #[inline]
    fn default() -> Self {
        Self {
            falloff_radius: 1.0,
            thickness: 0.5,
        }
    }
}

impl HorizonParams {
    /// Returns a copy with every field clamped to its valid, finite range.
    ///
    /// `thickness` is clamped to `[0, 1]`; a non-finite `falloff_radius` is
    /// replaced by `0` (falloff disabled).  Used internally so the search body
    /// can assume sane inputs.
    #[inline]
    fn sanitized(self) -> Self {
        let falloff_radius = if self.falloff_radius.is_finite() && self.falloff_radius > 0.0 {
            self.falloff_radius
        } else {
            0.0
        };
        let thickness = if self.thickness.is_finite() {
            self.thickness.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            falloff_radius,
            thickness,
        }
    }
}

/// Normalises `v`, returning `None` for a degenerate (near-zero) input.
#[inline]
fn try_normalize(v: Vec3) -> Option<Vec3> {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > MIN_SEPARATION_SQ {
        Some(v * len_sq.sqrt().recip())
    } else {
        None
    }
}

/// Smooth distance weight in `[0, 1]`: `1` at zero distance, `0` at/after the
/// radius.
///
/// Uses the quadratic `1 - (dist / radius)^2` so the attenuation is gentle near
/// the shading point and flattens into zero contribution past the radius.  A
/// disabled radius (`<= 0`) returns `1`, i.e. no attenuation.
#[inline]
fn falloff_weight(dist: f32, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 1.0;
    }
    let t = dist / radius;
    let w = 1.0 - t * t;
    w.clamp(0.0, 1.0)
}

/// Searches one side of a slice and returns its horizon magnitude in
/// `[0, PI/2]`.
///
/// `position` and `view` are the (already normalised) shading point and view
/// direction; `samples` are the view-space occluder positions on this side in
/// marching order.  The running maximum of the (falloff-weighted) view-cosine
/// is relaxed toward falling samples by `params.thickness`, then converted to
/// an angle via `acos`.
fn side_horizon(position: Vec3, view: Vec3, samples: &[Vec3], params: HorizonParams) -> f32 {
    // Baseline cosine `0` corresponds to a horizon at `PI/2`: fully unoccluded.
    // Only samples in front of the grazing plane (positive cosine) can raise it.
    let mut max_cos = 0.0_f32;
    for &sample in samples {
        let Some(dir) = try_normalize(sample - position) else {
            continue;
        };
        let raw_cos = view.dot(dir);
        if !raw_cos.is_finite() || raw_cos <= 0.0 {
            // At or behind grazing: contributes no occlusion.
            continue;
        }
        let dist = (sample - position).length();
        let weight = falloff_weight(dist, params.falloff_radius);
        // Attenuate the candidate toward the unoccluded baseline (cosine `0`).
        let candidate = (raw_cos * weight).clamp(0.0, 1.0);
        if candidate > max_cos {
            // A higher occluder: raise the horizon immediately.
            max_cos = candidate;
        } else {
            // A lower (farther) occluder: relax toward it by `thickness`, so a
            // thin surface lets the horizon fall back instead of latching.
            max_cos += (candidate - max_cos) * params.thickness;
        }
    }
    let max_cos = max_cos.clamp(0.0, 1.0);
    // acos maps [0, 1] -> [PI/2, 0]; clamp defends against round-off overshoot.
    ops::acos(max_cos).clamp(0.0, FRAC_PI_2)
}

/// Searches both sides of one slice and returns the horizon magnitudes
/// `(h1, h2)`.
///
/// `h1` is the horizon on the negative-tangent side, `h2` on the
/// positive-tangent side; both are magnitudes in `[0, PI/2]` measured from the
/// view direction (see the module conventions).  `neg_side` and `pos_side`
/// hold the view-space occluder positions on each side in marching order.  A
/// degenerate view direction (non-normalisable) yields the fully unoccluded
/// pair `(PI/2, PI/2)`.
pub fn search_horizons(
    position: Vec3,
    view: Vec3,
    neg_side: &[Vec3],
    pos_side: &[Vec3],
    params: HorizonParams,
) -> (f32, f32) {
    let params = params.sanitized();
    let Some(view) = try_normalize(view) else {
        return (FRAC_PI_2, FRAC_PI_2);
    };
    if !position.is_finite() {
        return (FRAC_PI_2, FRAC_PI_2);
    }
    let h1 = side_horizon(position, view, neg_side, params);
    let h2 = side_horizon(position, view, pos_side, params);
    (h1, h2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn empty_sides_are_unoccluded() {
        let (h1, h2) = search_horizons(Vec3::ZERO, Vec3::Z, &[], &[], HorizonParams::default());
        assert!(approx(h1, FRAC_PI_2, 1.0e-6));
        assert!(approx(h2, FRAC_PI_2, 1.0e-6));
    }

    #[test]
    fn degenerate_view_falls_back_to_unoccluded() {
        let occluder = [Vec3::new(0.0, 0.0, -1.0)];
        let (h1, h2) =
            search_horizons(Vec3::ZERO, Vec3::ZERO, &occluder, &occluder, HorizonParams::default());
        assert_eq!((h1, h2), (FRAC_PI_2, FRAC_PI_2));
    }

    #[test]
    fn occluder_on_view_ray_fully_occludes() {
        // View points +Z (toward camera); an occluder straight along the view
        // ray gives cosine 1 -> horizon 0.
        let view = Vec3::Z;
        let occluder = [Vec3::new(0.0, 0.0, 0.5)];
        // Disable falloff so the on-axis occluder reaches the horizon exactly.
        let params = HorizonParams {
            falloff_radius: -1.0,
            thickness: 0.0,
        };
        let (h1, _) = search_horizons(Vec3::ZERO, view, &occluder, &[], params);
        assert!(approx(h1, 0.0, 1.0e-4), "h1 = {h1}");
    }

    #[test]
    fn grazing_sample_does_not_occlude() {
        // A sample perpendicular to the view direction has cosine 0.
        let view = Vec3::Z;
        let occluder = [Vec3::new(1.0, 0.0, 0.0)];
        let (h1, _) = search_horizons(Vec3::ZERO, view, &occluder, &[], HorizonParams::default());
        assert!(approx(h1, FRAC_PI_2, 1.0e-5));
    }

    #[test]
    fn horizon_is_monotonic_in_occluder_height() {
        // Moving a sample closer to the view ray (raising its cosine) must not
        // increase the horizon magnitude: more occlusion => smaller angle.
        let view = Vec3::Z;
        let params = HorizonParams {
            falloff_radius: 100.0,
            thickness: 0.0,
        };
        let mut prev = f32::INFINITY;
        for k in 0..=8 {
            // Interpolate the sample direction from grazing (+X) toward +Z.
            let t = k as f32 / 8.0;
            let dir = Vec3::new(1.0 - t, 0.0, t).normalize();
            let (h1, _) = search_horizons(Vec3::ZERO, view, &[dir], &[], params);
            assert!(h1 <= prev + 1.0e-6, "not monotonic at k={k}: {h1} > {prev}");
            prev = h1;
        }
    }

    #[test]
    fn falloff_reduces_occlusion_with_distance() {
        let view = Vec3::Z;
        let params = HorizonParams {
            falloff_radius: 2.0,
            thickness: 0.0,
        };
        // Same direction, two distances: the nearer occluder blocks more
        // (smaller horizon) than the farther one.
        let near = [Vec3::new(0.0, 0.3, 0.6)];
        let far = [Vec3::new(0.0, 0.9, 1.8)];
        let (h_near, _) = search_horizons(Vec3::ZERO, view, &near, &[], params);
        let (h_far, _) = search_horizons(Vec3::ZERO, view, &far, &[], params);
        assert!(h_near < h_far, "near {h_near} should occlude more than far {h_far}");
    }

    #[test]
    fn sample_beyond_radius_is_ignored() {
        let view = Vec3::Z;
        let params = HorizonParams {
            falloff_radius: 1.0,
            thickness: 0.0,
        };
        let far = [Vec3::new(0.0, 0.0, 5.0)];
        let (h1, _) = search_horizons(Vec3::ZERO, view, &far, &[], params);
        assert!(approx(h1, FRAC_PI_2, 1.0e-5), "far sample should not occlude: {h1}");
    }

    #[test]
    fn thickness_lets_horizon_fall_past_thin_occluder() {
        let view = Vec3::Z;
        // First (near) sample raises the horizon high; the second (farther)
        // sample sits lower.  With thickness 1 the horizon follows it back
        // down (less occlusion => larger angle) than with thickness 0.
        let samples = [Vec3::new(0.0, 0.2, 0.8), Vec3::new(0.0, 0.9, 0.9)];
        let p = HorizonParams {
            falloff_radius: 100.0,
            thickness: 0.0,
        };
        let p_thin = HorizonParams {
            falloff_radius: 100.0,
            thickness: 1.0,
        };
        let (h_thick, _) = search_horizons(Vec3::ZERO, view, &samples, &[], p);
        let (h_thin, _) = search_horizons(Vec3::ZERO, view, &samples, &[], p_thin);
        assert!(h_thin > h_thick, "thin {h_thin} should occlude less than thick {h_thick}");
    }

    #[test]
    fn results_are_finite_and_in_range() {
        let view = Vec3::new(0.1, -0.2, 1.0);
        let samples = [
            Vec3::new(0.0, 0.0, 0.0),      // coincident -> skipped
            Vec3::new(1.0e20, 0.0, 1.0e20),
            Vec3::new(-0.3, 0.4, 0.5),
        ];
        let (h1, h2) = search_horizons(Vec3::ZERO, view, &samples, &samples, HorizonParams::default());
        for h in [h1, h2] {
            assert!(h.is_finite());
            assert!((0.0..=FRAC_PI_2 + 1.0e-6).contains(&h));
        }
    }

    #[test]
    fn negative_radius_disables_falloff() {
        let view = Vec3::Z;
        let params = HorizonParams {
            falloff_radius: -1.0,
            thickness: 0.0,
        };
        let far = [Vec3::new(0.0, 0.0, 50.0)];
        let (h1, _) = search_horizons(Vec3::ZERO, view, &far, &[], params);
        // Falloff disabled -> the distant on-axis sample still fully occludes.
        assert!(approx(h1, 0.0, 1.0e-3), "h1 = {h1}");
    }
}
