//! Specular occlusion and horizon occlusion — CPU golden.
//!
//! A scalar ambient-occlusion (AO) term is derived for the *diffuse* response
//! (the full cosine hemisphere).  Feeding that same term to the specular
//! response double-darkens tight reflections, because a glossy lobe only
//! integrates a narrow cone around the reflection vector, not the whole
//! hemisphere.  Lagarde & de Rousiers' *Moving Frostbite to Physically Based
//! Rendering* (SIGGRAPH 2014) give a cheap analytic remap that recovers a
//! view- and roughness-dependent *specular* occlusion from the diffuse AO, and
//! a separate *horizon occlusion* term that fades reflections which point into
//! already-occluded directions.  This module is the backend-neutral reference
//! for both, plus the cone-overlap primitive they share.
//!
//! * [`specular_occlusion`] is the Lagarde remap
//!   `saturate(pow(NoV + ao, exp2(-16*roughness - 1)) - 1 + ao)`.
//! * [`cone_cone_intersection`] estimates the fraction of one spherical cap
//!   (cone A) that lies inside another (cone B) — the analytic workhorse for
//!   soft cone visibility.
//! * [`horizon_occlusion`] expresses how much of a reflection survives the
//!   bent-normal visibility cone, by treating the mirror reflection as a thin
//!   cone and delegating to [`cone_cone_intersection`].
//!
//! # Conventions
//! * All inputs are defensively clamped: `n_dot_v`, `ao`, and `roughness` to
//!   `[0, 1]`; cone apertures to `[0, PI]` radians (half-angles); directions
//!   are normalised with a `+Y` fallback for the zero vector.
//! * Every returned factor is clamped to `[0, 1]` and is finite for all inputs
//!   (no `NaN`, no division by zero).
//! * `roughness` is perceptual (linear) roughness, matching the Frostbite
//!   remap's `exp2(-16*roughness - 1)` exponent; it is *not* `alpha`.
//! * Every function is a deterministic, allocation-free pure function so the
//!   WESL/GPU twin reproduces it bit-for-bit under the same `f32` arithmetic.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Fraction of the bent-normal cone's half-angle used as the specular lobe's
/// rim half-width in [`horizon_occlusion`].  A wider visibility cone therefore
/// has a correspondingly softer horizon.
const SPECULAR_RIM_FRACTION: f32 = 0.5;

/// Minimum specular rim half-width (radians) so the horizon response stays
/// well-defined even for a razor-thin visibility cone.
const MIN_SPECULAR_RIM: f32 = 0.02;

/// Lagarde's specular-occlusion remap of a diffuse AO term.
///
/// Returns a specular occlusion factor in `[0, 1]` from the view cosine
/// `n_dot_v` (`N·V`), the scalar diffuse ambient occlusion `ao`, and the
/// perceptual `roughness`:
///
/// ```text
/// SO = saturate(pow(NoV + ao, exp2(-16 * roughness - 1)) - 1 + ao)
/// ```
///
/// Behaviour at the limits (which the tests pin down):
/// * `ao = 1` (nothing occluded) saturates to `SO = 1` for every view angle and
///   roughness, so a fully lit surface is never darkened.
/// * As `roughness -> 1` the exponent `exp2(-16*roughness - 1) -> 0`, the `pow`
///   term tends to `1`, and `SO -> ao`: a rough lobe is wide enough that the
///   diffuse AO is already the right answer.
/// * As `roughness -> 0` the exponent tends to `exp2(-1) = 0.5`, giving the
///   view-dependent sharpening of a mirror lobe.
///
/// All three inputs are clamped to `[0, 1]`; the `pow` base is clamped to be
/// non-negative, so the result is always finite.
#[inline]
pub fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    let n_dot_v = n_dot_v.clamp(0.0, 1.0);
    let ao = ao.clamp(0.0, 1.0);
    let roughness = roughness.clamp(0.0, 1.0);

    let exponent = ops::exp2(-16.0 * roughness - 1.0);
    let base = (n_dot_v + ao).max(0.0);
    (ops::powf(base, exponent) - 1.0 + ao).clamp(0.0, 1.0)
}

/// Smooth Hermite interpolation of `t` clamped to `[0, 1]`.
#[inline]
fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Normalises `v`, returning `fallback` for a degenerate (near-zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Estimates the fraction of cone A's solid angle that lies inside cone B.
///
/// `dir_a` / `aperture_a` describe the first spherical cap (half-angle
/// `aperture_a` about `dir_a`) and `dir_b` / `aperture_b` the second.  The
/// result is the (approximate) fraction of cap A that overlaps cap B, in
/// `[0, 1]`:
///
/// * `1.0` when A lies entirely inside B (the angle between the axes is at most
///   `aperture_b - aperture_a`).
/// * The cap-area ratio `(1 - cos aperture_b) / (1 - cos aperture_a)` when B is
///   the *smaller* cone fully inside A, so the fraction of A covered saturates
///   below `1`.
/// * `0.0` when the caps are disjoint (axis angle at least
///   `aperture_a + aperture_b`).
/// * A smooth [`smoothstep01`] ramp between those regimes otherwise.
///
/// This is the standard real-time soft cone-overlap approximation (cf. Oat &
/// Sander, *Ambient Aperture Lighting*): it is exact at the containment and
/// disjoint boundaries and monotonic in between, which is all the horizon
/// term below needs.  Apertures are clamped to `[0, PI]` and directions are
/// normalised, so the result is always finite.
pub fn cone_cone_intersection(
    dir_a: Vec3,
    aperture_a: f32,
    dir_b: Vec3,
    aperture_b: f32,
) -> f32 {
    let a = aperture_a.clamp(0.0, PI);
    let b = aperture_b.clamp(0.0, PI);
    let da = normalize_or(dir_a, Vec3::Y);
    let db = normalize_or(dir_b, Vec3::Y);

    let cos_angle = da.dot(db).clamp(-1.0, 1.0);
    let angle = ops::acos(cos_angle);

    let inner = (a - b).abs(); // containment boundary
    let outer = a + b; // disjoint boundary

    // When B is the smaller cone fully inside A, the overlap saturates at the
    // ratio of cap areas rather than at 1.
    let cap_a = (1.0 - ops::cos(a)).max(0.0);
    let cap_b = (1.0 - ops::cos(b)).max(0.0);
    let contained_fraction = if a <= b {
        1.0
    } else if cap_a > f32::MIN_POSITIVE {
        (cap_b / cap_a).clamp(0.0, 1.0)
    } else {
        0.0
    };

    if angle <= inner {
        return contained_fraction;
    }
    if angle >= outer {
        return 0.0;
    }
    let span = outer - inner;
    if span <= f32::MIN_POSITIVE {
        // Degenerate band (one aperture is zero): hard step at the boundary.
        return 0.0;
    }
    // t = 1 at the containment boundary, 0 at the disjoint boundary.
    let t = ((outer - angle) / span).clamp(0.0, 1.0);
    smoothstep01(t) * contained_fraction
}

/// Horizon occlusion of a reflection against a bent-normal visibility cone.
///
/// Given the mirror `reflection_dir`, the `bent_normal` axis of the visible
/// cone, and that cone's half-angle `cone_aperture`, this returns how much of
/// the reflection survives, in `[0, 1]`:
///
/// * A reflection well inside the cone returns `1.0` (unoccluded).
/// * A reflection outside the cone fades toward `0.0` as the angle between the
///   reflection and the bent normal grows past the cone edge.
/// * A fully open cone (`cone_aperture >= PI`) never occludes (`1.0`); a fully
///   closed cone (`cone_aperture <= 0`) always occludes (`0.0`).
///
/// The reflection is modelled as a thin cone whose rim half-width scales with
/// the visibility aperture (see [`SPECULAR_RIM_FRACTION`]), and the overlap is
/// evaluated through [`cone_cone_intersection`], so the fade is smooth rather
/// than a hard step.  Inputs are clamped and the direction normalised, so the
/// result is always finite.
pub fn horizon_occlusion(reflection_dir: Vec3, bent_normal: Vec3, cone_aperture: f32) -> f32 {
    let aperture = cone_aperture.clamp(0.0, PI);
    // Essentially closed cone: nothing is visible, so the reflection is fully
    // occluded regardless of where it points.
    if aperture <= MIN_SPECULAR_RIM {
        return 0.0;
    }
    // Essentially open cone: the whole sphere is visible, so no occlusion.
    if aperture >= PI - 1.0e-3 {
        return 1.0;
    }
    // Thin specular lobe whose soft rim scales with the visibility cone.
    let rim = (aperture * SPECULAR_RIM_FRACTION).clamp(MIN_SPECULAR_RIM, PI);
    cone_cone_intersection(reflection_dir, rim, bent_normal, aperture)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ao_one_gives_full_specular_occlusion() {
        for &nov in &[0.0, 0.1, 0.5, 0.9, 1.0] {
            for &r in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                let so = specular_occlusion(nov, 1.0, r);
                assert!((so - 1.0).abs() < 1e-6, "nov {nov} r {r} so {so}");
            }
        }
    }

    #[test]
    fn high_roughness_approaches_ao() {
        for &ao in &[0.0, 0.2, 0.5, 0.8, 1.0] {
            for &nov in &[0.0, 0.3, 0.7, 1.0] {
                let so = specular_occlusion(nov, ao, 1.0);
                // exponent = exp2(-17) ~ 7.6e-6, so pow(base, e) ~ 1 -> SO ~ ao.
                assert!((so - ao).abs() < 1e-3, "ao {ao} nov {nov} so {so}");
            }
        }
    }

    #[test]
    fn specular_occlusion_is_bounded_and_finite() {
        // Sweep well outside the valid range to exercise the clamps.
        for i in -2..=12 {
            for j in -2..=12 {
                for k in -2..=12 {
                    let nov = i as f32 * 0.1;
                    let ao = j as f32 * 0.1;
                    let r = k as f32 * 0.1;
                    let so = specular_occlusion(nov, ao, r);
                    assert!(so.is_finite() && (0.0..=1.0).contains(&so), "so {so}");
                }
            }
        }
    }

    #[test]
    fn specular_occlusion_is_monotonic_in_ao() {
        // Darkening the diffuse AO must never brighten the specular occlusion.
        let nov = 0.4;
        let r = 0.3;
        let mut prev = -1.0;
        for i in 0..=10 {
            let ao = i as f32 / 10.0;
            let so = specular_occlusion(nov, ao, r);
            assert!(so + 1e-6 >= prev, "non-monotonic: {prev} -> {so}");
            prev = so;
        }
    }

    #[test]
    fn specular_occlusion_is_deterministic() {
        assert_eq!(
            specular_occlusion(0.37, 0.62, 0.21),
            specular_occlusion(0.37, 0.62, 0.21)
        );
    }

    #[test]
    fn cone_identical_cones_fully_overlap() {
        let v = Vec3::new(0.2, 0.5, 0.8);
        assert!((cone_cone_intersection(v, 0.5, v, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cone_small_inside_large_is_one() {
        // A narrow cone centred inside a wide one is fully contained.
        let v = Vec3::Z;
        let got = cone_cone_intersection(v, 0.1, v, 1.0);
        assert!((got - 1.0).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn cone_large_around_small_saturates_below_one() {
        // B (small) fully inside A (large): fraction of A covered = capB/capA.
        let v = Vec3::Z;
        let a = 1.0f32; // wide query cone
        let b = 0.3f32; // narrow visibility cone
        let got = cone_cone_intersection(v, a, v, b);
        let cap_a = 1.0 - ops::cos(a);
        let cap_b = 1.0 - ops::cos(b);
        let expected = (cap_b / cap_a).clamp(0.0, 1.0);
        assert!((got - expected).abs() < 1e-6, "got {got} expected {expected}");
        assert!(got < 1.0);
    }

    #[test]
    fn cone_disjoint_is_zero() {
        // Opposite axes, apertures too small to meet.
        let got = cone_cone_intersection(Vec3::Z, 0.3, Vec3::NEG_Z, 0.3);
        assert_eq!(got, 0.0);
    }

    #[test]
    fn cone_overlap_decreases_with_separation() {
        // Equal cones pulled apart: overlap must fall monotonically to 0.
        let a = 0.6f32;
        let axis_a = Vec3::Z;
        let mut prev = 2.0f32;
        for i in 0..=24 {
            // angle from 0 to just past 2a (disjoint boundary) via a tilt in xz.
            let angle = (i as f32 / 24.0) * (2.0 * a + 0.2);
            let axis_b = Vec3::new(ops::sin(angle), 0.0, ops::cos(angle));
            let got = cone_cone_intersection(axis_a, a, axis_b, a);
            assert!(got.is_finite() && (0.0..=1.0).contains(&got), "got {got}");
            assert!(got <= prev + 1e-6, "non-monotonic at {angle}: {prev} -> {got}");
            prev = got;
        }
        assert_eq!(prev, 0.0);
    }

    #[test]
    fn cone_intersection_is_bounded_everywhere() {
        let dirs = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::ZERO,
        ];
        for &da in &dirs {
            for &db in &dirs {
                for &aa in &[0.0, 0.3, 1.0, PI] {
                    for &bb in &[0.0, 0.3, 1.0, PI] {
                        let got = cone_cone_intersection(da, aa, db, bb);
                        assert!(got.is_finite() && (0.0..=1.0).contains(&got), "got {got}");
                    }
                }
            }
        }
    }

    #[test]
    fn horizon_reflection_inside_cone_is_unoccluded() {
        // Reflection along the bent normal, moderately open cone.
        let got = horizon_occlusion(Vec3::Z, Vec3::Z, 1.0);
        assert!((got - 1.0).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn horizon_reflection_outside_cone_is_occluded() {
        // Reflection pointing opposite the visible cone.
        let got = horizon_occlusion(Vec3::NEG_Z, Vec3::Z, 0.6);
        assert_eq!(got, 0.0);
    }

    #[test]
    fn horizon_closed_cone_fully_occludes() {
        let got = horizon_occlusion(Vec3::Z, Vec3::Z, 0.0);
        assert_eq!(got, 0.0);
    }

    #[test]
    fn horizon_open_cone_never_occludes() {
        // A whole-sphere cone keeps every reflection fully visible.
        for &refl in &[Vec3::Z, Vec3::X, Vec3::NEG_Z, Vec3::new(0.3, -0.7, 0.2)] {
            let got = horizon_occlusion(refl, Vec3::Z, PI);
            assert!((got - 1.0).abs() < 1e-6, "refl {refl:?} got {got}");
        }
    }

    #[test]
    fn horizon_fades_monotonically_with_angle() {
        let aperture = 1.0f32;
        let bent = Vec3::Z;
        let mut prev = 2.0f32;
        for i in 0..=20 {
            let angle = (i as f32 / 20.0) * PI;
            let refl = Vec3::new(ops::sin(angle), 0.0, ops::cos(angle));
            let got = horizon_occlusion(refl, bent, aperture);
            assert!(got.is_finite() && (0.0..=1.0).contains(&got), "got {got}");
            assert!(got <= prev + 1e-6, "non-monotonic at {angle}: {prev} -> {got}");
            prev = got;
        }
    }

    #[test]
    fn horizon_is_deterministic() {
        let refl = Vec3::new(0.3, 0.4, 0.86);
        let bent = Vec3::new(0.1, 0.2, 0.97);
        assert_eq!(
            horizon_occlusion(refl, bent, 0.7),
            horizon_occlusion(refl, bent, 0.7)
        );
    }
}
