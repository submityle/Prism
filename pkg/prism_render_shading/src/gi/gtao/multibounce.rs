//! Multi-bounce lightening and AO post-processing for GTAO.
//!
//! A single-scattering ambient-occlusion term darkens creases too aggressively:
//! in reality light bounces between nearby surfaces and partially fills them
//! back in, the more so the brighter (higher-albedo) the surface.  Jimenez et
//! al. (2016) approximate this *multi-bounce* behaviour with a cheap per-channel
//! cubic polynomial fitted to a path-traced ground truth, driven by the surface
//! albedo.  This module is the backend-neutral CPU reference for that fit plus
//! the small post-processing helpers that shape a raw visibility value into the
//! AO term the lighting pass consumes, and the [`GtaoResult`] that bundles AO
//! with the bent normal from [`super::integral`].
//!
//! # Conventions
//! * AO / visibility is a multiplier in `[0, 1]`: `1` is fully lit (unoccluded),
//!   `0` is fully occluded.
//! * Albedo is in `[0, 1]` per channel; higher albedo brightens more.
//! * The multi-bounce fit is `a*x^3 - b*x^2 + c*x` with
//!   `a = 2.0404*albedo - 0.3324`, `b = 4.7951*albedo - 0.6417`,
//!   `c = 2.7552*albedo + 0.6903`, and the result is `max(x, poly)` so the
//!   lightening never darkens below the input AO.
//! * Transcendental math via [`bevy_math::ops`].  Every helper clamps
//!   defensively and never emits `NaN`.
//! * Deterministic pure functions: no RNG, no I/O, no GPU, no allocation.

use bevy_math::{ops, Vec3};

/// Combined GTAO output ready for the lighting pass.
///
/// Bundles the scalar ambient-occlusion / visibility term with the unit bent
/// normal so a shader can both scale ambient light and bias its ambient /
/// indirect lookups toward the unoccluded direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GtaoResult {
    /// Scalar ambient-occlusion / visibility multiplier in `[0, 1]`.
    pub visibility: f32,
    /// Unit-length bent normal (mean unoccluded direction).
    pub bent_normal: Vec3,
}

impl GtaoResult {
    /// A fully lit result with the bent normal pointing along `+Y`.
    pub const UNOCCLUDED: Self = Self {
        visibility: 1.0,
        bent_normal: Vec3::Y,
    };
}

/// Clamps a raw integrated visibility into a usable AO multiplier in `[0, 1]`.
///
/// Non-finite inputs collapse to `0` (treated as fully occluded) so a bad
/// upstream value darkens rather than propagating a `NaN`.
#[inline]
pub fn visibility_to_ao(visibility: f32) -> f32 {
    if visibility.is_finite() {
        visibility.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Evaluates the single-channel GTAO multi-bounce fit for albedo `albedo`.
///
/// Returns a lightened AO value `>= ao`: the brighter the albedo, the more the
/// occlusion is filled back in.  `ao` and `albedo` are clamped to `[0, 1]`
/// first, so the output is always finite and in `[0, 1]`.
#[inline]
pub fn multi_bounce(ao: f32, albedo: f32) -> f32 {
    let x = visibility_to_ao(ao);
    let albedo = if albedo.is_finite() {
        albedo.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let a = 2.0404 * albedo - 0.3324;
    let b = 4.7951 * albedo - 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    // Horner form of `a*x^3 - b*x^2 + c*x`.
    let poly = ((a * x - b) * x + c) * x;
    let lit = x.max(poly);
    lit.clamp(0.0, 1.0)
}

/// Per-channel multi-bounce lightening for an RGB albedo.
///
/// Applies [`multi_bounce`] independently to each channel of `albedo`, so a
/// coloured surface tints the filled-in occlusion toward its own hue.
#[inline]
pub fn multi_bounce_rgb(ao: f32, albedo: Vec3) -> Vec3 {
    Vec3::new(
        multi_bounce(ao, albedo.x),
        multi_bounce(ao, albedo.y),
        multi_bounce(ao, albedo.z),
    )
}

/// Applies a power curve and intensity blend to an AO value.
///
/// `power` sharpens (`> 1`) or softens (`< 1`) the contrast via
/// `ao^power`; `intensity` blends between fully lit (`1`) and the shaped AO:
/// `1 - intensity*(1 - ao^power)`.  `power` is clamped non-negative and
/// `intensity` to `[0, 1]`; the result is clamped to `[0, 1]`.
#[inline]
pub fn power_intensity(ao: f32, power: f32, intensity: f32) -> f32 {
    let x = visibility_to_ao(ao);
    let power = if power.is_finite() { power.max(0.0) } else { 1.0 };
    let intensity = if intensity.is_finite() {
        intensity.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let shaped = ops::powf(x, power);
    let shaped = if shaped.is_finite() {
        shaped.clamp(0.0, 1.0)
    } else {
        x
    };
    (1.0 - intensity * (1.0 - shaped)).clamp(0.0, 1.0)
}

/// Combines a scalar AO term and a bent normal into a [`GtaoResult`].
///
/// The AO is clamped to `[0, 1]`; the bent normal is renormalised, falling back
/// to `+Y` when it is degenerate so the stored direction is always unit length.
#[inline]
pub fn combine(ao: f32, bent_normal: Vec3) -> GtaoResult {
    let visibility = visibility_to_ao(ao);
    let len_sq = bent_normal.length_squared();
    let bent_normal = if len_sq.is_finite() && len_sq > 1.0e-12 {
        bent_normal * len_sq.sqrt().recip()
    } else {
        Vec3::Y
    };
    GtaoResult {
        visibility,
        bent_normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn visibility_clamps_to_unit_range() {
        assert_eq!(visibility_to_ao(-2.0), 0.0);
        assert_eq!(visibility_to_ao(5.0), 1.0);
        assert_eq!(visibility_to_ao(f32::NAN), 0.0);
        assert!(approx(visibility_to_ao(0.37), 0.37, 1.0e-7));
    }

    #[test]
    fn multi_bounce_never_darkens() {
        for k in 0..=10 {
            let ao = k as f32 / 10.0;
            for a in [0.0, 0.25, 0.5, 0.8, 1.0] {
                let lit = multi_bounce(ao, a);
                assert!(lit + 1.0e-6 >= ao, "ao={ao} albedo={a} lit={lit}");
                assert!((0.0..=1.0).contains(&lit));
            }
        }
    }

    #[test]
    fn multi_bounce_brightens_for_bright_albedo() {
        // At mid AO a bright surface should fill in more than a dark one.
        let dark = multi_bounce(0.5, 0.05);
        let bright = multi_bounce(0.5, 0.95);
        assert!(bright > dark, "bright={bright} dark={dark}");
        // A black surface adds essentially no bounce light.
        assert!(approx(multi_bounce(0.5, 0.0), 0.5, 1.0e-3));
    }

    #[test]
    fn multi_bounce_is_monotonic_in_ao() {
        let albedo = 0.9;
        let mut prev = -1.0;
        for k in 0..=20 {
            let ao = k as f32 / 20.0;
            let lit = multi_bounce(ao, albedo);
            assert!(lit >= prev - 1.0e-6, "not monotonic at ao={ao}: {lit} < {prev}");
            prev = lit;
        }
    }

    #[test]
    fn multi_bounce_endpoints() {
        for a in [0.0, 0.5, 1.0] {
            assert!(approx(multi_bounce(0.0, a), 0.0, 1.0e-6));
            assert!(approx(multi_bounce(1.0, a), 1.0, 1.0e-5));
        }
    }

    #[test]
    fn multi_bounce_rgb_matches_scalar_per_channel() {
        let albedo = Vec3::new(0.1, 0.5, 0.9);
        let rgb = multi_bounce_rgb(0.4, albedo);
        assert!(approx(rgb.x, multi_bounce(0.4, 0.1), 1.0e-7));
        assert!(approx(rgb.y, multi_bounce(0.4, 0.5), 1.0e-7));
        assert!(approx(rgb.z, multi_bounce(0.4, 0.9), 1.0e-7));
        // Brighter channel is lit at least as much.
        assert!(rgb.z >= rgb.x);
    }

    #[test]
    fn power_intensity_identity_at_unit_power_full_intensity() {
        for k in 0..=10 {
            let ao = k as f32 / 10.0;
            assert!(approx(power_intensity(ao, 1.0, 1.0), ao, 1.0e-6));
        }
    }

    #[test]
    fn power_intensity_zero_intensity_is_fully_lit() {
        for ao in [0.0, 0.3, 0.7, 1.0] {
            assert!(approx(power_intensity(ao, 3.0, 0.0), 1.0, 1.0e-6));
        }
    }

    #[test]
    fn power_sharpening_darkens_midtones() {
        // Higher power pushes mid AO darker before the intensity blend.
        let soft = power_intensity(0.5, 1.0, 1.0);
        let sharp = power_intensity(0.5, 2.0, 1.0);
        assert!(sharp < soft, "sharp={sharp} soft={soft}");
    }

    #[test]
    fn power_intensity_handles_degenerate_inputs() {
        let v = power_intensity(f32::NAN, f32::NAN, f32::NAN);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v));
        let v = power_intensity(0.5, -3.0, 2.0);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v));
    }

    #[test]
    fn combine_normalizes_bent_normal() {
        let r = combine(0.6, Vec3::new(0.0, 0.0, 4.0));
        assert!(approx(r.visibility, 0.6, 1.0e-6));
        assert!(approx(r.bent_normal.length(), 1.0, 1.0e-6));
        assert!(approx(r.bent_normal.z, 1.0, 1.0e-6));
    }

    #[test]
    fn combine_degenerate_bent_normal_falls_back() {
        let r = combine(2.0, Vec3::ZERO);
        assert_eq!(r.visibility, 1.0);
        assert_eq!(r.bent_normal, Vec3::Y);
    }

    #[test]
    fn unoccluded_constant_is_fully_lit() {
        assert_eq!(GtaoResult::UNOCCLUDED.visibility, 1.0);
        assert!(approx(GtaoResult::UNOCCLUDED.bent_normal.length(), 1.0, 1.0e-6));
    }
}
