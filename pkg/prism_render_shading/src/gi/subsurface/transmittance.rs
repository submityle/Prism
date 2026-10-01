//! Jimenez thin-slab translucency transmittance — the CPU golden reference for
//! back-lit light transport through skin and other thin dielectric slabs.
//!
//! When a light sits *behind* a thin surface (an ear, a nostril, a finger held
//! to the sun) a fraction of its radiance survives the trip through the slab
//! and emerges on the viewer's side, tinted red by the deeper scattering of
//! long wavelengths.  Jimenez et al. (*Real-Time Realistic Skin Translucency*,
//! 2010) precompute this as a 1-D transmittance profile `T(s)` of the local
//! slab thickness `s`, fit by a sum of Gaussians — the same multi-pole skin
//! profile used by d'Eon & Luebke:
//!
//! ```text
//!   T(s) = Σ_i w_i · exp(-s² / v_i)
//! ```
//!
//! Each lobe `i` has a per-channel weight `w_i` and a falloff `v_i`; the weights
//! sum to (approximately) one per channel, so `T(0) ≈ 1` (a zero-thickness slab
//! transmits everything) and `T` decays monotonically toward zero as the slab
//! thickens.  The red channel carries the widest lobes, so thick regions pass
//! predominantly red light — the signature translucent glow of skin.
//!
//! Both the canonical six-Gaussian fit and a cheaper four-Gaussian reduction are
//! provided, along with a convenience that folds `T` into a back-facing wrap
//! term to produce transmitted radiance.
//!
//! # Conventions
//! * `thickness` is a slab path length in the same units as the profile falloffs
//!   (millimetres by convention); its magnitude is used, so the sign only
//!   encodes front/back and never produces negative attenuation.
//! * `scale` converts world-space thickness into profile units; it is clamped
//!   strictly positive.
//! * `T(s)` is non-negative, monotonically non-increasing in `|s|`, and clamped
//!   into `[0, 1]` on every channel (energy clamp: a slab never transmits more
//!   than the incident radiance).
//! * `back_cosine` is `dot(-N, L)` — positive when the light is behind the
//!   surface; it is clamped to `[0, 1]`.
//! * Spectral quantities are linear-RGB [`Vec3`]s matching the GPU twin.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   and no `unsafe`.  Transcendental maths goes through [`bevy_math::ops`].

use bevy_math::{ops, Vec3};

/// Largest squared-thickness-over-falloff exponent before the lobe is zero.
const MAX_EXPONENT: f32 = 80.0;

/// One Gaussian lobe of a transmittance profile (`weight · exp(-s²/falloff)`).
#[derive(Clone, Copy, Debug)]
pub struct TransmittanceLobe {
    /// Falloff `v` (squared length); larger values transmit through thicker slabs.
    pub falloff: f32,
    /// Per-channel linear-RGB weight.
    pub weight: Vec3,
}

/// Canonical six-Gaussian Jimenez / d'Eon skin transmittance profile.
///
/// Per-channel weights sum to `≈ (1.0, 1.0, 0.993)`, giving `T(0) ≈ 1`.  The
/// widest lobes carry only red, reproducing the deep-red translucency of skin.
pub const SKIN_SIX_GAUSSIAN: [TransmittanceLobe; 6] = [
    TransmittanceLobe { falloff: 0.0064, weight: Vec3::new(0.233, 0.455, 0.649) },
    TransmittanceLobe { falloff: 0.0484, weight: Vec3::new(0.100, 0.336, 0.344) },
    TransmittanceLobe { falloff: 0.1870, weight: Vec3::new(0.118, 0.198, 0.000) },
    TransmittanceLobe { falloff: 0.5670, weight: Vec3::new(0.113, 0.007, 0.007) },
    TransmittanceLobe { falloff: 1.9900, weight: Vec3::new(0.358, 0.004, 0.000) },
    TransmittanceLobe { falloff: 7.4100, weight: Vec3::new(0.078, 0.000, 0.000) },
];

/// Reduced four-Gaussian skin transmittance fit (cheaper real-time variant).
///
/// Per-channel weights sum to `≈ (1, 1, 1)`; it preserves the overall decay and
/// red bias of [`SKIN_SIX_GAUSSIAN`] with two fewer exponentials.
pub const SKIN_FOUR_GAUSSIAN: [TransmittanceLobe; 4] = [
    TransmittanceLobe { falloff: 0.0064, weight: Vec3::new(0.233, 0.455, 0.649) },
    TransmittanceLobe { falloff: 0.0484, weight: Vec3::new(0.100, 0.336, 0.344) },
    TransmittanceLobe { falloff: 0.1870, weight: Vec3::new(0.187, 0.205, 0.007) },
    TransmittanceLobe { falloff: 1.9900, weight: Vec3::new(0.480, 0.004, 0.000) },
];

/// Clamp a possibly non-finite scalar into `[lo, hi]`, mapping `NaN` to `lo`.
#[inline]
fn clamp_finite(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        lo
    }
}

/// Evaluate a transmittance profile `T(s) = Σ w_i exp(-s²/v_i)` at `thickness`.
///
/// Uses `|thickness|`, so the sign is informational.  Each lobe's exponent is
/// clamped to avoid overflow; the per-channel result is clamped into `[0, 1]`
/// (energy clamp) and is always finite and non-negative.
pub fn transmittance_with(profile: &[TransmittanceLobe], thickness: f32) -> Vec3 {
    let s = clamp_finite(thickness, f32::MIN, f32::MAX).abs();
    let s2 = s * s;
    let mut acc = Vec3::ZERO;
    for lobe in profile {
        let v = lobe.falloff.max(1.0e-8);
        let exponent = (s2 / v).min(MAX_EXPONENT);
        acc += lobe.weight.max(Vec3::ZERO) * ops::exp(-exponent);
    }
    acc.clamp(Vec3::ZERO, Vec3::ONE)
}

/// Canonical skin transmittance `T(thickness)` using [`SKIN_SIX_GAUSSIAN`].
///
/// `thickness` is in profile units (millimetres).  See [`transmittance_with`]
/// for the guarantees (non-negative, monotonic in `|thickness|`, clamped to
/// `[0, 1]`).
#[inline]
pub fn transmittance(thickness: f32) -> Vec3 {
    transmittance_with(&SKIN_SIX_GAUSSIAN, thickness)
}

/// Reduced four-Gaussian skin transmittance `T(thickness)`.
#[inline]
pub fn transmittance_reduced(thickness: f32) -> Vec3 {
    transmittance_with(&SKIN_FOUR_GAUSSIAN, thickness)
}

/// Transmittance of a world-space `thickness` after conversion by `scale`.
///
/// Computes `T(scale · |thickness|)`; `scale` (profile units per world unit) is
/// clamped strictly positive.  Shares the guarantees of [`transmittance_with`].
#[inline]
pub fn transmittance_scaled(profile: &[TransmittanceLobe], thickness: f32, scale: f32) -> Vec3 {
    let scale = clamp_finite(scale, 1.0e-6, f32::MAX);
    let s = clamp_finite(thickness, f32::MIN, f32::MAX).abs();
    transmittance_with(profile, s * scale)
}

/// Back-lit transmitted radiance `L = light · T(thickness) · wrap(back_cosine)`.
///
/// Folds the transmittance profile into a clamped back-facing cosine so a light
/// behind a thin slab contributes diffuse glow on the shaded side.  `light` is
/// a non-negative linear-RGB radiance, `back_cosine = dot(-N, L)` is clamped to
/// `[0, 1]`, and the result is non-negative and finite.
pub fn transmitted_radiance(
    light: Vec3,
    profile: &[TransmittanceLobe],
    thickness: f32,
    back_cosine: f32,
) -> Vec3 {
    let light = light.max(Vec3::ZERO);
    let t = transmittance_with(profile, thickness);
    let wrap = clamp_finite(back_cosine, 0.0, 1.0);
    (light * t * wrap).max(Vec3::ZERO)
}

/// Per-channel sum of a profile's lobe weights — its `T(0)` before clamping.
///
/// Useful to verify a profile is energy-bounded (`≤ 1` per channel).
pub fn profile_energy(profile: &[TransmittanceLobe]) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for lobe in profile {
        acc += lobe.weight.max(Vec3::ZERO);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn zero_thickness_transmits_everything() {
        let t = transmittance(0.0);
        assert!(approx(t.x, 1.0, 1e-3), "red={}", t.x);
        assert!(approx(t.y, 1.0, 1e-3), "green={}", t.y);
        // Blue weights sum to ~0.993.
        assert!(t.z > 0.98 && t.z <= 1.0, "blue={}", t.z);
    }

    #[test]
    fn monotonically_decreasing_in_thickness() {
        for profile in [&SKIN_SIX_GAUSSIAN[..], &SKIN_FOUR_GAUSSIAN[..]] {
            let mut prev = transmittance_with(profile, 0.0);
            let mut s = 0.1f32;
            while s <= 12.0 {
                let t = transmittance_with(profile, s);
                assert!(t.x <= prev.x + 1e-6, "red not decreasing at s={s}");
                assert!(t.y <= prev.y + 1e-6, "green not decreasing at s={s}");
                assert!(t.z <= prev.z + 1e-6, "blue not decreasing at s={s}");
                prev = t;
                s += 0.1;
            }
        }
    }

    #[test]
    fn symmetric_in_sign() {
        for s in [0.3f32, 1.0, 2.5, 5.0] {
            assert_eq!(transmittance(s), transmittance(-s));
        }
    }

    #[test]
    fn non_negative_and_bounded() {
        for i in 0..=120 {
            let s = i as f32 * 0.1;
            let t = transmittance(s);
            for c in [t.x, t.y, t.z] {
                assert!((0.0..=1.0).contains(&c), "c={c} s={s}");
            }
        }
    }

    #[test]
    fn red_outlasts_blue_through_thick_slabs() {
        // Deep tissue passes red far more than blue.
        let t = transmittance(3.0);
        assert!(t.x > t.z, "expected red>blue, got {t:?}");
        assert!(t.x > t.y, "expected red>green, got {t:?}");
    }

    #[test]
    fn energy_is_bounded_by_one() {
        let e6 = profile_energy(&SKIN_SIX_GAUSSIAN);
        let e4 = profile_energy(&SKIN_FOUR_GAUSSIAN);
        for e in [e6, e4] {
            assert!(e.x <= 1.0 + 1e-4, "red energy {}", e.x);
            assert!(e.y <= 1.0 + 1e-4, "green energy {}", e.y);
            assert!(e.z <= 1.0 + 1e-4, "blue energy {}", e.z);
        }
    }

    #[test]
    fn scaled_matches_manual_scale() {
        let a = transmittance_scaled(&SKIN_SIX_GAUSSIAN, 2.0, 1.5);
        let b = transmittance_with(&SKIN_SIX_GAUSSIAN, 3.0);
        assert!((a - b).length() < 1e-6, "a={a:?} b={b:?}");
    }

    #[test]
    fn transmitted_radiance_scales_with_backlight() {
        let light = Vec3::new(2.0, 2.0, 2.0);
        let full = transmitted_radiance(light, &SKIN_SIX_GAUSSIAN, 1.0, 1.0);
        let half = transmitted_radiance(light, &SKIN_SIX_GAUSSIAN, 1.0, 0.5);
        let none = transmitted_radiance(light, &SKIN_SIX_GAUSSIAN, 1.0, 0.0);
        assert!((full - half * 2.0).length() < 1e-5, "full={full:?} half={half:?}");
        assert_eq!(none, Vec3::ZERO);
    }

    #[test]
    fn reduced_tracks_canonical() {
        // The four-Gaussian fit should stay close to the six-Gaussian reference.
        for s in [0.0f32, 0.5, 1.0, 2.0, 4.0] {
            let six = transmittance(s);
            let four = transmittance_reduced(s);
            assert!((six - four).length() < 0.12, "s={s} six={six:?} four={four:?}");
        }
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(transmittance(1.3), transmittance(1.3));
        assert_eq!(transmittance_reduced(0.7), transmittance_reduced(0.7));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        assert!(transmittance(f32::NAN).is_finite());
        assert!(transmittance(f32::INFINITY).is_finite());
        assert!(transmittance_reduced(f32::NAN).is_finite());
        assert!(transmittance_scaled(&SKIN_SIX_GAUSSIAN, f32::NAN, f32::NAN).is_finite());
        assert!(transmitted_radiance(Vec3::splat(f32::NAN), &SKIN_SIX_GAUSSIAN, f32::NAN, f32::NAN).is_finite());
        assert!(profile_energy(&SKIN_SIX_GAUSSIAN).is_finite());
    }
}
