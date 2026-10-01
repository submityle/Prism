//! Beer-Lambert path absorption and the Fresnel reflect/transmit energy split.
//!
//! This module is the CPU golden reference for the *radiometric* half of
//! screen-space refraction.  Once [`crate::gi::refraction::bend`] has decided
//! *where* to sample the background buffer, two physical effects remain before
//! the transmitted colour is correct:
//!
//! 1. **Volume absorption.** As light travels a distance `dist` through a
//!    coloured medium it is attenuated per wavelength by the Beer-Lambert law
//!    `T(dist) = exp(-sigma_a * dist)`, where `sigma_a` is the per-channel
//!    absorption coefficient.  This is what tints thick glass green, a ruby
//!    red, or deep water blue — the attenuation is exponential in depth, so the
//!    colour deepens with path length.
//! 2. **The Fresnel split.** Only part of the incident radiance is transmitted
//!    through the interface; the rest is reflected.  The split `(R, T)` is
//!    governed by the Fresnel equations and must conserve energy: `R + T = 1`
//!    for a lossless dielectric (and `T = 0`, `R = 1` under total internal
//!    reflection).
//!
//! Both the exact unpolarised dielectric Fresnel reflectance and the cheap
//! Schlick approximation are provided so the GPU twin pass can be validated
//! against the physical ground truth.  The Schlick evaluator reuses the shared
//! lobe helper [`crate::gi::spec_gi::ggx_lobe::fresnel_schlick_scalar`] rather
//! than re-deriving it.
//!
//! # Conventions
//! * Radiance and absorption coefficients are per-channel [`Vec3`] in linear
//!   RGB.  `sigma_a` has units of inverse distance and is clamped to `>= 0`.
//! * `cos_i` is the cosine of the incidence angle (between the view ray and the
//!   surface normal), clamped to `[0, 1]`.  `n_i`, `n_t` are the incident and
//!   transmitted indices of refraction, clamped to the dielectric range
//!   `[1, inf)`.
//! * `no_std`: math via `bevy_math`; the exponential via `bevy_math::ops::exp`
//!   and the logarithm via `bevy_math::ops::ln` (never `f32::exp`); square
//!   roots via the `f32::sqrt` method.  Nothing is allocated.
//! * Every transmittance is in `[0, 1]` per channel and decreases monotonically
//!   with distance; every reflect/transmit pair is non-negative and sums to at
//!   most one.  All results are finite — no `NaN`, no infinity.
//! * Every function is a deterministic, allocation-free pure function.

use crate::gi::spec_gi::ggx_lobe::fresnel_schlick_scalar;
use bevy_math::{ops, Vec3};

/// Smallest index of refraction accepted; physical dielectrics have `n >= 1`.
const MIN_IOR: f32 = 1.0;

/// Lower bound on the Beer-Lambert exponent, i.e. `-sigma_a * dist`, before
/// exponentiation.  `exp(-80) ~= 1.8e-35` is already indistinguishable from
/// zero, so clamping here keeps the result finite for enormous optical depths
/// without changing any observable value.
const MIN_EXPONENT: f32 = -80.0;

/// Clamps an index of refraction to the physical dielectric range `[1, inf)`.
#[inline]
fn clamp_ior(n: f32) -> f32 {
    if n.is_finite() {
        n.max(MIN_IOR)
    } else {
        MIN_IOR
    }
}

/// Clamps a cosine of an incidence angle to `[0, 1]`.
#[inline]
fn clamp_cos(cos: f32) -> f32 {
    if cos.is_finite() {
        cos.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Sanitises a scalar that must be non-negative; non-finite inputs become `0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Per-channel `exp(-x)` with the exponent floored by [`MIN_EXPONENT`].
///
/// The input `x` is the optical depth `sigma_a * dist` (expected `>= 0`); the
/// result is a transmittance in `(0, 1]`.
#[inline]
fn exp_neg(x: Vec3) -> Vec3 {
    Vec3::new(
        ops::exp((-x.x).max(MIN_EXPONENT)),
        ops::exp((-x.y).max(MIN_EXPONENT)),
        ops::exp((-x.z).max(MIN_EXPONENT)),
    )
}

/// Beer-Lambert transmittance of a coloured medium over a path.
///
/// Returns `exp(-sigma_a * dist)` per channel: the fraction of radiance that
/// survives travelling `dist` through a medium with absorption coefficient
/// `sigma_a`.  Both inputs are clamped non-negative, so the result is always in
/// `(0, 1]` and decreases monotonically as either the distance or the
/// absorption grows.  At `dist == 0` (or `sigma_a == 0`) the medium is
/// perfectly clear and the transmittance is [`Vec3::ONE`].
#[inline]
pub fn beer_lambert_transmittance(sigma_a: Vec3, dist: f32) -> Vec3 {
    let sigma = Vec3::new(
        sanitize_nonneg(sigma_a.x),
        sanitize_nonneg(sigma_a.y),
        sanitize_nonneg(sigma_a.z),
    );
    let d = sanitize_nonneg(dist);
    exp_neg(sigma * d)
}

/// Applies Beer-Lambert absorption to a radiance travelling through a medium.
///
/// Convenience wrapper: `radiance * beer_lambert_transmittance(sigma_a, dist)`,
/// with the incoming radiance clamped non-negative per channel so no negative
/// energy leaks through.
#[inline]
pub fn absorb(radiance: Vec3, sigma_a: Vec3, dist: f32) -> Vec3 {
    let r = Vec3::new(
        sanitize_nonneg(radiance.x),
        sanitize_nonneg(radiance.y),
        sanitize_nonneg(radiance.z),
    );
    r * beer_lambert_transmittance(sigma_a, dist)
}

/// Recovers an absorption coefficient from a desired transmission tint.
///
/// Artists usually author "the colour thick glass should be" rather than a raw
/// `sigma_a`.  Given the `tint` the medium should exhibit at a reference depth
/// `ref_dist`, this inverts Beer-Lambert to `sigma_a = -ln(tint) / ref_dist`
/// per channel.  The tint is clamped to `(0, 1]` (a channel of `0` would demand
/// infinite absorption, so it is floored to a tiny positive value) and
/// `ref_dist` is floored to a small positive distance.  Feeding the result back
/// into [`beer_lambert_transmittance`] at `ref_dist` reproduces the tint.
#[inline]
pub fn absorption_from_tint(tint: Vec3, ref_dist: f32) -> Vec3 {
    const MIN_TINT: f32 = 1.0e-4;
    const MIN_DIST: f32 = 1.0e-4;
    let d = sanitize_nonneg(ref_dist).max(MIN_DIST);
    let channel = |c: f32| -> f32 {
        let c = if c.is_finite() { c.clamp(MIN_TINT, 1.0) } else { MIN_TINT };
        -ops::ln(c) / d
    };
    Vec3::new(channel(tint.x), channel(tint.y), channel(tint.z))
}

/// Normal-incidence reflectance `F0` for a dielectric interface.
///
/// Returns `((n_t - n_i) / (n_t + n_i))^2`, the fraction of power reflected at
/// perpendicular incidence.  For an air -> glass interface (`1 -> 1.5`) this is
/// `~= 0.04`, the familiar seed for the Schlick approximation.
#[inline]
pub fn f0_dielectric(n_i: f32, n_t: f32) -> f32 {
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);
    let r = (n_t - n_i) / (n_t + n_i);
    (r * r).clamp(0.0, 1.0)
}

/// Exact unpolarised Fresnel reflectance of a dielectric interface.
///
/// Averages the s- and p-polarised power reflectances for light crossing from
/// medium `n_i` into medium `n_t` at incident cosine `cos_i`.  Beyond the
/// critical angle (total internal reflection, only possible when `n_i > n_t`)
/// the surface reflects everything and the result is `1`.  This is the physical
/// ground truth that [`fresnel_schlick_dielectric`] approximates.
#[inline]
pub fn fresnel_dielectric(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    let cos_i = clamp_cos(cos_i);
    let n_i = clamp_ior(n_i);
    let n_t = clamp_ior(n_t);

    let sin_i2 = (1.0 - cos_i * cos_i).max(0.0);
    let eta = n_i / n_t;
    let sin_t2 = eta * eta * sin_i2;
    if sin_t2 >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin_t2).max(0.0).sqrt();

    let r_s = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t);
    let r_p = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t);
    (0.5 * (r_s * r_s + r_p * r_p)).clamp(0.0, 1.0)
}

/// Schlick's approximation of the dielectric Fresnel reflectance.
///
/// Seeds the shared [`fresnel_schlick_scalar`] lobe helper with the
/// normal-incidence reflectance [`f0_dielectric`] of the `n_i -> n_t`
/// interface.  Fast and widely used for GI; exact at normal incidence and
/// increasingly divergent from [`fresnel_dielectric`] only near grazing angles.
#[inline]
pub fn fresnel_schlick_dielectric(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    fresnel_schlick_scalar(f0_dielectric(n_i, n_t), clamp_cos(cos_i))
}

/// The energy-conserving split of incident radiance at a dielectric interface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelSplit {
    /// Fraction of power reflected; in `[0, 1]`.
    pub reflect: f32,
    /// Fraction of power transmitted; `1 - reflect`, in `[0, 1]`.
    pub transmit: f32,
}

/// Exact reflect/transmit weights for a dielectric interface.
///
/// Returns a [`FresnelSplit`] whose `reflect` weight is the exact Fresnel
/// reflectance at incident cosine `cos_i` for the `n_i -> n_t` interface and
/// whose `transmit` weight is `1 - reflect`.  The two always sum to one, so no
/// energy is created or lost; under total internal reflection the pair is
/// `(1, 0)`.
#[inline]
pub fn reflect_transmit_split(cos_i: f32, n_i: f32, n_t: f32) -> FresnelSplit {
    let reflect = fresnel_dielectric(cos_i, n_i, n_t);
    FresnelSplit {
        reflect,
        transmit: (1.0 - reflect).clamp(0.0, 1.0),
    }
}

/// Cheap reflect/transmit weights using the Schlick Fresnel approximation.
///
/// Same contract as [`reflect_transmit_split`] but seeded by
/// [`fresnel_schlick_dielectric`]; `reflect + transmit == 1`.
#[inline]
pub fn reflect_transmit_split_schlick(cos_i: f32, n_i: f32, n_t: f32) -> FresnelSplit {
    let reflect = fresnel_schlick_dielectric(cos_i, n_i, n_t);
    FresnelSplit {
        reflect,
        transmit: (1.0 - reflect).clamp(0.0, 1.0),
    }
}

/// Reflected and transmitted radiance for a dielectric fragment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitRadiance {
    /// Radiance redirected into the reflected lobe.
    pub reflected: Vec3,
    /// Radiance that enters the medium (before volume absorption).
    pub transmitted: Vec3,
}

/// Splits incident radiance into reflected and transmitted parts.
///
/// Scales `incident` (clamped non-negative) by the exact Fresnel reflect and
/// transmit weights at `cos_i` for the `n_i -> n_t` interface.  Because the
/// weights sum to one, `reflected + transmitted == incident` channel-wise: the
/// split conserves energy.  Volume absorption of the transmitted part is a
/// separate step (see [`absorb`]).
#[inline]
pub fn split_radiance(incident: Vec3, cos_i: f32, n_i: f32, n_t: f32) -> SplitRadiance {
    let r = Vec3::new(
        sanitize_nonneg(incident.x),
        sanitize_nonneg(incident.y),
        sanitize_nonneg(incident.z),
    );
    let split = reflect_transmit_split(cos_i, n_i, n_t);
    SplitRadiance {
        reflected: r * split.reflect,
        transmitted: r * split.transmit,
    }
}

/// Full transmitted contribution of a refracted background sample.
///
/// Combines the Fresnel transmit weight with Beer-Lambert absorption over the
/// in-medium path: `transmit * exp(-sigma_a * dist) * background`.  This is the
/// value a screen-space refraction pass adds for the transmitted lobe once the
/// background has been sampled at the bent UV.
#[inline]
pub fn transmitted_background(
    background: Vec3,
    cos_i: f32,
    n_i: f32,
    n_t: f32,
    sigma_a: Vec3,
    dist: f32,
) -> Vec3 {
    let split = reflect_transmit_split(cos_i, n_i, n_t);
    absorb(background, sigma_a, dist) * split.transmit
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-4;
    const AIR: f32 = 1.0;
    const GLASS: f32 = 1.5;

    fn assert_finite(v: Vec3) {
        assert!(v.x.is_finite() && v.y.is_finite() && v.z.is_finite(), "{v:?}");
    }

    #[test]
    fn transmittance_is_one_at_zero_distance() {
        let t = beer_lambert_transmittance(Vec3::new(0.5, 1.0, 2.0), 0.0);
        assert!((t - Vec3::ONE).length() < EPS, "{t:?}");
    }

    #[test]
    fn transmittance_decreases_monotonically_with_distance() {
        let sigma = Vec3::splat(0.7);
        let near = beer_lambert_transmittance(sigma, 0.5);
        let far = beer_lambert_transmittance(sigma, 3.0);
        assert!(far.x < near.x && far.y < near.y && far.z < near.z);
        // Still strictly within (0, 1].
        assert!(far.x > 0.0 && near.x <= 1.0);
    }

    #[test]
    fn transmittance_is_per_channel_colored() {
        // Red absorbs least, blue most -> warm transmitted tint.
        let sigma = Vec3::new(0.1, 0.4, 0.9);
        let t = beer_lambert_transmittance(sigma, 2.0);
        assert!(t.x > t.y && t.y > t.z, "{t:?}");
    }

    #[test]
    fn transmittance_stays_finite_for_huge_optical_depth() {
        let t = beer_lambert_transmittance(Vec3::splat(1.0e9), 1.0e9);
        assert_finite(t);
        assert!(t.x >= 0.0 && t.x <= 1.0);
    }

    #[test]
    fn negative_and_nan_inputs_are_sanitised() {
        let t = beer_lambert_transmittance(Vec3::new(-1.0, f32::NAN, 0.5), -2.0);
        assert_finite(t);
        // dist clamps to 0, so the whole thing is clear.
        assert!((t - Vec3::ONE).length() < EPS);
    }

    #[test]
    fn absorption_from_tint_round_trips() {
        let tint = Vec3::new(0.8, 0.5, 0.2);
        let ref_dist = 1.5;
        let sigma = absorption_from_tint(tint, ref_dist);
        let back = beer_lambert_transmittance(sigma, ref_dist);
        assert!((back - tint).length() < EPS, "{back:?} vs {tint:?}");
    }

    #[test]
    fn absorb_never_amplifies() {
        let r = absorb(Vec3::new(1.0, 1.0, 1.0), Vec3::splat(0.3), 2.0);
        assert!(r.x <= 1.0 && r.y <= 1.0 && r.z <= 1.0);
        assert!(r.x >= 0.0);
    }

    #[test]
    fn f0_matches_known_glass_value() {
        // Air -> glass F0 ~= 0.04.
        assert!((f0_dielectric(AIR, GLASS) - 0.04).abs() < 2.0e-3);
    }

    #[test]
    fn energy_is_conserved_exact() {
        for &cos in &[0.05f32, 0.3, 0.6, 0.9, 1.0] {
            let s = reflect_transmit_split(cos, AIR, GLASS);
            assert!((s.reflect + s.transmit - 1.0).abs() < EPS, "cos={cos}");
            assert!(s.reflect >= 0.0 && s.reflect <= 1.0);
            assert!(s.transmit >= 0.0 && s.transmit <= 1.0);
        }
    }

    #[test]
    fn energy_is_conserved_schlick() {
        for &cos in &[0.1f32, 0.5, 1.0] {
            let s = reflect_transmit_split_schlick(cos, AIR, GLASS);
            assert!((s.reflect + s.transmit - 1.0).abs() < EPS, "cos={cos}");
        }
    }

    #[test]
    fn schlick_tracks_exact_away_from_grazing() {
        // The two Fresnel models should agree closely near normal incidence.
        let exact = fresnel_dielectric(0.9, AIR, GLASS);
        let schlick = fresnel_schlick_dielectric(0.9, AIR, GLASS);
        assert!((exact - schlick).abs() < 1.0e-2, "{exact} vs {schlick}");
    }

    #[test]
    fn grazing_incidence_reflects_everything() {
        let s = reflect_transmit_split(0.0, AIR, GLASS);
        assert!((s.reflect - 1.0).abs() < EPS);
        assert!(s.transmit < EPS);
    }

    #[test]
    fn total_internal_reflection_transmits_nothing() {
        // Glass -> air past the critical angle: cos_i small enough to TIR.
        let s = reflect_transmit_split(0.2, GLASS, AIR);
        assert_eq!(s.reflect, 1.0);
        assert_eq!(s.transmit, 0.0);
    }

    #[test]
    fn split_radiance_conserves_energy() {
        let incident = Vec3::new(0.9, 0.6, 0.3);
        let s = split_radiance(incident, 0.7, AIR, GLASS);
        let sum = s.reflected + s.transmitted;
        assert!((sum - incident).length() < EPS, "{sum:?} vs {incident:?}");
        assert!(s.reflected.min_element() >= 0.0);
        assert!(s.transmitted.min_element() >= 0.0);
    }

    #[test]
    fn transmitted_background_is_attenuated_and_bounded() {
        let bg = Vec3::ONE;
        let near = transmitted_background(bg, 0.8, AIR, GLASS, Vec3::splat(0.4), 0.5);
        let far = transmitted_background(bg, 0.8, AIR, GLASS, Vec3::splat(0.4), 4.0);
        // Deeper path -> darker transmitted colour.
        assert!(far.x < near.x);
        // Never exceeds the (already <= 1) Fresnel transmit weight.
        assert!(near.x <= 1.0 && far.x >= 0.0);
    }
}
