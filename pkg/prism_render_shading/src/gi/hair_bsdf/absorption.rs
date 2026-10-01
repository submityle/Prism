//! Absorption coefficients and appearance-parameter mapping for the
//! Marschner/Chiang hair BSDF (CPU golden reference).
//!
//! Artists do not want to author raw absorption coefficients `sigma_a`; they
//! want to pick a *color* and have the fiber reproduce it after the many
//! internal bounces that multiple scattering produces.  d'Eon (2011) and Chiang
//! (2016) fit a closed-form inverse that maps a desired multiple-scattering
//! albedo `c in (0, 1]` (per RGB channel) back to the `sigma_a` that yields it:
//!
//! ```text
//! sigma_a(c) = ( ln(c) / D(beta_n) )^2
//! D(beta_n) = 5.969 - 0.215 beta_n + 2.532 beta_n^2
//!           - 10.73 beta_n^3 + 5.574 beta_n^4 + 0.245 beta_n^5
//! ```
//!
//! where `beta_n` is the azimuthal roughness (rougher fibers scatter more and
//! therefore need a different `sigma_a` to hit the same apparent color).  The
//! forward map `c(sigma_a) = exp(-sqrt(sigma_a) * D)` is the exact algebraic
//! inverse, so a `sigma_a -> color -> sigma_a` round trip is lossless (verified
//! in the tests).
//!
//! For physically grounded authoring we also expose the classic *melanin*
//! parameterisation: real hair color comes from two pigments, brown-black
//! **eumelanin** and reddish-yellow **pheomelanin**, each with a fixed RGB
//! absorption spectrum.  A concentration mix reproduces the full natural range
//! from blonde through red to black.
//!
//! # Conventions
//! * `no_std`: RGB stored as [`bevy_math::Vec3`]; all transcendentals via
//!   [`bevy_math::ops`] (never `f32::exp`), square roots via `f32::sqrt`.
//! * Reflectances are clamped into `(epsilon, 1]` before the logarithm, and
//!   every returned `sigma_a` is forced finite and non-negative, so pure black
//!   (`c = 0`) maps to a large-but-finite absorption rather than `+inf`.
//! * Every function is a deterministic, allocation-free pure function with no
//!   RNG, I/O, GPU, or global state.

use bevy_math::{ops, Vec3};

/// Smallest reflectance accepted before taking its logarithm, so pure black
/// maps to a large but finite `sigma_a` instead of `+inf`.
const MIN_REFLECTANCE: f32 = 1.0e-4;

/// Largest `sigma_a` returned per channel, bounding the inversion of very dark
/// colors to a finite, well-conditioned value.
const MAX_SIGMA_A: f32 = 1.0e3;

/// Per-channel RGB absorption of pure **eumelanin** (brown-black pigment),
/// from the Chiang/d'Eon fit.
pub const EUMELANIN_SIGMA_A: [f32; 3] = [0.419, 0.697, 1.37];

/// Per-channel RGB absorption of pure **pheomelanin** (red-yellow pigment).
pub const PHEOMELANIN_SIGMA_A: [f32; 3] = [0.187, 0.4, 1.05];

/// Clamps an azimuthal roughness to `(0, 1]`.
#[inline]
fn clamp_beta_n(beta_n: f32) -> f32 {
    if beta_n.is_finite() {
        beta_n.clamp(1.0e-3, 1.0)
    } else {
        1.0e-3
    }
}

/// Clamps a reflectance channel to `(MIN_REFLECTANCE, 1]`.
#[inline]
fn clamp_reflectance(c: f32) -> f32 {
    if c.is_finite() {
        c.clamp(MIN_REFLECTANCE, 1.0)
    } else {
        MIN_REFLECTANCE
    }
}

/// Clamps a `sigma_a` channel to `[0, MAX_SIGMA_A]`.
#[inline]
fn clamp_sigma(s: f32) -> f32 {
    if s.is_finite() {
        s.clamp(0.0, MAX_SIGMA_A)
    } else {
        MAX_SIGMA_A
    }
}

/// The roughness-dependent denominator `D(beta_n)` of the color inversion.
///
/// This quintic polynomial captures how the apparent albedo saturates as the
/// azimuthal roughness grows; it is strictly positive over `beta_n in (0, 1]`.
#[inline]
pub fn azimuthal_denominator(beta_n: f32) -> f32 {
    let b = clamp_beta_n(beta_n);
    let b2 = b * b;
    let b3 = b2 * b;
    let b4 = b3 * b;
    let b5 = b4 * b;
    let d = 5.969 - 0.215 * b + 2.532 * b2 - 10.73 * b3 + 5.574 * b4 + 0.245 * b5;
    d.max(1.0e-2)
}

/// Inverts a desired per-channel multiple-scattering reflectance to the fiber
/// absorption `sigma_a`.
///
/// `reflectance` is the target RGB color (each channel clamped to
/// `(epsilon, 1]`) and `beta_n` the azimuthal roughness.  Returns the RGB
/// `sigma_a` with every channel finite and in `[0, MAX_SIGMA_A]`.
#[inline]
pub fn color_to_sigma_a(reflectance: Vec3, beta_n: f32) -> Vec3 {
    let d = azimuthal_denominator(beta_n);
    let channel = |c: f32| -> f32 {
        let c = clamp_reflectance(c);
        let s = ops::ln(c) / d;
        clamp_sigma(s * s)
    };
    Vec3::new(channel(reflectance.x), channel(reflectance.y), channel(reflectance.z))
}

/// Forward map: the multiple-scattering reflectance produced by a given
/// `sigma_a` at roughness `beta_n`.
///
/// This is the exact algebraic inverse of [`color_to_sigma_a`]:
/// `c = exp(-sqrt(sigma_a) * D(beta_n))`.  Each channel is returned in
/// `[0, 1]`.
#[inline]
pub fn sigma_a_to_color(sigma_a: Vec3, beta_n: f32) -> Vec3 {
    let d = azimuthal_denominator(beta_n);
    let channel = |s: f32| -> f32 {
        let s = clamp_sigma(s);
        ops::exp(-s.sqrt() * d).clamp(0.0, 1.0)
    };
    Vec3::new(channel(sigma_a.x), channel(sigma_a.y), channel(sigma_a.z))
}

/// Maps **eumelanin** and **pheomelanin** concentrations to an RGB `sigma_a`.
///
/// `eumelanin` and `pheomelanin` are non-negative concentrations; the result is
/// their pigment-weighted absorption sum, clamped finite and non-negative.
/// A pure-eumelanin fiber trends brown-black, pure pheomelanin trends red.
#[inline]
pub fn melanin_to_sigma_a(eumelanin: f32, pheomelanin: f32) -> Vec3 {
    let ce = if eumelanin.is_finite() { eumelanin.max(0.0) } else { 0.0 };
    let cp = if pheomelanin.is_finite() { pheomelanin.max(0.0) } else { 0.0 };
    Vec3::new(
        clamp_sigma(ce * EUMELANIN_SIGMA_A[0] + cp * PHEOMELANIN_SIGMA_A[0]),
        clamp_sigma(ce * EUMELANIN_SIGMA_A[1] + cp * PHEOMELANIN_SIGMA_A[1]),
        clamp_sigma(ce * EUMELANIN_SIGMA_A[2] + cp * PHEOMELANIN_SIGMA_A[2]),
    )
}

/// Convenience wrapper that maps a single scalar melanin amount with a
/// pheomelanin *ratio* `r in [0, 1]` into `sigma_a`.
///
/// `melanin` scales the overall pigment load; `redness` biases the eumelanin
/// (`r = 0`, brown-black) / pheomelanin (`r = 1`, red) split.
#[inline]
pub fn pigment_to_sigma_a(melanin: f32, redness: f32) -> Vec3 {
    let m = if melanin.is_finite() { melanin.max(0.0) } else { 0.0 };
    let r = if redness.is_finite() { redness.clamp(0.0, 1.0) } else { 0.0 };
    melanin_to_sigma_a(m * (1.0 - r), m * r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denominator_is_positive_over_range() {
        for bk in 0..=20 {
            let b = bk as f32 / 20.0;
            let d = azimuthal_denominator(b);
            assert!(d.is_finite() && d > 0.0, "beta_n={b} d={d}");
        }
    }

    #[test]
    fn sigma_a_is_non_negative_and_finite() {
        for ck in 1..=10 {
            let c = ck as f32 / 10.0;
            let sigma = color_to_sigma_a(Vec3::splat(c), 0.3);
            assert!(sigma.x >= 0.0 && sigma.x.is_finite(), "sigma={sigma:?}");
            assert!(sigma.y >= 0.0 && sigma.y.is_finite(), "sigma={sigma:?}");
            assert!(sigma.z >= 0.0 && sigma.z.is_finite(), "sigma={sigma:?}");
        }
    }

    #[test]
    fn darker_color_means_more_absorption() {
        // Monotonic: as the target reflectance drops, sigma_a must rise.
        let mut prev = -1.0_f32;
        for ck in (1..=10).rev() {
            let c = ck as f32 / 10.0; // 1.0 down to 0.1
            let sigma = color_to_sigma_a(Vec3::splat(c), 0.3).x;
            assert!(sigma >= prev, "c={c} sigma={sigma} prev={prev}");
            prev = sigma;
        }
    }

    #[test]
    fn roundtrip_sigma_color_sigma_is_lossless() {
        // sigma -> color -> sigma must recover the original within tolerance.
        for &sx in &[0.02_f32, 0.1, 0.3, 0.6] {
            let sigma = Vec3::new(sx, sx * 1.2, sx * 1.5);
            let color = sigma_a_to_color(sigma, 0.3);
            let back = color_to_sigma_a(color, 0.3);
            assert!((back - sigma).length() < 1.0e-2, "sigma={sigma:?} back={back:?}");
        }
    }

    #[test]
    fn roundtrip_color_sigma_color_is_lossless() {
        for &c in &[0.1_f32, 0.3, 0.6, 0.9] {
            let color = Vec3::new(c, c * 0.8, c * 0.5);
            let sigma = color_to_sigma_a(color, 0.4);
            let back = sigma_a_to_color(sigma, 0.4);
            assert!((back - color).length() < 1.0e-3, "color={color:?} back={back:?}");
        }
    }

    #[test]
    fn melanin_mix_matches_pigment_spectra() {
        // Pure eumelanin reproduces its spectrum exactly.
        let eu = melanin_to_sigma_a(1.0, 0.0);
        assert!((eu.x - EUMELANIN_SIGMA_A[0]).abs() < 1.0e-6);
        assert!((eu.z - EUMELANIN_SIGMA_A[2]).abs() < 1.0e-6);
        // Pure pheomelanin likewise.
        let ph = melanin_to_sigma_a(0.0, 1.0);
        assert!((ph.y - PHEOMELANIN_SIGMA_A[1]).abs() < 1.0e-6);
        // Both pigments absorb blue most strongly (z channel largest).
        assert!(eu.z > eu.x && ph.z > ph.x);
    }

    #[test]
    fn more_melanin_absorbs_more() {
        let light = melanin_to_sigma_a(0.5, 0.0);
        let dark = melanin_to_sigma_a(4.0, 0.0);
        assert!(dark.x > light.x && dark.y > light.y && dark.z > light.z);
    }

    #[test]
    fn pigment_wrapper_blends_endpoints() {
        // redness = 0 is pure eumelanin, redness = 1 is pure pheomelanin.
        let brown = pigment_to_sigma_a(2.0, 0.0);
        let red = pigment_to_sigma_a(2.0, 1.0);
        let expect_brown = melanin_to_sigma_a(2.0, 0.0);
        let expect_red = melanin_to_sigma_a(0.0, 2.0);
        assert!((brown - expect_brown).length() < 1.0e-6);
        assert!((red - expect_red).length() < 1.0e-6);
    }

    #[test]
    fn determinism_and_extreme_inputs_are_finite() {
        let a = color_to_sigma_a(Vec3::new(0.2, 0.5, 0.8), 0.37);
        let b = color_to_sigma_a(Vec3::new(0.2, 0.5, 0.8), 0.37);
        assert_eq!(a, b);
        // Black and NaN/inf inputs stay finite.
        let black = color_to_sigma_a(Vec3::ZERO, 0.3);
        assert!(black.x.is_finite() && black.x > 0.0);
        let wild = color_to_sigma_a(Vec3::splat(f32::NAN), f32::INFINITY);
        assert!(wild.x.is_finite() && wild.y.is_finite() && wild.z.is_finite());
        let wild_s = melanin_to_sigma_a(f32::INFINITY, f32::NAN);
        assert!(wild_s.x.is_finite() && wild_s.y.is_finite() && wild_s.z.is_finite());
    }
}
