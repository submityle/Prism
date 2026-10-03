//! Fresnel integrals and the UTD transition function `F(X)`.
//!
//! The Uniform Theory of Diffraction (UTD, Kouyoumjian and Pathak 1974)
//! expresses a wedge diffraction coefficient as a sum of cotangent terms, each
//! multiplied by a *transition function* `F(X)` that smoothly removes the
//! singularities the geometrical-optics cotangents have at the shadow and
//! reflection boundaries. `F(X)` is built from the Fresnel integrals, so this
//! module provides both: a numerically robust `f32` evaluation of the Fresnel
//! cosine and sine integrals `C(z)` and `S(z)`, and the complex UTD transition
//! function `F(X)` defined on top of them.
//!
//! # Fresnel integrals
//!
//! With the standard normalisation
//!
//! `C(z) = integral from 0 to z of cos(pi/2 * t^2) dt`,
//! `S(z) = integral from 0 to z of sin(pi/2 * t^2) dt`,
//!
//! both are odd functions and tend to `1/2` as `z -> +infinity`. A single-
//! precision power series suffers catastrophic cancellation for moderate `z`,
//! so this module uses the Abramowitz and Stegun 7.3.32 / 7.3.33 auxiliary
//! functions `f(z)` and `g(z)` for `z < 4` and the large-argument asymptotic
//! expansion for `z >= 4`, recombined through
//!
//! `C(z) = 1/2 + f(z) * sin(a) - g(z) * cos(a)`,
//! `S(z) = 1/2 - f(z) * cos(a) - g(z) * sin(a)`,  with `a = pi/2 * z^2`.
//!
//! # Transition function
//!
//! The UTD transition function is
//!
//! `F(X) = 2j * sqrt(X) * exp(jX) * integral from sqrt(X) to infinity of exp(-j tau^2) d tau`,
//!
//! defined for `X >= 0` (and taken as `0` for `X <= 0`). Substituting
//! `tau = t * sqrt(pi/2)` rewrites the tail integral in terms of `C` and `S`:
//!
//! `integral from sqrt(X) to inf of exp(-j tau^2) d tau`
//! `  = sqrt(pi/2) * [ (1/2 - C(w)) - j (1/2 - S(w)) ]`,  with `w = sqrt(2X/pi)`.
//!
//! As `X -> infinity` the magnitude approaches `1` and the phase approaches
//! `0` (geometrical optics is recovered); as `X -> 0` the magnitude behaves
//! like `sqrt(pi * X)` with phase `pi/4`.
//!
//! # Determinism and real-time safety
//!
//! Every routine operates on stack scalars, allocates nothing, never locks and
//! never panics. All transcendental math routes through [`bevy_math::ops`]
//! (libm-backed) rather than `f32` intrinsics, so results are bit-reproducible
//! across targets and can be golden-compared.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code**. The Fresnel approximations are the published Abramowitz and
//! Stegun formulas and the transition function is the textbook UTD definition.

use bevy_math::ops;

use core::f32::consts::{FRAC_PI_2, PI};

use prism_audio_core::math::Sample;

/// Crossover between the rational approximation (small argument) and the
/// asymptotic expansion (large argument) for the Fresnel integrals.
const FRESNEL_ASYMPTOTIC_CUTOFF: Sample = 4.0;

/// A complex value returned by the UTD [`transition`] function.
///
/// The real and imaginary parts are stored directly; [`TransitionValue::magnitude`]
/// and [`TransitionValue::phase`] derive the polar form.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TransitionValue {
    /// Real part of `F(X)`.
    pub re: Sample,
    /// Imaginary part of `F(X)`.
    pub im: Sample,
}

impl TransitionValue {
    /// The additive identity `0 + 0j`, returned when `X <= 0`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Returns the magnitude `|F(X)|`.
    #[must_use]
    pub fn magnitude(&self) -> Sample {
        ops::hypot(self.re, self.im)
    }

    /// Returns the phase `arg(F(X))` in radians, in `(-pi, pi]`.
    ///
    /// A zero value reports a phase of `0`.
    #[must_use]
    pub fn phase(&self) -> Sample {
        if self.re == 0.0 && self.im == 0.0 {
            0.0
        } else {
            ops::atan2(self.im, self.re)
        }
    }
}

/// Evaluates the Fresnel cosine and sine integrals `(C(z), S(z))`.
///
/// Both integrals use the `pi/2` normalisation, are odd in `z`, and converge
/// to `1/2` as `z -> +infinity`. The implementation blends the Abramowitz and
/// Stegun rational auxiliary functions for `|z| < 4` with the large-argument
/// asymptotic expansion for `|z| >= 4`; the worst-case absolute error stays
/// well below `3e-3` across the whole range.
///
/// Non-finite input returns `(0, 0)`.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::fresnel_transition::fresnel_integrals;
///
/// let (c, s) = fresnel_integrals(1.0);
/// assert!((c - 0.7799).abs() < 3e-3);
/// assert!((s - 0.4383).abs() < 3e-3);
///
/// // Odd symmetry.
/// let (cn, sn) = fresnel_integrals(-1.0);
/// assert!((cn + c).abs() < 1e-6);
/// assert!((sn + s).abs() < 1e-6);
/// ```
#[must_use]
pub fn fresnel_integrals(z: Sample) -> (Sample, Sample) {
    if !z.is_finite() {
        return (0.0, 0.0);
    }

    let az = ops::abs(z);
    let (f, g) = fresnel_auxiliary(az);

    let a = FRAC_PI_2 * az * az;
    let sin_a = ops::sin(a);
    let cos_a = ops::cos(a);

    let c = 0.5 + f * sin_a - g * cos_a;
    let s = 0.5 - f * cos_a - g * sin_a;

    if z < 0.0 {
        (-c, -s)
    } else {
        (c, s)
    }
}

/// The Abramowitz and Stegun auxiliary functions `f(z)` and `g(z)` for `z >= 0`.
///
/// For `z < 4` this uses the 7.3.32 / 7.3.33 rational approximations; for
/// `z >= 4` it switches to the asymptotic expansion, which is more accurate
/// there and avoids the rational form drifting.
fn fresnel_auxiliary(z: Sample) -> (Sample, Sample) {
    if z < FRESNEL_ASYMPTOTIC_CUTOFF {
        let z2 = z * z;
        let z3 = z2 * z;
        let f = (1.0 + 0.926 * z) / (2.0 + 1.792 * z + 3.104 * z2);
        let g = 1.0 / (2.0 + 4.142 * z + 3.492 * z2 + 6.670 * z3);
        (f, g)
    } else {
        let pi2 = PI * PI;
        let pi4 = pi2 * pi2;
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let f = (1.0 / (PI * z)) * (1.0 - 3.0 / (pi2 * z4) + 105.0 / (pi4 * z8));
        let g = (1.0 / (pi2 * z3)) * (1.0 - 15.0 / (pi2 * z4) + 945.0 / (pi4 * z8));
        (f, g)
    }
}

/// Evaluates the UTD transition function `F(X)`.
///
/// For `X <= 0` the result is [`TransitionValue::ZERO`]. For `X > 0` the
/// function returns the complex value whose magnitude approaches `1` and whose
/// phase approaches `0` as `X` grows, modelling the smooth recovery of
/// geometrical optics away from the shadow and reflection boundaries.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::fresnel_transition::transition;
///
/// // Deep lit/shadow region: F -> 1 with vanishing phase.
/// let far = transition(60.0);
/// assert!((far.magnitude() - 1.0).abs() < 0.05);
/// assert!(far.phase().abs() < 0.05);
///
/// // Near a boundary F collapses toward zero.
/// let near = transition(1e-4);
/// assert!(near.magnitude() < 0.05);
/// ```
#[must_use]
pub fn transition(x: Sample) -> TransitionValue {
    if !x.is_finite() || x <= 0.0 {
        return TransitionValue::ZERO;
    }

    let sx = ops::sqrt(x);
    let w = ops::sqrt(2.0 * x / PI);
    let (c, s) = fresnel_integrals(w);

    // Tail integral: sqrt(pi/2) * [ (1/2 - C(w)) - j (1/2 - S(w)) ].
    let k = ops::sqrt(FRAC_PI_2);
    let inner_re = k * (0.5 - c);
    let inner_im = -k * (0.5 - s);

    // 2j * sqrt(X) * exp(jX) = 2 * sqrt(X) * (j * exp(jX)); j*exp(jX) = -sin X + j cos X.
    let jexp_re = -ops::sin(x);
    let jexp_im = ops::cos(x);

    // Complex product (jexp) * (inner).
    let prod_re = jexp_re * inner_re - jexp_im * inner_im;
    let prod_im = jexp_re * inner_im + jexp_im * inner_re;

    let scale = 2.0 * sx;
    TransitionValue {
        re: scale * prod_re,
        im: scale * prod_im,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Sample = 3e-3;

    #[test]
    fn fresnel_at_origin_is_zero() {
        let (c, s) = fresnel_integrals(0.0);
        assert!(c.abs() < 1e-6);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn fresnel_known_values() {
        let (c, s) = fresnel_integrals(0.5);
        assert!((c - 0.4923).abs() < TOL, "C(0.5)={c}");
        assert!((s - 0.0647).abs() < TOL, "S(0.5)={s}");

        let (c, s) = fresnel_integrals(1.0);
        assert!((c - 0.7799).abs() < TOL, "C(1)={c}");
        assert!((s - 0.4383).abs() < TOL, "S(1)={s}");
    }

    #[test]
    fn fresnel_tends_to_half() {
        let (c, s) = fresnel_integrals(8.0);
        assert!((c - 0.5).abs() < 0.05, "C(8)={c}");
        assert!((s - 0.5).abs() < 0.05, "S(8)={s}");
    }

    #[test]
    fn fresnel_is_odd() {
        for &z in &[0.3, 0.9, 1.7, 3.3, 6.0] {
            let (cp, sp) = fresnel_integrals(z);
            let (cn, sn) = fresnel_integrals(-z);
            assert!((cp + cn).abs() < 1e-6);
            assert!((sp + sn).abs() < 1e-6);
        }
    }

    #[test]
    fn fresnel_non_finite_is_zero() {
        let (c, s) = fresnel_integrals(Sample::NAN);
        assert_eq!((c, s), (0.0, 0.0));
        let (c, s) = fresnel_integrals(Sample::INFINITY);
        assert_eq!((c, s), (0.0, 0.0));
    }

    #[test]
    fn transition_non_positive_is_zero() {
        assert_eq!(transition(0.0), TransitionValue::ZERO);
        assert_eq!(transition(-1.0), TransitionValue::ZERO);
        assert_eq!(transition(Sample::NAN), TransitionValue::ZERO);
    }

    #[test]
    fn transition_large_argument_recovers_optics() {
        let f = transition(80.0);
        assert!((f.magnitude() - 1.0).abs() < 0.05, "|F|={}", f.magnitude());
        assert!(f.phase().abs() < 0.05, "phase={}", f.phase());
    }

    #[test]
    fn transition_small_argument_asymptotics() {
        let x = 1e-4;
        let f = transition(x);
        let expected_mag = ops::sqrt(PI * x);
        assert!(
            (f.magnitude() - expected_mag).abs() < 2e-3,
            "|F|={} expected {expected_mag}",
            f.magnitude()
        );
        assert!((f.phase() - core::f32::consts::FRAC_PI_4).abs() < 0.1);
    }

    #[test]
    fn transition_is_finite_and_bounded() {
        let mut x = 0.01;
        while x < 100.0 {
            let f = transition(x);
            assert!(f.re.is_finite() && f.im.is_finite());
            assert!(f.magnitude() <= 1.3, "|F({x})|={}", f.magnitude());
            x *= 1.3;
        }
    }
}
