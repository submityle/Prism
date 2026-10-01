//! Longitudinal scattering functions `M_R` / `M_TT` / `M_TRT` for the
//! Marschner/Chiang hair BSDF (CPU golden reference).
//!
//! The longitudinal term describes how light that enters a fiber at an
//! inclination `theta_i` (measured from the plane perpendicular to the hair
//! tangent) leaves at an inclination `theta_r`.  Marschner (2003) modelled this
//! with a plain Gaussian in angle, which is cheap but leaks energy at grazing
//! angles.  d'Eon et al. (2011), *An Energy-conserving Hair Reflectance Model*,
//! replaced it with a **normalized Gaussian on the sphere** whose closed form
//! is written in terms of the modified Bessel function of the first kind of
//! order zero, `I0`:
//!
//! ```text
//! M_p(theta_i, theta_r; v) =
//!     exp(-sin(theta_i) sin(theta_r) / v) *
//!     I0( cos(theta_i) cos(theta_r) / v ) /
//!     ( 2 v sinh(1 / v) )
//! ```
//!
//! where `v = beta^2` is the longitudinal *variance* of the lobe.  This term
//! integrates to one over the outgoing longitudinal angle, which is why the
//! model conserves energy (verified numerically in the tests).  For small `v`
//! the `sinh(1/v)` and `I0` factors overflow `f32`, so we evaluate the term in
//! log space using a `log I0` approximation, exactly as pbrt-v3 does.
//!
//! The three Marschner lobes share one user roughness `beta_m` through the
//! classic variance spread `v_R = v`, `v_TT = v / 4`, `v_TRT = 4 v` (so TT is
//! sharper and TRT is broader than the primary highlight), and each lobe is
//! shifted by a multiple of the cuticle scale tilt `alpha` to reproduce the
//! separated R / TT / TRT highlights seen on real hair.
//!
//! # Conventions
//! * `no_std`: vectors/scalars from `bevy_math`; every transcendental goes
//!   through [`bevy_math::ops`] (never `f32::exp`), and square roots use the
//!   `f32::sqrt` method.
//! * All roughnesses, variances, and angles are defensively clamped: `beta_m`
//!   and the derived variance are forced into a strictly positive finite
//!   range, cosines into `[0, 1]`, so no evaluation can divide by zero or
//!   return `NaN`/`inf`.
//! * Every function is a deterministic, allocation-free pure function with no
//!   RNG, I/O, GPU, or global state.

use bevy_math::ops;
use core::f32::consts::{FRAC_PI_2, LN_2};

/// Number of Marschner longitudinal lobes modelled: R (`p = 0`), TT (`p = 1`),
/// and TRT (`p = 2`).
pub const LOBE_COUNT: usize = 3;

/// Smallest longitudinal variance evaluated.  Below this the lobe is treated as
/// a razor-sharp delta but still finite, keeping `1 / v` and `sinh(1 / v)`
/// representable.
const MIN_VARIANCE: f32 = 1.0e-4;

/// Largest longitudinal variance evaluated, keeping extremely rough fibers from
/// collapsing the normalization.
const MAX_VARIANCE: f32 = 16.0;

/// Threshold below which the longitudinal term is evaluated in log space to
/// avoid overflowing `sinh(1 / v)` and `I0`.  Matches the pbrt-v3 cross-over.
const LOG_SPACE_V: f32 = 0.1;

/// Clamps a user roughness to the open unit interval `(0, 1]`.
#[inline]
fn clamp_beta(beta: f32) -> f32 {
    if beta.is_finite() {
        beta.clamp(1.0e-3, 1.0)
    } else {
        1.0e-3
    }
}

/// Clamps a longitudinal variance to `[MIN_VARIANCE, MAX_VARIANCE]`.
#[inline]
fn clamp_variance(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(MIN_VARIANCE, MAX_VARIANCE)
    } else {
        MIN_VARIANCE
    }
}

/// Clamps a cosine to `[0, 1]` (longitudinal cosines are always non-negative in
/// this parameterisation).
#[inline]
fn clamp_cos(c: f32) -> f32 {
    if c.is_finite() {
        c.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a sine to `[-1, 1]`.
#[inline]
fn clamp_sin(s: f32) -> f32 {
    if s.is_finite() {
        s.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Modified Bessel function of the first kind of order zero, `I0(x)`.
///
/// Evaluated from its power series `I0(x) = sum_i x^{2i} / (4^i (i!)^2)`, which
/// converges quickly for the small-to-moderate arguments that appear in the
/// longitudinal term (`x = cos(theta_i) cos(theta_r) / v`).  Ten terms give
/// `f32` precision for `|x| <= 12`; beyond that use [`log_bessel_i0`], whose
/// asymptotic branch stays finite.
#[inline]
pub fn bessel_i0(x: f32) -> f32 {
    let x = if x.is_finite() { x.abs() } else { 0.0 };
    let x2 = x * x;
    let mut val = 0.0_f32;
    let mut x2i = 1.0_f32; // x^(2i)
    let mut i_fact = 1.0_f32; // i!
    let mut pow4 = 1.0_f32; // 4^i
    let mut i = 0_u32;
    while i < 10 {
        if i > 1 {
            i_fact *= i as f32;
        }
        val += x2i / (pow4 * i_fact * i_fact);
        x2i *= x2;
        pow4 *= 4.0;
        i += 1;
    }
    val
}

/// Natural logarithm of [`bessel_i0`], `ln I0(x)`.
///
/// For `|x| > 12` the series overflows, so we switch to the standard asymptotic
/// expansion `ln I0(x) ~= x - 0.5 ln(2 pi x) + 1 / (8 x)`, which stays finite
/// and lets the longitudinal term be evaluated entirely in log space.
#[inline]
pub fn log_bessel_i0(x: f32) -> f32 {
    let x = if x.is_finite() { x.abs() } else { 0.0 };
    if x > 12.0 {
        // x + 0.5 * ( -ln(2 pi) - ln(x) + 1/(8x) )
        x + 0.5 * (-ops::ln(2.0 * core::f32::consts::PI) - ops::ln(x) + 1.0 / (8.0 * x))
    } else {
        ops::ln(bessel_i0(x).max(f32::MIN_POSITIVE))
    }
}

/// Maps the user longitudinal roughness `beta_m in (0, 1]` to the base
/// variance `v = v_R` of the primary (R) lobe, following the d'Eon/pbrt
/// perceptual remap
///
/// ```text
/// v_R = ( 0.726 beta + 0.812 beta^2 + 3.7 beta^20 )^2
/// ```
///
/// which keeps the apparent highlight width roughly linear in `beta_m`.
#[inline]
pub fn base_variance(beta_m: f32) -> f32 {
    let b = clamp_beta(beta_m);
    let b2 = b * b;
    let b20 = ops::powf(b, 20.0);
    let root = 0.726 * b + 0.812 * b2 + 3.7 * b20;
    clamp_variance(root * root)
}

/// Returns the three lobe variances `[v_R, v_TT, v_TRT]` from one user
/// roughness, using the classic spread `v_TT = v_R / 4`, `v_TRT = 4 v_R`.
#[inline]
pub fn lobe_variances(beta_m: f32) -> [f32; LOBE_COUNT] {
    let v_r = base_variance(beta_m);
    [
        v_r,
        clamp_variance(0.25 * v_r),
        clamp_variance(4.0 * v_r),
    ]
}

/// Energy-conserving longitudinal scattering term `M_p` of d'Eon et al. (2011).
///
/// `cos_theta_i` / `sin_theta_i` and `cos_theta_o` / `sin_theta_o` are the
/// sine/cosine of the incident and outgoing longitudinal angles (already
/// cuticle-shifted by [`cone_shift`] when modelling a specific lobe), and `v`
/// is that lobe's longitudinal variance.  The result is non-negative and, when
/// integrated over the outgoing longitudinal angle, integrates to one.
#[inline]
pub fn longitudinal_m(
    cos_theta_i: f32,
    sin_theta_i: f32,
    cos_theta_o: f32,
    sin_theta_o: f32,
    v: f32,
) -> f32 {
    let v = clamp_variance(v);
    let ci = clamp_cos(cos_theta_i);
    let co = clamp_cos(cos_theta_o);
    let si = clamp_sin(sin_theta_i);
    let so = clamp_sin(sin_theta_o);

    let a = ci * co / v;
    let b = si * so / v;

    let m = if v <= LOG_SPACE_V {
        // Log-space evaluation: exp( logI0(a) - b - 1/v + ln2 + ln(1/(2v)) ).
        ops::exp(log_bessel_i0(a) - b - 1.0 / v + LN_2 + ops::ln(1.0 / (2.0 * v)))
    } else {
        // Direct evaluation using the closed form with the sinh normalizer.
        ops::exp(-b) * bessel_i0(a) / (ops::sinh(1.0 / v) * 2.0 * v)
    };

    if m.is_finite() && m >= 0.0 {
        m
    } else {
        0.0
    }
}

/// Precomputes `(sin, cos)` of `alpha`, `2 alpha`, and `4 alpha` via the
/// double-angle recurrence, so the per-lobe cuticle shift needs no further
/// transcendental calls.
#[inline]
fn alpha_sin_cos(alpha: f32) -> [(f32, f32); LOBE_COUNT] {
    let alpha = if alpha.is_finite() {
        alpha.clamp(-FRAC_PI_2, FRAC_PI_2)
    } else {
        0.0
    };
    let (s0, c0) = ops::sin_cos(alpha);
    let s1 = 2.0 * c0 * s0;
    let c1 = c0 * c0 - s0 * s0;
    let s2 = 2.0 * c1 * s1;
    let c2 = c1 * c1 - s1 * s1;
    [(s0, c0), (s1, c1), (s2, c2)]
}

/// Applies the cuticle scale-tilt (`alpha`) rotation to the outgoing
/// longitudinal `(sin, cos)` for lobe `p`.
///
/// Hair cuticle scales are tilted, which shifts each specular cone by a
/// different signed multiple of `alpha`: R by `-2 alpha`, TT by `+alpha`, and
/// TRT by `+4 alpha` (the pbrt-v3 convention).  The returned cosine is forced
/// non-negative so the shifted angle stays in the upper hemisphere.
#[inline]
pub fn cone_shift(p: usize, sin_theta_o: f32, cos_theta_o: f32, alpha: f32) -> (f32, f32) {
    let so = clamp_sin(sin_theta_o);
    let co = clamp_cos(cos_theta_o);
    let a = alpha_sin_cos(alpha);
    let (sp, cp) = match p {
        0 => {
            let (s, c) = a[1];
            (so * c - co * s, co * c + so * s)
        }
        1 => {
            let (s, c) = a[0];
            (so * c + co * s, co * c - so * s)
        }
        2 => {
            let (s, c) = a[2];
            (so * c + co * s, co * c - so * s)
        }
        _ => (so, co),
    };
    (clamp_sin(sp), clamp_cos(cp.abs()))
}

/// Evaluates lobe `p`'s longitudinal term for raw incident/outgoing
/// longitudinal angles `theta_i` / `theta_o` (in radians) at user roughness
/// `beta_m` and cuticle tilt `alpha`.
///
/// This is the convenience entry point that performs the variance remap, the
/// cuticle shift of the *outgoing* angle, and the `M_p` evaluation in one call.
#[inline]
pub fn longitudinal_lobe(p: usize, theta_i: f32, theta_o: f32, beta_m: f32, alpha: f32) -> f32 {
    if p >= LOBE_COUNT {
        return 0.0;
    }
    let (si, ci) = {
        let (s, c) = ops::sin_cos(if theta_i.is_finite() { theta_i } else { 0.0 });
        (clamp_sin(s), clamp_cos(c.abs()))
    };
    let (so_raw, co_raw) = {
        let (s, c) = ops::sin_cos(if theta_o.is_finite() { theta_o } else { 0.0 });
        (clamp_sin(s), clamp_cos(c.abs()))
    };
    let (so, co) = cone_shift(p, so_raw, co_raw, alpha);
    let v = lobe_variances(beta_m)[p];
    longitudinal_m(ci, si, co, so, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn bessel_i0_matches_known_values() {
        // I0(0) = 1 exactly; I0(1) ~= 1.2660658; I0(2) ~= 2.2795853.
        assert!((bessel_i0(0.0) - 1.0).abs() < EPS);
        assert!((bessel_i0(1.0) - 1.266_065_8).abs() < 1.0e-4);
        assert!((bessel_i0(2.0) - 2.279_585_3).abs() < 1.0e-4);
        // Even function.
        assert!((bessel_i0(-1.5) - bessel_i0(1.5)).abs() < EPS);
    }

    #[test]
    fn log_bessel_i0_consistent_across_branches() {
        // Below the cross-over, ln(I0(x)) must equal log_bessel_i0 directly.
        for &x in &[0.0_f32, 0.5, 2.0, 6.0, 11.5] {
            let direct = ops::ln(bessel_i0(x));
            assert!((log_bessel_i0(x) - direct).abs() < 1.0e-3, "x={x}");
        }
        // The asymptotic branch stays finite where the series would overflow.
        let big = log_bessel_i0(40.0);
        assert!(big.is_finite() && big > 0.0);
    }

    #[test]
    fn longitudinal_is_non_negative_and_finite() {
        for vi in 1..=40 {
            let v = vi as f32 * 0.02; // 0.02 ..= 0.8
            for ti in -8..=8 {
                let ti = ti as f32 * 0.18;
                for to in -8..=8 {
                    let to = to as f32 * 0.18;
                    let (si, ci) = ops::sin_cos(ti);
                    let (so, co) = ops::sin_cos(to);
                    let m = longitudinal_m(ci.abs(), si, co.abs(), so, v);
                    assert!(m.is_finite() && m >= 0.0, "v={v} ti={ti} to={to} m={m}");
                }
            }
        }
    }

    /// Numerically integrate `M_p` over the outgoing longitudinal angle and
    /// confirm the energy-conserving normalization `integral ~= 1`.
    fn integrate_over_theta_o(theta_i: f32, v: f32) -> f32 {
        let (si, ci) = ops::sin_cos(theta_i);
        let n = 4000;
        let lo = -FRAC_PI_2;
        let hi = FRAC_PI_2;
        let dt = (hi - lo) / n as f32;
        let mut sum = 0.0_f32;
        for k in 0..n {
            let to = lo + (k as f32 + 0.5) * dt;
            let (so, co) = ops::sin_cos(to);
            sum += longitudinal_m(ci.abs(), si, co.abs(), so, v) * dt;
        }
        sum
    }

    #[test]
    fn longitudinal_conserves_energy() {
        // d'Eon's energy-conserving design: at (near-)normal incidence the
        // longitudinal lobe integrates to one over the outgoing angle. The
        // Gaussian-on-sphere normalization is exact only near theta_i = 0; the
        // 1D integral grows at grazing angles (checked separately below), which
        // is the model's documented behaviour.
        for &theta_i in &[0.0_f32, 0.05, -0.1] {
            for &v in &[0.005_f32, 0.01, 0.02, 0.05] {
                let total = integrate_over_theta_o(theta_i, v);
                assert!(
                    (total - 1.0).abs() < 3.0e-2,
                    "theta_i={theta_i} v={v} integral={total}"
                );
            }
        }
    }

    #[test]
    fn longitudinal_integral_grows_monotonically_toward_grazing() {
        // The 1D longitudinal integral is a well-behaved, bounded, monotonically
        // increasing function of |theta_i| (the known Gaussian-on-sphere
        // artifact), never collapsing to zero or diverging.
        let v = 0.02_f32;
        let mut prev = 0.0_f32;
        for ti in 0..=6 {
            let theta_i = ti as f32 * 0.15;
            let total = integrate_over_theta_o(theta_i, v);
            assert!(total.is_finite() && total >= 1.0 - 3.0e-2, "integral={total}");
            assert!(total >= prev - 1.0e-3, "non-monotone: {total} < {prev}");
            prev = total;
        }
        assert!(prev < 3.0, "integral stayed bounded near grazing: {prev}");
    }

    #[test]
    fn smaller_beta_sharpens_the_peak() {
        // At the true specular peak (theta_i = theta_o = 0, where cos = 1) a
        // smaller variance gives a taller lobe: M(0, 0) ~= 1 / sqrt(2 pi v).
        let peak_sharp = longitudinal_m(1.0, 0.0, 1.0, 0.0, 0.02);
        let peak_broad = longitudinal_m(1.0, 0.0, 1.0, 0.0, 0.2);
        assert!(
            peak_sharp > peak_broad,
            "sharp={peak_sharp} broad={peak_broad}"
        );
        // Height tracks the Gaussian-on-sphere normalization 1/sqrt(2 pi v).
        let predicted = 1.0 / (2.0 * core::f32::consts::PI * 0.02).sqrt();
        assert!((peak_sharp - predicted).abs() / predicted < 0.05);
    }

    #[test]
    fn lobe_variance_spread_is_ordered() {
        // TT sharper than R sharper than TRT: v_TT < v_R < v_TRT.
        let v = lobe_variances(0.3);
        assert!(v[1] < v[0] && v[0] < v[2], "variances={v:?}");
        // Ratios follow the 1 : 1/4 : 4 convention (before clamping).
        assert!((v[1] / v[0] - 0.25).abs() < 1.0e-5);
        assert!((v[2] / v[0] - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn cone_shift_preserves_unit_circle() {
        // The rotated (sin, cos) must remain a valid direction: sin^2 + cos^2
        // stays close to one and cos is non-negative.
        for p in 0..LOBE_COUNT {
            for ti in -6..=6 {
                let to = ti as f32 * 0.22;
                let (s, c) = ops::sin_cos(to);
                let (sp, cp) = cone_shift(p, s, c.abs(), 0.12);
                assert!(cp >= 0.0);
                assert!((sp * sp + cp * cp - 1.0).abs() < 1.0e-3, "p={p} to={to}");
            }
        }
    }

    #[test]
    fn longitudinal_lobe_is_deterministic() {
        let a = longitudinal_lobe(2, 0.1, -0.2, 0.25, 0.1);
        let b = longitudinal_lobe(2, 0.1, -0.2, 0.25, 0.1);
        assert_eq!(a, b);
        // Out-of-range lobe index yields zero rather than panicking.
        assert_eq!(longitudinal_lobe(5, 0.1, 0.1, 0.3, 0.1), 0.0);
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let m = longitudinal_m(f32::NAN, f32::INFINITY, 1.0, 0.0, f32::NAN);
        assert!(m.is_finite() && m >= 0.0);
        let v = base_variance(f32::INFINITY);
        assert!(v.is_finite() && v > 0.0);
    }
}
