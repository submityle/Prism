//! Azimuthal scattering functions `N_R` / `N_TT` / `N_TRT` for the
//! Marschner/Chiang hair BSDF (CPU golden reference).
//!
//! Where the longitudinal term handles the inclination of light along the
//! fiber, the azimuthal term handles the angle *around* the fiber cross
//! section.  Light enters the circular cross section at a signed offset
//! `h in [-1, 1]`, refracts, bounces `p` times inside, and exits; the net
//! azimuthal deflection is the deterministic
//!
//! ```text
//! Phi(p, h) = 2 p * gamma_t(h) - 2 * gamma_o(h) + p * pi
//! ```
//!
//! with `gamma_o = asin(h)` the incident surface angle and `gamma_t` the
//! refracted angle through the Bravais *virtual* index of refraction.  Marschner
//! derived `Phi` from the cubic root of the azimuthal deflection; Chiang (2016),
//! *A Practical and Controllable Hair and Fur Model for Production Path
//! Tracing*, replaced Marschner's hard caustics with a **trimmed logistic**
//! roughening lobe `D_p` centred on `Phi(p, h)`, which is cheap, always
//! normalized, and controllable through a single azimuthal roughness `beta_n`.
//!
//! Each path order carries an *attenuation* `A_p` built from the dielectric
//! Fresnel reflectance and (for transmitted paths) the Beer-Lambert absorption
//! through the fiber interior, giving the RGB tint of hair.  The full azimuthal
//! response integrates the per-offset product `A_p(h) * D_p(phi - Phi(p, h))`
//! over `h`:
//!
//! ```text
//! N_p(phi) = (1/2) integral_{-1}^{1} A_p(h) * D_p(phi - Phi(p, h)) dh
//! ```
//!
//! # Conventions
//! * `no_std`: RGB stored as [`bevy_math::Vec3`]; all transcendentals via
//!   [`bevy_math::ops`] (never `f32::exp`), square roots via `f32::sqrt`.
//! * The index of refraction is clamped to `[1, 3]`, offsets to `[-1, 1]`,
//!   cosines to `[epsilon, 1]`, and every attenuation to `[0, 1]`, so grazing
//!   angles, total internal reflection, and `beta_n -> 0` can never divide by
//!   zero or emit `NaN`/`inf`.
//! * Every function is a deterministic, allocation-free pure function (no RNG,
//!   I/O, GPU, or global state); the `h` integral uses a fixed sample count.

use bevy_math::{ops, Vec3};
use core::f32::consts::PI;

/// Number of azimuthal lobes modelled: R (`p = 0`), TT (`p = 1`), TRT
/// (`p = 2`).  This matches the longitudinal [`LOBE_COUNT`].
///
/// [`LOBE_COUNT`]: crate::gi::hair_bsdf::longitudinal::LOBE_COUNT
pub const LOBE_COUNT: usize = 3;

/// `sqrt(pi / 8)`, the leading constant of Chiang's azimuthal roughness remap.
const SQRT_PI_OVER_8: f32 = 0.626_657_07;

/// Smallest cosine evaluated, keeping grazing Fresnel denominators finite.
const MIN_COS: f32 = 1.0e-4;

/// Smallest logistic scale evaluated, keeping `D_p` from collapsing to a delta
/// that would divide by zero.
const MIN_LOGISTIC_S: f32 = 1.0e-4;

/// Clamps an index of refraction to the physical hair range `[1, 3]`.
#[inline]
fn clamp_eta(eta: f32) -> f32 {
    if eta.is_finite() {
        eta.clamp(1.0, 3.0)
    } else {
        1.55
    }
}

/// Clamps a fiber offset `h` to `[-1, 1]`.
#[inline]
fn clamp_h(h: f32) -> f32 {
    if h.is_finite() {
        h.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a cosine to `[MIN_COS, 1]`.
#[inline]
fn clamp_cos(c: f32) -> f32 {
    if c.is_finite() {
        c.clamp(MIN_COS, 1.0)
    } else {
        MIN_COS
    }
}

/// `sqrt(max(0, 1 - x^2))` with a non-negative, finite guarantee.
#[inline]
fn safe_sqrt_complement(x: f32) -> f32 {
    (1.0 - x * x).max(0.0).sqrt()
}

/// `asin` with the argument clamped to `[-1, 1]` so it never returns `NaN`.
#[inline]
fn safe_asin(x: f32) -> f32 {
    ops::asin(x.clamp(-1.0, 1.0))
}

/// Unpolarised Fresnel reflectance of a smooth dielectric interface.
///
/// `cos_i` is the incident cosine (clamped to `[MIN_COS, 1]`) and `eta` the
/// relative index of refraction (outgoing / incoming).  Returns the average of
/// the s- and p-polarised intensity reflectances in `[0, 1]`, or `1` under
/// total internal reflection.
#[inline]
pub fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let eta = clamp_eta(eta);
    let cos_i = clamp_cos(cos_i);
    let sin_t2 = (1.0 - cos_i * cos_i) / (eta * eta);
    if sin_t2 >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin_t2).max(0.0).sqrt();
    let r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    (0.5 * (r_parl * r_parl + r_perp * r_perp)).clamp(0.0, 1.0)
}

/// Bravais *virtual* index of refraction `eta'(theta)` for a tilted fiber.
///
/// Because light strikes the cylinder at a longitudinal inclination, the
/// effective index seen in the normal plane differs from the material `eta`.
/// Returns the perpendicular virtual index
///
/// ```text
/// eta'(theta) = sqrt(eta^2 - sin^2(theta)) / cos(theta)
/// ```
///
/// clamped to a finite positive range.
#[inline]
pub fn bravais_virtual_eta(sin_theta: f32, cos_theta: f32, eta: f32) -> f32 {
    let eta = clamp_eta(eta);
    let cos_theta = clamp_cos(cos_theta.abs());
    let sin2 = (sin_theta * sin_theta).clamp(0.0, 1.0);
    let num = (eta * eta - sin2).max(0.0).sqrt();
    (num / cos_theta).clamp(1.0, 1.0e3)
}

/// Beer-Lambert transmittance of a single pass through the fiber interior.
///
/// `sigma_a` is the per-channel absorption coefficient (`>= 0`), and the path
/// length is `2 cos(gamma_t) / cos(theta_t)` fiber radii.  Returns the RGB
/// transmittance `exp(-sigma_a * path)`, each channel in `[0, 1]`.
#[inline]
pub fn transmittance(sigma_a: Vec3, cos_gamma_t: f32, cos_theta_t: f32) -> Vec3 {
    let cos_theta_t = clamp_cos(cos_theta_t);
    let cos_gamma_t = cos_gamma_t.clamp(0.0, 1.0);
    let path = 2.0 * cos_gamma_t / cos_theta_t;
    let sa = Vec3::new(sigma_a.x.max(0.0), sigma_a.y.max(0.0), sigma_a.z.max(0.0));
    Vec3::new(
        ops::exp(-sa.x * path).clamp(0.0, 1.0),
        ops::exp(-sa.y * path).clamp(0.0, 1.0),
        ops::exp(-sa.z * path).clamp(0.0, 1.0),
    )
}

/// Per-order attenuation terms `A_p` for `p = 0, 1, 2` (R, TT, TRT).
///
/// Follows Marschner's recurrence closed by the geometric tail of higher-order
/// internal reflections:
///
/// ```text
/// A_0 = f
/// A_1 = (1 - f)^2 * T
/// A_2 = (1 - f)^2 * f * T^2
/// ```
///
/// where `f` is the interface Fresnel reflectance and `T` the single-pass
/// absorption transmittance.  Each returned channel lies in `[0, 1]` and the
/// three orders together never exceed unit incident energy.
#[inline]
pub fn attenuation_ap(
    cos_theta_o: f32,
    sin_theta_o: f32,
    eta: f32,
    h: f32,
    sigma_a: Vec3,
) -> [Vec3; LOBE_COUNT] {
    let eta = clamp_eta(eta);
    let h = clamp_h(h);
    let cos_theta_o = clamp_cos(cos_theta_o.abs());

    // Refracted longitudinal angle (Snell on the inclination).
    let sin_theta_t = (sin_theta_o / eta).clamp(-1.0, 1.0);
    let cos_theta_t = safe_sqrt_complement(sin_theta_t);

    // Bravais virtual index and refracted azimuthal angle.
    let etap = bravais_virtual_eta(sin_theta_o, cos_theta_o, eta);
    let sin_gamma_t = (h / etap).clamp(-1.0, 1.0);
    let cos_gamma_t = safe_sqrt_complement(sin_gamma_t);

    // Fresnel at the first intersection; cos measured in the fiber frame.
    let cos_gamma_o = safe_sqrt_complement(h);
    let cos_at_surface = (cos_theta_o * cos_gamma_o).clamp(MIN_COS, 1.0);
    let f = fresnel_dielectric(cos_at_surface, eta);

    let t = transmittance(sigma_a, cos_gamma_t, cos_theta_t);
    let one_minus_f = (1.0 - f).max(0.0);
    let one_minus_f2 = one_minus_f * one_minus_f;

    let a0 = Vec3::splat(f);
    let a1 = t * one_minus_f2;
    let a2 = t * t * (one_minus_f2 * f);

    [clamp_vec01(a0), clamp_vec01(a1), clamp_vec01(a2)]
}

/// Clamps every channel of an RGB attenuation to `[0, 1]`.
#[inline]
fn clamp_vec01(v: Vec3) -> Vec3 {
    Vec3::new(
        if v.x.is_finite() { v.x.clamp(0.0, 1.0) } else { 0.0 },
        if v.y.is_finite() { v.y.clamp(0.0, 1.0) } else { 0.0 },
        if v.z.is_finite() { v.z.clamp(0.0, 1.0) } else { 0.0 },
    )
}

/// Deterministic azimuthal deflection `Phi(p, h)` of a path of order `p`.
///
/// ```text
/// Phi = 2 p * gamma_t - 2 * gamma_o + p * pi
/// ```
///
/// where `gamma_o = asin(h)` and `gamma_t = asin(h / eta')`.
#[inline]
pub fn azimuthal_phi(p: usize, gamma_o: f32, gamma_t: f32) -> f32 {
    let p = p as f32;
    2.0 * p * gamma_t - 2.0 * gamma_o + p * PI
}

/// Logistic probability density `l(x; s) = e^{-|x|/s} / (s (1 + e^{-|x|/s})^2)`.
#[inline]
pub fn logistic(x: f32, s: f32) -> f32 {
    let s = s.max(MIN_LOGISTIC_S);
    let e = ops::exp(-x.abs() / s);
    let d = 1.0 + e;
    (e / (s * d * d)).max(0.0)
}

/// Logistic cumulative distribution `L(x; s) = 1 / (1 + e^{-x/s})`.
#[inline]
pub fn logistic_cdf(x: f32, s: f32) -> f32 {
    let s = s.max(MIN_LOGISTIC_S);
    1.0 / (1.0 + ops::exp(-x / s))
}

/// Trimmed logistic density on `[a, b]`: the logistic renormalized so it
/// integrates to one over the finite support, used by Chiang's `D_p`.
#[inline]
pub fn trimmed_logistic(x: f32, s: f32, a: f32, b: f32) -> f32 {
    let s = s.max(MIN_LOGISTIC_S);
    let norm = (logistic_cdf(b, s) - logistic_cdf(a, s)).max(f32::MIN_POSITIVE);
    (logistic(x, s) / norm).max(0.0)
}

/// Maps the user azimuthal roughness `beta_n in (0, 1]` to the logistic scale
/// `s` of Chiang's `D_p`, via the perceptual remap
///
/// ```text
/// s = sqrt(pi/8) * ( 0.265 beta + 1.194 beta^2 + 5.372 beta^22 ).
/// ```
#[inline]
pub fn azimuthal_roughness_s(beta_n: f32) -> f32 {
    let b = if beta_n.is_finite() {
        beta_n.clamp(1.0e-3, 1.0)
    } else {
        1.0e-3
    };
    let b2 = b * b;
    let b22 = ops::powf(b, 22.0);
    (SQRT_PI_OVER_8 * (0.265 * b + 1.194 * b2 + 5.372 * b22)).max(MIN_LOGISTIC_S)
}

/// Chiang's azimuthal distribution `D_p(phi)` for lobe `p` at offset angles
/// `gamma_o` / `gamma_t`.
///
/// The relative azimuth `phi - Phi(p, h)` is wrapped into `[-pi, pi]` before
/// being fed to the trimmed logistic of scale `s`, giving a normalized,
/// non-negative roughening lobe.
#[inline]
pub fn azimuthal_np(phi: f32, p: usize, s: f32, gamma_o: f32, gamma_t: f32) -> f32 {
    let mut dphi = phi - azimuthal_phi(p, gamma_o, gamma_t);
    // Wrap into [-pi, pi].
    dphi %= 2.0 * PI;
    if dphi > PI {
        dphi -= 2.0 * PI;
    } else if dphi < -PI {
        dphi += 2.0 * PI;
    }
    trimmed_logistic(dphi, s, -PI, PI)
}

/// Full azimuthal scattering `N_p(phi)` for lobe `p`, integrating the per-offset
/// attenuated distribution over the fiber width `h in [-1, 1]`.
///
/// ```text
/// N_p(phi) = (1/2) integral_{-1}^{1} A_p(h) * D_p(phi - Phi(p, h)) dh
/// ```
///
/// Returns an RGB [`Vec3`]; the `A_p` factor carries the dielectric tint and the
/// `D_p` factor the azimuthal spread.  `samples` fixes the midpoint-rule
/// resolution of the `h` integral (clamped to at least 2).
#[inline]
pub fn azimuthal_scattering(
    phi: f32,
    p: usize,
    cos_theta_o: f32,
    sin_theta_o: f32,
    eta: f32,
    sigma_a: Vec3,
    beta_n: f32,
    samples: usize,
) -> Vec3 {
    if p >= LOBE_COUNT {
        return Vec3::ZERO;
    }
    let eta = clamp_eta(eta);
    let cos_theta_o = clamp_cos(cos_theta_o.abs());
    let s = azimuthal_roughness_s(beta_n);
    let n = samples.max(2);
    let dh = 2.0 / n as f32;
    let mut acc = Vec3::ZERO;
    for k in 0..n {
        let h = -1.0 + (k as f32 + 0.5) * dh;
        let etap = bravais_virtual_eta(sin_theta_o, cos_theta_o, eta);
        let gamma_o = safe_asin(h);
        let gamma_t = safe_asin((h / etap).clamp(-1.0, 1.0));
        let ap = attenuation_ap(cos_theta_o, sin_theta_o, eta, h, sigma_a)[p];
        let d = azimuthal_np(phi, p, s, gamma_o, gamma_t);
        acc += ap * d;
    }
    // (1/2) * integral => multiply the midpoint sum (sum * dh) by 1/2.
    let scaled = acc * (0.5 * dh);
    clamp_vec_non_negative(scaled)
}

/// Clamps every channel of an RGB value to be finite and non-negative.
#[inline]
fn clamp_vec_non_negative(v: Vec3) -> Vec3 {
    Vec3::new(
        if v.x.is_finite() { v.x.max(0.0) } else { 0.0 },
        if v.y.is_finite() { v.y.max(0.0) } else { 0.0 },
        if v.z.is_finite() { v.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_is_bounded_and_grazing_reflects_fully() {
        for ci in 1..=20 {
            let cos_i = ci as f32 / 20.0;
            let f = fresnel_dielectric(cos_i, 1.55);
            assert!(f.is_finite() && (0.0..=1.0).contains(&f), "f={f}");
        }
        // Grazing incidence reflects (almost) everything.
        let grazing = fresnel_dielectric(MIN_COS, 1.55);
        assert!(grazing > 0.9, "grazing={grazing}");
        // Normal incidence matches the Schlick base reflectance (eta=1.55 ->
        // R0 ~= 0.0465).
        let normal = fresnel_dielectric(1.0, 1.55);
        assert!((normal - 0.046_5).abs() < 2.0e-3, "normal={normal}");
    }

    #[test]
    fn attenuation_terms_are_bounded_and_sum_within_energy() {
        let sigma_a = Vec3::splat(0.2);
        for hk in -9..=9 {
            let h = hk as f32 / 10.0;
            let ap = attenuation_ap(0.9, 0.3, 1.55, h, sigma_a);
            let mut total = Vec3::ZERO;
            for a in ap {
                assert!(a.x >= 0.0 && a.x <= 1.0, "a.x={}", a.x);
                assert!(a.y >= 0.0 && a.y <= 1.0, "a.y={}", a.y);
                assert!(a.z >= 0.0 && a.z <= 1.0, "a.z={}", a.z);
                total += a;
            }
            // R + TT + TRT energy never exceeds the incident unit flux.
            assert!(total.x <= 1.0 + 1.0e-4, "total.x={}", total.x);
            assert!(total.y <= 1.0 + 1.0e-4, "total.y={}", total.y);
            assert!(total.z <= 1.0 + 1.0e-4, "total.z={}", total.z);
        }
    }

    #[test]
    fn zero_absorption_transmits_fully() {
        let t = transmittance(Vec3::ZERO, 0.8, 0.9);
        assert!((t - Vec3::ONE).length() < 1.0e-6, "t={t:?}");
        // Larger sigma_a strictly reduces transmittance.
        let t_dark = transmittance(Vec3::splat(1.0), 0.8, 0.9);
        assert!(t_dark.x < t.x);
    }

    #[test]
    fn logistic_integrates_to_one_on_support() {
        // The trimmed logistic must integrate to one over [-pi, pi].
        let s = 0.3_f32;
        let n = 4000;
        let dx = 2.0 * PI / n as f32;
        let mut sum = 0.0_f32;
        for k in 0..n {
            let x = -PI + (k as f32 + 0.5) * dx;
            sum += trimmed_logistic(x, s, -PI, PI) * dx;
        }
        assert!((sum - 1.0).abs() < 1.0e-2, "integral={sum}");
    }

    #[test]
    fn azimuthal_np_is_symmetric_about_phi_peak() {
        // D_p is an even function of the relative azimuth, so equal offsets on
        // either side of Phi give equal density.
        let s = azimuthal_roughness_s(0.3);
        let (gamma_o, gamma_t) = (0.1_f32, 0.05_f32);
        let center = azimuthal_phi(0, gamma_o, gamma_t);
        let left = azimuthal_np(center - 0.4, 0, s, gamma_o, gamma_t);
        let right = azimuthal_np(center + 0.4, 0, s, gamma_o, gamma_t);
        assert!((left - right).abs() < 1.0e-5, "left={left} right={right}");
    }

    #[test]
    fn azimuthal_phi_matches_closed_form() {
        // p = 0 reflection: Phi = -2 gamma_o, independent of gamma_t.
        let phi0 = azimuthal_phi(0, 0.3, 0.1);
        assert!((phi0 + 0.6).abs() < 1.0e-6, "phi0={phi0}");
        // p = 1 transmission adds 2 gamma_t + pi.
        let phi1 = azimuthal_phi(1, 0.3, 0.1);
        assert!((phi1 - (2.0 * 0.1 - 0.6 + PI)).abs() < 1.0e-6, "phi1={phi1}");
    }

    #[test]
    fn azimuthal_scattering_is_non_negative_finite_and_deterministic() {
        let sigma_a = Vec3::new(0.1, 0.3, 0.7);
        for p in 0..LOBE_COUNT {
            for pk in 0..12 {
                let phi = -PI + pk as f32 * (2.0 * PI / 12.0);
                let n = azimuthal_scattering(phi, p, 0.95, 0.2, 1.55, sigma_a, 0.3, 64);
                assert!(n.x >= 0.0 && n.x.is_finite(), "n={n:?}");
                assert!(n.y >= 0.0 && n.y.is_finite(), "n={n:?}");
                assert!(n.z >= 0.0 && n.z.is_finite(), "n={n:?}");
            }
        }
        let a = azimuthal_scattering(0.5, 1, 0.95, 0.2, 1.55, sigma_a, 0.3, 64);
        let b = azimuthal_scattering(0.5, 1, 0.95, 0.2, 1.55, sigma_a, 0.3, 64);
        assert_eq!(a, b);
    }

    #[test]
    fn azimuthal_integral_vs_direct_sum_agree() {
        // Integrating N_p over phi equals the h-averaged attenuation, because
        // D_p integrates to one over the azimuth. Cross-check the two routes.
        let sigma_a = Vec3::splat(0.25);
        let p = 1;
        let (cos_o, sin_o, eta, beta_n) = (0.95_f32, 0.2_f32, 1.55_f32, 0.3_f32);

        // Route A: integrate N_p(phi) over the full azimuth.
        let nphi = 240;
        let dphi = 2.0 * PI / nphi as f32;
        let mut route_a = Vec3::ZERO;
        for k in 0..nphi {
            let phi = -PI + (k as f32 + 0.5) * dphi;
            route_a += azimuthal_scattering(phi, p, cos_o, sin_o, eta, sigma_a, beta_n, 64) * dphi;
        }

        // Route B: average A_p over h directly (the azimuth-integrated energy).
        let nh = 64;
        let dh = 2.0 / nh as f32;
        let mut route_b = Vec3::ZERO;
        for k in 0..nh {
            let h = -1.0 + (k as f32 + 0.5) * dh;
            route_b += attenuation_ap(cos_o, sin_o, eta, h, sigma_a)[p] * (0.5 * dh);
        }

        assert!((route_a - route_b).length() < 2.0e-2, "a={route_a:?} b={route_b:?}");
    }

    #[test]
    fn degenerate_inputs_never_produce_nan() {
        let ap = attenuation_ap(f32::NAN, f32::INFINITY, f32::NAN, 5.0, Vec3::splat(-1.0));
        for a in ap {
            assert!(a.x.is_finite() && a.y.is_finite() && a.z.is_finite());
        }
        let n = azimuthal_scattering(f32::NAN, 0, 0.0, 2.0, 0.0, Vec3::splat(f32::NAN), -1.0, 8);
        assert!(n.x.is_finite() && n.y.is_finite() && n.z.is_finite());
    }
}
