//! Marschner/Chiang hair scattering BSDF (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! Implements the Marschner R/TT/TRT longitudinal-azimuthal hair model with
//! Chiang (2016) azimuthal roughness and physically-based absorption.
//!
//! The full fiber BSDF is the sum over the three scattering orders of a
//! separable product of a longitudinal term `M_p` (how light tilts along the
//! fiber) and an azimuthal term `N_p` (how it bends around the fiber):
//!
//! ```text
//! f(theta_i, theta_r, phi) = sum_{p in {R, TT, TRT}} M_p(theta_i, theta_r) * N_p(phi)
//! ```
//!
//! * [`longitudinal`] — energy-conserving `M_R` / `M_TT` / `M_TRT` (d'Eon
//!   2011), the modified Bessel `I0`, the variance spread, and the cuticle
//!   scale-tilt (`alpha`) cone shift.
//! * [`azimuthal`] — `N_R` / `N_TT` / `N_TRT` with dielectric Fresnel, the
//!   Bravais virtual IOR, Beer-Lambert absorption attenuation `A_p`, and
//!   Chiang's trimmed-logistic azimuthal roughening `D_p`.
//! * [`absorption`] — the color -> `sigma_a` inversion and the
//!   eumelanin/pheomelanin pigment parameterisation.

pub mod absorption;
pub mod azimuthal;
pub mod longitudinal;

pub use absorption::{color_to_sigma_a, melanin_to_sigma_a, sigma_a_to_color};
pub use azimuthal::azimuthal_scattering;
pub use longitudinal::longitudinal_lobe;

use bevy_math::{ops, Vec3};

/// Number of scattering orders summed by the fiber BSDF: R, TT, TRT.
pub const LOBE_COUNT: usize = longitudinal::LOBE_COUNT;

/// Default fixed-fiber-offset integration resolution for the azimuthal term.
const DEFAULT_H_SAMPLES: usize = 64;

/// Appearance parameters of a single hair fiber, laid out as plain `f32` fields
/// (and an RGB [`Vec3`]) to mirror the GPU twin's uniform block.
///
/// # Conventions
/// * `eta` is clamped to the physical hair range `[1, 3]` on use.
/// * `beta_m` (longitudinal) and `beta_n` (azimuthal) roughnesses are clamped
///   to `(0, 1]`.
/// * `sigma_a` is the per-channel absorption coefficient (`>= 0`); build it
///   from a target color via [`color_to_sigma_a`] or from pigment
///   concentrations via [`melanin_to_sigma_a`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairParams {
    /// Index of refraction of the fiber relative to the surrounding medium.
    pub eta: f32,
    /// Per-channel RGB absorption coefficient of the fiber interior.
    pub sigma_a: Vec3,
    /// Longitudinal roughness controlling the `M_p` lobe widths.
    pub beta_m: f32,
    /// Azimuthal roughness controlling the `N_p` (Chiang `D_p`) spread.
    pub beta_n: f32,
    /// Cuticle scale-tilt angle `alpha` (radians) separating the R/TT/TRT
    /// highlights.
    pub alpha: f32,
}

impl Default for HairParams {
    /// Brown hair defaults: `eta = 1.55`, a moderate pigment load, mid
    /// roughness, and a `2` degree cuticle tilt.
    #[inline]
    fn default() -> Self {
        Self {
            eta: 1.55,
            sigma_a: melanin_to_sigma_a(1.3, 0.0),
            beta_m: 0.3,
            beta_n: 0.3,
            alpha: 0.0349, // ~2 degrees
        }
    }
}

/// Evaluates the full Marschner/Chiang fiber BSDF as the sum of the three
/// R/TT/TRT lobes.
///
/// `theta_i` / `theta_r` are the incident and outgoing longitudinal angles
/// (radians, measured from the fiber normal plane) and `phi` is the relative
/// azimuth `phi_r - phi_i`.  Returns an RGB [`Vec3`] whose channels are finite
/// and non-negative; the azimuthal `N_p` carries the per-channel tint.
///
/// The result is the separable product summed over orders,
/// `sum_p M_p(theta_i, theta_r) * N_p(phi)`, with each `N_p` already integrated
/// over the fiber width `h`.
#[inline]
pub fn hair_bsdf(theta_i: f32, theta_r: f32, phi: f32, params: &HairParams) -> Vec3 {
    let theta_i = sanitize_angle(theta_i);
    let theta_r = sanitize_angle(theta_r);
    let phi = if phi.is_finite() { phi } else { 0.0 };

    // Outgoing longitudinal sine/cosine (shared by every azimuthal lobe).
    let (sin_theta_r, cos_theta_r) = ops::sin_cos(theta_r);
    let cos_theta_r = cos_theta_r.abs();

    let mut sum = Vec3::ZERO;
    for p in 0..LOBE_COUNT {
        let m = longitudinal_lobe(p, theta_i, theta_r, params.beta_m, params.alpha);
        if m <= 0.0 {
            continue;
        }
        let n = azimuthal_scattering(
            phi,
            p,
            cos_theta_r,
            sin_theta_r,
            params.eta,
            params.sigma_a,
            params.beta_n,
            DEFAULT_H_SAMPLES,
        );
        sum += n * m;
    }

    Vec3::new(
        if sum.x.is_finite() { sum.x.max(0.0) } else { 0.0 },
        if sum.y.is_finite() { sum.y.max(0.0) } else { 0.0 },
        if sum.z.is_finite() { sum.z.max(0.0) } else { 0.0 },
    )
}

/// Clamps a longitudinal angle to `(-pi/2, pi/2)` and replaces non-finite input
/// with zero, keeping all downstream trigonometry well-defined.
#[inline]
fn sanitize_angle(theta: f32) -> f32 {
    use core::f32::consts::FRAC_PI_2;
    if theta.is_finite() {
        theta.clamp(-FRAC_PI_2 + 1.0e-4, FRAC_PI_2 - 1.0e-4)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsdf_is_non_negative_and_finite_over_sweep() {
        let params = HairParams::default();
        for ti in -5..=5 {
            let theta_i = ti as f32 * 0.25;
            for tr in -5..=5 {
                let theta_r = tr as f32 * 0.25;
                for pk in 0..8 {
                    let phi = -core::f32::consts::PI
                        + pk as f32 * (2.0 * core::f32::consts::PI / 8.0);
                    let f = hair_bsdf(theta_i, theta_r, phi, &params);
                    assert!(f.x >= 0.0 && f.x.is_finite(), "f={f:?}");
                    assert!(f.y >= 0.0 && f.y.is_finite(), "f={f:?}");
                    assert!(f.z >= 0.0 && f.z.is_finite(), "f={f:?}");
                }
            }
        }
    }

    #[test]
    fn bsdf_is_deterministic() {
        let params = HairParams::default();
        let a = hair_bsdf(0.1, -0.2, 0.6, &params);
        let b = hair_bsdf(0.1, -0.2, 0.6, &params);
        assert_eq!(a, b);
    }

    #[test]
    fn bsdf_energy_does_not_explode() {
        // Integrating the BSDF over the full outgoing sphere must stay bounded
        // (energy is conserved, not amplified). We integrate f * cos over the
        // outgoing hemisphere-like domain and check the total is finite and
        // below a generous ceiling.
        let params = HairParams::default();
        let theta_i = 0.2_f32;
        let n_theta = 48;
        let n_phi = 48;
        let dtheta = core::f32::consts::PI / n_theta as f32; // -pi/2..pi/2
        let dphi = 2.0 * core::f32::consts::PI / n_phi as f32;
        let mut total = Vec3::ZERO;
        for it in 0..n_theta {
            let theta_r = -core::f32::consts::FRAC_PI_2 + (it as f32 + 0.5) * dtheta;
            let cos_r = ops::cos(theta_r).abs();
            for ip in 0..n_phi {
                let phi = -core::f32::consts::PI + (ip as f32 + 0.5) * dphi;
                let f = hair_bsdf(theta_i, theta_r, phi, &params);
                total += f * (cos_r * dtheta * dphi);
            }
        }
        assert!(total.x.is_finite() && total.y.is_finite() && total.z.is_finite());
        // Reflectance (dimensionless albedo) must not exceed unity by much.
        assert!(total.x <= 1.5 && total.y <= 1.5 && total.z <= 1.5, "total={total:?}");
    }

    #[test]
    fn darker_absorption_darkens_transmitted_lobes() {
        // Raising sigma_a cannot brighten the fiber: the integrated response of
        // a darker fiber is <= that of a lighter one.
        let light = HairParams {
            sigma_a: Vec3::splat(0.05),
            ..HairParams::default()
        };
        let dark = HairParams {
            sigma_a: Vec3::splat(2.0),
            ..HairParams::default()
        };
        let acc = |p: &HairParams| -> f32 {
            let mut s = 0.0;
            for pk in 0..16 {
                let phi = -core::f32::consts::PI
                    + pk as f32 * (2.0 * core::f32::consts::PI / 16.0);
                s += hair_bsdf(0.1, -0.1, phi, p).x;
            }
            s
        };
        assert!(acc(&dark) <= acc(&light) + 1.0e-4);
    }

    #[test]
    fn degenerate_params_are_handled() {
        let params = HairParams {
            eta: f32::NAN,
            sigma_a: Vec3::splat(f32::INFINITY),
            beta_m: -1.0,
            beta_n: f32::NAN,
            alpha: f32::INFINITY,
        };
        let f = hair_bsdf(f32::NAN, f32::INFINITY, f32::NAN, &params);
        assert!(f.x.is_finite() && f.y.is_finite() && f.z.is_finite());
    }

    #[test]
    fn reexports_are_wired() {
        // Smoke-test the public re-exports resolve to the submodule items.
        let sigma = color_to_sigma_a(Vec3::splat(0.5), 0.3);
        let color = sigma_a_to_color(sigma, 0.3);
        assert!(color.x.is_finite());
        let _ = melanin_to_sigma_a(1.0, 0.0);
        let _ = longitudinal_lobe(0, 0.1, 0.1, 0.3, 0.0);
        let _ = azimuthal_scattering(0.5, 0, 0.9, 0.2, 1.55, sigma, 0.3, 8);
    }
}
