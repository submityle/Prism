//! Smooth (perfectly specular) dielectric scattering for the reference tracer.
//!
//! A dielectric interface — glass, water, a clear coat — both reflects and
//! transmits light. Unlike a conductor, a non-absorbing dielectric lets most
//! of the incident energy pass through, bending it according to Snell's law,
//! while a view-angle-dependent fraction given by the Fresnel equations is
//! mirror-reflected. This module provides the two closed-form building blocks a
//! specular dielectric lobe needs:
//!
//! - [`fresnel_dielectric`] — the unpolarized Fresnel reflectance for an
//!   interface between two real refractive indices, including the total
//!   internal reflection (`TIR`) regime.
//! - [`refract`] — the refracted direction from Snell's law, or `None` when the
//!   geometry is in `TIR` and no transmission exists.
//!
//! Both follow the sign conventions of the path tracer: `wo` and the surface
//! normal `n` point *away* from the surface and lie in the same hemisphere, and
//! all returned directions are unit length. Only `sqrt` is used, matching the
//! crate's determinism policy (no `sin`/`cos`/`exp`).

use super::Vec3;

/// The unpolarized Fresnel reflectance of a smooth dielectric interface.
///
/// `cos_i` is the cosine of the angle between the incident direction and the
/// interface normal, taken on the incident side (so `cos_i >= 0`). `eta_i` is
/// the refractive index of the medium the light arrives through and `eta_t` the
/// index of the medium it would enter. The result is the fraction of energy
/// reflected, averaged over the parallel and perpendicular polarizations
/// (`Fresnel` equations); the transmitted fraction is `1 - R`.
///
/// When the incident geometry exceeds the critical angle (`sin^2(theta_t) >= 1`)
/// the interface is in total internal reflection and the function returns `1`.
#[must_use]
pub fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    // Snell's law in squared-sine form: sin^2(t) = (eta_i/eta_t)^2 sin^2(i).
    let eta = eta_i / eta_t;
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        // Beyond the critical angle: all energy is reflected.
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    // Parallel- and perpendicular-polarized amplitude reflection coefficients.
    let r_parl = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let r_perp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    // Unpolarized reflectance is the mean of the two squared amplitudes.
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// The refracted direction for a ray leaving along `wo` across an interface.
///
/// `wo` is the (unit) outgoing/view direction pointing away from the surface,
/// `n` is the unit interface normal oriented into the same hemisphere as `wo`,
/// and `eta` is the relative index `eta_i / eta_t` (incident over transmitted).
/// Returns the unit transmitted direction (on the far side of `n`), or `None`
/// in the total-internal-reflection regime where Snell's law has no real
/// solution.
#[must_use]
pub fn refract(wo: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = n.dot(wo);
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    // wt = -eta * wo + (eta cos_i - cos_t) n, pointing below the interface.
    let wt = wo.scale(-eta).add(n.scale(eta * cos_i - cos_t));
    Some(wt.normalize_or_zero())
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    #[test]
    fn normal_incidence_matches_schlick_f0() {
        // At normal incidence R = ((n1 - n2)/(n1 + n2))^2; for air->glass
        // (1.0 -> 1.5) this is the familiar F0 = 0.04.
        let r = fresnel_dielectric(1.0, 1.0, 1.5);
        assert!((r - 0.04).abs() < 1e-3, "normal-incidence R = {r}");
    }

    #[test]
    fn reflectance_rises_toward_grazing() {
        // Fresnel reflectance is monotonically increasing from F0 to 1 as the
        // incidence angle approaches grazing (cos_i -> 0).
        let near_normal = fresnel_dielectric(0.99, 1.0, 1.5);
        let mid = fresnel_dielectric(0.5, 1.0, 1.5);
        let grazing = fresnel_dielectric(0.02, 1.0, 1.5);
        assert!(near_normal < mid, "{near_normal} < {mid}");
        assert!(mid < grazing, "{mid} < {grazing}");
        assert!(grazing <= 1.0 + 1e-6);
    }

    #[test]
    fn total_internal_reflection_returns_full_reflectance() {
        // Going from dense (1.5) to rare (1.0), angles past the critical angle
        // (cos_i small enough) reflect all energy.
        let r = fresnel_dielectric(0.1, 1.5, 1.0);
        assert!((r - 1.0).abs() < 1e-6, "TIR reflectance = {r}");
    }

    #[test]
    fn refraction_at_normal_incidence_passes_straight() {
        // A ray along the normal is not bent; the transmitted direction is the
        // exact opposite of the view direction.
        let wo = N;
        let wt = refract(wo, N, 1.0 / 1.5).expect("transmission exists at normal incidence");
        let expect = N.negate();
        assert!(
            wt.sub(expect).length_squared() < 1e-10,
            "straight-through transmission {wt:?}"
        );
    }

    #[test]
    fn refraction_bends_toward_normal_entering_denser_medium() {
        // Entering a denser medium (air -> glass) bends the ray toward the
        // normal: the transmitted angle is smaller than the incident angle.
        let wo = Vec3::new(0.6, 0.8, 0.0).normalize_or_zero();
        let cos_i = N.dot(wo);
        let wt = refract(wo, N, 1.0 / 1.5).expect("transmission exists");
        // Transmitted direction is below the surface.
        assert!(
            wt.y < 0.0,
            "transmitted ray should go below surface: {wt:?}"
        );
        let cos_t = N.negate().dot(wt);
        assert!(
            cos_t > cos_i,
            "entering a denser medium, cos_t ({cos_t}) should exceed cos_i ({cos_i})"
        );
    }

    #[test]
    fn refraction_returns_none_in_total_internal_reflection() {
        // Dense -> rare at a shallow angle has no real transmission.
        let wo = Vec3::new(0.95, 0.3122499, 0.0).normalize_or_zero();
        let wt = refract(wo, N, 1.5 / 1.0);
        assert!(wt.is_none(), "expected TIR, got {wt:?}");
    }

    #[test]
    fn snells_law_sine_ratio_is_preserved() {
        // sin(theta_i) / sin(theta_t) must equal eta_t / eta_i for the
        // transmitted direction (here air->glass, so the ratio is 1.5).
        let wo = Vec3::new(0.5, 0.8660254, 0.0).normalize_or_zero();
        let eta = 1.0 / 1.5;
        let wt = refract(wo, N, eta).expect("transmission exists");
        let cos_i = N.dot(wo);
        let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
        let cos_t = N.negate().dot(wt);
        let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
        let ratio = sin_i / sin_t;
        assert!(
            (ratio - 1.5).abs() < 1e-4,
            "Snell ratio {ratio} should equal eta_t/eta_i = 1.5"
        );
    }
}
