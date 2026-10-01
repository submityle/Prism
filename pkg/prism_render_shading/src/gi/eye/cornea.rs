//! Corneal refraction and dielectric Fresnel — CPU golden reference.
//!
//! The cornea is the eye's dominant refractive surface: a thin, wet dielectric
//! shell (index of refraction `n ≈ 1.376`) separating air (`n = 1.0`) from the
//! aqueous humour (`n ≈ 1.336`) that fills the anterior chamber in front of the
//! iris.  Two optical effects matter for a AAA eye:
//!
//! * **Refraction.** The viewing ray bends as it crosses the air→cornea
//!   interface (Snell's law).  The bent ray is what actually samples the iris,
//!   so getting it right is what makes an eye read as a fluid-filled sphere
//!   instead of a flat decal.  Past the critical angle the ray cannot cross and
//!   is *totally internally reflected*; the physical refraction routine signals
//!   this with [`Option::None`].
//! * **Reflection.** The same interface reflects part of the incident energy —
//!   the specular "wet" highlight and catch-light.  This module evaluates the
//!   *exact* unpolarised dielectric Fresnel reflectance (the full Fresnel
//!   equations averaged over polarisation), and also exposes the cheaper
//!   Schlick approximation re-used from the shared GGX lobe for parity with the
//!   glossy specular path.
//!
//! # Conventions
//! * All directions are unit vectors in the eye's *local frame*, where the
//!   optical axis / apex normal is `+Z` pointing out of the eye toward the
//!   camera.  A direction's `z` component is therefore its cosine with the
//!   outward normal.
//! * [`refract`] takes the ray's *propagation* direction `incident` (pointing
//!   into the surface) and the interface `normal` on the incident-medium side
//!   (opposing the ray), with `eta = n_incident / n_transmitted`.  It returns
//!   the transmitted propagation direction, or `None` on total internal
//!   reflection.
//! * [`shade`] takes the shading `view` direction (pointing *away* from the
//!   surface, toward the camera, the standard shading convention) and converts
//!   to the propagation direction internally.
//! * Every helper is a deterministic pure function (no RNG, I/O, GPU, globals
//!   or `unsafe`).  Inputs are clamped, square-root arguments are floored at
//!   zero, denominators are kept away from zero and every result is finite so
//!   the reference can never inject `NaN`/`inf` energy.
//! * `f32` arithmetic throughout mirrors the WESL/GPU twin.  `sqrt` uses the
//!   inherent `f32::sqrt`; there are no transcendental calls on the hot path.
//!
//! # References
//! * Atchison & Smith, *Optics of the Human Eye* — corneal/aqueous indices.
//! * Born & Wolf, *Principles of Optics* — the Fresnel equations.
//! * Walter et al. 2007, *Microfacet Models for Refraction* — the vector form
//!   of Snell's law used by [`refract`].

use bevy_math::Vec3;

use crate::gi::spec_gi::ggx_lobe::fresnel_schlick_scalar;

/// Refractive index of air (the incident medium in front of the eye).
pub const IOR_AIR: f32 = 1.0;
/// Refractive index of the corneal stroma.
pub const IOR_CORNEA: f32 = 1.376;
/// Refractive index of the aqueous humour behind the cornea.
pub const IOR_AQUEOUS: f32 = 1.336;

/// Smallest magnitude a cosine/denominator is allowed to reach before it is
/// floored, keeping grazing-angle divisions finite.
const MIN_DENOM: f32 = 1.0e-5;

/// Physically correct refraction of a propagation direction across a dielectric
/// interface (the vector form of Snell's law).
///
/// `incident` is the unit direction the ray *travels* as it hits the surface
/// (pointing into the interface).  `normal` is the unit interface normal on the
/// incident-medium side, i.e. opposing `incident` so that `incident · normal`
/// is negative.  `eta = n_incident / n_transmitted` is the relative index.
///
/// Returns the unit transmitted propagation direction, or [`None`] when the
/// configuration is *total internal reflection* (only possible for `eta > 1`,
/// i.e. travelling into the rarer medium beyond the critical angle).
///
/// The normal is auto-oriented against the incident ray, so a caller that
/// hands in the outward normal regardless of ray side still gets a correct
/// result.
#[inline]
pub fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let i = safe_normalize(incident, Vec3::NEG_Z);
    let mut n = safe_normalize(normal, Vec3::Z);
    let eta = eta.max(MIN_DENOM);

    // Orient the normal against the ray so `cos_i >= 0`.
    let mut cos_i = -i.dot(n);
    if cos_i < 0.0 {
        n = -n;
        cos_i = -cos_i;
    }
    cos_i = cos_i.clamp(0.0, 1.0);

    // k = cos_t^2.  Negative means the squared transmitted cosine is imaginary:
    // total internal reflection.
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        return None;
    }
    let cos_t = k.sqrt();
    let t = eta * i + (eta * cos_i - cos_t) * n;
    Some(safe_normalize(t, i))
}

/// Mirror-reflects a propagation direction about `normal`.
///
/// `incident` points into the surface; the result points back out.  Useful as
/// the total-internal-reflection fallback for [`refract`].
#[inline]
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    let i = safe_normalize(incident, Vec3::NEG_Z);
    let n = safe_normalize(normal, Vec3::Z);
    i - 2.0 * i.dot(n) * n
}

/// Cosine of the critical angle for an interface with relative index `eta`.
///
/// Returns [`Some`] only when total internal reflection is possible (`eta > 1`,
/// travelling into the rarer medium); for `eta <= 1` no critical angle exists
/// and the function returns [`None`].  Incidence cosines *below* the returned
/// value undergo total internal reflection.
#[inline]
pub fn critical_angle_cos(eta: f32) -> Option<f32> {
    if eta <= 1.0 {
        return None;
    }
    // sin(theta_c) = 1 / eta  =>  cos(theta_c) = sqrt(1 - 1/eta^2).
    let inv = 1.0 / eta;
    Some((1.0 - inv * inv).max(0.0).sqrt())
}

/// Exact unpolarised dielectric Fresnel reflectance.
///
/// `cos_i` is the (clamped) incidence cosine and `eta = n_incident /
/// n_transmitted`.  Evaluates the full Fresnel equations for `s`- and
/// `p`-polarisation and averages them (unpolarised light).  Returns `1.0` under
/// total internal reflection, and always a value in `[0, 1]`.
#[inline]
pub fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let eta = eta.max(MIN_DENOM);

    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t >= 1.0 {
        // Beyond the critical angle: everything reflects.
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();

    // Divide numerator/denominator of the standard equations by n_transmitted
    // so only the ratio `eta` is needed.
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t).max(MIN_DENOM);
    let rp = (eta * cos_t - cos_i) / (eta * cos_t + cos_i).max(MIN_DENOM);
    (0.5 * (rs * rs + rp * rp)).clamp(0.0, 1.0)
}

/// Normal-incidence reflectance `F0` of a dielectric interface from the two
/// indices: `F0 = ((n_i - n_t) / (n_i + n_t))^2`.
#[inline]
pub fn ior_to_f0(n_i: f32, n_t: f32) -> f32 {
    let num = n_i - n_t;
    let den = (n_i + n_t).max(MIN_DENOM);
    let r = num / den;
    (r * r).clamp(0.0, 1.0)
}

/// Schlick-approximated dielectric Fresnel for parity with the glossy specular
/// path (re-uses [`fresnel_schlick_scalar`]).
///
/// Converts the two indices to `F0` via [`ior_to_f0`] and evaluates Schlick's
/// `F0 + (1 - F0)(1 - cosθ)^5`.  Cheaper and smoother than
/// [`fresnel_dielectric`] but misses the Brewster dip, so it is offered
/// alongside rather than instead of the exact form.
#[inline]
pub fn fresnel_schlick_ior(cos_i: f32, n_i: f32, n_t: f32) -> f32 {
    fresnel_schlick_scalar(ior_to_f0(n_i, n_t), cos_i)
}

/// A pair of refractive indices describing one corneal interface.
///
/// `outer` is the index on the camera side (air for the front surface) and
/// `inner` the index just past it (the corneal stroma, or — for an effective
/// single-surface model that pushes the ray straight to the iris — the aqueous
/// humour).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorneaInterface {
    /// Index on the incident (camera) side.
    pub outer: f32,
    /// Index on the transmitted (interior) side.
    pub inner: f32,
}

impl CorneaInterface {
    /// Builds an interface, flooring both indices at `1.0` (no sub-vacuum
    /// media) so [`Self::eta`] stays finite and physical.
    #[inline]
    pub fn new(outer: f32, inner: f32) -> Self {
        Self {
            outer: outer.max(1.0),
            inner: inner.max(1.0),
        }
    }

    /// The physically faithful air→cornea front surface.
    #[inline]
    pub fn front() -> Self {
        Self::new(IOR_AIR, IOR_CORNEA)
    }

    /// An effective single-surface air→aqueous interface that refracts the
    /// view ray straight toward the iris, skipping the optically thin stroma.
    #[inline]
    pub fn effective() -> Self {
        Self::new(IOR_AIR, IOR_AQUEOUS)
    }

    /// Relative index `eta = n_outer / n_inner` used by [`refract`].
    #[inline]
    pub fn eta(&self) -> f32 {
        self.outer / self.inner.max(MIN_DENOM)
    }

    /// Normal-incidence reflectance `F0` of this interface.
    #[inline]
    pub fn f0(&self) -> f32 {
        ior_to_f0(self.outer, self.inner)
    }
}

impl Default for CorneaInterface {
    #[inline]
    fn default() -> Self {
        Self::front()
    }
}

/// Result of evaluating one corneal interface for a given view direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorneaSample {
    /// Transmitted propagation direction (points into the eye), or [`None`]
    /// under total internal reflection.
    pub refracted: Option<Vec3>,
    /// Exact unpolarised Fresnel reflectance in `[0, 1]` (the specular/wet
    /// highlight weight).
    pub reflectance: f32,
    /// Clamped incidence cosine `view · normal` used for the evaluation.
    pub cos_incidence: f32,
    /// `true` when the ray was totally internally reflected.
    pub total_internal_reflection: bool,
}

/// Evaluates a corneal interface for a shading `view` direction.
///
/// `view` points *away* from the surface toward the camera (standard shading
/// convention); `normal` is the outward surface normal.  The function converts
/// `view` to the ray's propagation direction (`-view`), refracts it through the
/// interface and evaluates the exact Fresnel reflectance at the same incidence.
#[inline]
pub fn shade(view: Vec3, normal: Vec3, interface: CorneaInterface) -> CorneaSample {
    let v = safe_normalize(view, Vec3::Z);
    let n = safe_normalize(normal, Vec3::Z);
    let cos_i = v.dot(n).clamp(0.0, 1.0);
    let eta = interface.eta();

    let refracted = refract(-v, n, eta);
    let reflectance = fresnel_dielectric(cos_i, eta);

    CorneaSample {
        refracted,
        reflectance,
        cos_incidence: cos_i,
        total_internal_reflection: refracted.is_none(),
    }
}

/// Normalises `v`, returning `fallback` when `v` is degenerate (near zero or
/// non-finite), so downstream math never sees `NaN`.
#[inline]
fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > MIN_DENOM * MIN_DENOM {
        v / len_sq.sqrt()
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn straight_on_ray_passes_undeviated() {
        // A ray along the axis hits the apex head-on and does not bend.
        let incident = Vec3::NEG_Z; // travelling into the eye
        let normal = Vec3::Z;
        let t = refract(incident, normal, CorneaInterface::front().eta()).unwrap();
        assert!((t - Vec3::NEG_Z).length() < 1e-4, "t={t:?}");
    }

    #[test]
    fn refracted_direction_is_unit_and_bends_toward_normal() {
        // Entering a denser medium bends the ray toward the normal, so the
        // transmitted angle from the axis is smaller than the incident one.
        let incident = Vec3::new(0.6, 0.0, -0.8).normalize();
        let normal = Vec3::Z;
        let eta = CorneaInterface::front().eta(); // < 1 (air -> cornea)
        let t = refract(incident, normal, eta).unwrap();
        assert!((t.length() - 1.0).abs() < 1e-4);

        let sin_in = incident.x.abs(); // |x| for a unit vector in the xz-plane
        let sin_out = t.x.abs();
        assert!(sin_out < sin_in, "sin_out={sin_out} sin_in={sin_in}");
        // Snell's law: n_i sin_i = n_t sin_t  =>  eta * sin_i = sin_t.
        assert!((eta * sin_in - sin_out).abs() < 1e-4);
    }

    #[test]
    fn total_internal_reflection_returns_none() {
        // Dense -> rare (eta > 1) beyond the critical angle must fail.
        let eta = IOR_CORNEA / IOR_AIR; // ~1.376
        let cos_c = critical_angle_cos(eta).unwrap();
        // Pick an incidence just past the critical angle (smaller cosine).
        let cos_i = (cos_c - 0.05).max(0.0);
        let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
        let incident = Vec3::new(sin_i, 0.0, -cos_i).normalize();
        assert!(refract(incident, Vec3::Z, eta).is_none());
    }

    #[test]
    fn below_critical_angle_still_transmits() {
        let eta = IOR_CORNEA / IOR_AIR;
        let cos_c = critical_angle_cos(eta).unwrap();
        let cos_i = (cos_c + 0.1).min(1.0);
        let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
        let incident = Vec3::new(sin_i, 0.0, -cos_i).normalize();
        assert!(refract(incident, Vec3::Z, eta).is_some());
    }

    #[test]
    fn fresnel_normal_incidence_matches_f0() {
        let iface = CorneaInterface::front();
        let f = fresnel_dielectric(1.0, iface.eta());
        assert!((f - iface.f0()).abs() < 1e-4, "f={f} f0={}", iface.f0());
    }

    #[test]
    fn fresnel_grazing_goes_to_one() {
        let f = fresnel_dielectric(0.0, CorneaInterface::front().eta());
        assert!((f - 1.0).abs() < EPS, "f={f}");
    }

    #[test]
    fn fresnel_is_bounded_and_monotone_increasing_toward_grazing() {
        let eta = CorneaInterface::front().eta();
        let mut prev = fresnel_dielectric(1.0, eta);
        for i in 1..=20 {
            let cos_i = 1.0 - i as f32 / 20.0;
            let f = fresnel_dielectric(cos_i, eta);
            assert!((0.0..=1.0).contains(&f), "f={f}");
            assert!(f >= prev - 1e-4, "not monotone: f={f} prev={prev}");
            prev = f;
        }
    }

    #[test]
    fn tir_reflectance_is_total() {
        let eta = IOR_CORNEA / IOR_AIR;
        let cos_c = critical_angle_cos(eta).unwrap();
        let f = fresnel_dielectric((cos_c - 0.05).max(0.0), eta);
        assert_eq!(f, 1.0);
    }

    #[test]
    fn schlick_matches_exact_at_normal_incidence() {
        let a = fresnel_schlick_ior(1.0, IOR_AIR, IOR_CORNEA);
        let b = fresnel_dielectric(1.0, IOR_AIR / IOR_CORNEA);
        assert!((a - b).abs() < 1e-4, "schlick={a} exact={b}");
    }

    #[test]
    fn shade_reports_tir_flag_and_finite_outputs() {
        let s = shade(Vec3::Z, Vec3::Z, CorneaInterface::front());
        assert!(!s.total_internal_reflection);
        assert!(s.refracted.is_some());
        assert!(s.reflectance.is_finite());
        assert!((0.0..=1.0).contains(&s.reflectance));
    }

    #[test]
    fn degenerate_inputs_fall_back_without_nan() {
        let t = refract(Vec3::ZERO, Vec3::ZERO, 0.0);
        // Zero ray along -Z with +Z fallback normal => straight through.
        assert!(t.is_some());
        let s = shade(Vec3::ZERO, Vec3::ZERO, CorneaInterface::front());
        assert!(s.reflectance.is_finite());
    }

    #[test]
    fn reflect_mirrors_about_normal() {
        let r = reflect(Vec3::new(0.0, 0.0, -1.0), Vec3::Z);
        assert!((r - Vec3::Z).length() < 1e-5, "r={r:?}");
    }

    #[test]
    fn critical_angle_absent_entering_denser_medium() {
        assert!(critical_angle_cos(IOR_AIR / IOR_CORNEA).is_none());
    }
}
