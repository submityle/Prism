//! Physically exact rough conductor using the complex-index-of-refraction
//! `Fresnel` equations.
//!
//! [`crate::reference_pt::bsdf::Bsdf::GgxConductor`] colours its highlight with
//! Schlick's cheap rational `Fresnel` approximation driven by a single
//! normal-incidence reflectance `F0`. That is adequate for an art-directed
//! tint, but it cannot reproduce the characteristic *off-axis hue shift* of a
//! real metal: gold warms toward white at grazing angles, copper and aluminium
//! each bend differently because their reflectance is governed by a wavelength
//! dependent **complex** index of refraction `n~ = eta + i*k` (the real index
//! `eta` and the extinction coefficient `k`). This module evaluates the exact
//! unpolarized `Fresnel` reflectance of such an interface, so the offline
//! reference tracer owns a spectral metal oracle the real-time Cook-Torrance
//! path can be measured against.
//!
//! The specular lobe itself is the same isotropic `GGX` microfacet statistics
//! as [`crate::reference_pt::microfacet`] (distribution `D`, height-correlated
//! Smith masking `G2`, and visible-normal `VNDF` importance sampling); only the
//! `Fresnel` term is replaced. Because the exact `Fresnel` is spectral, the
//! three channels of `eta` and `k` carry the measured red/green/blue indices of
//! the metal (for example gold `eta ~ (0.14, 0.37, 1.44)`,
//! `k ~ (3.98, 2.39, 1.60)`), reproducing both the base colour and its angular
//! drift from first principles.
//!
//! The derivation follows the standard `PBRT` `FrComplex` formulation: the
//! squared magnitudes `a^2 + b^2` and `a` of the complex transmitted cosine are
//! recovered algebraically, which keeps the whole routine inside the crate's
//! "no transcendental" policy — only `sqrt`, multiplication, and division
//! appear, never `sin`/`cos`/`pow`.
//!
//! Conventions match the rest of the tracer: `wo`, `wi`, and `normal` are unit
//! vectors with `wo`/`wi` pointing away from the surface; the lobe is glossy
//! (non-delta), so it is importance-sampled and connected to lights by
//! next-event estimation like any rough surface.

use super::microfacet::GgxIsotropic;
use super::sampler::Rng;
use super::{Vec3, EPS_LEN_SQ};

/// The exact unpolarized `Fresnel` reflectance of a conductor interface for a
/// single wavelength channel, given the cosine of the incidence angle and the
/// channel's complex index `eta + i*k`.
///
/// `cos_theta_i` is clamped to `[0, 1]`; `eta` (real index) and `k` (extinction
/// coefficient) are taken non-negative. The result is the mean of the squared
/// `s`- and `p`-polarized reflection amplitudes `0.5 * (R_s + R_p)`, clamped to
/// `[0, 1]`. All intermediates use only `sqrt`, products, and quotients so no
/// transcendental call is needed.
fn fresnel_conductor_channel(cos_theta_i: f32, eta: f32, k: f32) -> f32 {
    let cos_i = cos_theta_i.clamp(0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    // The complex transmitted cosine squared decomposes into `a^2 + b^2` and
    // the real part `a`; both are real square roots of non-negative quantities.
    let t0 = eta2 - k2 - sin2;
    let a2_plus_b2 = (t0 * t0 + 4.0 * eta2 * k2).max(0.0).sqrt();
    let a = (0.5 * (a2_plus_b2 + t0)).max(0.0).sqrt();
    // s-polarized reflectance.
    let t1 = a2_plus_b2 + cos2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = t1 + t2;
    let r_s = if denom_s > 0.0 {
        (t1 - t2) / denom_s
    } else {
        1.0
    };
    // p-polarized reflectance, expressed relative to `r_s`.
    let t3 = cos2 * a2_plus_b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = if denom_p > 0.0 {
        r_s * (t3 - t4) / denom_p
    } else {
        r_s
    };
    (0.5 * (r_s + r_p)).clamp(0.0, 1.0)
}

/// The per-channel exact conductor `Fresnel` reflectance for the red/green/blue
/// complex indices `eta + i*k` at incidence cosine `cos_theta_i`.
///
/// This is the spectral counterpart of
/// [`crate::reference_pt::microfacet::fresnel_schlick`]: at normal incidence it
/// returns the metal's measured base colour, and it rises smoothly toward one
/// at grazing with the per-channel curvature that gives real metals their
/// angular hue shift.
#[must_use]
pub fn fresnel_conductor(eta: Vec3, k: Vec3, cos_theta_i: f32) -> Vec3 {
    Vec3::new(
        fresnel_conductor_channel(cos_theta_i, eta.x, k.x),
        fresnel_conductor_channel(cos_theta_i, eta.y, k.y),
        fresnel_conductor_channel(cos_theta_i, eta.z, k.z),
    )
}

/// Number of midpoint-quadrature nodes used to pre-integrate the average
/// `Fresnel` reflectance over the cosine-weighted hemisphere.
const AVG_FRESNEL_NODES: u32 = 32;

/// The hemispherical cosine-weighted average `Fresnel` reflectance
/// `F_avg = 2 * integral_0^1 F(mu) mu d mu`, evaluated by midpoint quadrature.
///
/// This is the per-channel reflectance a diffuse (`Lambertian`) distribution of
/// micro-facets would show, and it drives the colour of the energy recovered by
/// the Kulla-Conty multiple-scattering lobes in
/// [`crate::reference_pt::conductor_ms`] and
/// [`crate::reference_pt::conductor_aniso_ms`].
#[must_use]
pub fn average_fresnel_conductor(eta: Vec3, k: Vec3) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for i in 0..AVG_FRESNEL_NODES {
        let mu = (i as f32 + 0.5) / AVG_FRESNEL_NODES as f32;
        let weight = 2.0 * mu / AVG_FRESNEL_NODES as f32;
        acc = acc.add(fresnel_conductor(eta, k, mu).scale(weight));
    }
    acc
}

/// The outcome of importance-sampling a [`Conductor`] lobe.
#[derive(Clone, Copy, Debug)]
pub struct ConductorSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The `BRDF` value `f_r(wo, wi)` for the sampled pair (no cosine applied).
    pub value: Vec3,
    /// The solid-angle probability density of `direction`.
    pub pdf: f32,
}

/// A rough conductor `BRDF` driven by the exact complex-index `Fresnel` term and
/// an isotropic `GGX` microfacet lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conductor {
    /// Per-channel real index of refraction `eta` of the metal.
    eta: Vec3,
    /// Per-channel extinction coefficient `k` of the metal.
    k: Vec3,
    /// The isotropic `GGX` microfacet distribution of the specular lobe.
    dist: GgxIsotropic,
}

impl Conductor {
    /// Builds a conductor from its complex index `eta + i*k` and a perceptual
    /// `roughness` in `[0, 1]` (remapped to the `GGX` width `alpha = roughness^2`
    /// by [`GgxIsotropic::from_roughness`]).
    #[must_use]
    pub fn new(eta: Vec3, k: Vec3, roughness: f32) -> Self {
        Self {
            eta,
            k,
            dist: GgxIsotropic::from_roughness(roughness),
        }
    }

    /// Evaluates the rough-conductor `BRDF` `f_r = F * D * G2 / (4 cos_o cos_i)`.
    ///
    /// Returns [`Vec3::ZERO`] when either direction is below the surface or the
    /// half vector degenerates.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let half = wo.add(wi).normalize_or_zero();
        if half.length_squared() <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        let cos_h = normal.dot(half);
        if cos_h <= 0.0 {
            return Vec3::ZERO;
        }
        let d = self.dist.distribution(cos_h);
        let g2 = self.dist.g2(cos_o, cos_i);
        let fresnel = fresnel_conductor(self.eta, self.k, wo.dot(half).max(0.0));
        fresnel.scale(d * g2 / (4.0 * cos_o * cos_i))
    }

    /// The solid-angle density [`Self::sample`] assigns to `(wo, wi)`.
    ///
    /// Zero when either direction is below the surface or the half vector
    /// degenerates.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let half = wo.add(wi).normalize_or_zero();
        if half.length_squared() <= EPS_LEN_SQ {
            return 0.0;
        }
        let cos_h = normal.dot(half);
        self.dist.reflection_pdf(cos_o, cos_h)
    }

    /// Importance-samples an outgoing direction via visible-normal (`VNDF`)
    /// sampling of the `GGX` lobe.
    ///
    /// The returned `value`/`pdf` are the true `BRDF` and its solid-angle
    /// density, so the integrator's generic throughput update
    /// `value * cos_i / pdf` reduces to the clean microfacet weight
    /// `F * G2 / G1(wo)`. Returns `None` for a degenerate (grazing/zero-length)
    /// sample so the caller terminates the path rather than dividing by zero.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<ConductorSample> {
        // The integrator passes the raw geometric normal; orient it into the
        // view hemisphere so a back-facing hit still scatters.
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        let half = self.dist.sample_half_vector(wo, normal, rng)?;
        let woh = wo.dot(half);
        if woh <= 0.0 {
            return None;
        }
        // Reflect the view direction about the sampled microfacet normal.
        let wi = wo.negate().reflect(half).normalize_or_zero();
        let cos_i = normal.dot(wi);
        if cos_i <= 0.0 {
            return None;
        }
        let cos_h = normal.dot(half);
        let pdf = self.dist.reflection_pdf(cos_o, cos_h);
        if pdf <= 0.0 {
            return None;
        }
        let d = self.dist.distribution(cos_h);
        let g2 = self.dist.g2(cos_o, cos_i);
        let fresnel = fresnel_conductor(self.eta, self.k, woh);
        let value = fresnel.scale(d * g2 / (4.0 * cos_o * cos_i));
        Some(ConductorSample {
            direction: wi,
            value,
            pdf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_pt::microfacet::fresnel_schlick;

    /// Red/green/blue complex index of gold, a canonical warm metal.
    const GOLD_ETA: Vec3 = Vec3 {
        x: 0.143,
        y: 0.375,
        z: 1.442,
    };
    /// Extinction coefficient of gold.
    const GOLD_K: Vec3 = Vec3 {
        x: 3.983,
        y: 2.386,
        z: 1.603,
    };

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    /// Every channel of the exact `Fresnel` term stays within `[0, 1]` across
    /// the whole incidence range.
    #[test]
    fn fresnel_is_bounded_everywhere() {
        for i in 0..=100u32 {
            let cos_i = i as f32 / 100.0;
            let f = fresnel_conductor(GOLD_ETA, GOLD_K, cos_i);
            for c in [f.x, f.y, f.z] {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "fresnel {c} out of range at {cos_i}"
                );
            }
        }
    }

    /// At grazing incidence (`cos -> 0`) a conductor reflects essentially all
    /// energy on every channel, approaching a white edge highlight.
    #[test]
    fn grazing_reflectance_approaches_one() {
        let f = fresnel_conductor(GOLD_ETA, GOLD_K, 0.0);
        assert!(
            f.x > 0.99 && f.y > 0.99 && f.z > 0.99,
            "grazing not white: {f:?}"
        );
    }

    /// Gold is warm: at normal incidence its red reflectance exceeds its blue,
    /// which Schlick can only reproduce if the base colour is pre-tinted but the
    /// exact model derives from the measured index directly.
    #[test]
    fn gold_is_warm_at_normal_incidence() {
        let f = fresnel_conductor(GOLD_ETA, GOLD_K, 1.0);
        assert!(
            f.x > f.z + 0.1,
            "gold should be warmer in red than blue: {f:?}"
        );
    }

    /// The exact conductor `Fresnel` and the Schlick approximation seeded with
    /// the exact normal-incidence reflectance agree closely near normal
    /// incidence (they only diverge in the grazing hue drift).
    #[test]
    fn matches_schlick_near_normal_incidence() {
        let f0 = fresnel_conductor(GOLD_ETA, GOLD_K, 1.0);
        let exact = fresnel_conductor(GOLD_ETA, GOLD_K, 0.95);
        let schlick = fresnel_schlick(f0, 0.95);
        assert!(
            exact.sub(schlick).length() < 0.05,
            "exact {exact:?} vs schlick {schlick:?}"
        );
    }

    /// A sampled direction stays in the view hemisphere and reports the same
    /// value and density [`Conductor::evaluate`]/[`Conductor::pdf`] would.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let conductor = Conductor::new(GOLD_ETA, GOLD_K, 0.3);
        let wo = unit(0.4, 0.9, 0.2);
        let mut rng = Rng::seed(42);
        for _ in 0..512 {
            if let Some(s) = conductor.sample(wo, N, &mut rng) {
                assert!(N.dot(s.direction) > 0.0, "sample below surface");
                let eval = conductor.evaluate(wo, s.direction, N);
                let pdf = conductor.pdf(wo, s.direction, N);
                // The sharp `GGX` lobe amplifies tiny half-vector round-off in
                // the recomputed `evaluate`/`pdf` path, so compare relative to
                // the (potentially large) peak value rather than absolutely.
                let val_tol = 1e-3 * s.value.length().max(1.0);
                let pdf_tol = 1e-3 * pdf.max(1.0);
                assert!(s.value.sub(eval).length() <= val_tol, "value mismatch");
                assert!((s.pdf - pdf).abs() <= pdf_tol, "pdf mismatch");
            }
        }
    }

    /// White-furnace energy check: the single-scatter directional albedo of the
    /// `GGX` conductor never exceeds one (the microfacet masking only removes
    /// energy, never creates it).
    #[test]
    fn directional_albedo_never_exceeds_one() {
        let conductor = Conductor::new(GOLD_ETA, GOLD_K, 0.4);
        let wo = unit(0.3, 0.9, 0.0);
        let mut rng = Rng::seed(7);
        let samples = 40_000u32;
        let mut sum = 0.0f32;
        for _ in 0..samples {
            if let Some(s) = conductor.sample(wo, N, &mut rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                // f_r * cos_i / pdf is the unbiased throughput weight.
                sum += s.value.x * cos_i / s.pdf;
            }
        }
        let albedo = sum / samples as f32;
        assert!(albedo <= 1.0 + 1e-2, "furnace gained energy: {albedo}");
    }
}
