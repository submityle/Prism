//! Energy-conserving `Schlick`-`Fresnel` rough conductor (metallic workflow).
//!
//! The real-time renderer parameterises metals the glTF / `UE` "metallic"
//! way: a single `F0` reflectance colour feeding the `Schlick` `Fresnel`
//! approximation, rather than the measured complex index `eta + i*k` the
//! spectral [`crate::reference_pt::conductor_ms`] oracle uses. This module is
//! the matching energy-conserving oracle for that workflow: it wraps the
//! single-scatter `Schlick`-`Fresnel` `GGX` lobe (the exact lobe
//! [`crate::reference_pt::bsdf::Bsdf::GgxConductor`] evaluates) with the same
//! Kulla-Conty multiple-scattering compensation, so the offline reference and
//! the real-time Cook-Torrance path can be compared under identical inputs and
//! both conserve energy.
//!
//! The hemispherical-average `Fresnel` reflectance has a closed form for the
//! `Schlick` approximation, `F_avg = (20 * F0 + 1) / 21` (Kulla and Conty,
//! 2017), so unlike the spectral conductor this oracle needs no `Fresnel`
//! quadrature at all — the average is exact and transcendental-free.
//!
//! The combined `BRDF` is `f_ss + F_ms * f_ms`, matching
//! [`crate::reference_pt::conductor_ms`]: `f_ss` is the single-scatter
//! `Schlick` lobe and `f_ms` the near-diffuse recovery lobe tinted by `F_avg`.
//! Sampling uses the same one-sample multiple-importance strategy between the
//! `GGX` `VNDF` lobe and a cosine-weighted direction, always reporting the
//! combined density so the estimator stays unbiased.

use super::ggx_energy::{average_albedo, multiscatter_fresnel, multiscatter_lobe};
use super::microfacet::{fresnel_schlick, GgxIsotropic};
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, Rng};
use super::{Vec3, EPS_LEN_SQ};

/// The outcome of importance-sampling a [`SchlickMultiscatterConductor`] lobe.
#[derive(Clone, Copy, Debug)]
pub struct SchlickMultiscatterSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The combined `BRDF` value `f_r(wo, wi)` for the sampled pair.
    pub value: Vec3,
    /// The combined solid-angle probability density of `direction`.
    pub pdf: f32,
}

/// A `Schlick`-`Fresnel` rough conductor that adds Kulla-Conty
/// multiple-scattering compensation on top of the single-scatter lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SchlickMultiscatterConductor {
    /// Normal-incidence reflectance `F0` of the metal (per channel).
    reflectance: Vec3,
    /// The isotropic `GGX` lobe shared by the single- and multiple-scatter
    /// terms.
    ggx: GgxIsotropic,
    /// The `GGX` width `alpha` used to index the energy-compensation tables.
    alpha: f32,
    /// The closed-form `Schlick` hemispherical-average reflectance
    /// `F_avg = (20 * F0 + 1) / 21`.
    f_avg: Vec3,
}

/// The exact `Schlick` cosine-weighted average reflectance
/// `F_avg = (20 * F0 + 1) / 21`.
fn schlick_average_fresnel(f0: Vec3) -> Vec3 {
    let channel = |f: f32| (20.0 * f + 1.0) / 21.0;
    Vec3::new(channel(f0.x), channel(f0.y), channel(f0.z))
}

impl SchlickMultiscatterConductor {
    /// Builds an energy-conserving conductor from its normal-incidence
    /// reflectance `F0` and a perceptual `roughness` in `[0, 1]`.
    #[must_use]
    pub fn new(reflectance: Vec3, roughness: f32) -> Self {
        let ggx = GgxIsotropic::from_roughness(roughness);
        Self {
            reflectance,
            ggx,
            alpha: ggx.alpha,
            f_avg: schlick_average_fresnel(reflectance),
        }
    }

    /// The single-scatter `Schlick`-`Fresnel` `GGX` lobe
    /// `f_ss = F * D * G2 / (4 cos_o cos_i)`.
    fn single_scatter(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
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
        let d = self.ggx.distribution(cos_h);
        let g2 = self.ggx.g2(cos_o, cos_i);
        let fresnel = fresnel_schlick(self.reflectance, wo.dot(half).max(0.0));
        fresnel.scale(d * g2 / (4.0 * cos_o * cos_i))
    }

    /// Evaluates the combined `BRDF` `f_ss + F_ms * f_ms`.
    ///
    /// Returns [`Vec3::ZERO`] when either direction is below the surface.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let single = self.single_scatter(wo, wi, normal);
        let lobe = multiscatter_lobe(cos_o, cos_i, self.alpha);
        let tint = multiscatter_fresnel(self.f_avg, self.alpha);
        single.add(tint.scale(lobe))
    }

    /// The probability of choosing the single-scatter `GGX` strategy, clamped so
    /// both strategies retain enough density for a stable one-sample estimator.
    fn single_scatter_probability(&self) -> f32 {
        average_albedo(self.alpha).clamp(0.1, 0.9)
    }

    /// The combined solid-angle density [`Self::sample`] assigns to `(wo, wi)`.
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
        let ggx = self.ggx.reflection_pdf(cos_o, cos_h);
        let diffuse = cosine_hemisphere_pdf(normal, wi);
        let p_ss = self.single_scatter_probability();
        p_ss * ggx + (1.0 - p_ss) * diffuse
    }

    /// Importance-samples the combined lobe with one-sample multiple importance
    /// sampling between the `GGX` `VNDF` and cosine strategies.
    ///
    /// Returns `None` for a degenerate sample so the caller terminates the path
    /// rather than dividing by zero.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut Rng,
    ) -> Option<SchlickMultiscatterSample> {
        // The integrator passes the raw geometric normal; orient it toward the
        // viewer so a back-facing hit still scatters.
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        let p_ss = self.single_scatter_probability();
        // Draw the strategy selector first so the two branches consume the rng
        // deterministically.
        let direction = if rng.next_f32() < p_ss {
            let half = self.ggx.sample_half_vector(wo, normal, rng)?;
            if wo.dot(half) <= 0.0 {
                return None;
            }
            wo.negate().reflect(half).normalize_or_zero()
        } else {
            cosine_sample_hemisphere(normal, rng).direction
        };
        let cos_i = normal.dot(direction);
        if cos_i <= 0.0 {
            return None;
        }
        let pdf = self.pdf(wo, direction, normal);
        if pdf <= 0.0 {
            return None;
        }
        Some(SchlickMultiscatterSample {
            direction,
            value: self.evaluate(wo, direction, normal),
            pdf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    /// A near-perfect white reflector (`F0 = 1`) should return almost all of its
    /// energy once the compensation lobe is added, and clearly more than the
    /// bare single-scatter lobe, without ever exceeding the furnace ceiling.
    #[test]
    fn white_furnace_recovers_lost_energy() {
        let f0 = Vec3::ONE;
        let roughness = 0.85;
        let ms = SchlickMultiscatterConductor::new(f0, roughness);
        let wo = unit(0.4, 0.9, 0.0);
        let mut rng = Rng::seed(23);
        let samples = 80_000u32;
        let mut sum_ms = 0.0f64;
        let mut sum_ss = 0.0f64;
        for _ in 0..samples {
            if let Some(s) = ms.sample(wo, N, &mut rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                sum_ms += f64::from(s.value.x * cos_i / s.pdf);
                let ss = ms.single_scatter(wo, s.direction, N);
                sum_ss += f64::from(ss.x * cos_i / s.pdf);
            }
        }
        let albedo_ms = sum_ms / f64::from(samples);
        let albedo_ss = sum_ss / f64::from(samples);
        assert!(
            albedo_ms > albedo_ss + 0.05,
            "ms {albedo_ms} should exceed single-scatter {albedo_ss}"
        );
        assert!(albedo_ms > 0.9, "white furnace too dark: {albedo_ms}");
        assert!(
            albedo_ms <= 1.0 + 1e-2,
            "furnace gained energy: {albedo_ms}"
        );
    }

    /// A sampled direction stays in the view hemisphere and reports the same
    /// value and density [`SchlickMultiscatterConductor::evaluate`] and
    /// [`SchlickMultiscatterConductor::pdf`] recompute for the same pair.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let ms = SchlickMultiscatterConductor::new(Vec3::new(0.95, 0.64, 0.54), 0.6);
        let wo = unit(0.3, 0.9, 0.1);
        let mut rng = Rng::seed(44);
        for _ in 0..2000 {
            if let Some(s) = ms.sample(wo, N, &mut rng) {
                assert!(N.dot(s.direction) > 0.0, "sample below surface");
                let eval = ms.evaluate(wo, s.direction, N);
                let pdf = ms.pdf(wo, s.direction, N);
                let val_tol = 1e-3 * s.value.length().max(1.0);
                let pdf_tol = 1e-3 * pdf.max(1.0);
                assert!(s.value.sub(eval).length() <= val_tol, "value mismatch");
                assert!((s.pdf - pdf).abs() <= pdf_tol, "pdf mismatch");
            }
        }
    }

    /// The closed-form `Schlick` average reflectance reproduces a direct
    /// cosine-weighted midpoint integral of `fresnel_schlick`.
    #[test]
    fn average_fresnel_matches_numeric_integral() {
        let f0 = Vec3::new(0.95, 0.64, 0.54);
        let closed = schlick_average_fresnel(f0);
        let nodes = 4096u32;
        let mut acc = Vec3::ZERO;
        for i in 0..nodes {
            let mu = (i as f32 + 0.5) / nodes as f32;
            let weight = 2.0 * mu / nodes as f32;
            acc = acc.add(fresnel_schlick(f0, mu).scale(weight));
        }
        assert!(
            closed.sub(acc).length() <= 2e-3,
            "closed {closed:?} vs {acc:?}"
        );
    }

    /// A smooth conductor loses almost no energy, so the multiple-scatter lobe
    /// is negligible and the combined `BRDF` matches the single-scatter one.
    #[test]
    fn smooth_conductor_matches_single_scatter() {
        let ms = SchlickMultiscatterConductor::new(Vec3::new(0.95, 0.64, 0.54), 0.02);
        let wo = unit(0.2, 0.95, 0.1);
        let wi = unit(-0.15, 0.96, 0.1);
        let a = ms.evaluate(wo, wi, N);
        let b = ms.single_scatter(wo, wi, N);
        assert!(a.sub(b).length() <= 1e-2 * b.length().max(1.0));
    }
}
