//! Energy-conserving rough conductor with Kulla-Conty multiple scattering.
//!
//! [`crate::reference_pt::conductor::Conductor`] evaluates only the
//! single-scatter `GGX` term, so it leaks the energy that bounces several times
//! between micro-facets before escaping; rough gold or aluminium therefore
//! renders too dark and desaturated. This module wraps the single-scatter lobe
//! with the Kulla-Conty compensation lobe from
//! [`crate::reference_pt::ggx_energy`], recovering exactly the missing
//! `1 - E(mu)` energy so a white furnace integrates back to one, and tinting
//! the recovered light by the metal's average `Fresnel` reflectance so the
//! added bounces also deepen the colour the way real rough metals do.
//!
//! The combined `BRDF` is `f_ss + F_ms * f_ms`, where `f_ss` is the exact
//! complex-index single-scatter conductor and `f_ms` is the near-diffuse
//! compensation lobe. Sampling uses a one-sample multiple-importance strategy:
//! with probability tied to the average albedo it draws the sharp `GGX`
//! direction, otherwise a cosine-weighted direction for the broad
//! multiple-scatter lobe, and always reports the combined density so the
//! estimator stays unbiased.

use super::conductor::{fresnel_conductor, Conductor};
use super::ggx_energy::{average_albedo, multiscatter_fresnel, multiscatter_lobe};
use super::microfacet::GgxIsotropic;
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, Rng};
use super::Vec3;

/// Number of quadrature nodes used to pre-integrate the average `Fresnel`
/// reflectance over the cosine-weighted hemisphere.
const FRESNEL_NODES: u32 = 32;

/// The outcome of importance-sampling a [`MultiscatterConductor`] lobe.
#[derive(Clone, Copy, Debug)]
pub struct MultiscatterSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The combined `BRDF` value `f_r(wo, wi)` for the sampled pair.
    pub value: Vec3,
    /// The combined solid-angle probability density of `direction`.
    pub pdf: f32,
}

/// A rough conductor that adds Kulla-Conty multiple-scattering compensation on
/// top of the exact single-scatter complex-index lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiscatterConductor {
    /// The exact single-scatter conductor providing `f_ss`.
    base: Conductor,
    /// The `GGX` width `alpha` used to index the energy-compensation tables.
    alpha: f32,
    /// The hemispherical-average `Fresnel` reflectance driving the colour of
    /// the recovered energy.
    f_avg: Vec3,
}

/// The hemispherical cosine-weighted average `Fresnel` reflectance
/// `F_avg = 2 * integral_0^1 F(mu) mu d mu`, evaluated by midpoint quadrature.
fn average_fresnel(eta: Vec3, k: Vec3) -> Vec3 {
    let mut acc = Vec3::ZERO;
    for i in 0..FRESNEL_NODES {
        let mu = (i as f32 + 0.5) / FRESNEL_NODES as f32;
        let weight = 2.0 * mu / FRESNEL_NODES as f32;
        acc = acc.add(fresnel_conductor(eta, k, mu).scale(weight));
    }
    acc
}

impl MultiscatterConductor {
    /// Builds an energy-conserving conductor from its complex index `eta + i*k`
    /// and a perceptual `roughness` in `[0, 1]`.
    #[must_use]
    pub fn new(eta: Vec3, k: Vec3, roughness: f32) -> Self {
        Self {
            base: Conductor::new(eta, k, roughness),
            alpha: GgxIsotropic::from_roughness(roughness).alpha,
            f_avg: average_fresnel(eta, k),
        }
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
        let single = self.base.evaluate(wo, wi, normal);
        let lobe = multiscatter_lobe(cos_o, cos_i, self.alpha);
        let tint = multiscatter_fresnel(self.f_avg, self.alpha);
        single.add(tint.scale(lobe))
    }

    /// The probability of choosing the single-scatter `GGX` strategy, clamped so
    /// both strategies retain enough density for a stable one-sample estimator.
    fn single_scatter_probability(&self) -> f32 {
        average_albedo(self.alpha).clamp(0.1, 0.9)
    }

    /// The combined solid-angle density [`Self::sample`] assigns to `(wo, wi)`,
    /// mixing the `GGX` and cosine strategies by their selection probabilities.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let p_ss = self.single_scatter_probability();
        let ggx = self.base.pdf(wo, wi, normal);
        let diffuse = cosine_hemisphere_pdf(normal, wi);
        p_ss * ggx + (1.0 - p_ss) * diffuse
    }

    /// Importance-samples the combined lobe with one-sample multiple importance
    /// sampling between the `GGX` and cosine strategies.
    ///
    /// Returns `None` for a degenerate sample so the caller terminates the path
    /// rather than dividing by zero.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<MultiscatterSample> {
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
            self.base.sample(wo, normal, rng)?.direction
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
        Some(MultiscatterSample {
            direction,
            value: self.evaluate(wo, direction, normal),
            pdf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Red/green/blue complex index of gold.
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

    /// The combined directional albedo of a near-perfect-reflector (unit
    /// `Fresnel`) rough conductor is closer to one than the single-scatter lobe
    /// alone: the compensation lobe recovers the energy Smith masking drops.
    #[test]
    fn multiple_scattering_recovers_lost_energy() {
        // A high-reflectance metal so the furnace target is close to one.
        let eta = Vec3::new(0.05, 0.05, 0.05);
        let k = Vec3::new(5.0, 5.0, 5.0);
        let roughness = 0.8;
        let ms = MultiscatterConductor::new(eta, k, roughness);
        let base = Conductor::new(eta, k, roughness);
        let wo = unit(0.4, 0.9, 0.0);
        let mut rng = Rng::seed(11);
        let samples = 60_000u32;
        let mut sum_ms = 0.0f64;
        let mut sum_base = 0.0f64;
        for _ in 0..samples {
            if let Some(s) = ms.sample(wo, N, &mut rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                sum_ms += f64::from(s.value.x * cos_i / s.pdf);
            }
            if let Some(s) = base.sample(wo, N, &mut rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                sum_base += f64::from(s.value.x * cos_i / s.pdf);
            }
        }
        let albedo_ms = sum_ms / f64::from(samples);
        let albedo_base = sum_base / f64::from(samples);
        assert!(
            albedo_ms > albedo_base + 0.02,
            "ms {albedo_ms} should exceed single-scatter {albedo_base}"
        );
        assert!(
            albedo_ms <= 1.0 + 1e-2,
            "furnace gained energy: {albedo_ms}"
        );
    }

    /// A sampled direction stays in the view hemisphere and reports the same
    /// value and density [`MultiscatterConductor::evaluate`]/[`MultiscatterConductor::pdf`]
    /// recompute for the same pair.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let ms = MultiscatterConductor::new(GOLD_ETA, GOLD_K, 0.6);
        let wo = unit(0.3, 0.9, 0.1);
        let mut rng = Rng::seed(42);
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

    /// A smooth conductor loses almost no energy, so the multiple-scatter lobe
    /// is negligible and the combined `BRDF` matches the single-scatter one.
    #[test]
    fn smooth_conductor_matches_single_scatter() {
        let ms = MultiscatterConductor::new(GOLD_ETA, GOLD_K, 0.02);
        let base = Conductor::new(GOLD_ETA, GOLD_K, 0.02);
        let wo = unit(0.2, 0.95, 0.1);
        let wi = unit(-0.15, 0.96, 0.1);
        let a = ms.evaluate(wo, wi, N);
        let b = base.evaluate(wo, wi, N);
        assert!(a.sub(b).length() <= 1e-2 * b.length().max(1.0));
    }
}
