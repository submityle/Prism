//! Energy-conserving rough dielectric with Turquin multiple-scattering
//! compensation.
//!
//! [`crate::reference_pt::rough_dielectric::RoughDielectric`] evaluates only
//! the single-scatter microfacet lobes, so it leaks the energy that bounces
//! several times between micro-facets before escaping; rough glass and water
//! therefore render too dark as roughness rises. This module wraps that lobe
//! with the multiplicative compensation of
//! [`crate::reference_pt::dielectric_energy`]: it divides the single-scatter
//! throughput by its directional albedo `E` and multiplies by the smooth
//! ceiling `E_smooth`, restoring exactly the energy Smith masking drops while
//! respecting the radiance-mode `1 / eta^2` compression (so a lossless furnace
//! climbs back to the smooth ceiling, not past it).
//!
//! The compensation is a single view-dependent scalar
//! `k(eta, alpha, mu_o) = E_smooth / E`, so it leaves the sampling distribution
//! and the solid-angle density untouched and only rescales the lobe value. This
//! keeps importance sampling exactly as unbiased as the base lobe.
//!
//! The factor is *view-referenced* (it depends on the view cosine `mu_o` only),
//! which — as in Turquin's original formulation and the production energy terms
//! shipped by `UE` and Frostbite — makes the compensated lobe furnace-correct
//! but not strictly reciprocal. Reciprocity is traded for a closed-form,
//! single-lookup correction with no extra sampling cost.

use super::dielectric_energy::compensation_factor;
use super::microfacet::GgxIsotropic;
use super::rough_dielectric::{RoughDielectric, RoughDielectricSample};
use super::sampler::Rng;
use super::Vec3;

/// A rough dielectric that adds Turquin multiple-scattering compensation on top
/// of the single-scatter microfacet reflect-and-refract lobes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiscatterDielectric {
    /// The single-scatter rough dielectric providing the base lobes.
    base: RoughDielectric,
    /// Relative index `eta_t / eta_i`, used to index the energy tables.
    eta: f32,
    /// The `GGX` width `alpha` used to index the energy tables.
    alpha: f32,
}

impl MultiscatterDielectric {
    /// Builds an energy-conserving rough dielectric from its relative `ior`,
    /// per-channel reflected and transmitted tints, and the perceptual
    /// `roughness` in `[0, 1]`.
    #[must_use]
    pub fn new(ior: f32, reflectance: Vec3, transmittance: Vec3, roughness: f32) -> Self {
        Self {
            base: RoughDielectric::new(ior, reflectance, transmittance, roughness),
            eta: ior,
            alpha: GgxIsotropic::from_roughness(roughness).alpha,
        }
    }

    /// The view-dependent compensation scalar `E_smooth / E` for a view
    /// direction whose cosine with `normal` is `cos_o`.
    fn factor(&self, cos_o: f32) -> f32 {
        compensation_factor(self.eta, self.alpha, cos_o.abs())
    }

    /// Evaluates the compensated `BSDF` value `k(mu_o) * f_ss(wo, wi)`.
    ///
    /// Returns [`Vec3::ZERO`] wherever the base single-scatter lobe does.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let single = self.base.evaluate(wo, wi, normal);
        single.scale(self.factor(normal.dot(wo)))
    }

    /// The solid-angle density of a sampled direction.
    ///
    /// A scalar multiplier does not change the sampling distribution, so this is
    /// exactly the base single-scatter density.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        self.base.pdf(wo, wi, normal)
    }

    /// Importance-samples the compensated lobe.
    ///
    /// Draws a direction from the base single-scatter distribution and rescales
    /// its value by the compensation factor; the density is unchanged. Returns
    /// `None` for a degenerate base sample so the caller terminates the path.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<RoughDielectricSample> {
        let s = self.base.sample(wo, normal, rng)?;
        let k = self.factor(normal.dot(wo));
        Some(RoughDielectricSample {
            direction: s.direction,
            value: s.value.scale(k),
            pdf: s.pdf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The macroscopic surface normal used throughout the tests (`+y`).
    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Builds a clear (untinted, non-absorbing) compensated glass.
    fn clear_glass(roughness: f32) -> MultiscatterDielectric {
        MultiscatterDielectric::new(1.5, Vec3::ONE, Vec3::ONE, roughness)
    }

    /// Monte Carlo furnace albedo (reflected plus transmitted) for a `BSDF`
    /// exposing `sample`.
    fn furnace_albedo(
        sample: impl Fn(Vec3, &mut Rng) -> Option<RoughDielectricSample>,
        wo: Vec3,
        rng: &mut Rng,
        samples: u32,
    ) -> f32 {
        let mut sum = 0.0f64;
        for _ in 0..samples {
            if let Some(s) = sample(wo, rng) {
                let cos_i = N.dot(s.direction).abs();
                sum += f64::from(s.value.x * cos_i / s.pdf);
            }
        }
        (sum / f64::from(samples)) as f32
    }

    /// A rough compensated dielectric recovers energy: its furnace albedo
    /// exceeds the darkened single-scatter one and lands near the smooth
    /// ceiling `R + (1 - R) / eta^2`.
    #[test]
    fn multiple_scattering_recovers_lost_energy() {
        use super::super::dielectric_energy::smooth_albedo;
        let roughness = 0.95f32;
        let ms = clear_glass(roughness);
        let base = RoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, roughness);
        let mu = 0.6f32;
        let sin_o = (1.0 - mu * mu).max(0.0).sqrt();
        let wo = Vec3::new(sin_o, mu, 0.0).normalize_or_zero();
        let mut rng = Rng::seed(0x5c47_7e21);
        let n = 1u32 << 17;
        let a_ms = furnace_albedo(|w, r| ms.sample(w, N, r), wo, &mut rng, n);
        let a_base = furnace_albedo(|w, r| base.sample(w, N, r), wo, &mut rng, n);
        let ceiling = smooth_albedo(1.5, mu);
        assert!(
            a_ms > a_base + 0.02,
            "ms {a_ms} should exceed single-scatter {a_base}"
        );
        assert!(
            (a_ms - ceiling).abs() < 3.0e-2,
            "ms albedo {a_ms} should reach smooth ceiling {ceiling}"
        );
    }

    /// A smooth compensated dielectric has nothing to recover, so its lobe
    /// equals the single-scatter lobe.
    #[test]
    fn smooth_dielectric_matches_single_scatter() {
        let ms = clear_glass(0.02);
        let base = RoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, 0.02);
        let wo = Vec3::new(0.3, 0.9, 0.1).normalize_or_zero();
        let wi = Vec3::new(-0.25, 0.92, 0.05).normalize_or_zero();
        let a = ms.evaluate(wo, wi, N);
        let b = base.evaluate(wo, wi, N);
        assert!(a.sub(b).length() <= 1e-2 * b.length().max(1.0));
    }

    /// The sampled value and density agree with independent
    /// [`MultiscatterDielectric::evaluate`]/[`MultiscatterDielectric::pdf`].
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let ms = clear_glass(0.4);
        let wo = Vec3::new(0.3, 0.85, 0.1).normalize_or_zero();
        let mut rng = Rng::seed(0xC0FFEE);
        let mut checked = 0u32;
        for _ in 0..40_000 {
            let Some(s) = ms.sample(wo, N, &mut rng) else {
                continue;
            };
            let p = ms.pdf(wo, s.direction, N);
            assert!(
                (p - s.pdf).abs() <= 1e-3 * s.pdf.max(1.0) + 1e-5,
                "pdf mismatch: sample {} vs pdf {p}",
                s.pdf
            );
            let v = ms.evaluate(wo, s.direction, N);
            let tol = 1e-3 * s.value.length().max(1.0) + 1e-5;
            assert!(s.value.sub(v).length() <= tol, "value mismatch");
            checked += 1;
        }
        assert!(checked > 1000, "too few live samples: {checked}");
    }
}
