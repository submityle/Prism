//! Energy-conserving anisotropic rough dielectric with Turquin
//! multiple-scattering compensation.
//!
//! [`crate::reference_pt::rough_dielectric_aniso::AnisoRoughDielectric`]
//! evaluates only the single-scatter microfacet lobes, so it leaks the energy
//! that bounces several times between micro-facets before escaping; brushed or
//! drawn rough glass therefore darkens as roughness rises, exactly like its
//! isotropic sibling. This module wraps the anisotropic lobe with the same
//! multiplicative compensation used by
//! [`crate::reference_pt::dielectric_ms`]: it divides the single-scatter
//! throughput by its directional albedo `E` and multiplies by the smooth
//! ceiling `E_smooth`, restoring the energy Smith masking drops while
//! respecting the radiance-mode `1 / eta^2` compression.
//!
//! The directional-albedo tables of
//! [`crate::reference_pt::dielectric_energy`] are indexed by a single scalar
//! width, so the anisotropic lobe is reduced to its *isotropic-equivalent*
//! width `alpha = sqrt(alpha_x * alpha_y)` — the geometric mean that preserves
//! the lobe's projected microfacet area and therefore its total masked energy,
//! the same reduction the anisotropic conductor uses in
//! [`crate::reference_pt::conductor_aniso_ms`]. The compensation is a single
//! view-dependent scalar `k(eta, alpha, mu_o) = E_smooth / E`, so it leaves the
//! sampling distribution and the solid-angle density untouched and only
//! rescales the lobe value, keeping importance sampling exactly as unbiased as
//! the base lobe.
//!
//! As in Turquin's original formulation (and the production energy terms
//! shipped by `UE` and Frostbite) the factor is *view-referenced*, which makes
//! the compensated lobe furnace-correct but not strictly reciprocal; that is
//! traded for a closed-form, single-lookup correction with no extra sampling
//! cost.

use super::dielectric_energy::compensation_factor;
use super::microfacet_aniso::GgxAnisotropic;
use super::rough_dielectric::RoughDielectricSample;
use super::rough_dielectric_aniso::AnisoRoughDielectric;
use super::sampler::Rng;
use super::Vec3;

/// The geometric-mean isotropic-equivalent width of an anisotropic lobe.
fn equivalent_alpha(dist: GgxAnisotropic) -> f32 {
    (dist.alpha_x * dist.alpha_y).sqrt()
}

/// An anisotropic rough dielectric that adds Turquin multiple-scattering
/// compensation on top of the single-scatter anisotropic reflect-and-refract
/// lobes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisoMultiscatterDielectric {
    /// The single-scatter anisotropic rough dielectric providing the base lobes.
    base: AnisoRoughDielectric,
    /// Relative index `eta_t / eta_i`, used to index the energy tables.
    eta: f32,
    /// The isotropic-equivalent `GGX` width `sqrt(alpha_x * alpha_y)` used to
    /// index the scalar energy-compensation tables.
    alpha: f32,
}

impl AnisoMultiscatterDielectric {
    /// Builds an energy-conserving anisotropic rough dielectric from its
    /// relative `ior`, per-channel reflected and transmitted tints, and explicit
    /// `GGX` widths `alpha_x`/`alpha_y`.
    #[must_use]
    pub fn new(
        ior: f32,
        reflectance: Vec3,
        transmittance: Vec3,
        alpha_x: f32,
        alpha_y: f32,
    ) -> Self {
        let dist = GgxAnisotropic::new(alpha_x, alpha_y);
        Self {
            base: AnisoRoughDielectric::new(ior, reflectance, transmittance, alpha_x, alpha_y),
            eta: ior,
            alpha: equivalent_alpha(dist),
        }
    }

    /// Builds an energy-conserving anisotropic rough dielectric from a
    /// perceptual `roughness` and `anisotropy` pair, using the same Disney /
    /// `UE` (Burley) remap as
    /// [`AnisoRoughDielectric::from_roughness_anisotropy`].
    #[must_use]
    pub fn from_roughness_anisotropy(
        ior: f32,
        reflectance: Vec3,
        transmittance: Vec3,
        roughness: f32,
        anisotropy: f32,
    ) -> Self {
        let dist = GgxAnisotropic::from_roughness_anisotropy(roughness, anisotropy);
        Self {
            base: AnisoRoughDielectric::from_roughness_anisotropy(
                ior,
                reflectance,
                transmittance,
                roughness,
                anisotropy,
            ),
            eta: ior,
            alpha: equivalent_alpha(dist),
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
    use crate::reference_pt::dielectric_energy::smooth_albedo;

    /// The macroscopic surface normal used throughout the tests (`+y`).
    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Builds a clear (untinted, non-absorbing) compensated anisotropic glass.
    fn clear_glass(alpha_x: f32, alpha_y: f32) -> AnisoMultiscatterDielectric {
        AnisoMultiscatterDielectric::new(1.5, Vec3::ONE, Vec3::ONE, alpha_x, alpha_y)
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

    /// A rough compensated anisotropic dielectric recovers energy: its furnace
    /// albedo exceeds the darkened single-scatter one and lands near the smooth
    /// ceiling `R + (1 - R) / eta^2`.
    #[test]
    fn multiple_scattering_recovers_lost_energy() {
        let (ax, ay) = (0.9f32, 0.55f32);
        let ms = clear_glass(ax, ay);
        let base = AnisoRoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, ax, ay);
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
            (a_ms - ceiling).abs() < 4.0e-2,
            "ms albedo {a_ms} should reach smooth ceiling {ceiling}"
        );
    }

    /// A smooth compensated dielectric has nothing to recover, so its lobe
    /// equals the single-scatter lobe.
    #[test]
    fn smooth_dielectric_matches_single_scatter() {
        let ms = clear_glass(0.02, 0.02);
        let base = AnisoRoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, 0.02, 0.02);
        let wo = Vec3::new(0.3, 0.9, 0.1).normalize_or_zero();
        let wi = Vec3::new(-0.25, 0.92, 0.05).normalize_or_zero();
        let a = ms.evaluate(wo, wi, N);
        let b = base.evaluate(wo, wi, N);
        assert!(a.sub(b).length() <= 1e-2 * b.length().max(1.0));
    }

    /// The sampled value and density agree with independent
    /// [`AnisoMultiscatterDielectric::evaluate`]/[`AnisoMultiscatterDielectric::pdf`].
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let ms = clear_glass(0.4, 0.12);
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

    /// With equal widths the compensated anisotropic lobe must agree with the
    /// isotropic Turquin compensation of the same width.
    #[test]
    fn reduces_to_isotropic_when_widths_match() {
        use crate::reference_pt::dielectric_ms::MultiscatterDielectric;
        let roughness = 0.6f32;
        let alpha = roughness * roughness;
        let aniso = AnisoMultiscatterDielectric::new(1.5, Vec3::ONE, Vec3::ONE, alpha, alpha);
        let iso = MultiscatterDielectric::new(1.5, Vec3::ONE, Vec3::ONE, roughness);
        let wo = Vec3::new(0.3, 0.9, 0.1).normalize_or_zero();
        for wi in [
            Vec3::new(-0.2, 0.95, 0.1).normalize_or_zero(),
            Vec3::new(0.1, -0.9, 0.15).normalize_or_zero(),
        ] {
            let a = aniso.evaluate(wo, wi, N);
            let b = iso.evaluate(wo, wi, N);
            assert!(a.sub(b).length() < 1e-3, "{a:?} vs {b:?}");
        }
    }
}
