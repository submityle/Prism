//! Energy-aware clear-coat over a rough conductor, following the production
//! `UE` clear-coat shading model.
//!
//! The reusable dielectric slab lives in [`super::coat`]; this module layers it
//! over a metallic [`Conductor`] base. The coat contributes an additive `GGX`
//! dielectric highlight and attenuates the base lobe by the Fresnel
//! transmission entering and leaving the coat,
//! `(1 - F_coat(cos_o)) * (1 - F_coat(cos_i))`. Both lobes reflect into the
//! upper hemisphere, so importance sampling draws one lobe by a Fresnel-weighted
//! probability and reports the combined value and the combined one-sample
//! density, exactly like the multi-lobe `FresnelBlend` plastic.

use super::coat::CoatLayer;
use super::conductor::Conductor;
#[cfg(test)]
use super::sampler::Rng;
use super::sampler::SampleSource;
use super::Vec3;

/// The outcome of importance-sampling a [`ClearcoatConductor`].
#[derive(Clone, Copy, Debug)]
pub struct ClearcoatSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The combined `BSDF` value `f_r(wo, wi)` of both lobes (no cosine applied).
    pub value: Vec3,
    /// The combined one-sample solid-angle density of `direction`.
    pub pdf: f32,
}

/// A rough conductor viewed through a dielectric clear coat.
///
/// The base metal is an ordinary [`Conductor`]; the coat is a reusable
/// [`CoatLayer`] dielectric interface scaled by its own weight in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearcoatConductor {
    /// The rough-conductor base layer seen through the coat.
    base: Conductor,
    /// The thin dielectric clear coat layered over the base.
    coat: CoatLayer,
    /// Two-way normal-incidence transmittance of the coat medium.
    coat_color: Vec3,
    /// Relative index used for the coat's refracted substrate path.
    coat_ior: f32,
    /// Presence of the coat, used to mix its absorption with the uncovered base.
    coat_weight: f32,
}

impl ClearcoatConductor {
    /// Builds a clear-coated conductor from the base metal's complex index
    /// `eta + i*k` and perceptual `roughness`, plus the coat's perceptual
    /// `coat_roughness`, relative `coat_ior`, `coat_weight` in `[0, 1]`, and
    /// `coat_color` tint.
    #[must_use]
    pub fn new(
        eta: Vec3,
        k: Vec3,
        roughness: f32,
        coat_roughness: f32,
        coat_ior: f32,
        coat_weight: f32,
        coat_color: Vec3,
        coat_darkening: f32,
    ) -> Self {
        let base_albedo = super::conductor::average_fresnel_conductor(eta, k);
        let base_albedo = (base_albedo.x + base_albedo.y + base_albedo.z) / 3.0;
        Self {
            base: Conductor::new(eta, k, roughness),
            coat: CoatLayer::new(
                coat_roughness,
                coat_ior,
                coat_weight,
                Vec3::ONE,
                coat_darkening,
                base_albedo,
                roughness,
            ),
            coat_color: Vec3::new(
                coat_color.x.clamp(0.0, 1.0),
                coat_color.y.clamp(0.0, 1.0),
                coat_color.z.clamp(0.0, 1.0),
            ),
            coat_ior: coat_ior.max(1.0 + 1.0e-3),
            coat_weight: coat_weight.clamp(0.0, 1.0),
        }
    }

    /// Evaluates the combined clear-coat `BSDF`
    /// `f_coat + (1 - F_o)(1 - F_i) * f_base`.
    ///
    /// Returns [`Vec3::ZERO`] when either direction is below the surface.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        let t_o = self.coat.transmission(cos_o);
        let t_i = self.coat.transmission(cos_i);
        let absorption = coat_absorption(self.coat_color, cos_i, cos_o, self.coat_ior);
        let absorption = Vec3::ONE.add(absorption.sub(Vec3::ONE).scale(self.coat_weight));
        let base = self
            .base
            .evaluate(wo, wi, normal)
            .mul(absorption)
            .scale(t_o * t_i * self.coat.base_darkening(cos_o));
        base.add(self.coat.lobe(wo, wi, normal, cos_o, cos_i))
    }

    /// The combined one-sample solid-angle density of `(wo, wi)`.
    ///
    /// Mirrors the lobe-selection mixture used by [`Self::sample`], so the
    /// Monte Carlo estimator stays unbiased. Zero when either direction is below
    /// the surface.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let p_coat = self.coat.selection_prob(cos_o);
        let coat_pdf = self.coat.pdf(wo, wi, normal, cos_o);
        let base_pdf = self.base.pdf(wo, wi, normal);
        p_coat * coat_pdf + (1.0 - p_coat) * base_pdf
    }

    /// Importance-samples the clear-coat lobe mixture.
    ///
    /// Draws the coat lobe with probability [`CoatLayer::selection_prob`] and the
    /// base lobe otherwise, then reports the combined value and the combined
    /// density so the integrator's `value * cos_i / pdf` weight is correct for a
    /// one-sample multiple-importance mixture. Returns `None` for a degenerate
    /// sample so the caller terminates the path.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<ClearcoatSample> {
        // The integrator passes the raw geometric normal; orient it into the
        // view hemisphere so a back-facing hit still scatters.
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        let p_coat = self.coat.selection_prob(cos_o);
        let wi = if rng.next_f32() < p_coat {
            self.coat.sample_direction(wo, normal, rng)?
        } else {
            self.base.sample(wo, normal, rng)?.direction
        };
        let pdf = self.pdf(wo, wi, normal);
        if pdf <= 0.0 {
            return None;
        }
        Some(ClearcoatSample {
            direction: wi,
            value: self.evaluate(wo, wi, normal),
            pdf,
        })
    }
}

/// OpenPBR two-way coat transmittance applied to the substrate, never the
/// coat's own specular reflection. `coat_color` is defined at normal incidence.
fn coat_absorption(coat_color: Vec3, mu_i: f32, mu_o: f32, eta: f32) -> Vec3 {
    let refracted_cosine = |mu: f32| {
        let mu = mu.clamp(0.0, 1.0);
        (1.0 - (1.0 - mu * mu) / (eta * eta)).max(0.0).sqrt()
    };
    let ti = refracted_cosine(mu_i);
    let to = refracted_cosine(mu_o);
    if ti <= 0.0 || to <= 0.0 {
        return Vec3::ZERO;
    }
    let exponent = 0.5 * (ti.recip() + to.recip());
    Vec3::new(
        coat_color.x.powf(exponent),
        coat_color.y.powf(exponent),
        coat_color.z.powf(exponent),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_pt::metal::Metal;

    /// The macroscopic surface normal used throughout the tests (`+y`).
    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Builds a clear-coated gold with the given coat weight and roughness.
    fn coated_gold(coat_weight: f32, coat_roughness: f32) -> ClearcoatConductor {
        coated_gold_with_darkening(coat_weight, coat_roughness, 1.0)
    }

    fn coated_gold_with_darkening(
        coat_weight: f32,
        coat_roughness: f32,
        coat_darkening: f32,
    ) -> ClearcoatConductor {
        let (eta, k) = Metal::Gold.complex_ior();
        ClearcoatConductor::new(
            eta,
            k,
            0.3,
            coat_roughness,
            1.5,
            coat_weight,
            Vec3::ONE,
            coat_darkening,
        )
    }

    /// A view direction at the given cosine to the normal, tilted in the `x-y`
    /// plane.
    fn view_at(mu: f32) -> Vec3 {
        let sin_o = (1.0 - mu * mu).max(0.0).sqrt();
        Vec3::new(sin_o, mu, 0.0).normalize_or_zero()
    }

    /// Monte Carlo directional albedo (reflected hemisphere) of a clear coat.
    fn directional_albedo(coat: &ClearcoatConductor, wo: Vec3, rng: &mut Rng, samples: u32) -> f32 {
        let mut sum = 0.0f64;
        for _ in 0..samples {
            if let Some(s) = coat.sample(wo, N, rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                sum += f64::from(s.value.max_component() * cos_i / s.pdf);
            }
        }
        (sum / f64::from(samples)) as f32
    }

    /// A weightless coat must reproduce the bare conductor's value, density, and
    /// sampled directions exactly.
    #[test]
    fn zero_weight_coat_matches_bare_conductor() {
        let coat = coated_gold(0.0, 0.05);
        let (eta, k) = Metal::Gold.complex_ior();
        let bare = Conductor::new(eta, k, 0.3);
        let wo = view_at(0.8);
        let wi = Vec3::new(-0.4, 0.9, 0.1).normalize_or_zero();
        assert!(
            coat.evaluate(wo, wi, N)
                .sub(bare.evaluate(wo, wi, N))
                .length()
                < 1e-6
        );
        assert!((coat.pdf(wo, wi, N) - bare.pdf(wo, wi, N)).abs() < 1e-6);
    }

    #[test]
    fn coat_darkening_scales_the_conductor_substrate_but_not_the_coat_lobe() {
        let neutral = coated_gold_with_darkening(1.0, 0.2, 0.0);
        let darkened = coated_gold_with_darkening(1.0, 0.2, 1.0);
        let wo = view_at(0.6);
        let wi = Vec3::new(-0.2, 0.94, 0.27).normalize_or_zero();
        let cos_o = N.dot(wo).max(0.0);
        let cos_i = N.dot(wi).max(0.0);

        let transmission = darkened.coat.transmission(cos_o) * darkened.coat.transmission(cos_i);
        let base = darkened
            .base
            .evaluate(wo, wi, N)
            .scale(transmission * darkened.coat.base_darkening(cos_o));
        let coat_lobe = darkened.coat.lobe(wo, wi, N, cos_o, cos_i);
        let expected = base.add(coat_lobe);
        let actual = darkened.evaluate(wo, wi, N);
        assert!(actual.sub(expected).length() < 1.0e-6);
        assert!(darkened.coat.base_darkening(cos_o) < 1.0);
        assert!(
            actual.max_component() < neutral.evaluate(wo, wi, N).max_component(),
            "darkening should attenuate the substrate while preserving the coat highlight"
        );
    }

    #[test]
    fn partial_coat_darkening_is_applied_once_to_the_substrate() {
        let neutral = coated_gold_with_darkening(0.45, 0.2, 0.0);
        let darkened = coated_gold_with_darkening(0.45, 0.2, 1.0);
        let wo = view_at(0.31);
        let wi = Vec3::new(-0.25, 0.91, 0.32).normalize_or_zero();
        let cos_o = N.dot(wo);
        let cos_i = N.dot(wi);

        let throughput = darkened.coat.transmission(cos_o) * darkened.coat.transmission(cos_i);
        let substrate = darkened.base.evaluate(wo, wi, N);
        let darkening = darkened.coat.base_darkening(cos_o);
        let expected_base = substrate.scale(throughput * darkening);
        let expected_coat = darkened.coat.lobe(wo, wi, N, cos_o, cos_i);
        let expected = expected_base.add(expected_coat);
        let actual = darkened.evaluate(wo, wi, N);

        assert!(actual.sub(expected).length() < 1.0e-6);
        assert!(darkening < 1.0);
        assert!(
            expected_coat
                .sub(neutral.coat.lobe(wo, wi, N, cos_o, cos_i))
                .length()
                < 1.0e-6,
            "darkening must not affect the additive coat lobe"
        );
        assert!(
            actual.sub(neutral.evaluate(wo, wi, N)).length() > 1.0e-6,
            "partial coat darkening must measurably attenuate its substrate"
        );
    }

    #[test]
    fn coat_color_absorbs_the_substrate_without_tinting_the_highlight() {
        let neutral = ClearcoatConductor::new(
            Metal::Gold.complex_ior().0,
            Metal::Gold.complex_ior().1,
            0.3,
            0.2,
            1.5,
            0.45,
            Vec3::ONE,
            0.0,
        );
        let tinted = ClearcoatConductor::new(
            Metal::Gold.complex_ior().0,
            Metal::Gold.complex_ior().1,
            0.3,
            0.2,
            1.5,
            0.45,
            Vec3::new(0.25, 0.7, 0.9),
            0.0,
        );
        let wo = view_at(0.47);
        let wi = view_at(0.76);
        let cos_o = N.dot(wo);
        let cos_i = N.dot(wi);
        let transmission = tinted.coat.transmission(cos_o) * tinted.coat.transmission(cos_i);
        let absorption = coat_absorption(tinted.coat_color, cos_i, cos_o, tinted.coat_ior);
        let mixed_absorption = Vec3::ONE.add(absorption.sub(Vec3::ONE).scale(tinted.coat_weight));
        let substrate = tinted.base.evaluate(wo, wi, N).scale(transmission);
        let expected_delta = substrate.mul(mixed_absorption.sub(Vec3::ONE));
        let actual_delta = tinted.evaluate(wo, wi, N).sub(neutral.evaluate(wo, wi, N));

        assert!(actual_delta.sub(expected_delta).length() < 1.0e-6);
        assert!(actual_delta.x < actual_delta.y && actual_delta.y < actual_delta.z);
    }

    /// Adding a coat injects an extra specular highlight: near the mirror
    /// direction the coated surface reflects strictly more than the bare base.
    #[test]
    fn coat_adds_specular_highlight() {
        let coat = coated_gold(1.0, 0.02);
        let (eta, k) = Metal::Gold.complex_ior();
        let bare = Conductor::new(eta, k, 0.3);
        let wo = view_at(0.7);
        // The mirror direction of `wo` about the `+y` normal.
        let wi = Vec3::new(-wo.x, wo.y, -wo.z).normalize_or_zero();
        let c = coat.evaluate(wo, wi, N).max_component();
        let b = bare.evaluate(wo, wi, N).max_component();
        assert!(c > b, "coated {c} should exceed bare {b}");
    }

    /// The coat conserves energy: a white coat over gold never reflects more
    /// light than arrives, so its directional albedo stays at or below one.
    #[test]
    fn coat_is_energy_conserving() {
        let coat = coated_gold(1.0, 0.2);
        let mut rng = Rng::seed(0x0C0A_7C0A);
        for mu in [0.2f32, 0.5, 0.9] {
            let a = directional_albedo(&coat, view_at(mu), &mut rng, 1u32 << 16);
            assert!(a <= 1.0 + 2e-2, "albedo {a} at mu {mu} exceeds unity");
        }
    }

    /// Sampling is consistent with independent [`ClearcoatConductor::evaluate`]
    /// and [`ClearcoatConductor::pdf`] for both lobes.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let coat = coated_gold(0.7, 0.15);
        let wo = view_at(0.75);
        let mut rng = Rng::seed(0xC0A7_5EED);
        let mut checked = 0u32;
        for _ in 0..40_000 {
            let Some(s) = coat.sample(wo, N, &mut rng) else {
                continue;
            };
            let p = coat.pdf(wo, s.direction, N);
            assert!(
                (p - s.pdf).abs() <= 1e-3 * s.pdf.max(1.0) + 1e-5,
                "pdf mismatch: sample {} vs pdf {p}",
                s.pdf
            );
            let v = coat.evaluate(wo, s.direction, N);
            let tol = 1e-3 * s.value.length().max(1.0) + 1e-5;
            assert!(s.value.sub(v).length() <= tol, "value mismatch");
            checked += 1;
        }
        assert!(checked > 1000, "too few live samples: {checked}");
    }

    /// Every sampled direction reflects into the upper hemisphere and carries a
    /// finite, positive density.
    #[test]
    fn sampled_directions_reflect_upward() {
        let coat = coated_gold(0.9, 0.1);
        let wo = view_at(0.85);
        let mut rng = Rng::seed(0x5EED_0C0A);
        for _ in 0..5_000 {
            if let Some(s) = coat.sample(wo, N, &mut rng) {
                assert!(N.dot(s.direction) > 0.0, "below surface");
                assert!(s.pdf > 0.0 && s.pdf.is_finite(), "bad pdf {}", s.pdf);
                assert!(s.value.is_finite(), "non-finite value");
            }
        }
    }
}
