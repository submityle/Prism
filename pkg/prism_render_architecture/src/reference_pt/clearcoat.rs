//! Energy-aware clear-coat: a smooth-to-rough dielectric coat layered over a
//! rough-conductor base, following the production `UE` clear-coat shading
//! model.
//!
//! A clear coat is the thin, near-colourless lacquer sprayed over car paint,
//! lacquered wood, or carbon fibre: a second specular interface sitting above
//! the pigmented or metallic base. Physically it is a dielectric slab, so it
//! reflects a Fresnel fraction of the incident light with its own (usually
//! very smooth) microfacet lobe and transmits the rest down to the base, which
//! then reflects back up through the slab a second time.
//!
//! Rather than tracing light through the slab with a position-free random walk,
//! this oracle uses the closed-form two-lobe model shipped by real-time
//! engines: the coat contributes an additive `GGX` dielectric highlight, and
//! the base lobe is attenuated by the Fresnel transmission entering and leaving
//! the coat, `(1 - F_coat(cos_o)) * (1 - F_coat(cos_i))`. The coat is treated
//! as optically thin, so it attenuates and tints but does not bend the base
//! direction — the standard engine approximation that keeps the model a sum of
//! two microfacet lobes with no extra sampling dimensions.
//!
//! Both lobes reflect into the upper hemisphere, so importance sampling draws
//! one lobe by a Fresnel-weighted probability and reports the combined value
//! and the combined one-sample density, exactly like the multi-lobe
//! `FresnelBlend` plastic. Only `sqrt`-based arithmetic is used (through the
//! Fresnel and `GGX` primitives), so the model stays within the transcendental
//! budget of the reference path tracer.

use super::conductor::Conductor;
use super::dielectric::fresnel_dielectric;
use super::microfacet::GgxIsotropic;
use super::sampler::Rng;
use super::{Vec3, EPS_LEN_SQ};

/// The smallest relative coat index the model accepts, keeping the coat a
/// genuine dielectric interface (an index of exactly one would be invisible).
const MIN_COAT_IOR: f32 = 1.0 + 1.0e-3;

/// The clamp window for the coat-versus-base lobe-selection probability.
///
/// Selecting a lobe strictly in proportion to the coat Fresnel term would stop
/// sampling the coat near normal incidence (where its reflectance is a few
/// percent) and starve the base near grazing. Clamping the probability into a
/// central window keeps both lobes alive, which lowers variance without biasing
/// the estimator (the matching density is used in [`ClearcoatConductor::pdf`]).
const MIN_COAT_PROB: f32 = 0.1;
/// The upper bound of the lobe-selection window; see [`MIN_COAT_PROB`].
const MAX_COAT_PROB: f32 = 0.9;

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
/// The base metal is an ordinary [`Conductor`]; the coat is an isotropic `GGX`
/// dielectric interface of relative index [`ClearcoatConductor::coat_ior`] whose
/// presence is scaled by [`ClearcoatConductor::coat_weight`] in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearcoatConductor {
    /// The rough-conductor base layer seen through the coat.
    base: Conductor,
    /// The isotropic `GGX` microfacet distribution of the coat highlight.
    coat: GgxIsotropic,
    /// Relative index of refraction `eta_t / eta_i` of the coat over the
    /// exterior medium, clamped to at least [`MIN_COAT_IOR`].
    coat_ior: f32,
    /// Coat presence in `[0, 1]`: `0` is a bare conductor, `1` a full coat.
    coat_weight: f32,
    /// Per-channel tint of the coat highlight (usually white for clear lacquer).
    coat_color: Vec3,
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
    ) -> Self {
        Self {
            base: Conductor::new(eta, k, roughness),
            coat: GgxIsotropic::from_roughness(coat_roughness),
            coat_ior: coat_ior.max(MIN_COAT_IOR),
            coat_weight: coat_weight.clamp(0.0, 1.0),
            coat_color,
        }
    }

    /// The effective macroscopic coat reflectance `weight * Fresnel(cos)` for a
    /// direction whose cosine with the normal is `cos`.
    fn coat_reflectance(&self, cos: f32) -> f32 {
        self.coat_weight * fresnel_dielectric(cos, 1.0, self.coat_ior)
    }

    /// The probability of sampling the coat lobe for a view cosine `cos_o`,
    /// clamped into the central window `[MIN_COAT_PROB, MAX_COAT_PROB]`.
    ///
    /// Returns zero for a weightless coat so a bare conductor always samples its
    /// own lobe.
    fn coat_selection_prob(&self, cos_o: f32) -> f32 {
        if self.coat_weight <= 0.0 {
            return 0.0;
        }
        self.coat_reflectance(cos_o)
            .clamp(MIN_COAT_PROB, MAX_COAT_PROB)
    }

    /// The coat's additive `GGX` dielectric highlight
    /// `weight * F * D * G2 / (4 cos_o cos_i)`.
    ///
    /// Returns [`Vec3::ZERO`] for a weightless coat or a degenerate half vector.
    fn coat_lobe(&self, wo: Vec3, wi: Vec3, normal: Vec3, cos_o: f32, cos_i: f32) -> Vec3 {
        if self.coat_weight <= 0.0 {
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
        let d = self.coat.distribution(cos_h);
        let g2 = self.coat.g2(cos_o, cos_i);
        let f = self.coat_weight * fresnel_dielectric(wo.dot(half).max(0.0), 1.0, self.coat_ior);
        self.coat_color.scale(f * d * g2 / (4.0 * cos_o * cos_i))
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
        let t_o = 1.0 - self.coat_reflectance(cos_o);
        let t_i = 1.0 - self.coat_reflectance(cos_i);
        let base = self.base.evaluate(wo, wi, normal).scale(t_o * t_i);
        base.add(self.coat_lobe(wo, wi, normal, cos_o, cos_i))
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
        let p_coat = self.coat_selection_prob(cos_o);
        let coat_pdf = {
            let half = wo.add(wi).normalize_or_zero();
            if half.length_squared() <= EPS_LEN_SQ {
                0.0
            } else {
                self.coat.reflection_pdf(cos_o, normal.dot(half))
            }
        };
        let base_pdf = self.base.pdf(wo, wi, normal);
        p_coat * coat_pdf + (1.0 - p_coat) * base_pdf
    }

    /// Importance-samples the clear-coat lobe mixture.
    ///
    /// Draws the coat lobe with probability [`Self::coat_selection_prob`] and the
    /// base lobe otherwise, then reports the combined value and the combined
    /// density so the integrator's `value * cos_i / pdf` weight is correct for a
    /// one-sample multiple-importance mixture. Returns `None` for a degenerate
    /// sample so the caller terminates the path.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<ClearcoatSample> {
        // The integrator passes the raw geometric normal; orient it into the
        // view hemisphere so a back-facing hit still scatters.
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        let p_coat = self.coat_selection_prob(cos_o);
        let wi = if rng.next_f32() < p_coat {
            let half = self.coat.sample_half_vector(wo, normal, rng)?;
            if wo.dot(half) <= 0.0 {
                return None;
            }
            let wi = wo.negate().reflect(half).normalize_or_zero();
            if normal.dot(wi) <= 0.0 {
                return None;
            }
            wi
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
        let (eta, k) = Metal::Gold.complex_ior();
        ClearcoatConductor::new(eta, k, 0.3, coat_roughness, 1.5, coat_weight, Vec3::ONE)
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
