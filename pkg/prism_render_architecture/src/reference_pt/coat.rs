//! A reusable dielectric clear-coat layer shared by every coated material.
//!
//! A clear coat is the thin, near-colourless lacquer sprayed over car paint,
//! lacquered wood, or carbon fibre: a second specular interface sitting above a
//! pigmented or metallic base. Physically it is a dielectric slab, so it
//! reflects a Fresnel fraction of the incident light with its own (usually very
//! smooth) microfacet lobe and transmits the rest down to the base, which then
//! reflects back up through the slab a second time.
//!
//! Rather than tracing light through the slab with a position-free random walk,
//! this layer uses the closed-form two-lobe model shipped by real-time engines:
//! the coat contributes an additive `GGX` dielectric highlight, and the base
//! lobe is attenuated by the Fresnel transmission entering and leaving the
//! coat, `(1 - F_coat(cos_o)) * (1 - F_coat(cos_i))`. The coat is treated as
//! optically thin, so it attenuates and tints but does not bend the base
//! direction — the standard engine approximation that keeps a coated material a
//! sum of two microfacet lobes with no extra sampling dimensions.
//!
//! Both the coat highlight and the base reflect into the upper hemisphere, so a
//! coated material importance-samples one lobe by a Fresnel-weighted
//! probability and reports the combined value and the combined one-sample
//! density, exactly like the multi-lobe `FresnelBlend` plastic. Only
//! `sqrt`-based arithmetic is used (through the Fresnel and `GGX` primitives),
//! so the layer stays within the transcendental budget of the reference path
//! tracer.

use super::dielectric::fresnel_dielectric;
use super::microfacet::GgxIsotropic;
use super::sampler::SampleSource;
use super::{Vec3, EPS_LEN_SQ};

/// The smallest relative coat index the layer accepts, keeping the coat a
/// genuine dielectric interface (an index of exactly one would be invisible).
const MIN_COAT_IOR: f32 = 1.0 + 1.0e-3;

/// The clamp window for the coat-versus-base lobe-selection probability.
///
/// Selecting a lobe strictly in proportion to the coat Fresnel term would stop
/// sampling the coat near normal incidence (where its reflectance is a few
/// percent) and starve the base near grazing. Clamping the probability into a
/// central window keeps both lobes alive, which lowers variance without biasing
/// the estimator (the matching density is used in [`CoatLayer::pdf`]).
const MIN_COAT_PROB: f32 = 0.1;
/// The upper bound of the lobe-selection window; see [`MIN_COAT_PROB`].
const MAX_COAT_PROB: f32 = 0.9;

/// A thin dielectric clear coat layered over an opaque base.
///
/// The coat is an isotropic `GGX` dielectric interface of relative index
/// [`CoatLayer::coat_ior`] whose presence is scaled by
/// [`CoatLayer::coat_weight`] in `[0, 1]`. A coated material owns one of these
/// alongside its base lobe and routes the base lobe's Fresnel attenuation,
/// additive highlight, lobe-selection probability, and sampling through it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CoatLayer {
    /// The isotropic `GGX` microfacet distribution of the coat highlight.
    coat: GgxIsotropic,
    /// Relative index of refraction `eta_t / eta_i` of the coat over the
    /// exterior medium, clamped to at least [`MIN_COAT_IOR`].
    coat_ior: f32,
    /// Coat presence in `[0, 1]`: `0` is a bare base, `1` a full coat.
    coat_weight: f32,
    /// Per-channel tint of the coat highlight (usually white for clear lacquer).
    coat_color: Vec3,
}

impl CoatLayer {
    /// Builds a coat from its perceptual `coat_roughness`, relative `coat_ior`,
    /// `coat_weight` in `[0, 1]`, and `coat_color` tint.
    #[must_use]
    pub(super) fn new(
        coat_roughness: f32,
        coat_ior: f32,
        coat_weight: f32,
        coat_color: Vec3,
    ) -> Self {
        Self {
            coat: GgxIsotropic::from_roughness(coat_roughness),
            coat_ior: coat_ior.max(MIN_COAT_IOR),
            coat_weight: coat_weight.clamp(0.0, 1.0),
            coat_color,
        }
    }

    /// The effective macroscopic coat reflectance `weight * Fresnel(cos)` for a
    /// direction whose cosine with the normal is `cos`.
    #[must_use]
    pub(super) fn reflectance(&self, cos: f32) -> f32 {
        self.coat_weight * fresnel_dielectric(cos, 1.0, self.coat_ior)
    }

    /// The fraction of light transmitted through the coat at cosine `cos`,
    /// i.e. `1 - reflectance`, used to attenuate the base lobe.
    #[must_use]
    pub(super) fn transmission(&self, cos: f32) -> f32 {
        1.0 - self.reflectance(cos)
    }

    /// The probability of sampling the coat lobe for a view cosine `cos_o`,
    /// clamped into the central window `[MIN_COAT_PROB, MAX_COAT_PROB]`.
    ///
    /// Returns zero for a weightless coat so a bare base always samples its own
    /// lobe.
    #[must_use]
    pub(super) fn selection_prob(&self, cos_o: f32) -> f32 {
        if self.coat_weight <= 0.0 {
            return 0.0;
        }
        self.reflectance(cos_o).clamp(MIN_COAT_PROB, MAX_COAT_PROB)
    }

    /// The coat's additive `GGX` dielectric highlight
    /// `weight * F * D * G2 / (4 cos_o cos_i)`.
    ///
    /// Returns [`Vec3::ZERO`] for a weightless coat or a degenerate half vector.
    /// `cos_o` and `cos_i` are the view and light cosines with `normal`, assumed
    /// positive by the caller.
    #[must_use]
    pub(super) fn lobe(&self, wo: Vec3, wi: Vec3, normal: Vec3, cos_o: f32, cos_i: f32) -> Vec3 {
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

    /// The coat lobe's solid-angle density for the pair `(wo, wi)` with view
    /// cosine `cos_o`.
    ///
    /// Returns zero for a degenerate half vector so the mixed density stays
    /// finite.
    #[must_use]
    pub(super) fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3, cos_o: f32) -> f32 {
        let half = wo.add(wi).normalize_or_zero();
        if half.length_squared() <= EPS_LEN_SQ {
            return 0.0;
        }
        self.coat.reflection_pdf(cos_o, normal.dot(half))
    }

    /// Importance-samples a reflection direction off the coat's microfacet
    /// distribution.
    ///
    /// Returns `None` for a degenerate half vector or a direction that leaves
    /// the upper hemisphere so the caller terminates the path.
    #[must_use]
    pub(super) fn sample_direction(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<Vec3> {
        let half = self.coat.sample_half_vector(wo, normal, rng)?;
        if wo.dot(half) <= 0.0 {
            return None;
        }
        let wi = wo.negate().reflect(half).normalize_or_zero();
        if normal.dot(wi) <= 0.0 {
            return None;
        }
        Some(wi)
    }
}
