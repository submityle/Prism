//! Coupled diffuse-specular reflection (Ashikhmin-Shirley `Fresnel` blend).
//!
//! A clear-coated or painted "plastic" surface is a dielectric specular layer
//! sitting on top of a diffuse substrate. The two layers are not independent:
//! energy the specular layer reflects (a view-angle-dependent `Fresnel`
//! fraction) can never reach the diffuse base, and energy that does reach the
//! base must cross the interface twice. Treating the layers as two separate,
//! additive lobes therefore over-counts energy at grazing angles.
//!
//! Ashikhmin and Shirley ("An Anisotropic Phong `BRDF` Model", `JGT` 2000)
//! derived a single reciprocal, energy-conserving `BRDF` that couples the two:
//! the diffuse term is attenuated by `(1 - R_s)` and by a factor that vanishes
//! as either direction approaches grazing, exactly where the specular `Fresnel`
//! term saturates to one. This module implements that model with the shared
//! GGX microfacet core (see [`crate::reference_pt::microfacet`]) as the specular
//! distribution, so it validates the real-time clear-coat / plastic shading
//! paths against the same statistics.
//!
//! Conventions match the rest of the tracer: `wo`, `wi`, and `normal` are unit
//! vectors in the viewer hemisphere, and the model is sampled by a two-strategy
//! mixture (cosine diffuse plus visible-normal specular) whose combined density
//! is reported by [`FresnelBlend::pdf`]. Only `sqrt` is used; the Schlick
//! quintics are spelled out as repeated multiplications.

use super::microfacet::{fresnel_schlick, GgxIsotropic};
#[cfg(test)]
use super::sampler::Rng;
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, SampleSource};
use super::{Vec3, EPS_LEN_SQ};

/// The normalization constant `28 / (23 * pi)` of the Ashikhmin-Shirley diffuse
/// term, chosen so the coupled `BRDF` integrates to at most one (it conserves
/// energy and never creates it).
const DIFFUSE_NORM: f32 = 28.0 / (23.0 * core::f32::consts::PI);

/// The probability of choosing the diffuse sampling strategy in the two-lobe
/// mixture; the specular strategy is chosen with the complementary probability.
const DIFFUSE_SAMPLE_PROBABILITY: f32 = 0.5;

/// Raises `x` to the fifth power without a transcendental call.
fn quintic(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

/// A coupled diffuse-specular reflector parameterized by its diffuse and
/// specular reflectances and a GGX specular width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FresnelBlend {
    /// Diffuse (substrate) reflectance `R_d` per channel, in `[0, 1]`.
    diffuse: Vec3,
    /// Specular normal-incidence reflectance `R_s` per channel (the dielectric
    /// `F0`, e.g. `0.04` for a common clear coat).
    specular: Vec3,
    /// The GGX distribution of the specular layer.
    ggx: GgxIsotropic,
}

impl FresnelBlend {
    /// Builds a blend from diffuse reflectance `R_d`, specular reflectance
    /// `R_s`, and a perceptual `roughness` in `[0, 1]` (remapped to the GGX
    /// width `alpha = roughness^2`).
    #[must_use]
    pub fn new(diffuse: Vec3, specular: Vec3, roughness: f32) -> Self {
        Self {
            diffuse,
            specular,
            ggx: GgxIsotropic::from_roughness(roughness),
        }
    }

    /// Evaluates the coupled `BRDF` value `f_r(wo, wi)`.
    ///
    /// Returns [`Vec3::ZERO`] when either direction is below the surface. The
    /// result is the sum of the Ashikhmin-Shirley diffuse term and the GGX
    /// microfacet specular term sharing the same `Fresnel` factor.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }

        // Ashikhmin-Shirley diffuse term: Rd (1 - Rs) coupled by a factor that
        // falls to zero as either direction grazes (where specular -> 1).
        let fi = 1.0 - quintic(1.0 - 0.5 * cos_i);
        let fo = 1.0 - quintic(1.0 - 0.5 * cos_o);
        let diffuse = self
            .diffuse
            .mul(Vec3::ONE.sub(self.specular))
            .scale(DIFFUSE_NORM * fi * fo);

        // GGX microfacet specular term with a Schlick `Fresnel` tint.
        let specular = self.specular_term(wo, wi, normal, cos_o, cos_i);
        diffuse.add(specular)
    }

    /// The specular (microfacet) contribution of the coupled `BRDF`.
    ///
    /// Separated out so [`Self::evaluate`] reads as "diffuse plus specular". The
    /// cosines are passed in to avoid recomputing the dot products.
    fn specular_term(&self, wo: Vec3, wi: Vec3, normal: Vec3, cos_o: f32, cos_i: f32) -> Vec3 {
        let half = wo.add(wi).normalize_or_zero();
        if half.length_squared() <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        let cos_h = normal.dot(half);
        if cos_h <= 0.0 {
            return Vec3::ZERO;
        }
        let woh = wo.dot(half);
        let denom = 4.0 * woh.abs() * cos_o.max(cos_i);
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        let d = self.ggx.distribution(cos_h);
        let fresnel = fresnel_schlick(self.specular, woh.max(0.0));
        fresnel.scale(d / denom)
    }

    /// The solid-angle density that [`Self::sample`] assigns to `(wo, wi)`.
    ///
    /// This is the balanced mixture of the two sampling strategies: a cosine
    /// hemisphere density for the diffuse lobe and the GGX visible-normal
    /// reflection density for the specular lobe. Zero when `wi` is below the
    /// surface.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let diffuse_pdf = cosine_hemisphere_pdf(normal, wi);
        let half = wo.add(wi).normalize_or_zero();
        let specular_pdf = if half.length_squared() <= EPS_LEN_SQ {
            0.0
        } else {
            self.ggx.reflection_pdf(cos_o, normal.dot(half))
        };
        DIFFUSE_SAMPLE_PROBABILITY * diffuse_pdf + (1.0 - DIFFUSE_SAMPLE_PROBABILITY) * specular_pdf
    }

    /// Importance-samples an outgoing direction, returning it together with the
    /// `BRDF` value and the mixture density, or `None` for a degenerate sample.
    ///
    /// With probability [`DIFFUSE_SAMPLE_PROBABILITY`] a cosine-weighted diffuse
    /// direction is drawn; otherwise a visible-normal microfacet half vector is
    /// drawn and the view direction is mirrored about it. Either way the full
    /// coupled `BRDF` and the combined density are returned, so the integrator's
    /// generic `value * cos / pdf` weight stays unbiased.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<FresnelBlendSample> {
        // Orient the raw geometric normal into the view hemisphere.
        let normal = normal.faced_toward(wo);
        if normal.dot(wo) <= 0.0 {
            return None;
        }
        let wi = if rng.next_f32() < DIFFUSE_SAMPLE_PROBABILITY {
            cosine_sample_hemisphere(normal, rng).direction
        } else {
            let half = self.ggx.sample_half_vector(wo, normal, rng)?;
            wo.negate().reflect(half).normalize_or_zero()
        };
        if normal.dot(wi) <= 0.0 {
            return None;
        }
        let pdf = self.pdf(wo, wi, normal);
        if pdf <= 0.0 {
            return None;
        }
        Some(FresnelBlendSample {
            direction: wi,
            value: self.evaluate(wo, wi, normal),
            pdf,
        })
    }
}

/// The outcome of sampling a [`FresnelBlend`]: a direction, the coupled `BRDF`
/// value there, and the mixture solid-angle density.
#[derive(Clone, Copy, Debug)]
pub struct FresnelBlendSample {
    /// The sampled unit outgoing direction (away from the surface).
    pub direction: Vec3,
    /// The coupled `BRDF` value `f_r(wo, wi)` at the sampled pair.
    pub value: Vec3,
    /// The combined (two-strategy) solid-angle probability density.
    pub pdf: f32,
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
    fn evaluate_zero_below_surface() {
        let blend = FresnelBlend::new(Vec3::splat(0.5), Vec3::splat(0.04), 0.3);
        let below = Vec3::new(0.0, -1.0, 0.0);
        assert_eq!(blend.evaluate(N, below, N), Vec3::ZERO);
        assert_eq!(blend.evaluate(below, N, N), Vec3::ZERO);
    }

    #[test]
    fn brdf_is_reciprocal() {
        // A physically based BRDF must satisfy f(wo, wi) == f(wi, wo).
        let blend = FresnelBlend::new(Vec3::new(0.6, 0.4, 0.2), Vec3::splat(0.05), 0.35);
        let wo = Vec3::new(0.3, 0.9, 0.1).normalize_or_zero();
        let wi = Vec3::new(-0.4, 0.8, 0.2).normalize_or_zero();
        let a = blend.evaluate(wo, wi, N);
        let b = blend.evaluate(wi, wo, N);
        assert!(
            (a.x - b.x).abs() < 1e-6,
            "reciprocity x: {} vs {}",
            a.x,
            b.x
        );
        assert!((a.y - b.y).abs() < 1e-6);
        assert!((a.z - b.z).abs() < 1e-6);
    }

    #[test]
    fn sample_pdf_matches_reported_pdf() {
        let blend = FresnelBlend::new(Vec3::splat(0.5), Vec3::splat(0.04), 0.4);
        let mut rng = Rng::seed(9182);
        let wo = Vec3::new(0.2, 1.0, 0.1).normalize_or_zero();
        let mut checked = 0u32;
        for _ in 0..40_000 {
            let Some(s) = blend.sample(wo, N, &mut rng) else {
                continue;
            };
            checked += 1;
            let pdf = blend.pdf(wo, s.direction, N);
            assert!(
                (pdf - s.pdf).abs() <= 1e-4 * s.pdf.max(1.0),
                "pdf mismatch {pdf} vs {}",
                s.pdf
            );
            assert!(s.direction.is_finite());
            assert!(s.value.is_finite());
        }
        assert!(checked > 10_000, "too few valid samples ({checked})");
    }

    #[test]
    fn white_furnace_conserves_energy() {
        // With Rd = 1 and Rs = 0 the coupled BRDF reduces to the pure
        // Ashikhmin-Shirley diffuse term, whose directional-hemispherical
        // reflectance is below one (energy conserving, never amplifying).
        let blend = FresnelBlend::new(Vec3::ONE, Vec3::ZERO, 0.5);
        let mut rng = Rng::seed(2718);
        let wo = Vec3::new(0.3, 1.0, 0.0).normalize_or_zero();
        let count = 300_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            let Some(s) = blend.sample(wo, N, &mut rng) else {
                continue;
            };
            let cos_i = N.dot(s.direction);
            sum += f64::from(s.value.x * cos_i / s.pdf);
        }
        let reflectance = sum / f64::from(count);
        assert!(
            reflectance > 0.0 && reflectance < 1.0,
            "coupled-diffuse reflectance {reflectance} must lie in (0, 1)"
        );
    }

    #[test]
    fn full_blend_reflectance_stays_below_one() {
        // A realistic plastic (diffuse base plus a dielectric coat) must not
        // reflect more energy than it receives at any view angle.
        let blend = FresnelBlend::new(Vec3::splat(0.9), Vec3::splat(0.04), 0.25);
        let mut rng = Rng::seed(1414);
        for &cos in &[0.1f32, 0.4, 0.7, 0.95] {
            let sin = (1.0 - cos * cos).max(0.0).sqrt();
            let wo = Vec3::new(sin, cos, 0.0).normalize_or_zero();
            let count = 200_000u32;
            let mut sum = 0.0f64;
            for _ in 0..count {
                let Some(s) = blend.sample(wo, N, &mut rng) else {
                    continue;
                };
                let cos_i = N.dot(s.direction);
                sum += f64::from(s.value.x * cos_i / s.pdf);
            }
            let reflectance = sum / f64::from(count);
            assert!(
                reflectance < 1.0 + 1e-2,
                "reflectance {reflectance} at cos {cos} must not exceed one"
            );
            assert!(reflectance > 0.0);
        }
    }

    #[test]
    fn specular_sharpens_as_roughness_drops() {
        // Near the mirror direction, a smoother coat concentrates more specular
        // energy, so the BRDF value there grows as roughness falls.
        fn peak_value(roughness: f32) -> f32 {
            let blend = FresnelBlend::new(Vec3::splat(0.3), Vec3::splat(0.5), roughness);
            // Mirror configuration about the normal: wo and wi symmetric.
            let wo = Vec3::new(0.3, 0.9539392, 0.0).normalize_or_zero();
            let wi = Vec3::new(-0.3, 0.9539392, 0.0).normalize_or_zero();
            blend.evaluate(wo, wi, N).x
        }
        let rough = peak_value(0.6);
        let smooth = peak_value(0.1);
        assert!(
            smooth > rough,
            "smoother specular peak {smooth} should exceed rougher {rough}"
        );
    }
}
