//! Energy-aware clear-coat over a diffuse base, following the production `UE`
//! clear-coat shading model.
//!
//! This is the paint-and-lacquer material: a pigmented matte base (modelled by
//! an [`OrenNayar`] diffuse lobe) sealed under the thin dielectric clear coat
//! from [`super::coat`]. It is the diffuse counterpart of
//! [`super::clearcoat::ClearcoatConductor`] and reuses the exact same
//! [`CoatLayer`], so the coat contributes an additive `GGX` dielectric highlight
//! and attenuates the diffuse lobe by the Fresnel transmission entering and
//! leaving the coat, `(1 - F_coat(cos_o)) * (1 - F_coat(cos_i))`.
//!
//! Both lobes reflect into the upper hemisphere, so importance sampling draws
//! one lobe by a Fresnel-weighted probability and reports the combined value and
//! the combined one-sample density, keeping the Monte Carlo estimator unbiased.
//! Only `sqrt`-based arithmetic is used, so the model stays within the
//! transcendental budget of the reference path tracer.

use super::coat::CoatLayer;
use super::oren_nayar::OrenNayar;
use super::sampler::Rng;
use super::Vec3;

/// The outcome of importance-sampling a [`ClearcoatDiffuse`].
#[derive(Clone, Copy, Debug)]
pub struct ClearcoatDiffuseSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The combined `BSDF` value `f_r(wo, wi)` of both lobes (no cosine applied).
    pub value: Vec3,
    /// The combined one-sample solid-angle density of `direction`.
    pub pdf: f32,
}

/// A diffuse base viewed through a dielectric clear coat.
///
/// The base is an [`OrenNayar`] rough-diffuse lobe; the coat is a reusable
/// [`CoatLayer`] dielectric interface scaled by its own weight in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearcoatDiffuse {
    /// The rough-diffuse base layer seen through the coat.
    base: OrenNayar,
    /// The thin dielectric clear coat layered over the base.
    coat: CoatLayer,
}

impl ClearcoatDiffuse {
    /// Builds a clear-coated diffuse from the base `albedo` and Oren-Nayar
    /// roughness `sigma` (radians; `0` is Lambertian), plus the coat's
    /// perceptual `coat_roughness`, relative `coat_ior`, `coat_weight` in
    /// `[0, 1]`, and `coat_color` tint.
    #[must_use]
    pub fn new(
        albedo: Vec3,
        sigma: f32,
        coat_roughness: f32,
        coat_ior: f32,
        coat_weight: f32,
        coat_color: Vec3,
    ) -> Self {
        Self {
            base: OrenNayar::new(albedo, sigma),
            coat: CoatLayer::new(coat_roughness, coat_ior, coat_weight, coat_color),
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
        let base = self.base.evaluate(wo, wi, normal).scale(t_o * t_i);
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
    /// diffuse base otherwise, then reports the combined value and the combined
    /// density so the integrator's `value * cos_i / pdf` weight is correct for a
    /// one-sample multiple-importance mixture. Returns `None` for a degenerate
    /// sample so the caller terminates the path.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<ClearcoatDiffuseSample> {
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
        Some(ClearcoatDiffuseSample {
            direction: wi,
            value: self.evaluate(wo, wi, normal),
            pdf,
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

    /// A mid-grey base albedo shared by the tests.
    const ALBEDO: Vec3 = Vec3 {
        x: 0.6,
        y: 0.5,
        z: 0.4,
    };

    /// Builds a clear-coated matte paint with the given coat weight and
    /// roughness over a Lambertian base.
    fn coated_paint(coat_weight: f32, coat_roughness: f32) -> ClearcoatDiffuse {
        ClearcoatDiffuse::new(ALBEDO, 0.0, coat_roughness, 1.5, coat_weight, Vec3::ONE)
    }

    /// A view direction at the given cosine to the normal, tilted in the `x-y`
    /// plane.
    fn view_at(mu: f32) -> Vec3 {
        let sin_o = (1.0 - mu * mu).max(0.0).sqrt();
        Vec3::new(sin_o, mu, 0.0).normalize_or_zero()
    }

    /// Monte Carlo directional albedo (reflected hemisphere) of a coated paint.
    fn directional_albedo(coat: &ClearcoatDiffuse, wo: Vec3, rng: &mut Rng, samples: u32) -> f32 {
        let mut sum = 0.0f64;
        for _ in 0..samples {
            if let Some(s) = coat.sample(wo, N, rng) {
                let cos_i = N.dot(s.direction).max(0.0);
                sum += f64::from(s.value.max_component() * cos_i / s.pdf);
            }
        }
        (sum / f64::from(samples)) as f32
    }

    /// A weightless coat must reproduce the bare Oren-Nayar value and density.
    #[test]
    fn zero_weight_coat_matches_bare_diffuse() {
        let coat = coated_paint(0.0, 0.05);
        let bare = OrenNayar::new(ALBEDO, 0.0);
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

    /// Adding a coat injects a specular highlight the matte base cannot produce:
    /// near the mirror direction the coated surface reflects strictly more.
    #[test]
    fn coat_adds_specular_highlight() {
        let coat = coated_paint(1.0, 0.02);
        let bare = OrenNayar::new(ALBEDO, 0.0);
        let wo = view_at(0.7);
        // The mirror direction of `wo` about the `+y` normal.
        let wi = Vec3::new(-wo.x, wo.y, -wo.z).normalize_or_zero();
        let c = coat.evaluate(wo, wi, N).max_component();
        let b = bare.evaluate(wo, wi, N).max_component();
        assert!(c > b, "coated {c} should exceed bare {b}");
    }

    /// The coat attenuates the diffuse base away from the highlight: at an
    /// off-specular direction the coated base reflects no more than the bare
    /// base (the `(1 - F)` transmission factors are at most one).
    #[test]
    fn coat_attenuates_diffuse_off_specular() {
        let coat = coated_paint(1.0, 0.2);
        let bare = OrenNayar::new(ALBEDO, 0.0);
        let wo = view_at(0.6);
        // A direction well away from the mirror lobe, still above the surface.
        let wi = Vec3::new(0.5, 0.8, 0.33).normalize_or_zero();
        let c = coat.evaluate(wo, wi, N).max_component();
        let b = bare.evaluate(wo, wi, N).max_component();
        assert!(c <= b + 1e-6, "coated {c} should not exceed bare {b}");
    }

    /// The material conserves energy: a white coat over a grey base never
    /// reflects more light than arrives, so its directional albedo stays at or
    /// below one.
    #[test]
    fn material_is_energy_conserving() {
        let coat = coated_paint(1.0, 0.2);
        let mut rng = Rng::seed(0x0C0A_D1FF);
        for mu in [0.2f32, 0.5, 0.9] {
            let a = directional_albedo(&coat, view_at(mu), &mut rng, 1u32 << 16);
            assert!(a <= 1.0 + 2e-2, "albedo {a} at mu {mu} exceeds unity");
        }
    }

    /// Sampling is consistent with independent [`ClearcoatDiffuse::evaluate`]
    /// and [`ClearcoatDiffuse::pdf`] for both lobes.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let coat = coated_paint(0.7, 0.15);
        let wo = view_at(0.75);
        let mut rng = Rng::seed(0xD1FF_5EED);
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
        let coat = coated_paint(0.9, 0.1);
        let wo = view_at(0.85);
        let mut rng = Rng::seed(0x5EED_D1FF);
        for _ in 0..5_000 {
            if let Some(s) = coat.sample(wo, N, &mut rng) {
                assert!(N.dot(s.direction) > 0.0, "below surface");
                assert!(s.pdf > 0.0 && s.pdf.is_finite(), "bad pdf {}", s.pdf);
                assert!(s.value.is_finite(), "non-finite value");
            }
        }
    }
}
