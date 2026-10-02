//! Energy-conserving anisotropic rough conductor (brushed / satin metal).
//!
//! [`crate::reference_pt::conductor_aniso::AnisoConductor`] stretches the
//! `GGX` highlight into a grain-aligned streak but, like every single-scatter
//! microfacet lobe, it still discards the energy that bounces several times
//! between facets before escaping. Brushed aluminium and satin steel therefore
//! darken and desaturate as the grain roughens, exactly the Kulla-Conty deficit
//! the isotropic [`crate::reference_pt::conductor_ms`] already fixes. This
//! module adds the same compensation on top of the anisotropic lobe so the
//! brushed-metal oracle also conserves energy under a white furnace.
//!
//! The Kulla-Conty tables in [`crate::reference_pt::ggx_energy`] are indexed by
//! a single scalar `alpha`. The single-scatter directional albedo depends only
//! very weakly on the *direction* of anisotropy, so the standard and accurate
//! choice is to drive the compensation lobe with the isotropic-equivalent width
//! `alpha = sqrt(alpha_x * alpha_y)` — the geometric mean that preserves the
//! micro-facet slope area. The compensation lobe is itself azimuthally smooth
//! (near-diffuse), so collapsing the two widths to their mean introduces no
//! visible directional error while keeping the furnace exactly balanced.
//!
//! The combined `BRDF` is `f_ss + F_ms * f_ms`, matching
//! [`crate::reference_pt::conductor_ms`]: `f_ss` is the exact anisotropic
//! complex-index lobe and `f_ms` the near-diffuse recovery lobe tinted by the
//! metal's average `Fresnel` reflectance. Sampling uses the same one-sample
//! multiple-importance strategy between the anisotropic `VNDF` lobe and a
//! cosine-weighted direction, always reporting the combined density so the
//! estimator stays unbiased.

use super::conductor::average_fresnel_conductor;
use super::conductor_aniso::AnisoConductor;
use super::ggx_energy::{average_albedo, multiscatter_fresnel, multiscatter_lobe};
use super::microfacet_aniso::GgxAnisotropic;
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, Rng};
use super::Vec3;

/// The outcome of importance-sampling a [`MultiscatterAnisoConductor`] lobe.
#[derive(Clone, Copy, Debug)]
pub struct MultiscatterAnisoSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The combined `BRDF` value `f_r(wo, wi)` for the sampled pair.
    pub value: Vec3,
    /// The combined solid-angle probability density of `direction`.
    pub pdf: f32,
}

/// An anisotropic rough conductor that adds Kulla-Conty multiple-scattering
/// compensation on top of the exact single-scatter anisotropic lobe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiscatterAnisoConductor {
    /// The exact single-scatter anisotropic conductor providing `f_ss`.
    base: AnisoConductor,
    /// The isotropic-equivalent `GGX` width `sqrt(alpha_x * alpha_y)` used to
    /// index the scalar energy-compensation tables.
    alpha: f32,
    /// The hemispherical-average `Fresnel` reflectance driving the colour of the
    /// recovered energy.
    f_avg: Vec3,
}

/// The geometric-mean isotropic-equivalent width of an anisotropic lobe.
fn equivalent_alpha(dist: GgxAnisotropic) -> f32 {
    (dist.alpha_x * dist.alpha_y).sqrt()
}

impl MultiscatterAnisoConductor {
    /// Builds an energy-conserving anisotropic conductor from its complex index
    /// `eta + i*k` and explicit `GGX` widths `alpha_x`/`alpha_y`.
    #[must_use]
    pub fn new(eta: Vec3, k: Vec3, alpha_x: f32, alpha_y: f32) -> Self {
        let dist = GgxAnisotropic::new(alpha_x, alpha_y);
        Self {
            base: AnisoConductor::new(eta, k, alpha_x, alpha_y),
            alpha: equivalent_alpha(dist),
            f_avg: average_fresnel_conductor(eta, k),
        }
    }

    /// Builds an energy-conserving anisotropic conductor from its complex index
    /// and a perceptual `roughness`/`anisotropy` pair, using the same Disney /
    /// `UE` (Burley) remap as
    /// [`AnisoConductor::from_roughness_anisotropy`].
    #[must_use]
    pub fn from_roughness_anisotropy(eta: Vec3, k: Vec3, roughness: f32, anisotropy: f32) -> Self {
        let dist = GgxAnisotropic::from_roughness_anisotropy(roughness, anisotropy);
        Self {
            base: AnisoConductor::from_roughness_anisotropy(eta, k, roughness, anisotropy),
            alpha: equivalent_alpha(dist),
            f_avg: average_fresnel_conductor(eta, k),
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

    /// The probability of choosing the single-scatter anisotropic strategy,
    /// clamped so both strategies retain enough density for a stable one-sample
    /// estimator.
    fn single_scatter_probability(&self) -> f32 {
        average_albedo(self.alpha).clamp(0.1, 0.9)
    }

    /// The combined solid-angle density [`Self::sample`] assigns to `(wo, wi)`,
    /// mixing the anisotropic and cosine strategies by their selection
    /// probabilities.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return 0.0;
        }
        let p_ss = self.single_scatter_probability();
        let aniso = self.base.pdf(wo, wi, normal);
        let diffuse = cosine_hemisphere_pdf(normal, wi);
        p_ss * aniso + (1.0 - p_ss) * diffuse
    }

    /// Importance-samples the combined lobe with one-sample multiple importance
    /// sampling between the anisotropic `VNDF` and cosine strategies.
    ///
    /// Returns `None` for a degenerate sample so the caller terminates the path
    /// rather than dividing by zero.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<MultiscatterAnisoSample> {
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
        Some(MultiscatterAnisoSample {
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

    /// The compensated anisotropic conductor returns more energy than the bare
    /// single-scatter anisotropic lobe for a rough, strongly anisotropic metal,
    /// and never exceeds the white-furnace ceiling of one.
    #[test]
    fn multiple_scattering_recovers_lost_energy() {
        // A near-perfect reflector so the furnace target sits close to one.
        let eta = Vec3::new(0.05, 0.05, 0.05);
        let k = Vec3::new(5.0, 5.0, 5.0);
        let roughness = 0.8;
        let anisotropy = 0.8;
        let ms =
            MultiscatterAnisoConductor::from_roughness_anisotropy(eta, k, roughness, anisotropy);
        let base = AnisoConductor::from_roughness_anisotropy(eta, k, roughness, anisotropy);
        let wo = unit(0.4, 0.9, 0.0);
        let mut rng = Rng::seed(17);
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
    /// value and density [`MultiscatterAnisoConductor::evaluate`] and
    /// [`MultiscatterAnisoConductor::pdf`] recompute for the same pair.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let ms = MultiscatterAnisoConductor::from_roughness_anisotropy(GOLD_ETA, GOLD_K, 0.6, 0.6);
        let wo = unit(0.3, 0.9, 0.1);
        let mut rng = Rng::seed(43);
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

    /// With `anisotropy = 0` the compensated anisotropic conductor collapses to
    /// the isotropic multiple-scattering conductor it generalises.
    #[test]
    fn isotropic_limit_matches_scalar_multiscatter() {
        use crate::reference_pt::conductor_ms::MultiscatterConductor;
        let roughness = 0.5;
        let aniso =
            MultiscatterAnisoConductor::from_roughness_anisotropy(GOLD_ETA, GOLD_K, roughness, 0.0);
        let iso = MultiscatterConductor::new(GOLD_ETA, GOLD_K, roughness);
        let wo = unit(0.25, 0.92, 0.08);
        let wi = unit(-0.2, 0.95, 0.05);
        let a = aniso.evaluate(wo, wi, N);
        let b = iso.evaluate(wo, wi, N);
        assert!(
            a.sub(b).length() <= 1e-3 * b.length().max(1.0),
            "aniso {a:?} should match isotropic {b:?}"
        );
    }

    /// A smooth anisotropic conductor loses almost no energy, so the combined
    /// `BRDF` matches the single-scatter anisotropic lobe.
    #[test]
    fn smooth_conductor_matches_single_scatter() {
        let ms = MultiscatterAnisoConductor::from_roughness_anisotropy(GOLD_ETA, GOLD_K, 0.02, 0.5);
        let base = AnisoConductor::from_roughness_anisotropy(GOLD_ETA, GOLD_K, 0.02, 0.5);
        let wo = unit(0.2, 0.95, 0.1);
        let wi = unit(-0.15, 0.96, 0.1);
        let a = ms.evaluate(wo, wi, N);
        let b = base.evaluate(wo, wi, N);
        assert!(a.sub(b).length() <= 1e-2 * b.length().max(1.0));
    }
}
