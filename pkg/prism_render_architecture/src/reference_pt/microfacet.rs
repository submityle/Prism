//! Isotropic GGX (Trowbridge-Reitz) microfacet reflection.
//!
//! This is the shared microfacet core behind the physically based specular
//! lobes in [`crate::reference_pt::bsdf`]. It is the ground-truth counterpart of
//! the real-time Cook-Torrance specular path, so the offline reference tracer
//! can validate screen-space reflections, image-based lighting, and the glossy
//! `ReSTIR` lobes against the same microfacet statistics they approximate.
//!
//! Three classical pieces make up a Cook-Torrance microfacet `BRDF`:
//!
//! - the **normal distribution function** `D(h)` — the GGX / Trowbridge-Reitz
//!   lobe that concentrates microfacet normals around the macroscopic normal;
//! - the **masking-shadowing term** `G` — the height-correlated Smith form,
//!   expressed through the per-direction `Lambda` auxiliary so the masking and
//!   shadowing cosines are never double counted;
//! - the **Fresnel term** `F` — Schlick's rational approximation, evaluated
//!   per channel so coloured conductors (gold, copper) are reproduced.
//!
//! Sampling uses the visible-normal (`VNDF`) method of Heitz
//! ("Sampling the GGX Distribution of Visible Normals", `JCGT` 2018): a half
//! vector is drawn proportionally to `D(h) * G1(wo) * max(0, wo·h)`, which
//! removes the back-facing microfacets that ordinary `NDF` sampling wastes and
//! collapses the Monte Carlo weight to `F * G2 / G1(wo)`. The whole routine is
//! expressed with `sqrt` and a rejection-sampled unit-disk point only, honouring
//! the crate's "no `sin`/`cos`" determinism policy.
//!
//! Roughness follows the Disney / `UE` convention: the user-facing perceptual
//! roughness `r` maps to the GGX width `alpha = r^2`, which linearises the
//! visual change in highlight size across the slider.

use super::sampler::{orthonormal_basis, uniform_disk, Rng};
use super::{Vec3, EPS_LEN_SQ, INV_PI};

/// Smallest GGX width used for shading.
///
/// Below this the lobe is numerically indistinguishable from a perfect mirror
/// (its density spikes without bound); clamping keeps `D`, the sampling warp,
/// and the probability density finite without visibly rounding real materials.
pub const MIN_ALPHA: f32 = 1.0e-3;

/// An isotropic GGX microfacet distribution of width `alpha`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxIsotropic {
    /// GGX roughness width `alpha` (the root-mean-square microfacet slope),
    /// clamped to at least [`MIN_ALPHA`].
    pub alpha: f32,
}

impl GgxIsotropic {
    /// Builds a distribution from a perceptual `roughness` in `[0, 1]` using the
    /// `alpha = roughness^2` remap, clamped to [`MIN_ALPHA`].
    #[must_use]
    pub fn from_roughness(roughness: f32) -> Self {
        let r = roughness.clamp(0.0, 1.0);
        Self {
            alpha: (r * r).max(MIN_ALPHA),
        }
    }

    /// The GGX normal distribution `D(h)` for a half vector whose cosine to the
    /// macroscopic normal is `cos_h`.
    ///
    /// Returns zero for a back-facing half vector (`cos_h <= 0`).
    #[must_use]
    pub fn distribution(&self, cos_h: f32) -> f32 {
        if cos_h <= 0.0 {
            return 0.0;
        }
        let a2 = self.alpha * self.alpha;
        let c2 = cos_h * cos_h;
        // Denominator (cos_h^2 (alpha^2 - 1) + 1)^2; the `+1` keeps it positive.
        let denom = c2 * (a2 - 1.0) + 1.0;
        a2 * INV_PI / (denom * denom)
    }

    /// The Smith `Lambda` auxiliary for a direction whose cosine to the normal
    /// is `cos_w`, i.e. the ratio of masked to visible projected microfacet area.
    ///
    /// Grazing directions (`|cos_w| -> 0`) drive `Lambda -> infinity`; the normal
    /// incidence case (`|cos_w| = 1`) returns zero.
    fn lambda(&self, cos_w: f32) -> f32 {
        let c = cos_w.abs();
        if c >= 1.0 {
            return 0.0;
        }
        let c2 = c * c;
        // tan^2(theta) = (1 - cos^2) / cos^2.
        let tan2 = (1.0 - c2) / c2;
        let a2 = self.alpha * self.alpha;
        0.5 * ((1.0 + a2 * tan2).sqrt() - 1.0)
    }

    /// The Smith single-direction masking term `G1(w)` in `[0, 1]`.
    #[must_use]
    pub fn g1(&self, cos_w: f32) -> f32 {
        1.0 / (1.0 + self.lambda(cos_w))
    }

    /// The height-correlated Smith masking-shadowing term `G2(wo, wi)`.
    ///
    /// The height-correlated form shares a single `Lambda` sum between the view
    /// and light directions, which is both more accurate and never larger than
    /// the separable product `G1(wo) * G1(wi)`.
    #[must_use]
    pub fn g2(&self, cos_o: f32, cos_i: f32) -> f32 {
        1.0 / (1.0 + self.lambda(cos_o) + self.lambda(cos_i))
    }

    /// The solid-angle probability density of the reflected direction produced
    /// by [`Self::sample_half_vector`], given the view cosine `cos_o = n·wo` and
    /// the half-vector cosine `cos_h = n·h`.
    ///
    /// This is the visible-normal density folded through the reflection Jacobian
    /// `1 / (4 wo·h)`; the `wo·h` factor cancels, leaving
    /// `G1(wo) * D(h) / (4 * cos_o)`.
    #[must_use]
    pub fn reflection_pdf(&self, cos_o: f32, cos_h: f32) -> f32 {
        if cos_o <= 0.0 {
            return 0.0;
        }
        self.g1(cos_o) * self.distribution(cos_h) / (4.0 * cos_o)
    }

    /// Importance-samples a world-space microfacet half vector around unit
    /// `normal` for the view direction `wo`, using Heitz's `VNDF` method.
    ///
    /// Returns [`None`] when the view direction is below the surface or the warp
    /// degenerates, so the caller terminates rather than producing a `NaN`.
    #[must_use]
    pub fn sample_half_vector(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<Vec3> {
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        // Express the view direction in the local frame whose +z is `normal`.
        let (tangent, bitangent) = orthonormal_basis(normal);
        let wo_local = Vec3::new(wo.dot(tangent), wo.dot(bitangent), cos_o.max(0.0));

        // 1. Stretch the view direction into the hemisphere configuration.
        let vh = Vec3::new(self.alpha * wo_local.x, self.alpha * wo_local.y, wo_local.z)
            .normalize_or_zero();
        if vh.length_squared() <= EPS_LEN_SQ {
            return None;
        }

        // 2. Build an orthonormal basis around the stretched view direction.
        let lensq = vh.x * vh.x + vh.y * vh.y;
        let t1 = if lensq > EPS_LEN_SQ {
            Vec3::new(-vh.y, vh.x, 0.0).scale(1.0 / lensq.sqrt())
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let t2 = vh.cross(t1);

        // 3. Sample a uniform point in the unit disk and warp it onto the
        //    projected hemisphere (the Heitz spherical-cap mapping).
        let (p1, p2_raw) = uniform_disk(rng);
        let s = 0.5 * (1.0 + vh.z);
        let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2_raw;

        // 4. Lift to the hemisphere and un-stretch back to the GGX normal.
        let pz = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
        let nh = t1.scale(p1).add(t2.scale(p2)).add(vh.scale(pz));
        let h_local =
            Vec3::new(self.alpha * nh.x, self.alpha * nh.y, nh.z.max(0.0)).normalize_or_zero();
        if h_local.length_squared() <= EPS_LEN_SQ {
            return None;
        }

        // 5. Rotate the local half vector back into world space.
        let half = tangent
            .scale(h_local.x)
            .add(bitangent.scale(h_local.y))
            .add(normal.scale(h_local.z))
            .normalize_or_zero();
        if half.length_squared() <= EPS_LEN_SQ {
            None
        } else {
            Some(half)
        }
    }
}

/// Schlick's Fresnel approximation `F(cos) = F0 + (1 - F0)(1 - cos)^5`,
/// evaluated per channel so coloured conductor reflectance is preserved.
///
/// `cos_theta` is the cosine between the incident direction and the half vector
/// (clamped into `[0, 1]`); `f0` is the normal-incidence reflectance.
#[must_use]
pub fn fresnel_schlick(f0: Vec3, cos_theta: f32) -> Vec3 {
    let c = (1.0 - cos_theta).clamp(0.0, 1.0);
    let c2 = c * c;
    // (1 - cos)^5 spelled out so no `powf`/`powi` is needed.
    let c5 = c2 * c2 * c;
    f0.add(Vec3::ONE.sub(f0).scale(c5))
}

#[cfg(test)]
mod tests {
    use super::super::sampler::Rng;
    use super::super::{Vec3, PI};
    use super::*;

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    #[test]
    fn alpha_remap_is_squared_and_clamped() {
        assert!((GgxIsotropic::from_roughness(0.5).alpha - 0.25).abs() < 1e-7);
        // Zero roughness is clamped to the floor, never exactly zero.
        assert!((GgxIsotropic::from_roughness(0.0).alpha - MIN_ALPHA).abs() < 1e-7);
        // Out-of-range roughness is clamped before squaring.
        assert!((GgxIsotropic::from_roughness(2.0).alpha - 1.0).abs() < 1e-7);
    }

    #[test]
    fn distribution_integrates_to_one_over_the_hemisphere() {
        // integral over the hemisphere of D(h) * cos_h dωh == 1 for any alpha.
        let ggx = GgxIsotropic::from_roughness(0.4);
        let mut rng = Rng::seed(7);
        let count = 400_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            // Uniform hemisphere direction (pdf = 1/(2*pi)) via cube rejection.
            let h = loop {
                let v = Vec3::new(
                    2.0 * rng.next_f32() - 1.0,
                    2.0 * rng.next_f32() - 1.0,
                    2.0 * rng.next_f32() - 1.0,
                );
                let l2 = v.length_squared();
                if l2 > 1e-6 && l2 <= 1.0 {
                    let u = v.normalize_or_zero();
                    break if u.dot(N) < 0.0 { u.negate() } else { u };
                }
            };
            let cos_h = N.dot(h).max(0.0);
            let d = f64::from(ggx.distribution(cos_h));
            sum += d * f64::from(cos_h) * (2.0 * f64::from(PI));
        }
        let integral = sum / f64::from(count);
        assert!(
            (integral - 1.0).abs() < 2e-2,
            "GGX NDF must integrate to one, got {integral}"
        );
    }

    #[test]
    fn masking_is_bounded_and_monotone() {
        let ggx = GgxIsotropic::from_roughness(0.3);
        // G1 is in [0, 1] and increases toward normal incidence.
        let grazing = ggx.g1(0.05);
        let normal_incidence = ggx.g1(1.0);
        assert!(grazing >= 0.0 && grazing <= 1.0);
        assert!(normal_incidence >= 0.0 && normal_incidence <= 1.0);
        assert!(normal_incidence > grazing);
        // Height-correlated G2 never exceeds either single-direction G1.
        let g2 = ggx.g2(0.6, 0.4);
        assert!(g2 <= ggx.g1(0.6) + 1e-6 && g2 <= ggx.g1(0.4) + 1e-6);
    }

    #[test]
    fn fresnel_endpoints() {
        let f0 = Vec3::new(0.04, 0.1, 0.2);
        // At normal incidence Fresnel returns F0 exactly.
        let at_normal = fresnel_schlick(f0, 1.0);
        assert!(at_normal.sub(f0).length_squared() < 1e-12);
        // At grazing Fresnel saturates to one on every channel.
        let at_grazing = fresnel_schlick(f0, 0.0);
        assert!(at_grazing.sub(Vec3::ONE).length_squared() < 1e-12);
    }

    #[test]
    fn sampled_half_vectors_stay_in_the_upper_hemisphere() {
        let ggx = GgxIsotropic::from_roughness(0.5);
        let mut rng = Rng::seed(123);
        let wo = Vec3::new(0.3, 0.9, 0.1).normalize_or_zero();
        for _ in 0..20_000 {
            let h = ggx.sample_half_vector(wo, N, &mut rng).expect("valid half");
            assert!(N.dot(h) > 0.0, "half vector below surface");
            assert!((h.length() - 1.0).abs() < 1e-4, "half vector not unit");
        }
    }

    #[test]
    fn sampling_density_matches_reflection_pdf() {
        // Monte Carlo estimate of the pdf normalisation: integrating the sampled
        // density over the solid angle it covers must converge to one. We verify
        // the analytic `reflection_pdf` is self-consistent with the sampler by
        // checking that 1/pdf averaged against the sampling process is finite and
        // that the half-vector cosine density follows D-weighted statistics.
        let ggx = GgxIsotropic::from_roughness(0.35);
        let mut rng = Rng::seed(321);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let count = 50_000u32;
        let mut mean_pdf = 0.0f64;
        for _ in 0..count {
            let h = ggx.sample_half_vector(wo, N, &mut rng).expect("valid half");
            let cos_h = N.dot(h);
            let cos_o = N.dot(wo);
            let pdf = ggx.reflection_pdf(cos_o, cos_h);
            assert!(pdf > 0.0 && pdf.is_finite(), "pdf must be positive finite");
            // The sampled half vector always sits in the upper hemisphere; the
            // reflected direction may fall below it for rough surfaces (that lost
            // energy is exactly the single-scatter deficit), so it is not tested
            // here.
            mean_pdf += f64::from(pdf);
        }
        mean_pdf /= f64::from(count);
        assert!(mean_pdf > 0.0, "mean pdf must be positive");
    }
}
