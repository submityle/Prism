//! Oren-Nayar rough-diffuse `BRDF` for the reference path tracer.
//!
//! Lambert assumes a perfectly smooth diffuse surface, but real matte materials
//! (unfinished clay, concrete, the Moon) are rough at the micro-scale: they are
//! modelled as a dense field of Lambertian V-cavity microfacets whose
//! masking, shadowing, and single inter-reflection flatten the limb and add a
//! grazing-angle retro-reflection that Lambert cannot reproduce. Oren and Nayar
//! ("Generalization of Lambert's Reflectance Model", `SIGGRAPH` 1994) derived
//! the reflectance of that cavity field; this module implements the trig-free
//! "qualitative" form (Gotanda / Fujii), which keeps the dominant `A` and `B`
//! lobes and expresses the azimuthal coupling purely with dot products so no
//! transcendental call is needed.
//!
//! It is the path-traced oracle for the real-time Oren-Nayar shader in
//! [`crate::particle::oren_nayar`]: both evaluate the identical analytic lobe,
//! so a converged render here is the ground truth the `GPU` twin is checked
//! against. The lobe is a non-delta diffuse reflector, so it is cosine-sampled
//! and connected to lights by next-event estimation exactly like
//! [`crate::reference_pt::bsdf::Bsdf::Lambert`].
//!
//! Conventions match the rest of the tracer: `wo`, `wi`, and `normal` are unit
//! vectors with `wo` and `wi` pointing away from the surface, and the model is
//! reciprocal because its `s` numerator and `t = max(cos_o, cos_i)` denominator
//! are both symmetric in the two directions. Only `sqrt` (inside the shared
//! cosine sampler) and division are used.

#[cfg(test)]
use super::sampler::Rng;
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, SampleSource};
use super::{Vec3, INV_PI};

/// Denominator offset of the Oren-Nayar `A` (base) coefficient: the `0.33` in
/// `A = 1 - 0.5 * sigma^2 / (sigma^2 + 0.33)`.
const A_OFFSET: f32 = 0.33;

/// Denominator offset of the Oren-Nayar `B` (inter-reflection) coefficient: the
/// `0.09` in `B = 0.45 * sigma^2 / (sigma^2 + 0.09)`.
const B_OFFSET: f32 = 0.09;

/// Leading factor of the Oren-Nayar `A` coefficient (`0.5`).
const A_SCALE: f32 = 0.5;

/// Leading factor of the Oren-Nayar `B` coefficient (`0.45`).
const B_SCALE: f32 = 0.45;

/// Minimum value of the `s / t` denominator, guarding the degenerate case where
/// `s > 0` yet `max(cos_o, cos_i)` is numerically zero, which would otherwise
/// produce `inf * 0 = NaN`. The lobe is clamped non-negative afterwards, so this
/// floor only removes the `NaN`; it never lowers a valid value.
const MIN_T: f32 = 1e-6;

/// The outcome of importance-sampling an [`OrenNayar`] lobe.
#[derive(Clone, Copy, Debug)]
pub struct OrenNayarSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The `BRDF` value `f_r(wo, wi)` for the sampled pair (no cosine applied).
    pub value: Vec3,
    /// The cosine-weighted solid-angle density `cos(theta_i) / pi` of
    /// `direction`.
    pub pdf: f32,
}

/// A rough Lambertian (Oren-Nayar) diffuse reflector parameterized by its
/// hemispherical `albedo` and a micro-slope roughness `sigma` in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrenNayar {
    /// Per-channel diffuse reflectance (hemispherical albedo) in `[0, 1]`.
    albedo: Vec3,
    /// Micro-slope standard deviation (radians); `0` collapses to Lambert.
    sigma: f32,
}

impl OrenNayar {
    /// Builds a rough-diffuse lobe from an `albedo` and roughness `sigma`
    /// (radians). Negative `sigma` is treated as `0` (Lambert) by [`Self::coeffs`].
    #[must_use]
    pub fn new(albedo: Vec3, sigma: f32) -> Self {
        Self { albedo, sigma }
    }

    /// The Oren-Nayar `A` (base lobe) and `B` (inter-reflection lobe)
    /// coefficients for this surface's `sigma`.
    ///
    /// `A = 1 - 0.5 * sigma^2 / (sigma^2 + 0.33)` and
    /// `B = 0.45 * sigma^2 / (sigma^2 + 0.09)`; at `sigma = 0` this is `(1, 0)`,
    /// so the lobe collapses to Lambert. `sigma` is clamped non-negative.
    fn coeffs(&self) -> (f32, f32) {
        let s = self.sigma.max(0.0);
        let sigma_sq = s * s;
        let a = 1.0 - A_SCALE * sigma_sq / (sigma_sq + A_OFFSET);
        let b = B_SCALE * sigma_sq / (sigma_sq + B_OFFSET);
        (a, b)
    }

    /// The `BRDF` value `albedo / pi * (A + B * s / t)` for a direction pair
    /// already known to lie in the hemisphere of unit `normal`.
    ///
    /// `cos_o` and `cos_i` are the (positive) cosines of `wo` and `wi` to
    /// `normal`; `s = dot(wo, wi) - cos_o * cos_i` is the azimuthal coupling and
    /// `t = max(cos_o, cos_i)` when `s > 0`, else `1` (the Fujii direction-only
    /// denominator). The lobe is clamped non-negative.
    fn lobe(&self, wo: Vec3, wi: Vec3, cos_o: f32, cos_i: f32) -> Vec3 {
        let (a, b) = self.coeffs();
        let s = wo.dot(wi) - cos_o * cos_i;
        let t = if s > 0.0 {
            cos_o.max(cos_i).max(MIN_T)
        } else {
            1.0
        };
        let lobe = (a + b * s / t).max(0.0);
        self.albedo.scale(INV_PI * lobe)
    }

    /// Evaluates the `BRDF` value `f_r(wo, wi)` (no cosine applied).
    ///
    /// Returns [`Vec3::ZERO`] when either direction is on or below the surface.
    /// `normal` is the shading normal already oriented toward `wo`.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
            return Vec3::ZERO;
        }
        self.lobe(wo, wi, cos_o, cos_i)
    }

    /// The cosine-weighted solid-angle density this lobe assigns to `(wo, wi)`.
    ///
    /// Zero when `wo` is below the surface; otherwise `max(cos_i, 0) / pi`.
    /// `normal` is the shading normal already oriented toward `wo`.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        if normal.dot(wo) > 0.0 {
            cosine_hemisphere_pdf(normal, wi)
        } else {
            0.0
        }
    }

    /// Importance-samples an outgoing direction with a cosine-weighted
    /// hemisphere about the view-facing `normal`.
    ///
    /// The raw geometric `normal` is oriented toward `wo` so a back-facing hit
    /// still scatters. Returns `None` for a degenerate (grazing/zero-length)
    /// sample so the caller can terminate the path instead of dividing by zero.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<OrenNayarSample> {
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        let hemi = cosine_sample_hemisphere(normal, rng);
        let cos_i = normal.dot(hemi.direction);
        if cos_i <= 0.0 || hemi.pdf <= 0.0 {
            return None;
        }
        Some(OrenNayarSample {
            direction: hemi.direction,
            value: self.lobe(wo, hemi.direction, cos_o, cos_i),
            pdf: hemi.pdf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_pt::sampler::orthonormal_basis;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    fn vec_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    /// At `sigma = 0` the lobe must equal Lambert `albedo / pi` for every pair.
    #[test]
    fn sigma_zero_is_lambert() {
        let albedo = Vec3::new(0.2, 0.5, 0.9);
        let on = OrenNayar::new(albedo, 0.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let dirs = [
            unit(0.0, 0.0, 1.0),
            unit(0.5, 0.0, 1.0),
            unit(-0.3, 0.4, 1.0),
            unit(0.7, -0.2, 0.8),
        ];
        for &wo in &dirs {
            for &wi in &dirs {
                let got = on.evaluate(wo, wi, normal);
                assert!(vec_approx_eq(got, albedo.scale(INV_PI)));
            }
        }
    }

    /// Below-horizon directions on either side return exactly zero.
    #[test]
    fn back_facing_is_zero() {
        let on = OrenNayar::new(Vec3::splat(0.8), 0.5);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let up = unit(0.3, 0.1, 1.0);
        let down = unit(0.3, 0.1, -1.0);
        assert_eq!(on.evaluate(down, up, normal), Vec3::ZERO);
        assert_eq!(on.evaluate(up, down, normal), Vec3::ZERO);
    }

    /// The `BRDF` is reciprocal: `f_r(wo, wi) == f_r(wi, wo)`.
    #[test]
    fn is_reciprocal() {
        let on = OrenNayar::new(Vec3::new(0.6, 0.6, 0.6), 0.7);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let wo = unit(0.6, -0.1, 0.8);
        let wi = unit(-0.4, 0.5, 0.6);
        let a = on.evaluate(wo, wi, normal);
        let b = on.evaluate(wi, wo, normal);
        assert!(vec_approx_eq(a, b));
    }

    /// Roughening the surface brightens the grazing retro-reflection (the lobe
    /// grows where `wo == wi` at a shallow angle), never darkening it below
    /// Lambert there.
    #[test]
    fn roughness_brightens_grazing_retroreflection() {
        let albedo = Vec3::splat(0.8);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let grazing = unit(0.95, 0.0, 0.31);
        let smooth = OrenNayar::new(albedo, 0.0).evaluate(grazing, grazing, normal);
        let rough = OrenNayar::new(albedo, 0.8).evaluate(grazing, grazing, normal);
        assert!(rough.x > smooth.x + CMP_EPS);
    }

    /// A sampled direction must stay in the view hemisphere, report the matching
    /// cosine density, and carry the same value [`OrenNayar::evaluate`] would.
    #[test]
    fn sample_is_consistent_with_pdf_and_evaluate() {
        let on = OrenNayar::new(Vec3::new(0.3, 0.6, 0.9), 0.6);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let wo = unit(0.4, 0.2, 0.9);
        let mut rng = Rng::seed(7);
        for _ in 0..256 {
            if let Some(s) = on.sample(wo, normal, &mut rng) {
                assert!(normal.dot(s.direction) > 0.0);
                assert!(approx_eq(s.pdf, on.pdf(wo, s.direction, normal)));
                assert!(vec_approx_eq(s.value, on.evaluate(wo, s.direction, normal)));
            }
        }
    }

    /// Monte Carlo directional albedo must not exceed the input albedo: the
    /// qualitative lobe conserves (never creates) energy for a white furnace.
    #[test]
    fn directional_albedo_does_not_exceed_input() {
        let albedo = Vec3::splat(1.0);
        let on = OrenNayar::new(albedo, 1.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let (tangent, _bitangent) = orthonormal_basis(normal);
        // A fixed oblique view direction.
        let wo = tangent
            .scale(0.6)
            .add(normal.scale(0.8))
            .normalize_or_zero();
        let mut rng = Rng::seed(123);
        let samples = 20_000u32;
        let mut sum = 0.0f32;
        for _ in 0..samples {
            if let Some(s) = on.sample(wo, normal, &mut rng) {
                let cos_i = normal.dot(s.direction).max(0.0);
                // Monte Carlo of integral f_r * cos_i d_omega with cosine pdf:
                // each term is f_r * cos_i / pdf = f_r * pi (pdf = cos_i / pi).
                sum += s.value.x * cos_i / s.pdf;
            }
        }
        let directional_albedo = sum / (samples as f32);
        assert!(directional_albedo <= 1.0 + 1e-2);
        // And it is a meaningful (non-trivial) fraction of the hemisphere.
        assert!(directional_albedo > 0.5);
    }
}
