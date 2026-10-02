//! Surface scattering models for the reference path tracer.
//!
//! Two classical lobes are provided, each exposing the same three operations a
//! `Monte Carlo` integrator needs:
//!
//! - **evaluate** — the `BRDF` value `f_r(wo, wi)` for a fixed pair of
//!   directions (zero for a perfectly specular lobe, whose energy lives in a
//!   Dirac delta that cannot be evaluated for an arbitrary `wi`).
//! - **sample** — importance-sample an outgoing direction `wi` given the view
//!   direction `wo`, returning the direction, the `BRDF` value, and its density.
//! - **pdf** — the solid-angle probability density of a given `(wo, wi)` pair,
//!   consistent with `sample` (zero for a specular lobe).
//!
//! Directions follow the usual convention: `wo` and `wi` both point *away* from
//! the surface, and `normal` is the (viewer-facing) shading normal. All
//! quantities are linear radiance scales, never gamma-encoded.

use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, Rng};
use super::{Vec3, INV_PI};

/// A surface scattering model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bsdf {
    /// Ideal Lambertian diffuse reflector with a spectral `albedo` in `[0, 1]`
    /// per channel. Its `BRDF` is the constant `albedo / pi`.
    Lambert {
        /// Per-channel diffuse reflectance (hemispherical albedo).
        albedo: Vec3,
    },
    /// Perfect specular mirror with a per-channel `reflectance` in `[0, 1]`.
    /// All energy leaves along the single mirror direction, so the lobe is a
    /// Dirac delta (its [`Bsdf::evaluate`]/[`Bsdf::pdf`] are zero).
    Mirror {
        /// Per-channel specular reflectance.
        reflectance: Vec3,
    },
}

/// The outcome of importance-sampling a [`Bsdf`].
#[derive(Clone, Copy, Debug)]
pub struct BsdfSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The `BRDF` value `f_r(wo, wi)` for the sampled pair. For a specular lobe
    /// this is the reflectance divided by `cos(theta_i)` so that the Monte
    /// Carlo weight `value * cos / pdf` equals the reflectance exactly.
    pub value: Vec3,
    /// The solid-angle probability density of `direction`. For a specular lobe
    /// this is `1` (the delta is folded into `value`).
    pub pdf: f32,
    /// `true` when the lobe is a Dirac delta (specular); the integrator then
    /// skips next-event estimation and multiple-importance weighting.
    pub specular: bool,
}

impl Bsdf {
    /// `true` when this lobe is a perfectly specular (delta) reflector.
    #[must_use]
    pub const fn is_specular(&self) -> bool {
        matches!(self, Self::Mirror { .. })
    }

    /// Evaluates the `BRDF` value `f_r(wo, wi)` for a fixed direction pair.
    ///
    /// Returns [`Vec3::ZERO`] when `wo` and `wi` are not in the same hemisphere
    /// as `normal`, and always zero for a specular lobe (its energy is a delta
    /// that an arbitrary `wi` misses).
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        match self {
            Self::Lambert { albedo } => {
                if normal.dot(wo) > 0.0 && normal.dot(wi) > 0.0 {
                    albedo.scale(INV_PI)
                } else {
                    Vec3::ZERO
                }
            }
            Self::Mirror { .. } => Vec3::ZERO,
        }
    }

    /// The solid-angle density `sample` would assign to `(wo, wi)`.
    ///
    /// Zero for a specular lobe and zero when `wi` is below the surface.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        match self {
            Self::Lambert { .. } => {
                if normal.dot(wo) > 0.0 {
                    cosine_hemisphere_pdf(normal, wi)
                } else {
                    0.0
                }
            }
            Self::Mirror { .. } => 0.0,
        }
    }

    /// Importance-samples an outgoing direction given the view direction `wo`
    /// and the viewer-facing shading `normal`.
    ///
    /// Returns `None` when the sample is degenerate (grazing/zero-length), so
    /// the caller terminates the path rather than dividing by zero.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<BsdfSample> {
        match self {
            Self::Lambert { albedo } => {
                if normal.dot(wo) <= 0.0 {
                    return None;
                }
                let hemi = cosine_sample_hemisphere(normal, rng);
                let cos_i = normal.dot(hemi.direction);
                if cos_i <= 0.0 || hemi.pdf <= 0.0 {
                    return None;
                }
                Some(BsdfSample {
                    direction: hemi.direction,
                    value: albedo.scale(INV_PI),
                    pdf: hemi.pdf,
                    specular: false,
                })
            }
            Self::Mirror { reflectance } => {
                let cos_o = normal.dot(wo);
                if cos_o <= 0.0 {
                    return None;
                }
                // Mirror the view direction about the normal.
                let wi = wo.negate().reflect(normal).normalize_or_zero();
                let cos_i = normal.dot(wi);
                if cos_i <= 0.0 {
                    return None;
                }
                // Fold the delta: weight = value * cos_i / pdf must equal
                // `reflectance`, with pdf = 1, so value = reflectance / cos_i.
                Some(BsdfSample {
                    direction: wi,
                    value: reflectance.scale(1.0 / cos_i),
                    pdf: 1.0,
                    specular: true,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::sampler::Rng;
    use super::*;

    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    #[test]
    fn lambert_brdf_is_albedo_over_pi() {
        let bsdf = Bsdf::Lambert {
            albedo: Vec3::splat(0.5),
        };
        let v = bsdf.evaluate(N, N, N);
        assert!((v.x - 0.5 * INV_PI).abs() < 1e-7);
    }

    #[test]
    fn lambert_brdf_zero_below_surface() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let below = Vec3::new(0.0, -1.0, 0.0);
        assert_eq!(bsdf.evaluate(N, below, N), Vec3::ZERO);
    }

    #[test]
    fn lambert_energy_conservation() {
        // Hemispherical integral of f_r * cos must equal the albedo (<= 1).
        let albedo = 0.8f32;
        let bsdf = Bsdf::Lambert {
            albedo: Vec3::splat(albedo),
        };
        let mut rng = Rng::seed(11);
        let count = 400_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            // Uniform hemisphere sample via cube rejection, pdf = 1/(2*pi).
            let wi = loop {
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
            let cos_i = N.dot(wi).max(0.0);
            let fr = bsdf.evaluate(N, wi, N).x as f64;
            // estimator = f_r * cos / pdf = f_r * cos * 2*pi
            sum += fr * cos_i as f64 * (2.0 * super::super::PI as f64);
        }
        let reflectance = sum / f64::from(count);
        assert!(
            (reflectance - albedo as f64).abs() < 2e-2,
            "directional-hemispherical reflectance {reflectance} should equal albedo {albedo}"
        );
        assert!(reflectance <= 1.0 + 1e-2, "reflectance must not exceed 1");
    }

    #[test]
    fn lambert_sample_matches_pdf() {
        let bsdf = Bsdf::Lambert {
            albedo: Vec3::splat(0.3),
        };
        let mut rng = Rng::seed(77);
        let wo = Vec3::new(0.2, 1.0, 0.1).normalize_or_zero();
        for _ in 0..10_000 {
            let s = bsdf.sample(wo, N, &mut rng).expect("diffuse sample valid");
            let pdf = bsdf.pdf(wo, s.direction, N);
            assert!(
                (pdf - s.pdf).abs() < 1e-5,
                "pdf mismatch {pdf} vs {}",
                s.pdf
            );
            assert!(s.direction.is_finite());
            assert!(!s.specular);
        }
    }

    #[test]
    fn lambert_sample_weight_is_albedo() {
        // The Monte Carlo throughput factor value*cos/pdf must equal the albedo.
        let bsdf = Bsdf::Lambert {
            albedo: Vec3::splat(0.6),
        };
        let mut rng = Rng::seed(999);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..10_000 {
            let s = bsdf.sample(wo, N, &mut rng).expect("valid");
            let cos_i = N.dot(s.direction);
            let weight = s.value.scale(cos_i / s.pdf);
            assert!(
                (weight.x - 0.6).abs() < 1e-4,
                "weight {} != albedo",
                weight.x
            );
        }
    }

    #[test]
    fn mirror_reflects_about_normal() {
        let bsdf = Bsdf::Mirror {
            reflectance: Vec3::splat(0.9),
        };
        let mut rng = Rng::seed(1);
        // View coming from above-and-to-the-side.
        let wo = Vec3::new(1.0, 1.0, 0.0).normalize_or_zero();
        let s = bsdf.sample(wo, N, &mut rng).expect("mirror sample valid");
        // Perfect mirror of wo about +Y is (-1, 1, 0) normalized.
        let expect = Vec3::new(-1.0, 1.0, 0.0).normalize_or_zero();
        assert!(
            s.direction.sub(expect).length_squared() < 1e-8,
            "mirror direction {:?} != {:?}",
            s.direction,
            expect
        );
        assert!(s.specular);
        // Delta lobe does not evaluate for an arbitrary direction.
        assert_eq!(bsdf.evaluate(wo, s.direction, N), Vec3::ZERO);
        assert!(bsdf.pdf(wo, s.direction, N) < 1e-9);
        // Throughput weight = value * cos / pdf must equal the reflectance.
        let cos_i = N.dot(s.direction);
        let weight = s.value.scale(cos_i / s.pdf);
        assert!((weight.x - 0.9).abs() < 1e-5);
    }

    #[test]
    fn sample_below_surface_returns_none() {
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(3);
        let wo = Vec3::new(0.0, -1.0, 0.0);
        assert!(bsdf.sample(wo, N, &mut rng).is_none());
    }
}
