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

use super::dielectric::{fresnel_dielectric, refract};
use super::microfacet::{fresnel_schlick, GgxIsotropic};
use super::sampler::{cosine_hemisphere_pdf, cosine_sample_hemisphere, Rng};
use super::{Vec3, EPS_LEN_SQ, INV_PI};

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
    /// Rough conductor (metal) described by an isotropic GGX microfacet lobe.
    ///
    /// `reflectance` is the normal-incidence Fresnel reflectance `F0` (the
    /// characteristic metallic tint, e.g. gold or copper), and `roughness` is
    /// the perceptual roughness in `[0, 1]` remapped to the GGX width
    /// `alpha = roughness^2`. As `roughness -> 0` the lobe narrows toward the
    /// [`Bsdf::Mirror`] limit, but it is never a Dirac delta, so it is sampled
    /// and connected to lights like any glossy surface.
    GgxConductor {
        /// Per-channel normal-incidence reflectance `F0`.
        reflectance: Vec3,
        /// Perceptual roughness in `[0, 1]`.
        roughness: f32,
    },
    /// Smooth (perfectly specular) dielectric interface: glass, water, or a
    /// clear coat. Light is either mirror-reflected or refracted through the
    /// surface, with the split governed by the angle-dependent `Fresnel`
    /// equations (see [`crate::reference_pt::dielectric`]). Both outgoing lobes
    /// are Dirac deltas, so [`Bsdf::evaluate`]/[`Bsdf::pdf`] are zero and the
    /// surface is sampled, never connected to lights by next-event estimation.
    Dielectric {
        /// Relative index of refraction `eta_t / eta_i` of the interior medium
        /// over the exterior (e.g. `1.5` for air-to-glass). Must be positive
        /// and not equal to `1` for the interface to refract.
        ior: f32,
        /// Per-channel tint applied to the mirror-reflected component.
        reflectance: Vec3,
        /// Per-channel tint applied to the refracted (transmitted) component.
        transmittance: Vec3,
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
        matches!(self, Self::Mirror { .. } | Self::Dielectric { .. })
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
            // A perfect mirror and a specular dielectric both keep their energy
            // in Dirac deltas that an arbitrary `wi` misses.
            Self::Mirror { .. } | Self::Dielectric { .. } => Vec3::ZERO,
            Self::GgxConductor {
                reflectance,
                roughness,
            } => Self::ggx_evaluate(*reflectance, *roughness, wo, wi, normal),
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
            Self::Mirror { .. } | Self::Dielectric { .. } => 0.0,
            Self::GgxConductor { roughness, .. } => {
                let cos_o = normal.dot(wo);
                let cos_i = normal.dot(wi);
                if cos_o <= 0.0 || cos_i <= 0.0 {
                    return 0.0;
                }
                let half = wo.add(wi).normalize_or_zero();
                if half.length_squared() <= EPS_LEN_SQ {
                    return 0.0;
                }
                let cos_h = normal.dot(half);
                GgxIsotropic::from_roughness(*roughness).reflection_pdf(cos_o, cos_h)
            }
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
                // The integrator passes the raw geometric normal; orient it into
                // the view hemisphere so a back-facing hit still scatters.
                let normal = normal.faced_toward(wo);
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
                // Orient the raw geometric normal into the view hemisphere.
                let normal = normal.faced_toward(wo);
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
            Self::GgxConductor {
                reflectance,
                roughness,
            } => Self::ggx_sample(*reflectance, *roughness, wo, normal, rng),
            Self::Dielectric {
                ior,
                reflectance,
                transmittance,
            } => Self::dielectric_sample(*ior, *reflectance, *transmittance, wo, normal, rng),
        }
    }

    /// Evaluates the GGX rough-conductor `BRDF` `f_r = F * D * G2 / (4 cos_o cos_i)`.
    ///
    /// Returns [`Vec3::ZERO`] when either direction is below the surface or the
    /// half vector degenerates.
    fn ggx_evaluate(reflectance: Vec3, roughness: f32, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o <= 0.0 || cos_i <= 0.0 {
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
        let ggx = GgxIsotropic::from_roughness(roughness);
        let d = ggx.distribution(cos_h);
        let g2 = ggx.g2(cos_o, cos_i);
        let fresnel = fresnel_schlick(reflectance, wo.dot(half).max(0.0));
        fresnel.scale(d * g2 / (4.0 * cos_o * cos_i))
    }

    /// Importance-samples the GGX rough conductor via visible-normal sampling.
    ///
    /// The returned `value`/`pdf` are the true `BRDF` and its solid-angle
    /// density, so the integrator's generic throughput update
    /// `value * cos_i / pdf` reduces to the clean microfacet weight
    /// `F * G2 / G1(wo)`.
    fn ggx_sample(
        reflectance: Vec3,
        roughness: f32,
        wo: Vec3,
        normal: Vec3,
        rng: &mut Rng,
    ) -> Option<BsdfSample> {
        // Orient the raw geometric normal into the view hemisphere.
        let normal = normal.faced_toward(wo);
        let cos_o = normal.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        let ggx = GgxIsotropic::from_roughness(roughness);
        let half = ggx.sample_half_vector(wo, normal, rng)?;
        let woh = wo.dot(half);
        if woh <= 0.0 {
            return None;
        }
        // Reflect the view direction about the sampled microfacet normal.
        let wi = wo.negate().reflect(half).normalize_or_zero();
        let cos_i = normal.dot(wi);
        if cos_i <= 0.0 {
            return None;
        }
        let cos_h = normal.dot(half);
        let pdf = ggx.reflection_pdf(cos_o, cos_h);
        if pdf <= 0.0 {
            return None;
        }
        let d = ggx.distribution(cos_h);
        let g2 = ggx.g2(cos_o, cos_i);
        let fresnel = fresnel_schlick(reflectance, woh);
        let value = fresnel.scale(d * g2 / (4.0 * cos_o * cos_i));
        Some(BsdfSample {
            direction: wi,
            value,
            pdf,
            specular: false,
        })
    }

    /// Importance-samples a smooth dielectric by stochastically choosing the
    /// reflected or refracted lobe in proportion to the `Fresnel` reflectance.
    ///
    /// The interface side is recovered from the *raw* geometric `normal`: a
    /// positive `normal . wo` means the ray is outside entering the medium, a
    /// negative one means it is inside leaving it. The randomly chosen branch
    /// probability (`fr` for reflection, `1 - fr` for transmission) cancels the
    /// matching `Fresnel` factor, so the returned `value` carries only the tint
    /// and the radiance-space solid-angle compression `eta^2` for transmission.
    fn dielectric_sample(
        ior: f32,
        reflectance: Vec3,
        transmittance: Vec3,
        wo: Vec3,
        normal: Vec3,
        rng: &mut Rng,
    ) -> Option<BsdfSample> {
        // Decide which side of the interface the view ray is on, then orient the
        // normal to face the view direction (the incident side).
        let entering = normal.dot(wo) > 0.0;
        let n = if entering { normal } else { normal.negate() };
        let cos_o = n.dot(wo);
        if cos_o <= 0.0 {
            return None;
        }
        // Exterior index is 1 (vacuum/air); interior index is `ior`.
        let (eta_i, eta_t) = if entering { (1.0, ior) } else { (ior, 1.0) };
        let fr = fresnel_dielectric(cos_o, eta_i, eta_t);
        if rng.next_f32() < fr {
            // Reflect: mirror the view direction about the oriented normal. The
            // chosen-branch probability `fr` cancels the Fresnel reflectance, so
            // only the tint survives in the folded delta weight.
            let wi = wo.negate().reflect(n).normalize_or_zero();
            let cos_i = n.dot(wi);
            if cos_i <= 0.0 {
                return None;
            }
            Some(BsdfSample {
                direction: wi,
                value: reflectance.scale(1.0 / cos_i),
                pdf: 1.0,
                specular: true,
            })
        } else {
            // Refract through the interface. `eta` is the incident-over-
            // transmitted index ratio; `refract` returns `None` under total
            // internal reflection (handled above by `fr == 1`, but guarded).
            let eta = eta_i / eta_t;
            let wi = refract(wo, n, eta)?;
            let cos_i = n.dot(wi).abs();
            if cos_i <= 0.0 {
                return None;
            }
            // Radiance transport across an index change is compressed by the
            // square of the relative index (`PBRT` radiance-mode `eta^2`). The
            // `(1 - fr)` branch probability cancels the transmitted Fresnel
            // factor `1 - fr`.
            let factor = eta * eta / cos_i;
            Some(BsdfSample {
                direction: wi,
                value: transmittance.scale(factor),
                pdf: 1.0,
                specular: true,
            })
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
    fn sample_degenerate_view_returns_none() {
        // The sampler now orients the raw normal toward `wo`, so a back-facing
        // view still scatters; only a zero-length view direction is degenerate
        // (cos_o == 0) and must terminate the path.
        let bsdf = Bsdf::Lambert { albedo: Vec3::ONE };
        let mut rng = Rng::seed(3);
        assert!(bsdf.sample(Vec3::ZERO, N, &mut rng).is_none());
        // A back-facing geometric normal is re-oriented and still yields a valid
        // diffuse sample for a legitimate view direction.
        let wo = Vec3::new(0.0, -1.0, 0.0);
        let back = Vec3::new(0.0, 1.0, 0.0);
        assert!(bsdf.sample(wo, back, &mut rng).is_some());
    }

    #[test]
    fn dielectric_is_specular() {
        let glass = Bsdf::Dielectric {
            ior: 1.5,
            reflectance: Vec3::ONE,
            transmittance: Vec3::ONE,
        };
        assert!(glass.is_specular());
        // Delta lobes never evaluate or report a density for an arbitrary pair.
        assert_eq!(glass.evaluate(N, N, N), Vec3::ZERO);
        assert!(glass.pdf(N, N, N) < 1e-9);
    }

    #[test]
    fn dielectric_normal_incidence_mostly_transmits() {
        // At normal incidence only ~4% of air->glass energy reflects, so the
        // overwhelming majority of samples refract straight through.
        let glass = Bsdf::Dielectric {
            ior: 1.5,
            reflectance: Vec3::ONE,
            transmittance: Vec3::ONE,
        };
        let mut rng = Rng::seed(321);
        let wo = N;
        let count = 200_000u32;
        let mut transmitted = 0u32;
        for _ in 0..count {
            let s = glass.sample(wo, N, &mut rng).expect("glass always samples");
            assert!(s.specular);
            if s.direction.y < 0.0 {
                transmitted += 1;
            }
        }
        let frac = f64::from(transmitted) / f64::from(count);
        assert!(
            (frac - 0.96).abs() < 1e-2,
            "transmitted fraction {frac} should be ~0.96 at normal incidence"
        );
    }

    #[test]
    fn dielectric_total_internal_reflection_never_transmits() {
        // From inside glass (ior reversed) a shallow view angle is past the
        // critical angle, so every sample reflects back into the medium.
        let glass = Bsdf::Dielectric {
            ior: 1.5,
            reflectance: Vec3::ONE,
            transmittance: Vec3::ONE,
        };
        let mut rng = Rng::seed(654);
        // View from below the surface at a shallow angle (inside the medium).
        let wo = Vec3::new(0.95, -0.3122499, 0.0).normalize_or_zero();
        for _ in 0..50_000 {
            let s = glass.sample(wo, N, &mut rng).expect("TIR still reflects");
            // Reflection stays on the incident (below-surface) side.
            assert!(
                s.direction.y < 0.0,
                "TIR must reflect, got {:?}",
                s.direction
            );
            assert!(s.specular);
        }
    }

    #[test]
    fn dielectric_reflection_carries_reflectance_tint() {
        // Force the reflection branch via a grazing view (high Fresnel) and a
        // distinct tint, then confirm the folded delta weight equals the tint.
        let tint = Vec3::new(0.8, 0.6, 0.4);
        let glass = Bsdf::Dielectric {
            ior: 1.5,
            reflectance: tint,
            transmittance: Vec3::ONE,
        };
        let mut rng = Rng::seed(42);
        let wo = Vec3::new(0.9998, 0.02, 0.0).normalize_or_zero();
        let mut saw_reflection = false;
        for _ in 0..5_000 {
            let s = glass.sample(wo, N, &mut rng).expect("sample");
            if s.direction.y > 0.0 {
                // Reflected lobe: weight value*cos/pdf must equal the tint.
                let cos_i = N.dot(s.direction);
                let weight = s.value.scale(cos_i / s.pdf);
                assert!((weight.x - tint.x).abs() < 1e-5);
                assert!((weight.y - tint.y).abs() < 1e-5);
                assert!((weight.z - tint.z).abs() < 1e-5);
                saw_reflection = true;
                break;
            }
        }
        assert!(
            saw_reflection,
            "a grazing view should reflect at least once"
        );
    }

    #[test]
    fn dielectric_transmission_weight_includes_radiance_compression() {
        // Entering a denser medium at normal incidence, the transmitted
        // radiance weight is transmittance * eta^2 / cos_i; with eta = 1/1.5
        // and cos_i = 1 that is below one (radiance is compressed into a
        // narrower solid angle).
        let tint = Vec3::splat(0.9);
        let glass = Bsdf::Dielectric {
            ior: 1.5,
            reflectance: Vec3::ONE,
            transmittance: tint,
        };
        let mut rng = Rng::seed(7);
        let wo = N;
        let mut saw_transmission = false;
        for _ in 0..5_000 {
            let s = glass.sample(wo, N, &mut rng).expect("sample");
            if s.direction.y < 0.0 {
                let cos_i = N.negate().dot(s.direction);
                let weight = s.value.scale(cos_i / s.pdf);
                let eta = 1.0f32 / 1.5;
                let expected = tint.x * eta * eta;
                assert!(
                    (weight.x - expected).abs() < 1e-5,
                    "transmission weight {} should equal {expected}",
                    weight.x
                );
                assert!(weight.x < tint.x, "radiance compression must dim the tint");
                saw_transmission = true;
                break;
            }
        }
        assert!(saw_transmission, "normal incidence should transmit");
    }

    #[test]
    fn ggx_conductor_is_not_specular() {
        let bsdf = Bsdf::GgxConductor {
            reflectance: Vec3::splat(0.9),
            roughness: 0.3,
        };
        assert!(!bsdf.is_specular());
    }

    #[test]
    fn ggx_evaluate_zero_below_surface() {
        let bsdf = Bsdf::GgxConductor {
            reflectance: Vec3::ONE,
            roughness: 0.4,
        };
        let below = Vec3::new(0.0, -1.0, 0.0);
        assert_eq!(bsdf.evaluate(N, below, N), Vec3::ZERO);
        assert_eq!(bsdf.evaluate(below, N, N), Vec3::ZERO);
    }

    #[test]
    fn ggx_sample_pdf_matches_reported_pdf() {
        let bsdf = Bsdf::GgxConductor {
            reflectance: Vec3::splat(0.95),
            roughness: 0.35,
        };
        let mut rng = Rng::seed(4242);
        let wo = Vec3::new(0.4, 1.0, 0.1).normalize_or_zero();
        let mut checked = 0u32;
        for _ in 0..20_000 {
            // Some samples reflect below the horizon and are terminated (`None`);
            // only the valid glossy samples are checked for pdf consistency.
            let Some(s) = bsdf.sample(wo, N, &mut rng) else {
                continue;
            };
            checked += 1;
            let pdf = bsdf.pdf(wo, s.direction, N);
            assert!(
                (pdf - s.pdf).abs() <= 1e-4 * s.pdf.max(1.0),
                "pdf mismatch {pdf} vs {}",
                s.pdf
            );
            assert!(s.direction.is_finite());
            assert!(!s.specular);
        }
        assert!(checked > 1_000, "too few valid glossy samples ({checked})");
    }

    #[test]
    fn ggx_white_furnace_weight_is_masking_ratio() {
        // With F0 = 1 the Monte Carlo throughput value*cos/pdf collapses to the
        // Smith ratio G2(wo, wi) / G1(wo), which is in (0, 1]: single-scatter
        // GGX conserves or loses energy but never creates it.
        let bsdf = Bsdf::GgxConductor {
            reflectance: Vec3::ONE,
            roughness: 0.5,
        };
        let mut rng = Rng::seed(2024);
        let wo = Vec3::new(0.3, 1.0, 0.0).normalize_or_zero();
        let count = 200_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            // Below-horizon reflections terminate the path and contribute zero,
            // which is exactly the single-scatter energy deficit.
            let Some(s) = bsdf.sample(wo, N, &mut rng) else {
                continue;
            };
            let cos_i = N.dot(s.direction);
            let weight = s.value.x * cos_i / s.pdf;
            assert!(
                weight <= 1.0 + 1e-3,
                "single-scatter weight {weight} must not exceed 1"
            );
            assert!(weight >= 0.0);
            sum += f64::from(weight);
        }
        let reflectance = sum / f64::from(count);
        // Directional-hemispherical reflectance of a white rough conductor is
        // below one (energy lost to multiple scattering is not re-added here).
        assert!(
            reflectance > 0.0 && reflectance < 1.0,
            "white furnace reflectance {reflectance} should be in (0, 1)"
        );
    }

    #[test]
    fn ggx_smoother_conductor_reflects_more_energy() {
        // Less roughness means less masking-shadowing loss, so the single-scatter
        // directional-hemispherical reflectance of a white conductor rises as the
        // surface gets smoother.
        fn reflectance(roughness: f32) -> f64 {
            let bsdf = Bsdf::GgxConductor {
                reflectance: Vec3::ONE,
                roughness,
            };
            let mut rng = Rng::seed(999 + (roughness * 1000.0) as u64);
            let wo = Vec3::new(0.5, 1.0, 0.0).normalize_or_zero();
            let count = 200_000u32;
            let mut sum = 0.0f64;
            for _ in 0..count {
                let Some(s) = bsdf.sample(wo, N, &mut rng) else {
                    continue;
                };
                let cos_i = N.dot(s.direction);
                sum += f64::from(s.value.x * cos_i / s.pdf);
            }
            sum / f64::from(count)
        }
        let rough = reflectance(0.6);
        let smooth = reflectance(0.1);
        assert!(
            smooth > rough,
            "smoother reflectance {smooth} should exceed rougher {rough}"
        );
    }

    #[test]
    fn ggx_colored_fresnel_tints_reflection() {
        // A copper-like F0 must preserve its spectral tint in the sampled value.
        let f0 = Vec3::new(0.95, 0.64, 0.54);
        let bsdf = Bsdf::GgxConductor {
            reflectance: f0,
            roughness: 0.2,
        };
        let mut rng = Rng::seed(77);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        let mut sample = None;
        for _ in 0..1_000 {
            if let Some(s) = bsdf.sample(wo, N, &mut rng) {
                sample = Some(s);
                break;
            }
        }
        let s = sample.expect("a smooth conductor must yield a valid sample");
        // Near normal incidence the Fresnel tint keeps red brightest, blue dimmest.
        assert!(s.value.x > s.value.y && s.value.y > s.value.z);
    }
}
