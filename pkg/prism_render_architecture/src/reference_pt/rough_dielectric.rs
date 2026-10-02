//! Rough dielectric scattering (microfacet refraction) for the reference tracer.
//!
//! A rough dielectric is frosted glass: the same reflect-or-transmit split as a
//! smooth dielectric (see [`crate::reference_pt::dielectric`]), but the two
//! lobes are blurred by a GGX microfacet distribution instead of collapsing to
//! Dirac deltas. This is the microfacet model of Walter et al.
//! ("Microfacet Models for Refraction through Rough Surfaces", `EGSR` 2007) in
//! the generalized half-vector form used by `PBRT` v4's `DielectricBxDF`.
//!
//! Three physical pieces drive each lobe:
//!
//! - the GGX normal distribution `D(wm)` of microfacet normals (shared with the
//!   conductor path in [`crate::reference_pt::microfacet`]);
//! - the height-correlated Smith masking-shadowing `G2`;
//! - the unpolarized Fresnel reflectance `F`, which splits incident energy
//!   between the reflected and transmitted microfacet lobes.
//!
//! The reflected lobe uses the ordinary half vector `wm ∝ wi + wo`. The
//! transmitted lobe uses the *generalized* half vector `wm ∝ eta*wi + wo`,
//! whose change-of-variables Jacobian carries the characteristic
//! `1 / (eta*wi·wm + wo·wm)^2` compression. Radiance transport across the
//! interface is additionally scaled by `1 / eta^2` (the `PBRT` radiance-mode
//! solid-angle compression).
//!
//! All computation uses a fixed shading frame whose `+z` axis is the geometric
//! `normal` passed by the caller, exactly as `PBRT` does, so the signed cosines
//! are read straight off the dot products. Only `sqrt`/`abs`/`clamp` and
//! squares are used, honouring the crate's determinism policy (no
//! `sin`/`cos`/`exp`).

use super::microfacet::GgxIsotropic;
use super::sampler::SampleSource;
use super::{Vec3, EPS_LEN_SQ};

/// A direction cosine below this magnitude is treated as a grazing degeneracy
/// and short-circuits to zero, keeping the microfacet denominators finite.
const COS_EPS: f32 = 1.0e-8;

/// Returns `x` squared, spelled out so the banned `powi`/`powf` are avoided.
fn sqr(x: f32) -> f32 {
    x * x
}

/// The unpolarized Fresnel reflectance of a dielectric interface in `PBRT` v4's
/// relative-index form (`FrDielectric`).
///
/// `cos_i` is the signed cosine between the incident direction and the
/// microfacet normal and `eta` is the relative index `eta_t / eta_i`
/// (transmitted over incident). A negative `cos_i` means the ray strikes the
/// back of the microfacet, so the index ratio is inverted and the cosine
/// flipped before the standard formula is applied. Beyond the critical angle
/// (total internal reflection, `TIR`) the function returns `1`.
pub(super) fn fr_dielectric(cos_i: f32, eta: f32) -> f32 {
    let mut cos_i = cos_i.clamp(-1.0, 1.0);
    let mut eta = eta;
    if cos_i < 0.0 {
        eta = 1.0 / eta;
        cos_i = -cos_i;
    }
    // Snell's law in squared-sine form: sin^2(t) = sin^2(i) / eta^2.
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = sin2_i / (eta * eta);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    // Parallel- and perpendicular-polarized amplitude reflection coefficients.
    let r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Refracts the direction `wi` across a microfacet with normal `n` and relative
/// index `eta` (`eta_t / eta_i`), following `PBRT` v4's `Refract`.
///
/// Returns the unit transmitted direction together with the possibly-inverted
/// relative index `etap` actually used (needed by the transmission Jacobian),
/// or [`None`] under total internal reflection (`TIR`). A back-facing incidence
/// (`n·wi < 0`) inverts the index ratio and flips the normal so the refraction
/// is always computed from the incident side.
pub(super) fn refract_through(wi: Vec3, n: Vec3, eta: f32) -> Option<(Vec3, f32)> {
    let mut cos_i = n.dot(wi);
    let mut eta = eta;
    let mut n = n;
    if cos_i < 0.0 {
        eta = 1.0 / eta;
        cos_i = -cos_i;
        n = n.negate();
    }
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = sin2_i / (eta * eta);
    if sin2_t >= 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    // wt = -wi / eta + (cos_i / eta - cos_t) n, pointing to the far side of n.
    let wt = wi
        .negate()
        .scale(1.0 / eta)
        .add(n.scale(cos_i / eta - cos_t));
    let wt = wt.normalize_or_zero();
    if wt.length_squared() <= EPS_LEN_SQ {
        return None;
    }
    Some((wt, eta))
}

/// The outcome of importance-sampling a [`RoughDielectric`] lobe.
///
/// Mirrors the shape of [`crate::reference_pt::bsdf::BsdfSample`] without the
/// `specular` flag (a rough dielectric is always glossy), so the enclosing
/// [`crate::reference_pt::bsdf::Bsdf`] can forward it with `specular: false`.
#[derive(Clone, Copy, Debug)]
pub struct RoughDielectricSample {
    /// The sampled outgoing direction (away from the surface), unit length.
    pub direction: Vec3,
    /// The full `BSDF` value `f(wo, wi)` for the sampled pair (reflection
    /// `BRDF` or transmission `BTDF`), carrying the Fresnel split so the
    /// integrator's generic `value * |cos_i| / pdf` throughput update is
    /// unbiased.
    pub value: Vec3,
    /// The solid-angle probability density of `direction`, consistent with
    /// [`RoughDielectric::pdf`].
    pub pdf: f32,
}

/// A rough dielectric interface: a GGX-blurred reflect-and-refract surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoughDielectric {
    /// Relative index of refraction `eta_t / eta_i` of the interior over the
    /// exterior (e.g. `1.5` for air-to-glass).
    ior: f32,
    /// Per-channel tint applied to the reflected microfacet lobe.
    reflectance: Vec3,
    /// Per-channel tint applied to the transmitted microfacet lobe.
    transmittance: Vec3,
    /// The shared isotropic GGX microfacet distribution.
    ggx: GgxIsotropic,
}

impl RoughDielectric {
    /// Builds a rough dielectric from its relative `ior`, per-channel reflected
    /// and transmitted tints, and the perceptual `roughness` in `[0, 1]`
    /// (remapped to the GGX width `alpha = roughness^2`, clamped to a small
    /// floor so the lobe never becomes a true Dirac delta).
    #[must_use]
    pub fn new(ior: f32, reflectance: Vec3, transmittance: Vec3, roughness: f32) -> Self {
        Self {
            ior,
            reflectance,
            transmittance,
            ggx: GgxIsotropic::from_roughness(roughness),
        }
    }

    /// The visible-normal (`VNDF`) solid-angle density of the microfacet normal
    /// `wm` for the view direction `wo`, in the frame whose `+z` is `normal`.
    ///
    /// This is `D(wm) * G1(wo) * |wo·wm| / |n·wo|`, i.e. the density the GGX
    /// visible-normal sampler draws half vectors from, before the reflect or
    /// refract change of variables is applied.
    fn visible_normal_pdf(&self, wo: Vec3, wm: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        if cos_o.abs() < COS_EPS {
            return 0.0;
        }
        let d = self.ggx.distribution(normal.dot(wm).abs());
        let g1 = self.ggx.g1(cos_o);
        d * g1 * wo.dot(wm).abs() / cos_o.abs()
    }

    /// Evaluates the full `BSDF` value `f(wo, wi)` for a fixed direction pair.
    ///
    /// Reflection (`wo`, `wi` on the same side of `normal`) uses the ordinary
    /// half vector; transmission (opposite sides) uses the generalized half
    /// vector `eta*wi + wo`. Returns [`Vec3::ZERO`] for grazing degeneracies
    /// and for direction pairs that fall behind the chosen microfacet.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o.abs() < COS_EPS || cos_i.abs() < COS_EPS {
            return Vec3::ZERO;
        }
        let reflect = cos_i * cos_o > 0.0;
        // Relative index on the generalized half-vector side.
        let etap = if reflect {
            1.0
        } else if cos_o > 0.0 {
            self.ior
        } else {
            1.0 / self.ior
        };
        let wm = wi.scale(etap).add(wo);
        if wm.length_squared() <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        // Face the half vector toward the macroscopic normal (the `+z` axis).
        let wm = wm.normalize_or_zero().faced_toward(normal);
        // Discard direction pairs that lie behind the chosen microfacet.
        if wm.dot(wi) * cos_i < 0.0 || wm.dot(wo) * cos_o < 0.0 {
            return Vec3::ZERO;
        }
        let d = self.ggx.distribution(normal.dot(wm).abs());
        let g2 = self.ggx.g2(cos_o, cos_i);
        let f = fr_dielectric(wo.dot(wm), self.ior);
        if reflect {
            let denom = 4.0 * (cos_i * cos_o).abs();
            if denom <= 0.0 {
                return Vec3::ZERO;
            }
            self.reflectance.scale(d * g2 * f / denom)
        } else {
            let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap) * cos_i * cos_o;
            if denom.abs() < COS_EPS {
                return Vec3::ZERO;
            }
            // Transmission `BTDF` with the radiance-mode 1/eta^2 compression.
            let ft = d * (1.0 - f) * g2 * (wi.dot(wm) * wo.dot(wm) / denom).abs() / sqr(etap);
            self.transmittance.scale(ft)
        }
    }

    /// The solid-angle density [`Self::sample`] would assign to `(wo, wi)`.
    ///
    /// Combines the microfacet `VNDF` density with the reflect/refract Jacobian
    /// and the Fresnel-proportional lobe-selection probability, so it is a
    /// proper mixture density. Returns zero for grazing or back-facing pairs.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let cos_o = normal.dot(wo);
        let cos_i = normal.dot(wi);
        if cos_o.abs() < COS_EPS || cos_i.abs() < COS_EPS {
            return 0.0;
        }
        let reflect = cos_i * cos_o > 0.0;
        let etap = if reflect {
            1.0
        } else if cos_o > 0.0 {
            self.ior
        } else {
            1.0 / self.ior
        };
        let wm = wi.scale(etap).add(wo);
        if wm.length_squared() <= EPS_LEN_SQ {
            return 0.0;
        }
        let wm = wm.normalize_or_zero().faced_toward(normal);
        if wm.dot(wi) * cos_i < 0.0 || wm.dot(wo) * cos_o < 0.0 {
            return 0.0;
        }
        let f = fr_dielectric(wo.dot(wm), self.ior);
        let pr = f;
        let pt = 1.0 - f;
        if pr + pt <= 0.0 {
            return 0.0;
        }
        let vndf = self.visible_normal_pdf(wo, wm, normal);
        if reflect {
            let woh = wo.dot(wm).abs();
            if woh < COS_EPS {
                return 0.0;
            }
            vndf / (4.0 * woh) * pr / (pr + pt)
        } else {
            let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);
            if denom <= 0.0 {
                return 0.0;
            }
            let dwm_dwi = wi.dot(wm).abs() / denom;
            vndf * dwm_dwi * pt / (pr + pt)
        }
    }

    /// Importance-samples an outgoing direction for the view direction `wo` and
    /// the fixed shading `normal` (the frame `+z`).
    ///
    /// A microfacet normal is drawn by the GGX visible-normal sampler, faced to
    /// the hemisphere of `wo`; the reflected or transmitted lobe is then chosen
    /// in proportion to the microfacet Fresnel reflectance. Returns [`None`]
    /// when the sample is degenerate (grazing, zero-length, or landing in the
    /// wrong hemisphere), so the caller terminates the path.
    #[must_use]
    pub fn sample(
        &self,
        wo: Vec3,
        normal: Vec3,
        rng: &mut impl SampleSource,
    ) -> Option<RoughDielectricSample> {
        let cos_o = normal.dot(wo);
        if cos_o.abs() < COS_EPS {
            return None;
        }
        // Sample a microfacet normal faced into the hemisphere of `wo`.
        let nf = normal.faced_toward(wo);
        let wm = self.ggx.sample_half_vector(wo, nf, rng)?;
        let f = fr_dielectric(wo.dot(wm), self.ior);
        let pr = f;
        let pt = 1.0 - f;
        if pr + pt <= 0.0 {
            return None;
        }
        if rng.next_f32() < pr / (pr + pt) {
            // Reflected lobe: mirror `wo` about the microfacet normal.
            let wi = wo.negate().reflect(wm).normalize_or_zero();
            let cos_i = normal.dot(wi);
            // The reflected direction must stay on `wo`'s side of the surface.
            if cos_i * cos_o <= 0.0 {
                return None;
            }
            let woh = wo.dot(wm).abs();
            if woh < COS_EPS {
                return None;
            }
            let vndf = self.visible_normal_pdf(wo, wm, normal);
            let pdf = vndf / (4.0 * woh) * pr / (pr + pt);
            if pdf <= 0.0 {
                return None;
            }
            let d = self.ggx.distribution(normal.dot(wm).abs());
            let g2 = self.ggx.g2(cos_o, cos_i);
            let denom = 4.0 * (cos_i * cos_o).abs();
            if denom <= 0.0 {
                return None;
            }
            Some(RoughDielectricSample {
                direction: wi,
                value: self.reflectance.scale(d * g2 * f / denom),
                pdf,
            })
        } else {
            // Transmitted lobe: refract `wo` through the microfacet normal.
            let (wi, etap) = refract_through(wo, wm, self.ior)?;
            let cos_i = normal.dot(wi);
            // The transmitted direction must cross to the far side of the surface.
            if cos_i * cos_o >= 0.0 {
                return None;
            }
            let denom_j = sqr(wi.dot(wm) + wo.dot(wm) / etap);
            if denom_j <= 0.0 {
                return None;
            }
            let dwm_dwi = wi.dot(wm).abs() / denom_j;
            let vndf = self.visible_normal_pdf(wo, wm, normal);
            let pdf = vndf * dwm_dwi * pt / (pr + pt);
            if pdf <= 0.0 {
                return None;
            }
            let d = self.ggx.distribution(normal.dot(wm).abs());
            let g2 = self.ggx.g2(cos_o, cos_i);
            let denom = denom_j * cos_i * cos_o;
            if denom.abs() < COS_EPS {
                return None;
            }
            let ft = d * (1.0 - f) * g2 * (wi.dot(wm) * wo.dot(wm) / denom).abs() / sqr(etap);
            Some(RoughDielectricSample {
                direction: wi,
                value: self.transmittance.scale(ft),
                pdf,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::sampler::Rng;
    use super::*;

    /// The macroscopic surface normal used throughout the tests (`+y`).
    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Builds a clear (untinted, non-absorbing) glass of the given roughness.
    fn clear_glass(roughness: f32) -> RoughDielectric {
        RoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, roughness)
    }

    #[test]
    fn fresnel_relative_matches_known_endpoints() {
        // Normal incidence air->glass (eta = 1.5): R = ((1.5-1)/(1.5+1))^2 = 0.04.
        let r = fr_dielectric(1.0, 1.5);
        assert!((r - 0.04).abs() < 1e-3, "normal-incidence R = {r}");
        // A back-facing incidence inverts eta: glass->air at normal incidence is
        // the same 0.04 (reciprocity of the interface).
        let r_back = fr_dielectric(-1.0, 1.5);
        assert!((r_back - 0.04).abs() < 1e-3, "back-face R = {r_back}");
    }

    #[test]
    fn total_internal_reflection_is_full() {
        // Dense (1.5) to rare (1.0): eta_rel for the internal side is 1/1.5.
        // A shallow internal angle exceeds the critical angle and reflects all.
        let r = fr_dielectric(0.1, 1.0 / 1.5);
        assert!((r - 1.0).abs() < 1e-6, "TIR reflectance = {r}");
        // The refraction routine agrees: it reports no transmitted direction.
        let wo = Vec3::new(0.98, 0.2, 0.0).normalize_or_zero();
        assert!(
            refract_through(wo, N, 1.0 / 1.5).is_none(),
            "shallow internal ray must be in TIR"
        );
    }

    #[test]
    fn sample_pdf_and_value_are_self_consistent() {
        // For a glossy lobe the sampled `pdf`/`value` must equal the independent
        // `pdf`/`evaluate` for the same pair (catches Jacobian/sign bugs).
        let rd = clear_glass(0.35);
        let mut rng = Rng::seed(0xC0FFEE);
        let wo = Vec3::new(0.3, 0.85, 0.1).normalize_or_zero();
        let mut checked = 0u32;
        for _ in 0..40_000 {
            let Some(s) = rd.sample(wo, N, &mut rng) else {
                continue;
            };
            let p = rd.pdf(wo, s.direction, N);
            assert!(
                (p - s.pdf).abs() <= 1e-3 * s.pdf.max(1.0) + 1e-5,
                "pdf mismatch: sample {} vs pdf {p}",
                s.pdf
            );
            let v = rd.evaluate(wo, s.direction, N);
            let diff = v.sub(s.value).length();
            let scale = s.value.length().max(1.0);
            assert!(
                diff <= 1e-3 * scale + 1e-5,
                "value mismatch: sample {:?} vs evaluate {:?}",
                s.value.to_array(),
                v.to_array()
            );
            checked += 1;
        }
        assert!(checked > 1000, "expected many valid samples, got {checked}");
    }

    #[test]
    fn produces_both_reflection_and_transmission() {
        // A clear glass must both reflect and transmit over many samples.
        let rd = clear_glass(0.3);
        let mut rng = Rng::seed(7);
        let wo = Vec3::new(0.2, 0.9, 0.0).normalize_or_zero();
        let mut reflected = 0u32;
        let mut transmitted = 0u32;
        for _ in 0..20_000 {
            if let Some(s) = rd.sample(wo, N, &mut rng) {
                if N.dot(s.direction) > 0.0 {
                    reflected += 1;
                } else {
                    transmitted += 1;
                }
            }
        }
        assert!(reflected > 100, "expected reflections, got {reflected}");
        assert!(
            transmitted > 100,
            "expected transmissions, got {transmitted}"
        );
    }

    #[test]
    fn microfacet_masking_never_adds_energy() {
        // The single-scatter throughput weight `value * |cos_i| / pdf`, averaged
        // over the sampler, is the directional albedo. For a lossless glass the
        // physically correct (radiance-mode) ceiling is `R + (1 - R) / eta^2`;
        // the Smith masking term only ever removes energy, so the measured
        // albedo must stay at or below that ceiling at every roughness and
        // decrease monotonically as the surface roughens.
        let ior = 1.5_f32;
        let wo = Vec3::new(0.1, 0.99, 0.0).normalize_or_zero();
        let cos_o = N.dot(wo);
        let r = f64::from(fr_dielectric(cos_o, ior));
        let ceiling = r + (1.0 - r) / f64::from(ior * ior);
        let mut previous = f64::INFINITY;
        for &roughness in &[0.05_f32, 0.3_f32, 0.6_f32] {
            let rd = RoughDielectric::new(ior, Vec3::ONE, Vec3::ONE, roughness);
            let mut rng = Rng::seed(0x5EED ^ u64::from(roughness.to_bits()));
            let count = 60_000u32;
            let mut sum = 0.0f64;
            for _ in 0..count {
                if let Some(s) = rd.sample(wo, N, &mut rng) {
                    let cos_i = N.dot(s.direction).abs();
                    let w = s.value.scale(cos_i / s.pdf);
                    sum += f64::from((w.x + w.y + w.z) / 3.0);
                }
            }
            let albedo = sum / f64::from(count);
            assert!(
                albedo <= ceiling + 2e-2,
                "roughness {roughness}: albedo {albedo} exceeds lossless ceiling {ceiling}"
            );
            // Rougher surfaces mask more energy (allow Monte Carlo slack).
            assert!(
                albedo <= previous + 2e-2,
                "roughness {roughness}: albedo {albedo} rose above the smoother {previous}"
            );
            previous = albedo;
        }
    }

    #[test]
    fn radiance_mode_transmission_compresses_by_eta_squared() {
        // Camera-side radiance transport scales the transmitted lobe by 1/eta^2,
        // so the measured directional albedo of a near-smooth glass is the
        // analytic `R + (1 - R) / eta^2`, not unity. This pins down the
        // radiance-mode factor and the Fresnel split together.
        let ior = 1.5_f32;
        let rd = RoughDielectric::new(ior, Vec3::ONE, Vec3::ONE, 0.03);
        let mut rng = Rng::seed(0xABCD);
        let wo = Vec3::new(0.1, 0.99, 0.0).normalize_or_zero();
        let cos_o = N.dot(wo);
        let r = f64::from(fr_dielectric(cos_o, ior));
        let predicted = r + (1.0 - r) / f64::from(ior * ior);
        let count = 80_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            if let Some(s) = rd.sample(wo, N, &mut rng) {
                let cos_i = N.dot(s.direction).abs();
                let w = s.value.scale(cos_i / s.pdf);
                sum += f64::from((w.x + w.y + w.z) / 3.0);
            }
        }
        let albedo = sum / f64::from(count);
        assert!(
            (albedo - predicted).abs() < 3e-2,
            "albedo {albedo} should match radiance-mode prediction {predicted}"
        );
    }

    #[test]
    fn evaluate_and_pdf_vanish_for_grazing() {
        let rd = clear_glass(0.4);
        let grazing = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(rd.evaluate(grazing, N, N), Vec3::ZERO);
        assert!(rd.pdf(grazing, N, N) <= 0.0);
    }
}
