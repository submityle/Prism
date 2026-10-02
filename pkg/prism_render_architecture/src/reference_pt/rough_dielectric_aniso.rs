//! Anisotropic rough dielectric scattering (microfacet refraction with a grain).
//!
//! [`crate::reference_pt::rough_dielectric`] blurs the reflect-and-refract split
//! of a smooth dielectric with an *isotropic* `GGX` lobe, so its frosted
//! highlight and its blurred transmission are both rotationally symmetric about
//! the normal. Real brushed or drawn transparent media — stretched acrylic,
//! satin-etched glass, the fine grooves on a machined light pipe — instead have
//! a *grain* that elongates both the reflected and the transmitted lobe into a
//! streak. This module reuses the anisotropic `GGX` distribution of
//! [`crate::reference_pt::microfacet_aniso`] for both lobes, giving the offline
//! reference tracer a brushed-glass oracle the real-time path can be measured
//! against.
//!
//! The physics is identical to the isotropic rough dielectric — the same
//! generalized half vector `wm` for transmission, the same radiance-mode
//! `1 / eta^2` compression, the same unpolarized Fresnel split (both reused
//! verbatim from the sibling module so there is a single source of truth) — but
//! the microfacet distribution `D`, the Smith masking `G1`/`G2`, and the
//! visible-normal (`VNDF`) density are evaluated in a local shading frame with
//! independent widths `alpha_x`/`alpha_y` along the tangent and bitangent axes.
//!
//! Every direction and half vector is projected into that local frame, whose
//! `+z` axis is the macroscopic normal and whose `x`/`y` axes carry the grain.
//! Because dot products and cosines are invariant under the orthonormal
//! change of basis, every Jacobian (the reflection `1 / (4 wo·wm)`, the
//! transmission `|wi·wm| / (wi·wm + wo·wm/etap)^2`, and the `1 / eta^2`
//! compression) is numerically identical to the isotropic module; only the
//! lobe shape and its masking become elliptical. When
//! `alpha_x == alpha_y` every result collapses exactly to
//! [`crate::reference_pt::rough_dielectric::RoughDielectric`].
//!
//! The anisotropy axes are bound to the geometric tangent basis built from the
//! shading normal, so this oracle validates the `BRDF`/`BTDF` mathematics
//! itself, independently of any mesh `UV` parameterisation, exactly as the
//! conductor twin in [`crate::reference_pt::conductor_aniso`] does. Only
//! `sqrt`/`abs`/`clamp` and squares appear, honouring the crate's determinism
//! policy.

use super::microfacet_aniso::GgxAnisotropic;
use super::rough_dielectric::{fr_dielectric, refract_through, RoughDielectricSample};
use super::sampler::{orthonormal_basis, Rng};
use super::{Vec3, EPS_LEN_SQ};

/// A direction cosine below this magnitude is treated as a grazing degeneracy
/// and short-circuits to zero, keeping the microfacet denominators finite.
const COS_EPS: f32 = 1.0e-8;

/// Returns `x` squared, spelled out so the banned `powi`/`powf` are avoided.
fn sqr(x: f32) -> f32 {
    x * x
}

/// A local-frame triad `(tangent, bitangent, normal)` built from the shading
/// normal, used to project directions into the frame the anisotropic lobe lives
/// in.
///
/// Mirrors the frame of [`crate::reference_pt::conductor_aniso`]; it is kept
/// local to this module so the two oracles stay independently testable.
struct LocalFrame {
    /// The local `x` axis (tangent, carrying `alpha_x`).
    tangent: Vec3,
    /// The local `y` axis (bitangent, carrying `alpha_y`).
    bitangent: Vec3,
    /// The local `z` axis (the macroscopic shading normal).
    normal: Vec3,
}

impl LocalFrame {
    /// Builds the shading frame around `normal`.
    fn new(normal: Vec3) -> Self {
        let (tangent, bitangent) = orthonormal_basis(normal);
        Self {
            tangent,
            bitangent,
            normal,
        }
    }

    /// Projects a world-space vector into the local shading frame.
    fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            v.dot(self.tangent),
            v.dot(self.bitangent),
            v.dot(self.normal),
        )
    }

    /// Lifts a local-frame vector back into world space.
    fn to_world(&self, v: Vec3) -> Vec3 {
        self.tangent
            .scale(v.x)
            .add(self.bitangent.scale(v.y))
            .add(self.normal.scale(v.z))
    }
}

/// A rough anisotropic dielectric interface (brushed/drawn frosted glass).
///
/// Reflected and transmitted microfacet lobes share one elliptical `GGX`
/// distribution; the reflect-or-transmit split and the transmission Jacobian
/// are identical to the isotropic sibling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisoRoughDielectric {
    /// Relative index of refraction `eta_t / eta_i` of the interior medium over
    /// the exterior (e.g. `1.5` for air-to-glass).
    ior: f32,
    /// Per-channel tint applied to the reflected microfacet lobe.
    reflectance: Vec3,
    /// Per-channel tint applied to the transmitted microfacet lobe.
    transmittance: Vec3,
    /// The anisotropic `GGX` microfacet distribution shared by both lobes.
    dist: GgxAnisotropic,
}

impl AnisoRoughDielectric {
    /// Builds an anisotropic rough dielectric from its relative `ior`,
    /// per-channel reflected and transmitted tints, and explicit `GGX` widths
    /// `alpha_x`/`alpha_y` (each clamped to the lobe's minimum width).
    #[must_use]
    pub fn new(
        ior: f32,
        reflectance: Vec3,
        transmittance: Vec3,
        alpha_x: f32,
        alpha_y: f32,
    ) -> Self {
        Self {
            ior,
            reflectance,
            transmittance,
            dist: GgxAnisotropic::new(alpha_x, alpha_y),
        }
    }

    /// Builds an anisotropic rough dielectric from a perceptual `roughness` and
    /// an `anisotropy`, both in `[0, 1]`, using the Disney / `UE` (Burley)
    /// remap in [`GgxAnisotropic::from_roughness_anisotropy`].
    #[must_use]
    pub fn from_roughness_anisotropy(
        ior: f32,
        reflectance: Vec3,
        transmittance: Vec3,
        roughness: f32,
        anisotropy: f32,
    ) -> Self {
        Self {
            ior,
            reflectance,
            transmittance,
            dist: GgxAnisotropic::from_roughness_anisotropy(roughness, anisotropy),
        }
    }

    /// The visible-normal (`VNDF`) solid-angle density of the local half vector
    /// `wm_local` for the local view direction `wo_local`.
    ///
    /// This is `D(wm) * G1(wo) * |wo·wm| / |wo.z|`, the elliptical analogue of
    /// the isotropic sibling's density, before the reflect or refract change of
    /// variables is applied. `wm_local` must already be faced into the upper
    /// hemisphere so the anisotropic `D` does not short-circuit to zero.
    fn visible_normal_pdf(&self, wo_local: Vec3, wm_local: Vec3) -> f32 {
        let cos_o = wo_local.z;
        if cos_o.abs() < COS_EPS {
            return 0.0;
        }
        let d = self.dist.distribution(wm_local);
        let g1 = self.dist.g1(wo_local);
        d * g1 * wo_local.dot(wm_local).abs() / cos_o.abs()
    }

    /// Evaluates the full `BSDF` value `f(wo, wi)` for a fixed direction pair.
    ///
    /// Reflection (`wo`, `wi` on the same side of `normal`) uses the ordinary
    /// half vector; transmission (opposite sides) uses the generalized half
    /// vector `eta*wi + wo`. Returns [`Vec3::ZERO`] for grazing degeneracies
    /// and for direction pairs that fall behind the chosen microfacet.
    #[must_use]
    pub fn evaluate(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> Vec3 {
        let frame = LocalFrame::new(normal);
        let wo_local = frame.to_local(wo);
        let wi_local = frame.to_local(wi);
        let cos_o = wo_local.z;
        let cos_i = wi_local.z;
        if cos_o.abs() < COS_EPS || cos_i.abs() < COS_EPS {
            return Vec3::ZERO;
        }
        let reflect = cos_i * cos_o > 0.0;
        let etap = if reflect {
            1.0
        } else if cos_o > 0.0 {
            self.ior
        } else {
            1.0 / self.ior
        };
        let wm = wi_local.scale(etap).add(wo_local);
        if wm.length_squared() <= EPS_LEN_SQ {
            return Vec3::ZERO;
        }
        // Face the half vector into the upper hemisphere (`+z`) so the elliptical
        // `D` is evaluated on its supported domain.
        let wm = wm
            .normalize_or_zero()
            .faced_toward(Vec3::new(0.0, 0.0, 1.0));
        // Discard direction pairs that lie behind the chosen microfacet.
        if wm.dot(wi_local) * cos_i < 0.0 || wm.dot(wo_local) * cos_o < 0.0 {
            return Vec3::ZERO;
        }
        let d = self.dist.distribution(wm);
        let g2 = self.dist.g2(wo_local, wi_local);
        let f = fr_dielectric(wo_local.dot(wm), self.ior);
        if reflect {
            let denom = 4.0 * (cos_i * cos_o).abs();
            if denom <= 0.0 {
                return Vec3::ZERO;
            }
            self.reflectance.scale(d * g2 * f / denom)
        } else {
            let denom = sqr(wi_local.dot(wm) + wo_local.dot(wm) / etap) * cos_i * cos_o;
            if denom.abs() < COS_EPS {
                return Vec3::ZERO;
            }
            let ft = d * (1.0 - f) * g2 * (wi_local.dot(wm) * wo_local.dot(wm) / denom).abs()
                / sqr(etap);
            self.transmittance.scale(ft)
        }
    }

    /// The solid-angle density [`Self::sample`] would assign to `(wo, wi)`.
    ///
    /// Combines the anisotropic `VNDF` density with the reflect/refract Jacobian
    /// and the Fresnel-proportional lobe-selection probability, so it is a
    /// proper mixture density. Returns zero for grazing or back-facing pairs.
    #[must_use]
    pub fn pdf(&self, wo: Vec3, wi: Vec3, normal: Vec3) -> f32 {
        let frame = LocalFrame::new(normal);
        let wo_local = frame.to_local(wo);
        let wi_local = frame.to_local(wi);
        let cos_o = wo_local.z;
        let cos_i = wi_local.z;
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
        let wm = wi_local.scale(etap).add(wo_local);
        if wm.length_squared() <= EPS_LEN_SQ {
            return 0.0;
        }
        let wm = wm
            .normalize_or_zero()
            .faced_toward(Vec3::new(0.0, 0.0, 1.0));
        if wm.dot(wi_local) * cos_i < 0.0 || wm.dot(wo_local) * cos_o < 0.0 {
            return 0.0;
        }
        let f = fr_dielectric(wo_local.dot(wm), self.ior);
        let pr = f;
        let pt = 1.0 - f;
        if pr + pt <= 0.0 {
            return 0.0;
        }
        let vndf = self.visible_normal_pdf(wo_local, wm);
        if reflect {
            let woh = wo_local.dot(wm).abs();
            if woh < COS_EPS {
                return 0.0;
            }
            vndf / (4.0 * woh) * pr / (pr + pt)
        } else {
            let denom = sqr(wi_local.dot(wm) + wo_local.dot(wm) / etap);
            if denom <= 0.0 {
                return 0.0;
            }
            let dwm_dwi = wi_local.dot(wm).abs() / denom;
            vndf * dwm_dwi * pt / (pr + pt)
        }
    }

    /// Importance-samples an outgoing direction for the view direction `wo` and
    /// the shading `normal`.
    ///
    /// A microfacet normal is drawn by the anisotropic `GGX` visible-normal
    /// sampler in the view hemisphere; the reflected or transmitted lobe is then
    /// chosen in proportion to the microfacet Fresnel reflectance. Returns
    /// [`None`] when the sample is degenerate (grazing, zero-length, or landing
    /// in the wrong hemisphere), so the caller terminates the path.
    #[must_use]
    pub fn sample(&self, wo: Vec3, normal: Vec3, rng: &mut Rng) -> Option<RoughDielectricSample> {
        // Orient the frame into the view hemisphere so a back-facing hit still
        // scatters; the microfacet-relative Fresnel and refraction remain
        // correct because they key off the sign of `wo·wm`.
        let frame = LocalFrame::new(normal.faced_toward(wo));
        let wo_local = frame.to_local(wo);
        let cos_o = wo_local.z;
        if cos_o < COS_EPS {
            return None;
        }
        let wm = self.dist.sample_half_vector(wo_local, rng)?;
        let f = fr_dielectric(wo_local.dot(wm), self.ior);
        let pr = f;
        let pt = 1.0 - f;
        if pr + pt <= 0.0 {
            return None;
        }
        if rng.next_f32() < pr / (pr + pt) {
            // Reflected lobe: mirror `wo` about the microfacet normal.
            let wi_local = wo_local.negate().reflect(wm).normalize_or_zero();
            let cos_i = wi_local.z;
            if cos_i * cos_o <= 0.0 {
                return None;
            }
            let woh = wo_local.dot(wm).abs();
            if woh < COS_EPS {
                return None;
            }
            let vndf = self.visible_normal_pdf(wo_local, wm);
            let pdf = vndf / (4.0 * woh) * pr / (pr + pt);
            if pdf <= 0.0 {
                return None;
            }
            let d = self.dist.distribution(wm);
            let g2 = self.dist.g2(wo_local, wi_local);
            let denom = 4.0 * (cos_i * cos_o).abs();
            if denom <= 0.0 {
                return None;
            }
            Some(RoughDielectricSample {
                direction: frame.to_world(wi_local),
                value: self.reflectance.scale(d * g2 * f / denom),
                pdf,
            })
        } else {
            // Transmitted lobe: refract `wo` through the microfacet normal.
            let (wi_local, etap) = refract_through(wo_local, wm, self.ior)?;
            let cos_i = wi_local.z;
            if cos_i * cos_o >= 0.0 {
                return None;
            }
            let denom_j = sqr(wi_local.dot(wm) + wo_local.dot(wm) / etap);
            if denom_j <= 0.0 {
                return None;
            }
            let dwm_dwi = wi_local.dot(wm).abs() / denom_j;
            let vndf = self.visible_normal_pdf(wo_local, wm);
            let pdf = vndf * dwm_dwi * pt / (pr + pt);
            if pdf <= 0.0 {
                return None;
            }
            let d = self.dist.distribution(wm);
            let g2 = self.dist.g2(wo_local, wi_local);
            let denom = denom_j * cos_i * cos_o;
            if denom.abs() < COS_EPS {
                return None;
            }
            let ft = d * (1.0 - f) * g2 * (wi_local.dot(wm) * wo_local.dot(wm) / denom).abs()
                / sqr(etap);
            Some(RoughDielectricSample {
                direction: frame.to_world(wi_local),
                value: self.transmittance.scale(ft),
                pdf,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_pt::rough_dielectric::RoughDielectric;

    /// The macroscopic shading normal shared by the direction fixtures.
    const N: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// A unit vector from raw components.
    fn unit(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize_or_zero()
    }

    /// A clear anisotropic glass with unit tints at the given widths.
    fn clear_glass(alpha_x: f32, alpha_y: f32) -> AnisoRoughDielectric {
        AnisoRoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, alpha_x, alpha_y)
    }

    /// With equal widths the anisotropic dielectric must match the isotropic
    /// [`RoughDielectric`] of the same width for both the reflected and the
    /// transmitted lobe.
    #[test]
    fn reduces_to_isotropic_when_widths_match() {
        let roughness = 0.3_f32;
        let alpha = roughness * roughness;
        let aniso = AnisoRoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, alpha, alpha);
        let iso = RoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, roughness);
        let wo = unit(0.3, 0.9, 0.1);
        // Reflected pair (same side as `wo`) and transmitted pair (far side).
        for wi in [unit(-0.2, 0.95, 0.1), unit(0.1, -0.9, 0.15)] {
            let a = aniso.evaluate(wo, wi, N);
            let b = iso.evaluate(wo, wi, N);
            assert!(
                a.sub(b).length() < 1e-3,
                "aniso {:?} vs iso {:?}",
                a.to_array(),
                b.to_array()
            );
            let pa = aniso.pdf(wo, wi, N);
            let pb = iso.pdf(wo, wi, N);
            assert!((pa - pb).abs() < 1e-3, "pdf aniso {pa} vs iso {pb}");
        }
    }

    /// A sampled direction reports the same value and density that
    /// [`AnisoRoughDielectric::evaluate`]/[`AnisoRoughDielectric::pdf`]
    /// recompute for the same pair, over both lobes.
    #[test]
    fn sample_is_consistent_with_evaluate_and_pdf() {
        let rd = clear_glass(0.35, 0.1);
        let wo = unit(0.2, 0.9, 0.0);
        let mut rng = Rng::seed(42);
        let mut checked = 0u32;
        for _ in 0..40_000 {
            let Some(s) = rd.sample(wo, N, &mut rng) else {
                continue;
            };
            let p = rd.pdf(wo, s.direction, N);
            // The sharp lobe amplifies tiny half-vector round-off, so compare
            // the recomputed density relative to its peak.
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

    /// A clear anisotropic glass must both reflect and transmit over many
    /// samples.
    #[test]
    fn produces_both_reflection_and_transmission() {
        let rd = clear_glass(0.3, 0.12);
        let mut rng = Rng::seed(7);
        let wo = unit(0.2, 0.9, 0.0);
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

    /// The single-scatter throughput weight `value * |cos_i| / pdf`, averaged
    /// over the sampler, is the directional albedo. For a lossless glass the
    /// physically correct (radiance-mode) ceiling is `R + (1 - R) / eta^2`; the
    /// Smith masking only ever removes energy, so the measured albedo of an
    /// anisotropic glass must stay at or below that ceiling.
    #[test]
    fn microfacet_masking_never_adds_energy() {
        let ior = 1.5_f32;
        let wo = unit(0.1, 0.99, 0.0);
        let cos_o = N.dot(wo);
        let r = f64::from(fr_dielectric(cos_o, ior));
        let ceiling = r + (1.0 - r) / f64::from(ior * ior);
        for &(ax, ay) in &[(0.5_f32, 0.1_f32), (0.2_f32, 0.5_f32), (0.3_f32, 0.3_f32)] {
            let rd = AnisoRoughDielectric::new(ior, Vec3::ONE, Vec3::ONE, ax, ay);
            let mut rng = Rng::seed(0x5EED ^ u64::from(ax.to_bits()) ^ u64::from(ay.to_bits()));
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
                albedo <= ceiling + 2e-2,
                "widths ({ax},{ay}): albedo {albedo} exceeds lossless ceiling {ceiling}"
            );
        }
    }

    /// The frosted streak is directional: a reflected probe aligned with the
    /// narrow axis produces a different value than one aligned with the wide
    /// axis, confirming the lobe is genuinely anisotropic.
    #[test]
    fn highlight_depends_on_azimuth() {
        let rd = AnisoRoughDielectric::new(1.5, Vec3::ONE, Vec3::ONE, 0.6, 0.08);
        let wo = unit(0.0, 0.9, 0.0);
        // Two reflected directions tilted the same polar amount but 90 degrees
        // apart in azimuth around the normal.
        let wi_x = unit(0.5, 0.86, 0.0);
        let wi_z = unit(0.0, 0.86, 0.5);
        let fx = rd.evaluate(wo, wi_x, N).length();
        let fz = rd.evaluate(wo, wi_z, N).length();
        assert!(
            (fx - fz).abs() > 1e-3 * fx.max(fz).max(1e-3),
            "anisotropic highlight should vary with azimuth: {fx} vs {fz}"
        );
    }

    /// Grazing incidence yields no value and no density.
    #[test]
    fn evaluate_and_pdf_vanish_for_grazing() {
        let rd = clear_glass(0.4, 0.2);
        let grazing = unit(1.0, 0.0, 0.0);
        assert_eq!(rd.evaluate(grazing, N, N), Vec3::ZERO);
        assert!(rd.pdf(grazing, N, N) <= 0.0);
    }
}
