//! Anisotropic specular BRDF CPU golden references.
//!
//! Deterministic, GPU-free anisotropic GGX lobe for brushed metal, hair-like
//! highlights, and directional roughness.  Distinct from the isotropic GGX
//! ReSTIR lobe in [`crate::gi::spec_gi`]: here roughness differs along the
//! tangent/bitangent frame.
//!
//! * [`ndf`] — anisotropic GGX normal-distribution + Smith height-correlated
//!   masking-shadowing in the tangent frame.
//! * [`remap`] — Burley/Disney anisotropy↔(alpha_t, alpha_b) remapping and the
//!   tangent-frame construction from a surface tangent.
//! * [`sample`] — anisotropic VNDF importance sampling and its PDF.

pub mod ndf;
pub mod remap;
pub mod sample;

use bevy_math::Vec3;

use crate::gi::spec_gi::ggx_lobe::fresnel_schlick;
use ndf::{ndf_anisotropic, smith_g2};
use remap::{TangentFrame, anisotropy_to_alpha, orthonormal_tangent_frame, rotate_tangent_frame};

/// Floor applied to cosine denominators in the BRDF so grazing geometry stays
/// finite instead of dividing by zero.
const MIN_DENOM: f32 = 1.0e-6;

/// Artist-facing parameters for the anisotropic GGX specular lobe.
///
/// `roughness` and `anisotropy` are remapped to the GGX widths `(α_t, α_b)` via
/// [`remap::anisotropy_to_alpha`]; `rotation` spins the anisotropy direction in
/// the surface plane (glTF `KHR_materials_anisotropy`); `f0` is the RGB normal-
/// incidence specular reflectance fed to the Schlick Fresnel term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnisotropicGgxParams {
    /// Perceptual roughness in `[0, 1]` (`α = roughness²`).
    pub roughness: f32,
    /// Anisotropy strength in `[0, 1]`; `0` is isotropic.
    pub anisotropy: f32,
    /// Rotation of the tangent (anisotropy direction) about the normal, radians.
    pub rotation: f32,
    /// RGB normal-incidence specular reflectance `F0`.
    pub f0: Vec3,
}

impl Default for AnisotropicGgxParams {
    #[inline]
    fn default() -> Self {
        Self {
            roughness: 0.3,
            anisotropy: 0.0,
            rotation: 0.0,
            f0: Vec3::splat(0.04),
        }
    }
}

/// Builds the oriented shading frame and GGX widths for `params` from a world
/// `normal` and `tangent`.
///
/// Returns the right-handed [`TangentFrame`] (after applying
/// [`AnisotropicGgxParams::rotation`]) together with `(α_t, α_b)`.  Shared by
/// the BRDF evaluator and exposed so sampling callers can reuse the exact frame.
#[inline]
pub fn shading_frame(
    normal: Vec3,
    tangent: Vec3,
    params: &AnisotropicGgxParams,
) -> (TangentFrame, f32, f32) {
    let base = orthonormal_tangent_frame(normal, tangent);
    let frame = if params.rotation != 0.0 {
        rotate_tangent_frame(base, params.rotation)
    } else {
        base
    };
    let (alpha_t, alpha_b) = anisotropy_to_alpha(params.roughness, params.anisotropy);
    (frame, alpha_t, alpha_b)
}

/// Evaluates the anisotropic GGX specular BRDF for world-space incident `wi`,
/// outgoing `wo`, shading `normal` `n`, surface `tangent` `t`, and `params`.
///
/// Returns the pure BRDF value (excluding the `n·wi` cosine), i.e.
/// `f = D · G2 · F / (4 (n·wo) (n·wi))`, as an RGB [`Vec3`].  Returns
/// [`Vec3::ZERO`] for below-horizon configurations or a degenerate half vector,
/// and never emits `NaN`/`inf`.
#[inline]
pub fn anisotropic_ggx_brdf(
    wi: Vec3,
    wo: Vec3,
    n: Vec3,
    t: Vec3,
    params: &AnisotropicGgxParams,
) -> Vec3 {
    let (frame, alpha_t, alpha_b) = shading_frame(n, t, params);
    let nrm = frame.normal;

    let n_dot_wo = wo.dot(nrm);
    let n_dot_wi = wi.dot(nrm);
    if n_dot_wo <= 0.0 || n_dot_wi <= 0.0 {
        return Vec3::ZERO;
    }

    let h = (wo + wi).normalize_or_zero();
    if h.length_squared() == 0.0 {
        return Vec3::ZERO;
    }

    let d = ndf_anisotropic(h, frame.tangent, frame.bitangent, nrm, alpha_t, alpha_b);
    let g = smith_g2(wo, wi, frame.tangent, frame.bitangent, nrm, alpha_t, alpha_b);
    let f = fresnel_schlick(params.f0, wo.dot(h).max(0.0));
    let denom = (4.0 * n_dot_wo * n_dot_wi).max(MIN_DENOM);
    let v = f * (d * g / denom);
    if v.is_finite() { v.max(Vec3::ZERO) } else { Vec3::ZERO }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gi::spec_gi::ggx_lobe::ggx_brdf;
    use ndf::to_tangent_space;

    #[test]
    fn brdf_is_nonnegative_and_finite() {
        let params = AnisotropicGgxParams {
            roughness: 0.4,
            anisotropy: 0.6,
            rotation: 0.0,
            f0: Vec3::splat(0.04),
        };
        let n = Vec3::Z;
        let t = Vec3::X;
        let wo = Vec3::new(0.3, 0.1, 0.95).normalize();
        for i in 0..24u32 {
            for j in 0..24u32 {
                let theta = (i as f32 + 0.5) / 24.0 * core::f32::consts::FRAC_PI_2;
                let phi = (j as f32 + 0.5) / 24.0 * core::f32::consts::TAU;
                let (st, ct) = bevy_math::ops::sin_cos(theta);
                let (sp, cp) = bevy_math::ops::sin_cos(phi);
                let wi = Vec3::new(st * cp, st * sp, ct);
                let v = anisotropic_ggx_brdf(wi, wo, n, t, &params);
                assert!(v.is_finite(), "non-finite: {v:?}");
                assert!(v.x >= 0.0 && v.y >= 0.0 && v.z >= 0.0, "negative: {v:?}");
            }
        }
    }

    #[test]
    fn below_horizon_is_zero() {
        let params = AnisotropicGgxParams::default();
        let n = Vec3::Z;
        let t = Vec3::X;
        let wo = Vec3::new(0.0, 0.0, 0.9).normalize();
        let wi = Vec3::new(0.1, 0.1, -0.9).normalize();
        assert_eq!(anisotropic_ggx_brdf(wi, wo, n, t, &params), Vec3::ZERO);
    }

    #[test]
    fn isotropic_matches_reference_ggx_lobe() {
        // With anisotropy = 0 our world-space BRDF must agree with the isotropic
        // GGX lobe reference evaluated in its local frame.
        let params = AnisotropicGgxParams {
            roughness: 0.5,
            anisotropy: 0.0,
            rotation: 0.0,
            f0: Vec3::splat(0.04),
        };
        let n = Vec3::Z;
        let t = Vec3::X;
        let frame = orthonormal_tangent_frame(n, t);
        let (at, ab) = anisotropy_to_alpha(params.roughness, params.anisotropy);
        let wo = Vec3::new(0.2, 0.1, 0.97).normalize();
        for &wi in &[
            Vec3::new(-0.2, 0.1, 0.97).normalize(),
            Vec3::new(0.4, -0.3, 0.85).normalize(),
            Vec3::new(0.0, 0.5, 0.86).normalize(),
        ] {
            let ours = anisotropic_ggx_brdf(wi, wo, n, t, &params);
            let wo_l = to_tangent_space(wo, frame.tangent, frame.bitangent, frame.normal);
            let wi_l = to_tangent_space(wi, frame.tangent, frame.bitangent, frame.normal);
            let reference = ggx_brdf(wo_l, wi_l, at, ab, params.f0);
            assert!((ours - reference).length() < 1e-4, "ours={ours:?} ref={reference:?}");
        }
    }

    #[test]
    fn anisotropy_breaks_azimuthal_symmetry() {
        // A strongly anisotropic lobe must respond differently to half vectors
        // tilted along the tangent vs the bitangent.
        let params = AnisotropicGgxParams {
            roughness: 0.3,
            anisotropy: 0.9,
            rotation: 0.0,
            f0: Vec3::splat(0.04),
        };
        let n = Vec3::Z;
        let t = Vec3::X;
        let wo = Vec3::Z;
        // Reflect symmetric incident directions in the two principal planes.
        let wi_t = Vec3::new(0.3, 0.0, 0.954).normalize();
        let wi_b = Vec3::new(0.0, 0.3, 0.954).normalize();
        let along_t = anisotropic_ggx_brdf(wi_t, wo, n, t, &params);
        let along_b = anisotropic_ggx_brdf(wi_b, wo, n, t, &params);
        assert!(
            (along_t - along_b).length() > 1e-3,
            "lobe should be directional: t={along_t:?} b={along_b:?}"
        );
    }

    #[test]
    fn rotation_realigns_the_lobe() {
        // Rotating the tangent by 90° swaps the tangent/bitangent response, so a
        // rotated evaluation along the bitangent matches the unrotated one along
        // the tangent.
        let base = AnisotropicGgxParams {
            roughness: 0.3,
            anisotropy: 0.9,
            rotation: 0.0,
            f0: Vec3::splat(0.04),
        };
        let rotated = AnisotropicGgxParams {
            rotation: core::f32::consts::FRAC_PI_2,
            ..base
        };
        let n = Vec3::Z;
        let t = Vec3::X;
        let wo = Vec3::Z;
        let wi_t = Vec3::new(0.3, 0.0, 0.954).normalize();
        let wi_b = Vec3::new(0.0, 0.3, 0.954).normalize();
        let base_along_t = anisotropic_ggx_brdf(wi_t, wo, n, t, &base);
        let rot_along_b = anisotropic_ggx_brdf(wi_b, wo, n, t, &rotated);
        assert!(
            (base_along_t - rot_along_b).length() < 1e-3,
            "rotation did not realign: base_t={base_along_t:?} rot_b={rot_along_b:?}"
        );
    }

    #[test]
    fn default_params_are_isotropic_and_smooth() {
        let p = AnisotropicGgxParams::default();
        let (at, ab) = anisotropy_to_alpha(p.roughness, p.anisotropy);
        assert!((at - ab).abs() < 1e-6, "default must be isotropic");
    }
}
