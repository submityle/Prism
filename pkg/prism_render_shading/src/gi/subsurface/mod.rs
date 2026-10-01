//! Pre-integrated subsurface scattering CPU golden references.
//!
//! Screen-space skin shading that complements the physically based diffusion
//! profiles in [`crate::gi::translucency`].  Where `translucency` samples the
//! Burley searchlight profile directly, this module provides the cheap
//! real-time approximations used by AAA skin pipelines:
//!
//! * [`preintegrated`] — Penner 2011 curvature-driven pre-integrated BRDF: the
//!   `NdotL` wrap lookup baked from a 1-D diffusion profile and curvature.
//! * [`separable`] — Jimenez separable screen-space SSS Gaussian weights.
//! * [`transmittance`] — Jimenez thin-slab translucency transmittance profile.
//!
//! The [`shade_skin`] helper fuses all three into a single skin response:
//! curvature-softened front-lit diffuse plus back-lit transmitted glow.
//!
//! # Conventions
//! * `ndotl` is the clamped `[-1, 1]` front cosine `N·L`; `back_cosine` is the
//!   clamped `[0, 1]` back-facing cosine `dot(-N, L)` used for transmission.
//! * `curvature` is `1/radius` in inverse world units; `thickness` is a slab
//!   path length in profile units (millimetres).
//! * All colours are non-negative linear-RGB [`Vec3`]s; every output is finite
//!   and non-negative.
//! * Every item is a deterministic, GPU-free pure function with unit tests.

pub mod preintegrated;
pub mod separable;
pub mod transmittance;

use bevy_math::Vec3;

/// Inputs to [`shade_skin`], the fused pre-integrated skin response.
#[derive(Clone, Copy, Debug)]
pub struct SkinShadingInput {
    /// Front-lit cosine `N·L`, clamped to `[-1, 1]`.
    pub ndotl: f32,
    /// Local surface curvature `κ = 1/radius` (inverse world units), `≥ 0`.
    pub curvature: f32,
    /// Diffuse skin albedo (linear RGB), clamped to `[0, 1]`.
    pub albedo: Vec3,
    /// Incident light radiance (linear RGB), clamped non-negative.
    pub light_color: Vec3,
    /// Local slab thickness toward the light (profile units), magnitude used.
    pub back_thickness: f32,
    /// Back-facing cosine `dot(-N, L)`, clamped to `[0, 1]`.
    pub back_cosine: f32,
}

impl Default for SkinShadingInput {
    fn default() -> Self {
        Self {
            ndotl: 1.0,
            curvature: 0.0,
            albedo: Vec3::splat(0.5),
            light_color: Vec3::ONE,
            back_thickness: f32::INFINITY,
            back_cosine: 0.0,
        }
    }
}

/// Fused skin shading: curvature-softened front diffuse plus back translucency.
///
/// Combines [`preintegrated::preintegrated_diffuse`] (front-lit, curvature-aware
/// terminator softening, tinted by `albedo`) with
/// [`transmittance::transmitted_radiance`] (back-lit glow through a thin slab).
/// Every input is clamped into its valid domain; the returned radiance is
/// non-negative and finite.
pub fn shade_skin(input: SkinShadingInput) -> Vec3 {
    let albedo = input.albedo.clamp(Vec3::ZERO, Vec3::ONE);
    let light = input.light_color.max(Vec3::ZERO);

    let front_response = preintegrated::preintegrated_diffuse(input.ndotl, input.curvature);
    let front = albedo * light * front_response;

    let back = transmittance::transmitted_radiance(
        light,
        &transmittance::SKIN_SIX_GAUSSIAN,
        input.back_thickness,
        input.back_cosine,
    );

    (front + back).max(Vec3::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_only_matches_preintegrated() {
        // No back lighting ⇒ pure front-lit curvature diffuse, tinted by albedo.
        let input = SkinShadingInput {
            ndotl: 0.6,
            curvature: 2.0,
            albedo: Vec3::new(0.8, 0.5, 0.4),
            light_color: Vec3::new(1.0, 0.9, 0.8),
            back_thickness: f32::INFINITY,
            back_cosine: 0.0,
        };
        let got = shade_skin(input);
        let d = preintegrated::preintegrated_diffuse(0.6, 2.0);
        let expect = input.albedo * input.light_color * d;
        assert!((got - expect).length() < 1e-6, "got={got:?} expect={expect:?}");
    }

    #[test]
    fn back_lit_adds_transmitted_glow() {
        // Light behind the surface (negative N·L) still produces transmitted red.
        let input = SkinShadingInput {
            ndotl: -0.8,
            curvature: 1.0,
            albedo: Vec3::splat(0.6),
            light_color: Vec3::ONE,
            back_thickness: 1.0,
            back_cosine: 0.8,
        };
        let got = shade_skin(input);
        assert!(got.x > 0.0, "expected transmitted glow, got {got:?}");
        assert!(got.x >= got.z, "transmission should be red-biased: {got:?}");
    }

    #[test]
    fn flat_fully_lit_approaches_albedo() {
        // Flat surface, head-on light, no transmission ⇒ ~ albedo·light.
        let input = SkinShadingInput {
            ndotl: 1.0,
            curvature: 0.0,
            albedo: Vec3::new(0.7, 0.6, 0.5),
            light_color: Vec3::ONE,
            back_thickness: f32::INFINITY,
            back_cosine: 0.0,
        };
        let got = shade_skin(input);
        assert!((got - input.albedo).length() < 3e-2, "got={got:?}");
    }

    #[test]
    fn is_deterministic() {
        let input = SkinShadingInput::default();
        assert_eq!(shade_skin(input), shade_skin(input));
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        let input = SkinShadingInput {
            ndotl: f32::NAN,
            curvature: f32::NAN,
            albedo: Vec3::splat(f32::NAN),
            light_color: Vec3::splat(f32::NAN),
            back_thickness: f32::NAN,
            back_cosine: f32::NAN,
        };
        assert!(shade_skin(input).is_finite());
    }
}
