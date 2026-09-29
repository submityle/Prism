//! The four-frontend lighting-response fork (design §5).
//!
//! Geometry, simulation, refraction routing, and the entire shared advanced
//! base ([`super::SharedBaseServices`]) are identical across every frontend;
//! the frontends diverge *only* in how they interpret the resolved lighting.
//! This module owns that divergence and nothing else.
//!
//! - [`pbr`] — physically based: `Schlick` `Fresnel` from the water `IOR`, the
//!   `SSR` -> `RT` -> probe reflection tier, a `GGX` micro-surface roughness
//!   raised by the `Jacobian` foam mask, and grazing subsurface transmission.
//!   Mirrors `UE5` Single Layer Water and `Crest`/`WaveWorks` shading.
//! - [`npr`] — stylized (illumination axis): ramp-quantized water color, toon
//!   specular blocks, hand-drawn shoreline foam edges, `halftone` caustics, and
//!   flow-aligned stylization lines. Mirrors `Zelda`/`Genshin`/`Okami` water.
//! - [`hybrid`] — per-region blend of the two by depth and shore proximity plus
//!   an orthogonal stylized overlay (`Arcane`-style hand-paint over a physical
//!   base). All four frontends remain first-class citizens.
//!
//! Every planner here is a pure, deterministic function of a static
//! [`ShadingProfile`] slice and per-view [`SurfaceShadingInputs`]: identical
//! arguments always yield identical responses, keeping `GPU` uploads stable
//! across frames. This mirrors the [`super::optics`] planning contract.

use super::underwater::RgbColor;
use super::ShadingFrontend;

pub mod hybrid;
pub mod npr;
pub mod pbr;

/// Reflection source selected for a physically based water surface.
///
/// The `PBR` frontend walks a three-tier fallback: screen-space reflections
/// when the `HZB` trace is confident, hardware ray tracing when the `RT` budget
/// allows it, and a prefiltered probe otherwise. This mirrors the `SSR` ->
/// `RT` -> probe routing used by `UE5` and `Crest`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ReflectionTier {
    /// Screen-space reflection via the hierarchical depth buffer.
    ScreenSpace,
    /// Hardware ray-traced reflection (`SSR` miss fallback).
    RayTraced,
    /// Prefiltered reflection probe (final fallback).
    Probe,
}

/// Per-view lighting inputs shared by every frontend.
///
/// These are the resolved, frontend-agnostic quantities the renderer samples
/// each frame. The frontends read the subset they need and respond
/// differently; the inputs themselves never encode a frontend decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceShadingInputs {
    /// Cosine of the angle between the view ray and the surface normal, in
    /// `0..=1` (`1` looking straight down, `0` at grazing).
    pub cos_view: f32,
    /// Water index of refraction (`IOR`), expected `> 1`.
    pub ior: f32,
    /// Surface `Jacobian`; values below the fold threshold indicate compression
    /// and drive the micro-surface foam mask.
    pub jacobian: f32,
    /// Base water `RGB` color before the frontend response.
    pub water_color: RgbColor,
    /// Raw specular highlight intensity from the lighting pass, in `0..=1`.
    pub specular_intensity: f32,
    /// Raw caustic coverage from the caustics pass, in `0..=1`.
    pub caustic_intensity: f32,
    /// Surface flow speed in m/s, driving stylized flow lines.
    pub flow_speed: f32,
    /// Water-column depth in meters, driving the hybrid depth mask.
    pub depth: f32,
    /// Distance to the shoreline in meters, driving the hybrid shore mask and
    /// the `NPR` hand-drawn foam edge.
    pub dist_to_shore: f32,
    /// Screen-space reflection hit confidence, in `0..=1`.
    pub ssr_confidence: f32,
    /// Ray-tracing budget availability, in `0..=1`.
    pub ray_budget: f32,
}

/// Static `PBR` tuning for one water body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbrShadingParams {
    /// Explicit base reflectance `F0`; when `<= 0` the response derives `F0`
    /// from the `IOR` via the `Schlick`/`Fresnel` relation.
    pub f0_override: f32,
    /// `Jacobian` at or above which no micro-surface foam appears; the mask
    /// ramps to full coverage as the `Jacobian` drops to zero.
    pub foam_fold_threshold: f32,
    /// Base `GGX` roughness for calm water, in `0..=1`.
    pub base_roughness: f32,
    /// Grazing subsurface back-transmission strength (浪尖背光透光), in `0..=1`.
    pub grazing_transmission: f32,
    /// Minimum screen-space confidence to accept the `SSR` tier.
    pub ssr_min_confidence: f32,
    /// Minimum ray budget to accept the `RT` tier when `SSR` misses.
    pub rt_min_budget: f32,
}

/// Static `NPR` tuning for one water body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NprShadingParams {
    /// Number of quantization bands for the stylized water-color ramp.
    pub color_bands: u32,
    /// Specular intensity at or above which the toon highlight block turns on.
    pub specular_threshold: f32,
    /// Shore distance in meters over which the hand-drawn foam edge fades out.
    pub foam_edge_width: f32,
    /// Screen-space dot scale for the `halftone` stylized caustics.
    pub halftone_scale: f32,
    /// Flow-line intensity gain per unit flow speed.
    pub flow_line_gain: f32,
}

/// Static blend tuning for the hybrid frontend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridBlendParams {
    /// Shore distance in meters over which the response fades from `NPR` (at
    /// the shore) to `PBR` (offshore).
    pub shore_blend_dist: f32,
    /// Water depth in meters over which the response fades from `NPR` (shallow)
    /// to `PBR` (deep).
    pub deep_blend_depth: f32,
    /// Opacity of the orthogonal stylized overlay laid over the physical base,
    /// in `0..=1`.
    pub overlay_opacity: f32,
}

/// The complete per-body shading tuning: one slice per frontend plus the shared
/// hybrid blend knobs and the custom closure identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadingProfile {
    /// Physically based tuning.
    pub pbr: PbrShadingParams,
    /// Stylized tuning.
    pub npr: NprShadingParams,
    /// Hybrid blend tuning.
    pub hybrid: HybridBlendParams,
    /// Author-injected closure identity for the custom frontend; the shader
    /// specialization keyed to this id runs on top of the physical base.
    pub custom_hook_id: u32,
}

impl ShadingProfile {
    /// A coherent default matching the clear open-water optics fixture.
    #[must_use]
    pub fn physical_water() -> Self {
        Self {
            pbr: PbrShadingParams {
                f0_override: 0.0,
                foam_fold_threshold: 1.0,
                base_roughness: 0.08,
                grazing_transmission: 0.6,
                ssr_min_confidence: 0.5,
                rt_min_budget: 0.25,
            },
            npr: NprShadingParams {
                color_bands: 4,
                specular_threshold: 0.7,
                foam_edge_width: 1.5,
                halftone_scale: 8.0,
                flow_line_gain: 0.5,
            },
            hybrid: HybridBlendParams {
                shore_blend_dist: 3.0,
                deep_blend_depth: 2.0,
                overlay_opacity: 0.35,
            },
            custom_hook_id: 0,
        }
    }
}

/// Resolved physically based response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PbrResponse {
    /// Base reflectance `F0` used by the `Fresnel` term.
    pub f0: f32,
    /// `Schlick` `Fresnel` reflectance at the view angle, in `0..=1`.
    pub fresnel: f32,
    /// Selected reflection tier.
    pub reflection_tier: ReflectionTier,
    /// Micro-surface foam coverage from the `Jacobian` fold, in `0..=1`.
    pub foam_mask: f32,
    /// Effective `GGX` roughness after foam roughening, in `0..=1`.
    pub specular_roughness: f32,
    /// Grazing subsurface back-transmission, in `0..=1`.
    pub subsurface_transmission: f32,
}

/// Resolved stylized response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NprResponse {
    /// Ramp-quantized stylized water color.
    pub ramp_color: RgbColor,
    /// Toon specular block, `1` when the highlight is on, `0` otherwise.
    pub toon_specular: f32,
    /// Hand-drawn shoreline foam edge coverage, in `0..=1`.
    pub foam_edge: f32,
    /// `Halftone` caustic dot coverage, in `0..=1`.
    pub halftone_coverage: f32,
    /// Screen-space dot scale carried through for the `halftone` pattern.
    pub halftone_scale: f32,
    /// Flow-aligned stylization line intensity, in `0..=1`.
    pub flow_line: f32,
}

/// Resolved hybrid response: both sub-responses plus their blend weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridResponse {
    /// Physically based sub-response.
    pub pbr: PbrResponse,
    /// Stylized sub-response.
    pub npr: NprResponse,
    /// Weight toward the `NPR` response, in `0..=1`.
    pub npr_weight: f32,
    /// Weight toward the `PBR` response; `pbr_weight + npr_weight == 1`.
    pub pbr_weight: f32,
    /// Orthogonal stylized overlay opacity laid over the blended base.
    pub overlay_opacity: f32,
}

/// Resolved custom response: the author closure identity over a physical base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomResponse {
    /// Author-injected closure identity to specialize in the shader.
    pub hook_id: u32,
    /// The physically based base the custom closure builds on.
    pub base: PbrResponse,
}

/// The frontend-tagged shading response for one water body this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterShadingResponse {
    /// Physically based response.
    Pbr(PbrResponse),
    /// Stylized response.
    Npr(NprResponse),
    /// Custom-closure response over a physical base.
    Custom(CustomResponse),
    /// Per-region hybrid blend response.
    Hybrid(HybridResponse),
}

/// Dispatch the lighting-response fork for one body.
///
/// Selects the frontend planner and returns its frontend-tagged response. Pure
/// and deterministic: geometry/simulation/base services are untouched here, so
/// this is the single point where the four frontends diverge.
#[must_use]
pub fn plan_shading(
    frontend: ShadingFrontend,
    profile: ShadingProfile,
    inputs: SurfaceShadingInputs,
) -> WaterShadingResponse {
    match frontend {
        ShadingFrontend::Pbr => WaterShadingResponse::Pbr(pbr::plan_pbr(profile.pbr, inputs)),
        ShadingFrontend::Npr => WaterShadingResponse::Npr(npr::plan_npr(profile.npr, inputs)),
        ShadingFrontend::Custom => WaterShadingResponse::Custom(CustomResponse {
            hook_id: profile.custom_hook_id,
            base: pbr::plan_pbr(profile.pbr, inputs),
        }),
        ShadingFrontend::Hybrid => WaterShadingResponse::Hybrid(hybrid::plan_hybrid(
            profile.hybrid,
            profile.pbr,
            profile.npr,
            inputs,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_inputs() -> SurfaceShadingInputs {
        SurfaceShadingInputs {
            cos_view: 0.5,
            ior: 1.33,
            jacobian: 0.4,
            water_color: RgbColor {
                r: 0.1,
                g: 0.3,
                b: 0.5,
            },
            specular_intensity: 0.8,
            caustic_intensity: 0.6,
            flow_speed: 1.2,
            depth: 1.0,
            dist_to_shore: 1.0,
            ssr_confidence: 0.7,
            ray_budget: 0.5,
        }
    }

    #[test]
    fn physical_water_profile_is_deterministic() {
        assert_eq!(
            ShadingProfile::physical_water(),
            ShadingProfile::physical_water()
        );
    }

    #[test]
    fn dispatch_selects_matching_variant() {
        let profile = ShadingProfile::physical_water();
        let inputs = fixture_inputs();
        assert!(matches!(
            plan_shading(ShadingFrontend::Pbr, profile, inputs),
            WaterShadingResponse::Pbr(_)
        ));
        assert!(matches!(
            plan_shading(ShadingFrontend::Npr, profile, inputs),
            WaterShadingResponse::Npr(_)
        ));
        assert!(matches!(
            plan_shading(ShadingFrontend::Custom, profile, inputs),
            WaterShadingResponse::Custom(_)
        ));
        assert!(matches!(
            plan_shading(ShadingFrontend::Hybrid, profile, inputs),
            WaterShadingResponse::Hybrid(_)
        ));
    }

    #[test]
    fn custom_carries_hook_and_physical_base() {
        let mut profile = ShadingProfile::physical_water();
        profile.custom_hook_id = 42;
        let inputs = fixture_inputs();
        let WaterShadingResponse::Custom(custom) =
            plan_shading(ShadingFrontend::Custom, profile, inputs)
        else {
            panic!("expected a custom response");
        };
        assert_eq!(custom.hook_id, 42);
        // The base equals the standalone PBR plan: custom never fakes lighting.
        assert_eq!(custom.base, pbr::plan_pbr(profile.pbr, inputs));
    }

    #[test]
    fn dispatch_is_deterministic() {
        let profile = ShadingProfile::physical_water();
        let inputs = fixture_inputs();
        for frontend in [
            ShadingFrontend::Pbr,
            ShadingFrontend::Npr,
            ShadingFrontend::Custom,
            ShadingFrontend::Hybrid,
        ] {
            assert_eq!(
                plan_shading(frontend, profile, inputs),
                plan_shading(frontend, profile, inputs)
            );
        }
    }
}
