//! HDR gamut-mapping post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::gamut_map`] and the shader
//! twin in `shaders/gamut_map.wesl`; this module is the render-world plumbing
//! that runs them. Gamut mapping reads the resolved, *pre-exposed* linear HDR
//! radiance (see the exposure subsystem), after colour grading and *before* the
//! display tone-map curve, and compresses colours that fall outside the target
//! display gamut (linear `sRGB` / Rec. 709) smoothly back toward the achromatic
//! axis — the `OpenColorIO` / `ACES` gamut-compress approach — rather than
//! hard-clamping them and shifting their hue.
//!
//! Gamut mapping is a shared post-processing base, not a peer of the PBR/NPR
//! shading fronts: every illumination model — physically based or stylized —
//! writes into the same HDR buffer this pass consumes, so one gamut map serves
//! them all. The subsystem is the standard five-file split — the
//! [`GpuGamutMapParams`](abi::GpuGamutMapParams) immediate block, the
//! [`PrismGamutMapSettings`] resource, the compute pipeline, the per-view output
//! texture and its bind group, and the `Core3d` dispatch — each committed with
//! its first live consumer so no ABI is dead, matching the
//! SSR / SSGI / exposure / bloom / colour-grade precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_gamut_map_bind_groups;
pub(crate) use dispatch::gamut_map_pass;
pub(crate) use pipeline::init_gamut_map_pipeline;
pub(crate) use resources::prepare_gamut_map_textures;
pub(crate) use settings::PrismGamutMapSettings;

#[cfg(test)]
mod shader_tests;
