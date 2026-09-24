//! GPU shadow-visibility subsystem for the shading resolve pass.
//!
//! The shadow evaluation math lives in `shaders/shadow.wesl`, the WESL twin of
//! the CPU golden reference in `prism_render_shading::shadow`.  It computes the
//! analytic `visibility` term (directional cascaded shadow maps with PCF/PCSS,
//! and omnidirectional point-light cube distance maps) that the resolve pass
//! multiplies into each light's contribution.
//!
//! This module owns the WESL compilation coverage for both the shadow-sampling
//! shader (`shadow.wesl`) and the shadow-map depth-raster pass
//! (`shadow_depth.wesl`) that fills the atlas.  The depth pass renders scene
//! geometry from each shadow view (a directional cascade, a spot frustum, or a
//! point-light cube face) and stores NDC depth or range-normalized distance in
//! the atlas layer's `.r` channel.  Directional cascade matrices come from
//! `prism_render_shading::shadow::csm`; atlas layer assignment comes from
//! `prism_render_shading::shadow::atlas`.
//!
//! The device-side pipeline object construction, per-view uniform upload,
//! draw submission per atlas layer, and the resolve-side wiring that binds the
//! atlas texture and feeds these matrices are built in later slices and require
//! on-device validation for numerical parity against the CPU reference.

mod abi;
mod bindings;
mod depth_pass;
mod extract;
mod pipeline;
mod resources;
mod settings;
mod systems;

pub(crate) use bindings::ShadowBindGroup;
pub(crate) use extract::extract_shadows;
pub(crate) use resources::{
    ExtractedShadows, ShadowAtlas, ShadowAtlasConfig, ShadowGpuBuffers,
    DEFAULT_SHADOW_ATLAS_LAYERS, DEFAULT_SHADOW_ATLAS_RESOLUTION,
};
pub(crate) use depth_pass::{
    prepare_shadow_depth_uniform, queue_shadow_depth, shadow_depth_pass, ShadowDepthDrawList,
    ShadowDepthViewOffsets,
};
pub(crate) use pipeline::{
    init_shadow_depth_pipeline, register_shadow_depth_shader, ShadowDepthPipeline,
    ShadowDepthViewUniform,
};
pub(crate) use settings::PrismShadowSettings;
pub(crate) use systems::{
    ensure_shadow_atlas, prepare_shadow_bind_group, rebuild_shadow_buffers, write_shadow_buffers,
};

#[cfg(test)]
mod shader_tests;
