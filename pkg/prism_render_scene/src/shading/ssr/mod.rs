//! GPU screen-space reflection (SSR) subsystem.
//!
//! The CPU golden and shader twins live in
//! [`prism_render_shading::screen_space`] and `shaders/ssr*.wesl`; this module
//! is the render-world plumbing that runs them. Because Prism is a
//! visibility-buffer deferred renderer with no depth/normal G-buffer, SSR first
//! *rebuilds* the inputs its trace needs, then reflects the view ray off each
//! surface, marches the reflected ray across the reverse-Z HZB "nearest depth"
//! pyramid, reprojects a hit into the previous frame's colour, and produces a
//! reflected-radiance-plus-confidence buffer the shading resolve blends over
//! the prefiltered IBL specular wherever the trace is reliable.
//!
//! Following the rest of the shading pipeline, the plumbing lands as cohesive
//! files across successive slices:
//!
//! * [`abi`] — the immediate blocks shared with the SSR shaders.
//! * [`resources`] — the per-view device-depth / view-normal prepass targets,
//!   the reverse-Z Hi-Z pyramid (and its per-mip views), and their
//!   viewport-sized allocator.
//! * [`pipeline`] — the geometry-prepass compute pipeline plus bind-group
//!   layouts.
//! * [`bind_groups`] — the per-view visibility/scene prepass bind groups.
//! * [`dispatch`] — the `Core3d` node recording the prepass dispatch.
//! * [`hzb`] — the Hi-Z pyramid build pipelines, per-level bind groups and the
//!   `Core3d` node recording the copy + 2x2-max reductions.
//! * [`repack`] — the material-roughness repack pipeline, its per-view and
//!   scene-table bind groups, and the `Core3d` node that reconstructs each
//!   covered pixel's interpolated UV from the visibility buffer, samples the
//!   metallic-roughness texture through the shared bindless heap
//!   (`material_sample`), and folds the prepass view-normal plus that
//!   texture-modulated roughness into the trace's packed `normal_roughness`
//!   input — all on the current frame in this single pass.
//! * [`color_mips`] — the copy + 2x2-average mip-build pipelines, per-level bind
//!   groups and the `Core3d` node that builds the current-frame scene-colour
//!   pyramid (level 0 = `scene_color`, coarser levels = box-filtered pre-blur)
//!   the trace samples for reflected radiance, run after the resolve.
//!
//! * [`trace`] — the screen-space reflection march pipeline, its per-view
//!   bind group and the `Core3d` node that reflects the view ray off each
//!   reconstructed surface, marches the reverse-Z Hi-Z pyramid and samples the
//!   roughness-selected colour-pyramid mip at the hit, writing reflected
//!   radiance plus a blend confidence into the reflection output.
//! * [`reconstruct`] — the spatial-reconstruction (bilateral resolve) pipeline,
//!   its per-view bind group and the `Core3d` node that denoises the noisy
//!   multi-ray trace with an edge-aware neighbourhood filter, writing the
//!   resolved reflection the composite reads.
//! * [`temporal`] — the cross-frame accumulation pipeline, its persistent
//!   per-view ping-pong history, per-view bind group and the `Core3d` node
//!   that camera-reprojects the previous frame's accumulated reflection,
//!   colour-box-clips it against ghosting and exponentially blends it with the
//!   resolve, feeding the composite a temporally stable buffer.
//! * [`composite`] — the pipeline, per-view bind group and `Core3d` node that
//!   fold that reflection output back over the shaded `scene_color` (reading
//!   the untouched base from colour-pyramid level 0 to avoid storage-image
//!   read/write aliasing) before the main pass presents it.

mod abi;
mod bind_groups;
mod color_mips;
mod composite;
mod dispatch;
mod hzb;
mod pipeline;
mod reconstruct;
mod temporal;
mod repack;
mod trace;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_ssr_prepass_bind_groups;
pub(crate) use dispatch::ssr_prepass_pass;
pub(crate) use hzb::{init_ssr_hzb_pipeline, prepare_ssr_hzb_bind_groups, ssr_hzb_pass};
pub(crate) use color_mips::{
    init_ssr_color_mips_pipeline, prepare_ssr_color_mips_bind_groups, ssr_color_mips_pass,
};
pub(crate) use composite::{
    init_ssr_composite_pipeline, prepare_ssr_composite_bind_groups, ssr_composite_pass,
};
pub(crate) use trace::{init_ssr_trace_pipeline, prepare_ssr_trace_bind_groups, ssr_trace_pass};
pub(crate) use reconstruct::{
    init_ssr_reconstruct_pipeline, prepare_ssr_reconstruct_bind_groups, ssr_reconstruct_pass,
};
pub(crate) use temporal::{
    init_ssr_temporal_pipeline, prepare_ssr_temporal_bind_groups, prepare_ssr_temporal_textures,
    ssr_temporal_pass,
};
pub(crate) use repack::{init_ssr_repack_pipeline, prepare_ssr_repack_bind_groups, ssr_repack_pass};
pub(crate) use pipeline::init_ssr_prepass_pipeline;
pub(crate) use resources::prepare_ssr_textures;
