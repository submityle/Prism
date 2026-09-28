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
//!
//! The material-roughness repack, the trace and the resolve's consumption of
//! the reflection buffer land in following slices.

mod abi;
mod bind_groups;
mod dispatch;
mod hzb;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_ssr_prepass_bind_groups;
pub(crate) use dispatch::ssr_prepass_pass;
pub(crate) use hzb::{init_ssr_hzb_pipeline, prepare_ssr_hzb_bind_groups, ssr_hzb_pass};
pub(crate) use pipeline::init_ssr_prepass_pipeline;
pub(crate) use resources::prepare_ssr_textures;
