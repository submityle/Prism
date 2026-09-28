//! GPU temporal anti-aliasing (TAA) subsystem.
//!
//! The CPU golden twin lives in [`prism_render_shading::taa`] and the shader in
//! `shaders/taa_resolve.wesl`; this module is the render-world plumbing that
//! runs them. TAA runs as a compute pass on the composited HDR `scene_color`
//! (opaque + SSR, after `ssr_composite` and before the main pass): each frame
//! the sub-pixel-jittered current colour is blended with the motion-reprojected
//! previous frame so the jitter integrates into a supersampled image, fighting
//! both geometric aliasing and specular firefly flicker. The forward
//! transparency composite runs afterwards so alpha stays sharp.
//!
//! Following the rest of the shading pipeline, the plumbing lands as cohesive
//! files:
//!
//! * [`abi`] — the immediate block shared with `taa_resolve.wesl`.
//! * [`resources`] — the persistent per-view ping-pong history pair and the
//!   [`ViewTaa`] the resolve reads/writes each frame.
//! * [`pipeline`] — the resolve compute pipeline, its group-0 layout and the
//!   filtering history sampler.
//! * [`bind_groups`] — the per-view resolve bind group.
//! * [`dispatch`] — the `Core3d` node recording the resolve dispatch.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_taa_bind_groups;
pub(crate) use dispatch::taa_resolve_pass;
pub(crate) use pipeline::init_taa_resolve_pipeline;
pub(crate) use resources::{prepare_taa_textures, ViewTaa};
