//! GPU Ground-Truth Ambient Occlusion (GTAO) subsystem.
//!
//! CPU golden and shader twin live in [`prism_render_shading::ao`] and
//! `shaders/gtao.wesl`; this module is the render-world plumbing that runs the
//! twin.  GTAO slots between the visibility raster and the shading resolve as
//! two compute steps:
//!
//! 1. a geometry prepass (`shaders/gtao_prepass.wesl`) decodes the visibility
//!    buffer into a linear view-depth texture and a view-space normal texture,
//!    and
//! 2. the GTAO kernel reads those to write per-pixel ambient visibility, which
//!    the resolve stage multiplies into its indirect/ambient term.
//!
//! Following the rest of the shading pipeline, it is split into cohesive files:
//!
//! * [`resources`] — the per-view [`resources::ViewGtaoTextures`] and the
//!   prepare system that keeps them sized to the viewport.
//! * [`abi`] — the immediate block shared with `gtao_prepass.wesl`.
//! * [`pipeline`] — the prepass compute pipeline + bind-group layouts.
//! * [`bind_groups`] — per-view prepass bind-group construction.
//! * [`dispatch`] — the `Core3d` node recording the prepass dispatch.
//!
//! The GTAO kernel pipeline and the resolve's consumption of the AO texture
//! land in following slices.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_gtao_prepass_bind_groups;
pub(crate) use dispatch::gtao_prepass_pass;
pub(crate) use pipeline::init_gtao_prepass_pipeline;
pub(crate) use resources::prepare_gtao_textures;
