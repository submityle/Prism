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
//! * The kernel pipeline/bind-group/dispatch live alongside the prepass in
//!   [`pipeline`], [`bind_groups`], and [`dispatch`]. The resolve's
//!   consumption of the AO texture lands in a following slice.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod temporal;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::{
    prepare_gtao_denoise_bind_groups, prepare_gtao_kernel_bind_groups,
    prepare_gtao_prepass_bind_groups,
};
pub(crate) use dispatch::{gtao_compute_pass, gtao_denoise_pass, gtao_prepass_pass};
pub(crate) use pipeline::{
    init_gtao_denoise_pipeline, init_gtao_kernel_pipeline, init_gtao_prepass_pipeline,
};
pub(crate) use resources::{prepare_gtao_textures, ViewGtaoTextures};
pub(crate) use temporal::{
    gtao_temporal_pass, init_gtao_temporal_pipeline, prepare_gtao_temporal_bind_groups,
    prepare_gtao_temporal_textures,
};
