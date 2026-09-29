//! Depth-of-field (`DoF`) post-processing subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::dof`] and the shader twin in
//! `shaders/dof.wesl`; this module is the render-world plumbing that runs them.
//! `DoF` reads the resolved, *pre-exposed* HDR radiance (see the exposure
//! subsystem) plus the scene depth, computes each pixel's thin-lens circle of
//! confusion from the physical camera (aperture / focal length / focus
//! distance / sensor width), separates the near and far fields and blends a
//! soft-edged bokeh gather over the sharp image by the `CoC`.
//!
//! `DoF` is a shared post-processing base, not a peer of the PBR/NPR shading
//! fronts: every illumination model — physically based or stylized — writes
//! into the same HDR buffer this pass consumes, so one implementation serves
//! them all.
//!
//! The subsystem is a three-pass compute chain (a scatter-as-gather disk bokeh):
//!
//! * [`pipeline`] builds the `dof_coc` / `dof_gather` / `dof_composite` compute
//!   pipelines and their group-0 layouts at `RenderStartup`.
//! * [`resources`] allocates the per-view `CoC` / blurred / output textures during
//!   `PrepareResources`, gated on the enable and the resident visibility + SSR
//!   buffers it reads.
//! * [`bind_groups`] wires those textures into the three group-0 bind groups
//!   during `PrepareBindGroups`.
//! * [`dispatch`] records the `CoC` -> gather -> composite chain as a `Core3d`
//!   scheduling system, then copies the composited output back over
//!   `scene_color`.
//!
//! [`settings`] is the single render-world resource gating the subsystem and
//! feeding the golden tunables into the [`abi`] immediate blocks, mirroring the
//! golden [`prism_render_shading::DofParams`] defaults.
//!
//! Depth is the SSR geometry prepass's reverse-Z device depth, unprojected to a
//! linear view distance — reusing that pass rather than duplicating a depth
//! prepass, the same coupling motion blur relies on.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_dof_bind_groups;
pub(crate) use dispatch::dof_pass;
pub(crate) use pipeline::init_dof_pipeline;
pub(crate) use resources::prepare_dof_textures;
pub(crate) use settings::PrismDofSettings;

#[cfg(test)]
mod shader_tests;
