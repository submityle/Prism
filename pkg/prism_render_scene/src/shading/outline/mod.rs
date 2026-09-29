//! NPR silhouette/crease outline subsystem.
//!
//! The CPU golden lives in [`prism_render_shading::outline`] and the shader
//! twin in `shaders/outline.wesl`; this module is the render-world plumbing
//! that will run them. The outline is a *screen-space* post decision: for each
//! shaded pixel it reads a fixed four-neighbour cross of the geometry buffer
//! (vis-buffer material id, linear depth and packed normal) and raises an ink
//! line on any discontinuity — an authored outline/material-id boundary, a
//! large relative depth jump (silhouette against the background or a fold), or
//! a sharp normal turn with no depth jump (interior crease). The three edges
//! combine as a union, so the pass emits a single `[0, 1]` coverage the
//! composite stage `mix`es toward the authored line colour.
//!
//! Because the outline reuses the vis-buffer / G-buffer the opaque pass already
//! resolves, the extra plumbing is a single screen-space compute pass; it lands
//! across cohesive slices matching the rest of the pipeline:
//!
//! * *(following slices)* — the shared `OutlineParams` immediate ABI, the
//!   per-view line target, the compute pipeline and bind group over the
//!   material-id/depth/normal reads, the `Core3d` dispatch node, and the
//!   composite consumption that blends the coverage over the shaded colour. The
//!   ABI struct lands with that first live consumer so no committed ABI is
//!   dead, matching the SSR / SSGI precedent.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_outline_bind_groups;
pub(crate) use dispatch::outline_pass;
pub(crate) use pipeline::init_outline_pipeline;
pub(crate) use resources::prepare_outline_textures;
pub(crate) use settings::PrismOutlineSettings;

#[cfg(test)]
mod shader_tests;
