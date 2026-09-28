//! GPU image-based lighting (IBL) subsystem.
//!
//! CPU goldens and shader twins live in [`prism_render_shading::environment`]
//! and `shaders/{brdf_lut,env_prefilter}.wesl`; this module is the render-world
//! plumbing that runs the twins.  Real-time IBL splits the reflectance integral
//! into two precomputed halves that the shading resolve recombines:
//!
//! 1. a **split-sum environment BRDF** ("DFG") lookup table
//!    (`shaders/brdf_lut.wesl`) that depends only on view angle and roughness,
//!    and
//! 2. a **prefiltered radiance** cube-map mip chain (a following slice) that
//!    pre-convolves the environment for each roughness level.
//!
//! Following the rest of the shading pipeline, the plumbing is split into
//! cohesive files as each slice lands:
//!
//! * [`abi`] — the immediate block shared with `brdf_lut.wesl`.
//! * [`resources`] — the global [`resources::DfgLutTexture`] and its
//!   `RenderStartup` allocator.
//! * [`pipeline`] — the DFG precompute compute pipeline + bind-group layout.
//! * [`bind_groups`] — the one-shot DFG storage bind group.
//! * [`dispatch`] — the `Core3d` node recording the one-shot DFG integration.
//!
//! The prefiltered-radiance precompute and the resolve's consumption of both
//! tables arrive in the following slices.

mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::prepare_dfg_lut_bind_group;
pub(crate) use dispatch::dfg_lut_precompute_pass;
pub(crate) use pipeline::init_brdf_lut_pipeline;
pub(crate) use resources::init_dfg_lut_texture;
