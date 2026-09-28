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
//! * [`extract`] — mirrors the active probe's radiance source into the
//!   render world for the prefilter pass.
//!
//! The shading resolve binds both tables into its view bind group and
//! samples the prefiltered cube (at a roughness-selected mip) weighted by the
//! DFG term for the real split-sum specular reflection.

mod abi;
mod bind_groups;
mod dispatch;
mod extract;
mod pipeline;
mod resources;

#[cfg(test)]
mod shader_tests;

pub(crate) use bind_groups::{prepare_dfg_lut_bind_group, prepare_env_prefilter_bind_groups};
pub(crate) use bind_groups::EnvPrefilterBindGroups;
pub(crate) use dispatch::{dfg_lut_precompute_pass, env_prefilter_precompute_pass};
pub(crate) use extract::{extract_ibl_source, ExtractedIblSource};
pub(crate) use pipeline::{init_brdf_lut_pipeline, init_env_prefilter_pipeline};
pub(crate) use resources::{
    init_dfg_lut_texture, init_prefiltered_env_map, DfgLutTexture, PrefilteredEnvironmentMap,
};
