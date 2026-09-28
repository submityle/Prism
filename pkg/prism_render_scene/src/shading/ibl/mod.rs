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
//! cohesive files as each slice lands (the DFG table's pipeline, bind groups,
//! and dispatch arrive with the resolve consumption slice).

#[cfg(test)]
mod shader_tests;
