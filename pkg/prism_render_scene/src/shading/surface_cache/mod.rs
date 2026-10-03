//! Lumen-style surfel radiance surface cache: the render-world plumbing that
//! runs the persistent surfel cache on device.
//!
//! The `CPU` golden lives in [`prism_render_shading::gi::surface_cache`] and
//! the shader twins in `shaders/surface_cache_{alloc,update,spatial_filter,
//! coverage}.wesl` plus `shaders/surface_cache_composite.wesl`; this module is
//! the plumbing that schedules them. Unlike [`super::world_space_gi`] the
//! surface cache is *persistent across frames*: the per-surfel radiance `EMA`
//! state must survive the per-frame render-world entity rebuild, so the surfel
//! atlas buffers live in a [`RetainedViewEntity`]-keyed [`SurfaceCacheBuffers`]
//! `Resource` (mirroring the virtual-shadow persistent caches) rather than a
//! per-view `Component`. The cache is read and written in the same frame's
//! passes, so there is no +1-frame latency.
//!
//! The pipeline is four same-frame compute passes over the persistent surfel
//! buffer plus a two-pass composite:
//!
//! * `surface_cache_alloc_main` seeds one scratch surfel per screen tile from
//!   the `SSR` prepass depth / normal and the pre-exposed scene colour.
//! * `surface_cache_update_main` blends each scratch surfel into the persistent
//!   surfel via the golden confidence-weighted `EMA`, resetting history on
//!   disocclusion.
//! * `surface_cache_spatial_filter_main` runs the golden bilateral filter over
//!   geometrically compatible neighbour surfels to hide Monte-Carlo noise.
//! * `surface_cache_coverage_main` gathers the filtered surfels per pixel and
//!   writes the coverage-weighted diffuse `GI` into an `rgba16float` export.
//!
//! The export (`gi_out`) is then folded back over `scene_color` under the same
//! energy-conserving substitution the world-space `GI` / `SSGI` composites use
//! (`scene = base + confidence * albedo * (gi_out - ambient)`), swapping the
//! flat `IBL` ambient for the surfel gather under confidence.
//!
//! The subsystem is opt-in (see [`PrismSurfaceCacheSettings`]); when disabled
//! the passes allocate nothing and dispatch nothing.

mod abi;
mod bind_groups;
mod composite;
mod dispatch;
mod pipeline;
mod resources;
mod settings;

pub(crate) use bind_groups::prepare_surface_cache_bind_groups;
pub(crate) use composite::{
    init_surface_cache_composite_pipeline, prepare_surface_cache_composite_bind_groups,
    surface_cache_composite_pass,
};
pub(crate) use dispatch::surface_cache_pass;
pub(crate) use pipeline::init_surface_cache_pipeline;
pub(crate) use resources::{prepare_surface_cache_resources, SurfaceCacheBuffers};
pub(crate) use settings::PrismSurfaceCacheSettings;

#[cfg(test)]
mod shader_tests;
