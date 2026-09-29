//! Virtual-shadow-map physical page atlas -- the depth texture the resolve-side
//! sampler (`shaders/vsm_sample.wesl`) reads a resident page's stored NDC depth
//! from.
//!
//! Once the page-request pass (`super::page_mark`) has marked the pages a frame
//! touches and the allocator has made a bounded working set resident, those
//! resident physical pages are packed into one square depth texture: the
//! `physical_atlas`. `physical_pages` pages tile a grid `physical_pages_per_edge`
//! on a side, so physical page `P` occupies tile
//! `(P % physical_pages_per_edge, P / physical_pages_per_edge)` and the atlas
//! edge is `physical_pages_per_edge * page_size` texels. This is exactly the
//! layout `vsm_sample.wesl` addresses (its `physical_atlas` binding, whose `.r`
//! channel is the stored NDC depth) and the golden
//! [`prism_render_shading::shadow::virtual_sm`] sampling path assumes.
//!
//! This slice owns only the *resource*: the per-view atlas depth texture, its
//! sampling view, the filtering sampler `vsm_sample.wesl` binds at binding `2`,
//! and the tiling math (kept as pure, testable functions). It deliberately does
//! not run any pass -- the shadow-depth draw that fills the atlas tiles and the
//! resolve pass that samples it are wired by later slices.
//!
//! * [`resources`] -- the [`resources::ViewVsmPhysicalAtlas`] component, the
//!   [`resources::VsmPhysicalAtlasCache`] host resource, the
//!   [`resources::prepare_vsm_physical_atlas`] `PrepareResources` system, and
//!   the `atlas_tile_origin` / `physical_pages_per_edge` tiling functions.
//!
//! The resource, the caster-depth raster pass, and their re-exports are all
//! consumed by the parent `virtual_shadow` slice's plugin / render-graph wiring.

mod resources;

// Caster-depth raster fill: the pass that rasterizes shadow casters into each
// resident physical page's atlas tile. Every symbol these modules expose is
// `pub(crate)` and is consumed by the parent slice's plugin / render-graph
// wiring. `abi`, `bind_groups`, and `projection` are pure leaf modules whose
// items are referenced transitively by the re-exported `dispatch` / `pipeline`
// / `raster` surface (and by their own unit tests).
mod abi;
mod bind_groups;
mod dispatch;
mod pipeline;
mod projection;
mod raster;

#[cfg(test)]
mod shader_tests;

// Re-exported for the parent module: `prepare_vsm_physical_atlas` is registered
// by the plugin, the rest are consumed by the physical-atlas raster pass and the
// resolve VSM sampling bind group.
pub(crate) use resources::{
    atlas_tile_origin, physical_pages_per_edge, prepare_vsm_physical_atlas, ViewVsmPhysicalAtlas,
    VsmPhysicalAtlasCache,
};

// The caster-depth pass's public surface consumed by the parent
// `virtual_shadow/mod.rs` wiring: the shader / pipeline init hooks, the feeder
// systems, the render-graph node, and the resources / per-view components the
// plugin inserts and schedules. Symbols used only inside this `atlas` module
// (e.g. `VsmCasterDepthPipelineKey`, `ViewVsmCasterDepth`) are reached through
// their defining submodule and are intentionally not re-exported here.
pub(crate) use dispatch::{
    prepare_vsm_caster_depth_targets, prepare_vsm_caster_depth_views, queue_vsm_caster_depth,
    vsm_caster_depth_pass,
};
pub(crate) use pipeline::{
    init_vsm_caster_depth_pipeline, register_vsm_caster_depth_shader, VsmCasterDepthPipeline,
    VsmCasterDepthViewUniform,
};
pub(crate) use raster::{
    ViewVsmCasterPages, VsmCasterDepthDrawList, VsmCasterDepthTargets, VsmCasterPage,
};
