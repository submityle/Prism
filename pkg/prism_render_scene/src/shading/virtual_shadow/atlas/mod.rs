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
//! Until the draw / resolve wiring lands this resource has no in-crate consumer
//! outside its own tests, so the submodule and its re-exports are
//! `dead_code`-/`unused`-expected rather than trimmed. **Remove these `expect`
//! guards when the plugin / graph wiring lands** (they become unfulfilled
//! expectations, hence warnings, once a real consumer exists).

#[expect(
    dead_code,
    reason = "physical-atlas resource whose only non-test consumer is the draw / resolve wiring landing in a later slice; remove when wired"
)]
mod resources;

// Caster depth raster fill (Slice A): the pass that rasterizes shadow casters
// into each resident physical page's atlas tile. Every symbol these modules
// expose is `pub(crate)` and is consumed only by the parent slice's plugin /
// render-graph wiring (out of scope for this module).
//
// `abi`, `bind_groups`, and `projection` are pure leaf modules whose items are
// referenced transitively by the re-exported `dispatch` / `pipeline` / `raster`
// surface (and by their own unit tests), so they are live in every build and
// need no guard. `dispatch`, `pipeline`, and `raster` additionally carry private
// records / helpers with no in-crate consumer until the parent wiring schedules
// them, so those three keep a module-level `dead_code` guard. **Remove each guard
// when the parent wiring lands** -- the expectation then goes unfulfilled and
// flags the now-stale guard.
mod abi;
mod bind_groups;
#[expect(
    dead_code,
    reason = "caster-depth feeder systems + render-graph node; scheduled by the parent VSM raster wiring slice added separately. Remove when that wiring lands"
)]
mod dispatch;
#[expect(
    dead_code,
    reason = "caster-depth device pipeline + per-page uniform buffer; initialised by the parent VSM raster wiring slice added separately. Remove when that wiring lands"
)]
mod pipeline;
mod projection;
#[expect(
    dead_code,
    reason = "caster-depth data records + per-view components + transient targets; inserted / consumed by the parent VSM raster wiring slice added separately. Remove when that wiring lands"
)]
mod raster;

#[cfg(test)]
mod shader_tests;

// Re-exported for the parent module (and the upcoming raster / resolve wiring);
// `prepare_vsm_physical_atlas` is registered by the plugin, the rest are consumed
// by the physical-atlas raster pass.
pub(crate) use resources::{
    atlas_tile_origin, physical_pages_per_edge, prepare_vsm_physical_atlas, ViewVsmPhysicalAtlas,
    VsmPhysicalAtlasCache,
};

// The caster-depth pass's public surface for the parent wiring slice: the shader
// / pipeline init hooks, the three feeder systems, the render-graph node, and the
// resources / per-view components the parent inserts and schedules. Unused until
// `virtual_shadow/mod.rs` re-exports them (out of scope here); remove the guard
// when that wiring lands.
#[expect(
    unused_imports,
    reason = "caster-depth public surface re-exported by the parent VSM raster wiring slice added separately; remove when that wiring lands"
)]
pub(crate) use dispatch::{
    prepare_vsm_caster_depth_targets, prepare_vsm_caster_depth_views, queue_vsm_caster_depth,
    vsm_caster_depth_pass,
};
#[expect(
    unused_imports,
    reason = "caster-depth public surface re-exported by the parent VSM raster wiring slice added separately; remove when that wiring lands"
)]
pub(crate) use pipeline::{
    init_vsm_caster_depth_pipeline, register_vsm_caster_depth_shader, VsmCasterDepthPipeline,
    VsmCasterDepthPipelineKey, VsmCasterDepthViewUniform,
};
#[expect(
    unused_imports,
    reason = "caster-depth public surface re-exported by the parent VSM raster wiring slice added separately; remove when that wiring lands"
)]
pub(crate) use raster::{
    ViewVsmCasterDepth, ViewVsmCasterPages, VsmCasterDepthDrawList, VsmCasterDepthTargets,
    VsmCasterPage,
};
