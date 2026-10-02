//! The `bevy` plugin that installs the `GPU` water compute subsystem.
//!
//! This plugin wires the cohesive halves of the module into a running app:
//!
//! * It embeds the `WESL` compute shaders (`water_ocean.wesl`,
//!   `water_surface_reconstruct.wesl`, `water_pbf.wesl`, `water_surface.wesl`,
//!   `water_render_fx.wesl`, the ocean `FFT` pair, the staggered `MAC`
//!   `FLIP`/`APIC` chain and the surface mesh/raster shaders) so
//!   [`init_water_compute_pipelines`](super::pipeline::init_water_compute_pipelines)
//!   can load them by their stable asset paths and `naga` validates every
//!   compute entry point the moment the render app boots. (The `water.wesl`
//!   surface `BSDF` library is a shading-side asset owned by the shading plugin,
//!   not a compute kernel, so it is not embedded here.)
//! * It builds every water pipeline and its bind-group layouts once at
//!   `RenderStartup`, inserting the shared
//!   [`WaterComputePipelines`](super::pipeline::WaterComputePipelines) resource.
//! * It installs the [`WaterGpuBodies`](super::resources::WaterGpuBodies) and
//!   [`ExtractedWater`](super::body::ExtractedWater) render resources (empty by
//!   default) and schedules the
//!   [`dispatch_water`](super::dispatch::dispatch_water) compute node into the
//!   `Core3d` graph before the main pass, matching every other Prism compute
//!   pass.
//! * It runs the full end-to-end loop each frame:
//!   [`extract_water_bodies`](super::extract::extract_water_bodies) in
//!   [`ExtractSchedule`] snapshots the main-world
//!   [`WaterBody`](super::body::WaterBody)s into `ExtractedWater`, then
//!   [`prepare_water_bodies`](super::prepare::prepare_water_bodies) in
//!   [`RenderSystems::PrepareResources`] expands each body's golden dispatch
//!   schedule, allocates its resident buffers and bind groups and pushes
//!   the resident body the dispatch node records.
//!
//! When no water body is spawned every stage is an honest no-op: the extracted
//! set is empty, the prepare stage builds no body and the dispatch node records
//! nothing (it never fabricates a solve). The pipelines still build and the
//! shaders still validate, so this plugin is the real, load-bearing wiring for
//! the device-side solver rather than a placeholder.

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::render_resource::SpecializedRenderPipelines;
use bevy_render::{
    init_gpu_resource, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

use super::body::ExtractedWater;
use super::dispatch::dispatch_water;
use super::extract::extract_water_bodies;
use super::pipeline::init_water_compute_pipelines;
use super::prepare::prepare_water_bodies;
use super::resources::WaterGpuBodies;
use super::surface_draw::draw_water_surface;
use super::surface_motion::{prepare_water_surface_motion, WaterMotionHistory};
use super::surface_pipeline::{
    init_water_surface_pipelines, prepare_water_surface_pipelines, WaterSurfacePipelines,
};
use super::surface_ssr::init_water_ssr_fallback;
use super::surface_vsm::init_water_vsm_fallback;
use super::surface_world_space_gi::init_water_world_space_gi_fallback;
use crate::lighting::LightBindGroup;

/// Installs the `GPU` water compute subsystem into an app.
///
/// Idempotent at the call site: the top-level scene plugin guards against a
/// double add, matching the sibling render sub-plugins.
#[derive(Default)]
pub(crate) struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        // Embed every compute shader the pipeline init systems load through
        // `load_embedded_asset!`, plus the surface raster shader, next to this
        // module so they resolve by their stable `../shaders/*.wesl` asset
        // paths regardless of the working directory. Every path here must stay
        // in lockstep with the `load_embedded_asset!` calls in
        // `pipeline.rs::init_water_compute_pipelines`; a missing embed panics
        // at pipeline-init time on a real device even though the sandbox's
        // `include_str!`-based shader tests never exercise this path.
        embedded_asset!(app, "../shaders/water_ocean.wesl");
        embedded_asset!(app, "../shaders/water_surface_reconstruct.wesl");
        embedded_asset!(app, "../shaders/water_pbf.wesl");
        embedded_asset!(app, "../shaders/water_surface.wesl");
        embedded_asset!(app, "../shaders/water_render_fx.wesl");
        embedded_asset!(app, "../shaders/water_spectrum_fft.wesl");
        embedded_asset!(app, "../shaders/water_butterfly.wesl");
        embedded_asset!(app, "../shaders/water_flip_mac_p2g.wesl");
        embedded_asset!(app, "../shaders/water_flip_mac.wesl");
        embedded_asset!(app, "../shaders/water_flip_mac_g2p.wesl");
        embedded_asset!(app, "../shaders/water_surface_mesh.wesl");
        embedded_asset!(app, "../shaders/water_surface_raster.wesl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            // Default-empty: the extract stage rebuilds the resident bodies
            // each frame from the extracted bodies; an empty set makes
            // `dispatch_water` a genuine no-op rather than a fabricated solve.
            .init_resource::<WaterGpuBodies>()
            // Default-empty snapshot the extract stage refills every frame.
            .init_resource::<ExtractedWater>()
            // Build the sixteen pipelines + twelve layouts once, then insert the
            // shared `WaterComputePipelines` resource the prepare and dispatch
            // stages read.
            .add_systems(RenderStartup, init_water_compute_pipelines)
            // Cache the per-view, per-target-format specialized surface raster
            // pipelines, matching the shading composite pass's specialization
            // registry.
            .init_resource::<SpecializedRenderPipelines<WaterSurfacePipelines>>()
            // Build after the shared light table resource exists: the surface
            // pipeline clones `LightBindGroup`'s `group(1)` layout descriptor so
            // its fragment stage reads the engine's directional/punctual/
            // environment tables, exactly like the opaque resolve pass.
            .add_systems(
                RenderStartup,
                init_water_surface_pipelines.after(init_gpu_resource::<LightBindGroup>),
            )
            // Build the surface pass's `@group(2)` virtual-shadow-map fallback
            // objects (dummy page table + 1x1 atlas + sampler). Depends only on
            // `RenderDevice`, which is live by `RenderStartup`, so no ordering
            // constraint is needed.
            .add_systems(RenderStartup, init_water_vsm_fallback)
            // Likewise the SSR fallback Hi-Z view: a 1x1 `R32Float`
            // texture bound into `@group(3)` on any view without a
            // resident depth pyramid, with the march's `sample_enable`
            // bit cleared so the shader keeps the image-based reflection.
            .add_systems(RenderStartup, init_water_ssr_fallback)
            // And the world-space GI fallback probe buffer: a single dummy
            // probe bound into `@group(8)` on any view without a resident
            // `ViewWorldSpaceGi` field, with the gather's `sample_enable` bit
            // cleared so the shader skips the probe gather entirely.
            .add_systems(RenderStartup, init_water_world_space_gi_fallback)
            // Snapshot the main-world water bodies into the render world each
            // frame.
            .add_systems(ExtractSchedule, extract_water_bodies)
            // Turn each extracted body into a resident `GPU` body before the
            // dispatch node records the solve, in the standard
            // `PrepareResources` set every other Prism compute prepare uses.
            .add_systems(
                Render,
                prepare_water_bodies.in_set(RenderSystems::PrepareResources),
            )
            // Specialize the four surface frontends for every visibility-path
            // view and stash the concrete pipeline ids on the view, in the
            // standard `Prepare` set Bevy's own fullscreen passes use.
            .add_systems(
                Render,
                prepare_water_surface_pipelines.in_set(RenderSystems::Prepare),
            )
            // Per-view history of the previous-frame view-projection, refreshed
            // wholesale each frame so a vanished view cannot leak its matrix.
            .init_resource::<WaterMotionHistory>()
            // Build the per-view `@group(4)` motion-vector uniform (current +
            // previous `clip_from_world`) the surface raster's second render
            // target reads, in the same `Prepare` set as the pipeline
            // specialization so it is resident before the draw node records.
            .add_systems(
                Render,
                prepare_water_surface_motion.in_set(RenderSystems::Prepare),
            )
            // Record the solve in the `Core3d` graph before the main pass, the
            // same ordering every other Prism compute pass uses.
            .add_systems(
                bevy_core_pipeline::Core3d,
                dispatch_water.before(bevy_core_pipeline::Core3dSystems::MainPass),
            )
            // Rasterize the displaced surface *after* the shading composite (so
            // the opaque radiance it refracts is present in `scene_color`) and
            // before Bevy's post-process, matching the composite -> OIT order.
            .add_systems(
                bevy_core_pipeline::Core3d,
                draw_water_surface
                    .after(crate::shading::composite_shading)
                    .before(bevy_core_pipeline::Core3dSystems::PostProcess),
            );
    }
}
