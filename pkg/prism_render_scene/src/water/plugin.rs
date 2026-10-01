//! The `bevy` plugin that installs the `GPU` water compute subsystem.
//!
//! This plugin wires the cohesive halves of the module into a running app:
//!
//! * It embeds the five `WESL` compute shaders (`water_ocean.wesl`,
//!   `water_flip.wesl`, `water_pbf.wesl`, `water_surface.wesl`,
//!   `water_render_fx.wesl`) so
//!   [`init_water_compute_pipelines`](super::pipeline::init_water_compute_pipelines)
//!   can load them by their stable asset paths and `naga` validates all sixteen
//!   compute entry points the moment the render app boots. (The `water.wesl`
//!   surface `BSDF` library is a shading-side asset owned by the shading plugin,
//!   not a compute kernel, so it is not embedded here.)
//! * It builds the sixteen pipelines and twelve bind-group layouts once at
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
//!   schedule, allocates its resident buffers and twelve bind groups and pushes
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
use bevy_render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

use super::body::ExtractedWater;
use super::dispatch::dispatch_water;
use super::extract::extract_water_bodies;
use super::pipeline::init_water_compute_pipelines;
use super::prepare::prepare_water_bodies;
use super::resources::WaterGpuBodies;
use super::surface_pipeline::init_water_surface_pipelines;

/// Installs the `GPU` water compute subsystem into an app.
///
/// Idempotent at the call site: the top-level scene plugin guards against a
/// double add, matching the sibling render sub-plugins.
#[derive(Default)]
pub(crate) struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        // Embed the five compute shaders plus the surface raster shader next
        // to this module so the pipeline init systems can load them by their
        // stable `../shaders/*.wesl` asset paths regardless of the working
        // directory.
        embedded_asset!(app, "../shaders/water_ocean.wesl");
        embedded_asset!(app, "../shaders/water_flip.wesl");
        embedded_asset!(app, "../shaders/water_pbf.wesl");
        embedded_asset!(app, "../shaders/water_surface.wesl");
        embedded_asset!(app, "../shaders/water_render_fx.wesl");
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
            .add_systems(RenderStartup, init_water_surface_pipelines)
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
            // Record the solve in the `Core3d` graph before the main pass, the
            // same ordering every other Prism compute pass uses.
            .add_systems(
                bevy_core_pipeline::Core3d,
                dispatch_water.before(bevy_core_pipeline::Core3dSystems::MainPass),
            );
    }
}
