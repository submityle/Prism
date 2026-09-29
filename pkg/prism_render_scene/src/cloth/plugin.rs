//! The `bevy` plugin that installs the `GPU` cloth compute subsystem.
//!
//! This plugin wires the three cohesive halves of the module into a running
//! app:
//!
//! * It embeds the three `WESL` compute shaders
//!   (`cloth_sim.wesl`, `cloth_collision.wesl`, `cloth_embed.wesl`) so
//!   [`init_cloth_compute_pipelines`](super::pipeline::init_cloth_compute_pipelines)
//!   can load them by their stable asset paths and `naga` validates them the
//!   moment the render app boots.
//! * It builds the eleven pipelines and five bind-group layouts once at
//!   `RenderStartup`, inserting the shared
//!   [`ClothComputePipelines`](super::pipeline::ClothComputePipelines) resource.
//! * It installs the [`ClothGpuPieces`](super::resources::ClothGpuPieces)
//!   render resource (empty by default) and schedules the
//!   [`dispatch_cloth`](super::dispatch::dispatch_cloth) compute node into the
//!   `Core3d` graph before the main pass, matching every other Prism compute
//!   pass.
//!
//! The extract stage that snapshots main-world garments into `ClothGpuPieces`
//! is a deliberately separate concern: until a main-world cloth component is
//! designed the resource stays empty and the dispatch node is an honest no-op
//! (it never fabricates a solve). The pipelines still build and the shaders
//! still validate, so this plugin is the real, load-bearing wiring for the
//! device-side solver rather than a placeholder.

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{RenderApp, RenderStartup};

use super::dispatch::dispatch_cloth;
use super::pipeline::init_cloth_compute_pipelines;
use super::resources::ClothGpuPieces;

/// Installs the `GPU` cloth compute subsystem into an app.
///
/// Idempotent at the call site: the top-level scene plugin guards against a
/// double add, matching the sibling render sub-plugins.
#[derive(Default)]
pub(crate) struct ClothPlugin;

impl Plugin for ClothPlugin {
    fn build(&self, app: &mut App) {
        // Embed the three compute shaders next to this module so the pipeline
        // init system can load them by their stable `../shaders/*.wesl` asset
        // paths regardless of the working directory.
        embedded_asset!(app, "../shaders/cloth_sim.wesl");
        embedded_asset!(app, "../shaders/cloth_collision.wesl");
        embedded_asset!(app, "../shaders/cloth_embed.wesl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            // Default-empty: the extract stage (a separate slice) rebuilds this
            // each frame; an empty set makes `dispatch_cloth` a genuine no-op.
            .init_resource::<ClothGpuPieces>()
            // Build the eleven pipelines + five layouts once, then insert the
            // shared `ClothComputePipelines` resource the dispatch node reads.
            .add_systems(RenderStartup, init_cloth_compute_pipelines)
            // Record the solve in the `Core3d` graph before the main pass, the
            // same ordering every other Prism compute pass uses.
            .add_systems(
                bevy_core_pipeline::Core3d,
                dispatch_cloth.before(bevy_core_pipeline::Core3dSystems::MainPass),
            );
    }
}
