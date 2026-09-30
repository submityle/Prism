//! The `bevy` plugin that installs the `GPU` cloth compute subsystem.
//!
//! This plugin wires the three cohesive halves of the module into a running
//! app:
//!
//! * It embeds the five `WESL` compute shaders
//!   (`cloth_sim.wesl`, `cloth_collision.wesl`, `cloth_embed.wesl`,
//!   `cloth_aerodynamics_snapshot.wesl`, `cloth_aerodynamics.wesl`) so
//!   [`init_cloth_compute_pipelines`](super::pipeline::init_cloth_compute_pipelines)
//!   can load them by their stable asset paths and `naga` validates them the
//!   moment the render app boots.
//! * It builds the thirteen pipelines and seven bind-group layouts once at
//!   `RenderStartup`, inserting the shared
//!   [`ClothComputePipelines`](super::pipeline::ClothComputePipelines) resource.
//! * It installs the [`ClothGpuPieces`](super::resources::ClothGpuPieces) and
//!   [`ExtractedCloth`](super::garment::ExtractedCloth) render resources (empty
//!   by default) and schedules the
//!   [`dispatch_cloth`](super::dispatch::dispatch_cloth) compute node into the
//!   `Core3d` graph before the main pass, matching every other Prism compute
//!   pass.
//! * It installs the main-world screen-coverage estimator
//!   [`update_cloth_coverage`](super::coverage::update_cloth_coverage) in
//!   `PostUpdate` (after the camera projection refresh), which projects each
//!   garment's world bounds and writes back a live `0..=1` coverage that the
//!   render-world LOD gate resolves against.
//! * It runs the full end-to-end loop each frame:
//!   [`extract_cloth_garments`](super::extract::extract_cloth_garments) in
//!   [`ExtractSchedule`] snapshots the main-world
//!   [`ClothGarment`](super::garment::ClothGarment)s into `ExtractedCloth`, then
//!   [`prepare_cloth_pieces`](super::prepare::prepare_cloth_pieces) in
//!   [`RenderSystems::PrepareResources`] builds each garment's device-free solve
//!   plan, allocates its resident buffers and bind groups and pushes the
//!   resident piece the dispatch node records.
//!
//! When no garment is spawned every stage is an honest no-op: the extracted set
//! is empty, the prepare stage builds no piece and the dispatch node records
//! nothing (it never fabricates a solve). The pipelines still build and the
//! shaders still validate, so this plugin is the real, load-bearing wiring for
//! the device-side solver rather than a placeholder.

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::embedded_asset;
use bevy_camera::CameraUpdateSystems;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems};

use super::budget::ClothDeformationBudget;
use super::coverage::update_cloth_coverage;
use super::dispatch::dispatch_cloth;
use super::extract::extract_cloth_garments;
use super::garment::ExtractedCloth;
use super::pipeline::init_cloth_compute_pipelines;
use super::prepare::prepare_cloth_pieces;
use super::resources::ClothGpuPieces;

/// Installs the `GPU` cloth compute subsystem into an app.
///
/// Idempotent at the call site: the top-level scene plugin guards against a
/// double add, matching the sibling render sub-plugins.
#[derive(Default)]
pub(crate) struct ClothPlugin;

impl Plugin for ClothPlugin {
    fn build(&self, app: &mut App) {
        // Embed the five compute shaders next to this module so the pipeline
        // init system can load them by their stable `../shaders/*.wesl` asset
        // paths regardless of the working directory.
        embedded_asset!(app, "../shaders/cloth_sim.wesl");
        embedded_asset!(app, "../shaders/cloth_collision.wesl");
        embedded_asset!(app, "../shaders/cloth_embed.wesl");
        embedded_asset!(app, "../shaders/cloth_aerodynamics_snapshot.wesl");
        embedded_asset!(app, "../shaders/cloth_aerodynamics.wesl");

        // Main-world screen-coverage estimator: refresh every garment's projected
        // on-screen size each frame so the render-world LOD gate resolves against
        // a live coverage rather than a static authored value. It runs after the
        // camera projection is recomputed (`CameraUpdateSystems`) and before the
        // extract stage snapshots the garments into the render world.
        app.add_systems(PostUpdate, update_cloth_coverage.after(CameraUpdateSystems));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            // Default-empty: the extract stage rebuilds the resident pieces each
            // frame from the extracted garments; an empty set makes
            // `dispatch_cloth` a genuine no-op.
            .init_resource::<ClothGpuPieces>()
            // Default-empty snapshot the extract stage refills every frame.
            .init_resource::<ExtractedCloth>()
            // Shared per-frame deformation budget the prepare stage arbitrates
            // simulated garments against. Defaults to unlimited, so the budget
            // gate is a transparent pass-through until a scene lowers it.
            .init_resource::<ClothDeformationBudget>()
            // Build the thirteen pipelines + seven layouts once, then insert the
            // shared `ClothComputePipelines` resource the prepare and dispatch
            // stages read.
            .add_systems(RenderStartup, init_cloth_compute_pipelines)
            // Snapshot the main-world garments into the render world each frame.
            .add_systems(ExtractSchedule, extract_cloth_garments)
            // Turn each extracted garment into a resident `GPU` piece before the
            // dispatch node records the solve, in the standard
            // `PrepareResources` set every other Prism compute prepare uses.
            .add_systems(
                Render,
                prepare_cloth_pieces.in_set(RenderSystems::PrepareResources),
            )
            // Record the solve in the `Core3d` graph before the main pass, the
            // same ordering every other Prism compute pass uses.
            .add_systems(
                bevy_core_pipeline::Core3d,
                dispatch_cloth.before(bevy_core_pipeline::Core3dSystems::MainPass),
            );
    }
}
