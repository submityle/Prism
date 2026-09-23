use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    init_gpu_resource,
    renderer::{RenderGraph, RenderGraphSystems},
    sync_world::SyncToRenderWorld,
    ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

use crate::{
    buffers::{
        prepare_gpu_scene_bind_group, write_gpu_scene_buffers, GpuSceneBindGroup, GpuSceneBuffers,
    },
    compare::{compare_scene_mirror, GpuSceneParityDiagnostics},
    completion::{reclaim_completed_handles, track_submission, GpuCompletionTracker},
    diagnostics::{GpuSceneDiagnostics, GpuSceneUploadSettings},
    extract::{
        apply_extracted_scene_changes, extract_scene_instances, retire_unused_geometry,
        PrismGpuSceneEntity,
    },
    scene::RenderGpuScene,
};

/// Installs the retained GPU Scene into Bevy's render sub-application.
pub struct PrismGpuScenePlugin;

impl PrismGpuScenePlugin {
    /// Component bundle that opts a mesh into the retained scene and Bevy's
    /// render-world synchronization without changing Bevy's required-component
    /// registrations.
    pub fn entity(config: PrismGpuSceneEntity) -> (PrismGpuSceneEntity, SyncToRenderWorld) {
        (config, SyncToRenderWorld::default())
    }
}

/// Controls whether opt-in entities use the retained GPU Scene or remain on
/// the legacy renderer-only path.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuSceneMode {
    /// Stops extraction/consumption and keeps the legacy renderer active.
    Disabled,
    /// Enables the GPU Scene; installed consumers replace their legacy phase.
    #[default]
    Enabled,
    /// Enables the same consumers while publishing CPU/GPU parity diagnostics.
    /// Pixel A/B rendering is intentionally owned by tooling, not duplicated
    /// into the production opaque phase.
    Compare,
}

impl Plugin for PrismGpuScenePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::material::PrismMaterialPlugin>() {
            app.add_plugins(crate::material::PrismMaterialPlugin);
        }
        embedded_asset!(app, "shaders/gpu_scene.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<GpuSceneMode>()
            .init_resource::<GpuSceneDiagnostics>()
            .init_resource::<GpuSceneParityDiagnostics>()
            .init_resource::<GpuSceneUploadSettings>()
            .init_resource::<crate::extract::lifecycle::ExtractionClock>()
            .init_resource::<RenderGpuScene>()
            .init_resource::<GpuCompletionTracker>()
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<GpuSceneBuffers>,
                    init_gpu_resource::<GpuSceneBindGroup>,
                    rebuild_gpu_scene_after_device_startup,
                )
                    .chain(),
            )
            .add_systems(
                ExtractSchedule,
                (
                    extract_scene_instances,
                    crate::material::systems::invalidate_scene_materials
                        .after(crate::material::systems::extract_standard_materials)
                        .after(extract_scene_instances),
                    retire_unused_geometry,
                ),
            )
            .add_systems(
                Render,
                (
                    apply_extracted_scene_changes.in_set(RenderSystems::PrepareResources),
                    compare_scene_mirror
                        .after(apply_extracted_scene_changes)
                        .in_set(RenderSystems::PrepareResources),
                    write_gpu_scene_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_gpu_scene_bind_group
                        .after(write_gpu_scene_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                RenderGraph,
                (
                    track_submission.in_set(RenderGraphSystems::Finish),
                    reclaim_completed_handles
                        .after(track_submission)
                        .in_set(RenderGraphSystems::Finish),
                    crate::material::systems::reclaim_completed_materials
                        .after(track_submission)
                        .in_set(RenderGraphSystems::Finish),
                ),
            );
    }
}

fn rebuild_gpu_scene_after_device_startup(
    mut scene: bevy_ecs::prelude::ResMut<RenderGpuScene>,
    mut buffers: bevy_ecs::prelude::ResMut<GpuSceneBuffers>,
    mut diagnostics: bevy_ecs::prelude::ResMut<GpuSceneDiagnostics>,
) {
    scene.rebuild_gpu_buffers(&mut buffers);
    diagnostics.buffer_version = scene.buffer_version();
    diagnostics.buffer_rebuilds = diagnostics.buffer_rebuilds.saturating_add(1);
}
