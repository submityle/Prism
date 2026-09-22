use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    renderer::{RenderGraph, RenderGraphSystems},
    sync_world::SyncToRenderWorld,
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
};

use crate::{
    buffers::{write_gpu_scene_buffers, GpuSceneBuffers},
    completion::{reclaim_completed_handles, track_submission, GpuCompletionTracker},
    diagnostics::GpuSceneDiagnostics,
    extract::{apply_extracted_scene_changes, extract_scene_instances, PrismGpuSceneEntity},
    scene::RenderGpuScene,
};

/// Installs the retained GPU Scene into Bevy's render sub-application.
pub struct PrismGpuScenePlugin;

/// Controls whether opt-in entities use the retained GPU Scene or remain on
/// the legacy renderer-only path.
#[derive(Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuSceneMode {
    Disabled,
    #[default]
    Enabled,
    Compare,
}

impl Plugin for PrismGpuScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<PrismGpuSceneEntity, SyncToRenderWorld>();
        embedded_asset!(app, "shaders/gpu_scene.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<GpuSceneMode>()
            .init_resource::<GpuSceneDiagnostics>()
            .init_resource::<RenderGpuScene>()
            .init_gpu_resource::<GpuSceneBuffers>()
            .init_resource::<GpuCompletionTracker>()
            .add_systems(ExtractSchedule, extract_scene_instances)
            .add_systems(
                Render,
                (
                    apply_extracted_scene_changes.in_set(RenderSystems::PrepareResources),
                    write_gpu_scene_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                ),
            )
            .add_systems(
                RenderGraph,
                (
                    track_submission.in_set(RenderGraphSystems::Finish),
                    reclaim_completed_handles
                        .after(track_submission)
                        .in_set(RenderGraphSystems::Finish),
                ),
            );
    }
}
