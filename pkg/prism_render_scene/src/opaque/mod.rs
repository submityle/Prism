mod draw;
mod pipeline;
mod queue;

use bevy_app::{App, Plugin};
use bevy_core_pipeline::core_3d::Opaque3d;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_pbr::MeshPipelineSystems;
use bevy_render::{
    render_phase::AddRenderCommand, render_resource::SpecializedMeshPipelines, Render, RenderApp,
    RenderStartup, RenderSystems,
};

use draw::DrawGpuSceneOpaque;
use pipeline::{init_opaque_pipeline, GpuSceneOpaquePipeline};
use queue::queue_gpu_scene_opaque;

/// Standard opaque mesh consumer backed by Prism's retained GPU Scene.
pub struct PrismGpuSceneOpaquePlugin;

impl Plugin for PrismGpuSceneOpaquePlugin {
    fn build(&self, app: &mut App) {
        pipeline::embed_opaque_shader(app);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SpecializedMeshPipelines<GpuSceneOpaquePipeline>>()
            .add_render_command::<Opaque3d, DrawGpuSceneOpaque>()
            .add_systems(
                RenderStartup,
                init_opaque_pipeline.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                queue_gpu_scene_opaque.in_set(RenderSystems::QueueMeshes),
            );
    }
}
