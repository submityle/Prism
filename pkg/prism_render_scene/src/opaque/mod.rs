mod draw;
mod indirect;
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
pub use pipeline::GpuSceneDebugView;
use pipeline::{init_opaque_pipeline, GpuSceneOpaquePipeline};
use queue::queue_gpu_scene_opaque;

/// Orders the Prism opaque replacement after Bevy's material queue.
#[derive(bevy_ecs::schedule::SystemSet, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct GpuSceneOpaqueQueue;

/// Controls whether the experimental opaque consumer replaces Bevy PBR.
#[derive(bevy_ecs::resource::Resource, Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSceneOpaqueEnabled(pub bool);

/// Opt-in production gate for the indirect opaque consumer. Capability
/// support alone cannot prove runtime draw parity, so the default is disabled.
#[derive(bevy_ecs::resource::Resource, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSceneOpaqueIndirectEnabled(pub bool);

impl Default for GpuSceneOpaqueEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// Standard opaque mesh consumer backed by Prism's retained GPU Scene.
pub struct PrismGpuSceneOpaquePlugin;

impl Plugin for PrismGpuSceneOpaquePlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<bevy_pbr::PbrPlugin>(),
            "PrismGpuSceneOpaquePlugin must be added after Bevy PbrPlugin"
        );
        pipeline::embed_opaque_shader(app);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SpecializedMeshPipelines<GpuSceneOpaquePipeline>>()
            .init_resource::<GpuSceneDebugView>()
            .init_resource::<GpuSceneOpaqueEnabled>()
            .init_resource::<GpuSceneOpaqueIndirectEnabled>()
            .add_render_command::<Opaque3d, DrawGpuSceneOpaque>()
            .add_systems(
                RenderStartup,
                init_opaque_pipeline.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                queue_gpu_scene_opaque
                    .in_set(RenderSystems::QueueMeshes)
                    .in_set(GpuSceneOpaqueQueue)
                    .after(bevy_pbr::queue_material_meshes),
            );
    }
}
