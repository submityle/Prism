use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{init_gpu_resource, Render, RenderApp, RenderStartup, RenderSystems};

use super::{
    buffers::UnifiedVisibilityBuffers,
    graph::visibility_frame_graph,
    runtime::{
        PrismVisibilityDiagnostics, UnifiedVisibilityEnabled, UnifiedVisibilitySettings,
        UnifiedVisibilityState, VisibilityFrameGraph,
    },
    systems::{build_unified_visibility, rebuild_unified_visibility, upload_unified_visibility},
};

pub struct PrismVisibilityPlugin;

impl Plugin for PrismVisibilityPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "../shaders/visibility.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let compiled_graph = visibility_frame_graph()
            .compile()
            .expect("Prism visibility frame graph must be valid");
        render_app
            .init_resource::<UnifiedVisibilityEnabled>()
            .init_resource::<UnifiedVisibilitySettings>()
            .init_resource::<UnifiedVisibilityState>()
            .init_resource::<PrismVisibilityDiagnostics>()
            .insert_resource(VisibilityFrameGraph {
                compiled: compiled_graph,
            })
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<UnifiedVisibilityBuffers>,
                    rebuild_unified_visibility,
                )
                    .chain(),
            )
            .add_systems(
                Render,
                (
                    build_unified_visibility.in_set(RenderSystems::PrepareResources),
                    upload_unified_visibility
                        .after(build_unified_visibility)
                        .in_set(RenderSystems::PrepareResourcesFlush),
                ),
            );
    }
}
