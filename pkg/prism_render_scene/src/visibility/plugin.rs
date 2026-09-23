use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    init_gpu_resource,
    renderer::{RenderGraph, RenderGraphSystems},
    Render, RenderApp, RenderStartup, RenderSystems,
};

use super::{
    buffers::UnifiedVisibilityBuffers,
    gpu::{
        init_visibility_compute_pipeline, prepare_visibility_compute_bind_group,
        VisibilityComputeBindGroup,
    },
    graph::visibility_frame_graph,
    hzb::{inspect_hzb_history, install_hzb_schedule, prepare_hzb_history},
    hzb_gpu::{init_hzb_visibility_pipeline, inspect_hzb_visibility_pipeline},
    readback::{
        collect_visibility_parity_readback, map_submitted_visibility_parity_readback,
        request_visibility_parity_readback, VisibilityParityReadback,
    },
    runtime::{
        PrismVisibilityDiagnostics, UnifiedVisibilityEnabled, UnifiedVisibilitySettings,
        UnifiedVisibilityState, VisibilityFrameGraph,
    },
    systems::{
        build_unified_visibility, dispatch_unified_visibility, rebuild_unified_visibility,
        upload_unified_visibility,
    },
};

pub struct PrismVisibilityPlugin;

fn detect_visibility_capabilities(
    device: bevy_ecs::prelude::Res<bevy_render::renderer::RenderDevice>,
    mut settings: bevy_ecs::prelude::ResMut<UnifiedVisibilitySettings>,
) {
    settings.indirect_first_instance = device
        .features()
        .contains(bevy_render::render_resource::WgpuFeatures::INDIRECT_FIRST_INSTANCE);
}

impl Plugin for PrismVisibilityPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "../shaders/visibility.wesl");
        embedded_asset!(app, "../shaders/hzb_visibility.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        install_hzb_schedule(render_app);
        let compiled_graph = visibility_frame_graph()
            .compile()
            .expect("Prism visibility frame graph must be valid");
        render_app
            .init_resource::<UnifiedVisibilityEnabled>()
            .init_resource::<UnifiedVisibilitySettings>()
            .init_resource::<UnifiedVisibilityState>()
            .init_resource::<PrismVisibilityDiagnostics>()
            .init_resource::<VisibilityParityReadback>()
            .insert_resource(VisibilityFrameGraph {
                compiled: compiled_graph,
            })
            .add_systems(
                RenderStartup,
                (
                    detect_visibility_capabilities,
                    init_gpu_resource::<UnifiedVisibilityBuffers>,
                    init_gpu_resource::<VisibilityComputeBindGroup>,
                    init_visibility_compute_pipeline,
                    init_hzb_visibility_pipeline,
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
                    prepare_visibility_compute_bind_group
                        .after(upload_unified_visibility)
                        .after(crate::geometry::prepare_geometry_bind_group)
                        .in_set(RenderSystems::PrepareBindGroups),
                    (prepare_hzb_history, inspect_hzb_history)
                        .chain()
                        .in_set(RenderSystems::PrepareResources),
                    inspect_hzb_visibility_pipeline.in_set(RenderSystems::Prepare),
                ),
            );
        render_app.add_systems(
            RenderGraph,
            (
                collect_visibility_parity_readback.in_set(RenderGraphSystems::Begin),
                dispatch_unified_visibility
                    .after(collect_visibility_parity_readback)
                    .in_set(RenderGraphSystems::Begin),
                request_visibility_parity_readback
                    .after(dispatch_unified_visibility)
                    .in_set(RenderGraphSystems::Begin),
                map_submitted_visibility_parity_readback.in_set(RenderGraphSystems::Finish),
            ),
        );
    }
}
