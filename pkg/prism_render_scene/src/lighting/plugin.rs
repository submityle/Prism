//! Installs the GPU light system into Bevy's render sub-application.
//!
//! The plugin mirrors the material plugin's lifecycle: GPU resources are
//! created at [`RenderStartup`] (after the device exists), lights are extracted
//! every frame in [`ExtractSchedule`], and the buffers are staged, flushed, and
//! bound across the `PrepareResources`, `PrepareResourcesFlush`, and
//! `PrepareBindGroups` render sets so the resolve pass sees a valid bind group.

use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    init_gpu_resource, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

use super::{
    bindings::LightBindGroup,
    buffers::LightGpuBuffers,
    cluster::{
        extract_cluster_view, prepare_cluster_bind_group, rebuild_cluster_buffers,
        write_cluster_buffers, ClusterBindGroup, ClusterConfig, ClusterGpuBuffers,
        ExtractedClusterView,
    },
    extract::{extract_lights, ExtractedLights},
    probe::EnvironmentProbeCache,
    stylized_config::StylizedLighting,
    systems::{prepare_light_bind_group, rebuild_light_buffers, write_light_buffers},
};

/// Extracts Bevy lights into the flat GPU light ABI and keeps the storage
/// buffers and bind group up to date for the resolve pass.
pub struct PrismLightingPlugin;

impl Plugin for PrismLightingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "../shaders/lighting.wesl");
        // The stylized (NPR) look control lives in the main world; extraction
        // reads it via `Extract`.  Its default is the legacy toon lobe, so
        // registering it here is a behavioural no-op until an app mutates it.
        app.init_resource::<StylizedLighting>();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedLights>()
            .init_resource::<EnvironmentProbeCache>()
            .init_resource::<ExtractedClusterView>()
            .init_resource::<ClusterConfig>()
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<LightGpuBuffers>,
                    init_gpu_resource::<LightBindGroup>,
                    init_gpu_resource::<ClusterGpuBuffers>,
                    init_gpu_resource::<ClusterBindGroup>,
                )
                    .chain(),
            )
            .add_systems(ExtractSchedule, (extract_lights, extract_cluster_view))
            .add_systems(
                Render,
                (
                    rebuild_light_buffers.in_set(RenderSystems::PrepareResources),
                    write_light_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_light_bind_group
                        .after(write_light_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                    // The clustered tables build from the same extracted lights
                    // and share the resolve pass's prepare lifecycle.
                    rebuild_cluster_buffers.in_set(RenderSystems::PrepareResources),
                    write_cluster_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_cluster_bind_group
                        .after(write_cluster_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}
