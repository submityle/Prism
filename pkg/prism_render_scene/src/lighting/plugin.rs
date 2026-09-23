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
    extract::{extract_lights, ExtractedLights},
    systems::{prepare_light_bind_group, rebuild_light_buffers, write_light_buffers},
};

/// Extracts Bevy lights into the flat GPU light ABI and keeps the storage
/// buffers and bind group up to date for the resolve pass.
pub struct PrismLightingPlugin;

impl Plugin for PrismLightingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "../shaders/lighting.wesl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<ExtractedLights>()
            .add_systems(
                RenderStartup,
                (
                    init_gpu_resource::<LightGpuBuffers>,
                    init_gpu_resource::<LightBindGroup>,
                )
                    .chain(),
            )
            .add_systems(ExtractSchedule, extract_lights)
            .add_systems(
                Render,
                (
                    rebuild_light_buffers.in_set(RenderSystems::PrepareResources),
                    write_light_buffers.in_set(RenderSystems::PrepareResourcesFlush),
                    prepare_light_bind_group
                        .after(write_light_buffers)
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            );
    }
}
