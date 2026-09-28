//! Exposure histogram build: pipeline, per-view bind group, and the `Core3d`
//! dispatch node.
//!
//! First of the two auto-exposure passes and the GPU twin of the histogram half
//! of [`prism_render_shading::exposure`]. For every covered pixel of the
//! resolved HDR `scene_color` it computes the Rec.709 luminance, maps it to a
//! log2 bin over the artist's `[min, max]` window, and `atomicAdd`s into a
//! 64-bin storage histogram. The ragged tile edge is dropped by a per-pixel
//! coverage guard.
//!
//! It reads one bind group (group 0, matching `build_histogram` in
//! `shaders/exposure.wesl`):
//!
//! * `0` the resolved HDR `scene_color` (SSGI/SSR composited, pre-TAA — the true
//!   radiance to meter, `textureLoad`ed), and
//! * `1` the write-accumulated 64-bin histogram storage buffer.
//!
//! The log2 window + framebuffer extent travel in the
//! [`GpuExposureHistogramConfig`] immediate block. It runs after the SSGI/SSR
//! composite (so `scene_color` holds the final radiance) and before the resolve
//! pass that reduces the histogram, both ahead of the main pass so the exposure
//! multiplier is ready when the composite consumes it.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_sized, texture_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
        TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
};
use bevy_shader::Shader;
use prism_render_shading::HistogramRange;

use super::super::resources::ViewVisibilityBuffer;
use super::super::runtime::PrismShadingSettings;
use super::abi::{GpuExposureHistogramConfig, EXPOSURE_HISTOGRAM_WORKGROUP_SIZE};
use super::resources::ViewExposureBuffers;

/// Compute pipeline and its owned group-0 layout for the histogram build.
#[derive(Resource)]
pub(crate) struct ExposureHistogramPipeline {
    /// `build_histogram` compute entry point, specialized against the group-0
    /// layout and the 16-byte [`GpuExposureHistogramConfig`] immediate block.
    build: CachedComputePipelineId,
    /// group 0: the resolved HDR `scene_color` read + the write-accumulated
    /// histogram storage buffer.
    layout: BindGroupLayout,
}

/// group-0 layout mirroring `build_histogram`: one non-filterable float read
/// (`scene_color`, `textureLoad`ed) then the read-write 64-bin histogram
/// storage buffer.
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`ExposureHistogramPipeline`].
pub(crate) fn init_exposure_histogram_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism exposure histogram", &entries);
    let layout = device.create_bind_group_layout("prism exposure histogram", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/exposure.wesl");

    let build = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism exposure histogram".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuExposureHistogramConfig>() as u32,
        shader,
        entry_point: Some("build_histogram".into()),
        ..Default::default()
    });

    commands.insert_resource(ExposureHistogramPipeline { build, layout });
}

/// The histogram build's group-0 bind group for a single view. Present only
/// when exposure is enabled and the view has both a visibility buffer (for the
/// `scene_color` to meter) and its persistent exposure buffers.
#[derive(Component)]
pub(crate) struct ViewExposureHistogramBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building the histogram bind group for every view
/// with a resident `scene_color` and exposure buffers, gated on
/// `enable_exposure` so a disabled frame allocates nothing and the stale bind
/// group is dropped.
pub(crate) fn prepare_exposure_histogram_bind_groups(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    pipeline: Res<ExposureHistogramPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewExposureBuffers)>,
) {
    for (entity, visibility, exposure) in &views {
        if !settings.enable_exposure {
            commands
                .entity(entity)
                .remove::<ViewExposureHistogramBindGroup>();
            continue;
        }
        let group = device.create_bind_group(
            "prism exposure histogram",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                visibility.scene_color_view(),
                exposure.histogram_buffer().as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewExposureHistogramBindGroup { group });
    }
}

/// `Core3d` node recording the histogram build dispatch for every view.
///
/// Runs after the SSGI/SSR composite fills `scene_color` with the final
/// radiance and before the resolve pass reduces the histogram. Dispatches one
/// workgroup per 8x8 pixel tile; the shader's coverage guard drops the ragged
/// edge. Gated on `enable_exposure`.
pub(crate) fn exposure_histogram_pass(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(&ViewVisibilityBuffer, &ViewExposureHistogramBindGroup)>,
    pipeline: Res<ExposureHistogramPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_exposure {
        return;
    }
    let (visibility, group) = view.into_inner();

    let Some(build) = cache.get_compute_pipeline(pipeline.build) else {
        return;
    };

    let size = visibility.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let config = GpuExposureHistogramConfig::from_view(
        HistogramRange {
            min_log2_luminance: settings.exposure_histogram_min_log2,
            max_log2_luminance: settings.exposure_histogram_max_log2,
        },
        size,
    );

    let workgroups_x = size.x.div_ceil(EXPOSURE_HISTOGRAM_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(EXPOSURE_HISTOGRAM_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism exposure histogram"),
            timestamp_writes: None,
        });
    pass.set_pipeline(build);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
