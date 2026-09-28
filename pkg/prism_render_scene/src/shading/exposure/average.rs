//! Exposure resolve: pipeline, per-view bind group, and the single-invocation
//! `Core3d` dispatch node.
//!
//! Second of the two auto-exposure passes and the GPU twin of the reduction +
//! eye-adaptation half of [`prism_render_shading::exposure`]. A single compute
//! invocation copies the 64-bin histogram into registers, computes the
//! percentile-trimmed average luminance (discarding the darkest `low` and
//! brightest `high` fraction so background darkness and pinprick highlights do
//! not drag the metering), eases the persistent adapted luminance toward it
//! with the exponential eye-adaptation response, resolves the clamped +
//! compensated exposure multiplier, writes it to the persistent state, and
//! zeroes the histogram so the next frame accumulates from empty.
//!
//! It reads one bind group (group 0, matching `resolve_exposure` in
//! `shaders/exposure.wesl`):
//!
//! * `0` the 64-bin histogram storage buffer (reduced, then zeroed), and
//! * `1` the persistent [`GpuExposureState`] (last frame's adaptation in, the
//!   new multiplier out).
//!
//! The log2 window, percentile trim, EV clamp/compensation, adaptation speeds
//! and the wall-clock frame delta travel in the [`GpuExposureResolveConfig`]
//! immediate block. The delta is measured on the CPU with a
//! [`std::time::Instant`] kept in a [`Local`] because the render world exposes
//! no frame clock to this pass. It runs after the histogram build and before
//! the main pass so the multiplier is ready when the composite consumes it.

use std::time::Instant;

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{binding_types::storage_buffer_sized, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
};
use bevy_shader::Shader;
use prism_render_shading::{
    AutoExposureSettings, EyeAdaptation, HistogramPercentiles, HistogramRange,
};

use super::super::runtime::PrismShadingSettings;
use super::abi::GpuExposureResolveConfig;
use super::resources::ViewExposureBuffers;

/// Compute pipeline and its owned group-0 layout for the exposure resolve.
#[derive(Resource)]
pub(crate) struct ExposureAveragePipeline {
    /// `resolve_exposure` compute entry point, specialized against the group-0
    /// layout and the 48-byte [`GpuExposureResolveConfig`] immediate block.
    resolve: CachedComputePipelineId,
    /// group 0: the histogram (reduced then zeroed) + the persistent exposure
    /// state (read/write), both read-write storage buffers.
    layout: BindGroupLayout,
}

/// group-0 layout mirroring `resolve_exposure`: the read-write histogram then
/// the read-write persistent exposure state, both runtime-sized storage
/// buffers.
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`ExposureAveragePipeline`].
pub(crate) fn init_exposure_average_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism exposure resolve", &entries);
    let layout = device.create_bind_group_layout("prism exposure resolve", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/exposure.wesl");

    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism exposure resolve".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuExposureResolveConfig>() as u32,
        shader,
        entry_point: Some("resolve_exposure".into()),
        ..Default::default()
    });

    commands.insert_resource(ExposureAveragePipeline { resolve, layout });
}

/// The exposure resolve's group-0 bind group for a single view. Present only
/// when exposure is enabled and the view has its persistent exposure buffers.
#[derive(Component)]
pub(crate) struct ViewExposureAverageBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building the resolve bind group for every view
/// with resident exposure buffers, gated on `enable_exposure` so a disabled
/// frame allocates nothing and the stale bind group is dropped.
pub(crate) fn prepare_exposure_average_bind_groups(
    mut commands: Commands,
    settings: Res<PrismShadingSettings>,
    pipeline: Res<ExposureAveragePipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewExposureBuffers)>,
) {
    for (entity, exposure) in &views {
        if !settings.enable_exposure {
            commands
                .entity(entity)
                .remove::<ViewExposureAverageBindGroup>();
            continue;
        }
        let group = device.create_bind_group(
            "prism exposure resolve",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                exposure.histogram_buffer().as_entire_binding(),
                exposure.state_buffer().as_entire_binding(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewExposureAverageBindGroup { group });
    }
}

/// `Core3d` node recording the single-invocation exposure resolve for every
/// view.
///
/// Runs after the histogram build (which fills the bins) and before the main
/// pass (so the multiplier is ready for the composite). The wall-clock delta is
/// measured across invocations with a [`Local`] [`Instant`]; the first frame
/// falls back to a 60 fps step so adaptation still integrates. Gated on
/// `enable_exposure`.
pub(crate) fn exposure_average_pass(
    settings: Res<PrismShadingSettings>,
    mut last: Local<Option<Instant>>,
    view: ViewQuery<&ViewExposureAverageBindGroup>,
    pipeline: Res<ExposureAveragePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_exposure {
        // Keep the clock fresh so a re-enable does not integrate a huge delta.
        *last = Some(Instant::now());
        return;
    }
    let group = view.into_inner();

    let Some(resolve) = cache.get_compute_pipeline(pipeline.resolve) else {
        return;
    };

    let now = Instant::now();
    let delta_seconds = match *last {
        Some(previous) => now.duration_since(previous).as_secs_f32().clamp(0.0, 1.0),
        None => 1.0 / 60.0,
    };
    *last = Some(now);

    let config = GpuExposureResolveConfig::new(
        HistogramRange::default(),
        HistogramPercentiles::default(),
        AutoExposureSettings::default(),
        EyeAdaptation::default(),
        delta_seconds,
    );

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism exposure resolve"),
            timestamp_writes: None,
        });
    pass.set_pipeline(resolve);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(1, 1, 1);
}
