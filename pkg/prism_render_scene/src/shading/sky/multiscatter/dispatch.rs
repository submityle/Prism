//! One-shot dispatch of the physical sky LUT bake.
use super::{
    abi::{GpuSkyMultiscatterParams, SkyMultiscatterSampleBudget, WORKGROUP_SIZE},
    bind_groups::SkyMultiscatterBindGroup,
    pipeline::SkyMultiscatterPipeline,
    resources::{SkyMultiscatterGeneration, SkyMultiscatterLut},
};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};
pub(crate) fn sky_multiscatter_lut_pass(
    settings: Res<super::super::super::runtime::PrismShadingSettings>,
    pipeline: Res<SkyMultiscatterPipeline>,
    bind: Res<SkyMultiscatterBindGroup>,
    lut: Res<SkyMultiscatterLut>,
    mut generation: ResMut<SkyMultiscatterGeneration>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let budget = SkyMultiscatterSampleBudget::from(settings.lobe_sample_budget());
    if !generation.needs_rebuild(budget) {
        return;
    }
    let Some(p) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let params = GpuSkyMultiscatterParams::for_budget(lut.width, lut.height, budget);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism sky multiple-scattering LUT"),
            timestamp_writes: None,
        });
    pass.set_pipeline(p);
    pass.set_bind_group(0, &bind.0, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(
        lut.width.div_ceil(WORKGROUP_SIZE),
        lut.height.div_ceil(WORKGROUP_SIZE),
        1,
    );
    drop(pass);
    generation.mark_generated(budget);
}
