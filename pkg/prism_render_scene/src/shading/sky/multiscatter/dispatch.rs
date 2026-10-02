//! One-shot dispatch of the physical sky LUT bake.
use super::{
    abi::{GpuSkyMultiscatterParams, WORKGROUP_SIZE},
    bind_groups::SkyMultiscatterBindGroup,
    pipeline::SkyMultiscatterPipeline,
    resources::SkyMultiscatterLut,
};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};
pub(crate) fn sky_multiscatter_lut_pass(
    pipeline: Res<SkyMultiscatterPipeline>,
    bind: Res<SkyMultiscatterBindGroup>,
    lut: Res<SkyMultiscatterLut>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
    mut generated: Local<bool>,
) {
    if *generated {
        return;
    }
    let Some(p) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let params = GpuSkyMultiscatterParams::new(lut.width, lut.height, 64, 16);
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
    *generated = true;
}
