//! One-shot dispatch of the physical sky transmittance LUT bake.
use super::{
    abi::{GpuSkyTransmittanceParams, WORKGROUP_SIZE},
    bind_groups::SkyTransmittanceBindGroup,
    pipeline::SkyTransmittancePipeline,
    resources::SkyTransmittanceLut,
};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};
pub(crate) fn sky_transmittance_lut_pass(
    pipeline: Res<SkyTransmittancePipeline>,
    bind: Res<SkyTransmittanceBindGroup>,
    lut: Res<SkyTransmittanceLut>,
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
    let params = GpuSkyTransmittanceParams::new(lut.width, lut.height, 40);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism sky transmittance LUT"),
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
