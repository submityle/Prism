//! One-shot dispatch of the physical sky sky-view LUT bake.
use super::{
    abi::{GpuSkyViewParams, WORKGROUP_SIZE},
    bind_groups::SkyViewBindGroup,
    pipeline::SkyViewPipeline,
    resources::SkyViewLut,
};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

fn pending_revision(source_revision: Option<u64>, generated_revision: Option<u64>) -> Option<u64> {
    source_revision.filter(|revision| generated_revision != Some(*revision))
}

pub(crate) fn sky_view_lut_pass(
    multiscatter_generation: Res<super::super::multiscatter::SkyMultiscatterGeneration>,
    pipeline: Res<SkyViewPipeline>,
    bind: Res<SkyViewBindGroup>,
    lut: Res<SkyViewLut>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
    mut generated_revision: Local<Option<u64>>,
) {
    let Some(multiscatter_revision) =
        pending_revision(multiscatter_generation.revision(), *generated_revision)
    else {
        return;
    };
    let Some(p) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let params = GpuSkyViewParams::new(lut.width, lut.height, 48, 16);
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism sky-view LUT"),
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
    *generated_revision = Some(multiscatter_revision);
}

#[cfg(test)]
mod tests {
    use super::pending_revision;

    #[test]
    fn waits_for_the_multiscatter_source_to_be_generated() {
        assert_eq!(pending_revision(None, None), None);
        assert_eq!(pending_revision(None, Some(7)), None);
    }

    #[test]
    fn first_source_revision_requests_a_bake() {
        assert_eq!(pending_revision(Some(1), None), Some(1));
    }

    #[test]
    fn unchanged_source_revision_does_not_request_another_bake() {
        assert_eq!(pending_revision(Some(4), Some(4)), None);
    }

    #[test]
    fn changed_source_revision_requests_a_rebake() {
        assert_eq!(pending_revision(Some(5), Some(4)), Some(5));
    }
}
