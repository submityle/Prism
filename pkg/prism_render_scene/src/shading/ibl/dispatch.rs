//! `Core3d` graph node recording the one-shot DFG lookup-table precompute.
//!
//! The split-sum environment BRDF is independent of the scene, the view, and
//! time, so the table only needs to be integrated once: the node gates itself
//! on a [`Local`] flag and becomes a no-op after the first successful dispatch.
//! It runs before the shading resolve that samples the table, and is skipped
//! entirely while image-based lighting is disabled.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use super::abi::{GpuBrdfLutConfig, BRDF_LUT_WORKGROUP_SIZE};
use super::bind_groups::DfgLutBindGroup;
use super::pipeline::BrdfLutPipeline;
use super::resources::DfgLutTexture;

/// Records the DFG table integration once, then early-returns forever after.
///
/// `generated` persists across frames because render-graph node systems keep
/// their [`Local`] state; it is only set once the dispatch has actually been
/// recorded, so transient gaps (pipeline still compiling, bind group not yet
/// built) are retried on later frames rather than silently skipped.
pub(crate) fn dfg_lut_precompute_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    pipeline: Res<BrdfLutPipeline>,
    texture: Res<DfgLutTexture>,
    bind_group: Option<Res<DfgLutBindGroup>>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
    mut generated: Local<bool>,
) {
    if *generated || !settings.enable_ibl {
        return;
    }

    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let Some(bind_group) = bind_group else {
        return;
    };

    let config = GpuBrdfLutConfig::new(settings.ibl_dfg_sample_count);
    let groups = texture.resolution.div_ceil(BRDF_LUT_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism DFG LUT precompute"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, &bind_group.0, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&config));
    pass.dispatch_workgroups(groups, groups, 1);
    drop(pass);

    *generated = true;
}
