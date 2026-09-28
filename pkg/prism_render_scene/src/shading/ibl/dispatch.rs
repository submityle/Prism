//! `Core3d` graph nodes recording the IBL precompute passes.
//!
//! * [`dfg_lut_precompute_pass`] integrates the view-independent DFG table once
//!   and then early-returns forever after.
//! * [`env_prefilter_precompute_pass`] convolves the active environment probe
//!   into the prefiltered radiance cube, reconvolving only when the probe
//!   changes.
//!
//! Both run before the shading resolve that samples the tables and are skipped
//! entirely while image-based lighting is disabled.

use bevy_asset::AssetId;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use super::abi::{
    GpuBrdfLutConfig, GpuPrefilterConfig, BRDF_LUT_WORKGROUP_SIZE, ENV_PREFILTER_WORKGROUP_SIZE,
};
use super::bind_groups::{DfgLutBindGroup, EnvPrefilterBindGroups};
use super::pipeline::{BrdfLutPipeline, EnvPrefilterPipeline};
use super::resources::{DfgLutTexture, PrefilteredEnvironmentMap};

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

/// Records the prefiltered-radiance convolution for the active probe.
///
/// One dispatch per output mip bakes a fixed roughness (mip 0 = mirror,
/// last mip = fully rough), each covering all six cube faces via `z = 6`
/// workgroups.  `generated_source` remembers which source cube was last
/// convolved so the (expensive) full mip chain is only re-recorded when the
/// probe actually changes; clearing it when nothing is bound lets a probe that
/// disappears and reappears reconvolve.
pub(crate) fn env_prefilter_precompute_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    pipeline: Res<EnvPrefilterPipeline>,
    target: Res<PrefilteredEnvironmentMap>,
    bind_groups: Res<EnvPrefilterBindGroups>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
    mut generated_source: Local<Option<AssetId<Image>>>,
) {
    if !settings.enable_ibl {
        return;
    }
    // Nothing bound yet (no probe, or its GPU image is not uploaded): idle and
    // forget the prior source so a re-appearing probe reconvolves.
    if bind_groups.mips.is_empty() {
        *generated_source = None;
        return;
    }
    // Already convolved this exact source cube.
    if *generated_source == bind_groups.source {
        return;
    }

    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    let mip_count = target.mip_count();
    // Map mip index -> perceptual roughness in [0, 1]; guard the single-mip case.
    let denom = mip_count.saturating_sub(1).max(1) as f32;
    let sample_count = settings.ibl_prefilter_sample_count;

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism env prefilter precompute"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    for mip in 0..mip_count {
        let Some(bind_group) = bind_groups.mips.get(mip as usize) else {
            continue;
        };
        let mip_size = target.mip_size(mip);
        let roughness = mip as f32 / denom;
        let config = GpuPrefilterConfig::new(roughness, sample_count, mip_size);
        let groups = mip_size.div_ceil(ENV_PREFILTER_WORKGROUP_SIZE);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&config));
        // z = 6: one workgroup layer per cube face.
        pass.dispatch_workgroups(groups, groups, 6);
    }
    drop(pass);

    *generated_source = bind_groups.source;
}
