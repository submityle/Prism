//! The `Core3d` bloom node: records the whole dual-filter pyramid for a view.
//!
//! A single system records every dispatch of the chain into one compute pass so
//! the graph carries just one node regardless of how deep the pyramid is:
//!
//! 1. `copy` — lift `scene_color` into the full-res base (so the combine can
//!    read the untouched scene while writing back into `scene_color`).
//! 2. `prefilter` — Karis soft-threshold prefilter + partial-Karis first
//!    downsample into the half-res mip 0.
//! 3. `downsample` x `(levels - 1)` — energy-preserving COD 13-tap downsamples
//!    building the rest of the pyramid.
//! 4. `upsample` x `(levels - 1)` — 3x3 tent upsamples accumulating the halo
//!    from the coarsest mip back up to mip 0.
//! 5. `combine` — blend the accumulated bloom over the base by the artist
//!    intensity, back into `scene_color`.
//!
//! Each dispatch carries one [`GpuBloomParams`]: `src_size` scales the sampling
//! offsets (`x2` in the downsamples, `x0.5` in the upsamples, so the shader
//! multiplies the source texel size to reach the neighbour taps) and `dst_size`
//! both bounds the per-texel coverage guard and drives the `8x8` workgroup
//! grid. Runs after the exposure resolve (so metering saw the clean scene) and
//! before the main pass; gated on `enable_bloom`.

use bevy_ecs::prelude::*;
use bevy_math::UVec2;
use bevy_render::{
    render_resource::{
        BindGroup, ComputePass, ComputePassDescriptor, ComputePipeline, PipelineCache,
    },
    renderer::{RenderContext, ViewQuery},
};
use prism_render_shading::BloomParams;

use super::super::runtime::PrismShadingSettings;
use super::abi::{GpuBloomParams, BLOOM_WORKGROUP_SIZE};
use super::bind_groups::ViewBloomBindGroups;
use super::pipeline::BloomPipelines;
use super::resources::ViewBloomTextures;

/// Records one bloom dispatch: bind the pipeline + group, push the immediate,
/// and cover `dst` with `8x8` workgroups (rounding up so a partial edge tile is
/// still launched).
fn record(
    pass: &mut ComputePass<'_>,
    pipeline: &ComputePipeline,
    group: &BindGroup,
    params: GpuBloomParams,
    dst: UVec2,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(
        dst.x.div_ceil(BLOOM_WORKGROUP_SIZE),
        dst.y.div_ceil(BLOOM_WORKGROUP_SIZE),
        1,
    );
}

/// `Core3d` node recording the full bloom pyramid for every view with a
/// resident pyramid + bind groups. Gated on `enable_bloom`; returns early until
/// all five pipelines have finished compiling so a half-ready frame is a no-op
/// rather than a partial pass.
pub(crate) fn bloom_pass(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(&ViewBloomTextures, &ViewBloomBindGroups)>,
    pipelines: Res<BloomPipelines>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_bloom {
        return;
    }
    let (textures, groups) = view.into_inner();

    let (Some(copy), Some(prefilter), Some(downsample), Some(upsample), Some(combine)) = (
        cache.get_compute_pipeline(pipelines.copy),
        cache.get_compute_pipeline(pipelines.prefilter),
        cache.get_compute_pipeline(pipelines.downsample),
        cache.get_compute_pipeline(pipelines.upsample),
        cache.get_compute_pipeline(pipelines.combine),
    ) else {
        return;
    };

    let bloom = BloomParams {
        threshold: settings.bloom_threshold,
        knee: settings.bloom_knee,
        intensity: settings.bloom_intensity,
        radius: settings.bloom_radius,
    };

    let chain = textures.chain();
    let full = chain.full;
    let mips = &chain.mips;
    let levels = mips.len();
    if levels == 0 {
        return;
    }

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism bloom"),
            timestamp_writes: None,
        });

    // 1. Full-res copy of scene_color into the base (dst guard only).
    record(&mut pass, copy, &groups.copy, GpuBloomParams::new(full, full, bloom), full);

    // 2. Prefilter + partial-Karis first downsample: full-res source -> mip 0.
    record(
        &mut pass,
        prefilter,
        &groups.prefilter,
        GpuBloomParams::new(full, mips[0], bloom),
        mips[0],
    );

    // 3. Deeper downsamples: down[k] -> down[k + 1].
    for k in 0..levels - 1 {
        record(
            &mut pass,
            downsample,
            &groups.downs[k],
            GpuBloomParams::new(mips[k], mips[k + 1], bloom),
            mips[k + 1],
        );
    }

    // 4. Upsamples coarse -> fine, accumulating into up[i]. The coarse source
    // is half the destination extent, so its texel size (`inv_src`) is the
    // sampling scale the shader halves.
    for i in (0..levels - 1).rev() {
        record(
            &mut pass,
            upsample,
            &groups.ups[i],
            GpuBloomParams::new(mips[i + 1], mips[i], bloom),
            mips[i],
        );
    }

    // 5. Combine the accumulated bloom over the base, back into scene_color.
    // The bloom is sampled with normalised UVs, so `src_size` is the full
    // extent regardless of the bloom mip's own size.
    record(
        &mut pass,
        combine,
        &groups.combine,
        GpuBloomParams::new(full, full, bloom),
        full,
    );
}
