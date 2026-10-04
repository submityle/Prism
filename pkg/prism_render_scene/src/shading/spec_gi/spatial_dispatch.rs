//! `Core3d` dispatch node recording the glossy-specular ReSTIR **spatial**
//! reuse compute pass for every view.
//!
//! One invocation per framebuffer texel reconstructs the pixel's view-space
//! glossy point from the SSR prepass reverse-Z depth + packed `normal_roughness`
//! (bound in [`super::spatial_bind_groups`]), seeds a working reservoir from the
//! pixel's own post-temporal reservoir, pools a frame-jittered Fibonacci-spiral
//! disc of neighbour reservoirs onto this pixel's GGX lobe through the golden
//! `specr_merge_glossy`, clamps the pooled confidence with the
//! roughness-tightened M-cap and overwrites the resolved specular + confidence
//! target (the reservoir table itself is left as the pure post-temporal history
//! so spatial correlation never feeds back into next frame's temporal reuse).
//!
//! Unlike [`super::dispatch`] (whose reuse kernel reads its config from a bound
//! uniform), this node pushes the [`GpuSpecGiSpatialParams`] in an immediate
//! block (like the SSGI / composite dispatches), so it records a
//! `set_immediates` before the dispatch.
//!
//! Ordering (wired in the plugin slice): it runs after the reuse dispatch that
//! writes this frame's reservoir table + resolved target (so the read-only
//! snapshot is complete behind the inserted render-graph barrier) and before
//! the `spec_denoise` reproject that consumes the spatially pooled resolve.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use super::abi::{GpuSpecGiSpatialParams, SPEC_GI_WORKGROUP_SIZE};
use super::resources::ViewSpecGiReuse;
use super::spatial_bind_groups::ViewSpecGiSpatialBindGroup;
use super::spatial_pipeline::SpecGiSpatialPipeline;

/// `Core3d` node recording the glossy-specular spatial reuse dispatch for every
/// view whose reuse resources and spatial bind group are resident.
///
/// Gated on `enable_spec_gi && enable_spec_gi_spatial`; the per-view
/// [`ViewSpecGiSpatialBindGroup`] is only present when the subsystem's full gate
/// (SSR + visibility buffer + single-sample) already held in resource prep and
/// spatial reuse is enabled, so a resident bind group plus the enable gate is
/// sufficient to dispatch. Dispatches one workgroup per
/// [`SPEC_GI_WORKGROUP_SIZE`]×[`SPEC_GI_WORKGROUP_SIZE`] tile; the kernel
/// bounds-checks every invocation against the params extent.
pub(crate) fn spec_gi_spatial_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(
        &ViewSpecGiReuse,
        &ViewSpecGiSpatialBindGroup,
        &ExtractedView,
    )>,
    pipeline: Res<SpecGiSpatialPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_spec_gi || !settings.enable_spec_gi_spatial {
        return;
    }
    let (spec_gi, bind_group, extracted) = view.into_inner();

    let Some(spatial) = cache.get_compute_pipeline(pipeline.spatial()) else {
        return;
    };

    let size = spec_gi.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // The inverse projection rebuilds the view-space glossy point from the SSR
    // prepass reverse-Z depth, identically to the reuse pass's reconstruction.
    let params = GpuSpecGiSpatialParams::new(
        extracted.clip_from_view.inverse(),
        size.x,
        size.y,
        settings.spec_gi_spatial_radius,
        settings.spec_gi_spatial_sample_count,
        spec_gi.frame(),
        settings.spec_gi_temporal_m_cap,
        settings.spec_gi_roughness_cap_base,
        settings.spec_gi_sigma_roughness,
    );

    let workgroups_x = size.x.div_ceil(SPEC_GI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SPEC_GI_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism spec_gi spatial"),
            timestamp_writes: None,
        });
    pass.set_pipeline(spatial);
    pass.set_bind_group(0, bind_group.group(), &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_count_rounds_up_to_cover_every_texel() {
        // Same 8x8 tiling as the reuse / composite dispatches: a 1920x1080
        // target needs ceil(1920/8)=240 x ceil(1080/8)=135 groups, and any
        // non-multiple extent still rounds up so no edge texels drop.
        assert_eq!(1920u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 240);
        assert_eq!(1080u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 135);
        assert_eq!(9u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 2);
    }
}
