//! `Core3d` dispatch node recording the glossy-specular ReSTIR *reuse* compute
//! pass for every view.
//!
//! One invocation per framebuffer texel reconstructs the pixel's view-space
//! glossy point from the SSR prepass reverse-Z depth + packed `normal_roughness`
//! (bound in [`super::bind_groups`]), streams the current-frame screen-space GGX
//! candidate radiance into a fresh reservoir, temporally merges the same-pixel
//! prior-frame reservoir under the roughness-tightened confidence cap, finalises
//! the unbiased contribution weight and writes both the packed reservoir
//! (ping-pong storage, for next frame) and the resolved specular + confidence
//! (storage texture, for `spec_denoise` and the energy-conserving composite).
//!
//! Unlike [`super::super::ssgi::trace::ssgi_trace_pass`] and the world-space
//! `ReSTIR` fill — both of which push their dispatch config in an immediate
//! block — the reuse kernel reads its config from the bound uniform the
//! bind-group slice uploaded, so this node records **no** `set_immediates`.
//!
//! Ordering (wired in the plugin slice): it runs after the SSR trace/repack
//! that fill its depth/normal/candidate inputs and the reuse resource prep that
//! advances the ping-pong flip, and before the energy-conserving composite that
//! folds the resolved specular back into `scene_color`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::abi::SPEC_GI_WORKGROUP_SIZE;
use super::bind_groups::ViewSpecGiReuseBindGroup;
use super::pipeline::SpecGiReusePipeline;
use super::resources::ViewSpecGiReuse;

/// `Core3d` node recording the glossy-specular reuse dispatch for every view
/// whose reuse resources and bind group are resident.
///
/// Gated on `enable_spec_gi`; the per-view [`ViewSpecGiReuseBindGroup`] is only
/// present when the subsystem's full gate (SSR + visibility buffer +
/// single-sample) already held in resource prep, so a resident bind group is
/// sufficient to dispatch. Dispatches one workgroup per
/// [`SPEC_GI_WORKGROUP_SIZE`]×[`SPEC_GI_WORKGROUP_SIZE`] tile; the kernel
/// bounds-checks every invocation against the config extent.
#[allow(dead_code)] // added to the Core3d schedule by the plugin slice (next).
pub(crate) fn spec_gi_reuse_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSpecGiReuse, &ViewSpecGiReuseBindGroup)>,
    pipeline: Res<SpecGiReusePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_spec_gi {
        return;
    }
    let (spec_gi, bind_group) = view.into_inner();

    let Some(reuse) = cache.get_compute_pipeline(pipeline.reuse()) else {
        return;
    };

    let size = spec_gi.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let workgroups_x = size.x.div_ceil(SPEC_GI_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SPEC_GI_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism spec_gi reuse"),
            timestamp_writes: None,
        });
    pass.set_pipeline(reuse);
    pass.set_bind_group(0, bind_group.group(), &[]);
    // No `set_immediates`: the config travels in the bound uniform at binding 0.
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_count_rounds_up_to_cover_every_texel() {
        // The dispatch must cover a partial trailing tile: a 1920x1080 target at
        // the 8x8 workgroup needs ceil(1920/8)=240 x ceil(1080/8)=135 groups,
        // and a non-multiple extent still rounds up (so no edge texels drop).
        assert_eq!(1920u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 240);
        assert_eq!(1080u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 135);
        assert_eq!(1u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 1);
        assert_eq!(9u32.div_ceil(SPEC_GI_WORKGROUP_SIZE), 2);
    }
}
