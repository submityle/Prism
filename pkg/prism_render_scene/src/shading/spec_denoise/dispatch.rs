//! `Core3d` dispatch node recording the specular-GI *spatial denoise* compute
//! pass for every view.
//!
//! One invocation per framebuffer texel reconstructs the pixel's view-space
//! position from the SSR prepass reverse-Z depth (bound in
//! [`super::bind_groups`]), selects an anisotropic, contact-hardened blur
//! footprint from the surface roughness and the SSR hit distance, then runs the
//! edge-aware cross-bilateral gather (gated on depth, normal and roughness) that
//! matches the CPU golden's `spatial_filter`, and writes the denoised specular +
//! passthrough confidence into the filtered target the composite consumes.
//!
//! Like [`super::super::spec_gi::dispatch::spec_gi_reuse_pass`], the kernel
//! reads its config from the bound uniform the bind-group slice uploaded, so
//! this node records **no** `set_immediates`.
//!
//! Ordering (wired in the plugin slice): it runs after the `spec_gi` reuse pass
//! that produces the resolve it filters, and before the energy-conserving
//! composite that folds the filtered specular back into `scene_color`.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
};

use super::abi::SPEC_DENOISE_WORKGROUP_SIZE;
use super::bind_groups::ViewSpecDenoiseBindGroup;
use super::pipeline::SpecDenoiseSpatialPipeline;
use super::resources::ViewSpecDenoise;

/// `Core3d` node recording the spatial-denoise dispatch for every view whose
/// spatial-denoise target and bind group are resident.
///
/// Gated on `enable_spec_gi`; the per-view [`ViewSpecDenoiseBindGroup`] is only
/// present when the subsystem's full gate (SSR + visibility buffer +
/// single-sample) already held in resource prep, so a resident bind group is
/// sufficient to dispatch. Dispatches one workgroup per
/// [`SPEC_DENOISE_WORKGROUP_SIZE`]×[`SPEC_DENOISE_WORKGROUP_SIZE`] tile; the
/// kernel bounds-checks every invocation against the config extent.
pub(crate) fn spec_denoise_spatial_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSpecDenoise, &ViewSpecDenoiseBindGroup)>,
    pipeline: Res<SpecDenoiseSpatialPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_spec_gi {
        return;
    }
    let (denoise, bind_group) = view.into_inner();

    let Some(spatial) = cache.get_compute_pipeline(pipeline.spatial()) else {
        return;
    };

    let size = denoise.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let workgroups_x = size.x.div_ceil(SPEC_DENOISE_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SPEC_DENOISE_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism spec_denoise spatial"),
            timestamp_writes: None,
        });
    pass.set_pipeline(spatial);
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
        assert_eq!(1920u32.div_ceil(SPEC_DENOISE_WORKGROUP_SIZE), 240);
        assert_eq!(1080u32.div_ceil(SPEC_DENOISE_WORKGROUP_SIZE), 135);
        assert_eq!(1u32.div_ceil(SPEC_DENOISE_WORKGROUP_SIZE), 1);
        assert_eq!(9u32.div_ceil(SPEC_DENOISE_WORKGROUP_SIZE), 2);
    }
}
