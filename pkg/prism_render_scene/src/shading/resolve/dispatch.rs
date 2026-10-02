//! `Core3d` graph node recording the shading-resolve dispatches.
//!
//! `material_classification.wesl` already wrote one indirect
//! [`GpuShadingDispatchArgs`](super::super::classification_gpu::GpuShadingDispatchArgs)
//! slot per shading class into `dispatch_args`.  This node binds the four
//! resolve groups once and issues one `dispatch_workgroups_indirect` per class,
//! swapping only the `shading_class` immediate between passes so the shader can
//! select the matching worklist slice and BRDF lobe.
//!
//! No buffer is cleared here: the classification stage owns
//! `clear_transient_state`, and this stage only *reads* those buffers and
//! *writes* the HDR storage texture.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, ViewQuery},
    view::ExtractedView,
};

use prism_render_shading::MAX_SHADING_CLASSES;

use crate::{ClusterBindGroup, LightBindGroup, MaterialBindGroup};

use super::super::shadow::ShadowBindGroup;

use super::super::classification_gpu::GpuShadingDispatchArgs;
use super::super::ibl::EnvPrefilterBindGroups;
use super::super::resources::ViewShadingBuffers;
use super::abi::{GpuShadingResolveParams, RESOLVE_FLAG_GTAO, RESOLVE_FLAG_IBL_SPECULAR};
use super::bind_groups::ViewResolveBindGroups;
use super::pipeline::ShadingResolvePipeline;

// The resolve pass reuses the *same* indirect dispatch-argument buffer that
// `material_classification.wesl` fills (one `GpuShadingDispatchArgs` per
// class, each `= pixel_count / CLASSIFICATION_WORKGROUP_SIZE` workgroups).
// That is only correct if this shader declares the identical workgroup size,
// so pin the two constants together at compile time.
const _: () = assert!(
    super::abi::RESOLVE_WORKGROUP_SIZE
        == super::super::classification_gpu::CLASSIFICATION_WORKGROUP_SIZE,
    "shading_resolve @workgroup_size must match the classification dispatch stride",
);

pub(crate) fn dispatch_shading_resolve(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewShadingBuffers, &ViewResolveBindGroups, &ExtractedView)>,
    material_bindings: Res<MaterialBindGroup>,
    light_bindings: Res<LightBindGroup>,
    shadow_bindings: Res<ShadowBindGroup>,
    cluster_bindings: Res<ClusterBindGroup>,
    prefilter_bind_groups: Res<EnvPrefilterBindGroups>,
    pipeline: Res<ShadingResolvePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let (buffers, groups, extracted) = view.into_inner();

    // Every bind group must be resident: the two pass-owned ones plus the
    // shared material/light groups the classification stage also depends on.
    let (Some(materials_group), Some(lights_group), Some(shadow_group), Some(cluster_group)) = (
        material_bindings.bind_group.as_ref(),
        light_bindings.bind_group.as_ref(),
        shadow_bindings.bind_group.as_ref(),
        cluster_bindings.bind_group.as_ref(),
    ) else {
        return;
    };
    let Some(resolve) = cache.get_compute_pipeline(pipeline.resolve) else {
        return;
    };

    let translation = extracted.world_from_view.translation();
    // Pack the feature bits the resolve shader ANDs against. The prefiltered
    // specular path is enabled only when the prefilter pass has bind groups
    // built for a resident probe source; otherwise the shader falls back to the
    // low-frequency SH-radiance specular so an IBL scene without a specular map
    // still shades.
    let mut flags = 0u32;
    if settings.enable_gtao {
        flags |= RESOLVE_FLAG_GTAO;
    }
    if prefilter_bind_groups.source.is_some() {
        flags |= RESOLVE_FLAG_IBL_SPECULAR;
    }
    let mut params = GpuShadingResolveParams {
        shading_class: 0,
        width: buffers.size.x,
        height: buffers.size.y,
        flags,
        view_position: [translation.x, translation.y, translation.z, 0.0],
    };
    if params.width == 0 || params.height == 0 {
        return;
    }

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism GPU shading resolve"),
            timestamp_writes: None,
        });
    pass.set_pipeline(resolve);
    pass.set_bind_group(0, &groups.view, &[]);
    pass.set_bind_group(1, materials_group, &[]);
    pass.set_bind_group(2, &groups.scene, &[]);
    pass.set_bind_group(3, lights_group, &[]);
    pass.set_bind_group(4, shadow_group, &[]);
    pass.set_bind_group(5, cluster_group, &[]);
    // group 6: virtual-shadow-map sample bindings. Always bound (the
    // pipeline layout includes group 6); real when the feature is on for
    // this view, else the pass-owned fallbacks with the uniform's `enable`
    // bit clear so the shader stays on the cascaded-shadow path.
    pass.set_bind_group(6, &groups.vsm, &[]);

    let stride = size_of::<GpuShadingDispatchArgs>() as u64;
    for class in 0..MAX_SHADING_CLASSES as u32 {
        params.shading_class = class;
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups_indirect(&buffers.dispatch_args, class as u64 * stride);
    }
}
