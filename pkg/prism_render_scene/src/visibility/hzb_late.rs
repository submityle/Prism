use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer, storage_buffer_read_only},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, BufferId, CachedComputePipelineId,
        ComputePipelineDescriptor, ShaderStages,
    },
    renderer::{RenderContext, RenderDevice},
};
use bevy_shader::Shader;

use crate::geometry::RenderGeometryBuffers;

#[derive(Resource)]
pub(crate) struct HzbLateCompactPipeline {
    pub(crate) pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
    bind_group_layout: BindGroupLayout,
}

#[derive(Resource, Default)]
pub(crate) struct HzbLateCompactBindGroup {
    pub(crate) bind_group: Option<BindGroup>,
    ids: Option<[BufferId; 8]>,
}

pub(crate) fn init_hzb_late_compact_pipeline(
    mut commands: Commands,
    cache: Res<bevy_render::render_resource::PipelineCache>,
    device: Res<RenderDevice>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only::<u32>(false),
            storage_buffer_read_only::<u32>(false),
            storage_buffer::<super::rows::RenderDrawBinHeader>(false),
            storage_buffer::<super::rows::RenderVisibilityIndirect>(false),
            storage_buffer::<super::rows::RenderVisibilityNonIndexedIndirect>(false),
            storage_buffer::<super::rows::RenderVisibilityCounter>(false),
            storage_buffer_read_only::<super::super::geometry::rows::RenderGeometryHeader>(false),
            storage_buffer_read_only::<super::super::geometry::rows::RenderGeometryLod>(false),
        ),
    );
    let layout = BindGroupLayoutDescriptor::new("prism hzb late compact", &entries);
    let bind_group_layout = device.create_bind_group_layout("prism hzb late compact", &entries);
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/hzb_late_compact.wesl");
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism hzb late compact".into()),
        layout: vec![layout.clone()],
        immediate_size: 32,
        shader,
        entry_point: Some("compact_late_hzb".into()),
        ..Default::default()
    });
    commands.insert_resource(HzbLateCompactPipeline {
        pipeline,
        layout,
        bind_group_layout,
    });
}

pub(crate) fn inspect_hzb_late_pipeline(
    pipeline: Res<HzbLateCompactPipeline>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
) {
    let _ = (
        &pipeline.layout,
        &pipeline.bind_group_layout,
        cache.get_compute_pipeline(pipeline.pipeline),
    );
}

pub(crate) fn prepare_hzb_late_bind_group(
    pipeline: Res<HzbLateCompactPipeline>,
    hzb: Res<super::hzb_gpu::HzbVisibilityBuffers>,
    visibility: Res<super::buffers::UnifiedVisibilityBuffers>,
    geometry_buffers: Res<RenderGeometryBuffers>,
    device: Res<RenderDevice>,
    mut bindings: ResMut<HzbLateCompactBindGroup>,
) {
    let (_, stages) = hzb.bindings();
    let Some(candidate_bins) = visibility.candidate_bin_buffer() else {
        return;
    };
    let Some((late_counters, late_indexed, late_non_indexed, late_bins)) =
        visibility.late_compute_buffers()
    else {
        return;
    };
    let Some((geometry_headers, geometry_lods)) = geometry_buffers.buffers() else {
        return;
    };
    let ids = [
        stages.id(),
        candidate_bins.id(),
        late_bins.id(),
        late_indexed.id(),
        late_non_indexed.id(),
        late_counters.id(),
        geometry_headers.id(),
        geometry_lods.id(),
    ];
    if bindings.ids == Some(ids) {
        return;
    }
    bindings.bind_group = Some(device.create_bind_group(
        "prism hzb late compact",
        &pipeline.bind_group_layout,
        &BindGroupEntries::sequential((
            stages.as_entire_binding(),
            candidate_bins.as_entire_binding(),
            late_bins.as_entire_binding(),
            late_indexed.as_entire_binding(),
            late_non_indexed.as_entire_binding(),
            late_counters.as_entire_binding(),
            geometry_headers.as_entire_binding(),
            geometry_lods.as_entire_binding(),
        )),
    ));
    bindings.ids = Some(ids);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LateCompactDispatch {
    candidate_count: u32,
    stage_start: u32,
    candidate_bin_start: u32,
    bin_start: u32,
    command_start: u32,
    command_end: u32,
    indirect_first_instance: u32,
    counter_index: u32,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Late compaction consumes the per-view HZB, bin, and command contracts."
)]
pub(crate) fn dispatch_hzb_late_compact(
    view: bevy_render::renderer::ViewQuery<&bevy_render::view::ExtractedView>,
    enabled: Res<super::runtime::UnifiedVisibilityEnabled>,
    settings: Res<super::runtime::UnifiedVisibilitySettings>,
    state: Res<super::runtime::UnifiedVisibilityState>,
    hzb: Res<super::hzb_gpu::HzbVisibilityBuffers>,
    pipeline: Res<HzbLateCompactPipeline>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
    bindings: Res<HzbLateCompactBindGroup>,
    visibility: Res<super::buffers::UnifiedVisibilityBuffers>,
    mut ctx: RenderContext,
    mut diagnostics: ResMut<super::runtime::PrismVisibilityDiagnostics>,
) {
    if !enabled.0 || !settings.hzb_occlusion {
        return;
    }
    let retained = view.into_inner().retained_view_entity;
    let Some((stage_start, candidate_count)) = hzb.view_range(retained) else {
        return;
    };
    let Some(view_bins) = state
        .draw_bins
        .iter()
        .find(|bins| state.retained_view(bins.view) == Some(retained))
    else {
        return;
    };
    let Some(counter_index) = state
        .views
        .iter()
        .position(|view| state.retained_view(view.handle) == Some(retained))
        .and_then(|index| u32::try_from(index).ok())
    else {
        return;
    };
    let (Some(compute_pipeline), Some(bind_group)) = (
        cache.get_compute_pipeline(pipeline.pipeline),
        bindings.bind_group.as_ref(),
    ) else {
        return;
    };
    let immediates = LateCompactDispatch {
        candidate_count,
        stage_start,
        candidate_bin_start: view_bins.global_candidate_start,
        bin_start: view_bins.global_bin_start,
        command_start: view_bins.command_buffer_start,
        command_end: view_bins
            .command_buffer_start
            .saturating_add(visibility.gpu_slots_per_view()),
        indirect_first_instance: u32::from(settings.indirect_first_instance),
        counter_index,
    };
    let mut pass = ctx.command_encoder().begin_compute_pass(
        &bevy_render::render_resource::ComputePassDescriptor {
            label: Some("prism late hzb command compaction"),
            timestamp_writes: None,
        },
    );
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&immediates));
    pass.dispatch_workgroups(candidate_count.div_ceil(64), 1, 1);
    diagnostics.hzb_late_visibility_dispatches += 1;
}

#[cfg(test)]
mod tests {
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("late compact shader is WESL"),
        }
    }

    #[test]
    fn late_compact_wesl_compiles() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d48_5a42_4c41_5445_434f_4d01),
        };
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                include_str!("../shaders/hzb_late_compact.wesl"),
                "shaders/prism_hzb_late_compact.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("late compact shader failed: {error}"));
    }
}
