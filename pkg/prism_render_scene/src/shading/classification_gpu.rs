//! GPU-side material classification ABI.
//!
//! The buffers are intentionally per-view.  This keeps visibility IDs,
//! generation checks, counters and compact worklists isolated when more than
//! one camera is rendered in the same frame.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer, texture_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupEntries, BindGroupLayout, CachedComputePipelineId, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, ShaderStages, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
};
use bevy_shader::Shader;
use bytemuck::{Pod, Zeroable};

use prism_render_shading::MAX_SHADING_CLASSES;

pub const CLASSIFICATION_WORKGROUP_SIZE: u32 = 64;
#[cfg(test)]
const INVALID_CLASS: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub(crate) struct GpuShadingClassificationParams {
    pub width: u32,
    pub height: u32,
    pub pixel_count: u32,
    pub material_capacity: u32,
    pub work_capacity: u32,
    pub class_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub(crate) struct GpuShadingPixelClass {
    pub class: u32,
    pub work_index: u32,
    pub material_index: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub(crate) struct GpuShadingDispatchArgs {
    pub workgroups_x: u32,
    pub workgroups_y: u32,
    pub workgroups_z: u32,
}

pub(crate) const fn classification_dispatch(pixel_count: u32) -> GpuShadingDispatchArgs {
    GpuShadingDispatchArgs {
        workgroups_x: pixel_count.div_ceil(CLASSIFICATION_WORKGROUP_SIZE),
        workgroups_y: 1,
        workgroups_z: 1,
    }
}

#[cfg(test)]
fn class_prefix_sum(counts: [u32; MAX_SHADING_CLASSES]) -> ([u32; MAX_SHADING_CLASSES], u32) {
    let mut offsets = [0; MAX_SHADING_CLASSES];
    let mut cursor = 0_u32;
    let mut index = 0;
    while index < MAX_SHADING_CLASSES {
        offsets[index] = cursor;
        cursor = cursor.saturating_add(counts[index]);
        index += 1;
    }
    (offsets, cursor)
}

#[derive(Resource)]
pub(crate) struct MaterialClassificationPipeline {
    classify_count: CachedComputePipelineId,
    prefix_classes: CachedComputePipelineId,
    scatter_work: CachedComputePipelineId,
    visibility_layout: BindGroupLayout,
    output_layout: BindGroupLayout,
}

pub(crate) fn init_material_classification_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    material_bindings: Res<crate::MaterialBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let visibility_entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
        ),
    );
    let output_entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer::<u32>(false),
            storage_buffer::<u32>(false),
            storage_buffer::<u32>(false),
            storage_buffer::<u32>(false),
            storage_buffer::<u32>(false),
            storage_buffer::<u32>(false),
            storage_buffer::<u32>(false),
        ),
    );
    let visibility_descriptor =
        BindGroupLayoutDescriptor::new("prism classification visibility", &visibility_entries);
    let output_descriptor =
        BindGroupLayoutDescriptor::new("prism classification output", &output_entries);
    let visibility_layout =
        device.create_bind_group_layout("prism classification visibility", &visibility_entries);
    let output_layout =
        device.create_bind_group_layout("prism classification output", &output_entries);
    let shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/material_classification.wesl"
    );
    let pipeline = |label: &'static str, entry: &'static str| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![
                visibility_descriptor.clone(),
                material_bindings.layout_descriptor.clone(),
                output_descriptor.clone(),
            ],
            immediate_size: size_of::<GpuShadingClassificationParams>() as u32,
            shader: shader.clone(),
            entry_point: Some(entry.into()),
            ..Default::default()
        })
    };
    commands.insert_resource(MaterialClassificationPipeline {
        classify_count: pipeline("prism material classify/count", "classify_count"),
        prefix_classes: pipeline("prism material class prefix", "prefix_classes"),
        scatter_work: pipeline("prism material work scatter", "scatter_work"),
        visibility_layout,
        output_layout,
    });
}

pub(crate) fn prepare_material_classification_bind_groups(
    mut views: Query<(
        &super::resources::ViewVisibilityBuffer,
        &mut super::resources::ViewShadingBuffers,
    )>,
    pipeline: Res<MaterialClassificationPipeline>,
    device: Res<RenderDevice>,
) {
    for (visibility, mut buffers) in &mut views {
        let (ids, metadata) = visibility.attachments();
        buffers.input_bind_group = Some(device.create_bind_group(
            "prism classification visibility",
            &pipeline.visibility_layout,
            &BindGroupEntries::sequential((ids, metadata)),
        ));
        buffers.output_bind_group = Some(device.create_bind_group(
            "prism classification output",
            &pipeline.output_layout,
            &BindGroupEntries::sequential((
                buffers.pixel_classes.as_entire_binding(),
                buffers.work_items.as_entire_binding(),
                buffers.class_counts.as_entire_binding(),
                buffers.class_offsets.as_entire_binding(),
                buffers.class_cursors.as_entire_binding(),
                buffers.dispatch_args.as_entire_binding(),
                buffers.diagnostics.as_entire_binding(),
            )),
        ));
    }
}

pub(crate) fn dispatch_material_classification(
    settings: Res<super::runtime::PrismShadingSettings>,
    view: ViewQuery<&super::resources::ViewShadingBuffers>,
    material_bindings: Res<crate::MaterialBindGroup>,
    materials: Res<crate::material::runtime::RenderMaterialRegistry>,
    pipeline: Res<MaterialClassificationPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let buffers = view.into_inner();
    let (Some(input), Some(output), Some(materials_group)) = (
        buffers.input_bind_group.as_ref(),
        buffers.output_bind_group.as_ref(),
        material_bindings.bind_group.as_ref(),
    ) else {
        return;
    };
    let (Some(classify), Some(prefix), Some(scatter)) = (
        cache.get_compute_pipeline(pipeline.classify_count),
        cache.get_compute_pipeline(pipeline.prefix_classes),
        cache.get_compute_pipeline(pipeline.scatter_work),
    ) else {
        return;
    };
    let params = GpuShadingClassificationParams {
        width: buffers.size.x,
        height: buffers.size.y,
        pixel_count: buffers.size.x.saturating_mul(buffers.size.y),
        material_capacity: materials.registry.capacity(),
        work_capacity: buffers.capacity,
        class_count: MAX_SHADING_CLASSES as u32,
    };
    if params.pixel_count == 0 {
        return;
    }
    let groups = classification_dispatch(params.pixel_count).workgroups_x;
    buffers.clear_transient_state(ctx.command_encoder());
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism GPU material classify/count"),
                timestamp_writes: None,
            });
        pass.set_bind_group(0, input, &[]);
        pass.set_bind_group(1, materials_group, &[]);
        pass.set_bind_group(2, output, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.set_pipeline(classify);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism GPU material class prefix"),
                timestamp_writes: None,
            });
        pass.set_bind_group(0, input, &[]);
        pass.set_bind_group(1, materials_group, &[]);
        pass.set_bind_group(2, output, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.set_pipeline(prefix);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism GPU material work scatter"),
                timestamp_writes: None,
            });
        pass.set_bind_group(0, input, &[]);
        pass.set_bind_group(1, materials_group, &[]);
        pass.set_bind_group(2, output, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.set_pipeline(scatter);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("classification shader is WESL"),
        }
    }

    #[test]
    fn abi_sizes_and_dispatch_use_the_expected_contract() {
        assert_eq!(size_of::<GpuShadingClassificationParams>(), 24);
        assert_eq!(size_of::<GpuShadingPixelClass>(), 16);
        assert_eq!(size_of::<GpuShadingDispatchArgs>(), 12);
        assert_eq!(INVALID_CLASS, u32::MAX);
        assert_eq!(classification_dispatch(0).workgroups_x, 0);
        assert_eq!(classification_dispatch(65).workgroups_x, 2);
    }

    #[test]
    fn prefix_sum_is_contiguous_and_bounded() {
        let mut counts = [0; MAX_SHADING_CLASSES];
        counts[0] = 3;
        counts[2] = 4;
        let (offsets, total) = class_prefix_sum(counts);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 3);
        assert_eq!(offsets[2], 3);
        assert_eq!(offsets[3], 7);
        assert_eq!(total, 7);
    }

    #[test]
    fn classification_wesl_compiles_with_the_three_stage_abi() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d43_4c41_5353_4946_5900_0001),
        };
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                include_str!("../shaders/material_classification.wesl"),
                "shaders/prism_material_classification.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("material classification shader failed: {error}"));
    }
}
