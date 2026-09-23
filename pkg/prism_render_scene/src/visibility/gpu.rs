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
        ComputePipelineDescriptor, PipelineCache, ShaderStages,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use crate::{buffers::GpuSceneBindGroup, GeometryBindGroup, MaterialBindGroup};

use super::{
    buffers::UnifiedVisibilityBuffers,
    rows::{RenderVisibilityCounter, RenderVisibilityView},
};

#[derive(Resource)]
pub(crate) struct VisibilityComputePipeline {
    pub pipeline: CachedComputePipelineId,
    pub output_layout: BindGroupLayout,
}

#[derive(Resource, Default)]
pub(crate) struct VisibilityComputeBindGroup {
    pub bind_group: Option<BindGroup>,
    buffer_ids: Option<[BufferId; 6]>,
}

pub(crate) fn init_visibility_compute_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    scene: Res<GpuSceneBindGroup>,
    materials: Res<MaterialBindGroup>,
    geometry: Res<GeometryBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only::<RenderVisibilityView>(false),
            storage_buffer::<RenderVisibilityCounter>(false),
            storage_buffer::<super::rows::RenderVisibilityWorkItem>(false),
            storage_buffer::<super::rows::RenderVisibilityRange>(false),
            storage_buffer::<super::rows::RenderVisibilityIndirect>(false),
            storage_buffer::<u32>(false),
        ),
    );
    let output_descriptor = BindGroupLayoutDescriptor::new("prism visibility output", &entries);
    let output_layout = device.create_bind_group_layout("prism visibility output", &entries);
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/visibility.wesl");
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism unified visibility".into()),
        layout: vec![
            scene.layout_descriptor.clone(),
            materials.layout_descriptor.clone(),
            geometry.layout_descriptor.clone(),
            output_descriptor.clone(),
        ],
        immediate_size: 16,
        shader,
        entry_point: Some("cull_instances".into()),
        ..Default::default()
    });
    commands.insert_resource(VisibilityComputePipeline {
        pipeline,
        output_layout,
    });
}

pub(crate) fn prepare_visibility_compute_bind_group(
    pipeline: Res<VisibilityComputePipeline>,
    buffers: Res<UnifiedVisibilityBuffers>,
    device: Res<RenderDevice>,
    mut bindings: ResMut<VisibilityComputeBindGroup>,
) {
    let Some((views, counters, work, ranges, indirect, overflow)) = buffers.compute_buffers()
    else {
        return;
    };
    let ids = [
        views.id(),
        counters.id(),
        work.id(),
        ranges.id(),
        indirect.id(),
        overflow.id(),
    ];
    if bindings.buffer_ids == Some(ids) {
        return;
    }
    bindings.bind_group = Some(device.create_bind_group(
        "prism visibility output",
        &pipeline.output_layout,
        &BindGroupEntries::sequential((
            views.as_entire_binding(),
            counters.as_entire_binding(),
            work.as_entire_binding(),
            ranges.as_entire_binding(),
            indirect.as_entire_binding(),
            overflow.as_entire_binding(),
        )),
    ));
    bindings.buffer_ids = Some(ids);
}

#[cfg(test)]
mod tests {
    use super::{
        storage_buffer, storage_buffer_read_only, RenderVisibilityCounter, RenderVisibilityView,
    };
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("visibility shader is WESL"),
        }
    }

    #[test]
    fn visibility_compute_wesl_compiles() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d56_4953_4942_494c_4954_5901),
        };
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                include_str!("../shaders/visibility.wesl"),
                "shaders/prism_visibility.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("visibility compute shader failed: {error}"));
    }

    #[test]
    fn output_layout_matches_shader_access_modes() {
        use bevy_material::bind_group_layout_entries::BindGroupLayoutEntries;
        use bevy_render::render_resource::ShaderStages;
        use bevy_render::render_resource::{BindingType, BufferBindingType};

        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only::<RenderVisibilityView>(false),
                storage_buffer::<RenderVisibilityCounter>(false),
                storage_buffer::<super::super::rows::RenderVisibilityWorkItem>(false),
                storage_buffer::<super::super::rows::RenderVisibilityRange>(false),
                storage_buffer::<super::super::rows::RenderVisibilityIndirect>(false),
                storage_buffer::<u32>(false),
            ),
        );
        assert!(matches!(
            entries[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                ..
            }
        ));
        assert!(entries[1..].iter().all(|entry| matches!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                ..
            }
        )));
    }
}
