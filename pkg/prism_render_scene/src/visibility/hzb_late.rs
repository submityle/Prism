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
        CachedComputePipelineId, ComputePipelineDescriptor, ShaderStages,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

#[derive(Resource)]
pub(crate) struct HzbLateCompactPipeline {
    pub(crate) pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
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
        ),
    );
    let layout = BindGroupLayoutDescriptor::new("prism hzb late compact", &entries);
    let _ = device.create_bind_group_layout("prism hzb late compact", &entries);
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
    commands.insert_resource(HzbLateCompactPipeline { pipeline, layout });
}

pub(crate) fn inspect_hzb_late_pipeline(
    pipeline: Res<HzbLateCompactPipeline>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
) {
    let _ = (&pipeline.layout, cache.get_compute_pipeline(pipeline.pipeline));
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
