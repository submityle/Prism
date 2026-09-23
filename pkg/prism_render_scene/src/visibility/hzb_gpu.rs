use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer, storage_buffer_read_only, texture_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, CachedComputePipelineId, ComputePipelineDescriptor,
        ShaderStages, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_render::render_resource::ShaderType;
use bevy_shader::Shader;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bevy_render::render_resource::ShaderType)]
pub(crate) struct RenderHzbCullInput {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub nearest_depth: f32,
    pub projected_velocity: f32,
}

#[derive(Resource)]
pub(crate) struct HzbVisibilityPipeline {
    pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub(crate) struct HzbVisibilityBuffers {
    candidates: Buffer,
    stages: Buffer,
    capacity: u32,
}

impl FromWorld for HzbVisibilityBuffers {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        Self::with_capacity(device, 1)
    }
}

impl HzbVisibilityBuffers {
    fn with_capacity(device: &RenderDevice, capacity: u32) -> Self {
        let capacity = capacity.max(1);
        Self {
            candidates: device.create_buffer(&BufferDescriptor {
                label: Some("prism hzb candidates"),
                size: capacity as u64 * RenderHzbCullInput::min_size().get(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stages: device.create_buffer(&BufferDescriptor {
                label: Some("prism hzb stages"),
                size: capacity as u64 * size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            capacity,
        }
    }

    pub(crate) fn ensure_capacity(&mut self, device: &RenderDevice, capacity: u32) {
        if capacity > self.capacity {
            *self = Self::with_capacity(device, capacity.next_power_of_two());
        }
    }

    pub(crate) fn bindings(&self) -> (&Buffer, &Buffer) {
        (&self.candidates, &self.stages)
    }
}

pub(crate) fn inspect_hzb_visibility_pipeline(
    pipeline: Res<HzbVisibilityPipeline>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
    mut buffers: ResMut<HzbVisibilityBuffers>,
    device: Res<RenderDevice>,
) {
    buffers.ensure_capacity(&device, 1);
    let _ = buffers.bindings();
    let _ = (&pipeline.layout, cache.get_compute_pipeline(pipeline.pipeline));
}

pub(crate) fn init_hzb_visibility_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only::<RenderHzbCullInput>(false),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer::<u32>(false),
        ),
    );
    let layout = BindGroupLayoutDescriptor::new("prism hzb visibility", &entries);
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/hzb_visibility.wesl");
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism hzb visibility".into()),
        layout: vec![layout.clone()],
        immediate_size: 32,
        shader,
        entry_point: Some("classify_hzb".into()),
        ..Default::default()
    });
    // Force layout validation against the actual device during startup.
    let _ = device.create_bind_group_layout("prism hzb visibility", &entries);
    commands.insert_resource(HzbVisibilityPipeline { pipeline, layout });
}

#[cfg(test)]
mod tests {
    use super::RenderHzbCullInput;
    use bevy_render::render_resource::ShaderType;
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("HZB shader is WESL"),
        }
    }

    #[test]
    fn hzb_visibility_wesl_compiles() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d48_5a42_5649_5349_4249_4c01),
        };
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                include_str!("../shaders/hzb_visibility.wesl"),
                "shaders/prism_hzb_visibility.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("HZB visibility shader failed: {error}"));
    }

    #[test]
    fn hzb_rows_match_shader_layout() {
        assert_eq!(RenderHzbCullInput::min_size().get(), 24);
    }
}
