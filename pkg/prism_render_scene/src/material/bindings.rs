use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, storage_buffer_read_only_sized, texture_2d, texture_2d_array},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, Buffer, BufferId, BufferInitDescriptor,
        BufferUsages, SamplerBindingType, ShaderStages, TextureSampleType,
    },
    renderer::{RenderDevice, RenderQueue},
};
use core::num::NonZero;
use prism_render_material::{GpuMaterialHeader, GpuMaterialTexture, EXECUTION_PLAN_ROW_WORDS};

use super::texture_heap::RESERVED_GENERATION;
use super::{
    buffers::MaterialGpuBuffers,
    runtime::RenderMaterialRegistry,
    texture_upload::MaterialTextureArrays,
    virtual_texture::{MaterialVirtualTextureGpuBundle, MaterialVirtualTextureGpuConfig},
};

/// Shared binding contract for raster, visibility, shadows, GI, ray, and
/// offline consumers of the unified Material ABI.
///
/// Bindings 0/1/2 are the material header / parameter / texture storage tables.
/// When the device supports bindless textures the layout is extended with
/// bindings 3/4 — the `binding_array<texture_2d<f32>>` and
/// `binding_array<sampler>` that `material_sample.wesl` samples — sized to the
/// heap's device-clamped capacity. Bindings 5–8 are always present for the
/// dedicated Material-VT page table, config, atlas, and sampler; the disabled
/// resource bundle supplies a valid fallback until residency is enabled.
#[derive(Resource)]
pub struct MaterialBindGroup {
    pub layout: BindGroupLayout,
    pub layout_descriptor: BindGroupLayoutDescriptor,
    pub bind_group: Option<BindGroup>,
    bindless: bool,
    generation_table_buffer: Option<Buffer>,
    execution_plan_buffer: Buffer,
    execution_program_buffer: Buffer,
    execution_package_buffer: Buffer,
    buffer_ids: Option<[BufferId; 4]>,
    virtual_texture_buffer_ids: Option<[BufferId; 2]>,
    generation_table_buffer_id: Option<BufferId>,
    buffer_version: u32,
    texture_version: u32,
    execution_table_version: u32,
}

impl FromWorld for MaterialBindGroup {
    fn from_world(world: &mut World) -> Self {
        let (bindless, capacity) = {
            let arrays = world.resource::<MaterialTextureArrays>();
            (arrays.bindless(), arrays.capacity())
        };
        let device = world.resource::<RenderDevice>();
        let stages = ShaderStages::COMPUTE | ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let header = storage_buffer_read_only_sized(
            false,
            NonZero::new(size_of::<GpuMaterialHeader>() as u64),
        );
        // The parameter table is now a flat `u32` word heap holding
        // variable-length packed surface blocks (ABI v4), not a fixed-stride
        // `GpuSurfaceParameters` array. The minimum binding size is therefore a
        // single word; shaders decode `parameter_size` words at each material's
        // `parameter_offset` using its `lobe_mask`.
        let parameter =
            storage_buffer_read_only_sized(false, NonZero::new(size_of::<u32>() as u64));
        let texture = storage_buffer_read_only_sized(
            false,
            NonZero::new(size_of::<GpuMaterialTexture>() as u64),
        );
        let virtual_texture_page_table =
            storage_buffer_read_only_sized(false, NonZero::new(size_of::<u32>() as u64));
        // wgpu disallows a uniform buffer in a bind group that also contains
        // binding arrays. The fixed 32-byte VT record is read-only, so use
        // the portable storage-buffer form for both bindless and fallback
        // layouts.
        let virtual_texture_config = storage_buffer_read_only_sized(
            false,
            NonZero::new(MaterialVirtualTextureGpuConfig::byte_size()),
        );
        let virtual_texture_atlas = texture_2d_array(TextureSampleType::Float { filterable: true });
        let virtual_texture_sampler = sampler(SamplerBindingType::Filtering);
        let generation_table_buffer = bindless.then(|| {
            // Initialise every slot, not only the current high-water prefix.
            // Reserved fallback slots are valid from frame zero. Dynamic slots
            // that have never been materialised stay at generation zero, which
            // is the shader's invalid/stale sentinel; otherwise an arbitrary
            // `(index, generation=1)` row could pass the guard for an index
            // that the heap has never allocated.
            let mut initial_generations = vec![0_u32; capacity as usize];
            initial_generations[..super::texture_heap::FIRST_DYNAMIC_SLOT as usize]
                .fill(RESERVED_GENERATION);
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("prism material texture generation table"),
                contents: bytemuck::cast_slice(&initial_generations),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })
        });
        let execution_plan_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism material execution plans"),
            contents: bytemuck::cast_slice(&[[u32::MAX; EXECUTION_PLAN_ROW_WORDS]]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let execution_program_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism material execution programs"),
            contents: bytemuck::cast_slice(&[[0_u32; 5]]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let execution_package_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism material execution packages"),
            contents: bytemuck::cast_slice(&[0_u32]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let (layout, layout_descriptor) = if bindless {
            let count = NonZero::new(capacity)
                .expect("bindless capacity is validated to exceed the reserved slots");
            let entries = BindGroupLayoutEntries::sequential(
                stages,
                (
                    header,
                    parameter,
                    texture,
                    texture_2d(TextureSampleType::Float { filterable: true }).count(count),
                    sampler(SamplerBindingType::Filtering).count(count),
                    virtual_texture_page_table,
                    virtual_texture_config,
                    virtual_texture_atlas,
                    virtual_texture_sampler,
                    storage_buffer_read_only_sized(false, NonZero::new(size_of::<u32>() as u64)),
                    storage_buffer_read_only_sized(
                        false,
                        NonZero::new(
                            (EXECUTION_PLAN_ROW_WORDS * core::mem::size_of::<u32>()) as u64,
                        ),
                    ),
                    storage_buffer_read_only_sized(false, NonZero::new(16)),
                    storage_buffer_read_only_sized(false, NonZero::new(4)),
                    storage_buffer_read_only_sized(false, NonZero::new(64)),
                ),
            );
            (
                device.create_bind_group_layout("prism materials", &entries),
                BindGroupLayoutDescriptor::new("prism materials", &entries),
            )
        } else {
            let entries = BindGroupLayoutEntries::with_indices(
                stages,
                (
                    (0, header),
                    (1, parameter),
                    (2, texture),
                    (5, virtual_texture_page_table),
                    (6, virtual_texture_config),
                    (7, virtual_texture_atlas),
                    (8, virtual_texture_sampler),
                    (
                        10,
                        storage_buffer_read_only_sized(
                            false,
                            NonZero::new(
                                (EXECUTION_PLAN_ROW_WORDS * core::mem::size_of::<u32>()) as u64,
                            ),
                        ),
                    ),
                    (11, storage_buffer_read_only_sized(false, NonZero::new(16))),
                    (12, storage_buffer_read_only_sized(false, NonZero::new(4))),
                    (13, storage_buffer_read_only_sized(false, NonZero::new(64))),
                ),
            );
            (
                device.create_bind_group_layout("prism materials", &entries),
                BindGroupLayoutDescriptor::new("prism materials", &entries),
            )
        };

        Self {
            layout,
            layout_descriptor,
            bind_group: None,
            bindless,
            generation_table_buffer,
            execution_plan_buffer,
            execution_program_buffer,
            execution_package_buffer,
            buffer_ids: None,
            virtual_texture_buffer_ids: None,
            generation_table_buffer_id: None,
            buffer_version: 0,
            texture_version: 0,
            execution_table_version: 0,
        }
    }
}

impl MaterialBindGroup {
    /// Whether the bound layout carries the bindless texture/sampler arrays.
    #[must_use]
    pub(crate) const fn bindless(&self) -> bool {
        self.bindless
    }

    pub(crate) fn prepare(
        &mut self,
        device: &RenderDevice,
        buffers: &MaterialGpuBuffers,
        runtime: &RenderMaterialRegistry,
        arrays: &MaterialTextureArrays,
        virtual_texture: &MaterialVirtualTextureGpuBundle,
        queue: &RenderQueue,
    ) {
        let Some((headers, parameters, textures, surface_contexts)) = buffers.buffers() else {
            return;
        };
        let ids = [
            headers.id(),
            parameters.id(),
            textures.id(),
            surface_contexts.id(),
        ];
        let version = runtime.registry.snapshot().buffer_version;
        let texture_version = arrays.version();
        let execution_table_version = runtime.registry.execution_table_version();
        let generation_table_buffer_id = self.generation_table_buffer.as_ref().map(Buffer::id);
        let virtual_texture_buffer_ids = [
            virtual_texture.page_table_buffer().id(),
            virtual_texture.config_buffer().id(),
        ];
        if self.buffer_ids == Some(ids)
            && self.virtual_texture_buffer_ids == Some(virtual_texture_buffer_ids)
            && self.buffer_version == version
            && self.texture_version == texture_version
            && self.execution_table_version == execution_table_version
            && self.generation_table_buffer_id == generation_table_buffer_id
        {
            return;
        }
        if self.bindless && self.texture_version != texture_version {
            if let Some(buffer) = self.generation_table_buffer.as_ref() {
                queue.write_buffer(
                    buffer,
                    0,
                    bytemuck::cast_slice(arrays.published_generations()),
                );
            }
        }
        if self.execution_table_version != execution_table_version {
            let plans = runtime.registry.execution_plan_rows();
            let programs = runtime.registry.execution_program_rows();
            let packages = [runtime.registry.execution_package_count()];
            queue.write_buffer(
                &self.execution_plan_buffer,
                0,
                bytemuck::cast_slice(if plans.is_empty() {
                    &[[u32::MAX; EXECUTION_PLAN_ROW_WORDS]]
                } else {
                    plans
                }),
            );
            queue.write_buffer(
                &self.execution_program_buffer,
                0,
                bytemuck::cast_slice(if programs.is_empty() {
                    &[[0_u32; 5]]
                } else {
                    programs
                }),
            );
            queue.write_buffer(
                &self.execution_package_buffer,
                0,
                bytemuck::cast_slice(&packages),
            );
        }
        self.bind_group = Some(if self.bindless {
            // The bindless arrays extend the same bind group with the texture
            // and sampler `binding_array`s at bindings 3/4. `as_slice` yields
            // the `&[&wgpu::TextureView]` / `&[&wgpu::Sampler]` slices wgpu binds
            // as `TextureViewArray` / `SamplerArray`.
            let views = arrays.view_array();
            let samplers = arrays.sampler_array();
            device.create_bind_group(
                "prism materials",
                &self.layout,
                &BindGroupEntries::sequential((
                    headers.as_entire_binding(),
                    parameters.as_entire_binding(),
                    textures.as_entire_binding(),
                    views.as_slice(),
                    samplers.as_slice(),
                    virtual_texture.page_table_buffer().as_entire_binding(),
                    virtual_texture.config_buffer().as_entire_binding(),
                    virtual_texture.atlas_view(),
                    virtual_texture.sampler(),
                    self.generation_table_buffer
                        .as_ref()
                        .expect("bindless material layout requires generation table")
                        .as_entire_binding(),
                    self.execution_plan_buffer.as_entire_binding(),
                    self.execution_program_buffer.as_entire_binding(),
                    self.execution_package_buffer.as_entire_binding(),
                    surface_contexts.as_entire_binding(),
                )),
            )
        } else {
            device.create_bind_group(
                "prism materials",
                &self.layout,
                &BindGroupEntries::with_indices((
                    (0, headers.as_entire_binding()),
                    (1, parameters.as_entire_binding()),
                    (2, textures.as_entire_binding()),
                    (13, surface_contexts.as_entire_binding()),
                    (5, virtual_texture.page_table_buffer().as_entire_binding()),
                    (6, virtual_texture.config_buffer().as_entire_binding()),
                    (7, virtual_texture.atlas_view()),
                    (8, virtual_texture.sampler()),
                    (10, self.execution_plan_buffer.as_entire_binding()),
                    (11, self.execution_program_buffer.as_entire_binding()),
                    (12, self.execution_package_buffer.as_entire_binding()),
                )),
            )
        });
        self.buffer_ids = Some(ids);
        self.virtual_texture_buffer_ids = Some(virtual_texture_buffer_ids);
        self.generation_table_buffer_id = generation_table_buffer_id;
        self.buffer_version = version;
        self.texture_version = texture_version;
        self.execution_table_version = execution_table_version;
    }
}

#[cfg(test)]
mod tests {
    use super::MaterialBindGroup;
    use bevy_asset::{uuid::Uuid, AssetEvent, AssetId, Assets, Handle};
    use bevy_ecs::{message::Messages, schedule::Schedule, world::FromWorld};
    use bevy_image::Image;
    use bevy_pbr::StandardMaterial;
    use bevy_platform::future::block_on;
    use bevy_render::{
        render_asset::RenderAssets,
        renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
        texture::GpuImage,
        MainWorld,
    };
    use bevy_shader::{
        Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ShaderDefVal, ValidateShader,
    };
    use prism_render_architecture::gpu_scene::GpuCompletionValue;
    use prism_render_material::{
        MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot, NO_VIRTUAL_TEXTURE,
    };
    use std::mem::size_of;
    use wgpu::util::{BufferInitDescriptor, DeviceExt};
    use wgpu::{
        BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
        CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
        DeviceDescriptor, Extent3d, FilterMode, Instance, InstanceDescriptor, InstanceFlags,
        MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType,
        RequestAdapterOptions, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource,
        ShaderStages, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        TextureViewDescriptor,
    };

    #[test]
    fn generation_table_upload_uses_only_published_texture_arrays() {
        let source = include_str!("bindings.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("bindings production section");
        assert!(production.contains("arrays.published_generations()"));
        assert!(!production.contains("bytemuck::cast_slice(runtime.texture_generation_table())"));
    }

    #[test]
    fn surface_context_binding_is_generation_guarded_and_shared_by_texture_sampling() {
        let source = include_str!("../shaders/shading_resolve.wesl");
        assert!(
            source.contains("@group(1) @binding(13) var<storage, read> material_surface_contexts")
        );
        assert!(source.contains("surface_context.generation != material_header.generation"));
        assert!(source.contains("let material_uv = shading_resolve_transform_uv"));
        assert!(source.contains("sample_texture_slot(record, material_uv, lod)"));
        assert!(source.contains("material_uv + uv_gradients[0]"));
        assert!(source.contains("material_uv + uv_gradients[1]"));
    }

    fn require_gpu() -> bool {
        std::env::var_os("PRISM_REQUIRE_GPU").is_some()
    }

    fn try_device() -> Option<(Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = match block_on(instance.request_adapter(&RequestAdapterOptions::default())) {
            Ok(adapter) => adapter,
            Err(error) => {
                assert!(
                    !require_gpu(),
                    "PRISM_REQUIRE_GPU set: Material-VT production closure adapter request failed: {error}"
                );
                return None;
            }
        };
        match block_on(adapter.request_device(&DeviceDescriptor::default())) {
            Ok((device, queue)) => Some((instance, adapter, device, queue)),
            Err(error) => {
                assert!(
                    !require_gpu(),
                    "PRISM_REQUIRE_GPU set: Material-VT production closure device request failed: {error}"
                );
                None
            }
        }
    }

    fn source_snapshot(texture_id: u32, seed: u8) -> MaterialVtSourceSnapshot {
        const EXTENT: u32 = super::super::vt_source::DEFAULT_MATERIAL_VT_PAGE_SIZE;
        const MIP_COUNT: u32 = 8;
        let descriptor = MaterialVtSourceDescriptor::new(
            texture_id,
            EXTENT,
            EXTENT,
            1,
            EXTENT,
            MIP_COUNT as u8,
            MaterialVtFormat::Rgba8Unorm,
        )
        .expect("valid default-producer source descriptor");
        let mut bytes = Vec::new();
        for mip in 0..MIP_COUNT {
            let extent = (EXTENT >> mip).max(1);
            bytes.extend(std::iter::repeat_n(seed, (extent * extent * 4) as usize));
        }
        MaterialVtSourceSnapshot::new(descriptor, bytes)
            .expect("valid default-producer source payload")
    }

    fn material_texture_row(
        world: &bevy_ecs::world::World,
        material_id: AssetId<StandardMaterial>,
    ) -> prism_render_material::GpuMaterialTexture {
        let runtime = world.resource::<super::RenderMaterialRegistry>();
        let handle = runtime
            .material_handle(material_id)
            .expect("published material handle");
        let record = runtime
            .registry
            .get(handle)
            .expect("published material record");
        assert_eq!(record.textures.len(), 1);
        record.textures[0]
    }

    fn upload_material_buffers(world: &mut bevy_ecs::world::World) {
        world.resource_scope(
            |world, mut buffers: bevy_ecs::prelude::Mut<super::MaterialGpuBuffers>| {
                let mut runtime = world.resource_mut::<super::RenderMaterialRegistry>();
                buffers.apply_dirty(&mut runtime);
            },
        );
        world.resource_scope(
            |world, mut buffers: bevy_ecs::prelude::Mut<super::MaterialGpuBuffers>| {
                let device = world.resource::<RenderDevice>();
                let queue = world.resource::<RenderQueue>();
                buffers.headers.write_buffers(&device, &queue);
                buffers.parameters.write_buffers(&device, &queue);
                buffers.textures.write_buffers(&device, &queue);
                buffers.surface_contexts.write_buffers(&device, &queue);
            },
        );
    }

    fn prepare_bind_group(world: &mut bevy_ecs::world::World) {
        world.resource_scope(
            |world, mut bindings: bevy_ecs::prelude::Mut<MaterialBindGroup>| {
                bindings.prepare(
                    &world.resource::<RenderDevice>(),
                    &world.resource::<super::MaterialGpuBuffers>(),
                    &world.resource::<super::RenderMaterialRegistry>(),
                    &world.resource::<super::MaterialTextureArrays>(),
                    &world.resource::<super::MaterialVirtualTextureGpuBundle>(),
                    &world.resource::<RenderQueue>(),
                );
            },
        );
    }

    const MATERIAL_VT_PROBE_WESL: &str = r#"
import prism_render_material::shaders::material::{
    PrismMaterialHeader,
    PrismMaterialTexture,
};
import prism_render_material::shaders::material_sample::{sample_texture_slot};

struct ProbeInput {
    material_index: u32,
}

@group(0) @binding(0) var<storage, read> inputs: array<ProbeInput>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(1) @binding(0) var<storage, read> material_headers: array<PrismMaterialHeader>;
@group(1) @binding(2) var<storage, read> material_textures: array<PrismMaterialTexture>;

@compute @workgroup_size(1)
fn material_vt_production_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&inputs) { return; }
    let material_index = inputs[id.x].material_index;
    let header = material_headers[material_index];
    if header.texture_count == 0u {
        output[id.x] = vec4<f32>(-1.0);
        return;
    }
    let record = material_textures[header.texture_offset];
    output[id.x] = sample_texture_slot(record, vec2<f32>(0.5), 0.0);
}
"#;

    fn keep_wgsl(
        _: &(),
        source: ShaderCacheSource,
        _: &ValidateShader,
    ) -> Result<String, ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("Material-VT probe uses WESL"),
        }
    }

    fn shader_id(uuid: u128) -> AssetId<Shader> {
        AssetId::Uuid {
            uuid: Uuid::from_u128(uuid),
        }
    }

    fn compile_material_vt_probe(bindless: bool) -> String {
        let mut cache = ShaderCache::new((), keep_wgsl);
        let material = shader_id(0x5052_4953_4d5f_5052_4f42_455f_4d41_0001);
        let virtual_texture = shader_id(0x5052_4953_4d5f_5052_4f42_455f_5654_0001);
        let sample = shader_id(0x5052_4953_4d5f_5052_4f42_455f_5341_0001);
        let probe = shader_id(0x5052_4953_4d5f_5052_4f42_455f_5052_0001);
        cache.set_shader(
            material,
            Shader::from_wesl(
                prism_render_material::shaders::MATERIAL_SHADER_SOURCE,
                "embedded://prism_render_material/shaders/material.wesl",
            ),
        );
        cache.set_shader(
            virtual_texture,
            Shader::from_wesl(
                prism_render_material::shaders::MATERIAL_VIRTUAL_TEXTURE_SHADER_SOURCE,
                "embedded://prism_render_material/shaders/material_virtual_texture.wesl",
            ),
        );
        cache.set_shader(
            sample,
            Shader::from_wesl(
                prism_render_material::shaders::MATERIAL_SAMPLE_SHADER_SOURCE,
                "embedded://prism_render_material/shaders/material_sample.wesl",
            ),
        );
        cache.set_shader(
            probe,
            Shader::from_wesl(
                MATERIAL_VT_PROBE_WESL,
                "embedded://prism_render_scene/tests/material_vt_production_probe.wesl",
            ),
        );
        let defs = if bindless {
            vec![ShaderDefVal::Bool("MATERIAL_BINDLESS".into(), true)]
        } else {
            Vec::new()
        };
        cache
            .get(0, probe, &defs)
            .unwrap_or_else(|error| panic!("production Material-VT probe failed to link: {error}"))
            .as_ref()
            .clone()
    }

    fn readback_rgba(device: &wgpu::Device, buffer: &wgpu::Buffer) -> [f32; 4] {
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(PollType::wait_indefinitely())
            .expect("Material-VT production probe poll");
        receiver
            .recv()
            .expect("Material-VT production probe callback")
            .expect("Material-VT production probe mapping");
        let mapped = slice
            .get_mapped_range()
            .expect("Material-VT production probe mapped range");
        let result = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped)[0];
        drop(mapped);
        buffer.unmap();
        result
    }

    fn add_gpu_image_for_bindless_generation(
        world: &mut bevy_ecs::world::World,
        image: AssetId<Image>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let texture_descriptor = TextureDescriptor {
            label: Some("material-vt production probe direct image"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let texture = device.create_texture(&texture_descriptor);
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("material-vt production probe direct sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        queue.write_texture(
            texture.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            texture_descriptor.size,
        );
        world.insert_resource(RenderAssets::<GpuImage>::default());
        world.resource_mut::<RenderAssets<GpuImage>>().insert(
            image,
            GpuImage {
                texture: texture.into(),
                texture_view: texture_view.into(),
                sampler: sampler.into(),
                texture_descriptor,
                texture_view_descriptor: None,
                had_data: true,
            },
        );
    }

    fn compose_material_texture_arrays(world: &mut bevy_ecs::world::World) {
        world.resource_scope(
            |world,
             mut arrays: bevy_ecs::prelude::Mut<
                super::super::texture_upload::MaterialTextureArrays,
            >| {
                let runtime = world.resource::<super::RenderMaterialRegistry>();
                let images = world.resource::<RenderAssets<GpuImage>>();
                arrays.compose(runtime, images);
            },
        );
    }

    #[test]
    fn material_vt_producer_publication_rebuilds_and_samples_production_bind_group() {
        let Some((_instance, adapter, device, queue)) = try_device() else {
            eprintln!(
                "material_vt_producer_publication_rebuilds_production_bind_group: no GPU adapter; skipped"
            );
            return;
        };
        let mut world = bevy_ecs::world::World::new();
        world.insert_resource(RenderDevice::from(device.clone()));
        world.insert_resource(RenderQueue::new(queue.clone()));
        world.insert_resource(RenderAdapterInfo::new(adapter.get_info()));
        let texture_arrays = super::MaterialTextureArrays::from_world(&mut world);
        world.insert_resource(texture_arrays);
        let virtual_texture_bundle = super::MaterialVirtualTextureGpuBundle::from_world(&mut world);
        world.insert_resource(virtual_texture_bundle);
        let material_bind_group = MaterialBindGroup::from_world(&mut world);
        world.insert_resource(material_bind_group);
        let material_buffers = super::MaterialGpuBuffers::from_world(&mut world);
        world.insert_resource(material_buffers);
        world.insert_resource(super::RenderMaterialRegistry::default());
        world.insert_resource(super::super::vt_source::MaterialVtSourceRegistry::default());
        world.insert_resource(super::super::vt_source::MaterialVtActivePublication::default());
        world.insert_resource(super::super::vt_source::MaterialVtSourceChangeQueue::default());
        world.insert_resource(super::super::vt_source::MaterialVtMaterialSourceQueries::default());
        world.insert_resource(super::super::virtual_texture::MaterialVtDemandRegistry::default());
        world.insert_resource(super::super::vt_producer::MaterialVtProducer::default());
        world.insert_resource(crate::completion::GpuCompletionTracker::default());
        world.insert_resource(super::super::runtime::PrismMaterialDiagnostics::default());

        let image: Handle<Image> = Handle::from(Uuid::from_u128(0x5052_4953_4d5f_5654_0000_0001));
        let material_id: AssetId<StandardMaterial> =
            Handle::from(Uuid::from_u128(0x5052_4953_4d5f_4d41_0000_0001)).id();
        let identity = world
            .resource_mut::<super::super::vt_source::MaterialVtSourceRegistry>()
            .insert_snapshot(image.id(), source_snapshot(7, 173));
        let material = StandardMaterial {
            base_color_texture: Some(image.clone()),
            ..Default::default()
        };
        let mut materials = Assets::<StandardMaterial>::default();
        materials
            .insert(material_id, material)
            .expect("UUID material insertion should succeed");
        let mut material_events = Messages::<AssetEvent<StandardMaterial>>::default();
        material_events.write(AssetEvent::Added { id: material_id });
        let mut main_world = MainWorld::default();
        main_world.insert_resource(materials);
        main_world.insert_resource(material_events);
        world.insert_resource(main_world);

        let mut material_schedule = Schedule::default();
        material_schedule.add_systems(super::super::systems::extract_standard_materials);
        material_schedule.run(&mut world);
        assert_eq!(
            material_texture_row(&world, material_id).virtual_texture_id,
            NO_VIRTUAL_TEXTURE,
            "a CPU source alone must not advertise VT before completion publication"
        );
        upload_material_buffers(&mut world);
        prepare_bind_group(&mut world);
        let (first_group, first_vt_buffers) = {
            let bindings = world.resource::<MaterialBindGroup>();
            (
                bindings
                    .bind_group
                    .as_ref()
                    .expect("initial production material bind group")
                    .id(),
                bindings
                    .virtual_texture_buffer_ids
                    .expect("initial VT buffer identity"),
            )
        };
        assert_eq!(
            world
                .resource::<super::MaterialVirtualTextureGpuBundle>()
                .active_slot(),
            0
        );

        let mut producer_schedule = Schedule::default();
        producer_schedule.add_systems(super::super::vt_producer::prepare_material_virtual_texture);
        producer_schedule.run(&mut world);
        assert_eq!(
            world
                .resource::<super::MaterialVirtualTextureGpuBundle>()
                .active_slot(),
            0,
            "the staging slice must remain unpublished before queue completion"
        );
        assert!(!world
            .resource::<super::super::vt_source::MaterialVtActivePublication>()
            .contains(identity));

        let tracked = {
            let tracker = world.resource::<crate::completion::GpuCompletionTracker>();
            tracker.track_host(&world.resource::<RenderQueue>())
        };
        assert_eq!(tracked, GpuCompletionValue(1));
        let encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("material-VT production closure completion"),
        });
        queue.submit(Some(encoder.finish()));
        device
            .poll(PollType::wait_indefinitely())
            .expect("Material-VT production closure completion poll");
        assert_eq!(
            world
                .resource::<crate::completion::GpuCompletionTracker>()
                .completed_value(),
            tracked
        );

        producer_schedule.run(&mut world);
        assert_eq!(
            world
                .resource::<super::MaterialVirtualTextureGpuBundle>()
                .active_slot(),
            1,
            "completion must flip the uploaded GPU slice active"
        );
        assert!(world
            .resource::<super::super::vt_source::MaterialVtActivePublication>()
            .contains(identity));

        material_schedule.run(&mut world);
        assert_eq!(
            material_texture_row(&world, material_id).virtual_texture_id,
            7,
            "the republished material row must reference the completion-published VT source"
        );
        assert_eq!(
            world
                .resource::<super::super::runtime::PrismMaterialDiagnostics>()
                .updated,
            1,
            "the production material extraction system must consume the VT source-change queue"
        );
        upload_material_buffers(&mut world);
        prepare_bind_group(&mut world);

        let bindings = world.resource::<MaterialBindGroup>();
        let second_vt_buffers = bindings
            .virtual_texture_buffer_ids
            .expect("published VT buffer identity");
        let second_group = bindings
            .bind_group
            .as_ref()
            .expect("published production material bind group")
            .id();
        assert_ne!(first_vt_buffers, second_vt_buffers);
        assert_ne!(first_group, second_group);
        let active_slot = {
            let bundle = world.resource::<super::MaterialVirtualTextureGpuBundle>();
            assert_eq!(
                second_vt_buffers,
                [bundle.page_table_buffer().id(), bundle.config_buffer().id()],
                "the rebuilt group must bind the newly active VT slice"
            );
            bundle.active_slot()
        };

        // Bindless material rows are guarded by the same generation table even
        // when the successful branch is VT. Supply a real uploaded GpuImage so
        // the test follows the production publication rule instead of mutating
        // the generation table by hand.
        let bindless = world.resource::<super::MaterialTextureArrays>().bindless();
        if bindless {
            add_gpu_image_for_bindless_generation(&mut world, image.id(), &device, &queue);
            compose_material_texture_arrays(&mut world);
            prepare_bind_group(&mut world);
        }

        let handle = world
            .resource::<super::RenderMaterialRegistry>()
            .material_handle(material_id)
            .expect("published material handle for GPU probe");
        let wgsl = compile_material_vt_probe(bindless);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("material-vt production probe"),
            source: ShaderSource::Wgsl(wgsl.into()),
        });
        let cases_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("material-vt production probe inputs"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let material_layout = world.resource::<MaterialBindGroup>().layout.clone();
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("material-vt production probe layout"),
            bind_group_layouts: &[Some(&cases_layout), Some(&material_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("material-vt production probe pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("material_vt_production_probe"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let input = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("material-vt production probe input"),
            contents: bytemuck::bytes_of(&handle.index),
            usage: BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&BufferDescriptor {
            label: Some("material-vt production probe output"),
            size: size_of::<[f32; 4]>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("material-vt production probe readback"),
            size: size_of::<[f32; 4]>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cases = device.create_bind_group(&BindGroupDescriptor {
            label: Some("material-vt production probe inputs"),
            layout: &cases_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let materials = world
            .resource::<MaterialBindGroup>()
            .bind_group
            .as_ref()
            .expect("production material bind group must be available for sampling");
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("material-vt production probe encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("material-vt production probe dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &cases, &[]);
            pass.set_bind_group(1, materials, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size_of::<[f32; 4]>() as u64);
        queue.submit([encoder.finish()]);
        let got = readback_rgba(&device, &readback);
        let expected = 173.0 / 255.0;
        for (channel, value) in got.iter().copied().enumerate() {
            assert!(
                (value - expected).abs() <= 1.0 / 255.0 + 1.0e-5,
                "production publication -> sampling -> readback channel {channel}: got {value}, expected {expected}; active_slot={active_slot} bindless={bindless}",
            );
        }
    }
}
