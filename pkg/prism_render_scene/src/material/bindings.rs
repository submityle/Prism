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
use prism_render_material::{GpuMaterialHeader, GpuMaterialTexture};

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
    buffer_ids: Option<[BufferId; 3]>,
    virtual_texture_buffer_ids: Option<[BufferId; 2]>,
    generation_table_buffer_id: Option<BufferId>,
    buffer_version: u32,
    texture_version: u32,
    generation_table_version: u32,
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
            contents: bytemuck::cast_slice(&[[u32::MAX; 2]]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let execution_program_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("prism material execution programs"),
            contents: bytemuck::cast_slice(&[[0_u32; 4]]),
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
                    storage_buffer_read_only_sized(false, NonZero::new(8)),
                    storage_buffer_read_only_sized(false, NonZero::new(16)),
                    storage_buffer_read_only_sized(false, NonZero::new(4)),
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
                    (10, storage_buffer_read_only_sized(false, NonZero::new(8))),
                    (11, storage_buffer_read_only_sized(false, NonZero::new(16))),
                    (12, storage_buffer_read_only_sized(false, NonZero::new(4))),
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
            generation_table_version: 0,
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
        let Some((headers, parameters, textures)) = buffers.buffers() else {
            return;
        };
        let ids = [headers.id(), parameters.id(), textures.id()];
        let version = runtime.registry.snapshot().buffer_version;
        let texture_version = arrays.version();
        let generation_table_version = runtime.texture_generation_version();
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
            && self.generation_table_version == generation_table_version
            && self.execution_table_version == execution_table_version
            && self.generation_table_buffer_id == generation_table_buffer_id
        {
            return;
        }
        if self.bindless && self.generation_table_version != generation_table_version {
            if let Some(buffer) = self.generation_table_buffer.as_ref() {
                queue.write_buffer(
                    buffer,
                    0,
                    bytemuck::cast_slice(runtime.texture_generation_table()),
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
                    &[[u32::MAX; 2]]
                } else {
                    plans
                }),
            );
            queue.write_buffer(
                &self.execution_program_buffer,
                0,
                bytemuck::cast_slice(if programs.is_empty() {
                    &[[0_u32; 4]]
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
        self.generation_table_version = generation_table_version;
        self.execution_table_version = execution_table_version;
    }
}
