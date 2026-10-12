use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, texture_2d, texture_2d_array,
            uniform_buffer_sized,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, BufferId, SamplerBindingType, ShaderStages,
        TextureSampleType,
    },
    renderer::RenderDevice,
};
use core::num::NonZero;
use prism_render_material::{GpuMaterialHeader, GpuMaterialTexture};

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
    buffer_ids: Option<[BufferId; 3]>,
    virtual_texture_buffer_ids: Option<[BufferId; 2]>,
    buffer_version: u32,
    texture_version: u32,
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
        let virtual_texture_config = uniform_buffer_sized(
            false,
            NonZero::new(MaterialVirtualTextureGpuConfig::byte_size()),
        );
        let virtual_texture_atlas = texture_2d_array(TextureSampleType::Float { filterable: true });
        let virtual_texture_sampler = sampler(SamplerBindingType::Filtering);

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
            buffer_ids: None,
            virtual_texture_buffer_ids: None,
            buffer_version: 0,
            texture_version: 0,
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
    ) {
        let Some((headers, parameters, textures)) = buffers.buffers() else {
            return;
        };
        let ids = [headers.id(), parameters.id(), textures.id()];
        let version = runtime.registry.snapshot().buffer_version;
        let texture_version = arrays.version();
        let virtual_texture_buffer_ids = [
            virtual_texture.page_table_buffer().id(),
            virtual_texture.config_buffer().id(),
        ];
        if self.buffer_ids == Some(ids)
            && self.virtual_texture_buffer_ids == Some(virtual_texture_buffer_ids)
            && self.buffer_version == version
            && self.texture_version == texture_version
        {
            return;
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
                )),
            )
        });
        self.buffer_ids = Some(ids);
        self.virtual_texture_buffer_ids = Some(virtual_texture_buffer_ids);
        self.buffer_version = version;
        self.texture_version = texture_version;
    }
}
