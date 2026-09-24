use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, storage_buffer_read_only_sized, texture_2d},
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
use prism_render_material::{GpuMaterialHeader, GpuMaterialTexture, GpuSurfaceParameters};

use super::{
    buffers::MaterialGpuBuffers, runtime::RenderMaterialRegistry,
    texture_upload::MaterialTextureArrays,
};

/// Shared binding contract for raster, visibility, shadows, GI, ray, and
/// offline consumers of the unified Material ABI.
///
/// Bindings 0/1/2 are the material header / parameter / texture storage tables.
/// When the device supports bindless textures the layout is extended with
/// bindings 3/4 — the `binding_array<texture_2d<f32>>` and
/// `binding_array<sampler>` that `material_sample.wesl` samples — sized to the
/// heap's device-clamped capacity. On non-bindless devices only the three
/// storage bindings are present.
#[derive(Resource)]
pub struct MaterialBindGroup {
    pub layout: BindGroupLayout,
    pub layout_descriptor: BindGroupLayoutDescriptor,
    pub bind_group: Option<BindGroup>,
    bindless: bool,
    buffer_ids: Option<[BufferId; 3]>,
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
        let parameter = storage_buffer_read_only_sized(
            false,
            NonZero::new(size_of::<GpuSurfaceParameters>() as u64),
        );
        let texture = storage_buffer_read_only_sized(
            false,
            NonZero::new(size_of::<GpuMaterialTexture>() as u64),
        );

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
                ),
            );
            (
                device.create_bind_group_layout("prism materials", &entries),
                BindGroupLayoutDescriptor::new("prism materials", &entries),
            )
        } else {
            let entries = BindGroupLayoutEntries::sequential(stages, (header, parameter, texture));
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
            buffer_version: 0,
            texture_version: 0,
        }
    }
}

impl MaterialBindGroup {
    pub(crate) fn prepare(
        &mut self,
        device: &RenderDevice,
        buffers: &MaterialGpuBuffers,
        runtime: &RenderMaterialRegistry,
        arrays: &MaterialTextureArrays,
    ) {
        let Some((headers, parameters, textures)) = buffers.buffers() else {
            return;
        };
        let ids = [headers.id(), parameters.id(), textures.id()];
        let version = runtime.registry.snapshot().buffer_version;
        let texture_version = arrays.version();
        if self.buffer_ids == Some(ids)
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
                )),
            )
        } else {
            device.create_bind_group(
                "prism materials",
                &self.layout,
                &BindGroupEntries::sequential((
                    headers.as_entire_binding(),
                    parameters.as_entire_binding(),
                    textures.as_entire_binding(),
                )),
            )
        });
        self.buffer_ids = Some(ids);
        self.buffer_version = version;
        self.texture_version = texture_version;
    }
}
