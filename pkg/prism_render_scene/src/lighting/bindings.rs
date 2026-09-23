//! Bind-group contract exposing the extracted light buffers to the resolve
//! compute shader (and any future clustered/forward light consumers).
//!
//! All three buffers are bound as read-only storage.  The environment is a
//! single-element storage array rather than a uniform so its `std430` layout
//! matches the CPU-side `#[repr(C)]` record exactly, side-stepping the
//! `std140` vec3 padding rules a uniform block would impose.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::storage_buffer_read_only_sized, BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{BindGroup, BindGroupEntries, BindGroupLayout, BufferId, ShaderStages},
    renderer::RenderDevice,
};
use core::num::NonZero;

use super::{
    abi::{GpuDirectionalLight, GpuLightEnvironment, GpuPunctualLight},
    buffers::LightGpuBuffers,
};

/// Shared binding contract for consumers of the extracted light tables.
#[derive(Resource)]
pub struct LightBindGroup {
    /// The GPU layout the resolve pipeline is created against.
    pub layout: BindGroupLayout,
    /// The reflected descriptor, kept for pipeline specialization/validation.
    pub layout_descriptor: BindGroupLayoutDescriptor,
    /// The most recently prepared bind group, if the buffers have uploaded.
    pub bind_group: Option<BindGroup>,
    buffer_ids: Option<[BufferId; 3]>,
    buffer_version: u32,
}

impl FromWorld for LightBindGroup {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuDirectionalLight>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuPunctualLight>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuLightEnvironment>() as u64),
                ),
            ),
        );
        let layout_descriptor = BindGroupLayoutDescriptor::new("prism lights", &entries);
        Self {
            layout: device.create_bind_group_layout("prism lights", &entries),
            layout_descriptor,
            bind_group: None,
            buffer_ids: None,
            buffer_version: 0,
        }
    }
}

impl LightBindGroup {
    pub(crate) fn prepare(&mut self, device: &RenderDevice, buffers: &LightGpuBuffers) {
        let Some((directionals, punctuals, environment)) = buffers.buffers() else {
            return;
        };
        let ids = [directionals.id(), punctuals.id(), environment.id()];
        let version = buffers.version();
        if self.buffer_ids == Some(ids) && self.buffer_version == version {
            return;
        }
        self.bind_group = Some(device.create_bind_group(
            "prism lights",
            &self.layout,
            &BindGroupEntries::sequential((
                directionals.as_entire_binding(),
                punctuals.as_entire_binding(),
                environment.as_entire_binding(),
            )),
        ));
        self.buffer_ids = Some(ids);
        self.buffer_version = version;
    }
}
