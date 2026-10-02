//! Bind-group contract exposing the clustered-light tables to the resolve pass.
//!
//! All three buffers are bound as read-only storage: the single-element grid
//! record, the per-cluster `[offset, count]` table, and the flat light-index
//! list.  The grid is a single-element storage array rather than a uniform so
//! its `std430` layout matches the CPU-side `#[repr(C)]` record exactly,
//! matching the sibling light bind group's convention.

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

use super::abi::GpuClusterGrid;
use super::buffers::ClusterGpuBuffers;

/// Shared binding contract for consumers of the clustered light tables.
#[derive(Resource)]
pub struct ClusterBindGroup {
    /// The GPU layout the resolve pipeline is created against.
    pub layout: BindGroupLayout,
    /// The reflected descriptor, kept for pipeline specialization/validation.
    pub layout_descriptor: BindGroupLayoutDescriptor,
    /// The most recently prepared bind group, if the buffers have uploaded.
    pub bind_group: Option<BindGroup>,
    buffer_ids: Option<[BufferId; 3]>,
    buffer_version: u32,
}

impl FromWorld for ClusterBindGroup {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuClusterGrid>() as u64),
                ),
                storage_buffer_read_only_sized(false, NonZero::new(size_of::<[u32; 2]>() as u64)),
                storage_buffer_read_only_sized(false, NonZero::new(size_of::<u32>() as u64)),
            ),
        );
        let layout_descriptor = BindGroupLayoutDescriptor::new("prism clustered lights", &entries);
        Self {
            layout: device.create_bind_group_layout("prism clustered lights", &entries),
            layout_descriptor,
            bind_group: None,
            buffer_ids: None,
            buffer_version: 0,
        }
    }
}

impl ClusterBindGroup {
    pub(crate) fn prepare(&mut self, device: &RenderDevice, buffers: &ClusterGpuBuffers) {
        let Some((grid, offsets, indices)) = buffers.buffers() else {
            return;
        };
        let ids = [grid.id(), offsets.id(), indices.id()];
        let version = buffers.version();
        if self.buffer_ids == Some(ids) && self.buffer_version == version {
            return;
        }
        self.bind_group = Some(device.create_bind_group(
            "prism clustered lights",
            &self.layout,
            &BindGroupEntries::sequential((
                grid.as_entire_binding(),
                offsets.as_entire_binding(),
                indices.as_entire_binding(),
            )),
        ));
        self.buffer_ids = Some(ids);
        self.buffer_version = version;
    }
}
