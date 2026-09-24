//! Bind-group contract exposing the shadow atlas and shadow tables to the
//! resolve compute shader.
//!
//! The resolve pass reserves group 4 for shadows (group 0 view / 1 materials /
//! 2 scene / 3 lights are already taken).  This layout binds, in order:
//!
//! * binding 0 — the `texture_2d_array<f32>` shadow atlas,
//! * binding 1 — the linear `sampler` used for its manual depth comparison,
//! * binding 2 — the directional-shadow storage array,
//! * binding 3 — the point-shadow storage array,
//! * binding 4 — the single-element shadow globals record.
//!
//! The three buffers are read-only storage (not uniforms) so their `std430`
//! layout matches the CPU-side `#[repr(C)]` records exactly, side-stepping the
//! `std140` padding rules a uniform block would impose — the same choice the
//! light bind group makes.

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::bind_group_layout_entries::{
    binding_types::{sampler, storage_buffer_read_only_sized, texture_2d_array},
    BindGroupLayoutEntries,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, BufferId, SamplerBindingType, ShaderStages,
        TextureSampleType, TextureViewId,
    },
    renderer::RenderDevice,
};
use core::num::NonZero;

use super::{
    abi::{GpuDirectionalShadow, GpuPointShadow, GpuShadowGlobals},
    resources::{ShadowAtlas, ShadowGpuBuffers},
};

/// Shared binding contract for the resolve pass's shadow inputs.
#[derive(Resource)]
pub(crate) struct ShadowBindGroup {
    /// The GPU layout the resolve pipeline is created against.
    pub layout: BindGroupLayout,
    /// The most recently prepared bind group, once the atlas exists and the
    /// buffers have uploaded.
    pub bind_group: Option<BindGroup>,
    atlas_view_id: Option<TextureViewId>,
    buffer_ids: Option<[BufferId; 3]>,
    buffer_version: u32,
}

impl FromWorld for ShadowBindGroup {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuDirectionalShadow>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuPointShadow>() as u64),
                ),
                storage_buffer_read_only_sized(
                    false,
                    NonZero::new(size_of::<GpuShadowGlobals>() as u64),
                ),
            ),
        );
        Self {
            layout: device.create_bind_group_layout("prism shadows", &entries),
            bind_group: None,
            atlas_view_id: None,
            buffer_ids: None,
            buffer_version: 0,
        }
    }
}

impl ShadowBindGroup {
    /// (Re)builds the shadow bind group once the atlas exists and the shadow
    /// buffers have uploaded, caching on the atlas view identity plus the
    /// buffer ids and version so it only rebuilds when the GPU state changes.
    pub(crate) fn prepare(
        &mut self,
        device: &RenderDevice,
        atlas: &ShadowAtlas,
        buffers: &ShadowGpuBuffers,
    ) {
        let Some((directionals, points, globals)) = buffers.buffers() else {
            return;
        };
        let atlas_view = atlas.view();
        let atlas_view_id = atlas_view.id();
        let ids = [directionals.id(), points.id(), globals.id()];
        let version = buffers.version();
        if self.atlas_view_id == Some(atlas_view_id)
            && self.buffer_ids == Some(ids)
            && self.buffer_version == version
        {
            return;
        }
        self.bind_group = Some(device.create_bind_group(
            "prism shadows",
            &self.layout,
            &BindGroupEntries::sequential((
                atlas_view,
                atlas.sampler(),
                directionals.as_entire_binding(),
                points.as_entire_binding(),
                globals.as_entire_binding(),
            )),
        ));
        self.atlas_view_id = Some(atlas_view_id);
        self.buffer_ids = Some(ids);
        self.buffer_version = version;
    }
}
