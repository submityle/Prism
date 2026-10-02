//! The world-space `ReSTIR` fill compute pipeline, its owned group-0 layout,
//! and the `RenderStartup` initializer that queues it.
//!
//! Mirrors [`super::super::world_space_gi::pipeline`], scaled down to the
//! single fill pass:
//!
//! * `fill_main` (`world_restir_fill.wesl`): one invocation per resident
//!   reservoir-table slot. Its group-0 binds last frame's reservoir table
//!   read-only (`src`, binding 0) and this frame's table read-write (`dst`,
//!   binding 1); empty slots copy through unchanged while occupied slots
//!   recompute their cell key, merge a jittered ring of golden `GRIS`
//!   neighbours and finalise the slot's contribution weight. The grid tunables
//!   and the per-frame seed arrive in the [`GpuWorldRestirFillParams`]
//!   immediate block, so no uniform buffer is bound.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_read_only_sized, storage_buffer_sized},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::GpuWorldRestirFillParams;

/// The world-space `ReSTIR` fill compute pipeline and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct WorldRestirPipeline {
    /// `fill_main` entry: per-slot spatial `GRIS` reuse + finalise.
    fill: CachedComputePipelineId,
    /// group 0 for `fill_main`: the previous reservoir table (read-only, 0) and
    /// the next reservoir table (read-write, 1).
    fill_layout: BindGroupLayout,
}

impl WorldRestirPipeline {
    /// The `fill_main` compute pipeline id.
    pub(crate) fn fill(&self) -> CachedComputePipelineId {
        self.fill
    }

    /// group-0 layout for the `fill_main` dispatch.
    pub(crate) fn fill_layout(&self) -> &BindGroupLayout {
        &self.fill_layout
    }
}

/// `fill_main` group-0 layout: the previous frame's reservoir table bound
/// read-only (0) and this frame's table bound read-write (1). Both are
/// unsized `array<WorldRestirReservoir>` storage buffers whose stride is the
/// frozen [`super::abi::WORLD_RESTIR_RESERVOIR_STRIDE`].
fn fill_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`WorldRestirPipeline`].
pub(crate) fn init_world_restir_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let fill_entries = fill_layout_entries();
    let fill_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space ReSTIR fill", &fill_entries);
    let fill_layout =
        device.create_bind_group_layout("prism world-space ReSTIR fill", &fill_entries);

    let fill_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/world_restir_fill.wesl");

    let fill = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR fill".into()),
        layout: vec![fill_descriptor],
        immediate_size: size_of::<GpuWorldRestirFillParams>() as u32,
        shader: fill_shader,
        entry_point: Some("fill_main".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldRestirPipeline { fill, fill_layout });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_immediate_block_is_the_abi_size() {
        // The pipeline reserves exactly the frozen fill immediate block; a
        // drift here would mismatch `set_immediates` against the shader's
        // `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirFillParams>() as u32, 64);
    }
}
