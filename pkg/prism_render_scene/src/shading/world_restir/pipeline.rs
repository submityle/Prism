//! The world-space `ReSTIR` seed and fill compute pipelines, their owned
//! group-0 layouts, and the `RenderStartup` initializer that queues them.
//!
//! Mirrors [`super::super::world_space_gi::pipeline`], scaled down to the two
//! world-space `ReSTIR` passes:
//!
//! * `seed_main` (`world_restir_seed.wesl`): one invocation per resident
//!   reservoir-table slot. Its group-0 binds last frame's table read-only
//!   (`src`, binding 0), this frame's table read-write (`dst`, binding 1) and
//!   the per-frame candidate light list read-only (`lights`, binding 2); each
//!   occupied slot streams a short `RIS` over the candidate lights towards its
//!   visible point and writes the surviving reservoir, while empty slots pass
//!   through. The grid tunables, the active light count and the per-frame seed
//!   arrive in the [`GpuWorldRestirSeedParams`] immediate block.
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

use super::abi::{GpuWorldRestirFillParams, GpuWorldRestirInjectParams, GpuWorldRestirSeedParams};

/// The world-space `ReSTIR` seed and fill compute pipelines and their owned
/// group-0 layouts.
#[derive(Resource)]
pub(crate) struct WorldRestirPipeline {
    /// `seed_main` entry: per-slot streaming `RIS` over the candidate lights.
    seed: CachedComputePipelineId,
    /// group 0 for `seed_main`: the previous reservoir table (read-only, 0),
    /// the next reservoir table (read-write, 1) and the candidate light list
    /// (read-only, 2).
    seed_layout: BindGroupLayout,
    /// `fill_main` entry: per-slot spatial `GRIS` reuse + finalise.
    fill: CachedComputePipelineId,
    /// group 0 for `fill_main`: the previous reservoir table (read-only, 0) and
    /// the next reservoir table (read-write, 1).
    fill_layout: BindGroupLayout,
    /// `inject_main` entry: one invocation per visible point, open-address
    /// claims the point's `SHARC` cell and pre-seeds the slot geometry.
    inject: CachedComputePipelineId,
    /// group 0 for `inject_main`: the per-frame visible-point list (read-only,
    /// 0), this frame's reservoir table (read-write, 1) and the parallel
    /// per-slot claim-guard array (atomic read-write, 2).
    inject_layout: BindGroupLayout,
}

impl WorldRestirPipeline {
    /// The `seed_main` compute pipeline id. Recorded by the seed dispatch in a
    /// follow-up slice.
    #[expect(
        dead_code,
        reason = "the seed dispatch records this pipeline in a follow-up slice;                   no render-graph node reads it yet"
    )]
    pub(crate) fn seed(&self) -> CachedComputePipelineId {
        self.seed
    }

    /// group-0 layout for the `seed_main` dispatch. The seed bind group builds
    /// against it in a follow-up slice.
    #[expect(
        dead_code,
        reason = "the seed bind group builds against this layout in a follow-up                   slice; no host path reads it yet"
    )]
    pub(crate) fn seed_layout(&self) -> &BindGroupLayout {
        &self.seed_layout
    }

    /// The `fill_main` compute pipeline id.
    pub(crate) fn fill(&self) -> CachedComputePipelineId {
        self.fill
    }

    /// group-0 layout for the `fill_main` dispatch.
    pub(crate) fn fill_layout(&self) -> &BindGroupLayout {
        &self.fill_layout
    }

    /// The `inject_main` compute pipeline id. Recorded by the inject dispatch
    /// in a follow-up slice.
    #[expect(
        dead_code,
        reason = "the inject dispatch records this pipeline in a follow-up slice; no render-graph node reads it yet"
    )]
    pub(crate) fn inject(&self) -> CachedComputePipelineId {
        self.inject
    }

    /// group-0 layout for the `inject_main` dispatch. The inject bind group
    /// builds against it in a follow-up slice.
    #[expect(
        dead_code,
        reason = "the inject bind group builds against this layout in a follow-up slice; no host path reads it yet"
    )]
    pub(crate) fn inject_layout(&self) -> &BindGroupLayout {
        &self.inject_layout
    }
}

/// `seed_main` group-0 layout: the previous frame's reservoir table bound
/// read-only (0), this frame's table bound read-write (1) and the per-frame
/// candidate light list bound read-only (2). The reservoir tables are unsized
/// `array<WorldRestirReservoir>` buffers at the frozen
/// [`super::abi::WORLD_RESTIR_RESERVOIR_STRIDE`]; the light list is an unsized
/// `array<WorldRestirLight>` at the frozen
/// [`super::abi::WORLD_RESTIR_LIGHT_STRIDE`].
fn seed_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
        ),
    )
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

/// `inject_main` group-0 layout: the per-frame visible-point list bound
/// read-only (0), this frame's reservoir table bound read-write (1) and the
/// parallel per-slot claim-guard array bound atomic read-write (2). The point
/// list is an unsized `array<InjectPoint>` at the frozen
/// [`super::abi::WORLD_RESTIR_INJECT_POINT_STRIDE`]; the reservoir table is an
/// unsized `array<WorldRestirReservoir>` at the frozen
/// [`super::abi::WORLD_RESTIR_RESERVOIR_STRIDE`]; the guard array is an unsized
/// `array<atomic<u32>>` (an atomic storage buffer uses the same plain
/// `storage_buffer_sized` layout entry as a non-atomic one in `wgpu`).
fn inject_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
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
    let seed_entries = seed_layout_entries();
    let seed_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space ReSTIR seed", &seed_entries);
    let seed_layout =
        device.create_bind_group_layout("prism world-space ReSTIR seed", &seed_entries);

    let seed_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/world_restir_seed.wesl");

    let seed = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR seed".into()),
        layout: vec![seed_descriptor],
        immediate_size: size_of::<GpuWorldRestirSeedParams>() as u32,
        shader: seed_shader,
        entry_point: Some("seed_main".into()),
        ..Default::default()
    });

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

    let inject_entries = inject_layout_entries();
    let inject_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space ReSTIR inject", &inject_entries);
    let inject_layout =
        device.create_bind_group_layout("prism world-space ReSTIR inject", &inject_entries);

    let inject_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/world_restir_inject.wesl");

    let inject = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space ReSTIR inject".into()),
        layout: vec![inject_descriptor],
        immediate_size: size_of::<GpuWorldRestirInjectParams>() as u32,
        shader: inject_shader,
        entry_point: Some("inject_main".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldRestirPipeline {
        seed,
        seed_layout,
        fill,
        fill_layout,
        inject,
        inject_layout,
    });
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

    #[test]
    fn seed_immediate_block_is_the_abi_size() {
        // The seed pipeline reserves exactly the frozen seed immediate block;
        // a drift here would mismatch `set_immediates` against the seed
        // shader's `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirSeedParams>() as u32, 32);
    }

    #[test]
    fn inject_immediate_block_is_the_abi_size() {
        // The inject pipeline reserves exactly the frozen inject immediate
        // block; a drift here would mismatch `set_immediates` against the
        // inject shader's `var<immediate>` block.
        assert_eq!(size_of::<GpuWorldRestirInjectParams>() as u32, 48);
    }
}
