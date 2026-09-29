//! The two world-space GI compute pipelines, their owned group-0 layouts, and
//! the `RenderStartup` initializer that queues them.
//!
//! Mirrors [`super::super::color_grade::pipeline`], scaled to a two-pass
//! pipeline:
//!
//! * `probe_update_main` (`world_space_gi_probe_update.wesl`): one invocation
//!   per screen probe. Its group-0 binds the reverse-Z scene depth, the packed
//!   `normal_roughness`, the pre-exposed scene colour (all non-filterable
//!   float, `textureLoad`ed) and the probe storage buffer (read-write).
//! * `resolve_main` (`world_space_gi_resolve.wesl`): one invocation per pixel.
//!   Its group-0 binds the scene depth, `normal_roughness`, the probe storage
//!   buffer (read-only) and the write-only `rgba16float` GI export target.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_storage_2d,
        },
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::{GpuWorldSpaceGiProbeParams, GpuWorldSpaceGiResolveParams};
use super::super::resources::SCENE_COLOR_FORMAT;

/// The two world-space GI compute pipelines and their owned group-0 layouts.
#[derive(Resource)]
pub(crate) struct WorldSpaceGiPipeline {
    /// `probe_update_main` entry: per-probe screen-probe SH capture.
    probe_update: CachedComputePipelineId,
    /// `resolve_main` entry: per-pixel probe interpolation into the GI buffer.
    resolve: CachedComputePipelineId,
    /// group 0 for `probe_update_main`: depth + normal + colour reads and the
    /// probe storage buffer (read-write).
    probe_update_layout: BindGroupLayout,
    /// group 0 for `resolve_main`: depth + normal reads, the probe storage
    /// buffer (read-only) and the GI export storage write.
    resolve_layout: BindGroupLayout,
}

impl WorldSpaceGiPipeline {
    /// The `probe_update_main` compute pipeline id.
    pub(crate) fn probe_update(&self) -> CachedComputePipelineId {
        self.probe_update
    }

    /// The `resolve_main` compute pipeline id.
    pub(crate) fn resolve(&self) -> CachedComputePipelineId {
        self.resolve
    }

    /// group-0 layout for the `probe_update_main` dispatch.
    pub(crate) fn probe_update_layout(&self) -> &BindGroupLayout {
        &self.probe_update_layout
    }

    /// group-0 layout for the `resolve_main` dispatch.
    pub(crate) fn resolve_layout(&self) -> &BindGroupLayout {
        &self.resolve_layout
    }
}

/// `probe_update_main` group-0 layout: reverse-Z scene depth (0), packed
/// `normal_roughness` (1), pre-exposed scene colour (2) — all non-filterable
/// float — and the read-write probe storage buffer (3).
fn probe_update_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `resolve_main` group-0 layout: reverse-Z scene depth (0), packed
/// `normal_roughness` (1), the read-only probe storage buffer (2) and the
/// write-only `rgba16float` GI export target (3).
fn resolve_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer_read_only_sized(false, None),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`WorldSpaceGiPipeline`].
pub(crate) fn init_world_space_gi_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let probe_update_entries = probe_update_layout_entries();
    let probe_update_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space GI probe update", &probe_update_entries);
    let probe_update_layout =
        device.create_bind_group_layout("prism world-space GI probe update", &probe_update_entries);

    let resolve_entries = resolve_layout_entries();
    let resolve_descriptor =
        BindGroupLayoutDescriptor::new("prism world-space GI resolve", &resolve_entries);
    let resolve_layout =
        device.create_bind_group_layout("prism world-space GI resolve", &resolve_entries);

    let probe_update_shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/world_space_gi_probe_update.wesl"
    );
    let resolve_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/world_space_gi_resolve.wesl");

    let probe_update = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space GI probe update".into()),
        layout: vec![probe_update_descriptor],
        immediate_size: size_of::<GpuWorldSpaceGiProbeParams>() as u32,
        shader: probe_update_shader,
        entry_point: Some("probe_update_main".into()),
        ..Default::default()
    });

    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism world-space GI resolve".into()),
        layout: vec![resolve_descriptor],
        immediate_size: size_of::<GpuWorldSpaceGiResolveParams>() as u32,
        shader: resolve_shader,
        entry_point: Some("resolve_main".into()),
        ..Default::default()
    });

    commands.insert_resource(WorldSpaceGiPipeline {
        probe_update,
        resolve,
        probe_update_layout,
        resolve_layout,
    });
}
