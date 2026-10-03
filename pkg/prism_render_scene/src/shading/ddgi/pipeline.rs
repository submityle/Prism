//! The DDGI irradiance-volume sample compute pipeline, its owned group-0
//! layout, and the `RenderStartup` initializer that queues it.
//!
//! Mirrors [`super::super::world_space_gi::pipeline`], scaled (for now) to the
//! single per-pixel `sample_main` pass; the probe-update / relocation pipelines
//! land in later DDGI blocks and extend this same resource.
//!
//! `sample_main` (`ddgi_sample.wesl`) runs one invocation per framebuffer
//! pixel. Its group-0 binds the reverse-Z scene depth (0) and packed
//! `normal_roughness` (1) — both non-filterable float, `textureLoad`ed — the
//! lattice + field-metadata uniform (2), the per-probe [`ProbeMeta`] storage
//! buffer read-only (3), the octahedral irradiance (4) + depth / visibility (5)
//! atlases (non-filterable float), and the write-only `rgba16float` GI export
//! target (6). The clip->world reconstruction transform travels as the
//! immediate ([`GpuDdgiSampleParams`]).

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{
            storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_storage_2d,
            uniform_buffer_sized,
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

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::{GpuDdgiSampleParams, GpuDdgiUpdateParams};
use super::resources::DDGI_ATLAS_FORMAT;

/// The DDGI sample compute pipeline and its owned group-0 layout.
#[derive(Resource)]
pub(crate) struct DdgiPipeline {
    /// `sample_main` entry: per-pixel eight-probe irradiance reconstruction.
    sample: CachedComputePipelineId,
    /// group 0 for `sample_main`: depth + normal reads, the lattice uniform,
    /// the probe-meta storage buffer (read-only), the two octahedral atlases
    /// and the GI export storage write.
    sample_layout: BindGroupLayout,
    /// `probe_update_main` entry: one workgroup per probe, traces 64 rays,
    /// temporally blends irradiance + depth moments, relocates / classifies the
    /// probe and writes the two octahedral atlases.
    probe_update: CachedComputePipelineId,
    /// group 0 for `probe_update_main`: depth / normal / scene-colour reads, the
    /// lattice uniform, the probe-meta + irradiance / depth history storage
    /// buffers (read_write) and the two octahedral atlas storage writes.
    probe_update_layout: BindGroupLayout,
}

impl DdgiPipeline {
    /// The `sample_main` compute pipeline id.
    pub(crate) fn sample(&self) -> CachedComputePipelineId {
        self.sample
    }

    /// group-0 layout for the `sample_main` dispatch.
    pub(crate) fn sample_layout(&self) -> &BindGroupLayout {
        &self.sample_layout
    }

    /// The `probe_update_main` compute pipeline id.
    pub(crate) fn probe_update(&self) -> CachedComputePipelineId {
        self.probe_update
    }

    /// group-0 layout for the `probe_update_main` dispatch.
    pub(crate) fn probe_update_layout(&self) -> &BindGroupLayout {
        &self.probe_update_layout
    }
}

/// `sample_main` group-0 layout: reverse-Z scene depth (0), packed
/// `normal_roughness` (1) — both non-filterable float — the lattice + field
/// metadata uniform (2), the read-only [`ProbeMeta`] storage buffer (3), the
/// octahedral irradiance (4) + depth / visibility (5) atlases (non-filterable
/// float) and the write-only `rgba16float` GI export target (6).
fn sample_layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            uniform_buffer_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `probe_update_main` group-0 layout: reverse-Z scene depth (0), packed
/// `normal_roughness` (1) and the pre-exposed scene colour (2) — all
/// non-filterable float, `textureLoad`ed — the lattice + field metadata uniform
/// (3), the [`ProbeMeta`] (4) + irradiance history (5) + depth history (6)
/// `read_write` storage buffers, and the write-only `rgba16float` octahedral
/// irradiance (7) + depth / visibility (8) atlases.
fn probe_update_layout_entries() -> BindGroupLayoutEntries<9> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            uniform_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            storage_buffer_sized(false, None),
            texture_storage_2d(DDGI_ATLAS_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(DDGI_ATLAS_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`DdgiPipeline`].
pub(crate) fn init_ddgi_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let sample_entries = sample_layout_entries();
    let sample_descriptor = BindGroupLayoutDescriptor::new("prism DDGI sample", &sample_entries);
    let sample_layout = device.create_bind_group_layout("prism DDGI sample", &sample_entries);

    let sample_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ddgi_sample.wesl");

    let sample = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism DDGI sample".into()),
        layout: vec![sample_descriptor],
        immediate_size: size_of::<GpuDdgiSampleParams>() as u32,
        shader: sample_shader,
        entry_point: Some("sample_main".into()),
        ..Default::default()
    });

    let probe_update_entries = probe_update_layout_entries();
    let probe_update_descriptor =
        BindGroupLayoutDescriptor::new("prism DDGI probe update", &probe_update_entries);
    let probe_update_layout =
        device.create_bind_group_layout("prism DDGI probe update", &probe_update_entries);

    let probe_update_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ddgi_probe_update.wesl");

    let probe_update = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism DDGI probe update".into()),
        layout: vec![probe_update_descriptor],
        immediate_size: size_of::<GpuDdgiUpdateParams>() as u32,
        shader: probe_update_shader,
        entry_point: Some("probe_update_main".into()),
        ..Default::default()
    });

    commands.insert_resource(DdgiPipeline {
        sample,
        sample_layout,
        probe_update,
        probe_update_layout,
    });
}
