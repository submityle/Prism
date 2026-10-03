//! The four surface-cache compute pipelines, their owned group-0 layouts, and
//! the `RenderStartup` initializer that queues them.
//!
//! Mirrors [`super::super::world_space_gi::pipeline`], scaled to the surfel
//! radiance cache's four same-frame passes:
//!
//! * `surface_cache_alloc_main` (`surface_cache_alloc.wesl`): one invocation
//!   per surfel. Its group-0 binds the reverse-Z scene depth, the packed
//!   `normal_roughness` and the pre-exposed scene colour (all non-filterable
//!   float, `textureLoad`ed) and the freshly sampled scratch surfel buffer
//!   (read-write).
//! * `surface_cache_update_main` (`surface_cache_update.wesl`): one invocation
//!   per surfel. Its group-0 binds the fresh scratch buffer (read-only) and the
//!   persistent surfel buffer (read-write) holding the `EMA` state.
//! * `surface_cache_spatial_filter_main` (`surface_cache_spatial_filter.wesl`):
//!   one invocation per surfel. Its group-0 binds the persistent surfel buffer
//!   (read-only) and the filtered scratch buffer (read-write).
//! * `surface_cache_coverage_main` (`surface_cache_coverage.wesl`): one
//!   invocation per pixel. Its group-0 binds the scene depth, the packed
//!   `normal_roughness`, the filtered surfel buffer (read-only) and the
//!   write-only `rgba16float` `GI` export target.

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

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::{
    GpuSurfaceCacheAllocParams, GpuSurfaceCacheCoverageParams, GpuSurfaceCacheFilterParams,
    GpuSurfaceCacheUpdateParams,
};

/// The four surface-cache compute pipelines and their owned group-0 layouts.
#[derive(Resource)]
pub(crate) struct SurfaceCachePipeline {
    /// `surface_cache_alloc_main` entry: per-surfel fresh sample capture.
    alloc: CachedComputePipelineId,
    /// `surface_cache_update_main` entry: per-surfel confidence-weighted `EMA`.
    update: CachedComputePipelineId,
    /// `surface_cache_spatial_filter_main` entry: per-surfel bilateral filter.
    filter: CachedComputePipelineId,
    /// `surface_cache_coverage_main` entry: per-pixel coverage gather.
    coverage: CachedComputePipelineId,
    /// group 0 for `surface_cache_alloc_main`: depth + normal + colour reads
    /// and the scratch surfel buffer (read-write).
    alloc_layout: BindGroupLayout,
    /// group 0 for `surface_cache_update_main`: the fresh scratch buffer
    /// (read-only) and the persistent surfel buffer (read-write).
    update_layout: BindGroupLayout,
    /// group 0 for `surface_cache_spatial_filter_main`: the persistent surfel
    /// buffer (read-only) and the filtered scratch buffer (read-write).
    filter_layout: BindGroupLayout,
    /// group 0 for `surface_cache_coverage_main`: depth + normal reads, the
    /// filtered surfel buffer (read-only) and the `GI` export storage write.
    coverage_layout: BindGroupLayout,
}

impl SurfaceCachePipeline {
    /// The `surface_cache_alloc_main` compute pipeline id.
    pub(crate) fn alloc(&self) -> CachedComputePipelineId {
        self.alloc
    }

    /// The `surface_cache_update_main` compute pipeline id.
    pub(crate) fn update(&self) -> CachedComputePipelineId {
        self.update
    }

    /// The `surface_cache_spatial_filter_main` compute pipeline id.
    pub(crate) fn filter(&self) -> CachedComputePipelineId {
        self.filter
    }

    /// The `surface_cache_coverage_main` compute pipeline id.
    pub(crate) fn coverage(&self) -> CachedComputePipelineId {
        self.coverage
    }

    /// group-0 layout for the `surface_cache_alloc_main` dispatch.
    pub(crate) fn alloc_layout(&self) -> &BindGroupLayout {
        &self.alloc_layout
    }

    /// group-0 layout for the `surface_cache_update_main` dispatch.
    pub(crate) fn update_layout(&self) -> &BindGroupLayout {
        &self.update_layout
    }

    /// group-0 layout for the `surface_cache_spatial_filter_main` dispatch.
    pub(crate) fn filter_layout(&self) -> &BindGroupLayout {
        &self.filter_layout
    }

    /// group-0 layout for the `surface_cache_coverage_main` dispatch.
    pub(crate) fn coverage_layout(&self) -> &BindGroupLayout {
        &self.coverage_layout
    }
}

/// `surface_cache_alloc_main` group-0 layout: reverse-Z scene depth (0), packed
/// `normal_roughness` (1), pre-exposed scene colour (2) — all non-filterable
/// float — and the read-write scratch surfel buffer (3).
fn alloc_layout_entries() -> BindGroupLayoutEntries<4> {
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

/// `surface_cache_update_main` group-0 layout: the fresh scratch surfel buffer
/// (0, read-only) and the persistent surfel buffer (1, read-write).
fn update_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `surface_cache_spatial_filter_main` group-0 layout: the persistent surfel
/// buffer (0, read-only) and the filtered scratch buffer (1, read-write).
fn filter_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_sized(false, None),
        ),
    )
}

/// `surface_cache_coverage_main` group-0 layout: reverse-Z scene depth (0),
/// packed `normal_roughness` (1), the read-only filtered surfel buffer (2) and
/// the write-only `rgba16float` `GI` export target (3).
fn coverage_layout_entries() -> BindGroupLayoutEntries<4> {
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

/// `RenderStartup` initializer for [`SurfaceCachePipeline`].
pub(crate) fn init_surface_cache_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let alloc_entries = alloc_layout_entries();
    let alloc_descriptor =
        BindGroupLayoutDescriptor::new("prism surface cache alloc", &alloc_entries);
    let alloc_layout = device.create_bind_group_layout("prism surface cache alloc", &alloc_entries);

    let update_entries = update_layout_entries();
    let update_descriptor =
        BindGroupLayoutDescriptor::new("prism surface cache update", &update_entries);
    let update_layout =
        device.create_bind_group_layout("prism surface cache update", &update_entries);

    let filter_entries = filter_layout_entries();
    let filter_descriptor =
        BindGroupLayoutDescriptor::new("prism surface cache spatial filter", &filter_entries);
    let filter_layout =
        device.create_bind_group_layout("prism surface cache spatial filter", &filter_entries);

    let coverage_entries = coverage_layout_entries();
    let coverage_descriptor =
        BindGroupLayoutDescriptor::new("prism surface cache coverage", &coverage_entries);
    let coverage_layout =
        device.create_bind_group_layout("prism surface cache coverage", &coverage_entries);

    let alloc_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/surface_cache_alloc.wesl");
    let update_shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/surface_cache_update.wesl"
    );
    let filter_shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/surface_cache_spatial_filter.wesl"
    );
    let coverage_shader: Handle<Shader> = load_embedded_asset!(
        asset_server.as_ref(),
        "../shaders/surface_cache_coverage.wesl"
    );

    let alloc = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism surface cache alloc".into()),
        layout: vec![alloc_descriptor],
        immediate_size: size_of::<GpuSurfaceCacheAllocParams>() as u32,
        shader: alloc_shader,
        entry_point: Some("surface_cache_alloc_main".into()),
        ..Default::default()
    });

    let update = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism surface cache update".into()),
        layout: vec![update_descriptor],
        immediate_size: size_of::<GpuSurfaceCacheUpdateParams>() as u32,
        shader: update_shader,
        entry_point: Some("surface_cache_update_main".into()),
        ..Default::default()
    });

    let filter = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism surface cache spatial filter".into()),
        layout: vec![filter_descriptor],
        immediate_size: size_of::<GpuSurfaceCacheFilterParams>() as u32,
        shader: filter_shader,
        entry_point: Some("surface_cache_spatial_filter_main".into()),
        ..Default::default()
    });

    let coverage = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism surface cache coverage".into()),
        layout: vec![coverage_descriptor],
        immediate_size: size_of::<GpuSurfaceCacheCoverageParams>() as u32,
        shader: coverage_shader,
        entry_point: Some("surface_cache_coverage_main".into()),
        ..Default::default()
    });

    commands.insert_resource(SurfaceCachePipeline {
        alloc,
        update,
        filter,
        coverage,
        alloc_layout,
        update_layout,
        filter_layout,
        coverage_layout,
    });
}
