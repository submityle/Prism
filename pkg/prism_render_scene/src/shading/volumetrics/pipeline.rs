//! Froxel volumetric-fog compute pipelines and their owned group-0 layouts.
//!
//! Two chained compute entries realise the fog over the view-frustum-fitted 3D
//! grid, mirroring the two-entry `exposure.wesl` precedent (each entry is a
//! distinct pipeline with its own layout, reusing `@group(0)` binding slots with
//! different types):
//!
//! * **scatter** (`volumetrics_scatter`) builds every froxel's medium and the
//!   light it in-scatters toward the eye, writing the per-froxel source radiance and
//!   slice thickness, plus the per-froxel extinction into two write-only
//!   `rgba16float` storage volumes. Its immediate block is the 96-byte
//!   [`GpuVolumetricsScatterParams`].
//! * **integrate** (`volumetrics_integrate`) reads those two scatter volumes back
//!   and marches each froxel column front-to-back, writing the running
//!   camera-to-slice accumulated in-scattering and transmittance into two more
//!   write-only `rgba16float` storage volumes. Its immediate block is the
//!   12-byte [`GpuVolumetricsIntegrateParams`].
//!
//! Both layouts are `D3` (3D) throughout, matching the froxel storage textures
//! [`super::resources`] allocates.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_3d, texture_storage_3d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::{GpuVolumetricsIntegrateParams, GpuVolumetricsScatterParams};

/// The `rgba16float` storage format shared by every froxel volume (source +
/// thickness, extinction, integrated in-scattering, integrated transmittance).
pub(crate) const VOLUMETRICS_FROXEL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The two froxel-fog compute pipelines and their owned group-0 layouts.
#[derive(Resource)]
pub(crate) struct VolumetricsPipeline {
    /// `volumetrics_scatter` compute entry, specialized against the scatter
    /// layout and the 96-byte [`GpuVolumetricsScatterParams`] immediate block.
    scatter: CachedComputePipelineId,
    /// `volumetrics_integrate` compute entry, specialized against the integrate
    /// layout and the 12-byte [`GpuVolumetricsIntegrateParams`] immediate block.
    integrate: CachedComputePipelineId,
    /// group 0 (scatter): two write-only `rgba16float` froxel volumes.
    scatter_layout: BindGroupLayout,
    /// group 0 (integrate): the two scatter volumes read + two write-only
    /// integrated volumes.
    integrate_layout: BindGroupLayout,
}

impl VolumetricsPipeline {
    /// The `volumetrics_scatter` compute pipeline id.
    pub(crate) fn scatter(&self) -> CachedComputePipelineId {
        self.scatter
    }

    /// The `volumetrics_integrate` compute pipeline id.
    pub(crate) fn integrate(&self) -> CachedComputePipelineId {
        self.integrate
    }

    /// The scatter pass's group-0 bind-group layout.
    pub(crate) fn scatter_layout(&self) -> &BindGroupLayout {
        &self.scatter_layout
    }

    /// The integrate pass's group-0 bind-group layout.
    pub(crate) fn integrate_layout(&self) -> &BindGroupLayout {
        &self.integrate_layout
    }
}

/// group-0 layout for `volumetrics_scatter`: two write-only `rgba16float` froxel
/// volumes (source + thickness at binding 0, extinction at binding 1).
fn scatter_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_storage_3d(VOLUMETRICS_FROXEL_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_3d(VOLUMETRICS_FROXEL_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// group-0 layout for `volumetrics_integrate`: the two scatter volumes read as
/// non-filterable `texture_3d` (`textureLoad`ed per froxel) at bindings 0/1, and
/// two write-only `rgba16float` integrated volumes at bindings 2/3.
fn integrate_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_3d(TextureSampleType::Float { filterable: false }),
            texture_3d(TextureSampleType::Float { filterable: false }),
            texture_storage_3d(VOLUMETRICS_FROXEL_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_3d(VOLUMETRICS_FROXEL_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`VolumetricsPipeline`]. Queues both compute
/// pipelines against the shared `volumetrics.wesl` shader and their owned
/// layouts.
pub(crate) fn init_volumetrics_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let scatter_entries = scatter_layout_entries();
    let scatter_descriptor =
        BindGroupLayoutDescriptor::new("prism volumetrics scatter", &scatter_entries);
    let scatter_layout =
        device.create_bind_group_layout("prism volumetrics scatter", &scatter_entries);

    let integrate_entries = integrate_layout_entries();
    let integrate_descriptor =
        BindGroupLayoutDescriptor::new("prism volumetrics integrate", &integrate_entries);
    let integrate_layout =
        device.create_bind_group_layout("prism volumetrics integrate", &integrate_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/volumetrics.wesl");

    let scatter = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetrics scatter".into()),
        layout: vec![scatter_descriptor],
        immediate_size: size_of::<GpuVolumetricsScatterParams>() as u32,
        shader: shader.clone(),
        entry_point: Some("volumetrics_scatter".into()),
        ..Default::default()
    });

    let integrate = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetrics integrate".into()),
        layout: vec![integrate_descriptor],
        immediate_size: size_of::<GpuVolumetricsIntegrateParams>() as u32,
        shader,
        entry_point: Some("volumetrics_integrate".into()),
        ..Default::default()
    });

    commands.insert_resource(VolumetricsPipeline {
        scatter,
        integrate,
        scatter_layout,
        integrate_layout,
    });
}
