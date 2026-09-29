//! Froxel volumetric-fog compute pipelines and their owned group-0 layouts.
//!
//! Three compute entries realise the fog over the view-frustum-fitted 3D grid,
//! mirroring the two-entry `exposure.wesl` precedent (each entry is a distinct
//! pipeline with its own layout, reusing `@group(0)` binding slots with
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
//! * **apply** (`volumetrics_apply`) resolves the fog per screen pixel: it
//!   reconstructs each surface's froxel from the scene depth, trilinearly
//!   samples the two integrated volumes and folds them over the lit
//!   `scene_color` (`background * transmittance + in_scattering`) into a
//!   dedicated full-res target the dispatch blits back over `scene_color`. Its
//!   immediate block is the 96-byte [`GpuVolumetricsApplyParams`].
//!
//! The scatter/integrate layouts are `D3` (3D) throughout, matching the froxel
//! storage textures [`super::resources`] allocates; the apply layout mixes the
//! two integrated `texture_3d` volumes (filtered through a linear sampler) with
//! the 2D scene depth/colour and the 2D `rgba16float` composite output.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_2d, texture_3d, texture_storage_2d, texture_storage_3d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        AddressMode, BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor,
        FilterMode, MipmapFilterMode, PipelineCache, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, StorageTextureAccess, TextureFormat, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::{
    GpuVolumetricsApplyParams, GpuVolumetricsIntegrateParams, GpuVolumetricsScatterParams,
};

/// The `rgba16float` storage format shared by every froxel volume (source +
/// thickness, extinction, integrated in-scattering, integrated transmittance).
pub(crate) const VOLUMETRICS_FROXEL_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// The three froxel-fog compute pipelines, their owned group-0 layouts and the
/// shared linear-clamp sampler the apply pass fetches the froxel volumes with.
#[derive(Resource)]
pub(crate) struct VolumetricsPipeline {
    /// `volumetrics_scatter` compute entry, specialized against the scatter
    /// layout and the 96-byte [`GpuVolumetricsScatterParams`] immediate block.
    scatter: CachedComputePipelineId,
    /// `volumetrics_integrate` compute entry, specialized against the integrate
    /// layout and the 12-byte [`GpuVolumetricsIntegrateParams`] immediate block.
    integrate: CachedComputePipelineId,
    /// `volumetrics_apply` compute entry, specialized against the apply layout
    /// and the 96-byte [`GpuVolumetricsApplyParams`] immediate block.
    apply: CachedComputePipelineId,
    /// group 0 (scatter): two write-only `rgba16float` froxel volumes.
    scatter_layout: BindGroupLayout,
    /// group 0 (integrate): the two scatter volumes read + two write-only
    /// integrated volumes.
    integrate_layout: BindGroupLayout,
    /// group 0 (apply): scene depth + the two integrated volumes + linear
    /// sampler + scene colour + the write-only composite output.
    apply_layout: BindGroupLayout,
    /// Linear-clamp sampler for the apply pass's trilinear froxel fetch.
    linear_sampler: Sampler,
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

    /// The `volumetrics_apply` compute pipeline id.
    pub(crate) fn apply(&self) -> CachedComputePipelineId {
        self.apply
    }

    /// The scatter pass's group-0 bind-group layout.
    pub(crate) fn scatter_layout(&self) -> &BindGroupLayout {
        &self.scatter_layout
    }

    /// The integrate pass's group-0 bind-group layout.
    pub(crate) fn integrate_layout(&self) -> &BindGroupLayout {
        &self.integrate_layout
    }

    /// The apply pass's group-0 bind-group layout.
    pub(crate) fn apply_layout(&self) -> &BindGroupLayout {
        &self.apply_layout
    }

    /// The shared linear-clamp sampler the apply pass fetches the froxel
    /// volumes with.
    pub(crate) fn linear_sampler(&self) -> &Sampler {
        &self.linear_sampler
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

/// group-0 layout for `volumetrics_apply` (sequential bindings 0-5):
///   0 = scene depth (reverse-Z `R32Float`, non-filterable, `textureLoad`ed),
///   1 = integrated in-scattering (`texture_3d`, filtered),
///   2 = integrated transmittance (`texture_3d`, filtered),
///   3 = linear-clamp sampler (trilinear froxel fetch),
///   4 = scene colour (lit HDR, non-filterable, `textureLoad`ed),
///   5 = fog-applied output (write-only `rgba16float` full-res composite).
fn apply_layout_entries() -> BindGroupLayoutEntries<6> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_3d(TextureSampleType::Float { filterable: true }),
            texture_3d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`VolumetricsPipeline`]. Queues all three
/// compute pipelines against the shared `volumetrics.wesl` shader, their owned
/// layouts, and the linear-clamp sampler the apply pass fetches with.
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

    let apply_entries = apply_layout_entries();
    let apply_descriptor =
        BindGroupLayoutDescriptor::new("prism volumetrics apply", &apply_entries);
    let apply_layout = device.create_bind_group_layout("prism volumetrics apply", &apply_entries);

    // Linear-clamp sampler for the apply pass's trilinear froxel fetch: the
    // integrated volumes are `rgba16float` (filterable) and clamped at the grid
    // edges so a pixel past the last froxel column reuses the border slice.
    let linear_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism volumetrics linear-clamp sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..Default::default()
    });

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
        shader: shader.clone(),
        entry_point: Some("volumetrics_integrate".into()),
        ..Default::default()
    });

    let apply = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism volumetrics apply".into()),
        layout: vec![apply_descriptor],
        immediate_size: size_of::<GpuVolumetricsApplyParams>() as u32,
        shader,
        entry_point: Some("volumetrics_apply".into()),
        ..Default::default()
    });

    commands.insert_resource(VolumetricsPipeline {
        scatter,
        integrate,
        apply,
        scatter_layout,
        integrate_layout,
        apply_layout,
        linear_sampler,
    });
}
