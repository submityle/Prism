//! Temporal-upscale compute pipelines, their owned group-0 layouts, and the
//! filtering sampler the reconstruction reads the reprojected history through.
//!
//! Mirrors [`super::super::taa::pipeline`]: two `@workgroup_size(8,8,1)` compute
//! entries specialized against their group-0 layouts and immediate blocks.
//!
//! * `reconstruct` (`upscale_reconstruct.wesl`) resolves the low-resolution
//!   current frame onto the display grid and blends it with the reprojected
//!   history. Its ten-binding layout matches the shader: the low-res
//!   `render_color`, the *filterable* display-res `history_color` + its
//!   sampler, the `motion_vectors`, the render-res `depth`, the display-res
//!   `history_depth` and `history_meta`, and the three write-only outputs
//!   `color_out` (`rgba16float`), `meta_out` (`rgba16float`) and `depth_out`
//!   (`r32float`, this frame's depth persisted for next frame's reprojection).
//! * `rcas` (`upscale_rcas.wesl`) sharpens the resolved image; its two-binding
//!   layout is the reconstructed `input_color` and the `rgba16float` output.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{sampler, texture_2d, texture_storage_2d},
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

use super::abi::{GpuUpscaleRcasParams, GpuUpscaleReconstructParams};
use super::super::resources::SCENE_COLOR_FORMAT;

/// The display-resolution history depth format, matching the reconstruction
/// shader's `depth_out` / `history_depth` (`r32float`, the SSR device-depth
/// encoding the reconstruction consumes).
pub(crate) const UPSCALE_DEPTH_FORMAT: TextureFormat = TextureFormat::R32Float;

/// The two temporal-upscale compute pipelines, their owned group-0 layouts, and
/// the bilinear history sampler the reconstruction reads the reprojected
/// history through.
#[derive(Resource)]
pub(crate) struct UpscalePipeline {
    /// `upscale_reconstruct` compute entry, specialized against the ten-binding
    /// reconstruction layout and the 56-byte [`GpuUpscaleReconstructParams`].
    reconstruct: CachedComputePipelineId,
    /// `upscale_rcas` compute entry, specialized against the two-binding RCAS
    /// layout and the 16-byte [`GpuUpscaleRcasParams`].
    rcas: CachedComputePipelineId,
    /// group-0 layout for the reconstruction pass.
    reconstruct_layout: BindGroupLayout,
    /// group-0 layout for the RCAS pass.
    rcas_layout: BindGroupLayout,
    /// Bilinear clamp sampler bound at reconstruction binding 2 so the
    /// reprojected history UV interpolates.
    sampler: Sampler,
}

impl UpscalePipeline {
    /// The `upscale_reconstruct` compute pipeline id.
    pub(crate) fn reconstruct(&self) -> CachedComputePipelineId {
        self.reconstruct
    }

    /// The `upscale_rcas` compute pipeline id.
    pub(crate) fn rcas(&self) -> CachedComputePipelineId {
        self.rcas
    }

    /// The reconstruction group-0 bind-group layout.
    pub(crate) fn reconstruct_layout(&self) -> &BindGroupLayout {
        &self.reconstruct_layout
    }

    /// The RCAS group-0 bind-group layout.
    pub(crate) fn rcas_layout(&self) -> &BindGroupLayout {
        &self.rcas_layout
    }

    /// The bilinear clamp history sampler bound at reconstruction binding 2.
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// group-0 layout mirroring `upscale_reconstruct.wesl` binding-for-binding: the
/// non-filterable low-res `render_color`, the *filterable* display-res
/// `history_color` + its filtering sampler, the non-filterable `motion_vectors`
/// / render-res `depth` / display-res `history_depth` / `history_meta` reads,
/// and the three write-only outputs `color_out` / `meta_out` (`rgba16float`)
/// and `depth_out` (`r32float`).
fn reconstruct_layout_entries() -> BindGroupLayoutEntries<10> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(UPSCALE_DEPTH_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// group-0 layout mirroring `upscale_rcas.wesl`: the non-filterable
/// reconstructed `input_color` and the write-only `rgba16float` output.
fn rcas_layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`UpscalePipeline`], queuing both compute
/// pipelines and building the bilinear clamp history sampler.
pub(crate) fn init_upscale_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let reconstruct_entries = reconstruct_layout_entries();
    let reconstruct_descriptor =
        BindGroupLayoutDescriptor::new("prism upscale reconstruct", &reconstruct_entries);
    let reconstruct_layout =
        device.create_bind_group_layout("prism upscale reconstruct", &reconstruct_entries);

    let rcas_entries = rcas_layout_entries();
    let rcas_descriptor = BindGroupLayoutDescriptor::new("prism upscale rcas", &rcas_entries);
    let rcas_layout = device.create_bind_group_layout("prism upscale rcas", &rcas_entries);

    // Bilinear clamp: linear min/mag so the reprojected history UV interpolates.
    // History is single-mip, so mip filtering never engages.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism upscale history sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let reconstruct_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/upscale_reconstruct.wesl");
    let rcas_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/upscale_rcas.wesl");

    let reconstruct = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism upscale reconstruct".into()),
        layout: vec![reconstruct_descriptor],
        immediate_size: size_of::<GpuUpscaleReconstructParams>() as u32,
        shader: reconstruct_shader,
        entry_point: Some("upscale_reconstruct".into()),
        ..Default::default()
    });

    let rcas = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism upscale rcas".into()),
        layout: vec![rcas_descriptor],
        immediate_size: size_of::<GpuUpscaleRcasParams>() as u32,
        shader: rcas_shader,
        entry_point: Some("upscale_rcas".into()),
        ..Default::default()
    });

    commands.insert_resource(UpscalePipeline {
        reconstruct,
        rcas,
        reconstruct_layout,
        rcas_layout,
        sampler,
    });
}
