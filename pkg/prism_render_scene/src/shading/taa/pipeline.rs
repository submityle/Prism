//! Temporal anti-aliasing resolve pipeline, its owned group-0 layout, and the
//! filtering sampler the resolve reads the reprojected history through.
//!
//! Mirrors [`super::super::ssr::temporal`]'s pipeline plumbing: a single
//! `resolve_taa` compute entry specialized against a group-0 layout of five
//! bindings and the 20-byte [`GpuTaaResolveParams`] immediate block. The TAA
//! layout drops SSR temporal's separate device-depth gate (a resolve of the
//! composited `scene_color` needs no real-surface mask), so it carries five
//! bindings against SSR temporal's six.

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
        SamplerDescriptor, ShaderStages, StorageTextureAccess, TextureSampleType,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::super::resources::SCENE_COLOR_FORMAT;
use super::abi::GpuTaaResolveParams;

/// Compute pipeline, its owned group-0 layout, and the filtering sampler the
/// resolve reads the reprojected history through.
#[derive(Resource)]
pub(crate) struct TaaResolvePipeline {
    /// `resolve_taa` compute entry point, specialized against the group-0
    /// layout and the 20-byte [`GpuTaaResolveParams`] immediate block.
    resolve: CachedComputePipelineId,
    /// group 0: composited `scene_color` + motion-vector reads, the filterable
    /// history + its sampler, and the write-only resolved output.
    layout: BindGroupLayout,
    /// Bilinear clamp sampler bound at binding 2 so the reprojected history UV
    /// interpolates.
    sampler: Sampler,
}

impl TaaResolvePipeline {
    /// The `resolve_taa` compute pipeline id.
    pub(crate) fn resolve(&self) -> CachedComputePipelineId {
        self.resolve
    }

    /// The group-0 bind-group layout shared with the resolve dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    /// The bilinear clamp history sampler bound at binding 2.
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// group-0 layout mirroring `taa_resolve.wesl`: the composited `scene_color`
/// (non-filterable float, `textureLoad`ed as both the anchor and the 3x3
/// neighbourhood box), the *filterable* history + its filtering sampler
/// (sampled at the reprojected UV), the write-only `rgba16float` resolved
/// output, and the non-filterable motion-vector G-buffer the reprojection adds
/// back to reach last frame's UV.
fn layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    )
}

/// `RenderStartup` initializer for [`TaaResolvePipeline`].
pub(crate) fn init_taa_resolve_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism TAA resolve", &entries);
    let layout = device.create_bind_group_layout("prism TAA resolve", &entries);

    // Bilinear clamp: linear min/mag so the reprojected history UV interpolates.
    // History is single-mip, so mip filtering never engages.
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism TAA resolve history sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/taa_resolve.wesl");

    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism TAA resolve".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuTaaResolveParams>() as u32,
        shader,
        entry_point: Some("resolve_taa".into()),
        ..Default::default()
    });

    commands.insert_resource(TaaResolvePipeline {
        resolve,
        layout,
        sampler,
    });
}
