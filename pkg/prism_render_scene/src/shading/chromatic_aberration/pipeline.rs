//! The chromatic-aberration compute pipeline, its owned group-0 layout, the
//! shared linear-clamp sampler and the `RenderStartup` initializer that queues
//! them.
//!
//! The subsystem is a single full-screen compute pass (`chromatic_aberration_main`
//! from `chromatic_aberration.wesl`), specialized against one group-0 layout and
//! one immediate block. Unlike the `textureLoad`-based `DoF` chain, the
//! aberration pass fetches the resolved scene colour at three *sub-texel*
//! radially split coordinates, so it binds a filterable scene-colour texture and
//! a linear-clamp sampler and reads with `textureSampleLevel` — the same
//! filtered-fetch pattern [`super::super::volumetrics`]'s apply pass uses.

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

use super::abi::GpuChromaticAberrationParams;
use super::super::resources::SCENE_COLOR_FORMAT;

/// The chromatic-aberration compute pipeline, its owned group-0 layout and the
/// shared linear-clamp sampler the pass fetches the scene colour with.
#[derive(Resource)]
pub(crate) struct ChromaticAberrationPipeline {
    /// `chromatic_aberration_main` entry: full-screen radial channel split.
    pipeline: CachedComputePipelineId,
    /// group 0: filterable scene colour + linear sampler + write-only output.
    layout: BindGroupLayout,
    /// Linear-clamp sampler for the three per-channel scene-colour fetches.
    linear_sampler: Sampler,
}

impl ChromaticAberrationPipeline {
    /// The chromatic-aberration compute pipeline id.
    pub(crate) fn id(&self) -> CachedComputePipelineId {
        self.pipeline
    }

    /// The group-0 layout for the aberration dispatch.
    pub(crate) fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    /// The shared linear-clamp sampler the pass fetches the scene colour with.
    pub(crate) fn linear_sampler(&self) -> &Sampler {
        &self.linear_sampler
    }
}

/// group-0 layout for `chromatic_aberration_main` (sequential bindings 0-2):
///   0 = the resolved HDR scene colour (filterable float, sampled),
///   1 = the linear-clamp sampler (the three per-channel fetches),
///   2 = the write-only `rgba16float` aberrated output.
fn layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: true }),
            sampler(SamplerBindingType::Filtering),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`ChromaticAberrationPipeline`]. Queues the
/// compute pipeline against `chromatic_aberration.wesl`, its owned layout, and
/// the linear-clamp sampler the per-channel fetches read with.
pub(crate) fn init_chromatic_aberration_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism chromatic aberration", &entries);
    let layout = device.create_bind_group_layout("prism chromatic aberration", &entries);

    // Linear-clamp sampler for the three radially split per-channel fetches: the
    // scene colour is `rgba16float` (filterable) and clamped at the framebuffer
    // edges so a split that lands past the border reuses the edge texel rather
    // than wrapping.
    let linear_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("prism chromatic aberration linear-clamp sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..Default::default()
    });

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/chromatic_aberration.wesl");

    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism chromatic aberration".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuChromaticAberrationParams>() as u32,
        shader,
        entry_point: Some("chromatic_aberration_main".into()),
        ..Default::default()
    });

    commands.insert_resource(ChromaticAberrationPipeline {
        pipeline,
        layout,
        linear_sampler,
    });
}
