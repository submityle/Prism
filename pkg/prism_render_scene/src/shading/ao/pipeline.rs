//! Compute pipeline + bind-group layouts for the GTAO geometry prepass.
//!
//! The prepass (`shaders/gtao_prepass.wesl`) reads two bind groups:
//!
//! * **group 0** — the two visibility textures (ids/metadata) read, plus the
//!   two view-space geometry targets (linear depth + normal) written.  Owned
//!   here because it is unique to this pass.
//! * **group 1** — the scene-instance and shading-geometry tables the surface
//!   reconstruction walks.  A strict subset of the resolve pass's group 2
//!   (no worklist), declared here with matching `None` min-binding-sizes so
//!   the shader's own generation/bounds guards remain the sole gate.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer_read_only_sized, texture_2d, texture_storage_2d},
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

use super::abi::{GpuGtaoConfig, GpuGtaoDenoiseConfig, GpuGtaoPrepassParams};
use super::resources::{GTAO_AO_FORMAT, GTAO_DEPTH_FORMAT, GTAO_NORMAL_FORMAT};

/// Compute pipeline and the two owned bind-group layouts for the GTAO prepass.
#[derive(Resource)]
pub(crate) struct GtaoPrepassPipeline {
    /// `gtao_prepass` compute entry point, specialized against both group
    /// layouts and the 80-byte immediate block.
    pub(crate) prepass: CachedComputePipelineId,
    /// group 0: visibility ids/metadata textures + linear-depth/view-normal
    /// storage outputs.
    pub(crate) view_layout: BindGroupLayout,
    /// group 1: scene instances + geometry headers/vertices/primitives.
    pub(crate) scene_layout: BindGroupLayout,
}

/// group-0 layout: two `texture_2d<u32>` visibility inputs followed by the
/// write-only linear-depth (`r32float`) and view-normal (`rgba16float`)
/// storage textures.
fn view_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
            texture_storage_2d(GTAO_DEPTH_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(GTAO_NORMAL_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// group-1 layout: four read-only storage buffers (scene instances, geometry
/// headers, geometry vertices, geometry primitives).
fn scene_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`GtaoPrepassPipeline`].
pub(crate) fn init_gtao_prepass_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = view_layout_entries();
    let scene_entries = scene_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism GTAO prepass view", &view_entries);
    let scene_descriptor =
        BindGroupLayoutDescriptor::new("prism GTAO prepass scene", &scene_entries);
    let view_layout = device.create_bind_group_layout("prism GTAO prepass view", &view_entries);
    let scene_layout = device.create_bind_group_layout("prism GTAO prepass scene", &scene_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/gtao_prepass.wesl");

    let prepass = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism GTAO prepass".into()),
        layout: vec![view_descriptor, scene_descriptor],
        immediate_size: size_of::<GpuGtaoPrepassParams>() as u32,
        shader,
        entry_point: Some("gtao_prepass".into()),
        ..Default::default()
    });

    commands.insert_resource(GtaoPrepassPipeline {
        prepass,
        view_layout,
        scene_layout,
    });
}

/// Compute pipeline and the single owned bind-group layout for the GTAO kernel.
///
/// The kernel (`shaders/gtao.wesl`) reads the two view-space geometry targets
/// produced by the prepass and writes ambient visibility, all through one bind
/// group; its projection/horizon tunables arrive in the 32-byte immediate
/// [`GpuGtaoConfig`] block.
#[derive(Resource)]
pub(crate) struct GtaoKernelPipeline {
    /// `compute_gtao` entry point, specialized against [`Self::view_layout`]
    /// and the 32-byte immediate config block.
    pub(crate) kernel: CachedComputePipelineId,
    /// group 0: linear-depth + view-normal sampled inputs followed by the
    /// write-only ambient-visibility (`r32float`) storage output.
    pub(crate) view_layout: BindGroupLayout,
}

/// group-0 layout for the kernel: the linear-depth and view-normal targets as
/// float-sampled textures, then the write-only ambient-visibility storage
/// texture.
fn kernel_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(GTAO_AO_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`GtaoKernelPipeline`].
pub(crate) fn init_gtao_kernel_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = kernel_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism GTAO kernel view", &view_entries);
    let view_layout = device.create_bind_group_layout("prism GTAO kernel view", &view_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/gtao.wesl");

    let kernel = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism GTAO kernel".into()),
        layout: vec![view_descriptor],
        immediate_size: size_of::<GpuGtaoConfig>() as u32,
        shader,
        entry_point: Some("compute_gtao".into()),
        ..Default::default()
    });

    commands.insert_resource(GtaoKernelPipeline {
        kernel,
        view_layout,
    });
}

/// Compute pipeline and the single owned bind-group layout for the GTAO spatial
/// denoiser.
///
/// The denoiser (`shaders/gtao_denoise.wesl`) reads the raw ambient-visibility
/// target the kernel produced plus the linear-depth and view-normal prepass
/// targets (for edge stopping) and writes the denoised visibility the resolve
/// samples, all through one bind group; its bilateral tunables arrive in the
/// 16-byte immediate [`GpuGtaoDenoiseConfig`] block.
#[derive(Resource)]
pub(crate) struct GtaoDenoisePipeline {
    /// `denoise_gtao` entry point, specialized against [`Self::view_layout`]
    /// and the 16-byte immediate config block.
    pub(crate) denoise: CachedComputePipelineId,
    /// group 0: raw-AO + linear-depth + view-normal sampled inputs followed by
    /// the write-only denoised-AO (`r32float`) storage output.
    pub(crate) view_layout: BindGroupLayout,
}

/// group-0 layout for the denoiser: the raw ambient visibility, linear depth,
/// and view normal as float-sampled textures, then the write-only denoised
/// ambient-visibility storage texture.
fn denoise_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(GTAO_AO_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`GtaoDenoisePipeline`].
pub(crate) fn init_gtao_denoise_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = denoise_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism GTAO denoise view", &view_entries);
    let view_layout = device.create_bind_group_layout("prism GTAO denoise view", &view_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/gtao_denoise.wesl");

    let denoise = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism GTAO denoise".into()),
        layout: vec![view_descriptor],
        immediate_size: size_of::<GpuGtaoDenoiseConfig>() as u32,
        shader,
        entry_point: Some("denoise_gtao".into()),
        ..Default::default()
    });

    commands.insert_resource(GtaoDenoisePipeline {
        denoise,
        view_layout,
    });
}
