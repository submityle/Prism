//! Compute pipeline + bind-group layouts for the SSR geometry prepass.
//!
//! The prepass (`shaders/ssr_prepass.wesl`) reads two bind groups:
//!
//! * **group 0** — the two visibility textures (ids/metadata) read, plus the
//!   reverse-Z device-depth and view-normal targets written. Owned here because
//!   it is unique to this pass.
//! * **group 1** — the scene-instance and shading-geometry tables the surface
//!   reconstruction walks, plus the per-instance current `world_from_local`
//!   transforms that lift the reconstructed local-space surface into world
//!   space. A subset of the resolve pass's scene group (no worklist, no
//!   previous-frame transforms), declared here with matching `None`
//!   min-binding-sizes so the shader's own generation/bounds guards remain the
//!   sole gate.

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

use super::abi::GpuSsrPrepassParams;
use super::resources::{SSR_DEPTH_FORMAT, SSR_NORMAL_FORMAT};

/// Compute pipeline and the two owned bind-group layouts for the SSR prepass.
#[derive(Resource)]
pub(crate) struct SsrPrepassPipeline {
    /// `ssr_prepass` compute entry point, specialized against both group
    /// layouts and the 144-byte immediate block.
    pub(crate) prepass: CachedComputePipelineId,
    /// group 0: visibility ids/metadata textures + device-depth/view-normal
    /// storage outputs.
    pub(crate) view_layout: BindGroupLayout,
    /// group 1: scene instances + geometry headers/vertices/primitives.
    pub(crate) scene_layout: BindGroupLayout,
}

/// group-0 layout: two `texture_2d<u32>` visibility inputs followed by the
/// write-only device-depth (`r32float`) and view-normal (`rgba16float`) storage
/// textures.
fn view_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
            texture_storage_2d(SSR_DEPTH_FORMAT, StorageTextureAccess::WriteOnly),
            texture_storage_2d(SSR_NORMAL_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// group-1 layout: five read-only storage buffers (scene instances, geometry
/// headers, geometry vertices, geometry primitives, then the per-instance
/// current `world_from_local` transforms used to lift the reconstructed
/// local-space surface into world space).
fn scene_layout_entries() -> BindGroupLayoutEntries<5> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrPrepassPipeline`].
pub(crate) fn init_ssr_prepass_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = view_layout_entries();
    let scene_entries = scene_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism SSR prepass view", &view_entries);
    let scene_descriptor =
        BindGroupLayoutDescriptor::new("prism SSR prepass scene", &scene_entries);
    let view_layout = device.create_bind_group_layout("prism SSR prepass view", &view_entries);
    let scene_layout = device.create_bind_group_layout("prism SSR prepass scene", &scene_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_prepass.wesl");

    let prepass = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR prepass".into()),
        layout: vec![view_descriptor, scene_descriptor],
        immediate_size: size_of::<GpuSsrPrepassParams>() as u32,
        shader,
        entry_point: Some("ssr_prepass".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrPrepassPipeline {
        prepass,
        view_layout,
        scene_layout,
    });
}
