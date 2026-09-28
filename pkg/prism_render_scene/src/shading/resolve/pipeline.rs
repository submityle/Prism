//! Compute pipeline + bind-group layouts for the shading-resolve pass.
//!
//! The resolve entry point (`shaders/shading_resolve.wesl`) is dispatched once
//! per [`MaterialShadingClass`](prism_render_shading::MaterialShadingClass)
//! with an indirect argument buffer.  It reads four bind groups:
//!
//! * **group 0** — visibility inputs + the HDR storage-texture output.  Owned
//!   here because it is unique to this pass.
//! * **group 1** — the shared material tables, reusing [`MaterialBindGroup`]'s
//!   layout so the same buffers bind byte-for-byte.
//! * **group 2** — the compacted per-class worklist plus the scene/geometry
//!   tables the surface reconstruction walks.  Owned here.
//! * **group 3** — the analytic light tables, reusing [`LightBindGroup`]'s
//!   layout.
//! * **group 4** — the shadow atlas + shadow tables, reusing
//!   [`ShadowBindGroup`]'s layout so the resolve pass samples the exact atlas
//!   the depth pass fills.
//! * **group 5** — the clustered-light tables, reusing [`ClusterBindGroup`]'s
//!   layout so each fragment iterates only the punctual lights assigned to its
//!   froxel instead of the whole scene.  Falls back to the full light list when
//!   a neutral single-cluster grid is bound.
//!
//! Reusing the material/light *layout descriptors* (rather than re-declaring
//! them) guarantees the resolve pipeline and those bind groups can never drift
//! out of sync.

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

use crate::{ClusterBindGroup, LightBindGroup, MaterialBindGroup};

use super::super::shadow::ShadowBindGroup;

use super::abi::GpuShadingResolveParams;
use super::super::resources::SCENE_COLOR_FORMAT;

/// Compute pipeline and the two owned bind-group layouts for the resolve pass.
#[derive(Resource)]
pub(crate) struct ShadingResolvePipeline {
    /// `shading_resolve` compute entry point, specialized against all four
    /// group layouts and the 32-byte immediate block.
    pub(crate) resolve: CachedComputePipelineId,
    /// group 0: visibility ids/metadata textures + HDR storage-texture output.
    pub(crate) view_layout: BindGroupLayout,
    /// group 2: compacted worklist + scene/geometry tables.
    pub(crate) scene_layout: BindGroupLayout,
}

/// Builds the group-0 layout entries: two `texture_2d<u32>` visibility inputs
/// followed by the write-only `rgba16float` storage texture.
fn view_layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// Builds the group-2 layout entries: seven read-only storage buffers
/// (work items, class offsets, class counts, scene instances, geometry
/// headers/vertices/primitives).  `None` min-binding-size keeps the layout
/// agnostic to the run-time array length; the shader guards every index.
fn scene_layout_entries() -> BindGroupLayoutEntries<7> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
            storage_buffer_read_only_sized(false, None),
        ),
    )
}

/// `RenderStartup` initializer.  Must run after both [`MaterialBindGroup`] and
/// [`LightBindGroup`] exist so their reflected layout descriptors are available
/// to clone into the pipeline's layout list.
pub(crate) fn init_shading_resolve_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    material_bindings: Res<MaterialBindGroup>,
    light_bindings: Res<LightBindGroup>,
    shadow_bindings: Res<ShadowBindGroup>,
    cluster_bindings: Res<ClusterBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = view_layout_entries();
    let scene_entries = scene_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism resolve view", &view_entries);
    let scene_descriptor = BindGroupLayoutDescriptor::new("prism resolve scene", &scene_entries);
    let view_layout = device.create_bind_group_layout("prism resolve view", &view_entries);
    let scene_layout = device.create_bind_group_layout("prism resolve scene", &scene_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/shading_resolve.wesl");

    let resolve = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism shading resolve".into()),
        layout: vec![
            view_descriptor,
            material_bindings.layout_descriptor.clone(),
            scene_descriptor,
            light_bindings.layout_descriptor.clone(),
            shadow_bindings.layout_descriptor.clone(),
            cluster_bindings.layout_descriptor.clone(),
        ],
        immediate_size: size_of::<GpuShadingResolveParams>() as u32,
        shader,
        entry_point: Some("shading_resolve".into()),
        ..Default::default()
    });

    commands.insert_resource(ShadingResolvePipeline {
        resolve,
        view_layout,
        scene_layout,
    });
}
