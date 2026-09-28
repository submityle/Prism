//! SSR material-roughness repack: pipeline, per-view bind group, and the
//! `Core3d` dispatch node.
//!
//! The trace samples a single `normal_roughness` texture, but the geometry
//! prepass can only reconstruct the signed view-space normal — it has no
//! material bind group, so it cannot fetch roughness. This stage sits between
//! them: it re-reads the prepass normal, resolves the covered pixel's material
//! through the shared material tables (identical generation/bounds guards to the
//! resolve stage), and packs `rgb = normal * 0.5 + 0.5`, `a = perceptual
//! roughness` into the trace's input, via `shaders/ssr_repack.wesl`.
//!
//! It reads two bind groups:
//!
//! * **group 0** — the two visibility textures (ids/metadata) read, the
//!   prepass view-normal read, and the packed `normal_roughness` written. Owned
//!   here because it is unique to this pass.
//! * **group 1** — the shared material tables, reusing [`MaterialBindGroup`]'s
//!   layout descriptor so the header/parameter buffers bind byte-for-byte, and
//!   bound straight from the shared resource at dispatch time (exactly like the
//!   resolve pass). The kernel declares only the header + parameter bindings; the
//!   further material-texture (and bindless array) bindings the layout carries
//!   are simply unused here.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{texture_2d, texture_storage_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
        StorageTextureAccess, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
};
use bevy_shader::Shader;

use crate::MaterialBindGroup;

use super::super::resources::ViewVisibilityBuffer;
use super::abi::{GpuSsrRepackParams, SSR_WORKGROUP_SIZE};
use super::resources::{ViewSsrTextures, SSR_NORMAL_ROUGHNESS_FORMAT};

/// Compute pipeline and the owned group-0 layout for the SSR repack. Group 1
/// reuses [`MaterialBindGroup`]'s layout descriptor and needs no owned layout.
#[derive(Resource)]
pub(crate) struct SsrRepackPipeline {
    /// `ssr_repack` compute entry point, specialized against the group-0 layout,
    /// the reused material layout and the 16-byte immediate block.
    repack: CachedComputePipelineId,
    /// group 0: visibility ids/metadata + prepass view-normal read, packed
    /// `normal_roughness` written.
    view_layout: BindGroupLayout,
}

/// group-0 layout: two `texture_2d<u32>` visibility inputs, the non-filterable
/// float view-normal read, then the write-only `rgba16float` `normal_roughness`
/// storage output.
fn view_layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Uint),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSR_NORMAL_ROUGHNESS_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrRepackPipeline`]. Must run after
/// [`MaterialBindGroup`] exists so its reflected layout descriptor is available
/// to clone into the pipeline's group-1 slot.
pub(crate) fn init_ssr_repack_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    material_bindings: Res<MaterialBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let view_entries = view_layout_entries();
    let view_descriptor = BindGroupLayoutDescriptor::new("prism SSR repack view", &view_entries);
    let view_layout = device.create_bind_group_layout("prism SSR repack view", &view_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_repack.wesl");

    let repack = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR repack".into()),
        layout: vec![view_descriptor, material_bindings.layout_descriptor.clone()],
        immediate_size: size_of::<GpuSsrRepackParams>() as u32,
        shader,
        entry_point: Some("ssr_repack".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrRepackPipeline {
        repack,
        view_layout,
    });
}

/// The pass-owned group-0 bind group for one view's SSR repack. Present only
/// when the visibility buffer and SSR textures are both resident.
#[derive(Component)]
pub(crate) struct ViewSsrRepackBindGroup {
    view: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsrRepackBindGroup`] for every view
/// that has both a visibility buffer and SSR textures. Group 1 is the shared
/// material bind group, bound directly at dispatch, so it is not stored here.
pub(crate) fn prepare_ssr_repack_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrRepackPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsrTextures)>,
) {
    for (entity, visibility, textures) in &views {
        let (ids, metadata) = visibility.attachments();
        let view = device.create_bind_group(
            "prism SSR repack view",
            &pipeline.view_layout,
            &BindGroupEntries::sequential((
                ids,
                metadata,
                textures.view_normal_view(),
                textures.normal_roughness_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsrRepackBindGroup { view });
    }
}

/// `Core3d` node recording the repack dispatch for every view.
///
/// Runs after the geometry prepass (which fills `view_normal`) and before the
/// trace/resolve that consume `normal_roughness`. Independent of the Hi-Z build,
/// so the two may run in either order. Binds the pass-owned group 0 plus the
/// shared material group and dispatches one workgroup per 8x8 pixel tile.
pub(crate) fn ssr_repack_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrRepackBindGroup)>,
    material_bindings: Res<MaterialBindGroup>,
    pipeline: Res<SsrRepackPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
        return;
    }
    let (textures, group) = view.into_inner();

    // The shared material bind group must be resident; without the material
    // tables there is no roughness to fold in.
    let Some(materials_group) = material_bindings.bind_group.as_ref() else {
        return;
    };
    let Some(repack) = cache.get_compute_pipeline(pipeline.repack) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuSsrRepackParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(SSR_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSR_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSR repack"),
            timestamp_writes: None,
        });
    pass.set_pipeline(repack);
    pass.set_bind_group(0, &group.view, &[]);
    pass.set_bind_group(1, materials_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
