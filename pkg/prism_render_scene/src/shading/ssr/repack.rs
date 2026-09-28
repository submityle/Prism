//! SSR material-roughness repack: pipeline, per-view bind groups, and the
//! `Core3d` dispatch node.
//!
//! The trace samples a single `normal_roughness` texture, but the geometry
//! prepass can only reconstruct the signed view-space *geometric* normal - it
//! has no material bind group, so it cannot fetch roughness or apply the normal
//! map. This stage sits between them: it re-reads the prepass normal,
//! reconstructs the covered pixel's interpolated world-space surface from the
//! visibility buffer (walking the same scene/geometry tables the resolve does),
//! samples the material's metallic-roughness *and* normal textures through the
//! shared bindless heap, rotates a bound normal map into world space against the
//! interpolated basis and back into the view frame (exactly like the resolve),
//! and packs `rgb = normal * 0.5 + 0.5`, `a = texture-modulated perceptual
//! roughness` into the trace's input, via `shaders/ssr_repack.wesl`. The trace
//! therefore marches against the same normal-mapped surface the resolve shades.
//!
//! It reads three bind groups:
//!
//! * **group 0** - the two visibility textures (ids/metadata) read, the
//!   prepass view-normal read, and the packed `normal_roughness` written. Owned
//!   here because it is unique to this pass.
//! * **group 1** - the shared material tables, reusing [`MaterialBindGroup`]'s
//!   layout descriptor so the header/parameter/texture buffers and the bindless
//!   texture + sampler heaps bind byte-for-byte, and bound straight from the
//!   shared resource at dispatch time (exactly like the resolve pass). The
//!   kernel now declares the header + parameter + texture-record bindings and
//!   the heap arrays (via `material_sample.wesl`); the layout carries no more.
//! * **group 2** - the scene-instance and shading-geometry tables the surface
//!   reconstruction walks to recover the interpolated UV, plus the per-instance
//!   current `world_from_local` transforms that lift the reconstructed
//!   local-space surface into world space for the normal-map basis. Owned here
//!   (a subset of the resolve's group 2, which also carries the per-class
//!   worklist the repack does not need).

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
        BindGroup, BindGroupEntries, BindGroupLayout, CachedComputePipelineId,
        ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
        StorageTextureAccess, TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::ExtractedView,
};
use bevy_shader::Shader;

use crate::{GpuSceneBuffers, MaterialBindGroup, RenderShadingGeometryBuffers};

use super::super::resources::ViewVisibilityBuffer;
use super::abi::{GpuSsrRepackParams, SSR_WORKGROUP_SIZE};
use super::resources::{ViewSsrTextures, SSR_NORMAL_ROUGHNESS_FORMAT};

/// Compute pipeline and the two owned bind-group layouts for the SSR repack.
/// Group 1 reuses [`MaterialBindGroup`]'s layout descriptor and needs no owned
/// layout.
#[derive(Resource)]
pub(crate) struct SsrRepackPipeline {
    /// `ssr_repack` compute entry point, specialized against the group-0 layout,
    /// the reused material layout, the owned group-2 scene/geometry layout and
    /// the 80-byte immediate block (the `view_from_world` matrix + extent).
    repack: CachedComputePipelineId,
    /// group 0: visibility ids/metadata + prepass view-normal read, packed
    /// `normal_roughness` written.
    view_layout: BindGroupLayout,
    /// group 2: scene-instance + shading-geometry tables plus the per-instance
    /// current transforms (five read-only storage buffers) walked to recover the
    /// interpolated UV and world-space normal-map basis.
    scene_layout: BindGroupLayout,
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

/// group-2 layout: five read-only storage buffers (scene instances, the
/// geometry headers/vertices/primitives, then the per-instance current
/// `world_from_local` transforms used to lift the reconstructed local-space
/// surface into world space for the normal-map basis). `None` min-binding-size
/// keeps the layout agnostic to the run-time array length; the shader guards
/// every index. A subset of the resolve's group 2 (the repack needs no worklist
/// and no previous-frame transforms).
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

    let scene_entries = scene_layout_entries();
    let scene_descriptor = BindGroupLayoutDescriptor::new("prism SSR repack scene", &scene_entries);
    let scene_layout = device.create_bind_group_layout("prism SSR repack scene", &scene_entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_repack.wesl");

    let repack = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR repack".into()),
        layout: vec![
            view_descriptor,
            material_bindings.layout_descriptor.clone(),
            scene_descriptor,
        ],
        immediate_size: size_of::<GpuSsrRepackParams>() as u32,
        shader,
        entry_point: Some("ssr_repack".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrRepackPipeline {
        repack,
        view_layout,
        scene_layout,
    });
}

/// The pass-owned bind groups (group 0 + group 2) for one view's SSR repack.
/// Present only when the visibility buffer, SSR textures and the shared
/// scene/geometry tables are all resident. Group 1 is the shared material bind
/// group, bound directly at dispatch, so it is not stored here.
#[derive(Component)]
pub(crate) struct ViewSsrRepackBindGroup {
    view: BindGroup,
    scene: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsrRepackBindGroup`] for every view
/// that has both a visibility buffer and SSR textures, provided the shared
/// scene-instance and shading-geometry tables have uploaded. Group 1 is the
/// shared material bind group, bound directly at dispatch, so it is not stored
/// here.
pub(crate) fn prepare_ssr_repack_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrRepackPipeline>,
    device: Res<RenderDevice>,
    scene: Res<GpuSceneBuffers>,
    geometry: Res<RenderShadingGeometryBuffers>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsrTextures)>,
) {
    // Scene/geometry tables are shared across all views; without them there is
    // no UV to reconstruct, so clear any stale group and skip this frame.
    let (
        Some(instances),
        Some(current_transforms),
        Some((geo_headers, geo_vertices, geo_primitives)),
    ) = (
        scene.instances(),
        scene.current_transforms(),
        geometry.buffers(),
    ) else {
        for (entity, _, _) in &views {
            commands.entity(entity).remove::<ViewSsrRepackBindGroup>();
        }
        return;
    };

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
        let scene_group = device.create_bind_group(
            "prism SSR repack scene",
            &pipeline.scene_layout,
            &BindGroupEntries::sequential((
                instances.as_entire_binding(),
                geo_headers.as_entire_binding(),
                geo_vertices.as_entire_binding(),
                geo_primitives.as_entire_binding(),
                current_transforms.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(ViewSsrRepackBindGroup {
            view,
            scene: scene_group,
        });
    }
}

/// `Core3d` node recording the repack dispatch for every view.
///
/// Runs after the geometry prepass (which fills `view_normal`) and before the
/// trace/resolve that consume `normal_roughness`. Independent of the Hi-Z build,
/// so the two may run in either order. Binds the pass-owned group 0 + group 2
/// plus the shared material group and dispatches one workgroup per 8x8 pixel
/// tile.
pub(crate) fn ssr_repack_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrRepackBindGroup, &ExtractedView)>,
    material_bindings: Res<MaterialBindGroup>,
    pipeline: Res<SsrRepackPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
        return;
    }
    let (textures, group, extracted) = view.into_inner();

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

    // Same world->view transform the prepass used, so a normal-mapped world-space
    // normal lands in the identical camera-at-origin frame as the geometric
    // normal the prepass wrote.
    let view_from_world = extracted.world_from_view.to_matrix().inverse();
    let params = GpuSsrRepackParams::new(view_from_world, size.x, size.y);
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
    pass.set_bind_group(2, &group.scene, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
