//! SSR spatial reconstruction: pipeline, per-view bind group, and the `Core3d`
//! dispatch node that bilateral-denoises the noisy multi-ray trace.
//!
//! The GGX-importance-sampled trace (`ssr.wesl`) shoots a handful of rays per
//! pixel, which still leaves visible noise on rough surfaces. Rather than pay
//! for hundreds of rays, this stage resolves the trace with an edge-aware
//! neighbourhood filter, the GPU twin of
//! [`prism_render_shading::screen_space::resolve_reflection`]: every nearby
//! pixel's reflection is a valid extra sample of the *same* lobe, so a bilateral
//! blur that respects surface normals, view-space depth and each neighbour's own
//! trace confidence collapses the noise without bleeding across silhouettes.
//!
//! It reads one bind group (group 0, matching `shaders/ssr_resolve.wesl`):
//!
//! * `0` the noisy trace output (`textureLoad`ed per neighbour),
//! * `1` the packed view-normal + roughness the repack produced,
//! * `2` the full-resolution device depth (for the linear-depth bilateral term),
//!   and
//! * `3` the write-only `rgba16float` resolved-reflection output.
//!
//! The inverse projection (for view-space depth reconstruction), framebuffer
//! extent, kernel radius and the shared golden tunables travel in the
//! [`GpuSsrResolveParams`] immediate block. Runs after the trace (its noisy
//! input) and before the composite, which now reads this resolved buffer.

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
    view::ExtractedView,
};
use bevy_shader::Shader;

use super::abi::{GpuSsrResolveParams, SSR_WORKGROUP_SIZE};
use super::resources::{ViewSsrTextures, SSR_OUT_FORMAT};

/// Compute pipeline and its owned group-0 layout for the SSR spatial resolve.
#[derive(Resource)]
pub(crate) struct SsrReconstructPipeline {
    /// `resolve_ssr` compute entry point, specialized against the group-0
    /// layout and the 96-byte [`GpuSsrResolveParams`] immediate block.
    reconstruct: CachedComputePipelineId,
    /// group 0: noisy trace + `normal_roughness` + depth reads, resolved output.
    layout: BindGroupLayout,
}

/// group-0 layout mirroring `ssr_resolve.wesl`: three non-filterable float reads
/// (the noisy trace output, the packed normal/roughness and device depth, all
/// `textureLoad`ed) then the write-only `rgba16float` resolved output.
fn layout_entries() -> BindGroupLayoutEntries<4> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSR_OUT_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrReconstructPipeline`].
pub(crate) fn init_ssr_reconstruct_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSR reconstruct", &entries);
    let layout = device.create_bind_group_layout("prism SSR reconstruct", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_resolve.wesl");

    let reconstruct = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR reconstruct".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSsrResolveParams>() as u32,
        shader,
        entry_point: Some("resolve_ssr".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrReconstructPipeline {
        reconstruct,
        layout,
    });
}

/// The resolve's group-0 bind group for a single view. Present only when every
/// backing SSR texture (noisy trace, `normal_roughness`, depth, resolved output)
/// is resident.
#[derive(Component)]
pub(crate) struct ViewSsrReconstructBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsrReconstructBindGroup`] for every
/// view with resident [`ViewSsrTextures`].
pub(crate) fn prepare_ssr_reconstruct_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrReconstructPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures)>,
) {
    for (entity, textures) in &views {
        let group = device.create_bind_group(
            "prism SSR reconstruct",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                textures.ssr_out_view(),
                textures.normal_roughness_view(),
                textures.scene_depth_sampled(),
                textures.ssr_resolved_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsrReconstructBindGroup { group });
    }
}

/// `Core3d` node recording the spatial-resolve dispatch for every view.
///
/// Runs after the trace (its noisy input) and before the composite that now
/// reads the resolved buffer. Dispatches one workgroup per 8x8 pixel tile; the
/// shader bounds-checks every invocation and skips background pixels.
pub(crate) fn ssr_reconstruct_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrReconstructBindGroup, &ExtractedView)>,
    pipeline: Res<SsrReconstructPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
        return;
    }
    let (textures, group, extracted) = view.into_inner();

    let Some(reconstruct) = cache.get_compute_pipeline(pipeline.reconstruct) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    // Inverse projection matches the trace's `view_from_clip` so the resolve
    // reconstructs the identical linear view-space depth for its bilateral term.
    let view_from_clip = extracted.clip_from_view.inverse();
    let params = GpuSsrResolveParams::new(view_from_clip, size.x, size.y);

    let workgroups_x = size.x.div_ceil(SSR_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSR_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSR reconstruct"),
            timestamp_writes: None,
        });
    pass.set_pipeline(reconstruct);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
