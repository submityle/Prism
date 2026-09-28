//! SSR composite: pipeline, per-view bind group, and the `Core3d` dispatch
//! node that folds the traced reflection buffer back over the shaded scene.
//!
//! The trace (`ssr.wesl`) leaves `rgb` = reflected radiance and `a` = a
//! `[0, 1]` blend confidence in its reflection output. This stage blends that
//! reflection into the shaded HDR `scene_color` the main pass presents, the GPU
//! twin of the composite step in
//! [`prism_render_shading::screen_space`], computing
//! `scene_color.rgb = mix(scene_color.rgb, reflection.rgb, reflection.a)`.
//!
//! `scene_color` is an `rgba16float` storage image, which is not read-write
//! storage-capable, so the *base* colour is read from colour-pyramid level 0 -
//! a byte-for-byte copy of `scene_color` the colour-mip build takes immediately
//! before this pass - and only the write side touches `scene_color`, keeping the
//! pass free of any read/write aliasing hazard on a single resource.
//!
//! It reads one bind group (group 0, matching `shaders/ssr_composite.wesl`):
//!
//! * `0` colour-pyramid level 0 (the untouched shaded colour, `textureLoad`ed),
//! * `1` the reflection output (`textureLoad`ed) — the temporally accumulated
//!   buffer when temporal accumulation is active, else the raw spatial resolve,
//!   and
//! * `2` the write-only `rgba16float` `scene_color` blended in place.
//!
//! The framebuffer extent travels in the [`GpuSsrCompositeParams`] immediate
//! block. Runs after the trace (its reflection input) and before the main pass
//! that presents `scene_color`.

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

use super::super::resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT};
use super::abi::{GpuSsrCompositeParams, SSR_WORKGROUP_SIZE};
use super::resources::ViewSsrTextures;
use super::temporal::ViewSsrTemporal;

/// Compute pipeline and its owned group-0 layout for the SSR composite.
#[derive(Resource)]
pub(crate) struct SsrCompositePipeline {
    /// `ssr_composite` compute entry point, specialized against the group-0
    /// layout and the 16-byte [`GpuSsrCompositeParams`] immediate block.
    composite: CachedComputePipelineId,
    /// group 0: colour-pyramid level 0 + reflection reads, `scene_color`
    /// blended in place.
    layout: BindGroupLayout,
}

/// group-0 layout mirroring `ssr_composite.wesl`: two non-filterable float reads
/// (the colour-pyramid level-0 copy and the reflection output, both
/// `textureLoad`ed) then the write-only `rgba16float` `scene_color` output.
fn layout_entries() -> BindGroupLayoutEntries<3> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SCENE_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrCompositePipeline`].
pub(crate) fn init_ssr_composite_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSR composite", &entries);
    let layout = device.create_bind_group_layout("prism SSR composite", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_composite.wesl");

    let composite = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR composite".into()),
        layout: vec![descriptor],
        immediate_size: size_of::<GpuSsrCompositeParams>() as u32,
        shader,
        entry_point: Some("ssr_composite".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrCompositePipeline { composite, layout });
}

/// The composite's group-0 bind group for a single view. Present only when both
/// the colour pyramid's level 0 and the reflection output are resident.
#[derive(Component)]
pub(crate) struct ViewSsrCompositeBindGroup {
    group: BindGroup,
}

/// `PrepareBindGroups` system building [`ViewSsrCompositeBindGroup`] for every
/// view that has both a visibility buffer (for `scene_color`) and resident SSR
/// textures (the colour pyramid + reflection output). Reads the temporally
/// accumulated reflection when a [`ViewSsrTemporal`] is present, else the raw
/// spatial resolve. Clears any stale group when the colour pyramid has no level
/// 0 to lift the base colour from.
pub(crate) fn prepare_ssr_composite_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrCompositePipeline>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        &ViewSsrTextures,
        Option<&ViewSsrTemporal>,
    )>,
) {
    for (entity, visibility, textures, temporal) in &views {
        let Some(color_l0) = textures.color_mip_view(0) else {
            commands
                .entity(entity)
                .remove::<ViewSsrCompositeBindGroup>();
            continue;
        };
        // With temporal accumulation the composite reads the stabilised
        // history output; otherwise it folds in the raw spatial resolve.
        let reflection = temporal
            .map(ViewSsrTemporal::write_view)
            .unwrap_or_else(|| textures.ssr_resolved_view());
        let group = device.create_bind_group(
            "prism SSR composite",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                color_l0,
                reflection,
                visibility.scene_color_view(),
            )),
        );
        commands
            .entity(entity)
            .insert(ViewSsrCompositeBindGroup { group });
    }
}

/// `Core3d` node recording the composite dispatch for every view.
///
/// Runs after the trace (its reflection input) and before the main pass that
/// presents `scene_color`. Dispatches one workgroup per 8x8 pixel tile; the
/// shader bounds-checks every invocation against the framebuffer extent.
pub(crate) fn ssr_composite_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrCompositeBindGroup)>,
    pipeline: Res<SsrCompositePipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
        return;
    }
    let (textures, group) = view.into_inner();

    let Some(composite) = cache.get_compute_pipeline(pipeline.composite) else {
        return;
    };

    let size = textures.size;
    if size.x == 0 || size.y == 0 {
        return;
    }

    let params = GpuSsrCompositeParams::new(size.x, size.y);
    let workgroups_x = size.x.div_ceil(SSR_WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(SSR_WORKGROUP_SIZE);

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism SSR composite"),
            timestamp_writes: None,
        });
    pass.set_pipeline(composite);
    pass.set_bind_group(0, &group.group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&params));
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}
