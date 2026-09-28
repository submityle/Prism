//! SSR current-frame scene-colour mip-chain build: pipelines, per-view bind
//! groups, and the `Core3d` dispatch node.
//!
//! Once the shading resolve has written the current frame's HDR radiance into
//! `scene_color`, this stage copies it into the trace's current-frame colour
//! pyramid ([`ViewSsrTextures`]'s `color` target) and builds that pyramid's
//! full mip chain, one mip level per dispatch, via the two
//! `shaders/ssr_color_mips.wesl` entry points:
//!
//! * `ssr_color_copy`   lifts `scene_color` into pyramid level 0 (a 1:1 copy).
//! * `ssr_color_reduce` writes each coarser level as the 2x2 *average* (box
//!   downsample) of the finer one, so the trace samples a progressively
//!   pre-blurred reflection for rougher surfaces.
//!
//! The pyramid is a frame-transient ([`bevy_render::texture::TextureCache`])
//! target: the trace samples the *current* frame's own shaded colour at each
//! reflected-ray hit — never a reprojected previous frame — so there is no
//! cross-frame history, motion vectors, or reprojection matrices involved. The
//! roughness-selected mip is purely a spatial pre-blur of this frame's radiance.
//!
//! Both entry points share one bind-group layout -- a sampled source mip plus an
//! `rgba16float` storage destination mip -- so the two pipelines differ only in
//! entry point. Each level runs in its own compute pass so the finer level's
//! writes are visible before the next level samples them, mirroring
//! [`super::hzb`]. Because it consumes the resolve's `scene_color`, the node is
//! scheduled after the resolve rather than before it.

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

use super::abi::{GpuSsrHzbParams, SSR_WORKGROUP_SIZE};
use super::resources::{ViewSsrTextures, SSR_COLOR_FORMAT};
use super::super::resources::ViewVisibilityBuffer;

/// The two mip-build pipelines and the bind-group layout they share.
#[derive(Resource)]
pub(crate) struct SsrColorMipsPipeline {
    /// `ssr_color_copy`: lifts `scene_color` into pyramid level 0.
    copy: CachedComputePipelineId,
    /// `ssr_color_reduce`: 2x2 box average of the finer level into a coarser one.
    reduce: CachedComputePipelineId,
    /// Shared layout: sampled source mip (binding 0) + `rgba16float` storage
    /// destination mip (binding 1).
    layout: BindGroupLayout,
}

/// Shared build layout: a non-filterable float source mip (read via
/// `textureLoad`) followed by the write-only `rgba16float` destination mip.
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSR_COLOR_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrColorMipsPipeline`].
pub(crate) fn init_ssr_color_mips_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSR colour mips", &entries);
    let layout = device.create_bind_group_layout("prism SSR colour mips", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_color_mips.wesl");
    let params_size = size_of::<GpuSsrHzbParams>() as u32;

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR colour copy".into()),
        layout: vec![descriptor.clone()],
        immediate_size: params_size,
        shader: shader.clone(),
        entry_point: Some("ssr_color_copy".into()),
        ..Default::default()
    });
    let reduce = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR colour reduce".into()),
        layout: vec![descriptor],
        immediate_size: params_size,
        shader,
        entry_point: Some("ssr_color_reduce".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrColorMipsPipeline {
        copy,
        reduce,
        layout,
    });
}

/// One bind group per colour-pyramid mip level for a single view.
///
/// `levels[0]` copies `scene_color` into pyramid level 0; `levels[i]` (for
/// `i >= 1`) reduces pyramid level `i - 1` into level `i`. Present only when the
/// scene-colour source and every colour-pyramid mip view are resident.
#[derive(Component)]
pub(crate) struct ViewSsrColorMipsBindGroups {
    levels: Vec<BindGroup>,
}

/// `PrepareBindGroups` system building the per-level colour-mip bind groups for
/// every view with a resident SSR colour pyramid and visibility buffer.
pub(crate) fn prepare_ssr_color_mips_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrColorMipsPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewVisibilityBuffer, &ViewSsrTextures)>,
) {
    for (entity, visibility, textures) in &views {
        let mip_count = textures.color_mip_count();
        let mut levels = Vec::with_capacity(mip_count as usize);
        let mut complete = true;

        for level in 0..mip_count {
            let Some(dst) = textures.color_mip_view(level) else {
                complete = false;
                break;
            };
            // Level 0 lifts the resolve's full-resolution scene colour; coarser
            // levels reduce the finer pyramid level written by the prior dispatch.
            let src = if level == 0 {
                visibility.scene_color_view()
            } else {
                match textures.color_mip_view(level - 1) {
                    Some(view) => view,
                    None => {
                        complete = false;
                        break;
                    }
                }
            };
            levels.push(device.create_bind_group(
                "prism SSR colour mip level",
                &pipeline.layout,
                &BindGroupEntries::sequential((src, dst)),
            ));
        }

        if complete && !levels.is_empty() {
            commands
                .entity(entity)
                .insert(ViewSsrColorMipsBindGroups { levels });
        } else {
            commands
                .entity(entity)
                .remove::<ViewSsrColorMipsBindGroups>();
        }
    }
}

/// `Core3d` node recording the colour-pyramid mip build for every view.
///
/// Runs *after* the shading resolve (which fills `scene_color`) so it snapshots
/// the current frame the trace then samples. Each level is its own compute pass
/// so the previous level's writes are synchronised before this level samples
/// them.
pub(crate) fn ssr_color_mips_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrColorMipsBindGroups)>,
    pipeline: Res<SsrColorMipsPipeline>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if !settings.enable_ssr {
        return;
    }
    let (textures, groups) = view.into_inner();

    let (Some(copy), Some(reduce)) = (
        cache.get_compute_pipeline(pipeline.copy),
        cache.get_compute_pipeline(pipeline.reduce),
    ) else {
        return;
    };

    for (level, bind_group) in groups.levels.iter().enumerate() {
        let level = level as u32;
        let dst_size = textures.color_mip_size(level);
        if dst_size.x == 0 || dst_size.y == 0 {
            continue;
        }
        // Level 0 copies from the full-resolution scene colour (same extent);
        // coarser levels reduce from the finer level's extent.
        let src_size = if level == 0 {
            dst_size
        } else {
            textures.color_mip_size(level - 1)
        };
        let params = GpuSsrHzbParams::new(dst_size, src_size);
        let workgroups_x = dst_size.x.div_ceil(SSR_WORKGROUP_SIZE);
        let workgroups_y = dst_size.y.div_ceil(SSR_WORKGROUP_SIZE);

        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism SSR colour mip level"),
                timestamp_writes: None,
            });
        pass.set_pipeline(if level == 0 { copy } else { reduce });
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
