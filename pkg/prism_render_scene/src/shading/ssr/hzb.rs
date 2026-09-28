//! SSR Hi-Z ("nearest depth") pyramid build: pipelines, per-view bind groups,
//! and the `Core3d` dispatch node.
//!
//! The screen-space trace marches a reverse-Z depth pyramid so it can skip
//! empty screen regions in sub-linear time. This stage builds that pyramid from
//! the geometry prepass's `scene_depth`, one mip level per dispatch, via the two
//! `shaders/ssr_hzb.wesl` entry points:
//!
//! * `ssr_hzb_copy`   lifts `scene_depth` into pyramid level 0 (a 1:1 copy).
//! * `ssr_hzb_reduce` writes each coarser level as the 2x2 max-reduction of the
//!   finer one (reverse-Z keeps the *nearest* surface as the per-cell maximum),
//!   reproducing `prism_render_shading::DepthPyramid::from_nearest_reduction`.
//!
//! Both entry points share one bind-group layout — a sampled source mip plus an
//! `r32float` storage destination mip — so the two pipelines differ only in
//! entry point. Each level runs in its own compute pass so the finer level's
//! writes are visible before the next level samples them (dispatches inside a
//! single compute pass are not ordered against each other).

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
use super::resources::{ViewSsrTextures, SSR_HZB_FORMAT};

/// The two pyramid-build pipelines and the bind-group layout they share.
#[derive(Resource)]
pub(crate) struct SsrHzbPipeline {
    /// `ssr_hzb_copy`: lifts `scene_depth` into pyramid level 0.
    copy: CachedComputePipelineId,
    /// `ssr_hzb_reduce`: 2x2 max-reduction of the finer level into a coarser one.
    reduce: CachedComputePipelineId,
    /// Shared layout: sampled source mip (binding 0) + storage destination mip
    /// (binding 1).
    layout: BindGroupLayout,
}

/// Shared build layout: a non-filterable float source mip followed by the
/// write-only `r32float` destination mip.
fn layout_entries() -> BindGroupLayoutEntries<2> {
    BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_storage_2d(SSR_HZB_FORMAT, StorageTextureAccess::WriteOnly),
        ),
    )
}

/// `RenderStartup` initializer for [`SsrHzbPipeline`].
pub(crate) fn init_ssr_hzb_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = layout_entries();
    let descriptor = BindGroupLayoutDescriptor::new("prism SSR HZB", &entries);
    let layout = device.create_bind_group_layout("prism SSR HZB", &entries);

    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/ssr_hzb.wesl");
    let params_size = size_of::<GpuSsrHzbParams>() as u32;

    let copy = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR HZB copy".into()),
        layout: vec![descriptor.clone()],
        immediate_size: params_size,
        shader: shader.clone(),
        entry_point: Some("ssr_hzb_copy".into()),
        ..Default::default()
    });
    let reduce = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism SSR HZB reduce".into()),
        layout: vec![descriptor],
        immediate_size: params_size,
        shader,
        entry_point: Some("ssr_hzb_reduce".into()),
        ..Default::default()
    });

    commands.insert_resource(SsrHzbPipeline {
        copy,
        reduce,
        layout,
    });
}

/// One bind group per pyramid level for a single view.
///
/// `levels[0]` copies `scene_depth` into pyramid level 0; `levels[i]` (for
/// `i >= 1`) reduces level `i - 1` into level `i`. Present only when every
/// backing mip view is resident.
#[derive(Component)]
pub(crate) struct ViewSsrHzbBindGroups {
    levels: Vec<BindGroup>,
}

/// `PrepareBindGroups` system building the per-level HZB bind groups for every
/// view that has resident [`ViewSsrTextures`].
pub(crate) fn prepare_ssr_hzb_bind_groups(
    mut commands: Commands,
    pipeline: Res<SsrHzbPipeline>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ViewSsrTextures)>,
) {
    for (entity, textures) in &views {
        let mip_count = textures.hzb_mip_count();
        let mut levels = Vec::with_capacity(mip_count as usize);
        let mut complete = true;

        for level in 0..mip_count {
            let Some(dst) = textures.hzb_mip_view(level) else {
                complete = false;
                break;
            };
            // Level 0 lifts the full-resolution device depth; coarser levels
            // reduce the finer pyramid level written by the previous dispatch.
            let src = if level == 0 {
                textures.scene_depth_sampled()
            } else {
                match textures.hzb_mip_view(level - 1) {
                    Some(view) => view,
                    None => {
                        complete = false;
                        break;
                    }
                }
            };
            levels.push(device.create_bind_group(
                "prism SSR HZB level",
                &pipeline.layout,
                &BindGroupEntries::sequential((src, dst)),
            ));
        }

        if complete && !levels.is_empty() {
            commands
                .entity(entity)
                .insert(ViewSsrHzbBindGroups { levels });
        } else {
            commands.entity(entity).remove::<ViewSsrHzbBindGroups>();
        }
    }
}

/// `Core3d` node recording the pyramid build for every view.
///
/// Runs after the geometry prepass (which fills `scene_depth`) and before the
/// trace. Each level is its own compute pass so the previous level's writes are
/// synchronized before this level samples them.
pub(crate) fn ssr_hzb_pass(
    settings: Res<super::super::runtime::PrismShadingSettings>,
    view: ViewQuery<(&ViewSsrTextures, &ViewSsrHzbBindGroups)>,
    pipeline: Res<SsrHzbPipeline>,
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
        let dst_size = textures.hzb_mip_size(level);
        if dst_size.x == 0 || dst_size.y == 0 {
            continue;
        }
        // Level 0 copies from the full-resolution depth (same extent); coarser
        // levels reduce from the finer level's extent.
        let src_size = if level == 0 {
            dst_size
        } else {
            textures.hzb_mip_size(level - 1)
        };
        let params = GpuSsrHzbParams::new(dst_size, src_size);
        let workgroups_x = dst_size.x.div_ceil(SSR_WORKGROUP_SIZE);
        let workgroups_y = dst_size.y.div_ceil(SSR_WORKGROUP_SIZE);

        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism SSR HZB level"),
                timestamp_writes: None,
            });
        pass.set_pipeline(if level == 0 { copy } else { reduce });
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
