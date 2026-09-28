//! Queue system for the transparent forward (WBOIT) draw pass.
//!
//! Mirrors [`super::super::raster::queue_visibility_raster`] but selects the
//! transparent slice of the unified visibility work list
//! ([`RenderPassMask::TRANSPARENT`]) instead of the opaque one, and specializes
//! [`OitForwardPipeline`] (which needs the mesh's optional normal/uv attributes
//! to keep shader defs and vertex layout aligned). Retained-phase bookkeeping
//! (add newly-visible, remove no-longer-visible) is identical.

use bevy_ecs::prelude::*;
use bevy_mesh::Mesh;
use bevy_pbr::{MeshPipelineKey, RenderMeshInstances, ViewKeyCache};
use bevy_render::{
    mesh::{RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_phase::{
        BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, ViewBinnedRenderPhases,
    },
    render_resource::{PipelineCache, SpecializedMeshPipelines},
    view::ExtractedView,
};
use prism_render_visibility::RenderPassMask;

use crate::{GpuSceneInstanceAddress, RenderGpuScene};

use super::super::runtime::PrismShadingSettings;
use super::draw::DrawTransparentOit;
use super::phase::{TransparentOit3d, TransparentOitBatchSetKey, TransparentOitBinKey};
use super::pipeline::{OitForwardPipeline, OitForwardPipelineKey};

#[expect(
    clippy::too_many_arguments,
    reason = "Transparent queue joins retained scene, view, and mesh state."
)]
pub(crate) fn queue_transparent_oit(
    settings: Res<PrismShadingSettings>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<OitForwardPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<OitForwardPipeline>>,
    draw_functions: Res<DrawFunctions<TransparentOit3d>>,
    mut phases: ResMut<ViewBinnedRenderPhases<TransparentOit3d>>,
    views: Query<&ExtractedView>,
    view_keys: Res<ViewKeyCache>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_instances: Res<RenderMeshInstances>,
    scene_instances: Query<&GpuSceneInstanceAddress>,
    visibility: Res<crate::visibility::runtime::UnifiedVisibilityState>,
    scene: Res<RenderGpuScene>,
    mut previously_queued: Local<
        bevy_platform::collections::HashMap<
            bevy_render::view::RetainedViewEntity,
            bevy_render::sync_world::MainEntityHashSet,
        >,
    >,
) {
    if !settings.enable_visibility_buffer {
        phases.clear();
        previously_queued.clear();
        return;
    }
    let draw_function = draw_functions.read().id::<DrawTransparentOit>();
    for view in &views {
        if view_keys
            .get(&view.retained_view_entity)
            .is_none_or(|key| key.msaa_samples() != 1)
        {
            if let Some(phase) = phases.get_mut(&view.retained_view_entity)
                && let Some(previous) = previously_queued.remove(&view.retained_view_entity)
            {
                for entity in previous {
                    phase.remove(entity);
                }
            }
            continue;
        }
        let Some(view_output) = visibility
            .frame
            .views
            .iter()
            .find(|(handle, _)| {
                visibility.retained_view(**handle) == Some(view.retained_view_entity)
            })
            .map(|(_, output)| output)
        else {
            if let Some(phase) = phases.get_mut(&view.retained_view_entity)
                && let Some(previous) = previously_queued.remove(&view.retained_view_entity)
            {
                for entity in previous {
                    phase.remove(entity);
                }
            }
            continue;
        };
        let Some(&view_key) = view_keys.get(&view.retained_view_entity) else {
            continue;
        };
        phases.prepare_for_new_frame(
            view.retained_view_entity,
            bevy_render::batching::gpu_preprocessing::GpuPreprocessingMode::None,
        );
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let previous = previously_queued.remove(&view.retained_view_entity);
        let mut current = bevy_render::sync_world::MainEntityHashSet::default();
        let mut candidates = Vec::new();
        for work in &visibility.frame.work_items[view_output.visible_instances.start as usize
            ..view_output
                .visible_instances
                .start
                .saturating_add(view_output.visible_instances.count) as usize]
        {
            if work.pass_mask.0 & RenderPassMask::TRANSPARENT.0 == 0 {
                continue;
            }
            let Some((render_entity, main_entity)) = scene.entity_binding_for_handle(work.scene)
            else {
                continue;
            };
            candidates.push((render_entity, main_entity));
        }
        for (render_entity, main_entity) in candidates {
            let Ok(address) = scene_instances.get(render_entity) else {
                continue;
            };
            let Some(mesh_id) = render_instances.mesh_asset_id(main_entity) else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_id) else {
                continue;
            };
            let key = view_key
                | MeshPipelineKey::from_primitive_topology_and_strip_index(
                    mesh.primitive_topology(),
                    mesh.index_format(),
                );
            let Ok(pipeline_id) = pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                OitForwardPipelineKey {
                    mesh: key,
                    has_normals: mesh.layout.0.contains(Mesh::ATTRIBUTE_NORMAL),
                    has_uvs: mesh.layout.0.contains(Mesh::ATTRIBUTE_UV_0),
                },
                &mesh.layout,
            ) else {
                continue;
            };
            current.insert(main_entity);
            phase.add(
                TransparentOitBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function,
                    indexed: matches!(mesh.buffer_info, RenderMeshBufferInfo::Indexed { .. }),
                },
                TransparentOitBinKey(mesh_id.into()),
                (render_entity, main_entity),
                InputUniformIndex(address.index),
                BinnedRenderPhaseType::NonMesh,
            );
        }
        if let Some(previous) = previous {
            for entity in previous {
                if !current.contains(&entity) {
                    phase.remove(entity);
                }
            }
        }
        previously_queued.insert(view.retained_view_entity, current);
    }
}
