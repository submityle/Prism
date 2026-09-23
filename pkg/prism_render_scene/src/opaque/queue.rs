use bevy_core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_mesh::{Mesh, Mesh3d};
use bevy_pbr::{MeshPipelineKey, RenderMeshInstances, ViewKeyCache};
use bevy_render::{
    camera::DirtySpecializations,
    mesh::{allocator::MeshAllocator, RenderMesh},
    render_asset::RenderAssets,
    render_phase::{
        BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, ViewBinnedRenderPhases,
    },
    render_resource::{PipelineCache, SpecializedMeshPipelines},
    view::{ExtractedView, RenderVisibleEntities},
};

use super::GpuSceneOpaqueEnabled;
use super::{
    draw::DrawGpuSceneOpaque,
    pipeline::{GpuSceneDebugView, GpuSceneOpaquePipeline, GpuSceneOpaquePipelineKey},
};
use crate::{
    visibility::runtime::UnifiedVisibilityState, GpuSceneDiagnostics, GpuSceneInstanceAddress,
    GpuSceneMode,
};

#[derive(SystemParam)]
pub(crate) struct OpaqueQueueControl<'w> {
    mode: Res<'w, GpuSceneMode>,
    enabled: Res<'w, GpuSceneOpaqueEnabled>,
    debug_view: Res<'w, GpuSceneDebugView>,
    diagnostics: ResMut<'w, GpuSceneDiagnostics>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Render queue joins view, mesh, pipeline, and scene state."
)]
pub(crate) fn queue_gpu_scene_opaque(
    mut control: OpaqueQueueControl,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<GpuSceneOpaquePipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<GpuSceneOpaquePipeline>>,
    draw_functions: Res<DrawFunctions<Opaque3d>>,
    mut phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    views: Query<(&RenderVisibleEntities, &ExtractedView)>,
    view_keys: Res<ViewKeyCache>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_instances: Res<RenderMeshInstances>,
    mesh_allocator: Res<MeshAllocator>,
    scene_instances: Query<&GpuSceneInstanceAddress>,
    dirty: Res<DirtySpecializations>,
    mut previously_queued: Local<bevy_render::sync_world::MainEntityHashSet>,
    visibility: Res<UnifiedVisibilityState>,
    scene: Res<crate::RenderGpuScene>,
) {
    control.diagnostics.opaque_visible = 0;
    control.diagnostics.opaque_queued = 0;
    control.diagnostics.opaque_skipped = 0;
    if *control.mode == GpuSceneMode::Disabled || !control.enabled.0 {
        let queued: Vec<_> = previously_queued.drain().collect();
        for phase in phases.values_mut() {
            for &entity in &queued {
                phase.remove(entity);
            }
        }
        return;
    }
    let draw_function = draw_functions.read().id::<DrawGpuSceneOpaque>();
    for (visible, view) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let Some(&view_key) = view_keys.get(&view.retained_view_entity) else {
            continue;
        };
        if let Some(visible_meshes) = visible.get::<Mesh3d>() {
            for &main_entity in dirty.iter_to_dequeue(view.retained_view_entity, visible_meshes) {
                phase.remove(main_entity);
                previously_queued.remove(&main_entity);
            }
        }
        let Some(unified) = visibility
            .frame
            .views
            .iter()
            .find(|(handle, _)| {
                visibility.retained_view(**handle) == Some(view.retained_view_entity)
            })
            .map(|(_, output)| output.visible_instances)
        else {
            continue;
        };
        let unified_entities: Vec<_> = visibility.frame.work_items
            [unified.start as usize..(unified.start + unified.count) as usize]
            .iter()
            .filter(|work| {
                work.pass_mask.0 & prism_render_visibility::RenderPassMask::OPAQUE.0 != 0
            })
            .filter_map(|work| scene.entity_binding_for_handle(work.scene))
            .collect();
        for (render_entity, main_entity) in unified_entities {
            control.diagnostics.opaque_visible += 1;
            if queue_one(
                phase,
                render_entity,
                main_entity,
                view_key,
                *control.debug_view,
                draw_function,
                &pipeline_cache,
                &pipeline,
                &mut pipelines,
                &render_meshes,
                &render_instances,
                &mesh_allocator,
                &scene_instances,
            ) {
                previously_queued.insert(main_entity);
                control.diagnostics.opaque_queued += 1;
            } else {
                control.diagnostics.opaque_skipped += 1;
            }
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Queues one mesh using already-borrowed render resources."
)]
fn queue_one(
    phase: &mut bevy_render::render_phase::BinnedRenderPhase<Opaque3d>,
    render_entity: Entity,
    main_entity: bevy_render::sync_world::MainEntity,
    view_key: MeshPipelineKey,
    debug_view: GpuSceneDebugView,
    draw_function: bevy_material::labels::DrawFunctionId,
    pipeline_cache: &PipelineCache,
    pipeline: &GpuSceneOpaquePipeline,
    pipelines: &mut SpecializedMeshPipelines<GpuSceneOpaquePipeline>,
    render_meshes: &RenderAssets<RenderMesh>,
    render_instances: &RenderMeshInstances,
    mesh_allocator: &MeshAllocator,
    scene_instances: &Query<&GpuSceneInstanceAddress>,
) -> bool {
    let Ok(address) = scene_instances.get(render_entity) else {
        return false;
    };
    let Some(mesh_id) = render_instances.mesh_asset_id(main_entity) else {
        return false;
    };
    let Some(mesh) = render_meshes.get(mesh_id) else {
        return false;
    };
    let Some(slabs) = mesh_allocator.mesh_slabs(&mesh_id) else {
        return false;
    };
    let key = view_key
        | MeshPipelineKey::from_primitive_topology_and_strip_index(
            mesh.primitive_topology(),
            mesh.index_format(),
        );
    let Ok(pipeline_id) = pipelines.specialize(
        pipeline_cache,
        pipeline,
        GpuSceneOpaquePipelineKey {
            mesh: key,
            debug: debug_view,
            has_normals: mesh.layout.0.contains(Mesh::ATTRIBUTE_NORMAL),
            has_uvs: mesh.layout.0.contains(Mesh::ATTRIBUTE_UV_0),
        },
        &mesh.layout,
    ) else {
        return false;
    };
    phase.add(
        Opaque3dBatchSetKey {
            pipeline: pipeline_id,
            draw_function,
            material_bind_group_index: None,
            slabs,
            lightmap_slab: None,
        },
        Opaque3dBinKey {
            asset_id: mesh_id.into(),
        },
        (render_entity, main_entity),
        InputUniformIndex(address.index),
        BinnedRenderPhaseType::NonMesh,
    );
    true
}
