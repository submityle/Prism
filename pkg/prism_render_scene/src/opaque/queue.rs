use bevy_core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy_ecs::prelude::*;
use bevy_mesh::Mesh3d;
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

use super::{
    draw::DrawGpuSceneOpaque,
    pipeline::{GpuSceneDebugView, GpuSceneOpaquePipeline, GpuSceneOpaquePipelineKey},
};
use crate::{GpuSceneDiagnostics, GpuSceneInstanceAddress, GpuSceneMode};

#[expect(
    clippy::too_many_arguments,
    reason = "Render queue joins view, mesh, pipeline, and scene state."
)]
pub(crate) fn queue_gpu_scene_opaque(
    mode: Res<GpuSceneMode>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<GpuSceneOpaquePipeline>,
    debug_view: Res<GpuSceneDebugView>,
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
    mut diagnostics: ResMut<GpuSceneDiagnostics>,
) {
    diagnostics.opaque_visible = 0;
    diagnostics.opaque_queued = 0;
    diagnostics.opaque_skipped = 0;
    if *mode == GpuSceneMode::Disabled {
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
        let Some(visible_meshes) = visible.get::<Mesh3d>() else {
            continue;
        };
        for &main_entity in dirty.iter_to_dequeue(view.retained_view_entity, visible_meshes) {
            phase.remove(main_entity);
            previously_queued.remove(&main_entity);
        }
        for &(render_entity, main_entity) in &visible_meshes.entities_cpu_culling {
            diagnostics.opaque_visible += 1;
            if queue_one(
                phase,
                render_entity,
                main_entity,
                view_key,
                *debug_view,
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
                diagnostics.opaque_queued += 1;
            } else {
                diagnostics.opaque_skipped += 1;
            }
        }
        for (&main_entity, &render_entity) in &visible_meshes.entities_gpu_culling {
            diagnostics.opaque_visible += 1;
            if queue_one(
                phase,
                render_entity,
                main_entity,
                view_key,
                *debug_view,
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
                diagnostics.opaque_queued += 1;
            } else {
                diagnostics.opaque_skipped += 1;
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
