use alloc::collections::BTreeMap;

use bevy_camera::{primitives::Frustum, visibility::RenderLayers};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{CommandEncoderDescriptor, ComputePassDescriptor, PipelineCache},
    renderer::PendingCommandBuffers,
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};
use prism_render_visibility::{
    build_view_draw_bins, cull_view, DrawBinCandidate, DrawBinKey, GeometryLod,
    GeometryLodChain, GpuViewRecord, ViewFlags, VisibilityFrame, VisibilityInput,
};

use crate::{
    material::runtime::RenderMaterialRegistry,
    scene::RenderGpuScene,
    visibility::{
        buffers::UnifiedVisibilityBuffers,
        gpu::{VisibilityComputeBindGroup, VisibilityComputePipeline},
        rows::{
            RenderVisibilityDispatch, RenderVisibilityRange, RenderVisibilityView,
            RenderVisibilityWorkItem,
        },
        runtime::{
            PrismVisibilityDiagnostics, UnifiedVisibilityEnabled, UnifiedVisibilitySettings,
            UnifiedVisibilityState, VisibilityFrameGraph,
        },
    },
};

const VISIBILITY_WORKGROUP_SIZE: u32 = 64;

pub(crate) fn build_unified_visibility(
    enabled: Res<UnifiedVisibilityEnabled>,
    settings: Res<UnifiedVisibilitySettings>,
    scene: Res<RenderGpuScene>,
    materials: Res<RenderMaterialRegistry>,
    geometries: Res<crate::RenderGeometryRegistry>,
    mut state: ResMut<UnifiedVisibilityState>,
    mut buffers: ResMut<UnifiedVisibilityBuffers>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
    views: Query<(&ExtractedView, Option<&Frustum>, Option<&RenderLayers>)>,
    frame_graph: Res<VisibilityFrameGraph>,
) {
    state.views.clear();
    state.frame = VisibilityFrame::default();
    state.draw_bins.clear();
    state.begin_frame();
    *diagnostics = PrismVisibilityDiagnostics {
        buffer_version: buffers.version(),
        ..Default::default()
    };
    if !enabled.0 {
        buffers.stage([], [], [], 0);
        return;
    }
    debug_assert!(!frame_graph.compiled.execution_order.is_empty());

    let handles = scene.mirror().live_handles();
    let geometry = geometry_lods(scene.mirror(), &handles, &geometries);
    let material_records = material_records(&materials, scene.mirror(), &handles);
    let previous_lods = state.previous_lods().clone();
    let occluded = state.occluded().clone();

    for (extracted, frustum, layers) in &views {
        let retained = extracted.retained_view_entity;
        let handle = state.handle(retained);
        let clip = extracted.clip_from_world.unwrap_or_else(|| {
            extracted.clip_from_view * extracted.world_from_view.to_matrix().inverse()
        });
        let clip_array = clip.to_cols_array_2d();
        let position = extracted.world_from_view.translation().to_array();
        let (previous_clip, previous_position) = state.previous(retained);
        let previous_history_epoch = state.previous_history_epoch(retained);
        let camera_cut = previous_position.is_none_or(|previous| {
            squared_distance(previous, position)
                > settings.camera_cut_distance * settings.camera_cut_distance
        });
        let mut flags = ViewFlags::REVERSE_Z;
        if camera_cut {
            flags |= ViewFlags::CAMERA_CUT;
            diagnostics.camera_cuts += 1;
        }
        let layer_mask = layers
            .and_then(|layers| layers.bits().first().copied())
            .unwrap_or(1) as u32;
        let record = GpuViewRecord {
            handle,
            clip_from_world: clip_array,
            previous_clip_from_world: previous_clip.unwrap_or(clip_array),
            world_position: position,
            lod_scale: extracted.viewport.w.max(1) as f32,
            viewport: extracted.viewport.to_array(),
            frustum_planes: frustum
                .map(|value| value.half_spaces.map(|plane| plane.normal_d().to_array()))
                .unwrap_or([[0.0; 4]; 6]),
            layer_mask,
            flags,
            history_epoch: previous_history_epoch
                .unwrap_or(0)
                .saturating_add(u64::from(camera_cut)),
        };
        let (work, stats) = cull_view(
            &record,
            VisibilityInput {
                scene: scene.mirror(),
                handles: &handles,
                geometry: &geometry,
                materials: &material_records,
                previous_lods: &previous_lods,
                occluded: &occluded,
                capacity: settings
                    .max_work_items
                    .saturating_sub(state.frame.work_items.len() as u32),
                previous_history_epoch,
            },
        );
        diagnostics.input_instances += stats.input_instances;
        diagnostics.visible_instances += stats.visible_instances;
        diagnostics.material_bins += stats.material_bins;
        diagnostics.frustum_rejected += stats.frustum_rejected;
        diagnostics.occlusion_rejected += stats.occlusion_rejected;
        diagnostics.stale_handles += stats.stale_handles;
        diagnostics.missing_geometry += stats.missing_geometry;
        diagnostics.missing_material += stats.missing_material;
        diagnostics.overflows += u32::from(stats.overflowed);
        state.frame.push_view(handle, work, stats);
        let output = state.frame.views[&handle].visible_instances;
        let candidates: Vec<_> = state.frame.work_items
            [output.start as usize..(output.start + output.count) as usize]
            .iter()
            .filter_map(|work| draw_bin_candidate(work, &geometries, &material_records))
            .collect();
        let mut bins = build_view_draw_bins(
            handle,
            scene.mirror().capacity() as u32,
            candidates,
        );
        bins.global_bin_start = state.draw_bins.iter().map(|view| view.bins.len() as u32).sum();
        bins.global_candidate_start = state
            .draw_bins
            .iter()
            .map(|view| view.candidate_bins.len() as u32)
            .sum();
        state.draw_bins.push(bins);
        state.views.push(record);
        state.remember(retained, clip_array, position, record.history_epoch);
    }

    diagnostics.views = state.views.len() as u32;
    diagnostics.work_items = state.frame.work_items.len() as u32;
    diagnostics.cpu_reference_frames = 1;
    diagnostics.draw_bins = state.draw_bins.iter().map(|view| view.bins.len() as u32).sum();
    diagnostics.draw_bin_capacity = state.draw_bins.iter().map(|view| view.command_count).sum();
    state.commit_lods();
    state.retire_missing_views();
    let ranges = state.frame.views.iter().map(|(view, output)| {
        RenderVisibilityRange::new(
            *view,
            output.visible_instances.start,
            output.visible_instances.count,
        )
    });
    buffers.stage(
        state.views.iter().map(RenderVisibilityView::from),
        state
            .frame
            .work_items
            .iter()
            .copied()
            .map(RenderVisibilityWorkItem::from),
        ranges,
        (handles.len() as u32).min(settings.gpu_parity_max_items_per_view),
    );
    buffers.stage_draw_bins(&state.draw_bins);
}

fn draw_bin_candidate(
    work: &prism_render_visibility::GpuRenderWorkItem,
    geometries: &crate::RenderGeometryRegistry,
    materials: &BTreeMap<
        prism_render_architecture::gpu_scene::SceneMaterialHandle,
        prism_render_material::MaterialRecord,
    >,
) -> Option<DrawBinCandidate> {
    let geometry = geometries.record(work.geometry)?;
    let lod = geometry.resolve_lod(work.lod_or_cluster)?;
    let material = materials.get(&work.material)?;
    Some(DrawBinCandidate {
        scene: work.scene,
        key: DrawBinKey {
            geometry: work.geometry,
            pipeline_class: ((material.shading_model as u32) << 16)
                | material.render_class as u32,
            vertex_buffer_class: geometry.vertex_buffer_class,
            index_buffer_class: geometry.index_buffer_class,
            indexed: lod.primitive_kind
                == prism_render_architecture::geometry::GeometryPrimitiveKind::Indexed,
        },
    })
}

pub(crate) fn upload_unified_visibility(
    mut buffers: ResMut<UnifiedVisibilityBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffers.upload(&device, &queue);
}

pub(crate) fn rebuild_unified_visibility(
    mut buffers: ResMut<UnifiedVisibilityBuffers>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
) {
    buffers.reset_after_device_loss();
    diagnostics.buffer_version = buffers.version();
}

#[expect(
    clippy::too_many_arguments,
    reason = "Compute dispatch joins scene/material/output bindings and diagnostics."
)]
pub(crate) fn dispatch_unified_visibility(
    enabled: Res<UnifiedVisibilityEnabled>,
    settings: Res<UnifiedVisibilitySettings>,
    scene: Res<RenderGpuScene>,
    state: Res<UnifiedVisibilityState>,
    pipeline: Res<VisibilityComputePipeline>,
    cache: Res<PipelineCache>,
    scene_bindings: Res<crate::buffers::GpuSceneBindGroup>,
    material_bindings: Res<crate::MaterialBindGroup>,
    geometry_bindings: Res<crate::GeometryBindGroup>,
    output_bindings: Res<VisibilityComputeBindGroup>,
    buffers: Res<UnifiedVisibilityBuffers>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
) {
    if !enabled.0 || state.views.is_empty() || scene.mirror().capacity() == 0 {
        return;
    }
    let Some(compute_pipeline) = cache.get_compute_pipeline(pipeline.pipeline) else {
        diagnostics.pipeline_not_ready = 1;
        return;
    };
    let (
        Some(scene_bind_group),
        Some(material_bind_group),
        Some(geometry_bind_group),
        Some(output_bind_group),
    ) = (
        scene_bindings.bind_group.as_ref(),
        material_bindings.bind_group.as_ref(),
        geometry_bindings.bind_group.as_ref(),
        output_bindings.bind_group.as_ref(),
    ) else {
        return;
    };
    let candidate_count = scene.mirror().capacity() as u32;
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism unified visibility"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism unified visibility"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute_pipeline);
        pass.set_bind_group(0, scene_bind_group, &[]);
        pass.set_bind_group(1, material_bind_group, &[]);
        pass.set_bind_group(2, geometry_bind_group, &[]);
        pass.set_bind_group(3, output_bind_group, &[]);
        for (view_index, _) in state.views.iter().enumerate() {
            let output_start = (view_index as u32).saturating_mul(buffers.gpu_slots_per_view());
            let immediates = RenderVisibilityDispatch {
                view_index: view_index as u32,
                candidate_count,
                output_start,
                output_end: output_start.saturating_add(buffers.gpu_slots_per_view()),
                indirect_first_instance: u32::from(settings.indirect_first_instance),
                bin_start: state.draw_bins[view_index].global_bin_start,
                candidate_bin_start: state.draw_bins[view_index].global_candidate_start,
                _padding: 0,
            };
            pass.set_immediates(0, bytemuck::bytes_of(&immediates));
            pass.dispatch_workgroups(candidate_count.div_ceil(VISIBILITY_WORKGROUP_SIZE), 1, 1);
            diagnostics.gpu_compute_dispatches += 1;
            diagnostics.gpu_candidates += candidate_count;
            diagnostics.indirect_identity_fallbacks +=
                u32::from(!settings.indirect_first_instance) * candidate_count;
        }
    }
    pending.push_encoder(encoder, "prism unified visibility");
}

fn geometry_lods(
    scene: &prism_render_architecture::gpu_scene::CpuRenderScene,
    handles: &[prism_render_architecture::gpu_scene::SceneHandle],
    registry: &crate::RenderGeometryRegistry,
) -> BTreeMap<prism_render_architecture::gpu_scene::GeometryHandle, GeometryLodChain> {
    handles
        .iter()
        .filter_map(|handle| scene.get(*handle).map(|instance| instance.geometry))
        .map(|geometry| {
            (
                geometry,
                GeometryLodChain {
                    geometry,
                    lods: registry.record(geometry).map_or_else(
                        Vec::new,
                        |record| {
                            record
                                .lods
                                .iter()
                                .map(|lod| GeometryLod {
                                    level: lod.level as u16,
                                    screen_error: lod.screen_error,
                                    resident: lod.resident,
                                    fallback: lod.fallback,
                                })
                                .collect()
                        },
                    ),
                },
            )
        })
        .collect()
}

fn material_records(
    materials: &RenderMaterialRegistry,
    scene: &prism_render_architecture::gpu_scene::CpuRenderScene,
    handles: &[prism_render_architecture::gpu_scene::SceneHandle],
) -> BTreeMap<
    prism_render_architecture::gpu_scene::SceneMaterialHandle,
    prism_render_material::MaterialRecord,
> {
    handles
        .iter()
        .filter_map(|handle| scene.get(*handle).map(|instance| instance.material))
        .map(|handle| (handle, materials.registry.record_or_fallback(handle)))
        .collect()
}

fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.into_iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum()
}

#[cfg(test)]
mod tests {
    use prism_render_architecture::{
        abi::GenerationalHandle,
        gpu_scene::{InstanceRecord, SceneOperation, SceneTransaction},
    };

    use super::*;

    #[test]
    fn helper_tables_cover_live_scene_and_stale_material_fallback() {
        let handle = GenerationalHandle {
            index: 1,
            generation: 1,
        };
        let geometry = GenerationalHandle {
            index: 7,
            generation: 2,
        };
        let stale_material = GenerationalHandle {
            index: 9,
            generation: 4,
        };
        let mut scene = prism_render_architecture::gpu_scene::CpuRenderScene::default();
        scene.apply(&SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 1,
            operations: vec![SceneOperation::Create {
                handle,
                record: InstanceRecord {
                    geometry,
                    material: stale_material,
                    ..Default::default()
                },
            }],
        });
        let handles = scene.live_handles();
        assert_eq!(handles, vec![handle]);
        let mut geometries = crate::RenderGeometryRegistry::default();
        geometries.upsert(
            bevy_asset::AssetId::Uuid {
                uuid: bevy_asset::uuid::Uuid::from_u128(7),
            },
            prism_render_architecture::geometry::GeometryRecord {
                handle: geometry,
                lods: vec![prism_render_architecture::geometry::GeometryLodRecord {
                    resident: true,
                    fallback: true,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        assert_eq!(geometry_lods(&scene, &handles, &geometries)[&geometry].lods.len(), 1);

        let materials = RenderMaterialRegistry::default();
        let table = material_records(&materials, &scene, &handles);
        assert_eq!(
            table[&stale_material].render_class,
            prism_render_material::MaterialRenderClass::Opaque
        );
        let work = prism_render_visibility::GpuRenderWorkItem {
            scene: handle,
            geometry,
            material: stale_material,
            lod_or_cluster: 0,
            pass_mask: prism_render_visibility::RenderPassMask::OPAQUE,
            sort_key: prism_render_visibility::WorkSortKey::default(),
        };
        let candidate = draw_bin_candidate(&work, &geometries, &table).unwrap();
        assert_eq!(candidate.key.geometry, geometry);
        assert!(candidate.key.indexed);
    }
}
