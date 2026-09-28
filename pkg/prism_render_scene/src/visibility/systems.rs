use alloc::collections::BTreeMap;

use bevy_camera::{primitives::Frustum, visibility::RenderLayers};
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::{RenderContext, RenderDevice, RenderQueue},
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
    let gpu_slots_per_view =
        (handles.len() as u32).min(settings.gpu_parity_max_items_per_view);
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
        bins.command_buffer_start = (state.draw_bins.len() as u32)
            .saturating_mul(gpu_slots_per_view);
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
        gpu_slots_per_view,
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
            lod_or_cluster: lod.level,
            pipeline_class: ((material.illumination as u32) << 16)
                | material.render_class as u32,
            vertex_buffer_class: geometry.vertex_buffer_class,
            index_buffer_class: geometry.index_buffer_class,
            indexed: lod.primitive_kind
                == prism_render_architecture::geometry::GeometryPrimitiveKind::Indexed,
            primitive_kind: lod.primitive_kind,
            pass_mask: work.pass_mask.0,
        },
        visibility_stages: work.visibility_stages,
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
pub(crate) fn dispatch_unified_visibility_for_view(
    current_view: bevy_render::renderer::ViewQuery<&ExtractedView>,
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
    hzb_buffers: Res<super::hzb_gpu::HzbVisibilityBuffers>,
    buffers: Res<UnifiedVisibilityBuffers>,
    mut ctx: RenderContext,
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
    let retained = current_view.into_inner().retained_view_entity;
    let candidate_count = scene.mirror().capacity() as u32;
    let hzb_stage_start = hzb_buffers
        .view_range(retained)
        .map_or(0, |(start, _)| start);
    let Some(immediates) = visibility_dispatch_for_view(
        &state,
        &buffers,
        retained,
        candidate_count,
        settings.indirect_first_instance,
        hzb_stage_start,
    ) else {
        return;
    };
    let mut pass = ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism unified visibility"),
        timestamp_writes: None,
    });
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, scene_bind_group, &[]);
    pass.set_bind_group(1, material_bind_group, &[]);
    pass.set_bind_group(2, geometry_bind_group, &[]);
    pass.set_bind_group(3, output_bind_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&immediates));
    pass.dispatch_workgroups(candidate_count.div_ceil(VISIBILITY_WORKGROUP_SIZE), 1, 1);
    diagnostics.gpu_compute_dispatches += 1;
    diagnostics.gpu_candidates += candidate_count;
    diagnostics.indirect_identity_fallbacks +=
        u32::from(!settings.indirect_first_instance) * candidate_count;
}

fn visibility_dispatch_for_view(
    state: &UnifiedVisibilityState,
    buffers: &UnifiedVisibilityBuffers,
    retained: bevy_render::view::RetainedViewEntity,
    candidate_count: u32,
    indirect_first_instance: bool,
    hzb_stage_start: u32,
) -> Option<RenderVisibilityDispatch> {
    let view_index = state
        .views
        .iter()
        .position(|view| state.retained_view(view.handle) == Some(retained))?;
    let view_bins = state.draw_bins.get(view_index)?;
    let output_start = u32::try_from(view_index)
        .ok()?
        .saturating_mul(buffers.gpu_slots_per_view());
    Some(RenderVisibilityDispatch {
        view_index: u32::try_from(view_index).ok()?,
        candidate_count,
        output_start,
        output_end: output_start.saturating_add(buffers.gpu_slots_per_view()),
        indirect_first_instance: u32::from(indirect_first_instance),
        bin_start: view_bins.global_bin_start,
        candidate_bin_start: view_bins.global_candidate_start,
        hzb_stage_start,
    })
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
    use bevy_render::view::RetainedViewEntity;
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
            visibility_stages: prism_render_visibility::VisibilityStageMask::EARLY,
            sort_key: prism_render_visibility::WorkSortKey::default(),
        };
        let candidate = draw_bin_candidate(&work, &geometries, &table).unwrap();
        assert_eq!(candidate.key.geometry, geometry);
        assert!(candidate.key.indexed);
    }

    #[test]
    fn per_view_dispatch_uses_matching_multiview_partitions() {
        let mut world = World::new();
        let mut buffers = UnifiedVisibilityBuffers::from_world(&mut world);
        buffers.stage(
            [RenderVisibilityView::default(), RenderVisibilityView::default()],
            [],
            [],
            8,
        );
        let first_retained = RetainedViewEntity::new(
            bevy_render::sync_world::MainEntity::from(Entity::from_bits(1)),
            None,
            0,
        );
        let second_retained = RetainedViewEntity::new(
            bevy_render::sync_world::MainEntity::from(Entity::from_bits(2)),
            None,
            0,
        );
        let mut state = UnifiedVisibilityState::default();
        let first_handle = state.handle(first_retained);
        let second_handle = state.handle(second_retained);
        let test_view = |handle| GpuViewRecord {
            handle,
            clip_from_world: [[0.0; 4]; 4],
            previous_clip_from_world: [[0.0; 4]; 4],
            world_position: [0.0; 3],
            lod_scale: 1.0,
            viewport: [0, 0, 1, 1],
            frustum_planes: [[0.0; 4]; 6],
            layer_mask: 1,
            flags: ViewFlags::REVERSE_Z,
            history_epoch: 1,
        };
        state.views = vec![test_view(first_handle), test_view(second_handle)];
        state.draw_bins = vec![
            prism_render_visibility::ViewDrawBins {
                global_bin_start: 3,
                global_candidate_start: 40,
                ..Default::default()
            },
            prism_render_visibility::ViewDrawBins {
                global_bin_start: 7,
                global_candidate_start: 80,
                ..Default::default()
            },
        ];
        let dispatch = visibility_dispatch_for_view(
            &state,
            &buffers,
            second_retained,
            64,
            true,
            128,
        )
        .unwrap();

        assert_eq!(dispatch.view_index, 1);
        assert_eq!(dispatch.output_start, 8);
        assert_eq!(dispatch.output_end, 16);
        assert_eq!(dispatch.bin_start, 7);
        assert_eq!(dispatch.candidate_bin_start, 80);
        assert_eq!(dispatch.indirect_first_instance, 1);
        assert_eq!(dispatch.hzb_stage_start, 128);
    }
}
