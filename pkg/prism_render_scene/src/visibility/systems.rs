use alloc::collections::BTreeMap;

use bevy_camera::{primitives::Frustum, visibility::RenderLayers};
use bevy_ecs::prelude::*;
use bevy_render::{
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};
use prism_render_visibility::{
    cull_view, GeometryLod, GeometryLodChain, GpuViewRecord, ViewFlags, VisibilityFrame,
    VisibilityInput,
};

use crate::{
    material::runtime::RenderMaterialRegistry,
    scene::RenderGpuScene,
    visibility::{
        buffers::UnifiedVisibilityBuffers,
        rows::{RenderVisibilityRange, RenderVisibilityView, RenderVisibilityWorkItem},
        runtime::{
            PrismVisibilityDiagnostics, UnifiedVisibilityEnabled, UnifiedVisibilitySettings,
            UnifiedVisibilityState, VisibilityFrameGraph,
        },
    },
};

pub(crate) fn build_unified_visibility(
    enabled: Res<UnifiedVisibilityEnabled>,
    settings: Res<UnifiedVisibilitySettings>,
    scene: Res<RenderGpuScene>,
    materials: Res<RenderMaterialRegistry>,
    mut state: ResMut<UnifiedVisibilityState>,
    mut buffers: ResMut<UnifiedVisibilityBuffers>,
    mut diagnostics: ResMut<PrismVisibilityDiagnostics>,
    views: Query<(&ExtractedView, Option<&Frustum>, Option<&RenderLayers>)>,
    frame_graph: Res<VisibilityFrameGraph>,
) {
    state.views.clear();
    state.frame = VisibilityFrame::default();
    *diagnostics = PrismVisibilityDiagnostics {
        buffer_version: buffers.version(),
        ..Default::default()
    };
    if !enabled.0 {
        buffers.stage([], [], []);
        return;
    }
    debug_assert!(!frame_graph.compiled.execution_order.is_empty());

    let handles = scene.mirror().live_handles();
    let geometry = geometry_lods(scene.mirror(), &handles);
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
        state.views.push(record);
        state.remember(retained, clip_array, position, record.history_epoch);
    }

    diagnostics.views = state.views.len() as u32;
    diagnostics.work_items = state.frame.work_items.len() as u32;
    diagnostics.cpu_reference_frames = 1;
    state.commit_lods();
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
    );
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

fn geometry_lods(
    scene: &prism_render_architecture::gpu_scene::CpuRenderScene,
    handles: &[prism_render_architecture::gpu_scene::SceneHandle],
) -> BTreeMap<prism_render_architecture::gpu_scene::GeometryHandle, GeometryLodChain> {
    handles
        .iter()
        .filter_map(|handle| scene.get(*handle).map(|instance| instance.geometry))
        .map(|geometry| {
            (
                geometry,
                GeometryLodChain {
                    geometry,
                    lods: vec![GeometryLod {
                        level: 0,
                        screen_error: 0.0,
                        resident: true,
                        fallback: true,
                    }],
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
        assert!(geometry_lods(&scene, &handles).contains_key(&geometry));

        let materials = RenderMaterialRegistry::default();
        let table = material_records(&materials, &scene, &handles);
        assert_eq!(
            table[&stale_material].render_class,
            prism_render_material::MaterialRenderClass::Opaque
        );
    }
}
