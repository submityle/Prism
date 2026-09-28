use crate::*;
use alloc::collections::{BTreeMap, BTreeSet};
use prism_render_architecture::{abi::GenerationalHandle, gpu_scene::*};
use prism_render_material::*;

fn view() -> GpuViewRecord {
    GpuViewRecord {
        handle: GenerationalHandle {
            index: 1,
            generation: 1,
        },
        clip_from_world: [[0.0; 4]; 4],
        previous_clip_from_world: [[0.0; 4]; 4],
        world_position: [0.0, 0.0, -10.0],
        lod_scale: 1.0,
        viewport: [0, 0, 1920, 1080],
        frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
        layer_mask: 1,
        flags: ViewFlags::REVERSE_Z,
        history_epoch: 1,
    }
}
fn material(handle: GenerationalHandle) -> MaterialRecord {
    MaterialRecord {
        handle,
        revision: 1,
        domain: MaterialDomain::Surface,
        render_class: MaterialRenderClass::Opaque,
        illumination: Illumination::Lit,
        features: Default::default(),
        closure_mask: 1,
        surface: Default::default(),
        textures: Vec::new(),
        custom_program: None,
    }
}

#[test]
fn culling_validates_scene_lod_material_and_occlusion() {
    let handle = GenerationalHandle {
        index: 1,
        generation: 1,
    };
    let geometry = GenerationalHandle {
        index: 1,
        generation: 1,
    };
    let material_handle = GenerationalHandle {
        index: 1,
        generation: 1,
    };
    let mut scene = CpuRenderScene::default();
    scene.apply(&SceneTransaction {
        frame_epoch: 1,
        sequence: 1,
        producer: 1,
        operations: vec![SceneOperation::Create {
            handle,
            record: InstanceRecord {
                bounds: SceneBounds {
                    center: [0.0; 3],
                    radius: 1.0,
                    half_extents: [1.0; 3],
                    _padding: 0.0,
                },
                geometry,
                material: material_handle,
                render_layers: 1,
                ..Default::default()
            },
        }],
    });
    let geometry_map = BTreeMap::from([(
        geometry,
        GeometryLodChain {
            geometry,
            lods: vec![GeometryLod {
                level: 0,
                screen_error: 0.01,
                resident: true,
                fallback: true,
            }],
        },
    )]);
    let materials = BTreeMap::from([(material_handle, material(material_handle))]);
    let previous = BTreeMap::new();
    let occluded = BTreeSet::new();
    let (work, stats) = cull_view(
        &view(),
        VisibilityInput {
            scene: &scene,
            handles: &[handle],
            geometry: &geometry_map,
            materials: &materials,
            previous_lods: &previous,
            occluded: &occluded,
            capacity: 8,
            previous_history_epoch: Some(1),
        },
    );
    assert_eq!(work.len(), 1);
    assert_eq!(stats.visible_instances, 1);
    assert_eq!(work[0].scene, handle);
    assert_ne!(work[0].pass_mask.0 & RenderPassMask::SHADOW.0, 0);
}

#[test]
fn camera_cut_resets_occlusion_history() {
    let mut cut = view();
    cut.flags |= ViewFlags::CAMERA_CUT;
    assert_eq!(cut.history_policy(Some(1)), HistoryPolicy::Reset);
    assert_eq!(view().history_policy(Some(1)), HistoryPolicy::Reuse);
}

#[test]
fn visibility_frame_sorts_work_and_publishes_ranges() {
    let mut frame = VisibilityFrame::default();
    let view = GenerationalHandle {
        index: 1,
        generation: 1,
    };
    let item = |key| GpuRenderWorkItem {
        scene: GenerationalHandle {
            index: key,
            generation: 1,
        },
        geometry: GenerationalHandle {
            index: 1,
            generation: 1,
        },
        material: GenerationalHandle {
            index: 1,
            generation: 1,
        },
        lod_or_cluster: 0,
        pass_mask: RenderPassMask::OPAQUE,
        visibility_stages: VisibilityStageMask::EARLY,
        sort_key: WorkSortKey(key as u64),
    };
    frame.push_view(
        view,
        vec![item(9), item(2)],
        VisibilityDiagnostics {
            material_bins: 1,
            ..Default::default()
        },
    );
    assert_eq!(frame.work_items[0].sort_key, WorkSortKey(2));
    assert_eq!(frame.views[&view].visible_instances.count, 2);
}
