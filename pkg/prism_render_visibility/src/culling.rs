use crate::{
    GeometryLodChain, GpuRenderWorkItem, GpuViewRecord, RenderPassMask, VisibilityDiagnostics,
    VisibilityStageMask, WorkSortKey,
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use prism_render_architecture::gpu_scene::{CpuRenderScene, SceneHandle};
use prism_render_material::{MaterialRecord, MaterialRenderClass};

pub struct VisibilityInput<'a> {
    pub scene: &'a CpuRenderScene,
    pub handles: &'a [SceneHandle],
    pub geometry:
        &'a BTreeMap<prism_render_architecture::gpu_scene::GeometryHandle, GeometryLodChain>,
    pub materials:
        &'a BTreeMap<prism_render_architecture::gpu_scene::SceneMaterialHandle, MaterialRecord>,
    pub previous_lods: &'a BTreeMap<(crate::ViewHandle, SceneHandle), u16>,
    pub occluded: &'a BTreeSet<(crate::ViewHandle, SceneHandle)>,
    pub capacity: u32,
    pub previous_history_epoch: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CullReason {
    StaleHandle,
    LayerMask,
    Frustum,
    Occluded,
    MissingGeometry,
    MissingMaterial,
    LodUnavailable,
    Overflow,
}

pub fn cull_view(
    view: &GpuViewRecord,
    input: VisibilityInput<'_>,
) -> (Vec<GpuRenderWorkItem>, VisibilityDiagnostics) {
    let mut work = Vec::new();
    let mut stats = VisibilityDiagnostics {
        input_instances: input.handles.len() as u32,
        ..Default::default()
    };
    let use_occlusion =
        view.history_policy(input.previous_history_epoch) == crate::HistoryPolicy::Reuse;
    for &handle in input.handles {
        let Some(instance) = input.scene.get(handle) else {
            stats.stale_handles += 1;
            continue;
        };
        if instance.render_layers & view.layer_mask == 0 {
            stats.layer_rejected += 1;
            continue;
        }
        if !sphere_visible(
            &view.frustum_planes,
            instance.bounds.center,
            instance.bounds.radius,
        ) {
            stats.frustum_rejected += 1;
            continue;
        }
        if use_occlusion && input.occluded.contains(&(view.handle, handle)) {
            stats.occlusion_rejected += 1;
            continue;
        }
        let Some(geometry) = input.geometry.get(&instance.geometry) else {
            stats.missing_geometry += 1;
            continue;
        };
        let Some(material) = input.materials.get(&instance.material) else {
            stats.missing_material += 1;
            continue;
        };
        let distance = squared_distance(view.world_position, instance.bounds.center)
            .sqrt()
            .max(1.0e-4);
        let projected_radius = instance.bounds.radius / distance;
        let Some(lod) = geometry.select(
            projected_radius,
            view.lod_scale,
            input.previous_lods.get(&(view.handle, handle)).copied(),
        ) else {
            stats.missing_geometry += 1;
            continue;
        };
        if work.len() >= input.capacity as usize {
            stats.overflowed = true;
            break;
        }
        if lod.used_fallback {
            stats.lod_fallbacks += 1;
        }
        let pass_mask = pass_mask(material.render_class, instance.flags, view.flags);
        stats.shadow_casters += u32::from(pass_mask.0 & RenderPassMask::SHADOW.0 != 0);
        stats.ray_scene_instances += u32::from(pass_mask.0 & RenderPassMask::RAY_SCENE.0 != 0);
        work.push(GpuRenderWorkItem {
            scene: handle,
            geometry: instance.geometry,
            material: instance.material,
            lod_or_cluster: lod.level as u32,
            pass_mask,
            visibility_stages: VisibilityStageMask::EARLY,
            sort_key: WorkSortKey::new(
                pass_class(material.render_class),
                material.illumination as u8,
                material.render_class as u8,
                instance.geometry.index as u16,
                depth_bucket(distance),
            ),
        });
    }
    stats.visible_instances = work.len() as u32;
    stats.material_bins = work
        .iter()
        .map(|item| item.material)
        .collect::<BTreeSet<_>>()
        .len() as u32;
    (work, stats)
}

fn sphere_visible(planes: &[[f32; 4]; 6], center: [f32; 3], radius: f32) -> bool {
    planes
        .iter()
        .all(|p| p[0] * center[0] + p[1] * center[1] + p[2] * center[2] + p[3] >= -radius)
}
fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}
fn depth_bucket(distance: f32) -> u16 {
    distance.min(u16::MAX as f32) as u16
}
fn pass_class(class: MaterialRenderClass) -> u8 {
    match class {
        MaterialRenderClass::Opaque | MaterialRenderClass::OpaqueTwoSided => 0,
        MaterialRenderClass::Masked | MaterialRenderClass::MaskedTwoSided => 1,
        _ => 2,
    }
}
fn pass_mask(class: MaterialRenderClass, flags: u32, view: crate::ViewFlags) -> RenderPassMask {
    let mut mask = match pass_class(class) {
        0 => RenderPassMask::OPAQUE,
        1 => RenderPassMask::MASKED,
        _ => RenderPassMask::TRANSPARENT,
    };
    if flags & 1 == 0 {
        mask |= RenderPassMask::SHADOW;
    }
    mask |= RenderPassMask::GI | RenderPassMask::RAY_SCENE | RenderPassMask::PICKING;
    if view.contains(crate::ViewFlags::OFFLINE) {
        mask |= RenderPassMask::OFFLINE;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GeometryLod, HistoryPolicy, ViewFlags};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        CpuRenderScene, InstanceRecord, SceneBounds, SceneOperation, SceneTransaction,
    };
    use prism_render_material::{
        Illumination, MaterialDomain, MaterialRecord, MaterialRenderClass,
    };

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    fn view() -> GpuViewRecord {
        GpuViewRecord {
            handle: handle(1),
            clip_from_world: [[0.0; 4]; 4],
            previous_clip_from_world: [[0.0; 4]; 4],
            world_position: [0.0, 0.0, -10.0],
            lod_scale: 1.0,
            viewport: [0, 0, 1920, 1080],
            // All planes evaluate to `1.0 >= -radius`, i.e. always inside.
            frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            layer_mask: 1,
            flags: ViewFlags::REVERSE_Z,
            history_epoch: 1,
        }
    }

    fn instance(geometry: GenerationalHandle, material: GenerationalHandle) -> InstanceRecord {
        InstanceRecord {
            bounds: SceneBounds {
                center: [0.0; 3],
                radius: 1.0,
                half_extents: [1.0; 3],
                _padding: 0.0,
            },
            geometry,
            material,
            render_layers: 1,
            ..Default::default()
        }
    }

    fn scene_with(scene_handle: GenerationalHandle, record: InstanceRecord) -> CpuRenderScene {
        let mut scene = CpuRenderScene::default();
        scene.apply(&SceneTransaction {
            frame_epoch: 1,
            sequence: 1,
            producer: 1,
            operations: vec![SceneOperation::Create {
                handle: scene_handle,
                record,
            }],
        });
        scene
    }

    fn material(material_handle: GenerationalHandle) -> MaterialRecord {
        MaterialRecord {
            handle: material_handle,
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

    fn lod_chain(geometry: GenerationalHandle, resident: bool, fallback: bool) -> GeometryLodChain {
        GeometryLodChain {
            geometry,
            lods: vec![GeometryLod {
                level: 0,
                screen_error: 0.01,
                resident,
                fallback,
            }],
        }
    }

    #[test]
    fn stale_layer_and_frustum_rejections_are_counted_separately() {
        let geometry = handle(1);
        let material_handle = handle(1);
        let materials = BTreeMap::from([(material_handle, material(material_handle))]);
        let geometry_map = BTreeMap::from([(geometry, lod_chain(geometry, true, true))]);
        let previous = BTreeMap::new();
        let occluded = BTreeSet::new();

        // Stale: the handle is not present in the scene at all.
        let empty_scene = CpuRenderScene::default();
        let (work, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &empty_scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert!(work.is_empty());
        assert_eq!(stats.stale_handles, 1);
        assert_eq!(stats.input_instances, 1);

        // Layer mask: the instance shares no bit with the view layer mask.
        let mut rec = instance(geometry, material_handle);
        rec.render_layers = 0b10;
        let scene = scene_with(handle(1), rec);
        let (_, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert_eq!(stats.layer_rejected, 1);
        assert_eq!(stats.visible_instances, 0);

        // Frustum: place the center far outside a half-space plane.
        let mut culling_view = view();
        culling_view.frustum_planes[0] = [1.0, 0.0, 0.0, 0.0];
        let mut rec = instance(geometry, material_handle);
        rec.bounds.center = [-100.0, 0.0, 0.0];
        let scene = scene_with(handle(1), rec);
        let (_, stats) = cull_view(
            &culling_view,
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert_eq!(stats.frustum_rejected, 1);
    }

    #[test]
    fn occlusion_is_only_applied_when_history_is_reused() {
        let geometry = handle(1);
        let material_handle = handle(1);
        let scene = scene_with(handle(1), instance(geometry, material_handle));
        let geometry_map = BTreeMap::from([(geometry, lod_chain(geometry, true, true))]);
        let materials = BTreeMap::from([(material_handle, material(material_handle))]);
        let previous = BTreeMap::new();
        let occluded = BTreeSet::from([(view().handle, handle(1))]);

        // Reused history (matching epoch, no camera cut) honours occlusion.
        assert_eq!(view().history_policy(Some(1)), HistoryPolicy::Reuse);
        let (work, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert!(work.is_empty());
        assert_eq!(stats.occlusion_rejected, 1);

        // A reset history (missing previous epoch) ignores the occlusion set.
        let (work, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: None,
            },
        );
        assert_eq!(stats.occlusion_rejected, 0);
        assert_eq!(work.len(), 1);
    }

    #[test]
    fn missing_geometry_material_and_unavailable_lod_are_handled() {
        let geometry = handle(1);
        let material_handle = handle(1);
        let scene = scene_with(handle(1), instance(geometry, material_handle));
        let previous = BTreeMap::new();
        let occluded = BTreeSet::new();

        // Missing geometry entry.
        let empty_geometry = BTreeMap::new();
        let materials = BTreeMap::from([(material_handle, material(material_handle))]);
        let (_, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &empty_geometry,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert_eq!(stats.missing_geometry, 1);

        // Missing material entry.
        let geometry_map = BTreeMap::from([(geometry, lod_chain(geometry, true, true))]);
        let empty_materials = BTreeMap::new();
        let (_, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &empty_materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert_eq!(stats.missing_material, 1);

        // Unavailable LOD: no resident level and no fallback -> select fails
        // and the instance is charged to the missing-geometry counter.
        let no_lod = BTreeMap::from([(geometry, lod_chain(geometry, false, false))]);
        let (_, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &no_lod,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert_eq!(stats.missing_geometry, 1);
    }

    #[test]
    fn capacity_limit_sets_overflow_and_stops_emitting_work() {
        let geometry = handle(1);
        let material_handle = handle(1);
        let scene = scene_with(handle(1), instance(geometry, material_handle));
        let geometry_map = BTreeMap::from([(geometry, lod_chain(geometry, true, true))]);
        let materials = BTreeMap::from([(material_handle, material(material_handle))]);
        let previous = BTreeMap::new();
        let occluded = BTreeSet::new();

        let (work, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 0,
                previous_history_epoch: Some(1),
            },
        );
        assert!(work.is_empty());
        assert!(stats.overflowed);
        assert_eq!(stats.visible_instances, 0);
    }

    #[test]
    fn visible_instance_emits_offline_pass_only_when_view_is_offline() {
        let geometry = handle(1);
        let material_handle = handle(1);
        let scene = scene_with(handle(1), instance(geometry, material_handle));
        let geometry_map = BTreeMap::from([(geometry, lod_chain(geometry, true, true))]);
        let materials = BTreeMap::from([(material_handle, material(material_handle))]);
        let previous = BTreeMap::new();
        let occluded = BTreeSet::new();

        // Online view: no OFFLINE bit, but opaque + shadow are present.
        let (work, stats) = cull_view(
            &view(),
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
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
        assert_eq!(work[0].pass_mask.0 & RenderPassMask::OFFLINE.0, 0);
        assert_ne!(work[0].pass_mask.0 & RenderPassMask::OPAQUE.0, 0);
        assert_ne!(work[0].pass_mask.0 & RenderPassMask::SHADOW.0, 0);
        assert_eq!(stats.shadow_casters, 1);
        assert_eq!(stats.material_bins, 1);

        // Offline view: the OFFLINE pass is added to every emitted work item.
        let mut offline = view();
        offline.flags |= ViewFlags::OFFLINE;
        let (work, _) = cull_view(
            &offline,
            VisibilityInput {
                scene: &scene,
                handles: &[handle(1)],
                geometry: &geometry_map,
                materials: &materials,
                previous_lods: &previous,
                occluded: &occluded,
                capacity: 8,
                previous_history_epoch: Some(1),
            },
        );
        assert_ne!(work[0].pass_mask.0 & RenderPassMask::OFFLINE.0, 0);
    }

    #[test]
    fn cull_reason_variants_are_distinct_and_copyable() {
        let reasons = [
            CullReason::StaleHandle,
            CullReason::LayerMask,
            CullReason::Frustum,
            CullReason::Occluded,
            CullReason::MissingGeometry,
            CullReason::MissingMaterial,
            CullReason::LodUnavailable,
            CullReason::Overflow,
        ];
        // Copy semantics: comparing a value with its copy holds.
        let copied = reasons[0];
        assert_eq!(copied, CullReason::StaleHandle);
        // Every listed variant is pairwise distinct.
        for (i, a) in reasons.iter().enumerate() {
            for (j, b) in reasons.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }
}
