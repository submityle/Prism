use std::collections::{BTreeMap, BTreeSet};

use prism_app::{App, Plugin, PostUpdate};
use prism_camera_ecs::{CameraSystemSet, PublishedViews};
use prism_ecs::{
    entity::Entity,
    schedule::{IntoSystemConfigs, SystemSet, SystemSetId},
    world::World,
};
use prism_math::{Affine3, Frustum, Mat4, Plane, Vec3};
use prism_render_architecture::gpu_scene::{InstanceRecord, SceneOperation, SceneTransaction};
use prism_render_visibility::{cull_view, ViewFlags, VisibilityInput};
use prism_transform::{
    ecs::{propagate_transforms, LocalBounds},
    GlobalTransform,
};

use super::{
    lower::{instance_record, view_record},
    PrismCamera, PrismFrustum, PrismRenderScene, PrismRenderable, PrismViewResult,
    PrismViewVisibility,
};

/// Ordering anchors for main-world visibility in `PostUpdate`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PrismVisibilitySystems {
    /// Propagate native transforms and reconcile complete renderable membership.
    Sync,
    /// Cull complete cameras, update temporal state and write the visibility union.
    Cull,
}
impl SystemSet for PrismVisibilitySystems {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::with::<Self>(*self as u64)
    }
}

/// CPU-only native replacement for the Bevy visibility bridge. Preserves an
/// already configured scene resource. No render plugin or device is required.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrismVisibilityPlugin;
impl Plugin for PrismVisibilityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PrismRenderScene>().add_systems(
            PostUpdate,
            (
                sync_scene
                    .in_set(PrismVisibilitySystems::Sync)
                    .after(CameraSystemSet::Publication),
                cull_views.in_set(PrismVisibilitySystems::Cull),
            )
                .chain(),
        );
    }
}

/// Propagate transforms, then reconcile the complete required-component set.
/// Explicit comparison also detects removals without depending on event readers.
/// A propagation error publishes an empty scene rather than culling stale poses.
pub fn sync_scene(world: &mut World) {
    let error = propagate_transforms(world).err();
    let mut records = BTreeMap::<Entity, InstanceRecord>::new();
    if error.is_none() {
        let mut query = world.query::<(Entity, &PrismRenderable, &LocalBounds, &GlobalTransform)>();
        records.extend(
            query
                .iter(world)
                .map(|(entity, renderable, bounds, global)| {
                    (entity, instance_record(renderable, bounds, global))
                }),
        );
    }
    let mut scene = world
        .remove_resource::<PrismRenderScene>()
        .expect("visibility scene installed");
    scene.transform_error = error;
    scene.frame = scene
        .frame
        .checked_add(1)
        .expect("visibility frame exhausted");
    let mut transaction = SceneTransaction {
        frame_epoch: scene.frame,
        sequence: scene.frame,
        ..Default::default()
    };
    let removed: Vec<_> = scene
        .entities
        .iter()
        .filter(|(entity, _)| !records.contains_key(entity))
        .map(|(&entity, &handle)| (entity, handle))
        .collect();
    for (entity, handle) in removed {
        transaction
            .operations
            .push(SceneOperation::Destroy { handle });
        scene.entities.remove(&entity);
        scene.owners.remove(&handle);
        scene.forget_instance(handle);
        scene.retire(handle);
    }
    for (entity, record) in records {
        let Some(handle) = scene.handle_for(entity) else {
            let handle = scene.allocate();
            transaction
                .operations
                .push(SceneOperation::Create { handle, record });
            scene.entities.insert(entity, handle);
            scene.owners.insert(handle, entity);
            continue;
        };
        let old = *scene.scene.get(handle).expect("mapped instance is live");
        let ops = &mut transaction.operations;
        // One settling update advances previous to current after motion stops.
        if old.current_transform != record.current_transform
            || old.previous_transform != old.current_transform
        {
            ops.push(SceneOperation::SetTransform {
                handle,
                current: record.current_transform,
            });
        }
        if old.bounds != record.bounds {
            ops.push(SceneOperation::SetBounds {
                handle,
                bounds: record.bounds,
            });
        }
        if old.geometry != record.geometry {
            ops.push(SceneOperation::SetGeometry {
                handle,
                geometry: record.geometry,
            });
            scene.forget_instance(handle);
        }
        if old.material != record.material {
            ops.push(SceneOperation::SetMaterial {
                handle,
                material: record.material,
            });
        }
        if old.render_layers != record.render_layers {
            ops.push(SceneOperation::SetRenderLayers {
                handle,
                render_layers: record.render_layers,
            });
        }
        if old.flags != record.flags {
            ops.push(SceneOperation::SetFlags {
                handle,
                mask: u32::MAX,
                value: record.flags,
            });
        }
    }
    scene.report = scene.scene.apply(&transaction);
    assert!(
        scene.report.errors.is_empty(),
        "native visibility transaction rejected: {:?}",
        scene.report.errors
    );
    scene.transaction = transaction;
    world.insert_resource(scene);
}

/// Run real core culling per complete camera, retain only current LOD pairs,
/// and write boolean changes in the main world. Call after [`sync_scene`].
pub fn cull_views(world: &mut World) {
    // Published camera snapshots are the canonical input when CameraPlugin is
    // installed. The legacy native components remain an independent renderer
    // input only for applications which do not install the camera runtime.
    let cameras: BTreeMap<_, _> = if let Some(published) = world.get_resource::<PublishedViews>() {
        published
            .iter()
            .filter_map(|published| {
                let view = published.data;
                let origin = view.world_origin.as_vec3();
                let absolute_from_relative = Mat4::from_translation(-origin);
                let clip_from_world = view.clip_from_world * absolute_from_relative;
                let mut planes = [Plane::new(Vec3::ZERO, 1.0); 6];
                for (index, half_space) in view.frustum.planes[..view.frustum.count as usize]
                    .iter()
                    .enumerate()
                {
                    planes[index] = Plane::new(
                        half_space.normal,
                        half_space.d - half_space.normal.dot(origin),
                    );
                }
                let mut flags = ViewFlags::default();
                if view.teleport || !view.history_valid {
                    flags |= ViewFlags::CAMERA_CUT;
                }
                let camera = PrismCamera {
                    clip_from_world,
                    viewport: [
                        view.viewport_px.origin[0],
                        view.viewport_px.origin[1],
                        view.viewport_px.size[0],
                        view.viewport_px.size[1],
                    ],
                    lod_scale: view.lod_scale,
                    layer_mask: view.layer_mask,
                    flags,
                    history_epoch: view.history_epoch,
                };
                let frustum = PrismFrustum(Frustum::from_planes(planes));
                let mut world_from_view = view.world_from_view;
                world_from_view.translation += origin;
                let global = GlobalTransform(world_from_view);
                Some((published.entity, (camera, frustum, global)))
            })
            .collect()
    } else {
        let mut query = world.query::<(Entity, &PrismCamera, &PrismFrustum, &GlobalTransform)>();
        query
            .iter(world)
            .map(|(entity, camera, frustum, global)| (entity, (*camera, *frustum, *global)))
            .collect()
    };
    let mut scene = world
        .remove_resource::<PrismRenderScene>()
        .expect("visibility scene installed");
    let cameras = if scene.transform_error.is_some() {
        BTreeMap::new()
    } else {
        cameras
    };
    scene.views.retain(|entity, _| cameras.contains_key(entity));
    let live_views: BTreeSet<_> = scene
        .views
        .values()
        .map(|result| result.view.handle)
        .collect();
    scene
        .previous_lods
        .retain(|&(view, _), _| live_views.contains(&view));
    let handles: Vec<_> = scene.owners.keys().copied().collect();
    let occluded = BTreeSet::new();
    let mut visible = BTreeSet::new();
    for (entity, (camera, frustum, global)) in cameras {
        let previous = scene.views.remove(&entity);
        let (handle, previous_epoch, mut previous_clip) = match &previous {
            Some(result) => (
                result.view.handle,
                Some(result.view.history_epoch),
                result.view.clip_from_world,
            ),
            None => {
                let handle = scene.allocate_view();
                let view = view_record(handle, &camera, &frustum, &global, camera.clip_from_world);
                (handle, None, view.clip_from_world)
            }
        };
        let reset = previous_epoch != Some(camera.history_epoch)
            || camera.flags.contains(ViewFlags::CAMERA_CUT);
        let mut view = view_record(handle, &camera, &frustum, &global, camera.clip_from_world);
        if reset {
            scene.previous_lods.retain(|&(view, _), _| view != handle);
            previous_clip = view.clip_from_world;
        }
        view.previous_clip_from_world = previous_clip;
        let (work, diagnostics) = cull_view(
            &view,
            VisibilityInput {
                scene: &scene.scene,
                handles: &handles,
                geometry: &scene.geometry,
                materials: &scene.materials,
                previous_lods: &scene.previous_lods,
                occluded: &occluded,
                capacity: scene.capacity,
                previous_history_epoch: previous_epoch,
            },
        );
        // Replace, rather than accumulate, so culled/removed pairs cannot linger.
        scene.previous_lods.retain(|&(view, _), _| view != handle);
        for item in &work {
            visible.insert(item.scene);
            scene
                .previous_lods
                .insert((handle, item.scene), item.lod_or_cluster as u16);
        }
        scene.views.insert(
            entity,
            PrismViewResult {
                view,
                previous_history_epoch: previous_epoch,
                work,
                diagnostics,
            },
        );
    }
    let mut query = world.query::<(Entity, &PrismViewVisibility)>();
    let flips: Vec<_> = query
        .iter(world)
        .filter_map(|(entity, old)| {
            let visible = scene
                .handle_for(entity)
                .is_some_and(|handle| visible.contains(&handle));
            (old.visible != visible).then_some((entity, visible))
        })
        .collect();
    world.insert_resource(scene);
    for (entity, visible) in flips {
        if let Some(mut target) = world.get_mut::<PrismViewVisibility>(entity) {
            target.visible = visible;
        }
    }
}

#[cfg(test)]
mod camera_publication_tests {
    use super::*;
    use crate::visibility::{GeometryLod, GeometryLodChain};
    use prism_app::App;
    use prism_camera::{CameraIntent, CameraPose};
    use prism_camera_ecs::{
        CameraBundle, CameraIntentComponent, CameraPlugin, CameraPoseComponent, PublishedViews,
    };
    use prism_math::{Aabb3, Vec3};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
    use prism_render_material::fallback_material_record;

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle::new(index, 1)
    }

    fn spawn_instance(app: &mut App, material: SceneMaterialHandle, layer: u32) -> Entity {
        app.world_mut().spawn((
            PrismRenderable {
                geometry: handle(1),
                material,
                render_layers: layer,
                flags: 0,
            },
            LocalBounds(Aabb3::from_center_half_extents(
                Vec3::new(0.0, 0.0, -5.0),
                Vec3::splat(0.5),
            )),
            GlobalTransform::IDENTITY,
            PrismViewVisibility::default(),
        ))
    }

    #[test]
    fn camera_plugin_publication_drives_independent_visibility_views() {
        let mut app = App::new();
        app.add_plugins((CameraPlugin::new([640, 480]), PrismVisibilityPlugin));
        {
            let scene = app.world_mut().resource_mut::<PrismRenderScene>();
            let geometry: GeometryHandle = handle(1);
            let first_material: SceneMaterialHandle = handle(1);
            let second_material: SceneMaterialHandle = handle(2);
            scene.insert_geometry(GeometryLodChain {
                geometry,
                lods: vec![GeometryLod {
                    level: 0,
                    screen_error: 0.0,
                    resident: true,
                    fallback: true,
                }],
            });
            scene.insert_material(fallback_material_record(first_material, 1));
            scene.insert_material(fallback_material_record(second_material, 1));
        }

        let mut first_intent = CameraIntent::default();
        first_intent.layer_mask = 1;
        first_intent.lod_scale = 0.5;
        let first_origin = prism_math::DVec3::new(10_000.0, 0.0, 0.0);
        let first = app.world_mut().spawn(CameraBundle {
            intent: CameraIntentComponent(first_intent),
            pose: CameraPoseComponent(CameraPose {
                position: first_origin,
                world_origin: first_origin,
                ..CameraPose::default()
            }),
            ..Default::default()
        });
        let mut second_intent = CameraIntent::default();
        second_intent.layer_mask = 2;
        second_intent.lod_scale = 2.0;
        let second = app.world_mut().spawn(CameraBundle {
            intent: CameraIntentComponent(second_intent),
            pose: CameraPoseComponent(CameraPose::default()),
            ..Default::default()
        });
        let first_instance = spawn_instance(&mut app, handle(1), 1);
        let second_instance = spawn_instance(&mut app, handle(2), 2);
        app.world_mut()
            .get_mut::<LocalBounds>(first_instance)
            .unwrap()
            .0 = Aabb3::from_center_half_extents(Vec3::new(10_000.0, 0.0, -5.0), Vec3::splat(0.5));

        app.update();

        let (expected_matrix, expected_epoch) = {
            let published = app.world().resource::<PublishedViews>();
            assert_eq!(published.len(), 2);
            let first_view = published.iter().find(|view| view.entity == first).unwrap();
            let second_view = published.iter().find(|view| view.entity == second).unwrap();
            assert_eq!(first_view.data.layer_mask, 1);
            assert_eq!(second_view.data.layer_mask, 2);
            assert_eq!(first_view.data.lod_scale, 0.5);
            assert_eq!(second_view.data.lod_scale, 2.0);
            let clip_from_absolute_world =
                first_view.data.clip_from_world * Mat4::from_translation(-first_origin.as_vec3());

            (
                [
                    clip_from_absolute_world.x_axis.to_array(),
                    clip_from_absolute_world.y_axis.to_array(),
                    clip_from_absolute_world.z_axis.to_array(),
                    clip_from_absolute_world.w_axis.to_array(),
                ],
                first_view.data.history_epoch,
            )
        };

        let mut render_world = World::new();
        crate::surface_views::extract_surface_views(app.world_mut(), &mut render_world);
        let surfaces = render_world.resource::<crate::surface_views::ExtractedSurfaceViews>();
        let first_surface = surfaces.get(first).expect("published camera surface view");
        let second_surface = surfaces.get(second).expect("second camera surface view");
        assert_eq!(first_surface.viewport, [0, 0, 640, 480]);
        assert_eq!(second_surface.viewport, [0, 0, 640, 480]);
        assert_eq!(first_surface.history_epoch, expected_epoch);
        assert_eq!(first_surface.clip_from_world, expected_matrix);

        let scene = app.world().resource::<PrismRenderScene>();
        let first_scene = scene.handle_for(first_instance).unwrap();
        let second_scene = scene.handle_for(second_instance).unwrap();
        let first_result = scene
            .view_results()
            .find(|(camera, _)| *camera == first)
            .unwrap()
            .1;
        let second_result = scene
            .view_results()
            .find(|(camera, _)| *camera == second)
            .unwrap()
            .1;
        assert_ne!(first_result.view.handle, second_result.view.handle);
        assert_eq!(first_result.view.layer_mask, 1);
        assert_eq!(second_result.view.layer_mask, 2);
        assert_eq!(first_result.view.lod_scale, 0.5);
        assert_eq!(second_result.view.lod_scale, 2.0);
        assert_eq!(first_result.view.clip_from_world, expected_matrix);
        assert_eq!(first_result.work.len(), 1);
        assert_eq!(second_result.work.len(), 1);
        assert_eq!(first_result.work[0].scene, first_scene);
        assert_eq!(second_result.work[0].scene, second_scene);
    }
}
