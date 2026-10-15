use std::collections::{BTreeMap, BTreeSet};

use prism_app::{App, Plugin, PostUpdate};
use prism_ecs::{
    entity::Entity,
    schedule::{IntoSystemConfigs, SystemSet, SystemSetId},
    world::World,
};
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
                sync_scene.in_set(PrismVisibilitySystems::Sync),
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
    let mut query = world.query::<(Entity, &PrismCamera, &PrismFrustum, &GlobalTransform)>();
    let cameras: BTreeMap<_, _> = query
        .iter(world)
        .map(|(entity, camera, frustum, global)| (entity, (*camera, *frustum, *global)))
        .collect();
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
