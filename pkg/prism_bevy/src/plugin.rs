//! The Bevy plugin and systems that keep [`PrismRenderScene`] in sync with the
//! ECS world and run Prism's visibility culling each frame.

use alloc::collections::BTreeSet;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_camera::primitives::{Aabb, Frustum};
use bevy_transform::components::GlobalTransform;

use prism_render_architecture::gpu_scene::{SceneHandle, SceneOperation, SceneTransactionBuilder};
use prism_render_visibility::bevy_bridge::{
    BevyViewParams, instance_record, scene_bounds, scene_transform, view_record,
};
use prism_render_visibility::{VisibilityInput, cull_view};

use crate::components::{PrismCamera, PrismRenderable, PrismViewVisibility};
use crate::scene::{PrismRenderScene, ViewState};

/// System sets for the Prism visibility pipeline, run in order each frame.
#[derive(SystemSet, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PrismVisibilitySystems {
    /// Mirrors tagged entities into [`PrismRenderScene`].
    Sync,
    /// Runs `cull_view` per camera and writes [`PrismViewVisibility`].
    Cull,
}

/// Adds GPU-driven visibility backed by Prism's render core to a Bevy app.
///
/// The plugin inserts a default [`PrismRenderScene`] resource (register backend
/// geometry/materials on it before spawning renderables) and schedules the
/// [`PrismVisibilitySystems::Sync`] and [`PrismVisibilitySystems::Cull`] sets in
/// [`PostUpdate`], after transform propagation has produced final world
/// transforms.
#[derive(Default, Clone, Copy, Debug)]
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

/// Mirrors every [`PrismRenderable`] entity into the Prism scene: new entities
/// are created, moved/resized entities get transform+bounds updates, and
/// despawned entities are destroyed.
fn sync_scene(
    mut scene: ResMut<PrismRenderScene>,
    renderables: Query<(
        Entity,
        Ref<GlobalTransform>,
        Ref<Aabb>,
        Ref<PrismRenderable>,
    )>,
    mut removed: RemovedComponents<PrismRenderable>,
) {
    let scene = &mut *scene;
    scene.frame += 1;
    scene.seq += 1;

    let mut builder = SceneTransactionBuilder::new(scene.frame, scene.seq);
    let mut has_ops = false;
    let mut membership_changed = false;

    for entity in removed.read() {
        if let Some(handle) = scene.entity_to_handle.remove(&entity) {
            scene.handle_to_entity.remove(&handle);
            builder.push(SceneOperation::Destroy { handle });
            has_ops = true;
            membership_changed = true;
        }
    }

    for (entity, transform, aabb, renderable) in &renderables {
        match scene.entity_to_handle.get(&entity).copied() {
            None => {
                let handle = scene.allocate_handle();
                scene.entity_to_handle.insert(entity, handle);
                scene.handle_to_entity.insert(handle, entity);
                let record = instance_record(
                    &aabb,
                    &transform,
                    renderable.geometry,
                    renderable.material,
                    renderable.render_layers,
                    renderable.flags,
                );
                builder.push(SceneOperation::Create { handle, record });
                has_ops = true;
                membership_changed = true;
            }
            Some(handle) => {
                if transform.is_changed() || aabb.is_changed() {
                    builder.push(SceneOperation::SetTransform {
                        handle,
                        current: scene_transform(&transform),
                    });
                    builder.push(SceneOperation::SetBounds {
                        handle,
                        bounds: scene_bounds(&aabb, &transform),
                    });
                    has_ops = true;
                }
            }
        }
    }

    if has_ops {
        let transaction = builder.finish();
        scene.scene.apply(&transaction);
    }
    if membership_changed {
        scene.rebuild_handles();
    }
}

/// Runs `cull_view` for every [`PrismCamera`] and writes the union of visible
/// instances back to each entity's [`PrismViewVisibility`].
fn cull_views(
    mut scene: ResMut<PrismRenderScene>,
    cameras: Query<(Entity, &Frustum, &GlobalTransform, &PrismCamera)>,
    mut targets: Query<(Entity, &mut PrismViewVisibility)>,
) {
    let scene = &mut *scene;
    let mut visible: BTreeSet<SceneHandle> = BTreeSet::new();

    for (camera_entity, frustum, camera_transform, prism_camera) in &cameras {
        let (view_handle, previous_clip) = match scene.views.get(&camera_entity) {
            Some(state) => (state.handle, state.previous_clip),
            None => {
                let handle = scene.allocate_view_handle();
                scene.views.insert(
                    camera_entity,
                    ViewState {
                        handle,
                        previous_clip: prism_camera.clip_from_world,
                    },
                );
                (handle, prism_camera.clip_from_world)
            }
        };

        let params = BevyViewParams {
            handle: view_handle,
            clip_from_world: prism_camera.clip_from_world,
            previous_clip_from_world: previous_clip,
            world_position: camera_transform.translation(),
            viewport: prism_camera.viewport,
            lod_scale: prism_camera.lod_scale,
            layer_mask: prism_camera.layer_mask,
            flags: prism_camera.flags,
            history_epoch: 0,
        };
        let view = view_record(frustum, &params);

        let (work, _diagnostics) = cull_view(
            &view,
            VisibilityInput {
                scene: &scene.scene,
                handles: &scene.handles,
                geometry: &scene.geometry,
                materials: &scene.materials,
                previous_lods: &scene.previous_lods,
                occluded: &scene.occluded,
                capacity: scene.capacity,
                previous_history_epoch: None,
            },
        );
        for item in work {
            visible.insert(item.scene);
        }

        if let Some(state) = scene.views.get_mut(&camera_entity) {
            state.previous_clip = prism_camera.clip_from_world;
        }
    }

    for (entity, mut view_visibility) in &mut targets {
        let now_visible = scene
            .entity_to_handle
            .get(&entity)
            .is_some_and(|handle| visible.contains(handle));
        if view_visibility.visible != now_visible {
            view_visibility.visible = now_visible;
        }
    }
}
