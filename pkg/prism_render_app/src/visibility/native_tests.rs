use prism_app::{App, Plugin};
use prism_ecs::entity::Entity;
use prism_math::{Aabb3, Frustum, Plane, Vec3};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
use prism_render_material::{fallback_material_record, MaterialRecord};
use prism_render_visibility::{GeometryLod, GeometryLodChain};
use prism_transform::ecs::LocalBounds;
use prism_transform::{GlobalTransform, Transform};

use super::{
    PrismCamera, PrismFrustum, PrismRenderScene, PrismRenderable, PrismViewVisibility,
    PrismVisibilityPlugin,
};

fn handle(index: u32) -> GenerationalHandle {
    GenerationalHandle::new(index, 1)
}

fn geometry(index: u32) -> GeometryLodChain {
    GeometryLodChain {
        geometry: handle(index),
        lods: vec![GeometryLod {
            level: 0,
            screen_error: 0.0,
            resident: true,
            fallback: true,
        }],
    }
}

fn material(index: u32) -> MaterialRecord {
    fallback_material_record(handle(index), 1)
}

fn frustum() -> PrismFrustum {
    PrismFrustum(Frustum::from_planes([
        Plane::new(Vec3::new(1.0, 0.0, 0.0), 10.0),
        Plane::new(Vec3::new(-1.0, 0.0, 0.0), 10.0),
        Plane::new(Vec3::new(0.0, -1.0, 0.0), 10.0),
        Plane::new(Vec3::new(0.0, 1.0, 0.0), 10.0),
        Plane::new(Vec3::new(0.0, 0.0, -1.0), -1.0),
        Plane::new(Vec3::new(0.0, 0.0, 1.0), 100.0),
    ]))
}

fn bounds() -> LocalBounds {
    LocalBounds(Aabb3::from_center_half_extents(
        Vec3::ZERO,
        Vec3::splat(1.0),
    ))
}

fn global(z: f32) -> GlobalTransform {
    GlobalTransform::from_transform(&Transform::from_xyz(0.0, 0.0, z))
}

fn app() -> App {
    let mut app = App::new();
    app.add_plugins(PrismVisibilityPlugin);
    {
        let scene = app.world_mut().resource_mut::<PrismRenderScene>();
        scene.insert_geometry(geometry(7));
        scene.insert_geometry(geometry(8));
        scene.insert_material(material(1));
        scene.insert_material(material(2));
    }
    app
}

fn camera(app: &mut App, layer_mask: u32) -> Entity {
    let mut view = PrismCamera::default();
    view.layer_mask = layer_mask;
    app.world_mut().spawn((frustum(), global(0.0), view))
}

fn renderable(app: &mut App, geometry: GeometryHandle, material: SceneMaterialHandle) -> Entity {
    app.world_mut().spawn((
        PrismRenderable::new(geometry, material),
        bounds(),
        global(-5.0),
        PrismViewVisibility::default(),
    ))
}

#[test]
fn native_cull_writes_union_and_prunes_removed_camera_history() {
    let mut app = app();
    let first_camera = camera(&mut app, 1);
    let second_camera = camera(&mut app, 2);
    let entity = renderable(&mut app, handle(7), handle(1));

    app.update();
    let scene = app.world().resource::<PrismRenderScene>();
    assert!(
        app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );
    assert_eq!(scene.view_count(), 2);
    assert_eq!(scene.history_count(), 2);

    app.world_mut().despawn(first_camera);
    app.update();
    let scene = app.world().resource::<PrismRenderScene>();
    assert_eq!(scene.view_count(), 1);
    assert_eq!(scene.history_count(), 1);
    assert!(scene.view_result(second_camera).is_some());

    app.world_mut().despawn(second_camera);
    app.update();
    assert_eq!(app.world().resource::<PrismRenderScene>().view_count(), 0);
    assert_eq!(
        app.world().resource::<PrismRenderScene>().history_count(),
        0
    );
    assert!(
        !app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );
}

#[test]
fn native_field_edits_reuse_handle_and_change_culling_contract() {
    let mut app = app();
    let camera = camera(&mut app, 1);
    let entity = renderable(&mut app, handle(7), handle(1));
    app.update();
    let old_handle = app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(entity)
        .unwrap();
    assert!(
        app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );

    app.world_mut().insert(
        entity,
        PrismRenderable {
            geometry: handle(8),
            material: handle(2),
            render_layers: 2,
            flags: 1,
        },
    );
    app.update();
    let scene = app.world().resource::<PrismRenderScene>();
    assert_eq!(scene.handle_for(entity), Some(old_handle));
    assert!(
        !app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );
    assert_eq!(
        scene
            .view_result(camera)
            .unwrap()
            .diagnostics
            .layer_rejected,
        1
    );

    app.world_mut()
        .get_mut::<PrismCamera>(camera)
        .unwrap()
        .layer_mask = 2;
    app.update();
    let result = app
        .world()
        .resource::<PrismRenderScene>()
        .view_result(camera)
        .unwrap();
    assert_eq!(result.work.len(), 1);
    assert_eq!(result.work[0].geometry, handle(8));
    assert_eq!(result.work[0].material, handle(2));
    assert!(result.work[0].pass_mask.0 & prism_render_visibility::RenderPassMask::SHADOW.0 == 0);
}

#[test]
fn native_bounds_or_global_removal_destroys_scene_instance_and_boolean() {
    let mut app = app();
    camera(&mut app, u32::MAX);
    let entity = renderable(&mut app, handle(7), handle(1));
    app.update();
    let old = app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(entity)
        .unwrap();

    app.world_mut().remove::<LocalBounds>(entity);
    app.update();
    let scene = app.world().resource::<PrismRenderScene>();
    assert_eq!(scene.handle_for(entity), None);
    assert_eq!(scene.scene().get(old), None);
    assert!(
        !app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );

    app.world_mut().insert(entity, bounds());
    app.update();
    assert!(app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(entity)
        .is_some());
    app.world_mut().remove::<GlobalTransform>(entity);
    app.update();
    assert_eq!(
        app.world()
            .resource::<PrismRenderScene>()
            .handle_for(entity),
        None
    );
    assert!(
        !app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );
}

#[test]
fn native_despawn_reclaims_instance_and_scene_slot() {
    let mut app = app();
    camera(&mut app, u32::MAX);
    let entity = renderable(&mut app, handle(7), handle(1));
    app.update();
    assert_eq!(
        app.world().resource::<PrismRenderScene>().instance_count(),
        1
    );

    app.world_mut().despawn(entity);
    app.update();
    let scene = app.world().resource::<PrismRenderScene>();
    assert_eq!(scene.instance_count(), 0);
    assert_eq!(scene.scene().live_count(), 0);
}

#[test]
fn native_camera_history_resets_on_epoch_and_geometry_replacement() {
    let mut app = app();
    let camera = camera(&mut app, u32::MAX);
    let entity = renderable(&mut app, handle(7), handle(1));
    app.update();
    let view = app
        .world()
        .resource::<PrismRenderScene>()
        .view_result(camera)
        .unwrap()
        .view
        .handle;
    let instance = app
        .world()
        .resource::<PrismRenderScene>()
        .handle_for(entity)
        .unwrap();
    assert_eq!(
        app.world()
            .resource::<PrismRenderScene>()
            .previous_lod(view, instance),
        Some(0)
    );

    app.world_mut()
        .get_mut::<PrismCamera>(camera)
        .unwrap()
        .history_epoch = 1;
    app.update();
    let scene = app.world().resource::<PrismRenderScene>();
    assert_eq!(
        scene.view_result(camera).unwrap().previous_history_epoch,
        Some(0)
    );
    assert_eq!(scene.previous_lod(view, instance), Some(0));

    scene_check_geometry_invalidation(&mut app, camera, entity, view, instance);
}

#[test]
fn native_geometry_epoch_tracks_replacement_and_removal() {
    let mut scene = PrismRenderScene::default();
    assert_eq!(scene.geometry_epoch(), 0);
    assert_eq!(scene.geometry_records().count(), 0);

    scene.insert_geometry(geometry(7));
    assert_eq!(scene.geometry_epoch(), 1);
    assert_eq!(scene.geometry_records().count(), 1);

    scene.insert_geometry(geometry(7));
    assert_eq!(scene.geometry_epoch(), 1);
    assert_eq!(scene.geometry_records().count(), 1);

    let mut replacement = geometry(7);
    replacement.lods[0].screen_error = 0.5;
    scene.insert_geometry(replacement.clone());
    assert_eq!(scene.geometry_epoch(), 2);
    assert_eq!(scene.geometry_records().next(), Some(&replacement));

    assert_eq!(scene.remove_geometry(handle(7)), Some(replacement));
    assert_eq!(scene.geometry_epoch(), 3);
    assert_eq!(scene.geometry_records().count(), 0);

    assert_eq!(scene.remove_geometry(handle(7)), None);
    assert_eq!(scene.geometry_epoch(), 3);
}

fn scene_check_geometry_invalidation(
    app: &mut App,
    camera: Entity,
    entity: Entity,
    view: prism_render_visibility::ViewHandle,
    instance: prism_render_architecture::gpu_scene::SceneHandle,
) {
    app.world_mut()
        .resource_mut::<PrismRenderScene>()
        .insert_geometry(geometry(7));
    assert_eq!(
        app.world()
            .resource::<PrismRenderScene>()
            .previous_lod(view, instance),
        None
    );
    app.update();
    assert_eq!(
        app.world()
            .resource::<PrismRenderScene>()
            .previous_lod(view, instance),
        Some(0)
    );
    assert!(app
        .world()
        .resource::<PrismRenderScene>()
        .view_result(camera)
        .is_some());
    assert!(
        app.world()
            .get::<PrismViewVisibility>(entity)
            .unwrap()
            .visible
    );
}
