//! Host-level coverage for the production `prism_app::App` assembly path.

use prism_app::{App, PostUpdate};
use prism_ecs::prelude::{Entity, IntoSystemConfigs, Resource, World};
use prism_math::{Aabb3, Vec3};
use prism_transform::ecs::LocalBounds;
use prism_transform::{GlobalTransform, Transform};
use prism_transform_ecs::{
    SpatialIndexFrame, SpatialIndexes, TransformSpatialPlugin, TransformSpatialSystems,
};

#[derive(Default, Resource)]
struct MoveRequest(Option<(Entity, f32)>);

fn apply_requested_move(world: &mut World) {
    let request = world.resource_mut::<MoveRequest>().0.take();
    if let Some((entity, x)) = request {
        world
            .get_mut::<Transform>(entity)
            .expect("move target must exist")
            .translation
            .x = x;
    }
}

fn local_bounds() -> LocalBounds {
    LocalBounds(Aabb3::new(Vec3::ZERO, Vec3::ONE))
}

fn host_app() -> App {
    let mut app = App::new();
    app.add_plugins(TransformSpatialPlugin::default())
        .init_resource::<MoveRequest>()
        .add_systems(
            PostUpdate,
            apply_requested_move.before(TransformSpatialSystems::Propagate),
        );
    app
}

fn assert_backend_hits(world: &World, x: f32, expected: &[Entity]) {
    let indexes = world.resource::<SpatialIndexes>();
    #[cfg(feature = "bvh")]
    {
        let query = Aabb3::new(Vec3::new(x, -1.0, -1.0), Vec3::new(x + 1.0, 2.0, 2.0));
        assert_eq!(indexes.bvh().unwrap().query_aabb(query), expected);
        assert!(indexes.bvh().unwrap().tree().validate());
    }
    #[cfg(feature = "grid")]
    assert_eq!(
        indexes.grid().unwrap().query_circle(x + 0.5, 0.5, 0.25),
        expected
    );
    let _ = (indexes, x, expected);
}

#[test]
fn app_plugin_tracks_spawn_same_frame_move_and_despawn() {
    let mut app = host_app();
    let entity = app.world_mut().spawn((Transform::IDENTITY, local_bounds()));

    app.update();
    let frame = app.world().resource::<SpatialIndexFrame>();
    assert_eq!(frame.sequence(), 1);
    assert_eq!(frame.batch().stats.inserts, 1);
    assert_eq!(frame.error(), None);
    assert_eq!(app.world().resource::<SpatialIndexes>().len(), 1);
    assert_backend_hits(app.world(), 0.0, &[entity]);

    app.world_mut().resource_mut::<MoveRequest>().0 = Some((entity, 12.0));
    app.update();
    let frame = app.world().resource::<SpatialIndexFrame>();
    assert_eq!(frame.sequence(), 2);
    assert_eq!(frame.batch().stats.updates, 1);
    assert_eq!(
        app.world()
            .resource::<SpatialIndexes>()
            .bounds(entity)
            .unwrap()
            .min
            .x,
        12.0
    );
    assert_backend_hits(app.world(), 0.0, &[]);
    assert_backend_hits(app.world(), 12.0, &[entity]);

    assert!(app.world_mut().despawn(entity));
    app.update();
    let frame = app.world().resource::<SpatialIndexFrame>();
    assert_eq!(frame.sequence(), 3);
    assert_eq!(frame.batch().stats.removes, 1);
    assert!(app.world().resource::<SpatialIndexes>().is_empty());
    assert_backend_hits(app.world(), 12.0, &[]);
}

#[test]
fn app_plugin_reconciles_snapshot_restore_against_live_indexes() {
    let mut app = host_app();
    app.world_mut().register_snapshot_component::<Transform>();
    app.world_mut()
        .register_snapshot_component::<GlobalTransform>();
    app.world_mut().register_snapshot_component::<LocalBounds>();

    let moved = app.world_mut().spawn((Transform::IDENTITY, local_bounds()));
    let resurrected = app
        .world_mut()
        .spawn((Transform::from_xyz(8.0, 0.0, 0.0), local_bounds()));
    app.update();
    let snapshot = app.world().snapshot();

    app.world_mut().resource_mut::<MoveRequest>().0 = Some((moved, 12.0));
    app.update();
    assert!(app.world_mut().despawn(resurrected));
    let abandoned = app
        .world_mut()
        .spawn((Transform::from_xyz(20.0, 0.0, 0.0), local_bounds()));
    app.update();
    assert_backend_hits(app.world(), 12.0, &[moved]);
    assert_backend_hits(app.world(), 20.0, &[abandoned]);

    app.world_mut().restore(&snapshot);
    app.update();

    let frame = app.world().resource::<SpatialIndexFrame>();
    assert_eq!(
        (
            frame.batch().stats.inserts,
            frame.batch().stats.updates,
            frame.batch().stats.removes,
        ),
        (1, 1, 1)
    );
    assert_eq!(frame.error(), None);
    let indexes = app.world().resource::<SpatialIndexes>();
    assert_eq!(indexes.len(), 2);
    assert_eq!(indexes.bounds(moved).unwrap().min.x, 0.0);
    assert_eq!(indexes.bounds(resurrected).unwrap().min.x, 8.0);
    assert!(indexes.bounds(abandoned).is_none());
    assert_backend_hits(app.world(), 0.0, &[moved]);
    assert_backend_hits(app.world(), 8.0, &[resurrected]);
    assert_backend_hits(app.world(), 12.0, &[]);
    assert_backend_hits(app.world(), 20.0, &[]);
}

#[cfg(feature = "grid")]
#[test]
fn app_plugin_applies_validated_grid_configuration() {
    use prism_transform_ecs::TransformSpatialConfig;

    let config = TransformSpatialConfig::default()
        .with_grid_cell_size(2.5)
        .unwrap();
    let mut app = App::new();
    app.add_plugins(TransformSpatialPlugin::new(config));
    assert_eq!(
        app.world()
            .resource::<SpatialIndexes>()
            .grid()
            .unwrap()
            .grid()
            .cell_size(),
        2.5
    );
    assert!(TransformSpatialConfig::default()
        .with_grid_cell_size(0.0)
        .is_err());
}
