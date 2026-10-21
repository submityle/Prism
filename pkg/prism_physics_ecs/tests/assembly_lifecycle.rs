//! ECS integration tests for assembly ownership and generational remapping.
use std::time::Duration;

use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_ecs::core::{BodyDesc, BodyHandle, DistanceJoint, JointKind};
use prism_physics_ecs::*;
use prism_transform::Transform;

fn app() -> App {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins(PrismPhysicsPlugin);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .gravity = Vec3::ZERO;
    app
}

fn connected_pair(app: &mut App) -> (Entity, Entity, BodyHandle, BodyHandle) {
    let (a, b) = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        let a = physics.spawn(BodyDesc::dynamic_at(Vec3::new(-1.0, 0.0, 0.0)));
        let b = physics.spawn(BodyDesc::dynamic_at(Vec3::new(1.0, 0.0, 0.0)));
        physics
            .connect_assembly(
                a,
                b,
                None,
                JointKind::Distance(DistanceJoint::rigid(2.0)),
                None,
            )
            .unwrap();
        (a, b)
    };
    let first = app.world_mut().spawn((AssemblyBody { handle: a },));
    let second = app.world_mut().spawn((AssemblyBody { handle: b },));
    (first, second, a, b)
}

#[test]
fn merge_split_rebinds_each_entity_and_reverse_lookup_deterministically() {
    let mut app = app();
    let (first, second, original_a, original_b) = connected_pair(&mut app);

    app.world_mut()
        .resource_mut::<AssemblyRequestQueue>()
        .merge([second, first]);
    app.update();

    let proxy = app
        .world()
        .resource::<PhysicsBindings>()
        .body(first)
        .unwrap();
    assert_eq!(
        app.world().resource::<PhysicsBindings>().body(second),
        Some(proxy)
    );
    assert_ne!(proxy, original_a);
    assert_ne!(proxy, original_b);
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(proxy),
        Some(first.min(second))
    );
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(original_a));
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(original_b));

    app.world_mut()
        .resource_mut::<AssemblyRequestQueue>()
        .split(second);
    app.update();

    let split_a = app
        .world()
        .resource::<PhysicsBindings>()
        .body(first)
        .unwrap();
    let split_b = app
        .world()
        .resource::<PhysicsBindings>()
        .body(second)
        .unwrap();
    assert_ne!(split_a, split_b);
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(split_a),
        Some(first)
    );
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(split_b),
        Some(second)
    );
    assert_eq!(
        app.world().get::<AssemblyBody>(first).unwrap().handle,
        split_a
    );
    assert_eq!(
        app.world().get::<AssemblyBody>(second).unwrap().handle,
        split_b
    );
    assert!(app
        .world()
        .resource::<AssemblyDiagnostics>()
        .issues
        .is_empty());
}

#[test]
fn ordinary_rigid_body_ownership_is_not_taken_over_by_assembly_requests() {
    let mut app = app();
    let (_, second, a, _) = connected_pair(&mut app);
    let owned = app.world_mut().spawn((
        AssemblyBody { handle: a },
        RigidBody::dynamic(),
        Transform::IDENTITY,
    ));
    app.world_mut()
        .resource_mut::<AssemblyRequestQueue>()
        .merge([owned, second]);
    app.update();

    assert!(app
        .world()
        .resource::<AssemblyDiagnostics>()
        .issues
        .contains(&AssemblyDiagnostic {
            entity: Some(owned),
            issue: AssemblyIssue::RigidBodyOwned,
        }));
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(owned)
        .is_some());
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(a));
}

#[test]
fn despawning_a_member_splits_proxy_and_rebinds_surviving_entities() {
    let mut app = app();
    let (first, second, _, _) = connected_pair(&mut app);
    app.world_mut()
        .resource_mut::<AssemblyRequestQueue>()
        .merge([first, second]);
    app.update();
    let proxy = app
        .world()
        .resource::<PhysicsBindings>()
        .body(first)
        .unwrap();

    assert!(app.world_mut().despawn(second));
    app.update();

    let survivor = app
        .world()
        .resource::<PhysicsBindings>()
        .body(first)
        .unwrap();
    assert_ne!(survivor, proxy);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(survivor));
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(survivor),
        Some(first)
    );
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(second)
        .is_none());
}
