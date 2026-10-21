use std::time::Duration;

use prism_app::{App, FixedPostUpdate, FixedUpdate, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_ecs::core::{BodyDesc, BodyHandle, ColliderShape, QueryFilter, SoftBody};
use prism_physics_ecs::geometry::Ray;
use prism_physics_ecs::*;
use prism_transform::{GlobalTransform, Transform};

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

fn body(app: &App, entity: Entity) -> BodyHandle {
    app.world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .unwrap()
}

fn position(app: &App, entity: Entity) -> Vec3 {
    app.world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .position(body(app, entity))
        .unwrap()
}

fn near(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
}

fn has_issue(app: &App, entity: Option<Entity>, issue: PhysicsIssue) -> bool {
    app.world()
        .resource::<PhysicsDiagnostics>()
        .issues
        .contains(&PhysicsDiagnostic { entity, issue })
}

#[test]
fn real_app_fixed_accumulator_zero_and_multiple_ticks_and_ordering() {
    #[derive(Default, Resource)]
    struct Samples(Vec<f32>);
    fn author(world: &mut World) {
        let query = world.query::<Entity>();
        let entities: Vec<_> = query.iter(world).collect();
        for e in entities {
            if let Some(mut v) = world.get_mut::<Velocity>(e) {
                v.linear = Vec3::X;
            }
        }
    }
    fn observe(world: &mut World) {
        let x = world
            .query::<&Transform>()
            .iter(world)
            .next()
            .unwrap()
            .translation
            .x;
        world.resource_mut::<Samples>().0.push(x);
    }
    let mut app = app();
    app.init_resource::<Samples>()
        .add_systems(FixedUpdate, author.before(PhysicsSystems::Sync))
        .add_systems(FixedPostUpdate, observe)
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(3)));
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
    ));
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_none());
    assert_eq!(app.world().resource::<PhysicsDiagnostics>().tick, 0);
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(27)));
    app.update();
    assert_eq!(app.world().resource::<PhysicsDiagnostics>().tick, 3);
    let samples = &app.world().resource::<Samples>().0;
    assert_eq!(samples.len(), 3);
    near(samples[0], 0.01);
    near(samples[2], 0.03);
    near(position(&app, entity).x, 0.03);
}

#[test]
fn rigid_lifecycle_collider_edits_removal_readd_and_entity_despawn() {
    let mut app = app();
    let entity = app
        .world_mut()
        .spawn((RigidBody::dynamic(), Transform::IDENTITY));
    app.update();
    let first = body(&app, entity);
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(first),
        Some(entity)
    );
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .collider(first)
        .is_none());
    app.world_mut().insert(entity, (Collider::sphere(0.5),));
    app.update();
    assert_eq!(body(&app, entity), first);
    let original_shape = app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .collider(first)
        .unwrap();
    app.world_mut().get_mut::<Collider>(entity).unwrap().shape =
        CollisionShape::Analytic(ColliderShape::Cuboid {
            half_extents: Vec3::ONE,
        });
    app.world_mut().get_mut::<Collider>(entity).unwrap().sensor = true;
    app.update();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert_ne!(physics.bodies.collider(first), Some(original_shape));
    assert_eq!(physics.bodies.is_sensor(first), Some(true));
    assert!(app.world_mut().remove::<Collider>(entity));
    app.update();
    assert_eq!(body(&app, entity), first);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .collider(first)
        .is_none());

    assert!(app.world_mut().remove::<RigidBody>(entity));
    app.world_mut().insert(entity, (RigidBody::dynamic(),));
    app.update();
    let second = body(&app, entity);
    assert_ne!(
        first, second,
        "remove/readd between ticks is a new lifecycle"
    );
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(first));
    assert!(app.world_mut().remove::<Transform>(entity));
    app.update();
    assert!(has_issue(
        &app,
        Some(entity),
        PhysicsIssue::MissingTransform
    ));
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_none());
    app.world_mut()
        .insert(entity, (Transform::from_xyz(4.0, 0.0, 0.0),));
    app.update();
    near(position(&app, entity).x, 4.0);
    assert!(app.world_mut().despawn(entity));
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_none());
    assert_eq!(
        app.world().resource::<PhysicsWorldResource>().bodies.len(),
        0
    );
}

#[test]
fn stale_generation_cleanup_does_not_remove_an_unmanaged_reused_slot() {
    let mut app = app();
    let entity = app
        .world_mut()
        .spawn((RigidBody::dynamic(), Transform::IDENTITY));
    app.update();
    let stale = body(&app, entity);
    let unmanaged = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        assert!(physics.remove_body(stale));
        physics.spawn(BodyDesc::static_at(Vec3::new(50.0, 0.0, 0.0)))
    };
    assert_eq!(stale.index(), unmanaged.index());
    assert_ne!(stale.generation(), unmanaged.generation());
    app.update();
    let fresh = body(&app, entity);
    assert_ne!(fresh, stale);
    assert_ne!(fresh, unmanaged);
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(stale),
        None
    );
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(unmanaged),
        None
    );
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(unmanaged));
    assert!(has_issue(&app, Some(entity), PhysicsIssue::StaleBody));
}

#[test]
fn teleport_velocity_edits_mass_rebuild_and_hierarchy_writeback() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity {
            linear: Vec3::X,
            angular: Vec3::ZERO,
        },
    ));
    let child = app.world_mut().spawn((Transform::from_xyz(0.0, 2.0, 0.0),));
    app.world_mut().add_relation::<ChildOf>(child, entity);
    app.update();
    near(position(&app, entity).x, 0.01);
    let first = body(&app, entity);
    app.world_mut()
        .get_mut::<Transform>(entity)
        .unwrap()
        .translation
        .x = 10.0;
    app.world_mut().get_mut::<Velocity>(entity).unwrap().linear = Vec3::new(2.0, 0.0, 0.0);
    app.update();
    assert_eq!(body(&app, entity), first);
    near(position(&app, entity).x, 10.02);
    near(
        app.world()
            .get::<GlobalTransform>(entity)
            .unwrap()
            .translation()
            .x,
        10.02,
    );
    near(
        app.world()
            .get::<GlobalTransform>(child)
            .unwrap()
            .translation()
            .x,
        10.02,
    );
    near(
        app.world()
            .get::<GlobalTransform>(child)
            .unwrap()
            .translation()
            .y,
        2.0,
    );
    near(
        app.world().get::<Transform>(child).unwrap().translation.x,
        0.0,
    );
    app.world_mut()
        .get_mut::<RigidBody>(entity)
        .unwrap()
        .mass
        .inv_mass = 0.5;
    app.update();
    assert_ne!(body(&app, entity), first);
    near(position(&app, entity).x, 10.04);
    near(app.world().get::<Velocity>(entity).unwrap().linear.x, 2.0);
    app.world_mut().remove::<Velocity>(entity);
    app.update();
    near(position(&app, entity).x, 10.04);
    assert!(!app.world().has::<Velocity>(entity));
}

#[test]
fn parent_scale_and_invalid_authoring_are_rejected_and_recover() {
    let mut app = app();
    let parent = app.world_mut().spawn((Transform::from_xyz(3.0, 0.0, 0.0),));
    let entity = app
        .world_mut()
        .spawn((RigidBody::dynamic(), Transform::IDENTITY));
    app.world_mut().add_relation::<ChildOf>(entity, parent);
    app.update();
    assert!(has_issue(&app, Some(entity), PhysicsIssue::ParentedBody));
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_none());
    app.world_mut().remove_relation::<ChildOf>(entity, parent);
    app.world_mut()
        .get_mut::<Transform>(entity)
        .unwrap()
        .scale
        .x = 2.0;
    app.update();
    assert!(has_issue(&app, Some(entity), PhysicsIssue::NonUnitScale));
    app.world_mut()
        .get_mut::<Transform>(entity)
        .unwrap()
        .scale
        .x = 1.0;
    app.update();
    let old = body(&app, entity);
    app.world_mut()
        .get_mut::<RigidBody>(entity)
        .unwrap()
        .mass
        .inv_mass = f32::NAN;
    app.update();
    assert!(has_issue(
        &app,
        Some(entity),
        PhysicsIssue::InvalidAuthoring
    ));
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(old));
    *app.world_mut().get_mut::<RigidBody>(entity).unwrap() = RigidBody::fixed();
    app.update();
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .kind(body(&app, entity)),
        Some(core::BodyKind::Static)
    );
}

#[test]
fn sensor_events_keep_entity_identity_on_despawn_and_handle_reuse_and_queries_work() {
    let mut app = app();
    let mut sensor = Collider::sphere(1.0);
    sensor.sensor = true;
    let sensor_entity = app
        .world_mut()
        .spawn((RigidBody::fixed(), Transform::IDENTITY, sensor));
    let other = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::from_xyz(1.0, 0.0, 0.0),
        Collider::sphere(0.5),
    ));
    let mut cursor = EventCursor::<PhysicsContactEvent>::default();
    app.update();
    let events: Vec<_> = cursor
        .read(app.world().resource::<Events<PhysicsContactEvent>>())
        .copied()
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ContactEventKind::TriggerEntered);
    assert_eq!(events[0].first.entity, Some(sensor_entity));
    assert_eq!(events[0].second.entity, Some(other));
    let stale = body(&app, other);
    let hit = app
        .world()
        .resource::<PhysicsWorldResource>()
        .raycast(
            &Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X),
            &QueryFilter::ALL,
        )
        .unwrap();
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(hit.body),
        Some(sensor_entity)
    );
    app.world_mut().despawn(other);
    let replacement = app.world_mut().spawn((
        RigidBody::fixed(),
        Transform::from_xyz(20.0, 0.0, 0.0),
        Collider::sphere(0.5),
    ));
    app.update();
    let events: Vec<_> = cursor
        .read(app.world().resource::<Events<PhysicsContactEvent>>())
        .copied()
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].kind, ContactEventKind::TriggerExited);
    assert_eq!(events[0].second.body, stale);
    assert_eq!(events[0].second.entity, Some(other));
    assert_ne!(other, replacement);
    assert_eq!(
        app.world().resource::<PhysicsBindings>().entity(stale),
        None
    );
    assert_ne!(body(&app, replacement), stale);
}

#[test]
fn rigid_body_aero_lifecycle_registers_replaces_and_removes_fixed_step_wrenches() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        RigidBodyAero(
            core::AeroBodyConfig::default().with_thruster(core::Thruster::new(
                Vec3::ZERO,
                Vec3::X,
                10.0,
            )),
        ),
    ));

    app.update();
    let handle = body(&app, entity);
    near(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .linear_velocity(handle)
            .unwrap()
            .x,
        0.1,
    );
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle)
        .is_some());

    app.world_mut().get_mut::<RigidBodyAero>(entity).unwrap().0 = core::AeroBodyConfig::default()
        .with_thruster(core::Thruster::new(Vec3::ZERO, Vec3::X, 20.0));
    app.update();
    near(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .linear_velocity(handle)
            .unwrap()
            .x,
        0.3,
    );

    app.world_mut()
        .get_mut::<RigidBody>(entity)
        .unwrap()
        .mass
        .inv_mass = 0.5;
    app.update();
    let rebuilt = body(&app, entity);
    assert_ne!(rebuilt, handle);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(rebuilt)
        .is_some());

    assert!(app.world_mut().remove::<RigidBodyAero>(entity));
    app.update();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!(physics.aero_body(rebuilt).is_none());
    near(physics.bodies.linear_velocity(rebuilt).unwrap().x, 0.4);
}

#[test]
fn persistent_force_applies_every_substep_and_world_impulse_is_not_overwritten() {
    for substeps in [1, 4, 8] {
        let mut app = app();
        app.world_mut()
            .resource_mut::<PhysicsWorldResource>()
            .config
            .default_substeps = substeps;
        let entity = app.world_mut().spawn((
            RigidBody::dynamic(),
            Transform::IDENTITY,
            Velocity::default(),
            ExternalForce {
                force: Vec3::new(10.0, 0.0, 0.0),
                torque: Vec3::ZERO,
            },
        ));
        app.update();
        near(app.world().get::<Velocity>(entity).unwrap().linear.x, 0.1);
        app.update();
        near(app.world().get::<Velocity>(entity).unwrap().linear.x, 0.2);
        app.world_mut().remove::<ExternalForce>(entity);
        let handle = body(&app, entity);
        let point = position(&app, entity);
        assert!(app
            .world_mut()
            .resource_mut::<PhysicsWorldResource>()
            .apply_impulse(handle, Vec3::X, point));
        app.update();
        near(app.world().get::<Velocity>(entity).unwrap().linear.x, 1.2);
    }
}

#[test]
fn soft_body_binding_simulates_edits_removal_readd_and_stale_recovery() {
    let mut app = app();
    let mut authored = SoftBody::with_default_config();
    let particle = authored.spawn(Vec3::new(0.0, 3.0, 0.0), 1.0);
    let entity = app.world_mut().spawn((
        SoftBodyBinding(authored.clone()),
        Transform::from_xyz(100.0, 0.0, 0.0),
    ));
    app.update();
    let first = app
        .world()
        .resource::<PhysicsBindings>()
        .soft_body(entity)
        .unwrap();
    assert_eq!(
        app.world().resource::<PhysicsBindings>().soft_entity(first),
        Some(entity)
    );
    let p = app
        .world()
        .resource::<PhysicsWorldResource>()
        .soft_bodies
        .get(first)
        .unwrap()
        .particles
        .position(particle)
        .unwrap();
    assert!(p.y < 3.0);
    near(p.x, 0.0);
    assert_eq!(
        app.world().get::<SoftBodyBinding>(entity).unwrap().0,
        authored,
        "live simulation must not overwrite authoring"
    );
    app.update();
    assert_eq!(
        app.world().resource::<PhysicsBindings>().soft_body(entity),
        Some(first)
    );
    app.world_mut()
        .get_mut::<SoftBodyBinding>(entity)
        .unwrap()
        .0
        .config
        .gravity = Vec3::ZERO;
    app.update();
    let second = app
        .world()
        .resource::<PhysicsBindings>()
        .soft_body(entity)
        .unwrap();
    assert_ne!(first, second);
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .soft_bodies
        .contains(first));
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .remove_soft_body(second);
    app.update();
    assert!(has_issue(&app, Some(entity), PhysicsIssue::StaleSoftBody));
    let third = app
        .world()
        .resource::<PhysicsBindings>()
        .soft_body(entity)
        .unwrap();
    app.world_mut().remove::<SoftBodyBinding>(entity);
    app.world_mut().insert(entity, (SoftBodyBinding(authored),));
    app.update();
    assert_ne!(
        app.world().resource::<PhysicsBindings>().soft_body(entity),
        Some(third)
    );
    app.world_mut().despawn(entity);
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .soft_body(entity)
        .is_none());
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .soft_bodies
        .is_empty());
}

#[test]
fn invalid_configuration_reports_no_advance_then_recovers() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity {
            linear: Vec3::X,
            angular: Vec3::ZERO,
        },
    ));
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 0;
    app.update();
    near(position(&app, entity).x, 0.0);
    assert!(has_issue(&app, None, PhysicsIssue::InvalidConfiguration));
    assert_eq!(
        app.world().resource::<PhysicsDiagnostics>().simulated_dt,
        0.0
    );
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 4;
    app.update();
    near(position(&app, entity).x, 0.01);
    assert!(app
        .world()
        .resource::<PhysicsDiagnostics>()
        .issues
        .is_empty());
}

fn read_events(
    app: &App,
    cursor: &mut EventCursor<PhysicsContactEvent>,
) -> Vec<PhysicsContactEvent> {
    cursor
        .read(app.world().resource::<Events<PhysicsContactEvent>>())
        .copied()
        .collect()
}

#[test]
fn solid_collision_resolves_penetration_and_maps_transitions() {
    let mut app = app();
    let fixed = app.world_mut().spawn((
        RigidBody::fixed(),
        Transform::IDENTITY,
        Collider::sphere(1.0),
    ));
    let moving = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::from_xyz(1.25, 0.0, 0.0),
        Collider::sphere(0.5),
        Velocity::default(),
    ));
    let mut cursor = EventCursor::default();
    app.update();
    near(position(&app, fixed).x, 0.0);
    assert!(
        position(&app, moving).x >= 1.499,
        "solid contact must project the dynamic body"
    );
    let events = read_events(&app, &mut cursor);
    let start = events
        .iter()
        .find(|e| e.kind == ContactEventKind::CollisionStarted)
        .unwrap();
    assert_eq!(start.first.entity, Some(fixed));
    assert_eq!(start.second.entity, Some(moving));
    // The separating position correction can end the pair within this tick.
    app.world_mut()
        .get_mut::<Transform>(moving)
        .unwrap()
        .translation
        .x = 10.0;
    app.update();
    let later = read_events(&app, &mut cursor);
    assert!(events
        .iter()
        .chain(&later)
        .any(|e| e.kind == ContactEventKind::CollisionEnded
            && e.first.entity == Some(fixed)
            && e.second.entity == Some(moving)));
}

#[test]
fn sensor_role_swap_and_solid_sensor_edits_balance_events() {
    let mut app = app();
    let mut sensor = Collider::sphere(1.0);
    sensor.sensor = true;
    let a = app
        .world_mut()
        .spawn((RigidBody::fixed(), Transform::IDENTITY, sensor));
    // Keep a dynamic broad-phase participant while locking all response DOFs,
    // so solid/sensor policy changes cannot separate this event-test pair.
    let b = app.world_mut().spawn((
        RigidBody {
            kind: core::BodyKind::Dynamic,
            ..RigidBody::fixed()
        },
        Transform::from_xyz(1.0, 0.0, 0.0),
        Collider::sphere(1.0),
    ));
    let mut cursor = EventCursor::default();
    app.update();
    assert_eq!(read_events(&app, &mut cursor)[0].first.entity, Some(a));
    app.world_mut().get_mut::<Collider>(a).unwrap().sensor = false;
    app.world_mut().get_mut::<Collider>(b).unwrap().sensor = true;
    app.update();
    let events = read_events(&app, &mut cursor);
    assert_eq!(events.len(), 2);
    assert_eq!(
        (events[0].kind, events[0].first.entity),
        (ContactEventKind::TriggerExited, Some(a))
    );
    assert_eq!(
        (events[1].kind, events[1].first.entity),
        (ContactEventKind::TriggerEntered, Some(b))
    );
    app.update();
    assert!(read_events(&app, &mut cursor).is_empty());
    app.world_mut().get_mut::<Collider>(b).unwrap().sensor = false;
    app.update();
    let events = read_events(&app, &mut cursor);
    assert_eq!(
        events.iter().map(|e| e.kind).collect::<Vec<_>>(),
        [
            ContactEventKind::TriggerExited,
            ContactEventKind::CollisionStarted
        ]
    );
    app.world_mut().get_mut::<Collider>(a).unwrap().sensor = true;
    app.update();
    let events = read_events(&app, &mut cursor);
    assert_eq!(
        events.iter().map(|e| e.kind).collect::<Vec<_>>(),
        [
            ContactEventKind::CollisionEnded,
            ContactEventKind::TriggerEntered
        ]
    );
    app.world_mut().despawn(b);
    app.update();
    let events = read_events(&app, &mut cursor);
    assert_eq!(events.len(), 1);
    assert_eq!(
        (
            events[0].kind,
            events[0].first.entity,
            events[0].second.entity
        ),
        (ContactEventKind::TriggerExited, Some(a), Some(b))
    );
}

#[test]
fn static_and_kinematic_poses_are_authored_not_velocity_integrated() {
    let mut app = app();
    let mut entities = Vec::new();
    for rigid in [RigidBody::fixed(), RigidBody::kinematic()] {
        entities.push(app.world_mut().spawn((
            rigid,
            Transform::IDENTITY,
            Velocity {
                linear: Vec3::X * 10.0,
                angular: Vec3::Z,
            },
            ExternalForce {
                force: Vec3::Y * 100.0,
                torque: Vec3::Z,
            },
        )));
    }
    app.update();
    for &entity in &entities {
        assert_eq!(position(&app, entity), Vec3::ZERO);
        assert_eq!(
            app.world()
                .get::<Transform>(entity)
                .unwrap()
                .rotation
                .to_array(),
            [0.0, 0.0, 0.0, 1.0]
        );
        app.world_mut()
            .get_mut::<Transform>(entity)
            .unwrap()
            .translation
            .y = 3.0;
    }
    app.update();
    for entity in entities {
        near(position(&app, entity).y, 3.0);
    }
}

#[test]
fn invalid_colliders_recover_registered_shapes_work_and_analytic_shapes_are_deduplicated() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        RigidBody::fixed(),
        Transform::IDENTITY,
        Collider::sphere(-1.0),
    ));
    app.update();
    assert!(has_issue(
        &app,
        Some(entity),
        PhysicsIssue::InvalidAuthoring
    ));
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_none());
    *app.world_mut().get_mut::<Collider>(entity).unwrap() = Collider::sphere(0.5);
    app.update();
    let shape = app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .collider(body(&app, entity))
        .unwrap();
    for _ in 0..3 {
        app.world_mut().remove::<RigidBody>(entity);
        app.world_mut().insert(entity, (RigidBody::fixed(),));
        app.update();
    }
    assert_eq!(
        app.world().resource::<PhysicsWorldResource>().shapes.len(),
        1
    );
    *app.world_mut().get_mut::<Collider>(entity).unwrap() = Collider::registered(shape);
    app.update();
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .collider(body(&app, entity)),
        Some(shape)
    );
    *app.world_mut().get_mut::<Collider>(entity).unwrap() =
        Collider::registered(core::ColliderHandle(u32::MAX));
    app.update();
    assert!(has_issue(&app, Some(entity), PhysicsIssue::MissingShape));
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_none());
    *app.world_mut().get_mut::<Collider>(entity).unwrap() = Collider::registered(shape);
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .is_some());
}

#[test]
fn late_parenting_transform_readd_and_velocity_readd_are_observed() {
    let mut app = app();
    let parent = app.world_mut().spawn((Transform::IDENTITY,));
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity {
            linear: Vec3::X,
            angular: Vec3::ZERO,
        },
    ));
    app.update();
    let first = body(&app, entity);
    app.world_mut().add_relation::<ChildOf>(entity, parent);
    app.update();
    assert!(has_issue(&app, Some(entity), PhysicsIssue::ParentedBody));
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .bodies
        .contains(first));
    app.world_mut().remove_relation::<ChildOf>(entity, parent);
    app.update();
    let second = body(&app, entity);
    app.world_mut().remove::<Transform>(entity);
    app.world_mut()
        .insert(entity, (Transform::from_xyz(2.0, 0.0, 0.0),));
    app.update();
    assert_ne!(second, body(&app, entity));
    near(position(&app, entity).x, 2.01);
    // Reauthoring the same old value must still override a resource-side edit.
    let handle = body(&app, entity);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .bodies
        .set_linear_velocity(handle, Vec3::X * 50.0);
    app.world_mut().remove::<Velocity>(entity);
    app.world_mut().insert(
        entity,
        (Velocity {
            linear: Vec3::X,
            angular: Vec3::ZERO,
        },),
    );
    app.update();
    near(app.world().get::<Velocity>(entity).unwrap().linear.x, 1.0);
}

#[test]
fn torque_writes_rotation_and_zero_wrench_preserves_sleep() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
        ExternalForce {
            force: Vec3::ZERO,
            torque: Vec3::Z * 10.0,
        },
    ));
    app.update();
    near(app.world().get::<Velocity>(entity).unwrap().angular.z, 0.1);
    let rotation = app.world().get::<Transform>(entity).unwrap().rotation;
    assert!(rotation.to_array()[2] > 0.0);
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .orientation(body(&app, entity))
            .unwrap()
            .to_array(),
        rotation.to_array()
    );
    *app.world_mut().get_mut::<ExternalForce>(entity).unwrap() = ExternalForce::default();
    *app.world_mut().get_mut::<Velocity>(entity).unwrap() = Velocity::default();
    app.update();
    let handle = body(&app, entity);
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        let view = physics.bodies.solver_view_mut();
        view.sleeping[handle.index() as usize] = true;
        view.sleep_timers[handle.index() as usize] = 1.0;
    }
    app.update();
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .is_sleeping(handle),
        Some(true)
    );
}

#[test]
fn mass_rebuild_preserves_direct_live_state_and_removes_disabled_world_joints() {
    let mut app = app();
    let a = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
    ));
    let b = app
        .world_mut()
        .spawn((RigidBody::fixed(), Transform::from_xyz(10.0, 0.0, 0.0)));
    app.update();
    let old = body(&app, a);
    let other = body(&app, b);
    let joint = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.bodies.set_position(old, Vec3::X * 5.0);
        physics.bodies.set_linear_velocity(old, Vec3::X * 2.0);
        physics.spawn_joint(
            core::JointDesc::fixed(
                core::JointAnchor::at_center(old),
                core::JointAnchor::at_center(other),
            )
            .with_enabled(false),
        )
    };
    app.world_mut()
        .get_mut::<RigidBody>(a)
        .unwrap()
        .mass
        .inv_mass = 0.5;
    app.update();
    assert_ne!(body(&app, a), old);
    near(position(&app, a).x, 5.02);
    near(app.world().get::<Velocity>(a).unwrap().linear.x, 2.0);
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .joints
        .contains(joint));
}
