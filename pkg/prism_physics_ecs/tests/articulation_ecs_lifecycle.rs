//! Public native ECS articulation lifecycle and publication-sequence contracts.
//! XPBD owns simulation; this bridge adds no local solver or duplicate ABA step.

use std::time::Duration;

use glam::{Quat, Vec3};
use prism_app::{App, FixedPostUpdate, FixedUpdate, TimeUpdateStrategy};
use prism_ecs::prelude::*;
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuildError, ArticulationHandle, ArticulationJoint,
    ArticulationLink, JointMotor, LinkContactImpulse, SpatialInertia, SpatialMotion,
    SpatialTransform,
};
use prism_physics_ecs::articulation::{
    ArticulationBinding, ArticulationBindings, ArticulationDiagnostic, ArticulationDiagnostics,
    ArticulationIssue, ArticulationSnapshot, PrismArticulationPlugin,
};
use prism_physics_ecs::{
    PhysicsDiagnostics, PhysicsSystems, PhysicsWorldResource, PrismPhysicsPlugin,
};

fn app() -> App {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins((PrismPhysicsPlugin, PrismArticulationPlugin));
    let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
    physics.config.gravity = Vec3::ZERO;
    physics.config.default_substeps = 1;
    app
}

fn authored() -> ArticulationBinding {
    let mut binding = ArticulationBinding::default();
    binding.links.push(ArticulationLink {
        parent: None,
        x_tree: SpatialTransform::IDENTITY,
        inertia: SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.2)),
        joint: ArticulationJoint::revolute(Vec3::Z),
    });
    binding.velocities = vec![0.7];
    binding
}

fn handle(app: &App, entity: Entity) -> ArticulationHandle {
    app.world()
        .resource::<ArticulationBindings>()
        .articulation(entity)
        .unwrap()
}

fn snapshot(app: &App, entity: Entity) -> &ArticulationSnapshot {
    app.world().get::<ArticulationSnapshot>(entity).unwrap()
}

fn has_issue(app: &App, entity: Option<Entity>, issue: ArticulationIssue) -> bool {
    app.world()
        .resource::<ArticulationDiagnostics>()
        .issues
        .contains(&ArticulationDiagnostic { entity, issue })
}

#[test]
fn complete_lifecycle_edits_remove_readd_remove_and_despawn() {
    let mut app = app();
    let initial = authored();
    let entity = app.world_mut().spawn((initial.clone(),));
    app.update();
    let first = handle(&app, entity);
    assert_eq!(
        app.world().resource::<ArticulationBindings>().entity(first),
        Some(entity)
    );
    assert_eq!(snapshot(&app, entity).version, 1);
    assert_eq!(
        app.world().get::<ArticulationBinding>(entity),
        Some(&initial)
    );
    app.update();
    assert_eq!(handle(&app, entity), first);
    assert_eq!(snapshot(&app, entity).version, 2);

    let mut edited = initial.clone();
    edited.base_pose = Isometry::from_translation(Vec3::new(2.0, 3.0, 4.0));
    edited.positions = vec![0.4];
    edited.velocities = vec![0.0];
    app.world_mut().insert(entity, (edited.clone(),));
    app.update();
    let second = handle(&app, entity);
    assert_ne!(first, second);
    assert_eq!(snapshot(&app, entity).version, 1);
    assert_eq!(snapshot(&app, entity).positions, vec![0.4]);
    assert_eq!(snapshot(&app, entity).base_pose, edited.base_pose);
    assert_eq!(
        app.world().resource::<ArticulationBindings>().entity(first),
        None
    );
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(first));

    assert!(app.world_mut().remove::<ArticulationBinding>(entity));
    app.world_mut().insert(entity, (edited.clone(),));
    app.update();
    let third = handle(&app, entity);
    assert_ne!(
        third, second,
        "remove/readd with identical authoring is a new lifetime"
    );
    assert_eq!(snapshot(&app, entity).version, 1);

    assert!(app.world_mut().remove::<ArticulationBinding>(entity));
    app.update();
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
    assert!(app
        .world()
        .resource::<ArticulationBindings>()
        .articulation(entity)
        .is_none());
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(third));
    app.world_mut().insert(entity, (initial.clone(),));
    app.update();
    let fourth = handle(&app, entity);
    assert!(app.world_mut().despawn(entity));
    // Exercise ECS generation reuse too, without identifying an entity by index.
    let replacement = app.world_mut().spawn((initial,));
    assert_ne!(replacement, entity);
    app.update();
    let fresh = handle(&app, replacement);
    let bindings = app.world().resource::<ArticulationBindings>();
    assert_eq!(bindings.articulation(entity), None);
    assert_eq!(bindings.entity(fourth), None);
    assert_eq!(bindings.entity(fresh), Some(replacement));
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .articulations
            .len(),
        1
    );
}

#[test]
fn equal_value_authoring_writes_preserve_runtime_and_publication_sequence() {
    let mut app = app();
    let initial = authored();
    let entity = app.world_mut().spawn((initial.clone(),));
    app.update();
    let managed = handle(&app, entity);
    let mut expected = initial.build().unwrap();
    expected.set_position(0, 0.35);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .articulations
        .get_mut(managed)
        .unwrap()
        .set_position(0, 0.35);

    for tick in 2..=5 {
        match tick {
            2 => {
                assert!(app.world_mut().insert(entity, (initial.clone(),)));
            }
            3 => {
                // A mutable write stamps a new change tick even if its value
                // is equal. That alone must not restart the simulation.
                app.world_mut()
                    .get_mut::<ArticulationBinding>(entity)
                    .unwrap()
                    .velocities[0] = initial.velocities[0];
            }
            _ => {}
        }
        expected.step(0.01);
        app.update();
        let actual = snapshot(&app, entity);
        assert_eq!(handle(&app, entity), managed);
        assert_eq!(actual.positions, expected.positions());
        assert_eq!(actual.velocities, expected.velocities());
        assert_eq!((actual.tick, actual.version), (tick, tick));
        assert!(app
            .world()
            .resource::<ArticulationDiagnostics>()
            .issues
            .is_empty());
    }
}

#[test]
fn snapshot_removal_or_version_edits_do_not_reset_internal_sequence() {
    let mut app = app();
    let initial = authored();
    let entity = app.world_mut().spawn((initial.clone(),));
    app.update();
    let managed = handle(&app, entity);
    app.world_mut()
        .get_mut::<ArticulationSnapshot>(entity)
        .unwrap()
        .version = u64::MAX;
    app.update();
    assert_eq!(snapshot(&app, entity).version, 2);
    assert_eq!(handle(&app, entity), managed);
    assert!(app.world_mut().remove::<ArticulationSnapshot>(entity));
    app.update();
    assert_eq!(snapshot(&app, entity).version, 3);
    assert_eq!(handle(&app, entity), managed);

    assert!(app.world_mut().remove::<ArticulationBinding>(entity));
    app.world_mut().insert(entity, (initial,));
    app.update();
    assert_ne!(handle(&app, entity), managed);
    assert_eq!(snapshot(&app, entity).version, 1);
    assert_eq!(snapshot(&app, entity).tick, 4);
}

#[test]
fn public_resource_removal_replacement_and_edits_do_not_reset_runtime_sequences() {
    let mut app = app();
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    let managed = handle(&app, entity);
    let foreign = app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .spawn_articulation(authored().build().unwrap());
    for tick in 2..=7 {
        match tick {
            2 => {
                app.world_mut().remove_resource::<ArticulationBindings>();
            }
            3 => {
                app.world_mut()
                    .insert_resource(ArticulationBindings::default());
            }
            4 => {
                app.world_mut().remove_resource::<ArticulationDiagnostics>();
            }
            5 => {
                app.world_mut()
                    .insert_resource(ArticulationDiagnostics::default());
            }
            6 | 7 => {
                app.world_mut()
                    .resource_mut::<ArticulationDiagnostics>()
                    .tick = if tick == 6 { u64::MAX } else { 0 };
            }
            _ => unreachable!(),
        }
        let before = snapshot(&app, entity).positions[0];
        app.update();
        let state = snapshot(&app, entity);
        assert_eq!((state.tick, state.version), (tick, tick));
        assert_eq!(state.handle, managed);
        assert!(state.positions[0] > before, "live state must not restart");
        let diagnostics = app.world().resource::<ArticulationDiagnostics>();
        assert_eq!(diagnostics.tick, tick);
        assert!(diagnostics.issues.is_empty());
        let bindings = app.world().resource::<ArticulationBindings>();
        assert_eq!(bindings.articulation(entity), Some(managed));
        assert_eq!(bindings.entity(managed), Some(entity));
        assert_eq!(bindings.entity(foreign), None);
        let physics = app.world().resource::<PhysicsWorldResource>();
        assert!(physics.articulations.contains(managed));
        assert!(physics.articulations.contains(foreign));
        assert_eq!(physics.articulations.len(), 2);
    }

    // Removing the public views must not lose the private cleanup record.
    app.world_mut().remove_resource::<ArticulationBindings>();
    app.world_mut().remove_resource::<ArticulationDiagnostics>();
    assert!(app.world_mut().remove::<ArticulationBinding>(entity));
    app.update();
    assert_eq!(app.world().resource::<ArticulationDiagnostics>().tick, 8);
    let bindings = app.world().resource::<ArticulationBindings>();
    assert_eq!(bindings.articulation(entity), None);
    assert_eq!(bindings.entity(managed), None);
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!(!physics.articulations.contains(managed));
    assert!(physics.articulations.contains(foreign));
    assert_eq!(physics.articulations.len(), 1);
}

#[test]
fn stale_handle_recovers_without_removing_foreign_reused_generation() {
    let mut app = app();
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    let stale = handle(&app, entity);
    let foreign = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        assert!(physics.remove_articulation(stale));
        physics.spawn_articulation(authored().build().unwrap())
    };
    assert_eq!(foreign.index(), stale.index());
    assert_ne!(foreign.generation(), stale.generation());
    app.update();
    let fresh = handle(&app, entity);
    assert_ne!(fresh, stale);
    assert_ne!(fresh, foreign);
    assert!(has_issue(
        &app,
        Some(entity),
        ArticulationIssue::StaleHandle(stale)
    ));
    let bindings = app.world().resource::<ArticulationBindings>();
    assert_eq!(bindings.entity(stale), None);
    assert_eq!(bindings.entity(foreign), None);
    assert_eq!(bindings.entity(fresh), Some(entity));
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(foreign));
    assert_eq!(snapshot(&app, entity).version, 1);
    assert!(app.world_mut().despawn(entity));
    app.update();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!(physics.articulations.contains(foreign));
    assert_eq!(physics.articulations.len(), 1);
}

#[test]
fn fixed_accumulator_zero_multiple_ticks_and_public_ordering() {
    #[derive(Resource, Default)]
    struct Samples(Vec<(u64, f32)>);
    fn author(world: &mut World) {
        let entities: Vec<_> = world
            .query::<(Entity, &ArticulationBinding)>()
            .iter(world)
            .map(|(entity, _)| entity)
            .collect();
        for entity in entities {
            world
                .get_mut::<ArticulationBinding>(entity)
                .unwrap()
                .velocities = vec![1.0];
        }
    }
    fn observe(world: &mut World) {
        let sample = world
            .query::<&ArticulationSnapshot>()
            .iter(world)
            .next()
            .map(|s| (s.tick, s.positions[0]))
            .unwrap();
        world.resource_mut::<Samples>().0.push(sample);
    }
    let mut app = app();
    app.init_resource::<Samples>()
        .add_systems(FixedUpdate, author.before(PhysicsSystems::PreSync))
        .add_systems(FixedPostUpdate, observe)
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(3)));
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(entity),
        None
    );
    assert_eq!(app.world().resource::<ArticulationDiagnostics>().tick, 0);
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(27)));
    app.update();
    let samples = &app.world().resource::<Samples>().0;
    assert_eq!(samples.len(), 3);
    for (index, &(tick, position)) in samples.iter().enumerate() {
        assert_eq!(tick, index as u64 + 1);
        assert!((position - (index + 1) as f32 * 0.01).abs() < 1.0e-5);
    }
    assert_eq!(snapshot(&app, entity).version, 3);
}

#[test]
fn xpbd_advances_exactly_once_per_substep_and_preserves_resource_side_state() {
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for substeps in [1, 2, 7] {
            let mut app = app();
            app.world_mut()
                .resource_mut::<PhysicsWorldResource>()
                .config
                .default_substeps = substeps;
            let mut config = authored();
            config.base = base;
            config.base_inertia = SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::ONE);
            if base == ArticulationBase::Floating {
                config.base_velocity = SpatialMotion::new(Vec3::new(0.1, 0.2, 0.0), Vec3::X);
            }
            let mut expected = config.build().unwrap();
            let entity = app.world_mut().spawn((config.clone(),));
            app.update();
            for _ in 0..substeps {
                expected.step(0.01 / substeps as f32);
            }
            let managed = handle(&app, entity);
            let impulse = LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: Vec3::Y * 0.1,
            };
            assert!(app
                .world_mut()
                .resource_mut::<PhysicsWorldResource>()
                .queue_articulation_contact_impulse(managed, impulse));
            // This mutation is not an authoring edit and must not reset on sync.
            expected.set_position(0, 0.25);
            let _ = expected.apply_contact_impulses(&[impulse]);
            app.world_mut()
                .resource_mut::<PhysicsWorldResource>()
                .articulations
                .get_mut(managed)
                .unwrap()
                .set_position(0, 0.25);
            for tick in 2..=4 {
                app.update();
                for _ in 0..substeps {
                    expected.step(0.01 / substeps as f32);
                }
                let actual = snapshot(&app, entity);
                assert_eq!(actual.handle, managed);
                assert_eq!(actual.positions, expected.positions());
                assert_eq!(actual.velocities, expected.velocities());
                assert_eq!(actual.base_pose, expected.base_pose());
                assert_eq!(actual.base_velocity, expected.base_velocity());
                assert_eq!(actual.link_poses, expected.forward_kinematics());
                assert_eq!(actual.link_velocities, expected.link_velocities());
                assert_eq!(actual.tick, tick);
                assert_eq!(actual.version, tick);
                assert_eq!(actual.simulated_dt, 0.01);
                assert_eq!(
                    app.world().get::<ArticulationBinding>(entity),
                    Some(&config)
                );
            }
        }
    }
}

#[test]
fn invalid_edit_stops_old_runtime_reports_exact_cause_and_repair_recovers() {
    let mut app = app();
    let valid = authored();
    let entity = app.world_mut().spawn((valid.clone(),));
    app.update();
    let old = handle(&app, entity);
    app.world_mut()
        .get_mut::<ArticulationBinding>(entity)
        .unwrap()
        .links[0]
        .joint = ArticulationJoint::revolute(Vec3::ZERO);
    app.update();
    let error = ArticulationIssue::BuildFailed(ArticulationBuildError::InvalidAxis { link: 0 });
    assert!(has_issue(&app, Some(entity), error.clone()));
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(entity),
        None
    );
    assert_eq!(
        app.world().resource::<ArticulationBindings>().entity(old),
        None
    );
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(old));
    app.update();
    assert!(has_issue(&app, Some(entity), error));
    assert_eq!(
        app.world()
            .resource::<ArticulationDiagnostics>()
            .issues
            .len(),
        1
    );
    app.world_mut().insert(entity, (valid,));
    app.update();
    assert_ne!(handle(&app, entity), old);
    assert_eq!(snapshot(&app, entity).version, 1);
    assert!(app
        .world()
        .resource::<ArticulationDiagnostics>()
        .issues
        .is_empty());
}

#[test]
fn invalid_authoring_fields_and_core_topology_errors_are_not_silently_accepted() {
    let invalid = |field, link| ArticulationIssue::InvalidAuthoring { field, link };
    let mut cases = Vec::new();
    let mut config = authored();
    config.positions = vec![0.0, 1.0];
    cases.push((config, invalid("positions", None)));
    let mut config = authored();
    config.velocities[0] = f32::NAN;
    cases.push((config, invalid("velocities", None)));
    let mut config = authored();
    config.gravity.x = f32::INFINITY;
    cases.push((config, invalid("gravity", None)));
    let mut config = authored();
    config.base_pose.rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    cases.push((config, invalid("base_pose", None)));
    let mut config = authored();
    config.base_velocity.linear = Vec3::X;
    cases.push((config, invalid("base_velocity", None)));
    let mut config = authored();
    config.links[0].x_tree.translation.x = f32::NAN;
    cases.push((config, invalid("x_tree", Some(0))));
    let mut config = authored();
    config.links[0].joint.damping = -1.0;
    cases.push((config, invalid("joint_dynamics", Some(0))));
    let mut config = authored();
    config.links[0].joint.motor = Some(JointMotor::new(
        Vec3::ZERO,
        Vec3::ZERO,
        f32::NAN,
        0.0,
        Vec3::ONE,
    ));
    cases.push((config, invalid("motor", Some(0))));
    let mut config = authored();
    config.links[0].parent = Some(0);
    cases.push((
        config,
        ArticulationIssue::BuildFailed(ArticulationBuildError::ParentMustPrecedeChild {
            link: 0,
            parent: 0,
        }),
    ));
    let mut config = authored();
    config.base = ArticulationBase::Floating;
    cases.push((
        config,
        ArticulationIssue::BuildFailed(ArticulationBuildError::InvalidBaseInertia),
    ));
    for (config, issue) in cases {
        assert_eq!(config.build().unwrap_err(), issue);
        let mut app = app();
        let entity = app.world_mut().spawn((config,));
        app.update();
        assert!(has_issue(&app, Some(entity), issue));
        assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
        assert!(app
            .world()
            .resource::<PhysicsWorldResource>()
            .articulations
            .is_empty());
    }
}

#[test]
fn invalid_shared_configuration_reports_rejected_step_and_keeps_snapshot_unstepped() {
    let mut app = app();
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    let before = snapshot(&app, entity).positions.clone();
    let managed = handle(&app, entity);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 0;
    app.update();
    assert!(has_issue(&app, None, ArticulationIssue::StepRejected));
    assert_eq!(snapshot(&app, entity).positions, before);
    assert_eq!(snapshot(&app, entity).simulated_dt, 0.0);
    assert_eq!(snapshot(&app, entity).version, 2);
    assert_eq!(handle(&app, entity), managed);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 2;
    app.update();
    assert!(snapshot(&app, entity).positions[0] > before[0]);
    assert!(app
        .world()
        .resource::<ArticulationDiagnostics>()
        .issues
        .is_empty());
}

#[test]
fn invalid_runtime_output_is_retired_then_rebuilt_from_authoring() {
    let mut app = app();
    let mut config = authored();
    // The authored state is valid at its initial angular velocity, while a
    // later finite-but-extreme velocity overflows the derived link velocity.
    // This exercises output validation without bypassing the core setters.
    config.links.push(ArticulationLink {
        parent: Some(0),
        x_tree: SpatialTransform::new(Quat::IDENTITY, Vec3::new(4.0, 0.0, 0.0)),
        inertia: SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.2)),
        joint: ArticulationJoint::fixed(),
    });
    let entity = app.world_mut().spawn((config,));
    app.update();
    let corrupt = handle(&app, entity);
    assert!(app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .articulations
        .get_mut(corrupt)
        .unwrap()
        .set_velocity(0, f32::MAX));
    // Stop the core before it can consume the intentionally extreme velocity.
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 0;
    app.update();
    assert!(
        has_issue(
            &app,
            Some(entity),
            ArticulationIssue::InvalidOutput(corrupt)
        ),
        "unexpected diagnostics: {:?}",
        app.world().resource::<ArticulationDiagnostics>().issues
    );
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(entity),
        None
    );
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(corrupt));
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 1;
    app.update();
    assert_ne!(handle(&app, entity), corrupt);
    assert_eq!(snapshot(&app, entity).version, 1);
    assert!(snapshot(&app, entity).positions[0].is_finite());
}

#[test]
fn breakage_is_handed_off_without_draining_core_events_or_rebuilding_aba() {
    let mut app = app();
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    let managed = handle(&app, entity);
    assert!(app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .request_articulation_break(managed, 0));
    app.update();
    let converted = app
        .world()
        .resource::<ArticulationBindings>()
        .handoff(entity)
        .unwrap()
        .clone();
    assert_eq!(converted.report.articulation, managed);
    assert_eq!(converted.report.broken_link, 0);
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
    assert!(has_issue(
        &app,
        Some(entity),
        ArticulationIssue::Degraded {
            handle: managed,
            link: 0
        }
    ));
    app.update();
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(entity),
        None
    );
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .handoff(entity),
        Some(&converted)
    );
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(managed));
    let events = app
        .world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .drain_articulation_degradations();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].articulation, managed);
    assert_eq!(events[0].degradation.broken_link, 0);
}

#[test]
fn resources_remain_visible_to_snapshot_observers() {
    #[derive(Resource, Default)]
    struct Observed(usize);
    let mut app = app();
    app.init_resource::<Observed>();
    app.world_mut()
        .observe::<ArticulationSnapshot, _>(LifecycleEvent::Insert, |ctx| {
            assert!(ctx.world.contains_resource::<PhysicsWorldResource>());
            assert!(ctx.world.contains_resource::<PhysicsDiagnostics>());
            assert!(ctx.world.contains_resource::<ArticulationBindings>());
            assert!(ctx.world.contains_resource::<ArticulationDiagnostics>());
            ctx.world.resource_mut::<Observed>().0 += 1;
        });
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    app.update();
    assert_eq!(app.world().resource::<Observed>().0, 2);
    assert_eq!(snapshot(&app, entity).version, 2);
}

#[test]
fn snapshot_observer_remove_readd_cannot_publish_or_keep_the_old_lifetime() {
    #[derive(Resource, Default)]
    struct Replaced(bool);
    let mut app = app();
    app.init_resource::<Replaced>();
    let a = app.world_mut().spawn((authored(),));
    let b = app.world_mut().spawn((authored(),));
    let (first, later) = if a < b { (a, b) } else { (b, a) };
    app.update();
    let old = handle(&app, later);
    app.world_mut()
        .observe::<ArticulationSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
            if ctx.entity == first && !ctx.world.resource::<Replaced>().0 {
                ctx.world.resource_mut::<Replaced>().0 = true;
                assert!(ctx.world.remove::<ArticulationBinding>(later));
                assert!(ctx.world.insert(later, (authored(),)));
            }
        });
    app.update();
    assert_eq!(snapshot(&app, first).version, 2);
    assert!(app.world().get::<ArticulationSnapshot>(later).is_none());
    let bindings = app.world().resource::<ArticulationBindings>();
    assert_eq!(bindings.articulation(later), None);
    assert_eq!(bindings.entity(old), None);
    assert!(!app
        .world()
        .resource::<PhysicsWorldResource>()
        .articulations
        .contains(old));

    app.update();
    assert_ne!(handle(&app, later), old);
    assert_eq!(snapshot(&app, later).version, 1);
    assert_eq!(snapshot(&app, later).tick, 3);
}

#[test]
fn publication_repairs_observer_deleted_or_replaced_public_views_between_entities() {
    #[derive(Resource, Default)]
    struct Calls(usize);
    for replace in [false, true] {
        let mut app = app();
        app.init_resource::<Calls>();
        let entities = [
            app.world_mut().spawn((authored(),)),
            app.world_mut().spawn((authored(),)),
        ];
        app.world_mut()
            .observe::<ArticulationSnapshot, _>(LifecycleEvent::Insert, move |ctx| {
                assert!(ctx.world.contains_resource::<ArticulationBindings>());
                assert!(ctx.world.contains_resource::<ArticulationDiagnostics>());
                ctx.world.remove_resource::<ArticulationBindings>().unwrap();
                ctx.world
                    .remove_resource::<ArticulationDiagnostics>()
                    .unwrap();
                if replace {
                    ctx.world.insert_resource(ArticulationBindings::default());
                    ctx.world
                        .insert_resource(ArticulationDiagnostics::default());
                }
                ctx.world.resource_mut::<Calls>().0 += 1;
            });
        for tick in 1..=3 {
            app.update();
            assert_eq!(app.world().resource::<Calls>().0, tick as usize * 2);
            assert_eq!(
                app.world()
                    .resource::<PhysicsWorldResource>()
                    .articulations
                    .len(),
                2
            );
            for entity in entities {
                let state = snapshot(&app, entity);
                assert_eq!((state.tick, state.version), (tick, tick));
                assert_eq!(handle(&app, entity), state.handle);
                assert_eq!(
                    app.world()
                        .resource::<ArticulationBindings>()
                        .entity(state.handle),
                    Some(entity)
                );
            }
            for resource in ["ArticulationBindings", "ArticulationDiagnostics"] {
                assert!(has_issue(
                    &app,
                    None,
                    ArticulationIssue::ObserverResourceRemoved { resource }
                ));
            }
            assert!(!has_issue(&app, None, ArticulationIssue::LifecycleDeferred));
            assert!(!has_issue(&app, None, ArticulationIssue::MissingPhysics));
        }
    }
}

#[test]
fn missing_base_plugin_is_an_explicit_integration_diagnostic() {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins(PrismArticulationPlugin);
    let entity = app.world_mut().spawn((authored(),));
    app.update();
    assert!(has_issue(&app, None, ArticulationIssue::MissingPhysics));
    assert_eq!(
        app.world()
            .resource::<ArticulationBindings>()
            .articulation(entity),
        None
    );
    assert!(app.world().get::<ArticulationSnapshot>(entity).is_none());
}
