//! ECS integration tests for deterministic force-field and aero component lifecycle.
use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::Entity;
use prism_physics_ecs::core::dynamics::field::{
    Field, FieldEffect, FieldFalloff, FieldShape, FieldTargetMask,
};
use prism_physics_ecs::*;
use prism_transform::Transform;
use std::time::Duration;

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

fn field(force: Vec3) -> Field {
    Field {
        id: core::dynamics::field::FieldId(0),
        shape: FieldShape::Global,
        falloff: FieldFalloff::Constant,
        effect: FieldEffect::ConstantForce(force),
        target_mask: FieldTargetMask::ALL,
    }
}

#[test]
fn field_registration_order_update_identity_and_removal_are_deterministic() {
    let mut app = app();
    let first = app.world_mut().spawn(ForceField(field(Vec3::X)));
    let second = app.world_mut().spawn(ForceField(field(Vec3::Y * 2.0)));
    let body = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
    ));

    app.update();
    let (first_id, second_id) = {
        let bindings = app.world().resource::<FieldBindings>();
        (
            bindings.field(first).unwrap(),
            bindings.field(second).unwrap(),
        )
    };
    assert!(first_id < second_id);
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .fields
            .fields()
            .iter()
            .map(|field| field.id)
            .collect::<Vec<_>>(),
        vec![first_id, second_id]
    );
    let velocity = app.world().get::<Velocity>(body).unwrap().linear;
    assert!((velocity.x - 0.01).abs() < 1e-5);
    assert!((velocity.y - 0.02).abs() < 1e-5);

    app.world_mut().get_mut::<ForceField>(first).unwrap().0 = field(Vec3::X * 3.0);
    app.update();
    assert_eq!(
        app.world().resource::<FieldBindings>().field(first),
        Some(first_id)
    );
    let fields = &app.world().resource::<PhysicsWorldResource>().fields;
    assert_eq!(fields.fields()[0].id, first_id);
    assert_eq!(
        fields.fields()[0].effect,
        FieldEffect::ConstantForce(Vec3::X * 3.0)
    );

    app.world_mut().remove::<ForceField>(second);
    app.update();
    assert_eq!(app.world().resource::<FieldBindings>().field(second), None);
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .fields
            .fields()
            .len(),
        1
    );
    app.world_mut().despawn(first);
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .fields
        .fields()
        .is_empty());
}

#[test]
fn field_equal_id_range_preserves_duplicate_manual_id_matching_and_replacement() {
    let mut app = app();
    let lower = app.world_mut().spawn(ForceField(field(Vec3::Z)));
    let source = app.world_mut().spawn(ForceField(field(Vec3::X)));
    app.update();
    let id = app
        .world()
        .resource::<FieldBindings>()
        .field(source)
        .unwrap();
    let lower_id = app
        .world()
        .resource::<FieldBindings>()
        .field(lower)
        .unwrap();
    assert!(lower_id < id);
    let with_id = |force| Field { id, ..field(force) };
    let high = Field {
        id: core::dynamics::field::FieldId(id.0 + 10),
        ..field(Vec3::Y * 9.0)
    };
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.fields.remove(id).unwrap();
        // The first matching ID has the wrong value; the second is exact.
        physics.fields.insert(with_id(Vec3::Y));
        physics.fields.insert(with_id(Vec3::X));
        physics.fields.insert(high);
    }
    let accepted = app
        .world()
        .resource::<PhysicsWorldResource>()
        .fields
        .fields()
        .to_vec();
    for _ in 0..3 {
        app.update();
        assert_eq!(
            app.world()
                .resource::<PhysicsWorldResource>()
                .fields
                .fields(),
            accepted.as_slice()
        );
    }

    app.world_mut().get_mut::<ForceField>(source).unwrap().0 = field(Vec3::X * 3.0);
    app.update();
    // Preserve the existing core contract: replacement removes only the first
    // equal-ID entry and appends the new value, not all manual duplicates.
    let replaced = vec![
        field_with_id(Vec3::Z, lower_id),
        with_id(Vec3::X),
        with_id(Vec3::X * 3.0),
        high,
    ];
    for _ in 0..3 {
        assert_eq!(
            app.world()
                .resource::<PhysicsWorldResource>()
                .fields
                .fields(),
            replaced.as_slice()
        );
        assert_eq!(
            app.world().resource::<FieldBindings>().field(source),
            Some(id)
        );
        app.update();
    }

    // A live ID alone is insufficient when NONE of its values matches.
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.fields.remove(id).unwrap();
        physics.fields.remove(id).unwrap();
        physics.fields.insert(with_id(Vec3::Y));
    }
    app.update();
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .fields
            .fields(),
        &[
            field_with_id(Vec3::Z, lower_id),
            with_id(Vec3::X * 3.0),
            high
        ]
    );
    // An absent ID is repaired as well; it must not match the higher neighbor.
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .fields
        .remove(id)
        .unwrap();
    app.update();
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .fields
            .fields(),
        &[
            field_with_id(Vec3::Z, lower_id),
            with_id(Vec3::X * 3.0),
            high
        ]
    );
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .fields
        .insert(with_id(Vec3::Y * 4.0));
    app.world_mut().remove::<ForceField>(source);
    app.update();
    assert_eq!(app.world().resource::<FieldBindings>().field(source), None);
    // Retirement still removes only the first duplicate, leaving the manual
    // registration with the same ID intact and no longer managed by this ECS.
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .fields
            .fields(),
        &[
            field_with_id(Vec3::Z, lower_id),
            with_id(Vec3::Y * 4.0),
            high
        ]
    );
}

fn field_with_id(force: Vec3, id: core::dynamics::field::FieldId) -> Field {
    Field { id, ..field(force) }
}

#[test]
fn aero_functional_elements_bind_to_core_config_and_follow_component_lifecycle() {
    let mut app = app();
    let entity = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
        RigidBodyAero(core::AeroBodyConfig::default()),
        AeroFunctionalComponents {
            thrusters: vec![core::Thruster::new(Vec3::ZERO, Vec3::X, 10.0)],
            ..AeroFunctionalComponents::default()
        },
    ));

    app.update();
    let body = app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .unwrap();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert_eq!(physics.aero_body(body).unwrap().thrusters.len(), 1);
    assert!((physics.bodies.linear_velocity(body).unwrap().x - 0.1).abs() < 1e-5);

    app.world_mut()
        .get_mut::<AeroFunctionalComponents>(entity)
        .unwrap()
        .thrusters[0]
        .thrust = 20.0;
    app.update();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!((physics.bodies.linear_velocity(body).unwrap().x - 0.3).abs() < 1e-5);

    app.world_mut().remove::<AeroFunctionalComponents>(entity);
    app.update();
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .aero_body(body),
        Some(&core::AeroBodyConfig::default())
    );
}

fn spawn_body(app: &mut App) -> Entity {
    app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
    ))
}

fn handle(app: &App, entity: Entity) -> core::BodyHandle {
    app.world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .unwrap()
}

fn aero(app: &App, entity: Entity) -> core::AeroBodyConfig {
    app.world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle(app, entity))
        .unwrap()
        .clone()
}

fn thruster(thrust: f32) -> core::Thruster {
    core::Thruster::new(Vec3::ZERO, Vec3::X, thrust)
}

fn surface() -> core::AeroSurface {
    core::AeroSurface::new(Vec3::new(0.0, 2.0, 0.0), Vec3::Y, Vec3::X, 0.0)
}

fn volume() -> core::BuoyancyVolume {
    core::BuoyancyVolume::box_volume(Vec3::ZERO, Vec3::ZERO, core::aero::FluidKind::Water)
}

fn issues(app: &App) -> &[AeroDiagnostic] {
    &app.world().resource::<AeroDiagnostics>().issues
}

#[test]
fn independent_families_edit_remove_and_despawn_without_an_aggregate() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let panel = app.world_mut().spawn((
        AeroSurfaceComponent {
            target,
            surface: surface(),
        },
        Transform::from_xyz(100.0, 200.0, 300.0),
    ));
    let engine = app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: thruster(10.0),
    });
    let float = app.world_mut().spawn(AeroBuoyancyComponent {
        target,
        volume: volume(),
    });
    app.update();
    assert_eq!(aero(&app, target).surfaces, vec![surface()]);
    assert_eq!(aero(&app, target).thrusters, vec![thruster(10.0)]);
    assert_eq!(aero(&app, target).buoyancy, vec![volume()]);
    assert!(issues(&app).is_empty());

    let mut edited_surface = surface();
    edited_surface.local_position = Vec3::Z;
    let mut edited_volume = volume();
    edited_volume.local_center = Vec3::Y;
    app.world_mut()
        .get_mut::<AeroSurfaceComponent>(panel)
        .unwrap()
        .surface = edited_surface;
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(engine)
        .unwrap()
        .thruster = thruster(-20.0);
    app.world_mut()
        .get_mut::<AeroBuoyancyComponent>(float)
        .unwrap()
        .volume = edited_volume;
    app.update();
    assert_eq!(aero(&app, target).surfaces, vec![edited_surface]);
    assert_eq!(aero(&app, target).thrusters, vec![thruster(-20.0)]);
    assert_eq!(aero(&app, target).buoyancy, vec![edited_volume]);

    app.world_mut().remove::<AeroSurfaceComponent>(panel);
    app.update();
    assert!(aero(&app, target).surfaces.is_empty());
    assert_eq!(aero(&app, target).thrusters.len(), 1);
    assert_eq!(aero(&app, target).buoyancy.len(), 1);
    app.world_mut().remove::<AeroThrusterComponent>(engine);
    app.update();
    assert!(aero(&app, target).thrusters.is_empty());
    assert_eq!(aero(&app, target).buoyancy.len(), 1);
    app.world_mut().despawn(float);
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle(&app, target))
        .is_none());
}

#[test]
fn composition_is_additive_ordered_and_applies_once_per_substep() {
    for substeps in [1, 4, 8] {
        let mut app = app();
        app.world_mut()
            .resource_mut::<PhysicsWorldResource>()
            .config
            .default_substeps = substeps;
        let target = spawn_body(&mut app);
        let mut base = core::AeroBodyConfig::default();
        base.model.air_density = 2.0;
        base.thrusters = vec![thruster(1.0), thruster(2.0)];
        app.world_mut().insert(
            target,
            (
                RigidBodyAero(base.clone()),
                AeroFunctionalComponents {
                    thrusters: vec![thruster(3.0), thruster(4.0)],
                    ..Default::default()
                },
                AeroThrusterComponent {
                    target,
                    thruster: thruster(5.0),
                },
            ),
        );
        let first = app.world_mut().spawn((Transform::IDENTITY,));
        let second = app.world_mut().spawn(AeroThrusterComponent {
            target,
            thruster: thruster(7.0),
        });
        // Insert in reverse order; query/archetype order must not decide summation.
        app.world_mut().insert(
            first,
            (AeroThrusterComponent {
                target,
                thruster: thruster(6.0),
            },),
        );
        for tick in 1..=3 {
            app.update();
            assert_eq!(
                aero(&app, target).thrusters,
                (1..=7).map(|n| thruster(n as f32)).collect::<Vec<_>>()
            );
            assert_eq!(aero(&app, target).model, base.model);
            let physics = app.world().resource::<PhysicsWorldResource>();
            assert_eq!(
                physics
                    .aero_bodies
                    .iter()
                    .filter(|entry| entry.body == handle(&app, target))
                    .count(),
                1
            );
            assert!(
                (app.world().get::<Velocity>(target).unwrap().linear.x - tick as f32 * 0.28).abs()
                    < 1e-5
            );
        }
        // Identical parts remain intentional distinct contributions.
        app.world_mut()
            .get_mut::<AeroThrusterComponent>(second)
            .unwrap()
            .thruster = thruster(6.0);
        app.world_mut().remove::<RigidBodyAero>(target);
        app.update();
        assert_eq!(
            aero(&app, target).thrusters,
            [3., 4., 5., 6., 6.]
                .into_iter()
                .map(thruster)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            aero(&app, target).model,
            core::AeroBodyConfig::default().model
        );
        app.world_mut().remove::<AeroFunctionalComponents>(target);
        app.update();
        assert_eq!(
            aero(&app, target).thrusters,
            [5., 6., 6.].into_iter().map(thruster).collect::<Vec<_>>()
        );
    }
}

#[test]
fn retarget_and_target_removal_release_only_the_previous_generational_body() {
    let mut app = app();
    let first = spawn_body(&mut app);
    let second = spawn_body(&mut app);
    let part = app.world_mut().spawn(AeroThrusterComponent {
        target: first,
        thruster: thruster(10.0),
    });
    app.update();
    let old = handle(&app, first);
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(part)
        .unwrap()
        .target = second;
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(old)
        .is_none());
    assert_eq!(aero(&app, second).thrusters, vec![thruster(10.0)]);
    assert!((app.world().get::<Velocity>(first).unwrap().linear.x - 0.1).abs() < 1e-5);
    assert!((app.world().get::<Velocity>(second).unwrap().linear.x - 0.1).abs() < 1e-5);

    let second_body = handle(&app, second);
    app.world_mut().remove::<RigidBody>(second);
    app.update();
    assert_eq!(issues(&app)[0].issue, AeroIssue::TargetNotRigidBody);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(second_body)
        .is_none());
    app.world_mut().insert(second, (RigidBody::dynamic(),));
    app.update();
    assert_ne!(handle(&app, second), second_body);
    assert_eq!(aero(&app, second).thrusters, vec![thruster(10.0)]);
    assert!(issues(&app).is_empty());
}

#[test]
fn rebuild_and_external_body_removal_rebind_parts_to_live_handles_only() {
    let mut app = app();
    let target = spawn_body(&mut app);
    app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: thruster(10.0),
    });
    app.update();
    let old = handle(&app, target);
    app.world_mut()
        .get_mut::<RigidBody>(target)
        .unwrap()
        .mass
        .inv_mass = 0.5;
    app.update();
    let rebuilt = handle(&app, target);
    assert_ne!(old, rebuilt);
    assert_eq!(aero(&app, target).thrusters, vec![thruster(10.0)]);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(old)
        .is_none());
    assert!((app.world().get::<Velocity>(target).unwrap().linear.x - 0.15).abs() < 1e-5);
    let foreign_config = core::AeroBodyConfig::default();
    let foreign = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        assert!(physics.remove_body(rebuilt));
        let body = physics.spawn(core::BodyDesc::static_at(Vec3::ZERO));
        physics
            .register_aero_body(body, foreign_config.clone())
            .unwrap();
        body
    };
    assert_eq!(foreign.index(), rebuilt.index());
    assert_ne!(foreign.generation(), rebuilt.generation());
    app.update();
    assert_ne!(handle(&app, target), rebuilt);
    assert_ne!(handle(&app, target), foreign);
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .aero_body(foreign),
        Some(&foreign_config)
    );
    assert_eq!(aero(&app, target).thrusters, vec![thruster(10.0)]);
    assert!(issues(&app).is_empty());
}

#[test]
fn despawned_target_generation_never_aliases_a_reused_entity_slot() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let part = app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: thruster(10.0),
    });
    app.update();
    let old = handle(&app, target);
    app.world_mut().despawn(target);
    let replacement = spawn_body(&mut app);
    assert_eq!(target.index(), replacement.index());
    assert_ne!(target.generation(), replacement.generation());
    app.update();
    assert_eq!(
        issues(&app),
        &[AeroDiagnostic {
            entity: part,
            target,
            source: AeroSource::Thruster,
            issue: AeroIssue::MissingTarget
        }]
    );
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!(physics.aero_body(old).is_none());
    assert!(physics.aero_body(handle(&app, replacement)).is_none());
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(part)
        .unwrap()
        .target = replacement;
    app.update();
    assert!(issues(&app).is_empty());
    assert_eq!(aero(&app, replacement).thrusters, vec![thruster(10.0)]);
}

#[test]
fn invalid_independent_sources_reject_the_whole_target_and_recover_atomically() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let other = spawn_body(&mut app);
    let panel = app.world_mut().spawn(AeroSurfaceComponent {
        target,
        surface: surface(),
    });
    let engine = app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: thruster(10.0),
    });
    let float = app.world_mut().spawn(AeroBuoyancyComponent {
        target,
        volume: volume(),
    });
    let other_engine = app.world_mut().spawn(AeroThrusterComponent {
        target: other,
        thruster: thruster(1.0),
    });
    app.update();
    let accepted = aero(&app, target);
    app.world_mut()
        .get_mut::<AeroSurfaceComponent>(panel)
        .unwrap()
        .surface
        .control = Some(core::aero::ControlSurface {
        effectiveness: 2.0,
        ..Default::default()
    });
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(engine)
        .unwrap()
        .thruster
        .local_direction = Vec3::ZERO;
    app.world_mut()
        .get_mut::<AeroBuoyancyComponent>(float)
        .unwrap()
        .volume
        .samples_per_axis = 0;
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(other_engine)
        .unwrap()
        .thruster
        .thrust = 2.0;
    app.update();
    assert_eq!(aero(&app, target), accepted);
    assert_eq!(aero(&app, other).thrusters, vec![thruster(2.0)]);
    assert_eq!(
        issues(&app),
        &[
            AeroDiagnostic {
                entity: panel,
                target,
                source: AeroSource::Surface,
                issue: AeroIssue::InvalidParameter {
                    parameter: "surface.control.effectiveness",
                    index: Some(0)
                }
            },
            AeroDiagnostic {
                entity: engine,
                target,
                source: AeroSource::Thruster,
                issue: AeroIssue::InvalidParameter {
                    parameter: "thruster.local_direction",
                    index: Some(0)
                }
            },
            AeroDiagnostic {
                entity: float,
                target,
                source: AeroSource::Buoyancy,
                issue: AeroIssue::InvalidParameter {
                    parameter: "buoyancy.samples_per_axis",
                    index: Some(0)
                }
            },
        ]
    );
    // Removing one source during another invalid edit does not partially commit.
    app.world_mut().remove::<AeroSurfaceComponent>(panel);
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(engine)
        .unwrap()
        .thruster = thruster(20.0);
    app.update();
    assert_eq!(aero(&app, target), accepted);
    assert_eq!(issues(&app).len(), 1);
    app.world_mut()
        .get_mut::<AeroBuoyancyComponent>(float)
        .unwrap()
        .volume = volume();
    app.update();
    assert!(issues(&app).is_empty());
    assert!(aero(&app, target).surfaces.is_empty());
    assert_eq!(aero(&app, target).thrusters, vec![thruster(20.0)]);
    assert_eq!(aero(&app, target).buoyancy, vec![volume()]);
}

#[test]
fn invalid_base_sync_is_rolled_back_but_not_replayed_after_body_rebuild() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let base = core::AeroBodyConfig {
        thrusters: vec![thruster(1.0)],
        ..Default::default()
    };
    app.world_mut()
        .insert(target, (RigidBodyAero(base.clone()),));
    app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: thruster(10.0),
    });
    app.update();
    let accepted = aero(&app, target);
    // Base Sync tries to install this before field Sync; Step must still see
    // exactly the previous full configuration, not the base or sanitized NaNs.
    app.world_mut()
        .get_mut::<RigidBodyAero>(target)
        .unwrap()
        .0
        .model
        .air_density = f32::NAN;
    app.update();
    assert_eq!(aero(&app, target), accepted);
    assert_eq!(
        issues(&app)[0].issue,
        AeroIssue::InvalidParameter {
            parameter: "model.air_density",
            index: None
        }
    );
    assert!((app.world().get::<Velocity>(target).unwrap().linear.x - 0.22).abs() < 1e-5);
    let old = handle(&app, target);
    app.world_mut()
        .get_mut::<RigidBody>(target)
        .unwrap()
        .mass
        .inv_mass = 0.5;
    app.update();
    let rebuilt = handle(&app, target);
    assert_ne!(old, rebuilt);
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert!(physics.aero_body(old).is_none());
    assert!(physics.aero_body(rebuilt).is_none());
    assert!((app.world().get::<Velocity>(target).unwrap().linear.x - 0.22).abs() < 1e-5);
    app.world_mut().get_mut::<RigidBodyAero>(target).unwrap().0 = base;
    app.update();
    assert_eq!(aero(&app, target), accepted);
    assert!(issues(&app).is_empty());
}

#[test]
fn invalid_initial_authoring_and_unavailable_targets_are_reported_without_forces() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let mut invalid = thruster(10.0);
    invalid.thrust = f32::INFINITY;
    let engine = app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: invalid,
    });
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle(&app, target))
        .is_none());
    assert_eq!(
        app.world().get::<Velocity>(target).unwrap().linear,
        Vec3::ZERO
    );
    assert_eq!(issues(&app)[0].entity, engine);
    assert_eq!(
        issues(&app)[0].issue,
        AeroIssue::InvalidParameter {
            parameter: "thruster.thrust",
            index: Some(0)
        }
    );
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(engine)
        .unwrap()
        .thruster = thruster(10.0);
    // Invalid rigid-body authoring deactivates the ordinary binding.
    app.world_mut()
        .get_mut::<RigidBody>(target)
        .unwrap()
        .mass
        .inv_mass = f32::NAN;
    app.update();
    assert_eq!(issues(&app)[0].issue, AeroIssue::TargetUnavailable);
    app.world_mut()
        .get_mut::<RigidBody>(target)
        .unwrap()
        .mass
        .inv_mass = 1.0;
    app.update();
    assert!(issues(&app).is_empty());
    assert_eq!(aero(&app, target).thrusters, vec![thruster(10.0)]);
}

#[test]
fn aggregate_nested_parameters_are_validated_with_source_and_element_index() {
    type Edit = fn(&mut core::AeroBodyConfig);
    let cases: &[(&str, Option<usize>, Edit)] = &[
        ("surface.lift_curve.post_stall_slope", Some(1), |c| {
            c.surfaces[1].lift_curve.post_stall_slope = f32::NAN
        }),
        ("thruster.slipstream.radius", Some(1), |c| {
            c.thrusters[1].slipstream = Some(core::aero::FlowJet::new(
                Vec3::ZERO,
                Vec3::X,
                -1.0,
                1.0,
                1.0,
            ))
        }),
        ("buoyancy.local_half_extents", Some(1), |c| {
            c.buoyancy[1].local_half_extents.x = -1.0
        }),
        ("model.water_plane.normal", None, |c| {
            c.model.water_plane = Some(core::aero::FluidPlane {
                point: Vec3::ZERO,
                normal: Vec3::ZERO,
                density: 1000.0,
            })
        }),
        ("model.added_mass.linear_coefficients", None, |c| {
            c.model.added_mass = Some(core::aero::AddedMass {
                linear_coefficients: Vec3::splat(-1.0),
                ..Default::default()
            })
        }),
        ("model.magnus.coefficient", None, |c| {
            c.model.magnus = Some(core::aero::MagnusParams {
                coefficient: f32::NAN,
                ..Default::default()
            })
        }),
    ];
    for &(parameter, index, edit) in cases {
        let mut app = app();
        let target = spawn_body(&mut app);
        let valid = core::AeroBodyConfig {
            surfaces: vec![surface(), surface()],
            thrusters: vec![thruster(1.0), thruster(2.0)],
            buoyancy: vec![volume(), volume()],
            ..Default::default()
        };
        app.world_mut()
            .insert(target, (RigidBodyAero(valid.clone()),));
        app.update();
        edit(&mut app.world_mut().get_mut::<RigidBodyAero>(target).unwrap().0);
        app.update();
        assert_eq!(aero(&app, target), valid, "{parameter}");
        assert_eq!(
            issues(&app),
            &[AeroDiagnostic {
                entity: target,
                target,
                source: AeroSource::BodyConfig,
                issue: AeroIssue::InvalidParameter { parameter, index }
            }]
        );
        // The retained FunctionalComponents authoring uses the same validation,
        // even without RigidBodyAero. Invalid vectors are not partially installed.
        if index.is_some() {
            let invalid = app.world().get::<RigidBodyAero>(target).unwrap().0.clone();
            app.world_mut().remove::<RigidBodyAero>(target);
            app.world_mut().insert(
                target,
                (AeroFunctionalComponents {
                    surfaces: invalid.surfaces,
                    thrusters: invalid.thrusters,
                    buoyancy: invalid.buoyancy,
                },),
            );
            app.update();
            assert_eq!(aero(&app, target), valid);
            assert_eq!(issues(&app)[0].source, AeroSource::FunctionalComponents);
            assert_eq!(
                issues(&app)[0].issue,
                AeroIssue::InvalidParameter { parameter, index }
            );
        }
    }
}

#[test]
fn authoring_mutations_commit_only_on_fixed_sync() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let part = app.world_mut().spawn(AeroThrusterComponent {
        target,
        thruster: thruster(10.0),
    });
    app.update();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(5)));
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(part)
        .unwrap()
        .thruster
        .thrust = 20.0;
    app.update();
    assert_eq!(aero(&app, target).thrusters, vec![thruster(10.0)]);
    assert!((app.world().get::<Velocity>(target).unwrap().linear.x - 0.1).abs() < 1e-5);
    app.update();
    assert_eq!(aero(&app, target).thrusters, vec![thruster(20.0)]);
    assert!((app.world().get::<Velocity>(target).unwrap().linear.x - 0.3).abs() < 1e-5);
    app.world_mut().remove::<AeroThrusterComponent>(part);
    app.update();
    assert_eq!(aero(&app, target).thrusters, vec![thruster(20.0)]);
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle(&app, target))
        .is_none());
    assert!((app.world().get::<Velocity>(target).unwrap().linear.x - 0.3).abs() < 1e-5);
}

#[test]
fn managed_registration_order_is_stable_and_foreign_aero_is_preserved() {
    let mut app = app();
    let first = spawn_body(&mut app);
    let second = spawn_body(&mut app);
    // Only second initially registered; first must later sort before it.
    app.world_mut().spawn(AeroThrusterComponent {
        target: second,
        thruster: thruster(2.0),
    });
    app.update();
    let foreign_config = core::AeroBodyConfig {
        thrusters: vec![thruster(99.0)],
        ..Default::default()
    };
    let foreign = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        let body = physics.spawn(core::BodyDesc::static_at(Vec3::ZERO));
        physics
            .register_aero_body(body, foreign_config.clone())
            .unwrap();
        body
    };
    app.world_mut()
        .insert(first, (RigidBodyAero(core::AeroBodyConfig::default()),));
    let part = app.world_mut().spawn(AeroThrusterComponent {
        target: first,
        thruster: thruster(1.0),
    });
    for _ in 0..3 {
        app.update();
        let physics = app.world().resource::<PhysicsWorldResource>();
        let managed: Vec<_> = physics
            .aero_bodies
            .iter()
            .filter(|entry| entry.body != foreign)
            .map(|entry| entry.body)
            .collect();
        assert_eq!(managed, vec![handle(&app, first), handle(&app, second)]);
        assert_eq!(physics.aero_body(foreign), Some(&foreign_config));
    }
    app.world_mut().remove::<RigidBodyAero>(first);
    app.world_mut().despawn(part);
    app.world_mut().despawn(second);
    app.update();
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert_eq!(physics.aero_bodies.len(), 1);
    assert_eq!(physics.aero_bodies[0].body, foreign);
    assert_eq!(physics.aero_bodies[0].config, foreign_config);
}

#[test]
fn managed_slot_permutation_handles_multiple_cycles_without_moving_foreign_slots_or_cloning_configs(
) {
    let mut app = app();
    let mut targets: Vec<_> = (0..7).map(|_| spawn_body(&mut app)).collect();
    targets.sort();
    for (index, &target) in targets.iter().enumerate() {
        app.world_mut().insert(
            target,
            (AeroFunctionalComponents {
                surfaces: vec![surface()],
                thrusters: vec![thruster(index as f32 + 1.0)],
                buoyancy: vec![volume()],
            },),
        );
    }
    app.update();
    let managed: Vec<_> = targets.iter().map(|&target| handle(&app, target)).collect();
    let mut foreign = Vec::new();
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        for index in 0..4 {
            let body = physics.spawn(core::BodyDesc::static_at(Vec3::ZERO));
            physics
                .register_aero_body(
                    body,
                    core::AeroBodyConfig {
                        thrusters: vec![thruster(100.0 + index as f32)],
                        ..Default::default()
                    },
                )
                .unwrap();
            foreign.push(body);
        }
    }
    // Retain allocations as well as values: steady Sync must move registrations
    // without cloning/replacing their element vectors during canonicalization.
    let expected: Vec<_> = app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_bodies
        .iter()
        .map(|entry| {
            (
                entry.body,
                entry.config.clone(),
                entry.config.surfaces.as_ptr(),
                entry.config.thrusters.as_ptr(),
                entry.config.buoyancy.as_ptr(),
            )
        })
        .collect();
    for permutation in [
        [1, 2, 0, 4, 5, 3, 6], // Two independent 3-cycles and one fixed point.
        [2, 0, 1, 5, 3, 4, 6], // Inverse cycles.
        [6, 5, 4, 3, 2, 1, 0], // Multiple 2-cycles.
        [0, 1, 2, 3, 4, 5, 6], // Already canonical.
    ] {
        let mut layout = vec![foreign[0]];
        for (rank, original) in permutation.into_iter().enumerate() {
            layout.push(managed[original]);
            if rank == 1 {
                layout.push(foreign[1]);
            }
            if rank == 4 {
                layout.push(foreign[2]);
            }
        }
        layout.push(foreign[3]);
        {
            let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
            for (destination, body) in layout.into_iter().enumerate() {
                let current = physics
                    .aero_bodies
                    .iter()
                    .position(|entry| entry.body == body)
                    .unwrap();
                physics.aero_bodies.swap(destination, current);
            }
        }
        for _ in 0..3 {
            app.update();
            let physics = app.world().resource::<PhysicsWorldResource>();
            assert_eq!(physics.aero_bodies.len(), managed.len() + foreign.len());
            for (slot, body) in [0, 3, 7, 10].into_iter().zip(&foreign) {
                assert_eq!(physics.aero_bodies[slot].body, *body);
            }
            let actual: Vec<_> = physics
                .aero_bodies
                .iter()
                .filter(|entry| !foreign.contains(&entry.body))
                .map(|entry| entry.body)
                .collect();
            assert_eq!(actual, managed);
            for (body, config, surfaces, thrusters, buoyancy) in &expected {
                let actual = physics.aero_body(*body).unwrap();
                assert_eq!(actual, config);
                assert_eq!(actual.surfaces.as_ptr(), *surfaces);
                assert_eq!(actual.thrusters.as_ptr(), *thrusters);
                assert_eq!(actual.buoyancy.as_ptr(), *buoyancy);
            }
            assert!(issues(&app).is_empty());
        }
    }
}

#[test]
fn retarget_rolls_back_connected_targets_without_duplicate_force() {
    let mut app = app();
    let first = spawn_body(&mut app);
    let second = spawn_body(&mut app);
    let stationary = app.world_mut().spawn(AeroThrusterComponent {
        target: first,
        thruster: thruster(1.0),
    });
    let moved = app.world_mut().spawn(AeroThrusterComponent {
        target: first,
        thruster: thruster(10.0),
    });
    app.update();
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(stationary)
        .unwrap()
        .thruster
        .thrust = f32::NAN;
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(moved)
        .unwrap()
        .target = second;
    app.update();
    assert_eq!(
        aero(&app, first).thrusters,
        vec![thruster(1.0), thruster(10.0)]
    );
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle(&app, second))
        .is_none());
    assert_eq!(
        issues(&app)[1],
        AeroDiagnostic {
            entity: moved,
            target: second,
            source: AeroSource::Thruster,
            issue: AeroIssue::ReplacementBlocked {
                rejected_target: first
            }
        }
    );
    assert!((app.world().get::<Velocity>(first).unwrap().linear.x - 0.22).abs() < 1e-5);
    assert_eq!(
        app.world().get::<Velocity>(second).unwrap().linear,
        Vec3::ZERO
    );
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(stationary)
        .unwrap()
        .thruster = thruster(2.0);
    app.update();
    assert_eq!(aero(&app, first).thrusters, vec![thruster(2.0)]);
    assert_eq!(aero(&app, second).thrusters, vec![thruster(10.0)]);
    assert!(issues(&app).is_empty());
    assert!((app.world().get::<Velocity>(first).unwrap().linear.x - 0.24).abs() < 1e-5);
    assert!((app.world().get::<Velocity>(second).unwrap().linear.x - 0.1).abs() < 1e-5);
}

#[test]
fn rejected_retarget_preserves_a_pending_last_source_removal_until_recovery() {
    let mut app = app();
    let first = spawn_body(&mut app);
    let second = spawn_body(&mut app);
    let third = spawn_body(&mut app);
    let moved = app.world_mut().spawn(AeroThrusterComponent {
        target: first,
        thruster: thruster(10.0),
    });
    let next = app.world_mut().spawn(AeroThrusterComponent {
        target: second,
        thruster: thruster(20.0),
    });
    let invalid = app.world_mut().spawn(AeroThrusterComponent {
        target: third,
        thruster: thruster(30.0),
    });
    app.update();
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(moved)
        .unwrap()
        .target = second;
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(next)
        .unwrap()
        .target = third;
    app.world_mut()
        .get_mut::<AeroThrusterComponent>(invalid)
        .unwrap()
        .thruster
        .thrust = f32::NAN;
    app.update();
    assert_eq!(aero(&app, first).thrusters, vec![thruster(10.0)]);
    assert_eq!(aero(&app, second).thrusters, vec![thruster(20.0)]);
    assert_eq!(aero(&app, third).thrusters, vec![thruster(30.0)]);
    assert_eq!(issues(&app).len(), 3);
    assert_eq!(
        issues(&app)[0].issue,
        AeroIssue::ReplacementBlocked {
            rejected_target: third
        }
    );
    assert_eq!(
        issues(&app)[1].issue,
        AeroIssue::ReplacementBlocked {
            rejected_target: third
        }
    );
    app.world_mut().remove::<AeroThrusterComponent>(invalid);
    app.update();
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .aero_body(handle(&app, first))
        .is_none());
    assert_eq!(aero(&app, second).thrusters, vec![thruster(10.0)]);
    assert_eq!(aero(&app, third).thrusters, vec![thruster(20.0)]);
    assert!(issues(&app).is_empty());
}

#[test]
fn all_families_use_source_entity_order_including_recycled_generations() {
    let mut app = app();
    let target = spawn_body(&mut app);
    let spare = app.world_mut().spawn((Transform::IDENTITY,));
    let later = app.world_mut().spawn((Transform::IDENTITY,));
    app.world_mut().despawn(spare);
    let recycled = app.world_mut().spawn((Transform::IDENTITY,));
    assert_eq!(spare.index(), recycled.index());
    assert_ne!(spare.generation(), recycled.generation());
    // Reverse insertion order, with all three component kinds on each source.
    for (entity, marker) in [(later, 2.0), (recycled, 1.0)] {
        let mut s = surface();
        s.local_position.x = marker;
        let mut b = volume();
        b.local_center.x = marker;
        app.world_mut().insert(
            entity,
            (
                AeroSurfaceComponent { target, surface: s },
                AeroThrusterComponent {
                    target,
                    thruster: thruster(marker),
                },
                AeroBuoyancyComponent { target, volume: b },
            ),
        );
    }
    app.update();
    let config = aero(&app, target);
    assert_eq!(
        config
            .surfaces
            .iter()
            .map(|s| s.local_position.x)
            .collect::<Vec<_>>(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        config
            .thrusters
            .iter()
            .map(|t| t.thrust)
            .collect::<Vec<_>>(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        config
            .buoyancy
            .iter()
            .map(|b| b.local_center.x)
            .collect::<Vec<_>>(),
        vec![1.0, 2.0]
    );
    app.world_mut().despawn(recycled);
    app.update();
    assert_eq!(aero(&app, target).surfaces.len(), 1);
    assert_eq!(aero(&app, target).thrusters, vec![thruster(2.0)]);
    assert_eq!(aero(&app, target).buoyancy.len(), 1);
}
