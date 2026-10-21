//! Explicit GPU field authority, fixed-substep accounting and recovery.

use std::time::Duration;

use prism_app::{App, TimeUpdateStrategy};
use prism_ecs::prelude::Entity;
use prism_physics_ecs::core::dynamics::field::{
    Field, FieldBatchError, FieldEffect, FieldId, FieldShape,
};
use prism_physics_ecs::field_runtime::{FieldStepError, GpuFieldRuntime};
use prism_physics_ecs::*;
use prism_physics_gpu::context::GpuContext;
use prism_transform::Transform;

const DT: f32 = 0.01;

fn runtime() -> Option<FieldRuntimeResource> {
    let Some(context) = GpuContext::try_headless() else {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU is set but no usable real GPU adapter is available"
        );
        eprintln!("no usable GPU adapter; skipping explicit GPU field ECS test");
        return None;
    };
    eprintln!("field ECS GPU adapter: {:?}", context.adapter().get_info());
    Some(FieldRuntimeResource(
        GpuFieldRuntime::new(context).expect("field pipeline initialization"),
    ))
}

fn app(runtime: Option<FieldRuntimeResource>, substeps: u32) -> App {
    let mut app = App::new();
    // Preinserted runtime must survive plugin installation without acquiring
    // another device. Other tests insert/remove the same resource after ticks.
    if let Some(runtime) = runtime {
        app.world_mut().insert_resource(runtime);
    }
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_plugins(PrismPhysicsPlugin);
    let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
    physics.config.gravity = Vec3::ZERO;
    physics.config.default_substeps = substeps;
    app
}

fn force_field(app: &mut App, effect: FieldEffect) -> Entity {
    let mut field = Field::constant(FieldId(0), Vec3::ZERO);
    field.effect = effect;
    app.world_mut().spawn(ForceField(field))
}

fn body(app: &mut App) -> Entity {
    app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
        ExternalForce {
            force: Vec3::X * 4.0,
            torque: Vec3::Z * 2.0,
        },
    ))
}

fn velocity(app: &App, entity: Entity) -> Vec3 {
    app.world().get::<Velocity>(entity).unwrap().linear
}

fn close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

fn status(app: &App, gpu: bool, completed: u32, simulated_dt: f32) {
    let physics = app.world().resource::<PhysicsDiagnostics>();
    let fields = app.world().resource::<FieldRuntimeDiagnostics>();
    assert_eq!(fields.tick, physics.tick);
    assert_eq!(fields.gpu_authority, gpu);
    assert_eq!(fields.completed_substeps, completed);
    assert_eq!(physics.simulated_dt, simulated_dt);
}

#[test]
fn cpu_default_never_creates_a_gpu_runtime_and_keeps_per_substep_field_semantics() {
    for substeps in [1, 4, 8] {
        let mut app = app(None, substeps);
        let entity = body(&mut app);
        force_field(&mut app, FieldEffect::ConstantForce(Vec3::X * 3.0));
        force_field(&mut app, FieldEffect::Drag { coefficient: 2.0 });
        assert!(!app.world().contains_resource::<FieldRuntimeResource>());

        app.update();
        let h = DT / substeps as f32;
        let mut expected = 0.0;
        for _ in 0..substeps {
            expected += (7.0 - 2.0 * expected) * h;
        }
        close(velocity(&app, entity).x, expected);
        status(&app, false, substeps, DT);
        assert_eq!(
            app.world()
                .resource::<FieldRuntimeDiagnostics>()
                .requested_substeps,
            substeps
        );
        assert!(app
            .world()
            .resource::<FieldRuntimeDiagnostics>()
            .error
            .is_none());
        assert!(app
            .world()
            .resource::<PhysicsDiagnostics>()
            .issues
            .is_empty());
        assert!(!app.world().contains_resource::<FieldRuntimeResource>());
    }
}

#[test]
fn real_gpu_preinsertion_insertion_removal_and_reinsertion_match_cpu_substeps() {
    let Some(mut runtime) = runtime() else {
        return;
    };
    for substeps in [1, 4, 8] {
        let mut gpu = app(Some(runtime), substeps);
        let mut cpu = app(None, substeps);
        let gpu_body = body(&mut gpu);
        let cpu_body = body(&mut cpu);
        let gpu_field = force_field(&mut gpu, FieldEffect::ConstantForce(Vec3::X * 3.0));
        let cpu_field = force_field(&mut cpu, FieldEffect::ConstantForce(Vec3::X * 3.0));
        force_field(&mut gpu, FieldEffect::Drag { coefficient: 2.0 });
        force_field(&mut cpu, FieldEffect::Drag { coefficient: 2.0 });

        let mut detached = None;
        let mut bound_handle = None;
        for tick in 0..5 {
            if tick == 1 {
                detached = gpu.world_mut().remove_resource::<FieldRuntimeResource>();
            } else if tick == 3 {
                gpu.world_mut().insert_resource(detached.take().unwrap());
            }
            if tick == 2 {
                for (app, field) in [(&mut gpu, gpu_field), (&mut cpu, cpu_field)] {
                    app.world_mut()
                        .get_mut::<ForceField>(field)
                        .unwrap()
                        .0
                        .effect = FieldEffect::ConstantForce(Vec3::X * 9.0);
                }
            }
            gpu.update();
            cpu.update();
            status(&gpu, tick != 1 && tick != 2, substeps, DT);
            close(velocity(&gpu, gpu_body).x, velocity(&cpu, cpu_body).x);
            let gpu_pose = gpu.world().get::<Transform>(gpu_body).unwrap();
            let cpu_pose = cpu.world().get::<Transform>(cpu_body).unwrap();
            close(gpu_pose.translation.x, cpu_pose.translation.x);
            close(
                gpu.world().get::<Velocity>(gpu_body).unwrap().angular.z,
                cpu.world().get::<Velocity>(cpu_body).unwrap().angular.z,
            );
            assert!(gpu
                .world()
                .resource::<FieldRuntimeDiagnostics>()
                .error
                .is_none());
            assert!(gpu
                .world()
                .resource::<PhysicsDiagnostics>()
                .issues
                .is_empty());
            let current = gpu
                .world()
                .resource::<PhysicsBindings>()
                .body(gpu_body)
                .unwrap();
            assert_eq!(*bound_handle.get_or_insert(current), current);
        }
        runtime = gpu
            .world_mut()
            .remove_resource::<FieldRuntimeResource>()
            .unwrap();
    }
}

// A finite GPU sample plus a finite pending ECS force overflows ONLY in the
// core's atomic batch validation. No device failure/NaN behavior is assumed.
// Large mass keeps positions and velocities finite even on successful steps.
fn overflow_scene(runtime: FieldRuntimeResource) -> (App, Entity, Entity) {
    let mut app = app(Some(runtime), 4);
    let entity = body(&mut app);
    app.world_mut()
        .get_mut::<RigidBody>(entity)
        .unwrap()
        .mass
        .inv_mass = 1e-30;
    app.world_mut()
        .get_mut::<ExternalForce>(entity)
        .unwrap()
        .force = Vec3::X * 2e38;
    let field = force_field(&mut app, FieldEffect::ConstantForce(Vec3::X * 2e38));
    (app, entity, field)
}

fn assert_sampling_failure(app: &App, completed: u32) {
    status(app, true, completed, DT / 4.0 * completed as f32);
    assert_eq!(
        app.world().resource::<FieldRuntimeDiagnostics>().error,
        Some(FieldStepError::Sampling {
            completed_substeps: completed,
            source: FieldBatchError::AccumulationOverflow { index: 0 },
        })
    );
    assert!(app
        .world()
        .resource::<PhysicsDiagnostics>()
        .issues
        .iter()
        .any(|diagnostic| diagnostic.entity.is_none()
            && diagnostic.issue == PhysicsIssue::FieldSamplingFailed));
    // All resources temporarily removed by the bridge are usable after error,
    // including the original runtime, ready for removal or another attempt.
    assert!(app.world().contains_resource::<PhysicsWorldResource>());
    assert!(app.world().contains_resource::<PhysicsSolver>());
    assert!(app.world().contains_resource::<PhysicsBindings>());
    assert!(app.world().contains_resource::<FieldRuntimeResource>());
}

#[test]
fn real_gpu_failure_restores_resources_and_caller_forces_without_duplicate_ecs_wrenches() {
    let Some(runtime) = runtime() else {
        return;
    };
    let (mut app, entity, field) = overflow_scene(runtime);
    app.update();
    assert_sampling_failure(&app, 0);
    assert_eq!(velocity(&app, entity), Vec3::ZERO);
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .external_forces
        .is_empty());

    let handle = app
        .world()
        .resource::<PhysicsBindings>()
        .body(entity)
        .unwrap();
    let pending = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        assert!(physics.apply_force(handle, Vec3::Y * 5.0, Vec3::ZERO));
        assert!(physics.apply_torque(handle, Vec3::Z * 7.0));
        physics.external_forces.clone()
    };
    for _ in 0..2 {
        app.update();
        assert_sampling_failure(&app, 0);
        assert_eq!(
            app.world()
                .resource::<PhysicsWorldResource>()
                .external_forces,
            pending
        );
        assert_eq!(velocity(&app, entity), Vec3::ZERO);
    }

    app.world_mut().remove::<ForceField>(field);
    app.update();
    status(&app, true, 4, DT);
    close(velocity(&app, entity).x, 2e8 * DT);
    close(
        app.world().get::<Velocity>(entity).unwrap().angular.z,
        7.0 * DT / 4.0 + 2.0 * DT,
    );
    assert!(app
        .world()
        .resource::<FieldRuntimeDiagnostics>()
        .error
        .is_none());
    assert!(app
        .world()
        .resource::<PhysicsDiagnostics>()
        .issues
        .is_empty());
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .external_forces
        .is_empty());
}

#[test]
fn real_gpu_first_failure_restores_sleep_flags_idle_timers_and_pending_wrenches() {
    let Some(runtime) = runtime() else {
        return;
    };
    let (mut app, force_sleeper, field) = overflow_scene(runtime);
    app.world_mut()
        .get_mut::<ExternalForce>(force_sleeper)
        .unwrap()
        .torque = Vec3::ZERO;
    let torque_sleeper = body(&mut app);
    *app.world_mut()
        .get_mut::<ExternalForce>(torque_sleeper)
        .unwrap() = ExternalForce {
        force: Vec3::ZERO,
        torque: Vec3::Z * 3.0,
    };
    let awake = body(&mut app);
    let untouched_sleeper = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::IDENTITY,
        Velocity::default(),
    ));

    // Reconcile bodies without attempting a substep, then seed the exact
    // pre-attempt states, including nonzero idle timers on awake bodies.
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 0;
    app.update();
    let entities = [force_sleeper, torque_sleeper, awake, untouched_sleeper];
    let handles = entities.map(|entity| {
        app.world()
            .resource::<PhysicsBindings>()
            .body(entity)
            .unwrap()
    });
    let sleep_states = [(true, 0.75), (true, 1.25), (false, 0.125), (true, 2.0)];
    let pending = {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.config.default_substeps = 4;
        assert!(physics.apply_force(handles[0], Vec3::Y * 5.0, Vec3::ZERO));
        assert!(physics.apply_torque(handles[1], Vec3::Z * 7.0));
        let view = physics.bodies.solver_view_mut();
        for (handle, (sleeping, timer)) in handles.into_iter().zip(sleep_states) {
            let slot = handle.index() as usize;
            view.sleeping[slot] = sleeping;
            view.sleep_timers[slot] = timer;
        }
        physics.external_forces.clone()
    };
    for _ in 0..2 {
        app.update();
        assert_sampling_failure(&app, 0);
        let physics = app.world().resource::<PhysicsWorldResource>();
        assert_eq!(physics.external_forces, pending);
        for ((entity, handle), (sleeping, timer)) in
            entities.into_iter().zip(handles).zip(sleep_states)
        {
            assert_eq!(physics.bodies.is_sleeping(handle), Some(sleeping));
            assert_eq!(physics.bodies.sleep_timer(handle), Some(timer));
            assert_eq!(
                *app.world().get::<Velocity>(entity).unwrap(),
                Velocity::default()
            );
            assert_eq!(
                *app.world().get::<Transform>(entity).unwrap(),
                Transform::IDENTITY
            );
        }
    }

    // A successful retry must still wake the forced sleepers and consume each
    // caller wrench once, not suppress waking or duplicate failed attempts.
    app.world_mut().remove::<ForceField>(field);
    app.update();
    status(&app, true, 4, DT);
    let physics = app.world().resource::<PhysicsWorldResource>();
    assert_eq!(physics.bodies.is_sleeping(handles[0]), Some(false));
    assert_eq!(physics.bodies.is_sleeping(handles[1]), Some(false));
    assert_eq!(physics.bodies.is_sleeping(handles[3]), Some(true));
    assert_eq!(physics.bodies.sleep_timer(handles[3]), Some(2.0));
    assert!(physics.external_forces.is_empty());
    close(velocity(&app, force_sleeper).x, 2e8 * DT);
    close(
        app.world()
            .get::<Velocity>(torque_sleeper)
            .unwrap()
            .angular
            .z,
        7.0 * DT / 4.0 + 3.0 * DT,
    );
}

#[test]
fn real_gpu_partial_failure_reports_committed_time_and_writeback_then_recovers() {
    let Some(runtime) = runtime() else {
        return;
    };
    let (mut app, entity, field) = overflow_scene(runtime);
    let idle_body = app.world_mut().spawn((
        RigidBody::dynamic(),
        Transform::from_translation(prism_math::Vec3::Y * 2.0),
        Velocity::default(),
        ExternalForce {
            force: Vec3::ZERO,
            torque: Vec3::Z * 0.001,
        },
    ));
    // First substep starts outside. Its ECS force moves x to 1250, entering
    // the volume so the second GPU batch rejects the overflowing sum.
    app.world_mut()
        .get_mut::<ForceField>(field)
        .unwrap()
        .0
        .shape = FieldShape::Aabb {
        min: Vec3::new(100.0, -1.0, -1.0),
        max: Vec3::new(1e6, 1.0, 1.0),
    };
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 0;
    app.update();
    let handles = [entity, idle_body].map(|entity| {
        app.world()
            .resource::<PhysicsBindings>()
            .body(entity)
            .unwrap()
    });
    {
        let physics = app.world_mut().resource_mut::<PhysicsWorldResource>();
        physics.config.default_substeps = 4;
        // Sleep timers advance for constraint islands, not unconstrained free
        // bodies. A COM ball joint adds this quiet body to an island without
        // restricting the small torque's rotation or linking it to the mover.
        let anchor = physics.spawn(core::BodyDesc::static_at(Vec3::Y * 2.0));
        physics.spawn_joint(core::JointDesc::spherical(
            core::JointAnchor::at_point(handles[1], Vec3::ZERO),
            core::JointAnchor::at_point(anchor, Vec3::ZERO),
        ));
        let view = physics.bodies.solver_view_mut();
        for handle in handles {
            view.sleeping[handle.index() as usize] = true;
            view.sleep_timers[handle.index() as usize] = 0.75;
        }
    }
    app.update();
    assert_sampling_failure(&app, 1);
    // The committed first substep wakes both bodies. The quiet torque-only
    // body then earns one substep of idle time; the rejected second attempt
    // must preserve that timer instead of clearing it or restoring tick start.
    for handle in handles {
        assert_eq!(
            app.world()
                .resource::<PhysicsWorldResource>()
                .bodies
                .is_sleeping(handle),
            Some(false)
        );
    }
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .sleep_timer(handles[1]),
        Some(DT / 4.0)
    );
    close(velocity(&app, entity).x, 2e8 * DT / 4.0);
    close(
        app.world().get::<Transform>(entity).unwrap().translation.x,
        1250.0,
    );
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .external_forces
        .is_empty());

    let pose = *app.world().get::<Transform>(entity).unwrap();
    let velocity_before = *app.world().get::<Velocity>(entity).unwrap();
    app.update();
    assert_sampling_failure(&app, 0);
    assert_eq!(
        app.world()
            .resource::<PhysicsWorldResource>()
            .bodies
            .sleep_timer(handles[1]),
        Some(DT / 4.0)
    );
    assert_eq!(*app.world().get::<Transform>(entity).unwrap(), pose);
    assert_eq!(
        *app.world().get::<Velocity>(entity).unwrap(),
        velocity_before
    );
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .external_forces
        .is_empty());

    // Explicit removal is the caller's choice to return to CPU authority.
    app.world_mut()
        .remove_resource::<FieldRuntimeResource>()
        .unwrap();
    app.world_mut().remove::<ForceField>(field);
    app.update();
    status(&app, false, 4, DT);
    close(velocity(&app, entity).x, 2e8 * DT * 1.25);
    assert!(app
        .world()
        .resource::<FieldRuntimeDiagnostics>()
        .error
        .is_none());
    assert!(app
        .world()
        .resource::<PhysicsDiagnostics>()
        .issues
        .is_empty());
}

#[test]
fn real_gpu_invalid_configuration_neither_queues_forces_nor_reports_sampling_failure() {
    let Some(runtime) = runtime() else {
        return;
    };
    let (mut app, entity, _) = overflow_scene(runtime);
    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 0;
    app.update();
    status(&app, true, 0, 0.0);
    assert_eq!(velocity(&app, entity), Vec3::ZERO);
    assert!(app
        .world()
        .resource::<FieldRuntimeDiagnostics>()
        .error
        .is_none());
    assert!(app
        .world()
        .resource::<PhysicsDiagnostics>()
        .issues
        .iter()
        .any(|diagnostic| diagnostic.issue == PhysicsIssue::InvalidConfiguration));
    assert!(app.world().contains_resource::<FieldRuntimeResource>());
    assert!(app
        .world()
        .resource::<PhysicsWorldResource>()
        .external_forces
        .is_empty());

    app.world_mut()
        .resource_mut::<PhysicsWorldResource>()
        .config
        .default_substeps = 4;
    app.update();
    assert_sampling_failure(&app, 0);
}
