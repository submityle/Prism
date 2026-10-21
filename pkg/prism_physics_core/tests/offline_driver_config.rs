//! Real CPU-solver and complete-world replay regressions. No mock projection.
use glam::{Quat, Vec3};
use prism_physics_core::backend::{OfflineStepConfig, OfflineStepError};
use prism_physics_core::collider::ColliderShape;
use prism_physics_core::dynamics::field::{Field, FieldId};
use prism_physics_core::math::transform::Isometry;
use prism_physics_core::reduced::{
    Articulation, ArticulationBase, ArticulationBuilder, ArticulationColliderDesc,
    ArticulationJoint, LinkContactImpulse, SpatialInertia, SpatialTransform,
};
use prism_physics_core::snapshot::hash::hash_world;
use prism_physics_core::solver::{
    OfflineSolver, MAX_OFFLINE_SOLVER_ITERATIONS, MAX_OFFLINE_SOLVER_SUBSTEPS,
};
use prism_physics_core::{
    BodyDesc, BodyHandle, CacheHandle, ClothGrid, CpuBackend, DistanceConstraint, DistanceJoint,
    DriveMode, DriveOutcome, JointDesc, PhysicsBackend, PhysicsWorld, SimulationDriver, SoftBody,
    SoftBodyHandle, SoftSolverConfig, Solver, XpbdSolver,
};

const DT: f32 = 1.0 / 60.0;

fn chain_world() -> (PhysicsWorld, [BodyHandle; 3], SoftBodyHandle) {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    let rigid = [
        world.spawn(BodyDesc::static_at(Vec3::ZERO)),
        world.spawn(BodyDesc::dynamic_at(2.0 * Vec3::X).with_damping(0.0, 0.0)),
        world.spawn(BodyDesc::dynamic_at(4.0 * Vec3::X).with_damping(0.0, 0.0)),
    ];
    for pair in rigid.windows(2) {
        world.spawn_joint(JointDesc::distance(
            pair[0],
            Vec3::ZERO,
            pair[1],
            Vec3::ZERO,
            DistanceJoint::rigid(1.0),
        ));
    }
    let mut soft = SoftBody::new(SoftSolverConfig {
        gravity: Vec3::ZERO,
        damping: 0.0,
        iterations: 3,
        substeps: 7,
        ..SoftSolverConfig::default()
    });
    let particles = [
        soft.spawn_pinned(Vec3::ZERO),
        soft.spawn(2.0 * Vec3::X, 1.0),
        soft.spawn(4.0 * Vec3::X, 1.0),
    ];
    for pair in particles.windows(2) {
        soft.add_distance(DistanceConstraint::new(pair[0], pair[1], 1.0, 0.0));
    }
    let soft = world.spawn_soft_body(soft);
    (world, rigid, soft)
}

fn chain_error(positions: &[Vec3]) -> f32 {
    positions
        .windows(2)
        .map(|p| (p[0].distance(p[1]) - 1.0).abs())
        .sum()
}

fn articulation_contact_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    let shape = world.shapes.insert(ColliderShape::Sphere { radius: 0.5 });
    let inertia = SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.1));
    let mut builder = ArticulationBuilder::new(ArticulationBase::Floating, inertia)
        .with_base_pose(Isometry::IDENTITY)
        .with_gravity(Vec3::ZERO);
    builder
        .add_link(
            None,
            SpatialTransform::new(Quat::IDENTITY, Vec3::ZERO),
            inertia,
            ArticulationJoint::fixed(),
        )
        .unwrap();
    let articulation = world.spawn_articulation(builder.build().unwrap());
    world
        .attach_articulation_collider(ArticulationColliderDesc::new(articulation, Some(0), shape))
        .unwrap();
    for x in [-0.75, 0.75] {
        world.spawn(
            BodyDesc::dynamic_at(Vec3::X * x)
                .with_collider(shape)
                .with_damping(0.0, 0.0),
        );
    }
    world
}

fn assert_world_restored(actual: &PhysicsWorld, expected: &PhysicsWorld) {
    assert_eq!(
        actual.soft_bodies, expected.soft_bodies,
        "particles, history, constraints and topology"
    );
    assert_eq!(
        actual.fields, expected.fields,
        "field registry including next ID"
    );
    assert_eq!(actual.external_forces, expected.external_forces);
    assert_eq!(
        actual.articulation_degradations,
        expected.articulation_degradations
    );
    let actual_handles: Vec<_> = actual.articulations.handles().collect();
    let expected_handles: Vec<_> = expected.articulations.handles().collect();
    assert_eq!(actual_handles, expected_handles);
    for handle in expected_handles {
        let a = actual.articulations.get(handle).unwrap();
        let b = expected.articulations.get(handle).unwrap();
        assert_eq!(a.positions(), b.positions());
        assert_eq!(a.velocities(), b.velocities());
        assert_eq!(a.base_pose(), b.base_pose());
        assert_eq!(a.base_velocity(), b.base_velocity());
        assert_eq!(a.broken_joint(), b.broken_joint());
    }
    // Includes private allocator generations, queues and assembly bookkeeping;
    // randomized contact map iteration must not produce false mismatches.
    assert_eq!(hash_world(actual).unwrap(), hash_world(expected).unwrap());
}

#[test]
fn offline_iterations_improve_real_rigid_and_soft_constraint_convergence() {
    let (initial, rigid, soft) = chain_world();
    let solve = |iterations| {
        let mut world = initial.clone();
        let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 9);
        let driver = SimulationDriver::new(DriveMode::Offline {
            substeps: 1,
            iterations,
            target_frames: 1,
        });
        assert_eq!(
            driver.drive(&mut backend, &mut world, DT),
            DriveOutcome::OfflineFrame {
                frame: 0,
                total_frames: 1
            }
        );
        assert_eq!(backend.substeps(), 9);
        assert_eq!(
            world.soft_bodies.get(soft).unwrap().config,
            initial.soft_bodies.get(soft).unwrap().config
        );
        world
    };
    let one = solve(1);
    let many = solve(16);
    let rigid_error =
        |world: &PhysicsWorld| chain_error(&rigid.map(|h| world.bodies.position(h).unwrap()));
    let soft_error = |world: &PhysicsWorld| {
        chain_error(world.soft_bodies.get(soft).unwrap().particles.positions())
    };
    assert!(
        rigid_error(&one) > 0.1,
        "fixture must not converge in one sweep"
    );
    assert!(
        soft_error(&one) > 0.1,
        "fixture must not converge in one sweep"
    );
    assert!(rigid_error(&many) < rigid_error(&one) * 0.01);
    assert!(soft_error(&many) < soft_error(&one) * 0.01);
}

#[test]
fn offline_uses_exact_substeps_and_restores_realtime_solver_settings() {
    let (mut world, _, soft) = chain_world();
    world.config.gravity = -10.0 * Vec3::Y;
    // Free fall distinguishes three actual substeps from 3 * 11 substeps.
    world.spawn(BodyDesc::dynamic_at(10.0 * Vec3::Y).with_damping(0.0, 0.0));
    let mut reference = world.clone();
    let mut solver = XpbdSolver::new();
    solver.config.position_iterations = 2;
    solver.config.articulation_contact_iterations = 5;
    let mut backend = CpuBackend::new(Box::new(solver.clone()), 11);
    let mut driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 3,
        iterations: 7,
        target_frames: 1,
    });
    let original_soft_config = reference.soft_bodies.get(soft).unwrap().config;
    reference
        .soft_bodies
        .get_mut(soft)
        .unwrap()
        .config
        .iterations = 7;
    solver.config.position_iterations = 7;
    solver.config.articulation_contact_iterations = 7;
    solver.step(&mut reference, DT, 3);
    reference.soft_bodies.get_mut(soft).unwrap().config = original_soft_config;
    assert!(matches!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::OfflineFrame { frame: 0, .. }
    ));
    assert_world_restored(&world, &reference);
    assert_world_restored(
        &driver
            .recorded_cache()
            .unwrap()
            .frame(0)
            .unwrap()
            .snapshot()
            .fork_world(),
        &reference,
    );

    driver.mode = DriveMode::Realtime;
    solver.config.position_iterations = 2;
    solver.config.articulation_contact_iterations = 5;
    solver.step(&mut reference, DT, 11);
    assert_eq!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::Stepped
    );
    assert_world_restored(&world, &reference);
    assert!(driver.recorded_cache().is_none());
}

#[test]
fn invalid_or_unsupported_offline_requests_are_explicit_and_atomic() {
    let (mut world, _, _) = chain_world();
    let original = world.clone();
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 4);
    for substeps in [0, MAX_OFFLINE_SOLVER_SUBSTEPS + 1, u32::MAX] {
        assert_eq!(
            backend.step_offline(
                &mut world,
                DT,
                OfflineStepConfig {
                    substeps,
                    iterations: 1
                }
            ),
            Err(OfflineStepError::InvalidSubsteps {
                requested: substeps
            })
        );
        assert_world_restored(&world, &original);
    }
    for iterations in [0, MAX_OFFLINE_SOLVER_ITERATIONS + 1, u32::MAX] {
        assert_eq!(
            backend.step_offline(
                &mut world,
                DT,
                OfflineStepConfig {
                    substeps: 1,
                    iterations
                }
            ),
            Err(OfflineStepError::InvalidIterations {
                requested: iterations
            })
        );
        assert_world_restored(&world, &original);
        let mut driver = SimulationDriver::new(DriveMode::Offline {
            substeps: 1,
            iterations,
            target_frames: 1,
        });
        for _ in 0..2 {
            assert_eq!(
                driver.drive(&mut backend, &mut world, DT),
                DriveOutcome::OfflineRejected {
                    error: OfflineStepError::InvalidIterations {
                        requested: iterations
                    }
                }
            );
            assert!(driver.recorded_cache().unwrap().is_empty());
            assert_world_restored(&world, &original);
        }
        driver.mode = DriveMode::Offline {
            substeps: 1,
            iterations: 2,
            target_frames: 1,
        };
        assert!(matches!(
            driver.drive(&mut backend, &mut world, DT),
            DriveOutcome::OfflineFrame { frame: 0, .. }
        ));
        world = original.clone();
    }
    for dt in [0.0, -DT, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let config = OfflineStepConfig {
            substeps: 1,
            iterations: 2,
        };
        assert_eq!(
            backend.step_offline(&mut world, dt, config),
            Err(OfflineStepError::InvalidTimeStep)
        );
        let driver = SimulationDriver::new(DriveMode::Offline {
            substeps: 1,
            iterations: 2,
            target_frames: 1,
        });
        assert_eq!(
            driver.drive(&mut backend, &mut world, dt),
            DriveOutcome::OfflineRejected {
                error: OfflineStepError::InvalidTimeStep
            }
        );
        assert!(driver.recorded_cache().unwrap().is_empty());
        assert_world_restored(&world, &original);
        assert!(matches!(
            driver.drive(&mut backend, &mut world, DT),
            DriveOutcome::OfflineFrame { frame: 0, .. }
        ));
        world = original.clone();
    }
    let mut integrate_only = CpuBackend::with_defaults();
    let mut driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 2,
        iterations: 3,
        target_frames: 1,
    });
    assert_eq!(
        driver.drive(&mut integrate_only, &mut world, DT),
        DriveOutcome::OfflineRejected {
            error: OfflineStepError::UnsupportedSolver
        }
    );
    assert_world_restored(&world, &original);
    assert!(driver.recorded_cache().unwrap().is_empty());
    driver.mode = DriveMode::Realtime;
    assert_eq!(
        driver.drive(&mut integrate_only, &mut world, DT),
        DriveOutcome::Stepped
    );
}

fn articulation() -> Articulation {
    let mut builder = ArticulationBuilder::new(ArticulationBase::Fixed, SpatialInertia::default());
    for parent in [None, Some(0)] {
        builder
            .add_link(
                parent,
                SpatialTransform::IDENTITY,
                SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::splat(0.2)),
                ArticulationJoint::revolute(Vec3::Z),
            )
            .unwrap();
    }
    let mut result = builder.build().unwrap();
    assert!(result.set_velocity(0, 0.7));
    assert!(result.set_velocity(1, -0.2));
    result
}

#[test]
fn full_world_playback_restores_soft_fields_articulations_and_handle_generations() {
    let mut source = PhysicsWorld::with_gravity(-Vec3::Y);
    let mut rigid = source.spawn(BodyDesc::dynamic_at(Vec3::Y));
    let cloth = || {
        ClothGrid {
            columns: 3,
            rows: 3,
            origin: Vec3::Y * 4.0,
            ..ClothGrid::default()
        }
        .build_default()
        .body
    };
    let mut soft = source.spawn_soft_body(cloth());
    assert!(source.soft_bodies.get(soft).unwrap().topology.is_some());
    let mut art = source.spawn_articulation(articulation());
    let mut field = source
        .fields
        .insert(Field::constant(FieldId(0), 2.0 * Vec3::X));
    let driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 2,
        iterations: 5,
        target_frames: 3,
    });
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 4);
    let mut expected = Vec::new();
    for frame in 0..3 {
        if frame == 1 {
            let body = source.soft_bodies.get_mut(soft).unwrap();
            let point = body.particles.positions()[4];
            let events = body.puncture_topology(point, Vec3::Y, 1.0).unwrap();
            assert!(!events.is_empty());
            assert!(
                body.particles.len() > 9,
                "replay fixture must contain an actual split"
            );
            assert!(source.request_articulation_break(art, 1));
            assert!(source.fields.remove(field).is_some());
            field = source
                .fields
                .insert(Field::constant(FieldId(0), -3.0 * Vec3::Z));
        }
        if frame == 2 {
            assert!(source.remove_body(rigid));
            rigid = source.spawn(BodyDesc::dynamic_at(Vec3::Y * 2.0));
            assert!(source.remove_soft_body(soft));
            soft = source.spawn_soft_body(cloth());
            assert!(source.remove_articulation(art));
            art = source.spawn_articulation(articulation());
        }
        source
            .external_forces
            .accumulate_force(rigid, Vec3::Z)
            .unwrap();
        assert!(source.queue_articulation_contact_impulse(
            art,
            LinkContactImpulse {
                link: 0,
                point: Vec3::X,
                impulse: 0.1 * Vec3::Y
            }
        ));
        assert_eq!(
            driver.drive(&mut backend, &mut source, DT),
            DriveOutcome::OfflineFrame {
                frame,
                total_frames: 3
            }
        );
        expected.push(source.clone());
    }
    assert!(!expected[1].articulation_degradations.is_empty());
    assert!(
        expected[1].soft_bodies.get(soft).is_none(),
        "frame two must recycle the original handle generation"
    );
    let torn = expected[1].soft_bodies.handles().next().unwrap();
    assert!(expected[1].soft_bodies.get(torn).unwrap().particles.len() > 9);
    let cache = driver.recorded_cache().unwrap();
    let handle = CacheHandle(73);
    let playback = SimulationDriver::new(DriveMode::Playback(handle));
    playback.register_cache(handle, cache.clone()).unwrap();
    let mut destination = PhysicsWorld::with_gravity(Vec3::X * 99.0);
    for _ in 0..5 {
        destination.spawn(BodyDesc::static_at(Vec3::splat(99.0)));
        destination.spawn_soft_body(SoftBody::with_default_config());
        destination.spawn_articulation(articulation());
    }
    for (frame, expected) in expected.iter().enumerate() {
        assert_eq!(
            cache.frame(frame as u32).unwrap().snapshot().frame(),
            frame as u64
        );
        assert_eq!(
            playback.drive(&mut backend, &mut destination, 999.0),
            DriveOutcome::PlaybackFrame {
                frame: frame as u32,
                total_frames: 3
            }
        );
        assert_world_restored(&destination, expected);
        // Replay must recover simulation inputs/history, not just equal poses.
        let mut continued = destination.clone();
        let mut reference = expected.clone();
        CpuBackend::new(Box::new(XpbdSolver::new()), 4).step(&mut continued, DT);
        CpuBackend::new(Box::new(XpbdSolver::new()), 4).step(&mut reference, DT);
        assert_world_restored(&continued, &reference);
    }
    let before = destination.clone();
    assert_eq!(
        playback.drive(&mut backend, &mut destination, DT),
        DriveOutcome::Complete { frames: 3 }
    );
    assert_world_restored(&destination, &before);
    // A mutable fork never aliases an immutable cache frame.
    let mut fork = cache.frame(0).unwrap().snapshot().fork_world();
    fork.spawn_soft_body(SoftBody::with_default_config());
    playback.register_cache(handle, cache).unwrap();
    assert!(matches!(
        playback.drive(&mut backend, &mut destination, DT),
        DriveOutcome::PlaybackFrame { frame: 0, .. }
    ));
    assert_world_restored(&destination, &expected[0]);
}

#[test]
fn switching_modes_restarts_cursors_and_offline_runs_without_losing_registered_cache() {
    let (mut world, _, _) = chain_world();
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 3);
    let offline_mode = DriveMode::Offline {
        substeps: 2,
        iterations: 6,
        target_frames: 2,
    };
    let mut driver = SimulationDriver::new(offline_mode);
    for frame in 0..2 {
        assert_eq!(
            driver.drive(&mut backend, &mut world, DT),
            DriveOutcome::OfflineFrame {
                frame,
                total_frames: 2
            }
        );
    }
    let cache = driver.recorded_cache().unwrap();
    let first = cache.frame(0).unwrap().snapshot().fork_world();
    let handle = CacheHandle(17);
    driver.register_cache(handle, cache.clone()).unwrap();
    driver.mode = DriveMode::Playback(handle);
    assert!(driver.recorded_cache().is_none());
    assert_eq!(driver.playback_frame(), 0);
    assert!(matches!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::PlaybackFrame { frame: 0, .. }
    ));
    assert_world_restored(&world, &first);
    driver.mode = DriveMode::Realtime;
    assert_eq!(driver.playback_frame(), 0);
    let mut reference = first.clone();
    CpuBackend::new(Box::new(XpbdSolver::new()), 3).step(&mut reference, DT);
    assert_eq!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::Stepped
    );
    assert_world_restored(&world, &reference);
    driver.mode = DriveMode::Playback(handle);
    assert!(matches!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::PlaybackFrame { frame: 0, .. }
    ));
    assert_world_restored(&world, &first);
    assert_eq!(driver.playback_frame(), 1);
    driver.register_cache(handle, cache).unwrap();
    assert_eq!(driver.playback_frame(), 0);
    assert!(matches!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::PlaybackFrame { frame: 0, .. }
    ));

    driver.mode = offline_mode;
    assert!(matches!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::OfflineFrame { frame: 0, .. }
    ));
    driver.mode = DriveMode::Offline {
        substeps: 1,
        iterations: 1,
        target_frames: 1,
    };
    assert!(driver.recorded_cache().is_none());
    assert_eq!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::OfflineFrame {
            frame: 0,
            total_frames: 1
        }
    );
    assert_eq!(driver.recorded_cache().unwrap().frame_count(), 1);
    assert_eq!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::Complete { frames: 1 }
    );
    assert!(driver.cache(handle).is_some());
}

#[test]
fn direct_xpbd_offline_capability_restores_its_own_configuration() {
    let (mut world, _, soft) = chain_world();
    let mut solver = XpbdSolver::new();
    solver.config.position_iterations = 2;
    solver.config.articulation_contact_iterations = 5;
    let original = solver.config;
    let soft_config = world.soft_bodies.get(soft).unwrap().config;
    solver
        .step_offline(
            &mut world,
            DT,
            OfflineStepConfig {
                substeps: 2,
                iterations: 13,
            },
        )
        .unwrap();
    assert_eq!(solver.config, original);
    assert_eq!(world.soft_bodies.get(soft).unwrap().config, soft_config);
}

#[test]
fn offline_iterations_override_and_restore_articulation_contact_sweeps() {
    let initial = articulation_contact_world();
    let mut offline_world = initial.clone();
    let mut reference_world = initial.clone();
    let mut low_iteration_world = initial;
    let mut offline_solver = XpbdSolver::new();
    offline_solver.config.position_iterations = 2;
    offline_solver.config.articulation_contact_iterations = 1;
    let realtime_config = offline_solver.config;

    offline_solver
        .step_offline(
            &mut offline_world,
            DT,
            OfflineStepConfig {
                substeps: 1,
                iterations: 7,
            },
        )
        .unwrap();
    assert_eq!(offline_solver.config, realtime_config);

    let mut reference_solver = XpbdSolver::new();
    reference_solver.config.position_iterations = 7;
    reference_solver.config.articulation_contact_iterations = 7;
    reference_solver.step(&mut reference_world, DT, 1);
    assert_world_restored(&offline_world, &reference_world);

    let mut low_iteration_solver = reference_solver;
    low_iteration_solver.config.articulation_contact_iterations = 1;
    low_iteration_solver.step(&mut low_iteration_world, DT, 1);
    assert_ne!(
        hash_world(&low_iteration_world).unwrap(),
        hash_world(&reference_world).unwrap(),
        "fixture must distinguish one articulation sweep from seven"
    );
}

#[test]
fn rejected_direct_offline_requests_preserve_world_and_solver_config() {
    let initial = articulation_contact_world();
    let mut world = initial.clone();
    let mut solver = XpbdSolver::new();
    solver.config.position_iterations = 3;
    solver.config.articulation_contact_iterations = 6;
    let original_config = solver.config;
    let invalid = [
        (
            DT,
            OfflineStepConfig {
                substeps: 0,
                iterations: 1,
            },
            OfflineStepError::InvalidSubsteps { requested: 0 },
        ),
        (
            DT,
            OfflineStepConfig {
                substeps: 1,
                iterations: 0,
            },
            OfflineStepError::InvalidIterations { requested: 0 },
        ),
        (
            f32::NAN,
            OfflineStepConfig {
                substeps: 1,
                iterations: 1,
            },
            OfflineStepError::InvalidTimeStep,
        ),
    ];
    for (dt, config, expected) in invalid {
        assert_eq!(solver.step_offline(&mut world, dt, config), Err(expected));
        assert_eq!(solver.config, original_config);
        assert_world_restored(&world, &initial);
    }
}

#[test]
fn cache_records_effective_config_and_each_frame_duration() {
    let mut world = PhysicsWorld::default();
    let mut backend = CpuBackend::new(Box::new(XpbdSolver::new()), 9);
    let mut driver = SimulationDriver::new(DriveMode::Offline {
        substeps: 0,
        iterations: 3,
        target_frames: 2,
    });
    for (frame, dt) in [DT, DT * 2.0].into_iter().enumerate() {
        assert!(matches!(
            driver.drive(&mut backend, &mut world, dt),
            DriveOutcome::OfflineFrame { .. }
        ));
        let cache = driver.recorded_cache().unwrap();
        let recorded = cache.frame(frame as u32).unwrap();
        assert_eq!(recorded.dt(), dt);
        assert_eq!(
            recorded.config(),
            OfflineStepConfig {
                substeps: 1,
                iterations: 3
            }
        );
        assert_eq!(
            recorded.snapshot().state_hash().unwrap(),
            hash_world(&world).unwrap()
        );
    }
    driver.mode = DriveMode::Offline {
        substeps: u32::MAX,
        iterations: 1,
        target_frames: 1,
    };
    assert!(matches!(
        driver.drive(&mut backend, &mut world, DT),
        DriveOutcome::OfflineFrame { frame: 0, .. }
    ));
    assert_eq!(
        driver
            .recorded_cache()
            .unwrap()
            .frame(0)
            .unwrap()
            .config()
            .substeps,
        MAX_OFFLINE_SOLVER_SUBSTEPS
    );
}
