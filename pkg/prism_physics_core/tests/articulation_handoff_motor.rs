//! Explicit generalized-force drives are not XPBD position motors.
use glam::{Quat, Vec3};
use prism_physics_core::dynamics::assembly::BreakLimit;
use prism_physics_core::dynamics::field::{
    FieldBatchSampler, FieldBody, FieldRegistry, FieldSample,
};
use prism_physics_core::joint::{
    BoundedPd, BoundedPdFailure, JointAnchor, JointDesc, JointHandle, JointKind, LinearLimit,
    Motor, PrismaticJoint, RevoluteJoint,
};
use prism_physics_core::reduced::articulation::handoff::{
    ArticulationHandoffError, UnsupportedJointPolicy,
};
use prism_physics_core::reduced::{
    ArticulationBase, ArticulationBuilder, ArticulationJoint, JointLimits, JointMotor,
    SpatialForce, SpatialInertia, SpatialTransform,
};
use prism_physics_core::snapshot::{world_hash::hash_world, FullStateSnapshot};
use prism_physics_core::solver::xpbd::FieldStepError;
use prism_physics_core::solver::{Solver, XpbdConfig, XpbdSolver};
use prism_physics_core::state::body::{BodyDesc, MassProperties};
use prism_physics_core::state::handle::BodyHandle;
use prism_physics_core::world::PhysicsWorld;

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance,
        "{actual} != {expected}"
    );
}

fn drive() -> BoundedPd {
    BoundedPd::new(2.0, -3.0, 7.0, 5.0, 100.0).unwrap()
}

fn pair(hinge: bool, policy: BoundedPd) -> (PhysicsWorld, BodyHandle, BodyHandle, JointHandle) {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    let a = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    let b = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_damping(0.0, 0.0));
    let kind = if hinge {
        JointKind::Revolute(RevoluteJoint::new(Vec3::Z))
    } else {
        JointKind::Prismatic(PrismaticJoint::new(Vec3::X))
    };
    let joint = world.spawn_joint(
        JointDesc::new(JointAnchor::at_center(a), JointAnchor::at_center(b), kind)
            .with_bounded_pd(policy),
    );
    (world, a, b, joint)
}

fn solver(iterations: u32, parallel: bool, colored: bool) -> XpbdSolver {
    XpbdSolver::with_config(XpbdConfig {
        position_iterations: iterations,
        parallel_islands: parallel,
        parallel_within_island: colored,
        ..XpbdConfig::default()
    })
}

#[test]
fn surviving_scalar_handoff_preserves_both_targets_limits_and_continuous_coordinate() {
    for hinge in [false, true] {
        for broken in [0, 1] {
            let motor_link = 1 - broken;
            let q = if hinge { 20.0 } else { 1.0 };
            let motor = JointMotor::new(
                Vec3::splat(q + 2.0),
                Vec3::splat(-3.0),
                7.0,
                5.0,
                Vec3::splat(100.0),
            );
            let scalar = if hinge {
                ArticulationJoint::revolute(Vec3::Z)
            } else {
                ArticulationJoint::prismatic(Vec3::X)
            }
            .with_motor(motor)
            .with_limits(JointLimits::scalar(q - 1.0, q + 1.0));
            let inertia = SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE);
            let mut builder =
                ArticulationBuilder::new(ArticulationBase::Fixed, inertia).with_gravity(Vec3::ZERO);
            for link in 0usize..2 {
                builder
                    .add_link(
                        link.checked_sub(1),
                        SpatialTransform::IDENTITY,
                        inertia,
                        if link == motor_link {
                            scalar
                        } else {
                            ArticulationJoint::fixed()
                        },
                    )
                    .unwrap();
            }
            let mut tree = builder.build().unwrap();
            assert!(tree.set_position(0, q));
            assert!(tree.set_velocity(0, 1.0));
            tree.break_joint(broken);
            let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
            let source = world.spawn_articulation(tree);
            let report = world.handoff_articulation(source).unwrap();
            assert_eq!(report.joints.len(), 1);
            assert_eq!(report.joints[0].link, motor_link);
            let handle = report.joints[0].joint;
            let joint = world.joints.get(handle).unwrap();
            let policy = joint.bounded_pd.unwrap();
            assert_eq!(
                policy,
                BoundedPd::new(f64::from(q + 2.0), -3.0, 7.0, 5.0, 100.0).unwrap()
            );
            near(policy.effort(f64::from(q), 1.0).unwrap(), -6.0, 1.0e-12);
            match joint.kind {
                JointKind::Revolute(params) => {
                    assert!(params.motor.is_none());
                    assert_eq!(params.limit.unwrap().min, q - 1.0);
                    assert_eq!(joint.revolute_state().angle(), Some(f64::from(q)));
                }
                JointKind::Prismatic(params) => {
                    assert!(params.motor.is_none());
                    assert_eq!(params.limit.unwrap().max, q + 1.0);
                    let a = joint.anchor_a;
                    let b = joint.anchor_b;
                    let qa = world.bodies.orientation(a.body).unwrap();
                    let n = qa * a.local_frame * params.axis;
                    let pa = world.bodies.position(a.body).unwrap() + qa * a.local_point;
                    let pb = world.bodies.position(b.body).unwrap()
                        + world.bodies.orientation(b.body).unwrap() * b.local_point;
                    near(f64::from((pb - pa).dot(n)), f64::from(q), 1.0e-5);
                }
                _ => panic!("wrong surviving family"),
            }
            let mut replay = FullStateSnapshot::capture(0, &world).fork_world();
            let mut first = solver(4, false, false);
            let mut second = solver(4, false, false);
            for _ in 0..8 {
                first.step(&mut world, 0.002, 1);
                second.step(&mut replay, 0.002, 1);
                assert_eq!(hash_world(&world).unwrap(), hash_world(&replay).unwrap());
                assert_eq!(world.joints.get(handle).unwrap().bounded_pd_failure, None);
            }
        }
    }
}

#[test]
fn effort_budget_is_once_per_substep_not_per_iteration_and_survives_velocity_recovery() {
    for hinge in [false, true] {
        for iterations in [1, 12] {
            for substeps in [1, 4] {
                let cap = BoundedPd::new(100.0, 100.0, 3.0, 2.0, 2.0).unwrap();
                let (mut world, _, body, _) = pair(hinge, cap);
                solver(iterations, false, false).step(&mut world, 0.04, substeps);
                let speed = if hinge {
                    world.bodies.angular_velocity(body).unwrap().z
                } else {
                    world.bodies.linear_velocity(body).unwrap().x
                };
                near(f64::from(speed), 0.08, 4.0e-6);
                assert!(world.bodies.position(body).unwrap().is_finite());
            }
        }
    }
}

#[test]
fn simultaneous_feedback_targets_produce_the_sum_not_a_position_or_velocity_servo() {
    for hinge in [false, true] {
        let (mut world, a, b, joint) = pair(hinge, drive());
        if hinge {
            world.bodies.set_orientation(b, Quat::from_rotation_z(1.0));
            world.bodies.set_angular_velocity(b, Vec3::Z);
            assert!(world
                .joints
                .get_mut(joint)
                .unwrap()
                .set_revolute_coordinate(
                    1.0,
                    [
                        world.bodies.orientation(a).unwrap(),
                        world.bodies.orientation(b).unwrap()
                    ]
                ));
        } else {
            world.bodies.set_position(b, Vec3::X);
            world.bodies.set_linear_velocity(b, Vec3::X);
        }
        solver(8, false, false).step(&mut world, 0.01, 1);
        let speed = if hinge {
            world.bodies.angular_velocity(b).unwrap().z
        } else {
            world.bodies.linear_velocity(b).unwrap().x
        };
        near(f64::from(speed), 0.87, 3.0e-5);
    }
}

#[test]
fn sleeping_chain_wakes_before_prediction_and_stalled_drive_inhibits_sleep() {
    let (mut world, _, b, joint) = pair(false, BoundedPd::new(1.0, 0.0, 10.0, 0.0, 2.0).unwrap());
    world.config.sleep.enabled = true;
    world.config.sleep.time_to_sleep = 0.001;
    world.config.sleep.linear_velocity_threshold = 100.0;
    world.config.sleep.angular_velocity_threshold = 100.0;
    let c = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_damping(0.0, 0.0));
    world.spawn_joint(JointDesc::fixed(
        JointAnchor::at_center(b),
        JointAnchor::at_center(c),
    ));
    {
        let view = world.bodies.solver_view_mut();
        for body in [b, c] {
            let slot = body.index() as usize;
            view.sleeping[slot] = true;
            view.sleep_timers[slot] = 10.0;
            view.prev_positions[slot] = Vec3::splat(-1000.0);
        }
    }
    solver(8, false, false).step(&mut world, 0.01, 1);
    for body in [b, c] {
        assert_eq!(world.bodies.is_sleeping(body), Some(false));
        near(
            f64::from(world.bodies.linear_velocity(body).unwrap().x),
            0.01,
            1.0e-6,
        );
    }
    // Lock the free coordinate to a stop: nonzero effort remains activity,
    // even if the position solve exactly cancels the motor displacement.
    if let JointKind::Prismatic(ref mut params) = world.joints.get_mut(joint).unwrap().kind {
        params.limit = Some(LinearLimit::new(0.0, 0.0));
    }
    solver(8, false, false).step(&mut world, 0.01, 1);
    assert_eq!(world.bodies.is_sleeping(b), Some(false));
    assert_eq!(world.bodies.is_sleeping(c), Some(false));
    world.joints.get_mut(joint).unwrap().bounded_pd =
        Some(BoundedPd::new(1.0, 0.0, 10.0, 0.0, 0.0).unwrap());
    solver(8, false, false).step(&mut world, 0.01, 1);
    assert_eq!(world.bodies.is_sleeping(b), Some(true));
    assert_eq!(world.bodies.is_sleeping(c), Some(true));
    world.joints.get_mut(joint).unwrap().bounded_pd =
        Some(BoundedPd::new(1.0, 0.0, 10.0, 0.0, 2.0).unwrap());
    solver(8, false, false).step(&mut world, 0.01, 1);
    assert_eq!(world.bodies.is_sleeping(c), Some(false));
}

#[test]
fn actuator_loads_break_force_or_torque_policy_once_then_stop_actuating() {
    for hinge in [false, true] {
        for threshold in [1.0, 3.0] {
            let (mut world, _, b, joint) =
                pair(hinge, BoundedPd::new(100.0, 0.0, 1.0, 0.0, 2.0).unwrap());
            let limit = if hinge {
                BreakLimit::new(100.0, threshold)
            } else {
                BreakLimit::new(threshold, 100.0)
            };
            world.breakable_joints.register(joint, limit).unwrap();
            solver(10, false, false).step(&mut world, 0.01, 1);
            assert_eq!(world.joints.get(joint).unwrap().enabled, threshold > 2.0);
            let before = if hinge {
                world.bodies.angular_velocity(b).unwrap().z
            } else {
                world.bodies.linear_velocity(b).unwrap().x
            };
            solver(10, false, false).step(&mut world, 0.01, 1);
            let after = if hinge {
                world.bodies.angular_velocity(b).unwrap().z
            } else {
                world.bodies.linear_velocity(b).unwrap().x
            };
            near(
                f64::from(after - before),
                if threshold > 2.0 { 0.02 } else { 0.0 },
                2.0e-6,
            );
        }
    }
}

#[test]
fn snapshots_hash_policy_failure_and_disabled_hinge_history() {
    let (mut world, a, b, joint) = pair(true, BoundedPd::new(0.0, 0.0, 0.0, 0.0, 0.0).unwrap());
    world.joints.get_mut(joint).unwrap().enabled = false;
    world.bodies.set_angular_velocity(b, Vec3::Z * 8.0);
    for _ in 0..100 {
        solver(2, false, false).step(&mut world, 0.01, 1);
    }
    let angle = world
        .joints
        .get(joint)
        .unwrap()
        .revolute_state()
        .angle()
        .unwrap();
    assert!(angle > core::f64::consts::TAU);
    world.joints.get_mut(joint).unwrap().enabled = true;
    world.joints.get_mut(joint).unwrap().bounded_pd =
        Some(BoundedPd::new(angle + 1.0, 0.0, 3.0, 0.0, 5.0).unwrap());
    let baseline = hash_world(&world).unwrap();
    let mut restored = FullStateSnapshot::capture(100, &world).fork_world();
    assert_eq!(baseline, hash_world(&restored).unwrap());
    restored.joints.get_mut(joint).unwrap().bounded_pd_failure =
        Some(BoundedPdFailure::InvalidGeometry);
    assert_ne!(baseline, hash_world(&restored).unwrap());
    restored.joints.get_mut(joint).unwrap().bounded_pd_failure = None;
    restored.joints.get_mut(joint).unwrap().bounded_pd = Some(drive());
    assert_ne!(baseline, hash_world(&restored).unwrap());
    restored = FullStateSnapshot::capture(100, &world).fork_world();
    for _ in 0..16 {
        solver(4, false, false).step(&mut world, 0.005, 1);
        solver(4, false, false).step(&mut restored, 0.005, 1);
        assert_eq!(hash_world(&world).unwrap(), hash_world(&restored).unwrap());
    }
    assert_eq!(world.bodies.orientation(a), Some(Quat::IDENTITY));
}

struct FailAt {
    calls: u32,
    fail_at: u32,
}
impl FieldBatchSampler for FailAt {
    type Error = ();
    fn sample_batch(
        &mut self,
        _: &FieldRegistry,
        bodies: &[FieldBody],
    ) -> Result<Vec<FieldSample>, ()> {
        let call = self.calls;
        self.calls += 1;
        if call == self.fail_at {
            Err(())
        } else {
            Ok(vec![FieldSample::default(); bodies.len()])
        }
    }
}

#[test]
fn field_rejection_has_no_drive_sleep_or_history_mutation_in_the_failed_substep() {
    for hinge in [false, true] {
        for completed in [0, 1] {
            let (mut world, _, body, _) = pair(hinge, drive());
            world.config.sleep.enabled = true;
            world.bodies.solver_view_mut().sleeping[body.index() as usize] = true;
            let mut expected = FullStateSnapshot::capture(0, &world).fork_world();
            let solve = solver(5, false, false);
            let result = solve.step_with_field_sampler(
                &mut world,
                0.02,
                2,
                &mut FailAt {
                    calls: 0,
                    fail_at: completed,
                },
            );
            assert!(
                matches!(result, Err(FieldStepError::Sampling { completed_substeps, .. }) if completed_substeps == completed)
            );
            if completed == 1 {
                solve
                    .step_with_field_sampler(
                        &mut expected,
                        0.01,
                        1,
                        &mut FailAt {
                            calls: 0,
                            fail_at: 10,
                        },
                    )
                    .unwrap();
            }
            assert_eq!(hash_world(&world).unwrap(), hash_world(&expected).unwrap());
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn serial_and_parallel_islands_preserve_drives_history_loads_and_snapshot_hash() {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    for index in 0..12 {
        let origin = Vec3::Y * index as f32;
        let a = world.spawn(BodyDesc::static_at(origin));
        let b = world.spawn(BodyDesc::dynamic_at(origin).with_damping(0.0, 0.0));
        let kind = if index % 2 == 0 {
            JointKind::Revolute(RevoluteJoint::new(Vec3::Z))
        } else {
            JointKind::Prismatic(PrismaticJoint::new(Vec3::X))
        };
        let handle = world.spawn_joint(
            JointDesc::new(JointAnchor::at_center(a), JointAnchor::at_center(b), kind)
                .with_bounded_pd(drive()),
        );
        world
            .breakable_joints
            .register(handle, BreakLimit::new(1000.0, 1000.0).with_fatigue(0.1))
            .unwrap();
    }
    let mut parallel = FullStateSnapshot::capture(0, &world).fork_world();
    let mut colored = FullStateSnapshot::capture(0, &world).fork_world();
    for _ in 0..20 {
        solver(5, false, false).step(&mut world, 0.01, 2);
        solver(5, true, false).step(&mut parallel, 0.01, 2);
        // One joint per island makes the colored order equivalent as well.
        solver(5, true, true).step(&mut colored, 0.01, 2);
        assert_eq!(hash_world(&world).unwrap(), hash_world(&parallel).unwrap());
        assert_eq!(hash_world(&world).unwrap(), hash_world(&colored).unwrap());
    }
}

#[test]
fn independent_reduced_decay_rejects_atomically_with_or_without_a_motor() {
    let motor = JointMotor::new(Vec3::ONE, Vec3::ONE, 2.0, 1.0, Vec3::splat(10.0));
    for base in [ArticulationBase::Fixed, ArticulationBase::Floating] {
        for hinge in [false, true] {
            for has_motor in [false, true] {
                for broken in [0usize, 1] {
                    let surviving = 1 - broken;
                    let mut policy = if hinge {
                        ArticulationJoint::revolute(Vec3::Z)
                    } else {
                        ArticulationJoint::prismatic(Vec3::X)
                    }
                    .with_damping(0.25);
                    if has_motor {
                        policy = policy.with_motor(motor);
                    }
                    let inertia = SpatialInertia::diagonal(2.0, Vec3::ZERO, Vec3::splat(7.0));
                    let mut builder =
                        ArticulationBuilder::new(base, inertia).with_gravity(Vec3::ZERO);
                    for link in 0usize..2 {
                        builder
                            .add_link(
                                link.checked_sub(1),
                                SpatialTransform::IDENTITY,
                                inertia,
                                if link == surviving {
                                    policy
                                } else {
                                    ArticulationJoint::fixed()
                                },
                            )
                            .unwrap();
                    }
                    let mut tree = builder.build().unwrap();
                    assert!(tree.set_velocity(0, 1.0));
                    tree.break_joint(broken);
                    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
                    // Rejection must not affect unrelated live bodies either.
                    world.spawn(BodyDesc::dynamic_at(Vec3::ONE));
                    let source = world.spawn_articulation(tree);
                    let before = hash_world(&world).unwrap();
                    let mut replay = FullStateSnapshot::capture(0, &world).fork_world();
                    let expected = Err(ArticulationHandoffError::UnsupportedJointPolicy(
                        UnsupportedJointPolicy::InertOrExtraParameters { link: surviving },
                    ));
                    for _ in 0..2 {
                        assert_eq!(world.handoff_articulation(source), expected);
                        assert_eq!(replay.handoff_articulation(source), expected);
                        assert_eq!(hash_world(&world).unwrap(), before);
                        assert_eq!(hash_world(&replay).unwrap(), before);
                        assert_eq!(world.bodies.len(), 1);
                        assert!(world.joints.is_empty());
                        assert!(world.articulations.contains(source));
                    }
                }
            }
        }
    }
}

#[test]
fn reduced_decay_scales_accelerated_velocity_independently_of_mass_and_inertia() {
    for hinge in [false, true] {
        for effective_mass in [2.0f32, 7.0] {
            for h in [0.01f32, 0.04] {
                for initial_speed in [-1.0f32, 1.0] {
                    for acceleration in [0.0f32, 3.0] {
                        let damping = 1.25;
                        let inertia = SpatialInertia::diagonal(
                            effective_mass,
                            Vec3::ZERO,
                            Vec3::splat(effective_mass),
                        );
                        let joint = if hinge {
                            ArticulationJoint::revolute(Vec3::Z)
                        } else {
                            ArticulationJoint::prismatic(Vec3::X)
                        }
                        .with_damping(damping);
                        let mut builder =
                            ArticulationBuilder::new(ArticulationBase::Fixed, inertia)
                                .with_gravity(Vec3::ZERO);
                        builder
                            .add_link(None, SpatialTransform::IDENTITY, inertia, joint)
                            .unwrap();
                        let mut tree = builder.build().unwrap();
                        assert!(tree.set_velocity(0, initial_speed));
                        let effort = effective_mass * acceleration;
                        assert!(tree.set_link_force(
                            0,
                            if hinge {
                                SpatialForce::new(Vec3::Z * effort, Vec3::ZERO)
                            } else {
                                SpatialForce::new(Vec3::ZERO, Vec3::X * effort)
                            }
                        ));
                        let mut speed = initial_speed;
                        let mut position = 0.0;
                        for _ in 0..4 {
                            let report = tree.step(h);
                            assert!(!report.degraded);
                            near(
                                f64::from(report.joint_accelerations[0]),
                                f64::from(acceleration),
                                1.0e-5,
                            );
                            speed = (speed + h * acceleration) / (1.0 + h * damping);
                            position += h * speed;
                            near(f64::from(tree.velocities()[0]), f64::from(speed), 1.0e-6);
                            near(f64::from(tree.positions()[0]), f64::from(position), 1.0e-6);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn physical_joint_damping_is_not_reduced_implicit_decay() {
    let h = 0.04f32;
    let damping = 1.25;
    let reduced_speed = 1.0 / (1.0 + f64::from(h) * damping);
    for hinge in [false, true] {
        let mut observed = Vec::new();
        for effective_mass in [2.0f32, 7.0] {
            let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
            world.config.sleep.enabled = false;
            let a = world.spawn(BodyDesc::static_at(Vec3::ZERO));
            let b = world.spawn(
                BodyDesc::dynamic_at(Vec3::ZERO)
                    .with_damping(0.0, 0.0)
                    .with_mass_properties(MassProperties {
                        inv_mass: effective_mass.recip(),
                        inv_inertia: Vec3::splat(effective_mass.recip()),
                    }),
            );
            let kind = if hinge {
                world.bodies.set_angular_velocity(b, Vec3::Z);
                JointKind::Revolute(RevoluteJoint::new(Vec3::Z))
            } else {
                world.bodies.set_linear_velocity(b, Vec3::X);
                JointKind::Prismatic(PrismaticJoint::new(Vec3::X))
            };
            let policy = BoundedPd::new(0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap()
                .with_joint_damping(damping)
                .unwrap();
            let joint = world.spawn_joint(
                JointDesc::new(JointAnchor::at_center(a), JointAnchor::at_center(b), kind)
                    .with_bounded_pd(policy),
            );
            let mut replay = FullStateSnapshot::capture(0, &world).fork_world();
            solver(4, false, false).step(&mut world, h, 1);
            solver(4, false, false).step(&mut replay, h, 1);
            assert_eq!(hash_world(&world).unwrap(), hash_world(&replay).unwrap());
            assert_eq!(world.joints.get(joint).unwrap().bounded_pd_failure, None);
            let speed = if hinge {
                world.bodies.angular_velocity(b).unwrap().z
            } else {
                world.bodies.linear_velocity(b).unwrap().x
            };
            // Explicit physical effort changes velocity through inverse mass.
            let predicted = 1.0 - f64::from(h) * damping / f64::from(effective_mass);
            // XPBD's normalized first-order quaternion prediction/recovery is
            // a separate, known discretization effect, not joint damping.
            let expected = if hinge {
                predicted / (1.0 + (0.5 * f64::from(h) * predicted).powi(2)).sqrt()
            } else {
                predicted
            };
            near(f64::from(speed), expected, 2.0e-6);
            assert!((f64::from(speed) - reduced_speed).abs() > 0.01);
            observed.push(speed);
        }
        assert!(observed[1] > observed[0] + 0.01);
    }
}

#[test]
fn armature_and_non_equivalent_chart_policies_still_reject_atomically() {
    let motor = JointMotor::new(Vec3::ONE, Vec3::ONE, 2.0, 1.0, Vec3::splat(10.0));
    for policy in [
        ArticulationJoint::prismatic(Vec3::X)
            .with_motor(motor)
            .with_armature(0.1),
        ArticulationJoint::spherical().with_motor(motor),
        ArticulationJoint::fixed().with_limits(JointLimits::scalar(-1.0, 1.0)),
        ArticulationJoint::spherical().with_limits(JointLimits::scalar(-1.0, 1.0)),
    ] {
        let inertia = SpatialInertia::diagonal(1.0, Vec3::ZERO, Vec3::ONE);
        let mut builder =
            ArticulationBuilder::new(ArticulationBase::Fixed, inertia).with_gravity(Vec3::ZERO);
        builder
            .add_link(None, SpatialTransform::IDENTITY, inertia, policy)
            .unwrap();
        builder
            .add_link(
                Some(0),
                SpatialTransform::IDENTITY,
                inertia,
                ArticulationJoint::fixed(),
            )
            .unwrap();
        let mut tree = builder.build().unwrap();
        tree.break_joint(1);
        let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
        let source = world.spawn_articulation(tree);
        let before = hash_world(&world).unwrap();
        assert!(matches!(
            world.handoff_articulation(source),
            Err(ArticulationHandoffError::UnsupportedJointPolicy(
                UnsupportedJointPolicy::Motor { link: 0 }
                    | UnsupportedJointPolicy::InertOrExtraParameters { link: 0 }
            ))
        ));
        assert_eq!(before, hash_world(&world).unwrap());
    }
}

#[test]
fn unsupported_or_conflicting_native_drives_report_failures_instead_of_substitution() {
    let (mut world, _, _, joint) = pair(false, drive());
    if let JointKind::Prismatic(ref mut params) = world.joints.get_mut(joint).unwrap().kind {
        params.motor = Some(Motor::velocity(0.0, 1.0));
    }
    solver(1, false, false).step(&mut world, 0.01, 1);
    assert_eq!(
        world.joints.get(joint).unwrap().bounded_pd_failure,
        Some(BoundedPdFailure::ConflictingMotor)
    );
    world.joints.get_mut(joint).unwrap().kind = JointKind::Spherical(Default::default());
    solver(1, false, false).step(&mut world, 0.01, 1);
    assert_eq!(
        world.joints.get(joint).unwrap().bounded_pd_failure,
        Some(BoundedPdFailure::UnsupportedFamily)
    );
}

#[cfg(feature = "serialize")]
#[test]
fn policy_joint_and_failure_roundtrip_preserves_history_and_replay() {
    use prism_physics_core::joint::BoundedPdError;

    for hinge in [false, true] {
        let (mut world, a, b, handle) = pair(hinge, drive());
        if hinge {
            assert!(world
                .joints
                .get_mut(handle)
                .unwrap()
                .set_revolute_coordinate(
                    20.0,
                    [
                        world.bodies.orientation(a).unwrap(),
                        world.bodies.orientation(b).unwrap()
                    ],
                ));
        }
        let joint = *world.joints.get(handle).unwrap();
        let restored_joint = serde_tape::roundtrip(&joint);
        assert_eq!(joint, restored_joint);
        let desc = JointDesc::new(joint.anchor_a, joint.anchor_b, joint.kind)
            .with_bounded_pd(joint.bounded_pd.unwrap());
        assert_eq!(desc, serde_tape::roundtrip(&desc));
        let mut restored = FullStateSnapshot::capture(0, &world).fork_world();
        *restored.joints.get_mut(handle).unwrap() = restored_joint;
        for _ in 0..8 {
            solver(4, false, false).step(&mut world, 0.002, 2);
            solver(4, false, false).step(&mut restored, 0.002, 2);
            assert_eq!(hash_world(&world).unwrap(), hash_world(&restored).unwrap());
        }
    }
    for failure in [
        BoundedPdFailure::UnsupportedFamily,
        BoundedPdFailure::ConflictingMotor,
        BoundedPdFailure::InvalidEndpoints,
        BoundedPdFailure::InvalidGeometry,
        BoundedPdFailure::Sample(BoundedPdError::NonFiniteState),
        BoundedPdFailure::NonFiniteImpulse,
    ] {
        assert_eq!(failure, serde_tape::roundtrip(&failure));
    }
}

// Same dependency-free typed-token roundtrip harness as revolute_winding.
// unsupported types fail explicitly. This is test machinery, not a wire format.
#[cfg(feature = "serialize")]
mod serde_tape {
    use serde::de::{DeserializeSeed, EnumAccess, SeqAccess, VariantAccess, Visitor};
    use serde::ser::{
        Impossible, SerializeSeq, SerializeStruct, SerializeTuple, SerializeTupleStruct,
    };
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::VecDeque;
    use std::fmt;

    #[derive(Debug)]
    pub struct Error(String);
    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for Error {}
    impl serde::ser::Error for Error {
        fn custom<T: fmt::Display>(message: T) -> Self {
            Self(message.to_string())
        }
    }
    impl serde::de::Error for Error {
        fn custom<T: fmt::Display>(message: T) -> Self {
            Self(message.to_string())
        }
    }

    #[derive(Debug)]
    enum Token {
        Bool(bool),
        I(i64),
        U(u64),
        F32(f32),
        F64(f64),
        Unit,
        None,
        Some,
        Sequence(usize),
        Variant(u32),
    }
    #[derive(Default)]
    pub struct Tape(VecDeque<Token>);
    fn unsupported<T>() -> Result<T, Error> {
        Err(Error("unsupported tape type".into()))
    }

    pub fn roundtrip<T: Serialize + for<'de> Deserialize<'de>>(value: &T) -> T {
        let mut tape = Tape::default();
        value.serialize(&mut tape).unwrap();
        let result = T::deserialize(&mut tape).unwrap();
        assert!(tape.0.is_empty(), "unconsumed serialized fields");
        result
    }

    macro_rules! integer {
        ($method:ident, $ty:ty, $wide:ident, $convert:ty) => {
            fn $method(self, value: $ty) -> Result<(), Error> {
                self.$wide(<$convert>::from(value))
            }
        };
    }
    impl<'a> Serializer for &'a mut Tape {
        type Ok = ();
        type Error = Error;
        type SerializeSeq = Self;
        type SerializeTuple = Self;
        type SerializeTupleStruct = Self;
        type SerializeTupleVariant = Impossible<(), Error>;
        type SerializeMap = Impossible<(), Error>;
        type SerializeStruct = Self;
        type SerializeStructVariant = Impossible<(), Error>;

        fn serialize_bool(self, value: bool) -> Result<(), Error> {
            self.0.push_back(Token::Bool(value));
            Ok(())
        }
        integer!(serialize_i8, i8, serialize_i64, i64);
        integer!(serialize_i16, i16, serialize_i64, i64);
        integer!(serialize_i32, i32, serialize_i64, i64);
        integer!(serialize_u8, u8, serialize_u64, u64);
        integer!(serialize_u16, u16, serialize_u64, u64);
        integer!(serialize_u32, u32, serialize_u64, u64);
        fn serialize_i64(self, value: i64) -> Result<(), Error> {
            self.0.push_back(Token::I(value));
            Ok(())
        }
        fn serialize_u64(self, value: u64) -> Result<(), Error> {
            self.0.push_back(Token::U(value));
            Ok(())
        }
        fn serialize_f32(self, value: f32) -> Result<(), Error> {
            self.0.push_back(Token::F32(value));
            Ok(())
        }
        fn serialize_f64(self, value: f64) -> Result<(), Error> {
            self.0.push_back(Token::F64(value));
            Ok(())
        }
        fn serialize_char(self, _: char) -> Result<(), Error> {
            unsupported()
        }
        fn serialize_str(self, _: &str) -> Result<(), Error> {
            unsupported()
        }
        fn serialize_bytes(self, _: &[u8]) -> Result<(), Error> {
            unsupported()
        }
        fn serialize_unit(self) -> Result<(), Error> {
            self.0.push_back(Token::Unit);
            Ok(())
        }
        fn serialize_unit_struct(self, _: &'static str) -> Result<(), Error> {
            self.serialize_unit()
        }
        fn serialize_unit_variant(
            self,
            _: &'static str,
            index: u32,
            _: &'static str,
        ) -> Result<(), Error> {
            self.0.push_back(Token::Variant(index));
            self.serialize_unit()
        }
        fn serialize_none(self) -> Result<(), Error> {
            self.0.push_back(Token::None);
            Ok(())
        }
        fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), Error> {
            self.0.push_back(Token::Some);
            value.serialize(self)
        }
        fn serialize_newtype_struct<T: ?Sized + Serialize>(
            self,
            _: &'static str,
            value: &T,
        ) -> Result<(), Error> {
            value.serialize(self)
        }
        fn serialize_newtype_variant<T: ?Sized + Serialize>(
            self,
            _: &'static str,
            index: u32,
            _: &'static str,
            value: &T,
        ) -> Result<(), Error> {
            self.0.push_back(Token::Variant(index));
            value.serialize(self)
        }
        fn serialize_seq(self, len: Option<usize>) -> Result<Self, Error> {
            self.serialize_tuple(len.ok_or_else(|| Error("unknown length".into()))?)
        }
        fn serialize_tuple(self, len: usize) -> Result<Self, Error> {
            self.0.push_back(Token::Sequence(len));
            Ok(self)
        }
        fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Self, Error> {
            self.serialize_tuple(len)
        }
        fn serialize_struct(self, _: &'static str, len: usize) -> Result<Self, Error> {
            self.serialize_tuple(len)
        }
        fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Error> {
            unsupported()
        }
        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeTupleVariant, Error> {
            unsupported()
        }
        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            _: &'static str,
            _: usize,
        ) -> Result<Self::SerializeStructVariant, Error> {
            unsupported()
        }
    }
    impl SerializeSeq for &mut Tape {
        type Ok = ();
        type Error = Error;
        fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            value.serialize(&mut **self)
        }
        fn end(self) -> Result<(), Error> {
            Ok(())
        }
    }
    impl SerializeTuple for &mut Tape {
        type Ok = ();
        type Error = Error;
        fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            value.serialize(&mut **self)
        }
        fn end(self) -> Result<(), Error> {
            Ok(())
        }
    }
    impl SerializeTupleStruct for &mut Tape {
        type Ok = ();
        type Error = Error;
        fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            value.serialize(&mut **self)
        }
        fn end(self) -> Result<(), Error> {
            Ok(())
        }
    }
    impl SerializeStruct for &mut Tape {
        type Ok = ();
        type Error = Error;
        fn serialize_field<T: ?Sized + Serialize>(
            &mut self,
            _: &'static str,
            value: &T,
        ) -> Result<(), Error> {
            value.serialize(&mut **self)
        }
        fn end(self) -> Result<(), Error> {
            Ok(())
        }
    }

    impl<'de> Deserializer<'de> for &mut Tape {
        type Error = Error;
        fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
            match self
                .0
                .pop_front()
                .ok_or_else(|| Error("unexpected end of tape".into()))?
            {
                Token::Bool(value) => visitor.visit_bool(value),
                Token::I(value) => visitor.visit_i64(value),
                Token::U(value) => visitor.visit_u64(value),
                Token::F32(value) => visitor.visit_f32(value),
                Token::F64(value) => visitor.visit_f64(value),
                Token::Unit => visitor.visit_unit(),
                Token::None => visitor.visit_none(),
                Token::Some => visitor.visit_some(self),
                Token::Sequence(remaining) => visitor.visit_seq(Sequence {
                    tape: self,
                    remaining,
                }),
                Token::Variant(index) => visitor.visit_enum(Variant { tape: self, index }),
            }
        }
        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
            option unit unit_struct newtype_struct seq tuple tuple_struct map struct enum identifier ignored_any
        }
    }
    struct Sequence<'a> {
        tape: &'a mut Tape,
        remaining: usize,
    }
    impl<'de> SeqAccess<'de> for Sequence<'_> {
        type Error = Error;
        fn next_element_seed<T: DeserializeSeed<'de>>(
            &mut self,
            seed: T,
        ) -> Result<Option<T::Value>, Error> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;
            seed.deserialize(&mut *self.tape).map(Some)
        }
        fn size_hint(&self) -> Option<usize> {
            Some(self.remaining)
        }
    }
    struct Variant<'a> {
        tape: &'a mut Tape,
        index: u32,
    }
    impl<'a, 'de> EnumAccess<'de> for Variant<'a> {
        type Error = Error;
        type Variant = &'a mut Tape;
        fn variant_seed<T: DeserializeSeed<'de>>(
            self,
            seed: T,
        ) -> Result<(T::Value, Self::Variant), Error> {
            let index = serde::de::value::U32Deserializer::<Error>::new(self.index);
            Ok((seed.deserialize(index)?, self.tape))
        }
    }
    impl<'de> VariantAccess<'de> for &mut Tape {
        type Error = Error;
        fn unit_variant(self) -> Result<(), Error> {
            <()>::deserialize(self)
        }
        fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Error> {
            seed.deserialize(self)
        }
        fn tuple_variant<V: Visitor<'de>>(self, len: usize, visitor: V) -> Result<V::Value, Error> {
            self.deserialize_tuple(len, visitor)
        }
        fn struct_variant<V: Visitor<'de>>(
            self,
            fields: &'static [&'static str],
            visitor: V,
        ) -> Result<V::Value, Error> {
            self.deserialize_struct("variant", fields, visitor)
        }
    }
}
