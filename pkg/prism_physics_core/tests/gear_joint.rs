//! Position-level transmission, winding ownership and replay regressions.

use glam::{Quat, Vec3};
use prism_physics_core::snapshot::hash::{hash_state, hash_world};
use prism_physics_core::snapshot::world::FullStateSnapshot;
use prism_physics_core::snapshot::StateSnapshot;
use prism_physics_core::solver::xpbd::{gear, joint_constraint, XpbdConfig, XpbdSolver};
use prism_physics_core::solver::Solver;
use prism_physics_core::{
    BodyDesc, BodyHandle, GearJoint, GearJointError, JointAnchor, JointDesc, JointHandle,
    JointKind, MassProperties, PhysicsWorld, WorldConfig,
};

fn world() -> PhysicsWorld {
    let mut config = WorldConfig::with_gravity(Vec3::ZERO);
    config.sleep.enabled = false;
    PhysicsWorld::new(config)
}

fn pair(
    model: GearJoint,
    static_a: bool,
    static_b: bool,
) -> (PhysicsWorld, BodyHandle, BodyHandle, JointHandle) {
    let mut world = world();
    let desc = |is_static, position| {
        if is_static {
            BodyDesc::static_at(position)
        } else {
            BodyDesc::dynamic_at(position)
        }
    };
    let a = world.spawn(desc(static_a, Vec3::new(-2.0, 1.0, 0.0)));
    let b = world.spawn(desc(static_b, Vec3::new(3.0, -1.0, 2.0)));
    let joint = world.spawn_joint(JointDesc::gear(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        model,
    ));
    (world, a, b, joint)
}

fn project(world: &mut PhysicsWorld, iterations: usize, h: f32) {
    let PhysicsWorld { bodies, joints, .. } = world;
    let mut view = bodies.solver_view_mut();
    gear::begin_substep(&view, joints);
    for _ in 0..iterations {
        joint_constraint::solve_joints(&mut view, joints, h);
    }
    gear::finish_substep(&view, joints);
}

fn angles(world: &PhysicsWorld, joint: JointHandle) -> [f64; 2] {
    world
        .joints
        .get(joint)
        .unwrap()
        .gear_state()
        .angles()
        .unwrap()
}

fn error(world: &PhysicsWorld, joint: JointHandle) -> f64 {
    let JointKind::Gear(model) = world.joints.get(joint).unwrap().kind else {
        panic!("expected gear")
    };
    let [a, b] = angles(world, joint);
    b - f64::from(model.ratio) * a - model.phase
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected} (tolerance {tolerance})"
    );
}

fn rotate_driver(world: &mut PhysicsWorld, body: BodyHandle, increment: f32) {
    let q = world.bodies.orientation(body).unwrap();
    assert!(world
        .bodies
        .set_orientation(body, (Quat::from_rotation_z(increment) * q).normalize(),));
}

#[test]
fn signed_ratio_and_phase_correct_positions_from_zero_velocity() {
    for ratio in [2.0, -2.0] {
        let (mut world, a, b, joint) = pair(
            GearJoint::new(Vec3::Z, Vec3::Z, ratio).with_phase(0.6),
            false,
            false,
        );
        let positions = [world.bodies.position(a), world.bodies.position(b)];
        project(&mut world, 12, 1.0 / 60.0);
        near(error(&world, joint), 0.0, 2.0e-6);
        let [theta_a, theta_b] = angles(&world, joint);
        assert!(theta_a * f64::from(ratio) < 0.0);
        assert!(theta_b > 0.0);
        assert_eq!(
            positions,
            [world.bodies.position(a), world.bodies.position(b)]
        );
        // No velocity input or velocity-level correction is involved.
        assert_eq!(world.bodies.angular_velocity(a), Some(Vec3::ZERO));
        assert_eq!(world.bodies.angular_velocity(b), Some(Vec3::ZERO));
    }
}

#[test]
fn angular_projection_is_inertia_and_ratio_squared_weighted() {
    let mut world = world();
    let body = |inertia| {
        BodyDesc::dynamic_at(Vec3::ZERO).with_mass_properties(MassProperties {
            inv_mass: 1.0,
            inv_inertia: inertia,
        })
    };
    let a = world.spawn(body(Vec3::new(3.0, 5.0, 0.4)));
    let b = world.spawn(body(Vec3::new(7.0, 9.0, 2.3)));
    let ratio = -1.5;
    let phase = 0.3;
    let joint = world.spawn_joint(JointDesc::gear(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        GearJoint::new(Vec3::Z, Vec3::Z, ratio).with_phase(phase),
    ));
    project(&mut world, 1, 0.1);
    let lambda = phase / (f64::from(ratio).powi(2) * 0.4 + 2.3);
    let [theta_a, theta_b] = angles(&world, joint);
    near(theta_a, -f64::from(ratio) * 0.4 * lambda, 2.0e-7);
    near(theta_b, 2.3 * lambda, 2.0e-7);
    near(error(&world, joint), 0.0, 4.0e-7);
}

#[test]
fn static_end_on_either_side_and_two_static_ends() {
    for (static_a, static_b) in [(true, false), (false, true), (true, true)] {
        let (mut world, a, b, joint) = pair(
            GearJoint::new(Vec3::Z, Vec3::Z, -2.0).with_phase(0.4),
            static_a,
            static_b,
        );
        project(&mut world, 12, 0.1);
        if static_a {
            assert_eq!(world.bodies.orientation(a), Some(Quat::IDENTITY));
        }
        if static_b {
            assert_eq!(world.bodies.orientation(b), Some(Quat::IDENTITY));
        }
        if static_a && static_b {
            near(error(&world, joint), -0.4, 0.0);
        } else {
            near(error(&world, joint), 0.0, 2.0e-6);
        }
    }
}

#[test]
fn independent_local_axes_anchor_frames_and_world_zero_frames() {
    let reference_a = Quat::from_rotation_x(0.7);
    let reference_b = Quat::from_rotation_y(-0.4);
    let local_a = Quat::from_rotation_z(0.3);
    let local_b = Quat::from_rotation_y(0.6);
    let model = GearJoint::new(Vec3::Y * 4.0, -Vec3::X * 3.0, -0.5)
        .with_phase(0.1)
        .with_reference_frames(reference_a, reference_b);
    let (mut world, a, b, joint) = pair(model, false, false);
    {
        let live = world.joints.get_mut(joint).unwrap();
        live.anchor_a.local_frame = local_a;
        live.anchor_b.local_frame = local_b;
    }
    world.bodies.set_orientation(
        a,
        reference_a * Quat::from_rotation_y(0.35) * local_a.inverse(),
    );
    world.bodies.set_orientation(
        b,
        reference_b * Quat::from_rotation_x(0.2) * local_b.inverse(),
    );
    project(&mut world, 0, 0.1);
    near(angles(&world, joint)[0], 0.35, 2.0e-7);
    near(angles(&world, joint)[1], -0.2, 2.0e-7);
    project(&mut world, 16, 0.1);
    near(error(&world, joint), 0.0, 2.0e-6);
    let shaft_a = world.bodies.orientation(a).unwrap() * local_a * Vec3::Y;
    let shaft_b = world.bodies.orientation(b).unwrap() * local_b * -Vec3::X;
    assert!((shaft_a - reference_a * Vec3::Y).length() < 2.0e-6);
    assert!((shaft_b - reference_b * -Vec3::X).length() < 2.0e-6);
    // The unrelated, nonparallel shafts are not forcibly aligned.
    assert!(shaft_a.cross(shaft_b).length() > 0.5);
}

#[test]
fn compliance_keeps_lambda_across_position_iterations() {
    let model = GearJoint::new(Vec3::Z, Vec3::Z, 1.0)
        .with_phase(0.3)
        .with_compliance(0.02);
    let (mut once, _, _, joint) = pair(model, true, false);
    let mut many = once.clone();
    project(&mut once, 1, 0.1);
    project(&mut many, 32, 0.1);
    let expected = 0.3 / (1.0 + f64::from(model.compliance) / f64::from(0.1_f32).powi(2));
    near(angles(&once, joint)[1], expected, 2.0e-7);
    near(angles(&many, joint)[1], expected, 2.0e-7);
    assert!(
        error(&many, joint).abs() > 0.15,
        "iterations must not turn a compliant gear rigid"
    );
}

#[test]
fn fractional_ratios_cross_many_turns_and_quaternion_sign_seams() {
    for ratio in [2.25, -1.5] {
        let (mut world, a, b, joint) = pair(GearJoint::new(Vec3::Z, Vec3::Z, ratio), true, false);
        project(&mut world, 0, 0.01);
        let mut last_b = world.bodies.orientation(b).unwrap();
        for tick in 0..1200 {
            rotate_driver(&mut world, a, 0.07);
            if tick % 17 == 0 {
                // q and -q are identical physical poses, even at a winding seam.
                world
                    .bodies
                    .set_orientation(a, -world.bodies.orientation(a).unwrap());
                world
                    .bodies
                    .set_orientation(b, -world.bodies.orientation(b).unwrap());
            }
            project(&mut world, 6, 0.01);
            near(error(&world, joint), 0.0, 3.0e-6);
            let next_b = world.bodies.orientation(b).unwrap();
            assert!(
                last_b.dot(next_b).abs() > 0.99,
                "gear tooth jump at tick {tick}"
            );
            last_b = next_b;
        }
        let [theta_a, theta_b] = angles(&world, joint);
        near(theta_a, 84.0, 3.0e-4);
        near(theta_b, f64::from(ratio) * theta_a, 3.0e-6);
        assert!(theta_b.abs() > 10.0 * core::f64::consts::TAU);
    }
}

#[test]
fn phase_can_span_multiple_full_turns_without_wrapping_the_error() {
    let phase = 6.0 * core::f64::consts::TAU + 0.2;
    let (mut world, _, _, joint) = pair(
        GearJoint::new(Vec3::Z, Vec3::Z, -1.5).with_phase(phase),
        true,
        false,
    );
    project(&mut world, 120, 0.01);
    near(angles(&world, joint)[1], phase, 3.0e-6);
    near(error(&world, joint), 0.0, 3.0e-6);
}

#[test]
fn disabled_gears_observe_winding_and_reengage_without_rebasing() {
    let (mut world, a, _, joint) = pair(GearJoint::new(Vec3::Z, Vec3::Z, 2.25), true, false);
    world.joints.get_mut(joint).unwrap().enabled = false;
    project(&mut world, 0, 0.01);
    for _ in 0..60 {
        rotate_driver(&mut world, a, 0.2);
        project(&mut world, 1, 0.01);
    }
    let before = angles(&world, joint);
    near(before[0], 12.0, 3.0e-5);
    near(before[1], 0.0, 0.0);
    world.joints.get_mut(joint).unwrap().enabled = true;
    project(&mut world, 100, 0.01);
    near(angles(&world, joint)[0], before[0], 1.0e-12);
    near(angles(&world, joint)[1], 2.25 * before[0], 3.0e-6);
}

#[test]
fn retuning_preserves_history_and_rebinding_or_recycling_resets_it() {
    let model = GearJoint::new(Vec3::Z, Vec3::Z, -1.5);
    let (mut world, a, b, joint) = pair(model, true, false);
    project(&mut world, 0, 0.01);
    for _ in 0..80 {
        rotate_driver(&mut world, a, 0.1);
        project(&mut world, 3, 0.01);
    }
    let before = angles(&world, joint);
    if let JointKind::Gear(model) = &mut world.joints.get_mut(joint).unwrap().kind {
        model.ratio = 2.25;
        model.phase = 20.0;
        model.compliance = 0.01;
    }
    project(&mut world, 0, 0.01);
    assert_eq!(angles(&world, joint), before);
    world.joints.get_mut(joint).unwrap().anchor_a.local_frame = Quat::from_rotation_z(0.25);
    project(&mut world, 0, 0.01);
    assert!(angles(&world, joint)[0].abs() < core::f64::consts::PI);
    world.joints.get_mut(joint).unwrap().reset_gear_state();
    assert!(world
        .joints
        .get(joint)
        .unwrap()
        .gear_state()
        .angles()
        .is_none());
    assert!(world.joints.remove(joint));
    let next = world.spawn_joint(JointDesc::gear(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        model,
    ));
    assert_eq!(next.index(), joint.index());
    assert_ne!(next.generation(), joint.generation());
    assert!(world
        .joints
        .get(next)
        .unwrap()
        .gear_state()
        .angles()
        .is_none());
    project(&mut world, 0, 0.01);
    assert!(angles(&world, next)[0].abs() < core::f64::consts::PI);
}

#[test]
fn invalid_parameters_are_rejected_and_projection_is_a_finite_noop() {
    let good = GearJoint::new(Vec3::Z, Vec3::Z, 1.0);
    let mut cases = Vec::new();
    for ratio in [0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        cases.push((GearJoint { ratio, ..good }, GearJointError::InvalidRatio));
    }
    for axis in [
        Vec3::ZERO,
        Vec3::new(f32::NAN, 0.0, 1.0),
        Vec3::splat(f32::INFINITY),
    ] {
        cases.push((
            GearJoint {
                axis_a: axis,
                ..good
            },
            GearJointError::InvalidAxis,
        ));
        cases.push((
            GearJoint {
                axis_b: axis,
                ..good
            },
            GearJointError::InvalidAxis,
        ));
    }
    for phase in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.push((good.with_phase(phase), GearJointError::InvalidPhase));
    }
    for compliance in [-0.1, f32::NAN, f32::INFINITY] {
        cases.push((
            good.with_compliance(compliance),
            GearJointError::InvalidCompliance,
        ));
    }
    for frame in [
        Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
        Quat::from_xyzw(0.0, 0.0, 0.0, 2.0),
    ] {
        cases.push((
            good.with_reference_frames(frame, Quat::IDENTITY),
            GearJointError::InvalidReferenceFrame,
        ));
        cases.push((
            good.with_reference_frames(Quat::IDENTITY, frame),
            GearJointError::InvalidReferenceFrame,
        ));
    }
    for (model, expected) in cases {
        assert_eq!(model.validate(), Err(expected));
        let (mut world, a, b, joint) = pair(model, false, false);
        project(&mut world, 8, 0.01);
        assert_eq!(world.bodies.orientation(a), Some(Quat::IDENTITY));
        assert_eq!(world.bodies.orientation(b), Some(Quat::IDENTITY));
        assert!(world
            .joints
            .get(joint)
            .unwrap()
            .gear_state()
            .angles()
            .is_none());
    }
    // Normalization must not overflow or underflow on finite, nonzero axes.
    assert!(
        GearJoint::new(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN_POSITIVE), 1.0)
            .validate()
            .is_ok()
    );
}

#[test]
fn invalid_anchor_timestep_same_body_and_undefined_twist_are_safe() {
    let model = GearJoint::new(Vec3::Z, Vec3::Z, 1.0).with_phase(0.4);
    for h in [0.0, -0.1, f32::NAN, f32::INFINITY] {
        let (mut world, a, b, _) = pair(model, false, false);
        project(&mut world, 8, h);
        assert_eq!(world.bodies.orientation(a), Some(Quat::IDENTITY));
        assert_eq!(world.bodies.orientation(b), Some(Quat::IDENTITY));
    }
    let (mut world, a, b, joint) = pair(model, false, false);
    world.joints.get_mut(joint).unwrap().anchor_a.local_frame = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    project(&mut world, 8, 0.01);
    assert_eq!(world.bodies.orientation(b), Some(Quat::IDENTITY));
    world.joints.get_mut(joint).unwrap().anchor_a.local_frame = Quat::IDENTITY;
    world.joints.get_mut(joint).unwrap().anchor_b.body = a;
    project(&mut world, 8, 0.01);
    assert_eq!(world.bodies.orientation(a), Some(Quat::IDENTITY));
    world.joints.get_mut(joint).unwrap().anchor_b.body = b;
    let singular = Quat::from_rotation_x(core::f32::consts::PI);
    world.bodies.set_orientation(a, singular);
    project(&mut world, 8, 0.01);
    assert_eq!(world.bodies.orientation(a), Some(singular));
    assert_eq!(world.bodies.orientation(b), Some(Quat::IDENTITY));
}

fn spinning_world() -> (PhysicsWorld, JointHandle) {
    let (mut world, a, b, joint) = pair(GearJoint::new(Vec3::Z, Vec3::Z, -1.5), false, false);
    world.bodies.set_angular_velocity(a, Vec3::Z * 5.0);
    world.bodies.set_angular_velocity(b, Vec3::Z * -7.5);
    (world, joint)
}

fn solver() -> XpbdSolver {
    XpbdSolver::with_config(XpbdConfig {
        position_iterations: 6,
        parallel_islands: false,
        ..XpbdConfig::default()
    })
}

#[test]
fn multi_revolution_world_is_repeatable_and_full_snapshot_replays_bit_exactly() {
    let (mut world, joint) = spinning_world();
    let mut repeated = world.clone();
    let mut first_solver = solver();
    let mut repeated_solver = solver();
    for _ in 0..400 {
        first_solver.step(&mut world, 1.0 / 120.0, 1);
        repeated_solver.step(&mut repeated, 1.0 / 120.0, 1);
        near(error(&world, joint), 0.0, 3.0e-6);
    }
    assert!(angles(&world, joint)[0] > core::f64::consts::TAU);
    assert_eq!(hash_world(&world).unwrap(), hash_world(&repeated).unwrap());
    let snapshot = FullStateSnapshot::capture(400, &world);
    assert_eq!(snapshot.state_hash().unwrap(), hash_world(&world).unwrap());
    let mut replay = snapshot.fork_world();
    let mut fresh_solver = solver();
    for _ in 0..180 {
        first_solver.step(&mut world, 1.0 / 120.0, 1);
        fresh_solver.step(&mut replay, 1.0 / 120.0, 1);
    }
    assert_eq!(hash_world(&world).unwrap(), hash_world(&replay).unwrap());
    snapshot.restore_into(&mut replay);
    assert_eq!(hash_world(&replay).unwrap(), snapshot.state_hash().unwrap());
    assert_eq!(angles(&replay, joint), angles(&repeated, joint));
}

#[test]
fn full_hash_covers_gear_configuration_and_winding_not_just_body_poses() {
    let (mut world, a, _, joint) = pair(GearJoint::new(Vec3::Z, Vec3::Z, -1.5), true, false);
    project(&mut world, 0, 0.01);
    for _ in 0..80 {
        rotate_driver(&mut world, a, 0.1);
        project(&mut world, 3, 0.01);
    }
    let baseline = hash_world(&world).unwrap();
    let render = hash_state(&StateSnapshot::capture(&world));
    let mut rebased = world.clone();
    rebased.joints.get_mut(joint).unwrap().reset_gear_state();
    project(&mut rebased, 0, 0.01);
    assert_ne!(angles(&world, joint), angles(&rebased, joint));
    assert_ne!(baseline, hash_world(&rebased).unwrap());
    assert_eq!(render, hash_state(&StateSnapshot::capture(&rebased)));
    for field in 0..8 {
        let mut changed = world.clone();
        let live = changed.joints.get_mut(joint).unwrap();
        let JointKind::Gear(model) = &mut live.kind else {
            unreachable!()
        };
        match field {
            0 => model.ratio = 2.25,
            1 => model.phase = 0.25,
            2 => model.compliance = 0.001,
            3 => model.axis_a = Vec3::X,
            4 => model.axis_b = Vec3::Y,
            5 => model.reference_frame_a = Quat::from_rotation_x(0.2),
            6 => model.reference_frame_b = Quat::from_rotation_y(0.3),
            _ => live.anchor_a.local_frame = Quat::from_rotation_z(0.4),
        }
        assert_ne!(
            baseline,
            hash_world(&changed).unwrap(),
            "unhashed field {field}"
        );
        assert_eq!(render, hash_state(&StateSnapshot::capture(&changed)));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_islands_return_continuous_history_with_global_bindings() {
    let mut initial = world();
    initial.spawn(BodyDesc::dynamic_at(Vec3::new(100.0, 0.0, 0.0)));
    // Interleaved body slots force a nontrivial scratch-to-world remapping.
    let a0 = initial.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_angular_velocity(Vec3::Z * 5.0));
    let a1 = initial.spawn(BodyDesc::dynamic_at(Vec3::X).with_angular_velocity(Vec3::Z * 4.0));
    let b0 = initial.spawn(BodyDesc::dynamic_at(Vec3::Y).with_angular_velocity(Vec3::Z * -7.5));
    let b1 = initial.spawn(BodyDesc::dynamic_at(Vec3::Z).with_angular_velocity(Vec3::Z * 9.0));
    let mut handles = Vec::new();
    for (a, b, ratio) in [(a0, b0, -1.5), (a1, b1, 2.25)] {
        handles.push(initial.spawn_joint(JointDesc::gear(
            JointAnchor::at_center(a),
            JointAnchor::at_center(b),
            GearJoint::new(Vec3::Z, Vec3::Z, ratio),
        )));
    }
    let mut serial = initial.clone();
    let mut threaded = initial.clone();
    let mut coloured = initial;
    let mut serial_solver = solver();
    let mut threaded_solver = XpbdSolver::with_config(XpbdConfig {
        position_iterations: 6,
        parallel_islands: true,
        ..XpbdConfig::default()
    });
    let mut coloured_solver = XpbdSolver::with_config(XpbdConfig {
        position_iterations: 6,
        parallel_islands: true,
        parallel_within_island: true,
        ..XpbdConfig::default()
    });
    for tick in 0..500 {
        serial_solver.step(&mut serial, 1.0 / 120.0, 1);
        threaded_solver.step(&mut threaded, 1.0 / 120.0, 1);
        coloured_solver.step(&mut coloured, 1.0 / 120.0, 1);
        if tick % 50 == 0 {
            assert_eq!(hash_world(&serial).unwrap(), hash_world(&threaded).unwrap());
            // Each island has one gear: colouring preserves its operation order.
            assert_eq!(hash_world(&serial).unwrap(), hash_world(&coloured).unwrap());
        }
    }
    assert_eq!(hash_world(&serial).unwrap(), hash_world(&threaded).unwrap());
    assert_eq!(hash_world(&serial).unwrap(), hash_world(&coloured).unwrap());
    for joint in handles {
        assert_eq!(angles(&serial, joint), angles(&threaded, joint));
        assert!(angles(&threaded, joint)[0] > core::f64::consts::TAU);
        near(error(&threaded, joint), 0.0, 3.0e-6);
    }
}
