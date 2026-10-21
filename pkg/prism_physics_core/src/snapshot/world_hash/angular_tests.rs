//! Guard the recursive Debug encoding against dropping hidden angular state.
//! History is sampled directly so body poses, previous poses and velocities
//! remain bit-identical: a render-state change cannot mask a missing field.

use super::{hash_world, WorldHashError};
use crate::joint::{
    GearJoint, Joint, JointAnchor, JointDesc, JointHandle, JointKind, Motor, RevoluteJoint,
};
use crate::snapshot::{FullStateSnapshot, StateSnapshot};
use crate::solver::xpbd::XpbdConfig;
use crate::solver::{Solver, XpbdSolver};
use crate::state::{body::BodyDesc, handle::BodyHandle};
use crate::world::PhysicsWorld;
use glam::{Quat, Vec3};

struct Fixture {
    world: PhysicsWorld,
    joint: JointHandle,
    stale_a: BodyHandle,
    spare: BodyHandle,
}

fn fixture(kind: JointKind) -> Fixture {
    let mut world = PhysicsWorld::with_gravity(Vec3::ZERO);
    world.config.sleep.enabled = false;
    let stale_a = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    assert!(world.remove_body(stale_a));
    let a = world.spawn(BodyDesc::static_at(Vec3::ZERO));
    assert_eq!(a.index(), stale_a.index());
    assert_ne!(a.generation(), stale_a.generation());
    let b = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    let spare = world.spawn(BodyDesc::static_at(Vec3::X * 10.0));
    let joint = world.spawn_joint(JointDesc::new(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        kind,
    ));
    Fixture {
        world,
        joint,
        stale_a,
        spare,
    }
}

fn hinge() -> Fixture {
    fixture(JointKind::Revolute(RevoluteJoint::new(Vec3::Z)))
}

fn gear() -> Fixture {
    fixture(JointKind::Gear(GearJoint::new(Vec3::Z, Vec3::Z, -1.5)))
}

fn seed_hinge(world: &mut PhysicsWorld, handle: JointHandle, angle: f64) {
    assert!(world
        .joints
        .get_mut(handle)
        .unwrap()
        .set_revolute_coordinate(angle, [Quat::IDENTITY; 2]));
}

fn sample_gear(joint: &mut Joint, orientations: [Quat; 2]) {
    let JointKind::Gear(model) = joint.kind else {
        panic!("expected gear")
    };
    assert!(joint
        .gear_state
        .observe(model, joint.anchor_a, joint.anchor_b, orientations)
        .is_some());
}

fn wind_gear(world: &mut PhysicsWorld, handle: JointHandle, shaft: usize, direction: f32) {
    let joint = world.joints.get_mut(handle).unwrap();
    sample_gear(joint, [Quat::IDENTITY; 2]);
    // Every consecutive observation is less than pi apart; finish at exactly
    // identity rather than the rounded quaternion returned for 2*pi.
    for angle in [
        core::f32::consts::FRAC_PI_2,
        core::f32::consts::PI,
        -core::f32::consts::FRAC_PI_2,
        0.0,
    ] {
        let mut orientations = [Quat::IDENTITY; 2];
        orientations[shaft] = Quat::from_rotation_z(angle * direction);
        sample_gear(joint, orientations);
    }
    let angles = joint.gear_state().angles().unwrap();
    assert!((angles[shaft] - f64::from(direction) * core::f64::consts::TAU).abs() < 1.0e-6);
    assert_eq!(angles[1 - shaft], 0.0);
}

fn assert_hidden_difference(a: &PhysicsWorld, b: &PhysicsWorld, handle: JointHandle) {
    assert_eq!(StateSnapshot::capture(a), StateSnapshot::capture(b));
    // Authored parameters must not be responsible for the hash difference.
    let ja = a.joints.get(handle).unwrap();
    let jb = b.joints.get(handle).unwrap();
    assert_eq!(ja.anchor_a, jb.anchor_a);
    assert_eq!(ja.anchor_b, jb.anchor_b);
    assert_eq!(ja.kind, jb.kind);
    assert_eq!(ja.enabled, jb.enabled);
    assert_ne!(hash_world(a).unwrap(), hash_world(b).unwrap());
}

#[test]
fn revolute_turns_and_f64_precision_change_hash_without_changing_pose() {
    let Fixture {
        mut world, joint, ..
    } = hinge();
    seed_hinge(&mut world, joint, 0.0);
    for angle in [
        core::f64::consts::TAU,
        -core::f64::consts::TAU,
        1000.0 * core::f64::consts::TAU,
    ] {
        let mut changed = world.clone();
        seed_hinge(&mut changed, joint, angle);
        assert_eq!(
            world
                .joints
                .get(joint)
                .unwrap()
                .revolute_state()
                .principal_angle(),
            changed
                .joints
                .get(joint)
                .unwrap()
                .revolute_state()
                .principal_angle()
        );
        assert_hidden_difference(&world, &changed, joint);
    }
    let mut changed = world.clone();
    seed_hinge(&mut changed, joint, -0.0);
    assert_hidden_difference(&world, &changed, joint);
    // Values that would alias if history were narrowed to f32 before encoding.
    seed_hinge(&mut world, joint, 1000.0);
    seed_hinge(
        &mut changed,
        joint,
        f64::from_bits(1000.0_f64.to_bits() + 1),
    );
    assert_hidden_difference(&world, &changed, joint);
}

#[test]
fn revolute_binding_geometry_and_endpoint_generations_are_hashed() {
    let Fixture {
        mut world,
        joint,
        stale_a,
        spare,
    } = hinge();
    seed_hinge(&mut world, joint, core::f64::consts::TAU);
    for field in 0..6 {
        let mut changed = world.clone();
        let live = changed.joints.get_mut(joint).unwrap();
        let mut a = live.anchor_a;
        let mut b = live.anchor_b;
        let mut axis = Vec3::Z;
        match field {
            0 => a.body = spare,
            1 => b.body = spare,
            2 => a.local_frame = Quat::from_rotation_x(0.25),
            3 => b.local_frame = Quat::from_rotation_y(0.25),
            4 => axis = Vec3::X,
            _ => a.body = stale_a,
        }
        assert!(live
            .revolute_state
            .seed(core::f64::consts::TAU, axis, a, b, [Quat::IDENTITY; 2]));
        assert_eq!(live.revolute_state.angle(), Some(core::f64::consts::TAU));
        assert_eq!(live.revolute_state.principal_angle(), Some(0.0));
        assert_hidden_difference(&world, &changed, joint);
    }
}

#[test]
fn revolute_binding_presence_principal_sample_and_motor_scratch_are_hashed() {
    let Fixture {
        mut world, joint, ..
    } = hinge();
    let unbound = world.clone();
    seed_hinge(&mut world, joint, 0.0);
    assert_hidden_difference(&unbound, &world, joint);
    for field in 0..4 {
        let mut changed = world.clone();
        let live = changed.joints.get_mut(joint).unwrap();
        match field {
            0 => {
                // Same coordinate and binding, different last principal sample.
                assert!(live
                    .set_revolute_coordinate(0.0, [Quat::IDENTITY, Quat::from_rotation_z(0.25)]));
                assert_eq!(live.revolute_state.angle(), Some(0.0));
                assert!(live.revolute_state.principal_angle().unwrap() > 0.2);
            }
            1 => live.revolute_state.motor_lambda = 0.25,
            2 => live.revolute_state.motor_goal = Some(0.0),
            _ => live.revolute_state.motor_goal = Some(0.25),
        }
        assert_hidden_difference(&world, &changed, joint);
    }
}

#[test]
fn gear_turns_on_each_shaft_and_shaft_order_are_hashed_at_identical_pose() {
    let Fixture {
        mut world, joint, ..
    } = gear();
    sample_gear(world.joints.get_mut(joint).unwrap(), [Quat::IDENTITY; 2]);
    let mut wound_a = world.clone();
    wind_gear(&mut wound_a, joint, 0, 1.0);
    let mut wound_b = world.clone();
    wind_gear(&mut wound_b, joint, 1, 1.0);
    let mut negative = world.clone();
    wind_gear(&mut negative, joint, 0, -1.0);
    for changed in [&wound_a, &wound_b, &negative] {
        assert_hidden_difference(&world, changed, joint);
    }
    // The canonicalizer must not sort the two shaft coordinates as a set.
    assert_hidden_difference(&wound_a, &wound_b, joint);
    assert_hidden_difference(&wound_a, &negative, joint);
}

#[test]
fn gear_binding_encodes_both_anchors_axes_reference_frames_and_generations() {
    let Fixture {
        mut world,
        joint,
        stale_a,
        spare,
    } = gear();
    sample_gear(world.joints.get_mut(joint).unwrap(), [Quat::IDENTITY; 2]);
    for field in 0..11 {
        let mut changed = world.clone();
        let live = changed.joints.get_mut(joint).unwrap();
        let JointKind::Gear(mut model) = live.kind else {
            unreachable!()
        };
        let mut a = live.anchor_a;
        let mut b = live.anchor_b;
        match field {
            0 => a.body = spare,
            1 => b.body = spare,
            2 => a.local_point = Vec3::X,
            3 => b.local_point = Vec3::Y,
            4 => a.local_frame = Quat::from_rotation_x(0.25),
            5 => b.local_frame = Quat::from_rotation_y(0.25),
            6 => model.axis_a = Vec3::X,
            7 => model.axis_b = Vec3::Y,
            8 => model.reference_frame_a = Quat::from_rotation_x(0.25),
            9 => model.reference_frame_b = Quat::from_rotation_y(0.25),
            _ => a.body = stale_a,
        }
        assert!(live
            .gear_state
            .observe(model, a, b, [Quat::IDENTITY; 2])
            .is_some());
        // All numeric history is identical; only the cached binding differs.
        assert_eq!(live.gear_state.angles(), Some([0.0; 2]));
        assert_hidden_difference(&world, &changed, joint);
    }
}

#[test]
fn gear_binding_presence_and_substep_multiplier_are_hashed() {
    let Fixture {
        mut world, joint, ..
    } = gear();
    let unbound = world.clone();
    sample_gear(world.joints.get_mut(joint).unwrap(), [Quat::IDENTITY; 2]);
    assert_hidden_difference(&unbound, &world, joint);
    let mut changed = world.clone();
    changed.joints.get_mut(joint).unwrap().gear_state.lambda = 0.25;
    assert_hidden_difference(&world, &changed, joint);
}

#[test]
fn binding_hash_distinguishes_histories_that_rebind_differently_on_next_sample() {
    for is_gear in [false, true] {
        let Fixture {
            mut world,
            joint,
            stale_a,
            ..
        } = if is_gear { gear() } else { hinge() };
        if is_gear {
            wind_gear(&mut world, joint, 0, 1.0);
        } else {
            seed_hinge(&mut world, joint, core::f64::consts::TAU);
        }
        let mut rebound = world.clone();
        let live = rebound.joints.get_mut(joint).unwrap();
        let b = live.anchor_b.body;
        if is_gear {
            live.gear_state.remap_bodies(stale_a, b);
        } else {
            live.revolute_state.remap_bodies(stale_a, b);
        }
        assert_hidden_difference(&world, &rebound, joint);
        // Retain the mismatched binding itself, not just its numeric angles:
        // restoration must preserve the pending rebind on the next sample.
        let pending_rebind = FullStateSnapshot::capture(7, &rebound);
        let expected_joint = *rebound.joints.get(joint).unwrap();
        let expected_hash = hash_world(&rebound).unwrap();
        let mut restored = world.clone();
        pending_rebind.restore_into(&mut restored);
        let fork = pending_rebind.fork_world();
        for w in [&restored, &fork] {
            assert_eq!(w.joints.get(joint), Some(&expected_joint));
            assert_eq!(hash_world(w).unwrap(), expected_hash);
        }
        rebound = restored;
        for w in [&mut world, &mut rebound] {
            let live = w.joints.get_mut(joint).unwrap();
            if is_gear {
                sample_gear(live, [Quat::IDENTITY; 2]);
            } else {
                assert!(live
                    .revolute_state
                    .observe(Vec3::Z, live.anchor_a, live.anchor_b, [Quat::IDENTITY; 2])
                    .is_some());
            }
        }
        let coordinate = |w: &PhysicsWorld| {
            let live = w.joints.get(joint).unwrap();
            if is_gear {
                live.gear_state().angles().unwrap()[0]
            } else {
                live.revolute_state().angle().unwrap()
            }
        };
        assert!(coordinate(&world) > 6.0);
        assert_eq!(coordinate(&rebound), 0.0);
        assert_hidden_difference(&world, &rebound, joint);
    }
}

fn solver() -> XpbdSolver {
    XpbdSolver::with_config(XpbdConfig {
        position_iterations: 1,
        parallel_islands: false,
        ..XpbdConfig::default()
    })
}

#[test]
fn winding_hash_predicts_different_dynamics_from_identical_body_state() {
    for is_gear in [false, true] {
        let Fixture {
            mut world, joint, ..
        } = if is_gear { gear() } else { hinge() };
        if !is_gear {
            let live = world.joints.get_mut(joint).unwrap();
            live.kind = JointKind::Revolute(
                RevoluteJoint::new(Vec3::Z).with_motor(Motor::position(0.0, 1000.0)),
            );
            seed_hinge(&mut world, joint, 0.0);
        } else {
            sample_gear(world.joints.get_mut(joint).unwrap(), [Quat::IDENTITY; 2]);
        }
        let mut wound = world.clone();
        if is_gear {
            wind_gear(&mut wound, joint, 1, 1.0);
        } else {
            seed_hinge(&mut wound, joint, core::f64::consts::TAU);
        }
        assert_hidden_difference(&world, &wound, joint);
        solver().step(&mut world, 1.0 / 120.0, 1);
        solver().step(&mut wound, 1.0 / 120.0, 1);
        assert_ne!(
            StateSnapshot::capture(&world),
            StateSnapshot::capture(&wound)
        );
    }
}

#[test]
fn clone_and_full_snapshot_preserve_disabled_winding_bindings_and_replay() {
    for is_gear in [false, true] {
        let Fixture {
            mut world, joint, ..
        } = if is_gear { gear() } else { hinge() };
        if is_gear {
            wind_gear(&mut world, joint, 1, 1.0);
        } else {
            seed_hinge(&mut world, joint, 3.0 * core::f64::consts::TAU);
        }
        let live = world.joints.get_mut(joint).unwrap();
        live.enabled = false;
        // Fixed-boundary solvers normally clear these, but a complete clone
        // must also preserve nonzero scratch if captured between substeps.
        if is_gear {
            live.gear_state.lambda = 0.125;
        } else {
            live.revolute_state.motor_lambda = -0.125;
            live.revolute_state.motor_goal = Some(4.0 * core::f64::consts::TAU);
        }
        let expected_joint = *world.joints.get(joint).unwrap();
        let snapshot = FullStateSnapshot::capture(42, &world);
        let expected_hash = hash_world(&world).unwrap();
        let clone = world.clone();
        assert_eq!(clone.joints.get(joint), Some(&expected_joint));
        assert_eq!(hash_world(&clone).unwrap(), expected_hash);
        assert_eq!(snapshot.state_hash().unwrap(), expected_hash);

        let live = world.joints.get_mut(joint).unwrap();
        if is_gear {
            live.reset_gear_state();
        } else {
            live.reset_revolute_state();
        }
        assert_hidden_difference(&clone, &world, joint);
        let mut restored = world;
        snapshot.restore_into(&mut restored);
        let mut fork = snapshot.fork_world();
        for w in [&restored, &fork] {
            assert_eq!(w.joints.get(joint), Some(&expected_joint));
            assert_eq!(hash_world(w).unwrap(), expected_hash);
            assert_eq!(StateSnapshot::capture(w), *snapshot.render());
        }
        restored.joints.get_mut(joint).unwrap().enabled = true;
        fork.joints.get_mut(joint).unwrap().enabled = true;
        let mut original_solver = solver();
        let mut replay_solver = solver();
        for _ in 0..8 {
            original_solver.step(&mut restored, 1.0 / 120.0, 1);
            replay_solver.step(&mut fork, 1.0 / 120.0, 1);
            assert_eq!(hash_world(&restored).unwrap(), hash_world(&fork).unwrap());
            assert_eq!(restored.joints.get(joint), fork.joints.get(joint));
        }
        // Evolving either branch must not mutate the stored snapshot.
        assert_eq!(snapshot.state_hash().unwrap(), expected_hash);
        snapshot.restore_into(&mut restored);
        assert_eq!(restored.joints.get(joint), Some(&expected_joint));
        assert_eq!(hash_world(&restored).unwrap(), expected_hash);
    }
}

#[test]
fn nonfinite_hidden_angular_scratch_is_not_silently_equated() {
    for field in 0..3 {
        let Fixture {
            mut world, joint, ..
        } = if field == 2 { gear() } else { hinge() };
        let live = world.joints.get_mut(joint).unwrap();
        match field {
            0 => live.revolute_state.motor_lambda = f64::NAN,
            1 => live.revolute_state.motor_goal = Some(f64::NAN),
            _ => live.gear_state.lambda = f64::NAN,
        }
        assert_eq!(hash_world(&world), Err(WorldHashError::Nan));
        assert_eq!(
            FullStateSnapshot::capture(0, &world).state_hash(),
            Err(WorldHashError::Nan)
        );
    }
}
