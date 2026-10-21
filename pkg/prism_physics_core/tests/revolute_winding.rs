//! Continuous hinge coordinates are world state, not a wrapped pose query.
//! No dependency on reduced-coordinate handoff or its coordinate conventions.

use core::f64::consts::{PI, TAU};
use glam::{Quat, Vec3};
use prism_physics_core::snapshot::hash::{hash_state, hash_world};
use prism_physics_core::snapshot::world::FullStateSnapshot;
use prism_physics_core::snapshot::StateSnapshot;
use prism_physics_core::solver::xpbd::{joint_constraint, XpbdConfig, XpbdSolver};
use prism_physics_core::solver::Solver;
use prism_physics_core::{
    AngleLimit, BodyDesc, BodyHandle, JointAnchor, JointDesc, JointHandle, JointKind,
    MassProperties, Motor, PhysicsWorld, RevoluteJoint, WorldConfig,
};

fn world() -> PhysicsWorld {
    let mut config = WorldConfig::with_gravity(Vec3::ZERO);
    config.sleep.enabled = false;
    PhysicsWorld::new(config)
}

fn pair(
    model: RevoluteJoint,
    static_a: bool,
    static_b: bool,
) -> (PhysicsWorld, BodyHandle, BodyHandle, JointHandle) {
    let mut world = world();
    let body = |is_static| {
        if is_static {
            BodyDesc::static_at(Vec3::ZERO)
        } else {
            BodyDesc::dynamic_at(Vec3::ZERO).with_damping(0.0, 0.0)
        }
    };
    let a = world.spawn(body(static_a));
    let b = world.spawn(body(static_b));
    let joint = world.spawn_joint(JointDesc::revolute(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        model,
    ));
    (world, a, b, joint)
}

fn project(world: &mut PhysicsWorld, iterations: usize, h: f32) {
    let PhysicsWorld { bodies, joints, .. } = world;
    let mut view = bodies.solver_view_mut();
    joint_constraint::begin_revolute_substep(&view, joints, h);
    for _ in 0..iterations {
        joint_constraint::solve_joints(&mut view, joints, h);
    }
    joint_constraint::finish_revolute_substep(&view, joints);
}

fn angle(world: &PhysicsWorld, joint: JointHandle) -> f64 {
    world
        .joints
        .get(joint)
        .unwrap()
        .revolute_state()
        .angle()
        .unwrap()
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected} (tolerance {tolerance})"
    );
}

fn rotate(world: &mut PhysicsWorld, body: BodyHandle, increment: f32) {
    let q = world.bodies.orientation(body).unwrap();
    assert!(world
        .bodies
        .set_orientation(body, (Quat::from_rotation_z(increment) * q).normalize()));
}

fn seed(world: &mut PhysicsWorld, a: BodyHandle, b: BodyHandle, joint: JointHandle, q: f64) {
    let poses = [
        world.bodies.orientation(a).unwrap(),
        world.bodies.orientation(b).unwrap(),
    ];
    assert!(world
        .joints
        .get_mut(joint)
        .unwrap()
        .set_revolute_coordinate(q, poses));
}

fn solver(iterations: u32) -> XpbdSolver {
    XpbdSolver::with_config(XpbdConfig {
        position_iterations: iterations,
        parallel_islands: false,
        ..XpbdConfig::default()
    })
}

#[test]
fn both_pi_seams_many_turns_and_quaternion_signs_preserve_coordinate() {
    for direction in [-1.0_f32, 1.0] {
        let (mut world, a, b, joint) = pair(RevoluteJoint::new(Vec3::Z), true, false);
        project(&mut world, 0, 0.01);
        for tick in 0..900 {
            rotate(&mut world, b, direction * 0.07);
            if tick % 17 == 0 {
                for body in [a, b] {
                    let q = world.bodies.orientation(body).unwrap();
                    world.bodies.set_orientation(body, -q);
                }
            }
            project(&mut world, 1, 0.01);
            near(
                angle(&world, joint),
                f64::from(direction) * 0.07 * f64::from(tick + 1),
                2.0e-4,
            );
            let principal = world
                .joints
                .get(joint)
                .unwrap()
                .revolute_state()
                .principal_angle()
                .unwrap();
            assert!((-PI..PI).contains(&principal));
        }
        assert!(angle(&world, joint).abs() > 9.0 * TAU);
    }
}

#[test]
fn explicit_seed_preserves_offset_and_sub_f32_many_turn_precision() {
    let (mut world, a, b, joint) = pair(RevoluteJoint::new(Vec3::Z), true, false);
    // Pose and coordinate zero may differ; handoff defines the coordinate at
    // this exact pose. Never cast the accumulated coordinate to f32 first.
    let initial = 100_000_000.0 * TAU + 0.123;
    seed(&mut world, a, b, joint, initial);
    for _ in 0..100 {
        rotate(&mut world, b, 0.01);
        project(&mut world, 0, 0.01);
    }
    near(angle(&world, joint) - initial, 1.0, 2.0e-5);
}

#[test]
fn authored_ranges_outside_principal_interval_are_not_wrapped_or_rejected() {
    for initial in [4.0, -4.0, 6.0 * TAU + 0.3, -6.0 * TAU - 0.3] {
        let model = RevoluteJoint::new(Vec3::Z).with_limit(AngleLimit::new(
            (initial - 0.4) as f32,
            (initial + 0.4) as f32,
        ));
        let (mut world, a, b, joint) = pair(model, true, false);
        world
            .bodies
            .set_orientation(b, Quat::from_rotation_z(initial as f32));
        seed(&mut world, a, b, joint, initial);
        let before = world.bodies.orientation(b);
        project(&mut world, 12, 0.01);
        assert_eq!(
            world.bodies.orientation(b),
            before,
            "legal multi-turn pose moved"
        );
        near(angle(&world, joint), initial, 1.0e-10);
        let stop = (initial - 0.2) as f32;
        world.joints.get_mut(joint).unwrap().kind =
            JointKind::Revolute(model.with_limit(AngleLimit::new(stop, stop)));
        project(&mut world, 12, 0.01);
        near(angle(&world, joint), f64::from(stop), 2.0e-6);
    }
}

#[test]
fn position_motor_takes_unwrapped_direction_instead_of_shortest_pose_arc() {
    for (static_a, static_b) in [(true, false), (false, true), (false, false)] {
        for target in [(3.0 * TAU + 0.4) as f32, (-3.0 * TAU - 0.4) as f32] {
            let model =
                RevoluteJoint::new(Vec3::Z).with_motor(Motor::position(target, f32::INFINITY));
            let (mut world, a, b, joint) = pair(model, static_a, static_b);
            project(&mut world, 0, 0.01);
            let mut previous = 0.0;
            for _ in 0..80 {
                project(&mut world, 1, 0.01);
                let next = angle(&world, joint);
                assert!((next - previous) * f64::from(target.signum()) >= -2.0e-6);
                assert!(
                    (next - previous).abs() < 1.01,
                    "projection skipped observations"
                );
                previous = next;
            }
            near(angle(&world, joint), f64::from(target), 3.0e-6);
            if static_a {
                assert_eq!(world.bodies.orientation(a), Some(Quat::IDENTITY));
            }
            if static_b {
                assert_eq!(world.bodies.orientation(b), Some(Quat::IDENTITY));
            }
        }
    }
}

#[test]
fn motor_remeasures_after_limit_and_respects_its_unwrapped_stop() {
    for target in [0.2_f32, 2.0] {
        let model = RevoluteJoint::new(Vec3::Z)
            .with_limit(AngleLimit::new(-0.4, 0.4))
            .with_motor(Motor::position(target, f32::INFINITY));
        let (mut world, _, b, joint) = pair(model, true, false);
        world.bodies.set_orientation(b, Quat::from_rotation_z(0.8));
        project(&mut world, 1, 0.1);
        // Old code used the pre-limit coordinate for the motor and overshot.
        near(angle(&world, joint), f64::from(target.min(0.4)), 3.0e-7);
    }
    let model = RevoluteJoint::new(Vec3::Z)
        .with_limit(AngleLimit::new(3.0, 4.0))
        .with_motor(Motor::position(20.0, f32::INFINITY));
    let (mut world, a, b, joint) = pair(model, true, false);
    seed(&mut world, a, b, joint, 3.1);
    project(&mut world, 16, 0.1);
    near(angle(&world, joint), 4.0, 2.0e-6);
}

#[test]
fn motor_torque_budget_and_compliance_are_per_substep_not_per_iteration() {
    for motor in [
        Motor::position(0.3, 2.0),
        Motor::position(0.3, f32::INFINITY).with_compliance(0.02),
    ] {
        let (mut once, _, _, joint) =
            pair(RevoluteJoint::new(Vec3::Z).with_motor(motor), true, false);
        let mut many = once.clone();
        project(&mut once, 1, 0.1);
        project(&mut many, 32, 0.1);
        let h = f64::from(0.1_f32);
        let expected = if motor.max_force.is_finite() {
            2.0 * h * h
        } else {
            f64::from(0.3_f32) / (1.0 + f64::from(motor.compliance) / (h * h))
        };
        near(angle(&once, joint), expected, 2.0e-7);
        near(angle(&many, joint), expected, 2.0e-7);
        if motor.max_force.is_finite() {
            project(&mut many, 32, 0.1);
            near(angle(&many, joint), 2.0 * expected, 3.0e-7);
        }
    }
}

#[test]
fn velocity_motor_goal_is_frozen_before_prediction_and_across_iterations() {
    let model = RevoluteJoint::new(Vec3::Z).with_motor(Motor::velocity(3.0, f32::INFINITY));
    let (mut once, a, b, joint) = pair(model, true, false);
    seed(&mut once, a, b, joint, 5.0 * TAU);
    let mut many = once.clone();
    project(&mut once, 1, 0.1);
    project(&mut many, 32, 0.1);
    near(angle(&once, joint), 5.0 * TAU + 0.3, 3.0e-7);
    near(angle(&many, joint), angle(&once, joint), 3.0e-7);
    let (mut moving, _, b, joint) = pair(model, true, false);
    moving.bodies.set_angular_velocity(b, Vec3::Z * 3.0);
    let mut solver = solver(12);
    for tick in 1..=300 {
        solver.step(&mut moving, 0.01, 1);
        // A goal measured after predict would double the desired displacement.
        near(angle(&moving, joint), f64::from(tick) * 0.03, 2.0e-4);
    }
}

#[test]
fn disabled_hinges_track_turns_without_applying_motor_or_limit() {
    let model = RevoluteJoint::new(Vec3::Z)
        .with_limit(AngleLimit::symmetric(0.2))
        .with_motor(Motor::position(-1.0, f32::INFINITY));
    let (mut world, _, b, joint) = pair(model, true, false);
    world.joints.get_mut(joint).unwrap().enabled = false;
    project(&mut world, 0, 0.01);
    for tick in 1..=500 {
        rotate(&mut world, b, 0.06);
        let pose = world.bodies.orientation(b);
        project(&mut world, 8, 0.01);
        assert_eq!(world.bodies.orientation(b), pose);
        near(angle(&world, joint), f64::from(tick) * 0.06, 2.0e-4);
    }
    let target = (angle(&world, joint) + 0.15) as f32;
    let live = world.joints.get_mut(joint).unwrap();
    live.kind = JointKind::Revolute(
        RevoluteJoint::new(Vec3::Z)
            .with_limit(AngleLimit::new(target - 0.3, target + 0.3))
            .with_motor(Motor::position(target, f32::INFINITY)),
    );
    live.enabled = true;
    project(&mut world, 8, 0.01);
    near(angle(&world, joint), f64::from(target), 2.0e-6);
}

#[test]
fn disabled_pipeline_still_observes_predicted_poses() {
    let (mut world, _, b, joint) = pair(RevoluteJoint::new(Vec3::Z), true, false);
    world.joints.get_mut(joint).unwrap().enabled = false;
    world.bodies.set_angular_velocity(b, Vec3::Z * 6.0);
    let mut solver = solver(3);
    for _ in 0..300 {
        solver.step(&mut world, 0.01, 1);
    }
    assert!(angle(&world, joint) > 2.0 * TAU);
    let principal = world
        .joints
        .get(joint)
        .unwrap()
        .revolute_state()
        .principal_angle()
        .unwrap();
    near((angle(&world, joint) - principal) / TAU, 3.0, 2.0e-6);
}

#[test]
fn local_frames_and_body_frames_are_distinct_and_sign_invariant() {
    let (mut world, a, b, joint) = pair(RevoluteJoint::new(Vec3::Z * 4.0), true, false);
    let common = Quat::from_rotation_x(0.7) * Quat::from_rotation_y(-0.3);
    let local_a = Quat::from_rotation_y(0.4);
    let local_b = Quat::from_rotation_x(-0.5);
    {
        let live = world.joints.get_mut(joint).unwrap();
        live.anchor_a.local_frame = local_a;
        live.anchor_b.local_frame = local_b;
    }
    world.bodies.set_orientation(a, common * local_a.inverse());
    world
        .bodies
        .set_orientation(b, common * Quat::from_rotation_z(0.2) * local_b.inverse());
    seed(&mut world, a, b, joint, 4.0 * TAU + 0.2);
    for tick in 1..=300 {
        let q = world.bodies.orientation(b).unwrap();
        world.bodies.set_orientation(
            b,
            (Quat::from_axis_angle(common * Vec3::Z, 0.04) * q).normalize(),
        );
        if tick % 19 == 0 {
            let live = world.joints.get_mut(joint).unwrap();
            live.anchor_a.local_frame = -live.anchor_a.local_frame;
            live.anchor_b.local_frame = -live.anchor_b.local_frame;
        }
        project(&mut world, 0, 0.01);
        near(
            angle(&world, joint),
            4.0 * TAU + 0.2 + f64::from(tick) * 0.04,
            1.0e-4,
        );
    }
}

#[test]
fn retuning_and_anchor_points_preserve_history_but_geometry_rebinds() {
    let (mut world, a, b, joint) = pair(RevoluteJoint::new(Vec3::Z), true, false);
    seed(&mut world, a, b, joint, 3.0 * TAU + 0.1);
    {
        let live = world.joints.get_mut(joint).unwrap();
        live.anchor_a.local_point = Vec3::X;
        live.kind = JointKind::Revolute(
            RevoluteJoint::new(Vec3::Z * 2.0)
                .with_limit(AngleLimit::new(0.0, 100.0))
                .with_compliance(0.02),
        );
    }
    project(&mut world, 0, 0.01);
    near(angle(&world, joint), 3.0 * TAU + 0.1, 1.0e-10);
    world.joints.get_mut(joint).unwrap().anchor_b.local_frame = Quat::from_rotation_z(0.3);
    project(&mut world, 0, 0.01);
    near(angle(&world, joint), 0.3, 2.0e-7);
    seed(&mut world, a, b, joint, 9.0 * TAU);
    world.joints.get_mut(joint).unwrap().kind = JointKind::Revolute(RevoluteJoint::new(-Vec3::Z));
    project(&mut world, 0, 0.01);
    near(angle(&world, joint), -0.3, 2.0e-7);
    seed(&mut world, a, b, joint, 9.0 * TAU);
    world.joints.get_mut(joint).unwrap().reset_revolute_state();
    project(&mut world, 0, 0.01);
    near(angle(&world, joint), -0.3, 2.0e-7);
}

#[test]
fn seed_rejects_invalid_or_undefined_inputs_atomically() {
    let (mut world, a, b, joint) = pair(RevoluteJoint::new(Vec3::Z), true, false);
    seed(&mut world, a, b, joint, 7.0 * TAU + 0.2);
    let live = world.joints.get_mut(joint).unwrap();
    let before = *live.revolute_state();
    for q in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!live.set_revolute_coordinate(q, [Quat::IDENTITY; 2]));
        assert_eq!(*live.revolute_state(), before);
    }
    // Exactly pi swing has no unique twist (w == axis.dot(v) == 0).
    let singular = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);
    for orientations in [
        [Quat::IDENTITY, singular],
        [Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), Quat::IDENTITY],
    ] {
        assert!(!live.set_revolute_coordinate(0.0, orientations));
        assert_eq!(*live.revolute_state(), before);
    }
    live.kind = JointKind::Revolute(RevoluteJoint::new(Vec3::ZERO));
    assert!(!live.set_revolute_coordinate(0.0, [Quat::IDENTITY; 2]));
    assert_eq!(*live.revolute_state(), before);
}

#[test]
fn joint_slot_reuse_does_not_inherit_previous_turns() {
    let (mut world, a, b, old) = pair(RevoluteJoint::new(Vec3::Z), true, false);
    seed(&mut world, a, b, old, 12.0 * TAU);
    assert!(world.joints.remove(old));
    let fresh = world.spawn_joint(JointDesc::revolute(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        RevoluteJoint::new(Vec3::Z),
    ));
    assert_eq!(old.index(), fresh.index());
    assert_ne!(old.generation(), fresh.generation());
    assert_eq!(
        world.joints.get(fresh).unwrap().revolute_state().angle(),
        None
    );
    project(&mut world, 0, 0.01);
    near(angle(&world, fresh), 0.0, 1.0e-10);
}

#[test]
fn projection_is_weighted_by_both_endpoint_inertias() {
    let mut world = world();
    let mut body = |inverse: f32| {
        world.spawn(
            BodyDesc::dynamic_at(Vec3::ZERO).with_mass_properties(MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::splat(inverse),
            }),
        )
    };
    let a = body(0.4);
    let b = body(2.3);
    let joint = world.spawn_joint(JointDesc::revolute(
        JointAnchor::at_center(a),
        JointAnchor::at_center(b),
        RevoluteJoint::new(Vec3::Z).with_motor(Motor::position(0.3, f32::INFINITY)),
    ));
    project(&mut world, 1, 0.1);
    let theta_a = world.bodies.orientation(a).unwrap().to_scaled_axis().z;
    let theta_b = world.bodies.orientation(b).unwrap().to_scaled_axis().z;
    near(f64::from(theta_a), -0.3 * 0.4 / 2.7, 1.0e-7);
    near(f64::from(theta_b), 0.3 * 2.3 / 2.7, 1.0e-7);
    near(angle(&world, joint), 0.3, 2.0e-7);
}

#[cfg(feature = "serialize")]
#[test]
fn serde_joint_roundtrip_keeps_nonprincipal_coordinate_and_continuation() {
    let (mut original, a, b, joint) = pair(
        RevoluteJoint::new(Vec3::Z)
            .with_limit(AngleLimit::new(20.0, 60.0))
            .with_motor(Motor::velocity(4.0, 12.0)),
        true,
        false,
    );
    seed(&mut original, a, b, joint, 6.0 * TAU + 0.25);
    let mut restored = original.clone();
    let serialized_joint = *original.joints.get(joint).unwrap();
    let decoded = serde_tape::roundtrip(&serialized_joint);
    assert_eq!(serialized_joint, decoded);
    *restored.joints.get_mut(joint).unwrap() = decoded;
    let mut first = solver(6);
    let mut second = solver(6);
    for _ in 0..120 {
        first.step(&mut original, 0.01, 1);
        second.step(&mut restored, 0.01, 1);
    }
    assert_eq!(
        hash_world(&original).unwrap(),
        hash_world(&restored).unwrap()
    );
    assert!(angle(&restored, joint) > 6.0 * TAU);
}

// A small, typed in-memory serde format: exercise the actual derives without
// adding serde_json/bincode (or changing Cargo.toml in a shared worktree).
// It supports the structs, tuples, options and newtype enums used by Joint;
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

#[test]
fn clone_and_full_snapshot_resume_winding_with_fresh_solver() {
    let (mut world, _, b, joint) = pair(
        RevoluteJoint::new(Vec3::Z).with_motor(Motor::velocity(5.0, f32::INFINITY)),
        true,
        false,
    );
    world.bodies.set_angular_velocity(b, Vec3::Z * 5.0);
    let mut original_solver = solver(6);
    for _ in 0..400 {
        original_solver.step(&mut world, 0.01, 1);
    }
    assert!(angle(&world, joint) > 3.0 * TAU);
    let snapshot = FullStateSnapshot::capture(400, &world);
    let mut cloned = world.clone();
    let mut restored = snapshot.fork_world();
    let mut clone_solver = solver(6);
    let mut restored_solver = solver(6);
    for _ in 0..100 {
        original_solver.step(&mut world, 0.01, 1);
        clone_solver.step(&mut cloned, 0.01, 1);
        restored_solver.step(&mut restored, 0.01, 1);
    }
    assert_eq!(hash_world(&world).unwrap(), hash_world(&cloned).unwrap());
    assert_eq!(hash_world(&world).unwrap(), hash_world(&restored).unwrap());
    snapshot.restore_into(&mut restored);
    assert_eq!(
        hash_world(&restored).unwrap(),
        snapshot.state_hash().unwrap()
    );
}

#[test]
fn full_hash_distinguishes_winding_even_when_body_state_is_identical() {
    let (mut world, a, b, joint) = pair(RevoluteJoint::new(Vec3::Z), true, false);
    seed(&mut world, a, b, joint, 8.0 * TAU + 0.25);
    let mut branch = world.clone();
    seed(&mut branch, a, b, joint, 0.25);
    assert_eq!(
        hash_state(&StateSnapshot::capture(&world)),
        hash_state(&StateSnapshot::capture(&branch))
    );
    assert_ne!(hash_world(&world).unwrap(), hash_world(&branch).unwrap());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_and_colored_scratch_preserve_seed_motor_budget_and_global_binding() {
    let mut initial = world();
    initial.spawn(BodyDesc::dynamic_at(Vec3::splat(100.0)));
    // Interleaved slots and reused generations prevent accidental identity remaps.
    let hole = initial.spawn(BodyDesc::static_at(Vec3::ZERO));
    assert!(initial.remove_body(hole));
    let a0 = initial.spawn(BodyDesc::static_at(Vec3::ZERO));
    let a1 = initial.spawn(BodyDesc::static_at(Vec3::X));
    let b0 = initial.spawn(BodyDesc::dynamic_at(Vec3::ZERO).with_damping(0.0, 0.0));
    let b1 = initial.spawn(BodyDesc::dynamic_at(Vec3::X).with_damping(0.0, 0.0));
    let mut handles = Vec::new();
    for (a, b, rate) in [(a0, b0, 5.0), (a1, b1, -4.0)] {
        let joint = initial.spawn_joint(JointDesc::revolute(
            JointAnchor::at_center(a),
            JointAnchor::at_center(b),
            RevoluteJoint::new(Vec3::Z).with_motor(Motor::velocity(rate, 20.0)),
        ));
        seed(&mut initial, a, b, joint, 10.0 * TAU);
        handles.push((joint, rate));
    }
    let mut serial = initial.clone();
    let mut threaded = initial.clone();
    let mut colored = initial;
    let mut serial_solver = solver(6);
    let mut threaded_solver = XpbdSolver::with_config(XpbdConfig {
        position_iterations: 6,
        parallel_islands: true,
        ..XpbdConfig::default()
    });
    let mut colored_solver = XpbdSolver::with_config(XpbdConfig {
        position_iterations: 6,
        parallel_islands: true,
        parallel_within_island: true,
        ..XpbdConfig::default()
    });
    for tick in 0..400 {
        serial_solver.step(&mut serial, 0.01, 1);
        threaded_solver.step(&mut threaded, 0.01, 1);
        colored_solver.step(&mut colored, 0.01, 1);
        if tick % 40 == 0 {
            assert_eq!(hash_world(&serial).unwrap(), hash_world(&threaded).unwrap());
            // One joint per island: color-major and natural order coincide.
            assert_eq!(hash_world(&serial).unwrap(), hash_world(&colored).unwrap());
        }
    }
    for (joint, rate) in handles {
        assert_eq!(angle(&serial, joint), angle(&threaded, joint));
        assert_eq!(angle(&serial, joint), angle(&colored, joint));
        assert!((angle(&serial, joint) - 10.0 * TAU) * f64::from(rate.signum()) > TAU);
    }
}
