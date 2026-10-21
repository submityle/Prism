//! Public-contract coverage for compliant pose drives and anchor snapping.

use glam::{Quat, Vec3};
use prism_physics_core::{
    dynamics::{assembly::solve_snap, BodyState, GrabDrive, Pose},
    state::body::MassProperties,
    BodyDesc, PhysicsWorld,
};

fn fixture() -> (GrabDrive, BodyState) {
    let mut world = PhysicsWorld::default();
    let handle = world.spawn(BodyDesc::dynamic_at(Vec3::ZERO));
    (
        GrabDrive::soft(handle, Pose::identity()),
        BodyState {
            handle,
            pose: Pose::identity(),
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            mass_properties: MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ONE,
            },
        },
    )
}

fn assert_vec(actual: Vec3, expected: Vec3) {
    assert!(
        actual.abs_diff_eq(expected, 2.0e-5),
        "{actual:?} != {expected:?}"
    );
}

fn unbounded(drive: &mut GrabDrive) {
    drive.max_force = f32::INFINITY;
    drive.max_torque = f32::INFINITY;
    drive.max_linear_speed = f32::INFINITY;
    drive.max_angular_speed = f32::INFINITY;
}

#[test]
fn rejects_mismatched_handle_and_invalid_dt_without_mutating_inputs() {
    let (drive, mut state) = fixture();
    let before = state;
    for dt in [0.0, -0.0, -0.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(drive.sample(&state, dt), None);
        assert_eq!(state, before);
    }
    state.handle = prism_physics_core::BodyHandle::INVALID;
    assert_eq!(drive.sample(&state, 0.1), None);
}

#[test]
fn rejects_malformed_target_and_body_poses_instead_of_normalizing_them() {
    let (drive, state) = fixture();
    for orientation in [
        Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, 0.0, 0.0, 2.0),
        Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
        Quat::from_xyzw(0.0, f32::INFINITY, 0.0, 1.0),
        Quat::from_xyzw(f32::MAX, f32::MAX, f32::MAX, f32::MAX),
    ] {
        let mut invalid_drive = drive;
        invalid_drive.target_pose.orientation = orientation;
        assert_eq!(invalid_drive.sample(&state, 0.1), None);
        let mut invalid_state = state;
        invalid_state.pose.orientation = orientation;
        assert_eq!(drive.sample(&invalid_state, 0.1), None);
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid_drive = drive;
        invalid_drive.target_pose.position.x = value;
        assert_eq!(invalid_drive.sample(&state, 0.1), None);
        let mut invalid_state = state;
        invalid_state.pose.position.y = value;
        assert_eq!(drive.sample(&invalid_state, 0.1), None);
    }
}

#[test]
fn rejects_nonfinite_velocity_columns() {
    let (drive, state) = fixture();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = state;
        invalid.linear_velocity.x = value;
        assert_eq!(drive.sample(&invalid, 0.1), None);
        invalid = state;
        invalid.angular_velocity.z = value;
        assert_eq!(drive.sample(&invalid, 0.1), None);
    }
}

#[test]
fn rejects_nan_negative_caps_and_nonfinite_or_negative_compliance() {
    let (drive, state) = fixture();
    let cap_setters: [fn(&mut GrabDrive, f32); 4] = [
        |d, v| d.max_force = v,
        |d, v| d.max_torque = v,
        |d, v| d.max_linear_speed = v,
        |d, v| d.max_angular_speed = v,
    ];
    for set in cap_setters {
        for value in [f32::NAN, -1.0, -f32::from_bits(1), f32::NEG_INFINITY] {
            let mut invalid = drive;
            set(&mut invalid, value);
            assert_eq!(invalid.sample(&state, 0.1), None);
        }
    }
    let compliance_setters: [fn(&mut GrabDrive, f32); 2] = [
        |d, v| d.linear_compliance = v,
        |d, v| d.angular_compliance = v,
    ];
    for set in compliance_setters {
        for value in [
            f32::NAN,
            -1.0,
            -f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut invalid = drive;
            set(&mut invalid, value);
            assert_eq!(invalid.sample(&state, 0.1), None);
        }
    }
}

#[test]
fn rejects_invalid_inverse_mass_and_inertia_but_accepts_locked_axes() {
    let (drive, state) = fixture();
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid = state;
        invalid.mass_properties.inv_mass = value;
        assert_eq!(drive.sample(&invalid, 0.1), None);
    }
    for value in [
        -1.0,
        -f32::from_bits(1),
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        for axis in 0..3 {
            let mut invalid = state;
            invalid.mass_properties.inv_inertia[axis] = value;
            assert_eq!(drive.sample(&invalid, 0.1), None);
        }
    }
    let mut locked = state;
    locked.mass_properties.inv_inertia = Vec3::ZERO;
    locked.angular_velocity = Vec3::ONE;
    assert_eq!(drive.sample(&locked, 0.1).unwrap().torque, Vec3::ZERO);
}

#[test]
fn preserves_compliance_law_and_supports_explicit_infinite_caps() {
    let (mut drive, mut state) = fixture();
    unbounded(&mut drive);
    drive.target_pose.position = Vec3::new(4.0, -2.0, 1.0);
    drive.target_pose.orientation = Quat::from_rotation_x(0.5);
    drive.linear_compliance = 0.75;
    drive.angular_compliance = 0.25;
    state.linear_velocity = Vec3::new(2.0, 0.0, -2.0);
    state.angular_velocity = Vec3::X * 0.25;
    state.mass_properties.inv_mass = 0.5;
    state.mass_properties.inv_inertia = Vec3::new(0.25, 0.5, 1.0);
    let before = state;
    let wrench = drive.sample(&state, 0.5).unwrap();
    assert_vec(wrench.force, Vec3::new(6.0, -4.0, 4.0));
    assert_vec(wrench.torque, Vec3::X * 3.0);
    assert_eq!(state, before);
}

#[test]
fn zero_speed_caps_brake_while_zero_wrench_caps_disable_output() {
    let (mut drive, mut state) = fixture();
    unbounded(&mut drive);
    drive.linear_compliance = 0.0;
    drive.angular_compliance = 0.0;
    drive.target_pose.position = Vec3::splat(100.0);
    drive.target_pose.orientation = Quat::from_rotation_y(1.0);
    drive.max_linear_speed = 0.0;
    drive.max_angular_speed = 0.0;
    state.linear_velocity = Vec3::X * 2.0;
    state.angular_velocity = Vec3::Z * 3.0;
    let wrench = drive.sample(&state, 0.5).unwrap();
    assert_vec(wrench.force, Vec3::NEG_X * 4.0);
    assert_vec(wrench.torque, Vec3::NEG_Z * 6.0);
    drive.max_force = 0.0;
    drive.max_torque = 0.0;
    let wrench = drive.sample(&state, f32::from_bits(1)).unwrap();
    assert_eq!(wrench.force, Vec3::ZERO);
    assert_eq!(wrench.torque, Vec3::ZERO);
}

#[test]
fn speed_caps_limit_desired_motion_before_mass_and_inertia_conversion() {
    let (mut drive, mut state) = fixture();
    unbounded(&mut drive);
    drive.linear_compliance = 0.0;
    drive.angular_compliance = 0.0;
    drive.target_pose.position = Vec3::new(30.0, 40.0, 0.0);
    drive.target_pose.orientation = Quat::from_rotation_z(1.0);
    drive.max_linear_speed = 2.0;
    drive.max_angular_speed = 0.5;
    state.mass_properties.inv_mass = 0.5;
    state.mass_properties.inv_inertia = Vec3::splat(0.25);
    let wrench = drive.sample(&state, 0.5).unwrap();
    assert_vec(wrench.force, Vec3::new(4.8, 6.4, 0.0));
    assert_vec(wrench.torque, Vec3::Z * 4.0);
}

#[test]
fn quaternion_signs_do_not_change_shortest_arc_torque_including_pi_tie() {
    let (mut drive, mut state) = fixture();
    unbounded(&mut drive);
    drive.angular_compliance = 0.0;
    let current = Quat::from_rotation_y(0.4);
    let target = Quat::from_rotation_z(0.25) * current;
    for body_orientation in [current, -current] {
        for target_orientation in [target, -target] {
            state.pose.orientation = body_orientation;
            drive.target_pose.orientation = target_orientation;
            assert_vec(drive.sample(&state, 0.5).unwrap().torque, Vec3::Z);
        }
    }
    state.pose.orientation = Quat::IDENTITY;
    for target in [
        Quat::from_rotation_z(std::f32::consts::TAU - 0.25),
        -Quat::from_rotation_z(std::f32::consts::TAU - 0.25),
    ] {
        drive.target_pose.orientation = target;
        assert_vec(drive.sample(&state, 0.5).unwrap().torque, Vec3::NEG_Z);
    }
    for current in [Quat::IDENTITY, -Quat::IDENTITY] {
        state.pose.orientation = current;
        for target in [
            Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
            Quat::from_xyzw(-1.0, 0.0, 0.0, 0.0),
        ] {
            drive.target_pose.orientation = target;
            assert_vec(
                drive.sample(&state, 1.0).unwrap().torque,
                Vec3::X * std::f32::consts::PI,
            );
        }
    }
}

#[test]
fn rotated_anisotropic_inertia_and_zero_principal_axis_are_respected() {
    let (mut drive, mut state) = fixture();
    unbounded(&mut drive);
    drive.angular_compliance = 0.0;
    state.pose.orientation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    drive.target_pose = state.pose;
    state.mass_properties.inv_inertia = Vec3::new(0.5, 0.25, 0.0);
    state.angular_velocity = state.pose.orientation * Vec3::new(-1.0, -1.0, -1.0);
    let wrench = drive.sample(&state, 1.0).unwrap();
    assert_vec(
        wrench.torque,
        state.pose.orientation * Vec3::new(2.0, 4.0, 0.0),
    );
}

#[test]
fn extreme_finite_inputs_produce_directional_bounded_wrenches_not_nan_or_zero() {
    let (mut drive, mut state) = fixture();
    drive.target_pose.position = Vec3::splat(f32::MAX);
    drive.target_pose.orientation = Quat::from_rotation_x(0.75);
    drive.linear_compliance = 0.0;
    drive.angular_compliance = 0.0;
    drive.max_linear_speed = f32::INFINITY;
    drive.max_angular_speed = f32::INFINITY;
    state.pose.position = Vec3::splat(-f32::MAX);
    state.mass_properties.inv_mass = f32::from_bits(1);
    state.mass_properties.inv_inertia = Vec3::splat(f32::from_bits(1));
    for cap in [1.0, f32::MIN_POSITIVE, f32::MAX] {
        drive.max_force = cap;
        drive.max_torque = cap;
        for dt in [f32::from_bits(1), 1.0e-30, 0.1, f32::MAX] {
            let wrench = drive.sample(&state, dt).unwrap();
            for value in [wrench.force, wrench.torque] {
                let length = value.as_dvec3().length();
                assert!(value.is_finite());
                assert!(
                    length > 0.0 && length <= f64::from(cap),
                    "{value:?} cap={cap} dt={dt}"
                );
            }
            assert!(wrench.force.min_element() > 0.0);
            assert!(wrench.torque.x > 0.0);
        }
    }
}

#[test]
fn unbounded_unrepresentable_wrench_is_rejected_and_tiny_dt_at_rest_is_valid() {
    let (mut drive, mut state) = fixture();
    unbounded(&mut drive);
    for compliance in [0.0, 1.0e-3, f32::MAX] {
        drive.linear_compliance = compliance;
        drive.angular_compliance = compliance;
        for dt in [f32::from_bits(1), f32::MAX] {
            let wrench = drive.sample(&state, dt).unwrap();
            assert_eq!(wrench.force, Vec3::ZERO);
            assert_eq!(wrench.torque, Vec3::ZERO);
        }
    }
    drive.linear_compliance = 0.0;
    drive.angular_compliance = 0.0;
    assert_eq!(
        drive.sample(&state, f32::from_bits(1)).unwrap().force,
        Vec3::ZERO
    );
    drive.target_pose.position = Vec3::X * f32::MAX;
    assert_eq!(drive.sample(&state, f32::from_bits(1)), None);
    drive.target_pose.position = Vec3::ZERO;
    state.angular_velocity = Vec3::X * f32::MAX;
    assert_eq!(drive.sample(&state, f32::from_bits(1)), None);
}

#[test]
fn snap_rejects_nonfinite_inputs_nonunit_pose_and_zero_normals() {
    let current = Pose::identity();
    for invalid in [
        Vec3::ZERO,
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::INFINITY),
    ] {
        assert!(solve_snap(current, Vec3::ZERO, invalid, Vec3::ZERO, Vec3::Y).is_none());
        assert!(solve_snap(current, Vec3::ZERO, Vec3::Y, Vec3::ZERO, invalid).is_none());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let invalid = Vec3::splat(value);
        assert!(solve_snap(current, invalid, Vec3::Y, Vec3::ZERO, Vec3::Y).is_none());
        assert!(solve_snap(current, Vec3::ZERO, Vec3::Y, invalid, Vec3::Y).is_none());
        let bad_pose = Pose {
            position: invalid,
            ..current
        };
        assert!(solve_snap(bad_pose, Vec3::ZERO, Vec3::Y, Vec3::ZERO, Vec3::Y).is_none());
    }
    for orientation in [
        Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        Quat::from_xyzw(0.0, 0.0, 0.0, 2.0),
        Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
    ] {
        let bad_pose = Pose {
            orientation,
            ..current
        };
        assert!(solve_snap(bad_pose, Vec3::ZERO, Vec3::Y, Vec3::ZERO, Vec3::Y).is_none());
    }
}

#[test]
fn snap_handles_extreme_normal_magnitudes_and_nearly_opposed_normals() {
    let local_position = Vec3::new(1.0, 2.0, -3.0);
    let target_position = Vec3::new(4.0, -5.0, 6.0);
    for (local_normal, target_normal) in [
        (Vec3::X * f32::MAX, Vec3::X * f32::from_bits(1)),
        (Vec3::X * f32::from_bits(1), Vec3::X * f32::MAX),
        (Vec3::X, Vec3::new(1.0, 0.0001, 0.0)),
        (Vec3::X, Vec3::new(-1.0, 0.0001, 0.0)),
    ] {
        let result = solve_snap(
            Pose::identity(),
            local_position,
            local_normal,
            target_position,
            target_normal,
        )
        .unwrap();
        assert!(result.pose.position.is_finite());
        assert!(result.pose.orientation.is_normalized());
        assert!(result.score.is_finite() && (-1.0..=1.0).contains(&result.score));
        let local = local_normal.as_dvec3().normalize().as_vec3();
        let target = target_normal.as_dvec3().normalize().as_vec3();
        assert_vec(result.pose.orientation * local, -target);
        assert_vec(
            result.pose.position + result.pose.orientation * local_position,
            target_position,
        );
    }
}

#[test]
fn snap_accounts_for_current_orientation_and_rejects_unrepresentable_position() {
    let current = Pose {
        position: Vec3::splat(17.0),
        orientation: Quat::from_rotation_y(0.6),
    };
    let local_position = Vec3::new(1.0, 2.0, 3.0);
    let target_position = Vec3::new(4.0, 5.0, 6.0);
    let result = solve_snap(current, local_position, Vec3::X, target_position, Vec3::Y).unwrap();
    assert_vec(result.pose.orientation * Vec3::X, Vec3::NEG_Y);
    assert_vec(
        result.pose.position + result.pose.orientation * local_position,
        target_position,
    );
    assert!(solve_snap(
        Pose::identity(),
        Vec3::X * -f32::MAX,
        Vec3::Y,
        Vec3::X * f32::MAX,
        Vec3::NEG_Y
    )
    .is_none());
}
