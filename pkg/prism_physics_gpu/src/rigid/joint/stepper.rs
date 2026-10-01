//! Shared per-substep body phases for the `XPBD` joint steppers.
//!
//! Every joint `CPU` golden runs the identical substep schedule around its
//! joint-specific projection: snapshot the transforms, predict the bodies
//! forward under gravity and damping, project the constraints, then recover the
//! velocities from the net motion. Only the projection differs between joint
//! types, so the three body phases live here and are shared by `spherical_cpu`,
//! `revolute_cpu`, and every later joint golden. The `GPU` twins run the same
//! phases as the `snapshot` / `predict` / `recover` compute entry points.
//!
//! Provenance: the substep position-based scheme of Müller et al., "Detailed
//! Rigid Body Simulation with XPBD", over the quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use glam::{Quat, Vec3};

use super::super::body::RigidBodyState;
use super::math::{can_rotate, integrate_orientation, quat_array, quat_conj, quat_mul};

/// Copies the current positions and orientations into the per-substep snapshot
/// buffers the velocity recovery differences against.
pub(super) fn snapshot(
    state: &RigidBodyState,
    prev_positions: &mut [Vec3],
    prev_orientations: &mut [Quat],
) {
    prev_positions.copy_from_slice(&state.positions);
    prev_orientations.copy_from_slice(&state.orientations);
}

/// Predicts every body forward one substep: semi-implicit Euler with linear
/// damping for translation, and the quaternion kinematic equation with angular
/// damping for rotation. Static (zero inverse mass) bodies keep their position;
/// non-rotating (zero inverse inertia) bodies keep their orientation.
pub(super) fn predict(
    state: &mut RigidBodyState,
    gravity: Vec3,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    h: f32,
) {
    for i in 0..state.len() {
        if state.inverse_masses[i] > 0.0 {
            let velocity = (state.linear_velocities[i] + gravity * h) * linear_damping_scale;
            state.linear_velocities[i] = velocity;
            state.positions[i] += velocity * h;
        }
        let omega = state.angular_velocities[i] * angular_damping_scale;
        state.angular_velocities[i] = omega;
        if can_rotate(state.inverse_inertias[i]) {
            state.orientations[i] = integrate_orientation(state.orientations[i], omega, h);
        }
    }
}

/// Recovers each body's linear and angular velocity from the net motion over
/// the substep, the defining step of position-based dynamics: the velocity is
/// whatever moved the body from its snapshot to its post-projection transform.
pub(super) fn recover_velocities(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    inv_h: f32,
) {
    for i in 0..state.len() {
        if state.inverse_masses[i] > 0.0 {
            state.linear_velocities[i] = (state.positions[i] - prev_positions[i]) * inv_h;
        }
        if can_rotate(state.inverse_inertias[i]) {
            let delta = quat_mul(
                quat_array(state.orientations[i]),
                quat_conj(quat_array(prev_orientations[i])),
            );
            let mut omega = Vec3::new(delta[0], delta[1], delta[2]) * (2.0 * inv_h);
            if delta[3] < 0.0 {
                omega = -omega;
            }
            state.angular_velocities[i] = omega;
        }
    }
}
