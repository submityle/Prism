//! Authoritative `CPU` golden reference for the spherical-joint stepper.
//!
//! [`cpu_solve_joints_spherical`] is a *full stepper*: it owns the whole frame
//! advance, so a caller hands it the current [`RigidBodyState`], the joint set,
//! the shared [`IntegratorConfig`], the joint-specific [`JointSolverConfig`],
//! and the frame `dt`, and must **not** integrate the bodies itself. Within each
//! integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping (semi-implicit
//!    Euler for translation, the quaternion kinematic equation for rotation),
//! 3. resets each joint's `XPBD` Lagrange multiplier,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order so same-batch joints touch
//!    disjoint movable bodies, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! This is the substep-`XPBD` scheme of Müller et al., "Detailed Rigid Body
//! Simulation with XPBD": convergence comes from many small substeps rather than
//! many solver iterations, and each positional constraint is projected directly
//! against the two bodies' world-space inverse inertia.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored bit-for-bit by the `GPU` twin
//! (`GpuSphericalJointSolver`) and its shader. The quaternion helpers use the
//! same expanded sandwich product and Hamilton product as the rigid integrator
//! and contact solver so the three stages compose without drift. The `GPU`
//! kernel walks the identical reordered joint list and colour-batch order, which
//! is what lets the device reproduce this trajectory frame for frame.
//!
//! Provenance: the point-to-point (ball-socket) constraint and its substep
//! `XPBD` positional handling (Müller et al.), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use glam::{Quat, Vec3};

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::spherical::SphericalJoint;

/// Length below which a separation vector or an effective mass is treated as
/// degenerate and the correction is skipped. Matches the `EPSILON` the rigid
/// integrator and contact solver use, so the three stages share one notion of
/// "numerically zero".
pub(crate) const EPSILON: f32 = 1.192_092_9e-7;

/// Advances `state` by `dt` under the spherical joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects the joint constraints each substep. The joint set is coloured once
/// up front so same-batch joints write disjoint movable bodies; the batches are
/// then solved in order [`position_iterations`](JointSolverConfig::position_iterations)
/// times per substep.
///
/// # Errors
///
/// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree in
/// length or a joint references a body outside the state, and
/// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
/// batches than the colouring supports. Returns `Ok(())` with the state
/// untouched when there is nothing to do (`dt <= 0`, no bodies, or no joints).
pub fn cpu_solve_joints_spherical(
    state: &mut RigidBodyState,
    joints: &[SphericalJoint],
    integrator: &IntegratorConfig,
    joint_config: &JointSolverConfig,
    dt: f32,
) -> Result<(), RigidError> {
    integrator.validate()?;
    joint_config.validate()?;
    if !state.is_consistent() {
        return Err(RigidError::InconsistentState {
            reason: "per-body arrays must have equal length",
        });
    }
    if state.is_empty() || joints.is_empty() || dt <= 0.0 {
        return Ok(());
    }

    let movable = movable_mask(state);
    let colouring = JointColouring::build(joints, &movable)?;
    let ordered = colouring.reorder(joints);
    let ranges = colouring.ranges();

    let substeps = integrator.effective_substeps();
    let h = dt / substeps as f32;
    if h <= 0.0 {
        return Ok(());
    }
    let inv_h = 1.0 / h;
    let linear_damping_scale = (1.0 - integrator.linear_damping * h).max(0.0);
    let angular_damping_scale = (1.0 - integrator.angular_damping * h).max(0.0);
    let iterations = joint_config.effective_position_iterations();

    let mut prev_positions = state.positions.clone();
    let mut prev_orientations = state.orientations.clone();
    let mut lambda = vec![0.0f32; ordered.len()];

    for _ in 0..substeps {
        snapshot(state, &mut prev_positions, &mut prev_orientations);
        predict(
            state,
            integrator.gravity,
            linear_damping_scale,
            angular_damping_scale,
            h,
        );
        lambda.fill(0.0);
        for _ in 0..iterations {
            for &(start, end) in ranges {
                for k in start as usize..end as usize {
                    solve_one(state, &ordered[k], h, &mut lambda[k]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Copies the current positions and orientations into the per-substep snapshot
/// buffers the velocity recovery differences against.
fn snapshot(state: &RigidBodyState, prev_positions: &mut [Vec3], prev_orientations: &mut [Quat]) {
    prev_positions.copy_from_slice(&state.positions);
    prev_orientations.copy_from_slice(&state.orientations);
}

/// Predicts every body forward one substep: semi-implicit Euler with linear
/// damping for translation, and the quaternion kinematic equation with angular
/// damping for rotation. Static (zero inverse mass) bodies keep their position;
/// non-rotating (zero inverse inertia) bodies keep their orientation.
fn predict(
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
fn recover_velocities(
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

/// Projects one spherical joint for a single sweep, applying the `XPBD`
/// positional correction that drives the two world-space anchors together and
/// accumulating the joint's Lagrange multiplier in `lambda`.
fn solve_one(state: &mut RigidBodyState, joint: &SphericalJoint, h: f32, lambda: &mut f32) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];
    let r_a = rotate(q_a, joint.anchor_a);
    let r_b = rotate(q_b, joint.anchor_b);
    let dx = (state.positions[a] + r_a) - (state.positions[b] + r_b);
    let c = dx.length();
    if c < EPSILON {
        return;
    }
    let n = dx / c;

    let inv_m_a = state.inverse_masses[a];
    let inv_m_b = state.inverse_masses[b];
    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];

    let rn_a = r_a.cross(n);
    let rn_b = r_b.cross(n);
    let w_a = inv_m_a + rn_a.dot(world_inv_inertia_apply(q_a, ii_a, rn_a));
    let w_b = inv_m_b + rn_b.dot(world_inv_inertia_apply(q_b, ii_b, rn_b));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    let alpha_tilde = joint.compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    state.positions[a] += p * inv_m_a;
    state.positions[b] -= p * inv_m_b;
    let dw_a = world_inv_inertia_apply(q_a, ii_a, r_a.cross(p));
    state.orientations[a] = apply_rotation_delta(q_a, dw_a);
    let dw_b = world_inv_inertia_apply(q_b, ii_b, r_b.cross(p));
    state.orientations[b] = apply_rotation_delta(q_b, -dw_b);
}

/// Whether the solver may rotate a body with the given body-frame inverse
/// inertia (any non-zero principal axis).
fn can_rotate(inverse_inertia: Vec3) -> bool {
    inverse_inertia.x > 0.0 || inverse_inertia.y > 0.0 || inverse_inertia.z > 0.0
}

/// Advances an orientation by its world-space angular velocity over `h` via the
/// quaternion kinematic equation `q' = normalize(q + 0.5 h [omega, 0] q)`,
/// matching the rigid integrator. Falls back to the input orientation if the
/// integrated quaternion is degenerate.
fn integrate_orientation(orientation: Quat, omega: Vec3, h: f32) -> Quat {
    let q = quat_array(orientation);
    let dq = quat_mul([omega.x, omega.y, omega.z, 0.0], q);
    let half_h = 0.5 * h;
    let integrated = [
        q[0] + dq[0] * half_h,
        q[1] + dq[1] * half_h,
        q[2] + dq[2] * half_h,
        q[3] + dq[3] * half_h,
    ];
    normalize_or(integrated, orientation)
}

/// Applies a direct `XPBD` rotation delta `omega` (a rotation vector, not scaled
/// by any time step) to an orientation: `q' = normalize(q + 0.5 [omega, 0] q)`
/// (Müller et al.). Falls back to the input orientation if the result is
/// degenerate.
fn apply_rotation_delta(orientation: Quat, omega: Vec3) -> Quat {
    let q = quat_array(orientation);
    let dq = quat_mul([omega.x, omega.y, omega.z, 0.0], q);
    let integrated = [
        q[0] + 0.5 * dq[0],
        q[1] + 0.5 * dq[1],
        q[2] + 0.5 * dq[2],
        q[3] + 0.5 * dq[3],
    ];
    normalize_or(integrated, orientation)
}

/// Applies the world-space inverse inertia `R diag(inv_inertia) R^T` to `v`:
/// rotate `v` into the body frame, scale by the diagonal inverse inertia, and
/// rotate back. Mirrors the contact solver's identically named helper.
fn world_inv_inertia_apply(orientation: Quat, inv_inertia: Vec3, v: Vec3) -> Vec3 {
    let q = quat_array(orientation);
    let body = quat_rotate(quat_conj(q), v);
    quat_rotate(q, inv_inertia * body)
}

/// Rotates `v` by the unit quaternion `orientation`, routing through the
/// array-based [`quat_rotate`] so the `CPU` reference and the shader share one
/// rotation formula.
fn rotate(orientation: Quat, v: Vec3) -> Vec3 {
    quat_rotate(quat_array(orientation), v)
}

/// Normalises a quaternion stored as `(x, y, z, w)`, returning `fallback` when
/// the length is below [`EPSILON`].
fn normalize_or(q: [f32; 4], fallback: Quat) -> Quat {
    let length = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if length > EPSILON {
        Quat::from_xyzw(q[0] / length, q[1] / length, q[2] / length, q[3] / length)
    } else {
        fallback
    }
}

/// Unpacks a [`Quat`] into the `(x, y, z, w)` array the local quaternion
/// helpers operate on.
fn quat_array(q: Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

/// Hamilton product `a * b` of two quaternions stored as `(x, y, z, w)`.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Conjugate (inverse rotation) of a unit quaternion stored as `(x, y, z, w)`.
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotates `v` by the unit quaternion `q` via the expanded sandwich product
/// `2 (u . v) u + (s^2 - u . u) v + 2 s (u x v)` with `u = q.xyz`, `s = q.w`.
fn quat_rotate(q: [f32; 4], v: Vec3) -> Vec3 {
    let u = Vec3::new(q[0], q[1], q[2]);
    let s = q[3];
    u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Separation between a joint's two world-space anchors in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &SphericalJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    #[test]
    fn pin_to_static_keeps_anchor_coincident_under_gravity() {
        // Body 0 is a static pivot at the origin; body 1 is a unit mass at
        // (1, 0, 0) pinned back to the origin through its (-1, 0, 0) anchor. The
        // joint must hold the anchors coincident while the body swings down.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = SphericalJoint::new(0, 1, Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), 0.0);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_spherical(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
        }
        assert!(state.positions[1].y < -0.05, "pendulum never swung down");
    }

    #[test]
    fn satisfied_pair_at_rest_stays_put() {
        // Zero gravity, two dynamic bodies whose anchors already coincide at
        // (0.5, 0, 0): the solver must leave everything untouched.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = SphericalJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_spherical(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        assert!((state.positions[0] - Vec3::ZERO).length() < 1.0e-4);
        assert!((state.positions[1] - Vec3::new(1.0, 0.0, 0.0)).length() < 1.0e-4);
        assert!(state.linear_velocities[0].length() < 1.0e-4);
        assert!(state.linear_velocities[1].length() < 1.0e-4);
    }

    #[test]
    fn rigid_limit_closes_initial_separation() {
        // A rigid (zero-compliance) joint should pull a 0.4 m initial separation
        // essentially shut within a single frame of a few projection sweeps.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(0.4, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = SphericalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);

        assert!((anchor_separation(&state, &joint) - 0.4).abs() < 1.0e-6);
        cpu_solve_joints_spherical(&mut state, &[joint], &integrator, &config, 1.0 / 60.0).unwrap();
        assert!(
            anchor_separation(&state, &joint) < 1.0e-3,
            "rigid joint left a gap"
        );
    }

    #[test]
    fn compliant_joint_leaves_more_slack_than_rigid() {
        // The same frame with a soft (compliant) joint must leave strictly more
        // residual separation than the rigid one.
        let build = || {
            let mut state = RigidBodyState::new();
            state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
            state.push(
                Vec3::new(0.4, 0.0, 0.0),
                Quat::IDENTITY,
                1.0,
                Vec3::splat(1.0),
            );
            state
        };
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        let mut rigid_state = build();
        let rigid = SphericalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 0.0);
        cpu_solve_joints_spherical(&mut rigid_state, &[rigid], &integrator, &config, dt).unwrap();
        let rigid_gap = anchor_separation(&rigid_state, &rigid);

        let mut soft_state = build();
        let soft = SphericalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 0.01);
        cpu_solve_joints_spherical(&mut soft_state, &[soft], &integrator, &config, dt).unwrap();
        let soft_gap = anchor_separation(&soft_state, &soft);

        assert!(
            soft_gap > rigid_gap,
            "compliant joint ({soft_gap}) should keep more slack than rigid ({rigid_gap})"
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, one body given an initial velocity: the
        // joint's internal corrections are equal and opposite, so total linear
        // momentum must be conserved.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.5, 0.2, -0.1);
        let joint = SphericalJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_spherical(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 1.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }
}
