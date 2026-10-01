//! Authoritative `CPU` golden reference for the universal (Cardan) joint stepper.
//!
//! [`cpu_solve_joints_universal`] is a *full stepper* with the identical substep
//! schedule as [`cpu_solve_joints_fixed`](super::cpu_solve_joints_fixed): a
//! caller hands it the current [`RigidBodyState`], the joint set, the shared
//! [`IntegratorConfig`], the joint-specific [`JointSolverConfig`], and the frame
//! `dt`, and must **not** integrate the bodies itself. Within each integrator
//! substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's two `XPBD` Lagrange multipliers,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # The two constraints, projected perpendicular-first
//!
//! A universal joint pins the two bodies at a shared pivot and keeps their two
//! drive axes orthogonal. Each sweep projects the **perpendicularity**
//! constraint first — driving the angle between the two world-space drive axes
//! back to a right angle — then the **positional weld**, the full
//! point-to-point correction with no free direction. Projecting the gimbal
//! orientation before the anchors means the positional pass works against an
//! already-gimbaled frame.
//!
//! The perpendicularity constraint takes the signed angle error
//! `c = acos(clamp(u_a . u_b, -1, 1)) - pi/2` and drives it to zero by rotating
//! about `n = normalize(u_a x u_b)`, with the gradient `-n` on body `a` and
//! `+n` on body `b`. Because `u_b` is `u_a` rotated by `+phi` about `n`,
//! rotating body `b` about `+n` opens the angle and rotating body `a` about
//! `+n` closes it; the signed sign of `c` then pulls the axes apart when they
//! close past perpendicular and together when they open past it, holding the
//! right angle from either side. This reuses, bit for bit, the swing-cone
//! geometry of the [`SwingTwistJoint`](super::SwingTwistJoint) — the only
//! difference is the two-sided target `pi/2` in place of the one-sided cone rim.
//!
//! Each joint owns two Lagrange multipliers in the shared `lambda` buffer: slot
//! `2 * k` for the positional weld and slot `2 * k + 1` for the perpendicularity
//! constraint, where `k` is the joint's index in the colour-reordered list. The
//! buffer is therefore `2 * joints.len()` long and is reset to zero at the start
//! of every substep.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuUniversalJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical two-slot
//! multiplier layout. The `CPU` reference takes its inverse cosine through
//! [`bevy_math::ops::acos`] so the scalar matches the platform math library the
//! rest of the crate's `CPU` goldens use, while the shader uses the built-in
//! `acos`; the tight parity tolerance bounds the resulting reassociation.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the
//! perpendicularity (orthogonality) constraint over two body-fixed axes with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};
use super::universal::UniversalJoint;
use bevy_math::ops;
use glam::Vec3;
use std::f32::consts::FRAC_PI_2;

/// Advances `state` by `dt` under the universal joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's perpendicularity and positional-weld constraints every
/// substep. The joint set is coloured once up front so same-batch joints write
/// disjoint movable bodies; the batches are then solved in order
/// [`position_iterations`](JointSolverConfig::position_iterations) times per
/// substep.
///
/// # Errors
///
/// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree in
/// length or a joint references a body outside the state, and
/// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
/// batches than the colouring supports. Returns `Ok(())` with the state
/// untouched when there is nothing to do (`dt <= 0`, no bodies, or no joints).
pub fn cpu_solve_joints_universal(
    state: &mut RigidBodyState,
    joints: &[UniversalJoint],
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

    let mut prev_positions = vec![Vec3::ZERO; state.len()];
    let mut prev_orientations = vec![glam::Quat::IDENTITY; state.len()];

    // Two Lagrange multipliers per joint: slot `2k` for the positional weld,
    // slot `2k + 1` for the perpendicularity constraint. Reset to zero at the
    // start of every substep.
    let mut lambda = vec![0.0f32; 2 * ordered.len()];

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
                    solve_one(state, &ordered[k], h, &mut lambda[2 * k..2 * k + 2]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one universal joint for a single sweep: the perpendicularity
/// correction first (accumulating `lambda[1]`), then the positional-weld
/// correction (accumulating `lambda[0]`), each applied directly to the two
/// bodies' transforms.
fn solve_one(state: &mut RigidBodyState, joint: &UniversalJoint, h: f32, lambda: &mut [f32]) {
    solve_perpendicular(state, joint, h, &mut lambda[1]);
    solve_weld(state, joint, h, &mut lambda[0]);
}

/// Drives the angle between the two world-space drive axes back to a right
/// angle. With `u_a = rotate(q_a, axis_a)` and `u_b = rotate(q_b, axis_b)` the
/// unit drive axes, the signed violation `c = acos(clamp(u_a . u_b, -1, 1)) -
/// pi/2` is cancelled by rotating about `n = normalize(u_a x u_b)`, with the
/// gradient `-n` on body `a` and `+n` on body `b`. The constraint is two-sided:
/// `c < 0` (axes closer than perpendicular) and `c > 0` (opened past it) each
/// drive the pair back toward orthogonal.
fn solve_perpendicular(
    state: &mut RigidBodyState,
    joint: &UniversalJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let axis_a = rotate(q_a, joint.axis_a);
    let axis_b = rotate(q_b, joint.axis_b);
    let len_a = axis_a.length();
    let len_b = axis_b.length();
    if len_a < EPSILON || len_b < EPSILON {
        return;
    }
    let u_a = axis_a / len_a;
    let u_b = axis_b / len_b;

    // Signed angle error about the right angle: negative when the axes are
    // closer than perpendicular, positive when they have opened past it. `acos`
    // of the dot (rather than the cross-product magnitude) keeps the error
    // single-valued across the full `[0, pi]` range of the inter-axis angle.
    let phi = ops::acos((u_a.dot(u_b)).clamp(-1.0, 1.0));
    let c = phi - FRAC_PI_2;

    // Rotation axis that changes the angle between the two drive axes.
    let delta = u_a.cross(u_b);
    let delta_len = delta.length();
    if delta_len < EPSILON {
        return;
    }
    let n = delta / delta_len;

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = n.dot(world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = n.dot(world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    let alpha_tilde = joint.angular_compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    // `u_b` is `u_a` rotated by `+phi` about `n = normalize(u_a x u_b)`, so
    // rotating body `b` about `+n` *opens* the angle and rotating body `a`
    // about `+n` *closes* it. The constraint gradients are therefore `-n` on
    // body `a` and `+n` on body `b` — the same antisymmetric pattern the swing
    // cone uses.
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Drives the two world-space anchors together, cancelling the full separation
/// vector and locking all three relative translational degrees of freedom. This
/// is the point-to-point `XPBD` positional correction with no free direction,
/// identical to the fixed joint's weld.
fn solve_weld(state: &mut RigidBodyState, joint: &UniversalJoint, h: f32, lambda: &mut f32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    /// Separation between a joint's two world-space anchors.
    fn anchor_separation(state: &RigidBodyState, joint: &UniversalJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    /// Angle (radians) between the two world-space drive axes — `pi/2` when the
    /// perpendicularity constraint is satisfied.
    fn axis_angle(state: &RigidBodyState, joint: &UniversalJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let u_a = rotate(state.orientations[a], joint.axis_a).normalize();
        let u_b = rotate(state.orientations[b], joint.axis_b).normalize();
        ops::acos((u_a.dot(u_b)).clamp(-1.0, 1.0))
    }

    #[test]
    fn perpendicular_pulls_a_closed_gimbal_back_to_a_right_angle() {
        // Body 1's drive axis starts only 45 degrees off body 0's (closer than
        // perpendicular): the two-sided constraint must open it back toward the
        // right angle within a frame of a few sweeps.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // static mount
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        // Body 1 is rotated so its +Z axis tips toward body 0's +X axis, closing
        // the right angle between +X (body 0) and +Z (body 1) to 45 degrees.
        state.orientations[1] = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
        let joint = UniversalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::Z, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        let before = axis_angle(&state, &joint);
        assert!(
            (before - std::f32::consts::FRAC_PI_4).abs() < 0.02,
            "setup axis angle wrong: {before}"
        );
        cpu_solve_joints_universal(&mut state, &[joint], &integrator, &config, 1.0 / 60.0).unwrap();
        let after = axis_angle(&state, &joint);
        assert!(
            (after - FRAC_PI_2).abs() < 0.1,
            "perpendicular constraint left {after} rad (want pi/2)"
        );
    }

    #[test]
    fn perpendicular_pulls_an_opened_gimbal_back_to_a_right_angle() {
        // The mirror case: body 1's drive axis starts 135 degrees off body 0's
        // (opened past perpendicular); the constraint must close it back toward
        // the right angle.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.orientations[1] = Quat::from_axis_angle(Vec3::Y, -std::f32::consts::FRAC_PI_4);
        let joint = UniversalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::Z, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        let before = axis_angle(&state, &joint);
        assert!(
            (before - 3.0 * std::f32::consts::FRAC_PI_4).abs() < 0.02,
            "setup axis angle wrong: {before}"
        );
        cpu_solve_joints_universal(&mut state, &[joint], &integrator, &config, 1.0 / 60.0).unwrap();
        let after = axis_angle(&state, &joint);
        assert!(
            (after - FRAC_PI_2).abs() < 0.1,
            "perpendicular constraint left {after} rad (want pi/2)"
        );
    }

    #[test]
    fn gimbaled_shaft_stays_pinned_and_orthogonal_under_gravity() {
        // A dynamic shaft gimbaled off a static mount must keep its anchor pinned
        // and its drive axis orthogonal to the mount's even as gravity drags it,
        // while remaining free to swing/spin about the gimbal.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // mount
        state.push(
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        // Anchors coincide at the origin; mount drive axis +X, shaft drive axis
        // +Z start orthogonal.
        let joint = UniversalJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::X,
            Vec3::Z,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_universal(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "gimbaled shaft drifted off its pivot"
            );
            assert!(
                (axis_angle(&state, &joint) - FRAC_PI_2).abs() < 2.0e-2,
                "gimbal drive axes drifted off orthogonal"
            );
        }
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, both bodies given the same initial
        // velocity: the joint's internal corrections are equal and opposite, so
        // total linear momentum must be conserved.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.5, 0.2, -0.1);
        state.linear_velocities[1] = Vec3::new(0.5, 0.2, -0.1);
        let joint = UniversalJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::X,
            Vec3::Z,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_universal(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 1.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }

    #[test]
    fn empty_joint_set_is_a_no_op() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let before = state.positions[0];
        cpu_solve_joints_universal(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], before);
    }
}
