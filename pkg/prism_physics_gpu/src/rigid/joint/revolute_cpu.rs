//! Authoritative `CPU` golden reference for the revolute (hinge) joint stepper.
//!
//! [`cpu_solve_joints_revolute`] is a *full stepper* with the identical substep
//! schedule as [`cpu_solve_joints_spherical`](super::cpu_solve_joints_spherical):
//! a caller hands it the current [`RigidBodyState`], the joint set, the shared
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
//! # The two constraints, projected angle-first
//!
//! A revolute joint is the spherical joint's point-to-point positional weld plus
//! an axis-alignment angular restriction. Each sweep projects the **angular**
//! constraint first — realigning the two world-space hinge axes by cancelling
//! their cross product — then the **positional** constraint, which is numerically
//! identical to the spherical golden's `solve_one`. Projecting the axis before
//! the anchors means the positional pass works against an already-aligned hinge.
//!
//! Each joint owns two Lagrange multipliers in the shared `lambda` buffer: slot
//! `2 * k` for the positional weld and slot `2 * k + 1` for the axis alignment,
//! where `k` is the joint's index in the colour-reordered list. The buffer is
//! therefore `2 * joints.len()` long and is reset to zero at the start of every
//! substep.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuRevoluteJointSolver`) and its shader, which walk the identical reordered
//! joint list and colour-batch order with the identical two-slot multiplier
//! layout. The shared quaternion and inverse-inertia helpers keep the three
//! engine stages composing without drift.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the hinge
//! axis-alignment constraint with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::revolute::RevoluteJoint;
use super::stepper::{predict, recover_velocities, snapshot};

/// Advances `state` by `dt` under the revolute joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment and point-to-point constraints every
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
pub fn cpu_solve_joints_revolute(
    state: &mut RigidBodyState,
    joints: &[RevoluteJoint],
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
    // Two multipliers per joint: `[2k]` positional weld, `[2k + 1]` axis
    // alignment. Reset to zero at the start of every substep.
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

/// Projects one revolute joint for a single sweep: the axis-alignment angular
/// correction first (accumulating `lambda[1]`), then the point-to-point
/// positional correction (accumulating `lambda[0]`), each applied directly to
/// the two bodies' transforms.
fn solve_one(state: &mut RigidBodyState, joint: &RevoluteJoint, h: f32, lambda: &mut [f32]) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_point_to_point(state, joint, h, &mut lambda[0]);
}

/// Drives the two world-space hinge axes parallel by cancelling their cross
/// product, locking the two rotational degrees of freedom perpendicular to the
/// hinge while leaving the spin about it free.
fn solve_axis_alignment(
    state: &mut RigidBodyState,
    joint: &RevoluteJoint,
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

    // The alignment error is the cross product of the two unit hinge axes; its
    // magnitude is `sin(angle)` between them and its direction is the rotation
    // axis that drives them parallel.
    let delta = u_a.cross(u_b);
    let theta = delta.length();
    if theta < EPSILON {
        return;
    }
    let n = delta / theta;

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = n.dot(world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = n.dot(world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    let alpha_tilde = joint.angular_compliance / (h * h);
    let d_lambda = (-theta - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    state.orientations[a] = apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, -world_inv_inertia_apply(q_b, ii_b, p));
}

/// Drives the two world-space anchors together with the identical point-to-point
/// `XPBD` positional correction as the spherical golden.
fn solve_point_to_point(
    state: &mut RigidBodyState,
    joint: &RevoluteJoint,
    h: f32,
    lambda: &mut f32,
) {
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

    /// Separation between a joint's two world-space anchors in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &RevoluteJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    /// Sine of the angle between a joint's two world-space hinge axes in
    /// `state`: the magnitude of their unit cross product, which is `0` when
    /// perfectly aligned and `1` at `90` degrees. Avoids an inverse
    /// trigonometric call while staying a monotone misalignment measure over
    /// the `[0, 90]`-degree range these tests exercise.
    fn axis_misalignment_sin(state: &RigidBodyState, joint: &RevoluteJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let u_a = rotate(state.orientations[a], joint.axis_a).normalize();
        let u_b = rotate(state.orientations[b], joint.axis_b).normalize();
        u_a.cross(u_b).length().clamp(0.0, 1.0)
    }

    #[test]
    fn hinge_holds_axis_aligned_while_swinging() {
        // Body 0 is a static pivot at the origin; body 1 is a unit mass pinned
        // back to the origin and hinged about the shared world Z axis. As it
        // swings under gravity the anchors must stay coincident and the two
        // hinge axes parallel.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = RevoluteJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_revolute(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
            assert!(
                axis_misalignment_sin(&state, &joint) < 1.0e-2,
                "hinge axis drifted out of alignment"
            );
        }
        assert!(state.positions[1].y < -0.05, "hinge never swung down");
    }

    #[test]
    fn hinge_permits_free_spin_about_its_axis() {
        // A dynamic body hinged to a static pivot about the world Y axis, given
        // an initial spin about that same axis, must keep spinning freely: the
        // axis-alignment constraint locks only the two perpendicular rotational
        // degrees of freedom, so the Y spin survives.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.angular_velocities[1] = Vec3::new(0.0, 2.0, 0.0);
        let joint = RevoluteJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..60 {
            cpu_solve_joints_revolute(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        // The free spin about the hinge axis is preserved (little lost to the
        // constraint), while the axis stays put.
        assert!(
            state.angular_velocities[1].y > 1.5,
            "free hinge spin decayed: {:?}",
            state.angular_velocities[1]
        );
        assert!(
            axis_misalignment_sin(&state, &joint) < 1.0e-2,
            "hinge axis drifted under free spin"
        );
    }

    #[test]
    fn misaligned_axis_is_pulled_parallel() {
        // Two dynamic bodies whose hinge axes start 90 degrees apart: the rigid
        // (zero angular compliance) alignment must close most of that gap within
        // a single frame of a few sweeps.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        // Body 0 hinge axis is local X; body 1 hinge axis is local X but the body
        // is rotated 90 degrees about Z, so its world axis is Y.
        state.orientations[1] = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2);
        let joint = RevoluteJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::X, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        // Axes start 90 degrees apart, so their unit cross product has unit
        // length (sin 90 = 1).
        let before = axis_misalignment_sin(&state, &joint);
        assert!((before - 1.0).abs() < 1.0e-5);
        cpu_solve_joints_revolute(&mut state, &[joint], &integrator, &config, 1.0 / 60.0).unwrap();
        let after = axis_misalignment_sin(&state, &joint);
        assert!(
            after < 0.2,
            "axis alignment left sin(theta) = {after} of error"
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
        let joint = RevoluteJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_revolute(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 1.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }
}
