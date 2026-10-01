//! Authoritative `CPU` golden reference for the gear (angular ratio coupling)
//! joint stepper.
//!
//! [`cpu_solve_joints_gear`] is a *full stepper* with the identical substep
//! schedule as the other joint goldens: a caller hands it the current
//! [`RigidBodyState`], the joint set, the shared [`IntegratorConfig`], the
//! joint-specific [`JointSolverConfig`], and the frame `dt`, and must **not**
//! integrate the bodies itself. Within each integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's single `XPBD` Lagrange multiplier,
//! 4. projects the joint constraint
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # The single constraint
//!
//! A gear joint couples the spin of two bodies about their respective world
//! axes `u_a = rotate(orientation_a, axis_a)` and
//! `u_b = rotate(orientation_b, axis_b)` at a fixed ratio. It is one scalar
//! angular constraint expressed in the shared position-based stepper on the
//! per-substep relative angular displacement:
//!
//! ```text
//! C = ratio * (u_a . dphi_a) + (u_b . dphi_b)
//! ```
//!
//! where `dphi_a` / `dphi_b` are each body's angular displacement since the
//! substep snapshot (`angular_displacement`, the same small-angle extraction the
//! velocity recovery uses). Forcing `C -> 0` forces the recovered relative spin
//! rates onto `ratio * (u_a . omega_a) + (u_b . omega_b) = 0`.
//!
//! Each joint owns a single Lagrange multiplier in the shared `lambda` buffer at
//! slot `k`, where `k` is the joint's index in the colour-reordered list. The
//! buffer is therefore `joints.len()` long and is reset to zero at the start of
//! every substep.
//!
//! # The gear update
//!
//! The constraint gradients are `ratio * u_a` on body `a` and `u_b` on body `b`
//! — both carrying the plus sign, with the ratio coupling folded into body
//! `a`'s gradient. With
//! `w = (ratio * u_a) . (I_a^-1 (ratio * u_a)) + u_b . (I_b^-1 u_b)` the
//! effective inverse mass and `alpha_tilde = compliance / h^2` the
//! regularisation, each sweep applies
//!
//! ```text
//! d_lambda = (-C - alpha_tilde * lambda) / (w + alpha_tilde)
//! ```
//!
//! and rotates body `a` by `I_a^-1 (ratio * u_a) d_lambda` and body `b` by
//! `I_b^-1 u_b d_lambda`. A zero `compliance` collapses the update to the rigid
//! gear `d_lambda = -C / w`, forcing the rate coupling exactly each substep; a
//! positive compliance applies a finite coupling torque so the ratio is
//! satisfied asymptotically — a flexing belt. Because the two angular impulses
//! differ by the factor `ratio`, angular momentum is deliberately *not*
//! conserved between the two gears: the missing reaction torque flows into each
//! gear's mounting frame, exactly as in a real gear pair.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuGearJointSolver`) and its shader, which walk the identical reordered
//! joint list and colour-batch order with the identical single-slot multiplier
//! layout and read the same per-substep snapshot buffers for the relative
//! displacement term. There is no transcendental in the gear path — it reads
//! only dot products of rotation vectors — so the `CPU` reference and the shader
//! share one arithmetic path with only device reassociation separating them.
//!
//! Provenance: the angular ratio coupling expressed as a per-substep compliant
//! equality on the relative angular displacement about the two gear axes
//! (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
//! Dynamics"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::gear::GearJoint;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::stepper::{predict, recover_velocities, snapshot};
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the gear joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's ratio-coupling constraint every substep. The joint set
/// is coloured once up front so same-batch joints write disjoint movable bodies;
/// the batches are then solved in order
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
pub fn cpu_solve_joints_gear(
    state: &mut RigidBodyState,
    joints: &[GearJoint],
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
    for joint in joints {
        let (a, b) = joint.bodies();
        if a as usize >= state.len() || b as usize >= state.len() {
            return Err(RigidError::InconsistentState {
                reason: "joint references a body outside the state",
            });
        }
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
    let mut prev_orientations = vec![Quat::IDENTITY; state.len()];

    // One multiplier per joint: slot `k` is the ratio coupling. Reset to zero
    // every substep.
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
                    solve_one(state, &prev_orientations, &ordered[k], h, &mut lambda[k]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one gear joint for a single sweep: the ratio coupling about the two
/// world gear axes, applied directly to the two bodies' orientations.
fn solve_one(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &GearJoint,
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

    // Constraint gradients: `ratio * u_a` on body `a`, `u_b` on body `b`. The
    // ratio coupling is folded into body `a`'s gradient; both carry the plus
    // sign, with the signed correction magnitude carried by `d_lambda`.
    let ratio = joint.ratio;
    let grad_a = u_a * ratio;
    let grad_b = u_b;

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = grad_a.dot(world_inv_inertia_apply(q_a, ii_a, grad_a));
    let w_b = grad_b.dot(world_inv_inertia_apply(q_b, ii_b, grad_b));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    // Relative angular displacement about the two gear axes since the substep
    // snapshot, weighted by the ratio: `C = ratio * (u_a . dphi_a) +
    // (u_b . dphi_b)`.
    let dphi_a = angular_displacement(q_a, prev_orientations[a]);
    let dphi_b = angular_displacement(q_b, prev_orientations[b]);
    let c = ratio * u_a.dot(dphi_a) + u_b.dot(dphi_b);

    let alpha_tilde = joint.compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;

    state.orientations[a] =
        apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, grad_a * d_lambda));
    state.orientations[b] =
        apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, grad_b * d_lambda));
}

/// Relative rotation of a body since the substep snapshot, as a rotation vector:
/// twice the imaginary part of the delta quaternion `orientation *
/// conj(prev_orientation)`, hemisphere-corrected so the shortest arc is taken.
/// The gear joint dots this with each axis to read the relative angular rate
/// accumulated over the substep.
fn angular_displacement(orientation: Quat, prev_orientation: Quat) -> Vec3 {
    let delta = quat_mul(
        quat_array(orientation),
        quat_conj(quat_array(prev_orientation)),
    );
    let mut rotvec = Vec3::new(delta[0], delta[1], delta[2]) * 2.0;
    if delta[3] < 0.0 {
        rotvec = -rotvec;
    }
    rotvec
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Angular velocity of a body about a given world axis.
    fn rate_about(state: &RigidBodyState, body: usize, axis: Vec3) -> f32 {
        axis.normalize().dot(state.angular_velocities[body])
    }

    /// Two coaxial gears about the world `+Z` axis, both dynamic, body 0 spun up
    /// by an initial angular velocity. The gear joint ties their spins at the
    /// given ratio. Both start at identity orientation.
    fn gear_pair(ratio: f32, spin_a: f32) -> (RigidBodyState, GearJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 0
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1
        state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin_a);
        let joint = GearJoint::rigid(0, 1, Vec3::Z, Vec3::Z, ratio);
        (state, joint)
    }

    #[test]
    fn rigid_gear_locks_rate_ratio() {
        // A rigid gear forces `ratio * omega_a + omega_b = 0`, so body b spins at
        // `-ratio * omega_a` about Z. Positive ratio => counter-rotation.
        let ratio = 2.0;
        let spin_a = 1.5;
        let (mut state, joint) = gear_pair(ratio, spin_a);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_gear(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let wa = rate_about(&state, 0, Vec3::Z);
        let wb = rate_about(&state, 1, Vec3::Z);
        let residual = ratio * wa + wb;
        assert!(
            residual.abs() < 1.0e-2,
            "gear rate coupling violated: ratio*wa + wb = {residual} (wa = {wa}, wb = {wb})"
        );
        // The coupling must be live: body b should be counter-rotating, not
        // sitting still.
        assert!(wb < -1.0e-2, "driven gear failed to spin: wb = {wb}");
    }

    #[test]
    fn negative_ratio_co_rotates() {
        // A negative ratio makes the gears co-rotate (internal gear / belt):
        // `omega_b = -ratio * omega_a` is positive when ratio is negative.
        let ratio = -1.5;
        let spin_a = 2.0;
        let (mut state, joint) = gear_pair(ratio, spin_a);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_gear(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let wa = rate_about(&state, 0, Vec3::Z);
        let wb = rate_about(&state, 1, Vec3::Z);
        assert!(
            (ratio * wa + wb).abs() < 1.0e-2,
            "gear coupling violated for negative ratio"
        );
        assert!(
            wb > 1.0e-2,
            "co-rotating gear should spin the same way: wb = {wb}"
        );
    }

    #[test]
    fn static_gear_housing_drives_dynamic_gear_to_rest() {
        // Body 0 static (a locked driving shaft), body 1 dynamic with an initial
        // spin: the rigid gear against a static gear forces body 1's spin to the
        // coupled value `-ratio * 0 = 0`, braking it to rest.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: dynamic
        state.angular_velocities[1] = Vec3::new(0.0, 0.0, 3.0);
        let joint = GearJoint::rigid(0, 1, Vec3::Z, Vec3::Z, 2.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..60 {
            cpu_solve_joints_gear(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let wb = rate_about(&state, 1, Vec3::Z);
        assert!(
            wb.abs() < 1.0e-2,
            "static gear housing failed to brake the driven gear: wb = {wb}"
        );
    }

    #[test]
    fn soft_gear_couples_gradually() {
        // A soft gear applies a finite coupling torque, so after a short run the
        // driven gear has begun counter-rotating but has not fully reached the
        // rigid coupled rate.
        let ratio = 2.0;
        let spin_a = 2.0;
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 0
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1
        state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin_a);
        let joint = GearJoint::soft(0, 1, Vec3::Z, Vec3::Z, ratio, 50.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..3 {
            cpu_solve_joints_gear(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let wb = rate_about(&state, 1, Vec3::Z);
        assert!(
            wb < -1.0e-3,
            "soft gear should begin to drive the gear: wb = {wb}"
        );
    }

    #[test]
    fn empty_joint_set_is_a_no_op() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let before = state.positions[0];
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_gear(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], before);
    }

    #[test]
    fn joint_referencing_missing_body_errors() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let joint = GearJoint::rigid(0, 5, Vec3::Z, Vec3::Z, 1.0);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let err = cpu_solve_joints_gear(&mut state, &[joint], &integrator, &config, 1.0 / 60.0);
        assert!(matches!(err, Err(RigidError::InconsistentState { .. })));
    }
}
