//! Authoritative `CPU` golden reference for the angular `SLERP` drive joint stepper.
//!
//! [`cpu_solve_joints_angular_slerp_drive`] is a *full stepper* with the identical
//! substep schedule as the other joint goldens: a caller hands it the current
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
//! An angular `SLERP` drive servos body `b`'s orientation, relative to body `a`,
//! toward a commanded target relative rotation along the geodesic. It is a
//! purely rotational drive: the two bodies' positions are untouched. Each joint
//! owns a single Lagrange multiplier in the shared `lambda` buffer, slot `k`
//! for the joint's index `k` in the colour-reordered list; the buffer is
//! therefore `joints.len()` long and is reset to zero at the start of every
//! substep.
//!
//! # The drive update
//!
//! Body `b`'s target world orientation is `q_a * target_rotation`; the
//! world-space error rotation that carries `b` onto it is
//! `error = (q_a * target_rotation) * conj(q_b)`. Taking the shortest-arc
//! representative (non-negative scalar part), its rotation vector is twice the
//! imaginary part, `delta = 2 * vec(error)`; the geodesic axis is `n = delta /
//! |delta|` and the geodesic angle is `theta = |delta|`. The drive is the
//! bilateral compliant-and-damped `XPBD` equality `C = theta` about `n`. With
//! `w = n . (I_a^-1 n) + n . (I_b^-1 n)` the angular effective inverse mass
//! about `n`, `alpha_tilde = drive_compliance / h^2` the regularisation, and
//! `gamma = drive_compliance * drive_damping / h` the damping scale, each sweep
//! forms the unconstrained multiplier step
//!
//! ```text
//! d_lambda = (-theta - alpha_tilde * lambda - gamma * dv) / ((1 + gamma) * w + alpha_tilde)
//! ```
//!
//! where `dv = n . (angvec_a - angvec_b)` is the relative angular displacement
//! that increases `theta` since the substep snapshot (rotating `a` about `+n`
//! carries its target away, rotating `b` about `+n` closes the gap). The
//! accumulated multiplier `lambda` is the drive's net angular impulse this
//! substep, so a torque cap of `max_torque` clamps it to the box
//! `[-max_torque * h, +max_torque * h]`; only the change in the *clamped*
//! multiplier is applied to the bodies. A non-positive `max_torque` disables the
//! clamp. A zero `drive_compliance` collapses the step to the rigid servo
//! `d_lambda = -theta / w`, which the cap then saturates once the demanded
//! torque exceeds `max_torque`.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuAngularSlerpDriveJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical single-slot
//! multiplier layout, read the same per-substep snapshot buffers for the damping
//! term, and apply the identical multiplier clamp. The angular correction uses
//! the array-based `quat_mul` / `quat_conj` helpers (not glam's `Quat`
//! operators) so the `CPU` reference and the shader share one component order
//! and stay in lock-step.
//!
//! Provenance: the relative-orientation geodesic measurement and its substep
//! `XPBD` angular correction (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), with the bilateral compliant-and-damped drive and its
//! Macklin-style damping regularisation (Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics") and the box-limited multiplier
//! clamp realising the torque saturation, over the world-space inverse inertia
//! and quaternion kinematics of Baraff & Witkin. No Unreal Engine source or
//! derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::angular_slerp_drive::AngularSlerpDriveJoint;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, world_inv_inertia_apply, EPSILON,
};
use super::stepper::{predict, recover_velocities, snapshot};
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the angular `SLERP` drives in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's single angular-drive constraint every substep. The
/// joint set is coloured once up front so same-batch joints write disjoint
/// movable bodies; the batches are then solved in order
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
pub fn cpu_solve_joints_angular_slerp_drive(
    state: &mut RigidBodyState,
    joints: &[AngularSlerpDriveJoint],
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

    // One multiplier per joint: slot `k` is the geodesic angular drive. Reset to
    // zero every substep.
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
                    solve_drive(state, &prev_orientations, &ordered[k], h, &mut lambda[k]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one angular `SLERP` drive for a single sweep: measures the geodesic
/// error rotation from `b` to its target `q_a * target_rotation`, forms the
/// compliant-and-damped `XPBD` step, clamps the accumulated impulse to the
/// torque cap, and applies the equal-and-opposite rotation delta to the two
/// bodies.
fn solve_drive(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &AngularSlerpDriveJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    // Target world orientation of body `b`, and the world-space error rotation
    // that carries `b` onto it: `error = target * conj(q_b)`. Computed with the
    // array-based Hamilton product so the shader reproduces it component for
    // component.
    let target = quat_mul(quat_array(q_a), quat_array(joint.target_rotation));
    let mut error = quat_mul(target, quat_conj(quat_array(q_b)));
    // Take the shortest-arc representative: `q` and `-q` are the same rotation,
    // but only the hemisphere with a non-negative scalar part gives the small
    // rotation vector the linearisation assumes.
    if error[3] < 0.0 {
        error = [-error[0], -error[1], -error[2], -error[3]];
    }

    // The rotation vector is twice the imaginary part of the error quaternion.
    let delta = Vec3::new(2.0 * error[0], 2.0 * error[1], 2.0 * error[2]);
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

    let alpha_tilde = joint.drive_compliance / (h * h);
    // XPBD damping scale: gamma = compliance * damping / h (Macklin et al.). It
    // vanishes with the compliance, so a rigid servo carries no damping term.
    let gamma = (joint.drive_compliance * joint.drive_damping) / h;

    // Relative angular displacement that increases `theta` since the substep
    // snapshot: rotating `a` about `+n` carries its target away (opening the
    // error), rotating `b` about `+n` closes it, so `dv = n . (angvec_a -
    // angvec_b)`.
    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = n.dot(angvec_a - angvec_b);

    let d_lambda =
        (-theta - alpha_tilde * *lambda - gamma * dv) / ((1.0 + gamma) * w + alpha_tilde);
    // Box-limited projected step: clamp the accumulated impulse to the torque
    // cap, then apply only the admissible change. `max_torque <= 0` disables it.
    let old = *lambda;
    let unclamped = old + d_lambda;
    let new_lambda = if joint.max_torque > 0.0 {
        let max_impulse = joint.max_torque * h;
        unclamped.clamp(-max_impulse, max_impulse)
    } else {
        unclamped
    };
    *lambda = new_lambda;
    let p = n * (new_lambda - old);

    // Equal and opposite angular impulses: rotating `a` by `+I_a^-1 p` and `b`
    // by `-I_b^-1 p` reduces `theta` for the admissible (negative) step while
    // conserving angular momentum.
    state.orientations[a] = apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, -world_inv_inertia_apply(q_b, ii_b, p));
}

/// Relative rotation of a body since the substep snapshot, as a rotation vector:
/// twice the imaginary part of the delta quaternion `orientation *
/// conj(prev_orientation)`, hemisphere-corrected so the shortest arc is taken.
/// The drive's damping term dots this with the geodesic axis to read the
/// relative angular rate about the axis.
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

    /// Geodesic error angle (radians) of `joint` in `state`, measured the way
    /// the solver measures it: the shortest-arc rotation vector carrying `b`
    /// onto its world target `q_a * target_rotation`, whose length is the angle.
    fn geodesic_error(state: &RigidBodyState, joint: &AngularSlerpDriveJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let q_a = state.orientations[a];
        let q_b = state.orientations[b];
        let target = q_a * joint.target_rotation;
        let mut error = target * q_b.conjugate();
        if error.w < 0.0 {
            error = Quat::from_xyzw(-error.x, -error.y, -error.z, -error.w);
        }
        Vec3::new(2.0 * error.x, 2.0 * error.y, 2.0 * error.z).length()
    }

    /// A static reference body `a` at identity and a dynamic body `b` at
    /// `start`, both with isotropic unit inverse inertia. Body 0 is the static
    /// reference, body 1 the driven body.
    fn servo_pair(start: Quat) -> RigidBodyState {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static reference
        state.push(Vec3::ZERO, start, 1.0, Vec3::splat(1.0)); // 1: driven body
        state
    }

    #[test]
    fn drive_converges_toward_target() {
        // A rigid servo with a generous torque cap pulls the driven body's
        // orientation from identity onto a target 0.8 rad about a tilted axis.
        let target = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0).normalize(), 0.8);
        let joint = AngularSlerpDriveJoint::servo(0, 1, target, 1.0e4);
        let mut state = servo_pair(Quat::IDENTITY);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        let before = geodesic_error(&state, &joint);
        for _ in 0..200 {
            cpu_solve_joints_angular_slerp_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let after = geodesic_error(&state, &joint);
        assert!(before > 0.5, "sanity: the body should start off target");
        assert!(
            after < 1.0e-3,
            "rigid servo failed to reach the target: error = {after}"
        );
    }

    #[test]
    fn already_at_target_is_inactive() {
        // When the driven body already sits on the target the error vanishes, so
        // the drive makes no correction: the orientation is left unmoved.
        let target = Quat::from_axis_angle(Vec3::X, 0.4);
        let joint = AngularSlerpDriveJoint::servo(0, 1, target, 1.0e4);
        let mut state = servo_pair(target);
        let before = state.orientations[1];
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..60 {
            cpu_solve_joints_angular_slerp_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let drift = before.angle_between(state.orientations[1]);
        assert!(
            drift < 1.0e-4,
            "a body already at the target drifted by {drift} rad"
        );
    }

    #[test]
    fn empty_joint_set_is_a_no_op() {
        let mut state = RigidBodyState::new();
        state.push(
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::Y, 0.3),
            1.0,
            Vec3::splat(1.0),
        );
        let before = state.orientations[0];
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_angular_slerp_drive(&mut state, &[], &integrator, &config, 1.0 / 60.0)
            .unwrap();
        assert_eq!(state.orientations[0], before);
    }

    #[test]
    fn angular_momentum_is_conserved() {
        // Two dynamic bodies with isotropic unit inertia, starting at rest with
        // no gravity and a modest relative offset. The drive applies equal and
        // opposite angular impulses, so the net world angular momentum (the sum
        // of the angular velocities, since the inertia is isotropic unit) stays
        // at its initial value of zero. The offset is kept small so the
        // first-order equivalence between the applied rotation-vector impulses
        // and the recovered angular velocities holds tightly — a large relative
        // rotation would accumulate finite-rotation composition drift that is a
        // property of the kinematics, not of the equal-and-opposite drive.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::new(1.0, 0.0, 1.0).normalize(), 0.3),
            1.0,
            Vec3::splat(1.0),
        );
        let joint = AngularSlerpDriveJoint::drive(0, 1, Quat::IDENTITY, 80.0, 1.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        let mut peak = 0.0f32;
        for _ in 0..40 {
            cpu_solve_joints_angular_slerp_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
            let momentum = state.angular_velocities[0] + state.angular_velocities[1];
            peak = peak.max(momentum.length());
        }
        assert!(
            peak < 5.0e-3,
            "net angular momentum departed from zero by {peak}"
        );
    }

    #[test]
    fn rigid_servo_converges_faster_than_a_soft_spring() {
        // From the same large initial error, a rigid (zero-compliance) servo
        // closes the gap faster than a soft angular spring over the same few
        // steps, isolating the compliance as the cause of the lag.
        let target = Quat::from_axis_angle(Vec3::Z, 1.0);
        let rigid = AngularSlerpDriveJoint::servo(0, 1, target, 1.0e4);
        let soft = AngularSlerpDriveJoint::drive(0, 1, target, 20.0, 0.0, 1.0e4);
        let mut rigid_state = servo_pair(Quat::IDENTITY);
        let mut soft_state = servo_pair(Quat::IDENTITY);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        for _ in 0..6 {
            cpu_solve_joints_angular_slerp_drive(
                &mut rigid_state,
                &[rigid],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
            cpu_solve_joints_angular_slerp_drive(
                &mut soft_state,
                &[soft],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
        }
        let rigid_error = geodesic_error(&rigid_state, &rigid);
        let soft_error = geodesic_error(&soft_state, &soft);
        assert!(
            rigid_error < soft_error,
            "rigid servo did not lead the soft spring: rigid = {rigid_error}, soft = {soft_error}"
        );
    }

    #[test]
    fn tight_cap_lags_a_generous_cap() {
        // The same rigid servo from a large error, once with a torque cap far
        // below what the correction demands and once with an ample cap: the
        // tight cap can only nudge the orientation each substep, so after a few
        // steps it trails the generously-capped drive. Isolates the clamp.
        let target = Quat::from_axis_angle(Vec3::X, 1.2);
        let tight = AngularSlerpDriveJoint::servo(0, 1, target, 0.5);
        let generous = AngularSlerpDriveJoint::servo(0, 1, target, 1.0e6);
        let mut tight_state = servo_pair(Quat::IDENTITY);
        let mut generous_state = servo_pair(Quat::IDENTITY);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        for _ in 0..5 {
            cpu_solve_joints_angular_slerp_drive(
                &mut tight_state,
                &[tight],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
            cpu_solve_joints_angular_slerp_drive(
                &mut generous_state,
                &[generous],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
        }
        let tight_error = geodesic_error(&tight_state, &tight);
        let generous_error = geodesic_error(&generous_state, &generous);
        assert!(
            tight_error > generous_error + 0.1,
            "the tight cap did not visibly lag: tight = {tight_error}, generous = {generous_error}"
        );
    }

    #[test]
    fn disabled_cap_matches_an_unbinding_cap() {
        // With `max_torque <= 0` the clamp is disabled; the trajectory must match
        // a drive whose cap is set so high it never binds, confirming the two
        // branches agree in the unsaturated regime.
        let target = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.6);
        let unbounded = AngularSlerpDriveJoint::servo(0, 1, target, 0.0);
        let generous = AngularSlerpDriveJoint::servo(0, 1, target, 1.0e6);
        let mut unbounded_state = servo_pair(Quat::IDENTITY);
        let mut generous_state = servo_pair(Quat::IDENTITY);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_angular_slerp_drive(
                &mut unbounded_state,
                &[unbounded],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
            cpu_solve_joints_angular_slerp_drive(
                &mut generous_state,
                &[generous],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
        }
        let a = geodesic_error(&unbounded_state, &unbounded);
        let b = geodesic_error(&generous_state, &generous);
        assert!(
            (a - b).abs() < 1.0e-4,
            "disabled cap diverged from an unbinding cap: {a} vs {b}"
        );
    }
}
