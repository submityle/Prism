//! Authoritative `CPU` golden reference for the revolute (hinge) velocity-motor
//! joint stepper.
//!
//! [`cpu_solve_joints_revolute_motor`] is a *full stepper* with the identical
//! substep schedule as the other joint goldens: a caller hands it the current
//! [`RigidBodyState`], the joint set, the shared [`IntegratorConfig`], the
//! joint-specific [`JointSolverConfig`], and the frame `dt`, and must **not**
//! integrate the bodies itself. Within each integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's three `XPBD` Lagrange multipliers,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # The three constraints, projected axis-first
//!
//! A revolute motor joint welds the two bodies' anchors together, aligns their
//! hinge axes, and *drives the free spin's rate* about that axis toward a
//! commanded angular velocity. Each sweep projects, in order: the
//! **axis-alignment** angular constraint (realigning the two world-space hinge
//! axes), then the **velocity motor** (servoing the relative angular rate about
//! the freshly aligned axis onto `target_velocity`), then the
//! **point-to-point** positional weld (numerically identical to the spherical
//! golden's `solve_one`). Projecting the axis before the motor means the motor
//! works against an already-aligned hinge axis.
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer:
//! slot `3 * k` for the positional weld, slot `3 * k + 1` for the axis
//! alignment, and slot `3 * k + 2` for the velocity motor, where `k` is the
//! joint's index in the colour-reordered list. The buffer is therefore
//! `3 * joints.len()` long and is reset to zero at the start of every substep.
//!
//! # The motor update
//!
//! The motor is a *velocity-level* constraint expressed in the shared
//! position-based stepper as a per-substep equality on the relative angular
//! displacement about the world hinge axis `u`:
//!
//! ```text
//! C = u . (dphi_b - dphi_a) - target_velocity * h
//! ```
//!
//! where `dphi_a` / `dphi_b` are each body's angular displacement since the
//! substep snapshot (`angular_displacement`, the same small-angle extraction the
//! velocity recovery uses) and `h` is the substep duration. Forcing the relative
//! displacement over the substep onto `target_velocity * h` forces the recovered
//! relative angular velocity `u . (omega_b - omega_a)` onto `target_velocity`.
//! With `w = u . (I_a^-1 u) + u . (I_b^-1 u)` the angular effective inverse mass
//! about `u` and `alpha_tilde = motor_compliance / h^2` the regularisation, each
//! sweep applies
//!
//! ```text
//! d_lambda = (-C - alpha_tilde * lambda) / (w + alpha_tilde)
//! ```
//!
//! The constraint gradients are `-u` on body `a` and `+u` on body `b` (rotating
//! `b` by `+u` increases the relative displacement about `u`), so the correction
//! is equal and opposite and `C_dot = u . (omega_b - omega_a)`. A zero
//! `motor_compliance` collapses the update to the rigid motor
//! `d_lambda = -C / w`, forcing the relative rate exactly onto the target each
//! substep with unbounded torque; a positive compliance applies a finite torque
//! so the rate approaches the target asymptotically. The motor needs no explicit
//! dashpot — it *is* the rate-tracking term — so it carries no damping gain.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuRevoluteMotorJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout and read the same per-substep snapshot buffers for the
//! motor's relative-displacement term. There is no transcendental in the motor
//! path — it reads only a dot product of rotation vectors — so the `CPU`
//! reference and the shader share one arithmetic path with only device
//! reassociation separating them.
//!
//! Provenance: the point-to-point (ball-socket) constraint and the hinge
//! axis-alignment constraint (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), with the velocity-level motor expressed as a per-substep
//! compliant equality on the relative angular displacement about the hinge axis
//! (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
//! Dynamics"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::revolute_motor::RevoluteMotorJoint;
use super::stepper::{predict, recover_velocities, snapshot};
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the revolute motor joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment, velocity-motor, and point-to-point
/// constraints every substep. The joint set is coloured once up front so
/// same-batch joints write disjoint movable bodies; the batches are then solved
/// in order [`position_iterations`](JointSolverConfig::position_iterations)
/// times per substep.
///
/// # Errors
///
/// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree in
/// length or a joint references a body outside the state, and
/// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
/// batches than the colouring supports. Returns `Ok(())` with the state
/// untouched when there is nothing to do (`dt <= 0`, no bodies, or no joints).
pub fn cpu_solve_joints_revolute_motor(
    state: &mut RigidBodyState,
    joints: &[RevoluteMotorJoint],
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

    // Three multipliers per joint: `[3k]` positional weld, `[3k + 1]` axis
    // alignment, `[3k + 2]` velocity motor. Reset to zero every substep.
    let mut lambda = vec![0.0f32; 3 * ordered.len()];

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
                    solve_one(
                        state,
                        &prev_orientations,
                        &ordered[k],
                        h,
                        &mut lambda[3 * k..3 * k + 3],
                    );
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one revolute motor joint for a single sweep: axis alignment first
/// (accumulating `lambda[1]`), then the velocity motor (`lambda[2]`), then the
/// point-to-point weld (`lambda[0]`), each applied directly to the two bodies'
/// transforms.
fn solve_one(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &RevoluteMotorJoint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_motor(state, prev_orientations, joint, h, &mut lambda[2]);
    solve_point_to_point(state, joint, h, &mut lambda[0]);
}

/// Drives the two world-space hinge axes parallel by cancelling their cross
/// product, locking the two rotational degrees of freedom perpendicular to the
/// hinge while leaving the spin about it free. Identical to the revolute and
/// hinge-limit goldens' axis-alignment pass.
fn solve_axis_alignment(
    state: &mut RigidBodyState,
    joint: &RevoluteMotorJoint,
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

    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Servos the relative angular rate about the world hinge axis `u` onto
/// `target_velocity` with a bilateral compliant `XPBD` constraint on the
/// per-substep relative angular displacement. The violation
/// `c = u . (dphi_b - dphi_a) - target_velocity * h` is cancelled by a signed
/// rotation about `u`; `motor_compliance` softens the rigid rate-lock into a
/// finite-torque motor. There is no zero-angle reference and no absolute hinge
/// angle: the motor reads only the relative rotation accumulated since the
/// substep snapshot.
fn solve_motor(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &RevoluteMotorJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let axis = rotate(q_a, joint.axis_a);
    let axis_len = axis.length();
    if axis_len < EPSILON {
        return;
    }
    let u = axis / axis_len;

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = u.dot(world_inv_inertia_apply(q_a, ii_a, u));
    let w_b = u.dot(world_inv_inertia_apply(q_b, ii_b, u));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    // Relative angular displacement about `u` since the substep snapshot, with
    // the constraint gradient's sign (`-u` on `a`, `+u` on `b`), consistent with
    // the effective mass `w`.
    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = u.dot(angvec_b - angvec_a);

    // Target relative displacement over this substep: a rate `target_velocity`
    // held for the substep duration `h`.
    let target = joint.target_velocity * h;
    let c = dv - target;

    let alpha_tilde = joint.motor_compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = u * d_lambda;

    // Rotating body `b` by `+u` increases the relative displacement about `u`;
    // rotating body `a` by `+u` decreases it. The constraint gradients are
    // therefore `-u` on `a` and `+u` on `b`.
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Drives the two world-space anchors together with the identical point-to-point
/// `XPBD` positional correction as the spherical golden.
fn solve_point_to_point(
    state: &mut RigidBodyState,
    joint: &RevoluteMotorJoint,
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

/// Relative rotation of a body since the substep snapshot, as a rotation vector:
/// twice the imaginary part of the delta quaternion `orientation *
/// conj(prev_orientation)` (the same small-angle extraction the velocity
/// recovery uses), hemisphere-corrected so the shortest arc is taken. The motor
/// dots this with the hinge axis to read the relative angular rate about the
/// axis accumulated over the substep.
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

    /// Relative angular velocity of `joint`'s two bodies about its world hinge
    /// axis in `state`: `u . (omega_b - omega_a)`, the quantity the motor
    /// commands onto `target_velocity`.
    fn relative_rate(state: &RigidBodyState, joint: &RevoluteMotorJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let u = rotate(state.orientations[a], joint.axis_a).normalize();
        u.dot(state.angular_velocities[b] - state.angular_velocities[a])
    }

    /// Separation between a joint's two world-space anchors in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &RevoluteMotorJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    /// A dynamic rotor hinged to a static housing about the world Z axis, their
    /// anchors coincident at the origin. Body 0 is the static housing; body 1
    /// the dynamic rotor. Builds the joint from the given constructor so each
    /// test can pick a rigid or a soft motor.
    fn rotor(joint: RevoluteMotorJoint) -> (RigidBodyState, RevoluteMotorJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: housing
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 1: rotor
        (state, joint)
    }

    fn rigid_motor(target_velocity: f32) -> RevoluteMotorJoint {
        RevoluteMotorJoint::rigid(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            target_velocity,
        )
    }

    #[test]
    fn rigid_motor_spins_rotor_to_target() {
        // A rigid motor forces the relative hinge rate exactly onto the target
        // each substep, so a rotor against a static housing reaches the
        // commanded angular velocity almost immediately and holds it.
        let target = 3.0;
        let (mut state, joint) = rotor(rigid_motor(target));
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..10 {
            cpu_solve_joints_revolute_motor(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let rate = relative_rate(&state, &joint);
        assert!(
            (rate - target).abs() < 1.0e-2,
            "rigid motor failed to reach target rate: rate = {rate}, target = {target}"
        );
        assert!(
            anchor_separation(&state, &joint) < 1.0e-3,
            "anchor drifted apart"
        );
    }

    #[test]
    fn soft_motor_applies_finite_torque_then_reaches_target() {
        // A soft motor applies only a finite torque, so the rotor spins up
        // gradually: after a couple of frames it is turning but has not yet
        // reached the commanded speed, and after a long run it settles onto it.
        let target = 4.0;
        // A finite but firm torque stiffness: soft enough that the rotor spins
        // up gradually over tens of frames (a visible, finite-torque ramp rather
        // than the rigid motor's instant snap), yet firm enough that the
        // first-order relaxation toward the target settles within the run. The
        // per-substep relaxation rate of this velocity motor is
        // `1 / (1 + motor_compliance / h^2)`, so a too-soft motor would still be
        // ramping at the end of any practical horizon.
        let (mut state, joint) = rotor(RevoluteMotorJoint::soft(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Z,
            target,
            1500.0,
        ));
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..2 {
            cpu_solve_joints_revolute_motor(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let early = relative_rate(&state, &joint);
        assert!(
            early > 1.0e-3 && early < target,
            "soft motor should apply a finite spin-up torque: early rate = {early}, target = {target}"
        );

        for _ in 0..600 {
            cpu_solve_joints_revolute_motor(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let settled = relative_rate(&state, &joint);
        assert!(
            (settled - target).abs() < 5.0e-2,
            "soft motor failed to settle on the target rate: settled = {settled}, target = {target}"
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, one body given an initial velocity: all of
        // the motor's corrections are internal, so total linear momentum is
        // conserved up to solver noise.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.5, 0.2, -0.1);
        let joint = RevoluteMotorJoint::rigid(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            1.5,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_revolute_motor(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 5.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }

    #[test]
    fn empty_joint_set_is_a_no_op() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let before = state.positions[0];
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_revolute_motor(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], before);
    }

    #[test]
    fn alignment_pulls_a_tilted_axis_parallel() {
        // A static pivot and a dynamic motored rotor whose hinge axes start 30
        // degrees apart: the rigid axis alignment must pull them toward
        // *parallel* (`dot = +1`), never flip them to anti-parallel
        // (`dot = -1`). The signed `dot` guards the correction's sign, which a
        // `sin`-only measure cannot see because both parallel and anti-parallel
        // read ~0 there.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static pivot
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 1: dynamic rotor
        state.orientations[1] = Quat::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_6);
        let joint = rigid_motor(2.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..4 {
            cpu_solve_joints_revolute_motor(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }

        let u_a = rotate(state.orientations[0], joint.axis_a).normalize();
        let u_b = rotate(state.orientations[1], joint.axis_b).normalize();
        assert!(
            u_a.dot(u_b) > 0.9,
            "hinge axes converged anti-parallel: dot = {}",
            u_a.dot(u_b)
        );
    }
}
