//! Authoritative `CPU` golden reference for the revolute (hinge) torque-saturated
//! angular servo joint stepper.
//!
//! [`cpu_solve_joints_revolute_servo`] is a *full stepper* with the identical
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
//! A revolute servo joint welds the two bodies' anchors together, aligns their
//! hinge axes, and *actively servos* the free spin about that axis toward a
//! commanded angle with a **torque-limited** drive. Each sweep projects, in
//! order: the **axis-alignment** angular constraint (realigning the two
//! world-space hinge axes), then the **torque-saturated angular drive**
//! (servoing the signed hinge angle onto `target_angle` about the freshly
//! aligned axis, capping the drive torque at `max_torque`), then the
//! **point-to-point** positional weld (numerically identical to the spherical
//! golden's `solve_one`). Projecting the axis before the drive means the drive
//! works against an already-aligned hinge axis.
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer:
//! slot `3 * k` for the positional weld, slot `3 * k + 1` for the axis
//! alignment, and slot `3 * k + 2` for the angular drive, where `k` is the
//! joint's index in the colour-reordered list. The buffer is therefore
//! `3 * joints.len()` long and is reset to zero at the start of every substep.
//!
//! # The torque-saturated drive update
//!
//! The drive is the bilateral compliant-and-damped `XPBD` equality
//! `C = theta - target_angle`, where `theta` is the signed hinge angle measured
//! about the world hinge axis `u`. With `w = u . (I_a^-1 u) + u . (I_b^-1 u)`
//! the angular effective inverse mass about `u`, `alpha_tilde =
//! drive_compliance / h^2` the regularisation, and `gamma = drive_compliance *
//! drive_damping / h` the damping scale, each sweep forms the unconstrained
//! multiplier step
//!
//! ```text
//! d_lambda = (-C - alpha_tilde * lambda - gamma * dv) / ((1 + gamma) * w + alpha_tilde)
//! ```
//!
//! where `dv = u . (angvec_b - angvec_a)` is the relative angular displacement
//! about `u` since the substep snapshot. The accumulated multiplier `lambda` is
//! the drive's net angular impulse this substep, so a torque cap of `max_torque`
//! clamps it to the box `[-max_torque * h, +max_torque * h]`; only the change in
//! the *clamped* multiplier is applied to the bodies. A non-positive
//! `max_torque` disables the clamp, recovering the unbounded drive. A zero
//! `drive_compliance` collapses the step to the rigid servo `d_lambda = -C / w`,
//! which the cap then saturates once the demanded torque exceeds `max_torque`.
//!
//! # Measuring the hinge angle
//!
//! Each body carries a body-local reference direction, `ref_a` and `ref_b`,
//! nominally perpendicular to its hinge axis. Both are rotated to world space,
//! projected onto the plane perpendicular to the unit hinge axis `u`, and
//! normalised; the signed angle from `a`'s projection to `b`'s projection about
//! `u` is `theta = atan2((p_a x p_b) . u, p_a . p_b)`. The references need not be
//! exactly perpendicular to the axis — only non-parallel to it — because the
//! projection removes the axial component first.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuRevoluteServoJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout, read the same per-substep snapshot buffers for the damping
//! term, and apply the identical multiplier clamp. The angle is taken with
//! [`bevy_math::ops::atan2`] rather than `f32::atan2` so the `CPU` reference and
//! the shader share one transcendental path.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the hinge
//! axis-alignment constraint, and the signed hinge-angle measurement (Müller et
//! al., "Detailed Rigid Body Simulation with XPBD"), with the bilateral
//! compliant-and-damped angular drive and its Macklin-style damping
//! regularisation (Macklin et al., "XPBD: Position-Based Simulation of Compliant
//! Constrained Dynamics") and the box-limited multiplier clamp realising the
//! torque saturation, over the world-space inverse inertia and quaternion
//! kinematics of Baraff & Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::revolute_servo::RevoluteServoJoint;
use super::stepper::{predict, recover_velocities, snapshot};
use bevy_math::ops;
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the revolute servo joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment, torque-saturated angular-drive, and
/// point-to-point constraints every substep. The joint set is coloured once up
/// front so same-batch joints write disjoint movable bodies; the batches are
/// then solved in order
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
pub fn cpu_solve_joints_revolute_servo(
    state: &mut RigidBodyState,
    joints: &[RevoluteServoJoint],
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
    // alignment, `[3k + 2]` angular drive. Reset to zero every substep.
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

/// Projects one revolute servo joint for a single sweep: axis alignment first
/// (accumulating `lambda[1]`), then the torque-saturated angular drive
/// (`lambda[2]`), then the point-to-point weld (`lambda[0]`), each applied
/// directly to the two bodies' transforms.
fn solve_one(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &RevoluteServoJoint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_drive(state, prev_orientations, joint, h, &mut lambda[2]);
    solve_point_to_point(state, joint, h, &mut lambda[0]);
}

/// Drives the two world-space hinge axes parallel by cancelling their cross
/// product, locking the two rotational degrees of freedom perpendicular to the
/// hinge while leaving the spin about it free. Identical to the revolute and
/// hinge-limit goldens' axis-alignment pass.
fn solve_axis_alignment(
    state: &mut RigidBodyState,
    joint: &RevoluteServoJoint,
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

/// Servos the signed hinge angle `theta` onto `target_angle` with a bilateral
/// compliant-and-damped `XPBD` constraint about the world hinge axis `u`, then
/// saturates the drive torque at `max_torque`. The violation
/// `c = theta - target_angle` is cancelled by a signed rotation about `u`;
/// `drive_compliance` softens the servo into an angular spring and
/// `drive_damping` (scaled by the compliance) resists the hinge's closing rate.
/// The accumulated multiplier is clamped to `+/- max_torque * h` (the net
/// angular impulse a torque of `max_torque` can deliver over the substep), and
/// only the change in the clamped multiplier is applied. The angle is measured
/// from body `a`'s reference direction to body `b`'s, both projected onto the
/// plane perpendicular to `u`.
fn solve_drive(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &RevoluteServoJoint,
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

    let ra = rotate(q_a, joint.ref_a);
    let rb = rotate(q_b, joint.ref_b);
    let pa = ra - u * u.dot(ra);
    let pb = rb - u * u.dot(rb);
    let la = pa.length();
    let lb = pb.length();
    if la < EPSILON || lb < EPSILON {
        return;
    }
    let pa_n = pa / la;
    let pb_n = pb / lb;

    // Signed hinge angle from `a`'s reference to `b`'s about the hinge axis.
    let sin_theta = pa_n.cross(pb_n).dot(u);
    let cos_theta = pa_n.dot(pb_n);
    let theta = ops::atan2(sin_theta, cos_theta);

    // Bilateral: the drive is always active, pulling `theta` onto the target
    // from either side.
    let c = theta - joint.target_angle;

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = u.dot(world_inv_inertia_apply(q_a, ii_a, u));
    let w_b = u.dot(world_inv_inertia_apply(q_b, ii_b, u));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    let alpha_tilde = joint.drive_compliance / (h * h);
    // XPBD damping scale: gamma = compliance * damping / h (Macklin et al.). It
    // vanishes with the compliance, so a rigid servo carries no damping term.
    let gamma = (joint.drive_compliance * joint.drive_damping) / h;

    // Relative angular displacement about `u` since the substep snapshot, with
    // the constraint gradient's sign (`-u` on `a`, `+u` on `b`), consistent with
    // the effective mass `w`.
    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = u.dot(angvec_b - angvec_a);

    let d_lambda = (-c - alpha_tilde * *lambda - gamma * dv) / ((1.0 + gamma) * w + alpha_tilde);
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
    let p = u * (new_lambda - old);

    // Rotating body `b` by `+u` increases `theta`; rotating body `a` by `+u`
    // decreases it. The constraint gradients are therefore `-u` on `a` and `+u`
    // on `b`.
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Drives the two world-space anchors together with the identical point-to-point
/// `XPBD` positional correction as the spherical golden.
fn solve_point_to_point(
    state: &mut RigidBodyState,
    joint: &RevoluteServoJoint,
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
/// recovery uses), hemisphere-corrected so the shortest arc is taken. The hinge
/// drive's damping term dots this with the hinge axis to read the relative
/// angular rate about the axis.
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
    use core::f32::consts::FRAC_PI_4;

    /// Signed hinge angle (radians) of `joint` in `state`, measured the same way
    /// the solver measures it.
    fn hinge_angle(state: &RigidBodyState, joint: &RevoluteServoJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let q_a = state.orientations[a];
        let q_b = state.orientations[b];
        let u = rotate(q_a, joint.axis_a).normalize();
        let ra = rotate(q_a, joint.ref_a);
        let rb = rotate(q_b, joint.ref_b);
        let pa = (ra - u * u.dot(ra)).normalize();
        let pb = (rb - u * u.dot(rb)).normalize();
        let sin_theta = pa.cross(pb).dot(u);
        let cos_theta = pa.dot(pb);
        ops::atan2(sin_theta, cos_theta)
    }

    /// Separation between a joint's two world-space anchors in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &RevoluteServoJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    /// A dynamic body hinged to a static pivot about the world Z axis. Body 0 is
    /// the static pivot; body 1 the driven link, extending `+X` and anchored
    /// back to the origin.
    fn hinged_link(joint: RevoluteServoJoint) -> (RigidBodyState, RevoluteServoJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: pivot
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: link, extends +X
        (state, joint)
    }

    #[test]
    fn generous_cap_servo_holds_at_target_under_gravity() {
        // A rigid servo with a torque cap far above the gravitational load holds
        // the link level: the cap never binds, so it behaves as the unbounded
        // position servo.
        let joint = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            0.0,
            1.0e4,
        );
        let (mut state, joint) = hinged_link(joint);
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..300 {
            cpu_solve_joints_revolute_servo(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let theta = hinge_angle(&state, &joint);
        assert!(
            theta.abs() < 2.0e-2,
            "generously-capped servo failed to hold level: theta = {theta}"
        );
        assert!(
            anchor_separation(&state, &joint) < 1.0e-3,
            "anchor drifted apart"
        );
    }

    #[test]
    fn tight_cap_stalls_below_level() {
        // The hinge commanded to hold level (`target = 0`) against gravity, but
        // with a torque cap far below the gravitational load: the drive saturates
        // every sweep and can no longer resist the droop, so the link stalls well
        // below level instead of holding. An otherwise identical joint with an
        // ample cap holds level, isolating the clamp as the cause.
        let target = 0.0;
        let capped = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            target,
            0.005,
        );
        let (mut capped_state, capped) = hinged_link(capped);
        // An otherwise identical joint with an ample cap holds level.
        let free = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            target,
            1.0e4,
        );
        let (mut free_state, free) = hinged_link(free);

        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.1, 0.05);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..300 {
            cpu_solve_joints_revolute_servo(&mut capped_state, &[capped], &integrator, &config, dt)
                .unwrap();
            cpu_solve_joints_revolute_servo(&mut free_state, &[free], &integrator, &config, dt)
                .unwrap();
        }
        let capped_theta = hinge_angle(&capped_state, &capped);
        let free_theta = hinge_angle(&free_state, &free);
        assert!(
            free_theta.abs() < 2.0e-2,
            "amply-capped servo failed to hold level: theta = {free_theta}"
        );
        assert!(
            capped_theta < -2.0e-1,
            "tight cap failed to stall the servo below level: theta = {capped_theta}"
        );
        // The weld still holds regardless of the drive saturation.
        assert!(
            anchor_separation(&capped_state, &capped) < 1.0e-3,
            "anchor drifted apart under the tight cap"
        );
    }

    #[test]
    fn capped_spring_converges_when_cap_is_slack() {
        // A soft spring-damper onto +45 deg with no gravity and an ample cap:
        // the cap never binds, so the spring relaxes onto its target.
        let target = FRAC_PI_4;
        let joint = RevoluteServoJoint::drive(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            target,
            150.0,
            8.0,
            1.0e4,
        );
        let (mut state, joint) = hinged_link(joint);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..600 {
            cpu_solve_joints_revolute_servo(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let theta = hinge_angle(&state, &joint);
        assert!(
            (theta - target).abs() < 2.0e-2,
            "slack-capped spring failed to settle on the target: theta = {theta}, target = {target}"
        );
        assert!(
            anchor_separation(&state, &joint) < 1.0e-3,
            "anchor drifted apart"
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, one body given an initial velocity: all of
        // the joint's corrections are internal, so total linear momentum is
        // conserved up to solver noise, torque cap or not.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.5, 0.2, -0.1);
        let joint = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.3,
            20.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_revolute_servo(&mut state, &[joint], &integrator, &config, dt)
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
        cpu_solve_joints_revolute_servo(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], before);
    }

    #[test]
    fn alignment_pulls_a_tilted_axis_parallel() {
        // A static pivot and a dynamic driven link whose hinge axes start 30
        // degrees apart: the rigid axis alignment must pull them toward
        // *parallel* (`dot = +1`), never flip them anti-parallel. Independent of
        // the drive torque cap.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static pivot
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 1: dynamic link
        state.orientations[1] = Quat::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_6);
        let joint = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            0.0,
            5.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..4 {
            cpu_solve_joints_revolute_servo(&mut state, &[joint], &integrator, &config, dt)
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

    #[test]
    fn non_positive_cap_matches_an_amply_capped_drive() {
        // With `max_torque <= 0` the clamp is disabled; the trajectory must match
        // a drive whose cap is set so high it never binds, confirming the two
        // branches agree in the unsaturated regime.
        let target = FRAC_PI_4;
        let unbounded = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            target,
            0.0,
        );
        let (mut unbounded_state, unbounded) = hinged_link(unbounded);
        let generous = RevoluteServoJoint::servo(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            target,
            1.0e6,
        );
        let (mut generous_state, generous) = hinged_link(generous);

        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_revolute_servo(
                &mut unbounded_state,
                &[unbounded],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
            cpu_solve_joints_revolute_servo(
                &mut generous_state,
                &[generous],
                &integrator,
                &config,
                dt,
            )
            .unwrap();
        }
        let a = hinge_angle(&unbounded_state, &unbounded);
        let b = hinge_angle(&generous_state, &generous);
        assert!(
            (a - b).abs() < 1.0e-4,
            "disabled cap diverged from an unbinding cap: {a} vs {b}"
        );
    }
}
