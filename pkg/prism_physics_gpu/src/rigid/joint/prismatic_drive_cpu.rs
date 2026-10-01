//! Authoritative `CPU` golden reference for the prismatic drive (linear motor)
//! joint stepper.
//!
//! [`cpu_solve_joints_prismatic_drive`] is a *full stepper* with the identical
//! substep schedule as
//! [`cpu_solve_joints_prismatic`](super::cpu_solve_joints_prismatic): a caller
//! hands it the current [`RigidBodyState`], the joint set, the shared
//! [`IntegratorConfig`], the joint-specific [`JointSolverConfig`], and the frame
//! `dt`, and must **not** integrate the bodies itself. Within each integrator
//! substep the stepper
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
//! # The three constraints, projected angle-first
//!
//! A prismatic drive joint locks the relative orientation of the two bodies,
//! confines their anchors to a shared slide line, and *actively servos* the free
//! along-axis separation toward a commanded target. Each sweep projects the
//! **angular lock** first — cancelling the world-space error between body `b`'s
//! orientation and its target `q_a * rest_rotation` — then the **perpendicular
//! weld**, the point-to-point correction restricted to the plane perpendicular
//! to the world-space slide axis, then the **linear drive**, a bilateral
//! compliant-and-damped correction pulling the signed separation onto
//! `target_position`. Projecting the orientation before the anchors means the
//! perpendicular and drive passes work against an already-oriented slide axis.
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer:
//! slot `3 * k` for the perpendicular weld, slot `3 * k + 1` for the angular
//! lock, and slot `3 * k + 2` for the drive, where `k` is the joint's index in
//! the colour-reordered list. The buffer is therefore `3 * joints.len()` long
//! and is reset to zero at the start of every substep.
//!
//! # The drive update
//!
//! The drive is the compliant-and-damped `XPBD` equality `C = s -
//! target_position`, projected along the world slide axis `n`. With `w` the
//! effective inverse mass along `n`, `alpha_tilde = drive_compliance / h^2` the
//! regularisation, and `gamma = drive_compliance * drive_damping / h` the
//! damping scale, each sweep applies
//!
//! ```text
//! d_lambda = (-C - alpha_tilde * lambda - gamma * (n . dv)) / ((1 + gamma) * w + alpha_tilde)
//! ```
//!
//! where `n . dv` is the along-axis component of the net anchor displacement
//! since the substep snapshot (translation plus the rotation of the anchor
//! arm). A zero `drive_compliance` collapses this to the rigid update
//! `d_lambda = -C / w`, snapping the slide onto the target; the damping term
//! vanishes with it, as a rigid servo needs none.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuPrismaticDriveJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout and read the same per-substep snapshot buffers for the
//! damping term. The angular lock uses the array-based `quat_mul` / `quat_conj`
//! helpers (not glam's `Quat` operators) so the `CPU` reference and the shader
//! share one component order and stay in lock-step.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis, the relative-orientation lock, and the
//! bilateral along-axis drive with their substep compliant-and-damped `XPBD`
//! handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"; Macklin
//! et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"),
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::prismatic_drive::PrismaticDriveJoint;
use super::stepper::{predict, recover_velocities, snapshot};
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the prismatic drive joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's angular-lock, perpendicular-weld, and drive
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
pub fn cpu_solve_joints_prismatic_drive(
    state: &mut RigidBodyState,
    joints: &[PrismaticDriveJoint],
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

    // Three Lagrange multipliers per joint: slot `3k` for the perpendicular
    // weld, slot `3k + 1` for the angular lock, slot `3k + 2` for the drive.
    // Reset to zero at the start of every substep.
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
                        &prev_positions,
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

/// Projects one prismatic drive joint for a single sweep: the angular-lock
/// correction first (accumulating `lambda[1]`), then the perpendicular-weld
/// positional correction (accumulating `lambda[0]`), then the bilateral drive
/// correction (accumulating `lambda[2]`), each applied directly to the two
/// bodies' transforms.
fn solve_one(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    joint: &PrismaticDriveJoint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_angular_lock(state, joint, h, &mut lambda[1]);
    solve_perpendicular(state, joint, h, &mut lambda[0]);
    solve_drive(
        state,
        prev_positions,
        prev_orientations,
        joint,
        h,
        &mut lambda[2],
    );
}

/// Drives body `b`'s world orientation to its target `q_a * rest_rotation`,
/// locking all three relative rotational degrees of freedom. The world-space
/// error rotation is `target * conj(q_b)`; its imaginary part (times two) is the
/// rotation vector that carries `b` back onto the target, and the equal and
/// opposite impulse rotates `a`'s target to meet it.
fn solve_angular_lock(
    state: &mut RigidBodyState,
    joint: &PrismaticDriveJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let target = quat_mul(quat_array(q_a), quat_array(joint.rest_rotation));
    let mut error = quat_mul(target, quat_conj(quat_array(q_b)));
    if error[3] < 0.0 {
        error = [-error[0], -error[1], -error[2], -error[3]];
    }

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

    let alpha_tilde = joint.angular_compliance / (h * h);
    let d_lambda = (-theta - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    state.orientations[a] = apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, -world_inv_inertia_apply(q_b, ii_b, p));
}

/// Drives the two world-space anchors together in the plane perpendicular to the
/// world-space slide axis, leaving the along-axis separation free. This is the
/// point-to-point `XPBD` positional correction with the separation projected
/// onto the perpendicular plane before it is cancelled.
fn solve_perpendicular(
    state: &mut RigidBodyState,
    joint: &PrismaticDriveJoint,
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

    let axis = rotate(q_a, joint.axis_a);
    let axis_len = axis.length();
    if axis_len < EPSILON {
        return;
    }
    let axis_w = axis / axis_len;

    let dx_perp = dx - axis_w * dx.dot(axis_w);
    let c = dx_perp.length();
    if c < EPSILON {
        return;
    }
    let n = dx_perp / c;

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

/// Servos the signed along-axis separation `s = dx . axis_world` toward
/// `target_position` with a bilateral compliant-and-damped `XPBD` constraint.
/// The violation `c = s - target_position` is cancelled by an impulse along the
/// world slide axis; `drive_compliance` softens the servo into a spring and
/// `drive_damping` resists the slide rate so the approach settles without
/// ringing. The correction gradient is the point-to-point gradient restricted to
/// the single axis direction, so it feeds back into both bodies' translation and
/// rotation like the perpendicular weld.
fn solve_drive(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    joint: &PrismaticDriveJoint,
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

    let axis = rotate(q_a, joint.axis_a);
    let axis_len = axis.length();
    if axis_len < EPSILON {
        return;
    }
    let axis_w = axis / axis_len;

    let s = dx.dot(axis_w);
    // Bilateral: the drive is always active, pulling `s` onto the target from
    // either side.
    let c = s - joint.target_position;

    // The correction direction is the world slide axis.
    let n = axis_w;

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

    let alpha_tilde = joint.drive_compliance / (h * h);
    // XPBD damping scale: gamma = compliance * damping / h (Macklin et al.). It
    // vanishes with the compliance, so a rigid servo carries no damping term.
    let gamma = (joint.drive_compliance * joint.drive_damping) / h;

    // Along-axis component of the net anchor displacement since the substep
    // snapshot: translation plus the rotation of the anchor arm, consistent with
    // the point-to-point gradient `n` used for `w`.
    let dv_a = anchor_displacement(
        state.positions[a],
        prev_positions[a],
        q_a,
        prev_orientations[a],
        r_a,
    );
    let dv_b = anchor_displacement(
        state.positions[b],
        prev_positions[b],
        q_b,
        prev_orientations[b],
        r_b,
    );
    let dv = n.dot(dv_a - dv_b);

    let d_lambda = (-c - alpha_tilde * *lambda - gamma * dv) / ((1.0 + gamma) * w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    state.positions[a] += p * inv_m_a;
    state.positions[b] -= p * inv_m_b;
    let dw_a = world_inv_inertia_apply(q_a, ii_a, r_a.cross(p));
    state.orientations[a] = apply_rotation_delta(q_a, dw_a);
    let dw_b = world_inv_inertia_apply(q_b, ii_b, r_b.cross(p));
    state.orientations[b] = apply_rotation_delta(q_b, -dw_b);
}

/// Net displacement of a body's anchor point since the substep snapshot: the
/// centre-of-mass translation `position - prev_position` plus the rotation of
/// the anchor arm `rotvec(orientation * conj(prev_orientation)) x r`, where the
/// rotation vector is twice the imaginary part of the delta quaternion (the same
/// small-angle extraction the velocity recovery uses). The anchor arm `r` is the
/// current `rotate(orientation, anchor)`, matching the gradient the drive's
/// effective mass is built from.
fn anchor_displacement(
    position: Vec3,
    prev_position: Vec3,
    orientation: Quat,
    prev_orientation: Quat,
    r: Vec3,
) -> Vec3 {
    let delta = quat_mul(
        quat_array(orientation),
        quat_conj(quat_array(prev_orientation)),
    );
    let mut rotvec = Vec3::new(delta[0], delta[1], delta[2]) * 2.0;
    if delta[3] < 0.0 {
        rotvec = -rotvec;
    }
    (position - prev_position) + rotvec.cross(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a one-body vertical slider: a static pivot at the origin (body
    /// `b`, index 0) and a dynamic slider (body `a`, index 1) coincident with
    /// it, free to slide along `+Y`. Body `a` is the dynamic slider so the
    /// signed slide position `s = (p_slider - p_pivot) . axis` equals the
    /// slider's height, running negative as it descends under gravity.
    fn vertical_slider() -> RigidBodyState {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: pivot
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 1: slider
        state
    }

    fn slide_position(state: &RigidBodyState, joint: &PrismaticDriveJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let axis = state.orientations[a] * joint.axis_a;
        let axis_w = axis.normalize();
        let r_a = state.orientations[a] * joint.anchor_a;
        let r_b = state.orientations[b] * joint.anchor_b;
        let dx = (state.positions[a] + r_a) - (state.positions[b] + r_b);
        dx.dot(axis_w)
    }

    #[test]
    fn rigid_servo_holds_against_gravity() {
        // A rigid servo commanded to the start position must hold the slider at
        // s = 0 against gravity dragging it down the +Y axis.
        let joint = PrismaticDriveJoint::servo(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.0);
        let mut state = vertical_slider();
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_prismatic_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let s = slide_position(&state, &joint);
        assert!(
            s.abs() < 1.0e-3,
            "rigid servo drifted off its target: s {s}"
        );
    }

    #[test]
    fn rigid_servo_lifts_above_start() {
        // A rigid servo commanded above the start must lift the slider up to the
        // target and hold it there, working against gravity.
        let joint = PrismaticDriveJoint::servo(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.5);
        let mut state = vertical_slider();
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_prismatic_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let s = slide_position(&state, &joint);
        assert!(
            (s - 0.5).abs() < 1.0e-3,
            "rigid servo did not reach its raised target: s {s}"
        );
    }

    #[test]
    fn soft_spring_converges_to_target_without_gravity() {
        // A damped spring drive with no gravity must settle the slider onto its
        // commanded target.
        let joint =
            PrismaticDriveJoint::spring(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, 0.4, 300.0, 30.0);
        let mut state = vertical_slider();
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..180 {
            cpu_solve_joints_prismatic_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let s = slide_position(&state, &joint);
        assert!(
            (s - 0.4).abs() < 1.0e-2,
            "soft spring did not converge to its target: s {s}"
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum_along_axis() {
        // Zero gravity, equal masses, a drive commanding the two anchors apart:
        // the drive's internal corrections are equal and opposite, so total
        // linear momentum is conserved.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.0, -0.3, 0.0);
        state.linear_velocities[1] = Vec3::new(0.0, 0.7, 0.0);
        let joint =
            PrismaticDriveJoint::spring(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, -1.0, 200.0, 10.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_prismatic_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 1.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }
}
