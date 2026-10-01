//! Authoritative `CPU` golden reference for the cylindrical drive (linear motor)
//! joint stepper.
//!
//! [`cpu_solve_joints_cylindrical_drive`] is a *full stepper* with the identical
//! substep schedule as
//! [`cpu_solve_joints_cylindrical`](super::cpu_solve_joints_cylindrical): a
//! caller hands it the current [`RigidBodyState`], the joint set, the shared
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
//! # The three constraints, projected alignment-first
//!
//! A cylindrical drive joint keeps body `b`'s axis parallel to body `a`'s, pins
//! the two anchors to a common line, and *actively servos* the free along-axis
//! separation toward a commanded target, leaving the spin about the axis free.
//! Each sweep projects the **axis-alignment** constraint first — driving the two
//! world-space axes back to parallel — then the **point-on-line** weld, which
//! cancels only the component of the anchor separation perpendicular to the (now
//! re-aligned) axis, then the **linear drive**, a bilateral compliant-and-damped
//! correction pulling the signed separation onto `target_position`. Projecting
//! the axis orientation before the anchors means the positional and drive passes
//! work against an already-parallel axis.
//!
//! The axis-alignment constraint takes the cross product `delta = u_a x u_b` of
//! the two unit axes (whose magnitude is `sin` of the angle between them), drives
//! its length `theta` to zero by rotating about `n = delta / theta`, with the
//! gradient `+n` on body `a` and `-n` on body `b`. This reuses, bit for bit, the
//! alignment geometry of the [`RevoluteJoint`](super::RevoluteJoint). The
//! point-on-line weld removes only `dx_perp = dx - axis_w * dot(dx, axis_w)`, the
//! perpendicular part of the anchor separation about the world-space axis,
//! reusing the perpendicular weld of the [`PrismaticJoint`](super::PrismaticJoint).
//! The drive servos `s = dx . axis_w` onto `target_position`, reusing the
//! along-axis drive of the [`PrismaticDriveJoint`](super::PrismaticDriveJoint).
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer: slot
//! `3 * k` for the point-on-line weld, slot `3 * k + 1` for the axis alignment,
//! and slot `3 * k + 2` for the drive, where `k` is the joint's index in the
//! colour-reordered list. The buffer is therefore `3 * joints.len()` long and is
//! reset to zero at the start of every substep.
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
//! (`GpuCylindricalDriveJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout and read the same per-substep snapshot buffers for the
//! damping term. Neither the alignment, weld, nor drive takes an inverse cosine,
//! so the two paths need only match in finite-precision reassociation; the tight
//! parity tolerance bounds the result.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge, the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, and the bilateral along-axis drive shared
//! with the prismatic drive, with their substep compliant-and-damped `XPBD`
//! handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"; Macklin
//! et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"),
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::cylindrical_drive::CylindricalDriveJoint;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::stepper::{predict, recover_velocities, snapshot};
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the cylindrical drive joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment, point-on-line, and drive constraints
/// every substep. The joint set is coloured once up front so same-batch joints
/// write disjoint movable bodies; the batches are then solved in order
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
pub fn cpu_solve_joints_cylindrical_drive(
    state: &mut RigidBodyState,
    joints: &[CylindricalDriveJoint],
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

    // Three Lagrange multipliers per joint: slot `3k` for the point-on-line
    // weld, slot `3k + 1` for the axis alignment, slot `3k + 2` for the drive.
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

/// Projects one cylindrical drive joint for a single sweep: the axis-alignment
/// correction first (accumulating `lambda[1]`), then the point-on-line
/// correction (accumulating `lambda[0]`), then the bilateral drive correction
/// (accumulating `lambda[2]`), each applied directly to the two bodies'
/// transforms.
fn solve_one(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    joint: &CylindricalDriveJoint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_point_on_line(state, joint, h, &mut lambda[0]);
    solve_drive(
        state,
        prev_positions,
        prev_orientations,
        joint,
        h,
        &mut lambda[2],
    );
}

/// Drives body `b`'s axis parallel to body `a`'s. With `u_a = rotate(q_a,
/// axis_a)` and `u_b = rotate(q_b, axis_b)` the two unit axes, the alignment
/// error is their cross product `delta = u_a x u_b` — its magnitude is
/// `sin(angle)` between them and its direction is the rotation axis that drives
/// them parallel. The correction rotates about `n = delta / |delta|`, with the
/// gradient `+n` on body `a` and `-n` on body `b`, leaving the spin about the
/// shared axis free.
fn solve_axis_alignment(
    state: &mut RigidBodyState,
    joint: &CylindricalDriveJoint,
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

    state.orientations[a] = apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, -world_inv_inertia_apply(q_b, ii_b, p));
}

/// Pins the two world-space anchors to a common line. With `dx` the anchor
/// separation and `axis_w = rotate(q_a, axis_a)` the world-space shared axis,
/// only the perpendicular component `dx_perp = dx - axis_w * dot(dx, axis_w)` is
/// cancelled; the along-axis component is left free for the drive to command, so
/// the two anchors share a line rather than a point.
fn solve_point_on_line(
    state: &mut RigidBodyState,
    joint: &CylindricalDriveJoint,
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

    // World-space slide axis on body `a`; the free translational direction.
    let axis = rotate(q_a, joint.axis_a);
    let axis_len = axis.length();
    if axis_len < EPSILON {
        return;
    }
    let axis_w = axis / axis_len;

    // Remove the along-axis component: only the perpendicular separation is
    // cancelled, leaving the slide free.
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
    joint: &CylindricalDriveJoint,
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
    use bevy_math::ops;

    /// A static rod at the origin (body `0`) along `+Y` and a dynamic sleeve
    /// (body `1`) coincident with it, free to slide and spin along `+Y`. Body
    /// `a` of the joint is the sleeve, so the signed slide position
    /// `s = (p_sleeve - p_rod) . axis` equals the sleeve's height.
    fn rod_and_sleeve() -> RigidBodyState {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: rod
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 1: sleeve
        state
    }

    /// Signed slide position of a joint along its world axis.
    fn slide_position(state: &RigidBodyState, joint: &CylindricalDriveJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let axis_w = rotate(state.orientations[a], joint.axis_a).normalize();
        let r_a = rotate(state.orientations[a], joint.anchor_a);
        let r_b = rotate(state.orientations[b], joint.anchor_b);
        let dx = (state.positions[a] + r_a) - (state.positions[b] + r_b);
        dx.dot(axis_w)
    }

    /// Angle (radians) between the two world-space axes — zero when the
    /// alignment constraint is satisfied.
    fn axis_angle(state: &RigidBodyState, joint: &CylindricalDriveJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let u_a = rotate(state.orientations[a], joint.axis_a).normalize();
        let u_b = rotate(state.orientations[b], joint.axis_b).normalize();
        ops::asin(u_a.cross(u_b).length().clamp(-1.0, 1.0))
    }

    #[test]
    fn rigid_servo_holds_against_gravity() {
        // A rigid servo commanded to the start position must hold the sleeve at
        // s = 0 against gravity dragging it down the +Y axis.
        let joint =
            CylindricalDriveJoint::servo(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0);
        let mut state = rod_and_sleeve();
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_cylindrical_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        assert!(
            slide_position(&state, &joint).abs() < 1.0e-2,
            "rigid servo drifted off target: s = {}",
            slide_position(&state, &joint)
        );
    }

    #[test]
    fn rigid_servo_drives_to_a_nonzero_target() {
        // A rigid servo commanded to +0.5 must pull the sleeve up to that stroke
        // and hold it there, no gravity.
        let joint =
            CylindricalDriveJoint::servo(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.5);
        let mut state = rod_and_sleeve();
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_cylindrical_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        assert!(
            (slide_position(&state, &joint) - 0.5).abs() < 1.0e-2,
            "servo failed to reach target: s = {}",
            slide_position(&state, &joint)
        );
    }

    #[test]
    fn spin_about_the_axis_stays_free_under_drive() {
        // The sleeve carries an initial spin about the axis while a rigid servo
        // holds the slide at the start: the drive must not damp the spin, which
        // the cylindrical base leaves free.
        let joint =
            CylindricalDriveJoint::servo(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0);
        let mut state = rod_and_sleeve();
        state.angular_velocities[1] = Vec3::new(0.0, 2.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..30 {
            cpu_solve_joints_cylindrical_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        assert!(
            state.angular_velocities[1].y > 1.9,
            "spin about the axis was damped by the drive: {}",
            state.angular_velocities[1].y
        );
        assert!(
            axis_angle(&state, &joint) < 1.0e-3,
            "spin tilted the axis off parallel"
        );
    }

    #[test]
    fn soft_spring_pulls_toward_target_without_overshooting_rigidly() {
        // A soft spring-damper drive commanded to +0.3 should approach the target
        // monotonically-ish and settle near it, not snap in one substep.
        let joint = CylindricalDriveJoint::spring(
            1,
            0,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            0.3,
            200.0,
            20.0,
        );
        let mut state = rod_and_sleeve();
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        let start = slide_position(&state, &joint);
        assert!(start.abs() < 1.0e-6, "setup slide position wrong: {start}");
        for _ in 0..240 {
            cpu_solve_joints_cylindrical_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let settled = slide_position(&state, &joint);
        assert!(
            (settled - 0.3).abs() < 2.0e-2,
            "soft spring failed to settle near target: s = {settled}"
        );
    }

    #[test]
    fn alignment_holds_while_driving() {
        // A sleeve starting 20 degrees off parallel while a rigid servo drives
        // the slide: the alignment constraint must still pull the axis parallel.
        let joint =
            CylindricalDriveJoint::servo(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.25);
        let mut state = rod_and_sleeve();
        let tilt = std::f32::consts::FRAC_PI_6 * 0.666;
        state.orientations[1] = Quat::from_axis_angle(Vec3::X, tilt);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_cylindrical_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        assert!(
            axis_angle(&state, &joint) < 2.0e-2,
            "alignment failed under drive: angle = {}",
            axis_angle(&state, &joint)
        );
        assert!(
            (slide_position(&state, &joint) - 0.25).abs() < 2.0e-2,
            "drive failed to reach target while aligning: s = {}",
            slide_position(&state, &joint)
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum_with_zero_target() {
        // Zero gravity, equal masses, both bodies given the same initial
        // velocity, drive target zero: the joint's internal corrections are equal
        // and opposite, so total linear momentum must be conserved.
        let joint = CylindricalDriveJoint::servo(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            0.0,
        );
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
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_cylindrical_drive(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
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
        cpu_solve_joints_cylindrical_drive(&mut state, &[], &integrator, &config, 1.0 / 60.0)
            .unwrap();
        assert_eq!(state.positions[0], before);
    }
}
