//! Authoritative `CPU` golden reference for the configurable six-degree-of-
//! freedom (`D6`) joint stepper *with per-axis drives*.
//!
//! [`cpu_solve_joints_d6_driven`] is the driven counterpart of
//! [`cpu_solve_joints_d6`](super::d6_cpu::cpu_solve_joints_d6): the same full
//! stepper and substep schedule, but every joint additionally carries a
//! [`D6DriveSet`] of six per-axis actuators (a spring-damper on each of the
//! three linear and three angular degrees of freedom), and each substep projects
//! the passive constraints first and the drives second.
//!
//! # The schedule
//!
//! Within each integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's twelve `XPBD` Lagrange multipliers,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times — the six passive axes first (delegated verbatim to the passive
//!    golden's [`solve_passive`](super::d6_cpu::solve_passive)), then the six
//!    drives — walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # The drive update
//!
//! Each [`D6Drive`] is a parallel spring (gain `stiffness`, toward
//! `target_position`) and damper (gain `damping`, toward `target_velocity`) on
//! one scalar degree of freedom. A drive with positive stiffness is projected as
//! the compliant-and-damped `XPBD` equality `C = coord - target_position` with
//! compliance `1 / stiffness`, regularisation `alpha_tilde = compliance / h^2`,
//! and the Macklin velocity-damping term that bleeds the axis rate onto
//! `target_velocity`:
//!
//! ```text
//! gamma    = compliance * damping / h
//! c_vel    = dv - target_velocity * h
//! d_lambda = (-c_pos - alpha_tilde * lambda - gamma * c_vel)
//!            / ((1 + gamma) * w + alpha_tilde)
//! ```
//!
//! where `dv` is the relative displacement along the axis since the substep
//! snapshot and `w` the generalized effective inverse mass about the axis (the
//! same `w` the matching passive axis uses). Following Macklin et al. the
//! damping is coupled to the compliance through `gamma`, so a rigid spring
//! (`stiffness -> infinity`, `compliance -> 0`) carries a vanishing damping term.
//!
//! A drive with zero stiffness but positive damping is a *pure velocity motor*:
//! the position servo disappears and the constraint becomes `C = dv -
//! target_velocity * h`, regularised by `alpha_tilde = 1 / (h * damping)`. That
//! regularisation is exactly the `stiffness -> 0` limit of the position servo
//! above, so the two branches meet with no discontinuity as the spring is
//! softened away. A drive with both gains zero is inert and skipped.
//!
//! # Composition with the passive flags
//!
//! A drive is independent of its axis's [`D6Motion`](super::d6::D6Motion) flag
//! and the two compose: a `Free` axis with an active drive is a pure actuator; a
//! `Limited` axis with a drive is an actuator that still respects its mechanical
//! stop; a rigid `Locked` axis's weld, projected first and with zero compliance,
//! dominates any soft drive on the same axis, so driving a locked axis is a
//! no-op in practice. The stepper applies every drive unconditionally; the
//! passive projection simply wins where it is stiffer.
//!
//! # Lambda layout
//!
//! Each joint owns twelve multipliers in the shared `lambda` buffer: the first
//! six (`12k + 0 .. 12k + 6`) are the passive axes laid out exactly as the
//! passive golden (linear `x` / `y` / `z`, twist, swing1, swing2), and the next
//! six (`12k + 6 .. 12k + 12`) are the drives in the same axis order. The buffer
//! is `12 * joints.len()` long and is reset to zero at the start of every
//! substep.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin and its shader,
//! which walk the identical reordered joint list and colour-batch order with the
//! identical twelve-slot multiplier layout and read the same per-substep
//! snapshot buffers for the damping term. Angles are taken with
//! [`bevy_math::ops::atan2`] rather than `f32::atan2` so the `CPU` reference and
//! the shader share one transcendental path.
//!
//! Provenance: the per-axis spring-damper actuator of a general-purpose
//! constraint (Unreal Engine's `FConstraintDrive`, `PhysX`'s `PxD6JointDrive`),
//! realised as a compliant, velocity-damped `XPBD` constraint with Macklin-style
//! damping regularisation (Macklin et al., "XPBD: Position-Based Simulation of
//! Compliant Constrained Dynamics") over the passive `D6` constraints of Müller
//! et al. and the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code: only the public
//! spring-damper drive semantics are mirrored.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::d6::D6Joint;
use super::d6_cpu::{frame_world, solve_passive};
use super::d6_drive::{D6Drive, D6DriveSet};
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, quat_rotate, rotate,
    world_inv_inertia_apply, EPSILON,
};
use super::stepper::{predict, recover_velocities, snapshot};
use bevy_math::ops;
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the configurable `D6` joints in `joints`, each
/// actuated by the matching drive set in `drives`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest of
/// the crate (gravity, substep count, and damping from `integrator`), then every
/// substep projects each joint's six passive axes and six drives. The joint set
/// is coloured once up front so same-batch joints write disjoint movable bodies;
/// the drive sets are permuted by the same colouring so they stay aligned with
/// their joints. The batches are then solved in order
/// [`position_iterations`](JointSolverConfig::position_iterations) times per
/// substep.
///
/// # Errors
///
/// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree in
/// length, if `drives` and `joints` differ in length, or if a joint references a
/// body outside the state, and [`RigidError::TooManyJointBatches`] if the joint
/// graph needs more colour batches than the colouring supports. Returns `Ok(())`
/// with the state untouched when there is nothing to do (`dt <= 0`, no bodies,
/// or no joints).
pub fn cpu_solve_joints_d6_driven(
    state: &mut RigidBodyState,
    joints: &[D6Joint],
    drives: &[D6DriveSet],
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
    if drives.len() != joints.len() {
        return Err(RigidError::InconsistentState {
            reason: "each D6 joint needs exactly one drive set",
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
    let ordered_drives = colouring.reorder(drives);
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

    // Twelve multipliers per joint: `[12k + 0..6]` the six passive axes (linear
    // x/y/z, twist, swing1, swing2) laid out exactly as the passive golden, and
    // `[12k + 6..12]` the six drives in the same axis order. Reset every substep.
    let mut lambda = vec![0.0f32; 12 * ordered.len()];

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
                    let slots = &mut lambda[12 * k..12 * k + 12];
                    let (passive, drive) = slots.split_at_mut(6);
                    // Passive constraints first, then the drives, so each drive
                    // servos against a freshly projected passive configuration.
                    solve_passive(state, &ordered[k], h, passive);
                    solve_drives(
                        state,
                        &prev_positions,
                        &prev_orientations,
                        &ordered[k],
                        &ordered_drives[k],
                        h,
                        drive,
                    );
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one joint's six drives for a single sweep, in the same axis order as
/// the passive sweep — swing1 (`lambda[4]`), swing2 (`lambda[5]`), the twist
/// (`lambda[3]`), then the three linear axes (`lambda[0..3]`) — so the angular
/// servos act before the positional ones.
fn solve_drives(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    joint: &D6Joint,
    drives: &D6DriveSet,
    h: f32,
    lambda: &mut [f32],
) {
    drive_swing(
        state,
        prev_orientations,
        joint,
        drives.swing1,
        h,
        &mut lambda[4],
        0,
    );
    drive_swing(
        state,
        prev_orientations,
        joint,
        drives.swing2,
        h,
        &mut lambda[5],
        1,
    );
    drive_twist(
        state,
        prev_orientations,
        joint,
        drives.twist,
        h,
        &mut lambda[3],
    );
    drive_linear(
        state,
        prev_positions,
        prev_orientations,
        joint,
        drives.linear_x,
        h,
        &mut lambda[0],
        0,
    );
    drive_linear(
        state,
        prev_positions,
        prev_orientations,
        joint,
        drives.linear_y,
        h,
        &mut lambda[1],
        1,
    );
    drive_linear(
        state,
        prev_positions,
        prev_orientations,
        joint,
        drives.linear_z,
        h,
        &mut lambda[2],
        2,
    );
}

/// Computes the drive multiplier increment for one axis and accumulates it into
/// `lambda`, returning the impulse to apply along the axis.
///
/// `w` is the generalized effective inverse mass about the axis, `coord` the
/// current axis coordinate (metres of separation or radians of rotation), and
/// `dv` the relative displacement along the axis since the substep snapshot. A
/// positive-stiffness drive is the compliant-and-damped position servo `C =
/// coord - target_position`; a zero-stiffness, positive-damping drive is the
/// pure velocity motor `C = dv - target_velocity * h`, whose regularisation `1 /
/// (h * damping)` is the continuous `stiffness -> 0` limit of the servo. An
/// inert drive (both gains zero) or a singular axis (`w` below `EPSILON`)
/// contributes nothing.
fn drive_delta(w: f32, coord: f32, dv: f32, drive: D6Drive, h: f32, lambda: &mut f32) -> f32 {
    if !drive.is_active() || w < EPSILON {
        return 0.0;
    }

    // Relative displacement the drive wants over this substep to hold the
    // commanded rate; the damper resists deviation from it.
    let c_vel = dv - drive.target_velocity * h;

    let d_lambda = if drive.stiffness > 0.0 {
        let compliance = 1.0 / drive.stiffness;
        let alpha_tilde = compliance / (h * h);
        // XPBD damping scale gamma = compliance * damping / h (Macklin et al.);
        // it couples the damper to the spring and vanishes with the compliance.
        let gamma = if drive.damping > 0.0 {
            compliance * drive.damping / h
        } else {
            0.0
        };
        let c_pos = coord - drive.target_position;
        (-c_pos - alpha_tilde * *lambda - gamma * c_vel) / ((1.0 + gamma) * w + alpha_tilde)
    } else {
        // Pure velocity motor: the stiffness -> 0 limit of the servo above, with
        // the damper setting the regularisation 1 / (h * damping).
        let alpha_tilde = 1.0 / (h * drive.damping);
        (-c_vel - alpha_tilde * *lambda) / (w + alpha_tilde)
    };

    *lambda += d_lambda;
    d_lambda
}

/// Relative rotation of a body since the substep snapshot, as a rotation vector:
/// twice the imaginary part of the delta quaternion `orientation *
/// conj(prev_orientation)`, hemisphere-corrected so the shortest arc is taken.
/// An angular drive's damping term dots this with its correction axis to read
/// the relative angular rate about the axis.
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

/// Servos the twist angle onto its drive, about the world twist axis `t =
/// rotate(frame_a, X)`. The angle is measured exactly as the passive twist pass
/// measures it (the signed angle from body `a`'s frame `y` axis to body `b`'s,
/// both projected perpendicular to `t`), and the correction is a rotation about
/// `t` with the antisymmetric gradient `-t` on `a` and `+t` on `b`.
fn drive_twist(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &D6Joint,
    drive: D6Drive,
    h: f32,
    lambda: &mut f32,
) {
    if !drive.is_active() {
        return;
    }

    let a = joint.body_a as usize;
    let b = joint.body_b as usize;
    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let frame_a = frame_world(q_a, joint.basis_a);
    let frame_b = frame_world(q_b, joint.basis_b);
    let t_raw = quat_rotate(frame_a, Vec3::X);
    let t_len = t_raw.length();
    if t_len < EPSILON {
        return;
    }
    let t = t_raw / t_len;

    let ref_a = quat_rotate(frame_a, Vec3::Y);
    let ref_b = quat_rotate(frame_b, Vec3::Y);
    let pa = ref_a - t * t.dot(ref_a);
    let pb = ref_b - t * t.dot(ref_b);
    let la = pa.length();
    let lb = pb.length();
    if la < EPSILON || lb < EPSILON {
        return;
    }
    let pa_n = pa / la;
    let pb_n = pb / lb;
    let sin_theta = pa_n.cross(pb_n).dot(t);
    let cos_theta = pa_n.dot(pb_n);
    let theta = ops::atan2(sin_theta, cos_theta);

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = t.dot(world_inv_inertia_apply(q_a, ii_a, t));
    let w_b = t.dot(world_inv_inertia_apply(q_b, ii_b, t));
    let w = w_a + w_b;

    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = t.dot(angvec_b - angvec_a);

    let d_lambda = drive_delta(w, theta, dv, drive, h, lambda);
    let p = t * d_lambda;
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Servos swing1 (`which == 0`, tilt toward the frame `y` axis, corrected about
/// `+e_z`) or swing2 (`which == 1`, tilt toward the frame `z` axis, corrected
/// about `-e_y`) onto its drive. The angle is measured exactly as the passive
/// swing pass measures it, from body `b`'s twist axis decomposed in the frame
/// `(t, e_y, e_z)`.
fn drive_swing(
    state: &mut RigidBodyState,
    prev_orientations: &[Quat],
    joint: &D6Joint,
    drive: D6Drive,
    h: f32,
    lambda: &mut f32,
    which: usize,
) {
    if !drive.is_active() {
        return;
    }

    let a = joint.body_a as usize;
    let b = joint.body_b as usize;
    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let frame_a = frame_world(q_a, joint.basis_a);
    let frame_b = frame_world(q_b, joint.basis_b);
    let t_raw = quat_rotate(frame_a, Vec3::X);
    let e_y_raw = quat_rotate(frame_a, Vec3::Y);
    let e_z_raw = quat_rotate(frame_a, Vec3::Z);
    let t_len = t_raw.length();
    if t_len < EPSILON {
        return;
    }
    let t = t_raw / t_len;
    let e_y = e_y_raw.normalize_or_zero();
    let e_z = e_z_raw.normalize_or_zero();
    if e_y.length_squared() < 0.5 || e_z.length_squared() < 0.5 {
        return;
    }

    let u_b_raw = quat_rotate(frame_b, Vec3::X);
    let u_b_len = u_b_raw.length();
    if u_b_len < EPSILON {
        return;
    }
    let u_b = u_b_raw / u_b_len;

    let x_c = u_b.dot(t);
    let (angle, n) = if which == 0 {
        let y_c = u_b.dot(e_y);
        (ops::atan2(y_c, x_c), e_z)
    } else {
        let z_c = u_b.dot(e_z);
        (ops::atan2(z_c, x_c), -e_y)
    };

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = n.dot(world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = n.dot(world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;

    let angvec_a = angular_displacement(q_a, prev_orientations[a]);
    let angvec_b = angular_displacement(q_b, prev_orientations[b]);
    let dv = n.dot(angvec_b - angvec_a);

    let d_lambda = drive_delta(w, angle, dv, drive, h, lambda);
    let p = n * d_lambda;
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Servos the linear axis `axis` (`0` = frame `x`, `1` = `y`, `2` = `z`) onto
/// its drive. The signed anchor separation along the world frame axis is the
/// coordinate; the correction is an impulse along the frame axis applied at the
/// two anchors, exactly the passive linear pass's gradient.
#[expect(
    clippy::too_many_arguments,
    reason = "a single linear drive projection needs both bodies, both cached snapshots, the joint, the drive, the step, its multiplier, and the axis index"
)]
fn drive_linear(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    joint: &D6Joint,
    drive: D6Drive,
    h: f32,
    lambda: &mut f32,
    axis: usize,
) {
    if !drive.is_active() {
        return;
    }

    let a = joint.body_a as usize;
    let b = joint.body_b as usize;
    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let local = [Vec3::X, Vec3::Y, Vec3::Z][axis];
    let frame_a = frame_world(q_a, joint.basis_a);
    let axis_w = quat_rotate(frame_a, local);
    let axis_len = axis_w.length();
    if axis_len < EPSILON {
        return;
    }
    let n = axis_w / axis_len;

    let r_a = rotate(q_a, joint.anchor_a);
    let r_b = rotate(q_b, joint.anchor_b);
    let point_a = state.positions[a] + r_a;
    let point_b = state.positions[b] + r_b;
    let s = (point_a - point_b).dot(n);

    // Relative displacement of the two material anchor points since the snapshot,
    // projected onto the current frame axis, so the damper reads the along-axis
    // closing rate.
    let prev_point_a = prev_positions[a] + rotate(prev_orientations[a], joint.anchor_a);
    let prev_point_b = prev_positions[b] + rotate(prev_orientations[b], joint.anchor_b);
    let dv = ((point_a - prev_point_a) - (point_b - prev_point_b)).dot(n);

    let inv_m_a = state.inverse_masses[a];
    let inv_m_b = state.inverse_masses[b];
    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let rn_a = r_a.cross(n);
    let rn_b = r_b.cross(n);
    let w_a = inv_m_a + rn_a.dot(world_inv_inertia_apply(q_a, ii_a, rn_a));
    let w_b = inv_m_b + rn_b.dot(world_inv_inertia_apply(q_b, ii_b, rn_b));
    let w = w_a + w_b;

    let d_lambda = drive_delta(w, s, dv, drive, h, lambda);
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
    use super::super::d6::D6Motion;
    use super::super::d6_cpu::cpu_solve_joints_d6;
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    const LOCKED3: [D6Motion; 3] = [D6Motion::Locked; 3];
    const FREE3: [D6Motion; 3] = [D6Motion::Free; 3];
    const OFF3: [D6Drive; 3] = [D6Drive::OFF; 3];

    /// Signed anchor separation vector `p_a - p_b` in world space.
    fn anchor_delta(state: &RigidBodyState, joint: &D6Joint) -> Vec3 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let pa = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let pb = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        pa - pb
    }

    /// The two world-space joint frames of `joint` in `state`.
    fn world_frames(state: &RigidBodyState, joint: &D6Joint) -> ([f32; 4], [f32; 4]) {
        let fa = frame_world(state.orientations[joint.body_a as usize], joint.basis_a);
        let fb = frame_world(state.orientations[joint.body_b as usize], joint.basis_b);
        (fa, fb)
    }

    /// Twist angle (radians), measured exactly as the twist drive measures it.
    fn twist_angle(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let (fa, fb) = world_frames(state, joint);
        let t = quat_rotate(fa, Vec3::X).normalize();
        let ra = quat_rotate(fa, Vec3::Y);
        let rb = quat_rotate(fb, Vec3::Y);
        let pa = (ra - t * t.dot(ra)).normalize();
        let pb = (rb - t * t.dot(rb)).normalize();
        ops::atan2(pa.cross(pb).dot(t), pa.dot(pb))
    }

    /// Swing1 angle (radians), measured exactly as the swing1 drive measures it.
    fn swing1_angle(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let (fa, fb) = world_frames(state, joint);
        let t = quat_rotate(fa, Vec3::X).normalize();
        let e_y = quat_rotate(fa, Vec3::Y).normalize();
        let u_b = quat_rotate(fb, Vec3::X).normalize();
        ops::atan2(u_b.dot(e_y), u_b.dot(t))
    }

    /// Swing2 angle (radians), measured exactly as the swing2 drive measures it.
    fn swing2_angle(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let (fa, fb) = world_frames(state, joint);
        let t = quat_rotate(fa, Vec3::X).normalize();
        let e_z = quat_rotate(fa, Vec3::Z).normalize();
        let u_b = quat_rotate(fb, Vec3::X).normalize();
        ops::atan2(u_b.dot(e_z), u_b.dot(t))
    }

    /// A dynamic unit limb socketed off a static pivot with identity joint
    /// frames, so the frame axes are the world axes: body `a` (index 0) is the
    /// static socket at the origin and body `b` (index 1) is the dynamic unit
    /// limb above it, anchored back through its `-Y` end to the socket. The two
    /// anchors coincide at the origin, so every linear coordinate starts at zero
    /// and every joint angle starts at zero.
    fn socketed_limb(linear: [D6Motion; 3], angular: [D6Motion; 3]) -> (RigidBodyState, D6Joint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static socket
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: dynamic limb
        let joint = D6Joint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            Quat::IDENTITY,
            linear,
            angular,
            0.0,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        );
        (state, joint)
    }

    /// A free pair of equal unit masses straddling the origin along the world
    /// `x` axis, coupled by a `D6` joint whose anchors sit at each body's inner
    /// end so the linear `x` coordinate is their separation. Both bodies are
    /// dynamic, so any internal drive must conserve the pair's linear momentum.
    fn free_pair(linear: [D6Motion; 3]) -> (RigidBodyState, D6Joint) {
        let mut state = RigidBodyState::new();
        state.push(
            Vec3::new(-1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 0
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1
        let joint = D6Joint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Quat::IDENTITY,
            Quat::IDENTITY,
            linear,
            LOCKED3,
            0.0,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        );
        (state, joint)
    }

    fn integrator(gravity: Vec3) -> IntegratorConfig {
        IntegratorConfig::new(gravity, 8, 0.0, 0.0)
    }

    const DT: f32 = 1.0 / 60.0;

    #[test]
    fn inert_drive_matches_passive() {
        // A rest (all-inert) drive set must leave the passive stepper's output
        // untouched: the driven golden with no active drive is bit-for-bit the
        // passive golden on a representative mixed-freedom joint under gravity.
        let angular = [D6Motion::Free, D6Motion::Limited, D6Motion::Locked];
        let linear = [D6Motion::Limited, D6Motion::Free, D6Motion::Locked];
        let make = || {
            let mut state = RigidBodyState::new();
            state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
            state.push(
                Vec3::new(0.0, 1.0, 0.0),
                Quat::IDENTITY,
                1.0,
                Vec3::splat(1.0),
            );
            let joint = D6Joint::new(
                0,
                1,
                Vec3::ZERO,
                Vec3::new(0.0, -1.0, 0.0),
                Quat::IDENTITY,
                Quat::IDENTITY,
                linear,
                angular,
                0.3,
                (-FRAC_PI_2, FRAC_PI_2),
                (0.4, 0.4),
                (0.0, 0.0, 0.0),
            );
            (state, joint)
        };

        let (mut passive_state, joint) = make();
        let (mut driven_state, _) = make();
        let drives = [D6DriveSet::rest()];
        let config = JointSolverConfig::new(6);
        let integ = integrator(Vec3::new(2.0, -9.81, 1.0));

        for _ in 0..90 {
            cpu_solve_joints_d6(&mut passive_state, &[joint], &integ, &config, DT).unwrap();
            cpu_solve_joints_d6_driven(&mut driven_state, &[joint], &drives, &integ, &config, DT)
                .unwrap();
        }

        for i in 0..2 {
            assert_eq!(
                passive_state.positions[i], driven_state.positions[i],
                "body {i} position diverged from the passive golden"
            );
            assert_eq!(
                passive_state.orientations[i], driven_state.orientations[i],
                "body {i} orientation diverged from the passive golden"
            );
            assert_eq!(
                passive_state.linear_velocities[i], driven_state.linear_velocities[i],
                "body {i} linear velocity diverged from the passive golden"
            );
            assert_eq!(
                passive_state.angular_velocities[i], driven_state.angular_velocities[i],
                "body {i} angular velocity diverged from the passive golden"
            );
        }
    }

    #[test]
    fn linear_x_position_drive_converges() {
        // A stiff position servo on the free linear x axis must pull the anchor
        // separation to its commanded target and hold it there.
        let linear = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let (mut state, joint) = socketed_limb(linear, LOCKED3);
        let target = 0.35;
        let drives = [D6DriveSet::new(
            [D6Drive::position(5.0e3, target), D6Drive::OFF, D6Drive::OFF],
            OFF3,
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..120 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let s = anchor_delta(&state, &joint).x;
        assert!(
            (s - target).abs() < 2.0e-2,
            "linear x drive settled at {s}, want {target}"
        );
    }

    #[test]
    fn twist_position_drive_converges() {
        // A stiff twist servo must rotate the free-twisting limb to its target
        // angle about the twist axis.
        let angular = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let (mut state, joint) = socketed_limb(LOCKED3, angular);
        let target = 0.5;
        let drives = [D6DriveSet::new(
            OFF3,
            [D6Drive::position(5.0e3, target), D6Drive::OFF, D6Drive::OFF],
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..120 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let theta = twist_angle(&state, &joint);
        assert!(
            (theta - target).abs() < 2.0e-2,
            "twist drive settled at {theta}, want {target}"
        );
    }

    #[test]
    fn swing1_position_drive_converges() {
        // A stiff swing1 servo must tilt the limb to its target swing1 angle
        // while twist and swing2 stay welded shut.
        let angular = [D6Motion::Locked, D6Motion::Free, D6Motion::Locked];
        let (mut state, joint) = socketed_limb(LOCKED3, angular);
        let target = 0.3;
        let drives = [D6DriveSet::new(
            OFF3,
            [D6Drive::OFF, D6Drive::position(5.0e3, target), D6Drive::OFF],
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..120 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let angle = swing1_angle(&state, &joint);
        assert!(
            (angle - target).abs() < 3.0e-2,
            "swing1 drive settled at {angle}, want {target}"
        );
    }

    #[test]
    fn swing2_position_drive_converges() {
        // A stiff swing2 servo must tilt the limb to its target swing2 angle
        // while twist and swing1 stay welded shut.
        let angular = [D6Motion::Locked, D6Motion::Locked, D6Motion::Free];
        let (mut state, joint) = socketed_limb(LOCKED3, angular);
        let target = 0.3;
        let drives = [D6DriveSet::new(
            OFF3,
            [D6Drive::OFF, D6Drive::OFF, D6Drive::position(5.0e3, target)],
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..120 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let angle = swing2_angle(&state, &joint);
        assert!(
            (angle - target).abs() < 3.0e-2,
            "swing2 drive settled at {angle}, want {target}"
        );
    }

    #[test]
    fn twist_velocity_motor_reaches_rate() {
        // A pure velocity motor on the free twist axis must spin the limb up to
        // its commanded angular rate and then hold it there.
        let angular = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let (mut state, joint) = socketed_limb(LOCKED3, angular);
        let target = 2.0;
        let drives = [D6DriveSet::new(
            OFF3,
            [D6Drive::velocity(2.0e2, target), D6Drive::OFF, D6Drive::OFF],
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..150 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let rate = state.angular_velocities[1].x;
        assert!(
            (rate - target).abs() < 0.3,
            "velocity motor settled at rate {rate}, want {target}"
        );
    }

    #[test]
    fn stiffer_spring_sags_less() {
        // Under a steady along-axis load the position servo sags from its
        // target by load / stiffness, so a stiffer spring must sag strictly
        // less than a softer one.
        let linear = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let gravity = Vec3::new(-6.0, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let integ = integrator(gravity);

        let sag_for = |stiffness: f32| {
            let (mut state, joint) = socketed_limb(linear, LOCKED3);
            let drives = [D6DriveSet::new(
                [
                    D6Drive::position(stiffness, 0.0),
                    D6Drive::OFF,
                    D6Drive::OFF,
                ],
                OFF3,
            )];
            for _ in 0..180 {
                cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT)
                    .unwrap();
            }
            anchor_delta(&state, &joint).x.abs()
        };

        let soft = sag_for(1.0e2);
        let stiff = sag_for(1.0e3);
        assert!(
            stiff < soft,
            "stiffer spring sagged {stiff}, softer sagged {soft}"
        );
        assert!(soft > 0.0, "soft spring did not sag under load");
    }

    #[test]
    fn locked_axis_beats_drive() {
        // A rigid weld on the twist axis is projected first with zero
        // compliance, so even an aggressive twist servo cannot drag the locked
        // axis anywhere near its distant target.
        let (mut state, joint) = socketed_limb(LOCKED3, LOCKED3);
        let drives = [D6DriveSet::new(
            OFF3,
            [D6Drive::position(1.0e4, 1.0), D6Drive::OFF, D6Drive::OFF],
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..90 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let theta = twist_angle(&state, &joint);
        assert!(
            theta.abs() < 0.15,
            "weld lost to the drive: twist reached {theta}"
        );
    }

    #[test]
    fn limited_axis_respects_stop() {
        // A twist servo commanding past the mechanical stop must drive the axis
        // up to, but not through, the rigid upper limit.
        let angular = [D6Motion::Limited, D6Motion::Locked, D6Motion::Locked];
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let upper = 0.25;
        let joint = D6Joint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            Quat::IDENTITY,
            LOCKED3,
            angular,
            0.0,
            (-upper, upper),
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        );
        let drives = [D6DriveSet::new(
            OFF3,
            [D6Drive::position(5.0e3, 1.0), D6Drive::OFF, D6Drive::OFF],
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);

        for _ in 0..120 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
        }
        let theta = twist_angle(&state, &joint);
        assert!(
            theta > 0.1,
            "drive failed to push the limited axis toward its stop: {theta}"
        );
        assert!(
            theta < upper + 3.0e-2,
            "drive pushed through the mechanical stop: {theta} > {upper}"
        );
    }

    #[test]
    fn free_pair_linear_drive_conserves_momentum() {
        // A linear position drive between two free equal masses is an internal
        // force: it must pull the pair together without moving their centre of
        // mass or giving the system net linear momentum.
        let linear = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let (mut state, joint) = free_pair(linear);
        let drives = [D6DriveSet::new(
            [D6Drive::position(2.0e3, 0.5), D6Drive::OFF, D6Drive::OFF],
            OFF3,
        )];
        let config = JointSolverConfig::new(8);
        let integ = integrator(Vec3::ZERO);
        let com0 = 0.5 * (state.positions[0] + state.positions[1]);

        for _ in 0..90 {
            cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT).unwrap();
            let momentum = state.linear_velocities[0] + state.linear_velocities[1];
            assert!(
                momentum.length() < 1.0e-4,
                "internal drive created net momentum {momentum:?}"
            );
        }
        let com = 0.5 * (state.positions[0] + state.positions[1]);
        assert!(
            (com - com0).length() < 1.0e-3,
            "centre of mass drifted from {com0:?} to {com:?}"
        );
        let s = anchor_delta(&state, &joint).x;
        assert!(
            (s - 0.5).abs() < 3.0e-2,
            "pair did not reach the commanded separation: {s}"
        );
    }

    #[test]
    fn drive_count_mismatch_errors() {
        // Each joint needs exactly one drive set; a mismatched count is an
        // inconsistent-state error, not a silent partial solve.
        let (mut state, joint) = socketed_limb(FREE3, FREE3);
        let drives = [D6DriveSet::rest(), D6DriveSet::rest()];
        let config = JointSolverConfig::new(4);
        let integ = integrator(Vec3::ZERO);
        let err = cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT);
        assert!(matches!(err, Err(RigidError::InconsistentState { .. })));
    }

    #[test]
    fn body_out_of_range_errors() {
        // A joint that references a body outside the state is rejected before
        // any integration happens.
        let (mut state, _) = socketed_limb(FREE3, FREE3);
        let joint = D6Joint::new(
            0,
            9,
            Vec3::ZERO,
            Vec3::ZERO,
            Quat::IDENTITY,
            Quat::IDENTITY,
            FREE3,
            FREE3,
            0.0,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0, 0.0),
        );
        let drives = [D6DriveSet::rest()];
        let config = JointSolverConfig::new(4);
        let integ = integrator(Vec3::ZERO);
        let err = cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, DT);
        assert!(matches!(err, Err(RigidError::InconsistentState { .. })));
    }

    #[test]
    fn empty_state_is_noop() {
        // With no bodies there is nothing to solve and the call succeeds.
        let mut state = RigidBodyState::new();
        let config = JointSolverConfig::new(4);
        let integ = integrator(Vec3::new(0.0, -9.81, 0.0));
        cpu_solve_joints_d6_driven(&mut state, &[], &[], &integ, &config, DT).unwrap();
        assert!(state.is_empty());
    }

    #[test]
    fn no_joints_leaves_bodies_untouched() {
        // No joints means no drives: the state must come back exactly as it went
        // in, even under gravity.
        let (mut state, _) = socketed_limb(FREE3, FREE3);
        let before = state.positions[1];
        let config = JointSolverConfig::new(4);
        let integ = integrator(Vec3::new(0.0, -9.81, 0.0));
        cpu_solve_joints_d6_driven(&mut state, &[], &[], &integ, &config, DT).unwrap();
        assert_eq!(state.positions[1], before);
    }

    #[test]
    fn nonpositive_dt_is_noop() {
        // A zero or negative step advances nothing.
        let (mut state, joint) = socketed_limb(FREE3, FREE3);
        let before = state.positions[1];
        let drives = [D6DriveSet::new(
            [D6Drive::position(5.0e3, 1.0), D6Drive::OFF, D6Drive::OFF],
            OFF3,
        )];
        let config = JointSolverConfig::new(4);
        let integ = integrator(Vec3::new(0.0, -9.81, 0.0));
        cpu_solve_joints_d6_driven(&mut state, &[joint], &drives, &integ, &config, 0.0).unwrap();
        assert_eq!(state.positions[1], before);
    }
}
