//! Authoritative `CPU` golden reference for the configurable six-degree-of-
//! freedom (`D6`) joint stepper.
//!
//! [`cpu_solve_joints_d6`] is a *full stepper* with the identical substep
//! schedule as every other joint golden in the crate: the caller hands it the
//! current [`RigidBodyState`], the joint set, the shared [`IntegratorConfig`],
//! the joint-specific [`JointSolverConfig`], and the frame `dt`, and must **not**
//! integrate the bodies itself. Within each integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's six `XPBD` Lagrange multipliers,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # Per-axis freedoms
//!
//! A `D6` joint exposes six independent degrees of freedom between the two
//! bodies' joint frames: three linear (along the frame `x` / `y` / `z` axes) and
//! three angular (the twist about `x`, swing1 toward `y`, swing2 toward `z`).
//! Each axis is [`Locked`](D6Motion::Locked) (a compliant weld to zero),
//! [`Limited`](D6Motion::Limited) (free inside a symmetric band and clamped at
//! the bound), or [`Free`](D6Motion::Free) (no constraint). This single joint
//! therefore reproduces the whole lower-pair family: all six locked is a
//! [`FixedJoint`](super::FixedJoint); three linear locked plus two swings locked
//! leaving the twist free is a hinge; three linear locked leaving the three
//! angular free is a ball-and-socket; and so on.
//!
//! # The six constraints, projected angular-first
//!
//! Each sweep projects, in order, the two swings, the twist, then the three
//! linear axes — angular corrections first (as the elliptical cone-twist golden
//! does) so the positional weld closes last against a settled orientation. Each
//! projection is a single-axis `XPBD` correction identical in form to the ones
//! the hinge-limit, prismatic-limit, and cone-twist goldens already use; only
//! the measured error and the correction axis differ:
//!
//! * **Linear axis `i`** — the signed separation of the two world anchors along
//!   the world-space frame axis `e_i`, `s = (p_a - p_b) . e_i`. Locked drives
//!   `s` to zero (the compliant weld); limited clamps `s` into
//!   `[-linear_limit, +linear_limit]` with a free dead zone; free skips it. The
//!   correction is an impulse along `e_i` applied at the two anchors, exactly
//!   the prismatic limit's along-axis pass. Projecting all three locked linear
//!   axes in a sweep reproduces the point-to-point weld, because the three frame
//!   axes are orthonormal.
//! * **Twist** — the signed angle from body `a`'s frame `y` axis to body `b`'s,
//!   both projected onto the plane perpendicular to the world twist axis
//!   `t = rotate(q_a, basis_a . x)`, measured as
//!   `theta = atan2((p_a x p_b) . t, p_a . p_b)`. Locked drives `theta` to zero;
//!   limited clamps it into `[twist_min, twist_max]`; free skips it. The
//!   correction is a rotation about `t`, exactly the hinge limit's angular pass.
//! * **Swing1 / swing2** — the tilt of body `b`'s twist axis `u_b` toward body
//!   `a`'s frame `y` (swing1) and `z` (swing2) axes. With `u_b` decomposed in
//!   the orthonormal frame `(t, e_y, e_z)` as `(x_c, y_c, z_c)`, swing1 is
//!   `atan2(y_c, x_c)` (corrected by a rotation about `e_z`) and swing2 is
//!   `atan2(z_c, x_c)` (corrected by a rotation about `-e_y`). Locked drives the
//!   angle to zero; limited clamps it into `[-swing_limit, +swing_limit]`; free
//!   skips it. Treating the two swings as independent per-axis limits (a
//!   pyramidal swing region) is what lets every mixed `Locked` / `Limited` /
//!   `Free` combination be projected uniformly; the coupled *elliptical* swing
//!   cone is offered separately by
//!   [`EllipticalConeTwistJoint`](super::EllipticalConeTwistJoint). Both swing
//!   limits are assumed to stay below a right angle, as the measured
//!   `atan2(·, x_c)` angles do for every physical configuration of such a joint.
//!
//! # Lambda layout
//!
//! Each joint owns six Lagrange multipliers in the shared `lambda` buffer: slots
//! `6 * k + 0 .. 6 * k + 3` for the linear `x` / `y` / `z` axes, slot
//! `6 * k + 3` for the twist, `6 * k + 4` for swing1, and `6 * k + 5` for
//! swing2, where `k` is the joint's index in the colour-reordered list. The
//! buffer is `6 * joints.len()` long and is reset to zero at the start of every
//! substep. Axes whose motion is `Free`, or whose limit is not violated, leave
//! their multiplier untouched.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin and its shader,
//! which walk the identical reordered joint list and colour-batch order with the
//! identical six-slot multiplier layout. The transcendental angles are taken
//! with [`bevy_math::ops::atan2`] rather than the `f32` intrinsic so the `CPU`
//! reference and the device agree bit-for-bit on the branch an angle falls into.
//!
//! Provenance: the point-to-point and single-axis positional/angular `XPBD`
//! corrections (Müller et al.), the swing-twist decomposition about the joint
//! frame (as in `PhysX`'s `PxD6Joint` and Unreal's `FConstraintInstance`), over
//! the world-space inverse inertia and quaternion kinematics of Baraff & Witkin.
//! No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::d6::{D6Joint, D6Motion};
use super::math::{
    apply_rotation_delta, quat_array, quat_mul, quat_rotate, rotate, world_inv_inertia_apply,
    EPSILON,
};
use super::stepper::{predict, recover_velocities, snapshot};
use bevy_math::ops;
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the configurable `D6` joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's two swings, twist, and three linear axes every substep.
/// The joint set is coloured once up front so same-batch joints write disjoint
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
pub fn cpu_solve_joints_d6(
    state: &mut RigidBodyState,
    joints: &[D6Joint],
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

    // Six multipliers per joint: `[6k + 0..3]` linear x/y/z, `[6k + 3]` twist,
    // `[6k + 4]` swing1, `[6k + 5]` swing2. Reset to zero every substep.
    let mut lambda = vec![0.0f32; 6 * ordered.len()];

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
                    solve_passive(state, &ordered[k], h, &mut lambda[6 * k..6 * k + 6]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one `D6` joint for a single sweep: swing1 (`lambda[4]`), swing2
/// (`lambda[5]`), the twist (`lambda[3]`), then the three linear axes
/// (`lambda[0..3]`), each applied directly to the two bodies' transforms.
pub(super) fn solve_passive(
    state: &mut RigidBodyState,
    joint: &D6Joint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_swing(state, joint, h, &mut lambda[4], 0);
    solve_swing(state, joint, h, &mut lambda[5], 1);
    solve_twist(state, joint, h, &mut lambda[3]);
    solve_linear(state, joint, h, &mut lambda[0], 0);
    solve_linear(state, joint, h, &mut lambda[1], 1);
    solve_linear(state, joint, h, &mut lambda[2], 2);
}

/// The world-space joint-frame orientation of body `b` for `b in {a_index,
/// b_index}` — the body orientation composed with the body-local joint basis.
pub(super) fn frame_world(body_orientation: Quat, basis: Quat) -> [f32; 4] {
    quat_mul(quat_array(body_orientation), quat_array(basis))
}

/// Projects the linear axis `axis` (`0` = frame `x`, `1` = `y`, `2` = `z`) of a
/// `D6` joint. The signed anchor separation along the world-space frame axis is
/// driven to zero when the axis is [`Locked`](D6Motion::Locked), clamped into
/// `[-linear_limit, +linear_limit]` when [`Limited`](D6Motion::Limited), and
/// left untouched when [`Free`](D6Motion::Free). The correction is an impulse
/// along the frame axis applied at the two anchors — the prismatic limit's
/// along-axis pass restricted to one of the frame's orthonormal directions.
fn solve_linear(
    state: &mut RigidBodyState,
    joint: &D6Joint,
    h: f32,
    lambda: &mut f32,
    axis: usize,
) {
    let motion = [joint.linear_x, joint.linear_y, joint.linear_z][axis];
    if motion == D6Motion::Free {
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
    let dx = (state.positions[a] + r_a) - (state.positions[b] + r_b);
    let s = dx.dot(n);

    let limit = joint.linear_limit;
    let (c, compliance) = match motion {
        D6Motion::Locked => (s, joint.compliance),
        D6Motion::Limited => {
            if s > limit {
                (s - limit, joint.linear_limit_compliance)
            } else if s < -limit {
                (s + limit, joint.linear_limit_compliance)
            } else {
                return;
            }
        }
        D6Motion::Free => return,
    };

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

    let alpha_tilde = compliance / (h * h);
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

/// Projects the twist axis of a `D6` joint. The signed angle from body `a`'s
/// frame `y` axis to body `b`'s, both projected onto the plane perpendicular to
/// the world twist axis `t`, is driven to zero when [`Locked`](D6Motion::Locked),
/// clamped into `[twist_min, twist_max]` when [`Limited`](D6Motion::Limited),
/// and left untouched when [`Free`](D6Motion::Free). The correction is a rotation
/// about `t` — the hinge limit's angular pass.
fn solve_twist(state: &mut RigidBodyState, joint: &D6Joint, h: f32, lambda: &mut f32) {
    if joint.twist == D6Motion::Free {
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

    // Reference directions: each frame's `y` axis, projected onto the plane
    // perpendicular to the twist axis `t`.
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

    let (c, compliance) = match joint.twist {
        D6Motion::Locked => (theta, joint.compliance),
        D6Motion::Limited => {
            if theta < joint.twist_min {
                (theta - joint.twist_min, joint.angular_limit_compliance)
            } else if theta > joint.twist_max {
                (theta - joint.twist_max, joint.angular_limit_compliance)
            } else {
                return;
            }
        }
        D6Motion::Free => return,
    };

    apply_angular(state, a, b, q_a, q_b, t, c, compliance, h, lambda);
}

/// Projects swing1 (`which == 0`, tilt toward the frame `y` axis) or swing2
/// (`which == 1`, tilt toward the frame `z` axis) of a `D6` joint. With body
/// `b`'s twist axis `u_b` decomposed in the orthonormal frame `(t, e_y, e_z)` as
/// `(x_c, y_c, z_c)`, the swing angle is `atan2(y_c, x_c)` for swing1 (corrected
/// about `+e_z`) and `atan2(z_c, x_c)` for swing2 (corrected about `-e_y`); the
/// correction axis is the one about which rotating body `b` *increases* the
/// measured angle. Locked drives the angle to zero, limited clamps it into
/// `[-swing_limit, +swing_limit]`, and free skips it.
fn solve_swing(
    state: &mut RigidBodyState,
    joint: &D6Joint,
    h: f32,
    lambda: &mut f32,
    which: usize,
) {
    let motion = if which == 0 {
        joint.swing1
    } else {
        joint.swing2
    };
    if motion == D6Motion::Free {
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
    // swing1 tilts toward `e_y` (closed about `+e_z`); swing2 toward `e_z`
    // (closed about `-e_y`).
    let (angle, n, limit) = if which == 0 {
        let y_c = u_b.dot(e_y);
        (ops::atan2(y_c, x_c), e_z, joint.swing1_limit)
    } else {
        let z_c = u_b.dot(e_z);
        (ops::atan2(z_c, x_c), -e_y, joint.swing2_limit)
    };

    let (c, compliance) = match motion {
        D6Motion::Locked => (angle, joint.compliance),
        D6Motion::Limited => {
            if angle > limit {
                (angle - limit, joint.angular_limit_compliance)
            } else if angle < -limit {
                (angle + limit, joint.angular_limit_compliance)
            } else {
                return;
            }
        }
        D6Motion::Free => return,
    };

    apply_angular(state, a, b, q_a, q_b, n, c, compliance, h, lambda);
}

/// Applies a single-axis angular `XPBD` correction of signed violation `c`
/// about the unit world axis `n`, with the antisymmetric gradient `-n` on body
/// `a` and `+n` on body `b` (so a positive `c` measured as "rotating `b` about
/// `+n`" is driven shut). Shared by the twist and the two swing passes.
#[expect(
    clippy::too_many_arguments,
    reason = "a single-axis angular projection needs both bodies, both cached orientations, the axis, the violation, its compliance, the step, and its multiplier"
)]
fn apply_angular(
    state: &mut RigidBodyState,
    a: usize,
    b: usize,
    q_a: Quat,
    q_b: Quat,
    n: Vec3,
    c: f32,
    compliance: f32,
    h: f32,
    lambda: &mut f32,
) {
    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = n.dot(world_inv_inertia_apply(q_a, ii_a, n));
    let w_b = n.dot(world_inv_inertia_apply(q_b, ii_b, n));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    let alpha_tilde = compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    /// World-space anchor separation of `joint` in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let pa = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let pb = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (pa - pb).length()
    }

    /// Signed anchor separation vector `p_a - p_b`.
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

    /// Twist angle (radians), measured exactly as [`solve_twist`] measures it.
    fn twist_angle(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let (fa, fb) = world_frames(state, joint);
        let t = quat_rotate(fa, Vec3::X).normalize();
        let ra = quat_rotate(fa, Vec3::Y);
        let rb = quat_rotate(fb, Vec3::Y);
        let pa = (ra - t * t.dot(ra)).normalize();
        let pb = (rb - t * t.dot(rb)).normalize();
        ops::atan2(pa.cross(pb).dot(t), pa.dot(pb))
    }

    /// Swing1 angle (radians), measured exactly as [`solve_swing`] measures it.
    fn swing1_angle(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let (fa, fb) = world_frames(state, joint);
        let t = quat_rotate(fa, Vec3::X).normalize();
        let e_y = quat_rotate(fa, Vec3::Y).normalize();
        let u_b = quat_rotate(fb, Vec3::X).normalize();
        ops::atan2(u_b.dot(e_y), u_b.dot(t))
    }

    /// Swing2 angle (radians), measured exactly as [`solve_swing`] measures it.
    fn swing2_angle(state: &RigidBodyState, joint: &D6Joint) -> f32 {
        let (fa, fb) = world_frames(state, joint);
        let t = quat_rotate(fa, Vec3::X).normalize();
        let e_z = quat_rotate(fa, Vec3::Z).normalize();
        let u_b = quat_rotate(fb, Vec3::X).normalize();
        ops::atan2(u_b.dot(e_z), u_b.dot(t))
    }

    /// A dynamic limb socketed off a static pivot with identity joint frames, so
    /// the frame axes are the world axes: body `a` (index 0) is the static
    /// socket at the origin and body `b` (index 1) is the dynamic unit limb
    /// above it, anchored back through its `-Y` end to the socket. The six
    /// per-axis freedoms and the limits are configurable.
    fn socketed_limb(
        linear: [D6Motion; 3],
        angular: [D6Motion; 3],
        linear_limit: f32,
        twist_range: (f32, f32),
        swing_limits: (f32, f32),
    ) -> (RigidBodyState, D6Joint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: socket
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: limb
        let joint = D6Joint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            Quat::IDENTITY,
            linear,
            angular,
            linear_limit,
            twist_range,
            swing_limits,
            (0.0, 0.0, 0.0),
        );
        (state, joint)
    }

    const LOCKED3: [D6Motion; 3] = [D6Motion::Locked; 3];
    const FREE3: [D6Motion; 3] = [D6Motion::Free; 3];

    #[test]
    fn fixed_weld_pins_the_limb() {
        // All six axes locked: a weld against a static socket must hold the limb
        // rigidly in place even under an off-axis gravity that would otherwise
        // tilt and translate it.
        let (mut state, joint) = socketed_limb(LOCKED3, LOCKED3, 0.0, (0.0, 0.0), (0.0, 0.0));
        let integrator = IntegratorConfig::new(Vec3::new(4.0, -10.0, 2.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..60 {
            cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "weld anchor drifted apart"
            );
        }
        assert!(
            (state.positions[1] - Vec3::new(0.0, 1.0, 0.0)).length() < 2.0e-2,
            "welded limb translated: {:?}",
            state.positions[1]
        );
        assert!(twist_angle(&state, &joint).abs() < 2.0e-2, "weld twisted");
        assert!(swing1_angle(&state, &joint).abs() < 2.0e-2, "weld swung1");
        assert!(swing2_angle(&state, &joint).abs() < 2.0e-2, "weld swung2");
    }

    #[test]
    fn revolute_lets_the_twist_spin() {
        // Three linear axes and both swings locked, twist free: an initial spin
        // about the twist axis keeps turning while the anchor and the two swings
        // stay pinned.
        let angular = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let (mut state, joint) = socketed_limb(LOCKED3, angular, 0.0, (0.0, 0.0), (0.0, 0.0));
        state.angular_velocities[1] = Vec3::new(2.0, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..15 {
            cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "hinge anchor drifted apart"
            );
        }
        let twist = twist_angle(&state, &joint);
        assert!(twist > 0.15, "hinge never spun: twist = {twist}");
        assert!(twist < FRAC_PI_2, "hinge spun implausibly far: {twist}");
        assert!(swing1_angle(&state, &joint).abs() < 3.0e-2, "hinge swung1");
        assert!(swing2_angle(&state, &joint).abs() < 3.0e-2, "hinge swung2");
    }

    #[test]
    fn spherical_lets_the_limb_swing() {
        // Three linear axes locked, all three angular axes free: a horizontal
        // gravity swings the limb like a ball-socket pendulum while the anchor
        // stays shut.
        let (mut state, joint) = socketed_limb(LOCKED3, FREE3, 0.0, (0.0, 0.0), (0.0, 0.0));
        let integrator = IntegratorConfig::new(Vec3::new(3.0, 0.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        let mut max_swing: f32 = 0.0;
        for _ in 0..40 {
            cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "socket anchor drifted apart"
            );
            max_swing = max_swing.max(swing1_angle(&state, &joint).abs());
        }
        assert!(max_swing > 0.1, "ball-socket never swung: {max_swing}");
    }

    #[test]
    fn prismatic_lets_the_limb_slide() {
        // Linear `x` free, the other two linear axes and all three angular axes
        // locked: gravity along `x` slides the limb while the off-axis
        // separation and the orientation stay pinned.
        let linear = [D6Motion::Free, D6Motion::Locked, D6Motion::Locked];
        let (mut state, joint) = socketed_limb(linear, LOCKED3, 0.0, (0.0, 0.0), (0.0, 0.0));
        let integrator = IntegratorConfig::new(Vec3::new(5.0, 0.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..60 {
            cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let dx = anchor_delta(&state, &joint);
            assert!(
                dx.y.abs() < 1.0e-3 && dx.z.abs() < 1.0e-3,
                "slider left its axis: {dx:?}"
            );
        }
        assert!(
            state.positions[1].x > 0.1,
            "slider never slid: {}",
            state.positions[1].x
        );
        assert!(twist_angle(&state, &joint).abs() < 2.0e-2, "slider twisted");
        assert!(swing1_angle(&state, &joint).abs() < 2.0e-2, "slider swung1");
        assert!(swing2_angle(&state, &joint).abs() < 2.0e-2, "slider swung2");
    }

    #[test]
    fn twist_limit_arrests_both_signs() {
        // Twist limited to a symmetric band, everything else locked: a spin in
        // each direction settles against the near bound without blowing past it.
        for sign in [1.0_f32, -1.0] {
            let angular = [D6Motion::Limited, D6Motion::Locked, D6Motion::Locked];
            let (mut state, joint) = socketed_limb(LOCKED3, angular, 0.0, (-0.3, 0.3), (0.0, 0.0));
            state.angular_velocities[1] = Vec3::new(3.0 * sign, 0.0, 0.0);
            let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 4.0);
            let config = JointSolverConfig::new(8);
            let dt = 1.0 / 60.0;

            for _ in 0..400 {
                cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
                assert!(
                    anchor_separation(&state, &joint) < 1.0e-3,
                    "twist-limit anchor drifted apart"
                );
            }
            let twist = twist_angle(&state, &joint);
            assert!(
                (twist - 0.3 * sign).abs() < 5.0e-2,
                "twist did not settle at its bound: {twist} (sign {sign})"
            );
        }
    }

    #[test]
    fn swing1_limit_arrests_both_signs() {
        // Swing1 limited, twist and swing2 locked: a spin about the `+Z` axis
        // (which raises swing1) in each direction settles against the near bound.
        for sign in [1.0_f32, -1.0] {
            let angular = [D6Motion::Locked, D6Motion::Limited, D6Motion::Locked];
            let (mut state, joint) = socketed_limb(LOCKED3, angular, 0.0, (0.0, 0.0), (0.3, 0.0));
            state.angular_velocities[1] = Vec3::new(0.0, 0.0, 3.0 * sign);
            let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 4.0);
            let config = JointSolverConfig::new(8);
            let dt = 1.0 / 60.0;

            for _ in 0..400 {
                cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
                assert!(
                    anchor_separation(&state, &joint) < 1.0e-3,
                    "swing1-limit anchor drifted apart"
                );
            }
            let swing1 = swing1_angle(&state, &joint);
            assert!(
                (swing1 - 0.3 * sign).abs() < 5.0e-2,
                "swing1 did not settle at its bound: {swing1} (sign {sign})"
            );
            assert!(swing2_angle(&state, &joint).abs() < 3.0e-2, "swing2 leaked");
        }
    }

    #[test]
    fn swing2_limit_arrests_both_signs() {
        // Swing2 limited, twist and swing1 locked: a spin about the `-Y` axis
        // (which raises swing2) in each direction settles against the near bound.
        for sign in [1.0_f32, -1.0] {
            let angular = [D6Motion::Locked, D6Motion::Locked, D6Motion::Limited];
            let (mut state, joint) = socketed_limb(LOCKED3, angular, 0.0, (0.0, 0.0), (0.0, 0.3));
            state.angular_velocities[1] = Vec3::new(0.0, -3.0 * sign, 0.0);
            let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 4.0);
            let config = JointSolverConfig::new(8);
            let dt = 1.0 / 60.0;

            for _ in 0..400 {
                cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
                assert!(
                    anchor_separation(&state, &joint) < 1.0e-3,
                    "swing2-limit anchor drifted apart"
                );
            }
            let swing2 = swing2_angle(&state, &joint);
            assert!(
                (swing2 - 0.3 * sign).abs() < 5.0e-2,
                "swing2 did not settle at its bound: {swing2} (sign {sign})"
            );
            assert!(swing1_angle(&state, &joint).abs() < 3.0e-2, "swing1 leaked");
        }
    }

    #[test]
    fn linear_limit_arrests_both_signs() {
        // Linear `x` limited to a symmetric slab, everything else locked: gravity
        // along `x` in each direction drags the limb to the near stop and parks
        // it there.
        for sign in [1.0_f32, -1.0] {
            let linear = [D6Motion::Limited, D6Motion::Locked, D6Motion::Locked];
            let (mut state, joint) = socketed_limb(linear, LOCKED3, 0.25, (0.0, 0.0), (0.0, 0.0));
            let integrator = IntegratorConfig::new(Vec3::new(5.0 * sign, 0.0, 0.0), 8, 0.0, 4.0);
            let config = JointSolverConfig::new(8);
            let dt = 1.0 / 60.0;

            for _ in 0..400 {
                cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
            }
            let dx = anchor_delta(&state, &joint);
            assert!(
                (dx.x + 0.25 * sign).abs() < 5.0e-3,
                "linear slab did not park at its stop: {} (sign {sign})",
                dx.x
            );
            assert!(
                dx.y.abs() < 1.0e-3 && dx.z.abs() < 1.0e-3,
                "slab left its axis: {dx:?}"
            );
        }
    }

    #[test]
    fn free_pair_falls_independently() {
        // Every axis free: the limb is uncoupled and simply falls under gravity.
        let (mut state, joint) = socketed_limb(FREE3, FREE3, 0.0, (0.0, 0.0), (0.0, 0.0));
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -10.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_d6(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        assert!(
            state.positions[1].y < 0.9,
            "free limb did not fall: {:?}",
            state.positions[1]
        );
        assert!(
            state.positions[0] == Vec3::ZERO,
            "static socket moved: {:?}",
            state.positions[0]
        );
    }

    #[test]
    fn empty_joint_set_is_a_no_op() {
        let (mut state, _joint) = socketed_limb(LOCKED3, LOCKED3, 0.0, (0.0, 0.0), (0.0, 0.0));
        let before = state.positions.clone();
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -10.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_d6(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions, before);
    }

    #[test]
    fn out_of_range_body_is_rejected() {
        let (mut state, _joint) = socketed_limb(LOCKED3, LOCKED3, 0.0, (0.0, 0.0), (0.0, 0.0));
        let bad = D6Joint::fixed(0, 9, Vec3::ZERO, Vec3::ZERO, Quat::IDENTITY, Quat::IDENTITY);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let err = cpu_solve_joints_d6(&mut state, &[bad], &integrator, &config, 1.0 / 60.0);
        assert!(matches!(err, Err(RigidError::InconsistentState { .. })));
    }
}
