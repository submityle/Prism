//! Authoritative `CPU` golden reference for the swing-twist (cone-twist) joint
//! stepper.
//!
//! [`cpu_solve_joints_swing_twist`] is a *full stepper* with the identical
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
//! # The three constraints, projected swing-first
//!
//! A swing-twist joint welds the two bodies' anchors together, bounds the
//! lateral tilt of the twist axis within a cone, and bounds the axial spin
//! within a twist range. Each sweep projects, in order: the **swing cone**
//! (closing any over-cone tilt of the two world-space twist axes), then the
//! **twist limit** (clamping the signed spin about the freshly measured twist
//! axis), then the **point-to-point** positional weld (numerically identical to
//! the spherical golden's `solve_one`).
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer:
//! slot `3 * k` for the positional weld, slot `3 * k + 1` for the swing cone,
//! and slot `3 * k + 2` for the twist limit, where `k` is the joint's index in
//! the colour-reordered list. The buffer is therefore `3 * joints.len()` long
//! and is reset to zero at the start of every substep.
//!
//! # The swing cone
//!
//! With `u_a = rotate(q_a, twist_axis_a)` and `u_b = rotate(q_b, twist_axis_b)`
//! the two unit twist axes, the swing angle is the true angle between them,
//! `swing = acos(clamp(u_a . u_b, -1, 1))`, taken with `acos` rather than the
//! cross-product magnitude so the cone half-angle may exceed `90` degrees. While
//! `swing <= swing_limit` the cone is inactive. Past it the violation
//! `c = swing - swing_limit` is driven shut by a rotation about
//! `n = normalize(u_a x u_b)` — the same axis the revolute golden's
//! axis-alignment pass uses to close the gap between two directions — with the
//! constraint gradients (`-n` on body `a`, `+n` on body `b`, since rotating
//! body `a` about `+n` closes the cone while rotating body `b` about `+n`
//! widens it). The
//! effective inverse mass is `w = n . (I_a^-1 n) + n . (I_b^-1 n)`.
//!
//! # Measuring the twist angle
//!
//! The twist limit reuses the hinge limit's signed-angle machinery verbatim.
//! Each body carries a body-local reference direction, `ref_a` and `ref_b`,
//! nominally perpendicular to its twist axis. Both are rotated to world space,
//! projected onto the plane perpendicular to the unit twist axis
//! `u = rotate(q_a, twist_axis_a)`, and normalised; the signed angle from `a`'s
//! projection to `b`'s projection about `u` is
//! `theta = atan2((p_a x p_b) . u, p_a . p_b)`, clamped into
//! `[twist_min, twist_max]` with a free dead zone.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuSwingTwistJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout. The transcendental angles are taken with
//! [`bevy_math::ops::acos`] and [`bevy_math::ops::atan2`] rather than the `f32`
//! intrinsics so the `CPU` reference and the shader share one transcendental
//! path.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the signed angular
//! limit shared with the hinge limit, and the cone-swing limit with their
//! substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};
use super::swing_twist::SwingTwistJoint;
use bevy_math::ops;
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the swing-twist joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's swing-cone, twist-limit, and point-to-point
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
pub fn cpu_solve_joints_swing_twist(
    state: &mut RigidBodyState,
    joints: &[SwingTwistJoint],
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

    // Three multipliers per joint: `[3k]` positional weld, `[3k + 1]` swing
    // cone, `[3k + 2]` twist limit. Reset to zero every substep.
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
                    solve_one(state, &ordered[k], h, &mut lambda[3 * k..3 * k + 3]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one swing-twist joint for a single sweep: swing cone first
/// (accumulating `lambda[1]`), then the twist limit (`lambda[2]`), then the
/// point-to-point weld (`lambda[0]`), each applied directly to the two bodies'
/// transforms.
fn solve_one(state: &mut RigidBodyState, joint: &SwingTwistJoint, h: f32, lambda: &mut [f32]) {
    solve_swing_cone(state, joint, h, &mut lambda[1]);
    solve_twist_limit(state, joint, h, &mut lambda[2]);
    solve_point_to_point(state, joint, h, &mut lambda[0]);
}

/// Closes an over-cone tilt of the two world-space twist axes. The swing angle
/// `swing = acos(clamp(u_a . u_b, -1, 1))` between the unit twist axes is left
/// free while `swing <= swing_limit`; past the rim the violation
/// `swing - swing_limit` is driven to zero by a rotation about
/// `n = normalize(u_a x u_b)`, with the gradient `-n` on body `a` and `+n` on
/// body `b` (rotating `a` about `+n` closes the cone, rotating `b` about `+n`
/// widens it).
fn solve_swing_cone(state: &mut RigidBodyState, joint: &SwingTwistJoint, h: f32, lambda: &mut f32) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let axis_a = rotate(q_a, joint.twist_axis_a);
    let axis_b = rotate(q_b, joint.twist_axis_b);
    let len_a = axis_a.length();
    let len_b = axis_b.length();
    if len_a < EPSILON || len_b < EPSILON {
        return;
    }
    let u_a = axis_a / len_a;
    let u_b = axis_b / len_b;

    // True angle between the twist axes (may exceed 90 degrees, so `acos` of the
    // dot rather than the cross-product magnitude).
    let swing = ops::acos((u_a.dot(u_b)).clamp(-1.0, 1.0));
    if swing <= joint.swing_limit {
        return;
    }
    let c = swing - joint.swing_limit;

    // Rotation axis that closes the gap between the two twist axes.
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

    let alpha_tilde = joint.swing_compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = n * d_lambda;

    // `u_b` is `u_a` rotated by `+swing` about `n = normalize(u_a x u_b)`, so
    // rotating body `b` about `+n` *widens* the cone and rotating body `a`
    // about `+n` *closes* it. The constraint gradients are therefore `-n` on
    // body `a` and `+n` on body `b` — the same antisymmetric pattern the twist
    // limit uses — so the over-cone violation is driven shut (not open).
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Clamps the signed twist angle into `[twist_min, twist_max]` with a free dead
/// zone. The angle is measured from body `a`'s reference direction to body
/// `b`'s, both projected onto the plane perpendicular to the (unit) twist axis
/// `u = rotate(q_a, twist_axis_a)`, as `theta = atan2((p_a x p_b) . u,
/// p_a . p_b)`. Only a violated bound exerts a correction, applied as a signed
/// rotation about `u`. Identical to the hinge limit's angular-limit pass.
fn solve_twist_limit(
    state: &mut RigidBodyState,
    joint: &SwingTwistJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let axis = rotate(q_a, joint.twist_axis_a);
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

    // Signed twist angle from `a`'s reference to `b`'s about the twist axis.
    let sin_theta = pa_n.cross(pb_n).dot(u);
    let cos_theta = pa_n.dot(pb_n);
    let theta = ops::atan2(sin_theta, cos_theta);

    // Signed limit violation with a free dead zone inside the range.
    let c = if theta < joint.twist_min {
        theta - joint.twist_min
    } else if theta > joint.twist_max {
        theta - joint.twist_max
    } else {
        return;
    };

    let ii_a = state.inverse_inertias[a];
    let ii_b = state.inverse_inertias[b];
    let w_a = u.dot(world_inv_inertia_apply(q_a, ii_a, u));
    let w_b = u.dot(world_inv_inertia_apply(q_b, ii_b, u));
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    let alpha_tilde = joint.twist_compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;
    let p = u * d_lambda;

    // Rotating body `b` by `+u` increases `theta`; rotating body `a` by `+u`
    // decreases it. The constraint gradients are therefore `-u` on `a` and
    // `+u` on `b`.
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Drives the two world-space anchors together with the identical point-to-point
/// `XPBD` positional correction as the spherical golden.
fn solve_point_to_point(
    state: &mut RigidBodyState,
    joint: &SwingTwistJoint,
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
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

    /// Signed twist angle (radians) of `joint` in `state`, measured the same way
    /// the solver measures it.
    fn twist_angle(state: &RigidBodyState, joint: &SwingTwistJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let q_a = state.orientations[a];
        let q_b = state.orientations[b];
        let axis = rotate(q_a, joint.twist_axis_a);
        let u = axis / axis.length();
        let ra = rotate(q_a, joint.ref_a);
        let rb = rotate(q_b, joint.ref_b);
        let pa = (ra - u * u.dot(ra)).normalize();
        let pb = (rb - u * u.dot(rb)).normalize();
        ops::atan2(pa.cross(pb).dot(u), pa.dot(pb))
    }

    /// Swing angle (radians) between the two world-space twist axes.
    fn swing_angle(state: &RigidBodyState, joint: &SwingTwistJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let u_a = rotate(state.orientations[a], joint.twist_axis_a).normalize();
        let u_b = rotate(state.orientations[b], joint.twist_axis_b).normalize();
        ops::acos((u_a.dot(u_b)).clamp(-1.0, 1.0))
    }

    /// World-space anchor separation of `joint` in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &SwingTwistJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let pa = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let pb = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (pa - pb).length()
    }

    /// A dynamic `+Y` limb socketed off a static pivot: body `a` (index 0) is
    /// the static socket at the origin with twist axis `+Y`; body `b` (index 1)
    /// is the dynamic limb above it, anchored back to the socket.
    fn socketed_limb(swing_limit: f32, twist_limit: f32) -> (RigidBodyState, SwingTwistJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: socket
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: limb
        let joint = SwingTwistJoint::symmetric_cone(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            swing_limit,
            twist_limit,
        );
        (state, joint)
    }

    #[test]
    fn free_within_cone_lets_the_limb_swing() {
        // A wide cone (90 deg) with gravity pulling the +Y limb sideways: the
        // limb swings a meaningful amount while staying inside the cone and the
        // anchor never drifts apart.
        let (mut state, joint) = socketed_limb(FRAC_PI_2, FRAC_PI_2);
        let integrator = IntegratorConfig::new(Vec3::new(1.0, 0.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..40 {
            cpu_solve_joints_swing_twist(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
        }
        let swing = swing_angle(&state, &joint);
        assert!(swing > 0.05, "limb never swung: swing = {swing}");
        assert!(swing < FRAC_PI_2, "limb should not have hit the cone rim");
    }

    #[test]
    fn cone_rim_arrests_the_swing() {
        // A tight cone with a steady lateral gravity and angular damping: the
        // limb is dragged out to the rim and, as the damping bleeds off the
        // swing velocity, settles against the cone rather than tipping past it.
        // Damping is what makes the rest state quasi-static, so the rigid rim
        // can be checked to a tight tolerance without a velocity-driven slam
        // transient overshooting it.
        let swing_limit = 0.3;
        let (mut state, joint) = socketed_limb(swing_limit, FRAC_PI_2);
        let integrator = IntegratorConfig::new(Vec3::new(1.5, 0.0, 0.0), 8, 0.0, 4.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        // Let the limb settle against the rim.
        for _ in 0..400 {
            cpu_solve_joints_swing_twist(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
        }

        // At rest the lateral pull holds it on the rim: pinned at the limit from
        // above and reached from below.
        let swing = swing_angle(&state, &joint);
        assert!(
            swing < swing_limit + 5.0e-3,
            "limb settled past the cone rim at {swing}"
        );
        assert!(
            swing > swing_limit - 2.0e-2,
            "limb never reached its cone rim: swing = {swing}"
        );
    }

    #[test]
    fn twist_stop_arrests_a_spun_limb() {
        // Give the limb an axial spin about +Y and a tight twist range: the
        // twist limit must catch it near the bound rather than let it spin
        // through, while the wide cone leaves the swing free.
        let twist_limit = 0.4;
        let (mut state, joint) = socketed_limb(FRAC_PI_2, twist_limit);
        state.angular_velocities[1] = Vec3::new(0.0, 3.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_swing_twist(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let twist = twist_angle(&state, &joint);
            assert!(
                twist < twist_limit + 5.0e-3,
                "limb twisted past the upper stop to {twist}"
            );
        }
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, both bodies given one shared translational
        // velocity: all of the joint's corrections are internal, so total linear
        // momentum is conserved up to solver noise.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.3, 0.1, -0.2);
        state.linear_velocities[1] = Vec3::new(0.3, 0.1, -0.2);
        let joint = SwingTwistJoint::symmetric_cone(
            0,
            1,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            FRAC_PI_4,
            0.3,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_swing_twist(&mut state, &[joint], &integrator, &config, dt).unwrap();
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
        let (mut state, _joint) = socketed_limb(FRAC_PI_2, 0.5);
        let before = state.positions[1];
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_swing_twist(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[1], before, "no-op solve moved a body");
    }
}
