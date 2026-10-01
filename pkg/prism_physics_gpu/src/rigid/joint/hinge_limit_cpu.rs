//! Authoritative `CPU` golden reference for the hinge angular-limit joint
//! stepper.
//!
//! [`cpu_solve_joints_hinge_limit`] is a *full stepper* with the identical
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
//! A hinge-limit joint is the revolute joint's axis-alignment plus
//! point-to-point weld with a swing limit added about the hinge axis. Each sweep
//! projects, in order: the **axis-alignment** angular constraint (realigning the
//! two world-space hinge axes), then the **angular limit** (clamping the signed
//! hinge angle into `[min_angle, max_angle]` about the freshly aligned axis),
//! then the **point-to-point** positional weld (numerically identical to the
//! spherical golden's `solve_one`).
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer:
//! slot `3 * k` for the positional weld, slot `3 * k + 1` for the axis
//! alignment, and slot `3 * k + 2` for the angular limit, where `k` is the
//! joint's index in the colour-reordered list. The buffer is therefore
//! `3 * joints.len()` long and is reset to zero at the start of every substep.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuHingeLimitJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the hinge
//! axis-alignment constraint, and the one-sided angular limit with their substep
//! `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"),
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::hinge_limit::HingeLimitJoint;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};
use bevy_math::ops;

/// Advances `state` by `dt` under the hinge-limit joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment, angular-limit, and point-to-point
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
pub fn cpu_solve_joints_hinge_limit(
    state: &mut RigidBodyState,
    joints: &[HingeLimitJoint],
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
    // Three multipliers per joint: `[3k]` positional weld, `[3k + 1]` axis
    // alignment, `[3k + 2]` angular limit. Reset to zero every substep.
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

/// Projects one hinge-limit joint for a single sweep: axis alignment first
/// (accumulating `lambda[1]`), then the angular limit (`lambda[2]`), then the
/// point-to-point weld (`lambda[0]`), each applied directly to the two bodies'
/// transforms.
fn solve_one(state: &mut RigidBodyState, joint: &HingeLimitJoint, h: f32, lambda: &mut [f32]) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_angular_limit(state, joint, h, &mut lambda[2]);
    solve_point_to_point(state, joint, h, &mut lambda[0]);
}

/// Drives the two world-space hinge axes parallel by cancelling their cross
/// product, locking the two rotational degrees of freedom perpendicular to the
/// hinge while leaving the spin about it free. Identical to the revolute
/// golden's axis-alignment pass.
fn solve_axis_alignment(
    state: &mut RigidBodyState,
    joint: &HingeLimitJoint,
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

/// Clamps the signed hinge angle into `[min_angle, max_angle]` with a free dead
/// zone. The angle is measured from body `a`'s reference direction to body
/// `b`'s, both projected onto the plane perpendicular to the (unit) hinge axis
/// `u = rotate(q_a, axis_a)`, as `theta = atan2((p_a x p_b) . u, p_a . p_b)`.
/// Only a violated bound exerts a correction, applied as a signed rotation about
/// `u`.
fn solve_angular_limit(
    state: &mut RigidBodyState,
    joint: &HingeLimitJoint,
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

    // Signed limit violation with a free dead zone inside the range.
    let c = if theta < joint.min_angle {
        theta - joint.min_angle
    } else if theta > joint.max_angle {
        theta - joint.max_angle
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

    let alpha_tilde = joint.limit_compliance / (h * h);
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
    joint: &HingeLimitJoint,
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
    use std::f32::consts::FRAC_PI_2;

    /// Signed hinge angle (radians) of `joint` in `state`, measured the same way
    /// the solver measures it.
    fn hinge_angle(state: &RigidBodyState, joint: &HingeLimitJoint) -> f32 {
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
    fn anchor_separation(state: &RigidBodyState, joint: &HingeLimitJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    /// A dynamic body hinged to a static pivot about the world Z axis, free to
    /// swing in a limited arc. Body 0 is the static pivot; body 1 the swinging
    /// link, anchored back to the origin.
    fn hinged_link(min_angle: f32, max_angle: f32) -> (RigidBodyState, HingeLimitJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: pivot
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: link, extends +X
        let joint = HingeLimitJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::Z,
            Vec3::Z,
            Vec3::X,
            Vec3::X,
            min_angle,
            max_angle,
            0.0,
            0.0,
            0.0,
        );
        (state, joint)
    }

    #[test]
    fn free_within_range_lets_the_hinge_swing() {
        // A wide range (+/- 90 deg): the link falls under gravity and the angle
        // moves well away from zero without the limit arresting it, since it
        // never leaves the free interior over the arc tested.
        let (mut state, joint) = hinged_link(-FRAC_PI_2, FRAC_PI_2);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..40 {
            cpu_solve_joints_hinge_limit(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
        }
        // Gravity pulls the +X link downward (negative hinge angle about +Z);
        // it should swing a meaningful amount while staying inside the range.
        let theta = hinge_angle(&state, &joint);
        assert!(theta < -0.1, "hinge never swung: theta = {theta}");
        assert!(
            theta > -FRAC_PI_2,
            "hinge should not have hit the lower stop"
        );
    }

    #[test]
    fn lower_stop_arrests_the_swing() {
        // A tight lower stop just below the rest: the link cannot swing past it,
        // so the angle settles at (or just above) the lower bound and never
        // deeply penetrates it.
        let lower = -0.3;
        let (mut state, joint) = hinged_link(lower, FRAC_PI_2);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..240 {
            cpu_solve_joints_hinge_limit(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let theta = hinge_angle(&state, &joint);
            assert!(
                theta > lower - 5.0e-3,
                "hinge drove past the lower stop to {theta}"
            );
        }
        // It should actually have reached the stop under gravity.
        let theta = hinge_angle(&state, &joint);
        assert!(
            theta < lower + 5.0e-2,
            "hinge never descended to its lower stop: theta = {theta}"
        );
    }

    #[test]
    fn upper_stop_arrests_a_pushed_hinge() {
        // Give the link an upward (positive) spin about +Z and a tight upper
        // stop: the limit must catch it near the bound rather than let it spin
        // through.
        let upper = 0.4;
        let (mut state, joint) = hinged_link(-FRAC_PI_2, upper);
        state.angular_velocities[1] = Vec3::new(0.0, 0.0, 3.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_hinge_limit(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let theta = hinge_angle(&state, &joint);
            assert!(
                theta < upper + 5.0e-3,
                "hinge drove past the upper stop to {theta}"
            );
        }
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, one body given an initial velocity: all of
        // the joint's corrections are internal, so total linear momentum is
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
        let joint = HingeLimitJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            -0.2,
            0.2,
            0.0,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_hinge_limit(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 5.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }

    #[test]
    fn alignment_pulls_a_tilted_axis_parallel() {
        // A static pivot and a dynamic link whose hinge axes start 30 degrees
        // apart: the rigid axis alignment must pull them toward *parallel*
        // (`dot = +1`), never flip them to anti-parallel (`dot = -1`). The
        // signed `dot` guards the correction's sign, which a `sin`-only or
        // `|cross|`-only measure cannot see because both parallel and
        // anti-parallel read ~0 there.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static pivot
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 1: dynamic link
                                                                       // Hinge about the shared world Y axis; tilt body 1 by 30 degrees about X
                                                                       // so its world hinge axis leans off Y by that angle.
        state.orientations[1] = Quat::from_axis_angle(Vec3::X, std::f32::consts::FRAC_PI_6);
        let joint = HingeLimitJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            -FRAC_PI_2,
            FRAC_PI_2,
            0.0,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..4 {
            cpu_solve_joints_hinge_limit(&mut state, &[joint], &integrator, &config, dt).unwrap();
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
