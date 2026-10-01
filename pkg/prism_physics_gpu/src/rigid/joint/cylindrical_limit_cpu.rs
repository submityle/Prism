//! Authoritative `CPU` golden reference for the cylindrical limit joint stepper.
//!
//! [`cpu_solve_joints_cylindrical_limit`] is a *full stepper* with the identical
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
//! A cylindrical limit joint keeps body `b`'s axis parallel to body `a`'s, pins
//! the two anchors to a common line, and *clamps* the free along-axis separation
//! into `[min_distance, max_distance]`, leaving the spin about the axis free.
//! Each sweep projects the **axis-alignment** constraint first — driving the two
//! world-space axes back to parallel — then the **point-on-line** weld, which
//! cancels only the component of the anchor separation perpendicular to the (now
//! re-aligned) axis, then the **travel limit**, a one-sided along-axis
//! correction that is a no-op inside the range and pushes a violated bound shut
//! outside it. Projecting the axis orientation before the anchors means the
//! positional and limit passes work against an already-parallel axis.
//!
//! The axis-alignment constraint takes the cross product `delta = u_a x u_b` of
//! the two unit axes (whose magnitude is `sin` of the angle between them), drives
//! its length `theta` to zero by rotating about `n = delta / theta`, with the
//! gradient `+n` on body `a` and `-n` on body `b`. This reuses, bit for bit, the
//! alignment geometry of the [`RevoluteJoint`](super::RevoluteJoint). The
//! point-on-line weld removes only `dx_perp = dx - axis_w * dot(dx, axis_w)`, the
//! perpendicular part of the anchor separation about the world-space axis,
//! reusing the perpendicular weld of the [`PrismaticJoint`](super::PrismaticJoint).
//! The limit clamps `s = dx . axis_w` into the travel range, reusing the
//! one-sided stop of the [`PrismaticLimitJoint`](super::PrismaticLimitJoint).
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer: slot
//! `3 * k` for the point-on-line weld, slot `3 * k + 1` for the axis alignment,
//! and slot `3 * k + 2` for the travel limit, where `k` is the joint's index in
//! the colour-reordered list. The buffer is therefore `3 * joints.len()` long and
//! is reset to zero at the start of every substep.
//!
//! # The limit update
//!
//! The limit is the one-sided `XPBD` inequality clamping `s` into the range. The
//! signed violation is `c = s - min_distance` below the lower bound, `c = s -
//! max_distance` above the upper bound, and zero (dead zone) inside the range,
//! where the limit exerts no force and the slide is free. With `w` the effective
//! inverse mass along the world slide axis `n` and `alpha_tilde = limit_compliance
//! / h^2` the regularisation, the active sweep applies the standard positional
//! `XPBD` update `d_lambda = (-c - alpha_tilde * lambda) / (w + alpha_tilde)`
//! along `n`.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuCylindricalLimitJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout. Neither the alignment, weld, nor limit takes an inverse
//! cosine, so the two paths need only match in finite-precision reassociation;
//! the tight parity tolerance bounds the result.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge, the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, and the one-sided along-axis limit shared
//! with the prismatic limit, with their substep `XPBD` handling (Müller et al.,
//! "Detailed Rigid Body Simulation with XPBD"), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::cylindrical_limit::CylindricalLimitJoint;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};
use glam::Vec3;

/// Advances `state` by `dt` under the cylindrical limit joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment, point-on-line, and travel-limit
/// constraints every substep. The joint set is coloured once up front so
/// same-batch joints write disjoint movable bodies; the batches are then solved
/// in order [`position_iterations`](JointSolverConfig::position_iterations) times
/// per substep.
///
/// # Errors
///
/// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree in
/// length or a joint references a body outside the state, and
/// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
/// batches than the colouring supports. Returns `Ok(())` with the state
/// untouched when there is nothing to do (`dt <= 0`, no bodies, or no joints).
pub fn cpu_solve_joints_cylindrical_limit(
    state: &mut RigidBodyState,
    joints: &[CylindricalLimitJoint],
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
    let mut prev_orientations = vec![glam::Quat::IDENTITY; state.len()];

    // Three Lagrange multipliers per joint: slot `3k` for the point-on-line
    // weld, slot `3k + 1` for the axis alignment, slot `3k + 2` for the travel
    // limit. Reset to zero at the start of every substep.
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

/// Projects one cylindrical limit joint for a single sweep: the axis-alignment
/// correction first (accumulating `lambda[1]`), then the point-on-line
/// correction (accumulating `lambda[0]`), then the one-sided travel-limit
/// correction (accumulating `lambda[2]`), each applied directly to the two
/// bodies' transforms.
fn solve_one(
    state: &mut RigidBodyState,
    joint: &CylindricalLimitJoint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_point_on_line(state, joint, h, &mut lambda[0]);
    solve_limit(state, joint, h, &mut lambda[2]);
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
    joint: &CylindricalLimitJoint,
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

    // The alignment error is the cross product of the two unit axes; its
    // magnitude is `sin(angle)` between them and its direction is the rotation
    // axis that drives them parallel.
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

    // `d_lambda` is negative, so `p` points along `-n`. Rotating `u_a` toward
    // `u_b` requires rotating body `a` about `+n` (and body `b` about `-n`),
    // hence body `a` receives `-I^-1 p` and body `b` receives `+I^-1 p`.
    state.orientations[a] = apply_rotation_delta(q_a, -world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, world_inv_inertia_apply(q_b, ii_b, p));
}

/// Pins the two world-space anchors to a common line. With `dx` the anchor
/// separation and `axis_w = rotate(q_a, axis_a)` the world-space shared axis,
/// only the perpendicular component `dx_perp = dx - axis_w * dot(dx, axis_w)` is
/// cancelled; the along-axis component is left free, so the two anchors share a
/// line rather than a point and the slide along the axis stays free.
fn solve_point_on_line(
    state: &mut RigidBodyState,
    joint: &CylindricalLimitJoint,
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

/// Clamps the signed along-axis separation `s = dx . axis_world` into
/// `[min_distance, max_distance]`. Inside the range the limit is a no-op; below
/// the lower bound or above the upper bound the signed violation `c = s - bound`
/// is cancelled by an impulse along the world slide axis, pushing the anchors
/// back onto the active stop. The correction gradient is the point-to-point
/// gradient restricted to the single axis direction, so it feeds back into both
/// bodies' translation and rotation like the perpendicular weld.
fn solve_limit(
    state: &mut RigidBodyState,
    joint: &CylindricalLimitJoint,
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

    // Signed violation of the active bound; zero (dead zone) inside the range.
    let c = if s < joint.min_distance {
        s - joint.min_distance
    } else if s > joint.max_distance {
        s - joint.max_distance
    } else {
        return;
    };

    // The correction direction is the world slide axis; `c`'s sign selects which
    // bound is pushed shut.
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

    let alpha_tilde = joint.limit_compliance / (h * h);
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
    use bevy_math::ops;
    use glam::{Quat, Vec3};

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
    fn slide_position(state: &RigidBodyState, joint: &CylindricalLimitJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let r_a = rotate(state.orientations[a], joint.anchor_a);
        let r_b = rotate(state.orientations[b], joint.anchor_b);
        let dx = (state.positions[a] + r_a) - (state.positions[b] + r_b);
        let axis = rotate(state.orientations[a], joint.axis_a).normalize();
        dx.dot(axis)
    }

    /// Angle in radians between the two world-space axes — zero when aligned.
    fn axis_angle(state: &RigidBodyState, joint: &CylindricalLimitJoint) -> f32 {
        let u_a = rotate(state.orientations[joint.body_a as usize], joint.axis_a).normalize();
        let u_b = rotate(state.orientations[joint.body_b as usize], joint.axis_b).normalize();
        ops::acos(u_a.dot(u_b).clamp(-1.0, 1.0))
    }

    #[test]
    fn slide_is_free_inside_the_range() {
        // A sleeve dropped under gravity along the slide axis with a wide travel
        // range falls freely until it would hit a stop; over a short window it
        // stays inside the range, so the limit never fires.
        let joint =
            CylindricalLimitJoint::symmetric(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 1.0);
        let mut state = rod_and_sleeve();
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..10 {
            cpu_solve_joints_cylindrical_limit(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let s = slide_position(&state, &joint);
        assert!(
            s < 0.0 && s > -1.0,
            "sleeve should fall freely inside the range: s = {s}"
        );
        // It fell, so it moved off zero.
        assert!(s < -1.0e-3, "sleeve did not fall at all: s = {s}");
    }

    #[test]
    fn lower_bound_catches_a_falling_sleeve() {
        // A sleeve falling under axial gravity is caught at the lower stop and
        // held there; it must not sink appreciably past `min_distance`.
        let joint =
            CylindricalLimitJoint::symmetric(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.25);
        let mut state = rod_and_sleeve();
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            cpu_solve_joints_cylindrical_limit(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let s = slide_position(&state, &joint);
        assert!(
            (s - (-0.25)).abs() < 1.0e-3,
            "sleeve was not held at the lower stop: s = {s}"
        );
    }

    #[test]
    fn upper_bound_catches_a_rising_sleeve() {
        // A sleeve launched upward with an initial velocity is caught at the
        // upper stop and held there; it must not escape past `max_distance`.
        let joint =
            CylindricalLimitJoint::symmetric(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.3);
        let mut state = rod_and_sleeve();
        state.linear_velocities[1] = Vec3::new(0.0, 5.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_cylindrical_limit(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        let s = slide_position(&state, &joint);
        assert!(
            s <= 0.3 + 1.0e-3,
            "sleeve escaped past the upper stop: s = {s}"
        );
    }

    #[test]
    fn spin_about_axis_stays_free_at_a_stop() {
        // A sleeve pinned against its lower stop by axial gravity must still spin
        // freely about the axis; the alignment and limit must not bleed into the
        // free spin.
        let joint =
            CylindricalLimitJoint::symmetric(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.2);
        let mut state = rod_and_sleeve();
        state.angular_velocities[1] = Vec3::new(0.0, 2.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            cpu_solve_joints_cylindrical_limit(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        assert!(
            state.angular_velocities[1].y > 1.9,
            "spin about the axis was damped by the limit: {}",
            state.angular_velocities[1].y
        );
        assert!(
            axis_angle(&state, &joint) < 1.0e-3,
            "spin tilted the axis off parallel"
        );
    }

    #[test]
    fn alignment_holds_while_bounded() {
        // A sleeve starting 20 degrees off parallel, pinned at a stop by gravity:
        // the alignment constraint must still pull the axis parallel while the
        // limit holds the slide.
        let joint =
            CylindricalLimitJoint::symmetric(1, 0, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.15);
        let mut state = rod_and_sleeve();
        let tilt = std::f32::consts::FRAC_PI_6 * 0.666;
        state.orientations[1] = Quat::from_axis_angle(Vec3::X, tilt);
        let integrator = IntegratorConfig::new(Vec3::new(0.0, -9.81, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cpu_solve_joints_cylindrical_limit(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }
        assert!(
            axis_angle(&state, &joint) < 2.0e-2,
            "alignment failed under the limit: angle = {}",
            axis_angle(&state, &joint)
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum_at_a_stop() {
        // Zero gravity, equal masses, both bodies given the same initial velocity
        // large enough to drive the pair onto the upper stop: the joint's
        // internal corrections are equal and opposite, so total linear momentum
        // must be conserved.
        let joint = CylindricalLimitJoint::new(
            0,
            1,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::Y,
            Vec3::Y,
            -0.1,
            0.1,
            0.0,
            0.0,
            0.0,
        );
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.1, 0.2, -0.1);
        state.linear_velocities[1] = Vec3::new(0.1, 0.2, -0.1);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_cylindrical_limit(&mut state, &[joint], &integrator, &config, dt)
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
        cpu_solve_joints_cylindrical_limit(&mut state, &[], &integrator, &config, 1.0 / 60.0)
            .unwrap();
        assert_eq!(state.positions[0], before);
    }
}
