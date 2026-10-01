//! Authoritative `CPU` golden reference for the cylindrical joint stepper.
//!
//! [`cpu_solve_joints_cylindrical`] is a *full stepper* with the identical
//! substep schedule as [`cpu_solve_joints_universal`](super::cpu_solve_joints_universal):
//! a caller hands it the current [`RigidBodyState`], the joint set, the shared
//! [`IntegratorConfig`], the joint-specific [`JointSolverConfig`], and the frame
//! `dt`, and must **not** integrate the bodies itself. Within each integrator
//! substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's two `XPBD` Lagrange multipliers,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # The two constraints, projected alignment-first
//!
//! A cylindrical joint keeps body `b`'s axis parallel to body `a`'s and pins the
//! two anchors to a common line, leaving the slide along and the spin about that
//! axis free. Each sweep projects the **axis-alignment** constraint first —
//! driving the two world-space axes back to parallel — then the
//! **point-on-line** weld, which cancels only the component of the anchor
//! separation perpendicular to the (now re-aligned) axis. Projecting the axis
//! orientation before the anchors means the positional pass works against an
//! already-parallel axis.
//!
//! The axis-alignment constraint takes the cross product `delta = u_a x u_b` of
//! the two unit axes (whose magnitude is `sin` of the angle between them), drives
//! its length `theta` to zero by rotating about `n = delta / theta`, with the
//! gradient `+n` on body `a` and `-n` on body `b`. This reuses, bit for bit, the
//! alignment geometry of the [`RevoluteJoint`](super::RevoluteJoint). The
//! point-on-line weld removes only `dx_perp = dx - axis_w * dot(dx, axis_w)`, the
//! perpendicular part of the anchor separation about the world-space axis,
//! reusing the perpendicular weld of the [`PrismaticJoint`](super::PrismaticJoint).
//!
//! Each joint owns two Lagrange multipliers in the shared `lambda` buffer: slot
//! `2 * k` for the point-on-line weld and slot `2 * k + 1` for the axis
//! alignment, where `k` is the joint's index in the colour-reordered list. The
//! buffer is therefore `2 * joints.len()` long and is reset to zero at the start
//! of every substep.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuCylindricalJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical two-slot
//! multiplier layout. Neither constraint takes an inverse cosine, so the two
//! paths need only match in finite-precision reassociation; the tight parity
//! tolerance bounds the result.
//!
//! Provenance: the axis-alignment (orthogonality) angular constraint shared with
//! the revolute hinge and the perpendicular point-on-line positional constraint
//! shared with the prismatic slider, with their substep `XPBD` handling (Müller
//! et al., "Detailed Rigid Body Simulation with XPBD"), over the world-space
//! inverse inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine
//! source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::cylindrical::CylindricalJoint;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};
use glam::Vec3;

/// Advances `state` by `dt` under the cylindrical joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's axis-alignment and point-on-line constraints every
/// substep. The joint set is coloured once up front so same-batch joints write
/// disjoint movable bodies; the batches are then solved in order
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
pub fn cpu_solve_joints_cylindrical(
    state: &mut RigidBodyState,
    joints: &[CylindricalJoint],
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

    let mut prev_positions = vec![Vec3::ZERO; state.len()];
    let mut prev_orientations = vec![glam::Quat::IDENTITY; state.len()];

    // Two Lagrange multipliers per joint: slot `2k` for the point-on-line weld,
    // slot `2k + 1` for the axis alignment. Reset to zero at the start of every
    // substep.
    let mut lambda = vec![0.0f32; 2 * ordered.len()];

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
                    solve_one(state, &ordered[k], h, &mut lambda[2 * k..2 * k + 2]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one cylindrical joint for a single sweep: the axis-alignment
/// correction first (accumulating `lambda[1]`), then the point-on-line
/// correction (accumulating `lambda[0]`), each applied directly to the two
/// bodies' transforms.
fn solve_one(state: &mut RigidBodyState, joint: &CylindricalJoint, h: f32, lambda: &mut [f32]) {
    solve_axis_alignment(state, joint, h, &mut lambda[1]);
    solve_point_on_line(state, joint, h, &mut lambda[0]);
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
    joint: &CylindricalJoint,
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

    state.orientations[a] = apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, p));
    state.orientations[b] = apply_rotation_delta(q_b, -world_inv_inertia_apply(q_b, ii_b, p));
}

/// Pins the two world-space anchors to a common line. With `dx` the anchor
/// separation and `axis_w = rotate(q_a, axis_a)` the world-space shared axis,
/// only the perpendicular component `dx_perp = dx - axis_w * dot(dx, axis_w)` is
/// cancelled; the along-axis component is left free, so the two anchors share a
/// line rather than a point and the slide along the axis stays free.
fn solve_point_on_line(
    state: &mut RigidBodyState,
    joint: &CylindricalJoint,
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use glam::{Quat, Vec3};

    /// Perpendicular distance from a joint's two world-space anchors to their
    /// shared axis line — zero when the point-on-line weld is satisfied.
    fn perpendicular_offset(state: &RigidBodyState, joint: &CylindricalJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        let axis_w = rotate(state.orientations[a], joint.axis_a).normalize();
        let dx = p_a - p_b;
        (dx - axis_w * dx.dot(axis_w)).length()
    }

    /// Angle (radians) between the two world-space axes — zero when the
    /// alignment constraint is satisfied.
    fn axis_angle(state: &RigidBodyState, joint: &CylindricalJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let u_a = rotate(state.orientations[a], joint.axis_a).normalize();
        let u_b = rotate(state.orientations[b], joint.axis_b).normalize();
        ops::asin(u_a.cross(u_b).length().clamp(-1.0, 1.0))
    }

    #[test]
    fn alignment_pulls_a_tilted_axis_back_to_parallel() {
        // Body 1's axis starts 30 degrees off body 0's; the alignment constraint
        // must drive it back toward parallel within a frame of a few sweeps.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // static mount
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let tilt = std::f32::consts::FRAC_PI_6;
        state.orientations[1] = Quat::from_axis_angle(Vec3::X, tilt);
        let joint = CylindricalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        let before = axis_angle(&state, &joint);
        assert!(
            (before - tilt).abs() < 0.02,
            "setup axis angle wrong: {before}"
        );
        cpu_solve_joints_cylindrical(&mut state, &[joint], &integrator, &config, 1.0 / 60.0)
            .unwrap();
        let after = axis_angle(&state, &joint);
        assert!(
            after < 0.05,
            "alignment constraint left {after} rad (want ~0)"
        );
    }

    #[test]
    fn point_on_line_pulls_an_offset_sleeve_back_onto_the_axis() {
        // Body 1's anchor starts offset perpendicular to the shared axis; the
        // point-on-line weld must cancel that perpendicular separation.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // static rod
        state.push(
            Vec3::new(0.3, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        // Shared axis +Y; the body-1 anchor sits off the rod line by 0.3 in +X.
        let joint = CylindricalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        let before = perpendicular_offset(&state, &joint);
        assert!(
            (before - 0.3).abs() < 1.0e-4,
            "setup offset wrong: {before}"
        );
        cpu_solve_joints_cylindrical(&mut state, &[joint], &integrator, &config, 1.0 / 60.0)
            .unwrap();
        let after = perpendicular_offset(&state, &joint);
        assert!(after < 1.0e-3, "point-on-line weld left offset {after}");
    }

    #[test]
    fn sleeve_stays_on_line_and_parallel_under_gravity() {
        // A dynamic sleeve on a static rod along +Y must keep its anchor on the
        // rod line and its axis parallel even as gravity drags it, while
        // remaining free to slide along and spin about the axis.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // rod
        state.push(
            Vec3::new(0.0, -1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = CylindricalJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            0.0,
            0.0,
        );
        // Gravity along +X so it pushes the sleeve *off* the +Y line.
        let integrator = IntegratorConfig::new(Vec3::new(9.81, 0.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_cylindrical(&mut state, &[joint], &integrator, &config, dt).unwrap();
            assert!(
                perpendicular_offset(&state, &joint) < 1.0e-3,
                "sleeve drifted off the rod line"
            );
            assert!(
                axis_angle(&state, &joint) < 2.0e-2,
                "sleeve axis drifted off parallel"
            );
        }
    }

    #[test]
    fn slide_along_the_axis_is_free() {
        // With no gravity and an initial velocity along the shared axis, the
        // sleeve must slide freely: the point-on-line weld removes only the
        // perpendicular separation, so the along-axis motion is untouched.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // rod
        state.push(
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[1] = Vec3::new(0.0, 1.0, 0.0);
        let joint = CylindricalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_cylindrical(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        // The sleeve should have travelled freely along +Y (roughly v * t).
        assert!(
            state.positions[1].y > 0.4,
            "slide along the axis was impeded: y = {}",
            state.positions[1].y
        );
        assert!(
            state.linear_velocities[1].y > 0.9,
            "along-axis velocity was damped: {}",
            state.linear_velocities[1].y
        );
    }

    #[test]
    fn spin_about_the_axis_is_free() {
        // With an initial angular velocity about the shared axis, the sleeve must
        // spin freely: the alignment constraint only removes the tilt
        // perpendicular to the axis, leaving the spin about it untouched.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // rod
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.angular_velocities[1] = Vec3::new(0.0, 2.0, 0.0);
        let joint = CylindricalJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y, 0.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_cylindrical(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        assert!(
            state.angular_velocities[1].y > 1.9,
            "spin about the axis was damped: {}",
            state.angular_velocities[1].y
        );
        // The axis stays parallel: spin about +Y does not tilt +Y.
        assert!(
            axis_angle(&state, &joint) < 1.0e-3,
            "spin tilted the axis off parallel"
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, both bodies given the same initial
        // velocity: the joint's internal corrections are equal and opposite, so
        // total linear momentum must be conserved.
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
        let joint = CylindricalJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_cylindrical(&mut state, &[joint], &integrator, &config, dt).unwrap();
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
        cpu_solve_joints_cylindrical(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], before);
    }
}
