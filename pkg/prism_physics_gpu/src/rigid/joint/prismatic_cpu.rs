//! Authoritative `CPU` golden reference for the prismatic (slider) joint stepper.
//!
//! [`cpu_solve_joints_prismatic`] is a *full stepper* with the identical substep
//! schedule as [`cpu_solve_joints_revolute`](super::cpu_solve_joints_revolute):
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
//! # The two constraints, projected angle-first
//!
//! A prismatic joint locks the relative orientation of the two bodies to a fixed
//! rest offset and confines their anchors to a shared slide line. Each sweep
//! projects the **angular lock** first — cancelling the world-space error
//! between body `b`'s orientation and its target `q_a * rest_rotation` — then the
//! **perpendicular weld**, the point-to-point correction restricted to the plane
//! perpendicular to the world-space slide axis. Projecting the orientation
//! before the anchors means the perpendicular pass works against an
//! already-oriented slide axis.
//!
//! Each joint owns two Lagrange multipliers in the shared `lambda` buffer: slot
//! `2 * k` for the perpendicular weld and slot `2 * k + 1` for the angular lock,
//! where `k` is the joint's index in the colour-reordered list. The buffer is
//! therefore `2 * joints.len()` long and is reset to zero at the start of every
//! substep.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuPrismaticJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical two-slot
//! multiplier layout. The angular lock uses the array-based `quat_mul` /
//! `quat_conj` helpers (not glam's `Quat` operators) so the `CPU` reference and
//! the shader share one component order and stay in lock-step.
//!
//! Provenance: the point-to-point (ball-socket) constraint restricted to the
//! plane perpendicular to the slide axis and the relative-orientation lock with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. No Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::prismatic::PrismaticJoint;
use super::stepper::{predict, recover_velocities, snapshot};
use glam::Vec3;

/// Advances `state` by `dt` under the prismatic joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's angular-lock and perpendicular-weld constraints every
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
pub fn cpu_solve_joints_prismatic(
    state: &mut RigidBodyState,
    joints: &[PrismaticJoint],
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

    // Two Lagrange multipliers per joint: slot `2k` for the perpendicular weld,
    // slot `2k + 1` for the angular lock. Reset to zero at the start of every
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

/// Projects one prismatic joint for a single sweep: the angular-lock correction
/// first (accumulating `lambda[1]`), then the perpendicular-weld positional
/// correction (accumulating `lambda[0]`), each applied directly to the two
/// bodies' transforms.
fn solve_one(state: &mut RigidBodyState, joint: &PrismaticJoint, h: f32, lambda: &mut [f32]) {
    solve_angular_lock(state, joint, h, &mut lambda[1]);
    solve_perpendicular(state, joint, h, &mut lambda[0]);
}

/// Drives body `b`'s world orientation to its target `q_a * rest_rotation`,
/// locking all three relative rotational degrees of freedom. The world-space
/// error rotation is `target * conj(q_b)`; its imaginary part (times two) is the
/// rotation vector that carries `b` back onto the target, and the equal and
/// opposite impulse rotates `a`'s target to meet it.
fn solve_angular_lock(
    state: &mut RigidBodyState,
    joint: &PrismaticJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    // Target world orientation of body `b`, and the world-space error rotation
    // that carries `b` onto it: `error = target * conj(q_b)`. Computed with the
    // array-based Hamilton product so the shader reproduces it component for
    // component.
    let target = quat_mul(quat_array(q_a), quat_array(joint.rest_rotation));
    let mut error = quat_mul(target, quat_conj(quat_array(q_b)));
    // Take the shortest-arc representative: `q` and `-q` are the same rotation,
    // but only the hemisphere with a non-negative scalar part gives the small
    // rotation vector the linearisation assumes.
    if error[3] < 0.0 {
        error = [-error[0], -error[1], -error[2], -error[3]];
    }

    // The rotation vector is twice the imaginary part of the error quaternion.
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
    joint: &PrismaticJoint,
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
    use glam::{Quat, Vec3};

    /// Separation between a joint's two world-space anchors, split into the
    /// component along the world-space slide axis and the perpendicular
    /// component.
    fn anchor_separation(state: &RigidBodyState, joint: &PrismaticJoint) -> (f32, f32) {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        let dx = p_a - p_b;
        let axis_w = rotate(state.orientations[a], joint.axis_a).normalize();
        let along = dx.dot(axis_w);
        let perp = (dx - axis_w * along).length();
        (along.abs(), perp)
    }

    /// Sine of the angle between body `b`'s orientation and its target
    /// `q_a * rest_rotation`: the magnitude of the imaginary part of the relative
    /// error quaternion is `sin(theta / 2)`, a monotone misalignment measure that
    /// avoids an inverse trigonometric call.
    fn orientation_error_sin_half(state: &RigidBodyState, joint: &PrismaticJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let target = quat_mul(
            quat_array(state.orientations[a]),
            quat_array(joint.rest_rotation),
        );
        let error = quat_mul(target, quat_conj(quat_array(state.orientations[b])));
        Vec3::new(error[0], error[1], error[2]).length()
    }

    #[test]
    fn slider_permits_axial_translation() {
        // A dynamic body sliding along the world Y axis under gravity, anchored
        // to a static base. The along-axis separation must grow as it falls while
        // the perpendicular separation and the orientation stay locked.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let joint = PrismaticJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Quat::IDENTITY,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_prismatic(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let (_, perp) = anchor_separation(&state, &joint);
            assert!(perp < 1.0e-3, "slider drifted off its axis: perp {perp}");
            assert!(
                orientation_error_sin_half(&state, &joint) < 1.0e-2,
                "slider orientation drifted"
            );
        }
        // Gravity is along -Y, the slide axis, so the body must have slid down.
        assert!(state.positions[1].y < -0.05, "slider never slid down");
    }

    #[test]
    fn slider_blocks_perpendicular_translation() {
        // A dynamic body kicked sideways (perpendicular to its slide axis) must
        // be pulled back onto the axis line within a frame of a few sweeps.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(
            Vec3::new(0.3, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = PrismaticJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Quat::IDENTITY,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        let (_, before) = anchor_separation(&state, &joint);
        assert!(
            (before - 0.3).abs() < 1.0e-5,
            "setup perpendicular offset wrong"
        );
        cpu_solve_joints_prismatic(&mut state, &[joint], &integrator, &config, 1.0 / 60.0).unwrap();
        let (_, after) = anchor_separation(&state, &joint);
        assert!(after < 0.05, "perpendicular offset left {after} of error");
    }

    #[test]
    fn slider_locks_relative_rotation() {
        // A dynamic body whose orientation starts 90 degrees off its rest target
        // must be driven back toward the rest orientation by the rigid angular
        // lock within a frame of a few sweeps.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        // Body 1 is rotated 90 degrees about Z; the rest offset is identity, so
        // the lock must undo that rotation.
        state.orientations[1] = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2);
        let joint = PrismaticJoint::new(
            0,
            1,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::Y,
            Quat::IDENTITY,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(8);

        // sin(45 deg) = sqrt(2)/2 ~= 0.707 is the initial half-angle sine.
        let before = orientation_error_sin_half(&state, &joint);
        assert!(
            (before - 0.707).abs() < 0.02,
            "setup orientation error wrong: {before}"
        );
        cpu_solve_joints_prismatic(&mut state, &[joint], &integrator, &config, 1.0 / 60.0).unwrap();
        let after = orientation_error_sin_half(&state, &joint);
        assert!(
            after < 0.2,
            "angular lock left sin(theta/2) = {after} of error"
        );
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, one body given an initial velocity: the
        // joint's internal corrections are equal and opposite, so total linear
        // momentum must be conserved.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[0] = Vec3::new(0.5, 0.2, -0.1);
        let joint = PrismaticJoint::new(
            0,
            1,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::Y,
            Quat::IDENTITY,
            0.0,
            0.0,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_prismatic(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        assert!(
            (final_momentum - initial).length() < 1.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }
}
