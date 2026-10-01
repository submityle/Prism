//! Authoritative `CPU` golden reference for the distance (limit) joint stepper.
//!
//! [`cpu_solve_joints_distance`] is a *full stepper*: it owns the whole frame
//! advance, so a caller hands it the current [`RigidBodyState`], the joint set,
//! the shared [`IntegratorConfig`], the joint-specific [`JointSolverConfig`],
//! and the frame `dt`, and must **not** integrate the bodies itself. Within each
//! integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping (semi-implicit
//!    Euler for translation, the quaternion kinematic equation for rotation),
//! 3. resets each joint's `XPBD` Lagrange multiplier,
//! 4. projects the joint constraints
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order so same-batch joints touch
//!    disjoint movable bodies, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! This is the substep-`XPBD` scheme of Müller et al., "Detailed Rigid Body
//! Simulation with XPBD": convergence comes from many small substeps rather than
//! many solver iterations, and each positional constraint is projected directly
//! against the two bodies' world-space inverse inertia.
//!
//! # The limit constraint
//!
//! Each joint measures the length `d` of the world-space separation of its two
//! anchors and compares it against the configured range. The signed constraint
//! value is `d - min_distance` when `d < min_distance` (push apart),
//! `d - max_distance` when `d > max_distance` (pull together), and the joint is
//! skipped entirely when `d` lies inside `[min_distance, max_distance]`, so the
//! bodies move freely between the bounds. The correction direction is the unit
//! separation axis `n = (p_a - p_b) / d`, shared by both the lower and upper
//! limit; the signed value simply selects which way the single `XPBD`
//! multiplier drives the pair.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuDistanceJointSolver`) and its shader. The quaternion helpers use the
//! same expanded sandwich product and Hamilton product as the rigid integrator
//! and contact solver so the three stages compose without drift. The `GPU`
//! kernel walks the identical reordered joint list and colour-batch order, which
//! is what lets the device reproduce this trajectory frame for frame.
//!
//! Provenance: the point-to-point distance constraint and its substep `XPBD`
//! positional handling (Müller et al.), with the one-sided limit / dead-zone
//! treatment standard to distance joints, over the world-space inverse inertia
//! and quaternion kinematics of Baraff & Witkin. No Unreal Engine source or
//! derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::distance::DistanceJoint;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};

/// Advances `state` by `dt` under the distance joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects the joint limit constraints each substep. The joint set is coloured
/// once up front so same-batch joints write disjoint movable bodies; the batches
/// are then solved in order
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
pub fn cpu_solve_joints_distance(
    state: &mut RigidBodyState,
    joints: &[DistanceJoint],
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
    let mut lambda = vec![0.0f32; ordered.len()];

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
                for index in start as usize..end as usize {
                    solve_one(state, &ordered[index], h, &mut lambda[index]);
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one distance joint for a single sweep, applying the `XPBD`
/// positional correction that drives the violated separation bound shut and
/// accumulating the joint's Lagrange multiplier in `lambda`. When the current
/// separation lies within `[min_distance, max_distance]` the joint is inactive
/// and the function returns without touching the bodies or the multiplier.
fn solve_one(state: &mut RigidBodyState, joint: &DistanceJoint, h: f32, lambda: &mut f32) {
    let a = joint.body_a as usize;
    let b = joint.body_b as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];
    let r_a = rotate(q_a, joint.anchor_a);
    let r_b = rotate(q_b, joint.anchor_b);
    let dx = (state.positions[a] + r_a) - (state.positions[b] + r_b);
    let dist = dx.length();
    if dist < EPSILON {
        return;
    }
    let n = dx / dist;

    // Signed limit violation with a free dead zone between the two bounds.
    let c = if dist < joint.min_distance {
        dist - joint.min_distance
    } else if dist > joint.max_distance {
        dist - joint.max_distance
    } else {
        return;
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

    /// Separation between a joint's two world-space anchors in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &DistanceJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let p_a = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let p_b = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (p_a - p_b).length()
    }

    #[test]
    fn rigid_rod_holds_fixed_length_under_gravity() {
        // Body 0 is a static pivot at the origin; body 1 hangs a unit rod below
        // it. The rigid rod must keep the length at 1 m while the body swings.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = DistanceJoint::rigid_rod(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_distance(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let d = anchor_separation(&state, &joint);
            assert!((d - 1.0).abs() < 2.0e-3, "rod length drifted to {d}");
        }
        assert!(state.positions[1].y < -0.05, "pendulum never swung down");
    }

    #[test]
    fn rope_lets_slack_pair_approach_freely() {
        // Two dynamic bodies 0.5 m apart under a rope limit of 2 m: the pair is
        // well inside the limit, so with zero gravity and no initial velocity
        // nothing should move.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(0.5, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = DistanceJoint::rope(0, 1, Vec3::ZERO, Vec3::ZERO, 2.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_distance(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        assert!((state.positions[0] - Vec3::ZERO).length() < 1.0e-5);
        assert!((state.positions[1] - Vec3::new(0.5, 0.0, 0.0)).length() < 1.0e-5);
    }

    #[test]
    fn rope_catches_a_falling_body_at_its_length() {
        // Body 0 is a static anchor at the origin; body 1 starts coincident and
        // falls under gravity. A 1 m rope must arrest it near 1 m of drop and
        // never let it fall significantly past the limit.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO);
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let joint = DistanceJoint::rope(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..240 {
            cpu_solve_joints_distance(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let d = anchor_separation(&state, &joint);
            assert!(d < 1.0 + 5.0e-3, "rope stretched past its limit to {d}");
        }
        // It should actually have reached and be hanging at the limit.
        assert!(
            anchor_separation(&state, &joint) > 0.9,
            "body never descended to the rope limit"
        );
    }

    #[test]
    fn min_limit_pushes_a_close_pair_apart() {
        // Two dynamic bodies 0.2 m apart with a free dead zone of [1 m, 2 m]:
        // the lower limit must push the overlapping pair apart. Because the
        // dead zone exerts no force and there is no gravity or damping, the
        // outward velocity gained from the push coasts the pair across the gap
        // until the upper limit catches it. The physical invariants are that
        // the minimum separation is never violated and the pair never escapes
        // past the maximum.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(0.2, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        let joint = DistanceJoint::new(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0, 2.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_distance(&mut state, &[joint], &integrator, &config, dt).unwrap();
            let d = anchor_separation(&state, &joint);
            assert!(
                d >= 1.0 - 5.0e-3,
                "min limit was violated: pair at {d} < 1 m"
            );
            assert!(d <= 2.0 + 5.0e-3, "pair escaped past the max limit to {d}");
        }
    }

    #[test]
    fn free_pair_conserves_linear_momentum() {
        // Zero gravity, equal masses, one body kicked outward past the max limit:
        // the joint's internal corrections are equal and opposite, so total
        // linear momentum must be conserved.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        state.push(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        );
        state.linear_velocities[1] = Vec3::new(0.8, 0.2, -0.1);
        let joint = DistanceJoint::rigid_rod(0, 1, Vec3::ZERO, Vec3::ZERO, 1.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..60 {
            cpu_solve_joints_distance(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let final_momentum = state.linear_velocities[0] + state.linear_velocities[1];
        // The rod applies central, equal-and-opposite impulses, so total linear
        // momentum is conserved up to f32 solver noise accumulated over the sweep.
        assert!(
            (final_momentum - initial).length() < 5.0e-3,
            "linear momentum drifted by {}",
            (final_momentum - initial).length()
        );
    }
}
