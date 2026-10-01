//! Authoritative `CPU` golden reference for the rack-and-pinion joint stepper.
//!
//! [`cpu_solve_joints_rack_pinion`] is a *full stepper* with the identical
//! substep schedule as the other joint goldens: a caller hands it the current
//! [`RigidBodyState`], the joint set, the shared [`IntegratorConfig`], the
//! joint-specific [`JointSolverConfig`], and the frame `dt`, and must **not**
//! integrate the bodies itself. Within each integrator substep the stepper
//!
//! 1. snapshots every body's position and orientation,
//! 2. predicts the bodies forward under gravity and damping,
//! 3. resets every joint's single `XPBD` Lagrange multiplier,
//! 4. projects the joint constraint
//!    [`position_iterations`](super::config::JointSolverConfig::position_iterations)
//!    times, walking the colour batches in order, and
//! 5. recovers the linear and angular velocities from the net per-substep
//!    motion.
//!
//! # The single constraint
//!
//! A rack-and-pinion joint couples the spin of the pinion body about its world
//! axis `u_a = rotate(orientation_a, pinion_axis)` to the slide of the rack body
//! along its world axis `u_b = rotate(orientation_b, rack_axis)` at a fixed
//! ratio. It is one scalar mixed angular-linear constraint expressed in the
//! shared position-based stepper on the per-substep incremental motion:
//!
//! ```text
//! C = (u_a . dphi_a) - ratio * (u_b . dx_b)
//! ```
//!
//! where `dphi_a` is the pinion's angular displacement since the substep
//! snapshot (`angular_displacement`, the same small-angle extraction the
//! velocity recovery uses) and `dx_b` the rack's linear displacement since the
//! snapshot. Forcing `C -> 0` forces the recovered rates onto
//! `(u_a . omega_a) = ratio * (u_b . v_b)`.
//!
//! Each joint owns a single Lagrange multiplier in the shared `lambda` buffer at
//! slot `k`, where `k` is the joint's index in the colour-reordered list. The
//! buffer is therefore `joints.len()` long and is reset to zero at the start of
//! every substep.
//!
//! # The rack-and-pinion update
//!
//! The constraint gradients are `u_a` (angular) on the pinion and `-ratio * u_b`
//! (linear) on the rack. With
//! `w = u_a . (I_a^-1 u_a) + ratio^2 * inv_mass_b` the effective inverse mass
//! and `alpha_tilde = compliance / h^2` the regularisation, each sweep applies
//!
//! ```text
//! d_lambda = (-C - alpha_tilde * lambda) / (w + alpha_tilde)
//! ```
//!
//! then rotates the pinion by `I_a^-1 u_a d_lambda` and translates the rack by
//! `-ratio * inv_mass_b * u_b d_lambda`. A zero `compliance` collapses the
//! update to the rigid coupling `d_lambda = -C / w`, forcing the rate coupling
//! exactly each substep; a positive compliance applies a finite coupling force
//! so the ratio is satisfied asymptotically — a flexing transmission. The
//! pinion receives only an angular impulse and the rack only a linear one, which
//! are deliberately *not* a conserving action-reaction pair: the missing
//! reaction flows into each body's mounting frame, exactly as in a real
//! rack-and-pinion where the bearing and the rack guide carry it.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuRackPinionJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical single-slot
//! multiplier layout and read the same per-substep snapshot buffers for the
//! incremental motion terms. There is no transcendental in the rack-and-pinion
//! path — it reads only dot products of rotation and displacement vectors — so
//! the `CPU` reference and the shader share one arithmetic path with only device
//! reassociation separating them.
//!
//! Provenance: the rotation-to-translation ratio coupling expressed as a
//! per-substep compliant equality on the pinion's angular displacement and the
//! rack's linear displacement (Macklin et al., "XPBD: Position-Based Simulation
//! of Compliant Constrained Dynamics"), over the world-space inverse inertia,
//! inverse mass, and quaternion kinematics of Baraff & Witkin. No Unreal Engine
//! source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::math::{
    apply_rotation_delta, quat_array, quat_conj, quat_mul, rotate, world_inv_inertia_apply, EPSILON,
};
use super::rack_pinion::RackPinionJoint;
use super::stepper::{predict, recover_velocities, snapshot};
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the rack-and-pinion joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's coupling constraint every substep. The joint set is
/// coloured once up front so same-batch joints write disjoint movable bodies;
/// the batches are then solved in order
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
pub fn cpu_solve_joints_rack_pinion(
    state: &mut RigidBodyState,
    joints: &[RackPinionJoint],
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

    // One multiplier per joint: slot `k` is the rotation-to-translation
    // coupling. Reset to zero every substep.
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
                for k in start as usize..end as usize {
                    solve_one(
                        state,
                        &prev_positions,
                        &prev_orientations,
                        &ordered[k],
                        h,
                        &mut lambda[k],
                    );
                }
            }
        }
        recover_velocities(state, &prev_positions, &prev_orientations, inv_h);
    }

    Ok(())
}

/// Projects one rack-and-pinion joint for a single sweep: the
/// rotation-to-translation coupling of the pinion's spin and the rack's slide,
/// applied directly to the pinion's orientation and the rack's position.
fn solve_one(
    state: &mut RigidBodyState,
    prev_positions: &[Vec3],
    prev_orientations: &[Quat],
    joint: &RackPinionJoint,
    h: f32,
    lambda: &mut f32,
) {
    let a = joint.pinion as usize;
    let b = joint.rack as usize;

    let q_a = state.orientations[a];
    let q_b = state.orientations[b];

    let axis_a = rotate(q_a, joint.pinion_axis);
    let axis_b = rotate(q_b, joint.rack_axis);
    let len_a = axis_a.length();
    let len_b = axis_b.length();
    if len_a < EPSILON || len_b < EPSILON {
        return;
    }
    let u_a = axis_a / len_a;
    let u_b = axis_b / len_b;

    // Constraint gradients: `u_a` (angular) on the pinion and `-ratio * u_b`
    // (linear) on the rack. The ratio appears in the rack's gradient; the
    // signed correction magnitude is carried by `d_lambda`.
    let ratio = joint.ratio;

    let ii_a = state.inverse_inertias[a];
    let inv_m_b = state.inverse_masses[b];
    let w_a = u_a.dot(world_inv_inertia_apply(q_a, ii_a, u_a));
    let w_b = ratio * ratio * inv_m_b;
    let w = w_a + w_b;
    if w < EPSILON {
        return;
    }

    // Incremental motion since the substep snapshot: the pinion's angular
    // displacement about its spin axis and the rack's linear displacement along
    // its slide axis: `C = (u_a . dphi_a) - ratio * (u_b . dx_b)`.
    let dphi_a = angular_displacement(q_a, prev_orientations[a]);
    let dx_b = state.positions[b] - prev_positions[b];
    let c = u_a.dot(dphi_a) - ratio * u_b.dot(dx_b);

    let alpha_tilde = joint.compliance / (h * h);
    let d_lambda = (-c - alpha_tilde * *lambda) / (w + alpha_tilde);
    *lambda += d_lambda;

    // Pinion: angular impulse `u_a * d_lambda`. Rack: linear impulse
    // `-ratio * u_b * d_lambda`, scaled by the rack's inverse mass.
    state.orientations[a] =
        apply_rotation_delta(q_a, world_inv_inertia_apply(q_a, ii_a, u_a * d_lambda));
    state.positions[b] += u_b * (-ratio * inv_m_b * d_lambda);
}

/// Relative rotation of a body since the substep snapshot, as a rotation vector:
/// twice the imaginary part of the delta quaternion `orientation *
/// conj(prev_orientation)`, hemisphere-corrected so the shortest arc is taken.
/// The rack-and-pinion joint dots this with the pinion axis to read the relative
/// angular rate accumulated over the substep.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Angular velocity of a body about a given world axis.
    fn rate_about(state: &RigidBodyState, body: usize, axis: Vec3) -> f32 {
        axis.normalize().dot(state.angular_velocities[body])
    }

    /// Linear velocity of a body along a given world axis.
    fn slide_rate(state: &RigidBodyState, body: usize, axis: Vec3) -> f32 {
        axis.normalize().dot(state.linear_velocities[body])
    }

    /// A pinion spinning about `+Z` (body 0) coupled to a rack sliding along
    /// `+X` (body 1). The pinion is spun up by an initial angular velocity; the
    /// joint ties its spin to the rack's slide at the given ratio.
    fn rack_pinion_pair(ratio: f32, spin: f32) -> (RigidBodyState, RackPinionJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 0: pinion
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: rack
        state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin);
        let joint = RackPinionJoint::rigid(0, 1, Vec3::Z, Vec3::X, ratio);
        (state, joint)
    }

    #[test]
    fn rigid_coupling_locks_rotation_to_translation() {
        // A rigid rack-and-pinion forces `(u_a . omega_a) = ratio * (u_b . v_b)`,
        // so the rack slides at `v = omega / ratio` along X.
        let ratio = 2.0;
        let spin = 1.5;
        let (mut state, joint) = rack_pinion_pair(ratio, spin);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_rack_pinion(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let wa = rate_about(&state, 0, Vec3::Z);
        let vb = slide_rate(&state, 1, Vec3::X);
        let residual = wa - ratio * vb;
        assert!(
            residual.abs() < 1.0e-2,
            "rack-and-pinion coupling violated: wa - ratio*vb = {residual} (wa = {wa}, vb = {vb})"
        );
        // The coupling must be live: the rack should be sliding.
        assert!(vb.abs() > 1.0e-2, "rack failed to slide: vb = {vb}");
    }

    #[test]
    fn negative_ratio_reverses_slide() {
        // A negative ratio reverses the slide direction relative to the spin.
        let ratio = -2.0;
        let spin = 1.5;
        let (mut state_pos, joint_pos) = rack_pinion_pair(2.0, spin);
        let (mut state_neg, joint_neg) = rack_pinion_pair(ratio, spin);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..30 {
            cpu_solve_joints_rack_pinion(&mut state_pos, &[joint_pos], &integrator, &config, dt)
                .unwrap();
            cpu_solve_joints_rack_pinion(&mut state_neg, &[joint_neg], &integrator, &config, dt)
                .unwrap();
        }
        let vb_pos = slide_rate(&state_pos, 1, Vec3::X);
        let vb_neg = slide_rate(&state_neg, 1, Vec3::X);
        assert!(
            vb_pos * vb_neg < 0.0,
            "negative ratio should reverse the slide: vb_pos = {vb_pos}, vb_neg = {vb_neg}"
        );
    }

    #[test]
    fn static_pinion_locks_rack() {
        // Body 0 static (a locked pinion shaft), body 1 a dynamic rack with an
        // initial slide: the rigid coupling against a static pinion forces the
        // rack's slide to the coupled value `omega / ratio = 0`, braking it.
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: static pinion
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: dynamic rack
        state.linear_velocities[1] = Vec3::new(2.0, 0.0, 0.0);
        let joint = RackPinionJoint::rigid(0, 1, Vec3::Z, Vec3::X, 2.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..60 {
            cpu_solve_joints_rack_pinion(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let vb = slide_rate(&state, 1, Vec3::X);
        assert!(
            vb.abs() < 1.0e-2,
            "static pinion failed to brake the rack: vb = {vb}"
        );
    }

    #[test]
    fn soft_coupling_drives_rack_gradually() {
        // A soft coupling applies a finite force, so after a short run the rack
        // has begun to slide but has not reached the rigid coupled rate.
        let ratio = 2.0;
        let spin = 2.0;
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0)); // 0: pinion
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: rack
        state.angular_velocities[0] = Vec3::new(0.0, 0.0, spin);
        let joint = RackPinionJoint::soft(0, 1, Vec3::Z, Vec3::X, ratio, 50.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..3 {
            cpu_solve_joints_rack_pinion(&mut state, &[joint], &integrator, &config, dt).unwrap();
        }
        let vb = slide_rate(&state, 1, Vec3::X);
        assert!(
            vb > 1.0e-3,
            "soft coupling should begin to drive the rack: vb = {vb}"
        );
    }

    #[test]
    fn empty_joint_set_is_a_no_op() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let before = state.positions[0];
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_rack_pinion(&mut state, &[], &integrator, &config, 1.0 / 60.0).unwrap();
        assert_eq!(state.positions[0], before);
    }

    #[test]
    fn joint_referencing_missing_body_errors() {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 1.0, Vec3::splat(1.0));
        let joint = RackPinionJoint::rigid(0, 5, Vec3::Z, Vec3::X, 1.0);
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        let err =
            cpu_solve_joints_rack_pinion(&mut state, &[joint], &integrator, &config, 1.0 / 60.0);
        assert!(matches!(err, Err(RigidError::InconsistentState { .. })));
    }
}
