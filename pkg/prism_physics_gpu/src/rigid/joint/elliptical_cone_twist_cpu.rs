//! Authoritative `CPU` golden reference for the elliptical cone-twist joint
//! stepper.
//!
//! [`cpu_solve_joints_elliptical_cone_twist`] is a *full stepper* with the
//! identical substep schedule as the other joint goldens: a caller hands it the
//! current [`RigidBodyState`], the joint set, the shared [`IntegratorConfig`],
//! the joint-specific [`JointSolverConfig`], and the frame `dt`, and must **not**
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
//! An elliptical cone-twist joint welds the two bodies' anchors together, bounds
//! the lateral tilt of the twist axis within an *elliptical* cone (two
//! independent swing half-angles), and bounds the axial spin within a twist
//! range. Each sweep projects, in order: the **elliptical swing cone** (closing
//! any over-rim tilt of body `b`'s twist axis), then the **twist limit**
//! (clamping the signed spin about the freshly measured twist axis), then the
//! **point-to-point** positional weld (numerically identical to the spherical
//! golden's `solve_one`).
//!
//! Each joint owns three Lagrange multipliers in the shared `lambda` buffer:
//! slot `3 * k` for the positional weld, slot `3 * k + 1` for the elliptical
//! swing cone, and slot `3 * k + 2` for the twist limit, where `k` is the
//! joint's index in the colour-reordered list. The buffer is therefore
//! `3 * joints.len()` long and is reset to zero at the start of every substep.
//!
//! # The elliptical swing cone
//!
//! Let `t = rotate(q_a, twist_axis_a)` be body `a`'s unit twist axis and
//! `u_b = rotate(q_b, twist_axis_b)` body `b`'s. The swing plane spanned by
//! `(s1, s2)` is derived from body `a`'s reference direction: `s1` is `ref_a`
//! rotated to world and projected perpendicular to `t` (then normalised), and
//! `s2 = t x s1`. Decomposing `u_b = (x, y, z)` in the orthonormal frame
//! `(t, s1, s2)`, the tilt magnitude is `phi = acos(clamp(x, -1, 1))` and its
//! azimuth within the swing plane has `cos psi = y / rho`, `sin psi = z / rho`
//! with `rho = sqrt(y^2 + z^2)`. The elliptical rim at that azimuth is
//! `phi_max = 1 / sqrt((cos psi / swing1_limit)^2 + (sin psi / swing2_limit)^2)`,
//! which equals `swing1_limit` toward `+/- s1`, `swing2_limit` toward
//! `+/- s2`, and interpolates smoothly between. While `phi <= phi_max` the cone
//! is inactive; past it the violation `c = phi - phi_max` is driven shut by a
//! rotation about `n = normalize(t x u_b)` — the axis that closes the tilt
//! radially while holding the azimuth (and hence `phi_max`) fixed — with the
//! gradient `-n` on body `a` and `+n` on body `b`, exactly as the circular cone
//! closes its rim. When `swing1_limit == swing2_limit` the rim is circular and
//! this recovers the circular cone-twist exactly.
//!
//! # Why the radial azimuth projection is exact
//!
//! Rotating `u_b` about `n = normalize(t x u_b)` toward `t` reduces `phi` while
//! keeping `u_b` in the plane spanned by `(t, u_b)`; its projection onto the
//! `(s1, s2)` swing plane keeps its *direction* (the azimuth `psi`) and only
//! shrinks in magnitude. Because `phi_max(psi)` depends on the azimuth alone, it
//! is invariant under this correction, so the constraint `C = phi - phi_max(psi)`
//! is reduced purely through its `phi` term — the same stable single-axis
//! angular projection the circular cone uses, now with an azimuth-dependent rim.
//!
//! # Measuring the twist angle
//!
//! The twist limit reuses the hinge limit's signed-angle machinery verbatim.
//! Each body carries a body-local reference direction, `ref_a` and `ref_b`, both
//! rotated to world space, projected onto the plane perpendicular to the unit
//! twist axis `u = rotate(q_a, twist_axis_a)`, and normalised; the signed angle
//! from `a`'s projection to `b`'s projection about `u` is
//! `theta = atan2((p_a x p_b) . u, p_a . p_b)`, clamped into
//! `[twist_min, twist_max]` with a free dead zone.
//!
//! # Parity contract
//!
//! Every arithmetic step here is mirrored by the `GPU` twin
//! (`GpuEllipticalConeTwistJointSolver`) and its shader, which walk the identical
//! reordered joint list and colour-batch order with the identical three-slot
//! multiplier layout. The transcendental angles are taken with
//! [`bevy_math::ops::acos`] and [`bevy_math::ops::atan2`] rather than the `f32`
//! intrinsics, and the elliptical `phi_max` uses an explicit `sqrt` and division
//! (never a reciprocal-sqrt intrinsic), so the `CPU` reference and the shader
//! share one transcendental path.
//!
//! Provenance: the point-to-point (ball-socket) constraint, the signed angular
//! limit shared with the hinge limit, and the swing-cone limit generalised to an
//! elliptical rim, with their substep `XPBD` handling (Müller et al., "Detailed
//! Rigid Body Simulation with XPBD"), over the world-space inverse inertia and
//! quaternion kinematics of Baraff & Witkin. The elliptical two-angle swing
//! parametrisation follows the standard cone-twist of production engines. No
//! Unreal Engine source or derived code.

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::elliptical_cone_twist::EllipticalConeTwistJoint;
use super::math::{apply_rotation_delta, rotate, world_inv_inertia_apply, EPSILON};
use super::stepper::{predict, recover_velocities, snapshot};
use bevy_math::ops;
use glam::{Quat, Vec3};

/// Advances `state` by `dt` under the elliptical cone-twist joints in `joints`.
///
/// The stepper integrates the bodies on exactly the same schedule as the rest
/// of the crate (gravity, substep count, and damping from `integrator`), then
/// projects each joint's elliptical swing-cone, twist-limit, and
/// point-to-point constraints every substep. The joint set is coloured once up
/// front so same-batch joints write disjoint movable bodies; the batches are
/// then solved in order
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
pub fn cpu_solve_joints_elliptical_cone_twist(
    state: &mut RigidBodyState,
    joints: &[EllipticalConeTwistJoint],
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

    // Three multipliers per joint: `[3k]` positional weld, `[3k + 1]` elliptical
    // swing cone, `[3k + 2]` twist limit. Reset to zero every substep.
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

/// Projects one elliptical cone-twist joint for a single sweep: elliptical swing
/// cone first (accumulating `lambda[1]`), then the twist limit (`lambda[2]`),
/// then the point-to-point weld (`lambda[0]`), each applied directly to the two
/// bodies' transforms.
fn solve_one(
    state: &mut RigidBodyState,
    joint: &EllipticalConeTwistJoint,
    h: f32,
    lambda: &mut [f32],
) {
    solve_elliptical_swing(state, joint, h, &mut lambda[1]);
    solve_twist_limit(state, joint, h, &mut lambda[2]);
    solve_point_to_point(state, joint, h, &mut lambda[0]);
}

/// Closes an over-rim tilt of body `b`'s world-space twist axis within the
/// elliptical cone. With `t = rotate(q_a, twist_axis_a)` the swing axis of body
/// `a`, `u_b = rotate(q_b, twist_axis_b)` body `b`'s, and the swing plane
/// `(s1, s2)` derived from `ref_a` (where `s1 = normalize(ref_a - t (t . ref_a))`
/// and `s2 = t x s1`), the tilt magnitude `phi = acos(clamp(t . u_b, -1, 1))` is
/// left free while `phi <= phi_max(psi)`, the elliptical rim at the azimuth
/// `psi` of `u_b` within `(s1, s2)`. Past it the violation `phi - phi_max` is
/// driven to zero by a rotation about `n = normalize(t x u_b)`, with the
/// gradient `-n` on body `a` and `+n` on body `b` (rotating `a` about `+n`
/// closes the cone, rotating `b` about `+n` widens it). Rotating about `n`
/// holds the azimuth fixed, so `phi_max(psi)` is invariant and the constraint is
/// reduced purely through `phi`.
fn solve_elliptical_swing(
    state: &mut RigidBodyState,
    joint: &EllipticalConeTwistJoint,
    h: f32,
    lambda: &mut f32,
) {
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
    // `t`: body `a`'s unit twist axis (cone axis); `u_b`: body `b`'s unit twist
    // axis, whose tilt from `t` the elliptical rim bounds.
    let t = axis_a / len_a;
    let u_b = axis_b / len_b;

    // Swing-plane basis from `ref_a` projected perpendicular to `t`: `s1` is the
    // ellipse's first (swing1) axis and the twist limit's zero reference, and
    // `s2 = t x s1` the second (swing2).
    let r = rotate(q_a, joint.ref_a);
    let s1_raw = r - t * t.dot(r);
    let s1_len = s1_raw.length();
    if s1_len < EPSILON {
        return;
    }
    let s1 = s1_raw / s1_len;
    let s2 = t.cross(s1);

    // Decompose `u_b` in the orthonormal swing frame `(t, s1, s2)`.
    let x = u_b.dot(t);
    let y = u_b.dot(s1);
    let z = u_b.dot(s2);
    let phi = ops::acos(x.clamp(-1.0, 1.0));

    // Azimuth of the tilt within the `(s1, s2)` plane.
    let rho = (y * y + z * z).sqrt();
    if rho < EPSILON {
        return;
    }
    let cos_psi = y / rho;
    let sin_psi = z / rho;

    // Both swing half-angles must be positive; a locked swing axis belongs to a
    // general 6-DOF joint rather than an elliptical cone.
    if joint.swing1_limit <= EPSILON || joint.swing2_limit <= EPSILON {
        return;
    }
    // Elliptical rim half-angle at this azimuth:
    // `phi_max = 1 / sqrt((cos psi / s1_lim)^2 + (sin psi / s2_lim)^2)`.
    let e1 = cos_psi / joint.swing1_limit;
    let e2 = sin_psi / joint.swing2_limit;
    let inv = e1 * e1 + e2 * e2;
    let phi_max = 1.0 / inv.sqrt();
    if phi <= phi_max {
        return;
    }
    let c = phi - phi_max;

    // Rotation axis that closes the tilt radially (azimuth-preserving); equals
    // the circular cone's `normalize(u_a x u_b)` with `u_a = t`.
    let delta = t.cross(u_b);
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

    // `u_b` is `t` rotated by `+phi` about `n = normalize(t x u_b)`, so rotating
    // body `b` about `+n` *widens* the cone and rotating body `a` about `+n`
    // *closes* it. The constraint gradients are therefore `-n` on body `a` and
    // `+n` on body `b` — the same antisymmetric pattern the circular cone and the
    // twist limit use — so the over-rim violation is driven shut (not open).
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
    joint: &EllipticalConeTwistJoint,
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
    joint: &EllipticalConeTwistJoint,
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

    /// Signed twist angle (radians) of `joint` in `state`, measured the same way
    /// the solver measures it.
    fn twist_angle(state: &RigidBodyState, joint: &EllipticalConeTwistJoint) -> f32 {
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

    /// Swing angle (radians) `phi` between the two world-space twist axes — the
    /// tilt magnitude the elliptical rim bounds.
    fn swing_angle(state: &RigidBodyState, joint: &EllipticalConeTwistJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let t = rotate(state.orientations[a], joint.twist_axis_a).normalize();
        let u_b = rotate(state.orientations[b], joint.twist_axis_b).normalize();
        ops::acos((t.dot(u_b)).clamp(-1.0, 1.0))
    }

    /// World-space anchor separation of `joint` in `state`.
    fn anchor_separation(state: &RigidBodyState, joint: &EllipticalConeTwistJoint) -> f32 {
        let a = joint.body_a as usize;
        let b = joint.body_b as usize;
        let pa = state.positions[a] + rotate(state.orientations[a], joint.anchor_a);
        let pb = state.positions[b] + rotate(state.orientations[b], joint.anchor_b);
        (pa - pb).length()
    }

    /// A dynamic `+Y` limb socketed off a static pivot: body `a` (index 0) is the
    /// static socket at the origin with twist axis `+Y` and reference `+X`, so
    /// the ellipse's first swing axis `s1` is `+X` and `s2 = Y x X = -Z`; body
    /// `b` (index 1) is the dynamic limb above it, anchored back through its
    /// `-Y` end to the socket. The two swing half-angles and the symmetric twist
    /// range are configurable.
    fn socketed_limb(
        swing1_limit: f32,
        swing2_limit: f32,
        twist_limit: f32,
    ) -> (RigidBodyState, EllipticalConeTwistJoint) {
        let mut state = RigidBodyState::new();
        state.push(Vec3::ZERO, Quat::IDENTITY, 0.0, Vec3::ZERO); // 0: socket
        state.push(
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
            Vec3::splat(1.0),
        ); // 1: limb
        let joint = EllipticalConeTwistJoint::symmetric_cone(
            0,
            1,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            swing1_limit,
            swing2_limit,
            twist_limit,
        );
        (state, joint)
    }

    #[test]
    fn free_within_elliptical_cone_lets_the_limb_swing() {
        // A wide elliptical cone with gravity pulling the +Y limb toward +X (the
        // s1 axis): the limb swings a meaningful amount while staying inside the
        // cone and the anchor never drifts apart.
        let (mut state, joint) = socketed_limb(FRAC_PI_2, FRAC_PI_2, FRAC_PI_2);
        let integrator = IntegratorConfig::new(Vec3::new(1.0, 0.0, 0.0), 8, 0.0, 0.0);
        let config = JointSolverConfig::new(4);
        let dt = 1.0 / 60.0;

        for _ in 0..40 {
            cpu_solve_joints_elliptical_cone_twist(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
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
    fn major_axis_rim_arrests_the_swing() {
        // Pull toward +X = +s1 against a tight swing1 half-angle but a wide
        // swing2: the limb is dragged out along the ellipse's major reference
        // direction and, as angular damping bleeds off the swing velocity,
        // settles against the swing1 rim rather than tipping past it.
        let swing1 = 0.3;
        let (mut state, joint) = socketed_limb(swing1, FRAC_PI_2, FRAC_PI_2);
        let integrator = IntegratorConfig::new(Vec3::new(1.5, 0.0, 0.0), 8, 0.0, 4.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..400 {
            cpu_solve_joints_elliptical_cone_twist(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
        }

        // The pull is along s1, so the rim here is exactly `swing1`.
        let swing = swing_angle(&state, &joint);
        assert!(
            swing < swing1 + 5.0e-3,
            "limb settled past the swing1 rim at {swing}"
        );
        assert!(
            swing > swing1 - 2.0e-2,
            "limb never reached its swing1 rim: swing = {swing}"
        );
    }

    #[test]
    fn minor_axis_rim_arrests_the_swing() {
        // Pull toward +Z (the `s2 = -Z` axis) against a tight swing2 half-angle
        // but a wide swing1: the limb is dragged along the ellipse's minor axis
        // and settles against the swing2 rim. This exercises an azimuth
        // orthogonal to the swing1 test, so the azimuth-dependent `phi_max` is
        // validated at both ellipse extremes.
        let swing2 = 0.3;
        let (mut state, joint) = socketed_limb(FRAC_PI_2, swing2, FRAC_PI_2);
        let integrator = IntegratorConfig::new(Vec3::new(0.0, 0.0, 1.5), 8, 0.0, 4.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..400 {
            cpu_solve_joints_elliptical_cone_twist(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
            assert!(
                anchor_separation(&state, &joint) < 1.0e-3,
                "anchor drifted apart"
            );
        }

        let swing = swing_angle(&state, &joint);
        assert!(
            swing < swing2 + 5.0e-3,
            "limb settled past the swing2 rim at {swing}"
        );
        assert!(
            swing > swing2 - 2.0e-2,
            "limb never reached its swing2 rim: swing = {swing}"
        );
    }

    #[test]
    fn degenerate_elliptical_cone_matches_circular_rim() {
        // Equal swing half-angles make the rim circular, so an *oblique* pull
        // (equal +X and +Z) must settle at the same `phi_max = limit` as an
        // axis-aligned one — the strict superset recovering the circular cone.
        let limit = 0.4;
        let (mut state, joint) = socketed_limb(limit, limit, FRAC_PI_2);
        let integrator = IntegratorConfig::new(Vec3::new(1.2, 0.0, 1.2), 8, 0.0, 4.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..400 {
            cpu_solve_joints_elliptical_cone_twist(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
        }

        let swing = swing_angle(&state, &joint);
        assert!(
            swing < limit + 5.0e-3,
            "limb settled past the circular rim at {swing}"
        );
        assert!(
            swing > limit - 2.0e-2,
            "limb never reached its circular rim: swing = {swing}"
        );
    }

    #[test]
    fn twist_stop_arrests_a_spun_limb() {
        // Give the limb an axial spin about +Y and a tight twist range: the
        // twist limit must catch it near the bound rather than let it spin
        // through, while the wide elliptical cone leaves the swing free.
        let twist_limit = 0.4;
        let (mut state, joint) = socketed_limb(FRAC_PI_2, FRAC_PI_2, twist_limit);
        state.angular_velocities[1] = Vec3::new(0.0, 3.0, 0.0);
        let integrator = IntegratorConfig::new(Vec3::ZERO, 8, 0.0, 0.0);
        let config = JointSolverConfig::new(8);
        let dt = 1.0 / 60.0;

        for _ in 0..120 {
            cpu_solve_joints_elliptical_cone_twist(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
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
        let joint = EllipticalConeTwistJoint::symmetric_cone(
            0,
            1,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::Y,
            Vec3::Y,
            Vec3::X,
            Vec3::X,
            FRAC_PI_2,
            0.5,
            0.3,
        );
        let integrator = IntegratorConfig::new(Vec3::ZERO, 4, 0.0, 0.0);
        let config = JointSolverConfig::new(2);
        let dt = 1.0 / 60.0;

        let initial = state.linear_velocities[0] + state.linear_velocities[1];
        for _ in 0..30 {
            cpu_solve_joints_elliptical_cone_twist(&mut state, &[joint], &integrator, &config, dt)
                .unwrap();
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
        let (mut state, _joint) = socketed_limb(FRAC_PI_2, 0.5, 0.5);
        let before = state.positions[1];
        let integrator = IntegratorConfig::default();
        let config = JointSolverConfig::new(4);
        cpu_solve_joints_elliptical_cone_twist(&mut state, &[], &integrator, &config, 1.0 / 60.0)
            .unwrap();
        assert_eq!(state.positions[1], before, "no-op solve moved a body");
    }
}
