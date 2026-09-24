//! Position-level joint constraints for the XPBD solver.
//!
//! This module projects every active [`Joint`](crate::joint::Joint) once per
//! position iteration, mirroring the way
//! [`contact_constraint`](crate::solver::xpbd::contact_constraint) projects
//! contacts. Each joint family reduces to a small set of scalar or angular
//! constraints solved with the same compliant XPBD update the contact solver
//! uses:
//!
//! - a *positional* constraint drives a scalar error `c` (a distance or an
//!   offset along a direction) to zero via [`apply_positional`];
//! - an *angular* constraint drives a rotation-vector error `theta` to zero via
//!   [`apply_angular`].
//!
//! Higher-level joints compose these primitives: a hinge is a coincident-point
//! constraint plus a swing-axis alignment plus an optional twist limit and
//! motor; a slider is a perpendicular-offset lock plus a frame alignment plus an
//! optional slide limit and motor.
//!
//! # Provenance
//!
//! The compliant position update, the generalized-inverse-mass weighting, and
//! the angular-constraint algebra follow Müller et al., *Detailed Rigid Body
//! Simulation with Extended Position Based Dynamics* (2020). This file contains
//! **no Unreal Engine source or derived code**.

use super::rigid::{
    apply_angular_delta, apply_position_impulse, generalized_inverse_mass, inv_inertia_world,
};
use crate::joint::kind::{DistanceJoint, JointKind, PrismaticJoint, RevoluteJoint};
use crate::joint::motor::{Motor, MotorTarget};
use crate::joint::{Joint, JointStorage};
use crate::state::view::BodySolverView;
use glam::{Mat3, Quat, Vec3};

/// Threshold below which a length, an effective inverse mass, or an angle is
/// treated as numerically zero.
const SOLVE_EPS: f32 = 1.0e-9;

/// Solves every active joint in `joints` against the current poses in `view`.
///
/// This runs one projection pass; the caller loops it alongside the contact
/// position solve for the configured number of position iterations. Joints that
/// reference an inactive slot, an out-of-range slot, or a pair with no dynamic
/// body are skipped.
pub fn solve_joints(view: &mut BodySolverView<'_>, joints: &JointStorage, h: f32) {
    for joint in joints.active_joints() {
        solve_one(view, joint, h);
    }
}

/// Resolves a single joint's slots, guards them, and dispatches on its kind.
fn solve_one(view: &mut BodySolverView<'_>, joint: &Joint, h: f32) {
    let slot_a = joint.anchor_a.body.index() as usize;
    let slot_b = joint.anchor_b.body.index() as usize;
    if slot_a >= view.slot_count() || slot_b >= view.slot_count() {
        return;
    }
    if !view.is_active(slot_a) || !view.is_active(slot_b) {
        return;
    }
    if !view.is_dynamic(slot_a) && !view.is_dynamic(slot_b) {
        return;
    }

    let alpha = alpha_tilde(joint.kind.compliance(), h);
    match &joint.kind {
        JointKind::Fixed(_) => {
            solve_point(view, joint, slot_a, slot_b, alpha);
            solve_frame_align(view, joint, slot_a, slot_b, alpha);
        }
        JointKind::Distance(params) => solve_distance(view, joint, slot_a, slot_b, *params, alpha),
        JointKind::Spherical(_) => solve_point(view, joint, slot_a, slot_b, alpha),
        JointKind::Revolute(params) => {
            solve_revolute(view, joint, slot_a, slot_b, params, alpha, h);
        }
        JointKind::Prismatic(params) => {
            solve_prismatic(view, joint, slot_a, slot_b, params, alpha, h);
        }
    }
}

/// Holds the two anchor points coincident (ball-and-socket / weld position).
fn solve_point(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    alpha: f32,
) {
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let delta = world_a - world_b;
    let c = delta.length();
    if c <= SOLVE_EPS {
        return;
    }
    let dir = delta / c;
    apply_positional(view, slot_a, slot_b, r_a, r_b, dir, c, alpha, f32::INFINITY);
}

/// Keeps the anchor separation inside the distance joint's length range.
fn solve_distance(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    params: DistanceJoint,
    alpha: f32,
) {
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let delta = world_a - world_b;
    let d = delta.length();
    if d <= SOLVE_EPS {
        return;
    }
    let target = d.clamp(params.min_length, params.max_length);
    let c = d - target;
    if c.abs() <= SOLVE_EPS {
        return;
    }
    let dir = delta / d;
    apply_positional(view, slot_a, slot_b, r_a, r_b, dir, c, alpha, f32::INFINITY);
}

/// Aligns the two anchor reference frames (weld orientation / slider lock).
fn solve_frame_align(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    alpha: f32,
) {
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let frame_b = world_frame(view, slot_b, joint.anchor_b.local_frame);
    let theta = frame_error(frame_a, frame_b);
    apply_angular(view, slot_a, slot_b, theta, alpha, f32::INFINITY);
}

/// Solves a hinge: coincident anchors, aligned hinge axis, optional twist limit
/// and motor about that axis.
fn solve_revolute(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    params: &RevoluteJoint,
    alpha: f32,
    h: f32,
) {
    solve_point(view, joint, slot_a, slot_b, alpha);

    let axis_local = params.axis.normalize_or_zero();
    if axis_local.length_squared() <= SOLVE_EPS {
        return;
    }

    // Swing: force the world-space hinge axes of the two frames to coincide.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let frame_b = world_frame(view, slot_b, joint.anchor_b.local_frame);
    let axis_a = (frame_a * axis_local).normalize_or_zero();
    let axis_b = (frame_b * axis_local).normalize_or_zero();
    let swing = axis_a.cross(axis_b);
    apply_angular(view, slot_a, slot_b, swing, alpha, f32::INFINITY);

    if params.limit.is_none() && params.motor.is_none() {
        return;
    }

    // Twist: measure the relative rotation about the (re-read) hinge axis.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let frame_b = world_frame(view, slot_b, joint.anchor_b.local_frame);
    let hinge = (frame_a * axis_local).normalize_or_zero();
    if hinge.length_squared() <= SOLVE_EPS {
        return;
    }
    let angle = twist_angle(frame_a, frame_b, hinge);

    if let Some(limit) = params.limit {
        let c = angle - limit.clamp(angle);
        if c.abs() > SOLVE_EPS {
            apply_angular(view, slot_a, slot_b, hinge * c, alpha, f32::INFINITY);
        }
    }
    if let Some(motor) = params.motor {
        let c = motor_error(&motor, angle, h);
        let lambda_max = motor.max_force * h * h;
        let motor_alpha = alpha_tilde(motor.compliance, h);
        apply_angular(view, slot_a, slot_b, hinge * c, motor_alpha, lambda_max);
    }
}

/// Solves a slider: locks the perpendicular offset and the relative rotation,
/// with an optional slide limit and motor along the axis.
fn solve_prismatic(
    view: &mut BodySolverView<'_>,
    joint: &Joint,
    slot_a: usize,
    slot_b: usize,
    params: &PrismaticJoint,
    alpha: f32,
    h: f32,
) {
    let axis_local = params.axis.normalize_or_zero();
    if axis_local.length_squared() <= SOLVE_EPS {
        return;
    }

    // Perpendicular lock: cancel the anchor separation orthogonal to the axis.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let axis = (frame_a * axis_local).normalize_or_zero();
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let delta = world_a - world_b;
    let perp = delta - axis * delta.dot(axis);
    let perp_len = perp.length();
    if perp_len > SOLVE_EPS {
        let dir = perp / perp_len;
        apply_positional(
            view,
            slot_a,
            slot_b,
            r_a,
            r_b,
            dir,
            perp_len,
            alpha,
            f32::INFINITY,
        );
    }

    // Rotation lock: a slider keeps the two frames aligned.
    solve_frame_align(view, joint, slot_a, slot_b, alpha);

    if params.limit.is_none() && params.motor.is_none() {
        return;
    }

    // Slide coordinate along the (re-read) axis, with its position gradient.
    let frame_a = world_frame(view, slot_a, joint.anchor_a.local_frame);
    let axis = (frame_a * axis_local).normalize_or_zero();
    let (r_a, world_a) = world_anchor(view, slot_a, joint.anchor_a.local_point);
    let (r_b, world_b) = world_anchor(view, slot_b, joint.anchor_b.local_point);
    let slide = (world_b - world_a).dot(axis);
    let grad_a = -axis;

    if let Some(limit) = params.limit {
        let c = slide - limit.clamp(slide);
        if c.abs() > SOLVE_EPS {
            apply_positional(
                view,
                slot_a,
                slot_b,
                r_a,
                r_b,
                grad_a,
                c,
                alpha,
                f32::INFINITY,
            );
        }
    }
    if let Some(motor) = params.motor {
        let c = motor_error(&motor, slide, h);
        let lambda_max = motor.max_force * h * h;
        let motor_alpha = alpha_tilde(motor.compliance, h);
        apply_positional(
            view,
            slot_a,
            slot_b,
            r_a,
            r_b,
            grad_a,
            c,
            motor_alpha,
            lambda_max,
        );
    }
}

/// Applies one compliant positional correction for scalar error `c`.
///
/// `dir` is the unit gradient of `c` with respect to body `a`'s anchor
/// position; the correction moves `a` by `+p` and `b` by `-p` so that `c` is
/// driven toward zero. `lambda_max` bounds the impulse magnitude (use
/// [`f32::INFINITY`] for a hard constraint, or `max_force * h^2` for a motor).
#[expect(
    clippy::too_many_arguments,
    reason = "a positional constraint needs the body pair, both lever arms, the gradient, the error, the compliance, and the impulse cap"
)]
fn apply_positional(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    r_a: Vec3,
    r_b: Vec3,
    dir: Vec3,
    c: f32,
    alpha: f32,
    lambda_max: f32,
) {
    let inv_mass_a = inv_mass(view, slot_a);
    let inv_mass_b = inv_mass(view, slot_b);
    let inertia_a = inertia_world(view, slot_a);
    let inertia_b = inertia_world(view, slot_b);

    let w_a = generalized_inverse_mass(inv_mass_a, inertia_a, r_a, dir);
    let w_b = generalized_inverse_mass(inv_mass_b, inertia_b, r_b, dir);
    let w = w_a + w_b;
    if w <= SOLVE_EPS {
        return;
    }

    let delta_lambda = (-c / (w + alpha)).clamp(-lambda_max, lambda_max);
    let p = dir * delta_lambda;

    if view.is_dynamic(slot_a) {
        let mut x = view.positions[slot_a];
        let mut q = view.orientations[slot_a];
        apply_position_impulse(&mut x, &mut q, inv_mass_a, inertia_a, r_a, p);
        view.positions[slot_a] = x;
        view.orientations[slot_a] = q;
    }
    if view.is_dynamic(slot_b) {
        let mut x = view.positions[slot_b];
        let mut q = view.orientations[slot_b];
        apply_position_impulse(&mut x, &mut q, inv_mass_b, inertia_b, r_b, -p);
        view.positions[slot_b] = x;
        view.orientations[slot_b] = q;
    }
}

/// Applies one compliant angular correction for rotation-vector error `theta`.
///
/// `theta` points along the axis about which body `b` leads body `a`, scaled by
/// the angle error; the correction rotates `a` toward `b` and `b` toward `a` so
/// the error is driven to zero. `lambda_max` bounds the impulse magnitude.
fn apply_angular(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    theta: Vec3,
    alpha: f32,
    lambda_max: f32,
) {
    let c = theta.length();
    if c <= SOLVE_EPS {
        return;
    }
    let axis = theta / c;
    let inertia_a = inertia_world(view, slot_a);
    let inertia_b = inertia_world(view, slot_b);

    let w_a = if view.is_dynamic(slot_a) {
        axis.dot(inertia_a * axis)
    } else {
        0.0
    };
    let w_b = if view.is_dynamic(slot_b) {
        axis.dot(inertia_b * axis)
    } else {
        0.0
    };
    let w = w_a + w_b;
    if w <= SOLVE_EPS {
        return;
    }

    let delta_lambda = (c / (w + alpha)).clamp(-lambda_max, lambda_max);
    let p = axis * delta_lambda;

    if view.is_dynamic(slot_a) {
        let mut q = view.orientations[slot_a];
        apply_angular_delta(&mut q, inertia_a * p);
        view.orientations[slot_a] = q;
    }
    if view.is_dynamic(slot_b) {
        let mut q = view.orientations[slot_b];
        apply_angular_delta(&mut q, -(inertia_b * p));
        view.orientations[slot_b] = q;
    }
}

/// Returns the per-sub-step scalar goal a motor wants to null out.
///
/// A position motor drives the coordinate toward a fixed target; a velocity
/// motor asks for a `-v * h` increment each sub-step, which the compliant
/// projection converts into a bounded corrective impulse.
fn motor_error(motor: &Motor, current: f32, h: f32) -> f32 {
    match motor.target {
        MotorTarget::Position(target) => current - target,
        MotorTarget::Velocity(rate) => -rate * h,
    }
}

/// Converts a compliance (inverse stiffness) into the XPBD `alpha_tilde` term.
fn alpha_tilde(compliance: f32, h: f32) -> f32 {
    if h > 0.0 {
        compliance / (h * h)
    } else {
        0.0
    }
}

/// Returns the world lever arm `r` and world position of an anchor point.
fn world_anchor(view: &BodySolverView<'_>, slot: usize, local_point: Vec3) -> (Vec3, Vec3) {
    let r = view.orientations[slot] * local_point;
    (r, view.positions[slot] + r)
}

/// Returns the world-space orientation of an anchor reference frame.
fn world_frame(view: &BodySolverView<'_>, slot: usize, local_frame: Quat) -> Quat {
    view.orientations[slot] * local_frame
}

/// Returns the small-angle rotation vector that carries frame `a` onto frame
/// `b` (twice the vector part of the relative quaternion, hemisphere-aligned).
fn frame_error(a: Quat, b: Quat) -> Vec3 {
    let mut q_rel = b * a.inverse();
    if q_rel.w < 0.0 {
        q_rel = -q_rel;
    }
    2.0 * Vec3::new(q_rel.x, q_rel.y, q_rel.z)
}

/// Returns the signed twist angle of frame `b` relative to frame `a` about the
/// unit `hinge` axis, extracted by swing-twist decomposition of the relative
/// rotation. The result lies in `[-pi, pi]`.
fn twist_angle(frame_a: Quat, frame_b: Quat, hinge: Vec3) -> f32 {
    let mut rel = frame_b * frame_a.inverse();
    if rel.w < 0.0 {
        rel = -rel;
    }
    // Project the rotation's vector part onto the hinge axis to isolate twist.
    let proj = Vec3::new(rel.x, rel.y, rel.z).dot(hinge);
    let twist = Quat::from_xyzw(hinge.x * proj, hinge.y * proj, hinge.z * proj, rel.w);
    if twist.length_squared() <= SOLVE_EPS {
        return 0.0;
    }
    let twist = twist.normalize();
    let (axis, angle) = twist.to_axis_angle();
    let sign = axis.dot(hinge);
    if sign < 0.0 {
        -angle
    } else {
        angle
    }
}

/// Returns the effective inverse mass of `slot`, or `0` for non-dynamic bodies.
fn inv_mass(view: &BodySolverView<'_>, slot: usize) -> f32 {
    if view.is_dynamic(slot) {
        view.mass_props[slot].inv_mass
    } else {
        0.0
    }
}

/// Returns the world-space inverse inertia tensor of `slot`, or the zero tensor
/// for non-dynamic bodies (infinite inertia).
fn inertia_world(view: &BodySolverView<'_>, slot: usize) -> Mat3 {
    if view.is_dynamic(slot) {
        inv_inertia_world(view.mass_props[slot].inv_inertia, view.orientations[slot])
    } else {
        Mat3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::anchor::JointAnchor;
    use crate::joint::desc::JointDesc;
    use crate::joint::kind::{PrismaticJoint, RevoluteJoint};
    use crate::state::body::BodyDesc;
    use crate::state::storage::BodyStorage;

    fn dynamic(pos: Vec3) -> BodyDesc {
        BodyDesc::dynamic_at(pos)
    }

    #[test]
    fn distance_joint_converges_to_target_length() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::new(2.0, 0.0, 0.0)));
        let joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::distance(
                a,
                Vec3::ZERO,
                b,
                Vec3::ZERO,
                DistanceJoint::rigid(1.0),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..8 {
            solve_joints(&mut view, &joints, 1.0 / 60.0);
        }
        let d = (view.positions[a.index() as usize] - view.positions[b.index() as usize]).length();
        assert!(
            (d - 1.0).abs() < 1.0e-4,
            "distance settled at {d}, expected 1.0"
        );
    }

    #[test]
    fn spherical_joint_brings_anchors_together() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::new(1.0, 0.0, 0.0)));
        let joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::spherical(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..8 {
            solve_joints(&mut view, &joints, 1.0 / 60.0);
        }
        let gap =
            (view.positions[a.index() as usize] - view.positions[b.index() as usize]).length();
        assert!(gap < 1.0e-4, "anchors did not coincide, gap = {gap}");
    }

    #[test]
    fn fixed_joint_aligns_orientation() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::ZERO));
        s.set_orientation(b, Quat::from_rotation_y(0.4));
        let joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::fixed(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..40 {
            solve_joints(&mut view, &joints, 1.0 / 60.0);
        }
        let err = frame_error(
            view.orientations[a.index() as usize],
            view.orientations[b.index() as usize],
        )
        .length();
        assert!(err < 1.0e-2, "orientation not aligned, error = {err}");
    }

    #[test]
    fn revolute_joint_aligns_hinge_axis() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::ZERO));
        s.set_orientation(b, Quat::from_rotation_x(0.3));
        let joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::revolute(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
                RevoluteJoint::new(Vec3::Z),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..40 {
            solve_joints(&mut view, &joints, 1.0 / 60.0);
        }
        let axis_a = view.orientations[a.index() as usize] * Vec3::Z;
        let axis_b = view.orientations[b.index() as usize] * Vec3::Z;
        let misalignment = axis_a.cross(axis_b).length();
        assert!(
            misalignment < 1.0e-2,
            "hinge axes not aligned, cross = {misalignment}"
        );
    }

    #[test]
    fn prismatic_joint_removes_perpendicular_offset() {
        let mut s = BodyStorage::new();
        let a = s.insert(dynamic(Vec3::ZERO));
        let b = s.insert(dynamic(Vec3::new(0.5, 0.4, 0.3)));
        let joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::prismatic(
                JointAnchor::at_center(a),
                JointAnchor::at_center(b),
                PrismaticJoint::new(Vec3::X),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        for _ in 0..40 {
            solve_joints(&mut view, &joints, 1.0 / 60.0);
        }
        let axis = Vec3::X;
        let delta = view.positions[a.index() as usize] - view.positions[b.index() as usize];
        let perp = (delta - axis * delta.dot(axis)).length();
        assert!(perp < 1.0e-3, "perpendicular offset remains, perp = {perp}");
        // The free slide coordinate along the axis is preserved (unconstrained).
        assert!(
            delta.dot(axis).abs() > 0.1,
            "slide coordinate was wrongly removed"
        );
    }

    #[test]
    fn skips_joint_between_two_static_bodies() {
        let mut s = BodyStorage::new();
        let a = s.insert(BodyDesc::static_at(Vec3::ZERO));
        let b = s.insert(BodyDesc::static_at(Vec3::new(2.0, 0.0, 0.0)));
        let joints = {
            let mut j = JointStorage::new();
            j.insert(JointDesc::distance(
                a,
                Vec3::ZERO,
                b,
                Vec3::ZERO,
                DistanceJoint::rigid(1.0),
            ));
            j
        };
        let mut view = s.solver_view_mut();
        solve_joints(&mut view, &joints, 1.0 / 60.0);
        // Neither static body moved.
        assert_eq!(view.positions[a.index() as usize], Vec3::ZERO);
        assert_eq!(view.positions[b.index() as usize], Vec3::new(2.0, 0.0, 0.0));
    }
}
