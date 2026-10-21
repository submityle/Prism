//! Stateful position-level XPBD gear constraint.

use super::rigid::inv_inertia_world;
use crate::joint::{Joint, JointKind, JointStorage};
use crate::state::view::BodySolverView;
use glam::{Mat3, Quat, Vec3};

/// Begin one substep before pose prediction. Lambda belongs to this substep,
/// while shaft winding numbers belong to the live joint for its entire life.
/// Disabled gears are still observed: disengaging does not erase shaft turns.
/// Standalone users of `joint_constraint::solve_joint(s)` must bracket all
/// position iterations with `begin_substep` / `finish_substep` as the solver does.
pub fn begin_substep(view: &BodySolverView<'_>, joints: &mut JointStorage) {
    for joint in joints.live_joints_mut() {
        observe(view, joint);
        joint.gear_state.lambda = 0.0;
    }
}

/// Observe final poses after *all* position writers, including contacts and
/// coupling. The last position iteration may not be the last writer of a shaft.
/// Clearing lambda makes fixed-boundary snapshots independent of iteration
/// scratch, and guarantees that restore never reuses an old XPBD impulse.
pub fn finish_substep(view: &BodySolverView<'_>, joints: &mut JointStorage) {
    for joint in joints.live_joints_mut() {
        observe(view, joint);
        joint.gear_state.lambda = 0.0;
    }
}

fn observe(view: &BodySolverView<'_>, joint: &mut Joint) -> Option<[Vec3; 2]> {
    let JointKind::Gear(gear) = joint.kind else {
        joint.reset_gear_state();
        return None;
    };
    let a = joint.anchor_a.body.index() as usize;
    let b = joint.anchor_b.body.index() as usize;
    if a == b || !view.is_active(a) || !view.is_active(b) {
        return None;
    }
    joint.gear_state.observe(
        gear,
        joint.anchor_a,
        joint.anchor_b,
        [view.orientations[a], view.orientations[b]],
    )
}

/// Returns a finite torque multiplier for the existing breakage/load channel.
/// That channel exposes only one torque vector, so report the larger endpoint
/// reaction (A = -ratio*lambda*grad_a, B = lambda*grad_b), not their sum: a
/// reversing gear can exert unequal torques that would cancel in a sum.
pub(super) fn solve(view: &mut BodySolverView<'_>, joint: &mut Joint, h: f32) -> Vec3 {
    let JointKind::Gear(gear) = joint.kind else {
        return Vec3::ZERO;
    };
    let Some([axis_a, axis_b]) = observe(view, joint) else {
        return Vec3::ZERO;
    };
    let a = joint.anchor_a.body.index() as usize;
    let b = joint.anchor_b.body.index() as usize;
    let inertia = |slot: usize| {
        if view.is_dynamic(slot) {
            inv_inertia_world(view.mass_props[slot].inv_inertia, view.orientations[slot])
        } else {
            Mat3::ZERO
        }
    };
    let ia = inertia(a);
    let ib = inertia(b);
    // Work in f64 so extreme finite ratios cannot overflow the denominator,
    // and fractional-angle precision does not decay after many full turns.
    let ratio = f64::from(gear.ratio);
    let ga = axis_a.as_dvec3() * -ratio;
    let gb = axis_b.as_dvec3();
    let da = ia.as_dmat3() * ga;
    let db = ib.as_dmat3() * gb;
    let weight = ga.dot(da) + gb.dot(db);
    if !weight.is_finite() || weight <= 0.0 {
        return Vec3::ZERO;
    }
    let alpha = f64::from(gear.compliance) / f64::from(h).powi(2);
    let [theta_a, theta_b] = joint.gear_state.angles().expect("observed gear");
    let error = theta_b - ratio * theta_a - gear.phase;
    let mut delta = (-error - alpha * joint.gear_state.lambda) / (weight + alpha);
    // Every correction is observed immediately. Limiting the actual angular
    // displacement, not the constraint error, preserves winding even when the
    // initial phase error spans many turns or the ratio is very large.
    let max_rotation = da.length().max(db.length());
    if max_rotation > 0.0 {
        delta = delta.clamp(-0.5 / max_rotation, 0.5 / max_rotation);
    }
    let correction_a = (da * delta).as_vec3();
    let correction_b = (db * delta).as_vec3();
    let torque_a = (ga * delta).as_vec3();
    let torque_b = (gb * delta).as_vec3();
    if !delta.is_finite()
        || !correction_a.is_finite()
        || !correction_b.is_finite()
        || !torque_a.is_finite()
        || !torque_b.is_finite()
    {
        return Vec3::ZERO;
    }
    joint.gear_state.lambda += delta;
    // Exponential-map corrections preserve the requested angle exactly,
    // unlike a first-order quaternion increment for a large phase correction.
    if view.is_dynamic(a) {
        view.orientations[a] =
            (Quat::from_scaled_axis(correction_a) * view.orientations[a]).normalize();
    }
    if view.is_dynamic(b) {
        view.orientations[b] =
            (Quat::from_scaled_axis(correction_b) * view.orientations[b]).normalize();
    }
    observe(view, joint);
    if torque_a.as_dvec3().length_squared() >= torque_b.as_dvec3().length_squared() {
        torque_a
    } else {
        torque_b
    }
}
