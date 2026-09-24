//! Sub-step integration and velocity recovery for the XPBD solver.
//!
//! Position-based dynamics splits a sub-step into three phases: predict new
//! poses from the current velocities, solve position constraints against those
//! predicted poses, then recover velocities from how far each pose actually
//! moved. This module owns the first and third phases; the constraint solve
//! lives in [`contact_constraint`](super::contact_constraint).
//!
//! # Provenance
//!
//! The predict/solve/recover structure and the `v = (x - x_prev) / h` velocity
//! recovery are the standard "small steps" XPBD formulation of Müller et al.,
//! *Detailed Rigid Body Simulation with Extended Position Based Dynamics*
//! (2020). This file contains no Unreal Engine source or derived code.

use crate::state::view::BodySolverView;
use glam::{Quat, Vec3};

/// Threshold below which a squared quaternion length is treated as degenerate.
const QUAT_EPS: f32 = 1.0e-12;

/// Saves the pre-step pose of every active body and integrates the predicted
/// pose of every dynamic body forward by `h` seconds.
///
/// For each dynamic body this applies gravity and velocity damping, advances
/// the center of mass by the updated linear velocity, and integrates the
/// orientation quaternion with the current angular velocity. Static and
/// kinematic bodies keep their pose but still have their previous pose stamped
/// so friction anchors resolve to a zero relative drift.
pub fn predict(view: &mut BodySolverView<'_>, gravity: Vec3, h: f32) {
    for i in 0..view.slot_count() {
        if !view.is_active(i) {
            continue;
        }
        view.prev_positions[i] = view.positions[i];
        view.prev_orientations[i] = view.orientations[i];
        if !view.is_dynamic(i) {
            continue;
        }

        let mut linear = view.linear_velocities[i];
        let mut angular = view.angular_velocities[i];

        linear += gravity * h;
        let lin_factor = 1.0 / (1.0 + view.linear_damping[i] * h);
        let ang_factor = 1.0 / (1.0 + view.angular_damping[i] * h);
        linear *= lin_factor;
        angular *= ang_factor;

        view.linear_velocities[i] = linear;
        view.angular_velocities[i] = angular;

        view.positions[i] += linear * h;
        view.orientations[i] = integrate_orientation(view.orientations[i], angular, h);
    }
}

/// Recovers linear and angular velocities from the net pose change produced by
/// prediction plus the position solve.
///
/// After the position solve has moved each dynamic body, its velocity is the
/// finite difference `(x - x_prev) / h`; the angular velocity is extracted from
/// the relative rotation `q * q_prev^-1` mapped to an axis-angle rate.
pub fn recover_velocities(view: &mut BodySolverView<'_>, h: f32) {
    if h <= 0.0 {
        return;
    }
    let inv_h = 1.0 / h;
    for i in 0..view.slot_count() {
        if !view.is_dynamic(i) {
            continue;
        }
        view.linear_velocities[i] = (view.positions[i] - view.prev_positions[i]) * inv_h;
        view.angular_velocities[i] =
            angular_velocity_from_delta(view.prev_orientations[i], view.orientations[i], inv_h);
    }
}

/// Integrates an orientation quaternion by an angular velocity over `h`.
///
/// Uses the first-order update `q' = normalize(q + 0.5 * [w, 0] * q * h)`, the
/// same scheme the free-body integrator uses.
#[must_use]
fn integrate_orientation(q: Quat, angular: Vec3, h: f32) -> Quat {
    let omega = Quat::from_xyzw(angular.x, angular.y, angular.z, 0.0);
    let dq = omega.mul_quat(q);
    let half = 0.5 * h;
    let a = q.to_array();
    let d = dq.to_array();
    let updated = Quat::from_xyzw(
        a[0] + d[0] * half,
        a[1] + d[1] * half,
        a[2] + d[2] * half,
        a[3] + d[3] * half,
    );
    normalize_or(updated, q)
}

/// Extracts an axis-angle angular velocity from a relative rotation.
///
/// Computes `delta = current * prev^-1` and maps its vector part to an angular
/// rate, scaled by `inv_h`. The sign is chosen from the shortest arc so the
/// recovered spin never flips direction spuriously.
#[must_use]
fn angular_velocity_from_delta(prev: Quat, current: Quat, inv_h: f32) -> Vec3 {
    let delta = current.mul_quat(prev.inverse());
    let mut axis = Vec3::new(delta.x, delta.y, delta.z) * 2.0;
    if delta.w < 0.0 {
        axis = -axis;
    }
    axis * inv_h
}

/// Renormalizes `q`, falling back to `fallback` when it is degenerate.
#[must_use]
fn normalize_or(q: Quat, fallback: Quat) -> Quat {
    if q.length_squared() > QUAT_EPS {
        q.normalize()
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::body::{BodyDesc, BodyKind, MassProperties};
    use crate::state::storage::BodyStorage;

    fn dynamic_storage() -> BodyStorage {
        let mut s = BodyStorage::new();
        s.insert(
            BodyDesc::dynamic_at(Vec3::ZERO).with_mass_properties(MassProperties {
                inv_mass: 1.0,
                inv_inertia: Vec3::ONE,
            }),
        );
        s
    }

    #[test]
    fn predict_saves_prev_and_falls_under_gravity() {
        let mut s = dynamic_storage();
        let mut view = s.solver_view_mut();
        predict(&mut view, Vec3::new(0.0, -10.0, 0.0), 0.1);
        assert_eq!(view.prev_positions[0], Vec3::ZERO);
        assert!(view.positions[0].y < 0.0);
        assert!((view.linear_velocities[0].y - -1.0).abs() < 1e-5);
    }

    #[test]
    fn recover_round_trips_linear_velocity() {
        let mut s = dynamic_storage();
        let h = 0.05;
        {
            let mut view = s.solver_view_mut();
            predict(&mut view, Vec3::new(0.0, -10.0, 0.0), h);
            // Displace further as a stand-in for a position correction.
            view.positions[0].y -= 0.02;
            recover_velocities(&mut view, h);
            let expected = (view.positions[0].y - view.prev_positions[0].y) / h;
            assert!((view.linear_velocities[0].y - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn static_body_is_not_predicted() {
        let mut s = BodyStorage::new();
        s.insert(BodyDesc::static_at(Vec3::Y));
        let mut view = s.solver_view_mut();
        predict(&mut view, Vec3::new(0.0, -10.0, 0.0), 0.1);
        assert_eq!(view.positions[0], Vec3::Y);
        assert_eq!(view.kinds[0], BodyKind::Static);
    }
}
