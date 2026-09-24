//! Velocity-level contact solve: restitution and dynamic (Coulomb) friction.
//!
//! After the position solve has removed penetration and
//! [`recover_velocities`](super::integrate::recover_velocities) has rebuilt the
//! velocities from the net pose change, this pass applies the two remaining
//! velocity-level effects for each contact point:
//!
//! - **Restitution** drives the post-solve relative normal velocity toward
//!   `-e * vn_pre`, where `vn_pre` is the closing speed captured before the
//!   position solve and `e` is the combined restitution. Contacts whose closing
//!   speed is below [`XpbdConfig::restitution_threshold`] are treated as fully
//!   inelastic so resting stacks do not jitter.
//! - **Dynamic friction** removes tangential relative velocity, with the
//!   applied tangential impulse clamped to the Coulomb cone `mu * lambda_n / h`
//!   built from the accumulated normal position impulse.
//!
//! Because the normal impulse acts along the contact normal and the friction
//! impulse acts in the tangent plane, the two corrections are orthogonal and
//! are computed from a single sample of the relative velocity.
//!
//! # Provenance
//!
//! The restitution rule (`Δv = n(-vn + max(-e·vn_pre, 0))` reformulated for the
//! "positive means approaching" normal convention), the resting-contact
//! restitution cutoff, and the Coulomb friction bound `mu * lambda_n / h` follow
//! Müller et al., *Detailed Rigid Body Simulation with Extended Position Based
//! Dynamics* (2020). This file contains no Unreal Engine source or derived code.

use super::config::XpbdConfig;
use super::contact_constraint::ContactConstraint;
use super::rigid::{apply_velocity_impulse, generalized_inverse_mass, inv_inertia_world};
use crate::state::view::BodySolverView;
use glam::{Mat3, Vec3};

/// Threshold below which an effective inverse mass or a speed is treated as
/// numerically zero.
const SOLVE_EPS: f32 = 1.0e-9;

/// Applies the velocity-level restitution and dynamic-friction correction to
/// every contact point of every constraint.
///
/// This must run after the position solve and velocity recovery so the
/// accumulated normal impulse and the recovered velocities are available.
pub fn solve(
    view: &mut BodySolverView<'_>,
    constraints: &[ContactConstraint],
    config: &XpbdConfig,
    h: f32,
) {
    if h <= 0.0 {
        return;
    }
    let inv_h = 1.0 / h;
    for constraint in constraints {
        solve_constraint(view, constraint, config, inv_h);
    }
}

/// Solves the velocity-level restitution and dynamic friction for the contact
/// constraints named by `indices` (used by the per-island solver).
///
/// Each entry of `indices` is an index into `constraints`; out-of-range indices
/// are ignored. Because islands touch disjoint sets of dynamic bodies, solving
/// them one at a time here is numerically identical to a single global
/// [`solve`] pass.
pub fn solve_indexed(
    view: &mut BodySolverView<'_>,
    constraints: &[ContactConstraint],
    indices: &[usize],
    config: &XpbdConfig,
    h: f32,
) {
    if h <= 0.0 {
        return;
    }
    let inv_h = 1.0 / h;
    for &i in indices {
        if let Some(constraint) = constraints.get(i) {
            solve_constraint(view, constraint, config, inv_h);
        }
    }
}

/// Solves the velocity-level response for every point of a single constraint.
fn solve_constraint(
    view: &mut BodySolverView<'_>,
    constraint: &ContactConstraint,
    config: &XpbdConfig,
    inv_h: f32,
) {
    for index in 0..constraint.point_count() {
        solve_point(view, constraint, index, config, inv_h);
    }
}

/// Solves the restitution and dynamic-friction impulse for a single point.
fn solve_point(
    view: &mut BodySolverView<'_>,
    constraint: &ContactConstraint,
    index: usize,
    config: &XpbdConfig,
    inv_h: f32,
) {
    let slot_a = constraint.slot_a;
    let slot_b = constraint.slot_b;
    let normal = constraint.normal;

    let inv_mass_a = inv_mass(view, slot_a);
    let inv_mass_b = inv_mass(view, slot_b);
    let inertia_a = inertia_world(view, slot_a);
    let inertia_b = inertia_world(view, slot_b);

    let (r_a, r_b) = constraint.anchors(view, index);
    let vel_a = point_velocity(view, slot_a, r_a);
    let vel_b = point_velocity(view, slot_b, r_b);
    let v_rel = vel_a - vel_b;

    let vn = v_rel.dot(normal);
    let v_tangent = v_rel - normal * vn;

    apply_restitution(
        view,
        slot_a,
        slot_b,
        normal,
        inv_mass_a,
        inv_mass_b,
        inertia_a,
        inertia_b,
        r_a,
        r_b,
        vn,
        constraint.normal_velocity_pre(index),
        constraint.restitution,
        config.restitution_threshold,
    );

    apply_dynamic_friction(
        view,
        slot_a,
        slot_b,
        inv_mass_a,
        inv_mass_b,
        inertia_a,
        inertia_b,
        r_a,
        r_b,
        v_tangent,
        constraint.friction,
        constraint.normal_lambda(index),
        inv_h,
    );
}

/// Applies the normal restitution impulse for one contact point.
#[expect(
    clippy::too_many_arguments,
    reason = "restitution needs the full body pair, anchors, normal, and pre-solve velocity"
)]
fn apply_restitution(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    normal: Vec3,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inertia_a: Mat3,
    inertia_b: Mat3,
    r_a: Vec3,
    r_b: Vec3,
    vn: f32,
    vn_pre: f32,
    restitution: f32,
    threshold: f32,
) {
    // Suppress restitution for slow (resting) contacts so stacks stay quiet.
    let e = if vn_pre.abs() < threshold {
        0.0
    } else {
        restitution
    };
    // Positive `vn`/`vn_pre` means approaching (normal points a -> b); the
    // separating target is negative.
    let target_vn = -e * vn_pre;
    let delta_vn = target_vn - vn;
    // Only apply when the correction increases separation; never pull the
    // bodies back together.
    if delta_vn >= -SOLVE_EPS {
        return;
    }

    let w_a = generalized_inverse_mass(inv_mass_a, inertia_a, r_a, normal);
    let w_b = generalized_inverse_mass(inv_mass_b, inertia_b, r_b, normal);
    let w_sum = w_a + w_b;
    if w_sum <= SOLVE_EPS {
        return;
    }

    let lambda = delta_vn / w_sum;
    let impulse = normal * lambda;
    apply_pair(
        view, slot_a, slot_b, inv_mass_a, inv_mass_b, inertia_a, inertia_b, r_a, r_b, impulse,
    );
}

/// Applies the dynamic (Coulomb) friction impulse for one contact point.
#[expect(
    clippy::too_many_arguments,
    reason = "dynamic friction needs the full body pair, anchors, normal, and normal impulse"
)]
fn apply_dynamic_friction(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inertia_a: Mat3,
    inertia_b: Mat3,
    r_a: Vec3,
    r_b: Vec3,
    v_tangent: Vec3,
    friction: f32,
    normal_lambda: f32,
    inv_h: f32,
) {
    if friction <= 0.0 || normal_lambda <= 0.0 {
        return;
    }
    let speed = v_tangent.length();
    if speed <= SOLVE_EPS {
        return;
    }
    let tangent = v_tangent / speed;

    let w_a = generalized_inverse_mass(inv_mass_a, inertia_a, r_a, tangent);
    let w_b = generalized_inverse_mass(inv_mass_b, inertia_b, r_b, tangent);
    let w_sum = w_a + w_b;
    if w_sum <= SOLVE_EPS {
        return;
    }

    // Impulse that would fully cancel the tangential relative velocity.
    let full = speed / w_sum;
    // Coulomb cone: the tangential impulse cannot exceed `mu * lambda_n / h`.
    let bound = friction * normal_lambda * inv_h;
    let lambda_t = full.min(bound);

    // Friction opposes the relative tangential motion of `a` with respect to
    // `b`, so push `a` along `-tangent`.
    let impulse = tangent * (-lambda_t);
    apply_pair(
        view, slot_a, slot_b, inv_mass_a, inv_mass_b, inertia_a, inertia_b, r_a, r_b, impulse,
    );
}

/// Applies velocity impulse `+impulse` to `a` and `-impulse` to `b` at their
/// contact anchors, skipping non-dynamic bodies.
#[expect(
    clippy::too_many_arguments,
    reason = "applying a symmetric impulse touches both bodies plus their anchors and inertia"
)]
fn apply_pair(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inertia_a: Mat3,
    inertia_b: Mat3,
    r_a: Vec3,
    r_b: Vec3,
    impulse: Vec3,
) {
    if view.is_dynamic(slot_a) {
        let mut v = view.linear_velocities[slot_a];
        let mut w = view.angular_velocities[slot_a];
        apply_velocity_impulse(&mut v, &mut w, inv_mass_a, inertia_a, r_a, impulse);
        view.linear_velocities[slot_a] = v;
        view.angular_velocities[slot_a] = w;
    }
    if view.is_dynamic(slot_b) {
        let mut v = view.linear_velocities[slot_b];
        let mut w = view.angular_velocities[slot_b];
        apply_velocity_impulse(&mut v, &mut w, inv_mass_b, inertia_b, r_b, -impulse);
        view.linear_velocities[slot_b] = v;
        view.angular_velocities[slot_b] = w;
    }
}

/// Returns the world-space velocity of the point at lever arm `r` on `slot`.
#[must_use]
fn point_velocity(view: &BodySolverView<'_>, slot: usize, r: Vec3) -> Vec3 {
    view.linear_velocities[slot] + view.angular_velocities[slot].cross(r)
}

/// Returns the effective inverse mass of `slot`, or `0` for non-dynamic bodies.
#[must_use]
fn inv_mass(view: &BodySolverView<'_>, slot: usize) -> f32 {
    if view.is_dynamic(slot) {
        view.mass_props[slot].inv_mass
    } else {
        0.0
    }
}

/// Returns the world-space inverse inertia tensor of `slot`, or the zero tensor
/// for non-dynamic bodies (infinite inertia).
#[must_use]
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
    use crate::collide::{ContactManifold, ContactPoint};
    use crate::solver::xpbd::contact_constraint::{solve_positions, ContactConstraint};
    use crate::state::body::{BodyDesc, MassProperties};
    use crate::state::storage::BodyStorage;
    use crate::PhysicsMaterial;

    fn unit_dynamic(pos: Vec3) -> BodyDesc {
        BodyDesc::dynamic_at(pos).with_mass_properties(MassProperties {
            inv_mass: 1.0,
            inv_inertia: Vec3::ONE,
        })
    }

    #[test]
    fn resting_contact_has_no_restitution() {
        // Two boxes barely touching, tiny closing speed below threshold.
        let mut s = BodyStorage::new();
        let ha = s.insert(unit_dynamic(Vec3::ZERO).with_linear_velocity(Vec3::new(0.0, 0.05, 0.0)));
        let hb = s.insert(
            unit_dynamic(Vec3::new(0.0, 1.0, 0.0)).with_linear_velocity(Vec3::new(0.0, -0.05, 0.0)),
        );
        let mut manifold = ContactManifold::new(ha, hb, Vec3::Y);
        manifold.push(ContactPoint::new(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            0.0,
        ));
        let config = XpbdConfig::default();
        let mut view = s.solver_view_mut();
        let constraints = ContactConstraint::build(&view, &[manifold]);
        solve(&mut view, &constraints, &config, 0.016);
        // Closing speed 0.1 < 0.5 threshold => inelastic, velocities driven so
        // the relative normal velocity is (near) zero, not reversed.
        let vn = (view.linear_velocities[0] - view.linear_velocities[1]).dot(Vec3::Y);
        assert!(
            vn.abs() < 1e-3,
            "resting contact should not bounce, vn = {vn}"
        );
    }

    #[test]
    fn restitution_reverses_fast_normal_velocity() {
        let mut s = BodyStorage::new();
        // Fast closing speed (2 m/s each) above threshold, restitution 1.0.
        let bouncy = PhysicsMaterial::new(0.0, 1.0);
        let ha = s.insert(
            unit_dynamic(Vec3::ZERO)
                .with_linear_velocity(Vec3::new(0.0, 2.0, 0.0))
                .with_material(bouncy),
        );
        let hb = s.insert(
            unit_dynamic(Vec3::new(0.0, 1.0, 0.0))
                .with_linear_velocity(Vec3::new(0.0, -2.0, 0.0))
                .with_material(bouncy),
        );
        let mut manifold = ContactManifold::new(ha, hb, Vec3::Y);
        manifold.push(ContactPoint::new(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            0.0,
        ));
        let config = XpbdConfig::default();
        let mut view = s.solver_view_mut();
        let constraints = ContactConstraint::build(&view, &[manifold]);
        let vn_pre = (view.linear_velocities[0] - view.linear_velocities[1]).dot(Vec3::Y);
        solve(&mut view, &constraints, &config, 0.016);
        let vn_post = (view.linear_velocities[0] - view.linear_velocities[1]).dot(Vec3::Y);
        // Perfect restitution reverses the closing speed to a separating speed.
        assert!(vn_pre > 3.9);
        assert!(
            (vn_post + vn_pre).abs() < 1e-2,
            "vn_post = {vn_post}, vn_pre = {vn_pre}"
        );
    }

    #[test]
    fn dynamic_friction_opposes_sliding() {
        let mut s = BodyStorage::new();
        let ground =
            s.insert(BodyDesc::static_at(Vec3::ZERO).with_material(PhysicsMaterial::new(1.0, 0.0)));
        let hb = s.insert(
            unit_dynamic(Vec3::new(0.0, 0.0, 0.0))
                .with_linear_velocity(Vec3::new(1.0, -0.4, 0.0))
                .with_material(PhysicsMaterial::new(1.0, 0.0)),
        );
        // Ground is a, box is b; normal a -> b is +Y. Box sank 0.1 below.
        let mut manifold = ContactManifold::new(ground, hb, Vec3::Y);
        manifold.push(ContactPoint::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -0.1, 0.0),
            0.1,
        ));
        let config = XpbdConfig::default();
        let h = 0.016;
        let mut view = s.solver_view_mut();
        let mut constraints = ContactConstraint::build(&view, &[manifold]);
        // Position solve accumulates a normal impulse that bounds friction.
        solve_positions(&mut view, &mut constraints, &config, h);
        let vx_before = view.linear_velocities[1].x;
        solve(&mut view, &constraints, &config, h);
        let vx_after = view.linear_velocities[1].x;
        // Friction reduces the tangential (x) velocity of the sliding box.
        assert!(vx_after < vx_before, "vx {vx_before} -> {vx_after}");
        assert!(
            vx_after >= 0.0,
            "friction must not reverse sliding: {vx_after}"
        );
    }
}
