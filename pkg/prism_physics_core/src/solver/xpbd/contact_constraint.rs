//! Contact position constraints (penetration recovery and static friction).
//!
//! Each contact point becomes a one-dimensional position constraint that pushes
//! the two bodies apart along the manifold normal until they no longer overlap,
//! plus an optional static-friction constraint that cancels tangential drift of
//! the contact anchors while the required tangential impulse stays inside the
//! Coulomb cone (`lambda_t <= mu_s * lambda_n`).
//!
//! The working set is built once per sub-step from the narrow-phase manifolds
//! ([`ContactConstraint::build`]) so the anchors and the pre-solve normal
//! velocity are captured against the predicted poses, then solved in place by
//! [`solve_positions`]. The accumulated normal impulse `lambda_n` and the
//! captured pre-solve normal velocity are read back by the velocity pass.
//!
//! # Provenance
//!
//! The compliant position update (`delta_lambda = (c - alpha_tilde * lambda) /
//! (w + alpha_tilde)`), the generalized-inverse-mass weighting, and the
//! static-friction gate follow Müller et al., *Detailed Rigid Body Simulation
//! with Extended Position Based Dynamics* (2020). This file contains no Unreal
//! Engine source or derived code.

use super::config::XpbdConfig;
use super::rigid::{
    apply_position_impulse, generalized_inverse_mass, inv_inertia_world, inv_mass_world,
};
use crate::collide::contact::MAX_MANIFOLD_POINTS;
use crate::collide::ContactManifold;
use crate::collider::PhysicsMaterial;
use crate::state::view::BodySolverView;
use glam::{Mat3, Vec3};

/// Threshold below which an effective inverse mass or a length is treated as
/// numerically zero.
const SOLVE_EPS: f32 = 1.0e-9;

/// Per-point working state for a single contact within a constraint.
#[derive(Clone, Copy, Debug, Default)]
struct PointState {
    /// Contact anchor on body `a`, expressed in body `a`'s local frame.
    anchor_a_local: Vec3,
    /// Contact anchor on body `b`, expressed in body `b`'s local frame.
    anchor_b_local: Vec3,
    /// Relative normal velocity captured right after prediction, before the
    /// position solve. Positive means the bodies are approaching.
    normal_velocity_pre: f32,
    /// Accumulated normal position impulse for this point.
    normal_lambda: f32,
}

/// A solved contact between two bodies, addressed by storage slot index.
#[derive(Clone, Debug)]
pub struct ContactConstraint {
    /// Storage slot of body `a`.
    pub slot_a: usize,
    /// Storage slot of body `b`.
    pub slot_b: usize,
    /// Unit contact normal pointing from body `a` toward body `b`.
    pub normal: Vec3,
    /// Combined static/dynamic friction coefficient for the pair.
    pub friction: f32,
    /// Combined restitution coefficient for the pair.
    pub restitution: f32,
    /// Inline working state; cloning scratch contacts never allocates.
    points: [PointState; MAX_MANIFOLD_POINTS],
    /// Only this prefix of the inline buffer contains manifold points.
    point_count: usize,
}

impl ContactConstraint {
    /// Builds the sub-step contact working set from the narrow-phase manifolds.
    ///
    /// Anchors are captured in each body's local frame so the solve can follow
    /// the bodies as they rotate, and the pre-solve relative normal velocity is
    /// recorded for the restitution pass. Manifolds referencing an inactive
    /// slot or carrying no dynamic body are skipped.
    #[must_use]
    pub fn build(
        view: &BodySolverView<'_>,
        manifolds: &[ContactManifold],
    ) -> Vec<ContactConstraint> {
        let mut out = Vec::with_capacity(manifolds.len());
        for manifold in manifolds {
            let slot_a = manifold.body_a.index() as usize;
            let slot_b = manifold.body_b.index() as usize;
            if slot_a >= view.slot_count() || slot_b >= view.slot_count() {
                continue;
            }
            if !view.is_active(slot_a) || !view.is_active(slot_b) {
                continue;
            }
            if !view.is_dynamic(slot_a) && !view.is_dynamic(slot_b) {
                continue;
            }
            // Sensors (trigger volumes) participate in overlap detection and
            // event generation only; they never receive a physical response.
            if view.is_sensor(slot_a) || view.is_sensor(slot_b) {
                continue;
            }

            let material = PhysicsMaterial::combine(
                manifold.leaf_materials[0].unwrap_or(view.materials[slot_a]),
                manifold.leaf_materials[1].unwrap_or(view.materials[slot_b]),
            );
            let normal = manifold.normal;
            let q_a = view.orientations[slot_a];
            let q_b = view.orientations[slot_b];
            let inv_q_a = q_a.inverse();
            let inv_q_b = q_b.inverse();

            let mut points = [PointState::default(); MAX_MANIFOLD_POINTS];
            for (point, contact) in points.iter_mut().zip(manifold.points()) {
                let r_a = contact.point_a - view.positions[slot_a];
                let r_b = contact.point_b - view.positions[slot_b];
                let vel_a = point_velocity(view, slot_a, r_a);
                let vel_b = point_velocity(view, slot_b, r_b);
                let normal_velocity_pre = (vel_a - vel_b).dot(normal);
                *point = PointState {
                    anchor_a_local: inv_q_a * r_a,
                    anchor_b_local: inv_q_b * r_b,
                    normal_velocity_pre,
                    normal_lambda: 0.0,
                };
            }
            if manifold.is_empty() {
                continue;
            }
            out.push(ContactConstraint {
                slot_a,
                slot_b,
                normal,
                friction: material.friction,
                restitution: material.restitution,
                points,
                point_count: manifold.len(),
            });
        }
        out
    }

    /// Returns the accumulated normal impulse of point `index`, or `0` when the
    /// index is out of range.
    #[must_use]
    pub fn normal_lambda(&self, index: usize) -> f32 {
        self.points[..self.point_count]
            .get(index)
            .map_or(0.0, |p| p.normal_lambda)
    }

    /// Returns the pre-solve relative normal velocity of point `index`.
    #[must_use]
    pub fn normal_velocity_pre(&self, index: usize) -> f32 {
        self.points[..self.point_count]
            .get(index)
            .map_or(0.0, |p| p.normal_velocity_pre)
    }

    /// Returns the world-space contact anchors `(r_a, r_b)` of point `index` for
    /// the current orientations in `view`.
    #[must_use]
    pub fn anchors(&self, view: &BodySolverView<'_>, index: usize) -> (Vec3, Vec3) {
        let p = self.points[..self.point_count][index];
        (
            view.orientations[self.slot_a] * p.anchor_a_local,
            view.orientations[self.slot_b] * p.anchor_b_local,
        )
    }

    /// Returns the number of contact points in this constraint.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.point_count
    }
}

/// Runs one position-solve iteration over every contact constraint.
///
/// For each point the current penetration is recomputed from the live anchors,
/// resolved with a compliant XPBD position update along the normal, and then a
/// static-friction correction cancels the tangential drift of the anchors when
/// it stays inside the Coulomb cone.
/// Returns the compliant position-solve stiffness `alpha_tilde` for a
/// sub-step of length `h`.
pub(crate) fn contact_alpha_tilde(config: &XpbdConfig, h: f32) -> f32 {
    if h > 0.0 {
        config.contact_compliance / (h * h)
    } else {
        0.0
    }
}

/// Runs one position-solve iteration over a single contact constraint.
pub(crate) fn solve_constraint_positions(
    view: &mut BodySolverView<'_>,
    constraint: &mut ContactConstraint,
    alpha_tilde: f32,
) {
    let slot_a = constraint.slot_a;
    let slot_b = constraint.slot_b;
    let normal = constraint.normal;
    let friction = constraint.friction;
    for index in 0..constraint.point_count {
        solve_point(
            view,
            constraint,
            slot_a,
            slot_b,
            normal,
            friction,
            index,
            alpha_tilde,
        );
    }
}

/// Runs one position-solve iteration over every contact constraint.
///
/// This is the global entry point used when islands are not in play; it applies
/// the compliant normal + static-friction correction to each constraint in
/// order. The per-island [`solve_positions_indexed`] variant restricts the pass
/// to a subset of constraint indices.
pub fn solve_positions(
    view: &mut BodySolverView<'_>,
    constraints: &mut [ContactConstraint],
    config: &XpbdConfig,
    h: f32,
) {
    let alpha_tilde = contact_alpha_tilde(config, h);
    for constraint in constraints.iter_mut() {
        solve_constraint_positions(view, constraint, alpha_tilde);
    }
}

/// Runs one position-solve iteration over the contact constraints named by
/// `indices` (used by the per-island solver).
///
/// Each entry of `indices` is an index into `constraints`; out-of-range
/// indices are ignored. Solving disjoint islands through this entry point is
/// numerically identical to a single global [`solve_positions`] pass because
/// the islands touch disjoint sets of dynamic bodies.
pub fn solve_positions_indexed(
    view: &mut BodySolverView<'_>,
    constraints: &mut [ContactConstraint],
    indices: &[usize],
    config: &XpbdConfig,
    h: f32,
) {
    let alpha_tilde = contact_alpha_tilde(config, h);
    for &i in indices {
        if let Some(constraint) = constraints.get_mut(i) {
            solve_constraint_positions(view, constraint, alpha_tilde);
        }
    }
}

/// Solves the normal and static-friction correction for a single contact point.
#[expect(
    clippy::too_many_arguments,
    reason = "the per-contact-point position solve needs the full body pair, anchors, and constraint state"
)]
fn solve_point(
    view: &mut BodySolverView<'_>,
    constraint: &mut ContactConstraint,
    slot_a: usize,
    slot_b: usize,
    normal: Vec3,
    friction: f32,
    index: usize,
    alpha_tilde: f32,
) {
    let (inv_mass_a, inv_mass_b) = (
        inv_mass_world_for(view, slot_a),
        inv_mass_world_for(view, slot_b),
    );
    let (inertia_a, inertia_b) = (inertia_world(view, slot_a), inertia_world(view, slot_b));

    let anchor_a_local = constraint.points[index].anchor_a_local;
    let anchor_b_local = constraint.points[index].anchor_b_local;

    let r_a = view.orientations[slot_a] * anchor_a_local;
    let r_b = view.orientations[slot_b] * anchor_b_local;
    let world_a = view.positions[slot_a] + r_a;
    let world_b = view.positions[slot_b] + r_b;

    // Positive penetration means the anchors still overlap along the normal.
    let penetration = (world_a - world_b).dot(normal);
    if penetration <= 0.0 {
        return;
    }

    let w_a = generalized_inverse_mass(inv_mass_a, inertia_a, r_a, normal);
    let w_b = generalized_inverse_mass(inv_mass_b, inertia_b, r_b, normal);
    let w_sum = w_a + w_b;
    if w_sum <= SOLVE_EPS {
        return;
    }

    let lambda = constraint.points[index].normal_lambda;
    let delta_lambda = (penetration - alpha_tilde * lambda) / (w_sum + alpha_tilde);
    if delta_lambda <= 0.0 {
        return;
    }
    constraint.points[index].normal_lambda = lambda + delta_lambda;

    let impulse = normal * delta_lambda;
    // Normal points a -> b: push a back along -normal and b along +normal.
    apply_pair(
        view, slot_a, slot_b, inertia_a, inertia_b, r_a, r_b, -impulse,
    );

    apply_static_friction(
        view,
        constraint,
        slot_a,
        slot_b,
        normal,
        friction,
        index,
        anchor_a_local,
        anchor_b_local,
        inertia_a,
        inertia_b,
    );
}

/// Applies the static-friction position correction for one contact point.
#[expect(
    clippy::too_many_arguments,
    reason = "static friction operates on the full body pair, both anchors, and the tangent budget"
)]
fn apply_static_friction(
    view: &mut BodySolverView<'_>,
    constraint: &ContactConstraint,
    slot_a: usize,
    slot_b: usize,
    normal: Vec3,
    friction: f32,
    index: usize,
    anchor_a_local: Vec3,
    anchor_b_local: Vec3,
    inertia_a: Mat3,
    inertia_b: Mat3,
) {
    if friction <= 0.0 {
        return;
    }
    let r_a = view.orientations[slot_a] * anchor_a_local;
    let r_b = view.orientations[slot_b] * anchor_b_local;
    let world_a = view.positions[slot_a] + r_a;
    let world_b = view.positions[slot_b] + r_b;

    let prev_a = view.prev_positions[slot_a] + view.prev_orientations[slot_a] * anchor_a_local;
    let prev_b = view.prev_positions[slot_b] + view.prev_orientations[slot_b] * anchor_b_local;

    // Relative tangential drift of the anchors since the sub-step start.
    let drift = (world_a - prev_a) - (world_b - prev_b);
    let tangential = drift - normal * drift.dot(normal);
    let magnitude = tangential.length();
    if magnitude <= SOLVE_EPS {
        return;
    }
    let tangent = tangential / magnitude;

    let w_a = generalized_inverse_mass(inv_mass_world_for(view, slot_a), inertia_a, r_a, tangent);
    let w_b = generalized_inverse_mass(inv_mass_world_for(view, slot_b), inertia_b, r_b, tangent);
    let w_sum = w_a + w_b;
    if w_sum <= SOLVE_EPS {
        return;
    }
    let delta_lambda_t = magnitude / w_sum;

    // Static friction only holds while it stays inside the Coulomb cone.
    if delta_lambda_t >= friction * constraint.points[index].normal_lambda {
        return;
    }

    let impulse = tangent * delta_lambda_t;
    // Move a opposite to the drift and b along it to cancel the tangential slip.
    apply_pair(
        view, slot_a, slot_b, inertia_a, inertia_b, r_a, r_b, -impulse,
    );
}

/// Applies `+impulse` to body `a` and `-impulse` to body `b` at their anchors.
#[expect(
    clippy::too_many_arguments,
    reason = "applying a symmetric impulse touches both bodies plus their anchors and inertia"
)]
fn apply_pair(
    view: &mut BodySolverView<'_>,
    slot_a: usize,
    slot_b: usize,
    inertia_a: Mat3,
    inertia_b: Mat3,
    r_a: Vec3,
    r_b: Vec3,
    impulse: Vec3,
) {
    if view.is_dynamic(slot_a) {
        let inv_mass_a = inv_mass_world_for(view, slot_a);
        let mut x = view.positions[slot_a];
        let mut q = view.orientations[slot_a];
        apply_position_impulse(&mut x, &mut q, inv_mass_a, inertia_a, r_a, impulse);
        view.positions[slot_a] = x;
        view.orientations[slot_a] = q;
    }
    if view.is_dynamic(slot_b) {
        let inv_mass_b = inv_mass_world_for(view, slot_b);
        let mut x = view.positions[slot_b];
        let mut q = view.orientations[slot_b];
        apply_position_impulse(&mut x, &mut q, inv_mass_b, inertia_b, r_b, -impulse);
        view.positions[slot_b] = x;
        view.orientations[slot_b] = q;
    }
}

/// Returns the world-space velocity of the point at lever arm `r` on `slot`.
#[must_use]
fn point_velocity(view: &BodySolverView<'_>, slot: usize, r: Vec3) -> Vec3 {
    view.linear_velocities[slot] + view.angular_velocities[slot].cross(r)
}

/// Returns the effective inverse mass of `slot`, or `0` for non-dynamic bodies.
#[must_use]
fn inv_mass_world_for(view: &BodySolverView<'_>, slot: usize) -> Mat3 {
    if view.is_dynamic(slot) {
        inv_mass_world(view.mass_props[slot].inv_mass, view.orientations[slot])
    } else {
        Mat3::ZERO
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
    use crate::state::body::{BodyDesc, MassProperties, SolverMassProperties};
    use crate::state::storage::BodyStorage;
    use glam::Quat;

    const EPS: f32 = 1.0e-6;

    fn assert_scalar_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPS,
            "actual {actual}, expected {expected}"
        );
    }

    fn assert_vector_close(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).abs().max_element() < EPS,
            "actual {actual:?}, expected {expected:?}"
        );
    }

    fn assert_orientation_close(actual: Quat, expected: Quat) {
        for (actual, expected) in actual.to_array().into_iter().zip(expected.to_array()) {
            assert_scalar_close(actual, expected);
        }
    }

    // Zero lever arms isolate translational response. The positive overlap is
    // 0.2, with the normal pointing from a to b as required by the solver.
    fn centered_pair(normal: Vec3) -> (BodyStorage, ContactManifold) {
        let mut storage = BodyStorage::new();
        let ha =
            storage.insert(unit_dynamic(normal * 0.2).with_material(PhysicsMaterial::FRICTIONLESS));
        let hb =
            storage.insert(unit_dynamic(Vec3::ZERO).with_material(PhysicsMaterial::FRICTIONLESS));
        let mut manifold = ContactManifold::new(ha, hb, normal);
        manifold.push(ContactPoint::new(normal * 0.2, Vec3::ZERO, 0.2));
        (storage, manifold)
    }

    fn unit_dynamic(pos: Vec3) -> BodyDesc {
        BodyDesc::dynamic_at(pos).with_mass_properties(MassProperties {
            inv_mass: 1.0,
            inv_inertia: Vec3::ONE,
        })
    }

    #[test]
    fn inline_point_prefix_preserves_all_manifold_sizes_and_checked_access() {
        for count in 0..=MAX_MANIFOLD_POINTS {
            let mut bodies = BodyStorage::new();
            let a = bodies.insert(unit_dynamic(Vec3::ZERO));
            let b = bodies.insert(unit_dynamic(Vec3::Y));
            let mut manifold = ContactManifold::new(a, b, Vec3::Y);
            for i in 0..count {
                let anchor_a = Vec3::new(i as f32 * 0.1, 0.6, 0.0);
                assert!(manifold.push(ContactPoint::new(anchor_a, anchor_a - Vec3::Y * 0.2, 0.2)));
            }
            let view = bodies.solver_view_mut();
            let constraints = ContactConstraint::build(&view, &[manifold]);
            if count == 0 {
                assert!(constraints.is_empty());
                continue;
            }
            let constraint = constraints[0].clone();
            assert_eq!(constraint.point_count(), count);
            for (i, point) in manifold.points().iter().enumerate() {
                let (r_a, r_b) = constraint.anchors(&view, i);
                assert_eq!(r_a, point.point_a - view.positions[a.index() as usize]);
                assert_eq!(r_b, point.point_b - view.positions[b.index() as usize]);
                assert_eq!(constraint.normal_lambda(i), 0.0);
                assert_eq!(constraint.normal_velocity_pre(i), 0.0);
            }
            for invalid in [count, MAX_MANIFOLD_POINTS, usize::MAX] {
                assert_eq!(constraint.normal_lambda(invalid), 0.0);
                assert_eq!(constraint.normal_velocity_pre(invalid), 0.0);
            }
        }
    }

    #[test]
    fn normal_solve_separates_overlapping_bodies() {
        let mut s = BodyStorage::new();
        // b penetrates a by 0.2 along +Y (normal a -> b points +Y).
        let ha = s.insert(unit_dynamic(Vec3::ZERO));
        let hb = s.insert(unit_dynamic(Vec3::new(0.0, 0.8, 0.0)));
        let normal = Vec3::Y;
        let mut manifold = ContactManifold::new(ha, hb, normal);
        manifold.push(ContactPoint::new(
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.3, 0.0),
            0.2,
        ));
        let config = XpbdConfig::default();
        let mut view = s.solver_view_mut();
        let mut constraints = ContactConstraint::build(&view, &[manifold]);
        solve_positions(&mut view, &mut constraints, &config, 0.016);
        // Equal inverse mass: each body moves ~0.1 apart along the normal.
        assert!(view.positions[0].y < -0.05);
        assert!(view.positions[1].y > 0.85);
        assert!(constraints[0].normal_lambda(0) > 0.0);
    }

    #[test]
    fn separated_contact_is_not_pushed() {
        let mut s = BodyStorage::new();
        let ha = s.insert(unit_dynamic(Vec3::ZERO));
        let hb = s.insert(unit_dynamic(Vec3::new(0.0, 2.0, 0.0)));
        // Non-overlapping witness points => negative penetration => no move.
        let mut manifold = ContactManifold::new(ha, hb, Vec3::Y);
        manifold.push(ContactPoint::new(
            Vec3::new(0.0, 0.4, 0.0),
            Vec3::new(0.0, 0.6, 0.0),
            0.0,
        ));
        let config = XpbdConfig::default();
        let mut view = s.solver_view_mut();
        let mut constraints = ContactConstraint::build(&view, &[manifold]);
        solve_positions(&mut view, &mut constraints, &config, 0.016);
        assert_eq!(view.positions[0], Vec3::ZERO);
        assert_eq!(view.positions[1], Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn static_body_absorbs_all_correction() {
        let mut s = BodyStorage::new();
        let ground = s.insert(BodyDesc::static_at(Vec3::ZERO));
        let hb = s.insert(unit_dynamic(Vec3::new(0.0, -0.2, 0.0)));
        // Ground a faces up; dynamic box b sank 0.2 below the surface. The
        // witness points (point_a on the surface, point_b on the sunk box) and
        // penetration fix the contract normal to +Y: point_a == point_b + pen*n.
        let mut manifold = ContactManifold::new(ground, hb, Vec3::Y);
        manifold.push(ContactPoint::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -0.2, 0.0),
            0.2,
        ));
        let config = XpbdConfig::default();
        let mut view = s.solver_view_mut();
        let mut constraints = ContactConstraint::build(&view, &[manifold]);
        solve_positions(&mut view, &mut constraints, &config, 0.016);
        // Static ground unmoved; dynamic body lifted back to the surface.
        assert_eq!(view.positions[0], Vec3::ZERO);
        assert!((view.positions[1].y - 0.0).abs() < 1e-3);
    }

    #[test]
    fn anisotropic_solver_mass_changes_contact_projection() {
        let effective = SolverMassProperties {
            inv_mass: Vec3::new(0.1, 1.0, 0.25),
            inv_inertia: Vec3::ONE,
        };
        for q in [
            Quat::IDENTITY,
            Quat::from_rotation_z(core::f32::consts::FRAC_PI_4),
        ] {
            for normal in [Vec3::X, Vec3::Y, Vec3::Z, (Vec3::X + Vec3::Y).normalize()] {
                for alpha in [0.0, 0.25] {
                    let (mut storage, manifold) = centered_pair(normal);
                    assert!(storage.set_orientation(manifold.body_a, q));
                    assert!(storage.set_solver_mass_properties(manifold.body_a, effective));
                    // Only a is modified: otherwise equal weighting can mask a
                    // bad effective-mass implementation in a rigid projection.
                    let response = q * (effective.inv_mass * (q.conjugate() * normal));
                    let w_a = normal.dot(response);
                    let lambda = 0.2 / (w_a + 1.0 + alpha);
                    let h = 0.016;
                    let config = XpbdConfig {
                        contact_compliance: alpha * h * h,
                        ..XpbdConfig::default()
                    };
                    let mut view = storage.solver_view_mut();
                    let mut constraints = ContactConstraint::build(&view, &[manifold]);
                    solve_positions(&mut view, &mut constraints, &config, h);
                    assert_scalar_close(constraints[0].normal_lambda(0), lambda);
                    assert_vector_close(view.positions[0], normal * 0.2 - response * lambda);
                    assert_vector_close(view.positions[1], normal * lambda);
                    assert_orientation_close(view.orientations[0], q);
                    assert_orientation_close(view.orientations[1], Quat::IDENTITY);
                    let residual = (view.positions[0] - view.positions[1]).dot(normal);
                    assert_scalar_close(residual, alpha * lambda);
                }
            }
        }
    }

    #[test]
    fn anisotropic_equal_pair_keeps_rigid_projection_but_changes_lambda() {
        for normal in [Vec3::X, Vec3::Y, Vec3::Z] {
            for diagonal in [Vec3::ONE, Vec3::new(0.1, 1.0, 0.25)] {
                let (mut storage, manifold) = centered_pair(normal);
                let effective = SolverMassProperties {
                    inv_mass: diagonal,
                    inv_inertia: Vec3::ONE,
                };
                assert!(storage.set_solver_mass_properties(manifold.body_a, effective));
                assert!(storage.set_solver_mass_properties(manifold.body_b, effective));
                let mut view = storage.solver_view_mut();
                let mut constraints = ContactConstraint::build(&view, &[manifold]);
                solve_positions(&mut view, &mut constraints, &XpbdConfig::default(), 0.016);
                // w_a == w_b: each moves 0.1, even when both become heavier.
                assert_vector_close(view.positions[0], normal * 0.1);
                assert_vector_close(view.positions[1], normal * 0.1);
                assert_scalar_close(
                    constraints[0].normal_lambda(0),
                    0.2 / (2.0 * normal.dot(diagonal * normal)),
                );
            }
        }
    }

    #[test]
    fn anisotropic_tangent_mass_controls_friction_sharing_and_coulomb_gate() {
        let diagonal = Vec3::new(0.1, 1.0, 0.5);
        for q in [
            Quat::IDENTITY,
            Quat::from_rotation_y(core::f32::consts::FRAC_PI_2),
        ] {
            for tangent in [Vec3::X, Vec3::Z] {
                for drift in [0.02, 0.07] {
                    let (mut storage, manifold) = centered_pair(Vec3::Y);
                    let material = PhysicsMaterial::new(0.5, 0.0);
                    assert!(storage.set_material(manifold.body_a, material));
                    assert!(storage.set_material(manifold.body_b, material));
                    assert!(storage.set_orientation(manifold.body_a, q));
                    assert!(storage.set_prev_orientation(manifold.body_a, q));
                    assert!(
                        storage.set_prev_position(manifold.body_a, Vec3::Y * 0.2 - tangent * drift)
                    );
                    assert!(storage.set_solver_mass_properties(
                        manifold.body_a,
                        SolverMassProperties {
                            inv_mass: diagonal,
                            inv_inertia: Vec3::ONE
                        },
                    ));
                    let response = q * (diagonal * (q.conjugate() * tangent));
                    let lambda_t = drift / (1.0 + tangent.dot(response));
                    let holds = lambda_t < 0.5 * 0.1;
                    // At drift 0.07, w_t=1.1 slips but w_t=1.5 sticks.
                    let correction = if holds { lambda_t } else { 0.0 };
                    let mut view = storage.solver_view_mut();
                    let mut constraints = ContactConstraint::build(&view, &[manifold]);
                    solve_positions(&mut view, &mut constraints, &XpbdConfig::default(), 0.016);
                    assert_scalar_close(constraints[0].normal_lambda(0), 0.1);
                    assert_vector_close(view.positions[0], Vec3::Y * 0.1 - response * correction);
                    assert_vector_close(view.positions[1], Vec3::Y * 0.1 + tangent * correction);
                    let remaining_drift = (view.positions[0] - view.prev_positions[0])
                        - (view.positions[1] - view.prev_positions[1]);
                    assert_scalar_close(
                        remaining_drift.dot(tangent),
                        if holds { 0.0 } else { drift },
                    );
                    assert_orientation_close(view.orientations[0], q);
                }
            }
        }
    }

    #[test]
    fn anisotropic_static_body_ignores_nonzero_solver_mass_on_either_side() {
        for static_first in [false, true] {
            let mut storage = BodyStorage::new();
            let q = Quat::from_rotation_z(core::f32::consts::FRAC_PI_2);
            let ground = storage.insert(BodyDesc::static_at(Vec3::ZERO));
            let dynamic = storage
                .insert(unit_dynamic(-Vec3::Y * 0.2).with_material(PhysicsMaterial::FRICTIONLESS));
            assert!(storage.set_orientation(dynamic, q));
            let effective = SolverMassProperties {
                inv_mass: Vec3::new(0.1, 1.0, 0.25),
                inv_inertia: Vec3::new(2.0, 3.0, 4.0),
            };
            // Deliberately nonzero properties must not make a static body
            // contribute to either the denominator or the applied correction.
            assert!(storage.set_solver_mass_properties(ground, effective));
            assert!(storage.set_solver_mass_properties(dynamic, effective));
            let mut manifold = if static_first {
                ContactManifold::new(ground, dynamic, Vec3::Y)
            } else {
                ContactManifold::new(dynamic, ground, -Vec3::Y)
            };
            let ground_point = Vec3::X;
            let dynamic_point = ground_point - Vec3::Y * 0.2;
            let (a, b) = if static_first {
                (ground_point, dynamic_point)
            } else {
                (dynamic_point, ground_point)
            };
            manifold.push(ContactPoint::new(a, b, 0.2));
            let mut view = storage.solver_view_mut();
            let mut constraints = ContactConstraint::build(&view, &[manifold]);
            solve_positions(&mut view, &mut constraints, &XpbdConfig::default(), 0.016);
            // Dynamic arm X adds I^-1_zz=4; world M^-1_yy=0.1.
            let lambda = 0.2 / (0.1 + 4.0);
            assert_scalar_close(constraints[0].normal_lambda(0), lambda);
            assert_vector_close(view.positions[0], Vec3::ZERO);
            assert_orientation_close(view.orientations[0], Quat::IDENTITY);
            assert_vector_close(view.positions[1], Vec3::Y * (-0.2 + 0.1 * lambda));
            let rotation = Quat::from_xyzw(0.0, 0.0, 0.5 * 4.0 * lambda, 1.0).normalize();
            assert_orientation_close(view.orientations[1], rotation * q);
        }
    }

    #[test]
    fn anisotropic_rotated_inertia_controls_off_center_contact_projection() {
        let mass = SolverMassProperties {
            inv_mass: Vec3::new(0.25, 0.5, 1.0),
            inv_inertia: Vec3::new(2.0, 3.0, 5.0),
        };
        for q in [
            Quat::IDENTITY,
            Quat::from_rotation_x(core::f32::consts::FRAC_PI_2),
            Quat::from_rotation_y(core::f32::consts::FRAC_PI_4),
        ] {
            let mut storage = BodyStorage::new();
            let dynamic = storage
                .insert(unit_dynamic(Vec3::ZERO).with_material(PhysicsMaterial::FRICTIONLESS));
            let ground = storage.insert(BodyDesc::static_at(Vec3::X));
            assert!(storage.set_orientation(dynamic, q));
            assert!(storage.set_solver_mass_properties(dynamic, mass));
            let mut manifold = ContactManifold::new(dynamic, ground, Vec3::Y);
            manifold.push(ContactPoint::new(Vec3::X + Vec3::Y * 0.2, Vec3::X, 0.2));
            // r x n = Z. Compute expectations in body coordinates, independent
            // of the production world-tensor/generalized-mass helpers.
            let linear = q * (mass.inv_mass * (q.conjugate() * Vec3::Y));
            let angular = q * (mass.inv_inertia * (q.conjugate() * Vec3::Z));
            let lambda = 0.2 / (Vec3::Y.dot(linear) + Vec3::Z.dot(angular));
            let dphi = -angular * lambda;
            let rotation =
                Quat::from_xyzw(0.5 * dphi.x, 0.5 * dphi.y, 0.5 * dphi.z, 1.0).normalize();
            let mut view = storage.solver_view_mut();
            let mut constraints = ContactConstraint::build(&view, &[manifold]);
            solve_positions(&mut view, &mut constraints, &XpbdConfig::default(), 0.016);
            assert_scalar_close(constraints[0].normal_lambda(0), lambda);
            assert_vector_close(view.positions[0], -linear * lambda);
            assert_orientation_close(view.orientations[0], rotation * q);
            assert_vector_close(view.positions[1], Vec3::X);
            assert_orientation_close(view.orientations[1], Quat::IDENTITY);
        }
    }

    #[test]
    fn isotropic_solver_mass_reduces_to_scalar_contact_solution() {
        let normal = (Vec3::X + Vec3::Y).normalize();
        let tangent = Vec3::Z;
        let q = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.4);
        let (inv_a, inv_b) = (0.4, 0.8);
        for override_mass in [false, true] {
            for alpha in [0.0, 0.25] {
                let mut storage = BodyStorage::new();
                let mut a = unit_dynamic(normal * 0.2);
                a.orientation = q;
                a.mass_properties.inv_mass = inv_a;
                let mut b = unit_dynamic(Vec3::ZERO);
                b.orientation = q;
                b.mass_properties.inv_mass = inv_b;
                let ha = storage.insert(a);
                let hb = storage.insert(b);
                assert!(storage.set_prev_position(ha, a.position - tangent * 0.02));
                if override_mass {
                    for (body, inv_mass) in [(ha, inv_a), (hb, inv_b)] {
                        assert!(storage.set_solver_mass_properties(
                            body,
                            SolverMassProperties {
                                inv_mass: Vec3::splat(inv_mass),
                                inv_inertia: Vec3::ONE,
                            }
                        ));
                    }
                }
                let mut manifold = ContactManifold::new(ha, hb, normal);
                manifold.push(ContactPoint::new(a.position, b.position, 0.2));
                let h = 0.016;
                let config = XpbdConfig {
                    contact_compliance: alpha * h * h,
                    ..XpbdConfig::default()
                };
                let lambda_n = 0.2 / (inv_a + inv_b + alpha);
                let lambda_t = 0.02 / (inv_a + inv_b);
                assert!(lambda_t < 0.5 * lambda_n);
                let correction = normal * lambda_n + tangent * lambda_t;
                let mut view = storage.solver_view_mut();
                let mut constraints = ContactConstraint::build(&view, &[manifold]);
                solve_positions(&mut view, &mut constraints, &config, h);
                assert_scalar_close(constraints[0].normal_lambda(0), lambda_n);
                assert_vector_close(view.positions[0], a.position - correction * inv_a);
                assert_vector_close(view.positions[1], b.position + correction * inv_b);
                assert_orientation_close(view.orientations[0], q);
                assert_orientation_close(view.orientations[1], q);
            }
        }
    }
}
