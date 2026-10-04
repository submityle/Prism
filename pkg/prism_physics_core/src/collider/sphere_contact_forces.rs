//! Hertzian contact forces over a precomputed sphere-contact set.
//!
//! The monolithic [`resolve_hertz_contacts`](crate::collider::hertz_contact_resolver::resolve_hertz_contacts)
//! rediscovers the contacting pairs with its own internal grid every call. That
//! couples the geometry (which grains touch, and with what normal and overlap)
//! to the force law. This module separates the two: it takes an already-built
//! contact set — for example from
//! [`SphereNarrowPhase`](crate::collider::sphere_narrow_phase::SphereNarrowPhase)
//! — and evaluates the Hertz normal/tangential force on each contact, summing
//! the per-grain forces.
//!
//! Decoupling the narrow phase from the force law lets a single contact set be
//! shared by several responses (normal contact, friction, cohesion) without
//! recomputing the geometry, and lets either half be replaced independently.
//! The force law itself is reused verbatim from
//! [`evaluate_hertz_contact`](crate::collider::hertz_contact::evaluate_hertz_contact),
//! so there is a single source of truth for the contact physics.

use glam::Vec3;

use crate::collider::hertz_contact::{evaluate_hertz_contact, HertzModel};
use crate::collider::sphere_narrow_phase::SphereContact;

/// Per-grain forces and summary diagnostics from
/// [`resolve_sphere_contact_forces`].
#[derive(Clone, Debug, PartialEq)]
pub struct SphereContactForceResolution {
    forces: Vec<Vec3>,
    contact_count: u32,
    max_normal_force: f32,
    sliding_count: u32,
}

impl SphereContactForceResolution {
    /// Per-grain accumulated contact force (index-aligned with the cloud).
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// Number of contacts that produced a non-zero evaluation.
    pub fn contact_count(&self) -> u32 {
        self.contact_count
    }

    /// Largest single-contact normal force magnitude.
    pub fn max_normal_force(&self) -> f32 {
        self.max_normal_force
    }

    /// Number of contacts whose friction reached the Coulomb (sliding) limit.
    pub fn sliding_count(&self) -> u32 {
        self.sliding_count
    }

    /// Vector sum of all grain forces; near zero because contacts are
    /// equal-and-opposite.
    pub fn total_force(&self) -> Vec3 {
        self.forces.iter().copied().fold(Vec3::ZERO, |a, f| a + f)
    }
}

/// Evaluate Hertz contact forces over a precomputed contact set.
///
/// `radii` and `velocities` describe the grain cloud and must have equal
/// length; every radius must be finite and strictly positive and every velocity
/// finite. Each [`SphereContact`] references grain indices into that cloud; any
/// out-of-range index makes the call fail. Contacts with non-positive
/// penetration (near pairs reported under a detection margin) contribute no
/// force. Returns `None` on invalid input.
///
/// For a contact between `a` and `b` with unit normal pointing `a -> b`, the
/// force on `b` is `evaluate_hertz_contact(..)` and `a` receives its negation,
/// so linear momentum is conserved.
pub fn resolve_sphere_contact_forces(
    contacts: &[SphereContact],
    radii: &[f32],
    velocities: &[Vec3],
    model: &HertzModel,
) -> Option<SphereContactForceResolution> {
    let n = radii.len();
    if velocities.len() != n {
        return None;
    }
    for (&r, vel) in radii.iter().zip(velocities.iter()) {
        if !r.is_finite() || r <= 0.0 || !vel.is_finite() {
            return None;
        }
    }

    let mut forces = vec![Vec3::ZERO; n];
    let mut contact_count: u32 = 0;
    let mut max_normal_force = 0.0_f32;
    let mut sliding_count: u32 = 0;

    for contact in contacts.iter() {
        let a = contact.a as usize;
        let b = contact.b as usize;
        if a >= n || b >= n || a == b {
            return None;
        }
        if !(contact.penetration.is_finite() && contact.penetration > 0.0) {
            continue;
        }

        let radius_a = radii[a];
        let radius_b = radii[b];
        let effective_radius = (radius_a * radius_b) / (radius_a + radius_b);
        let rel_vel = velocities[b] - velocities[a];
        let force = evaluate_hertz_contact(
            model,
            contact.normal,
            contact.penetration,
            effective_radius,
            rel_vel,
        );
        if force.normal_magnitude <= 0.0 && force.tangential_magnitude <= 0.0 {
            continue;
        }

        forces[b] += force.force_on_b;
        forces[a] -= force.force_on_b;
        contact_count += 1;
        max_normal_force = max_normal_force.max(force.normal_magnitude);
        if force.sliding {
            sliding_count += 1;
        }
    }

    Some(SphereContactForceResolution {
        forces,
        contact_count,
        max_normal_force,
        sliding_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::hertz_contact_resolver::resolve_hertz_contacts;
    use crate::collider::sphere_narrow_phase::SphereNarrowPhase;

    fn model() -> HertzModel {
        HertzModel::new(1.0e7, 0.3, 10.0, 10.0, 0.5).unwrap()
    }

    fn contact(a: u32, b: u32, normal: Vec3, penetration: f32) -> SphereContact {
        SphereContact {
            a,
            b,
            normal,
            penetration,
            contact_point: Vec3::ZERO,
        }
    }

    #[test]
    fn empty_contacts_give_zero_forces() {
        let res =
            resolve_sphere_contact_forces(&[], &[1.0, 1.0], &[Vec3::ZERO, Vec3::ZERO], &model())
                .unwrap();
        assert_eq!(res.forces(), &[Vec3::ZERO, Vec3::ZERO]);
        assert_eq!(res.contact_count(), 0);
        assert_eq!(res.max_normal_force(), 0.0);
    }

    #[test]
    fn single_overlap_is_repulsive_and_newtonian() {
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let contacts = vec![contact(0, 1, Vec3::X, 0.1)];
        let res = resolve_sphere_contact_forces(&contacts, &radii, &velocities, &model()).unwrap();
        // Grain 0 pushed in -x, grain 1 in +x; equal and opposite.
        assert!(res.forces()[0].x < 0.0);
        assert!(res.forces()[1].x > 0.0);
        assert!((res.forces()[0] + res.forces()[1]).length() < 1.0e-6);
        assert!(res.max_normal_force() > 0.0);
        assert_eq!(res.contact_count(), 1);
    }

    #[test]
    fn matches_direct_force_law() {
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::new(-0.5, 0.0, 0.0)];
        let penetration = 0.1_f32;
        let contacts = vec![contact(0, 1, Vec3::X, penetration)];
        let res = resolve_sphere_contact_forces(&contacts, &radii, &velocities, &model()).unwrap();

        let effective_radius = (radii[0] * radii[1]) / (radii[0] + radii[1]);
        let rel_vel = velocities[1] - velocities[0];
        let direct =
            evaluate_hertz_contact(&model(), Vec3::X, penetration, effective_radius, rel_vel);
        assert!((res.forces()[1] - direct.force_on_b).length() < 1.0e-6);
        assert!((res.forces()[0] + direct.force_on_b).length() < 1.0e-6);
    }

    #[test]
    fn negative_penetration_contributes_no_force() {
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        // A near pair reported under a detection margin has negative penetration.
        let contacts = vec![contact(0, 1, Vec3::X, -0.05)];
        let res = resolve_sphere_contact_forces(&contacts, &radii, &velocities, &model()).unwrap();
        assert_eq!(res.forces(), &[Vec3::ZERO, Vec3::ZERO]);
        assert_eq!(res.contact_count(), 0);
    }

    #[test]
    fn rejects_out_of_range_index() {
        let radii = vec![1.0_f32, 1.0];
        let velocities = vec![Vec3::ZERO, Vec3::ZERO];
        let contacts = vec![contact(0, 5, Vec3::X, 0.1)];
        assert!(resolve_sphere_contact_forces(&contacts, &radii, &velocities, &model()).is_none());
    }

    #[test]
    fn rejects_invalid_cloud() {
        let model = model();
        let contacts = vec![contact(0, 1, Vec3::X, 0.1)];
        // Mismatched lengths.
        assert!(
            resolve_sphere_contact_forces(&contacts, &[1.0, 1.0], &[Vec3::ZERO], &model).is_none()
        );
        // Non-positive radius.
        assert!(resolve_sphere_contact_forces(
            &contacts,
            &[1.0, 0.0],
            &[Vec3::ZERO, Vec3::ZERO],
            &model
        )
        .is_none());
        // Non-finite velocity.
        assert!(resolve_sphere_contact_forces(
            &contacts,
            &[1.0, 1.0],
            &[Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)],
            &model
        )
        .is_none());
    }

    #[test]
    fn pipeline_matches_monolithic_resolver() {
        // Build a small overlapping cluster and compare the decoupled
        // narrow-phase + force pipeline against the monolithic resolver.
        let mut positions = Vec::new();
        let spacing = 1.8_f32; // < 2r so neighbours overlap
        for ix in 0..3 {
            for iy in 0..3 {
                positions.push(Vec3::new(ix as f32 * spacing, iy as f32 * spacing, 0.0));
            }
        }
        let n = positions.len();
        let radii = vec![1.0_f32; n];
        // Give grains distinct velocities so damping/friction terms are exercised.
        let mut velocities = Vec::with_capacity(n);
        for (i, _) in positions.iter().enumerate() {
            let s = i as f32;
            velocities.push(Vec3::new(0.1 * s, -0.05 * s, 0.0));
        }
        let model = model();

        let mut np = SphereNarrowPhase::new();
        let contacts = np.detect(&positions, &radii, 0.0).unwrap().to_vec();
        let decoupled =
            resolve_sphere_contact_forces(&contacts, &radii, &velocities, &model).unwrap();
        let monolithic = resolve_hertz_contacts(&positions, &radii, &velocities, &model).unwrap();

        assert_eq!(decoupled.forces().len(), monolithic.forces.len());
        for (d, m) in decoupled.forces().iter().zip(monolithic.forces.iter()) {
            assert!(
                (*d - *m).length() < 1.0e-3,
                "per-grain force mismatch: {d:?} vs {m:?}"
            );
        }
    }
}
