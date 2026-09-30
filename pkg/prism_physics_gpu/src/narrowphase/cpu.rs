//! `CPU` golden twin of the sphere-sphere narrow phase.
//!
//! [`cpu_narrowphase`] turns a set of broad-phase candidate pairs into contact
//! manifolds, running the shared [`sphere_sphere_contact`](super::sphere::sphere_sphere_contact)
//! geometry so its output matches the device kernel operation for operation. It
//! emits one slot per input pair, preserving order, which is what lets the
//! parity test line the `GPU` contacts up against this reference index by index.
//!
//! The twin is itself anchored against first principles in the unit tests: each
//! case checks the collision decision, normal direction, penetration depth, and
//! contact-point placement against hand-computed geometry, so correctness does
//! not rest on the `GPU` agreeing with it.
//!
//! Provenance: textbook sphere-sphere manifold; no Unreal Engine source or
//! derived code.

use super::contact::Contact;
use super::sphere::sphere_sphere_contact;
use crate::broadphase::{CandidatePair, Particle};

/// Generates sphere-sphere contacts for `pairs` over `particles`.
///
/// Returns one slot per pair, in input order: [`Some`] carrying the manifold
/// when the two spheres penetrate, or [`None`] when they are separated or
/// exactly touching. Keeping a slot per pair (rather than compacting) aligns the
/// contact index with the pair index for the device parity test; a later scan
/// stage compacts the survivors.
///
/// # Panics
///
/// Panics if a pair references a particle index outside `particles`, which is
/// never valid output from the broad phase over the same particle set.
#[must_use]
pub fn cpu_narrowphase(particles: &[Particle], pairs: &[CandidatePair]) -> Vec<Option<Contact>> {
    pairs
        .iter()
        .map(|pair| {
            let a = &particles[pair.a as usize];
            let b = &particles[pair.b as usize];
            sphere_sphere_contact(pair.a, pair.b, a.position, a.radius, b.position, b.radius)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// A particle at `(x, y, z)` with radius `r`.
    fn particle(x: f32, y: f32, z: f32, r: f32) -> Particle {
        Particle::new(Vec3::new(x, y, z), r)
    }

    #[test]
    fn separated_spheres_report_no_contact() {
        // Centres 4 apart, radii sum to 2: a clear gap, no overlap.
        let particles = [particle(0.0, 0.0, 0.0, 1.0), particle(4.0, 0.0, 0.0, 1.0)];
        let pairs = [CandidatePair::new(0, 1)];
        let contacts = cpu_narrowphase(&particles, &pairs);
        assert_eq!(contacts, vec![None]);
    }

    #[test]
    fn touching_spheres_report_no_contact() {
        // Centres exactly ra + rb apart: they share one boundary point but do
        // not penetrate, so the strict test rejects them.
        let particles = [particle(0.0, 0.0, 0.0, 1.0), particle(2.0, 0.0, 0.0, 1.0)];
        let pairs = [CandidatePair::new(0, 1)];
        assert_eq!(cpu_narrowphase(&particles, &pairs), vec![None]);
    }

    #[test]
    fn overlapping_spheres_build_the_expected_manifold() {
        // Centres 1 apart along +x, radii sum to 2: overlap depth 1.
        let particles = [particle(0.0, 0.0, 0.0, 1.0), particle(1.0, 0.0, 0.0, 1.0)];
        let pairs = [CandidatePair::new(0, 1)];
        let contacts = cpu_narrowphase(&particles, &pairs);
        let c = contacts[0].expect("overlapping spheres must contact");
        assert_eq!(c.a, 0);
        assert_eq!(c.b, 1);
        // Normal points from a to b, i.e. +x.
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        // Depth = (ra + rb) - dist = 2 - 1 = 1.
        assert!((c.depth - 1.0).abs() < 1.0e-6);
        // Point = pa + normal * (ra - depth / 2) = (1 - 0.5, 0, 0) = (0.5, 0, 0).
        assert!((c.point - Vec3::new(0.5, 0.0, 0.0)).length() < 1.0e-6);
    }

    #[test]
    fn normal_follows_pair_orientation_off_axis() {
        // A diagonal overlap so the normal is a genuine unit direction.
        let particles = [particle(0.0, 0.0, 0.0, 1.0), particle(1.0, 1.0, 0.0, 1.0)];
        let pairs = [CandidatePair::new(0, 1)];
        let c = cpu_narrowphase(&particles, &pairs)[0].expect("overlap");
        let dist = (2.0f32).sqrt();
        let expect_normal = Vec3::new(1.0, 1.0, 0.0) / dist;
        assert!((c.normal - expect_normal).length() < 1.0e-6);
        assert!((c.normal.length() - 1.0).abs() < 1.0e-6);
        assert!((c.depth - (2.0 - dist)).abs() < 1.0e-6);
    }

    #[test]
    fn coincident_centres_fall_back_to_a_stable_axis() {
        // Same centre: direction is undefined, so the test uses +x and full
        // depth rather than producing a NaN normal.
        let particles = [particle(1.0, 2.0, 3.0, 0.5), particle(1.0, 2.0, 3.0, 0.75)];
        let pairs = [CandidatePair::new(0, 1)];
        let c = cpu_narrowphase(&particles, &pairs)[0].expect("coincident overlap");
        assert_eq!(c.normal, Vec3::X);
        assert!((c.depth - 1.25).abs() < 1.0e-6);
        assert!(c.normal.is_finite());
    }

    #[test]
    fn one_slot_per_pair_in_order() {
        // Mix of hits and misses; every input pair keeps its slot and order.
        let particles = [
            particle(0.0, 0.0, 0.0, 1.0),
            particle(1.0, 0.0, 0.0, 1.0),
            particle(10.0, 0.0, 0.0, 1.0),
        ];
        let pairs = [CandidatePair::new(0, 1), CandidatePair::new(0, 2)];
        let contacts = cpu_narrowphase(&particles, &pairs);
        assert_eq!(contacts.len(), 2);
        assert!(contacts[0].is_some(), "0-1 overlap");
        assert!(contacts[1].is_none(), "0-2 far apart");
    }

    #[test]
    fn empty_pairs_yield_no_contacts() {
        let particles = [particle(0.0, 0.0, 0.0, 1.0)];
        assert!(cpu_narrowphase(&particles, &[]).is_empty());
    }
}
