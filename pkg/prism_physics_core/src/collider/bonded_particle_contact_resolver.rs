//! Broad-phase accumulation of soft-sphere discrete-element contacts.
//!
//! The pairwise contact law in
//! [`bonded_particle_contact`](super::bonded_particle_contact) answers the
//! question "what repulsive force acts between two overlapping spheres?". This
//! module answers the question one level up: "given a whole packing of moving
//! spheres, which pairs are in contact and what is the net force on each?". It
//! is the repulsive counterpart of the
//! [`bonded_particle_network_builder`](super::bonded_particle_network_builder):
//! the network builder installs *cohesive* bonds once over a rest packing,
//! whereas this resolver recomputes *repulsive* contacts every step as the
//! fragments move and collide.
//!
//! # Acceleration
//!
//! A brute-force scan is `O(n²)`. Instead the resolver bins particle centres
//! into a uniform spatial-hash grid whose cell size is the largest possible
//! contact distance `2·R_max`. Two spheres can only overlap when their centres
//! are closer than `R_a + R_b ≤ 2·R_max`, so an overlapping pair always falls
//! in the same or an adjacent cell and each particle need only test the 27
//! cells of its `3×3×3` neighbourhood. Each unordered pair is evaluated once,
//! with the lower index first, and the contacts are reported in a deterministic
//! order independent of grid-bucket iteration order.
//!
//! # Forces
//!
//! For each contacting pair the resolver calls
//! [`contact_between`](super::bonded_particle_contact::contact_between), which
//! returns the force on `b`; the equal and opposite force acts on `a`. The
//! per-particle `forces` therefore sum to zero (up to floating-point error),
//! so a resolved packing conserves linear momentum exactly.

use crate::collider::bonded_particle_contact::{contact_between, ContactModel};
use glam::Vec3;
use std::collections::HashMap;

/// A single resolved contact between two particles of a packing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Lower particle index of the contacting pair.
    pub a: u32,
    /// Higher particle index of the contacting pair.
    pub b: u32,
    /// Overlap `δ > 0` resolved by this contact.
    pub overlap: f32,
    /// Normal force magnitude `F_n ≥ 0`.
    pub normal_magnitude: f32,
    /// Tangential (friction) force magnitude actually applied.
    pub tangential_magnitude: f32,
    /// Whether the tangential force reached the Coulomb limit (sliding).
    pub sliding: bool,
}

/// Result of resolving every contact in a particle packing.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactResolution {
    /// Net contact force on each particle, indexed in parallel with the input
    /// `positions`; always the same length as the packing.
    pub forces: Vec<Vec3>,
    /// The resolved contacts, in deterministic `(a, b)` order.
    pub contacts: Vec<Contact>,
}

impl ContactResolution {
    /// Number of resolved contacts.
    #[must_use]
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Largest overlap across all contacts, or `0.0` when there are none. A
    /// useful stability diagnostic: a large maximum overlap means the step is
    /// letting spheres interpenetrate deeply before the penalty responds.
    #[must_use]
    pub fn max_overlap(&self) -> f32 {
        self.contacts
            .iter()
            .map(|c| c.overlap)
            .fold(0.0_f32, f32::max)
    }

    /// Largest normal force magnitude across all contacts, or `0.0` when there
    /// are none.
    #[must_use]
    pub fn max_normal_force(&self) -> f32 {
        self.contacts
            .iter()
            .map(|c| c.normal_magnitude)
            .fold(0.0_f32, f32::max)
    }

    /// Sum of all per-particle forces. The contact forces are internal and
    /// equal-and-opposite, so this is zero up to floating-point error; it is
    /// exposed as a conservation diagnostic.
    #[must_use]
    pub fn total_force(&self) -> Vec3 {
        self.forces.iter().copied().sum()
    }

    /// Number of contacts whose tangential force reached the Coulomb limit.
    #[must_use]
    pub fn sliding_count(&self) -> usize {
        self.contacts.iter().filter(|c| c.sliding).count()
    }
}

/// Validates that the packing is well formed: equal-length `positions`,
/// `radii`, and `velocities`; every coordinate finite; every radius finite and
/// strictly positive; every velocity component finite.
fn is_valid(positions: &[Vec3], radii: &[f32], velocities: &[Vec3]) -> bool {
    if positions.len() != radii.len() || positions.len() != velocities.len() {
        return false;
    }
    let finite_vec = |v: &Vec3| v.x.is_finite() && v.y.is_finite() && v.z.is_finite();
    positions.iter().all(finite_vec)
        && velocities.iter().all(finite_vec)
        && radii.iter().all(|&r| r.is_finite() && r > 0.0)
}

/// Integer grid cell coordinate of a point relative to the packing's minimum
/// corner, at the given cell size.
fn cell_of(point: Vec3, origin: Vec3, cell_size: f32) -> [i32; 3] {
    let rel = (point - origin) / cell_size;
    [
        rel.x.floor() as i32,
        rel.y.floor() as i32,
        rel.z.floor() as i32,
    ]
}

/// Resolves every soft-sphere contact in the packing described by `positions`,
/// `radii`, and `velocities` under the given contact `model`.
///
/// Returns `None` when the three slices disagree in length, when any coordinate
/// or velocity component is non-finite, or when any radius is not finite and
/// strictly positive. On success the returned [`ContactResolution`] carries the
/// net force on each particle (parallel to `positions`) and the list of
/// resolved contacts in deterministic order.
#[must_use]
pub fn resolve_contacts(
    positions: &[Vec3],
    radii: &[f32],
    velocities: &[Vec3],
    model: &ContactModel,
) -> Option<ContactResolution> {
    if !is_valid(positions, radii, velocities) {
        return None;
    }

    let n = positions.len();
    let mut forces = vec![Vec3::ZERO; n];
    let mut contacts = Vec::new();
    if n == 0 {
        return Some(ContactResolution { forces, contacts });
    }

    // Cell size = largest possible contact distance, so an overlapping pair is
    // always within the 3×3×3 neighbourhood of either particle.
    let max_radius = radii.iter().copied().fold(0.0_f32, f32::max);
    let cell_size = 2.0 * max_radius;
    // `radii > 0` guarantees `cell_size > 0`; guard anyway against overflow of
    // the floor cast for pathological inputs.
    if !(cell_size.is_finite() && cell_size > 0.0) {
        return None;
    }

    let origin = positions
        .iter()
        .copied()
        .reduce(Vec3::min)
        .unwrap_or(Vec3::ZERO);

    // Bin particle indices into their grid cells.
    let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for (i, &p) in positions.iter().enumerate() {
        grid.entry(cell_of(p, origin, cell_size))
            .or_default()
            .push(i as u32);
    }

    for (i, &pi) in positions.iter().enumerate() {
        let base = cell_of(pi, origin, cell_size);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let Some(bucket) = grid.get(&cell) else {
                        continue;
                    };
                    for &jraw in bucket {
                        let j = jraw as usize;
                        // Evaluate each unordered pair once, lower index first.
                        if j <= i {
                            continue;
                        }
                        let Some(contact) = contact_between(
                            model,
                            pi,
                            positions[j],
                            radii[i],
                            radii[j],
                            velocities[i],
                            velocities[j],
                        ) else {
                            continue;
                        };
                        forces[j] += contact.force_on_b;
                        forces[i] -= contact.force_on_b;
                        contacts.push(Contact {
                            a: i as u32,
                            b: j as u32,
                            overlap: contact.overlap,
                            normal_magnitude: contact.normal_magnitude,
                            tangential_magnitude: contact.tangential_magnitude,
                            sliding: contact.sliding,
                        });
                    }
                }
            }
        }
    }

    // Deterministic order independent of grid-bucket iteration order.
    contacts.sort_unstable_by_key(|c| (c.a, c.b));
    Some(ContactResolution { forces, contacts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ContactModel {
        ContactModel::new(1.0e6, 10.0, 10.0, 0.5).unwrap()
    }

    /// Reference `O(n²)` resolver used to cross-check the grid build.
    fn brute_force(
        positions: &[Vec3],
        radii: &[f32],
        velocities: &[Vec3],
        model: &ContactModel,
    ) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if contact_between(
                    model,
                    positions[i],
                    positions[j],
                    radii[i],
                    radii[j],
                    velocities[i],
                    velocities[j],
                )
                .is_some()
                {
                    pairs.push((i as u32, j as u32));
                }
            }
        }
        pairs
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let p = vec![Vec3::ZERO, Vec3::X];
        let r = vec![1.0];
        let v = vec![Vec3::ZERO, Vec3::ZERO];
        assert!(resolve_contacts(&p, &r, &v, &model()).is_none());
    }

    #[test]
    fn rejects_non_finite_and_bad_radius() {
        let p = vec![Vec3::new(f32::NAN, 0.0, 0.0), Vec3::X];
        let r = vec![1.0, 1.0];
        let v = vec![Vec3::ZERO, Vec3::ZERO];
        assert!(resolve_contacts(&p, &r, &v, &model()).is_none());

        let p = vec![Vec3::ZERO, Vec3::X];
        let r = vec![0.0, 1.0];
        assert!(resolve_contacts(&p, &r, &v, &model()).is_none());
    }

    #[test]
    fn empty_packing_resolves_to_nothing() {
        let res = resolve_contacts(&[], &[], &[], &model()).expect("valid");
        assert_eq!(res.contact_count(), 0);
        assert!(res.forces.is_empty());
        assert_eq!(res.total_force(), Vec3::ZERO);
    }

    #[test]
    fn separated_particles_have_no_contact() {
        // Centres 3 apart, radii 1 + 1 = 2 < 3 → separated.
        let p = vec![Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)];
        let r = vec![1.0, 1.0];
        let v = vec![Vec3::ZERO, Vec3::ZERO];
        let res = resolve_contacts(&p, &r, &v, &model()).expect("valid");
        assert_eq!(res.contact_count(), 0);
        assert_eq!(res.forces[0], Vec3::ZERO);
        assert_eq!(res.forces[1], Vec3::ZERO);
    }

    #[test]
    fn overlapping_pair_pushes_apart_and_conserves_momentum() {
        // Centres 1.5 apart, radii 1 + 1 = 2 → overlap 0.5.
        let p = vec![Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let r = vec![1.0, 1.0];
        let v = vec![Vec3::ZERO, Vec3::ZERO];
        let res = resolve_contacts(&p, &r, &v, &model()).expect("valid");
        assert_eq!(res.contact_count(), 1);
        let c = res.contacts[0];
        assert_eq!((c.a, c.b), (0, 1));
        assert!((c.overlap - 0.5).abs() < 1e-5);
        // b is pushed in +x, a in −x; momentum conserved.
        assert!(res.forces[1].x > 0.0);
        assert!(res.forces[0].x < 0.0);
        assert!(res.total_force().length() < 1e-1);
        assert!((res.max_normal_force() - c.normal_magnitude).abs() < 1e-3);
    }

    #[test]
    fn grid_matches_brute_force_on_a_dense_cloud() {
        // A jittered 4×4×4 lattice with spacing < 2R so many pairs overlap.
        let mut p = Vec::new();
        let mut r = Vec::new();
        let mut v = Vec::new();
        let spacing = 1.2_f32;
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    let jitter = Vec3::new(
                        ((ix * 7 + 1) % 5) as f32 * 0.01,
                        ((iy * 3 + 2) % 5) as f32 * 0.01,
                        ((iz * 5 + 4) % 5) as f32 * 0.01,
                    );
                    p.push(Vec3::new(ix as f32, iy as f32, iz as f32) * spacing + jitter);
                    r.push(0.7);
                    v.push(Vec3::new((ix - iz) as f32, (iy - ix) as f32, 0.5));
                }
            }
        }
        let res = resolve_contacts(&p, &r, &v, &model()).expect("valid");
        let grid_pairs: Vec<(u32, u32)> = res.contacts.iter().map(|c| (c.a, c.b)).collect();
        let mut expected = brute_force(&p, &r, &v, &model());
        expected.sort_unstable();
        assert_eq!(grid_pairs, expected);
        assert!(!grid_pairs.is_empty(), "cloud should produce contacts");
        // Internal forces must still cancel across the whole packing.
        assert!(res.total_force().length() < 1.0);
    }

    #[test]
    fn contacts_are_sorted_deterministically() {
        let p = vec![
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::ZERO,
        ];
        let r = vec![1.0, 1.0, 1.0];
        let v = vec![Vec3::ZERO; 3];
        let res = resolve_contacts(&p, &r, &v, &model()).expect("valid");
        let pairs: Vec<(u32, u32)> = res.contacts.iter().map(|c| (c.a, c.b)).collect();
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        assert_eq!(pairs, sorted);
    }
}
