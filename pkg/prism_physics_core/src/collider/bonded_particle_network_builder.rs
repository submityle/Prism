//! Installation of a parallel-bond network over a packing of discrete
//! particles.
//!
//! This is the discrete-element counterpart of the surface
//! [`cohesive_interface_builder`](super::cohesive_interface_builder): where the
//! cohesive builder splits a tetrahedral mesh to insert crack interfaces, this
//! builder *installs* parallel bonds between near-contacting particles so the
//! packing becomes a bonded solid that the
//! [`bonded_particle`](super::bonded_particle) law and the
//! [`bonded_particle_assembly`](super::bonded_particle_assembly) layer can then
//! drive and break.
//!
//! # Bond installation rule
//!
//! Following Potyondy & Cundall, a parallel bond is installed between particles
//! `i` and `j` when their *surface gap* does not exceed an installation
//! tolerance `g`:
//!
//! ```text
//!   gap_ij = ‖x_i − x_j‖ − (R_i + R_j)  ≤  g
//! ```
//!
//! so touching or slightly overlapping particles (`gap_ij ≤ 0`) always bond,
//! and a positive `g` additionally bonds particles separated by a thin gap —
//! the usual way to seed a bonded assembly from a non-contacting cloud.
//! Coincident centres (`‖x_i − x_j‖ ≈ 0`) are never bonded because the bond
//! axis would be undefined.
//!
//! # Acceleration
//!
//! A brute-force scan is `O(n²)`. Instead this builder bins particle centres
//! into a uniform spatial-hash grid whose cell size equals the largest possible
//! bond distance `2·R_max + g`. Any two particles that can bond therefore fall
//! in the same or an adjacent cell, so each particle only tests the 27 cells of
//! its `3×3×3` neighbourhood. Each unordered pair is emitted once, with the
//! lower index first, and the bonds are produced in a deterministic order.

use crate::collider::bonded_particle_assembly::ParticleBond;
use glam::Vec3;
use std::collections::HashMap;

/// An installed parallel-bond network over a particle packing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BondNetwork {
    /// Installed bonds, each with `a < b`, in deterministic order.
    pub bonds: Vec<ParticleBond>,
}

impl BondNetwork {
    /// Number of installed bonds.
    #[must_use]
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Per-particle coordination number (how many bonds touch each particle).
    ///
    /// The returned vector has one entry per particle; `particle_count` must be
    /// at least as large as the highest particle index referenced by a bond or
    /// this returns `None`.
    #[must_use]
    pub fn coordination_numbers(&self, particle_count: usize) -> Option<Vec<u32>> {
        let mut counts = vec![0u32; particle_count];
        for bond in &self.bonds {
            let a = bond.a as usize;
            let b = bond.b as usize;
            if a >= particle_count || b >= particle_count {
                return None;
            }
            counts[a] += 1;
            counts[b] += 1;
        }
        Some(counts)
    }

    /// Mean coordination number across `particle_count` particles, i.e.
    /// `2·bond_count / particle_count`, or `0.0` when there are no particles.
    #[must_use]
    pub fn mean_coordination(&self, particle_count: usize) -> f32 {
        if particle_count == 0 {
            return 0.0;
        }
        2.0 * self.bonds.len() as f32 / particle_count as f32
    }
}

/// Validates that the packing is well formed: equal-length `positions` and
/// `radii`, every coordinate finite, every radius finite and strictly positive,
/// and a finite non-negative installation `gap`.
fn is_valid(positions: &[Vec3], radii: &[f32], gap: f32) -> bool {
    if positions.len() != radii.len() {
        return false;
    }
    if !gap.is_finite() || gap < 0.0 {
        return false;
    }
    positions
        .iter()
        .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
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

/// Installs a parallel-bond network over the particle packing described by
/// `positions` and `radii`, bonding every pair whose surface gap is at most
/// `gap`.
///
/// Returns `None` when `positions` and `radii` disagree in length, when any
/// coordinate or radius is non-finite, when any radius is not strictly
/// positive, or when `gap` is negative or non-finite. The resulting bonds are
/// deterministic and free of duplicates, each stored with the lower particle
/// index first.
#[must_use]
pub fn build_bond_network(positions: &[Vec3], radii: &[f32], gap: f32) -> Option<BondNetwork> {
    if !is_valid(positions, radii, gap) {
        return None;
    }
    if positions.is_empty() {
        return Some(BondNetwork { bonds: Vec::new() });
    }

    // Cell size = largest possible bond distance, so a bondable pair is always
    // within the 3×3×3 neighbourhood of either particle.
    let max_radius = radii.iter().copied().fold(0.0_f32, f32::max);
    let cell_size = 2.0 * max_radius + gap;
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

    let mut bonds = Vec::new();
    for (i, &pi) in positions.iter().enumerate() {
        let ri = radii[i];
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
                        // Emit each unordered pair once, lower index first.
                        if j <= i {
                            continue;
                        }
                        let delta = pi - positions[j];
                        let dist = delta.length();
                        if dist <= f32::EPSILON {
                            continue;
                        }
                        let surface_gap = dist - (ri + radii[j]);
                        if surface_gap <= gap {
                            bonds.push(ParticleBond::new(i as u32, j as u32));
                        }
                    }
                }
            }
        }
    }

    // Deterministic order independent of grid-bucket iteration order.
    bonds.sort_unstable_by_key(|b| (b.a, b.b));
    Some(BondNetwork { bonds })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference `O(n²)` installation used to cross-check the grid build.
    fn brute_force(positions: &[Vec3], radii: &[f32], gap: f32) -> Vec<ParticleBond> {
        let mut bonds = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let dist = (positions[i] - positions[j]).length();
                if dist <= f32::EPSILON {
                    continue;
                }
                if dist - (radii[i] + radii[j]) <= gap {
                    bonds.push(ParticleBond::new(i as u32, j as u32));
                }
            }
        }
        bonds.sort_unstable_by_key(|b| (b.a, b.b));
        bonds
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let pos = vec![Vec3::ZERO, Vec3::X];
        let radii = vec![1.0];
        assert!(build_bond_network(&pos, &radii, 0.0).is_none());
    }

    #[test]
    fn rejects_bad_radius() {
        let pos = vec![Vec3::ZERO, Vec3::X];
        assert!(build_bond_network(&pos, &[1.0, 0.0], 0.0).is_none());
        assert!(build_bond_network(&pos, &[1.0, -1.0], 0.0).is_none());
        assert!(build_bond_network(&pos, &[1.0, f32::NAN], 0.0).is_none());
    }

    #[test]
    fn rejects_bad_gap() {
        let pos = vec![Vec3::ZERO, Vec3::X];
        assert!(build_bond_network(&pos, &[1.0, 1.0], -0.1).is_none());
        assert!(build_bond_network(&pos, &[1.0, 1.0], f32::INFINITY).is_none());
    }

    #[test]
    fn rejects_non_finite_position() {
        let pos = vec![Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)];
        assert!(build_bond_network(&pos, &[1.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn empty_packing_has_no_bonds() {
        let net = build_bond_network(&[], &[], 0.0).expect("valid");
        assert_eq!(net.bond_count(), 0);
        assert_eq!(net.mean_coordination(0), 0.0);
    }

    #[test]
    fn touching_spheres_bond() {
        // Centres 2 apart, radii 1 each → surfaces just touch (gap 0).
        let pos = vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let net = build_bond_network(&pos, &[1.0, 1.0], 0.0).expect("valid");
        assert_eq!(net.bonds, vec![ParticleBond::new(0, 1)]);
    }

    #[test]
    fn separated_spheres_do_not_bond_without_gap() {
        // Centres 2.5 apart, radii 1 → surface gap 0.5 > 0.
        let pos = vec![Vec3::ZERO, Vec3::new(2.5, 0.0, 0.0)];
        let net = build_bond_network(&pos, &[1.0, 1.0], 0.0).expect("valid");
        assert_eq!(net.bond_count(), 0);
    }

    #[test]
    fn gap_tolerance_bonds_near_pairs() {
        let pos = vec![Vec3::ZERO, Vec3::new(2.5, 0.0, 0.0)];
        // gap 0.5 just includes the 0.5 surface separation.
        let net = build_bond_network(&pos, &[1.0, 1.0], 0.5).expect("valid");
        assert_eq!(net.bonds, vec![ParticleBond::new(0, 1)]);
    }

    #[test]
    fn coincident_centres_never_bond() {
        let pos = vec![Vec3::ZERO, Vec3::ZERO];
        let net = build_bond_network(&pos, &[1.0, 1.0], 1.0).expect("valid");
        assert_eq!(net.bond_count(), 0);
    }

    #[test]
    fn chain_bonds_consecutive_particles() {
        // Four unit spheres spaced 2 apart along x: only neighbours touch.
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
        ];
        let radii = vec![1.0; 4];
        let net = build_bond_network(&pos, &radii, 0.0).expect("valid");
        assert_eq!(
            net.bonds,
            vec![
                ParticleBond::new(0, 1),
                ParticleBond::new(1, 2),
                ParticleBond::new(2, 3),
            ]
        );
        let coord = net.coordination_numbers(4).expect("in range");
        assert_eq!(coord, vec![1, 2, 2, 1]);
        assert!((net.mean_coordination(4) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn coordination_rejects_too_small_count() {
        let pos = vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let net = build_bond_network(&pos, &[1.0, 1.0], 0.0).expect("valid");
        assert!(net.coordination_numbers(1).is_none());
    }

    #[test]
    fn bonds_are_sorted_and_lower_index_first() {
        // A compact cluster where several pairs bond.
        let pos = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let radii = vec![0.75; 4];
        let net = build_bond_network(&pos, &radii, 0.0).expect("valid");
        for bond in &net.bonds {
            assert!(bond.a < bond.b, "lower index first");
        }
        let mut sorted = net.bonds.clone();
        sorted.sort_unstable_by_key(|b| (b.a, b.b));
        assert_eq!(net.bonds, sorted, "deterministic sorted order");
    }

    #[test]
    fn grid_matches_brute_force_on_a_lattice() {
        // A 5×5×5 grid of unit-spacing spheres with a small radius+gap so that
        // only axis neighbours bond; cross-check the accelerated build against
        // the O(n²) reference.
        let mut pos = Vec::new();
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    pos.push(Vec3::new(x as f32, y as f32, z as f32));
                }
            }
        }
        let radii = vec![0.4; pos.len()];
        let gap = 0.25; // 1.0 spacing - 0.8 (two radii) = 0.2 ≤ 0.25.
        let net = build_bond_network(&pos, &radii, gap).expect("valid");
        let reference = brute_force(&pos, &radii, gap);
        assert_eq!(net.bonds, reference);
        assert!(!net.bonds.is_empty());
    }

    #[test]
    fn grid_matches_brute_force_on_a_jittered_cloud() {
        // A deterministic pseudo-random cloud with heterogeneous radii stresses
        // the neighbourhood logic and the max-radius cell sizing.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32
        };
        let mut pos = Vec::new();
        let mut radii = Vec::new();
        for _ in 0..200 {
            pos.push(Vec3::new(next() * 6.0, next() * 6.0, next() * 6.0));
            radii.push(0.3 + next() * 0.5);
        }
        let gap = 0.1;
        let net = build_bond_network(&pos, &radii, gap).expect("valid");
        let reference = brute_force(&pos, &radii, gap);
        assert_eq!(net.bonds, reference);
    }
}
