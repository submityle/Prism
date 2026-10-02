//! Host-side deterministic preparation for the cloth point (vertex-vertex)
//! self-collision kernel.
//!
//! The GPU kernel does the floating-point separating-push arithmetic, but the
//! candidate set it resolves — and the per-particle reduction order it folds
//! those resolutions in — are built here on the host so they are bit-identical
//! to the [`prism_physics_core`] golden
//! ([`resolve_self_collision_jacobi`](prism_physics_core::resolve_self_collision_jacobi)
//! and its friction variant): the same integer cell assignment ([`cell_of`]),
//! the same ascending [`BTreeMap`] cell order with ascending in-bucket member
//! order, the same 27-cell neighborhood enumeration deduplicated into a single
//! ascending [`BTreeSet`] of unordered pairs, and the same ascending-by-pair
//! per-particle incidence list.
//!
//! Performing the broad phase on the host keeps the only GPU floating-point
//! work the separating push itself, which the parity suite checks within a
//! tight tolerance; the integer bucketing and pair enumeration never diverge.
//! Pairs farther apart than `thickness` are kept (the golden enumerates them
//! too and its `half_correction` zeroes them), so the candidate set matches
//! the golden's traversal exactly.
//!
//! # Provenance
//!
//! Uniform spatial hashing is the classical Teschner et al. 2003 scheme and the
//! inverse-mass-weighted separation is standard position-based dynamics. No
//! Unreal Engine source or derived code.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use glam::Vec3;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// A fully flattened, upload-ready snapshot of one cloth point self-collision
/// pass.
///
/// Every field is a plain-old-data array laid out for direct upload to a
/// storage buffer; the four-wide packing (`[_; 4]`) matches the `vec4` strides
/// the kernel binds. See [`build`] for the construction contract.
pub struct ClothSelfCollisionPointPrep {
    /// Number of addressable particles (phase-2 thread count).
    pub particle_count: u32,
    /// Number of candidate pairs (phase-1 thread count).
    pub pair_count: u32,
    /// Sanitized fabric thickness threaded to the kernel.
    pub thickness: f32,
    /// Sanitized Coulomb friction in `0..=1`; `0` selects the plain push.
    pub friction: f32,
    /// Frozen frame-end positions, padded to `vec4` (`w` carried through).
    pub positions: Vec<[f32; 4]>,
    /// Frozen frame-start positions, padded to `vec4` (`w` unused). Padded with
    /// the matching end position when the caller's slide column is short, so
    /// the kernel reads a zero slide there exactly as the golden does.
    pub prev_positions: Vec<[f32; 4]>,
    /// Index-aligned inverse masses (`0` marks a pinned particle).
    pub inverse_masses: Vec<f32>,
    /// Candidate pairs in ascending [`BTreeSet`] order; the pair index is the
    /// phase-1 slot and the per-particle accumulation order.
    pub pairs: Vec<[u32; 2]>,
    /// `CSR` offsets into [`vert_entries`](Self::vert_entries), length
    /// `particle_count + 1`.
    pub vert_offsets: Vec<u32>,
    /// Incident `(pair index, side)` records per particle, ascending by pair
    /// index so each particle folds in the golden's reduction order. `side`
    /// is `0` for the pair's first partner and `1` for the second.
    pub vert_entries: Vec<[u32; 2]>,
}

/// The integer cell of `pos`, matching `prism_physics_core`'s `cell_of`.
///
/// Floor division by `cell_size`, performed on the host so the GPU never has to
/// reproduce the boundary rounding of a float multiply-and-floor. Callers pass
/// a strictly positive `cell_size` (the sanitized value).
#[must_use]
fn cell_of(pos: Vec3, cell_size: Real) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    (
        (pos.x * inv).floor() as i32,
        (pos.y * inv).floor() as i32,
        (pos.z * inv).floor() as i32,
    )
}

/// Clamps `friction` to `0..=1`, mapping non-finite to `0`.
///
/// A crate-local copy of `prism_physics_core`'s crate-private
/// `sanitize_friction`, so the host builds the exact friction coefficient the
/// golden's `resolve_self_collision_with_friction_jacobi` would use.
#[must_use]
fn sanitize_friction(friction: Real) -> Real {
    if friction.is_finite() {
        friction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Builds the upload-ready [`ClothSelfCollisionPointPrep`] for one Jacobi point
/// self-collision pass, or returns [`None`] when the pass is a no-op.
///
/// The build mirrors `prism_physics_core`'s `resolve_self_collision_jacobi`
/// broad phase exactly: the pass is a no-op (returning [`None`], so the GPU
/// path leaves its inputs untouched) when `cell_size`/`thickness` is
/// non-positive, when `inverse_masses.len()` differs from `positions.len()`,
/// when fewer than two particles are addressable, or when no candidate pair
/// shares a 27-cell neighborhood. `friction` is sanitized to `0..=1`
/// (non-finite maps to `0`); `0` keeps the stored friction at `0` so the kernel
/// takes the plain normal-push branch. `prev_positions` shorter than
/// `positions` is padded with the matching end position, matching the golden's
/// `prev.get(i).unwrap_or(pos)` degradation.
#[must_use]
pub fn build(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    inverse_masses: &[Real],
    cell_size: Real,
    thickness: Real,
    friction: Real,
) -> Option<ClothSelfCollisionPointPrep> {
    if cell_size <= 0.0
        || thickness <= 0.0
        || positions.len() < 2
        || inverse_masses.len() != positions.len()
    {
        return None;
    }
    let count = positions.len();

    // Broad phase: bucket every particle into its single integer cell, keyed by
    // an ascending BTreeMap so cell order and in-bucket member order are fixed,
    // matching the golden's `build_grid`.
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, &pos) in positions.iter().enumerate() {
        let cell = cell_of(pos, cell_size);
        grid.entry(cell).or_default().push(index as u32);
    }

    // Every unordered pair whose cells lie within a 27-cell neighborhood,
    // deduplicated in a BTreeSet so the pair order is a single stable ascending
    // sequence. This is exactly the set the golden visits (for each `a`, every
    // `b != a` in the 27 cells around `a`), and the neighborhood is symmetric,
    // so storing each unordered pair once lets phase 1 compute both halves.
    let mut pair_set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for (&cell, bucket) in &grid {
        for &a in bucket {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b == a {
                                continue;
                            }
                            let lo = a.min(b);
                            let hi = a.max(b);
                            pair_set.insert((lo, hi));
                        }
                    }
                }
            }
        }
    }
    if pair_set.is_empty() {
        return None;
    }

    let pairs: Vec<[u32; 2]> = pair_set.iter().map(|&(a, b)| [a, b]).collect();

    // Phase-2 incidence: scanning pairs in ascending index order appends each
    // particle's contributions in ascending pair-index order, which is exactly
    // the golden's per-particle accumulation order.
    let mut per_vertex: Vec<Vec<[u32; 2]>> = Vec::with_capacity(count);
    per_vertex.resize_with(count, Vec::new);
    for (pair_index, pair) in pairs.iter().enumerate() {
        let pi = u32::try_from(pair_index).unwrap_or(u32::MAX);
        per_vertex[pair[0] as usize].push([pi, 0]);
        per_vertex[pair[1] as usize].push([pi, 1]);
    }
    let mut vert_offsets: Vec<u32> = Vec::with_capacity(count + 1);
    let mut vert_entries: Vec<[u32; 2]> = Vec::new();
    vert_offsets.push(0);
    for entries in &per_vertex {
        vert_entries.extend_from_slice(entries);
        vert_offsets.push(u32::try_from(vert_entries.len()).unwrap_or(u32::MAX));
    }

    let positions_packed: Vec<[f32; 4]> =
        positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let prev_packed: Vec<[f32; 4]> = (0..count)
        .map(|i| {
            let p = prev_positions.get(i).copied().unwrap_or(positions[i]);
            [p.x, p.y, p.z, 0.0]
        })
        .collect();

    Some(ClothSelfCollisionPointPrep {
        particle_count: u32::try_from(count).unwrap_or(u32::MAX),
        pair_count: u32::try_from(pairs.len()).unwrap_or(u32::MAX),
        thickness,
        friction: sanitize_friction(friction),
        positions: positions_packed,
        prev_positions: prev_packed,
        inverse_masses: inverse_masses.to_vec(),
        pairs,
        vert_offsets,
        vert_entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_inputs_are_a_no_op() {
        let positions = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        assert!(build(&positions, &prev, &im, 0.0, 1.0, 0.0).is_none());
        assert!(build(&positions, &prev, &im, 1.0, 0.0, 0.0).is_none());
        assert!(build(&positions[..1], &prev[..1], &im[..1], 1.0, 1.0, 0.0).is_none());
        assert!(build(&positions, &prev, &im[..1], 1.0, 1.0, 0.0).is_none());
    }

    #[test]
    fn separated_particles_produce_no_pairs() {
        // Two particles many cells apart share no 27-cell neighborhood.
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        assert!(build(&positions, &prev, &im, 1.0, 0.5, 0.0).is_none());
    }

    #[test]
    fn adjacent_pair_is_captured_with_ascending_csr() {
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        let prep = build(&positions, &prev, &im, 1.0, 0.5, 0.0).expect("non-degenerate");
        assert_eq!(prep.particle_count, 2);
        assert_eq!(prep.pair_count, 1);
        assert_eq!(prep.pairs, vec![[0, 1]]);
        assert_eq!(prep.vert_offsets, vec![0, 1, 2]);
        assert_eq!(prep.vert_entries, vec![[0, 0], [0, 1]]);
    }

    #[test]
    fn neighbouring_cells_still_pair() {
        // Two particles in adjacent cells (chebyshev distance 1) must pair even
        // though they do not co-occupy a cell.
        let positions = [Vec3::new(0.4, 0.0, 0.0), Vec3::new(0.6, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        let prep = build(&positions, &prev, &im, 0.5, 0.5, 0.0).expect("non-degenerate");
        assert_eq!(prep.pairs, vec![[0, 1]]);
    }

    #[test]
    fn cluster_lists_each_particle_in_ascending_pair_order() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::new(0.0, 0.05, 0.0),
        ];
        let prev = positions;
        let im = [1.0, 1.0, 1.0];
        let prep = build(&positions, &prev, &im, 1.0, 0.5, 0.0).expect("non-degenerate");
        assert_eq!(prep.pairs, vec![[0, 1], [0, 2], [1, 2]]);
        let p0 = &prep.vert_entries[prep.vert_offsets[0] as usize..prep.vert_offsets[1] as usize];
        let p1 = &prep.vert_entries[prep.vert_offsets[1] as usize..prep.vert_offsets[2] as usize];
        let p2 = &prep.vert_entries[prep.vert_offsets[2] as usize..prep.vert_offsets[3] as usize];
        assert_eq!(p0, &[[0, 0], [1, 0]]);
        assert_eq!(p1, &[[0, 1], [2, 0]]);
        assert_eq!(p2, &[[1, 1], [2, 1]]);
    }

    #[test]
    fn non_finite_friction_sanitises_to_zero() {
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)];
        let prev = positions;
        let im = [1.0, 1.0];
        let prep = build(&positions, &prev, &im, 1.0, 0.5, Real::NAN).expect("non-degenerate");
        assert_eq!(prep.friction, 0.0);
        let prep2 = build(&positions, &prev, &im, 1.0, 0.5, 5.0).expect("non-degenerate");
        assert_eq!(prep2.friction, 1.0);
    }
}
