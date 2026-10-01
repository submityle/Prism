//! Host-side deterministic preparation for the cloth continuous self-collision
//! (self-CCD) kernel.
//!
//! The GPU kernel does the floating-point swept-pair time-of-impact arithmetic,
//! but the candidate set it resolves — and the per-particle reduction order it
//! folds those resolutions in — are built here on the host so they are
//! bit-identical to the [`prism_physics_core`] golden
//! ([`resolve_self_ccd_jacobi`](prism_physics_core::resolve_self_ccd_jacobi)):
//! the same integer cell assignment ([`cell_of`]), the same ascending
//! [`BTreeMap`] cell order with ascending in-bucket member order, the same
//! [`BTreeSet`] candidate-pair order, and the same ascending-by-`pair_index`
//! per-particle incidence list.
//!
//! Performing the broad phase on the host keeps the only GPU floating-point
//! work the swept-pair resolution itself, which the parity suite checks within
//! a tight tolerance; the integer bucketing and pair enumeration never diverge.
//!
//! # Provenance
//!
//! Uniform spatial hashing is the classical Teschner et al. 2003 scheme and the
//! swept-pair TOI resolution is standard analytic continuous-collision
//! geometry. No Unreal Engine source or derived code.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::SelfCcdParams;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// A fully flattened, upload-ready snapshot of one cloth self-CCD pass.
///
/// Every field is a plain-old-data array laid out for direct upload to a
/// storage buffer; the four-wide packing (`[_; 4]`) matches the `vec4` strides
/// the kernel binds. See [`build`] for the construction contract.
pub struct ClothSelfCcdPrep {
    /// Number of addressable particles (the swept `count`).
    pub particle_count: u32,
    /// Number of candidate pairs (phase-1 thread count).
    pub pair_count: u32,
    /// Sanitized fabric thickness threaded to the kernel.
    pub thickness: f32,
    /// Sanitized normal restitution in `0..=1`.
    pub restitution: f32,
    /// Frozen frame-end positions, padded to `vec4` (`w` carried through).
    pub positions: Vec<[f32; 4]>,
    /// Frozen frame-start positions, padded to `vec4` (`w` unused).
    pub prev_positions: Vec<[f32; 4]>,
    /// Frozen velocities, padded to `vec4` (`w` carried through).
    pub velocities: Vec<[f32; 4]>,
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

/// Builds the upload-ready [`ClothSelfCcdPrep`] for one Jacobi self-CCD pass, or
/// returns [`None`] when the pass is a no-op.
///
/// The build mirrors `prism_physics_core`'s `resolve_self_ccd_jacobi` broad
/// phase exactly: `params` is [`SelfCcdParams::sanitized`] first, and the pass
/// is a no-op (returning [`None`], so the GPU path leaves its inputs untouched)
/// when the sweep is disabled, when `thickness` is non-positive, when fewer than
/// two particles are addressable (the shortest of `positions`,
/// `prev_positions`, and `inverse_masses`), or when no candidate pair shares a
/// swept-box cell. `velocities` shorter than the addressable count is padded
/// with zero exactly as the golden's frozen-snapshot read does.
#[must_use]
pub fn build(
    positions: &[Vec3],
    prev_positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[Real],
    params: SelfCcdParams,
) -> Option<ClothSelfCcdPrep> {
    let params = params.sanitized();
    if !params.enabled || params.thickness <= 0.0 {
        return None;
    }
    let count = positions
        .len()
        .min(prev_positions.len())
        .min(inverse_masses.len());
    if count < 2 {
        return None;
    }

    let thickness = params.thickness;
    let cell_size = params.cell_size;

    // Broad phase: bucket every swept box (grown by `thickness`) into a uniform
    // spatial hash, keyed by an ascending BTreeMap so cell order and in-bucket
    // member order are fixed, matching `collect_self_ccd_candidate_pairs`.
    let margin = Vec3::splat(thickness);
    let mut buckets: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for index in 0..count {
        let prev = prev_positions[index];
        let curr = positions[index];
        let lo = prev.min(curr) - margin;
        let hi = prev.max(curr) + margin;
        let (lx, ly, lz) = cell_of(lo, cell_size);
        let (hx, hy, hz) = cell_of(hi, cell_size);
        let mut cx = lx;
        while cx <= hx {
            let mut cy = ly;
            while cy <= hy {
                let mut cz = lz;
                while cz <= hz {
                    buckets.entry((cx, cy, cz)).or_default().push(index as u32);
                    cz += 1;
                }
                cy += 1;
            }
            cx += 1;
        }
    }

    // Every unordered pair that co-occupies a cell, deduplicated in a BTreeSet so
    // the pair order is a single stable ascending sequence (the golden consumes
    // the identical order).
    let mut pair_set: BTreeSet<(u32, u32)> = BTreeSet::new();
    for occupants in buckets.values() {
        for slot_a in 0..occupants.len() {
            for slot_b in (slot_a + 1)..occupants.len() {
                pair_set.insert((occupants[slot_a], occupants[slot_b]));
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

    let positions_packed: Vec<[f32; 4]> = positions[..count]
        .iter()
        .map(|p| [p.x, p.y, p.z, 0.0])
        .collect();
    let prev_packed: Vec<[f32; 4]> = prev_positions[..count]
        .iter()
        .map(|p| [p.x, p.y, p.z, 0.0])
        .collect();
    let velocities_packed: Vec<[f32; 4]> = (0..count)
        .map(|i| {
            let v = velocities.get(i).copied().unwrap_or(Vec3::ZERO);
            [v.x, v.y, v.z, 0.0]
        })
        .collect();

    Some(ClothSelfCcdPrep {
        particle_count: u32::try_from(count).unwrap_or(u32::MAX),
        pair_count: u32::try_from(pairs.len()).unwrap_or(u32::MAX),
        thickness,
        restitution: params.restitution,
        positions: positions_packed,
        prev_positions: prev_packed,
        velocities: velocities_packed,
        inverse_masses: inverse_masses[..count].to_vec(),
        pairs,
        vert_offsets,
        vert_entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(thickness: Real) -> SelfCcdParams {
        SelfCcdParams::new(0.2, thickness)
    }

    #[test]
    fn disabled_or_degenerate_inputs_are_no_ops() {
        let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let vel = [Vec3::ZERO, Vec3::ZERO];
        let im = [1.0, 1.0];

        let mut disabled = params(0.4);
        disabled.enabled = false;
        assert!(build(&positions, &prev, &vel, &im, disabled).is_none());

        // thickness <= 0 sanitizes to 0 -> no-op.
        assert!(build(&positions, &prev, &vel, &im, params(0.0)).is_none());
    }

    #[test]
    fn fewer_than_two_particles_is_a_no_op() {
        let positions = [Vec3::ZERO];
        let prev = [Vec3::ZERO];
        let vel = [Vec3::ZERO];
        let im = [1.0];
        assert!(build(&positions, &prev, &vel, &im, params(0.4)).is_none());
    }

    #[test]
    fn separated_particles_produce_no_pairs() {
        // Two particles far apart whose swept boxes never share a cell.
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)];
        let prev = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0)];
        let vel = [Vec3::ZERO, Vec3::ZERO];
        let im = [1.0, 1.0];
        assert!(build(&positions, &prev, &vel, &im, params(0.1)).is_none());
    }

    #[test]
    fn crossing_pair_is_captured_with_ascending_csr() {
        let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)];
        let prev = [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let vel = [Vec3::ZERO, Vec3::ZERO];
        let im = [1.0, 1.0];
        let prep = build(&positions, &prev, &vel, &im, params(0.4)).expect("non-degenerate");

        assert_eq!(prep.particle_count, 2);
        assert_eq!(prep.pair_count, 1);
        assert_eq!(prep.pairs, vec![[0, 1]]);
        // CSR frames every particle; each endpoint has exactly one incidence.
        assert_eq!(prep.vert_offsets, vec![0, 1, 2]);
        assert_eq!(prep.vert_entries, vec![[0, 0], [0, 1]]);
    }

    #[test]
    fn shared_cell_cluster_lists_each_particle_in_ascending_pair_order() {
        // Three mutually-near particles share a cell, forming pairs
        // (0,1), (0,2), (1,2) in ascending BTreeSet order.
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.05, 0.0, 0.0),
            Vec3::new(0.0, 0.05, 0.0),
        ];
        let prev = positions;
        let vel = [Vec3::ZERO; 3];
        let im = [1.0, 1.0, 1.0];
        let prep = build(&positions, &prev, &vel, &im, params(0.4)).expect("non-degenerate");

        assert_eq!(prep.pairs, vec![[0, 1], [0, 2], [1, 2]]);
        // Particle 0 is in pairs 0 and 1 (side 0 both); particle 1 in pairs 0
        // (side 1) and 2 (side 0); particle 2 in pairs 1 (side 1) and 2 (side 1).
        let p0 = &prep.vert_entries[prep.vert_offsets[0] as usize..prep.vert_offsets[1] as usize];
        let p1 = &prep.vert_entries[prep.vert_offsets[1] as usize..prep.vert_offsets[2] as usize];
        let p2 = &prep.vert_entries[prep.vert_offsets[2] as usize..prep.vert_offsets[3] as usize];
        assert_eq!(p0, &[[0, 0], [1, 0]]);
        assert_eq!(p1, &[[0, 1], [2, 0]]);
        assert_eq!(p2, &[[1, 1], [2, 1]]);
        // Each list ascending by pair index.
        for list in [p0, p1, p2] {
            for w in list.windows(2) {
                assert!(w[0][0] < w[1][0], "entries ascending by pair index");
            }
        }
    }
}
