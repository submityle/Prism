//! Host-side deterministic preparation for the multi-layer garment coupling
//! kernel.
//!
//! The GPU kernel does the floating-point separating-push arithmetic, but the
//! per-particle neighbor set it folds — and the order it folds them in — are
//! built here on the host so they are bit-identical to the
//! [`prism_physics_core`] golden
//! ([`resolve_layer_coupling_jacobi`](prism_physics_core::resolve_layer_coupling_jacobi)):
//! the same integer cell assignment ([`cell_of`]), the same ascending
//! [`BTreeMap`] cell order with ascending in-bucket member order, and the same
//! 27-cell neighborhood enumeration (dx/dy/dz nested, ascending buckets).
//!
//! Unlike the point self-collision tier — which stores unordered pairs and
//! resolves both halves in a per-pair pass — inter-layer coupling is a
//! *directed, per-particle* own-slot accumulation: particle `a` owns `out[a]`
//! and sums only its own half of the separating push against every penetrating
//! cross-layer neighbor `b != a`. The broad phase therefore emits a per-particle
//! `CSR` adjacency (`nbr_offsets` + `nbr_entries`) listing each particle's
//! neighbors in the golden's exact reduction order, and a single GPU pass folds
//! it. Same-layer neighbors (which the golden skips as self-collision) are
//! filtered out here so the kernel never has to re-test them.
//!
//! Performing the broad phase on the host keeps the only GPU floating-point
//! work the separating push itself, which the parity suite checks within a
//! tight tolerance; the integer bucketing and neighbor enumeration never
//! diverge.
//!
//! # Provenance
//!
//! Uniform spatial hashing is the classical Teschner et al. 2003 scheme and the
//! layer-number stacking constraint with inverse-mass-weighted separation is a
//! standard position-based-dynamics technique. No Unreal Engine source or
//! derived code.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::LayerParams;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// A fully flattened, upload-ready snapshot of one inter-layer coupling pass.
///
/// Every field is a plain-old-data array laid out for direct upload to a
/// storage buffer; the four-wide packing (`[_; 4]`) matches the `vec4` strides
/// the kernel binds. See [`build`] for the construction contract.
pub struct ClothLayerPrep {
    /// Number of addressable particles (kernel thread count).
    pub particle_count: u32,
    /// Sanitized minimum inter-layer separation threaded to the kernel.
    pub thickness: f32,
    /// `thickness * thickness`, precomputed on the host exactly as the golden
    /// does so the radial-fallback comparison matches bit-for-bit.
    pub thickness_sq: f32,
    /// Frozen positions, padded to `vec4` (`w` carried through).
    pub positions: Vec<[f32; 4]>,
    /// Index-aligned outward surface normals, padded to `vec4` (`w` unused).
    /// Padded with zero when the caller's column is short, so the kernel reads
    /// the radial-fallback branch there exactly as the golden does.
    pub normals: Vec<[f32; 4]>,
    /// Index-aligned inverse masses (`0` marks a pinned particle).
    pub inverse_masses: Vec<f32>,
    /// Index-aligned layer numbers (lower = inner). Padded with `0` beyond the
    /// caller's column; padded entries are never indexed because such particles
    /// carry no neighbors.
    pub layer_of: Vec<u32>,
    /// `CSR` offsets into [`nbr_entries`](Self::nbr_entries), length
    /// `particle_count + 1`.
    pub nbr_offsets: Vec<u32>,
    /// Each particle's cross-layer neighbors in the golden's reduction order
    /// (ascending cells, ascending buckets, 27-cell traversal), same-layer
    /// neighbors already filtered out.
    pub nbr_entries: Vec<u32>,
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

/// Builds the upload-ready [`ClothLayerPrep`] for one Jacobi inter-layer
/// coupling pass, or returns [`None`] when the pass is a no-op.
///
/// The build mirrors `prism_physics_core`'s
/// `accumulate_layer_jacobi_corrections` broad phase exactly: `params` is
/// [`sanitized`](LayerParams::sanitized) first, then the pass is a no-op
/// (returning [`None`], so the GPU path leaves its inputs untouched) when the
/// sanitized `thickness`/`cell_size` is non-positive, when
/// `inverse_masses.len()` differs from `positions.len()`, when fewer than two
/// particles are addressable, or when no cross-layer neighbor pair shares a
/// 27-cell neighborhood. Only particles carrying a `layer_of` entry are
/// bucketed, exactly as the golden does.
#[must_use]
pub fn build(
    positions: &[Vec3],
    inverse_masses: &[Real],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) -> Option<ClothLayerPrep> {
    let params = params.sanitized();
    if params.thickness <= 0.0
        || params.cell_size <= 0.0
        || positions.len() < 2
        || inverse_masses.len() != positions.len()
    {
        return None;
    }
    let count = positions.len();

    // Broad phase: bucket every particle that carries a layer number into its
    // single integer cell, keyed by an ascending BTreeMap so cell order and
    // in-bucket member order are fixed, matching the golden's grid build.
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, &pos) in positions.iter().enumerate() {
        if index < layer_of.len() {
            let cell = cell_of(pos, params.cell_size);
            grid.entry(cell).or_default().push(index as u32);
        }
    }

    // Per-particle directed adjacency: for each particle `a`, walk its 27-cell
    // neighborhood (dx/dy/dz nested, ascending neighbor buckets) and record
    // every cross-layer neighbor `b != a` in that exact order — the golden's
    // reduction order. Same-layer neighbors are skipped (self-collision's job),
    // so the kernel never re-tests them.
    let mut per_particle: Vec<Vec<u32>> = Vec::with_capacity(count);
    per_particle.resize_with(count, Vec::new);
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            let list = &mut per_particle[ai];
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
                            // Same-layer contacts belong to self-collision.
                            if layer_of[ai] == layer_of[b as usize] {
                                continue;
                            }
                            list.push(b);
                        }
                    }
                }
            }
        }
    }

    let total: usize = per_particle.iter().map(Vec::len).sum();
    if total == 0 {
        return None;
    }

    let mut nbr_offsets: Vec<u32> = Vec::with_capacity(count + 1);
    let mut nbr_entries: Vec<u32> = Vec::with_capacity(total);
    nbr_offsets.push(0);
    for list in &per_particle {
        nbr_entries.extend_from_slice(list);
        nbr_offsets.push(u32::try_from(nbr_entries.len()).unwrap_or(u32::MAX));
    }

    let positions_packed: Vec<[f32; 4]> =
        positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let normals_packed: Vec<[f32; 4]> = (0..count)
        .map(|i| {
            let n = normals.get(i).copied().unwrap_or(Vec3::ZERO);
            [n.x, n.y, n.z, 0.0]
        })
        .collect();
    let layer_packed: Vec<u32> = (0..count)
        .map(|i| layer_of.get(i).copied().unwrap_or(0))
        .collect();

    Some(ClothLayerPrep {
        particle_count: u32::try_from(count).unwrap_or(u32::MAX),
        thickness: params.thickness,
        thickness_sq: params.thickness * params.thickness,
        positions: positions_packed,
        normals: normals_packed,
        inverse_masses: inverse_masses.to_vec(),
        layer_of: layer_packed,
        nbr_offsets,
        nbr_entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LayerParams {
        LayerParams {
            thickness: 0.1,
            cell_size: 0.2,
        }
    }

    #[test]
    fn degenerate_inputs_are_a_no_op() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        // Disabled thickness.
        assert!(build(
            &positions,
            &im,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.0,
                cell_size: 0.2
            }
        )
        .is_none());
        // A zero cell size with a positive thickness is *not* a no-op: the
        // golden's `sanitized()` raises `cell_size` up to `thickness`, so the
        // pass still runs. Mirror that here rather than disabling.
        assert!(build(
            &positions,
            &im,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.0
            }
        )
        .is_some());
        // Non-positive thickness genuinely disables, even with a sized cell.
        assert!(build(
            &positions,
            &im,
            &layer_of,
            &normals,
            LayerParams {
                thickness: -1.0,
                cell_size: 0.2
            }
        )
        .is_none());
        // Fewer than two particles.
        assert!(build(&positions[..1], &im[..1], &layer_of[..1], &normals[..1], params()).is_none());
        // Mismatched inverse-mass length.
        assert!(build(&positions, &im[..1], &layer_of, &normals, params()).is_none());
    }

    #[test]
    fn separated_particles_produce_no_neighbors() {
        // Two particles many cells apart share no 27-cell neighborhood.
        let positions = [Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        assert!(build(&positions, &im, &layer_of, &normals, params()).is_none());
    }

    #[test]
    fn same_layer_pair_produces_no_neighbors() {
        // Co-located but same layer: all neighbors filtered, so a no-op.
        let positions = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [2u32, 2u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        assert!(build(&positions, &im, &layer_of, &normals, params()).is_none());
    }

    #[test]
    fn cross_layer_pair_is_captured_both_ways() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let prep = build(&positions, &im, &layer_of, &normals, params()).expect("non-degenerate");
        assert_eq!(prep.particle_count, 2);
        // Each particle sees the other exactly once.
        assert_eq!(prep.nbr_offsets, vec![0, 1, 2]);
        assert_eq!(prep.nbr_entries, vec![1, 0]);
    }

    #[test]
    fn short_normal_and_layer_columns_are_padded() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        // Normals column shorter than positions: padded with zero.
        let normals = [Vec3::new(0.0, 1.0, 0.0)];
        let prep = build(&positions, &im, &layer_of, &normals, params()).expect("non-degenerate");
        assert_eq!(prep.normals.len(), 2);
        assert_eq!(prep.normals[1], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(prep.layer_of, vec![0, 1]);
        assert_eq!(prep.thickness_sq, 0.1 * 0.1);
    }

    #[test]
    fn cluster_lists_each_particle_in_ascending_cell_order() {
        // One inner (layer 0) co-located with two outer (layer 1) particles.
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -0.03, 0.0),
            Vec3::new(0.01, -0.02, 0.0),
        ];
        let im = [1.0, 1.0, 1.0];
        let layer_of = [0u32, 1u32, 1u32];
        let normals = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let prep = build(&positions, &im, &layer_of, &normals, params()).expect("non-degenerate");
        // Particle 0 (inner) sees both outer particles; particles 1 and 2 each
        // see only particle 0 (the other same-layer neighbor is filtered).
        let p0 = &prep.nbr_entries[prep.nbr_offsets[0] as usize..prep.nbr_offsets[1] as usize];
        let p1 = &prep.nbr_entries[prep.nbr_offsets[1] as usize..prep.nbr_offsets[2] as usize];
        let p2 = &prep.nbr_entries[prep.nbr_offsets[2] as usize..prep.nbr_offsets[3] as usize];
        assert_eq!(p0, &[1, 2]);
        assert_eq!(p1, &[0]);
        assert_eq!(p2, &[0]);
    }
}
