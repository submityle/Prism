//! Host-side deterministic preparation for the cloth self-collision kernel.
//!
//! The GPU kernel does the floating-point separating-push arithmetic, but the
//! acceleration structure it walks is built here on the host so it is
//! bit-identical to the [`prism_physics_core`] golden: the same sample order
//! (every real particle, then in-range virtual particles in generation order),
//! the same integer cell assignment, the same ascending [`BTreeMap`] cell order
//! and ascending in-bucket member order, and the same ascending-by-sample
//! phase-2 incidence list. Performing cell assignment on the host keeps the
//! only GPU floating-point work the push arithmetic itself, which the parity
//! suite checks within a tight tolerance; the integer bucketing never diverges.
//!
//! # Provenance
//!
//! The virtual-particle technique is the published `NvCloth` method and uniform
//! spatial hashing is the classical Teschner et al. 2003 scheme. No Unreal
//! Engine source or derived code.

use alloc::collections::BTreeMap;

use glam::Vec3;
use prism_physics_core::VirtualParticle;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// A fully flattened, upload-ready snapshot of the cloth self-collision scene.
///
/// Every field is a plain-old-data array laid out for direct upload to a
/// storage buffer; the four-wide packing (`[_; 4]`) matches the `vec4` strides
/// the kernel binds. See [`build`] for the construction contract.
pub struct ClothPrep {
    /// Number of real particles (the leading samples).
    pub real_count: u32,
    /// Total samples: real particles followed by in-range virtual particles.
    pub sample_count: u32,
    /// Fabric thickness this scene was built against (threaded to the kernel).
    pub thickness: f32,
    /// Frozen positions, padded to `vec4` (`w` carried through untouched).
    pub positions: Vec<[f32; 4]>,
    /// Index-aligned inverse masses (`0` marks a pinned particle).
    pub inverse_masses: Vec<f32>,
    /// Per-sample vertex indices, padded to `vec4` (`w` unused).
    pub sample_verts: Vec<[u32; 4]>,
    /// Per-sample barycentric weights, padded to `vec4` (`w` unused).
    pub sample_weights: Vec<[f32; 4]>,
    /// Per-sample integer cell coordinate, padded to `vec4` (`w` unused).
    pub sample_cells: Vec<[i32; 4]>,
    /// Occupied cell coordinates, ascending in `BTreeMap` order, `w` unused.
    pub grid_keys: Vec<[i32; 4]>,
    /// `CSR` offsets into [`grid_members`](Self::grid_members), length
    /// `key_count + 1`.
    pub grid_offsets: Vec<u32>,
    /// Sample indices bucketed by cell, ascending within each bucket.
    pub grid_members: Vec<u32>,
    /// `CSR` offsets into [`vert_entries`](Self::vert_entries), length
    /// `real_count + 1`.
    pub vert_offsets: Vec<u32>,
    /// Incident `(sample index, weight slot)` pairs per vertex, ascending by
    /// sample index so each vertex folds in the golden's reduction order.
    pub vert_entries: Vec<[u32; 2]>,
}

impl ClothPrep {
    /// The number of occupied grid cells (length of
    /// [`grid_keys`](Self::grid_keys)).
    #[must_use]
    pub fn key_count(&self) -> u32 {
        u32::try_from(self.grid_keys.len()).unwrap_or(u32::MAX)
    }
}

/// The integer cell of `pos`, matching `prism_physics_core`'s `cell_of`.
///
/// Floor division by `cell_size`, performed on the host so the GPU never has to
/// reproduce the boundary rounding of a float multiply-and-floor.
#[must_use]
fn cell_of(pos: Vec3, cell_size: Real) -> [i32; 3] {
    let inv = 1.0 / cell_size;
    [
        (pos.x * inv).floor() as i32,
        (pos.y * inv).floor() as i32,
        (pos.z * inv).floor() as i32,
    ]
}

/// A barycentric sample: a real particle `[i, i, i] / [1, 0, 0]` or a virtual
/// particle carrying its triangle's three corners and weights.
#[derive(Clone, Copy)]
struct Sample {
    verts: [u32; 3],
    weights: [Real; 3],
}

impl Sample {
    fn real(index: u32) -> Sample {
        Sample {
            verts: [index, index, index],
            weights: [1.0, 0.0, 0.0],
        }
    }

    fn virtual_particle(vp: VirtualParticle) -> Sample {
        Sample {
            verts: vp.verts,
            weights: vp.weights,
        }
    }

    /// `Σ weights[k] * position[verts[k]]`, matching the golden sample position.
    fn position(&self, positions: &[Vec3]) -> Vec3 {
        let mut pos = Vec3::ZERO;
        for k in 0..3 {
            let w = self.weights[k];
            if w == 0.0 {
                continue;
            }
            pos += positions[self.verts[k] as usize] * w;
        }
        pos
    }
}

/// Builds the upload-ready [`ClothPrep`] for one Jacobi self-collision pass, or
/// returns [`None`] when the pass is a no-op.
///
/// The build mirrors `prism_physics_core`'s
/// `accumulate_virtual_jacobi_corrections` exactly: it returns [`None`] (the
/// GPU path then leaves positions untouched) when `cell_size`/`thickness` is
/// non-positive, when `inverse_masses` has a different length from `positions`,
/// or when fewer than two samples are in range. Otherwise the samples, the
/// uniform hash, and the phase-2 incidence list are all laid out in the golden's
/// deterministic order.
#[must_use]
pub fn build(
    positions: &[Vec3],
    inverse_masses: &[Real],
    virtuals: &[VirtualParticle],
    cell_size: Real,
    thickness: Real,
) -> Option<ClothPrep> {
    if cell_size <= 0.0 || thickness <= 0.0 || inverse_masses.len() != positions.len() {
        return None;
    }
    let real_count = positions.len();

    // Fixed sample order: real particles first, then in-range virtuals in
    // generation order — identical to both goldens.
    let mut samples: Vec<Sample> =
        Vec::with_capacity(real_count.saturating_add(virtuals.len()));
    for i in 0..real_count {
        samples.push(Sample::real(i as u32));
    }
    for &vp in virtuals {
        if vp.verts.iter().all(|&v| (v as usize) < real_count) {
            samples.push(Sample::virtual_particle(vp));
        }
    }
    if samples.len() < 2 {
        return None;
    }

    // Bucket samples by their frozen cell; ascending push keeps buckets stable.
    let mut grid: BTreeMap<[i32; 3], Vec<u32>> = BTreeMap::new();
    let mut sample_cells: Vec<[i32; 4]> = Vec::with_capacity(samples.len());
    for (index, sample) in samples.iter().enumerate() {
        let cell = cell_of(sample.position(positions), cell_size);
        sample_cells.push([cell[0], cell[1], cell[2], 0]);
        grid.entry(cell).or_default().push(index as u32);
    }

    // Flatten the ascending BTreeMap into sorted keys plus a CSR member list.
    let mut grid_keys: Vec<[i32; 4]> = Vec::with_capacity(grid.len());
    let mut grid_offsets: Vec<u32> = Vec::with_capacity(grid.len() + 1);
    let mut grid_members: Vec<u32> = Vec::new();
    grid_offsets.push(0);
    for (cell, bucket) in &grid {
        grid_keys.push([cell[0], cell[1], cell[2], 0]);
        grid_members.extend_from_slice(bucket);
        grid_offsets.push(u32::try_from(grid_members.len()).unwrap_or(u32::MAX));
    }

    // Phase-2 incidence: walking samples ascending gives each vertex its
    // contributions in ascending sample order, the golden's reduction order.
    let mut per_vertex: Vec<Vec<[u32; 2]>> =
        Vec::with_capacity(real_count);
    per_vertex.resize_with(real_count, Vec::new);
    for (ai, sample) in samples.iter().enumerate() {
        for k in 0..3 {
            if sample.weights[k] == 0.0 {
                continue;
            }
            let v = sample.verts[k] as usize;
            per_vertex[v].push([ai as u32, k as u32]);
        }
    }
    let mut vert_offsets: Vec<u32> = Vec::with_capacity(real_count + 1);
    let mut vert_entries: Vec<[u32; 2]> = Vec::new();
    vert_offsets.push(0);
    for entries in &per_vertex {
        vert_entries.extend_from_slice(entries);
        vert_offsets.push(u32::try_from(vert_entries.len()).unwrap_or(u32::MAX));
    }

    let positions_packed: Vec<[f32; 4]> =
        positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let sample_verts: Vec<[u32; 4]> = samples
        .iter()
        .map(|s| [s.verts[0], s.verts[1], s.verts[2], 0])
        .collect();
    let sample_weights: Vec<[f32; 4]> = samples
        .iter()
        .map(|s| [s.weights[0], s.weights[1], s.weights[2], 0.0])
        .collect();

    Some(ClothPrep {
        real_count: u32::try_from(real_count).unwrap_or(u32::MAX),
        sample_count: u32::try_from(samples.len()).unwrap_or(u32::MAX),
        thickness,
        positions: positions_packed,
        inverse_masses: inverse_masses.to_vec(),
        sample_verts,
        sample_weights,
        sample_cells,
        grid_keys,
        grid_offsets,
        grid_members,
        vert_offsets,
        vert_entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::{generate_virtual_particles, VirtualParticlePattern};

    #[test]
    fn non_positive_or_mismatched_inputs_are_no_ops() {
        let positions = vec![Vec3::ZERO, Vec3::X];
        let im = vec![1.0, 1.0];
        assert!(build(&positions, &im, &[], 0.0, 0.2).is_none());
        assert!(build(&positions, &im, &[], 1.0, 0.0).is_none());
        assert!(build(&positions, &vec![1.0], &[], 1.0, 0.2).is_none());
    }

    #[test]
    fn fewer_than_two_samples_is_a_no_op() {
        let positions = vec![Vec3::ZERO];
        let im = vec![1.0];
        assert!(build(&positions, &im, &[], 1.0, 0.2).is_none());
    }

    #[test]
    fn grid_csr_covers_every_sample_once_in_ascending_order() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        let im = vec![1.0, 1.0, 1.0];
        let prep = build(&positions, &im, &[], 1.0, 0.2).expect("non-degenerate build");

        // Keys ascending.
        for pair in prep.grid_keys.windows(2) {
            assert!(pair[0] < pair[1], "keys must be strictly ascending");
        }
        // CSR offsets monotonic and framing every member.
        assert_eq!(prep.grid_offsets.first(), Some(&0));
        assert_eq!(
            prep.grid_offsets.last(),
            Some(&(prep.grid_members.len() as u32))
        );
        // Every sample index appears exactly once across all buckets.
        let mut seen = prep.grid_members.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2]);
        // Each bucket is itself ascending.
        for w in prep.grid_offsets.windows(2) {
            let bucket = &prep.grid_members[w[0] as usize..w[1] as usize];
            for p in bucket.windows(2) {
                assert!(p[0] < p[1]);
            }
        }
    }

    #[test]
    fn real_samples_lead_virtual_samples() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        let im = vec![1.0, 1.0, 1.0];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        let prep = build(&positions, &im, &virtuals, 2.0, 0.5).expect("non-degenerate build");

        assert_eq!(prep.real_count, 3);
        assert_eq!(prep.sample_count as usize, 3 + virtuals.len());
        // The three leading samples are the real particles [i, i, i].
        for i in 0..3u32 {
            assert_eq!(prep.sample_verts[i as usize], [i, i, i, 0]);
            assert_eq!(prep.sample_weights[i as usize], [1.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn vertex_incidence_lists_are_ascending_by_sample() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        let im = vec![1.0, 1.0, 1.0];
        let virtuals =
            generate_virtual_particles(&[[0, 1, 2]], &VirtualParticlePattern::nvcloth_default());
        let prep = build(&positions, &im, &virtuals, 2.0, 0.5).expect("non-degenerate build");

        for v in 0..prep.real_count as usize {
            let lo = prep.vert_offsets[v] as usize;
            let hi = prep.vert_offsets[v + 1] as usize;
            let entries = &prep.vert_entries[lo..hi];
            // The real self-sample (sample v) always leads the vertex's list.
            assert_eq!(entries[0], [v as u32, 0]);
            for pair in entries.windows(2) {
                assert!(pair[0][0] < pair[1][0], "entries ascending by sample");
            }
        }
    }
}
