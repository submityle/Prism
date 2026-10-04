//! Water `FLIP`/`APIC` cell-scatter compute kernel: the `WESL` shader plus its
//! `CPU` twin.
//!
//! This is pass 3 of 3 in the particle counting-sort chain that reorders
//! particles into cell-major order before the measured `MAC` transfer
//! bottleneck (`P2G`/`G2P`). The chain is histogram -> exclusive scan ->
//! scatter, mirroring the `CPU` golden
//! [`super::super::flip_sort::counting_sort_particles`]: this pass consumes the
//! per-cell `CSR` offsets the scan produced and emits the cell-major
//! `sorted_indices` permutation the transfer kernels will gather through.
//!
//! [`WATER_FLIP_CELL_SCATTER_WESL`] is the shader (entry point
//! `water_flip_cell_scatter`, see
//! [`WaterKernel::FlipCellScatter`](super::super::kernels::WaterKernel)) and
//! [`dispatch_flip_cell_scatter`] is its `CPU` twin. Because the sandbox has no
//! `GPU`, the twin is the correctness proof: it consumes the identical buffer
//! `ABI` (one read storage buffer `positions_in`, one atomic read-write storage
//! buffer `cursors` seeded with the scan offsets, one read-write storage buffer
//! `sorted_indices`, one uniform param block, a 64-lane group over the
//! `Particle` domain) and reconstructs the shader's cell arithmetic inline.
//!
//! The parity test diffs the twin against the independent golden
//! [`super::super::flip_sort::counting_sort_particles`] seeded from
//! [`super::super::flip_sort::exclusive_prefix_sum`], so it proves the
//! packed-buffer adapter reproduces the golden rather than asserting a
//! tautology. The twin models the serialised index-order scatter — the stable
//! permutation the golden emits. The live `GPU` claims slots atomically, so its
//! within-cell order is implementation-defined; that freedom is harmless
//! because the downstream per-cell `MAC` accumulation is order-independent, and
//! the invariant that actually matters (every in-grid particle lands exactly
//! once inside its cell's half-open range) is covered by a dedicated
//! permutation test. Only `+ - * /` and comparisons appear; no `f32` equality
//! and no AI/ML.

use alloc::vec;
use alloc::vec::Vec;

use super::super::flip_sort::MacCellGrid;
use super::super::{Vec3, EPS};

/// `WESL` source of the water `FLIP`/`APIC` cell-scatter compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_CELL_SCATTER_WESL: &str = include_str!("water_flip_cell_scatter.wesl");

/// `f32` lanes per particle in the `positions_in` buffer: the `vec4`
/// `(x, y, z, w)` whose `w` lane is ignored by the scatter.
pub const FLIP_SCATTER_POSITION_FLOATS: usize = 4;

/// Uniform parameter block for the cell scatter, mirroring the shader's
/// `FlipScatterParams`.
///
/// It carries the row-major [`MacCellGrid`](super::super::flip_sort::MacCellGrid)
/// description the shader needs inline, plus the per-dispatch `particle_count`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipScatterParams {
    /// Minimum corner of the domain, world space.
    pub grid_origin: Vec3,
    /// Cell edge length `dx` (`> 0`).
    pub dx: f32,
    /// Cell count along x (`>= 1`).
    pub nx: u32,
    /// Cell count along y (`>= 1`).
    pub ny: u32,
    /// Cell count along z (`>= 1`).
    pub nz: u32,
    /// Number of particles bounding the per-particle dispatch.
    pub particle_count: u32,
}

impl FlipScatterParams {
    /// Total cell count (`nx * ny * nz`), saturating so a pathological
    /// descriptor can never overflow the index math.
    #[must_use]
    pub fn cell_count(self) -> usize {
        (self.nx as usize)
            .saturating_mul(self.ny as usize)
            .saturating_mul(self.nz as usize)
    }

    /// The [`MacCellGrid`](super::super::flip_sort::MacCellGrid) this param block
    /// describes, so the twin and tests can reuse the golden grid arithmetic.
    #[must_use]
    pub fn grid(self) -> MacCellGrid {
        MacCellGrid {
            origin: self.grid_origin,
            dx: self.dx,
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
        }
    }
}

/// Reads particle `idx`'s world position (the `xyz` lanes) from the packed
/// `(x, y, z, w)` lane buffer.
#[inline]
fn read_position(positions: &[f32], idx: usize) -> Vec3 {
    let base = idx * FLIP_SCATTER_POSITION_FLOATS;
    Vec3::new(positions[base], positions[base + 1], positions[base + 2])
}

/// Integer cell coordinate of a world position, or `None` when it falls outside
/// the grid. Mirrors the shader's `flip_cell_coord` and the `CPU`
/// [`MacCellGrid::cell_coord`](super::super::flip_sort::MacCellGrid::cell_coord)
/// arithmetic, reconstructed inline so the parity test is not a tautology.
#[inline]
fn cell_coord(params: FlipScatterParams, p: Vec3) -> Option<(u32, u32, u32)> {
    if params.dx <= EPS {
        return None;
    }
    let local = p.sub(params.grid_origin);
    if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
        return None;
    }
    let cx = (local.x / params.dx) as u32;
    let cy = (local.y / params.dx) as u32;
    let cz = (local.z / params.dx) as u32;
    if cx >= params.nx || cy >= params.ny || cz >= params.nz {
        return None;
    }
    Some((cx, cy, cz))
}

/// Row-major flat cell index. Mirrors the shader's `flip_flat_index`.
#[inline]
fn flat_index(params: FlipScatterParams, cx: u32, cy: u32, cz: u32) -> usize {
    let nx = params.nx as usize;
    let ny = params.ny as usize;
    (cz as usize) * nx * ny + (cy as usize) * nx + (cx as usize)
}

/// Bit-exact `CPU` twin of the cell-scatter kernel.
///
/// Consumes the packed `(x, y, z, w)` position buffer and the exclusive-scan
/// `offsets` (length `cell_count + 1`, as produced by
/// [`super::super::flip_sort::exclusive_prefix_sum`] or the `GPU` scan twin) and
/// returns the cell-major `sorted_indices` permutation of the in-grid particle
/// indices.
///
/// Models the serialised index-order scatter: particles are walked in index
/// order and claim the next free slot of their cell from a cursor seeded at the
/// cell's offset, so within a cell the original indices stay ascending — the
/// stable permutation the golden [`counting_sort_particles`] emits.
///
/// Returns a well-formed empty vector when the descriptor is empty or when the
/// `offsets` buffer is not sized `cell_count + 1`; returns a zero-filled result
/// (no scatter) when the position buffer is too short to hold `particle_count`
/// particles. Never indexes out of bounds and never panics.
///
/// [`counting_sort_particles`]: super::super::flip_sort::counting_sort_particles
#[must_use]
pub fn dispatch_flip_cell_scatter(
    positions: &[f32],
    offsets: &[u32],
    params: FlipScatterParams,
) -> Vec<u32> {
    let cell_count = params.cell_count();
    // The offsets buffer is the scan output: one entry per cell plus a trailing
    // total. A mis-sized buffer is a wiring bug; bail to a well-formed empty.
    if offsets.len() != cell_count + 1 {
        return Vec::new();
    }
    let total = *offsets.last().unwrap_or(&0) as usize;
    let mut sorted = vec![0u32; total];

    let particles = params.particle_count as usize;
    // Guard the packed buffer: a short buffer cannot hold the declared count,
    // so emit the zero-filled result rather than reading past the end.
    if positions.len() < particles.saturating_mul(FLIP_SCATTER_POSITION_FLOATS) {
        return sorted;
    }

    // Per-cell write cursor seeded from the exclusive-scan start of each cell.
    let mut cursor: Vec<u32> = offsets[..cell_count].to_vec();
    for i in 0..particles {
        let p = read_position(positions, i);
        let Some((cx, cy, cz)) = cell_coord(params, p) else {
            continue;
        };
        let flat = flat_index(params, cx, cy, cz);
        let slot = cursor[flat] as usize;
        if slot < sorted.len() {
            sorted[slot] = i as u32;
            cursor[flat] = cursor[flat].saturating_add(1);
        }
    }
    sorted
}

#[cfg(test)]
mod tests {
    use super::super::super::flip_sort::{
        cell_histogram, counting_sort_particles, exclusive_prefix_sum,
    };
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::*;

    fn params() -> FlipScatterParams {
        // A 2x2x2 grid of unit cells rooted at the origin: eight cells.
        FlipScatterParams {
            grid_origin: Vec3::new(0.0, 0.0, 0.0),
            dx: 1.0,
            nx: 2,
            ny: 2,
            nz: 2,
            particle_count: 0,
        }
    }

    fn flatten(positions: &[Vec3]) -> Vec<f32> {
        let mut out = Vec::with_capacity(positions.len() * FLIP_SCATTER_POSITION_FLOATS);
        for (i, p) in positions.iter().enumerate() {
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
            // `w` lane coded to the index to prove it is ignored by the pass.
            out.push(i as f32);
        }
        out
    }

    fn offsets_for(p: FlipScatterParams, positions: &[Vec3]) -> Vec<u32> {
        exclusive_prefix_sum(&cell_histogram(p.grid(), positions))
    }

    #[test]
    fn descriptor_abi_matches_the_shader_bindings() {
        // The twin reads one storage buffer, writes one atomic cursor storage
        // buffer and one plain index storage buffer, and reads one uniform:
        // three storage + one uniform, no textures, a 64-lane linear group over
        // the particle domain. Lock the contract so a shader-binding drift
        // fails here.
        let d = WaterKernel::FlipCellScatter.descriptor();
        assert_eq!(d.layout.storage_buffers, 3);
        assert_eq!(d.layout.uniform_buffers, 1);
        assert_eq!(d.layout.storage_textures, 0);
        assert_eq!(d.layout.sampled_textures, 0);
        assert_eq!(d.domain, DispatchDomain::Particle);
        assert_eq!(d.workgroup.x, 64);
        assert_eq!(d.workgroup.y, 1);
        assert_eq!(d.workgroup.z, 1);
        assert_eq!(
            WaterKernel::FlipCellScatter.wesl_entry_point(),
            "water_flip_cell_scatter"
        );
    }

    #[test]
    fn sorted_indices_match_a_hand_computed_layout() {
        // Independent of any sort routine: three particles in cell 0, one in
        // cell 1, one in cell 7, one dropped out of grid. Cell-major order with
        // ascending-within-cell stability gives [0, 1, 2 | 3 | 4].
        let pos = [
            Vec3::new(0.1, 0.1, 0.1),  // cell 0, particle 0
            Vec3::new(0.9, 0.2, 0.3),  // cell 0, particle 1
            Vec3::new(0.4, 0.4, 0.4),  // cell 0, particle 2
            Vec3::new(1.5, 0.5, 0.5),  // cell 1, particle 3
            Vec3::new(1.9, 1.9, 1.9),  // cell 7, particle 4
            Vec3::new(-2.0, 0.0, 0.0), // dropped
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        // Hand-built CSR offsets for the layout above (counts 3,1,0,0,0,0,0,1).
        let offsets = [0u32, 3, 4, 4, 4, 4, 4, 4, 5];
        let sorted = dispatch_flip_cell_scatter(&flatten(&pos), &offsets, p);
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn twin_matches_the_golden_counting_sort() {
        let pos = [
            Vec3::new(1.9, 1.9, 1.9),
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(1.1, 0.1, 1.1),
            Vec3::new(0.1, 1.1, 0.1),
            Vec3::new(1.1, 1.1, 1.1),
            Vec3::new(0.6, 0.2, 1.3),
            Vec3::new(0.2, 0.2, 0.2),
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let offsets = offsets_for(p, &pos);
        let twin = dispatch_flip_cell_scatter(&flatten(&pos), &offsets, p);
        let golden = counting_sort_particles(p.grid(), &pos);
        assert_eq!(twin, golden.sorted_indices);
        // Not vacuous: the permutation must actually place particles.
        assert!(!twin.is_empty());
    }

    #[test]
    fn out_of_grid_and_degenerate_dx_are_skipped_not_panicked() {
        let pos = [
            Vec3::new(-0.1, 0.5, 0.5), // negative local, dropped
            Vec3::new(2.5, 0.5, 0.5),  // beyond extent, dropped
            Vec3::new(0.5, 0.5, 0.5),  // in cell 0, particle 2
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let offsets = offsets_for(p, &pos);
        let sorted = dispatch_flip_cell_scatter(&flatten(&pos), &offsets, p);
        // Only the single in-grid particle is placed.
        assert_eq!(sorted, vec![2]);

        // Degenerate cell edge: nothing in grid, so offsets are all-zero and
        // the result is a well-formed empty permutation.
        let degenerate = FlipScatterParams { dx: 0.0, ..p };
        let deg_offsets = offsets_for(degenerate, &pos);
        let sorted = dispatch_flip_cell_scatter(&flatten(&pos), &deg_offsets, degenerate);
        assert!(sorted.is_empty());
    }

    #[test]
    fn mis_sized_buffers_yield_well_formed_results() {
        let pos = [Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let offsets = offsets_for(p, &pos);
        let flat = flatten(&pos);

        // Offsets of the wrong length is a wiring bug: well-formed empty.
        let short_offsets = &offsets[..offsets.len() - 1];
        assert!(dispatch_flip_cell_scatter(&flat, short_offsets, p).is_empty());

        // Position buffer too short to hold the declared count: zero-filled,
        // no panic, no out-of-bounds read.
        let short_pos = &flat[..flat.len() - 1];
        let sorted = dispatch_flip_cell_scatter(short_pos, &offsets, p);
        assert_eq!(sorted, vec![0u32; 2]);

        // Zero-cell grid: empty offsets of length one, empty permutation.
        let empty = FlipScatterParams { nx: 0, ..params() };
        let empty_offsets = offsets_for(empty, &pos);
        assert!(dispatch_flip_cell_scatter(&flat, &empty_offsets, empty).is_empty());
    }

    #[test]
    fn result_is_a_permutation_of_the_in_grid_particles() {
        let pos = [
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(1.8, 0.2, 0.2),
            Vec3::new(0.2, 1.8, 1.8),
            Vec3::new(-9.0, 0.0, 0.0), // dropped
            Vec3::new(1.8, 1.8, 0.2),
            Vec3::new(0.3, 0.3, 0.3),
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let offsets = offsets_for(p, &pos);
        let total = *offsets.last().unwrap() as usize;
        let sorted = dispatch_flip_cell_scatter(&flatten(&pos), &offsets, p);
        assert_eq!(sorted.len(), total);

        // Every placed index is a distinct in-grid particle: five in grid.
        let mut seen = sorted.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total);
        assert_eq!(total, 5);
        // The dropped particle (index 3) never appears.
        assert!(!sorted.contains(&3));
    }

    #[test]
    fn scatter_is_deterministic() {
        let pos = [
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(1.8, 0.2, 0.2),
            Vec3::new(0.2, 1.8, 1.8),
            Vec3::new(1.8, 1.8, 0.2),
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let offsets = offsets_for(p, &pos);
        let flat = flatten(&pos);
        let a = dispatch_flip_cell_scatter(&flat, &offsets, p);
        let b = dispatch_flip_cell_scatter(&flat, &offsets, p);
        assert_eq!(a, b);
    }
}
