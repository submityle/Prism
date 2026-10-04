//! Water `FLIP`/`APIC` cell-histogram compute kernel: the `WESL` shader plus
//! its bit-exact `CPU` twin.
//!
//! This is pass 1 of 3 in the particle counting-sort chain that reorders
//! particles into cell-major order before the measured `MAC` transfer
//! bottleneck (`P2G`/`G2P`). The chain is histogram -> exclusive scan ->
//! scatter, mirroring the `CPU` golden
//! [`super::super::flip_sort::counting_sort_particles`]: this pass fills the
//! per-cell particle count the scan turns into `CSR` write cursors.
//!
//! [`WATER_FLIP_CELL_HISTOGRAM_WESL`] is the shader (entry point
//! `water_flip_cell_histogram`, see
//! [`WaterKernel::FlipCellHistogram`](super::super::kernels::WaterKernel)) and
//! [`dispatch_flip_cell_histogram`] is its bit-exact `CPU` twin. Because the
//! sandbox has no `GPU`, the twin is the correctness proof: it consumes the
//! identical buffer `ABI` (one read storage buffer `positions_in`, one atomic
//! read-write storage buffer `counts`, one uniform param block, a 64-lane group
//! over the `Particle` domain) and reconstructs the shader's cell arithmetic
//! inline. The parity tests diff the twin lane-for-lane against the independent
//! golden [`super::super::flip_sort::cell_histogram`] (which walks a
//! [`MacCellGrid`](super::super::flip_sort::MacCellGrid) over `Vec3` input), so
//! they prove the packed-buffer adapter reproduces the golden rather than
//! asserting a tautology. Only `+ - * /` and comparisons appear; no `f32`
//! equality and no AI/ML.

use alloc::vec;
use alloc::vec::Vec;

use super::super::flip_sort::MacCellGrid;
use super::super::{Vec3, EPS};

/// `WESL` source of the water `FLIP`/`APIC` cell-histogram compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_CELL_HISTOGRAM_WESL: &str = include_str!("water_flip_cell_histogram.wesl");

/// `f32` lanes per particle in the `positions_in` buffer: the `vec4`
/// `(x, y, z, w)` whose `w` lane is ignored by the histogram.
pub const FLIP_HISTOGRAM_POSITION_FLOATS: usize = 4;

/// Uniform parameter block for the cell histogram, mirroring the shader's
/// `FlipHistogramParams`.
///
/// It carries the row-major [`MacCellGrid`](super::super::flip_sort::MacCellGrid)
/// description the shader needs inline, plus the per-dispatch `particle_count`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipHistogramParams {
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

impl FlipHistogramParams {
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
    let base = idx * FLIP_HISTOGRAM_POSITION_FLOATS;
    Vec3::new(positions[base], positions[base + 1], positions[base + 2])
}

/// Integer cell coordinate of a world position, or `None` when it falls outside
/// the grid. Mirrors the shader's `flip_cell_coord` and the `CPU`
/// [`MacCellGrid::cell_coord`](super::super::flip_sort::MacCellGrid::cell_coord)
/// arithmetic, reconstructed inline so the parity test is not a tautology.
#[inline]
fn cell_coord(params: FlipHistogramParams, p: Vec3) -> Option<(u32, u32, u32)> {
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
fn flat_index(params: FlipHistogramParams, cx: u32, cy: u32, cz: u32) -> usize {
    let nx = params.nx as usize;
    let ny = params.ny as usize;
    (cz as usize) * nx * ny + (cy as usize) * nx + (cx as usize)
}

/// Bit-exact `CPU` twin of the `water_flip_cell_histogram` kernel.
///
/// `positions` is the packed `(x, y, z, w)` lane buffer; the `w` lane is
/// ignored. Returns the `counts` buffer (`params.cell_count()` lanes), each
/// lane the number of in-grid particles whose centre falls in that cell,
/// row-major. The accumulation saturates, exactly as the golden
/// [`cell_histogram`](super::super::flip_sort::cell_histogram) does, so a
/// pathological bin can clamp rather than wrap. A degenerate request — a buffer
/// that cannot hold the declared particle count, or a zero-cell grid — yields
/// an all-zero (possibly empty) well-formed `counts` buffer, matching the
/// shader's "skip, do not crash" contract.
#[must_use]
pub fn dispatch_flip_cell_histogram(positions: &[f32], params: FlipHistogramParams) -> Vec<u32> {
    let cells = params.cell_count();
    let count = params.particle_count as usize;
    let mut counts = vec![0u32; cells];
    if cells == 0 || positions.len() < count * FLIP_HISTOGRAM_POSITION_FLOATS {
        return counts;
    }
    let mut i = 0usize;
    while i < count {
        let p = read_position(positions, i);
        if let Some((cx, cy, cz)) = cell_coord(params, p) {
            let flat = flat_index(params, cx, cy, cz);
            counts[flat] = counts[flat].saturating_add(1);
        }
        i += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::flip_sort::cell_histogram;
    use crate::water::kernels::{DispatchDomain, WaterKernel};

    fn params() -> FlipHistogramParams {
        FlipHistogramParams {
            grid_origin: Vec3::new(0.0, 0.0, 0.0),
            dx: 1.0,
            nx: 2,
            ny: 2,
            nz: 2,
            particle_count: 0,
        }
    }

    fn flatten(positions: &[Vec3]) -> Vec<f32> {
        let mut out = Vec::with_capacity(positions.len() * FLIP_HISTOGRAM_POSITION_FLOATS);
        for (i, p) in positions.iter().enumerate() {
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
            // `w` lane coded to the index to prove it is ignored by the pass.
            out.push(i as f32);
        }
        out
    }

    #[test]
    fn descriptor_abi_matches_the_shader_bindings() {
        // The twin reads one storage buffer, writes one atomic storage buffer,
        // and reads one uniform: two storage + one uniform, no textures, a
        // 64-lane linear group over the particle domain. Lock the contract so a
        // shader-binding drift fails here.
        let d = WaterKernel::FlipCellHistogram.descriptor();
        assert_eq!(d.layout.storage_buffers, 2);
        assert_eq!(d.layout.uniform_buffers, 1);
        assert_eq!(d.layout.storage_textures, 0);
        assert_eq!(d.layout.sampled_textures, 0);
        assert_eq!(d.domain, DispatchDomain::Particle);
        assert_eq!(d.workgroup.x, 64);
        assert_eq!(d.workgroup.y, 1);
        assert_eq!(d.workgroup.z, 1);
        assert_eq!(
            WaterKernel::FlipCellHistogram.wesl_entry_point(),
            "water_flip_cell_histogram"
        );
    }

    #[test]
    fn counts_match_a_hand_computed_layout() {
        // Independent of any histogram routine: three particles in cell 0, one
        // in cell 1, one in cell 7, one dropped out of grid.
        let pos = [
            Vec3::new(0.1, 0.1, 0.1),  // cell 0
            Vec3::new(0.9, 0.2, 0.3),  // cell 0
            Vec3::new(0.4, 0.4, 0.4),  // cell 0
            Vec3::new(1.5, 0.5, 0.5),  // cell 1
            Vec3::new(1.9, 1.9, 1.9),  // cell 7
            Vec3::new(-2.0, 0.0, 0.0), // dropped
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let counts = dispatch_flip_cell_histogram(&flatten(&pos), p);
        assert_eq!(counts.len(), 8);
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 1);
        assert_eq!(counts[7], 1);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[3], 0);
        assert_eq!(counts.iter().sum::<u32>(), 5);
    }

    #[test]
    fn twin_matches_the_golden_cell_histogram() {
        let pos = [
            Vec3::new(1.9, 1.9, 1.9),
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(1.1, 0.1, 1.1),
            Vec3::new(0.1, 1.1, 0.1),
            Vec3::new(1.1, 1.1, 1.1),
            Vec3::new(0.6, 0.2, 1.3),
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let twin = dispatch_flip_cell_histogram(&flatten(&pos), p);
        let golden = cell_histogram(p.grid(), &pos);
        assert_eq!(twin, golden);
        // Not vacuous: at least one bin must be populated.
        assert!(twin.iter().any(|&c| c > 0));
    }

    #[test]
    fn out_of_grid_and_degenerate_dx_are_skipped_not_panicked() {
        let pos = [
            Vec3::new(-0.1, 0.5, 0.5), // negative local
            Vec3::new(2.5, 0.5, 0.5),  // beyond extent
            Vec3::new(0.5, 0.5, 0.5),  // in cell 0
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let counts = dispatch_flip_cell_histogram(&flatten(&pos), p);
        assert_eq!(counts[0], 1);
        assert_eq!(counts.iter().sum::<u32>(), 1);

        // Degenerate cell edge: nothing counted, buffer still well-formed.
        let degenerate = FlipHistogramParams { dx: 0.0, ..p };
        let counts = dispatch_flip_cell_histogram(&flatten(&pos), degenerate);
        assert_eq!(counts, vec![0u32; 8]);
    }

    #[test]
    fn mis_sized_buffer_yields_well_formed_zeros() {
        let pos = [Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5)];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let flat = flatten(&pos);
        // Drop a lane so the buffer cannot hold the declared particle count.
        let short = &flat[..flat.len() - 1];
        let counts = dispatch_flip_cell_histogram(short, p);
        assert_eq!(counts, vec![0u32; 8]);

        // Zero-cell grid: empty, well-formed, no panic.
        let empty = FlipHistogramParams { nx: 0, ..params() };
        assert!(dispatch_flip_cell_histogram(&flat, empty).is_empty());
    }

    #[test]
    fn total_counts_equal_in_grid_particles_and_is_deterministic() {
        let pos = [
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(1.8, 0.2, 0.2),
            Vec3::new(0.2, 1.8, 1.8),
            Vec3::new(-9.0, 0.0, 0.0), // dropped
            Vec3::new(1.8, 1.8, 0.2),
        ];
        let mut p = params();
        p.particle_count = pos.len() as u32;
        let flat = flatten(&pos);
        let a = dispatch_flip_cell_histogram(&flat, p);
        let b = dispatch_flip_cell_histogram(&flat, p);
        assert_eq!(a, b);
        // Four of five particles are in-grid.
        assert_eq!(a.iter().sum::<u32>(), 4);
    }
}
