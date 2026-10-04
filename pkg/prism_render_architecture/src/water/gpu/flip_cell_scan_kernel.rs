//! Water `FLIP`/`APIC` cell-offset scan compute kernel: the `WESL` shader plus
//! its bit-exact `CPU` twin.
//!
//! This is pass 2 of 3 in the particle counting-sort chain that reorders
//! particles into cell-major order before the measured `MAC` transfer
//! bottleneck (`P2G`/`G2P`). The chain is histogram -> exclusive scan ->
//! scatter, mirroring the `CPU` golden
//! [`super::super::flip_sort::counting_sort_particles`]: this pass turns the
//! per-cell counts the histogram wrote into the `CSR` offsets the scatter pass
//! seeds its per-cell write cursor from.
//!
//! [`WATER_FLIP_CELL_SCAN_WESL`] is the shader (entry point
//! `water_flip_cell_scan`, see
//! [`WaterKernel::FlipCellScan`](super::super::kernels::WaterKernel)) and
//! [`dispatch_flip_cell_scan`] is its bit-exact `CPU` twin. Because the sandbox
//! has no `GPU`, the twin is the correctness proof: it consumes the identical
//! buffer `ABI` (one read storage buffer `counts`, one read-write storage
//! buffer `offsets`, one uniform param block, a single 256-lane workgroup over
//! the `Cells` domain) and reproduces the shader's `CSR` offset result. The
//! parity tests diff the twin against the independent golden
//! [`super::super::flip_sort::exclusive_prefix_sum`], so they prove the
//! packed-buffer adapter reproduces the golden rather than asserting a
//! tautology. Only integer `+` and comparisons appear; no `f32` equality and
//! no AI/ML.

use alloc::vec;
use alloc::vec::Vec;

/// `WESL` source of the water `FLIP`/`APIC` cell-offset scan compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_CELL_SCAN_WESL: &str = include_str!("water_flip_cell_scan.wesl");

/// Uniform parameter block for the cell-offset scan, mirroring the shader's
/// `FlipScanParams`.
///
/// It carries the flat pressure-cell count the scan sweeps; the output
/// `offsets` buffer holds `cell_count + 1` entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlipScanParams {
    /// Number of pressure cells (`nx * ny * nz`).
    pub cell_count: u32,
}

/// Exclusive prefix sum of `counts` into `CSR` cell offsets, the bit-exact
/// `CPU` twin of the `water_flip_cell_scan` shader.
///
/// Returns a vector of length `cell_count + 1`: `offsets[0] = 0`,
/// `offsets[c + 1] = sum(counts[0..=c])`, so `offsets[last]` is the number of
/// in-grid particles. The accumulation saturates, exactly as the golden
/// [`exclusive_prefix_sum`](super::super::flip_sort::exclusive_prefix_sum)
/// does, so a pathological total clamps rather than wrapping into a bogus
/// cursor. A degenerate request — a `counts` buffer that cannot hold the
/// declared cell count — yields an all-zero, well-formed `offsets` buffer,
/// matching the shader's "skip, do not crash" contract; a zero-cell grid
/// yields the single-element `[0]` the golden produces for an empty input.
#[must_use]
pub fn dispatch_flip_cell_scan(counts: &[u32], params: FlipScanParams) -> Vec<u32> {
    let cells = params.cell_count as usize;
    let mut offsets = vec![0u32; cells + 1];
    if counts.len() < cells {
        // Mis-sized buffer: cannot scan, return well-formed zeros.
        return offsets;
    }
    let mut running = 0u32;
    let mut c = 0usize;
    while c < cells {
        running = running.saturating_add(counts[c]);
        offsets[c + 1] = running;
        c += 1;
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::flip_sort::exclusive_prefix_sum;
    use crate::water::kernels::{DispatchDomain, WaterKernel};

    #[test]
    fn descriptor_abi_matches_the_shader_bindings() {
        // The twin reads one storage buffer (`counts`), writes one storage
        // buffer (`offsets`), and reads one uniform: two storage + one uniform,
        // no textures, a single 256-lane group over the `Cells` domain. Lock
        // the contract so a shader-binding drift fails here.
        let d = WaterKernel::FlipCellScan.descriptor();
        assert_eq!(d.layout.storage_buffers, 2);
        assert_eq!(d.layout.uniform_buffers, 1);
        assert_eq!(d.layout.storage_textures, 0);
        assert_eq!(d.layout.sampled_textures, 0);
        assert_eq!(d.domain, DispatchDomain::Cells);
        assert_eq!(d.workgroup.x, 256);
        assert_eq!(d.workgroup.y, 1);
        assert_eq!(d.workgroup.z, 1);
        assert_eq!(
            WaterKernel::FlipCellScan.wesl_entry_point(),
            "water_flip_cell_scan"
        );
    }

    #[test]
    fn offsets_match_a_hand_computed_csr_layout() {
        // Independent of any scan routine: three particles in cell 0, one in
        // cell 1, one in cell 7 (the histogram of the hand-computed layout the
        // histogram twin test uses).
        let counts = [3u32, 1, 0, 0, 0, 0, 0, 1];
        let offsets = dispatch_flip_cell_scan(
            &counts,
            FlipScanParams {
                cell_count: counts.len() as u32,
            },
        );
        assert_eq!(offsets, vec![0, 3, 4, 4, 4, 4, 4, 4, 5]);
        // `CSR`: cell 0 is `[0, 3)`, cell 1 is `[3, 4)`, cell 7 is `[4, 5)`.
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap(), 5);
    }

    #[test]
    fn twin_matches_the_golden_exclusive_prefix_sum() {
        let counts = [2u32, 0, 5, 1, 0, 3, 0, 0, 4, 7];
        let twin = dispatch_flip_cell_scan(
            &counts,
            FlipScanParams {
                cell_count: counts.len() as u32,
            },
        );
        let golden = exclusive_prefix_sum(&counts);
        assert_eq!(twin, golden);
        // Not vacuous: the running total must actually climb.
        assert!(twin.last() > twin.first());
    }

    #[test]
    fn mis_sized_or_zero_cell_is_well_formed_zeros() {
        // Buffer shorter than the declared cell count: skip, return zeros.
        let counts = [1u32, 2, 3];
        let offsets = dispatch_flip_cell_scan(&counts, FlipScanParams { cell_count: 8 });
        assert_eq!(offsets, vec![0u32; 9]);

        // Zero-cell grid: the single-element `[0]` the golden produces for an
        // empty input, no panic.
        let empty = dispatch_flip_cell_scan(&[], FlipScanParams { cell_count: 0 });
        assert_eq!(empty, vec![0u32]);
        assert_eq!(empty, exclusive_prefix_sum(&[]));
    }

    #[test]
    fn total_offset_equals_in_grid_particle_count_and_is_deterministic() {
        let counts = [4u32, 0, 1, 9, 0, 2];
        let params = FlipScanParams {
            cell_count: counts.len() as u32,
        };
        let a = dispatch_flip_cell_scan(&counts, params);
        let b = dispatch_flip_cell_scan(&counts, params);
        assert_eq!(a, b);
        // `offsets[last]` is the sum of all in-grid particles.
        assert_eq!(*a.last().unwrap(), counts.iter().sum::<u32>());
    }

    #[test]
    fn addition_saturates_rather_than_wrapping() {
        // Two near-max bins would overflow a wrapping add; the scan must clamp.
        let counts = [u32::MAX, 5];
        let offsets = dispatch_flip_cell_scan(
            &counts,
            FlipScanParams {
                cell_count: counts.len() as u32,
            },
        );
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], u32::MAX);
        assert_eq!(offsets[2], u32::MAX);
        assert_eq!(offsets, exclusive_prefix_sum(&counts));
    }
}
