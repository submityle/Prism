//! `GPU` compute twin of the `FLIP`/`APIC` cell-offset scan, pass 2 of 3 in the
//! particle counting-sort chain (histogram -> exclusive scan -> scatter).
//!
//! Compute twin of the `CPU` golden reference
//! `prism_render_architecture::water::flip_sort::exclusive_prefix_sum`. The
//! sandbox has no `GPU`, so this module is validated by `naga`/`wesl`
//! compilation and by its bit-exact `CPU` twin; it is pure integer bookkeeping
//! — no floats, no `f32` equality test, no AI/ML.
//!
//! The entry-point name below is byte-identical to the one the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits (`water_flip_cell_scan`), so the pipeline cache keys resolve
//! against this module. It is a single-workgroup pass dispatched at
//! `@workgroup_size(256)` over the `Cells` domain, binding one read storage
//! buffer (`counts`, one `u32` per cell), one read-write storage buffer
//! (`offsets`, `cells + 1` `u32`), and one uniform (`params`).
//!
//! ## What it does
//!
//! Converts the per-cell histogram into `CSR` cell offsets: `offsets[0] = 0`
//! and `offsets[c + 1] = sum(counts[0..=c])`, so `offsets[last]` is the number
//! of in-grid particles and `[offsets[c], offsets[c + 1])` is the slice the
//! scatter pass writes cell `c`'s particles into. One workgroup grid-strides
//! over the flat cell list in 256-wide tiles: each tile runs a work-efficient
//! `Hillis-Steele` inclusive scan in workgroup memory, the result is biased by
//! the running `carry` from earlier tiles, and the carry is advanced by the
//! tile total before the next tile. A single group keeps the prefix exact
//! across tile boundaries without a second global pass. The next pass
//! (`water_flip_cell_scatter`) seeds a per-cell write cursor from these
//! offsets.

/// Workgroup width of the single-group scan; also the tile stride.
const FLIP_SCAN_GROUP: u32 = 256u;

/// Grid description carrying the flat cell count the scan sweeps. Mirrors the
/// `CPU` `FlipScanParams`; scalar-only so it packs cleanly into a `uniform`.
struct FlipScanParams {
    /// Number of pressure cells (`nx * ny * nz`); `offsets` holds `cell_count
    /// + 1` entries.
    cell_count: u32,
}

// Per-cell particle counts (histogram output), row-major, length `cell_count`.
@group(0) @binding(0) var<storage, read> counts: array<u32>;
// `CSR` cell offsets, length `cell_count + 1`; `offsets[last]` is the in-grid
// particle total.
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
// Grid description (flat cell count).
@group(0) @binding(2) var<uniform> params: FlipScanParams;

// Scratch tile for the per-tile inclusive scan.
var<workgroup> tile: array<u32, 256u>;
// Running exclusive total of all cells scanned in earlier tiles.
var<workgroup> carry: u32;

/// Exclusive-prefix-sum scan: one workgroup turns `counts` into `CSR`
/// `offsets`. Launched as a single group; the host zeroes nothing (every
/// `offsets` entry is written).
@compute @workgroup_size(256)
fn water_flip_cell_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.cell_count;

    if tid == 0u {
        carry = 0u;
        offsets[0] = 0u;
    }
    workgroupBarrier();

    var base: u32 = 0u;
    loop {
        if base >= n {
            break;
        }

        let idx = base + tid;
        var v: u32 = 0u;
        if idx < n {
            v = counts[idx];
        }
        tile[tid] = v;
        workgroupBarrier();

        // Work-efficient inclusive scan over the tile (`Hillis-Steele`).
        var step: u32 = 1u;
        loop {
            if step >= FLIP_SCAN_GROUP {
                break;
            }
            var add: u32 = 0u;
            if tid >= step {
                add = tile[tid - step];
            }
            workgroupBarrier();
            tile[tid] = tile[tid] + add;
            workgroupBarrier();
            step = step * 2u;
        }

        // `tile[tid]` now holds the inclusive prefix within the tile; bias by
        // the carry so `offsets[idx + 1]` is the global inclusive prefix.
        if idx < n {
            offsets[idx + 1u] = carry + tile[tid];
        }
        workgroupBarrier();

        // Out-of-range lanes loaded zero, so lane 255 holds the whole-tile
        // total; fold it into the carry for the next tile.
        if tid == 0u {
            carry = carry + tile[FLIP_SCAN_GROUP - 1u];
        }
        workgroupBarrier();

        base = base + FLIP_SCAN_GROUP;
    }
}
