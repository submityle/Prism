//! `GPU` compute twin of the `FLIP`/`APIC` cell histogram, pass 1 of 3 in the
//! particle counting-sort chain (histogram -> exclusive scan -> scatter).
//!
//! Compute twin of the `CPU` golden reference
//! `prism_render_architecture::water::flip_sort::cell_histogram`. The sandbox
//! has no `GPU`, so this module is validated by `naga`/`wesl` compilation and
//! by its bit-exact `CPU` twin; it is pure integer bookkeeping — no floats
//! beyond the cell-coordinate division, no `f32` equality test, no AI/ML.
//!
//! The entry-point name below is byte-identical to the one the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits (`water_flip_cell_histogram`), so the pipeline cache keys
//! resolve against this module. It is a linear per-particle pass dispatched at
//! `@workgroup_size(64)` over the `Particle` domain, binding one read storage
//! buffer (`positions_in`), one atomic read-write storage buffer (`counts`),
//! and one uniform (`params`).
//!
//! ## What it does
//!
//! One invocation per particle resolves the particle's pressure-cell via the
//! same row-major `MacCellGrid` arithmetic the `CPU` golden uses, then does one
//! `atomicAdd(&counts[flat], 1u)`. Out-of-grid particles (negative local
//! coordinate, beyond extent, or a degenerate `dx`) are skipped, never indexed
//! out of bounds. The atomic makes the result independent of thread order, so
//! the histogram is a deterministic function of the input positions and matches
//! the golden `cell_histogram`, which walks particles in index order. The next
//! pass (`water_flip_cell_scan`) turns these counts into `CSR` write cursors.

/// Near-zero magnitude floor (mirrors the `CPU` `EPS`), guarding a degenerate
/// cell edge `dx`.
const FLIP_HIST_EPS: f32 = 1.0e-6;

/// Row-major pressure-cell grid of a `FLIP`/`APIC` `MAC` domain plus the
/// per-dispatch particle count. Mirrors the `CPU` `MacCellGrid`; scalar-only so
/// it packs cleanly into a `uniform` buffer.
struct FlipHistogramParams {
    /// Minimum corner of the domain, world space.
    grid_origin: vec3<f32>,
    /// Cell edge length `dx` (`> 0`).
    dx: f32,
    /// Cell count along x (`>= 1`).
    nx: u32,
    /// Cell count along y (`>= 1`).
    ny: u32,
    /// Cell count along z (`>= 1`).
    nz: u32,
    /// Number of particles bounding the per-particle dispatch.
    particle_count: u32,
}

// Particle world positions: `xyz` = position, `w` = carried scalar (ignored).
@group(0) @binding(0) var<storage, read> positions_in: array<vec4<f32>>;
// Per-cell particle count, row-major, length `nx * ny * nz`. Atomic because
// many particles land in the same cell concurrently.
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
// Grid description + particle count.
@group(0) @binding(2) var<uniform> params: FlipHistogramParams;

/// A resolved cell coordinate plus a validity flag (no `Option` in `WESL`).
struct FlipCell {
    cx: u32,
    cy: u32,
    cz: u32,
    valid: bool,
}

/// Integer cell coordinate of a world position, or `valid = false` when it
/// falls outside the grid (negative local, beyond extent, or degenerate `dx`).
/// Mirrors the `CPU` `MacCellGrid::cell_coord`.
fn flip_cell_coord(p: vec3<f32>) -> FlipCell {
    var out: FlipCell;
    out.cx = 0u;
    out.cy = 0u;
    out.cz = 0u;
    out.valid = false;
    if params.dx <= FLIP_HIST_EPS {
        return out;
    }
    let local = p - params.grid_origin;
    if local.x < 0.0 || local.y < 0.0 || local.z < 0.0 {
        return out;
    }
    let cx = u32(local.x / params.dx);
    let cy = u32(local.y / params.dx);
    let cz = u32(local.z / params.dx);
    if cx >= params.nx || cy >= params.ny || cz >= params.nz {
        return out;
    }
    out.cx = cx;
    out.cy = cy;
    out.cz = cz;
    out.valid = true;
    return out;
}

/// Row-major flat cell index. Mirrors the `CPU` `MacCellGrid::flat_index`.
fn flip_flat_index(cx: u32, cy: u32, cz: u32) -> u32 {
    return cz * params.nx * params.ny + cy * params.nx + cx;
}

/// Cell histogram: one invocation per particle adds one to its cell bin.
///
/// Out-of-range invocations and out-of-grid particles are skipped. The host
/// zeroes `counts` before dispatch; after this pass `counts[c]` is the number
/// of in-grid particles whose centre falls in cell `c`.
@compute @workgroup_size(64)
fn water_flip_cell_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }
    let cell = flip_cell_coord(positions_in[i].xyz);
    if !cell.valid {
        return;
    }
    let flat = flip_flat_index(cell.cx, cell.cy, cell.cz);
    atomicAdd(&counts[flat], 1u);
}
