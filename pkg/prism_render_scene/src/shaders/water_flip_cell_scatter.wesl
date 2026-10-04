//! `GPU` compute twin of the `FLIP`/`APIC` cell scatter, pass 3 of 3 in the
//! particle counting-sort chain (histogram -> exclusive scan -> scatter).
//!
//! Compute twin of the `CPU` golden reference
//! `prism_render_architecture::water::flip_sort::counting_sort_particles` (the
//! scatter half). The sandbox has no `GPU`, so this module is validated by
//! `naga`/`wesl` compilation and by its `CPU` twin; it is pure integer
//! bookkeeping — no floats beyond the cell-coordinate division, no `f32`
//! equality test, no AI/ML.
//!
//! The entry-point name below is byte-identical to the one the host
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! contract emits (`water_flip_cell_scatter`), so the pipeline cache keys
//! resolve against this module. It is a linear per-particle pass dispatched at
//! `@workgroup_size(64)` over the `Particle` domain, binding one read storage
//! buffer (`positions_in`), one atomic read-write storage buffer (`cursors`),
//! one plain read-write storage buffer (`sorted_indices`), and one uniform
//! (`params`).
//!
//! ## What it does
//!
//! The host seeds `cursors[c]` with the exclusive-scan start of cell `c` (the
//! `offsets` produced by `water_flip_cell_scan`). One invocation per particle
//! resolves its pressure-cell via the same row-major `MacCellGrid` arithmetic
//! the histogram pass uses, then claims a slot with
//! `atomicAdd(&cursors[flat], 1u)` and writes its original index into
//! `sorted_indices[slot]`. Out-of-grid particles are skipped. Because the slot
//! is claimed atomically, every in-grid particle lands in exactly one slot
//! inside its cell's `[offset, offset + count)` range, so the result is a valid
//! counting-sort permutation grouped cell-major. The within-cell ordering of
//! concurrent writers is implementation-defined (the atomic serialises claims
//! in device-scheduling order, not index order); the `CPU` twin models the
//! serialised index-order reference, which is the stable permutation the golden
//! `counting_sort_particles` emits. This ordering freedom is harmless: the
//! downstream `MAC` `P2G`/`G2P` transfers accumulate per cell and are
//! order-independent — only the cell-major grouping matters for coherence.

/// Near-zero magnitude floor (mirrors the `CPU` `EPS`), guarding a degenerate
/// cell edge `dx`.
const FLIP_SCATTER_EPS: f32 = 1.0e-6;

/// Row-major pressure-cell grid of a `FLIP`/`APIC` `MAC` domain plus the
/// per-dispatch particle count. Mirrors the `CPU` `MacCellGrid`; scalar-only so
/// it packs cleanly into a `uniform` buffer.
struct FlipScatterParams {
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
// Per-cell write cursor, row-major, length `nx * ny * nz`. Host-seeded with the
// exclusive-scan cell offsets; atomic because many particles in one cell race
// for consecutive slots.
@group(0) @binding(1) var<storage, read_write> cursors: array<atomic<u32>>;
// Sorted original particle indices, cell-major, length equal to the in-grid
// particle total (`offsets[cell_count]`).
@group(0) @binding(2) var<storage, read_write> sorted_indices: array<u32>;
// Grid description + particle count.
@group(0) @binding(3) var<uniform> params: FlipScatterParams;

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
    if params.dx <= FLIP_SCATTER_EPS {
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

/// Cell scatter: one invocation per particle claims its cell slot and writes
/// its original index into the cell-major `sorted_indices` buffer.
///
/// Out-of-range invocations and out-of-grid particles are skipped. The host
/// seeds `cursors` with the exclusive-scan offsets before dispatch; after this
/// pass `sorted_indices` is a cell-major counting-sort permutation of the
/// in-grid particles.
@compute @workgroup_size(64)
fn water_flip_cell_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count {
        return;
    }
    let cell = flip_cell_coord(positions_in[i].xyz);
    if !cell.valid {
        return;
    }
    let flat = flip_flat_index(cell.cx, cell.cy, cell.cz);
    let slot = atomicAdd(&cursors[flat], 1u);
    if slot < arrayLength(&sorted_indices) {
        sorted_indices[slot] = i;
    }
}
