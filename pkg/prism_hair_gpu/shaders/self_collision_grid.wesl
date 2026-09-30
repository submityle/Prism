//! `wgpu` compute twin of Prism's uniform-grid self-collision neighbor gather
//! (`prism_render_architecture::hair::self_collision_grid`).
//!
//! The `CPU` golden reifies the strand self-collision spatial hash as a
//! reusable `UniformGrid` and flattens it into a `GPU`-uploadable
//! compressed-sparse-row form (`GridCsr`): a lexicographically sorted
//! `cell_keys` table, a `cell_starts` prefix-sum offset array, and a flat
//! `indices` pool that concatenates every cell's ascending particle indices
//! (design §6.2/§9 自碰撞近似, `TressFX` 4 style). Its `neighbors(cell)` query
//! is the spatial lookup any `hair_self_collision.wesl` pass actually needs:
//! the ascending union of every particle index in the 27 cells around a query
//! cell. This kernel is the on-device twin of that query.
//!
//! One invocation owns one particle `p`: it reads `p`'s precomputed query cell
//! (hashed on the host by the golden `grid_cell_of`, never recomputed from a
//! float here so there is no cast to disagree on), iterates the 27-cell
//! neighborhood, and for each neighbor cell **binary-searches** the sorted
//! `cell_keys` table. On a hit it copies that cell's bucket
//! (`indices[cell_starts[k] .. cell_starts[k + 1]]`) into `p`'s own disjoint
//! output slice, then **insertion-sorts** the gathered slice ascending. Because
//! every particle sits in exactly one cell, the 27-cell union has no
//! duplicates, so the sorted slice is bit-identical to the golden's
//! `sort_unstable` on the same union.
//!
//! Bit-faithfulness: the kernel only compares integer cell keys and copies
//! `u32` particle indices — no arithmetic on the payload, no transcendental —
//! so a passing parity test proves the routing (which indices land in which
//! particle's neighbor list, and in what order) matches the golden exactly, not
//! merely to a tolerance. Each thread writes only its own disjoint slice, so
//! there is no cross-thread contention and no atomics.
//!
//! Portability note: the 27-cell offset add uses plain `i32` arithmetic rather
//! than the golden's `saturating_add`. The parity test constructs cells far
//! from the `i32` extremes, so `qx ± 1` never overflows; the saturating
//! boundary contract is exercised by the golden's own unit tests. The kernel
//! stays in the portable core-`WGSL` subset (no optional device feature), so it
//! runs unmodified on Metal, Vulkan and DX12.

/// Gather parameters. `particle_count` bounds the dispatch (one invocation per
/// particle); `cell_count` bounds the binary search over `cell_keys`. `16`-byte
/// scalar-packed so the uniform block is a multiple of 16.
struct HairSelfCollisionGridParams {
    particle_count: u32,
    cell_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One compacted per-particle output slice into `out_indices`: entries
/// `start .. start + count`, ascending after the per-particle sort. `count` is
/// unused by the kernel (it fills exactly `count` entries by construction) but
/// kept one-to-one with the host layout.
struct HairGridRange {
    start: u32,
    count: u32,
}

const NOT_FOUND: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: HairSelfCollisionGridParams;

// Lexicographically sorted occupied-cell coordinates, flattened three `i32`
// per key (`3*k`, `3*k+1`, `3*k+2` = x, y, z) to avoid `vec3<i32>` alignment.
@group(0) @binding(1) var<storage, read> cell_keys: array<i32>;

// Prefix-sum offsets; `cell_count + 1` entries. Bucket `k` occupies
// `indices[cell_starts[k] .. cell_starts[k + 1]]`.
@group(0) @binding(2) var<storage, read> cell_starts: array<u32>;

// Flat concatenation of every occupied cell's ascending particle indices.
@group(0) @binding(3) var<storage, read> indices: array<u32>;

// Per-particle query cell, flattened three `i32` per particle (host-hashed by
// the golden `grid_cell_of`).
@group(0) @binding(4) var<storage, read> query_cells: array<i32>;

// Per-particle compacted output ranges into `out_indices` (exclusive prefix).
@group(0) @binding(5) var<storage, read> out_ranges: array<HairGridRange>;

// Flat per-particle neighbor pool: `out_indices[start + k]` is the `k`-th
// (ascending) neighbor particle index of that particle.
@group(0) @binding(6) var<storage, read_write> out_indices: array<u32>;

// True when key `(ax, ay, az)` is lexicographically less than `(bx, by, bz)`.
fn key_less(ax: i32, ay: i32, az: i32, bx: i32, by: i32, bz: i32) -> bool {
    if ax != bx {
        return ax < bx;
    }
    if ay != by {
        return ay < by;
    }
    return az < bz;
}

// Binary-searches the sorted `cell_keys` table for `(kx, ky, kz)`, returning
// its cell index or `NOT_FOUND`.
fn find_cell(kx: i32, ky: i32, kz: i32) -> u32 {
    var lo = 0u;
    var hi = params.cell_count;
    while lo < hi {
        let mid = (lo + hi) / 2u;
        let base = 3u * mid;
        if key_less(cell_keys[base], cell_keys[base + 1u], cell_keys[base + 2u], kx, ky, kz) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if lo < params.cell_count {
        let base = 3u * lo;
        if cell_keys[base] == kx && cell_keys[base + 1u] == ky && cell_keys[base + 2u] == kz {
            return lo;
        }
    }
    return NOT_FOUND;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if p >= params.particle_count {
        return;
    }

    let qbase = 3u * p;
    let qx = query_cells[qbase];
    let qy = query_cells[qbase + 1u];
    let qz = query_cells[qbase + 2u];

    let out_start = out_ranges[p].start;
    var local = 0u;

    // Gather the 27-cell neighborhood via binary search into the sorted cell
    // table; copy each hit bucket into this particle's disjoint slice.
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let cell = find_cell(qx + dx, qy + dy, qz + dz);
                if cell != NOT_FOUND {
                    let bstart = cell_starts[cell];
                    let bend = cell_starts[cell + 1u];
                    for (var b = bstart; b < bend; b = b + 1u) {
                        out_indices[out_start + local] = indices[b];
                        local = local + 1u;
                    }
                }
            }
        }
    }

    // Insertion-sort the gathered slice ascending to match the golden's
    // `sort_unstable` on the unioned indices (particle indices are globally
    // unique, so the union holds no duplicates).
    for (var i = 1u; i < local; i = i + 1u) {
        let key = out_indices[out_start + i];
        var j = i;
        while j > 0u && out_indices[out_start + j - 1u] > key {
            out_indices[out_start + j] = out_indices[out_start + j - 1u];
            j = j - 1u;
        }
        out_indices[out_start + j] = key;
    }
}
