// Per-fragment broad-phase bounds builder.
//
// Five kernels turn a classified debris point cloud into one axis-aligned box
// and one bounding sphere per fragment cell:
//
//   * `clear` runs one invocation per cell and primes the extrema accumulators
//     (min = +INT_MAX, max = -INT_MAX, squared radius = 0).
//   * `scatter_aabb` runs one invocation per point and folds the quantised
//     position into its fragment's box extents with integer atomics.
//   * `finalize_center` runs one invocation per cell, de-quantises the box, and
//     writes the box and its centre (the sphere centre).
//   * `scatter_radius` runs one invocation per point, reads its fragment's
//     centre, and folds the quantised squared distance into the radius extremum.
//   * `finalize_radius` runs one invocation per cell and writes the square root
//     of the de-quantised squared radius.
//
// The kernels mirror the `cpu_bounds_fragments` golden twin: integer extrema
// are exact and order independent, so the de-quantised box is bit-identical
// across devices and only the final radius square root can diverge by low bits.
//
// Provenance: axis-aligned extrema and a box-centred bounding sphere are
// elementary geometry and fixed-point atomic reduction is a standard GPU
// technique. No Unreal Engine source or derived code.

struct Params {
    // Number of fragment cells.
    n_cells: u32,
    // Number of query points.
    n_points: u32,
    // Padding to keep the scalar block 16-byte aligned.
    _pad0: u32,
    _pad1: u32,
    // Fixed-point scale for a quantised coordinate.
    position_scale: f32,
    // Fixed-point scale for a quantised squared distance.
    radius_sq_scale: f32,
    // Padding to a 16-byte boundary.
    _pad2: f32,
    _pad3: f32,
};

// One broad-phase proxy per fragment, packed into vec4 lanes for a clean
// readback.
struct BoundsOut {
    // xyz = box minimum corner, w unused.
    aabb_min: vec4<f32>,
    // xyz = box maximum corner, w unused.
    aabb_max: vec4<f32>,
    // xyz = sphere centre, w = sphere radius.
    sphere: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Query points; xyz = position, w unused.
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
// Owning fragment cell per point (0xFFFFFFFF marks an unassigned point).
@group(0) @binding(2) var<storage, read> cells: array<u32>;
// Fixed-point box minima, three slots per cell (cell*3 + axis).
@group(0) @binding(3) var<storage, read_write> acc_min: array<atomic<i32>>;
// Fixed-point box maxima, three slots per cell (cell*3 + axis).
@group(0) @binding(4) var<storage, read_write> acc_max: array<atomic<i32>>;
// Fixed-point squared-radius extremum, one slot per cell.
@group(0) @binding(5) var<storage, read_write> acc_radius_sq: array<atomic<i32>>;
// Finalised per-fragment box and sphere.
@group(0) @binding(6) var<storage, read_write> out_bounds: array<BoundsOut>;

const NO_CELL: u32 = 0xFFFFFFFFu;
// Sentinels mirroring the CPU twin's inverted box; `i32` max/min literals.
const INT_MAX: i32 = 2147483647;
const INT_MIN: i32 = -2147483647 - 1;

// Quantises `value * scale` to the nearest integer (ties to even) for atomic
// reduction, matching the CPU twin's `round_ties_even`.
fn quantise(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

// Maps a raw cell tag to an in-range fragment index, returning `n_cells` (an
// out-of-range marker) for the sentinel or any out-of-range index.
fn fragment_index(cell: u32) -> u32 {
    if (cell == NO_CELL || cell >= params.n_cells) {
        return params.n_cells;
    }
    return cell;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= params.n_cells) {
        return;
    }
    let base3 = cid * 3u;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        atomicStore(&acc_min[base3 + axis], INT_MAX);
        atomicStore(&acc_max[base3 + axis], INT_MIN);
    }
    atomicStore(&acc_radius_sq[cid], 0);
}

@compute @workgroup_size(64)
fn scatter_aabb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) {
        return;
    }
    let cell = fragment_index(cells[idx]);
    if (cell >= params.n_cells) {
        return;
    }

    let p = points[idx].xyz;
    let qx = quantise(p.x, params.position_scale);
    let qy = quantise(p.y, params.position_scale);
    let qz = quantise(p.z, params.position_scale);

    let base3 = cell * 3u;
    atomicMin(&acc_min[base3 + 0u], qx);
    atomicMin(&acc_min[base3 + 1u], qy);
    atomicMin(&acc_min[base3 + 2u], qz);
    atomicMax(&acc_max[base3 + 0u], qx);
    atomicMax(&acc_max[base3 + 1u], qy);
    atomicMax(&acc_max[base3 + 2u], qz);
}

@compute @workgroup_size(64)
fn finalize_center(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= params.n_cells) {
        return;
    }

    let base3 = cid * 3u;
    let min_x = atomicLoad(&acc_min[base3 + 0u]);
    // An untouched cell keeps its inverted box, so report a zeroed proxy.
    if (min_x > atomicLoad(&acc_max[base3 + 0u])) {
        out_bounds[cid].aabb_min = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out_bounds[cid].aabb_max = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out_bounds[cid].sphere = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let scale = params.position_scale;
    let lo = vec3<f32>(
        f32(min_x) / scale,
        f32(atomicLoad(&acc_min[base3 + 1u])) / scale,
        f32(atomicLoad(&acc_min[base3 + 2u])) / scale,
    );
    let hi = vec3<f32>(
        f32(atomicLoad(&acc_max[base3 + 0u])) / scale,
        f32(atomicLoad(&acc_max[base3 + 1u])) / scale,
        f32(atomicLoad(&acc_max[base3 + 2u])) / scale,
    );
    let center = (lo + hi) * 0.5;

    out_bounds[cid].aabb_min = vec4<f32>(lo, 0.0);
    out_bounds[cid].aabb_max = vec4<f32>(hi, 0.0);
    // Stash the centre now; `finalize_radius` fills the w lane with the radius.
    out_bounds[cid].sphere = vec4<f32>(center, 0.0);
}

@compute @workgroup_size(64)
fn scatter_radius(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_points) {
        return;
    }
    let cell = fragment_index(cells[idx]);
    if (cell >= params.n_cells) {
        return;
    }

    let center = out_bounds[cell].sphere.xyz;
    let d = points[idx].xyz - center;
    let q = quantise(dot(d, d), params.radius_sq_scale);
    atomicMax(&acc_radius_sq[cell], q);
}

@compute @workgroup_size(64)
fn finalize_radius(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= params.n_cells) {
        return;
    }
    // Leave zeroed proxies untouched: their extrema never advanced.
    let base3 = cid * 3u;
    if (atomicLoad(&acc_min[base3 + 0u]) > atomicLoad(&acc_max[base3 + 0u])) {
        return;
    }
    let radius_sq = max(f32(atomicLoad(&acc_radius_sq[cid])) / params.radius_sq_scale, 0.0);
    out_bounds[cid].sphere.w = sqrt(radius_sq);
}
