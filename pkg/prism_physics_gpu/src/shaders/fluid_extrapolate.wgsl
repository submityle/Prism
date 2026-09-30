// GPU free-surface velocity extrapolation: one Jacobi sweep per dispatch that
// fills every still-unknown face with the mean of its already-known
// 6-neighbours, growing the known band outward one cell per sweep.
//
// Byte-for-byte-intent twin of the CPU golden reference in
// `src/fluid/cpu/extrapolate.rs`. Each sweep reads the `src` field and `known`
// mask and writes the `dst` field and mask, so no face relaxed in this sweep
// feeds another this sweep — the exact Jacobi structure of the reference. The
// host ping-pongs the two (field, known) buffer pairs across sweeps.
//
// A known face copies its value forward unchanged and stays known. An unknown
// face with at least one known neighbour takes the neighbour mean and becomes
// known; an unknown face with no known neighbour copies its (zero) value
// forward and stays unknown.
//
// Provenance: iterative velocity extrapolation from the known band into the air
// region is a standard free-surface technique (Bridson, *Fluid Simulation for
// Computer Graphics*; Zhu & Bridson 2005). No Unreal Engine source or derived
// code.

struct Params {
    // x=dx, y=dy, z=dz, w=count.
    dims: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field_src: array<f32>;
@group(0) @binding(2) var<storage, read> known_src: array<u32>;
@group(0) @binding(3) var<storage, read_write> field_dst: array<f32>;
@group(0) @binding(4) var<storage, read_write> known_dst: array<u32>;

fn flat(i: u32, j: u32, k: u32) -> u32 {
    return i + params.dims.x * (j + params.dims.y * k);
}

// Adds neighbour `(ti, tj, tk)` to the running mean when it is known.
fn take(ti: u32, tj: u32, tk: u32, acc: ptr<function, f32>, cnt: ptr<function, f32>) {
    let id = flat(ti, tj, tk);
    if (known_src[id] != 0u) {
        *acc = *acc + field_src[id];
        *cnt = *cnt + 1.0;
    }
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.dims.w) {
        return;
    }
    let dx = params.dims.x;
    let dy = params.dims.y;
    let dz = params.dims.z;
    let k = id / (dx * dy);
    let rem = id % (dx * dy);
    let j = rem / dx;
    let i = rem % dx;

    // A known face is carried forward unchanged.
    if (known_src[id] != 0u) {
        field_dst[id] = field_src[id];
        known_dst[id] = 1u;
        return;
    }

    var acc = 0.0;
    var cnt = 0.0;
    // Fixed neighbour order: -x, +x, -y, +y, -z, +z.
    if (i > 0u) {
        take(i - 1u, j, k, &acc, &cnt);
    }
    if (i + 1u < dx) {
        take(i + 1u, j, k, &acc, &cnt);
    }
    if (j > 0u) {
        take(i, j - 1u, k, &acc, &cnt);
    }
    if (j + 1u < dy) {
        take(i, j + 1u, k, &acc, &cnt);
    }
    if (k > 0u) {
        take(i, j, k - 1u, &acc, &cnt);
    }
    if (k + 1u < dz) {
        take(i, j, k + 1u, &acc, &cnt);
    }

    if (cnt > 0.0) {
        field_dst[id] = acc / cnt;
        known_dst[id] = 1u;
    } else {
        field_dst[id] = field_src[id];
        known_dst[id] = 0u;
    }
}
