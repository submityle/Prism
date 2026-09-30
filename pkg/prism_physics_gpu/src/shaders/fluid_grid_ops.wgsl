// GPU per-face grid operators that run between the particle-to-grid scatter and
// the pressure projection: the body-force integration (`add_gravity`) and the
// no-through-flow solid boundary condition (`enforce_u` / `enforce_v` /
// `enforce_w`).
//
// Byte-for-byte-intent twin of the CPU golden reference in
// `src/fluid/cpu/grid_ops.rs`. Every write is either a single add of a
// host-computed constant `g * dt` or a store of `0.0`, so both engines land on
// identical values and the parity test asserts exact equality.
//
// The three staggered face fields u, v, w are concatenated into one array laid
// out [u | v | w]; a field's base offset plus its local flat index gives the
// global index, matching `GridDims` on the host.
//
// Solid enforcement is expressed as a per-face *gather*: a face is zeroed when
// either of its two adjacent cells is solid. This is equivalent to the CPU
// scatter (which iterates solid cells and zeroes their six bounding faces)
// because the only value ever written is zero, so the operation is idempotent
// and order-independent. The gather form has no write-after-write hazard across
// invocations, which the scatter form would.
//
// Provenance: explicit body-force integration on a staggered MAC grid and the
// solid no-through-flow boundary are standard CFD constructs (Harlow & Welch
// 1965; Bridson, *Fluid Simulation for Computer Graphics*). No Unreal Engine
// source or derived code.

struct Params {
    // x=nx, y=ny, z=nz, w=cell_count.
    dims: vec4<u32>,
    // x=u_base, y=v_base, z=w_base, w=face_total.
    bases: vec4<u32>,
    // x/y/z = gravity * dt (per-axis face increment), w unused.
    dv: vec4<f32>,
};

const SOLID: u32 = 2u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_types: array<u32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;

fn cell_idx(i: u32, j: u32, k: u32) -> u32 {
    return i + params.dims.x * (j + params.dims.y * k);
}

fn u_idx(i: u32, j: u32, k: u32) -> u32 {
    return params.bases.x + i + (params.dims.x + 1u) * (j + params.dims.y * k);
}

fn v_idx(i: u32, j: u32, k: u32) -> u32 {
    return params.bases.y + i + params.dims.x * (j + (params.dims.y + 1u) * k);
}

fn w_idx(i: u32, j: u32, k: u32) -> u32 {
    return params.bases.z + i + params.dims.x * (j + params.dims.y * k);
}

fn is_solid(i: u32, j: u32, k: u32) -> bool {
    return cell_types[cell_idx(i, j, k)] == SOLID;
}

// Adds `g * dt` to every concatenated face: the x increment to the u region
// (indices [0, v_base)), the y increment to the v region ([v_base, w_base)),
// and the z increment to the w region ([w_base, face_total)).
@compute @workgroup_size(64)
fn add_gravity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= params.bases.w) {
        return;
    }
    if (f < params.bases.y) {
        velocity[f] = velocity[f] + params.dv.x;
    } else if (f < params.bases.z) {
        velocity[f] = velocity[f] + params.dv.y;
    } else {
        velocity[f] = velocity[f] + params.dv.z;
    }
}

// u faces have dims (nx + 1, ny, nz). Face (i, j, k) separates cells (i-1) and
// (i); it is zeroed when either exists and is solid.
@compute @workgroup_size(64)
fn enforce_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    let count = (nx + 1u) * ny * nz;
    if (gid.x >= count) {
        return;
    }
    let k = gid.x / ((nx + 1u) * ny);
    let rem = gid.x % ((nx + 1u) * ny);
    let j = rem / (nx + 1u);
    let i = rem % (nx + 1u);
    var solid = false;
    if (i < nx && is_solid(i, j, k)) {
        solid = true;
    }
    if (i > 0u && is_solid(i - 1u, j, k)) {
        solid = true;
    }
    if (solid) {
        velocity[u_idx(i, j, k)] = 0.0;
    }
}

// v faces have dims (nx, ny + 1, nz). Face (i, j, k) separates cells (j-1) and
// (j).
@compute @workgroup_size(64)
fn enforce_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    let count = nx * (ny + 1u) * nz;
    if (gid.x >= count) {
        return;
    }
    let k = gid.x / (nx * (ny + 1u));
    let rem = gid.x % (nx * (ny + 1u));
    let j = rem / nx;
    let i = rem % nx;
    var solid = false;
    if (j < ny && is_solid(i, j, k)) {
        solid = true;
    }
    if (j > 0u && is_solid(i, j - 1u, k)) {
        solid = true;
    }
    if (solid) {
        velocity[v_idx(i, j, k)] = 0.0;
    }
}

// w faces have dims (nx, ny, nz + 1). Face (i, j, k) separates cells (k-1) and
// (k).
@compute @workgroup_size(64)
fn enforce_w(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    let count = nx * ny * (nz + 1u);
    if (gid.x >= count) {
        return;
    }
    let k = gid.x / (nx * ny);
    let rem = gid.x % (nx * ny);
    let j = rem / nx;
    let i = rem % nx;
    var solid = false;
    if (k < nz && is_solid(i, j, k)) {
        solid = true;
    }
    if (k > 0u && is_solid(i, j, k - 1u)) {
        solid = true;
    }
    if (solid) {
        velocity[w_idx(i, j, k)] = 0.0;
    }
}
