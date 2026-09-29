// GPU pressure-projection kernels: a red-black successive-over-relaxation (SOR)
// Poisson solve on the staggered MAC grid, followed by the pressure-gradient
// subtraction that removes the divergent component of the face velocities.
//
// Byte-for-byte-intent twin of the CPU golden reference in
// `src/fluid/cpu/pressure.rs`. For each fluid cell the solve relaxes the
// discrete Poisson equation `laplacian(p) = (rho / dt) * div(u)` with a
// matrix-free SOR blend; solid neighbours drop out of the stencil (no flow
// through walls) and air neighbours impose the free-surface Dirichlet
// condition `p = 0`.
//
// Red-black ordering: within one colour a cell's six face-neighbours all have
// the opposite `(i + j + k)` parity, so no two same-colour cells interact. The
// two colours are relaxed by two separate dispatches (`sor_red` then
// `sor_black`); the implicit memory barrier between passes makes the red
// writes visible to the black sweep, reproducing the sequential-per-colour
// order the CPU twin runs in place. The gradient subtraction is split into one
// dispatch per staggered axis (`subtract_u`, `subtract_v`, `subtract_w`).
//
// The three staggered face fields u, v, w are concatenated into a single array
// laid out [u | v | w]; a field's base offset plus its local flat index gives
// the global index, matching `GridDims` on the host.
//
// Provenance: the MAC pressure-projection scheme, the solid / free-surface
// boundary handling, and the red-black SOR Poisson solve follow Bridson,
// *Fluid Simulation for Computer Graphics*, and Foster & Fedkiw 2001. No
// Unreal Engine source or derived code.

struct Params {
    // x=nx, y=ny, z=nz, w=cell_count.
    dims: vec4<u32>,
    // x=u_base, y=v_base, z=w_base, w=face_total.
    bases: vec4<u32>,
    // x=rhs_scale, y=omega, z=sub_scale, w=dx.
    coeff: vec4<f32>,
};

const AIR: u32 = 0u;
const FLUID: u32 = 1u;
const SOLID: u32 = 2u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cell_types: array<u32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure: array<f32>;

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

// The raw (undivided) divergence `du + dv + dw` at cell `(i, j, k)`.
fn raw_divergence(i: u32, j: u32, k: u32) -> f32 {
    let du = velocity[u_idx(i + 1u, j, k)] - velocity[u_idx(i, j, k)];
    let dv = velocity[v_idx(i, j + 1u, k)] - velocity[v_idx(i, j, k)];
    let dw = velocity[w_idx(i, j, k + 1u)] - velocity[w_idx(i, j, k)];
    return du + dv + dw;
}

// Accumulates one neighbour into the running diagonal / off-diagonal sum,
// mirroring the CPU `accumulate` closure exactly.
fn accumulate(ti: u32, tj: u32, tk: u32, diag: ptr<function, f32>, sum: ptr<function, f32>) {
    let t = cell_types[cell_idx(ti, tj, tk)];
    if (t == SOLID) {
        return;
    }
    *diag = *diag + 1.0;
    if (t == FLUID) {
        *sum = *sum + pressure[cell_idx(ti, tj, tk)];
    }
}

// Relaxes every fluid cell whose `(i + j + k)` parity equals `parity` once.
fn relax(gid: u32, parity: u32) {
    if (gid >= params.dims.w) {
        return;
    }
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    let k = gid / (nx * ny);
    let rem = gid % (nx * ny);
    let j = rem / nx;
    let i = rem % nx;

    if (((i + j + k) & 1u) != parity) {
        return;
    }
    if (cell_types[cell_idx(i, j, k)] != FLUID) {
        return;
    }

    var diag = 0.0;
    var sum = 0.0;
    // Fixed neighbour order: -x, +x, -y, +y, -z, +z.
    if (i > 0u) {
        accumulate(i - 1u, j, k, &diag, &sum);
    }
    if (i + 1u < nx) {
        accumulate(i + 1u, j, k, &diag, &sum);
    }
    if (j > 0u) {
        accumulate(i, j - 1u, k, &diag, &sum);
    }
    if (j + 1u < ny) {
        accumulate(i, j + 1u, k, &diag, &sum);
    }
    if (k > 0u) {
        accumulate(i, j, k - 1u, &diag, &sum);
    }
    if (k + 1u < nz) {
        accumulate(i, j, k + 1u, &diag, &sum);
    }
    if (diag <= 0.0) {
        return;
    }

    let raw = raw_divergence(i, j, k);
    let solved = (sum - params.coeff.x * raw) / diag;
    let cur = pressure[cell_idx(i, j, k)];
    pressure[cell_idx(i, j, k)] = cur + params.coeff.y * (solved - cur);
}

@compute @workgroup_size(64)
fn sor_red(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid.x, 0u);
}

@compute @workgroup_size(64)
fn sor_black(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax(gid.x, 1u);
}

// u faces between cells (i-1) and (i).
@compute @workgroup_size(64)
fn subtract_u(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    // Interior faces only: 1 <= i < nx (matches the CPU `1..nx` loop).
    if (i == 0u || i >= nx) {
        return;
    }
    let left = cell_types[cell_idx(i - 1u, j, k)];
    let right = cell_types[cell_idx(i, j, k)];
    let face = u_idx(i, j, k);
    if (left == SOLID || right == SOLID) {
        velocity[face] = 0.0;
    } else if (left == FLUID || right == FLUID) {
        let grad = pressure[cell_idx(i, j, k)] - pressure[cell_idx(i - 1u, j, k)];
        velocity[face] = velocity[face] - params.coeff.z * grad / params.coeff.w;
    }
}

// v faces between cells (j-1) and (j).
@compute @workgroup_size(64)
fn subtract_v(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    if (j == 0u || j >= ny) {
        return;
    }
    let down = cell_types[cell_idx(i, j - 1u, k)];
    let up = cell_types[cell_idx(i, j, k)];
    let face = v_idx(i, j, k);
    if (down == SOLID || up == SOLID) {
        velocity[face] = 0.0;
    } else if (down == FLUID || up == FLUID) {
        let grad = pressure[cell_idx(i, j, k)] - pressure[cell_idx(i, j - 1u, k)];
        velocity[face] = velocity[face] - params.coeff.z * grad / params.coeff.w;
    }
}

// w faces between cells (k-1) and (k).
@compute @workgroup_size(64)
fn subtract_w(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    if (k == 0u || k >= nz) {
        return;
    }
    let back = cell_types[cell_idx(i, j, k - 1u)];
    let front = cell_types[cell_idx(i, j, k)];
    let face = w_idx(i, j, k);
    if (back == SOLID || front == SOLID) {
        velocity[face] = 0.0;
    } else if (back == FLUID || front == FLUID) {
        let grad = pressure[cell_idx(i, j, k)] - pressure[cell_idx(i, j, k - 1u)];
        velocity[face] = velocity[face] - params.coeff.z * grad / params.coeff.w;
    }
}
