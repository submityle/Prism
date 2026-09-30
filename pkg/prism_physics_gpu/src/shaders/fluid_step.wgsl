// Fused full-step FLIP/APIC fluid kernels: every stage of one solver step in a
// single WGSL module over one resident set of grid buffers, so the host runs a
// complete step (P2G -> save -> gravity -> solids -> pressure projection ->
// extrapolation -> G2P -> advection) in one command submission with only the
// initial particle upload and the final particle read-back crossing the bus.
//
// This is the device twin of the CPU golden `fluid_step` in
// `src/fluid/cpu/step.rs`; each entry point reproduces the arithmetic of the
// corresponding standalone kernel (`fluid_transfer.wgsl`, `fluid_grid_ops.wgsl`,
// `fluid_pressure.wgsl`, `fluid_extrapolate.wgsl`, `fluid_advect.wgsl`) so the
// composed step matches the reference within the same tight tolerance the
// per-stage parity tests already establish.
//
// The three staggered face fields u, v, w are concatenated into single arrays
// laid out [u | v | w]; a field's base offset plus its local flat index gives
// the global index, matching `GridDims` on the host.
//
// Provenance: the P2G/G2P PIC/FLIP transfer (Zhu & Bridson 2005; Bridson), the
// MAC body-force and solid boundary operators (Harlow & Welch 1965; Bridson),
// the red-black SOR pressure projection (Bridson; Foster & Fedkiw 2001), the
// free-surface velocity extrapolation, and the second-order Runge-Kutta
// advection are all standard published techniques. No Unreal Engine source or
// derived code.

struct StepParams {
    // xyz = grid origin, w = cell size dx.
    origin_dx: vec4<f32>,
    // x=nx, y=ny, z=nz, w=particle_count.
    dims: vec4<u32>,
    // x=u_base, y=v_base, z=w_base, w=face_total.
    bases: vec4<u32>,
    // x=cell_count, yzw unused.
    cell: vec4<u32>,
    // x/y/z = gravity * dt (per-axis face increment), w = dt.
    grav_dt: vec4<f32>,
    // x=rhs_scale, y=omega, z=sub_scale, w=dx.
    coeff: vec4<f32>,
    // x=flip_blend, yzw unused.
    blend: vec4<f32>,
};

const MOMENTUM_SCALE: f32 = 65536.0;
const WEIGHT_SCALE: f32 = 65536.0;

const AIR: u32 = 0u;
const FLUID: u32 = 1u;
const SOLID: u32 = 2u;

@group(0) @binding(0) var<uniform> params: StepParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> positions_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> momentum: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> weight: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(8) var<storage, read_write> velocity_scratch: array<f32>;
@group(0) @binding(9) var<storage, read_write> saved: array<f32>;
@group(0) @binding(10) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(11) var<storage, read> cell_types: array<u32>;
@group(0) @binding(12) var<storage, read_write> known: array<u32>;
@group(0) @binding(13) var<storage, read_write> known_scratch: array<u32>;

// --- Shared helpers ---------------------------------------------------------

struct Stencil {
    lo: i32,
    hi: i32,
    frac: f32,
};

// Mirrors `axis_stencil` in `src/fluid/cpu/stencil.rs` exactly.
fn axis_stencil(coord: f32, nodes: i32) -> Stencil {
    let fi = floor(coord);
    let i0 = i32(fi);
    let frac = coord - fi;
    let maxn = nodes - 1;
    let lo = clamp(i0, 0, maxn);
    let hi = clamp(i0 + 1, 0, maxn);
    var s: Stencil;
    s.lo = lo;
    s.hi = hi;
    s.frac = clamp(frac, 0.0, 1.0);
    return s;
}

fn quantise(value: f32, scale: f32) -> i32 {
    return i32(round(value * scale));
}

fn cell_space(p: vec3<f32>) -> vec3<f32> {
    return (p - params.origin_dx.xyz) / params.origin_dx.w;
}

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

// --- Stage 1: particle-to-grid scatter --------------------------------------

fn scatter_axis(
    c: vec3<f32>,
    off: vec3<f32>,
    dimx: i32,
    dimy: i32,
    dimz: i32,
    base: u32,
    value: f32,
) {
    let sx = axis_stencil(c.x - off.x, dimx);
    let sy = axis_stencil(c.y - off.y, dimy);
    let sz = axis_stencil(c.z - off.z, dimz);
    let ix = array<i32, 2>(sx.lo, sx.hi);
    let iy = array<i32, 2>(sy.lo, sy.hi);
    let iz = array<i32, 2>(sz.lo, sz.hi);
    let wx = array<f32, 2>(1.0 - sx.frac, sx.frac);
    let wy = array<f32, 2>(1.0 - sy.frac, sy.frac);
    let wz = array<f32, 2>(1.0 - sz.frac, sz.frac);
    for (var a = 0; a < 2; a = a + 1) {
        for (var b = 0; b < 2; b = b + 1) {
            for (var d = 0; d < 2; d = d + 1) {
                let w = wx[a] * wy[b] * wz[d];
                if (w <= 0.0) {
                    continue;
                }
                let local = ix[a] + dimx * (iy[b] + dimy * iz[d]);
                let idx = base + u32(local);
                atomicAdd(&momentum[idx], quantise(w * value, MOMENTUM_SCALE));
                atomicAdd(&weight[idx], quantise(w, WEIGHT_SCALE));
            }
        }
    }
}

@compute @workgroup_size(64)
fn p2g_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let c = cell_space(positions[p].xyz);
    let vel = velocities_in[p].xyz;
    let nx = i32(params.dims.x);
    let ny = i32(params.dims.y);
    let nz = i32(params.dims.z);
    scatter_axis(c, vec3<f32>(0.0, 0.5, 0.5), nx + 1, ny, nz, params.bases.x, vel.x);
    scatter_axis(c, vec3<f32>(0.5, 0.0, 0.5), nx, ny + 1, nz, params.bases.y, vel.y);
    scatter_axis(c, vec3<f32>(0.5, 0.5, 0.0), nx, ny, nz + 1, params.bases.z, vel.z);
}

// --- Stage 2: normalise -----------------------------------------------------

@compute @workgroup_size(64)
fn normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.bases.w) {
        return;
    }
    let w = atomicLoad(&weight[i]);
    if (w > 0) {
        let m = atomicLoad(&momentum[i]);
        let mf = f32(m) / MOMENTUM_SCALE;
        let wf = f32(w) / WEIGHT_SCALE;
        velocity[i] = mf / wf;
    } else {
        velocity[i] = 0.0;
    }
}

// --- Stage 3: save the pre-projection field ---------------------------------

@compute @workgroup_size(64)
fn save_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.bases.w) {
        return;
    }
    saved[i] = velocity[i];
}

// --- Stage 4: body force ----------------------------------------------------

@compute @workgroup_size(64)
fn add_gravity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= params.bases.w) {
        return;
    }
    if (f < params.bases.y) {
        velocity[f] = velocity[f] + params.grav_dt.x;
    } else if (f < params.bases.z) {
        velocity[f] = velocity[f] + params.grav_dt.y;
    } else {
        velocity[f] = velocity[f] + params.grav_dt.z;
    }
}

// --- Stage 5: solid no-through-flow -----------------------------------------

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

// --- Stage 6: red-black SOR pressure solve ----------------------------------

fn raw_divergence(i: u32, j: u32, k: u32) -> f32 {
    let du = velocity[u_idx(i + 1u, j, k)] - velocity[u_idx(i, j, k)];
    let dv = velocity[v_idx(i, j + 1u, k)] - velocity[v_idx(i, j, k)];
    let dw = velocity[w_idx(i, j, k + 1u)] - velocity[w_idx(i, j, k)];
    return du + dv + dw;
}

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

fn relax(gid: u32, parity: u32) {
    if (gid >= params.cell.x) {
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

// --- Stage 7: free-surface extrapolation ------------------------------------

// The known band is seeded from the transfer weight accumulator: a face is
// known when its accumulated weight is positive.
@compute @workgroup_size(64)
fn init_known(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.bases.w) {
        return;
    }
    if (atomicLoad(&weight[i]) > 0) {
        known[i] = 1u;
    } else {
        known[i] = 0u;
    }
}

// Resolves the axis a concatenated face index belongs to, returning its local
// index and the axis node counts.
struct AxisFace {
    local: u32,
    dx: u32,
    dy: u32,
    dz: u32,
    base: u32,
};

fn resolve_axis(f: u32) -> AxisFace {
    let nx = params.dims.x;
    let ny = params.dims.y;
    let nz = params.dims.z;
    var a: AxisFace;
    if (f < params.bases.y) {
        a.local = f - params.bases.x;
        a.dx = nx + 1u;
        a.dy = ny;
        a.dz = nz;
        a.base = params.bases.x;
    } else if (f < params.bases.z) {
        a.local = f - params.bases.y;
        a.dx = nx;
        a.dy = ny + 1u;
        a.dz = nz;
        a.base = params.bases.y;
    } else {
        a.local = f - params.bases.z;
        a.dx = nx;
        a.dy = ny;
        a.dz = nz + 1u;
        a.base = params.bases.z;
    }
    return a;
}

fn extrap_take(base: u32, id: u32, acc: ptr<function, f32>, cnt: ptr<function, f32>) {
    let g = base + id;
    if (known_scratch[g] != 0u) {
        *acc = *acc + velocity_scratch[g];
        *cnt = *cnt + 1.0;
    }
}

// One Jacobi sweep. Reads the `*_scratch` (source) buffers, writes the primary
// `velocity` / `known` (destination) buffers, so the host copies the destination
// back into the source between sweeps for the next iteration.
@compute @workgroup_size(64)
fn extrap_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let f = gid.x;
    if (f >= params.bases.w) {
        return;
    }
    if (known_scratch[f] != 0u) {
        velocity[f] = velocity_scratch[f];
        known[f] = 1u;
        return;
    }
    let a = resolve_axis(f);
    let k = a.local / (a.dx * a.dy);
    let rem = a.local % (a.dx * a.dy);
    let j = rem / a.dx;
    let i = rem % a.dx;

    var acc = 0.0;
    var cnt = 0.0;
    if (i > 0u) {
        extrap_take(a.base, a.local - 1u, &acc, &cnt);
    }
    if (i + 1u < a.dx) {
        extrap_take(a.base, a.local + 1u, &acc, &cnt);
    }
    if (j > 0u) {
        extrap_take(a.base, a.local - a.dx, &acc, &cnt);
    }
    if (j + 1u < a.dy) {
        extrap_take(a.base, a.local + a.dx, &acc, &cnt);
    }
    if (k > 0u) {
        extrap_take(a.base, a.local - a.dx * a.dy, &acc, &cnt);
    }
    if (k + 1u < a.dz) {
        extrap_take(a.base, a.local + a.dx * a.dy, &acc, &cnt);
    }

    if (cnt > 0.0) {
        velocity[f] = acc / cnt;
        known[f] = 1u;
    } else {
        velocity[f] = velocity_scratch[f];
        known[f] = 0u;
    }
}

// --- Stage 8: grid-to-particle gather ---------------------------------------

fn sample_velocity_axis(
    c: vec3<f32>,
    off: vec3<f32>,
    dimx: i32,
    dimy: i32,
    dimz: i32,
    base: u32,
) -> f32 {
    let sx = axis_stencil(c.x - off.x, dimx);
    let sy = axis_stencil(c.y - off.y, dimy);
    let sz = axis_stencil(c.z - off.z, dimz);
    let ix = array<i32, 2>(sx.lo, sx.hi);
    let iy = array<i32, 2>(sy.lo, sy.hi);
    let iz = array<i32, 2>(sz.lo, sz.hi);
    let wx = array<f32, 2>(1.0 - sx.frac, sx.frac);
    let wy = array<f32, 2>(1.0 - sy.frac, sy.frac);
    let wz = array<f32, 2>(1.0 - sz.frac, sz.frac);
    var acc = 0.0;
    for (var a = 0; a < 2; a = a + 1) {
        for (var b = 0; b < 2; b = b + 1) {
            for (var d = 0; d < 2; d = d + 1) {
                let w = wx[a] * wy[b] * wz[d];
                let local = ix[a] + dimx * (iy[b] + dimy * iz[d]);
                acc = acc + w * velocity[base + u32(local)];
            }
        }
    }
    return acc;
}

fn sample_saved_axis(
    c: vec3<f32>,
    off: vec3<f32>,
    dimx: i32,
    dimy: i32,
    dimz: i32,
    base: u32,
) -> f32 {
    let sx = axis_stencil(c.x - off.x, dimx);
    let sy = axis_stencil(c.y - off.y, dimy);
    let sz = axis_stencil(c.z - off.z, dimz);
    let ix = array<i32, 2>(sx.lo, sx.hi);
    let iy = array<i32, 2>(sy.lo, sy.hi);
    let iz = array<i32, 2>(sz.lo, sz.hi);
    let wx = array<f32, 2>(1.0 - sx.frac, sx.frac);
    let wy = array<f32, 2>(1.0 - sy.frac, sy.frac);
    let wz = array<f32, 2>(1.0 - sz.frac, sz.frac);
    var acc = 0.0;
    for (var a = 0; a < 2; a = a + 1) {
        for (var b = 0; b < 2; b = b + 1) {
            for (var d = 0; d < 2; d = d + 1) {
                let w = wx[a] * wy[b] * wz[d];
                let local = ix[a] + dimx * (iy[b] + dimy * iz[d]);
                acc = acc + w * saved[base + u32(local)];
            }
        }
    }
    return acc;
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let c = cell_space(positions[p].xyz);
    let nx = i32(params.dims.x);
    let ny = i32(params.dims.y);
    let nz = i32(params.dims.z);
    let pic = vec3<f32>(
        sample_velocity_axis(c, vec3<f32>(0.0, 0.5, 0.5), nx + 1, ny, nz, params.bases.x),
        sample_velocity_axis(c, vec3<f32>(0.5, 0.0, 0.5), nx, ny + 1, nz, params.bases.y),
        sample_velocity_axis(c, vec3<f32>(0.5, 0.5, 0.0), nx, ny, nz + 1, params.bases.z),
    );
    let sav = vec3<f32>(
        sample_saved_axis(c, vec3<f32>(0.0, 0.5, 0.5), nx + 1, ny, nz, params.bases.x),
        sample_saved_axis(c, vec3<f32>(0.5, 0.0, 0.5), nx, ny + 1, nz, params.bases.y),
        sample_saved_axis(c, vec3<f32>(0.5, 0.5, 0.0), nx, ny, nz + 1, params.bases.z),
    );
    let old = velocities_in[p].xyz;
    let blend = clamp(params.blend.x, 0.0, 1.0);
    let flip = old + (pic - sav);
    let new_vel = blend * flip + (1.0 - blend) * pic;
    velocities_out[p] = vec4<f32>(new_vel, 0.0);
}

// --- Stage 9: RK2 advection -------------------------------------------------

fn sample_velocity_world(p: vec3<f32>) -> vec3<f32> {
    let c = cell_space(p);
    let nx = i32(params.dims.x);
    let ny = i32(params.dims.y);
    let nz = i32(params.dims.z);
    return vec3<f32>(
        sample_velocity_axis(c, vec3<f32>(0.0, 0.5, 0.5), nx + 1, ny, nz, params.bases.x),
        sample_velocity_axis(c, vec3<f32>(0.5, 0.0, 0.5), nx, ny + 1, nz, params.bases.y),
        sample_velocity_axis(c, vec3<f32>(0.5, 0.5, 0.0), nx, ny, nz + 1, params.bases.z),
    );
}

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let dt = params.grav_dt.w;
    let x0 = positions[p].xyz;
    let k1 = sample_velocity_world(x0);
    let mid = x0 + 0.5 * dt * k1;
    let k2 = sample_velocity_world(mid);
    let x1 = x0 + dt * k2;
    positions_out[p] = vec4<f32>(x1, 0.0);
}
