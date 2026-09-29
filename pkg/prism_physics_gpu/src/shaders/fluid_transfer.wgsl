// FLIP/APIC fluid transfer kernels: particle-to-grid scatter, face
// normalisation, and grid-to-particle gather.
//
// Byte-for-byte-intent twin of the CPU golden reference in `src/fluid/cpu/`.
// The particle scatter is a race: many particles add their trilinearly
// weighted momentum and weight to the same MAC face. WGSL atomics operate only
// on 32-bit integers, so each floating contribution is quantised (value * scale
// rounded to nearest even, matching Rust `round_ties_even`) and summed with an
// integer atomicAdd. Integer addition is exact and order-independent, so the
// device sum is deterministic and reproduces the CPU golden's accumulators
// bit-for-bit; only the final momentum/weight division and the gather are
// floating point, which the parity test bounds with a tight tolerance.
//
// The three staggered face fields u, v, w are concatenated into single arrays
// laid out [u | v | w]; a field's base offset (u_base / v_base / w_base) plus
// its local flat index gives the global index. This keeps normalisation to a
// single dispatch over `face_total` faces.
//
// Provenance: trilinear P2G/G2P with the PIC/FLIP blend (Zhu & Bridson 2005;
// Bridson), fixed-point atomic scatter (standard GPU technique). No Unreal
// Engine source or derived code.

struct Params {
    // xyz = grid origin, w = cell size dx.
    origin_dx: vec4<f32>,
    // x=nx, y=ny, z=nz, w=particle_count.
    dims: vec4<u32>,
    // x=u_base, y=v_base, z=w_base, w=face_total.
    bases: vec4<u32>,
    // x=flip_blend, yzw=padding.
    blend: vec4<f32>,
};

const MOMENTUM_SCALE: f32 = 65536.0;
const WEIGHT_SCALE: f32 = 65536.0;

@group(0) @binding(0) var<uniform> params: Params;
// Particle columns, xyz + pad.
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities_out: array<vec4<f32>>;
// Concatenated [u | v | w] accumulators and fields.
@group(0) @binding(4) var<storage, read_write> momentum: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> weight: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(7) var<storage, read> saved: array<f32>;

// One axis of a trilinear stencil after clamping into the valid node range.
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

// Stage 1: scatter one particle's velocity onto the staggered faces.
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

    // u faces: dims (nx+1, ny, nz), offset (0, 0.5, 0.5).
    scatter_axis(c, vec3<f32>(0.0, 0.5, 0.5), nx + 1, ny, nz, params.bases.x, vel.x);
    // v faces: dims (nx, ny+1, nz), offset (0.5, 0, 0.5).
    scatter_axis(c, vec3<f32>(0.5, 0.0, 0.5), nx, ny + 1, nz, params.bases.y, vel.y);
    // w faces: dims (nx, ny, nz+1), offset (0.5, 0.5, 0).
    scatter_axis(c, vec3<f32>(0.5, 0.5, 0.0), nx, ny, nz + 1, params.bases.z, vel.z);
}

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

// Stage 2: divide momentum by weight on every concatenated face.
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

// Stage 3: gather grid velocity back to particles with the PIC/FLIP blend.
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

// Trilinear gather from the normalised velocity field. WGSL does not portably
// support passing storage pointers to helpers, so the current and saved fields
// have dedicated samplers over the identical stencil.
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
