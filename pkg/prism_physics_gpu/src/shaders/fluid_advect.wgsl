// GPU marker-particle advection: one second-order Runge-Kutta (midpoint) step
// per particle through the staggered MAC velocity field.
//
// Byte-for-byte-intent twin of the CPU golden reference in
// `src/fluid/cpu/advect.rs`. Each particle is independent, so this is a single
// dispatch of one thread per marker; the only arithmetic is the two trilinear
// gathers shared with the grid-to-particle transfer, so the device reproduces
// the twin within the same tight tolerance.
//
// The three staggered face fields u, v, w are concatenated into one array laid
// out [u | v | w]; a field's base offset plus its local flat index gives the
// global index, matching `fluid_transfer.wgsl`.
//
// Provenance: second-order Runge-Kutta advection of markers through a sampled
// velocity field is a standard semi-Lagrangian technique (Bridson; Zhu &
// Bridson 2005). No Unreal Engine source or derived code.

struct Params {
    // xyz = grid origin, w = cell size dx.
    origin_dx: vec4<f32>,
    // x=nx, y=ny, z=nz, w=particle_count.
    dims: vec4<u32>,
    // x=u_base, y=v_base, z=w_base, w=face_total.
    bases: vec4<u32>,
    // x = dt, yzw = padding.
    step: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> positions_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> velocity: array<f32>;

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

fn cell_space(p: vec3<f32>) -> vec3<f32> {
    return (p - params.origin_dx.xyz) / params.origin_dx.w;
}

// Trilinear gather of one staggered face field at cell-space point `c`.
fn sample_axis(c: vec3<f32>, off: vec3<f32>, dimx: i32, dimy: i32, dimz: i32, base: u32) -> f32 {
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

// Full trilinear velocity sample at world position `p`.
fn sample_velocity(p: vec3<f32>) -> vec3<f32> {
    let c = cell_space(p);
    let nx = i32(params.dims.x);
    let ny = i32(params.dims.y);
    let nz = i32(params.dims.z);
    return vec3<f32>(
        sample_axis(c, vec3<f32>(0.0, 0.5, 0.5), nx + 1, ny, nz, params.bases.x),
        sample_axis(c, vec3<f32>(0.5, 0.0, 0.5), nx, ny + 1, nz, params.bases.y),
        sample_axis(c, vec3<f32>(0.5, 0.5, 0.0), nx, ny, nz + 1, params.bases.z),
    );
}

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.dims.w) {
        return;
    }
    let dt = params.step.x;
    let x0 = positions_in[p].xyz;
    let k1 = sample_velocity(x0);
    let mid = x0 + 0.5 * dt * k1;
    let k2 = sample_velocity(mid);
    let x1 = x0 + dt * k2;
    positions_out[p] = vec4<f32>(x1, 0.0);
}
