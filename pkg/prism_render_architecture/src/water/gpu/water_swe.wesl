// ===========================================================================
// Shallow-water step kernel (GPU twin of `swe::step`).
//
// One compute invocation advances one shallow-water cell by `dt`, reproducing
// the CPU golden `swe::step` byte-for-byte: conservative flux-form continuity
// with reflective (no-flux) walls, a central pressure gradient, first-order
// upwind self-advection of the velocity field, and linear velocity damping.
//
// Binding / dispatch ABI (mirrors `kernels::WaterKernel::SweStep.descriptor()`:
// 2 storage buffers, 1 uniform, 1 storage texture, 1 sampled texture, 8x8 tile,
// 2D grid domain). The classic SWE ping-pong maps cleanly onto it: the water
// depth lives in a texture pair (read the current frame, write the next) while
// the velocity field lives in a storage-buffer pair.
//   @binding(0) vel_in       current velocity, one vec4<f32> per cell: (u, v, _, _)
//   @binding(1) vel_out      next velocity, same packing (written)
//   @binding(2) swe_params   grid size + physical constants + dt
//   @binding(3) height_next  next water depth, one r32float texel (written)
//   @binding(4) height_curr  current water depth, one r32float texel (sampled)
//
// Determinism policy: only + - * /, and the reads above appear; no float
// intrinsics are used (the step needs none), matching `swe::step`.
// ===========================================================================

struct SweParams {
    // Cell size in meters (> 0); square cells.
    dx: f32,
    // Gravitational acceleration for the pressure gradient.
    gravity: f32,
    // Linear velocity damping per second, in 0..=1.
    damping: f32,
    // Explicit integration step in seconds.
    dt: f32,
    // Grid cell counts along x and z.
    nx: u32,
    nz: u32,
    // Padding to a 16-byte multiple for the uniform address space.
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> swe_params: SweParams;
@group(0) @binding(3) var height_next: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var height_curr: texture_2d<f32>;

// Current water depth at an in-range cell.
fn load_h(x: i32, z: i32) -> f32 {
    return textureLoad(height_curr, vec2<i32>(x, z), 0).x;
}

// Current x-velocity at an in-range cell.
fn load_u(x: i32, z: i32, nx: i32) -> f32 {
    return vel_in[u32(z * nx + x)].x;
}

// Current z-velocity at an in-range cell.
fn load_v(x: i32, z: i32, nx: i32) -> f32 {
    return vel_in[u32(z * nx + x)].y;
}

// First-order upwind advective derivative `u * dF/dx + v * dF/dz`, taking the
// gradient from the upwind neighbour (the direction the flow comes from).
// Mirrors `swe::upwind` branch-for-branch.
fn upwind(
    flow_u: f32,
    flow_v: f32,
    center: f32,
    xm: f32,
    xp: f32,
    zm: f32,
    zp: f32,
    inv_dx: f32,
) -> f32 {
    var ddx: f32;
    if (flow_u > 0.0) {
        ddx = (center - xm) * inv_dx;
    } else {
        ddx = (xp - center) * inv_dx;
    }
    var ddz: f32;
    if (flow_v > 0.0) {
        ddz = (center - zm) * inv_dx;
    } else {
        ddz = (zp - center) * inv_dx;
    }
    return flow_u * ddx + flow_v * ddz;
}

@compute @workgroup_size(8, 8, 1)
fn water_swe_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nx: u32 = swe_params.nx;
    let nz: u32 = swe_params.nz;
    if (gid.x >= nx || gid.y >= nz) {
        return;
    }

    let x: i32 = i32(gid.x);
    let z: i32 = i32(gid.y);
    let nxi: i32 = i32(nx);
    let nzi: i32 = i32(nz);
    let idx: u32 = gid.y * nx + gid.x;

    let inv_dx: f32 = 1.0 / swe_params.dx;
    let dt: f32 = swe_params.dt;
    let g: f32 = swe_params.gravity;

    // Center sample.
    let h_c: f32 = load_h(x, z);
    let u_c: f32 = load_u(x, z, nxi);
    let v_c: f32 = load_v(x, z, nxi);
    let hu_c: f32 = h_c * u_c;
    let hv_c: f32 = h_c * v_c;

    let has_xp: bool = x + 1 < nxi;
    let has_xm: bool = x >= 1;
    let has_zp: bool = z + 1 < nzi;
    let has_zm: bool = z >= 1;

    // Reflective-wall momentum flux: a shared interior face cancels across its
    // two cells, so total volume telescopes to zero change at the walls.
    var fx_right: f32 = 0.0;
    if (has_xp) {
        let hu_xp: f32 = load_h(x + 1, z) * load_u(x + 1, z, nxi);
        fx_right = 0.5 * (hu_c + hu_xp);
    }
    var fx_left: f32 = 0.0;
    if (has_xm) {
        let hu_xm: f32 = load_h(x - 1, z) * load_u(x - 1, z, nxi);
        fx_left = 0.5 * (hu_xm + hu_c);
    }
    var fz_up: f32 = 0.0;
    if (has_zp) {
        let hv_zp: f32 = load_h(x, z + 1) * load_v(x, z + 1, nxi);
        fz_up = 0.5 * (hv_c + hv_zp);
    }
    var fz_down: f32 = 0.0;
    if (has_zm) {
        let hv_zm: f32 = load_h(x, z - 1) * load_v(x, z - 1, nxi);
        fz_down = 0.5 * (hv_zm + hv_c);
    }
    let new_h: f32 = h_c - dt * inv_dx * ((fx_right - fx_left) + (fz_up - fz_down));

    // Reflective neighbour sampling for the momentum gradients.
    var h_xp: f32 = h_c;
    if (has_xp) { h_xp = load_h(x + 1, z); }
    var h_xm: f32 = h_c;
    if (has_xm) { h_xm = load_h(x - 1, z); }
    var h_zp: f32 = h_c;
    if (has_zp) { h_zp = load_h(x, z + 1); }
    var h_zm: f32 = h_c;
    if (has_zm) { h_zm = load_h(x, z - 1); }
    let dhdx: f32 = (h_xp - h_xm) * 0.5 * inv_dx;
    let dhdz: f32 = (h_zp - h_zm) * 0.5 * inv_dx;

    var u_xp: f32 = u_c;
    if (has_xp) { u_xp = load_u(x + 1, z, nxi); }
    var u_xm: f32 = u_c;
    if (has_xm) { u_xm = load_u(x - 1, z, nxi); }
    var u_zp: f32 = u_c;
    if (has_zp) { u_zp = load_u(x, z + 1, nxi); }
    var u_zm: f32 = u_c;
    if (has_zm) { u_zm = load_u(x, z - 1, nxi); }

    var v_xp: f32 = v_c;
    if (has_xp) { v_xp = load_v(x + 1, z, nxi); }
    var v_xm: f32 = v_c;
    if (has_xm) { v_xm = load_v(x - 1, z, nxi); }
    var v_zp: f32 = v_c;
    if (has_zp) { v_zp = load_v(x, z + 1, nxi); }
    var v_zm: f32 = v_c;
    if (has_zm) { v_zm = load_v(x, z - 1, nxi); }

    let adv_u: f32 = upwind(u_c, v_c, u_c, u_xm, u_xp, u_zm, u_zp, inv_dx);
    let adv_v: f32 = upwind(u_c, v_c, v_c, v_xm, v_xp, v_zm, v_zp, inv_dx);

    let damp: f32 = max(1.0 - swe_params.damping * dt, 0.0);
    let new_u: f32 = (u_c - dt * (adv_u + g * dhdx)) * damp;
    let new_v: f32 = (v_c - dt * (adv_v + g * dhdz)) * damp;

    vel_out[idx] = vec4<f32>(new_u, new_v, 0.0, 0.0);
    textureStore(height_next, vec2<i32>(x, z), vec4<f32>(new_h, 0.0, 0.0, 0.0));
}
