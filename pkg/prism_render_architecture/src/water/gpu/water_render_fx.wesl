// Water render-FX compute kernels (GPU twins of the CPU golden numerics in
// `prism_render_architecture::water::foam` and the sibling render-FX modules).
//
// This is the first real water-subsystem compute shader to land. It is the
// device companion to the CPU golden reference: the renderer runs this exact
// arithmetic on the GPU every frame against the resident foam-coverage field,
// while the CPU golden (`foam::step_foam`) and the bit-exact CPU twin
// (`render_fx_kernel::dispatch_foam_advect`) prove it correct in the sandbox,
// where there is no GPU.
//
// Entry point `water_foam_advect` mirrors `foam::step_foam` cell-for-cell:
// semi-Lagrangian backtrace + manual bilinear resample of the previous foam
// field, flow-aware exponential decay, additive reactive sources, and a final
// clamp to [0, 1] coverage. The arithmetic is only `+ - * /`, `sqrt`, and the
// hand-rolled `exp_approx` (identical to `water::exp_approx`), matching the
// workspace determinism policy that forbids every float intrinsic but `sqrt`.
//
// Buffer / binding ABI (mirrors `kernels::WaterKernel::FoamAdvect.descriptor()`
// byte-for-byte: 1 storage buffer, 1 uniform, 1 sampled texture, 1 storage
// texture, dispatched over an 8x8 texel tile on a 2D grid):
//   @binding(0) foam_prev   previous foam coverage, one r32float texel per cell
//   @binding(1) foam_next   next foam coverage, one r32float texel per cell
//   @binding(2) drive       per-cell vec4<f32>(u, v, source, pad), 16-byte stride
//   @binding(3) params      per-step scalars (grid size, dx, dt, decay model)

struct FoamParams {
    // Grid cell counts along x and z.
    nx: u32,
    nz: u32,
    // Cell size in meters (> 0).
    dx: f32,
    // Explicit integration step in seconds.
    dt: f32,
    // Baseline decay rate per second at or above `reference_speed`.
    base_decay: f32,
    // Fraction of `base_decay` that still applies in perfectly still water.
    persistence_floor: f32,
    // Flow speed at which decay reaches the full `base_decay` (> 0).
    reference_speed: f32,
    // Absolute tolerance guarding the division by `reference_speed`.
    eps: f32,
}

@group(0) @binding(0) var foam_prev: texture_2d<f32>;
@group(0) @binding(1) var foam_next: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read> drive: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: FoamParams;

// Hand-rolled exp via the limit identity exp(x) = (1 + x/4096)^4096, evaluated
// by twelve squarings. Bit-for-bit the same algorithm as `water::exp_approx`:
// the base is clamped to >= 0 so an extreme negative argument saturates to 0.
fn exp_approx(x: f32) -> f32 {
    var base: f32 = 1.0 + x / 4096.0;
    if (base < 0.0) {
        base = 0.0;
    }
    for (var i: i32 = 0; i < 12; i = i + 1) {
        base = base * base;
    }
    return base;
}

// Flow-aware decay rate, mirroring `foam::foam_decay_rate`: interpolates from
// `base_decay * persistence_floor` in still water up to the full `base_decay`
// at or beyond `reference_speed`, rising monotonically with flow speed.
fn foam_decay_rate(flow_speed: f32) -> f32 {
    let floor_v: f32 = clamp(params.persistence_floor, 0.0, 1.0);
    var reference: f32 = params.reference_speed;
    if (reference <= params.eps) {
        reference = params.eps;
    }
    let t: f32 = clamp(flow_speed / reference, 0.0, 1.0);
    return params.base_decay * (floor_v + (1.0 - floor_v) * t);
}

// Bilinearly samples the previous foam field at fractional cell coordinates,
// clamping into [0, nx-1] x [0, nz-1] exactly as `foam::sample_bilinear`, so a
// backtrace never reads out of bounds and a non-negative field stays
// non-negative. Uses `textureLoad` (not a sampler) so the four taps and the
// convex combination match the CPU golden tap-for-tap.
fn sample_bilinear_prev(px: f32, pz: f32) -> f32 {
    let max_x: f32 = f32(params.nx) - 1.0;
    let max_z: f32 = f32(params.nz) - 1.0;
    let cx: f32 = clamp(px, 0.0, max_x);
    let cz: f32 = clamp(pz, 0.0, max_z);
    let x0: i32 = i32(cx);
    let z0: i32 = i32(cz);
    let x1: i32 = min(x0 + 1, i32(params.nx) - 1);
    let z1: i32 = min(z0 + 1, i32(params.nz) - 1);
    let fx: f32 = cx - f32(x0);
    let fz: f32 = cz - f32(z0);
    let s00: f32 = textureLoad(foam_prev, vec2<i32>(x0, z0), 0).x;
    let s10: f32 = textureLoad(foam_prev, vec2<i32>(x1, z0), 0).x;
    let s01: f32 = textureLoad(foam_prev, vec2<i32>(x0, z1), 0).x;
    let s11: f32 = textureLoad(foam_prev, vec2<i32>(x1, z1), 0).x;
    let top: f32 = s00 + (s10 - s00) * fx;
    let bottom: f32 = s01 + (s11 - s01) * fx;
    return top + (bottom - top) * fz;
}

@compute @workgroup_size(8, 8, 1)
fn water_foam_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.nx || gid.y >= params.nz) {
        return;
    }
    let x: u32 = gid.x;
    let z: u32 = gid.y;
    let idx: u32 = z * params.nx + x;
    let inv_dx: f32 = 1.0 / params.dx;

    let drive_cell: vec4<f32> = drive[idx];
    let u: f32 = drive_cell.x;
    let v: f32 = drive_cell.y;
    let source: f32 = max(drive_cell.z, 0.0);

    // Semi-Lagrangian backtrace of this cell centre by -dt * velocity, then
    // resample the previous field there (`foam::advect_foam_field`).
    let px: f32 = f32(x) - params.dt * u * inv_dx;
    let pz: f32 = f32(z) - params.dt * v * inv_dx;
    let advected: f32 = sample_bilinear_prev(px, pz);

    // Flow-aware exponential decay (`foam::decay_foam`), then additive reactive
    // sources, clamped to physical [0, 1] coverage (`foam::step_foam`).
    let flow_speed: f32 = sqrt(u * u + v * v);
    let rate: f32 = foam_decay_rate(flow_speed);
    let decayed: f32 = max(advected, 0.0) * exp_approx(-rate * params.dt);
    let coverage: f32 = clamp(decayed + source, 0.0, 1.0);

    textureStore(foam_next, vec2<i32>(i32(x), i32(z)), vec4<f32>(coverage, 0.0, 0.0, 0.0));
}
