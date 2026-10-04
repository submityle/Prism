// `wgpu` compute twin of the dependency-free `CPU` golden full foam update
// (`prism_render_architecture::water::foam::step_foam`).
//
// Foam is a scalar coverage field in `0..=1` carried on the surface-flow grid.
// One step advects the field along the flow, applies a flow-aware exponential
// decay, adds this frame's foam sources, and clamps coverage back into
// `0..=1` — the churn-persists / calm-fades foam law the `Niagara`/`Crest`
// foam passes run on device. This twin fuses the whole update into one dispatch
// so the foam field evolves where the rest of the water subsystem already
// lives.
//
// The update is embarrassingly parallel: one invocation owns one output cell.
// The advection stage reads only the *previous* field (a pure gather, no write
// hazard), and the decay-plus-source stage depends solely on that cell's own
// advected value and its own flow sample, so there is no barrier and no
// workgroup memory. Each cell performs the identical floating-point operations
// the `CPU` golden performs, in the same order (backtrace + bilinear gather,
// `sqrt` flow speed, flow-aware rate, `exp_approx` decay, add source, clamp),
// so the readback is bit-identical to the golden up to the last-place slack of
// a `GPU` fused multiply-add. The kernel uses only the portable core-`WGSL`
// subset (multiply/add, `clamp`, `sqrt`, `floor`, index math), so it runs
// unmodified on Metal, Vulkan and DX12.

// Mirror of `prism_render_architecture::water::EPS` (the floating-point epsilon
// the foam decay uses to floor the reference speed).
const EPS: f32 = 1.0e-6;

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_foam_step.rs` (`32` bytes, 16-byte aligned).
struct Params {
    // Cell count along x (`>= 1`).
    nx: u32,
    // Cell count along z (`>= 1`).
    nz: u32,
    // Cell size in meters (`> 0`).
    dx: f32,
    // Step time in seconds.
    dt: f32,
    // Baseline decay rate per second at or above `reference_speed`.
    base_decay: f32,
    // Fraction of `base_decay` that still applies in perfectly still water.
    persistence_floor: f32,
    // Flow speed at which decay reaches the full `base_decay` (`> 0`).
    reference_speed: f32,
    // Padding to the 16-byte alignment the uniform address space requires.
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major `nx * nz` previous foam coverage.
@group(0) @binding(1) var<storage, read> density: array<f32>;
// Row-major `nx * nz` surface flow, x component (meters/second).
@group(0) @binding(2) var<storage, read> flow_u: array<f32>;
// Row-major `nx * nz` surface flow, z component (meters/second).
@group(0) @binding(3) var<storage, read> flow_v: array<f32>;
// Row-major `nx * nz` per-cell foam sources this step (already `dt`-scaled).
@group(0) @binding(4) var<storage, read> sources: array<f32>;
// Row-major `nx * nz` advanced foam coverage.
@group(0) @binding(5) var<storage, read_write> out_density: array<f32>;

// Bilinear sample of the previous foam field at fractional cell coordinates,
// mirroring the golden `sample_bilinear` tap order exactly.
fn sample_bilinear(px: f32, pz: f32) -> f32 {
    let nx = params.nx;
    let nz = params.nz;
    let max_x = f32(nx - 1u);
    let max_z = f32(nz - 1u);
    let cx = clamp(px, 0.0, max_x);
    let cz = clamp(pz, 0.0, max_z);
    // `cx`/`cz` are non-negative, so truncation equals floor, matching the
    // golden's `as usize` cast.
    let x0 = u32(cx);
    let z0 = u32(cz);
    var x1 = x0 + 1u;
    if x1 > nx - 1u {
        x1 = nx - 1u;
    }
    var z1 = z0 + 1u;
    if z1 > nz - 1u {
        z1 = nz - 1u;
    }
    let fx = cx - f32(x0);
    let fz = cz - f32(z0);
    let s00 = density[z0 * nx + x0];
    let s10 = density[z0 * nx + x1];
    let s01 = density[z1 * nx + x0];
    let s11 = density[z1 * nx + x1];
    let top = s00 + (s10 - s00) * fx;
    let bottom = s01 + (s11 - s01) * fx;
    return top + (bottom - top) * fz;
}

// Bit-for-bit mirror of the shared water `exp_approx`: evaluate the limit
// identity `exp(x) = (1 + x/4096)^4096` by twelve squarings, clamping the base
// to zero so an extreme negative argument saturates to `0`. Pure add/multiply.
fn exp_approx(x: f32) -> f32 {
    var base = 1.0 + x / 4096.0;
    if base < 0.0 {
        base = 0.0;
    }
    var i = 0u;
    loop {
        if i >= 12u {
            break;
        }
        base = base * base;
        i = i + 1u;
    }
    return base;
}

// Flow-aware decay rate (per second), mirroring the golden `foam_decay_rate`.
fn foam_decay_rate(flow_speed: f32) -> f32 {
    let floor = clamp(params.persistence_floor, 0.0, 1.0);
    var reference = EPS;
    if params.reference_speed > EPS {
        reference = params.reference_speed;
    }
    let t = clamp(flow_speed / reference, 0.0, 1.0);
    return params.base_decay * (floor + (1.0 - floor) * t);
}

// Exponential decay of a non-negative coverage, mirroring the golden
// `decay_foam`: `max(density, 0) * exp_approx(-rate * dt)`.
fn decay_foam(d_in: f32, rate: f32) -> f32 {
    var d = 0.0;
    if d_in > 0.0 {
        d = d_in;
    }
    return d * exp_approx(-rate * params.dt);
}

@compute @workgroup_size(256)
fn step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.nx * params.nz;
    if idx >= n {
        return;
    }
    let x = idx % params.nx;
    let z = idx / params.nx;
    // `inv_dx` is the golden's single `1.0 / dx` reciprocal; the backtrace keeps
    // the golden's `dt * vel * inv_dx` multiply order.
    let inv_dx = 1.0 / params.dx;
    let px = f32(x) - params.dt * flow_u[idx] * inv_dx;
    let pz = f32(z) - params.dt * flow_v[idx] * inv_dx;
    let advected = sample_bilinear(px, pz);

    let uu = flow_u[idx];
    let vv = flow_v[idx];
    let flow_speed = sqrt(uu * uu + vv * vv);
    let rate = foam_decay_rate(flow_speed);
    let decayed = decay_foam(advected, rate);
    var source = sources[idx];
    if source < 0.0 {
        source = 0.0;
    }
    out_density[idx] = clamp(decayed + source, 0.0, 1.0);
}
