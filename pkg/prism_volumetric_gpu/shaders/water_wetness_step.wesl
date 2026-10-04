// `wgpu` compute twin of the dependency-free `CPU` golden full moisture update
// (`prism_render_architecture::water::wetness::step_moisture`).
//
// Each surface cell carries two coupled fields driven by the same weather: a
// wetness saturation in `0..=1` (soaks under water contact or rain, dries when
// exposed and rain-free) and a standing puddle depth `>= 0` (rain fills,
// drainage empties). The companion `water_wetness_response` twin ports the
// stateless response kernels (albedo darkening, capillary height, puddle depth,
// puddle predicate) but deliberately leaves these time-stepping envelopes on
// the host; this twin ports exactly those envelopes so a wetness field can
// evolve where the rest of the water subsystem already lives.
//
// The update is embarrassingly parallel: one invocation owns one cell and
// reads only that cell's own state and drivers, so there is no write hazard, no
// barrier and no workgroup memory. Each cell performs the identical
// floating-point operations the `CPU` golden performs, in the same order, so
// the readback is bit-identical to the golden up to the last-place slack of a
// `GPU` fused multiply-add. The kernel uses only the portable core-`WGSL`
// subset (multiply/add, `clamp`, `min`/`max`, index math), so it runs
// unmodified on Metal, Vulkan and DX12.

// Mirror of `prism_render_architecture::water::EPS` (the floating-point epsilon
// the golden uses to gate the rain-wets-versus-dries branch).
const EPS: f32 = 1.0e-6;

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_wetness_step.rs` (`32` bytes, 16-byte aligned). Only the fields
// `step_moisture` actually reads are carried; the response-only `WetnessParams`
// fields (`max_capillary_height`, `darkening_strength`, `puddle_threshold`) do
// not influence a step and are intentionally omitted.
struct Params {
    // Number of cells (`>= 1`).
    count: u32,
    // Absorption rate per second under water contact or full rain.
    absorb_rate: f32,
    // Drying rate per second while exposed and rain-free.
    dry_rate: f32,
    // Puddle drainage rate per second.
    drain_rate: f32,
    // Step time in seconds.
    dt: f32,
    // Padding to the 16-byte alignment the uniform address space requires.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major per-cell wetness saturation (`0..=1`).
@group(0) @binding(1) var<storage, read> wetness: array<f32>;
// Row-major per-cell standing puddle depth (`>= 0`).
@group(0) @binding(2) var<storage, read> puddle: array<f32>;
// Row-major per-cell direct water-contact flag (`0` dry, non-zero in contact).
@group(0) @binding(3) var<storage, read> contact: array<u32>;
// Row-major per-cell normalized rain drive.
@group(0) @binding(4) var<storage, read> rain: array<f32>;
// Row-major per-cell advanced wetness saturation (`0..=1`).
@group(0) @binding(5) var<storage, read_write> wetness_out: array<f32>;
// Row-major per-cell advanced puddle depth (`>= 0`).
@group(0) @binding(6) var<storage, read_write> puddle_out: array<f32>;

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

// Mirror of the golden `absorb`: drive saturation toward `1`, the remaining dry
// fraction decaying exponentially. Keeps the golden's clamp and multiply order.
fn absorb(w: f32, rate: f32, dt: f32) -> f32 {
    let ww = clamp(w, 0.0, 1.0);
    let dry_fraction = (1.0 - ww) * exp_approx(-max(rate, 0.0) * max(dt, 0.0));
    return clamp(1.0 - dry_fraction, 0.0, 1.0);
}

// Mirror of the golden `dry`: drive saturation toward `0` via the shared
// non-negative `exp_approx`. Keeps the golden's clamp and multiply order.
fn dry(w: f32, rate: f32, dt: f32) -> f32 {
    let ww = clamp(w, 0.0, 1.0);
    return ww * exp_approx(-max(rate, 0.0) * max(dt, 0.0));
}

// Mirror of the golden `update_wetness`: water contact soaks at the full
// `absorb_rate`; otherwise rain wets at `absorb_rate * min(rain, 1)` and a
// rain-free surface dries at `dry_rate`. `rain_in` contributes only its
// non-negative part.
fn update_wetness(current: f32, water_contact: bool, rain_in: f32) -> f32 {
    if water_contact {
        return absorb(current, params.absorb_rate, params.dt);
    }
    let r = max(rain_in, 0.0);
    if r > EPS {
        return absorb(current, params.absorb_rate * min(r, 1.0), params.dt);
    }
    return dry(current, params.dry_rate, params.dt);
}

// Mirror of the golden `puddle_depth`: rain adds depth, drainage removes it,
// clamped at zero so a puddle never goes negative. Keeps the golden's add and
// `max` order.
fn puddle_depth(accumulated: f32, rain_in: f32) -> f32 {
    let base = max(accumulated, 0.0);
    return max(base + (max(rain_in, 0.0) - max(params.drain_rate, 0.0)) * max(params.dt, 0.0), 0.0);
}

@compute @workgroup_size(256)
fn step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }
    let water_contact = contact[idx] != 0u;
    let rain_in = rain[idx];
    wetness_out[idx] = update_wetness(wetness[idx], water_contact, rain_in);
    puddle_out[idx] = puddle_depth(puddle[idx], rain_in);
}
