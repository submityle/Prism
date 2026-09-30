// Pyrocumulus-buoyancy twin: evaluates on-device the pyrocumulus buoyant-lift
// amount from a normalized heat proxy, mirroring the CPU golden
// `pyrocumulus_buoyancy` (in `prism_render_architecture::volumetric::storm`).
//
// The cumulonimbus vertical-development model (design section 9b) drives
// fire-fed pyrocumulus growth from a heat proxy. `pyrocumulus_buoyancy` returns
// a saturating exponential ramp of buoyant lift for a normalized `heat`:
//   q = max(heat, 0)
//   w = saturate(1 - e^{-PYRO_BUOYANCY_GAIN * q})
// so more heat yields more lift with diminishing returns, monotonically
// increasing and bounded to 0..=1, and negative heat is floored at zero. The
// CPU golden `pyrocumulus_buoyancy` owns that math; this kernel is the
// on-device twin that runs one thread per query and reproduces the same value.
//
// The exponential uses the SAME hand-rolled `exp_approx` the reference uses
// (base-two range reduction: a fractional seven-term polynomial times an
// integer power assembled from the float exponent field), NOT the device-native
// `exp`, so the twin stays bit-close to the reference and the parity test can
// assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops,
// clamp/saturate and multiply/add - so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard fire-fed pyrocumulus buoyancy ramp; no Unreal Engine
// source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
// Buoyancy gain of the ramp, matching `PYRO_BUOYANCY_GAIN` on the CPU side.
const PYRO_BUOYANCY_GAIN: f32 = 2.5;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One pyrocumulus-buoyancy query: the normalized heat proxy. Matches `GpuQuery`
// in `src/pyrocumulus_buoyancy.rs`.
struct Query {
    heat: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return 3.4028235e38 * 3.4028235e38;
    }
    return bitcast<f32>(u32(k + 127) << 23u);
}

fn exp2_frac(f: f32) -> f32 {
    let c1 = 0.69314718;
    let c2 = 0.2402265;
    let c3 = 0.05550411;
    let c4 = 0.00961813;
    let c5 = 0.00133336;
    let c6 = 0.00015403;
    return 1.0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * (c5 + f * c6)))));
}

fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
}

// Pyrocumulus buoyant-lift amount from a heat proxy, mirroring
// `pyrocumulus_buoyancy`.
fn pyrocumulus_buoyancy(heat: f32) -> f32 {
    let q = max(heat, 0.0);
    return saturate(1.0 - exp_approx(-PYRO_BUOYANCY_GAIN * q));
}

@compute @workgroup_size(64)
fn pyrocumulus_buoyancy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = pyrocumulus_buoyancy(q.heat);
}
