// Beer-Lambert fog-transmittance twin: evaluates on-device the transmittance
// through a uniform fog slab, mirroring the CPU golden `fog_transmittance` (in
// `prism_render_architecture::volumetric::fog`).
//
// The unified volumetric fog (design section 9f) attenuates radiance through a
// uniform slab by Beer-Lambert: `T = e^{-density * distance}` saturated into
// `0..=1`. Negative inputs are floored at zero, so a zero distance or zero
// density yields exactly one and larger optical depths decay monotonically
// toward zero. The CPU golden `fog_transmittance` owns that math; this kernel
// is the on-device twin that runs one thread per query and reproduces the same
// value.
//
// The exponential uses the SAME hand-rolled `exp_approx` the reference uses
// (base-two range reduction: a fractional seven-term polynomial times an
// integer power assembled from the float exponent field), NOT the device-native
// `exp`, so the twin stays bit-close to the reference and the parity test can
// assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops and
// multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Beer-Lambert extinction; no Unreal Engine source or
// derived code.

const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One fog query: the ray distance through the slab and the slab extinction
// density. Matches `GpuQuery` in `src/fog.rs`.
struct Query {
    distance: f32,
    density: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return bitcast<f32>(0x7f800000u);
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

// Beer-Lambert transmittance through a uniform fog slab, mirroring
// `fog_transmittance`.
fn fog_transmittance(distance: f32, density: f32) -> f32 {
    let optical_depth = max(density, 0.0) * max(distance, 0.0);
    return saturate_s(exp_approx(-optical_depth));
}

@compute @workgroup_size(64)
fn fog_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = fog_transmittance(q.distance, q.density);
}
