// Overshooting-bump twin: evaluates on-device the cumulonimbus overshooting-top
// dome weight at a normalized height, mirroring the CPU golden
// `overshooting_bump` (in `prism_render_architecture::volumetric::storm`).
//
// The cumulonimbus vertical-development model (design section 9b) punches a
// dome through the anvil where a strong updraft overshoots the band top.
// `overshooting_bump` returns the dome weight at a normalized `height_fraction`
// for a dome strength `overshooting_top`:
//   top = saturate(overshooting_top)
//   d   = height_fraction - 1
//   w   = saturate(top * e^{-OVERSHOOT_SHARPNESS * d * d})
// a narrow Gaussian centred on the band top (height_fraction == 1) scaled by the
// dome strength. Heights above 1 are allowed so the dome can bulge past the
// tropopause; the output stays in 0..=1. The CPU golden `overshooting_bump`
// owns that math; this kernel is the on-device twin that runs one thread per
// query and reproduces the same value.
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
// Provenance: standard cumulonimbus overshooting-top dome; no Unreal Engine
// source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
// Gaussian sharpness of the dome, matching `OVERSHOOT_SHARPNESS` on the CPU side.
const OVERSHOOT_SHARPNESS: f32 = 12.0;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One overshooting-bump query: the normalized height plus the dome strength.
// Matches `GpuQuery` in `src/overshooting_bump.rs`.
struct Query {
    height_fraction: f32,
    overshooting_top: f32,
    pad0: f32,
    pad1: f32,
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

// Overshooting-top dome weight at a normalized height, mirroring
// `overshooting_bump`.
fn overshooting_bump(height_fraction: f32, overshooting_top: f32) -> f32 {
    let top = saturate(overshooting_top);
    let d = height_fraction - 1.0;
    return saturate(top * exp_approx(-OVERSHOOT_SHARPNESS * d * d));
}

@compute @workgroup_size(64)
fn overshooting_bump_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = overshooting_bump(q.height_fraction, q.overshooting_top);
}
