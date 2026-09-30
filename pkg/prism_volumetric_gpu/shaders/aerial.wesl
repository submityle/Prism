// Aerial-perspective fade twin: evaluates the normalised exponential opacity
// that blends distant geometry toward the atmosphere in-scatter, mirroring the
// CPU golden `aerial_perspective_weight` (in
// `prism_render_architecture::volumetric::atmosphere`).
//
// The weight is `(1 - e^{-k t}) / (1 - e^{-k})` with `t = distance /
// max_distance` clamped to `0..=1` and `k` = AERIAL_EXTINCTION, so it is
// monotone non-decreasing in distance, exactly 0 at the camera and exactly 1 at
// the far plane, mirroring the froxel fade without reading or rewriting the
// shared atmosphere LUT (design section 8). A non-positive `max_distance`
// degenerates safely: any positive distance maps to 1, otherwise 0. The CPU
// golden owns that math; this kernel is the on-device twin that runs one thread
// per query and reproduces the same value.
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
// Provenance: standard normalised exponential aerial-perspective fade; no
// Unreal Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
const EPS: f32 = 1e-6;
const AERIAL_EXTINCTION: f32 = 4.0;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One aerial query: the fragment distance and the fade's far distance. Matches
// `GpuQuery` in `src/aerial.rs`.
struct Query {
    distance: f32,
    max_distance: f32,
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

// Normalised exponential aerial-perspective weight, mirroring
// `aerial_perspective_weight`.
fn aerial_perspective_weight(distance: f32, max_distance: f32) -> f32 {
    if (max_distance <= EPS) {
        if (distance > 0.0) {
            return 1.0;
        }
        return 0.0;
    }
    let t = saturate_s(distance / max_distance);
    let numer = 1.0 - exp_approx(-AERIAL_EXTINCTION * t);
    let denom = 1.0 - exp_approx(-AERIAL_EXTINCTION);
    return saturate_s(numer / denom);
}

@compute @workgroup_size(64)
fn aerial_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = aerial_perspective_weight(q.distance, q.max_distance);
}
