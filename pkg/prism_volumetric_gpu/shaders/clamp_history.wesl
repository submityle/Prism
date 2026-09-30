// Clamp-history twin: evaluates on-device the temporal anti-aliasing history
// rectification that bounds a reprojected history sample into the current-frame
// spatial neighbourhood, mirroring the CPU golden `clamp_history` (in
// `prism_render_architecture::volumetric::temporal`).
//
// The temporal upsampler (design section 10) clamps a reprojected history
// sample to `[history_min, history_max]`, the range observed in the current
// frame's spatial neighbourhood. This is the cheapest, most robust anti-ghost
// rectification. When the bounds arrive inverted they are ordered first, so the
// window is always well-formed and the result never escapes it:
//   (lo, hi) = if history_min <= history_max { (history_min, history_max) }
//              else                           { (history_max, history_min) }
//   result   = clamp(history_sample, lo, hi)
//
// The `clamp` is expanded to the SAME branch form the CPU `math::clamp` uses
// (below the low bound -> low, above the high bound -> high, else pass through),
// so CPU and GPU evaluate the same algebra and the parity test can assert a
// tight tolerance.
//
// It uses only the portable core-WGSL subset - comparisons and select - so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard TAA neighbourhood history clamp; no Unreal Engine source
// or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One clamp-history query: the reprojected history sample plus the current
// frame's neighbourhood min and max. Matches `GpuQuery` in
// `src/clamp_history.rs`.
struct Query {
    history_sample: f32,
    history_min: f32,
    history_max: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Scalar clamp, mirroring `math::clamp` bit for bit.
fn clamp_approx(x: f32, lo: f32, hi: f32) -> f32 {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

// The neighbourhood history clamp, mirroring `clamp_history`.
fn clamp_history(history_sample: f32, history_min: f32, history_max: f32) -> f32 {
    let ordered = history_min <= history_max;
    let lo = select(history_max, history_min, ordered);
    let hi = select(history_min, history_max, ordered);
    return clamp_approx(history_sample, lo, hi);
}

@compute @workgroup_size(64)
fn clamp_history_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = clamp_history(q.history_sample, q.history_min, q.history_max);
}
