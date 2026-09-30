// Variance-clip twin: evaluates on-device the temporal anti-aliasing variance
// box clip of a reprojected history sample, mirroring the CPU golden
// `variance_clip` (in `prism_render_architecture::volumetric::temporal`).
//
// The temporal upsampler (design section 10) rectifies a reprojected history
// sample against the current-frame statistics to reject ghosting while keeping
// more history than a hard neighbourhood clamp:
//   sd   = max(std_dev, 0)
//   g    = max(gamma, 0)
//   half = sd * g
//   result = clamp(history_sample, mean - half, mean + half)
// `std_dev` and `gamma` are floored at zero, so the clip window is always
// well-formed (its lower bound never exceeds its upper bound) and the result
// lies inside `[mean - half, mean + half]`. The CPU golden `variance_clip` owns
// that math; this kernel is the on-device twin that runs one thread per query.
//
// The `clamp` is expanded to the SAME branch form the CPU `math::clamp` uses
// (below the low bound -> low, above the high bound -> high, else pass through),
// so CPU and GPU evaluate the same algebra and the parity test can assert a
// tight tolerance.
//
// It uses only the portable core-WGSL subset - comparisons and multiply/add -
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard TAA variance box clip; no Unreal Engine source or
// derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One variance-clip query: the reprojected history sample plus the current
// frame's neighbourhood mean, standard deviation and the gamma window scale.
// Matches `GpuQuery` in `src/variance_clip.rs`.
struct Query {
    history_sample: f32,
    mean: f32,
    std_dev: f32,
    gamma: f32,
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

// The variance box clip, mirroring `variance_clip`.
fn variance_clip(history_sample: f32, mean: f32, std_dev: f32, gamma: f32) -> f32 {
    let sd = max(std_dev, 0.0);
    let g = max(gamma, 0.0);
    let half = sd * g;
    return clamp_approx(history_sample, mean - half, mean + half);
}

@compute @workgroup_size(64)
fn variance_clip_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = variance_clip(q.history_sample, q.mean, q.std_dev, q.gamma);
}
