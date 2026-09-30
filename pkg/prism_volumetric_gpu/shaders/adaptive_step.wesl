// Adaptive-step twin: chooses the next raymarch stride for a sample on-device,
// mirroring the CPU golden `adaptive_step` (in
// `prism_render_architecture::volumetric::raymarch`).
//
// Volumetric clouds march with a variable stride so empty space is skipped in
// large jumps and dense medium is sampled finely (design section 6). Outside
// the cloud (not `in_cloud`, or density below `density_threshold`) the largest
// stride `max_step` is returned to skip empty space. Inside the cloud the step
// interpolates from `base_step` toward `min_step` as the saturated density
// rises, so the step is monotone non-increasing in density:
//   if !in_cloud || density < density_threshold -> clamp(max_step, min_step, max_step)
//   else t = saturate(density); step = lerp(base_step, min_step, t);
//        clamp(step, min_step, max_step)
// The result is always clamped into `[min_step, max_step]`. The CPU golden owns
// that logic; this kernel is the on-device twin that runs one thread per query
// and reproduces the same stride.
//
// The math is pure clamp/lerp/saturate (no exp), so this kernel mirrors the CPU
// `clamp` (branch form) and `lerp` exactly and the parity test asserts a tight
// tolerance. The result is one f32 per query. It uses only the portable
// core-WGSL subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard density-adaptive raymarch step selection; no Unreal
// Engine source or derived code.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One step query: the sample density, whether the sample is in-cloud (0/1) and
// the four raymarch-config bounds used by `adaptive_step`. Matches `GpuQuery`
// in `src/adaptive_step.rs`.
struct Query {
    current_density: f32,
    in_cloud: u32,
    base_step: f32,
    max_step: f32,
    min_step: f32,
    density_threshold: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Branch form mirroring the CPU `math::clamp` exactly (native WGSL `clamp` is
// implementation-defined when `lo > hi`).
fn clamp_approx(x: f32, lo: f32, hi: f32) -> f32 {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

fn saturate_s(x: f32) -> f32 {
    return clamp_approx(x, 0.0, 1.0);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Density-adaptive stride, mirroring `adaptive_step`.
fn adaptive_step(
    current_density: f32,
    in_cloud: u32,
    base_step: f32,
    max_step: f32,
    min_step: f32,
    density_threshold: f32,
) -> f32 {
    if (in_cloud == 0u || current_density < density_threshold) {
        return clamp_approx(max_step, min_step, max_step);
    }
    let t = saturate_s(current_density);
    let step = lerp(base_step, min_step, t);
    return clamp_approx(step, min_step, max_step);
}

@compute @workgroup_size(64)
fn adaptive_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = adaptive_step(
        q.current_density,
        q.in_cloud,
        q.base_step,
        q.max_step,
        q.min_step,
        q.density_threshold,
    );
}
