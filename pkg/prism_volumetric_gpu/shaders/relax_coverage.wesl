// Coverage-relaxation twin: relaxes a cell's current cloud coverage toward its
// target on-device, mirroring the CPU golden `relax_coverage` (in
// `prism_render_architecture::volumetric::weather`).
//
// The sky-state machine (Clear -> Fair -> Overcast -> Storm) drives a target
// coverage per cell; each cell approaches its target with a smooth exponential
// blend rather than snapping, coupling generation and decay to the weather
// field (design section 9). Over a step `dt` at rate `rate` (per unit time) the
// blend weight is `k = saturate(1 - exp(-max(rate,0) * max(dt,0)))` and the new
// coverage is `current + (target - current) * k`. Clamping `rate` and `dt` to
// non-negative and saturating `k` keeps the weight in `0..=1`, so the result is
// monotone toward `target`, never overshoots, and stays in `0..=1` whenever
// both `current` and `target` do.
//
// The CPU golden uses a hand-rolled `exp_approx` (no float intrinsic, for
// determinism across targets); this kernel mirrors that same polynomial
// exp so the on-device result matches bit-close, rather than calling the WGSL
// `exp` builtin. Result is one f32 per query. It uses only the portable
// core-WGSL subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard exponential relaxation / weather coverage coupling; no
// Unreal Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One relaxation query: the current coverage, its target, the relaxation rate
// and the timestep. Matches `GpuQuery` in `src/relax_coverage.rs`.
struct Query {
    current: f32,
    target: f32,
    rate: f32,
    dt: f32,
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

// Exponential relaxation toward target, mirroring `relax_coverage`.
fn relax_coverage(current: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let k = saturate_s(1.0 - exp_approx(-max(rate, 0.0) * max(dt, 0.0)));
    return current + (target - current) * k;
}

@compute @workgroup_size(64)
fn relax_coverage_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = relax_coverage(q.current, q.target, q.rate, q.dt);
}
