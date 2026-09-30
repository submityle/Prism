// Analytic single-scatter twin: evaluates the closed-form single-scattered
// radiance of a homogeneous medium on-device, mirroring the CPU golden
// `analytic_single_scatter` (in `prism_render_architecture::volumetric::reference`).
//
// This is the reference single-scatter value the delta/ratio tracking
// estimators must converge to (design section 9c reference). For a uniform slab
// lit by a constant light radiance the accumulated single scattering is
//   st       = max(sigma_t, EPS)          // avoid division by zero
//   dist     = max(distance, 0)
//   integral = (1 - exp(-st * dist)) / st // path integral of transmittance
//   L        = max(sigma_s, 0) * max(phase, 0) * max(light_radiance, 0) * integral
//
// The CPU golden owns that logic; this kernel is the on-device twin that runs
// one thread per query.
//
// The CPU golden uses a hand-rolled `exp_approx` (a polynomial, no float
// intrinsic, for cross-target determinism); this kernel mirrors that same
// polynomial exp so the on-device result matches bit-close, rather than calling
// the WGSL `exp` builtin. It uses only the portable core-WGSL subset, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard radiative-transfer single-scatter integral; no Unreal
// Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One single-scatter query. Matches `GpuQuery` in
// `src/analytic_single_scatter.rs`.
struct Query {
    sigma_t: f32,
    sigma_s: f32,
    phase: f32,
    light_radiance: f32,
    distance: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

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

// One single-scatter radiance, mirroring `analytic_single_scatter`.
fn analytic_single_scatter(q: Query) -> f32 {
    let st = max(q.sigma_t, EPS);
    let dist = max(q.distance, 0.0);
    let integral = (1.0 - exp_approx(-st * dist)) / st;
    return max(q.sigma_s, 0.0) * max(q.phase, 0.0) * max(q.light_radiance, 0.0) * integral;
}

@compute @workgroup_size(64)
fn analytic_single_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = analytic_single_scatter(queries[idx]);
}
