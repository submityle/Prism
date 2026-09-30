// Beer-Lambert transmittance twin: evaluates the closed-form transmittance of a
// homogeneous medium on-device, mirroring the CPU golden
// `analytic_transmittance` (in `prism_render_architecture::volumetric::reference`).
//
// This is the analytic value the delta/ratio tracking estimators must converge
// to (design section 9c reference). It returns `exp(-sigma_t * distance)`, with
// negative `sigma_t` and `distance` clamped to zero and the result saturated
// into [0, 1], so `distance == 0` yields exactly 1 and the value is never
// outside the physical range:
//   optical_depth = max(sigma_t, 0) * max(distance, 0)
//   transmittance = saturate(exp(-optical_depth))
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
// Provenance: standard Beer-Lambert transmittance; no Unreal Engine source or
// derived code.

const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One transmittance query: the extinction coefficient and the path distance.
// Matches `GpuQuery` in `src/analytic_transmittance.rs`.
struct Query {
    sigma_t: f32,
    distance: f32,
    pad0: u32,
    pad1: u32,
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

// One Beer-Lambert transmittance, mirroring `analytic_transmittance`.
fn analytic_transmittance(q: Query) -> f32 {
    let optical_depth = max(q.sigma_t, 0.0) * max(q.distance, 0.0);
    return saturate_s(exp_approx(-optical_depth));
}

@compute @workgroup_size(64)
fn analytic_transmittance_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = analytic_transmittance(queries[idx]);
}
