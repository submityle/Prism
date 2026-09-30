// Nubis powder dark-edge twin: evaluates the multiple-scattering darkening
// curve seen near dense cloud edges, mirroring the CPU golden `powder` (in
// `prism_render_architecture::volumetric::scatter`).
//
// The powder term fakes the darkening of dense cloud edges (design sections 5
// and 7) with `1 - exp(-2 * density * strength)`, with `density` (view-ray
// optical depth) and `strength` clamped non-negative, saturated into `[0, 1]`.
// The CPU golden owns that math; this kernel is the on-device twin that runs
// one thread per query and reproduces the same value.
//
// As with the shadow twin the extinction uses the SAME hand-rolled `exp_approx`
// the reference uses (base-two range reduction: a fractional seven-term
// polynomial times an integer power assembled from the float exponent field),
// NOT the device-native `exp`, so the twin stays bit-close to the reference and
// the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops and
// multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Nubis-style powder approximation; no Unreal Engine
// source or derived code.

const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One powder query: the view-ray optical depth and the powder strength. Matches
// `GpuQuery` in `src/powder.rs`.
struct Query {
    density: f32,
    strength: f32,
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

// Nubis powder dark-edge term, mirroring `powder`.
fn powder(density_along_view: f32, strength: f32) -> f32 {
    var density = density_along_view;
    if (density <= 0.0) {
        density = 0.0;
    }
    var s = strength;
    if (s <= 0.0) {
        s = 0.0;
    }
    return saturate_s(1.0 - exp_approx(-2.0 * density * s));
}

@compute @workgroup_size(64)
fn powder_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = powder(q.density, q.strength);
}
