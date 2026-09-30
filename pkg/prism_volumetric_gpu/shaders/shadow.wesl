// Cloud self-shadow twin: accumulates optical depth along each light ray and
// returns the surviving Beer-Lambert transmittance, mirroring the CPU golden
// `accumulate_shadow` (in `prism_render_architecture::volumetric::shadow`).
//
// Cloud self-shadowing (design section 12) marches the light ray through the
// density field, summing `max(density, 0) * step` into an optical depth, then
// maps it through Beer-Lambert extinction `exp(-od)` saturated into `[0, 1]`.
// The CPU golden owns that math; this kernel is the on-device twin that runs
// one thread per shadow ray and reproduces the same transmittance.
//
// The extinction is evaluated with the SAME hand-rolled `exp_approx` the
// reference uses (base-two range reduction: a fractional seven-term polynomial
// times an integer power assembled from the float exponent field), NOT the
// device-native `exp`. Mirroring the polynomial keeps the twin bit-close to the
// reference so the parity test can assert a tight tolerance and prove the whole
// extinction algorithm ported, not merely that the shapes roughly agree.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops and
// multiply/add, no `exp`, `pow` or optional feature - so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard Beer-Lambert volumetric shadowing; no Unreal Engine
// source or derived code.

const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    ray_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One shadow ray: where its density run starts, how many samples it spans and
// the march step. Matches `GpuRay` in `src/shadow.rs`.
struct Ray {
    offset: u32,
    sample_count: u32,
    step: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rays: array<Ray>;
@group(0) @binding(2) var<storage, read> densities: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// 2^k for integer k, assembled from the float exponent field, mirroring
// `exp2_int`.
fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return 3.4028235e38 * 3.4028235e38;
    }
    return bitcast<f32>(u32(k + 127) << 23u);
}

// 2^f for f in [0, 1] via the truncated exp(f*ln2) series, mirroring
// `exp2_frac`.
fn exp2_frac(f: f32) -> f32 {
    let c1 = 0.69314718;
    let c2 = 0.2402265;
    let c3 = 0.05550411;
    let c4 = 0.00961813;
    let c5 = 0.00133336;
    let c6 = 0.00015403;
    return 1.0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * (c5 + f * c6)))));
}

// exp via base-two range reduction, mirroring `exp_approx`.
fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
}

// Beer-Lambert transmittance for an accumulated optical depth, mirroring
// `shadow_transmittance`.
fn shadow_transmittance(optical_depth: f32) -> f32 {
    var od = optical_depth;
    if (od <= 0.0) {
        od = 0.0;
    }
    return saturate_s(exp_approx(-od));
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.ray_count) {
        return;
    }
    let ray = rays[idx];
    var step = ray.step;
    if (step <= 0.0) {
        step = 0.0;
    }
    var optical_depth = 0.0;
    var i = 0u;
    while (i < ray.sample_count) {
        let d = densities[ray.offset + i];
        var density = d;
        if (density <= 0.0) {
            density = 0.0;
        }
        optical_depth = optical_depth + density * step;
        i = i + 1u;
    }
    results[idx] = shadow_transmittance(optical_depth);
}
