// Contrail diffusion-kernel twin: evaluates on-device the normalized Gaussian
// cross-section of a condensation trail, mirroring the CPU golden
// `contrail_kernel` (in `prism_render_architecture::volumetric::fog`).
//
// The unified volumetric fog (design section 9f) injects the line-shaped
// contrails aircraft leave at altitude as a Gaussian cross-section that widens
// with age. `contrail_kernel` returns the unit-area kernel value at a
// cross-section `offset`:
//   sigma = max(0.5 * max(width, 0) + max(diffusion, 0) * max(age, 0), EPS)
//   T     = 1 / (sigma * sqrt(2*pi)) * e^{-0.5 * (offset/sigma)^2}
// Because the analytic normalization constant is applied, integrating the
// kernel across the full cross-section yields one, so injecting a contrail
// conserves its total mass no matter how far it has diffused. The CPU golden
// `contrail_kernel` owns that math; this kernel is the on-device twin that runs
// one thread per query and reproduces the same value.
//
// The exponential uses the SAME hand-rolled `exp_approx` the reference uses
// (base-two range reduction: a fractional seven-term polynomial times an
// integer power assembled from the float exponent field), NOT the device-native
// `exp`, so the twin stays bit-close to the reference and the parity test can
// assert a tight tolerance. The normalization constant uses the native `sqrt`,
// which both sides already share.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops,
// sqrt and multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard normalized-Gaussian condensation kernel; no Unreal
// Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
// 2*pi, matching `super::math::TWO_PI` on the CPU side.
const TWO_PI: f32 = 6.283185307179586;
// Floor for the effective sigma, matching `super::math::EPS`.
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One contrail-kernel query: the cross-section offset plus the trail's age,
// authored width and diffusion coefficient. Matches `GpuQuery` in
// `src/contrail_kernel.rs`.
struct Query {
    offset: f32,
    age: f32,
    width: f32,
    diffusion: f32,
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

// Effective Gaussian standard deviation of a contrail cross-section, floored at
// EPS so the normalization never divides by (near) zero. Mirrors the CPU
// `effective_sigma` helper.
fn effective_sigma(age: f32, width: f32, diffusion: f32) -> f32 {
    let half_width = 0.5 * max(width, 0.0);
    let spread = max(diffusion, 0.0) * max(age, 0.0);
    return max(half_width + spread, EPS);
}

// Normalized diffusion-kernel value at a cross-section offset, mirroring
// `contrail_kernel`.
fn contrail_kernel(offset: f32, age: f32, width: f32, diffusion: f32) -> f32 {
    let sigma = effective_sigma(age, width, diffusion);
    let norm = 1.0 / (sigma * sqrt(TWO_PI));
    let z = offset / sigma;
    return norm * exp_approx(-0.5 * z * z);
}

@compute @workgroup_size(64)
fn contrail_kernel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = contrail_kernel(q.offset, q.age, q.width, q.diffusion);
}
