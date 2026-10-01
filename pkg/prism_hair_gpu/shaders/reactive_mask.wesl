// Compute twin of Prism's fibre-aware temporal reactive-mask kernel
// (prism_render_architecture::hair::reactive_mask::reactivity_map, i.e.
// pixel_reactivity mapped over a span). One thread turns one pixel's
// (coverage, screen_velocity, depth_delta) triple into the TAA/TSR reactivity
// value in [0, max_reactivity]: a weighted sum of three monotone ramps —
// (1 - coverage), the velocity saturation ramp v/(v+k) and the depth saturation
// ramp d/(d+k) — clamped into range. Only +,-,*,/ and clamp are used (no
// exp/pow/sin), so the kernel mirrors the golden structurally and matches it
// within fma tolerance.
//
// The per-pixel weights and the max-reactivity clamp are uploaded raw as a
// uniform; the shader sanitizes them in-place bit-faithfully to the golden's
// `sanitized()` so the host never pre-folds the policy. The two saturation
// constants are mirrored verbatim from the golden as structural numbers.

struct Params {
    coverage_weight: f32,
    velocity_weight: f32,
    depth_weight: f32,
    max_reactivity: f32,
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-pixel inputs: x = coverage, y = screen_velocity, z = depth_delta (w pad).
@group(0) @binding(1) var<storage, read> inputs: array<vec4<f32>>;
// One reactivity value per pixel in [0, max_reactivity].
@group(0) @binding(2) var<storage, read_write> reactivity: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;
// Structural saturation constants mirrored verbatim from the golden source
// (not tunable magic): velocity reaches 0.5 at 2 px/frame, depth at 0.1.
const VELOCITY_SATURATION: f32 = 2.0;
const DEPTH_SATURATION: f32 = 0.1;

// Mirror the golden `is_finite()` test without a float equality on a literal:
// `x == x` rejects NaN, the finite-magnitude bound rejects +/-inf.
fn is_finite_f32(x: f32) -> bool {
    return x == x && x >= -MAX_FINITE_F32 && x <= MAX_FINITE_F32;
}

// Golden `sanitize_weight`: finite and strictly positive, else 0.
fn sanitize_weight(w: f32) -> f32 {
    if is_finite_f32(w) && w > 0.0 {
        return w;
    }
    return 0.0;
}

// Golden `sanitize_coverage`: clamp to [0,1]; non-finite -> fully covered (1).
fn sanitize_coverage(c: f32) -> f32 {
    if is_finite_f32(c) {
        return clamp(c, 0.0, 1.0);
    }
    return 1.0;
}

// Golden `sanitize_magnitude`: non-finite -> 0, else |v|.
fn sanitize_magnitude(v: f32) -> f32 {
    if is_finite_f32(v) {
        return abs(v);
    }
    return 0.0;
}

// Golden `max_reactivity` guard: clamp to [0,1]; non-finite -> 1.
fn sanitize_max(m: f32) -> f32 {
    if is_finite_f32(m) {
        return clamp(m, 0.0, 1.0);
    }
    return 1.0;
}

// Golden `saturation_ramp` v/(v+k): 0 at v=0, 0.5 at v=k, ->1 as v grows.
fn saturation_ramp(v: f32, k: f32) -> f32 {
    return v / (v + k);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    let pixel = inputs[idx];

    let cw = sanitize_weight(params.coverage_weight);
    let vw = sanitize_weight(params.velocity_weight);
    let dw = sanitize_weight(params.depth_weight);
    let max_reactivity = sanitize_max(params.max_reactivity);

    let coverage = sanitize_coverage(pixel.x);
    let velocity = sanitize_magnitude(pixel.y);
    let depth = sanitize_magnitude(pixel.z);

    let coverage_term = 1.0 - coverage;
    let velocity_term = saturation_ramp(velocity, VELOCITY_SATURATION);
    let depth_term = saturation_ramp(depth, DEPTH_SATURATION);

    let raw = cw * coverage_term + vw * velocity_term + dw * depth_term;
    reactivity[idx] = clamp(raw, 0.0, max_reactivity);
}
