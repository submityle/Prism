// Compute twin of Prism's hair BSDF lobe-importance pdf
// (prism_render_architecture::hair::scatter_lod::lobe_pdf composed with
// lobe_weights). One thread turns one fibre's optical inputs (fresnel proxy,
// absorption-path proxy) into the normalized R/TT/TRT lobe sampling pdf used to
// importance-sample the Marschner lobes. Every term is a product / rational /
// normalize of non-negative factors — no exp/pow/sin — so the kernel mirrors
// the golden bit-for-bit in structure and matches it within fma tolerance.
//
// The rational transmittance proxy `1 / (1 + absorption)` is a monotone,
// energy-ordering stand-in for Beer-Lambert `exp(-absorption)` (the real
// exponential lives in the shading closure, never here); it is used only to
// rank the lobes, so the proxy form is intentional and not an approximation of
// a transcendental we would otherwise evaluate on the GPU.

struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-fibre optical inputs: x = fresnel proxy, y = absorption-path proxy.
@group(0) @binding(1) var<storage, read> inputs: array<vec2<f32>>;
// Flattened output: three consecutive f32 probabilities per element in lobe
// order (R, TT, TRT); they are non-negative and sum to 1.
@group(0) @binding(2) var<storage, read_write> pdf: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;
// Structural guards mirrored verbatim from the golden source (not material
// magic constants): the all-zero-energy fallback threshold and lobe count.
const EPS: f32 = 1e-6;
const LOBE_COUNT: f32 = 3.0;

// Mirror the golden `is_finite()` test without a float equality on a literal:
// `x == x` rejects NaN, the finite-magnitude bound rejects +/-inf.
fn is_finite_f32(x: f32) -> bool {
    return x == x && x >= -MAX_FINITE_F32 && x <= MAX_FINITE_F32;
}

// Clamp to [0, 1], mapping non-finite inputs to 0 (golden `clamp01`).
fn clamp01(x: f32) -> f32 {
    if is_finite_f32(x) {
        return clamp(x, 0.0, 1.0);
    }
    return 0.0;
}

// Clamp to a finite >= 0 value, non-finite -> 0 (golden `sanitize_nonneg`).
fn sanitize_nonneg(x: f32) -> f32 {
    if is_finite_f32(x) {
        return max(x, 0.0);
    }
    return 0.0;
}

// Monotone rational transmittance proxy 1 / (1 + absorption) (golden
// `transmittance_proxy`).
fn transmittance_proxy(absorption: f32) -> f32 {
    return 1.0 / (1.0 + sanitize_nonneg(absorption));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    let optical = inputs[idx];

    // lobe_weights: R = F, TT = (1-F)^2 * T, TRT = (1-F)^2 * F * T^2.
    let f = clamp01(optical.x);
    let one_minus_f = 1.0 - f;
    let two_transmit = one_minus_f * one_minus_f;
    let t = transmittance_proxy(optical.y);
    let w_r = f;
    let w_tt = two_transmit * t;
    let w_trt = two_transmit * f * (t * t);

    // lobe_pdf: re-sanitize, normalize, uniform 1/3 fallback for a dead stack.
    let r = sanitize_nonneg(w_r);
    let tt = sanitize_nonneg(w_tt);
    let trt = sanitize_nonneg(w_trt);
    let sum = r + tt + trt;

    let base = idx * 3u;
    if sum <= EPS {
        let third = 1.0 / LOBE_COUNT;
        pdf[base + 0u] = third;
        pdf[base + 1u] = third;
        pdf[base + 2u] = third;
        return;
    }
    let inv = 1.0 / sum;
    pdf[base + 0u] = r * inv;
    pdf[base + 1u] = tt * inv;
    pdf[base + 2u] = trt * inv;
}
