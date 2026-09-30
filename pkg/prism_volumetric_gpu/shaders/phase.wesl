// Dual-lobe cloud-scattering phase twin: evaluates on-device the same
// anisotropic phase the CPU golden `dual_lobe_draine_phase` computes, one
// thread per query.
//
// Cloud single scattering biases light toward the forward direction (the
// silver-lining and glory response). The CPU golden
// `dual_lobe_draine_phase` (in
// `prism_render_architecture::volumetric::scatter`) owns that math; this kernel
// mirrors it exactly:
//
//   backward = hg_phase(cos_theta, g_backward)
//   forward  = hg_draine_phase(cos_theta, g_forward, g_forward, alpha,
//                              1 - saturate(draine_weight))
//   result   = lerp(backward, forward, saturate(blend))
//
// Every clamp, the `1 / (4*PI)` normalization, the HG second-moment Draine
// normalization `1 + alpha*(1 + 2*g^2)/3`, and the `denom * sqrt(denom)` form
// of the `^1.5` denominator are reproduced bit-for-bit, so the only divergence
// from the reference is fused-multiply-add contraction. It uses only the
// portable core-WGSL subset (sqrt/min/max/mul/add) - no exp, no pow, no
// optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Henyey-Greenstein / Draine (Jendersie and d'Eon 2023)
// dual-lobe cloud phase; no Unreal Engine source or derived code.

// `core::f32::consts::PI`, matching the reference's `PI`.
const PI: f32 = 3.1415927;
// Reciprocal of `4*PI`, the isotropic phase and leading normalization factor.
const INV_FOUR_PI: f32 = 1.0 / (4.0 * PI);
// `math::EPS`; guards the `^1.5` denominator away from zero.
const EPS: f32 = 1e-6;
// Largest anisotropy magnitude accepted by the phase functions.
const MAX_ABS_G: f32 = 0.999;

struct Params {
    // Number of valid entries in `queries` / outputs in `values`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One dual-lobe phase query. `24`-byte stride matching the `bytemuck` upload
// struct `PhaseQuery`.
struct Query {
    cos_theta: f32,
    g_forward: f32,
    g_backward: f32,
    alpha: f32,
    draine_weight: f32,
    blend: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> values: array<f32>;

// Clamps `x` into `[lo, hi]`, mirroring `math::clamp` (returns `lo` when the
// bounds are inverted, never `NaN`).
fn clamp_range(x: f32, lo: f32, hi: f32) -> f32 {
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    }
    return x;
}

// Clamps `x` into `[0, 1]`, mirroring `math::saturate`.
fn saturate_v(x: f32) -> f32 {
    return clamp_range(x, 0.0, 1.0);
}

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp` term order.
fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Normalized Henyey-Greenstein phase, mirroring `scatter::hg_phase`.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let u = clamp_range(cos_theta, -1.0, 1.0);
    let gc = clamp_range(g, -MAX_ABS_G, MAX_ABS_G);
    let g2 = gc * gc;
    let denom = max(1.0 + g2 - 2.0 * gc * u, EPS);
    return INV_FOUR_PI * (1.0 - g2) / (denom * sqrt(denom));
}

// Normalized Draine phase, mirroring `scatter::draine_phase`.
fn draine_phase(cos_theta: f32, g: f32, alpha: f32) -> f32 {
    let u = clamp_range(cos_theta, -1.0, 1.0);
    let gc = clamp_range(g, -MAX_ABS_G, MAX_ABS_G);
    let a = max(alpha, 0.0);
    let g2 = gc * gc;
    let denom = max(1.0 + g2 - 2.0 * gc * u, EPS);
    let norm = 1.0 + a * (1.0 + 2.0 * g2) / 3.0;
    return INV_FOUR_PI * (1.0 - g2) * (1.0 + a * u * u) / (norm * denom * sqrt(denom));
}

// Normalized HG-Draine blend, mirroring `scatter::hg_draine_phase`.
fn hg_draine_phase(cos_theta: f32, g_hg: f32, g_draine: f32, alpha: f32, hg_weight: f32) -> f32 {
    let hg = hg_phase(cos_theta, g_hg);
    let draine = draine_phase(cos_theta, g_draine, alpha);
    return lerp_v(draine, hg, saturate_v(hg_weight));
}

// Anisotropic dual-lobe cloud phase, mirroring `scatter::dual_lobe_draine_phase`.
fn dual_lobe_draine_phase(
    cos_theta: f32,
    g_forward: f32,
    g_backward: f32,
    alpha: f32,
    draine_weight: f32,
    blend: f32,
) -> f32 {
    let backward = hg_phase(cos_theta, g_backward);
    let forward = hg_draine_phase(
        cos_theta,
        g_forward,
        g_forward,
        alpha,
        1.0 - saturate_v(draine_weight),
    );
    return lerp_v(backward, forward, saturate_v(blend));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let q = queries[i];
    values[i] = dual_lobe_draine_phase(
        q.cos_theta,
        q.g_forward,
        q.g_backward,
        q.alpha,
        q.draine_weight,
        q.blend,
    );
}
