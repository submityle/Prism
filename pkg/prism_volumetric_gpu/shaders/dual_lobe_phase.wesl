// Non-Draine dual-lobe cloud-scattering phase twin: evaluates on-device the
// same anisotropic phase the CPU golden `dual_lobe_phase` computes, one thread
// per query.
//
// This is the plain-`Henyey-Greenstein` dual-lobe phase the reference uses
// when the sharper `Draine` shaping is disabled - distinct from the
// `dual_lobe_draine_phase` mirrored in `phase.wesl`. The CPU golden
// `dual_lobe_phase` (in
// `prism_render_architecture::volumetric::scatter`) owns that math; this kernel
// mirrors it exactly:
//
//   forward  = hg_phase(cos_theta, g_forward)
//   backward = hg_phase(cos_theta, g_backward)
//   result   = lerp(backward, forward, saturate(blend))
//
// Every clamp, the `1 / (4*PI)` normalization, and the `denom * sqrt(denom)`
// form of the `^1.5` denominator are reproduced bit-for-bit, so the only
// divergence from the reference is fused-multiply-add contraction. Because a
// convex blend of two normalized phases is itself normalized, this twin also
// satisfies `integral p dOmega = 1`. It uses only the portable core-WGSL subset
// (sqrt/min/max/mul/add) - no exp, no pow, no optional feature - so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Henyey-Greenstein dual-lobe cloud phase; no Unreal
// Engine source or derived code.

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

// One non-Draine dual-lobe phase query. `16`-byte stride matching the
// `bytemuck` upload struct `DualLobePhaseQuery`.
struct Query {
    cos_theta: f32,
    g_forward: f32,
    g_backward: f32,
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

// Non-Draine dual-lobe cloud phase, mirroring `scatter::dual_lobe_phase`. Each
// lobe clamps `cos_theta` independently inside `hg_phase`, exactly as the CPU
// golden does (no single pre-clamp).
fn dual_lobe_phase(cos_theta: f32, g_forward: f32, g_backward: f32, blend: f32) -> f32 {
    let forward = hg_phase(cos_theta, g_forward);
    let backward = hg_phase(cos_theta, g_backward);
    return lerp_v(backward, forward, saturate_v(blend));
}

@compute @workgroup_size(64)
fn dual_lobe_phase_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let q = queries[i];
    values[i] = dual_lobe_phase(q.cos_theta, q.g_forward, q.g_backward, q.blend);
}
