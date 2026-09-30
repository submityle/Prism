// Multiple-scattering energy-gain LUT bake twin: fills the three-axis
// energy-gain table on-device, mirroring the CPU golden
// `MultiScatterLut::build_energy_gain`
// (`prism_render_architecture::volumetric::multiscatter`).
//
// Single-scatter ray-marching loses the energy real clouds redistribute
// through many scattering events (design section 7b). Production engines
// recover it with a small pre-integrated table indexed by view-zenith cosine,
// accumulated optical depth and single-scatter albedo, each cell holding a
// bounded `[0, 1]` energy gain. The CPU golden owns the fill math; the sample
// side already has a device twin (`multiscatter_lut_sample.wesl`). This kernel
// is the on-device bake: one invocation per cell reconstructs that cell's three
// axis coordinates from its flat index and writes the same saturated
// `energy_gain` the reference stores.
//
// The kernel mirrors the reference exactly: the flat index is decomposed with
// the cos axis outermost, then optical_depth, then albedo (matching the golden
// `from_fn` fill order); each axis coordinate comes from the same `axis_value`
// lerp (a single-cell axis collapsing to its minimum); the gain is the product
// of the analytic isotropic diffusion reflectance `(1 - s) / (1 + s)` with
// `s = sqrt(1 - albedo)`, the optical-depth ramp `1 - exp(-depth)`, and the
// albedo-independent octave phase-weight modulation, saturated into `[0, 1]`.
// The octave phase weight reuses the shared octave-scatter decay and the
// Henyey-Greenstein phase already twinned elsewhere. `exp` uses the same
// base-two range-reduction polynomial as `math::exp_approx`; every other op is
// multiply-add plus compare, so this twin runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: Frostbite-style pre-integrated multiple-scattering energy gain
// plus Wrenninge-style octave decay and standard Henyey-Greenstein phase; no
// Unreal Engine source or derived code.

// `core::f32::consts::PI`, matching the reference's `PI`.
const PI: f32 = 3.1415927;
// Isotropic phase normalization `1 / (4*PI)`, matching `scatter::INV_FOUR_PI`.
const INV_FOUR_PI: f32 = 1.0 / (4.0 * PI);
// Small epsilon guarding divisions, matching `math::EPS`.
const EPS: f32 = 1e-6;
// Largest anisotropy magnitude accepted, matching `scatter::MAX_ABS_G`.
const MAX_ABS_G: f32 = 0.999;
// Representative forward eccentricity for the table, matching
// `multiscatter::DEFAULT_LUT_G`.
const DEFAULT_LUT_G: f32 = 0.7;
// log2(e), the base-two range-reduction constant for `exp_approx`.
const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    // Cell count on the view-zenith cosine axis (outermost).
    cos_dim: u32,
    // Cell count on the accumulated optical-depth axis (middle).
    depth_dim: u32,
    // Cell count on the single-scatter albedo axis (innermost).
    albedo_dim: u32,
    // Number of octaves summed by the phase weight.
    octave_count: u32,
    // Per-octave decay of the extinction coefficient (unused here but part of
    // the shared OctaveParams upload), saturated [0, 1].
    attenuation: f32,
    // Per-octave decay of the scattering coefficient, saturated [0, 1].
    contribution: f32,
    // Per-octave decay of the phase eccentricity, saturated [0, 1].
    eccentricity_attenuation: f32,
    // Upper bound of the optical-depth axis (`DEFAULT_MAX_OPTICAL_DEPTH`).
    max_optical_depth: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;

// Clamps `x` into `[lo, hi]`, mirroring `math::clamp`.
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

// `base` raised to the non-negative integer power `exp` by repeated
// multiplication, mirroring `scatter::pow_u` (`pow_u(_, 0) == 1`).
fn pow_u(base: f32, exp: u32) -> f32 {
    var acc = 1.0;
    var i = 0u;
    while (i < exp) {
        acc = acc * base;
        i = i + 1u;
    }
    return acc;
}

// 2^k for integer k, assembled from the float exponent field, mirroring
// `exp2_int`.
fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return bitcast<f32>(0x7f800000u);
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

// exp via base-two range reduction, mirroring `math::exp_approx`.
fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
}

// Normalized Henyey-Greenstein phase, mirroring `scatter::hg_phase`.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let u = clamp_range(cos_theta, -1.0, 1.0);
    let gc = clamp_range(g, -MAX_ABS_G, MAX_ABS_G);
    let g2 = gc * gc;
    let denom = max(1.0 + g2 - 2.0 * gc * u, EPS);
    return INV_FOUR_PI * (1.0 - g2) / (denom * sqrt(denom));
}

// Scattering-weighted average of normalized per-octave phase responses,
// mirroring `multiscatter::octave_phase_weight`. Each octave reuses
// `octave_scatter(1, 1, DEFAULT_LUT_G, n, params)`, of which only the
// scattering coefficient (weight) and eccentricity (phase) are needed.
fn octave_phase_weight(cos_theta: f32) -> f32 {
    var numerator = 0.0;
    var denominator = 0.0;
    // Shared octave-scatter reductions with base sigma_s = 1 and base g clamped.
    let sigma_s0 = max(1.0, 0.0);
    let g0 = clamp_range(DEFAULT_LUT_G, -MAX_ABS_G, MAX_ABS_G);
    let contribution = saturate_v(params.contribution);
    let eccentricity = saturate_v(params.eccentricity_attenuation);
    var last = 0u;
    if (params.octave_count > 0u) {
        last = params.octave_count - 1u;
    }
    var n = 0u;
    while (n < params.octave_count) {
        var i = n;
        if (i > last) {
            i = last;
        }
        let sigma_s = sigma_s0 * pow_u(contribution, i);
        let g = g0 * pow_u(eccentricity, i);
        let phase = hg_phase(cos_theta, g);
        let peak = hg_phase(1.0, g);
        var ratio = 1.0;
        if (peak > EPS) {
            ratio = saturate_v(phase / peak);
        }
        numerator = numerator + sigma_s * ratio;
        denominator = denominator + sigma_s;
        n = n + 1u;
    }
    if (denominator > EPS) {
        return saturate_v(numerator / denominator);
    }
    return 1.0;
}

// Bounded multiple-scatter energy gain for one cell, mirroring
// `multiscatter::energy_gain`.
fn energy_gain(cos_theta: f32, optical_depth: f32, albedo: f32) -> f32 {
    let a = saturate_v(albedo);
    let s = sqrt(saturate_v(1.0 - a));
    var reflectance = 1.0;
    if (1.0 + s > EPS) {
        reflectance = (1.0 - s) / (1.0 + s);
    }
    let depth = max(optical_depth, 0.0);
    let depth_ramp = saturate_v(1.0 - exp_approx(-depth));
    let phase_mod = octave_phase_weight(cos_theta);
    return saturate_v(reflectance * depth_ramp * phase_mod);
}

// Sample value of axis coordinate `i` on a `dim`-cell axis spanning `[mn, mx]`,
// mirroring `multiscatter::axis_value` (single-cell axis collapses to `mn`).
fn axis_value(mn: f32, mx: f32, dim: u32, i: u32) -> f32 {
    if (dim <= 1u) {
        return mn;
    }
    return lerp_v(mn, mx, f32(i) / f32(dim - 1u));
}

@compute @workgroup_size(64)
fn multiscatter_lut_build_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.cos_dim * params.depth_dim * params.albedo_dim;
    if (idx >= total) {
        return;
    }
    // Decompose the flat index with the cos axis outermost, then optical_depth,
    // then albedo, matching the golden `from_fn` fill order.
    let plane = params.depth_dim * params.albedo_dim;
    let ic = idx / plane;
    let rem = idx % plane;
    let id = rem / params.albedo_dim;
    let ia = rem % params.albedo_dim;

    let cos_theta = axis_value(-1.0, 1.0, params.cos_dim, ic);
    let depth = axis_value(0.0, params.max_optical_depth, params.depth_dim, id);
    let albedo = axis_value(0.0, 1.0, params.albedo_dim, ia);

    results[idx] = energy_gain(cos_theta, depth, albedo);
}
