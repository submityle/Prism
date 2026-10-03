//! Specular history clamping and accumulation control — the on-device twin of
//! the CPU golden `prism_render_shading::gi::spec_denoise::history_clamp`.
//!
//! Temporal accumulation of a specular signal must be far more cautious than
//! diffuse: a glossy highlight is view-dependent and can change completely in a
//! single frame. This module reproduces the four NVIDIA NRD ReBLUR/ReLAX
//! specular guards: (1) colour-space clipping toward the neighbourhood AABB
//! centre (Karis/DICE line clip) built from mean ± sigma·std moments,
//! (2) roughness-aware history length, (3) lobe-shift rejection comparing the
//! inter-frame normal change against the GGX lobe half-angle, and (4) a
//! fast/slow luminance divergence clamp. `sdh_specular_accumulation` fuses them
//! into one blend weight + clamped colour. Transcribed op-for-op from the
//! golden so `spec_denoise/history_clamp_tests.rs` asserts numeric parity
//! rather than merely compiling.
//!
//! WGSL has no `isFinite`; the golden's guards are reproduced with
//! `(x - x) == 0.0` and the `+Z` normalise fallback. The golden's INFINITY
//! sentinel for a degenerate colour box is reproduced with a `degenerate`
//! flag that returns the box centre, which is numerically identical. The
//! transcendentals `atan`/`acos`/`cos`/`exp` match the golden's `bevy_math::ops`
//! to tolerance.

const SDH_MIN_ALPHA: f32 = 1.0e-3;
const SDH_PI: f32 = 3.14159265358979323846;

struct SdhParams {
    clamp_sigma: f32,
    lobe_tolerance: f32,
    fast_sensitivity: f32,
    min_frames: u32,
    max_frames: u32,
}

struct SdhAabb {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

struct SdhClampResult {
    clamped_history: vec3<f32>,
    blend_weight: f32,
    frame_budget: u32,
}

// --- finite guards ------------------------------------------------------------

fn sdh_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

fn sdh_finite_or_zero(x: f32) -> f32 {
    if sdh_is_finite(x) {
        return x;
    }
    return 0.0;
}

fn sdh_sanitize_scalar(x: f32) -> f32 {
    return sdh_finite_or_zero(x);
}

fn sdh_sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    let v = vec3<f32>(
        sdh_finite_or_zero(c.x),
        sdh_finite_or_zero(c.y),
        sdh_finite_or_zero(c.z),
    );
    return max(v, vec3<f32>(0.0, 0.0, 0.0));
}

fn sdh_safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if sdh_is_finite(len_sq) && len_sq > 1.0e-24 {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn sdh_stable_exp(x: f32) -> f32 {
    if !sdh_is_finite(x) {
        return 0.0;
    }
    return exp(clamp(x, -80.0, 0.0));
}

/// Perceptual roughness -> isotropic GGX alpha (= roughness^2), clamped and
/// floored at MIN_ALPHA. Mirrors `roughness_to_alpha`.
fn sdh_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SDH_MIN_ALPHA);
}

// --- colour-space clipping ----------------------------------------------------

/// Build a colour AABB `(min, max) = mean ± sigma·std`, clamped non-negative.
fn sdh_variance_aabb(mean: vec3<f32>, std_dev: vec3<f32>, sigma: f32) -> SdhAabb {
    let m = sdh_sanitize_rgb(mean);
    let s = sdh_sanitize_rgb(std_dev) * max(sigma, 0.0);
    var box: SdhAabb;
    box.lo = max(m - s, vec3<f32>(0.0, 0.0, 0.0));
    box.hi = m + s;
    return box;
}

/// Clip `history` into `[aabb_min, aabb_max]` along the line toward the centre
/// (Karis/DICE clip-to-AABB). Mirrors `clip_history_aabb`.
fn sdh_clip_history_aabb(history: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let h = sdh_sanitize_rgb(history);
    let lo = sdh_sanitize_rgb(aabb_min);
    let hi = sdh_sanitize_rgb(aabb_max);
    let center = (lo + hi) * 0.5;
    let extent = (hi - lo) * 0.5;
    let offset = h - center;

    var max_ratio = 0.0;
    var degenerate = false;

    if extent.x <= 1.0e-12 {
        if abs(offset.x) > 1.0e-12 {
            degenerate = true;
        }
    } else {
        max_ratio = max(max_ratio, abs(offset.x) / extent.x);
    }
    if extent.y <= 1.0e-12 {
        if abs(offset.y) > 1.0e-12 {
            degenerate = true;
        }
    } else {
        max_ratio = max(max_ratio, abs(offset.y) / extent.y);
    }
    if extent.z <= 1.0e-12 {
        if abs(offset.z) > 1.0e-12 {
            degenerate = true;
        }
    } else {
        max_ratio = max(max_ratio, abs(offset.z) / extent.z);
    }

    if degenerate {
        return center;
    }
    if max_ratio <= 1.0 {
        return h;
    }
    return center + offset / max_ratio;
}

// --- accumulation scheduling --------------------------------------------------

/// Accumulated-frame budget for a roughness, in `[min_frames, max_frames]`.
fn sdh_roughness_history_length(roughness: f32, params: SdhParams) -> u32 {
    let r = clamp(roughness, 0.0, 1.0);
    let lo = max(params.min_frames, 1u);
    let hi = max(params.max_frames, lo);
    let span = f32(hi - lo);
    let frames = f32(lo) + span * r;
    let rounded = u32(frames + 0.5);
    return clamp(rounded, lo, hi);
}

/// GGX specular lobe half-angle (radians): `atan(alpha)` floored positive.
fn sdh_lobe_half_angle(roughness: f32) -> f32 {
    let alpha = sdh_roughness_to_alpha(roughness);
    return max(atan(alpha), 1.0e-4);
}

/// History-keep weight in `[0, 1]` from the inter-frame normal shift.
fn sdh_lobe_rejection(
    normal_curr: vec3<f32>,
    normal_prev: vec3<f32>,
    roughness: f32,
    params: SdhParams,
) -> f32 {
    let a = sdh_safe_normalize(normal_curr);
    let b = sdh_safe_normalize(normal_prev);
    let cos_t = clamp(dot(a, b), -1.0, 1.0);
    let angle = acos(cos_t);
    let window = sdh_lobe_half_angle(roughness) * max(params.lobe_tolerance, 1.0e-4);
    if angle <= 0.0 {
        return 1.0;
    }
    if angle >= window {
        return 0.0;
    }
    let t = clamp(angle / window, 0.0, 1.0);
    return clamp(0.5 * (1.0 + cos(SDH_PI * t)), 0.0, 1.0);
}

/// Divergence weight in `[0, 1]` between a fast and slow luminance history.
fn sdh_fast_history_factor(fast_luma: f32, slow_luma: f32, params: SdhParams) -> f32 {
    let f = max(sdh_sanitize_scalar(fast_luma), 0.0);
    let s = max(sdh_sanitize_scalar(slow_luma), 0.0);
    let diff = abs(f - s);
    let denom = (f + s) * 0.5 + 1.0e-4;
    let rel = diff / denom;
    return sdh_stable_exp(-max(params.fast_sensitivity, 0.0) * rel);
}

/// Blend weight for the new sample (ReLAX alpha), in `[0, 1]`.
fn sdh_specular_blend_weight(
    age: u32,
    frame_budget: u32,
    confidence: f32,
    lobe_keep: f32,
    fast_keep: f32,
) -> f32 {
    let budget = max(frame_budget, 1u);
    let effective = min(age + 1u, budget);
    let base = 1.0 / f32(effective);
    let keep = clamp(sdh_sanitize_scalar(confidence), 0.0, 1.0)
        * clamp(sdh_sanitize_scalar(lobe_keep), 0.0, 1.0)
        * clamp(sdh_sanitize_scalar(fast_keep), 0.0, 1.0);
    let alpha = 1.0 - keep * (1.0 - base);
    return clamp(alpha, 0.0, 1.0);
}

/// Full specular history guard for one pixel. Mirrors `specular_accumulation`.
fn sdh_specular_accumulation(
    history: vec3<f32>,
    neighbourhood_mean: vec3<f32>,
    neighbourhood_std: vec3<f32>,
    age: u32,
    roughness: f32,
    confidence: f32,
    normal_curr: vec3<f32>,
    normal_prev: vec3<f32>,
    fast_luma: f32,
    slow_luma: f32,
    params: SdhParams,
) -> SdhClampResult {
    let box = sdh_variance_aabb(neighbourhood_mean, neighbourhood_std, params.clamp_sigma);
    let clamped = sdh_clip_history_aabb(history, box.lo, box.hi);
    let frame_budget = sdh_roughness_history_length(roughness, params);
    let lobe_keep = sdh_lobe_rejection(normal_curr, normal_prev, roughness, params);
    let fast_keep = sdh_fast_history_factor(fast_luma, slow_luma, params);
    let blend = sdh_specular_blend_weight(age, frame_budget, confidence, lobe_keep, fast_keep);

    var out: SdhClampResult;
    out.clamped_history = sdh_sanitize_rgb(clamped);
    out.blend_weight = blend;
    out.frame_budget = frame_budget;
    return out;
}

// --- compile-coverage harness -------------------------------------------------
struct SdhProbe {
    value: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> sdh_probe: SdhProbe;

@compute @workgroup_size(1, 1, 1)
fn spec_denoise_history_clamp_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    var params: SdhParams;
    params.clamp_sigma = 2.0;
    params.lobe_tolerance = 2.0;
    params.fast_sensitivity = 4.0;
    params.min_frames = 2u;
    params.max_frames = 32u;

    let up = vec3<f32>(0.0, 0.0, 1.0);
    let res = sdh_specular_accumulation(
        vec3<f32>(0.6, 0.5, 0.4),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.1, 0.1, 0.1),
        4u,
        0.4,
        0.8,
        up,
        vec3<f32>(0.1, 0.0, 1.0),
        0.5,
        0.45,
        params,
    );

    let budget = sdh_roughness_history_length(0.5, params);
    let lobe = sdh_lobe_half_angle(0.5);
    let fast = sdh_fast_history_factor(0.5, 0.45, params);
    let blend = sdh_specular_blend_weight(4u, budget, 0.8, 0.9, fast);
    let box = sdh_variance_aabb(vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.1, 0.1, 0.1), 2.0);
    let clip = sdh_clip_history_aabb(vec3<f32>(2.0, 0.5, 0.5), box.lo, box.hi);

    sdh_probe.value = vec4<f32>(
        res.clamped_history + clip,
        res.blend_weight + f32(res.frame_budget) + f32(budget) + lobe + blend,
    );
}
