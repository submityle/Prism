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

// --- temporal history-clamp compute entry ------------------------------------
//
// One invocation per framebuffer texel. Fuses the four NRD ReBLUR/ReLAX
// specular guards (`sdh_specular_accumulation`) into a single exponential
// moving average over the reprojected specular history:
//
//   1. Build the current-frame neighbourhood colour box from a 3x3 gather of
//      the noisy `spec_gi` resolve (mean +/- clamp_sigma * std).
//   2. Clip the reprojected history toward that box (Karis/DICE line clip),
//      reject it by GGX lobe shift (current vs reprojected normal) and by
//      fast/slow luminance divergence, and derive the ReLAX blend weight.
//   3. Blend the clipped history with the new sample, advance the accumulated
//      age under the keep weight, and roll the fast/slow luminance EMAs.
//
// Writes four planes: the accumulated specular+age history and the
// current-frame normal+roughness and fast/slow luminance that become *next*
// frame's reprojection source, plus the denoised specular+confidence the
// spatial pass consumes. The accumulation maths matches the CPU golden
// `history_clamp` helpers op-for-op; the EMA/age/luma bookkeeping around them is
// the caller-side composition the golden documents as the caller's
// responsibility (NRD leaves it to the integrator), so it has no golden twin.

struct SpecDenoiseHistoryClampConfig {
    clamp_sigma: f32,
    lobe_tolerance: f32,
    fast_sensitivity: f32,
    // EMA rates for the fast (responsive) and slow (stable) luminance planes.
    fast_rate: f32,
    slow_rate: f32,
    min_frames: u32,
    max_frames: u32,
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> sdh_config: SpecDenoiseHistoryClampConfig;
// Reprojected specular history: rgb = reprojected radiance, a = reprojection
// confidence (both produced by `spec_denoise_reproject`).
@group(0) @binding(1) var sdh_reprojected: texture_2d<f32>;
// Reprojected history metadata: xyz = prior view-normal * 0.5 + 0.5, w = prior
// accumulated age (float frame count).
@group(0) @binding(2) var sdh_reproj_meta: texture_2d<f32>;
// Reprojected luminance history: r = fast luma, g = slow luma.
@group(0) @binding(3) var sdh_reproj_luma: texture_2d<f32>;
// Current-frame noisy `spec_gi` resolve: rgb = specular radiance, a = ReSTIR
// confidence. Also the source of the neighbourhood colour-box moments.
@group(0) @binding(4) var sdh_resolved: texture_2d<f32>;
// Packed G-buffer: rgb = view normal * 2 - 1, a = perceptual roughness.
@group(0) @binding(5) var sdh_nr: texture_2d<f32>;
// SSR prepass reverse-Z device depth (1.0 = near, 0.0 = background/sky).
@group(0) @binding(6) var sdh_depth: texture_2d<f32>;
// Accumulated specular history out: rgb = radiance, a = updated age (feeds next
// frame's reprojection).
@group(0) @binding(7) var sdh_out_history: texture_storage_2d<rgba16float, write>;
// Current normal+roughness out: xyz = view-normal * 0.5 + 0.5, w = roughness
// (becomes next frame's reprojected `sdh_reproj_meta` normal plane).
@group(0) @binding(8) var sdh_out_meta: texture_storage_2d<rgba16float, write>;
// Updated luminance EMAs out: r = fast luma, g = slow luma.
@group(0) @binding(9) var sdh_out_luma: texture_storage_2d<rgba16float, write>;
// Denoised specular out for the spatial pass: rgb = accumulated radiance,
// a = passthrough confidence.
@group(0) @binding(10) var sdh_out_denoised: texture_storage_2d<rgba16float, write>;

fn sdh_params_from_config() -> SdhParams {
    var p: SdhParams;
    p.clamp_sigma = sdh_config.clamp_sigma;
    p.lobe_tolerance = sdh_config.lobe_tolerance;
    p.fast_sensitivity = sdh_config.fast_sensitivity;
    p.min_frames = sdh_config.min_frames;
    p.max_frames = sdh_config.max_frames;
    return p;
}

// Rec. 709 luminance of a linear-RGB radiance, floored non-negative.
fn sdh_luma(c: vec3<f32>) -> f32 {
    let v = sdh_sanitize_rgb(c);
    return max(dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0);
}

// 3x3 neighbourhood colour moments of the noisy resolve around `pixel`, clamped
// to the framebuffer. Returns mean in `.mean`-equivalent and per-channel std.
struct SdhMoments {
    mean: vec3<f32>,
    std_dev: vec3<f32>,
}

fn sdh_neighbourhood_moments(pixel: vec2<i32>, max_coord: vec2<i32>) -> SdhMoments {
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var sum_sq = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = clamp(
                pixel + vec2<i32>(dx, dy),
                vec2<i32>(0, 0),
                max_coord,
            );
            let c = sdh_sanitize_rgb(textureLoad(sdh_resolved, tap, 0).xyz);
            sum = sum + c;
            sum_sq = sum_sq + c * c;
            count = count + 1.0;
        }
    }
    var out: SdhMoments;
    if count <= 0.0 {
        out.mean = vec3<f32>(0.0, 0.0, 0.0);
        out.std_dev = vec3<f32>(0.0, 0.0, 0.0);
        return out;
    }
    let inv = 1.0 / count;
    let mean = sum * inv;
    // Population variance, floored non-negative before the sqrt.
    let var_rgb = max(sum_sq * inv - mean * mean, vec3<f32>(0.0, 0.0, 0.0));
    out.mean = mean;
    out.std_dev = sqrt(var_rgb);
    return out;
}

@compute @workgroup_size(8, 8, 1)
fn spec_denoise_history_clamp(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sdh_config.width || id.y >= sdh_config.height {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let max_coord = vec2<i32>(i32(sdh_config.width) - 1, i32(sdh_config.height) - 1);

    let resolved = textureLoad(sdh_resolved, pixel, 0);
    let new_sample = sdh_sanitize_rgb(resolved.xyz);
    let confidence = clamp(sdh_sanitize_scalar(resolved.w), 0.0, 1.0);

    let nr = textureLoad(sdh_nr, pixel, 0);
    let curr_normal = sdh_safe_normalize(nr.xyz * 2.0 - 1.0);
    let roughness = clamp(nr.w, 0.0, 1.0);

    let device_depth = textureLoad(sdh_depth, pixel, 0).x;

    // Background / sky: no reflection history. Seed every plane from the current
    // sample at age 0 so a surface drifting into frame starts clean.
    if device_depth <= 0.0 {
        let seed_luma = sdh_luma(new_sample);
        textureStore(sdh_out_history, pixel, vec4<f32>(new_sample, 0.0));
        textureStore(sdh_out_meta, pixel, vec4<f32>(curr_normal * 0.5 + 0.5, roughness));
        textureStore(sdh_out_luma, pixel, vec4<f32>(seed_luma, seed_luma, 0.0, 0.0));
        textureStore(sdh_out_denoised, pixel, vec4<f32>(new_sample, confidence));
        return;
    }

    let reproj = textureLoad(sdh_reprojected, pixel, 0);
    let history = sdh_sanitize_rgb(reproj.xyz);
    let reproj_conf = clamp(sdh_sanitize_scalar(reproj.w), 0.0, 1.0);

    let prior_meta = textureLoad(sdh_reproj_meta, pixel, 0);
    let prev_normal = sdh_safe_normalize(prior_meta.xyz * 2.0 - 1.0);
    let age = u32(max(sdh_sanitize_scalar(prior_meta.w), 0.0) + 0.5);

    let luma = textureLoad(sdh_reproj_luma, pixel, 0);
    let prev_fast = max(sdh_sanitize_scalar(luma.x), 0.0);
    let prev_slow = max(sdh_sanitize_scalar(luma.y), 0.0);

    let moments = sdh_neighbourhood_moments(pixel, max_coord);
    let params = sdh_params_from_config();

    // Trust in the reprojected history gates both the colour clip input and the
    // blend: the reprojection confidence multiplies the resolve confidence.
    let fused_conf = clamp(reproj_conf * confidence, 0.0, 1.0);

    let res = sdh_specular_accumulation(
        history,
        moments.mean,
        moments.std_dev,
        age,
        roughness,
        fused_conf,
        curr_normal,
        prev_normal,
        prev_fast,
        prev_slow,
        params,
    );

    // ReLAX exponential moving average: blend the clipped history toward the new
    // sample by the guard's blend weight (1 = take the sample, 0 = keep history).
    let blend = clamp(res.blend_weight, 0.0, 1.0);
    let accumulated = sdh_sanitize_rgb(mix(res.clamped_history, new_sample, blend));

    // History-keep fraction: how much of the prior accumulation survived the
    // guards this frame. Advancing age by this fraction lets a stable pixel grow
    // its window toward the roughness budget while a rejected pixel resets.
    let keep = clamp(1.0 - blend, 0.0, 1.0);
    let advanced = min(age + 1u, max(res.frame_budget, 1u));
    let new_age = max(f32(advanced) * keep, 0.0);

    // Roll the fast/slow luminance EMAs toward this frame's accumulated luma so
    // the next frame's fast-history divergence detector has fresh references.
    let sample_luma = sdh_luma(accumulated);
    let fast_rate = clamp(sdh_config.fast_rate, 0.0, 1.0);
    let slow_rate = clamp(sdh_config.slow_rate, 0.0, 1.0);
    let new_fast = max(mix(prev_fast, sample_luma, max(fast_rate, blend)), 0.0);
    let new_slow = max(mix(prev_slow, sample_luma, max(slow_rate, blend)), 0.0);

    textureStore(sdh_out_history, pixel, vec4<f32>(accumulated, new_age));
    textureStore(sdh_out_meta, pixel, vec4<f32>(curr_normal * 0.5 + 0.5, roughness));
    textureStore(sdh_out_luma, pixel, vec4<f32>(new_fast, new_slow, 0.0, 0.0));
    textureStore(sdh_out_denoised, pixel, vec4<f32>(accumulated, fused_conf));
}

// --- compile-coverage harness -------------------------------------------------
struct SdhProbe {
    value: vec4<f32>,
}

// The probe lives in a disjoint group so it never collides with the real
// kernel's group-0 resource interface above (same module, two entries).
@group(1) @binding(0) var<storage, read_write> sdh_probe: SdhProbe;

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
