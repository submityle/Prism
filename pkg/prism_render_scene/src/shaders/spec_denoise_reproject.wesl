//! Specular denoiser — virtual-reflection history reprojection, the on-device
//! twin of the CPU golden `prism_render_shading::gi::spec_denoise::reproject`.
//!
//! A glossy highlight is the image of a point sitting `hit_distance` behind the
//! reflector along the reflected view ray (its "virtual" world position).
//! Reprojecting a specular history buffer from the surface position smears the
//! reflection into the classic lag trail as the camera moves; production
//! specular denoisers (NVIDIA NRD ReBLUR / ReLAX specular) instead reproject
//! along this virtual point, apply a parallax correction derived from the
//! camera's motion relative to that virtual image, and modulate the trust by
//! surface roughness. This module is the backend-neutral, deterministic twin of
//! that stage, transcribed op-for-op from the golden so
//! `spec_denoise/reproject_tests.rs` asserts numeric parity rather than merely
//! compiling.
//!
//! Conventions: world space is right-handed `f32`; `view_dir` points from the
//! surface toward the camera and is normalised defensively on entry;
//! `roughness` is perceptual (Disney). WGSL has no `isFinite`, so the golden's
//! finite guards are reproduced with `(x - x) == 0.0`, and every degenerate
//! input (zero-length vectors, non-finite depths, negative distances) falls
//! back to a safe identity instead of emitting NaN. Transcendentals use the
//! WGSL built-ins `pow`/`log`(natural)/`exp`, matching the golden's
//! `bevy_math::ops` to tolerance.

/// Tunables for the specular reprojection trust model. Mirrors `ReprojectParams`
/// (defaults: parallax_sensitivity = 8, virtual_exponent = 2, min_confidence =
/// 0), passed explicitly here because WESL has no `Default`.
struct SdrParams {
    parallax_sensitivity: f32,
    virtual_exponent: f32,
    min_confidence: f32,
}

/// Everything the reprojection stage produces for one pixel. Mirrors
/// `Reprojection`.
struct SdrReprojection {
    sample_position: vec3<f32>,
    virtual_point: vec3<f32>,
    parallax: f32,
    confidence: f32,
}

// --- finite guards (WGSL has no isFinite) -------------------------------------

fn sdr_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

fn sdr_finite_or_zero(x: f32) -> f32 {
    if sdr_is_finite(x) {
        return x;
    }
    return 0.0;
}

fn sdr_sanitize_scalar(x: f32) -> f32 {
    return sdr_finite_or_zero(x);
}

fn sdr_sanitize_vec(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sdr_finite_or_zero(v.x),
        sdr_finite_or_zero(v.y),
        sdr_finite_or_zero(v.z),
    );
}

/// Normalise `v`, falling back to `+Z` for zero-length / non-finite input.
fn sdr_safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if sdr_is_finite(len_sq) && len_sq > 1.0e-24 {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

/// Numerically safe `exp` on `(-inf, 0]`; never returns NaN / +inf.
fn sdr_stable_exp(x: f32) -> f32 {
    if !sdr_is_finite(x) {
        return 0.0;
    }
    return exp(clamp(x, -80.0, 0.0));
}

// --- reprojection maths -------------------------------------------------------

/// Reflect incident direction `i` about unit normal `n`: `i - 2 (i·n) n`.
fn sdr_reflect(i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let nn = sdr_safe_normalize(n);
    return i - nn * (2.0 * dot(i, nn));
}

/// Fraction of the full virtual-hit-distance parallax at a given roughness in
/// `[0, 1]`: `(1 - roughness)^virtual_exponent`.
fn sdr_virtual_history_amount(roughness: f32, params: SdrParams) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    let e = max(params.virtual_exponent, 0.0);
    return clamp(pow(max(1.0 - r, 0.0), e), 0.0, 1.0);
}

/// NRD/Frostbite specular dominant factor in `[0, 1]` (Lazarov closed form).
fn sdr_specular_dominant_factor(n_dot_v: f32, roughness: f32) -> f32 {
    let ndv = clamp(n_dot_v, 0.0, 1.0);
    let r = clamp(roughness, 0.0, 1.0);
    let a = 0.298475 * log(39.4115 - 39.0029 * r);
    let f = pow(max(1.0 - ndv, 0.0), 10.8649) * (1.0 - a) + a;
    return clamp(f, 0.0, 1.0);
}

/// World-space virtual reflection point used to reproject specular history.
fn sdr_virtual_reflection_point(
    surface_pos: vec3<f32>,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    hit_distance: f32,
    roughness: f32,
    params: SdrParams,
) -> vec3<f32> {
    let v = sdr_safe_normalize(view_dir);
    let n = sdr_safe_normalize(normal);
    let r = sdr_reflect(-v, n);
    let dist = max(sdr_sanitize_scalar(hit_distance), 0.0);
    let amount = sdr_virtual_history_amount(roughness, params);
    return sdr_sanitize_vec(surface_pos) + r * (dist * amount);
}

/// Tangent of the angle subtended at `point` by the camera's inter-frame motion
/// — the ReBLUR specular parallax measure (always `>= 0`).
fn sdr_view_parallax(prev_cam_pos: vec3<f32>, curr_cam_pos: vec3<f32>, point: vec3<f32>) -> f32 {
    let p = sdr_sanitize_vec(point);
    let a = sdr_sanitize_vec(curr_cam_pos) - p;
    let b = sdr_sanitize_vec(prev_cam_pos) - p;
    let la = length(a);
    let lb = length(b);
    if la <= 1.0e-12 || lb <= 1.0e-12 {
        return 0.0;
    }
    let cos_t = clamp(dot(a, b) / (la * lb), -1.0, 1.0);
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let tan_t = sin_t / max(abs(cos_t), 1.0e-4);
    if sdr_is_finite(tan_t) {
        return min(tan_t, 1.0e4);
    }
    return 1.0e4;
}

/// Reprojection confidence in `[min_confidence, 1]` for the virtual history.
fn sdr_reprojection_confidence(roughness: f32, parallax: f32, params: SdrParams) -> f32 {
    let px = max(sdr_sanitize_scalar(parallax), 0.0);
    let amount = sdr_virtual_history_amount(roughness, params);
    let sensitivity = max(params.parallax_sensitivity, 0.0) * amount;
    let conf = sdr_stable_exp(-sensitivity * px);
    let floor_c = clamp(params.min_confidence, 0.0, 1.0);
    return clamp(max(conf, floor_c), 0.0, 1.0);
}

/// Blend the surface-motion reprojection with the virtual-motion reprojection
/// by the shared roughness schedule.
fn sdr_blend_reprojected_position(
    surface_reproj: vec3<f32>,
    virtual_reproj: vec3<f32>,
    roughness: f32,
    params: SdrParams,
) -> vec3<f32> {
    let amount = sdr_virtual_history_amount(roughness, params);
    let s = sdr_sanitize_vec(surface_reproj);
    let v = sdr_sanitize_vec(virtual_reproj);
    return sdr_sanitize_vec(mix(s, v, vec3<f32>(amount)));
}

/// Full specular reprojection for one pixel. Mirrors `reproject_specular`.
fn sdr_reproject_specular(
    surface_pos: vec3<f32>,
    surface_reproj: vec3<f32>,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    hit_distance: f32,
    roughness: f32,
    prev_cam_pos: vec3<f32>,
    curr_cam_pos: vec3<f32>,
    params: SdrParams,
) -> SdrReprojection {
    let virtual_point = sdr_virtual_reflection_point(
        surface_pos,
        view_dir,
        normal,
        hit_distance,
        roughness,
        params,
    );
    let parallax = sdr_view_parallax(prev_cam_pos, curr_cam_pos, virtual_point);
    let confidence = sdr_reprojection_confidence(roughness, parallax, params);
    let sample_position = sdr_blend_reprojected_position(surface_reproj, virtual_point, roughness, params);
    var out: SdrReprojection;
    out.sample_position = sample_position;
    out.virtual_point = virtual_point;
    out.parallax = parallax;
    out.confidence = confidence;
    return out;
}

// --- real reprojection pass ---------------------------------------------------
// The scheduled entry point. One invocation per framebuffer texel reconstructs
// the pixel's world-space surface position from the SSR prepass reverse-Z depth,
// forms the virtual reflection point and its roughness-weighted reprojection
// confidence (via the parity-tested helpers above), reprojects the surface into
// the previous frame to fetch the specular history, and rejects disoccluded
// taps by a world-space distance guard. It writes the reprojected history
// (`rgb`) plus the reprojection confidence (`a`) the history-clamp pass blends.
//
// The specular history buffer is resident per surface pixel, so the history is
// fetched at the *surface* reprojection; the virtual-point parallax enters only
// through `reproj.confidence`, which down-weights fast-moving glossy highlights
// exactly as the golden `reproject_specular` prescribes. The group-0 contract
// is frozen here and mirrored by the Rust bind-group layout that drives it.

struct SpecDenoiseReprojectConfig {
    // Current camera: clip (NDC, reverse-Z) -> world, to rebuild the surface.
    world_from_clip: mat4x4<f32>,
    // Previous camera: world -> clip (NDC), to find the history texel.
    prev_clip_from_world: mat4x4<f32>,
    // Previous camera: clip (NDC) -> world, to rebuild the stored surface for
    // the world-space disocclusion guard.
    prev_world_from_clip: mat4x4<f32>,
    // Current camera: view -> world rotation, lifting the view-space G-buffer
    // normal into the world space the reprojection maths runs in.
    world_from_view: mat4x4<f32>,
    curr_cam_pos: vec3<f32>,
    parallax_sensitivity: f32,
    prev_cam_pos: vec3<f32>,
    virtual_exponent: f32,
    width: u32,
    height: u32,
    min_confidence: f32,
    // Relative world-space disocclusion tolerance (fraction of the view
    // distance): a reprojected tap is rejected once the stored prev surface has
    // drifted further than `depth_rejection * dist(cam, surface)`.
    depth_rejection: f32,
}

@group(0) @binding(0) var<uniform> sdr_config: SpecDenoiseReprojectConfig;
// Packed G-buffer: rgb = view normal * 2 - 1, a = perceptual roughness.
@group(0) @binding(1) var sdr_nr: texture_2d<f32>;
// SSR trace per-pixel hit distance (view-space units) in the red channel.
@group(0) @binding(2) var sdr_hit: texture_2d<f32>;
// SSR prepass reverse-Z device depth (1.0 = near, 0.0 = background/sky).
@group(0) @binding(3) var sdr_depth: texture_2d<f32>;
// Previous frame's accumulated specular history: rgb = radiance, a = age/conf.
@group(0) @binding(4) var sdr_prev_history: texture_2d<f32>;
// Previous frame's reverse-Z device depth, for the disocclusion guard.
@group(0) @binding(5) var sdr_prev_depth: texture_2d<f32>;
// Write-only reprojected history (rgb) + reprojection confidence (a).
@group(0) @binding(6) var sdr_reprojected: texture_storage_2d<rgba16float, write>;

fn sdr_params_from_config() -> SdrParams {
    var p: SdrParams;
    p.parallax_sensitivity = sdr_config.parallax_sensitivity;
    p.virtual_exponent = sdr_config.virtual_exponent;
    p.min_confidence = sdr_config.min_confidence;
    return p;
}

// Reconstruct the current-frame world-space position from framebuffer UV +
// reverse-Z device depth through the current `world_from_clip`.
fn sdr_reconstruct_world_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let world = sdr_config.world_from_clip * vec4<f32>(ndc, 1.0);
    return world.xyz / world.w;
}

// Reconstruct a previous-frame world position from a prev-frame UV + prev
// reverse-Z device depth through `prev_world_from_clip` (disocclusion guard).
fn sdr_reconstruct_prev_world_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let world = sdr_config.prev_world_from_clip * vec4<f32>(ndc, 1.0);
    return world.xyz / world.w;
}

// Project a world position into the previous frame, returning `(uv, device_z)`.
// Points behind the previous camera (`w <= 0`) return an out-of-range uv so the
// caller treats them as invalid.
fn sdr_project_prev(world_pos: vec3<f32>) -> vec3<f32> {
    let clip = sdr_config.prev_clip_from_world * vec4<f32>(world_pos, 1.0);
    if clip.w <= 1.0e-6 {
        return vec3<f32>(-1.0, -1.0, -1.0);
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    return vec3<f32>(uv, ndc.z);
}

@compute @workgroup_size(8, 8, 1)
fn spec_denoise_reproject(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sdr_config.width || id.y >= sdr_config.height {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let extent = vec2<f32>(f32(sdr_config.width), f32(sdr_config.height));
    let max_coord = vec2<i32>(i32(sdr_config.width) - 1, i32(sdr_config.height) - 1);

    let device_depth = textureLoad(sdr_depth, pixel, 0).x;
    if device_depth <= 0.0 {
        // Background / sky: no reflection history to reproject.
        textureStore(sdr_reprojected, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / extent;
    let world_pos = sdr_reconstruct_world_position(uv, device_depth);

    let nr = textureLoad(sdr_nr, pixel, 0);
    let view_normal = nr.xyz * 2.0 - 1.0;
    let world_normal = sdr_safe_normalize(
        (sdr_config.world_from_view * vec4<f32>(view_normal, 0.0)).xyz
    );
    let roughness = clamp(nr.w, 0.0, 1.0);
    let hit = textureLoad(sdr_hit, pixel, 0).x;

    let view_dir = sdr_safe_normalize(sdr_config.curr_cam_pos - world_pos);
    let params = sdr_params_from_config();

    // Virtual-reflection reprojection: the roughness-weighted virtual point and
    // its parallax drive the confidence; the surface position anchors the fetch.
    let reproj = sdr_reproject_specular(
        world_pos,
        world_pos,
        view_dir,
        world_normal,
        hit,
        roughness,
        sdr_config.prev_cam_pos,
        sdr_config.curr_cam_pos,
        params,
    );

    let prev = sdr_project_prev(world_pos);
    let in_bounds = prev.x >= 0.0 && prev.x <= 1.0 && prev.y >= 0.0 && prev.y <= 1.0;

    var history = vec3<f32>(0.0, 0.0, 0.0);
    var confidence = 0.0;
    if in_bounds {
        let prev_px = clamp(
            vec2<i32>(i32(prev.x * extent.x), i32(prev.y * extent.y)),
            vec2<i32>(0, 0),
            max_coord,
        );
        let prev_depth = textureLoad(sdr_prev_depth, prev_px, 0).x;
        if prev_depth > 0.0 {
            let prev_uv = (vec2<f32>(prev_px) + vec2<f32>(0.5)) / extent;
            let prev_world = sdr_reconstruct_prev_world_position(prev_uv, prev_depth);
            let drift = length(prev_world - world_pos);
            let scale = max(length(world_pos - sdr_config.curr_cam_pos), 1.0e-3);
            let tol = max(sdr_config.depth_rejection, 1.0e-4) * scale;
            if drift <= tol {
                let ph = textureLoad(sdr_prev_history, prev_px, 0);
                history = sdr_sanitize_vec(ph.xyz);
                confidence = reproj.confidence;
            }
        }
    }

    textureStore(sdr_reprojected, pixel, vec4<f32>(history, confidence));
}

// --- compile-coverage harness -------------------------------------------------
// A trivial entry point that folds every public helper into one output so the
// module validates standalone through the render world's `ShaderCache` (naga
// would otherwise prune the functions as dead code). Not scheduled.
struct SdrProbe {
    value: vec4<f32>,
}

@group(1) @binding(0) var<storage, read_write> sdr_probe: SdrProbe;

@compute @workgroup_size(1, 1, 1)
fn spec_denoise_reproject_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    var params: SdrParams;
    params.parallax_sensitivity = 8.0;
    params.virtual_exponent = 2.0;
    params.min_confidence = 0.0;

    let surface = vec3<f32>(0.0, 0.0, 0.0);
    let view = vec3<f32>(0.1, 0.0, 0.9);
    let n = vec3<f32>(0.0, 0.0, 1.0);
    let r = sdr_reproject_specular(
        surface,
        vec3<f32>(0.05, 0.0, 0.0),
        view,
        n,
        2.0,
        0.3,
        vec3<f32>(0.0, 0.0, 5.0),
        vec3<f32>(0.3, 0.0, 5.0),
        params,
    );
    let dom = sdr_specular_dominant_factor(0.8, 0.2);
    let refl = sdr_reflect(-sdr_safe_normalize(view), sdr_safe_normalize(n));
    sdr_probe.value = vec4<f32>(r.sample_position + r.virtual_point + refl, r.parallax + r.confidence + dom);
}
