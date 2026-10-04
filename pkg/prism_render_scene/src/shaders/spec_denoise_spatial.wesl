//! Specular denoiser — hit-distance-driven spatial filter, the on-device twin
//! of the CPU golden `prism_render_shading::gi::spec_denoise::spatial`.
//!
//! After temporal accumulation a specular signal still needs a geometry-aware
//! spatial pass (NVIDIA NRD ReBLUR specular): roughness sets the base kernel
//! size (a mirror stays razor sharp, a rough surface tolerates a wide blur),
//! hit distance performs contact hardening (near-surface reflections stay
//! sharp), the footprint is elongated anisotropically at grazing angles, and
//! bilateral edge-stops (plane distance, normal, roughness) keep the blur from
//! crossing geometry/material breaks. Transcribed op-for-op from the golden so
//! `spec_denoise/spatial_tests.rs` asserts numeric parity rather than merely
//! compiling.
//!
//! WGSL has no `isFinite`; the golden's guards are reproduced with
//! `(x - x) == 0.0` and the `+Z` normalise fallback. Transcendentals use the
//! WGSL built-ins `pow`/`exp`, matching the golden's `bevy_math::ops` to
//! tolerance. The on-device `sds_spatial_filter` gathers a bounded neighbour
//! array (device kernels tile the footprint); the parity harness drives the
//! same accumulation over an arbitrary-length slice.

const SDS_MIN_ALPHA: f32 = 1.0e-3;
const SDS_MAX_TAPS: u32 = 8u;

struct SdsParams {
    max_radius: f32,
    phi_depth: f32,
    phi_normal: f32,
    phi_roughness: f32,
    contact_hardening: f32,
    max_anisotropy: f32,
}

struct SdsTap {
    color: vec3<f32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    normalized_hit: f32,
}

struct SdsResult {
    color: vec3<f32>,
    normalized_hit: f32,
}

// --- finite guards ------------------------------------------------------------

fn sds_is_finite(x: f32) -> bool {
    return (x - x) == 0.0;
}

fn sds_finite_or_zero(x: f32) -> f32 {
    if sds_is_finite(x) {
        return x;
    }
    return 0.0;
}

fn sds_sanitize_scalar(x: f32) -> f32 {
    return sds_finite_or_zero(x);
}

fn sds_sanitize_vec(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        sds_finite_or_zero(v.x),
        sds_finite_or_zero(v.y),
        sds_finite_or_zero(v.z),
    );
}

fn sds_sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    return max(sds_sanitize_vec(c), vec3<f32>(0.0, 0.0, 0.0));
}

fn sds_safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if sds_is_finite(len_sq) && len_sq > 1.0e-24 {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn sds_stable_exp(x: f32) -> f32 {
    if !sds_is_finite(x) {
        return 0.0;
    }
    return exp(clamp(x, -80.0, 0.0));
}

/// Disney/Burley perceptual roughness -> isotropic GGX alpha (= roughness^2),
/// clamped to [0, 1] then floored at MIN_ALPHA. Mirrors `roughness_to_alpha`.
fn sds_roughness_to_alpha(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return max(r * r, SDS_MIN_ALPHA);
}

/// Construct a tap, sanitising colour/position and normalising the normal.
fn sds_make_tap(
    color: vec3<f32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    normalized_hit: f32,
) -> SdsTap {
    var tap: SdsTap;
    tap.color = sds_sanitize_rgb(color);
    tap.position = sds_sanitize_vec(position);
    tap.normal = sds_safe_normalize(normal);
    tap.roughness = clamp(roughness, 0.0, 1.0);
    tap.normalized_hit = clamp(sds_sanitize_scalar(normalized_hit), 0.0, 1.0);
    return tap;
}

// --- hit-distance normalisation ----------------------------------------------

/// Roughness-boosted multiple of the view depth, floored positive.
fn sds_hit_normalizer(roughness: f32, abs_view_z: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    let base = max(abs_view_z, 1.0e-3);
    return base * (0.5 + 1.5 * r);
}

fn sds_normalize_hit_distance(hit_distance: f32, view_z: f32, roughness: f32) -> f32 {
    let hit = max(sds_sanitize_scalar(hit_distance), 0.0);
    let vz = abs(sds_sanitize_scalar(view_z));
    let scale = sds_hit_normalizer(roughness, vz);
    let denom = hit + scale;
    if denom <= 1.0e-12 {
        return 0.0;
    }
    return clamp(hit / denom, 0.0, 1.0);
}

fn sds_denormalize_hit_distance(normalized: f32, view_z: f32, roughness: f32) -> f32 {
    let n = clamp(sds_sanitize_scalar(normalized), 0.0, 1.0 - 1.0e-4);
    let vz = abs(sds_sanitize_scalar(view_z));
    let scale = sds_hit_normalizer(roughness, vz);
    return max(n * scale / (1.0 - n), 0.0);
}

// --- kernel sizing ------------------------------------------------------------

fn sds_contact_hardening_factor(normalized_hit: f32, strength: f32) -> f32 {
    let n = clamp(sds_sanitize_scalar(normalized_hit), 0.0, 1.0);
    let s = clamp(strength, 0.0, 1.0);
    let hardened = sqrt(n);
    return (1.0 - s) + s * hardened;
}

fn sds_blur_radius(roughness: f32, normalized_hit: f32, params: SdsParams) -> f32 {
    let alpha = sds_roughness_to_alpha(roughness);
    let hardening = sds_contact_hardening_factor(normalized_hit, params.contact_hardening);
    let radius = max(params.max_radius, 0.0) * alpha * hardening;
    if sds_is_finite(radius) {
        return max(radius, 0.0);
    }
    return 0.0;
}

/// Split a scalar radius into anisotropic (major, minor) radii.
fn sds_anisotropic_radii(radius: f32, n_dot_v: f32, params: SdsParams) -> vec2<f32> {
    let r = max(radius, 0.0);
    let ndv = clamp(n_dot_v, 1.0e-3, 1.0);
    let max_aniso = max(params.max_anisotropy, 1.0);
    var aniso = 1.0 + (max_aniso - 1.0) * (1.0 - ndv);
    aniso = clamp(aniso, 1.0, max_aniso);
    let sqrt_aniso = sqrt(aniso);
    let major = r * sqrt_aniso;
    let minor = r / sqrt_aniso;
    return vec2<f32>(max(major, 0.0), max(minor, 0.0));
}

// --- bilateral edge-stops -----------------------------------------------------

fn sds_depth_weight(
    center_pos: vec3<f32>,
    center_normal: vec3<f32>,
    sample_pos: vec3<f32>,
    phi_depth: f32,
) -> f32 {
    let n = sds_safe_normalize(center_normal);
    let plane = abs(dot(sds_sanitize_vec(sample_pos) - sds_sanitize_vec(center_pos), n));
    let phi = max(phi_depth, 1.0e-6);
    return sds_stable_exp(-plane / phi);
}

fn sds_normal_weight(n0: vec3<f32>, n1: vec3<f32>, phi_normal: f32) -> f32 {
    let a = sds_safe_normalize(n0);
    let b = sds_safe_normalize(n1);
    let cosine = clamp(dot(a, b), 0.0, 1.0);
    return clamp(pow(cosine, max(phi_normal, 0.0)), 0.0, 1.0);
}

fn sds_roughness_weight(r0: f32, r1: f32, phi_roughness: f32) -> f32 {
    let diff = abs(clamp(r0, 0.0, 1.0) - clamp(r1, 0.0, 1.0));
    let phi = max(phi_roughness, 1.0e-6);
    return sds_stable_exp(-diff / phi);
}

fn sds_spatial_weight(center: SdsTap, sample: SdsTap, params: SdsParams) -> f32 {
    let w_depth = sds_depth_weight(center.position, center.normal, sample.position, params.phi_depth);
    let w_normal = sds_normal_weight(center.normal, sample.normal, params.phi_normal);
    let w_rough = sds_roughness_weight(center.roughness, sample.roughness, params.phi_roughness);
    return clamp(w_depth * w_normal * w_rough, 0.0, 1.0);
}

/// One edge-stopping spatial gather over a bounded specular neighbourhood.
/// `count <= SDS_MAX_TAPS` neighbours are active; each carries a precomputed
/// kernel weight. Mirrors `spatial_filter`.
fn sds_spatial_filter(
    center: SdsTap,
    center_kernel: f32,
    neighbours: array<SdsTap, 8>,
    kernels: array<f32, 8>,
    count: u32,
    params: SdsParams,
) -> SdsResult {
    let w_center = max(sds_sanitize_scalar(center_kernel), 0.0);
    var color_sum = center.color * w_center;
    var hit_sum = center.normalized_hit * w_center;
    var weight = w_center;

    let n = min(count, SDS_MAX_TAPS);
    for (var i = 0u; i < n; i = i + 1u) {
        let sample = neighbours[i];
        let k = max(sds_sanitize_scalar(kernels[i]), 0.0);
        if k == 0.0 {
            continue;
        }
        let w = k * sds_spatial_weight(center, sample, params);
        if w <= 0.0 {
            continue;
        }
        color_sum = color_sum + sample.color * w;
        hit_sum = hit_sum + sample.normalized_hit * w;
        weight = weight + w;
    }

    var out: SdsResult;
    if weight <= 1.0e-12 {
        out.color = center.color;
        out.normalized_hit = center.normalized_hit;
    } else {
        let inv = 1.0 / weight;
        out.color = sds_sanitize_rgb(color_sum * inv);
        out.normalized_hit = clamp(hit_sum * inv, 0.0, 1.0);
    }
    return out;
}

// --- compile-coverage harness -------------------------------------------------
struct SdsProbe {
    value: vec4<f32>,
}

// Probe lives in a disjoint group so it never collides with the real
// kernel's group-0 resource interface below (same module, two entries).
@group(1) @binding(0) var<storage, read_write> sds_probe: SdsProbe;

@compute @workgroup_size(1, 1, 1)
fn spec_denoise_spatial_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    var params: SdsParams;
    params.max_radius = 32.0;
    params.phi_depth = 0.5;
    params.phi_normal = 128.0;
    params.phi_roughness = 0.08;
    params.contact_hardening = 1.0;
    params.max_anisotropy = 3.0;

    let up = vec3<f32>(0.0, 0.0, 1.0);
    let center = sds_make_tap(vec3<f32>(0.5, 0.4, 0.3), vec3<f32>(0.0, 0.0, 0.0), up, 0.4, 0.3);
    var neighbours: array<SdsTap, 8>;
    var kernels: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        neighbours[i] = sds_make_tap(vec3<f32>(0.2, 0.2, 0.2), vec3<f32>(f32(i) * 0.1, 0.0, 0.0), up, 0.45, 0.35);
        kernels[i] = 0.5;
    }
    let res = sds_spatial_filter(center, 1.0, neighbours, kernels, 4u, params);

    let nh = sds_normalize_hit_distance(5.0, 10.0, 0.4);
    let dh = sds_denormalize_hit_distance(nh, 10.0, 0.4);
    let radius = sds_blur_radius(0.5, 0.5, params);
    let aniso = sds_anisotropic_radii(radius, 0.3, params);
    let sw = sds_spatial_weight(center, neighbours[0], params);

    sds_probe.value = vec4<f32>(res.color, res.normalized_hit + nh + dh + aniso.x + aniso.y + sw);
}


// --- spatial-denoise compute entry -------------------------------------------
//
// One invocation per framebuffer texel. Reconstructs the shading point's view
// position from the SSR prepass reverse-Z depth, selects an anisotropic,
// contact-hardened blur footprint from the surface roughness and the SSR hit
// distance, then runs the edge-aware cross-bilateral gather over a fixed
// unit-disk neighbourhood (gated on depth, normal and roughness) matching the
// CPU golden's `spatial_filter`. The ReSTIR confidence (the `spec_gi` resolve
// alpha) is passed through unchanged so the energy-conserving composite reads
// every GI source uniformly.

struct SpecDenoiseSpatialConfig {
    view_from_clip: mat4x4<f32>,
    width: u32,
    height: u32,
    max_radius: f32,
    phi_depth: f32,
    phi_normal: f32,
    phi_roughness: f32,
    contact_hardening: f32,
    max_anisotropy: f32,
    near: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> sds_config: SpecDenoiseSpatialConfig;
// `spec_gi` resolve: rgb = resolved glossy specular, a = ReSTIR confidence.
@group(0) @binding(1) var sds_resolved: texture_2d<f32>;
// SSR trace per-pixel hit distance (view-space units) in the red channel.
@group(0) @binding(2) var sds_hit: texture_2d<f32>;
// Packed G-buffer: rgb = view normal * 2 - 1, a = perceptual roughness.
@group(0) @binding(3) var sds_nr: texture_2d<f32>;
// SSR prepass reverse-Z device depth (depth 1.0 = near, 0.0 = background/sky).
@group(0) @binding(4) var sds_depth: texture_2d<f32>;
// Write-only filtered specular + passthrough confidence.
@group(0) @binding(5) var sds_filtered: texture_storage_2d<rgba16float, write>;

const SDS_TAP_COUNT: u32 = 8u;

// Fixed two-ring unit-disk neighbourhood (four axis + four diagonal taps),
// scaled by the anisotropic footprint at dispatch time. A static distribution
// keeps the gather deterministic and bounded (<= `SDS_MAX_TAPS`).
fn sds_tap_offset(i: u32) -> vec2<f32> {
    switch i {
        case 0u: {
            return vec2<f32>(1.0, 0.0);
        }
        case 1u: {
            return vec2<f32>(-1.0, 0.0);
        }
        case 2u: {
            return vec2<f32>(0.0, 1.0);
        }
        case 3u: {
            return vec2<f32>(0.0, -1.0);
        }
        case 4u: {
            return vec2<f32>(0.5, 0.5);
        }
        case 5u: {
            return vec2<f32>(-0.5, 0.5);
        }
        case 6u: {
            return vec2<f32>(0.5, -0.5);
        }
        default: {
            return vec2<f32>(-0.5, -0.5);
        }
    }
}

fn sds_params_from_config() -> SdsParams {
    var params: SdsParams;
    params.max_radius = sds_config.max_radius;
    params.phi_depth = sds_config.phi_depth;
    params.phi_normal = sds_config.phi_normal;
    params.phi_roughness = sds_config.phi_roughness;
    params.contact_hardening = sds_config.contact_hardening;
    params.max_anisotropy = sds_config.max_anisotropy;
    return params;
}

// Reconstruct the view-space position from framebuffer UV + reverse-Z depth,
// identical to `spec_gi_reuse.wesl::sgr_reconstruct_view_position` so both
// passes rebuild positions from the shared SSR depth consistently.
fn sds_reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = sds_config.view_from_clip * clip;
    return view.xyz / view.w;
}

@compute @workgroup_size(8, 8, 1)
fn spec_denoise_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= sds_config.width || id.y >= sds_config.height {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let extent = vec2<f32>(f32(sds_config.width), f32(sds_config.height));
    let max_coord = vec2<i32>(i32(sds_config.width) - 1, i32(sds_config.height) - 1);

    let center_resolved = textureLoad(sds_resolved, pixel, 0);
    let center_device_depth = textureLoad(sds_depth, pixel, 0).x;
    if center_device_depth <= 0.0 {
        // Background / sky: nothing to filter, pass the resolve through.
        textureStore(sds_filtered, pixel, center_resolved);
        return;
    }

    let params = sds_params_from_config();

    let center_uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / extent;
    let center_pos = sds_reconstruct_view_position(center_uv, center_device_depth);
    let center_nr = textureLoad(sds_nr, pixel, 0);
    let center_normal = sds_safe_normalize(center_nr.xyz * 2.0 - 1.0);
    let center_roughness = clamp(center_nr.w, 0.0, 1.0);
    let center_hit = textureLoad(sds_hit, pixel, 0).x;
    let center_norm_hit = sds_normalize_hit_distance(center_hit, center_pos.z, center_roughness);
    let center_tap = sds_make_tap(
        center_resolved.xyz,
        center_pos,
        center_normal,
        center_roughness,
        center_norm_hit,
    );

    // Anisotropic, contact-hardened footprint. The major axis follows the
    // surface normal's screen projection (the reflection footprint stretches
    // along it at grazing angles); the minor axis is perpendicular.
    let view_dir = sds_safe_normalize(-center_pos);
    let n_dot_v = max(dot(center_normal, view_dir), 1.0e-3);
    let radius = sds_blur_radius(center_roughness, center_norm_hit, params);
    let aniso = sds_anisotropic_radii(radius, n_dot_v, params);

    var major_dir = vec2<f32>(1.0, 0.0);
    let nxy = center_normal.xy;
    if dot(nxy, nxy) > 1.0e-12 {
        major_dir = normalize(nxy);
    }
    let minor_dir = vec2<f32>(-major_dir.y, major_dir.x);

    var neighbours: array<SdsTap, 8>;
    var kernels: array<f32, 8>;
    for (var i = 0u; i < SDS_TAP_COUNT; i = i + 1u) {
        let o = sds_tap_offset(i);
        let offset_px = major_dir * (o.x * aniso.x) + minor_dir * (o.y * aniso.y);
        let raw = pixel + vec2<i32>(i32(round(offset_px.x)), i32(round(offset_px.y)));
        let coord = clamp(raw, vec2<i32>(0, 0), max_coord);

        let s_resolved = textureLoad(sds_resolved, coord, 0);
        let s_depth = textureLoad(sds_depth, coord, 0).x;
        let s_nr = textureLoad(sds_nr, coord, 0);
        let s_normal = sds_safe_normalize(s_nr.xyz * 2.0 - 1.0);
        let s_roughness = clamp(s_nr.w, 0.0, 1.0);
        let s_uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / extent;
        var s_pos = center_pos;
        if s_depth > 0.0 {
            s_pos = sds_reconstruct_view_position(s_uv, s_depth);
        }
        let s_hit = textureLoad(sds_hit, coord, 0).x;
        let s_norm_hit = sds_normalize_hit_distance(s_hit, s_pos.z, s_roughness);
        neighbours[i] = sds_make_tap(s_resolved.xyz, s_pos, s_normal, s_roughness, s_norm_hit);
        // Spatial Gaussian over the unit-disk offset; the bilateral edge-stops
        // (depth/normal/roughness) are applied inside `sds_spatial_filter`.
        kernels[i] = exp(-2.0 * dot(o, o));
    }

    let res = sds_spatial_filter(center_tap, 1.0, neighbours, kernels, SDS_TAP_COUNT, params);
    textureStore(sds_filtered, pixel, vec4<f32>(res.color, center_resolved.w));
}
