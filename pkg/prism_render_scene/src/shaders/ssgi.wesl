//! Screen-space global illumination (SSGI) compute twin.
//!
//! GPU twin of `prism_render_shading::screen_space::gi`.  Where `ssr.wesl`
//! reflects a single specular ray, this kernel integrates one indirect
//! *diffuse* bounce: several cosine-weighted rays are cast around the surface
//! normal and marched across the *same* reverse-Z "nearest depth" HZB pyramid
//! the reflection trace builds.  A ray that intersects on-screen geometry picks
//! up that surface's already-shaded radiance (colour bleeding + one indirect
//! bounce); a ray that escapes the framebuffer contributes the ambient the
//! caller supplies (`ambient` = the IBL/SH diffuse irradiance the resolve
//! already evaluates), so energy stays consistent with the ambient term SSGI
//! augments.
//!
//! Cosine-weighted importance sampling collapses the Lambertian estimator
//! `∫ (albedo/π) L_i (n·l) dω` to `albedo · mean(L_i)`, so the gather is a plain
//! average of per-ray radiance — no per-sample cosine or pdf divide.  This
//! kernel outputs the *pre-albedo* mean radiance (rgb) plus a blend confidence
//! (a); the shading resolve multiplies by the surface albedo and blends over
//! the pure IBL/SH ambient under that confidence, exactly as it composites SSR
//! over the prefiltered IBL specular.  Occlusion is intentionally left to the
//! dedicated GTAO path, so no ambient-occlusion channel is emitted here.
//!
//! Determinism mirrors the reflection sampler: directions come from the fixed
//! low-discrepancy `hammersley` set (not a per-frame RNG), so this twin and the
//! CPU golden pick byte-identical directions for a given
//! `(sample_index, sample_count, normal)`.  Every constant, the near-plane clip
//! and the hierarchical march mirror the golden bit-for-bit within tolerance so
//! machines with and without a GPU agree.  The file is self-contained (no
//! `import`s), matching the `ssr.wesl` / `gtao.wesl` precedent.

// Per-dispatch constants (view/camera + gather tunables).  Field order matches
// `GpuSsgiConfig` byte-for-byte; `screen_size` leads the scalar block so the
// vec2 lands on its 8-byte alignment without an implicit pad.
struct SsgiConfig {
    // Projection (view -> clip, reverse-Z) and its inverse (clip -> view).
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<f32>,
    // Positive near-plane distance in front of the camera along -Z.
    near: f32,
    // View-space hemisphere-ray march length.
    max_distance: f32,
    // Device-depth thin-surface tolerance for accepting a hit.
    thickness: f32,
    // Hard iteration cap for the hierarchical march.
    max_iterations: u32,
    // Finest mip the march refines to (usually 0).
    most_detailed_mip: u32,
    // Artistic gain applied to the gathered indirect radiance.
    intensity: f32,
    // Normalized travel a hit fades out over, in (0, 1]; 1.0 disables falloff.
    distance_falloff: f32,
    // Number of cosine-weighted hemisphere rays traced per pixel.
    sample_count: u32,
    // Padding to the 16-byte immediate alignment the mat4x4 fields force.
    _pad0: u32,
    _pad1: u32,
}

// Reverse-Z "nearest depth" pyramid (per-cell max device depth over its mips).
@group(0) @binding(0) var hzb: texture_2d<f32>;
// Full-resolution device depth used to reconstruct the view-space position.
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
// View-space normal (rgb) plus perceptual roughness (a); SSGI reads the normal.
@group(0) @binding(2) var normal_roughness: texture_2d<f32>;
// Current-frame linear HDR colour (with mips) a hit ray gathers its bounce from.
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var color_sampler: sampler;
// Per-pixel IBL/SH diffuse irradiance; the miss/sky fallback the gather blends
// toward so unshadowed directions keep the ambient look.
@group(0) @binding(5) var ambient: texture_2d<f32>;
// Output: rgb = pre-albedo mean indirect radiance (x intensity), a = confidence.
@group(0) @binding(6) var ssgi_out: texture_storage_2d<rgba16float, write>;

var<immediate> config: SsgiConfig;

const PI: f32 = 3.1415926535897932;
const TAU: f32 = 6.2831853071795865;
const EPS: f32 = 1.0e-4;

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Normalize a vector, returning zero for a degenerate input (golden
// `Vec3::normalize_or_zero`) so callers can skip the ray.
fn norm_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len < 1.0e-8 {
        return vec3<f32>(0.0);
    }
    return v / len;
}

// Van der Corput radical inverse in base 2 (golden `radical_inverse_vdc`).
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = (bits_in << 16u) | (bits_in >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10; // 1 / 2^32
}

// The i-th point of the n-point Hammersley set (golden `hammersley`).
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    let x = select(f32(i) / f32(n), 0.0, n == 0u);
    return vec2<f32>(x, radical_inverse_vdc(i));
}

struct Basis {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
}

// Duff et al. (2017) branch-light orthonormal basis (golden
// `orthonormal_basis`); z axis is the unit normal.
fn orthonormal_basis(normal: vec3<f32>) -> Basis {
    let sign = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    var basis: Basis;
    basis.tangent = vec3<f32>(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    basis.bitangent = vec3<f32>(b, sign + normal.y * normal.y * a, -normal.y);
    return basis;
}

// Cosine-weighted hemisphere direction around unit `normal` (Malley's method;
// golden `cosine_sample_direction`).  Radius `sqrt(xi.x)` and height
// `sqrt(1 - xi.x)` give a cos-distributed elevation with pdf `(n·l)/π`.
fn cosine_sample_direction(xi: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    let n = norm_or_zero(normal);
    if all(n == vec3<f32>(0.0)) {
        return vec3<f32>(0.0);
    }
    let u = clamp(xi.x, 0.0, 1.0);
    let radius = sqrt(u);
    let phi = TAU * xi.y;
    let x = radius * cos(phi);
    let y = radius * sin(phi);
    let z = sqrt(max(1.0 - u, 0.0));
    let basis = orthonormal_basis(n);
    return norm_or_zero(basis.tangent * x + basis.bitangent * y + n * z);
}

// Reconstruct a view-space position from a UV + device depth via the inverse
// projection (golden `project_view_to_screen`, inverted).
fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = config.view_from_clip * clip;
    return view.xyz / view.w;
}

struct ScreenSample {
    uv: vec2<f32>,
    device_depth: f32,
    clip_w: f32,
}

// Project a view-space point to framebuffer UV + device depth (golden
// `project_view_to_screen`).
fn project_view(point_view: vec3<f32>) -> ScreenSample {
    let clip = config.clip_from_view * vec4<f32>(point_view, 1.0);
    let inv_w = 1.0 / clip.w;
    let ndc = clip.xyz * inv_w;
    var s: ScreenSample;
    s.uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    s.device_depth = ndc.z;
    s.clip_w = clip.w;
    return s;
}

// Smallest ray parameter > t at which the ray crosses into the next integer
// texel cell at `size` (golden `next_cell_boundary`).
fn next_cell_boundary(start: vec2<f32>, delta: vec2<f32>, size: vec2<f32>, t: f32) -> f32 {
    let eps = 1.0e-6;
    var best = 1.0;
    let sx = max(size.x, 1.0);
    let sy = max(size.y, 1.0);

    let dx = delta.x * sx;
    if abs(dx) > eps {
        let pos = (start.x + delta.x * t) * sx;
        let boundary = select(ceil(pos) - 1.0, floor(pos) + 1.0, dx > 0.0);
        let t_axis = (boundary / sx - start.x) / delta.x;
        if t_axis > t + eps && t_axis < best {
            best = t_axis;
        }
    }
    let dy = delta.y * sy;
    if abs(dy) > eps {
        let pos = (start.y + delta.y * t) * sy;
        let boundary = select(ceil(pos) - 1.0, floor(pos) + 1.0, dy > 0.0);
        let t_axis = (boundary / sy - start.y) / delta.y;
        if t_axis > t + eps && t_axis < best {
            best = t_axis;
        }
    }
    return min(best, 1.0);
}

struct MarchResult {
    hit: bool,
    uv: vec2<f32>,
    depth: f32,
    travel: f32,
}

// Hierarchical (Hi-Z) march across the reverse-Z nearest-depth pyramid
// (golden `march_hierarchical`).  Shared verbatim with `ssr.wesl`.
fn march_hierarchical(
    start_uv: vec2<f32>,
    start_depth: f32,
    end_uv: vec2<f32>,
    end_depth: f32,
) -> MarchResult {
    let delta = end_uv - start_uv;
    let mip_levels = i32(textureNumLevels(hzb));
    let base = u32(max(mip_levels - 1, 0));
    let most_detailed = min(config.most_detailed_mip, base);

    var level = most_detailed;
    var t = 0.0;
    var iterations = 0u;

    var result: MarchResult;
    result.hit = false;

    loop {
        if iterations >= config.max_iterations || t >= 1.0 {
            break;
        }
        iterations = iterations + 1u;

        let uv = start_uv + delta * t;
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            result.uv = uv;
            result.depth = mix(start_depth, end_depth, t);
            result.travel = t;
            return result;
        }

        let size = vec2<f32>(textureDimensions(hzb, i32(level)));
        let cell = vec2<i32>(uv * size);
        let t_next = next_cell_boundary(start_uv, delta, size, t);

        let depth_near = mix(start_depth, end_depth, t);
        let depth_far = mix(start_depth, end_depth, t_next);
        let cell_nearest = textureLoad(hzb, cell, i32(level)).x;

        if depth_far > cell_nearest {
            t = t_next;
            if level < base {
                level = level + 1u;
            }
            continue;
        }

        if level > most_detailed {
            level = level - 1u;
            continue;
        }

        if cell_nearest - depth_near <= config.thickness {
            result.hit = true;
            result.uv = uv;
            result.depth = cell_nearest;
            result.travel = t;
            return result;
        }
        t = t_next;
    }

    let clamped = min(t, 1.0);
    result.uv = start_uv + delta * clamped;
    result.depth = mix(start_depth, end_depth, clamped);
    result.travel = clamped;
    return result;
}

struct HemisphereRay {
    valid: bool,
    start_uv: vec2<f32>,
    start_depth: f32,
    end_uv: vec2<f32>,
    end_depth: f32,
    clipped_fraction: f32,
}

// Build the screen-space ray for a surface at `position_view` shooting along
// unit `direction_view` (golden `build_hemisphere_ray`): the SSR ray build with
// the reflect step removed.  `valid = false` for a behind-camera origin or a
// ray entirely behind the near plane.
fn build_hemisphere_ray(position_view: vec3<f32>, direction_view: vec3<f32>) -> HemisphereRay {
    var out: HemisphereRay;
    out.valid = false;

    if position_view.z >= -config.near {
        return out;
    }
    let dir = norm_or_zero(direction_view);
    if all(dir == vec3<f32>(0.0)) {
        return out;
    }

    let requested = max(config.max_distance, 1.0e-3);
    var travel = requested;

    let plane = -config.near;
    var end_view = position_view + dir * travel;
    if end_view.z > plane {
        let denom = dir.z;
        if abs(denom) <= 1.0e-6 {
            return out;
        }
        let clipped = (plane - position_view.z) / denom;
        if clipped <= EPS {
            return out;
        }
        travel = clipped;
    }

    let clipped_fraction = clamp(travel / requested, EPS, 1.0);
    end_view = position_view + dir * travel;

    let start = project_view(position_view);
    let end = project_view(end_view);
    if start.clip_w <= 0.0 || end.clip_w <= 0.0 {
        return out;
    }

    out.valid = true;
    out.start_uv = start.uv;
    out.start_depth = start.device_depth;
    out.end_uv = end.uv;
    out.end_depth = end.device_depth;
    out.clipped_fraction = clipped_fraction;
    return out;
}

// Falloff weight for a hit at normalized `travel` under `distance_falloff`
// (golden `distance_weight`).
fn distance_weight(travel: f32, distance_falloff: f32) -> f32 {
    let cutoff = clamp(distance_falloff, EPS, 1.0);
    return clamp(1.0 - travel / cutoff, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn trace_ssgi(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = vec2<i32>(id.xy);
    let dims = vec2<i32>(textureDimensions(ssgi_out));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: device depth 0 is the far plane / cleared background — no
    // surface to gather indirect light for.
    if device_depth <= 0.0 {
        textureStore(ssgi_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / config.screen_size;
    let nr = textureLoad(normal_roughness, pixel, 0);
    let normal_view = normalize(nr.xyz * 2.0 - 1.0);
    let position_view = reconstruct_view_position(uv, device_depth);

    // The ambient the resolve would otherwise apply: misses and the faded tail
    // of hits blend toward this so SSGI never punches a dark hole.
    let sky = textureLoad(ambient, pixel, 0).rgb;

    let sample_count = max(config.sample_count, 1u);
    var radiance_sum = vec3<f32>(0.0);
    var valid = 0.0;
    var hits = 0.0;

    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley(i, sample_count);
        let dir = cosine_sample_direction(xi, normal_view);
        if all(dir == vec3<f32>(0.0)) {
            continue;
        }
        let ray = build_hemisphere_ray(position_view, dir);
        if !ray.valid {
            continue;
        }

        valid = valid + 1.0;
        let march = march_hierarchical(ray.start_uv, ray.start_depth, ray.end_uv, ray.end_depth);
        if march.hit {
            hits = hits + 1.0;
            let travel_norm = saturate(march.travel * ray.clipped_fraction);
            let weight = distance_weight(travel_norm, config.distance_falloff);
            let radiance = textureSampleLevel(scene_color, color_sampler, march.uv, 0.0).rgb;
            // A faded hit blends toward the sky over the missing weight so a
            // distant bounce degrades to the ambient fallback, never to black.
            radiance_sum = radiance_sum + radiance * weight + sky * (1.0 - weight);
        } else {
            radiance_sum = radiance_sum + sky;
        }
    }

    if valid <= 0.0 {
        // No usable ray: emit the pure ambient with zero trust so the resolve
        // keeps its IBL/SH result untouched.
        textureStore(ssgi_out, pixel, vec4<f32>(sky * config.intensity, 0.0));
        return;
    }

    let mean_radiance = radiance_sum / valid;
    let confidence = saturate(valid / f32(sample_count));
    let indirect = mean_radiance * config.intensity;
    textureStore(ssgi_out, pixel, vec4<f32>(indirect, confidence));
}
