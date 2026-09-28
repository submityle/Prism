//! Screen-space reflection (SSR) compute twin.
//!
//! GPU twin of `prism_render_shading::screen_space`.  For every pixel this
//! kernel reflects the view ray off the reconstructed surface, marches the
//! reflected ray across the reverse-Z "nearest depth" HZB pyramid, reprojects a
//! hit into the previous frame's colour, and writes the reflected radiance plus
//! a `[0, 1]` blend confidence the resolve stage mixes over the prefiltered IBL
//! specular.
//!
//! Every constant, fade and the hierarchical march mirror the CPU golden
//! bit-for-bit within tolerance so machines with and without a GPU agree.  The
//! file is self-contained (no `import`s), matching the `gtao.wesl` /
//! `brdf_lut.wesl` precedent.

// Per-dispatch constants (view/camera + fade tunables).
struct SsrConfig {
    // Projection (view -> clip, reverse-Z) and its inverse (clip -> view).
    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    // Positive near-plane distance in front of the camera along -Z.
    near: f32,
    // View-space march length.
    max_distance: f32,
    // Device-depth thin-surface tolerance for accepting a hit.
    thickness: f32,
    // Hard iteration cap for the hierarchical march.
    max_iterations: u32,
    // Fade tunables (see the golden `SsrConfidenceParams`).
    edge_fade_start: f32,
    full_roughness: f32,
    max_roughness: f32,
    distance_fade_start: f32,
    // Finest mip refined to (usually 0) and number of colour mips available.
    most_detailed_mip: u32,
    color_max_mip: f32,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<f32>,
}

// Reverse-Z "nearest depth" pyramid (per-cell max device depth over its mips).
@group(0) @binding(0) var hzb: texture_2d<f32>;
// Full-resolution device depth used to reconstruct the view-space position.
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
// View-space normal (rgb) plus perceptual roughness (a).
@group(0) @binding(2) var normal_roughness: texture_2d<f32>;
// Previous frame's linear HDR colour (with mips for the roughness blur).
@group(0) @binding(3) var prev_color: texture_2d<f32>;
@group(0) @binding(4) var prev_sampler: sampler;
// Output: rgb = reflected radiance, a = blend confidence.
@group(0) @binding(5) var ssr_out: texture_storage_2d<rgba16float, write>;

var<immediate> config: SsrConfig;

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep matching the golden `smoothstep`.
fn smoothstep_ref(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return select(0.0, 1.0, x >= edge0);
    }
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

// Reflect an incident direction about a unit normal.
fn reflect_ref(incident: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, normal) * normal;
}

// Fade toward 0 near any screen border (golden `edge_fade`).
fn edge_fade(uv: vec2<f32>, fade_start: f32) -> f32 {
    let fade = max(fade_start, 1.0e-4);
    let dx = min(uv.x, 1.0 - uv.x);
    let dy = min(uv.y, 1.0 - uv.y);
    return smoothstep_ref(0.0, fade, min(dx, dy));
}

// Fade rays that point back toward the camera (golden `facing_fade`).
fn facing_fade(reflection: vec3<f32>, camera_to_surface: vec3<f32>) -> f32 {
    return smoothstep_ref(-0.15, 0.35, dot(reflection, camera_to_surface));
}

// Fade out as the surface roughens past a single mirror ray (golden
// `roughness_fade`).
fn roughness_fade(roughness: f32, full_roughness: f32, max_roughness: f32) -> f32 {
    let lo = min(full_roughness, max_roughness);
    let hi = max(max_roughness, full_roughness + 1.0e-4);
    return 1.0 - smoothstep_ref(lo, hi, roughness);
}

// Fade toward the far march limit (golden `distance_fade`).
fn distance_fade(travel: f32, fade_start: f32) -> f32 {
    let start = clamp(fade_start, 0.0, 1.0);
    return 1.0 - smoothstep_ref(start, 1.0, saturate(travel));
}

// Perceptual roughness -> fractional colour mip (golden `reflection_mip`).
fn reflection_mip(roughness: f32, max_mip: f32) -> f32 {
    let linear = saturate(roughness) * saturate(roughness);
    let mip = sqrt(linear) * max(max_mip, 0.0);
    return clamp(mip, 0.0, max(max_mip, 0.0));
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
// (golden `march_hierarchical`).
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

        if depth_near >= cell_nearest {
            let span = depth_near - depth_far;
            var t_hit = t;
            if span > 1.0e-8 {
                t_hit = t + (depth_near - cell_nearest) / span * (t_next - t);
            }
            result.hit = true;
            result.uv = start_uv + delta * t_hit;
            result.depth = cell_nearest;
            result.travel = t_hit;
            return result;
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

@compute @workgroup_size(8, 8)
fn trace_ssr(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = vec2<i32>(id.xy);
    let dims = vec2<i32>(textureDimensions(ssr_out));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / config.screen_size;
    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: device depth 0 is the far plane / cleared background.
    if device_depth <= 0.0 {
        textureStore(ssr_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let nr = textureLoad(normal_roughness, pixel, 0);
    let normal_view = normalize(nr.xyz * 2.0 - 1.0);
    let roughness = nr.w;

    let position_view = reconstruct_view_position(uv, device_depth);
    let incident = normalize(position_view);
    let reflection = normalize(reflect_ref(incident, normal_view));

    // Build the screen-space ray, clipping the far endpoint to the near plane.
    var travel = max(config.max_distance, 1.0e-3);
    var end_view = position_view + reflection * travel;
    let plane = -config.near;
    if end_view.z > plane {
        let denom = reflection.z;
        if abs(denom) <= 1.0e-6 {
            textureStore(ssr_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            return;
        }
        let clipped = (plane - position_view.z) / denom;
        if clipped <= 1.0e-4 {
            textureStore(ssr_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
            return;
        }
        travel = clipped;
    }
    let clipped_fraction = clamp(travel / max(config.max_distance, 1.0e-3), 1.0e-4, 1.0);
    end_view = position_view + reflection * travel;

    let start = project_view(position_view);
    let end = project_view(end_view);
    if start.clip_w <= 0.0 || end.clip_w <= 0.0 {
        textureStore(ssr_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let march = march_hierarchical(start.uv, start.device_depth, end.uv, end.device_depth);

    // Confidence fades (golden `trace_confidence`).
    var confidence = 0.0;
    var reflected = vec3<f32>(0.0);
    if march.hit {
        let travel_norm = clamp(march.travel * clipped_fraction, 0.0, 1.0);
        let edge = edge_fade(march.uv, config.edge_fade_start);
        let facing = facing_fade(reflection, incident);
        let rough = roughness_fade(roughness, config.full_roughness, config.max_roughness);
        let dist = distance_fade(travel_norm, config.distance_fade_start);
        confidence = saturate(edge * facing * rough * dist);

        let lod = reflection_mip(roughness, config.color_max_mip);
        reflected = textureSampleLevel(prev_color, prev_sampler, march.uv, lod).rgb;
    }

    textureStore(ssr_out, pixel, vec4<f32>(reflected, confidence));
}
