//! Ground-Truth Ambient Occlusion (GTAO) compute twin.
//!
//! GPU twin of `prism_render_shading::ao`.  For every covered pixel this kernel
//! sweeps `slice_count` directions through the view vector, searches each slice
//! for the highest occluding horizon on both sides, and integrates the
//! cosine-weighted visible arc analytically (Jimenez et al., SIGGRAPH 2016).
//! Averaging the slices yields ambient **visibility** in `[0, 1]` (`1`
//! unoccluded); the shading-resolve stage multiplies indirect/ambient light by
//! this factor.
//!
//! It consumes the linear view-depth prepass (`.x` = positive view distance,
//! `<= 0` marks background) and the view-space normal buffer, and every
//! constant and expression here mirrors the CPU golden bit-for-bit within
//! tolerance so the two agree on machines with and without a GPU.  The file is
//! self-contained (no `import`s), matching the `oit.wesl` precedent.

// `pi` and `pi / 2`; the widest a horizon can open on one side of the normal.
const GTAO_PI: f32 = 3.14159265358979323846;
const GTAO_HALF_PI: f32 = 1.57079632679489661923;
// Texture-space nudge used to derive the in-view slice axis; matches the
// golden `AXIS_EPSILON`.
const GTAO_AXIS_EPSILON: f32 = 1.0e-3;
// Guards against dividing by a near-zero horizon distance; matches the golden
// `DISTANCE_EPSILON`.
const GTAO_DISTANCE_EPSILON: f32 = 1.0e-4;

// Pinhole projection + horizon-search tunables, uploaded per dispatch. The two
// tangents are `1 / proj[0][0]` and `1 / proj[1][1]`; the counts are clamped to
// at least one on the CPU side exactly as the golden does.
struct GtaoConfig {
    tan_half_fov_x: f32,
    tan_half_fov_y: f32,
    world_radius: f32,
    falloff: f32,
    power: f32,
    slice_count: u32,
    steps_per_slice: u32,
    padding: u32,
}

// linear view depth (positive distance, <= 0 = background) in `.x`.
@group(0) @binding(0) var linear_depth_tex: texture_2d<f32>;
// unit view-space normal in `.xyz`.
@group(0) @binding(1) var view_normal_tex: texture_2d<f32>;
// ambient visibility output (1 = unoccluded) in `.x`.
@group(0) @binding(2) var ao_out: texture_storage_2d<r32float, write>;

var<immediate> config: GtaoConfig;

// Normalises `value`, falling back to `fallback` for degenerate/non-finite
// input. Mirrors `prism_render_shading::vecmath::normalize_or` (the `< 3.4e38`
// guard stands in for the golden's `is_finite`).
fn normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 && length_squared < 3.4e38 {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

// View-space position of a sample at texture coord `uv` (origin top-left, `y`
// down) whose positive linear view depth is `depth`. RH, camera at the origin
// looking down `-Z`. Mirrors `GtaoCamera::reconstruct`.
fn reconstruct(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = 2.0 * uv.x - 1.0;
    let ndc_y = 1.0 - 2.0 * uv.y;
    return vec3<f32>(
        ndc_x * config.tan_half_fov_x * depth,
        ndc_y * config.tan_half_fov_y * depth,
        -depth,
    );
}

// World search radius -> per-axis texture-space radius at `depth`. Mirrors
// `GtaoCamera::uv_radius`.
fn uv_radius(world_radius: f32, depth: f32) -> vec2<f32> {
    let inv = 1.0 / (2.0 * max(depth, 1.0e-4));
    return vec2<f32>(
        world_radius * inv / config.tan_half_fov_x,
        world_radius * inv / config.tan_half_fov_y,
    );
}

// Nearest-neighbour depth fetch. Returns the positive linear depth, or `-1.0`
// when the coordinate leaves the frame or lands on background/non-finite depth
// (the golden's `Option::None`). Mirrors `GtaoBuffers::sample_depth`.
fn sample_depth(uv: vec2<f32>) -> f32 {
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return -1.0;
    }
    let dims = vec2<f32>(textureDimensions(linear_depth_tex, 0));
    let max_x = i32(dims.x) - 1;
    let max_y = i32(dims.y) - 1;
    let xi = clamp(i32(round(uv.x * dims.x - 0.5)), 0, max_x);
    let yi = clamp(i32(round(uv.y * dims.y - 0.5)), 0, max_y);
    let d = textureLoad(linear_depth_tex, vec2<i32>(xi, yi), 0).x;
    if d > 0.0 && d < 3.4e38 {
        return d;
    }
    return -1.0;
}

// Distance attenuation applied to a horizon sample. Full weight inside
// `falloff_start`, quadratic ease-out to zero at `radius`. Mirrors
// `integral::distance_weight`.
fn distance_weight(dist: f32, falloff_start: f32, radius: f32) -> f32 {
    if dist <= falloff_start {
        return 1.0;
    }
    let span = max(radius - falloff_start, 1.0e-4);
    let t = clamp((dist - falloff_start) / span, 0.0, 1.0);
    return 1.0 - t * t;
}

// Closed-form visibility of one slice (Jimenez et al. 2016). Mirrors
// `integral::slice_visibility`.
fn slice_visibility(cos_h1: f32, cos_h2: f32, gamma: f32, proj_len: f32) -> f32 {
    let h1_raw = -acos(clamp(cos_h1, -1.0, 1.0));
    let h2_raw = acos(clamp(cos_h2, -1.0, 1.0));

    let h1 = gamma + max(h1_raw - gamma, -GTAO_HALF_PI);
    let h2 = gamma + min(h2_raw - gamma, GTAO_HALF_PI);

    let sin_gamma = sin(gamma);
    let cos_gamma = cos(gamma);

    let term1 = 0.25 * (-cos(2.0 * h1 - gamma) + cos_gamma + 2.0 * h1 * sin_gamma);
    let term2 = 0.25 * (-cos(2.0 * h2 - gamma) + cos_gamma + 2.0 * h2 * sin_gamma);

    return proj_len * (term1 + term2);
}

// Marches one side of a slice, returning the highest (distance-attenuated)
// horizon cosine found. `side` is `+1.0` (positive axis, `cos_h2`) or `-1.0`
// (negative axis, `cos_h1`). Mirrors `HorizonSearch::horizon`.
fn horizon(
    center_uv: vec2<f32>,
    position: vec3<f32>,
    view: vec3<f32>,
    radius_uv: vec2<f32>,
    radius_world: f32,
    falloff_start: f32,
    steps: u32,
    omega: vec2<f32>,
    side: f32,
) -> f32 {
    var cos_horizon = 0.0;
    for (var step = 0u; step < steps; step = step + 1u) {
        let s = (f32(step) + 1.0) / f32(steps);
        let sample_uv = center_uv + side * omega * radius_uv * s;
        let sampled_depth = sample_depth(sample_uv);
        if sampled_depth <= 0.0 {
            continue;
        }
        let sample_pos = reconstruct(sample_uv, sampled_depth);
        let horizon_vec = sample_pos - position;
        let dist_sq = dot(horizon_vec, horizon_vec);
        if dist_sq < GTAO_DISTANCE_EPSILON * GTAO_DISTANCE_EPSILON {
            continue;
        }
        let dist = sqrt(dist_sq);
        let cos_angle = dot(horizon_vec, view) / dist;
        if cos_angle <= 0.0 {
            continue;
        }
        let weighted = cos_angle * distance_weight(dist, falloff_start, radius_world);
        cos_horizon = max(cos_horizon, weighted);
    }
    return cos_horizon;
}

// Ambient visibility (1 = unoccluded) for the pixel at integer coord `coord`.
// Mirrors `gtao_pixel`.
fn gtao_pixel(coord: vec2<i32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(linear_depth_tex, 0));
    let depth = textureLoad(linear_depth_tex, coord, 0).x;
    if !(depth > 0.0 && depth < 3.4e38) {
        return 1.0;
    }

    let slice_count = max(config.slice_count, 1u);
    let steps = max(config.steps_per_slice, 1u);
    let radius_world = max(config.world_radius, GTAO_DISTANCE_EPSILON);
    let falloff_start = radius_world * clamp(config.falloff, 0.0, 1.0);

    let center_uv = vec2<f32>(
        (f32(coord.x) + 0.5) / dims.x,
        (f32(coord.y) + 0.5) / dims.y,
    );
    let position = reconstruct(center_uv, depth);
    let view = normalize_or(position * -1.0, vec3<f32>(0.0, 0.0, 1.0));
    let normal = textureLoad(view_normal_tex, coord, 0).xyz;

    let radius_uv = uv_radius(radius_world, depth);

    var visibility = 0.0;
    for (var slice = 0u; slice < slice_count; slice = slice + 1u) {
        let phi = GTAO_PI * f32(slice) / f32(slice_count);
        let omega = vec2<f32>(cos(phi), sin(phi));

        let nudged_uv = center_uv + omega * GTAO_AXIS_EPSILON;
        let nudged = reconstruct(nudged_uv, depth);
        let tangent = nudged - position;
        let axis = normalize_or(
            tangent - view * dot(tangent, view),
            vec3<f32>(1.0, 0.0, 0.0),
        );

        let n_view = dot(normal, view);
        let n_axis = dot(normal, axis);
        let proj_len = sqrt(n_view * n_view + n_axis * n_axis);
        if proj_len < GTAO_DISTANCE_EPSILON {
            continue;
        }
        let gamma = atan2(n_axis, n_view);

        let cos_h2 = horizon(
            center_uv, position, view, radius_uv, radius_world, falloff_start, steps, omega, 1.0);
        let cos_h1 = horizon(
            center_uv, position, view, radius_uv, radius_world, falloff_start, steps, omega, -1.0);

        visibility = visibility + slice_visibility(cos_h1, cos_h2, gamma, proj_len);
    }

    visibility = clamp(visibility / f32(slice_count), 0.0, 1.0);
    return pow(visibility, max(config.power, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn compute_gtao(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(ao_out);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let visibility = gtao_pixel(coord);
    textureStore(ao_out, coord, vec4<f32>(visibility, 0.0, 0.0, 1.0));
}
