// Surface-cache coverage pass: per-pixel surfel gather (the consumer).
//
// One invocation per framebuffer pixel. The invocation reconstructs the
// shading point's view-space position + normal from the `SSR` prepass
// (reverse-Z depth + packed `normal_roughness`), gathers the filtered surfels
// in a `gather_radius` box around the point's tile and writes the
// coverage-weighted mean radiance (`rgb`) plus the accumulated coverage
// confidence (`a`) into the `rgba16float` `GI` export the composite fold reads.
//
// The coverage kernel (radial x axial x normal) mirrors the golden
// `prism_render_shading::gi::surface_cache` `Surfel::coverage` verbatim, so a
// point no surfel covers resolves to zero radiance / zero confidence and the
// composite leaves the shaded colour untouched there.

const MIN_RADIUS: f32 = 1.0e-6;
const F32_FINITE_MAX: f32 = 3.40282347e38;

// Persistent per-surfel storage element; all-scalar std430 layout (48 bytes),
// mirroring `GpuSurfel`.
struct Surfel {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    radius: f32,
    radiance_x: f32,
    radiance_y: f32,
    radiance_z: f32,
    sample_count: u32,
    valid: u32,
}

// Immediate (push-constant) block mirroring `GpuSurfaceCacheCoverageParams`
// (112 bytes: the `mat4x4` leads for its 16-byte alignment, then the `vec2`s,
// then the scalar tail padded to the 16-byte immediate boundary).
struct CoverageParams {
    view_from_clip: mat4x4<f32>,
    screen_size: vec2<f32>,
    surfel_grid: vec2<u32>,
    tile: u32,
    near: f32,
    normal_sharpness: f32,
    axial_tolerance: f32,
    intensity: f32,
    gather_radius: u32,
    _pad0: u32,
    _pad1: u32,
}

var<immediate> params: CoverageParams;

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var surfels: array<Surfel>;
@group(0) @binding(3) var gi_out: texture_storage_2d<rgba16float, write>;

fn is_finite(x: f32) -> bool {
    return x == x && abs(x) <= F32_FINITE_MAX;
}

fn finite_or_zero(x: f32) -> f32 {
    if (is_finite(x)) {
        return x;
    }
    return 0.0;
}

fn sanitize_vec(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z));
}

fn sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    let s = vec3<f32>(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z));
    return max(s, vec3<f32>(0.0, 0.0, 0.0));
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (is_finite(len_sq) && len_sq > 1.0e-24) {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn stable_exp(x: f32) -> f32 {
    if (!is_finite(x)) {
        return 0.0;
    }
    return exp(clamp(x, -80.0, 0.0));
}

// In-plane (radial) distance from the surfel disc to `point` (golden
// `radial_distance`).
fn radial_distance(pos: vec3<f32>, normal: vec3<f32>, point: vec3<f32>) -> f32 {
    let delta = point - pos;
    let axial = dot(delta, normal);
    let in_plane = delta - normal * axial;
    let len_sq = dot(in_plane, in_plane);
    if (len_sq > 0.0) {
        return sqrt(len_sq);
    }
    return 0.0;
}

// Radial coverage weight `max(0, 1 - (r / radius)^2)` (golden `radial_weight`).
fn radial_weight(pos: vec3<f32>, normal: vec3<f32>, radius: f32, point: vec3<f32>) -> f32 {
    let t = clamp(radial_distance(pos, normal, point) / radius, 0.0, 1.0);
    return clamp(1.0 - t * t, 0.0, 1.0);
}

// Axial coverage weight `exp(-|axial| / (radius * tol))` (golden `axial_weight`).
fn axial_weight(pos: vec3<f32>, normal: vec3<f32>, radius: f32, point: vec3<f32>, tol: f32) -> f32 {
    let axial = abs(dot(point - pos, normal));
    let scale = radius * max(tol, 0.0);
    let denom = max(scale, MIN_RADIUS);
    return stable_exp(-axial / denom);
}

// Orientation-agreement weight `max(0, dot(a, b))^sharpness` (golden
// `normal_consistency`).
fn normal_consistency(a: vec3<f32>, b: vec3<f32>, sharpness: f32) -> f32 {
    let cosine = clamp(dot(safe_normalize(a), safe_normalize(b)), 0.0, 1.0);
    return clamp(pow(cosine, max(sharpness, 0.0)), 0.0, 1.0);
}

// Reconstruct a view-space position from a pixel uv and its reverse-Z device
// depth via the inverse projection.
fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

fn surfel_position(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.pos_x, s.pos_y, s.pos_z);
}

fn surfel_normal(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.normal_x, s.normal_y, s.normal_z);
}

fn surfel_radiance(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.radiance_x, s.radiance_y, s.radiance_z);
}

// Full coverage weight of `s` for a shading point (golden `Surfel::coverage`):
// radial x axial x normal, short-circuited to zero outside the disc.
fn coverage_weight(s: Surfel, point: vec3<f32>, point_normal: vec3<f32>) -> f32 {
    let pos = surfel_position(s);
    let normal = surfel_normal(s);
    let w_radial = radial_weight(pos, normal, s.radius, point);
    if (w_radial <= 0.0) {
        return 0.0;
    }
    let w_axial = axial_weight(pos, normal, s.radius, point, params.axial_tolerance);
    let w_normal = normal_consistency(normal, point_normal, params.normal_sharpness);
    return clamp(w_radial * w_axial * w_normal, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn surface_cache_coverage_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u32(params.screen_size.x) || id.y >= u32(params.screen_size.y)) {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));

    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: device depth <= 0 is the far plane / sky; nothing to gather.
    if (device_depth <= 0.0) {
        textureStore(gi_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5, 0.5)) / params.screen_size;
    let point = sanitize_vec(reconstruct_view_position(uv, device_depth));
    let packed_normal = textureLoad(normal_roughness, pixel, 0).xyz;
    let point_normal = safe_normalize(packed_normal * 2.0 - 1.0);

    let grid = params.surfel_grid;
    let tile = max(params.tile, 1u);
    let center_tile = vec2<i32>(i32(id.x / tile), i32(id.y / tile));
    let radius = i32(params.gather_radius);

    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var weight = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let nx = center_tile.x + dx;
            let ny = center_tile.y + dy;
            if (nx < 0 || ny < 0 || nx >= i32(grid.x) || ny >= i32(grid.y)) {
                continue;
            }
            let nidx = u32(ny) * grid.x + u32(nx);
            let s = surfels[nidx];
            if (s.valid == 0u) {
                continue;
            }
            let w = coverage_weight(s, point, point_normal);
            if (w > 0.0) {
                sum = sum + sanitize_rgb(surfel_radiance(s)) * w;
                weight = weight + w;
            }
        }
    }

    var out_rgb = vec3<f32>(0.0, 0.0, 0.0);
    if (weight > 1.0e-12) {
        out_rgb = sanitize_rgb(sum / weight);
    }
    textureStore(gi_out, pixel, vec4<f32>(out_rgb * params.intensity, clamp(weight, 0.0, 1.0)));
}
