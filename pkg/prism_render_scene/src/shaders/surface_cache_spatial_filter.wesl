// Surface-cache spatial-filter pass: bilateral neighbour reuse.
//
// One invocation per surfel. The invocation gathers the persistent surfels in
// a `filter_radius` box around the centre and forms a geometry-weighted mean of
// their radiance, writing the result into `scratch_filtered` (the `coverage`
// pass reads it). The geometry weight is the golden
// `Surfel::geometric_weight` (radial x axial x normal), so a locally constant
// signal is reproduced exactly and an incompatible neighbourhood falls back to
// the centre radiance rather than emitting `NaN`.
//
// Mirrors `prism_render_shading::gi::surface_cache::spatial_filter` verbatim.

const MIN_RADIUS: f32 = 1.0e-6;
const F32_FINITE_MAX: f32 = 3.40282347e38;

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

// Immediate block mirroring `GpuSurfaceCacheFilterParams` (32 bytes).
struct FilterParams {
    surfel_grid: vec2<u32>,
    normal_sharpness: f32,
    axial_tolerance: f32,
    count: u32,
    filter_radius: u32,
    _pad0: u32,
    _pad1: u32,
}

var<immediate> params: FilterParams;

@group(0) @binding(0) var surfels: array<Surfel>;
@group(0) @binding(1) var scratch_filtered: array<Surfel>;

fn is_finite(x: f32) -> bool {
    return x == x && abs(x) <= F32_FINITE_MAX;
}

fn finite_or_zero(x: f32) -> f32 {
    if (is_finite(x)) {
        return x;
    }
    return 0.0;
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

fn surfel_position(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.pos_x, s.pos_y, s.pos_z);
}

fn surfel_normal(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.normal_x, s.normal_y, s.normal_z);
}

fn surfel_radiance(s: Surfel) -> vec3<f32> {
    return vec3<f32>(s.radiance_x, s.radiance_y, s.radiance_z);
}

// Golden `Surfel::geometric_weight`: how far the neighbour centre lies off this
// disc (radial x axial, using the centre radius) times their normal agreement,
// short-circuited to zero outside the disc.
fn geometric_weight(center: Surfel, nb: Surfel) -> f32 {
    let pos = surfel_position(center);
    let normal = surfel_normal(center);
    let nb_pos = surfel_position(nb);
    let w_radial = radial_weight(pos, normal, center.radius, nb_pos);
    if (w_radial <= 0.0) {
        return 0.0;
    }
    let w_axial = axial_weight(pos, normal, center.radius, nb_pos, params.axial_tolerance);
    let w_normal = normal_consistency(normal, surfel_normal(nb), params.normal_sharpness);
    return clamp(w_radial * w_axial * w_normal, 0.0, 1.0);
}

// Writes the centre's geometry through with a replaced radiance.
fn write_filtered(idx: u32, center: Surfel, radiance: vec3<f32>) {
    scratch_filtered[idx].pos_x = center.pos_x;
    scratch_filtered[idx].pos_y = center.pos_y;
    scratch_filtered[idx].pos_z = center.pos_z;
    scratch_filtered[idx].normal_x = center.normal_x;
    scratch_filtered[idx].normal_y = center.normal_y;
    scratch_filtered[idx].normal_z = center.normal_z;
    scratch_filtered[idx].radius = center.radius;
    scratch_filtered[idx].radiance_x = radiance.x;
    scratch_filtered[idx].radiance_y = radiance.y;
    scratch_filtered[idx].radiance_z = radiance.z;
    scratch_filtered[idx].sample_count = center.sample_count;
    scratch_filtered[idx].valid = center.valid;
}

@compute @workgroup_size(64, 1, 1)
fn surface_cache_spatial_filter_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) {
        return;
    }

    let center = surfels[idx];
    // Invalid surfels copy through untouched.
    if (center.valid == 0u) {
        write_filtered(idx, center, surfel_radiance(center));
        return;
    }

    let grid = params.surfel_grid;
    let tile_x = i32(idx % grid.x);
    let tile_y = i32(idx / grid.x);
    let radius = i32(params.filter_radius);

    let center_rgb = sanitize_rgb(surfel_radiance(center));
    var sum = center_rgb;
    var weight = 1.0;

    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nx = tile_x + dx;
            let ny = tile_y + dy;
            if (nx < 0 || ny < 0 || nx >= i32(grid.x) || ny >= i32(grid.y)) {
                continue;
            }
            let nidx = u32(ny) * grid.x + u32(nx);
            let nb = surfels[nidx];
            if (nb.valid == 0u) {
                continue;
            }
            let w = geometric_weight(center, nb);
            if (w > 0.0) {
                sum = sum + sanitize_rgb(surfel_radiance(nb)) * w;
                weight = weight + w;
            }
        }
    }

    var out = center_rgb;
    if (weight > 1.0e-12) {
        out = sanitize_rgb(sum / weight);
    }
    write_filtered(idx, center, out);
}
