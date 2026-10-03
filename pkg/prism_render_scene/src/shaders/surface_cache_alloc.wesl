// Surface-cache alloc pass: seed one fresh surfel per screen tile.
//
// One invocation per surfel (screen tile). The invocation reconstructs the
// tile centre's view-space position + normal from the `SSR` prepass (reverse-Z
// depth + packed `normal_roughness`), samples the pre-exposed scene colour and
// writes a freshly observed surfel into the `scratch_current` buffer. The
// `update` pass later blends it into the persistent surfel buffer.
//
// Positions and normals are VIEW-space (recovered via the inverse projection
// and the view normals the prepass stores as `*0.5 + 0.5`). A sky / invalid
// pixel (reverse-Z device depth <= 0) writes an all-zero surfel with `valid`
// cleared so the `update` pass drops it.
//
// The geometry/sanitise helpers mirror the golden
// `prism_render_shading::gi::surface_cache` CPU twin verbatim so the device
// matches the CPU reference bit-for-bit on finite inputs.

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

// Immediate (push-constant) block mirroring `GpuSurfaceCacheAllocParams`
// (96 bytes: the `mat4x4` leads for its 16-byte alignment, then the `vec2`s,
// then the scalar tail padded to the 16-byte immediate boundary).
struct AllocParams {
    view_from_clip: mat4x4<f32>,
    screen_size: vec2<f32>,
    surfel_grid: vec2<u32>,
    tile: u32,
    near: f32,
    radius_scale: f32,
    _pad0: u32,
}

var<immediate> params: AllocParams;

@group(0) @binding(0) var scene_depth: texture_2d<f32>;
@group(0) @binding(1) var normal_roughness: texture_2d<f32>;
@group(0) @binding(2) var scene_color: texture_2d<f32>;
@group(0) @binding(3) var scratch_current: array<Surfel>;

// `true` when `x` is finite (neither `NaN` nor an infinity).
fn is_finite(x: f32) -> bool {
    return x == x && abs(x) <= F32_FINITE_MAX;
}

// Replace a non-finite scalar with zero (golden `finite_or_zero`).
fn finite_or_zero(x: f32) -> f32 {
    if (is_finite(x)) {
        return x;
    }
    return 0.0;
}

// Replace any non-finite component of a position with zero (golden
// `sanitize_vec`).
fn sanitize_vec(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z));
}

// Sanitise a linear-RGB triple finite and non-negative (golden `sanitize_rgb`).
fn sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    let s = vec3<f32>(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z));
    return max(s, vec3<f32>(0.0, 0.0, 0.0));
}

// Normalise a vector, falling back to `+z` for zero-length / non-finite input
// (golden `safe_normalize`).
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (is_finite(len_sq) && len_sq > 1.0e-24) {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// Reconstruct a view-space position from a pixel uv and its reverse-Z device
// depth via the inverse projection.
fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

// Writes an all-zero, invalid surfel into the scratch slot.
fn write_invalid(idx: u32) {
    scratch_current[idx].pos_x = 0.0;
    scratch_current[idx].pos_y = 0.0;
    scratch_current[idx].pos_z = 0.0;
    scratch_current[idx].normal_x = 0.0;
    scratch_current[idx].normal_y = 0.0;
    scratch_current[idx].normal_z = 1.0;
    scratch_current[idx].radius = MIN_RADIUS;
    scratch_current[idx].radiance_x = 0.0;
    scratch_current[idx].radiance_y = 0.0;
    scratch_current[idx].radiance_z = 0.0;
    scratch_current[idx].sample_count = 0u;
    scratch_current[idx].valid = 0u;
}

@compute @workgroup_size(64, 1, 1)
fn surface_cache_alloc_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let surfel_count = params.surfel_grid.x * params.surfel_grid.y;
    if (idx >= surfel_count) {
        return;
    }

    // Surfel -> tile coordinate -> tile-centre pixel (clamped to the frame).
    let tile_x = idx % params.surfel_grid.x;
    let tile_y = idx / params.surfel_grid.x;
    let half = params.tile / 2u;
    let max_x = u32(max(params.screen_size.x, 1.0)) - 1u;
    let max_y = u32(max(params.screen_size.y, 1.0)) - 1u;
    let px = min(tile_x * params.tile + half, max_x);
    let py = min(tile_y * params.tile + half, max_y);
    let pixel = vec2<i32>(i32(px), i32(py));

    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: device depth <= 0 is the far plane / sky; no surface here.
    if (device_depth <= 0.0) {
        write_invalid(idx);
        return;
    }

    let uv = (vec2<f32>(f32(px), f32(py)) + vec2<f32>(0.5, 0.5)) / params.screen_size;
    let view_pos = sanitize_vec(reconstruct_view_position(uv, device_depth));
    let packed_normal = textureLoad(normal_roughness, pixel, 0).xyz;
    let view_normal = safe_normalize(packed_normal * 2.0 - 1.0);
    let radiance = sanitize_rgb(textureLoad(scene_color, pixel, 0).rgb);

    // View-space surfel radius: the tile's half-extent back-projected through
    // the perspective divide (a tile covers ~`tile` px at `-near`, scaling with
    // view depth `-view_pos.z`), floored at `MIN_RADIUS` and gained by
    // `radius_scale`.
    let depth_scale = -view_pos.z;
    let radius = max(f32(params.tile) * 0.5 * depth_scale / max(params.near * params.screen_size.y, 1.0), MIN_RADIUS) * max(params.radius_scale, MIN_RADIUS);

    scratch_current[idx].pos_x = view_pos.x;
    scratch_current[idx].pos_y = view_pos.y;
    scratch_current[idx].pos_z = view_pos.z;
    scratch_current[idx].normal_x = view_normal.x;
    scratch_current[idx].normal_y = view_normal.y;
    scratch_current[idx].normal_z = view_normal.z;
    scratch_current[idx].radius = radius;
    scratch_current[idx].radiance_x = radiance.x;
    scratch_current[idx].radiance_y = radiance.y;
    scratch_current[idx].radiance_z = radiance.z;
    scratch_current[idx].sample_count = 1u;
    scratch_current[idx].valid = 1u;
}
