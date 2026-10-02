// Surfel spatial-filter producer kernel: the on-device twin of the CPU golden
// `spatial_filter` in
// `prism_render_shading::gi::surface_cache::integration`.
//
// A Lumen-style surface cache hides the per-surfel Monte-Carlo noise with a
// bilateral spatial pass: each surfel pulls radiance from its geometrically
// compatible neighbours, weighting every neighbour by the geometric reuse
// weight `Surfel::geometric_weight` (radial coverage x off-plane falloff x
// normal agreement, all evaluated against the centre surfel's own disc). The
// centre always contributes unit weight so the filter degrades to identity
// when every neighbour is incompatible, and a locally constant signal is
// reproduced exactly.
//
// This kernel runs one thread per centre surfel. Each centre carries a
// `[neighbor_offset, neighbor_offset + neighbor_count)` slice into the shared
// `neighbors` buffer, so the per-centre reduction is a sequential in-order sum
// — identical to the golden's `for` loop — and therefore bit-for-bit
// reproducible by the CPU mirror (which uses the same `exp`/`pow` as the
// golden). Only `+ - * /`, `sqrt`, `exp`, `pow`, `min`/`max`/`clamp` and
// comparisons are used — the portable core-WGSL subset — so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard bilateral surfel reuse driven by a geometric weight; no
// Unreal Engine source or derived code.

// Dispatch parameters. 16-byte `repr(C)` matching `GpuSpatialFilterParams`.
struct FilterParams {
    count: u32,
    normal_sharpness: f32,
    axial_tolerance: f32,
    reserved: u32,
}

// One centre surfel plus its radiance and its neighbour slice. 48-byte
// `repr(C)` matching `GpuSpatialCenter`; all scalar to avoid `vec3` straddle.
struct Center {
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
    neighbor_offset: u32,
    neighbor_count: u32,
}

// One neighbour surfel plus its radiance. 40-byte `repr(C)` matching
// `GpuSpatialNeighbor`.
struct Neighbor {
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
}

// Filtered radiance plus the accumulated total weight. 16-byte `repr(C)`
// matching `GpuSpatialResult`.
struct FilterResult {
    radiance_x: f32,
    radiance_y: f32,
    radiance_z: f32,
    total_weight: f32,
}

@group(0) @binding(0) var<uniform> params: FilterParams;
@group(0) @binding(1) var<storage, read> centers: array<Center>;
@group(0) @binding(2) var<storage, read> neighbors: array<Neighbor>;
@group(0) @binding(3) var<storage, read_write> results: array<FilterResult>;

// Smallest surfel radius; keeps the axial falloff division finite. Mirrors
// `MIN_RADIUS` in the surfel golden.
const MIN_RADIUS: f32 = 1.0e-6;

// Largest finite `f32`; used with the self-equality `NaN` test to reproduce the
// golden `f32::is_finite` guard without an `isFinite` builtin.
const F32_FINITE_MAX: f32 = 3.40282347e38;

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

// Sanitise a linear-RGB triple finite and non-negative (golden `sanitize_rgb`).
fn sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    let s = vec3<f32>(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z));
    return max(s, vec3<f32>(0.0, 0.0, 0.0));
}

// Normalise a vector, falling back to `+z` for zero-length / non-finite input,
// mirroring the golden `safe_normalize`.
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (is_finite(len_sq) && len_sq > 1.0e-24) {
        return v / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// Numerically safe `exp` saturating for large magnitudes (golden `stable_exp`).
fn stable_exp(x: f32) -> f32 {
    if (!is_finite(x)) {
        return 0.0;
    }
    return exp(clamp(x, -80.0, 0.0));
}

// In-plane (radial) distance from the centre disc to `point`, mirroring the
// golden `Surfel::radial_distance`.
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

// Geometric reuse weight of `nb` against the centre disc (golden
// `Surfel::geometric_weight`): radial x axial x normal, short-circuited to zero
// outside the disc.
fn geometric_weight(c: Center, nb: Neighbor) -> f32 {
    let pos = vec3<f32>(c.pos_x, c.pos_y, c.pos_z);
    let normal = vec3<f32>(c.normal_x, c.normal_y, c.normal_z);
    let nb_pos = vec3<f32>(nb.pos_x, nb.pos_y, nb.pos_z);
    let nb_normal = vec3<f32>(nb.normal_x, nb.normal_y, nb.normal_z);
    let w_radial = radial_weight(pos, normal, c.radius, nb_pos);
    if (w_radial <= 0.0) {
        return 0.0;
    }
    let w_axial = axial_weight(pos, normal, c.radius, nb_pos, params.axial_tolerance);
    let w_normal = normal_consistency(normal, nb_normal, params.normal_sharpness);
    return clamp(w_radial * w_axial * w_normal, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn surfel_spatial_filter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let c = centers[idx];
    let center_rgb = sanitize_rgb(vec3<f32>(c.radiance_x, c.radiance_y, c.radiance_z));

    var sum = center_rgb;
    var weight = 1.0;
    for (var i = 0u; i < c.neighbor_count; i = i + 1u) {
        let nb = neighbors[c.neighbor_offset + i];
        let w = geometric_weight(c, nb);
        if (w <= 0.0) {
            continue;
        }
        let nb_rgb = sanitize_rgb(vec3<f32>(nb.radiance_x, nb.radiance_y, nb.radiance_z));
        sum = sum + nb_rgb * w;
        weight = weight + w;
    }

    var out_rgb: vec3<f32>;
    if (weight <= 1.0e-12) {
        out_rgb = center_rgb;
    } else {
        out_rgb = sanitize_rgb(sum / weight);
    }

    var result: FilterResult;
    result.radiance_x = out_rgb.x;
    result.radiance_y = out_rgb.y;
    result.radiance_z = out_rgb.z;
    result.total_weight = weight;
    results[idx] = result;
}
