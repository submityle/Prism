// Surfel coverage-gather producer kernel: the on-device twin of the CPU golden
// `Surfel::coverage` in `prism_render_shading::gi::surface_cache::surfel`.
//
// This is the surface cache's *consumer* pass. A Lumen-style surface cache
// resolves indirect radiance at a shading point by gathering the surfels whose
// oriented discs cover it, weighting every surfel by its coverage kernel
// (radial coverage x off-plane falloff x point/surfel normal agreement) and
// returning the coverage-weighted mean of their cached radiance. A point that
// no surfel covers resolves to zero radiance with zero weight, so the host can
// fall back to another estimator.
//
// The kernel runs one thread per shading point. Each point carries a
// `[surfel_offset, surfel_offset + surfel_count)` slice into the shared
// `surfels` buffer, so the per-point reduction is a sequential in-order sum —
// identical to the golden's gather loop — and therefore bit-for-bit
// reproducible by the CPU mirror (which uses the same `exp`/`pow` as the
// golden). Only `+ - * /`, `sqrt`, `exp`, `pow`, `min`/`max`/`clamp` and
// comparisons are used — the portable core-WGSL subset — so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard surfel-coverage gather driven by a coverage kernel; no
// Unreal Engine source or derived code.

// Dispatch parameters. 16-byte `repr(C)` matching `GpuSurfelCoverageParams`.
struct CoverageParams {
    count: u32,
    normal_sharpness: f32,
    axial_tolerance: f32,
    reserved: u32,
}

// One shading point plus its surfel slice. 32-byte `repr(C)` matching
// `GpuCoveragePoint`; all scalar to avoid `vec3` straddle.
struct Point {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    surfel_offset: u32,
    surfel_count: u32,
}

// One candidate surfel plus its cached radiance. 40-byte `repr(C)` matching
// `GpuCoverageSurfel`.
struct CandidateSurfel {
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

// Gathered radiance plus the accumulated total coverage weight. 16-byte
// `repr(C)` matching `GpuCoverageResult`.
struct CoverageResult {
    radiance_x: f32,
    radiance_y: f32,
    radiance_z: f32,
    total_weight: f32,
}

@group(0) @binding(0) var<uniform> params: CoverageParams;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> surfels: array<CandidateSurfel>;
@group(0) @binding(3) var<storage, read_write> results: array<CoverageResult>;

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

// In-plane (radial) distance from the surfel disc to `point`, mirroring the
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

// Full coverage weight of `s` for a shading point (golden `Surfel::coverage`):
// radial x axial x normal, short-circuited to zero outside the disc.
fn coverage_weight(s: CandidateSurfel, point: vec3<f32>, point_normal: vec3<f32>) -> f32 {
    let pos = vec3<f32>(s.pos_x, s.pos_y, s.pos_z);
    let normal = vec3<f32>(s.normal_x, s.normal_y, s.normal_z);
    let w_radial = radial_weight(pos, normal, s.radius, point);
    if (w_radial <= 0.0) {
        return 0.0;
    }
    let w_axial = axial_weight(pos, normal, s.radius, point, params.axial_tolerance);
    let w_normal = normal_consistency(normal, point_normal, params.normal_sharpness);
    return clamp(w_radial * w_axial * w_normal, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn surfel_coverage_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = points[idx];
    let point = sanitize_vec(vec3<f32>(p.pos_x, p.pos_y, p.pos_z));
    let point_normal = vec3<f32>(p.normal_x, p.normal_y, p.normal_z);

    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var weight = 0.0;
    for (var i = 0u; i < p.surfel_count; i = i + 1u) {
        let s = surfels[p.surfel_offset + i];
        let w = coverage_weight(s, point, point_normal);
        if (w <= 0.0) {
            continue;
        }
        let rgb = sanitize_rgb(vec3<f32>(s.radiance_x, s.radiance_y, s.radiance_z));
        sum = sum + rgb * w;
        weight = weight + w;
    }

    var out_rgb: vec3<f32>;
    if (weight <= 1.0e-12) {
        out_rgb = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        out_rgb = sanitize_rgb(sum / weight);
    }

    var result: CoverageResult;
    result.radiance_x = out_rgb.x;
    result.radiance_y = out_rgb.y;
    result.radiance_z = out_rgb.z;
    result.total_weight = weight;
    results[idx] = result;
}
