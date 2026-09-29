//! GPU compute software rasterizer for thin hair strands (shared base stage).
//!
//! The CPU side `prism_render_architecture::hair::raster` is only a
//! *classifier/binner*: it decides, per strand segment, whether the segment is
//! thin enough for the compute software path and describes the rasterizer's ABI
//! (`HairSoftRasterAbi`: sub-pixel sample count, coverage epsilon, tile size).
//! It deliberately leaves the physical rasterization to the backend. This
//! shader is that backend: the compute software / visibility pass UE5 Groom and
//! AMD `TressFX` use for sub-pixel strands, honouring the same ABI constants so
//! the CPU classifier and the GPU rasterizer agree on behaviour.
//!
//! A render strand segment is a screen-space *capsule* (a thick line of radius
//! `width_px * 0.5`), not a triangle, so it is scattered into a per-pixel
//! vis-buffer with analytic sub-pixel coverage. Because baseline WGSL exposes
//! only 32-bit atomics, nearest-fragment resolution is a lock-free two-kernel
//! pass over the same segment scatter (the standard Nanite-style vis-buffer
//! without 64-bit atomics):
//!
//! 1. `hair_raster_depth`  — atomicMin the quantized segment depth into
//!    `vis_depth` for every covered pixel, keeping the nearest fragment.
//! 2. `hair_raster_resolve` — for every covered pixel whose winning depth equals
//!    this segment's quantized depth, publish the segment id and its analytic
//!    coverage into `vis_id` / `vis_coverage` for the deferred hair shade.
//!
//! One invocation owns one segment and scatters over the pixels in its
//! radius-expanded screen AABB; tiled binning (`tile_size_px`,
//! `max_segments_per_tile`) is an alternate dispatch scheduling concern handled
//! by the backend and does not change the per-pixel coverage math here.

/// 24-bit depth quantization range. Smaller quantized value == nearer, so
/// `atomicMin` keeps the closest fragment (matches a reversed-Z-agnostic
/// `0=near..1=far` convention chosen by the caller).
const HAIR_DEPTH_QUANT: f32 = 16777215.0; // 2^24 - 1
/// Fixed-point scale for the 0..=1 coverage stored in `vis_coverage`.
const HAIR_COVERAGE_QUANT: f32 = 65535.0; // 2^16 - 1
/// Sentinel "no fragment" depth (farther than any real quantized depth).
const HAIR_DEPTH_EMPTY: u32 = 0xFFFFFFFFu;

/// One projected strand segment handed to the software rasterizer. Screen-space
/// endpoints are in pixel coordinates; depths are the caller's `0=near..1=far`
/// values; `width_px` is the projected strand diameter; `seg_id` is the packed
/// strand/segment identity resolved for deferred shading.
struct HairRasterSegment {
    p0: vec2<f32>,
    p1: vec2<f32>,
    depth0: f32,
    depth1: f32,
    width_px: f32,
    seg_id: u32,
}

/// Global rasterizer parameters. Mirrors the fields of the CPU
/// `HairSoftRasterAbi` plus the viewport and dispatch bound.
struct HairRasterParamsGpu {
    /// Framebuffer dimensions in pixels (`x` = width, `y` = height).
    viewport: vec2<u32>,
    /// Coverage strictly below this is not published (mirrors
    /// `HairSoftRasterAbi::coverage_epsilon` / `DEFAULT_HAIR_MIN_COVERAGE`).
    coverage_epsilon: f32,
    /// Number of stratified sub-pixel samples averaged per pixel
    /// (`HairSoftRasterAbi::subpixel_samples`, default 8).
    subpixel_samples: u32,
    /// Number of segments in `segments`; bounds the dispatch.
    segment_count: u32,
}

@group(0) @binding(0) var<storage, read> segments: array<HairRasterSegment>;
// Per-pixel nearest quantized depth; pass 1 target, pass 2 read.
@group(0) @binding(1) var<storage, read_write> vis_depth: array<atomic<u32>>;
// Per-pixel winning segment id; published in pass 2.
@group(0) @binding(2) var<storage, read_write> vis_id: array<u32>;
// Per-pixel published coverage (fixed-point 0..=HAIR_COVERAGE_QUANT).
@group(0) @binding(3) var<storage, read_write> vis_coverage: array<u32>;

var<immediate> params: HairRasterParamsGpu;

/// Squared 2D distance from a point to a segment `[a, b]`, plus the clamped
/// parametric position `t` used to interpolate depth. Returns `vec3(dist2, t)`.
fn hair_point_segment(point: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec3<f32> {
    let ab = b - a;
    let ap = point - a;
    let denom = dot(ab, ab);
    var t = 0.0;
    if denom > 1.0e-12 {
        t = clamp(dot(ap, ab) / denom, 0.0, 1.0);
    }
    let closest = a + ab * t;
    let d = point - closest;
    return vec3<f32>(dot(d, d), t, 0.0);
}

/// Analytic sub-pixel coverage of a capsule of radius `radius` over the pixel
/// centred at `pixel_center`, averaged over `subpixel_samples` stratified
/// offsets. A one-pixel-wide analytic AA band on the capsule boundary keeps
/// thin strands from shimmering. Also returns the interpolated depth `t` at the
/// nearest sample. Result: `vec2(coverage, depth_t)`.
fn hair_capsule_coverage(
    pixel_center: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    radius: f32,
    sample_count: u32,
) -> vec2<f32> {
    let samples = max(sample_count, 1u);
    var inside = 0.0;
    var best_t = 0.0;
    var best_dist2 = 3.4e38;
    let inv = 1.0 / f32(samples);
    for (var i = 0u; i < samples; i = i + 1u) {
        // Cheap deterministic stratified offset in [-0.5, 0.5)^2 (van der
        // Corput style radical inverse on the sample index).
        let fi = f32(i);
        let ox = fract(fi * 0.618034) - 0.5;
        let oy = fract(fi * 0.381966) - 0.5;
        let sample_point = pixel_center + vec2<f32>(ox, oy);
        let ds = hair_point_segment(sample_point, a, b);
        let dist = sqrt(ds.x);
        // Analytic 1px AA band: fully inside at dist<=radius-0.5, fades to 0 by
        // dist=radius+0.5.
        let cov = clamp(radius + 0.5 - dist, 0.0, 1.0);
        inside = inside + cov * inv;
        if ds.x < best_dist2 {
            best_dist2 = ds.x;
            best_t = ds.y;
        }
    }
    return vec2<f32>(clamp(inside, 0.0, 1.0), best_t);
}

/// Quantizes a `0=near..1=far` depth into the 24-bit atomic domain.
fn hair_quantize_depth(depth: f32) -> u32 {
    return u32(clamp(depth, 0.0, 1.0) * HAIR_DEPTH_QUANT);
}

/// Shared preamble: validate the invocation's segment, returning whether it is
/// live plus its radius-expanded integer screen AABB in `vec4(x0,y0,x1,y1)`.
fn hair_segment_bounds(seg: HairRasterSegment) -> vec4<i32> {
    let radius = max(seg.width_px, 0.0) * 0.5;
    let lo = min(seg.p0, seg.p1) - vec2<f32>(radius + 1.0);
    let hi = max(seg.p0, seg.p1) + vec2<f32>(radius + 1.0);
    let x0 = max(i32(floor(lo.x)), 0);
    let y0 = max(i32(floor(lo.y)), 0);
    let x1 = min(i32(ceil(hi.x)), i32(params.viewport.x) - 1);
    let y1 = min(i32(ceil(hi.y)), i32(params.viewport.y) - 1);
    return vec4<i32>(x0, y0, x1, y1);
}

@compute @workgroup_size(64)
fn hair_raster_depth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.segment_count {
        return;
    }
    let seg = segments[s];
    let radius = max(seg.width_px, 0.0) * 0.5;
    if radius <= 0.0 || params.viewport.x == 0u || params.viewport.y == 0u {
        return;
    }
    let bounds = hair_segment_bounds(seg);
    for (var y = bounds.y; y <= bounds.w; y = y + 1) {
        for (var x = bounds.x; x <= bounds.z; x = x + 1) {
            let center = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
            let cov = hair_capsule_coverage(center, seg.p0, seg.p1, radius, params.subpixel_samples);
            if cov.x < params.coverage_epsilon {
                continue;
            }
            let depth = mix(seg.depth0, seg.depth1, cov.y);
            let quant = hair_quantize_depth(depth);
            let idx = u32(y) * params.viewport.x + u32(x);
            atomicMin(&vis_depth[idx], quant);
        }
    }
}

@compute @workgroup_size(64)
fn hair_raster_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.segment_count {
        return;
    }
    let seg = segments[s];
    let radius = max(seg.width_px, 0.0) * 0.5;
    if radius <= 0.0 || params.viewport.x == 0u || params.viewport.y == 0u {
        return;
    }
    let bounds = hair_segment_bounds(seg);
    for (var y = bounds.y; y <= bounds.w; y = y + 1) {
        for (var x = bounds.x; x <= bounds.z; x = x + 1) {
            let center = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
            let cov = hair_capsule_coverage(center, seg.p0, seg.p1, radius, params.subpixel_samples);
            if cov.x < params.coverage_epsilon {
                continue;
            }
            let depth = mix(seg.depth0, seg.depth1, cov.y);
            let quant = hair_quantize_depth(depth);
            let idx = u32(y) * params.viewport.x + u32(x);
            // Only the segment that won the depth race publishes its identity;
            // the equality test is exact because both sides are the same
            // quantization of the same interpolated depth.
            if atomicLoad(&vis_depth[idx]) == quant {
                vis_id[idx] = seg.seg_id;
                vis_coverage[idx] = u32(clamp(cov.x, 0.0, 1.0) * HAIR_COVERAGE_QUANT);
            }
        }
    }
}
