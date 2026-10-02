//! `wgpu` compute twin of the `Forward+` tile-light-cull numeric kernel
//! ([`tile_light_cull`](prism_render_architecture::particle::tile_light_cull),
//! particle design §8.3 *Scene* category).
//!
//! The `CPU` golden
//! [`tile_light_cull`](prism_render_architecture::particle::tile_light_cull)
//! partitions the framebuffer into fixed pixel tiles, back-projects each tile's
//! screen rectangle into a view-space sub-frustum, and bins every spherical
//! light into the tiles its footprint touches. This twin reproduces, one thread
//! per query, the module's pure numeric surface as a tagged batch so one
//! dispatch can mix vector algebra, plane, grid-rectangle, frustum-construction
//! and sphere-cull queries:
//!
//! - the [`Vec3`](prism_render_architecture::particle::tile_light_cull::Vec3)
//!   primitives `plus`, `minus`, `scale`, `dot`, `cross`, `length` and
//!   `normalize_or_zero`;
//! - [`SphereLight::is_active`](prism_render_architecture::particle::tile_light_cull::SphereLight::is_active),
//!   the strictly-positive-radius predicate;
//! - [`Plane::from_inward_normal`](prism_render_architecture::particle::tile_light_cull::Plane::from_inward_normal)
//!   and
//!   [`Plane::signed_distance`](prism_render_architecture::particle::tile_light_cull::Plane::signed_distance);
//! - [`DepthRange::new`](prism_render_architecture::particle::tile_light_cull::DepthRange::new),
//!   the edge-ordering constructor;
//! - [`TileGrid::tile_pixel_rect`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_pixel_rect),
//!   the clamped per-tile pixel rectangle;
//! - [`TileGrid::tile_frustum`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_frustum),
//!   the four apex-through side planes plus the clamped near/far pair;
//! - [`TileFrustum::intersects_sphere`](prism_render_architecture::particle::tile_light_cull::TileFrustum::intersects_sphere),
//!   the conservative sphere-vs-frustum cull predicate.
//!
//! [`GpuTileLightCull`] is the on-device twin: a passing real-device parity test
//! is direct evidence the ported kernel evaluates the same closed forms and
//! classifies the same degenerate geometry the reference does, not merely that
//! the shader compiles.
//!
//! # What stays on the host
//!
//! The reference binning pass
//! [`cull_tile_lights`](prism_render_architecture::particle::tile_light_cull::cull_tile_lights)
//! and its packed
//! [`TileLightList`](prism_render_architecture::particle::tile_light_cull::TileLightList)
//! are **not** twinned: they walk a variable-length light list over a tile grid
//! and append into a growing index `Vec` with a running prefix sum, which has no
//! place in the one-thread-per-element, no-aliasing kernel contract. The
//! `*_storage_bytes` `std430` byte utilities and the `width`/`height`-to-`u16`
//! range conversion the reference performs before the pixel-to-`NDC` divide also
//! stay host-side; the twin receives the framebuffer extents already within the
//! `u16` range as plain `u32`, exactly the values the host would clamp them to.
//! The twin instead exposes the pure per-tile numeric pieces the pass composes;
//! a host that needs the full bin assembles the index list itself and hands the
//! twin the per-tile geometry.
//!
//! # Degenerate regimes
//!
//! `normalize_or_zero` collapses a vector shorter than `GUARD_EPS` to the zero
//! vector rather than dividing by zero; `tile_frustum` floors the framebuffer
//! extents and the near plane to `GUARD_EPS` and pushes `far` strictly beyond
//! `near`; `tile_pixel_rect` clamps off-grid tile coordinates to the last valid
//! tile. The parity fixtures stay clear of these thresholds by rejection
//! sampling so the comparison exercises the live solve.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `dot`,
//! `sqrt` (reached only through the guarded `normalize_or_zero` and `length`,
//! exactly as the reference does), `select` and `+ - * /` — with no
//! transcendental call, no `u64`, no `u16` and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. Each thread performs a
//! fixed, bounded sequence (the sphere cull unrolls its four side planes), so it
//! provably terminates.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! guarded divides, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` lanes and an exact
//! match on the discrete pixel-rectangle integers and the `is_active` / cull
//! flags.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::tile_light_cull`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::tile_light_cull::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Operation tag: [`Vec3::plus`](prism_render_architecture::particle::tile_light_cull::Vec3::plus).
const OP_VEC3_PLUS: u32 = 0;
/// Operation tag: [`Vec3::minus`](prism_render_architecture::particle::tile_light_cull::Vec3::minus).
const OP_VEC3_MINUS: u32 = 1;
/// Operation tag: [`Vec3::scale`](prism_render_architecture::particle::tile_light_cull::Vec3::scale).
const OP_VEC3_SCALE: u32 = 2;
/// Operation tag: [`Vec3::dot`](prism_render_architecture::particle::tile_light_cull::Vec3::dot).
const OP_VEC3_DOT: u32 = 3;
/// Operation tag: [`Vec3::cross`](prism_render_architecture::particle::tile_light_cull::Vec3::cross).
const OP_VEC3_CROSS: u32 = 4;
/// Operation tag: [`Vec3::length`](prism_render_architecture::particle::tile_light_cull::Vec3::length).
const OP_VEC3_LENGTH: u32 = 5;
/// Operation tag: [`Vec3::normalize_or_zero`](prism_render_architecture::particle::tile_light_cull::Vec3::normalize_or_zero).
const OP_VEC3_NORMALIZE_OR_ZERO: u32 = 6;
/// Operation tag: [`SphereLight::is_active`](prism_render_architecture::particle::tile_light_cull::SphereLight::is_active).
const OP_SPHERE_IS_ACTIVE: u32 = 7;
/// Operation tag: [`Plane::from_inward_normal`](prism_render_architecture::particle::tile_light_cull::Plane::from_inward_normal).
const OP_PLANE_FROM_INWARD_NORMAL: u32 = 8;
/// Operation tag: [`Plane::signed_distance`](prism_render_architecture::particle::tile_light_cull::Plane::signed_distance).
const OP_PLANE_SIGNED_DISTANCE: u32 = 9;
/// Operation tag: [`DepthRange::new`](prism_render_architecture::particle::tile_light_cull::DepthRange::new).
const OP_DEPTH_RANGE_NEW: u32 = 10;
/// Operation tag: [`TileGrid::tile_pixel_rect`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_pixel_rect).
const OP_TILE_PIXEL_RECT: u32 = 11;
/// Operation tag: [`TileGrid::tile_frustum`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_frustum).
const OP_TILE_FRUSTUM: u32 = 12;
/// Operation tag: [`TileFrustum::intersects_sphere`](prism_render_architecture::particle::tile_light_cull::TileFrustum::intersects_sphere).
const OP_INTERSECTS_SPHERE: u32 = 13;

/// Discrete flag code written for an active light or a surviving sphere; decoded
/// with `== 1` so no `f32` equality is used.
const CODE_FLAG: u32 = 1;

/// The portable core-`WGSL` tile-light-cull numeric kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// dispatches on a per-query operation tag and mirrors the `CPU` golden
/// [`tile_light_cull`](prism_render_architecture::particle::tile_light_cull)
/// numeric surface; see the module documentation for the formulae.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tile_light_cull`。
const TILE_LIGHT_CULL_WGSL: &str = r#"
// tile_light_cull twin: one thread per query dispatches on an operation tag and
// reproduces the CPU golden `particle::tile_light_cull` numeric surface — the
// Vec3 plus / minus / scale / dot / cross / length / normalize_or_zero
// primitives, the sphere is_active predicate, the plane from_inward_normal and
// signed_distance, the depth-range edge ordering, the clamped per-tile pixel
// rectangle, the four apex-through side planes of a tile sub-frustum, and the
// conservative sphere-vs-frustum cull. It mirrors the reference branch for
// branch, uses only the portable core-WGSL subset (min/max/dot/sqrt/select and
// + - * /), takes no optional feature, uses no u64/u16, and runs unmodified on
// Metal, Vulkan and DX12. Each thread runs a fixed, bounded sequence, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::tile_light_cull;
// 无第三方引擎源码或衍生代码。

// Shared guard tolerance mirrored from the golden `GUARD_EPS`.
const GUARD_EPS: f32 = 1.0e-6;

// Operation tags (kept in lockstep with the host constants).
const OP_VEC3_PLUS: u32 = 0u;
const OP_VEC3_MINUS: u32 = 1u;
const OP_VEC3_SCALE: u32 = 2u;
const OP_VEC3_DOT: u32 = 3u;
const OP_VEC3_CROSS: u32 = 4u;
const OP_VEC3_LENGTH: u32 = 5u;
const OP_VEC3_NORMALIZE_OR_ZERO: u32 = 6u;
const OP_SPHERE_IS_ACTIVE: u32 = 7u;
const OP_PLANE_FROM_INWARD_NORMAL: u32 = 8u;
const OP_PLANE_SIGNED_DISTANCE: u32 = 9u;
const OP_DEPTH_RANGE_NEW: u32 = 10u;
const OP_TILE_PIXEL_RECT: u32 = 11u;
const OP_TILE_FRUSTUM: u32 = 12u;
const OP_INTERSECTS_SPHERE: u32 = 13u;

// Result-kind tags.
const KIND_SCALAR: u32 = 0u;
const KIND_VECTOR: u32 = 1u;
const KIND_BOOL: u32 = 2u;
const KIND_RANGE: u32 = 3u;
const KIND_RECT: u32 = 4u;
const KIND_FRUSTUM: u32 = 5u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which golden function this slot evaluates.
    op: u32,
    // Tile coordinate and grid integers (also drive the pixel rectangle).
    tile_x: u32,
    tile_y: u32,
    tile_size: u32,
    tile_count_x: u32,
    tile_count_y: u32,
    width: u32,
    height: u32,
    // Camera half-extent slopes and the near / far depth pair.
    slope_x: f32,
    slope_y: f32,
    near: f32,
    far: f32,
    // Scalar operands: sphere radius, scale factor, depth-range edges.
    radius: f32,
    factor: f32,
    range_a: f32,
    range_b: f32,
    // Primary vector slot: vec operand a / inward normal / sphere center.
    vec_a: vec3<f32>,
    pad_a: f32,
    // Secondary vector slot: vec operand b / query point.
    vec_b: vec3<f32>,
    pad_b: f32,
}

struct Result {
    // Result-kind tag selecting how the host decodes the lanes.
    kind: u32,
    // Discrete flag: is_active or sphere-survives.
    flag: u32,
    // Clamped pixel rectangle lanes (x0, y0, x1, y1).
    rect_x0: u32,
    rect_y0: u32,
    rect_x1: u32,
    rect_y1: u32,
    pad_u0: u32,
    pad_u1: u32,
    // Scalar result (dot / length / signed_distance) or depth-range min.
    scalar: f32,
    // Depth-range max.
    extra: f32,
    // Clamped tile-frustum near / far depth.
    near: f32,
    far: f32,
    pad_f0: f32,
    pad_f1: f32,
    pad_f2: f32,
    pad_f3: f32,
    // Vector result, or the first (left) side-plane normal.
    vec_r: vec3<f32>,
    pad_r: f32,
    // Right side-plane normal.
    plane1: vec3<f32>,
    pad_p1: f32,
    // Bottom side-plane normal.
    plane2: vec3<f32>,
    pad_p2: f32,
    // Top side-plane normal.
    plane3: vec3<f32>,
    pad_p3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Right-handed cross product, spelled out exactly like the golden `Vec3::cross`.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    );
}

// Guarded normalize, mirroring the reference `Vec3::normalize_or_zero`: a vector
// shorter than `GUARD_EPS` collapses to the zero vector so a degenerate
// direction never divides by zero.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len > GUARD_EPS) {
        return v * (1.0 / len);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Clamped per-tile pixel rectangle, mirroring the reference `tile_pixel_rect`.
// The grid guarantees `tile_count_x >= 1`, so the `- 1` never underflows for the
// fixtures the twin is handed.
fn tile_pixel_rect(q: Query) -> vec4<u32> {
    let tx = min(q.tile_x, q.tile_count_x - 1u);
    let ty = min(q.tile_y, q.tile_count_y - 1u);
    let x0 = min(tx * q.tile_size, q.width);
    let y0 = min(ty * q.tile_size, q.height);
    let x1 = min((tx + 1u) * q.tile_size, q.width);
    let y1 = min((ty + 1u) * q.tile_size, q.height);
    return vec4<u32>(x0, y0, x1, y1);
}

// One inward-oriented, unit-length side-plane normal, mirroring the reference
// `side` closure: the raw cross is flipped toward the tile center ray so the
// frustum interior always has a positive signed distance.
fn side_normal(edge_a: vec3<f32>, edge_b: vec3<f32>, center_ray: vec3<f32>) -> vec3<f32> {
    let raw = cross3(edge_a, edge_b);
    let oriented = select(raw, raw * -1.0, dot(raw, center_ray) < 0.0);
    return normalize_or_zero(oriented);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let vec_a = q.vec_a;
    let vec_b = q.vec_b;

    var out: Result;
    out.kind = KIND_SCALAR;
    out.flag = 0u;
    out.rect_x0 = 0u;
    out.rect_y0 = 0u;
    out.rect_x1 = 0u;
    out.rect_y1 = 0u;
    out.pad_u0 = 0u;
    out.pad_u1 = 0u;
    out.scalar = 0.0;
    out.extra = 0.0;
    out.near = 0.0;
    out.far = 0.0;
    out.pad_f0 = 0.0;
    out.pad_f1 = 0.0;
    out.pad_f2 = 0.0;
    out.pad_f3 = 0.0;
    out.vec_r = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_r = 0.0;
    out.plane1 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_p1 = 0.0;
    out.plane2 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_p2 = 0.0;
    out.plane3 = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_p3 = 0.0;

    switch q.op {
        case OP_VEC3_PLUS: {
            out.kind = KIND_VECTOR;
            out.vec_r = vec_a + vec_b;
        }
        case OP_VEC3_MINUS: {
            out.kind = KIND_VECTOR;
            out.vec_r = vec_a - vec_b;
        }
        case OP_VEC3_SCALE: {
            out.kind = KIND_VECTOR;
            out.vec_r = vec_a * q.factor;
        }
        case OP_VEC3_DOT: {
            out.kind = KIND_SCALAR;
            out.scalar = dot(vec_a, vec_b);
        }
        case OP_VEC3_CROSS: {
            out.kind = KIND_VECTOR;
            out.vec_r = cross3(vec_a, vec_b);
        }
        case OP_VEC3_LENGTH: {
            out.kind = KIND_SCALAR;
            out.scalar = sqrt(dot(vec_a, vec_a));
        }
        case OP_VEC3_NORMALIZE_OR_ZERO: {
            out.kind = KIND_VECTOR;
            out.vec_r = normalize_or_zero(vec_a);
        }
        case OP_SPHERE_IS_ACTIVE: {
            out.kind = KIND_BOOL;
            out.flag = select(0u, 1u, q.radius > GUARD_EPS);
        }
        case OP_PLANE_FROM_INWARD_NORMAL: {
            out.kind = KIND_VECTOR;
            out.vec_r = normalize_or_zero(vec_a);
        }
        case OP_PLANE_SIGNED_DISTANCE: {
            // `vec_a` is the plane's already-unit inward normal; the signed
            // distance to a point is simply their dot product.
            out.kind = KIND_SCALAR;
            out.scalar = dot(vec_a, vec_b);
        }
        case OP_DEPTH_RANGE_NEW: {
            out.kind = KIND_RANGE;
            out.scalar = min(q.range_a, q.range_b);
            out.extra = max(q.range_a, q.range_b);
        }
        case OP_TILE_PIXEL_RECT: {
            out.kind = KIND_RECT;
            let rect = tile_pixel_rect(q);
            out.rect_x0 = rect.x;
            out.rect_y0 = rect.y;
            out.rect_x1 = rect.z;
            out.rect_y1 = rect.w;
        }
        case OP_TILE_FRUSTUM, OP_INTERSECTS_SPHERE: {
            let rect = tile_pixel_rect(q);
            let inv_w = 1.0 / max(f32(q.width), GUARD_EPS);
            let inv_h = 1.0 / max(f32(q.height), GUARD_EPS);
            // Normalized-device edges: X grows right, Y is flipped so it grows up.
            let nx0 = 2.0 * (f32(rect.x) * inv_w) - 1.0;
            let nx1 = 2.0 * (f32(rect.z) * inv_w) - 1.0;
            // Pixel Y grows downward, so the smaller pixel Y is the larger NDC Y.
            let ny_hi = 1.0 - 2.0 * (f32(rect.y) * inv_h);
            let ny_lo = 1.0 - 2.0 * (f32(rect.w) * inv_h);

            let corner_ll = vec3<f32>(nx0 * q.slope_x, ny_lo * q.slope_y, 1.0);
            let corner_lr = vec3<f32>(nx1 * q.slope_x, ny_lo * q.slope_y, 1.0);
            let corner_ul = vec3<f32>(nx0 * q.slope_x, ny_hi * q.slope_y, 1.0);
            let corner_ur = vec3<f32>(nx1 * q.slope_x, ny_hi * q.slope_y, 1.0);
            let mid_x = (nx0 + nx1) * 0.5;
            let mid_y = (ny_lo + ny_hi) * 0.5;
            let center_ray = vec3<f32>(mid_x * q.slope_x, mid_y * q.slope_y, 1.0);

            let side_left = side_normal(corner_ll, corner_ul, center_ray);
            let side_right = side_normal(corner_lr, corner_ur, center_ray);
            let side_bottom = side_normal(corner_ll, corner_lr, center_ray);
            let side_top = side_normal(corner_ul, corner_ur, center_ray);

            let near_c = max(q.near, GUARD_EPS);
            let far_c = select(near_c + GUARD_EPS, q.far, q.far > near_c + GUARD_EPS);

            if (q.op == OP_TILE_FRUSTUM) {
                out.kind = KIND_FRUSTUM;
                out.vec_r = side_left;
                out.plane1 = side_right;
                out.plane2 = side_bottom;
                out.plane3 = side_top;
                out.near = near_c;
                out.far = far_c;
            } else {
                // Conservative sphere-vs-frustum cull: the sphere survives when
                // it is not entirely outside any side plane or the near / far
                // plane. `vec_a` is the sphere center, `radius` its radius.
                let center = vec_a;
                let radius = q.radius;
                var survives = true;
                if (dot(side_left, center) < -radius) {
                    survives = false;
                }
                if (dot(side_right, center) < -radius) {
                    survives = false;
                }
                if (dot(side_bottom, center) < -radius) {
                    survives = false;
                }
                if (dot(side_top, center) < -radius) {
                    survives = false;
                }
                if (center.z - near_c < -radius) {
                    survives = false;
                }
                if (far_c - center.z < -radius) {
                    survives = false;
                }
                out.kind = KIND_BOOL;
                out.flag = select(0u, 1u, survives);
            }
        }
        default: {
        }
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in [`TILE_LIGHT_CULL_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every `vec3` lane carries a trailing pad word so each stays `16`-byte aligned
/// on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting which golden function this slot evaluates.
    op: u32,
    /// Tile column coordinate.
    tile_x: u32,
    /// Tile row coordinate.
    tile_y: u32,
    /// Square pixel tile size.
    tile_size: u32,
    /// Number of tile columns.
    tile_count_x: u32,
    /// Number of tile rows.
    tile_count_y: u32,
    /// Framebuffer width in pixels (already within the `u16` range host-side).
    width: u32,
    /// Framebuffer height in pixels (already within the `u16` range host-side).
    height: u32,
    /// Horizontal camera half-extent slope.
    slope_x: f32,
    /// Vertical camera half-extent slope.
    slope_y: f32,
    /// Near depth plane distance.
    near: f32,
    /// Far depth plane distance.
    far: f32,
    /// Sphere radius operand.
    radius: f32,
    /// Scalar scale factor operand.
    factor: f32,
    /// First depth-range edge.
    range_a: f32,
    /// Second depth-range edge.
    range_b: f32,
    /// Primary vector slot.
    vec_a: [f32; 3],
    /// Pad lane after `vec_a`.
    pad_a: f32,
    /// Secondary vector slot.
    vec_b: [f32; 3],
    /// Pad lane after `vec_b`.
    pad_b: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Result-kind tag.
    kind: u32,
    /// Discrete flag (`is_active` or sphere-survives).
    flag: u32,
    /// Pixel-rectangle minimum `x`.
    rect_x0: u32,
    /// Pixel-rectangle minimum `y`.
    rect_y0: u32,
    /// Pixel-rectangle maximum `x`.
    rect_x1: u32,
    /// Pixel-rectangle maximum `y`.
    rect_y1: u32,
    /// Padding word.
    pad_u0: u32,
    /// Padding word.
    pad_u1: u32,
    /// Scalar result or depth-range minimum.
    scalar: f32,
    /// Depth-range maximum.
    extra: f32,
    /// Clamped tile-frustum near depth.
    near: f32,
    /// Clamped tile-frustum far depth.
    far: f32,
    /// Padding word.
    pad_f0: f32,
    /// Padding word.
    pad_f1: f32,
    /// Padding word.
    pad_f2: f32,
    /// Padding word.
    pad_f3: f32,
    /// Vector result, or the left side-plane normal.
    vec_r: [f32; 3],
    /// Pad lane after `vec_r`.
    pad_r: f32,
    /// Right side-plane normal.
    plane1: [f32; 3],
    /// Pad lane after `plane1`.
    pad_p1: f32,
    /// Bottom side-plane normal.
    plane2: [f32; 3],
    /// Pad lane after `plane2`.
    pad_p2: f32,
    /// Top side-plane normal.
    plane3: [f32; 3],
    /// Pad lane after `plane3`.
    pad_p3: f32,
}

/// One numeric query against the tile-light-cull twin.
///
/// Each variant mirrors one golden function. `SphereIsActive` and
/// `IntersectsSphere` return a discrete flag standing in for the reference
/// `bool`; `TilePixelRect` returns exact integer pixel bounds; `TileFrustum`
/// returns the four inward unit side-plane normals plus the clamped near / far
/// depth pair.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tile_light_cull`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TileLightCullQuery {
    /// Component-wise sum
    /// ([`Vec3::plus`](prism_render_architecture::particle::tile_light_cull::Vec3::plus)).
    Vec3Plus {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Component-wise difference
    /// ([`Vec3::minus`](prism_render_architecture::particle::tile_light_cull::Vec3::minus)).
    Vec3Minus {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Scalar multiple
    /// ([`Vec3::scale`](prism_render_architecture::particle::tile_light_cull::Vec3::scale)).
    Vec3Scale {
        /// Vector operand.
        a: Vec3,
        /// Scalar factor.
        factor: f32,
    },
    /// Dot product
    /// ([`Vec3::dot`](prism_render_architecture::particle::tile_light_cull::Vec3::dot)).
    Vec3Dot {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Right-handed cross product
    /// ([`Vec3::cross`](prism_render_architecture::particle::tile_light_cull::Vec3::cross)).
    Vec3Cross {
        /// Left operand.
        a: Vec3,
        /// Right operand.
        b: Vec3,
    },
    /// Euclidean length
    /// ([`Vec3::length`](prism_render_architecture::particle::tile_light_cull::Vec3::length)).
    Vec3Length {
        /// Vector operand.
        a: Vec3,
    },
    /// Guarded unit direction
    /// ([`Vec3::normalize_or_zero`](prism_render_architecture::particle::tile_light_cull::Vec3::normalize_or_zero)).
    Vec3NormalizeOrZero {
        /// Vector operand.
        a: Vec3,
    },
    /// Strictly-positive-radius predicate
    /// ([`SphereLight::is_active`](prism_render_architecture::particle::tile_light_cull::SphereLight::is_active)).
    SphereIsActive {
        /// Candidate influence radius.
        radius: f32,
    },
    /// Inward-normal plane constructor
    /// ([`Plane::from_inward_normal`](prism_render_architecture::particle::tile_light_cull::Plane::from_inward_normal)),
    /// returning the normalized inward normal.
    PlaneFromInwardNormal {
        /// Raw (not necessarily unit) inward normal.
        normal: Vec3,
    },
    /// Signed distance of a point to a plane through the origin
    /// ([`Plane::signed_distance`](prism_render_architecture::particle::tile_light_cull::Plane::signed_distance)).
    PlaneSignedDistance {
        /// The plane's unit inward normal.
        normal: Vec3,
        /// Point to measure.
        point: Vec3,
    },
    /// Edge-ordering depth-range constructor
    /// ([`DepthRange::new`](prism_render_architecture::particle::tile_light_cull::DepthRange::new)).
    DepthRangeNew {
        /// First depth edge.
        a: f32,
        /// Second depth edge.
        b: f32,
    },
    /// Clamped per-tile pixel rectangle
    /// ([`TileGrid::tile_pixel_rect`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_pixel_rect)).
    TilePixelRect {
        /// Tile column coordinate.
        tile_x: u32,
        /// Tile row coordinate.
        tile_y: u32,
        /// Square pixel tile size.
        tile_size: u32,
        /// Number of tile columns.
        tile_count_x: u32,
        /// Number of tile rows.
        tile_count_y: u32,
        /// Framebuffer width in pixels.
        width: u32,
        /// Framebuffer height in pixels.
        height: u32,
    },
    /// Tile sub-frustum construction
    /// ([`TileGrid::tile_frustum`](prism_render_architecture::particle::tile_light_cull::TileGrid::tile_frustum)),
    /// returning the four inward unit side-plane normals and clamped near / far.
    TileFrustum {
        /// Tile column coordinate.
        tile_x: u32,
        /// Tile row coordinate.
        tile_y: u32,
        /// Square pixel tile size.
        tile_size: u32,
        /// Number of tile columns.
        tile_count_x: u32,
        /// Number of tile rows.
        tile_count_y: u32,
        /// Framebuffer width in pixels.
        width: u32,
        /// Framebuffer height in pixels.
        height: u32,
        /// Horizontal camera half-extent slope.
        slope_x: f32,
        /// Vertical camera half-extent slope.
        slope_y: f32,
        /// Requested near depth.
        near: f32,
        /// Requested far depth.
        far: f32,
    },
    /// Conservative sphere-vs-frustum cull
    /// ([`TileFrustum::intersects_sphere`](prism_render_architecture::particle::tile_light_cull::TileFrustum::intersects_sphere)):
    /// builds the tile sub-frustum, then tests one bounding sphere against it.
    IntersectsSphere {
        /// Tile column coordinate.
        tile_x: u32,
        /// Tile row coordinate.
        tile_y: u32,
        /// Square pixel tile size.
        tile_size: u32,
        /// Number of tile columns.
        tile_count_x: u32,
        /// Number of tile rows.
        tile_count_y: u32,
        /// Framebuffer width in pixels.
        width: u32,
        /// Framebuffer height in pixels.
        height: u32,
        /// Horizontal camera half-extent slope.
        slope_x: f32,
        /// Vertical camera half-extent slope.
        slope_y: f32,
        /// Requested near depth.
        near: f32,
        /// Requested far depth.
        far: f32,
        /// Sphere center in view space.
        center: Vec3,
        /// Sphere radius.
        radius: f32,
    },
}

/// One resolved answer, mirroring whichever golden function the query selected.
///
/// `Bool` carries the `is_active` or sphere-survives flag; `Range` the ordered
/// depth edges; `PixelRect` the exact integer tile bounds; `Frustum` the four
/// inward unit side-plane normals and the clamped near / far depth pair.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tile_light_cull`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TileLightCullResult {
    /// A single scalar (`dot`, `length`, `signed_distance`).
    Scalar(f32),
    /// A single vector (`plus`, `minus`, `scale`, `cross`, `normalize_or_zero`,
    /// `from_inward_normal`).
    Vector(Vec3),
    /// A discrete boolean (`is_active`, `intersects_sphere`).
    Bool(bool),
    /// An ordered depth range.
    Range {
        /// Nearest view depth.
        min: f32,
        /// Farthest view depth.
        max: f32,
    },
    /// A clamped per-tile pixel rectangle `(x0, y0, x1, y1)`.
    PixelRect {
        /// Minimum pixel `x`.
        x0: u32,
        /// Minimum pixel `y`.
        y0: u32,
        /// Maximum pixel `x`.
        x1: u32,
        /// Maximum pixel `y`.
        y1: u32,
    },
    /// A tile sub-frustum: four inward unit side-plane normals (left, right,
    /// bottom, top) and the clamped near / far depth pair.
    Frustum {
        /// Inward unit side-plane normals (left, right, bottom, top).
        sides: [Vec3; 4],
        /// Clamped near depth.
        near: f32,
        /// Clamped far depth.
        far: f32,
    },
}

/// Converts a [`Vec3`] into its padded `std430` lane.
fn lane(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Reads a padded `std430` lane back into a [`Vec3`].
fn unlane(l: [f32; 3]) -> Vec3 {
    Vec3::new(l[0], l[1], l[2])
}

/// A fully zeroed query slot, filled per variant by [`encode_query`].
fn empty_query() -> GpuQuery {
    GpuQuery {
        op: 0,
        tile_x: 0,
        tile_y: 0,
        tile_size: 0,
        tile_count_x: 0,
        tile_count_y: 0,
        width: 0,
        height: 0,
        slope_x: 0.0,
        slope_y: 0.0,
        near: 0.0,
        far: 0.0,
        radius: 0.0,
        factor: 0.0,
        range_a: 0.0,
        range_b: 0.0,
        vec_a: [0.0; 3],
        pad_a: 0.0,
        vec_b: [0.0; 3],
        pad_b: 0.0,
    }
}

/// Encodes one [`TileLightCullQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(query: &TileLightCullQuery) -> GpuQuery {
    let mut q = empty_query();
    match query {
        TileLightCullQuery::Vec3Plus { a, b } => {
            q.op = OP_VEC3_PLUS;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        TileLightCullQuery::Vec3Minus { a, b } => {
            q.op = OP_VEC3_MINUS;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        TileLightCullQuery::Vec3Scale { a, factor } => {
            q.op = OP_VEC3_SCALE;
            q.vec_a = lane(*a);
            q.factor = *factor;
        }
        TileLightCullQuery::Vec3Dot { a, b } => {
            q.op = OP_VEC3_DOT;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        TileLightCullQuery::Vec3Cross { a, b } => {
            q.op = OP_VEC3_CROSS;
            q.vec_a = lane(*a);
            q.vec_b = lane(*b);
        }
        TileLightCullQuery::Vec3Length { a } => {
            q.op = OP_VEC3_LENGTH;
            q.vec_a = lane(*a);
        }
        TileLightCullQuery::Vec3NormalizeOrZero { a } => {
            q.op = OP_VEC3_NORMALIZE_OR_ZERO;
            q.vec_a = lane(*a);
        }
        TileLightCullQuery::SphereIsActive { radius } => {
            q.op = OP_SPHERE_IS_ACTIVE;
            q.radius = *radius;
        }
        TileLightCullQuery::PlaneFromInwardNormal { normal } => {
            q.op = OP_PLANE_FROM_INWARD_NORMAL;
            q.vec_a = lane(*normal);
        }
        TileLightCullQuery::PlaneSignedDistance { normal, point } => {
            q.op = OP_PLANE_SIGNED_DISTANCE;
            q.vec_a = lane(*normal);
            q.vec_b = lane(*point);
        }
        TileLightCullQuery::DepthRangeNew { a, b } => {
            q.op = OP_DEPTH_RANGE_NEW;
            q.range_a = *a;
            q.range_b = *b;
        }
        TileLightCullQuery::TilePixelRect {
            tile_x,
            tile_y,
            tile_size,
            tile_count_x,
            tile_count_y,
            width,
            height,
        } => {
            q.op = OP_TILE_PIXEL_RECT;
            q.tile_x = *tile_x;
            q.tile_y = *tile_y;
            q.tile_size = *tile_size;
            q.tile_count_x = *tile_count_x;
            q.tile_count_y = *tile_count_y;
            q.width = *width;
            q.height = *height;
        }
        TileLightCullQuery::TileFrustum {
            tile_x,
            tile_y,
            tile_size,
            tile_count_x,
            tile_count_y,
            width,
            height,
            slope_x,
            slope_y,
            near,
            far,
        } => {
            q.op = OP_TILE_FRUSTUM;
            q.tile_x = *tile_x;
            q.tile_y = *tile_y;
            q.tile_size = *tile_size;
            q.tile_count_x = *tile_count_x;
            q.tile_count_y = *tile_count_y;
            q.width = *width;
            q.height = *height;
            q.slope_x = *slope_x;
            q.slope_y = *slope_y;
            q.near = *near;
            q.far = *far;
        }
        TileLightCullQuery::IntersectsSphere {
            tile_x,
            tile_y,
            tile_size,
            tile_count_x,
            tile_count_y,
            width,
            height,
            slope_x,
            slope_y,
            near,
            far,
            center,
            radius,
        } => {
            q.op = OP_INTERSECTS_SPHERE;
            q.tile_x = *tile_x;
            q.tile_y = *tile_y;
            q.tile_size = *tile_size;
            q.tile_count_x = *tile_count_x;
            q.tile_count_y = *tile_count_y;
            q.width = *width;
            q.height = *height;
            q.slope_x = *slope_x;
            q.slope_y = *slope_y;
            q.near = *near;
            q.far = *far;
            q.vec_a = lane(*center);
            q.radius = *radius;
        }
    }
    q
}

/// Decodes one packed [`GpuResult`] into the public [`TileLightCullResult`],
/// using the originating `query` to select the result shape.
fn decode_result(query: &TileLightCullQuery, raw: &GpuResult) -> TileLightCullResult {
    match query {
        TileLightCullQuery::Vec3Dot { .. }
        | TileLightCullQuery::Vec3Length { .. }
        | TileLightCullQuery::PlaneSignedDistance { .. } => TileLightCullResult::Scalar(raw.scalar),
        TileLightCullQuery::Vec3Plus { .. }
        | TileLightCullQuery::Vec3Minus { .. }
        | TileLightCullQuery::Vec3Scale { .. }
        | TileLightCullQuery::Vec3Cross { .. }
        | TileLightCullQuery::Vec3NormalizeOrZero { .. }
        | TileLightCullQuery::PlaneFromInwardNormal { .. } => {
            TileLightCullResult::Vector(unlane(raw.vec_r))
        }
        TileLightCullQuery::SphereIsActive { .. } | TileLightCullQuery::IntersectsSphere { .. } => {
            TileLightCullResult::Bool(raw.flag == CODE_FLAG)
        }
        TileLightCullQuery::DepthRangeNew { .. } => TileLightCullResult::Range {
            min: raw.scalar,
            max: raw.extra,
        },
        TileLightCullQuery::TilePixelRect { .. } => TileLightCullResult::PixelRect {
            x0: raw.rect_x0,
            y0: raw.rect_y0,
            x1: raw.rect_x1,
            y1: raw.rect_y1,
        },
        TileLightCullQuery::TileFrustum { .. } => TileLightCullResult::Frustum {
            sides: [
                unlane(raw.vec_r),
                unlane(raw.plane1),
                unlane(raw.plane2),
                unlane(raw.plane3),
            ],
            near: raw.near,
            far: raw.far,
        },
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compiled, reusable tile-light-cull compute pipeline, twinning the `CPU`
/// golden
/// [`tile_light_cull`](prism_render_architecture::particle::tile_light_cull).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tile_light_cull`。
pub struct GpuTileLightCull {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTileLightCull {
    /// Compiles the tile-light-cull kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTileLightCull {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tile_light_cull"),
            source: ShaderSource::Wgsl(TILE_LIGHT_CULL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tile_light_cull_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tile_light_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tile_light_cull_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTileLightCull {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`TileLightCullResult`]
    /// per input, in order.
    ///
    /// The results match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TileLightCullQuery],
    ) -> Vec<TileLightCullResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tile_light_cull_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tile_light_cull_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tile_light_cull_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tile_light_cull_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tile_light_cull_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_tile_light_cull_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tile_light_cull_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
