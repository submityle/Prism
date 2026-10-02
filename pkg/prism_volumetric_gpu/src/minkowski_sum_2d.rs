//! `wgpu` compute twin of the 2D convex-polygon Minkowski-sum construction
//! ([`minkowski_sum_2d`](prism_render_architecture::particle::minkowski_sum_2d),
//! particle design §8.2, §10).
//!
//! The `CPU` golden
//! [`minkowski_sum`](prism_render_architecture::particle::minkowski_sum_2d::minkowski_sum)
//! *constructs* the swept boundary `A ⊕ B = { a + b : a ∈ A, b ∈ B }` of two
//! convex polygons as a fresh counter-clockwise (`CCW`) vertex ring: a splat
//! inflated by a brush radius, a collider grown by a probe's extent, or a
//! broadphase cell bounding every relative placement of two clusters. This
//! module is the on-device twin: one thread constructs the whole sum of one
//! polygon pair, so a passing real-device parity test is direct evidence the
//! ported pipeline builds the same ring the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! The thread reproduces the reference's private pipeline branch for branch:
//! each input is `normalize`d (duplicate drop, shoelace orientation to `CCW`,
//! collinear strip, rotate so the bottom-most vertex is first, with a fully
//! degenerate input reduced to a two-point segment or single point); its
//! `edges` are taken in cyclic order; the two edge sequences are merged by
//! increasing polar angle (`angle_leq`, compared without `atan` by half-plane
//! then cross-product sign); the chosen edges are accumulated onto a running
//! vertex from the summed bottom-most starts; and the result is cleaned the same
//! way `normalize` cleans an input. The output is the bottom-most-first `CCW`
//! ring, its vertex count, and the degenerate collapses, exactly as the
//! reference returns them.
//!
//! # Fixed-capacity rings
//!
//! The reference grows [`Vec`] rings; the kernel has no heap, so each ring is a
//! fixed-length lane array carrying a live-vertex count. Inputs carry up to
//! `16` vertices each (`MAX_A` / `MAX_B`), the accumulated ring up to
//! `MAX_A + MAX_B = 32` (`MAX_OUT`), and every working ring in the pipeline uses
//! the same `32`-lane capacity. The one convergence loop the reference writes
//! (`strip_collinear`, which repeats until the ring is stable) is bounded by a
//! fixed `MAX_OUT`-iteration `for` with an early `break` on a stable pass, so
//! the kernel provably terminates with no runaway loop.
//!
//! # Correctness model
//!
//! The pipeline is pure cross-product sign algebra and `+ - *` accumulation with
//! no division; the output vertex count and the ordering are discrete
//! classifications, so `CPU` and `GPU` must agree on
//! [`MinkowskiSum2dResult::count`] exactly and on the bottom-most-first `CCW`
//! vertex order exactly. The vertex coordinates thread through adds and
//! subtracts, so they are compared under tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`), tight enough to catch a wrong port
//! yet loose enough to admit legal fused multiply-add contraction. Every
//! classification compares against [`SUM_EPS`] rather than an `f32` `==`, and
//! fixtures stay well clear of those epsilon ties so both devices land on the
//! same side of each sign test.
//!
//! # Degenerate inputs
//!
//! A one-vertex input makes the sum a pure translation of the other shape, a
//! two-vertex (segment) input sweeps the other shape along its length, and a sum
//! that collapses to a line is returned as a two-point segment (or a single
//! point when both inputs are points) — all reduced to extreme endpoints exactly
//! as the reference does. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `+ - *`, comparisons and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `atan`, `exp`, `log`, `pow`, no `smoothstep`, no `round`, no `sqrt`
//! and no division, and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::minkowski_sum_2d`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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

/// Maximum vertex count the fixed-capacity layout carries for each input
/// polygon. A ring with more vertices is clamped to this many on upload.
const MAX_A: usize = 16;
/// Maximum vertex count for the second input polygon; mirrors [`MAX_A`].
const MAX_B: usize = 16;
/// Maximum vertex count of the constructed sum: `MAX_A + MAX_B`, the reference's
/// own bound on the result ring length.
const MAX_OUT: usize = MAX_A + MAX_B;

/// The portable core-`WGSL` Minkowski-sum kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`minkowski_sum`](prism_render_architecture::particle::minkowski_sum_2d::minkowski_sum)
/// private pipeline function for function; see the module documentation for the
/// algorithm.
const MINKOWSKI_SUM_2D_WGSL: &str = r#"
// 2D convex-polygon Minkowski-sum twin: one thread constructs the whole sum of
// one polygon pair. It mirrors the CPU golden
// `particle::minkowski_sum_2d::minkowski_sum` private pipeline function for
// function (normalize -> edges -> polar-angle merge -> accumulate -> finalize),
// uses only the portable core-WGSL subset (min/max/abs and + - * plus unsigned
// index math), needs no atan, no sqrt, no division and no transcendental call
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. The single convergence loop (strip_collinear) is bounded by a fixed
// MAX_OUT-iteration for with an early break, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::minkowski_sum_2d；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a coordinate difference, a cross product, or a signed
// area is treated as zero. Matches the reference `SUM_EPS`; the compare rule
// used instead of an f32 `==`.
const SUM_EPS: f32 = 1.0e-6;

// Fixed capacity of every working ring in the pipeline: MAX_A + MAX_B = 32, the
// reference's bound on the accumulated and result ring length.
const MAX_RING: u32 = 32u;

struct Params {
    // Number of polygon pairs in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First input ring: the first `a_count` lanes are live vertices.
    a_verts: array<vec2<f32>, 16>,
    // Second input ring: the first `b_count` lanes are live vertices.
    b_verts: array<vec2<f32>, 16>,
    a_count: u32,
    b_count: u32,
    pad0: u32,
    pad1: u32,
}

struct OutSlot {
    // Constructed sum ring: the first `output_count` lanes are live vertices,
    // bottom-most first and CCW.
    output_verts: array<vec2<f32>, 32>,
    output_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// A fixed-capacity ring threaded through the pipeline: `n` live vertices in the
// leading lanes of `v`. Replaces the reference's growable Vec.
struct Ring {
    n: u32,
    v: array<vec2<f32>, 32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<OutSlot>;

// 2D cross product `a x b`, i.e. the signed area spanned by the two vectors; its
// sign classifies the turn. Mirrors the reference `Vec2::cross`.
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// Returns true when two points coincide within SUM_EPS on both axes; mirrors the
// reference `points_equal`.
fn points_equal(a: vec2<f32>, b: vec2<f32>) -> bool {
    return abs(a.x - b.x) <= SUM_EPS && abs(a.y - b.y) <= SUM_EPS;
}

// Returns true when `a` precedes `b` in bottom-most order: strictly smaller y,
// or an equal y (within SUM_EPS) and strictly smaller x. Mirrors `is_lower`.
fn is_lower(a: vec2<f32>, b: vec2<f32>) -> bool {
    if (abs(a.y - b.y) <= SUM_EPS) {
        return a.x < b.x - SUM_EPS;
    }
    return a.y < b.y;
}

// Twice the signed area of a vertex ring via the shoelace formula; positive for
// a CCW ring, negative for a CW ring, zero for fewer than three vertices.
// Mirrors `signed_area2`.
fn signed_area2(r: Ring) -> f32 {
    if (r.n < 3u) {
        return 0.0;
    }
    var acc = 0.0;
    for (var i = 0u; i < r.n; i = i + 1u) {
        let j = (i + 1u) % r.n;
        acc = acc + cross2(r.v[i], r.v[j]);
    }
    return acc;
}

// Drops consecutive duplicate vertices (including the wrap-around pair) within
// SUM_EPS, preserving order. Mirrors `dedupe`.
fn dedupe(r: Ring) -> Ring {
    var out: Ring;
    out.n = 0u;
    for (var i = 0u; i < r.n; i = i + 1u) {
        let p = r.v[i];
        if (out.n > 0u && points_equal(out.v[out.n - 1u], p)) {
            // Skip the duplicate.
        } else {
            out.v[out.n] = p;
            out.n = out.n + 1u;
        }
    }
    if (out.n > 1u && points_equal(out.v[0], out.v[out.n - 1u])) {
        out.n = out.n - 1u;
    }
    return out;
}

// Rotates a ring so its bottom-most vertex is first, preserving cyclic order.
// Mirrors `rotate_to_lowest`.
fn rotate_to_lowest(r: Ring) -> Ring {
    var out: Ring;
    out.n = 0u;
    if (r.n == 0u) {
        return out;
    }
    var best = 0u;
    for (var i = 0u; i < r.n; i = i + 1u) {
        if (is_lower(r.v[i], r.v[best])) {
            best = i;
        }
    }
    for (var off = 0u; off < r.n; off = off + 1u) {
        out.v[off] = r.v[(best + off) % r.n];
    }
    out.n = r.n;
    return out;
}

// Removes every vertex collinear with its two neighbours, repeating until the
// ring is stable so collinear chains collapse fully. Mirrors `strip_collinear`;
// the reference's unbounded loop is bounded here by MAX_RING with an early break
// on a stable pass.
fn strip_collinear(r: Ring) -> Ring {
    var current = r;
    for (var ring_pass = 0u; ring_pass < MAX_RING; ring_pass = ring_pass + 1u) {
        let n = current.n;
        if (n < 3u) {
            return current;
        }
        var next: Ring;
        next.n = 0u;
        for (var i = 0u; i < n; i = i + 1u) {
            let prev = current.v[(i + n - 1u) % n];
            let cur = current.v[i];
            let after = current.v[(i + 1u) % n];
            let turn = cross2(cur - prev, after - cur);
            if (abs(turn) > SUM_EPS) {
                next.v[next.n] = cur;
                next.n = next.n + 1u;
            }
        }
        if (next.n == n || next.n < 3u) {
            return next;
        }
        current = next;
    }
    return current;
}

// Reduces a set of collinear (or coincident) points to its extreme endpoints
// along the dominant axis: a single point when they all coincide, a two-point
// segment otherwise, ordered bottom-most first. Mirrors `reduce_to_segment`.
fn reduce_to_segment(r: Ring) -> Ring {
    var out: Ring;
    out.n = 0u;
    if (r.n == 0u) {
        return out;
    }
    var min_x = r.v[0].x;
    var max_x = r.v[0].x;
    var min_y = r.v[0].y;
    var max_y = r.v[0].y;
    for (var i = 0u; i < r.n; i = i + 1u) {
        min_x = min(min_x, r.v[i].x);
        max_x = max(max_x, r.v[i].x);
        min_y = min(min_y, r.v[i].y);
        max_y = max(max_y, r.v[i].y);
    }
    let use_x = (max_x - min_x) >= (max_y - min_y);
    var lo = r.v[0];
    var hi = r.v[0];
    for (var i = 0u; i < r.n; i = i + 1u) {
        let p = r.v[i];
        var key: f32;
        var lo_key: f32;
        var hi_key: f32;
        if (use_x) {
            key = p.x;
            lo_key = lo.x;
            hi_key = hi.x;
        } else {
            key = p.y;
            lo_key = lo.y;
            hi_key = hi.y;
        }
        if (key < lo_key - SUM_EPS) {
            lo = p;
        }
        if (key > hi_key + SUM_EPS) {
            hi = p;
        }
    }
    if (points_equal(lo, hi)) {
        out.v[0] = lo;
        out.n = 1u;
        return out;
    }
    if (is_lower(hi, lo)) {
        out.v[0] = hi;
        out.v[1] = lo;
    } else {
        out.v[0] = lo;
        out.v[1] = hi;
    }
    out.n = 2u;
    return out;
}

// Reverses the vertex order of a ring, used to flip a CW ring to CCW.
fn reverse_ring(r: Ring) -> Ring {
    var out: Ring;
    out.n = r.n;
    for (var i = 0u; i < r.n; i = i + 1u) {
        out.v[i] = r.v[r.n - 1u - i];
    }
    return out;
}

// Normalizes an arbitrary convex ring into the canonical form the merge expects:
// a CCW ring, or a two-point segment / single point for a degenerate input,
// always bottom-most first with duplicate and collinear vertices removed.
// Mirrors `normalize`.
fn normalize_ring(poly: Ring) -> Ring {
    let deduped = dedupe(poly);
    if (deduped.n <= 1u) {
        return deduped;
    }
    let area2 = signed_area2(deduped);
    if (abs(area2) <= SUM_EPS) {
        return reduce_to_segment(deduped);
    }
    var oriented = deduped;
    if (area2 < 0.0) {
        oriented = reverse_ring(deduped);
    }
    let stripped = strip_collinear(oriented);
    if (stripped.n <= 2u) {
        return reduce_to_segment(stripped);
    }
    return rotate_to_lowest(stripped);
}

// The edge vectors of a normalized ring in cyclic order; a single point yields
// no edges, a two-point segment yields its two antiparallel edges. Mirrors
// `edges`.
fn edges_of(r: Ring) -> Ring {
    var out: Ring;
    out.n = 0u;
    if (r.n < 2u) {
        return out;
    }
    out.n = r.n;
    for (var i = 0u; i < r.n; i = i + 1u) {
        let j = (i + 1u) % r.n;
        out.v[i] = r.v[j] - r.v[i];
    }
    return out;
}

// Half-plane classifier for polar-angle ordering: 0 for the upper half
// [0, 180) (including the positive x axis) and 1 for the lower half. Mirrors
// `half_plane`.
fn half_plane(v: vec2<f32>) -> u32 {
    if (abs(v.y) <= SUM_EPS) {
        if (v.x >= 0.0) {
            return 0u;
        }
        return 1u;
    }
    if (v.y > 0.0) {
        return 0u;
    }
    return 1u;
}

// Returns true when edge `a` has a polar angle no greater than edge `b`, without
// atan: first by half-plane, then by cross-product sign. Mirrors `angle_leq`.
fn angle_leq(a: vec2<f32>, b: vec2<f32>) -> bool {
    let ha = half_plane(a);
    let hb = half_plane(b);
    if (ha != hb) {
        return ha < hb;
    }
    return cross2(a, b) >= -SUM_EPS;
}

// Cleans a freshly accumulated vertex ring into the canonical result: a CCW
// polygon (bottom-most first, no collinear vertices) or, when degenerate, a
// two-point segment or single point. Mirrors `finalize`.
fn finalize(verts: Ring) -> Ring {
    let deduped = dedupe(verts);
    if (deduped.n <= 2u) {
        return reduce_to_segment(deduped);
    }
    let area2 = signed_area2(deduped);
    if (abs(area2) <= SUM_EPS) {
        return reduce_to_segment(deduped);
    }
    var oriented = deduped;
    if (area2 < 0.0) {
        oriented = reverse_ring(deduped);
    }
    let stripped = strip_collinear(oriented);
    if (stripped.n <= 2u) {
        return reduce_to_segment(stripped);
    }
    return rotate_to_lowest(stripped);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var ra: Ring;
    ra.n = q.a_count;
    for (var i = 0u; i < q.a_count; i = i + 1u) {
        ra.v[i] = q.a_verts[i];
    }
    var rb: Ring;
    rb.n = q.b_count;
    for (var i = 0u; i < q.b_count; i = i + 1u) {
        rb.v[i] = q.b_verts[i];
    }

    let poly_a = normalize_ring(ra);
    let poly_b = normalize_ring(rb);

    var out: OutSlot;
    out.output_count = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    if (poly_a.n == 0u || poly_b.n == 0u) {
        results[idx] = out;
        return;
    }

    let edges_a = edges_of(poly_a);
    let edges_b = edges_of(poly_b);
    let n = edges_a.n;
    let m = edges_b.n;

    let start = poly_a.v[0] + poly_b.v[0];
    var cursor = start;
    var verts: Ring;
    verts.n = 0u;
    var i = 0u;
    var j = 0u;
    // Merge the two edge sequences by polar angle, accumulating each chosen edge
    // onto the running vertex. Bounded by n + m <= MAX_RING iterations.
    for (var step = 0u; step < MAX_RING; step = step + 1u) {
        if (!(i < n || j < m)) {
            break;
        }
        verts.v[verts.n] = cursor;
        verts.n = verts.n + 1u;
        var take_a: bool;
        if (i >= n) {
            take_a = false;
        } else if (j >= m) {
            take_a = true;
        } else {
            take_a = angle_leq(edges_a.v[i], edges_b.v[j]);
        }
        if (take_a) {
            cursor = cursor + edges_a.v[i];
            i = i + 1u;
        } else {
            cursor = cursor + edges_b.v[j];
            j = j + 1u;
        }
    }
    if (verts.n == 0u) {
        verts.v[0] = start;
        verts.n = 1u;
    }

    let result = finalize(verts);
    out.output_count = result.n;
    for (var k = 0u; k < result.n; k = k + 1u) {
        out.output_verts[k] = result.v[k];
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MINKOWSKI_SUM_2D_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid polygon pairs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one polygon pair, matching the `WGSL` `Query`
/// struct. Each vertex array is `32` floats, i.e. `16` interleaved `vec2<f32>`
/// lanes laid out as `[x0, y0, x1, y1, ...]`, so its `128` bytes map onto the
/// device `array<vec2<f32>, 16>` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First input ring: `MAX_A` interleaved `vec2<f32>` vertex lanes.
    a_verts: [f32; 32],
    /// Second input ring: `MAX_B` interleaved `vec2<f32>` vertex lanes.
    b_verts: [f32; 32],
    /// Live vertex count of the first ring.
    a_count: u32,
    /// Live vertex count of the second ring.
    b_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `OutSlot`
/// struct. The vertex array is `64` floats, i.e. `MAX_OUT` interleaved
/// `vec2<f32>` lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Constructed sum ring: `MAX_OUT` interleaved `vec2<f32>` vertex lanes.
    output_verts: [f32; 64],
    /// Live vertex count of the constructed ring.
    output_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One Minkowski-sum query: two convex polygons, each an open vertex ring in any
/// winding and starting vertex.
///
/// Only the first `MAX_A` / `MAX_B` vertices of each ring are read; a longer ring
/// is clamped on upload. The winding, starting vertex and degeneracy are all
/// handled by the twinned pipeline, so inputs need no pre-normalization.
#[derive(Clone, Debug, PartialEq)]
pub struct MinkowskiSum2dQuery {
    /// First convex polygon as an open ring of `[x, y]` vertices.
    pub a: Vec<[f32; 2]>,
    /// Second convex polygon as an open ring of `[x, y]` vertices.
    pub b: Vec<[f32; 2]>,
}

/// The constructed Minkowski sum of one query, mirroring the ring the reference
/// [`minkowski_sum`](prism_render_architecture::particle::minkowski_sum_2d::minkowski_sum)
/// returns: a bottom-most-first `CCW` polygon, or a two-point segment / single
/// point for a degenerate result.
#[derive(Clone, Debug, PartialEq)]
pub struct MinkowskiSum2dResult {
    /// The sum ring as `[x, y]` vertices, bottom-most first and `CCW`; its length
    /// equals [`MinkowskiSum2dResult::count`].
    pub verts: Vec<[f32; 2]>,
    /// The live vertex count of the ring, compared exactly against the reference.
    pub count: u32,
}

/// Encodes one [`MinkowskiSum2dQuery`] into its fixed-length `std430`
/// [`GpuQuery`] slot, clamping each ring to its capacity.
fn encode_query(query: &MinkowskiSum2dQuery) -> GpuQuery {
    let a_count = query.a.len().min(MAX_A);
    let b_count = query.b.len().min(MAX_B);
    let mut a_verts = [0.0_f32; 32];
    for (i, &p) in query.a.iter().take(MAX_A).enumerate() {
        a_verts[2 * i] = p[0];
        a_verts[2 * i + 1] = p[1];
    }
    let mut b_verts = [0.0_f32; 32];
    for (i, &p) in query.b.iter().take(MAX_B).enumerate() {
        b_verts[2 * i] = p[0];
        b_verts[2 * i + 1] = p[1];
    }
    GpuQuery {
        a_verts,
        b_verts,
        a_count: a_count as u32,
        b_count: b_count as u32,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MinkowskiSum2dResult`],
/// taking only the live `output_count` leading vertices.
fn decode_result(raw: &GpuResult) -> MinkowskiSum2dResult {
    let count = (raw.output_count as usize).min(MAX_OUT);
    let mut verts = Vec::with_capacity(count);
    for slot in raw.output_verts.chunks_exact(2).take(count) {
        verts.push([slot[0], slot[1]]);
    }
    MinkowskiSum2dResult {
        verts,
        count: raw.output_count,
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

/// A compiled, reusable 2D Minkowski-sum compute pipeline, twinning the `CPU`
/// golden
/// [`minkowski_sum`](prism_render_architecture::particle::minkowski_sum_2d::minkowski_sum).
pub struct GpuMinkowskiSum2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMinkowskiSum2d {
    /// Compiles the Minkowski-sum kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMinkowskiSum2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d"),
            source: ShaderSource::Wgsl(MINKOWSKI_SUM_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMinkowskiSum2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Constructs the Minkowski sum of every pair in `queries` and returns one
    /// [`MinkowskiSum2dResult`] per input, in order.
    ///
    /// The vertex count and the bottom-most-first `CCW` ordering equal the
    /// reference exactly for inputs clear of the epsilon ties; the vertex
    /// coordinates match to within the tolerance documented on this module. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MinkowskiSum2dQuery],
    ) -> Vec<MinkowskiSum2dResult> {
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
            label: Some("prism_volumetric_minkowski_sum_2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_bind_group"),
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
            label: Some("prism_volumetric_minkowski_sum_2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_minkowski_sum_2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_minkowski_sum_2d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per polygon pair, flattened to a 1-D dispatch.
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

        raw.iter().map(decode_result).collect()
    }
}
