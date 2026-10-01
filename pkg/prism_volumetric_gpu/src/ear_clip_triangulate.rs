#![forbid(unsafe_code)]
//! `wgpu` compute twin of the ear-clipping triangulation geometry predicates
//! ([`ear_clip_triangulate`](prism_render_architecture::particle::ear_clip_triangulate),
//! particle design §14, §24).
//!
//! The `CPU` golden
//! [`ear_clip_triangulate`](prism_render_architecture::particle::ear_clip_triangulate)
//! turns a flat simple polygon into a `CCW`-wound triangle soup by repeatedly
//! locating and clipping convex "ear" vertices. That host routine is a strictly
//! *sequential* editor of a shrinking index ring, so its outer loop is not a
//! data-parallel problem and is **not** twinned here. What *is* twinned is the
//! loop body's pure, per-element geometry algebra — the predicates every ear
//! clipper evaluates thousands of times — so a passing real-device parity test
//! is direct evidence the ported predicates classify the same geometry and the
//! same degenerate cases the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! One thread per query reproduces, branch for branch, the reference's
//! parallelisable predicates:
//!
//! * [`signed_area`](prism_render_architecture::particle::ear_clip_triangulate::signed_area)
//!   — the shoelace signed area of a whole polygon (positive `CCW`, negative
//!   `CW`, zero for fewer than three vertices).
//! * [`is_ccw`](prism_render_architecture::particle::ear_clip_triangulate::is_ccw)
//!   — the winding verdict, `signed_area > CMP_EPS`.
//! * [`point_in_triangle`](prism_render_architecture::particle::ear_clip_triangulate::point_in_triangle)
//!   — the orientation-agnostic containment test (interior or boundary within
//!   `CMP_EPS`).
//! * [`is_convex_vertex`](prism_render_architecture::particle::ear_clip_triangulate::is_convex_vertex)
//!   — the strict-convexity test of a vertex given its two polygon neighbours
//!   and the ring winding.
//! * [`is_ear`](prism_render_architecture::particle::ear_clip_triangulate::is_ear)
//!   — the full ear test for a vertex at a ring position: convex *and* its
//!   neighbour triangle contains no other ring vertex.
//!
//! # What is NOT twinned
//!
//! The host entry point
//! [`triangulate`](prism_render_architecture::particle::ear_clip_triangulate::triangulate)
//! is deliberately left on the `CPU`: it is a serial clip loop that mutates a
//! working ring (`remove` a vertex per iteration), an inherently sequential
//! orchestration with no per-element parallelism to port. Its correctness is
//! covered on the host by the reference's own tests; this crate only accelerates
//! the predicates it calls.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /` and unsigned integer / bit arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no `smoothstep`, no `sqrt` (every
//! predicate is cross-product sign algebra) and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each predicate is a fixed, non-reorderable sequence of multiplies, adds and
//! subtracts, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The boolean verdicts are returned as `u32` (`0` / `1`)
//! and compared exactly, while the continuous [`signed_area`] is compared with a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a
//! wrong port yet loose enough to admit legal fused multiply-add contraction.
//! Fixtures stay well away from the compare-epsilon ties so both devices land on
//! the same side of every sign test.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ear_clip_triangulate`；
//! 无第三方引擎源码或衍生代码。

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

/// Discriminant for a [`signed_area`](prism_render_architecture::particle::ear_clip_triangulate::signed_area) query.
const KIND_SIGNED_AREA: u32 = 0;
/// Discriminant for an [`is_ccw`](prism_render_architecture::particle::ear_clip_triangulate::is_ccw) query.
const KIND_IS_CCW: u32 = 1;
/// Discriminant for a [`point_in_triangle`](prism_render_architecture::particle::ear_clip_triangulate::point_in_triangle) query.
const KIND_POINT_IN_TRIANGLE: u32 = 2;
/// Discriminant for an [`is_convex_vertex`](prism_render_architecture::particle::ear_clip_triangulate::is_convex_vertex) query.
const KIND_IS_CONVEX_VERTEX: u32 = 3;
/// Discriminant for an [`is_ear`](prism_render_architecture::particle::ear_clip_triangulate::is_ear) query.
const KIND_IS_EAR: u32 = 4;

/// `u32` encoding of a `true` boolean verdict read back from the device.
const CODE_TRUE: u32 = 1;

/// The portable core-`WGSL` ear-clip predicate kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` dispatches
/// on a per-query `kind` discriminant to the predicate the reference exposes;
/// see the module documentation for the mapping and the algorithm.
const EAR_CLIP_TRIANGULATE_WGSL: &str = r#"
// Ear-clipping geometry-predicate twin: one thread per query reproduces the
// reference's parallelisable predicates (signed_area, is_ccw, point_in_triangle,
// is_convex_vertex, is_ear). The sequential `triangulate` clip loop is NOT
// ported; it stays on the host. The kernel mirrors the CPU golden
// `particle::ear_clip_triangulate` branch for branch, uses only the portable
// core-WGSL subset (min/max/clamp/abs and + - * / plus unsigned integer math),
// needs no sqrt or transcendental and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::ear_clip_triangulate；无第三方引擎源码或衍生代码。

// Epsilon used for tolerant float comparisons instead of an exact == / != on an
// f32: a cross product whose magnitude is at or below it is treated as zero
// (collinear) and a point within it of an edge counts as on the boundary.
// Matches the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

// Per-query discriminants, mirroring the host-side KIND_* constants.
const KIND_SIGNED_AREA: u32 = 0u;
const KIND_IS_CCW: u32 = 1u;
const KIND_POINT_IN_TRIANGLE: u32 = 2u;
const KIND_IS_CONVEX_VERTEX: u32 = 3u;
const KIND_IS_EAR: u32 = 4u;

struct Params {
    // Number of queries in the storage array.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Which predicate this query evaluates (one of the KIND_* constants).
    kind: u32,
    // Offset / count of this query's polygon inside the shared `verts` buffer;
    // used by signed_area, is_ccw and is_ear.
    poly_offset: u32,
    poly_count: u32,
    // Offset / count of this query's index ring inside the shared `ring` buffer;
    // used by is_ear only.
    ring_offset: u32,
    ring_count: u32,
    // Ring position of the candidate ear vertex; used by is_ear only.
    vertex_i: u32,
    // Ring winding flag (0 = CW, 1 = CCW); used by is_convex_vertex and is_ear.
    ccw: u32,
    pad0: u32,
    // Inline point operands. For point_in_triangle: (p, a, b, c). For
    // is_convex_vertex: (prev, cur, next, unused). Unused by the polygon kinds.
    pt0: vec2<f32>,
    pt1: vec2<f32>,
    pt2: vec2<f32>,
    pt3: vec2<f32>,
}

struct Result {
    // Signed area for KIND_SIGNED_AREA; zero otherwise.
    area: f32,
    // Boolean verdict (0 / 1) for the predicate kinds; zero for signed_area.
    flag: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> verts: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> ring: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<Result>;

// Twice the signed area of triangle (a, b, c): the z component of the edge
// cross product. Positive for a CCW triple, negative for CW, near zero when
// collinear. Mirrors the reference `tri_cross`.
fn tri_cross(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    let ex = b.x - a.x;
    let ey = b.y - a.y;
    let fx = c.x - b.x;
    let fy = c.y - b.y;
    return ex * fy - ey * fx;
}

// The side of directed edge a -> b that point p falls on: positive to the left,
// negative to the right, near zero on the line. Mirrors the reference
// `edge_side`.
fn edge_side(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// The shoelace signed area of the polygon stored at [offset, offset + count) in
// `verts`. Positive CCW, negative CW, zero for fewer than three vertices.
// Mirrors the reference `signed_area`; the (i + 1) % n wrap is open-coded as a
// branch so no integer modulo appears.
fn signed_area(offset: u32, count: u32) -> f32 {
    if (count < 3u) {
        return 0.0;
    }
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let p = verts[offset + i];
        var j: u32 = i + 1u;
        if (j >= count) {
            j = 0u;
        }
        let q = verts[offset + j];
        sum = sum + (p.x * q.y - q.x * p.y);
    }
    return sum * 0.5;
}

// True when point p is inside triangle (a, b, c) or on its boundary within
// CMP_EPS. Orientation-agnostic. Mirrors the reference `point_in_triangle`.
fn point_in_triangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
    let d1 = edge_side(p, a, b);
    let d2 = edge_side(p, b, c);
    let d3 = edge_side(p, c, a);
    let has_neg = d1 < -CMP_EPS || d2 < -CMP_EPS || d3 < -CMP_EPS;
    let has_pos = d1 > CMP_EPS || d2 > CMP_EPS || d3 > CMP_EPS;
    return !(has_neg && has_pos);
}

// True when vertex `cur` with neighbours `prev` / `next` is strictly convex for
// the given winding. Collinear vertices (cross magnitude at or below CMP_EPS)
// report false. Mirrors the reference `is_convex_vertex`.
fn is_convex_vertex(prev: vec2<f32>, cur: vec2<f32>, next: vec2<f32>, ccw: bool) -> bool {
    let cross = tri_cross(prev, cur, next);
    if (ccw) {
        return cross > CMP_EPS;
    }
    return cross < -CMP_EPS;
}

// True when the vertex at ring position `i` is an ear: convex and its neighbour
// triangle contains no other ring vertex. `ring` holds polygon-local indices
// into `verts`. Mirrors the reference `is_ear`; the ring wraps are open-coded as
// branches so no integer modulo appears.
fn is_ear(poly_offset: u32, ring_offset: u32, ring_count: u32, i: u32, ccw: bool) -> bool {
    let m = ring_count;
    if (m < 3u) {
        return false;
    }
    var ip: u32 = m - 1u;
    if (i > 0u) {
        ip = i - 1u;
    }
    var inx: u32 = i + 1u;
    if (inx >= m) {
        inx = 0u;
    }
    let a = verts[poly_offset + ring[ring_offset + ip]];
    let b = verts[poly_offset + ring[ring_offset + i]];
    let c = verts[poly_offset + ring[ring_offset + inx]];
    if (!is_convex_vertex(a, b, c, ccw)) {
        return false;
    }
    for (var k: u32 = 0u; k < m; k = k + 1u) {
        if (k == ip || k == i || k == inx) {
            continue;
        }
        let vi = ring[ring_offset + k];
        if (point_in_triangle(verts[poly_offset + vi], a, b, c)) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let ccw = q.ccw != 0u;

    var area: f32 = 0.0;
    var flag: u32 = 0u;
    switch (q.kind) {
        case KIND_SIGNED_AREA: {
            area = signed_area(q.poly_offset, q.poly_count);
        }
        case KIND_IS_CCW: {
            // is_ccw(poly) == signed_area(poly) > CMP_EPS.
            let s = signed_area(q.poly_offset, q.poly_count);
            if (s > CMP_EPS) {
                flag = 1u;
            }
        }
        case KIND_POINT_IN_TRIANGLE: {
            if (point_in_triangle(q.pt0, q.pt1, q.pt2, q.pt3)) {
                flag = 1u;
            }
        }
        case KIND_IS_CONVEX_VERTEX: {
            if (is_convex_vertex(q.pt0, q.pt1, q.pt2, ccw)) {
                flag = 1u;
            }
        }
        case KIND_IS_EAR: {
            if (is_ear(q.poly_offset, q.ring_offset, q.ring_count, q.vertex_i, ccw)) {
                flag = 1u;
            }
        }
        default: {
        }
    }

    var out: Result;
    out.area = area;
    out.flag = flag;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// One ear-clip predicate query. Each variant twins exactly one reference
/// predicate; the sequential
/// [`triangulate`](prism_render_architecture::particle::ear_clip_triangulate::triangulate)
/// clip loop has no variant because it is not ported.
#[derive(Clone, Copy, Debug)]
pub enum EarClipQuery<'a> {
    /// Twins
    /// [`signed_area`](prism_render_architecture::particle::ear_clip_triangulate::signed_area):
    /// the shoelace signed area of `polygon`.
    SignedArea {
        /// The polygon whose signed area is computed.
        polygon: &'a [[f32; 2]],
    },
    /// Twins
    /// [`is_ccw`](prism_render_architecture::particle::ear_clip_triangulate::is_ccw):
    /// whether `polygon` is wound counter-clockwise.
    IsCcw {
        /// The polygon whose winding is tested.
        polygon: &'a [[f32; 2]],
    },
    /// Twins
    /// [`point_in_triangle`](prism_render_architecture::particle::ear_clip_triangulate::point_in_triangle):
    /// whether `p` lies inside or on triangle `(a, b, c)`.
    PointInTriangle {
        /// The query point.
        p: [f32; 2],
        /// First triangle vertex.
        a: [f32; 2],
        /// Second triangle vertex.
        b: [f32; 2],
        /// Third triangle vertex.
        c: [f32; 2],
    },
    /// Twins
    /// [`is_convex_vertex`](prism_render_architecture::particle::ear_clip_triangulate::is_convex_vertex):
    /// whether `cur` is strictly convex for the given winding.
    IsConvexVertex {
        /// The polygon neighbour before `cur`.
        prev: [f32; 2],
        /// The vertex under test.
        cur: [f32; 2],
        /// The polygon neighbour after `cur`.
        next: [f32; 2],
        /// The ring winding: `true` when counter-clockwise.
        ccw: bool,
    },
    /// Twins
    /// [`is_ear`](prism_render_architecture::particle::ear_clip_triangulate::is_ear):
    /// whether the vertex at ring position `vertex` is an ear.
    IsEar {
        /// The polygon the ring indexes into.
        polygon: &'a [[f32; 2]],
        /// The current index ring (polygon-local indices).
        ring: &'a [usize],
        /// The ring position of the candidate ear vertex.
        vertex: usize,
        /// The ring winding: `true` when counter-clockwise.
        ccw: bool,
    },
}

/// The resolved answer for one [`EarClipQuery`], mirroring the reference
/// predicate's return value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EarClipAnswer {
    /// The shoelace signed area from an [`EarClipQuery::SignedArea`].
    SignedArea(f32),
    /// The winding verdict from an [`EarClipQuery::IsCcw`].
    IsCcw(bool),
    /// The containment verdict from an [`EarClipQuery::PointInTriangle`].
    PointInTriangle(bool),
    /// The convexity verdict from an [`EarClipQuery::IsConvexVertex`].
    IsConvexVertex(bool),
    /// The ear verdict from an [`EarClipQuery::IsEar`].
    IsEar(bool),
}

/// `repr(C)` `std430` image of one packed query: eight `u32` control words
/// followed by four `vec2<f32>` point slots — `64` bytes, laying each `vec2` on
/// its `8`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Predicate discriminant (one of the `KIND_*` constants).
    kind: u32,
    /// Offset of the polygon in the shared vertex buffer.
    poly_offset: u32,
    /// Vertex count of the polygon.
    poly_count: u32,
    /// Offset of the index ring in the shared ring buffer.
    ring_offset: u32,
    /// Entry count of the index ring.
    ring_count: u32,
    /// Ring position of the candidate ear vertex.
    vertex_i: u32,
    /// Ring winding flag (`0` = `CW`, `1` = `CCW`).
    ccw: u32,
    /// Padding word so the `vec2` slots start on a `16`-byte boundary.
    pad0: u32,
    /// First point operand (`p` or `prev`).
    pt0: [f32; 2],
    /// Second point operand (`a` or `cur`).
    pt1: [f32; 2],
    /// Third point operand (`b` or `next`).
    pt2: [f32; 2],
    /// Fourth point operand (`c`; unused by non-triangle kinds).
    pt3: [f32; 2],
}

impl GpuQuery {
    /// Builds an all-zero query image; callers overwrite the fields a given kind
    /// uses, leaving the rest at their harmless zero defaults.
    fn empty() -> GpuQuery {
        GpuQuery {
            kind: 0,
            poly_offset: 0,
            poly_count: 0,
            ring_offset: 0,
            ring_count: 0,
            vertex_i: 0,
            ccw: 0,
            pad0: 0,
            pt0: [0.0, 0.0],
            pt1: [0.0, 0.0],
            pt2: [0.0, 0.0],
            pt3: [0.0, 0.0],
        }
    }
}

/// `repr(C)` `std430` image of one result: the signed area, the boolean verdict
/// as a `u32`, then two pad words — `16` bytes matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed area for a signed-area query; zero otherwise.
    area: f32,
    /// Boolean verdict (`0` / `1`) for the predicate kinds.
    flag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage array.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable ear-clip predicate compute pipeline.
pub struct GpuEarClipTriangulate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEarClipTriangulate {
    /// Compiles the ear-clip predicate kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEarClipTriangulate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate"),
            source: ShaderSource::Wgsl(EAR_CLIP_TRIANGULATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEarClipTriangulate {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one [`EarClipAnswer`] per
    /// input, in order and tagged to match the query kind.
    ///
    /// Each answer equals the reference predicate
    /// ([`signed_area`](prism_render_architecture::particle::ear_clip_triangulate::signed_area),
    /// [`is_ccw`](prism_render_architecture::particle::ear_clip_triangulate::is_ccw),
    /// [`point_in_triangle`](prism_render_architecture::particle::ear_clip_triangulate::point_in_triangle),
    /// [`is_convex_vertex`](prism_render_architecture::particle::ear_clip_triangulate::is_convex_vertex)
    /// or [`is_ear`](prism_render_architecture::particle::ear_clip_triangulate::is_ear))
    /// — exactly for the boolean verdicts, within the documented tolerance for
    /// the signed area. An empty input returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[EarClipQuery<'_>]) -> Vec<EarClipAnswer> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        // Flatten the heterogeneous queries into one contiguous vertex buffer,
        // one contiguous index-ring buffer, and one packed query array that
        // carries each query's offsets into those shared buffers.
        let mut verts: Vec<[f32; 2]> = Vec::new();
        let mut ring: Vec<u32> = Vec::new();
        let mut packed: Vec<GpuQuery> = Vec::with_capacity(count);
        for query in queries {
            packed.push(pack_query(query, &mut verts, &mut ring));
        }
        // A WebGPU storage binding may not be zero-sized; pad the shared buffers
        // up to a single harmless element when no query contributed to them (for
        // example a batch of only point-in-triangle queries).
        if verts.is_empty() {
            verts.push([0.0, 0.0]);
        }
        if ring.is_empty() {
            ring.push(0);
        }

        let query_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_queries"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let vert_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_verts"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::STORAGE,
        });
        let ring_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_ring"),
            contents: bytemuck::cast_slice(&ring),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: query_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: vert_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: ring_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_ear_clip_triangulate_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_ear_clip_triangulate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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
            .map(|(query, result)| decode_answer(query, result))
            .collect()
    }
}

/// Packs one query into its `std430` image, appending its polygon to `verts` and
/// its index ring to `ring` and recording the offsets.
fn pack_query(
    query: &EarClipQuery<'_>,
    verts: &mut Vec<[f32; 2]>,
    ring: &mut Vec<u32>,
) -> GpuQuery {
    let mut gpu = GpuQuery::empty();
    match *query {
        EarClipQuery::SignedArea { polygon } => {
            gpu.kind = KIND_SIGNED_AREA;
            gpu.poly_offset = verts.len() as u32;
            gpu.poly_count = polygon.len() as u32;
            verts.extend_from_slice(polygon);
        }
        EarClipQuery::IsCcw { polygon } => {
            gpu.kind = KIND_IS_CCW;
            gpu.poly_offset = verts.len() as u32;
            gpu.poly_count = polygon.len() as u32;
            verts.extend_from_slice(polygon);
        }
        EarClipQuery::PointInTriangle { p, a, b, c } => {
            gpu.kind = KIND_POINT_IN_TRIANGLE;
            gpu.pt0 = p;
            gpu.pt1 = a;
            gpu.pt2 = b;
            gpu.pt3 = c;
        }
        EarClipQuery::IsConvexVertex {
            prev,
            cur,
            next,
            ccw,
        } => {
            gpu.kind = KIND_IS_CONVEX_VERTEX;
            gpu.pt0 = prev;
            gpu.pt1 = cur;
            gpu.pt2 = next;
            gpu.ccw = u32::from(ccw);
        }
        EarClipQuery::IsEar {
            polygon,
            ring: indices,
            vertex,
            ccw,
        } => {
            gpu.kind = KIND_IS_EAR;
            gpu.poly_offset = verts.len() as u32;
            gpu.poly_count = polygon.len() as u32;
            verts.extend_from_slice(polygon);
            gpu.ring_offset = ring.len() as u32;
            gpu.ring_count = indices.len() as u32;
            ring.extend(indices.iter().map(|&i| i as u32));
            gpu.vertex_i = vertex as u32;
            gpu.ccw = u32::from(ccw);
        }
    }
    gpu
}

/// Decodes one packed [`GpuResult`] into the public [`EarClipAnswer`], tagged to
/// match the originating query kind.
fn decode_answer(query: &EarClipQuery<'_>, raw: &GpuResult) -> EarClipAnswer {
    match query {
        EarClipQuery::SignedArea { .. } => EarClipAnswer::SignedArea(raw.area),
        EarClipQuery::IsCcw { .. } => EarClipAnswer::IsCcw(raw.flag == CODE_TRUE),
        EarClipQuery::PointInTriangle { .. } => {
            EarClipAnswer::PointInTriangle(raw.flag == CODE_TRUE)
        }
        EarClipQuery::IsConvexVertex { .. } => EarClipAnswer::IsConvexVertex(raw.flag == CODE_TRUE),
        EarClipQuery::IsEar { .. } => EarClipAnswer::IsEar(raw.flag == CODE_TRUE),
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
