//! `wgpu` compute twin of the 2D segment-intersection geometry contract
//! ([`segment_intersect_2d`](prism_render_architecture::particle::segment_intersect_2d),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`segment_intersect_2d`](prism_render_architecture::particle::segment_intersect_2d)
//! owns the analytic 2D segment/line intersection the collision broadphase, the
//! trail-ribbon clipper and the authoring gizmo all share: the `orientation`
//! triple predicate, the two-segment classifier
//! ([`intersect`](prism_render_architecture::particle::segment_intersect_2d::intersect)),
//! its parametric form
//! ([`intersect_params`](prism_render_architecture::particle::segment_intersect_2d::intersect_params))
//! and the infinite-line crossing
//! ([`line_intersection`](prism_render_architecture::particle::segment_intersect_2d::line_intersection)).
//! [`GpuSegmentIntersect2d`] is the on-device twin: one thread per segment pair
//! reproduces the classifier branch for branch, so a passing real-device parity
//! test is direct evidence the ported kernel classifies the same geometry and
//! the same degenerate cases the reference does, not merely that it compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference reports is reproduced: the
//! [`SegIntersect`](prism_render_architecture::particle::segment_intersect_2d::SegIntersect)
//! discriminant (`None`, `Point`, `Collinear`) as an exact classification code,
//! the derived "do they meet" boolean, the `Point` payload coordinate, the
//! `intersect_params` `(t, u)` parametric pair with its `Some`/`None` flag, and
//! the `line_intersection` crossing point with its `Some`/`None` flag. The
//! reference's regimes are mirrored branch for branch: a proper interior
//! crossing (all four turns non-zero and straddling), a fully `collinear` pair
//! (classified by its 1D interval overlap into a gap, an endpoint touch or a
//! shared sub-segment), the four boundary touches (an endpoint lying on the
//! other segment), and the disjoint fallthrough. Degenerate zero-length
//! segments route through the same guarded-length branches as the reference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `dot`, `select`, `+ - * /` and unsigned index comparison — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, `sqrt` or `smoothstep`, no `u64` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! The classification code and the "meet" boolean are pure sign/epsilon
//! decisions, so they match exactly and the parity test asserts `==` on them.
//! The `f32` coordinates and the `(t, u)` parameters are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place, so the parity test
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on them. Every
//! fixture is conditioned clear of a branch tie, so both devices share each
//! sign decision regardless of that slack.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`segment_intersect_2d`](prism_render_architecture::particle::segment_intersect_2d);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::segment_intersect_2d::SegIntersect;
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

/// Classification code for a disjoint pair, mirroring
/// [`SegIntersect::None`](prism_render_architecture::particle::segment_intersect_2d::SegIntersect::None).
pub const CODE_NONE: u32 = 0;

/// Classification code for a single meeting point, mirroring
/// [`SegIntersect::Point`](prism_render_architecture::particle::segment_intersect_2d::SegIntersect::Point).
pub const CODE_POINT: u32 = 1;

/// Classification code for a `collinear` overlap, mirroring
/// [`SegIntersect::Collinear`](prism_render_architecture::particle::segment_intersect_2d::SegIntersect::Collinear).
pub const CODE_COLLINEAR: u32 = 2;

/// The portable core-`WGSL` segment-intersection kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`segment_intersect_2d`](prism_render_architecture::particle::segment_intersect_2d)
/// branch for branch; see the module documentation for the algorithm.
const SEGMENT_INTERSECT_2D_WGSL: &str = r#"
// 2D segment-intersection twin: one thread per segment pair reproduces the
// orientation-triple classifier, its parametric (t, u) form and the infinite
// line crossing. It mirrors the CPU golden `particle::segment_intersect_2d`
// branch for branch, uses only the portable core-WGSL subset (min/max/abs/dot/
// select and + - * /), needs no sqrt and no transcendental, and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::segment_intersect_2d; no third-party engine source or derived code.

// Magnitude below which a cross product, a coordinate difference or a
// parametric denominator is treated as zero. This is the comparison rule used
// throughout instead of an exact == / != on an f32, matching the reference
// `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

const CODE_NONE: u32 = 0u;
const CODE_POINT: u32 = 1u;
const CODE_COLLINEAR: u32 = 2u;

struct Params {
    // Number of segment-pair queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First segment endpoints p1 -> p2.
    p1: vec2<f32>,
    p2: vec2<f32>,
    // Second segment endpoints p3 -> p4.
    p3: vec2<f32>,
    p4: vec2<f32>,
}

struct Result {
    // SegIntersect discriminant: CODE_NONE / CODE_POINT / CODE_COLLINEAR.
    code: u32,
    // 1 when the segments meet (code != CODE_NONE), else 0.
    meets: u32,
    // 1 when intersect_params returned Some (lines not parallel), else 0.
    has_params: u32,
    // 1 when line_intersection returned Some, else 0 (equals has_params).
    has_line: u32,
    // SegIntersect::Point payload; (0, 0) when code != CODE_POINT.
    point: vec2<f32>,
    // Infinite-line crossing point; (0, 0) when has_line == 0.
    line_point: vec2<f32>,
    // Parametric coordinate along p1 -> p2; 0 when has_params == 0.
    t: f32,
    // Parametric coordinate along p3 -> p4; 0 when has_params == 0.
    u: f32,
}

// The classifier verdict: the discriminant plus the single meeting point.
struct Verdict {
    code: u32,
    point: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The 2D cross product of two free vectors u x v.
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

// Classifies the turn a -> b -> c by the sign of the triangle's signed area:
// 1 for a left turn, -1 for a right turn, 0 when collinear within CMP_EPS.
// Matches the reference `orientation`.
fn orient(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> i32 {
    let v = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (v > CMP_EPS) {
        return 1;
    }
    if (v < -CMP_EPS) {
        return -1;
    }
    return 0;
}

// True when two points coincide within CMP_EPS on both axes.
fn points_eq(a: vec2<f32>, b: vec2<f32>) -> bool {
    return abs(a.x - b.x) <= CMP_EPS && abs(a.y - b.y) <= CMP_EPS;
}

// True when point p lies on segment a -> b, assuming near-collinearity:
// p must fall inside the CMP_EPS-widened bounding box and project between the
// endpoints. A degenerate segment is on-segment only for a coincident p.
// Matches the reference `on_segment`.
fn on_seg(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> bool {
    let min_x = min(a.x, b.x);
    let max_x = max(a.x, b.x);
    let min_y = min(a.y, b.y);
    let max_y = max(a.y, b.y);
    let in_box = p.x >= min_x - CMP_EPS
        && p.x <= max_x + CMP_EPS
        && p.y >= min_y - CMP_EPS
        && p.y <= max_y + CMP_EPS;
    if (!in_box) {
        return false;
    }
    let dir = b - a;
    let len2 = dot(dir, dir);
    if (len2 <= CMP_EPS) {
        return points_eq(a, p);
    }
    let proj = dot(p - a, dir);
    return proj >= -CMP_EPS && proj <= len2 + CMP_EPS;
}

// Classifies the overlap of two collinear segments along their shared line by
// intersecting the two projected 1D intervals. Matches the reference
// `overlap_direction` + `collinear_overlap`.
fn collinear_overlap(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, p4: vec2<f32>) -> Verdict {
    var out: Verdict;
    out.code = CODE_NONE;
    out.point = vec2<f32>(0.0, 0.0);

    let d1 = p2 - p1;
    let d2 = p4 - p3;
    var dir: vec2<f32> = vec2<f32>(0.0, 0.0);
    var have_dir = false;
    if (dot(d1, d1) > CMP_EPS) {
        dir = d1;
        have_dir = true;
    } else if (dot(d2, d2) > CMP_EPS) {
        dir = d2;
        have_dir = true;
    }

    if (!have_dir) {
        // Both segments are single points: they meet only if coincident.
        if (points_eq(p1, p3)) {
            out.code = CODE_POINT;
            out.point = p1;
        }
        return out;
    }

    let s1 = dot(p1, dir);
    let s2 = dot(p2, dir);
    let s3 = dot(p3, dir);
    let s4 = dot(p4, dir);
    let lo = max(min(s1, s2), min(s3, s4));
    let hi = min(max(s1, s2), max(s3, s4));

    if (lo > hi + CMP_EPS) {
        return out;
    }
    if (abs(hi - lo) <= CMP_EPS) {
        // A single shared point: return the endpoint whose projection matches.
        var pts = array<vec2<f32>, 4>(p1, p2, p3, p4);
        for (var i = 0u; i < 4u; i = i + 1u) {
            if (abs(dot(pts[i], dir) - lo) <= CMP_EPS) {
                out.code = CODE_POINT;
                out.point = pts[i];
                return out;
            }
        }
        return out;
    }
    out.code = CODE_COLLINEAR;
    return out;
}

// Classifies the intersection of segment p1 p2 with segment p3 p4. Matches the
// reference `intersect`.
fn classify(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, p4: vec2<f32>) -> Verdict {
    var out: Verdict;
    out.code = CODE_NONE;
    out.point = vec2<f32>(0.0, 0.0);

    let o1 = orient(p1, p2, p3);
    let o2 = orient(p1, p2, p4);
    let o3 = orient(p3, p4, p1);
    let o4 = orient(p3, p4, p2);

    // Proper interior crossing: each segment strictly straddles the other's
    // line, so all four turns are non-zero with opposite signs per segment.
    if (o1 != 0 && o2 != 0 && o3 != 0 && o4 != 0
        && (o1 > 0) != (o2 > 0) && (o3 > 0) != (o4 > 0)) {
        let d1 = p2 - p1;
        let d2 = p4 - p3;
        let denom = cross2(d1, d2);
        if (abs(denom) > CMP_EPS) {
            let diff = p3 - p1;
            let t = cross2(diff, d2) / denom;
            out.code = CODE_POINT;
            out.point = p1 + d1 * t;
            return out;
        }
    }

    // Fully collinear: classify the 1D overlap along the shared line.
    if (o1 == 0 && o2 == 0 && o3 == 0 && o4 == 0) {
        return collinear_overlap(p1, p2, p3, p4);
    }

    // Boundary touches: an endpoint of one segment lies on the other segment.
    if (o1 == 0 && on_seg(p1, p2, p3)) {
        out.code = CODE_POINT;
        out.point = p3;
        return out;
    }
    if (o2 == 0 && on_seg(p1, p2, p4)) {
        out.code = CODE_POINT;
        out.point = p4;
        return out;
    }
    if (o3 == 0 && on_seg(p3, p4, p1)) {
        out.code = CODE_POINT;
        out.point = p1;
        return out;
    }
    if (o4 == 0 && on_seg(p3, p4, p2)) {
        out.code = CODE_POINT;
        out.point = p2;
        return out;
    }

    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let p1 = q.p1;
    let p2 = q.p2;
    let p3 = q.p3;
    let p4 = q.p4;

    // intersect(...) classification plus the derived "meet" boolean.
    let verdict = classify(p1, p2, p3, p4);

    // intersect_params(...) and line_intersection(...): the shared parametric
    // solve, guarded by a near-zero denominator (parallel / degenerate).
    let d1 = p2 - p1;
    let d2 = p4 - p3;
    let denom = cross2(d1, d2);
    var has_params = 0u;
    var t = 0.0;
    var u = 0.0;
    var has_line = 0u;
    var line_point = vec2<f32>(0.0, 0.0);
    if (abs(denom) > CMP_EPS) {
        let diff = p3 - p1;
        t = cross2(diff, d2) / denom;
        u = cross2(diff, d1) / denom;
        has_params = 1u;
        has_line = 1u;
        line_point = p1 + d1 * t;
    }

    var out: Result;
    out.code = verdict.code;
    out.meets = select(0u, 1u, verdict.code != CODE_NONE);
    out.has_params = has_params;
    out.has_line = has_line;
    out.point = verdict.point;
    out.line_point = line_point;
    out.t = t;
    out.u = u;
    results[idx] = out;
}
"#;

/// One segment-pair intersection query: the two 2D segments `p1 -> p2` and
/// `p3 -> p4`, the same inputs the reference `intersect`, `intersect_params`
/// and `line_intersection` consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentIntersectQuery {
    /// First segment start endpoint.
    pub p1: [f32; 2],
    /// First segment end endpoint.
    pub p2: [f32; 2],
    /// Second segment start endpoint.
    pub p3: [f32; 2],
    /// Second segment end endpoint.
    pub p4: [f32; 2],
}

impl SegmentIntersectQuery {
    /// Builds a query from the four segment endpoints.
    #[must_use]
    pub const fn new(
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        p4: [f32; 2],
    ) -> SegmentIntersectQuery {
        SegmentIntersectQuery { p1, p2, p3, p4 }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its three twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentIntersectResult {
    /// The [`SegIntersect`] discriminant as a classification code, one of
    /// [`CODE_NONE`], [`CODE_POINT`] or [`CODE_COLLINEAR`].
    pub code: u32,
    /// Whether the two segments meet (`code` is not [`CODE_NONE`]), matching
    /// `intersect(...) != SegIntersect::None`.
    pub meets: bool,
    /// The single meeting point when `code` is [`CODE_POINT`], matching the
    /// `SegIntersect::Point` payload; `(0, 0)` otherwise.
    pub point: [f32; 2],
    /// The `intersect_params` parametric pair `(t, u)`, or `None` when the
    /// segment directions are parallel or degenerate.
    pub params: Option<(f32, f32)>,
    /// The `line_intersection` infinite-line crossing point, or `None` when the
    /// lines are parallel or coincident.
    pub line_point: Option<[f32; 2]>,
}

impl SegmentIntersectResult {
    /// Rebuilds the reference [`SegIntersect`] discriminant and payload from the
    /// classification code and the meeting point. A code outside the three
    /// documented values is treated as [`SegIntersect::None`].
    #[must_use]
    pub fn classification(&self) -> SegIntersect {
        match self.code {
            CODE_POINT => SegIntersect::Point(self.point),
            CODE_COLLINEAR => SegIntersect::Collinear,
            _ => SegIntersect::None,
        }
    }
}

/// `repr(C)` `std430` image of one packed query: four `vec2<f32>` endpoint slots
/// `(p1, p2, p3, p4)` — `32` bytes, each `vec2` on its `8`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First segment start endpoint.
    p1: [f32; 2],
    /// First segment end endpoint.
    p2: [f32; 2],
    /// Second segment start endpoint.
    p3: [f32; 2],
    /// Second segment end endpoint.
    p4: [f32; 2],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &SegmentIntersectQuery) -> GpuQuery {
        GpuQuery {
            p1: query.p1,
            p2: query.p2,
            p3: query.p3,
            p4: query.p4,
        }
    }
}

/// `repr(C)` `std430` image of one result: four `u32` flag words
/// `(code, meets, has_params, has_line)`, two `vec2<f32>` point slots
/// `(point, line_point)`, then the two parameters `(t, u)` — `40` bytes
/// matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `SegIntersect` discriminant as a classification code.
    code: u32,
    /// `1` when the segments meet, else `0`.
    meets: u32,
    /// `1` when `intersect_params` returned `Some`, else `0`.
    has_params: u32,
    /// `1` when `line_intersection` returned `Some`, else `0`.
    has_line: u32,
    /// `SegIntersect::Point` payload; `(0, 0)` when not a point.
    point: [f32; 2],
    /// Infinite-line crossing point; `(0, 0)` when there is none.
    line_point: [f32; 2],
    /// Parametric coordinate along the first segment.
    t: f32,
    /// Parametric coordinate along the second segment.
    u: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable segment-intersection compute pipeline.
pub struct GpuSegmentIntersect2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegmentIntersect2d {
    /// Compiles the segment-intersection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegmentIntersect2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d"),
            source: ShaderSource::Wgsl(SEGMENT_INTERSECT_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegmentIntersect2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`SegmentIntersectResult`]
    /// per input, in order.
    ///
    /// Each result matches the reference answers (`intersect`,
    /// `intersect_params` and `line_intersection`): the classification code and
    /// the `meets` boolean exactly, the coordinates and parameters to within the
    /// tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[SegmentIntersectQuery],
    ) -> Vec<SegmentIntersectResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_output"),
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
            label: Some("prism_volumetric_segment_intersect_2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_segment_intersect_2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_segment_intersect_2d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per segment pair, flattened to a 1-D dispatch.
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

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SegmentIntersectResult`].
fn decode_result(raw: &GpuResult) -> SegmentIntersectResult {
    let params = if raw.has_params == 0 {
        None
    } else {
        Some((raw.t, raw.u))
    };
    let line_point = if raw.has_line == 0 {
        None
    } else {
        Some(raw.line_point)
    };
    SegmentIntersectResult {
        code: raw.code,
        meets: raw.meets != 0,
        point: raw.point,
        params,
        line_point,
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
