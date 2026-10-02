//! `wgpu` compute twin of the analytic finite-segment vs *oriented bounding
//! box* (`OBB`) slab-clip golden
//! ([`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect),
//! design §8.2, §10, §14).
//!
//! The particle subsystem's collision-probe, trail-clip and analytic-primitive
//! contracts need, per `(segment, OBB)` pair, the clipped parameter span
//! `[t_enter, t_exit]` of the finite segment `p0 -> p1` against the box, the
//! unsigned overlap predicate and the standalone endpoint-containment reading.
//! The `CPU` golden
//! [`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_intersect),
//! [`segment_obb_overlaps`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_overlaps)
//! and
//! [`point_in_obb`](prism_render_architecture::particle::segment_obb_intersect::point_in_obb)
//! own that math; [`GpuSegmentObbIntersect`] is the on-device twin that runs one
//! thread per query and reproduces every lane. A passing real-device parity test
//! is therefore direct evidence the ported kernel folds the same three per-axis
//! slab clips, the same parallel-slab guards and the same running interval
//! intersection the reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The kernel transforms the segment endpoints and direction into the box frame
//! by dot products against the orthonormal `axes` and clips three independent
//! `1`D slabs, mirroring the reference guard for guard: the running span is
//! seeded to the segment's own `[0, 1]` range; an axis whose direction
//! projection magnitude is below [`EPS`](prism_render_architecture::particle::segment_obb_intersect::EPS)
//! is *parallel* and divides nothing — it contributes a miss when the start
//! projects outside that slab and leaves the span untouched otherwise; every
//! other axis forms the reciprocal `1 / f`, orders its `[t_near, t_far]` with
//! `min`/`max` and folds it into `[t_enter, t_exit]`, bailing out the instant
//! `t_enter > t_exit`. The hit is reported when every axis survives. From that
//! span the kernel also derives the two continuous *hit points*
//! `p0 + t * (p1 - p0)` and the standalone
//! [`point_in_obb`](prism_render_architecture::particle::segment_obb_intersect::point_in_obb)
//! readings on both endpoints.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `dot`, `+ - * /` and one guarded reciprocal — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `sqrt` or optional device feature (the box faces are flat, so
//! there is no quadratic and no normalization). It therefore runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of three guarded divisions
//! and a running interval intersection, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the span
//! and hit-point values yet an *exact* match on the discrete hit and
//! containment flags, which are routed through the same `EPS` magnitude band the
//! reference uses so a query placed clear of a face boundary folds the identical
//! boolean verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::segment_obb_intersect`；
//! standard slab-clip finite-segment/`OBB` intersection plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::segment_obb_intersect::{
    point_in_obb, segment_obb_intersect, segment_obb_overlaps, v_add, v_scale, v_sub, SegmentObbHit,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Discrete flag written by the kernel for a lane whose boolean reading is
/// `true`: matches the host `== 1` decode in [`decode_result`]. A direct `f32`
/// equality is forbidden, so the kernel emits an integer flag rather than a
/// sentinel float.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` segment/`OBB` slab-clip kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` golden
/// [`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect)
/// slab clip and its `point_in_obb` containment test guard for guard; see the
/// module documentation for the algorithm.
const SEGMENT_OBB_INTERSECT_WGSL: &str = r#"
// Finite segment vs OBB slab-clip twin: one thread per (segment, box) query
// transforms the endpoints and direction into the box frame by dot products
// against the orthonormal axes, folds the three per-axis slab intervals into the
// clipped span [t_enter, t_exit] seeded to the segment's own [0, 1] range, then
// writes the overlap flag, both endpoint-containment flags and the two
// continuous hit points. It mirrors the CPU golden
// `particle::segment_obb_intersect` guard for guard, uses only the portable
// core-WGSL subset (abs/min/max/dot, + - * / and one guarded reciprocal), and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard slab-clip finite-segment/OBB intersection; no
// third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 112-byte std430 stride matching the host `GpuQuery`: the two
// segment endpoints, the box center, the three orthonormal axes and the
// half-extents, each padded to a vec4 so the storage array needs no manual vec3
// alignment arithmetic.
struct Query {
    p0: vec4<f32>,
    p1: vec4<f32>,
    center: vec4<f32>,
    axis0: vec4<f32>,
    axis1: vec4<f32>,
    axis2: vec4<f32>,
    half: vec4<f32>,
}

// One result. 64-byte std430 stride matching the host `GpuResult`: the overlap
// flag and the two endpoint-containment flags as 0u/1u, three pad words, the
// clipped span endpoints, two more pad words and the two continuous hit points
// as vec4 lanes.
struct Result {
    overlaps: u32,
    p0_inside: u32,
    p1_inside: u32,
    pad0: u32,
    t_enter: f32,
    t_exit: f32,
    pad1: f32,
    pad2: f32,
    enter_point: vec4<f32>,
    exit_point: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Magnitude floor guarding the parallel-slab `1 / f` division, classifying a
// direction projection as parallel and giving a face-exact point an inside
// slack, matching the reference `EPS`. A direct f32 `==`/`!=` is forbidden, so
// the parallel and containment tests compare magnitudes against this floor
// instead of exact zero.
const EPS: f32 = 1.0e-6;

// The resolved clip: whether the three per-axis slab intervals overlap the
// segment's [0, 1] range and the ordered entry/exit parameters when they do.
struct Span {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
}

// Standalone OBB containment test: a point is inside exactly when the magnitude
// of its projection onto each axis is within that axis's half-extent (with an
// EPS slack so a face-exact point counts as inside), mirroring the reference
// `point_in_obb`.
fn point_in_obb(
    point: vec3<f32>,
    center: vec3<f32>,
    axis0: vec3<f32>,
    axis1: vec3<f32>,
    axis2: vec3<f32>,
    half: vec3<f32>,
) -> bool {
    let m = point - center;
    if (abs(dot(m, axis0)) > half.x + EPS) {
        return false;
    }
    if (abs(dot(m, axis1)) > half.y + EPS) {
        return false;
    }
    if (abs(dot(m, axis2)) > half.z + EPS) {
        return false;
    }
    return true;
}

// The shared slab clip: tightens the running span (seeded to [0, 1]) with each
// box axis and returns the ordered span when every axis survives. Mirrors the
// reference `segment_obb_intersect` guard for guard: an axis whose direction
// projection magnitude is below EPS is parallel and divides nothing,
// contributing a miss when the start projects outside that slab and leaving the
// span untouched otherwise; every other axis forms the reciprocal `1 / f`,
// orders [t_near, t_far] and folds it in, bailing the instant t_enter > t_exit.
fn clip_axis(
    e: f32,
    f: f32,
    h: f32,
    t_enter: ptr<function, f32>,
    t_exit: ptr<function, f32>,
) -> bool {
    if (abs(f) < EPS) {
        // Parallel to this slab: a start projected outside it can never enter.
        if (abs(e) > h + EPS) {
            return false;
        }
        return true;
    }
    let inv = 1.0 / f;
    let t1 = (-h - e) * inv;
    let t2 = (h - e) * inv;
    let t_near = min(t1, t2);
    let t_far = max(t1, t2);
    *t_enter = max(*t_enter, t_near);
    *t_exit = min(*t_exit, t_far);
    if (*t_enter > *t_exit) {
        return false;
    }
    return true;
}

fn segment_obb_intersect(
    p0: vec3<f32>,
    p1: vec3<f32>,
    center: vec3<f32>,
    axis0: vec3<f32>,
    axis1: vec3<f32>,
    axis2: vec3<f32>,
    half: vec3<f32>,
) -> Span {
    var out: Span;
    out.hit = false;
    out.t_enter = 0.0;
    out.t_exit = 0.0;

    let dir = p1 - p0;
    let m = p0 - center;
    var t_enter = 0.0;
    var t_exit = 1.0;

    if (!clip_axis(dot(m, axis0), dot(dir, axis0), half.x, &t_enter, &t_exit)) {
        return out;
    }
    if (!clip_axis(dot(m, axis1), dot(dir, axis1), half.y, &t_enter, &t_exit)) {
        return out;
    }
    if (!clip_axis(dot(m, axis2), dot(dir, axis2), half.z, &t_enter, &t_exit)) {
        return out;
    }

    out.hit = true;
    out.t_enter = t_enter;
    out.t_exit = t_exit;
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let p0 = queries[idx].p0.xyz;
    let p1 = queries[idx].p1.xyz;
    let center = queries[idx].center.xyz;
    let axis0 = queries[idx].axis0.xyz;
    let axis1 = queries[idx].axis1.xyz;
    let axis2 = queries[idx].axis2.xyz;
    let half = queries[idx].half.xyz;

    var res: Result;
    res.pad0 = 0u;
    res.pad1 = 0.0;
    res.pad2 = 0.0;

    let span = segment_obb_intersect(p0, p1, center, axis0, axis1, axis2, half);
    let dir = p1 - p0;
    if (span.hit) {
        res.overlaps = 1u;
        res.t_enter = span.t_enter;
        res.t_exit = span.t_exit;
        res.enter_point = vec4<f32>(p0 + dir * span.t_enter, 0.0);
        res.exit_point = vec4<f32>(p0 + dir * span.t_exit, 0.0);
    } else {
        res.overlaps = 0u;
        res.t_enter = 0.0;
        res.t_exit = 0.0;
        res.enter_point = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        res.exit_point = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    if (point_in_obb(p0, center, axis0, axis1, axis2, half)) {
        res.p0_inside = 1u;
    } else {
        res.p0_inside = 0u;
    }
    if (point_in_obb(p1, center, axis0, axis1, axis2, half)) {
        res.p1_inside = 1u;
    } else {
        res.p1_inside = 0u;
    }

    results[idx] = res;
}
"#;

/// One finite-segment vs `OBB` query: the segment endpoints and the box to clip
/// them against.
///
/// Mirrors a single reference
/// [`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_intersect)
/// call. The `OBB` is a `center`, three orthonormal `axes` and non-negative
/// `half` extents. Carrying the box per query lets one dispatch clip many
/// segments against distinct boxes. Derives only [`PartialEq`] (no `Eq`/`Hash`)
/// because it holds `f32` geometry.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::segment_obb_intersect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentObbQuery {
    /// The segment start point `p0` in world space.
    pub p0: [f32; 3],
    /// The segment end point `p1` in world space.
    pub p1: [f32; 3],
    /// The box center in world space.
    pub center: [f32; 3],
    /// The three mutually orthogonal unit axes of the box frame.
    pub axes: [[f32; 3]; 3],
    /// The non-negative half-extent along each axis.
    pub half: [f32; 3],
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `overlaps` is the unsigned overlap predicate
/// ([`segment_obb_overlaps`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_overlaps));
/// `span` carries the clipped `[t_enter, t_exit]` parameter interval reusing the
/// golden [`SegmentObbHit`] type and is `Some` exactly when the segment meets
/// the box; `p0_inside` and `p1_inside` are the standalone
/// [`point_in_obb`](prism_render_architecture::particle::segment_obb_intersect::point_in_obb)
/// readings on the two endpoints; `enter_point` and `exit_point` are the two
/// continuous hit points `p0 + t * (p1 - p0)` and are meaningful only when
/// `span` is `Some` (otherwise both hold a zero vector). Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` parameters.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::segment_obb_intersect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentObbResult {
    /// Whether the finite segment meets the box at all.
    pub overlaps: bool,
    /// Whether the start point `p0` lies inside (or on) the box.
    pub p0_inside: bool,
    /// Whether the end point `p1` lies inside (or on) the box.
    pub p1_inside: bool,
    /// The clipped parameter span, `Some` exactly when `overlaps`.
    pub span: Option<SegmentObbHit>,
    /// The continuous hit point at `t_enter`; meaningful only when `overlaps`.
    pub enter_point: [f32; 3],
    /// The continuous hit point at `t_exit`; meaningful only when `overlaps`.
    pub exit_point: [f32; 3],
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points so callers (and the parity test) can pin the twin lane for lane.
///
/// Returns the overlap flag, the two endpoint-containment flags, the clipped
/// span (as an [`Option`] because the reference returns `None` on a miss) and
/// the two continuous hit points (zero vectors on a miss), each matching
/// [`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_intersect),
/// [`segment_obb_overlaps`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_overlaps)
/// and
/// [`point_in_obb`](prism_render_architecture::particle::segment_obb_intersect::point_in_obb).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::segment_obb_intersect`.
#[must_use]
pub fn cpu_reference(query: &SegmentObbQuery) -> SegmentObbResult {
    let span = segment_obb_intersect(query.p0, query.p1, query.center, query.axes, query.half);
    let overlaps = segment_obb_overlaps(query.p0, query.p1, query.center, query.axes, query.half);
    let p0_inside = point_in_obb(query.p0, query.center, query.axes, query.half);
    let p1_inside = point_in_obb(query.p1, query.center, query.axes, query.half);
    let (enter_point, exit_point) = match span {
        Some(hit) => {
            let dir = v_sub(query.p1, query.p0);
            (
                v_add(query.p0, v_scale(dir, hit.t_enter)),
                v_add(query.p0, v_scale(dir, hit.t_exit)),
            )
        }
        None => ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    };
    SegmentObbResult {
        overlaps,
        p0_inside,
        p1_inside,
        span,
        enter_point,
        exit_point,
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

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`SEGMENT_OBB_INTERSECT_WGSL`]: the query count and three pad
/// words — `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `112`-byte `std430` stride matching `Query` in the
/// shader: the two segment endpoints, the box center, the three orthonormal
/// axes and the half-extents, each padded to a `vec4` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Segment start `p0` in `xyz`; the `w` lane is unused padding.
    p0: [f32; 4],
    /// Segment end `p1` in `xyz`; the `w` lane is unused padding.
    p1: [f32; 4],
    /// Box center in `xyz`; the `w` lane is unused padding.
    center: [f32; 4],
    /// First box axis in `xyz`; the `w` lane is unused padding.
    axis0: [f32; 4],
    /// Second box axis in `xyz`; the `w` lane is unused padding.
    axis1: [f32; 4],
    /// Third box axis in `xyz`; the `w` lane is unused padding.
    axis2: [f32; 4],
    /// Half-extents in `xyz`; the `w` lane is unused padding.
    half: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`SegmentObbQuery`] into the `std430` upload layout.
    fn from_query(query: &SegmentObbQuery) -> GpuQuery {
        let a = query.axes;
        GpuQuery {
            p0: [query.p0[0], query.p0[1], query.p0[2], 0.0],
            p1: [query.p1[0], query.p1[1], query.p1[2], 0.0],
            center: [query.center[0], query.center[1], query.center[2], 0.0],
            axis0: [a[0][0], a[0][1], a[0][2], 0.0],
            axis1: [a[1][0], a[1][1], a[1][2], 0.0],
            axis2: [a[2][0], a[2][1], a[2][2], 0.0],
            half: [query.half[0], query.half[1], query.half[2], 0.0],
        }
    }
}

/// One result as read back. `64`-byte `std430` stride matching `Result` in the
/// shader: the overlap flag and two endpoint-containment flags as `0`/`1`, three
/// pad words, the clipped span endpoints, two more pad words and the two
/// continuous hit points as `vec4` lanes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Overlap flag (`1` = hit).
    overlaps: u32,
    /// Start-point containment flag (`1` = inside).
    p0_inside: u32,
    /// End-point containment flag (`1` = inside).
    p1_inside: u32,
    /// Padding word.
    pad0: u32,
    /// Span entry parameter.
    t_enter: f32,
    /// Span exit parameter.
    t_exit: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
    /// Continuous hit point at `t_enter` in `xyz`; the `w` lane is padding.
    enter_point: [f32; 4],
    /// Continuous hit point at `t_exit` in `xyz`; the `w` lane is padding.
    exit_point: [f32; 4],
}

/// Maps one kernel `Result` lane back to the host [`SegmentObbResult`].
fn decode_result(raw: &GpuResult) -> SegmentObbResult {
    let span = if raw.overlaps == CODE_HIT {
        Some(SegmentObbHit {
            t_enter: raw.t_enter,
            t_exit: raw.t_exit,
        })
    } else {
        None
    };
    SegmentObbResult {
        overlaps: raw.overlaps == CODE_HIT,
        p0_inside: raw.p0_inside == CODE_HIT,
        p1_inside: raw.p1_inside == CODE_HIT,
        span,
        enter_point: [raw.enter_point[0], raw.enter_point[1], raw.enter_point[2]],
        exit_point: [raw.exit_point[0], raw.exit_point[1], raw.exit_point[2]],
    }
}

/// A compiled, reusable finite-segment/`OBB` slab-clip pipeline.
pub struct GpuSegmentObbIntersect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegmentObbIntersect {
    /// Compiles the segment/`OBB` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegmentObbIntersect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect"),
            source: ShaderSource::Wgsl(SEGMENT_OBB_INTERSECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegmentObbIntersect {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`SegmentObbResult`] per
    /// query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`segment_obb_intersect`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_intersect),
    /// [`segment_obb_overlaps`](prism_render_architecture::particle::segment_obb_intersect::segment_obb_overlaps)
    /// and
    /// [`point_in_obb`](prism_render_architecture::particle::segment_obb_intersect::point_in_obb)
    /// evaluated on `q`. An empty `queries` slice yields an empty result —
    /// storage buffers cannot be zero-sized, so it is handled by an early return
    /// before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SegmentObbQuery]) -> Vec<SegmentObbResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_segment_obb_intersect_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_segment_obb_intersect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
