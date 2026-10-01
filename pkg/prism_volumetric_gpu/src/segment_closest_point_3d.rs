//! `wgpu` compute twin of the 3D segment-closest-point geometry contract
//! ([`segment_closest_point_3d`](prism_render_architecture::particle::segment_closest_point_3d),
//! particle design §10, §14).
//!
//! The `CPU` golden
//! [`segment_closest_point_3d`](prism_render_architecture::particle::segment_closest_point_3d)
//! owns the pure proximity math between two finite 3D line segments. It answers
//! three questions per query, all re-derived from the closed form in Christer
//! Ericson's *Real-Time Collision Detection* §5.1.9: the clamped projection of a
//! point onto one segment ([`closest_point_on_segment`](prism_render_architecture::particle::segment_closest_point_3d::closest_point_on_segment)),
//! the squared distance of that projection
//! ([`point_segment_distance_squared`](prism_render_architecture::particle::segment_closest_point_3d::point_segment_distance_squared)),
//! and the mutually closest points of two segments
//! ([`closest_points_between_segments`](prism_render_architecture::particle::segment_closest_point_3d::closest_points_between_segments)).
//! [`GpuSegmentClosestPoint3d`] is the on-device twin: one thread per query
//! reproduces all three answers, so a passing real-device parity test is direct
//! evidence the ported kernel solves the same geometry and classifies the same
//! degenerate cases the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced: the clamped
//! segment parameter and the closest point it names, the point-to-segment
//! squared distance, and the full [`ClosestPoints`](prism_render_architecture::particle::segment_closest_point_3d::ClosestPoints)
//! record (`s`, `t`, `point_on_first`, `point_on_second`, `distance_squared`).
//! The reference's four regimes are mirrored branch for branch: both segments
//! degenerate (two points), the first degenerate (a point projected onto the
//! second), the second degenerate (a point projected onto the first), and the
//! general / parallel solve where a determinant at or below the compare epsilon
//! pins `s` to the segment start before the `t` recovery and its re-clamp place
//! both parameters inside `[0, 1]²`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /` and unsigned bit arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no `sqrt` (the twinned functions report
//! squared distances, never a length), and no optional device feature, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, tight enough
//! to catch a genuinely wrong port (a dropped branch, a swapped coefficient, a
//! wrong clamp) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`segment_closest_point_3d`](prism_render_architecture::particle::segment_closest_point_3d);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::segment_closest_point_3d::{Segment, Vec3};
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

/// The portable core-`WGSL` segment-closest-point kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`segment_closest_point_3d`](prism_render_architecture::particle::segment_closest_point_3d)
/// branch for branch; see the module documentation for the algorithm.
const SEGMENT_CLOSEST_POINT_3D_WGSL: &str = r#"
// 3D segment-closest-point twin: one thread per query reproduces the clamped
// point-to-segment projection, the point-to-segment squared distance, and the
// mutually closest points of two segments. It mirrors the CPU golden
// `particle::segment_closest_point_3d` branch for branch, uses only the portable
// core-WGSL subset (min/max/clamp/abs and + - * / plus unsigned bit math), needs
// no sqrt (every reported distance is squared) and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::segment_closest_point_3d; no third-party engine source or derived
// code.

// Epsilon used to guard divisions and to classify a segment direction or a
// system determinant as degenerate / parallel, without ever writing an exact
// == / != on an f32. Matches the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First segment start endpoint; a pad lane follows.
    first_a: vec3<f32>,
    pad0: f32,
    // First segment end endpoint; a pad lane follows.
    first_b: vec3<f32>,
    pad1: f32,
    // Second segment start endpoint; a pad lane follows.
    second_a: vec3<f32>,
    pad2: f32,
    // Second segment end endpoint; a pad lane follows.
    second_b: vec3<f32>,
    pad3: f32,
    // Query point for the point-to-first-segment projection; a pad lane follows.
    point: vec3<f32>,
    pad4: f32,
}

struct Result {
    // closest_point_on_segment parameter, point_segment_distance_squared, then
    // the ClosestPoints parameters s and t: four scalars filling one vec4 slot.
    seg_t: f32,
    seg_dist_sq: f32,
    s: f32,
    t: f32,
    // closest_points_between_segments squared distance, with three pad lanes.
    distance_squared: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    // closest_point_on_segment point on the first segment; a pad lane follows.
    seg_point: vec3<f32>,
    pad3: f32,
    // ClosestPoints point_on_first; a pad lane follows.
    point_on_first: vec3<f32>,
    pad4: f32,
    // ClosestPoints point_on_second; a pad lane follows.
    point_on_second: vec3<f32>,
    pad5: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The clamped parameter t in [0, 1] of the point on segment a -> b closest to
// p, mirroring the reference `closest_point_on_segment`. A degenerate segment
// (squared length at or below CMP_EPS) collapses to t = 0, the point a.
fn segment_param(a: vec3<f32>, b: vec3<f32>, p: vec3<f32>) -> f32 {
    let dir = b - a;
    let len_sq = dot(dir, dir);
    if (len_sq <= CMP_EPS) {
        return 0.0;
    }
    let raw = dot(p - a, dir) / len_sq;
    return clamp(raw, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a1 = q.first_a;
    let b1 = q.first_b;
    let a2 = q.second_a;
    let b2 = q.second_b;
    let p = q.point;

    // closest_point_on_segment(first, p) and point_segment_distance_squared.
    let seg_t = segment_param(a1, b1, p);
    let seg_point = a1 + (b1 - a1) * seg_t;
    let seg_delta = p - seg_point;
    let seg_dist_sq = dot(seg_delta, seg_delta);

    // closest_points_between_segments(first, second), Ericson §5.1.9.
    let d1 = b1 - a1;
    let d2 = b2 - a2;
    let r = a1 - a2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);

    let first_degenerate = a <= CMP_EPS;
    let second_degenerate = e <= CMP_EPS;

    var s: f32 = 0.0;
    var t: f32 = 0.0;
    if (first_degenerate && second_degenerate) {
        // Both segments are points: nothing to project.
        s = 0.0;
        t = 0.0;
    } else if (first_degenerate) {
        // First segment is a point; clamp its projection onto segment two.
        s = 0.0;
        t = clamp(f / e, 0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if (second_degenerate) {
            // Second segment is a point; clamp its projection onto segment one.
            t = 0.0;
            s = clamp(-c / a, 0.0, 1.0);
        } else {
            // The fully general non-degenerate case.
            let b = dot(d1, d2);
            let denom = a * e - b * b;

            // Not near zero => lines are not parallel: solve for the line-line
            // optimum along segment one; otherwise pin s to the start.
            var s_line: f32 = 0.0;
            if (denom > CMP_EPS) {
                s_line = clamp((b * f - c * e) / denom, 0.0, 1.0);
            } else {
                s_line = 0.0;
            }

            // Recover t for this s: t = (b*s + f) / e.
            let t_line = (b * s_line + f) / e;

            // If t fell outside [0, 1], clamp it and recompute s for the clamped
            // t via s = (b*t - c) / a, clamped back to [0, 1].
            if (t_line < 0.0) {
                t = 0.0;
                s = clamp(-c / a, 0.0, 1.0);
            } else if (t_line > 1.0) {
                t = 1.0;
                s = clamp((b - c) / a, 0.0, 1.0);
            } else {
                s = s_line;
                t = t_line;
            }
        }
    }

    let point_on_first = a1 + d1 * s;
    let point_on_second = a2 + d2 * t;
    let diff = point_on_first - point_on_second;
    let distance_squared = dot(diff, diff);

    var out: Result;
    out.seg_t = seg_t;
    out.seg_dist_sq = seg_dist_sq;
    out.s = s;
    out.t = t;
    out.distance_squared = distance_squared;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    out.seg_point = seg_point;
    out.pad3 = 0.0;
    out.point_on_first = point_on_first;
    out.pad4 = 0.0;
    out.point_on_second = point_on_second;
    out.pad5 = 0.0;
    results[idx] = out;
}
"#;

/// One segment-closest-point query: two [`Segment`]s plus a query point, the
/// same inputs the reference `closest_point_on_segment` (first segment and
/// point), `point_segment_distance_squared` (first segment and point) and
/// `closest_points_between_segments` (both segments) consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentClosestQuery {
    /// The first segment: the subject of the point projection and the first
    /// argument of the segment-segment solve.
    pub first: Segment,
    /// The second segment: the second argument of the segment-segment solve.
    pub second: Segment,
    /// The query point projected onto `first` by `closest_point_on_segment` and
    /// `point_segment_distance_squared`.
    pub point: Vec3,
}

impl SegmentClosestQuery {
    /// Builds a query from two segments and a projection query point.
    #[must_use]
    pub const fn new(first: Segment, second: Segment, point: Vec3) -> SegmentClosestQuery {
        SegmentClosestQuery {
            first,
            second,
            point,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its three twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentClosestResult {
    /// The clamped parameter in `[0, 1]` of the closest point on `first` to
    /// `point`, matching `closest_point_on_segment`'s first return value.
    pub seg_param: f32,
    /// The closest point on `first` to `point`, matching
    /// `closest_point_on_segment`'s second return value.
    pub seg_point: Vec3,
    /// The squared distance from `point` to `first`, matching
    /// `point_segment_distance_squared`.
    pub seg_distance_squared: f32,
    /// Parameter along `first` of the segment-segment closest point, matching
    /// `ClosestPoints::s`.
    pub s: f32,
    /// Parameter along `second` of the segment-segment closest point, matching
    /// `ClosestPoints::t`.
    pub t: f32,
    /// The closest point on `first`, matching `ClosestPoints::point_on_first`.
    pub point_on_first: Vec3,
    /// The closest point on `second`, matching `ClosestPoints::point_on_second`.
    pub point_on_second: Vec3,
    /// The squared distance between the two segment-segment closest points,
    /// matching `ClosestPoints::distance_squared`.
    pub distance_squared: f32,
}

/// `repr(C)` `std430` layout of one packed query: five `vec4` slots holding
/// `(first.a.xyz, pad)`, `(first.b.xyz, pad)`, `(second.a.xyz, pad)`,
/// `(second.b.xyz, pad)` and `(point.xyz, pad)` — `80` bytes, each `vec3` on its
/// `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First segment start endpoint.
    first_a: [f32; 3],
    /// Padding lane after the first start.
    pad0: f32,
    /// First segment end endpoint.
    first_b: [f32; 3],
    /// Padding lane after the first end.
    pad1: f32,
    /// Second segment start endpoint.
    second_a: [f32; 3],
    /// Padding lane after the second start.
    pad2: f32,
    /// Second segment end endpoint.
    second_b: [f32; 3],
    /// Padding lane after the second end.
    pad3: f32,
    /// Projection query point.
    point: [f32; 3],
    /// Padding lane after the query point.
    pad4: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &SegmentClosestQuery) -> GpuQuery {
        GpuQuery {
            first_a: [query.first.a.x, query.first.a.y, query.first.a.z],
            pad0: 0.0,
            first_b: [query.first.b.x, query.first.b.y, query.first.b.z],
            pad1: 0.0,
            second_a: [query.second.a.x, query.second.a.y, query.second.a.z],
            pad2: 0.0,
            second_b: [query.second.b.x, query.second.b.y, query.second.b.z],
            pad3: 0.0,
            point: [query.point.x, query.point.y, query.point.z],
            pad4: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(seg_t, seg_dist_sq, s, t)`, a `distance_squared` slot with three pad lanes,
/// then three `vec4` slots for the closest point on the first segment, the
/// segment-segment point on first and the segment-segment point on second —
/// `80` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped point-to-segment parameter.
    seg_t: f32,
    /// Point-to-segment squared distance.
    seg_dist_sq: f32,
    /// Segment-segment parameter along the first segment.
    s: f32,
    /// Segment-segment parameter along the second segment.
    t: f32,
    /// Segment-segment squared distance.
    distance_squared: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
    /// Closest point on the first segment to the query point.
    seg_point: [f32; 3],
    /// Padding lane after the closest point.
    pad3: f32,
    /// Segment-segment closest point on the first segment.
    point_on_first: [f32; 3],
    /// Padding lane after the point on first.
    pad4: f32,
    /// Segment-segment closest point on the second segment.
    point_on_second: [f32; 3],
    /// Padding lane after the point on second.
    pad5: f32,
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

/// A compiled, reusable segment-closest-point compute pipeline.
pub struct GpuSegmentClosestPoint3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegmentClosestPoint3d {
    /// Compiles the segment-closest-point kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegmentClosestPoint3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d"),
            source: ShaderSource::Wgsl(SEGMENT_CLOSEST_POINT_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegmentClosestPoint3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`SegmentClosestResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers (`closest_point_on_segment`,
    /// `point_segment_distance_squared` and `closest_points_between_segments`) to
    /// within the tolerance documented on this module. An empty input returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[SegmentClosestQuery],
    ) -> Vec<SegmentClosestResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_output"),
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
            label: Some("prism_volumetric_segment_closest_point_3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_bind_group"),
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
            label: Some("prism_volumetric_segment_closest_point_3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_segment_closest_point_3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_segment_closest_point_3d_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SegmentClosestResult`].
fn decode_result(raw: &GpuResult) -> SegmentClosestResult {
    SegmentClosestResult {
        seg_param: raw.seg_t,
        seg_point: Vec3::new(raw.seg_point[0], raw.seg_point[1], raw.seg_point[2]),
        seg_distance_squared: raw.seg_dist_sq,
        s: raw.s,
        t: raw.t,
        point_on_first: Vec3::new(
            raw.point_on_first[0],
            raw.point_on_first[1],
            raw.point_on_first[2],
        ),
        point_on_second: Vec3::new(
            raw.point_on_second[0],
            raw.point_on_second[1],
            raw.point_on_second[2],
        ),
        distance_squared: raw.distance_squared,
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
