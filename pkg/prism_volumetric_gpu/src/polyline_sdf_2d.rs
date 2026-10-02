//! `wgpu` compute twin of the 2D polyline / polygon signed-distance golden
//! ([`polyline_sdf_2d`](prism_render_architecture::particle::polyline_sdf_2d),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`polyline_sdf_2d`](prism_render_architecture::particle::polyline_sdf_2d)
//! owns a small, screen-space analytic contract: the clamped point-to-segment
//! distance, the unsigned distance to an *open* polyline
//! ([`polyline_distance`](prism_render_architecture::particle::polyline_sdf_2d::polyline_distance)),
//! the unsigned distance to a *closed* polygon boundary
//! ([`polygon_boundary_distance`](prism_render_architecture::particle::polyline_sdf_2d::polygon_boundary_distance)),
//! the integer winding number
//! ([`winding_number`](prism_render_architecture::particle::polyline_sdf_2d::winding_number)),
//! the non-zero-winding containment test
//! ([`point_in_polygon`](prism_render_architecture::particle::polyline_sdf_2d::point_in_polygon)),
//! and the signed polygon distance
//! ([`polygon_signed_distance`](prism_render_architecture::particle::polyline_sdf_2d::polygon_signed_distance))
//! that pairs the boundary magnitude with the winding sign (negative inside,
//! positive outside).
//!
//! [`GpuPolylineSdf2d`] is the on-device twin: one thread per `(point, polyline)`
//! query reproduces the same closed form branch for branch over a bounded
//! vertex ring, so a passing real-device parity test is direct evidence the
//! ported kernel evaluates the same geometry and classifies the same winding
//! and containment the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every value the reference exposes is reproduced per query into
//! [`GpuPolylineSdf`]: the open-polyline distance, the closed-boundary distance,
//! the integer `winding`, the `inside` flag (`1` when the winding is non-zero,
//! `0` otherwise) and the signed polygon distance. The reference's degenerate
//! branches are mirrored: a zero-length edge collapses its projection parameter
//! to the start endpoint, a ring of fewer than three vertices always winds `0`,
//! and a boundary-grazing sample (unsigned magnitude at or below [`CMP_EPS`]) is
//! snapped to a signed distance of exactly `0.0`.
//!
//! # Fixed capacity
//!
//! Each query carries up to [`MAX_VERTS`] vertices in a fixed `std430` block,
//! with `vertex_count` naming how many are live, so the kernel needs no dynamic
//! allocation and keeps the one-thread-per-element dispatch rule. The host
//! rejects any query whose vertex count exceeds the capacity.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `clamp`, `+ - * /`, the `dot` builtin, one `sqrt` for the genuine Euclidean
//! length and a `bitcast` to synthesise [`f32::INFINITY`] — with no `sin`,
//! `cos`, `atan`, `exp`, `log`, `pow`, `tan` and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt` per edge, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact for the continuous distances:
//! a `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The parity
//! test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`)
//! on the distances while demanding bit-exact agreement on the integer `winding`
//! and the `inside` classification code.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
//! standard point-to-segment distance plus the classic integer winding number
//! and `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::polyline_sdf_2d::{
    point_in_polygon, polygon_boundary_distance, polygon_signed_distance, polyline_distance,
    winding_number, Vec2,
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Fixed per-query vertex capacity. A query may carry up to this many vertices
/// in its `std430` block; `vertex_count` names how many are live. The host
/// rejects any query that exceeds this bound.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
/// no third-party engine source or derived code.
pub const MAX_VERTS: usize = 32;

/// The portable core-`WGSL` polyline / polygon-`SDF` kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`polyline_sdf_2d`](prism_render_architecture::particle::polyline_sdf_2d)
/// branch for branch; see the module documentation for the algorithm.
const POLYLINE_SDF_2D_WGSL: &str = r#"
// Polyline / polygon-SDF twin: one thread per (point, polyline) query
// reproduces the open-polyline distance, the closed-boundary distance, the
// integer winding number, the containment flag and the signed polygon
// distance. It mirrors the CPU golden particle::polyline_sdf_2d branch for
// branch, uses only the portable core-WGSL subset (min/max/abs/clamp and
// + - * / plus the dot builtin, one sqrt and a bitcast for infinity) and takes
// no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::polyline_sdf_2d; no
// third-party engine source or derived code.

// Magnitude below which a squared length or a distance is treated as zero. This
// is the comparison rule used instead of an exact == / != on an f32, matching
// the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

// Fixed per-query vertex capacity, matching the host `MAX_VERTS`.
const MAX_VERTS: u32 = 32u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point whose distances and containment are evaluated.
    point: vec2<f32>,
    // Number of live vertices in `verts`, at most MAX_VERTS.
    vertex_count: u32,
    pad0: u32,
    // Fixed-capacity vertex ring; only the first `vertex_count` are live.
    verts: array<vec2<f32>, 32u>,
}

struct Result {
    // Open-polyline distance, closed-boundary distance, integer winding, the
    // inside flag (0/1), the signed polygon distance and three pad lanes: eight
    // scalars filling two vec4 slots.
    polyline_dist: f32,
    boundary_dist: f32,
    winding: i32,
    inside: u32,
    signed_dist: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Positive infinity synthesised from its bit pattern, so the kernel never emits
// a transcendental and never writes an exact f32 literal for an unbounded best.
fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// One live vertex of query `qi`, read directly from storage so the dynamic
// index touches the backing buffer rather than a large value copy.
fn vert(qi: u32, i: u32) -> vec2<f32> {
    return queries[qi].verts[i];
}

// Unsigned distance from p to the segment a -> b, mirroring the reference
// `segment_distance`. The projection parameter is clamped to [0, 1] so points
// beyond either end measure to the nearer endpoint; a degenerate segment
// (squared length at or below CMP_EPS) collapses the parameter to 0, the
// endpoint a.
fn segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let len2 = dot(ab, ab);
    var t: f32 = 0.0;
    if (len2 <= CMP_EPS) {
        t = 0.0;
    } else {
        t = clamp(dot(p - a, ab) / len2, 0.0, 1.0);
    }
    let d = p - (a + ab * t);
    return sqrt(dot(d, d));
}

// Signed side of the directed edge a -> b that p lies on, i.e. the 2D cross
// product (b - a) x (p - a). Mirrors the reference `is_left`.
fn is_left(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    return ab.x * ap.y - ab.y * ap.x;
}

// Minimum unsigned distance from p to the OPEN polyline of query qi, mirroring
// the reference `polyline_distance`: infinity for no vertices, the point
// distance for one, otherwise the least segment distance over consecutive
// pairs (no closing edge).
fn polyline_distance(qi: u32, p: vec2<f32>, count: u32) -> f32 {
    if (count == 0u) {
        return inf();
    }
    if (count == 1u) {
        let d = p - vert(qi, 0u);
        return sqrt(dot(d, d));
    }
    var best = inf();
    var i = 0u;
    loop {
        if (i + 1u >= count) {
            break;
        }
        best = min(best, segment_distance(p, vert(qi, i), vert(qi, i + 1u)));
        i = i + 1u;
    }
    return best;
}

// Minimum unsigned distance from p to the CLOSED ring of query qi, mirroring
// the reference `polygon_boundary_distance`: infinity for no vertices, the
// point distance for one, otherwise the least segment distance over every edge
// including the wrap-around closing edge.
fn boundary_distance(qi: u32, p: vec2<f32>, count: u32) -> f32 {
    if (count == 0u) {
        return inf();
    }
    if (count == 1u) {
        let d = p - vert(qi, 0u);
        return sqrt(dot(d, d));
    }
    var best = inf();
    var i = 0u;
    loop {
        if (i >= count) {
            break;
        }
        var j = i + 1u;
        if (j == count) {
            j = 0u;
        }
        best = min(best, segment_distance(p, vert(qi, i), vert(qi, j)));
        i = i + 1u;
    }
    return best;
}

// Integer winding number of the closed ring of query qi around p, mirroring the
// reference `winding_number`: a ring of fewer than three vertices winds 0;
// otherwise each upward crossing with p to the left adds 1 and each downward
// crossing with p to the right subtracts 1.
fn winding_number(qi: u32, p: vec2<f32>, count: u32) -> i32 {
    if (count < 3u) {
        return 0;
    }
    var wn = 0;
    var i = 0u;
    loop {
        if (i >= count) {
            break;
        }
        let a = vert(qi, i);
        var jn = i + 1u;
        if (jn == count) {
            jn = 0u;
        }
        let b = vert(qi, jn);
        if (a.y <= p.y) {
            if (b.y > p.y && is_left(a, b, p) > 0.0) {
                wn = wn + 1;
            }
        } else {
            if (b.y <= p.y && is_left(a, b, p) < 0.0) {
                wn = wn - 1;
            }
        }
        i = i + 1u;
    }
    return wn;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = queries[idx].point;
    let count = queries[idx].vertex_count;

    let poly = polyline_distance(idx, p, count);
    let bound = boundary_distance(idx, p, count);
    let wn = winding_number(idx, p, count);
    let inside = select(0u, 1u, wn != 0);

    // Signed polygon distance: boundary magnitude snapped to 0 on the boundary,
    // left unsigned for a degenerate ring, else signed negative when inside.
    var sdist = bound;
    if (bound <= CMP_EPS) {
        sdist = 0.0;
    } else {
        if (count < 3u) {
            sdist = bound;
        } else {
            if (wn != 0) {
                sdist = -bound;
            } else {
                sdist = bound;
            }
        }
    }

    var out: Result;
    out.polyline_dist = poly;
    out.boundary_dist = bound;
    out.winding = wn;
    out.inside = inside;
    out.signed_dist = sdist;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One polyline / polygon-`SDF` query: the point `p` evaluated against an
/// ordered vertex ring of at most [`MAX_VERTS`] vertices — the same inputs the
/// reference
/// [`polyline_sdf_2d`](prism_render_architecture::particle::polyline_sdf_2d)
/// routines consume (open path for the polyline distance, closed ring for the
/// boundary, winding and signed distance).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
/// no third-party engine source or derived code.
#[derive(Clone, Debug, PartialEq)]
pub struct PolylineSdf2dQuery {
    /// The query point whose distances and containment are evaluated.
    pub point: Vec2,
    /// The ordered vertex ring. Its length must not exceed [`MAX_VERTS`].
    pub vertices: Vec<Vec2>,
}

impl PolylineSdf2dQuery {
    /// Builds a query from the evaluation point and its vertex ring.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(point: Vec2, vertices: Vec<Vec2>) -> PolylineSdf2dQuery {
        PolylineSdf2dQuery { point, vertices }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// exposes: the open-polyline distance, the closed-boundary distance, the
/// integer `winding`, the `inside` flag (`1` when the winding is non-zero) and
/// the signed polygon distance.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPolylineSdf {
    /// Unsigned distance to the open polyline, matching
    /// [`polyline_distance`](prism_render_architecture::particle::polyline_sdf_2d::polyline_distance).
    pub polyline_dist: f32,
    /// Unsigned distance to the closed boundary ring, matching
    /// [`polygon_boundary_distance`](prism_render_architecture::particle::polyline_sdf_2d::polygon_boundary_distance).
    pub boundary_dist: f32,
    /// Integer winding number, matching
    /// [`winding_number`](prism_render_architecture::particle::polyline_sdf_2d::winding_number).
    pub winding: i32,
    /// Containment flag (`1` inside, `0` outside), matching
    /// [`point_in_polygon`](prism_render_architecture::particle::polyline_sdf_2d::point_in_polygon).
    pub inside: u32,
    /// Signed polygon distance (negative inside, positive outside, `0` on the
    /// boundary), matching
    /// [`polygon_signed_distance`](prism_render_architecture::particle::polyline_sdf_2d::polygon_signed_distance).
    pub signed_dist: f32,
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`polyline_distance`](prism_render_architecture::particle::polyline_sdf_2d::polyline_distance),
/// [`polygon_boundary_distance`](prism_render_architecture::particle::polyline_sdf_2d::polygon_boundary_distance),
/// [`winding_number`](prism_render_architecture::particle::polyline_sdf_2d::winding_number),
/// [`point_in_polygon`](prism_render_architecture::particle::polyline_sdf_2d::point_in_polygon)
/// and
/// [`polygon_signed_distance`](prism_render_architecture::particle::polyline_sdf_2d::polygon_signed_distance)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &PolylineSdf2dQuery) -> GpuPolylineSdf {
    let p = query.point;
    let v = query.vertices.as_slice();
    GpuPolylineSdf {
        polyline_dist: polyline_distance(p, v),
        boundary_dist: polygon_boundary_distance(p, v),
        winding: winding_number(p, v),
        inside: u32::from(point_in_polygon(p, v)),
        signed_dist: polygon_signed_distance(p, v),
    }
}

/// `repr(C)` `std430` layout of one packed query: a `16`-byte header
/// `(point.xy, vertex_count, pad)` followed by the fixed [`MAX_VERTS`]-element
/// `vec2<f32>` ring (`8` bytes each) exactly as the `WGSL` `Query` struct reads
/// it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point.
    point: [f32; 2],
    /// Number of live vertices in `verts`.
    vertex_count: u32,
    /// Padding word completing the `16`-byte header.
    pad0: u32,
    /// Fixed-capacity vertex ring; only the first `vertex_count` are live.
    verts: [[f32; 2]; MAX_VERTS],
}

impl GpuQuery {
    /// Packs one query into its `std430` image, zero-filling the unused tail of
    /// the vertex ring.
    fn new(query: &PolylineSdf2dQuery) -> GpuQuery {
        let mut verts = [[0.0f32; 2]; MAX_VERTS];
        for (slot, v) in verts.iter_mut().zip(query.vertices.iter()) {
            *slot = [v.x, v.y];
        }
        GpuQuery {
            point: [query.point.x, query.point.y],
            vertex_count: query.vertices.len() as u32,
            pad0: 0,
            verts,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(polyline_dist, boundary_dist, winding, inside)` and
/// `(signed_dist, pad, pad, pad)` — `32` bytes matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Open-polyline distance.
    polyline_dist: f32,
    /// Closed-boundary distance.
    boundary_dist: f32,
    /// Integer winding number.
    winding: i32,
    /// Containment flag (`0`/`1`).
    inside: u32,
    /// Signed polygon distance.
    signed_dist: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
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

/// A compiled, reusable polyline / polygon-`SDF` compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
/// no third-party engine source or derived code.
pub struct GpuPolylineSdf2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPolylineSdf2d {
    /// Compiles the polyline / polygon-`SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPolylineSdf2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d"),
            source: ShaderSource::Wgsl(POLYLINE_SDF_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPolylineSdf2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`GpuPolylineSdf`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers to within the tolerance
    /// documented on this module (bit-exact on the integer `winding` and the
    /// `inside` code). An empty input returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics if any query carries more than [`MAX_VERTS`] vertices, since the
    /// fixed `std430` block cannot hold them.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::polyline_sdf_2d`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[PolylineSdf2dQuery]) -> Vec<GpuPolylineSdf> {
        if queries.is_empty() {
            return Vec::new();
        }
        assert!(
            queries.iter().all(|q| q.vertices.len() <= MAX_VERTS),
            "every query must carry at most MAX_VERTS vertices"
        );
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_output"),
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
            label: Some("prism_volumetric_polyline_sdf_2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_bind_group"),
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
            label: Some("prism_volumetric_polyline_sdf_2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_polyline_sdf_2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_polyline_sdf_2d_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`GpuPolylineSdf`].
fn decode_result(raw: &GpuResult) -> GpuPolylineSdf {
    GpuPolylineSdf {
        polyline_dist: raw.polyline_dist,
        boundary_dist: raw.boundary_dist,
        winding: raw.winding,
        inside: raw.inside,
        signed_dist: raw.signed_dist,
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
