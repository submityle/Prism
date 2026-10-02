//! `wgpu` compute twin of the 2D Marching Squares *per-cell* iso-contour
//! primitive
//! ([`marching_squares`](prism_render_architecture::particle::marching_squares),
//! particle design §8.2 authoring / field-visualization reference).
//!
//! The `CPU` golden
//! [`marching_squares`](prism_render_architecture::particle::marching_squares)
//! walks a `width × height` scalar grid and emits a variable-length list of
//! contour segments. That whole-grid assembly
//! ([`extract_contours`](prism_render_architecture::particle::marching_squares::extract_contours))
//! produces a growable `Vec` and is intentionally *not* twinned here: a
//! variable-length output does not fit a fixed one-thread-per-element parity
//! contract. Instead [`GpuMarchingSquares`] twins the fixed-size, per-cell
//! geometric kernel the whole-grid loop is built from — the exact
//! classification and interpolation each cell performs independently:
//! [`case_index`](prism_render_architecture::particle::marching_squares::case_index)
//! turning four corner values into a `4`-bit case, the saddle-aware edge table,
//! the guarded edge crossing parameter
//! ([`lerp_param`](prism_render_architecture::particle::marching_squares::lerp_param)),
//! and the up-to-two line segments a single cell contributes in grid
//! coordinates. One thread solves one cell, so a passing real-device parity test
//! is direct evidence the ported kernel classifies the same case, resolves the
//! same saddle topology, and places the same crossing points the reference does.
//!
//! # What is twinned
//!
//! For a batch of independent cell queries the kernel reproduces, per cell: the
//! `case_index` classification code (`0..=15`), the segment count (`0`, `1`, or
//! the two-segment saddle), the two line segments' endpoint coordinates built
//! from the reference edge table and `lerp_param`, and each produced segment's
//! Euclidean length (the module's only `sqrt`, mirroring
//! [`Segment::length`](prism_render_architecture::particle::marching_squares::Segment::length)
//! over
//! [`Point2::distance`](prism_render_architecture::particle::marching_squares::Point2::distance)).
//! The empty cases (`0` and `15`) emit a zero segment count. The whole-grid
//! `extract_contours` concatenation is deliberately excluded, since its growable
//! `Vec` output has no fixed per-thread shape.
//!
//! # Correctness model
//!
//! The case code and the segment count are discrete classifications built from
//! `f32` magnitude comparisons (`>= iso`), so for inputs clear of the `== iso`
//! classification boundary the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==`. The crossing coordinates and the segment lengths
//! thread through subtracts, one guarded division, adds and a `sqrt`, so `CPU`
//! and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous quantity.
//!
//! # Degenerate inputs
//!
//! When the two corner values along a crossed edge differ by less than
//! [`CMP_EPS`] the linear crossing parameter is ill-conditioned (a near-`0/0`),
//! so the kernel places the crossing at the edge midpoint `0.5` instead of
//! dividing by a vanishing denominator, matching the reference `lerp_param`. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `min`, `max`, `sqrt`, `+ - * /`, bitwise `or` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry and
//! no optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::marching_squares`；无第三方引擎源码或衍生代码。
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

/// Magnitude below which a crossed edge's corner-value difference is treated as
/// zero, mirroring the reference `EPS`. When the two corner values differ by
/// less than this the crossing is placed at the edge midpoint instead of
/// dividing by a vanishing denominator; the compare rule used instead of an
/// `f32` `==`.
pub const CMP_EPS: f32 = 1.0e-6;

/// The portable core-`WGSL` Marching Squares per-cell kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`marching_squares`](prism_render_architecture::particle::marching_squares)
/// per-cell path branch for branch; see the module documentation for the
/// algorithm.
const MARCHING_SQUARES_WGSL: &str = r#"
// Marching Squares per-cell twin: one thread per cell reproduces the case
// classification, the saddle-aware edge table, the guarded edge crossings and
// the up-to-two line segments (with lengths) a single cell contributes. It
// mirrors the CPU golden `particle::marching_squares` per-cell path branch for
// branch, uses only the portable core-WGSL subset (abs/clamp/min/max/sqrt and
// + - * / plus bitwise-or and unsigned index math) and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::marching_squares；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a crossed edge's corner-value difference is treated as
// zero. Matches the reference `EPS`; the compare rule used instead of an f32
// `==`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of cell queries in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Four corner scalar values in [bottom-left, bottom-right, top-right,
    // top-left] order (v0, v1, v2, v3).
    corners: vec4<f32>,
    // Bottom-left grid/world coordinate of the cell (cx, cy).
    base: vec2<f32>,
    // Iso threshold.
    iso: f32,
    pad: f32,
}

struct Result {
    // First segment endpoints.
    seg0_a: vec2<f32>,
    seg0_b: vec2<f32>,
    // Second segment endpoints (saddle cases only; zero otherwise).
    seg1_a: vec2<f32>,
    seg1_b: vec2<f32>,
    // Length of the first and second segment (zero when absent).
    len0: f32,
    len1: f32,
    // case_index classification code in 0..=15.
    code: u32,
    // Number of segments produced (0, 1, or 2).
    seg_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Classifies four corner values against iso into a 4-bit case; mirrors the
// reference `case_index`. Bit 0 is bottom-left, bit 1 bottom-right, bit 2
// top-right, bit 3 top-left; a bit is set when that corner is inside (>= iso).
fn case_index(v: vec4<f32>, iso: f32) -> u32 {
    var c: u32 = 0u;
    if (v.x >= iso) {
        c = c | 1u;
    }
    if (v.y >= iso) {
        c = c | 2u;
    }
    if (v.z >= iso) {
        c = c | 4u;
    }
    if (v.w >= iso) {
        c = c | 8u;
    }
    return c;
}

// Linear crossing parameter along an edge from value `va` to `vb`; mirrors the
// reference `lerp_param`. When the values differ by less than CMP_EPS the
// denominator is ill-conditioned, so the crossing is placed at the midpoint 0.5.
fn lerp_param(va: f32, vb: f32, iso: f32) -> f32 {
    let denom = vb - va;
    if (abs(denom) < CMP_EPS) {
        return 0.5;
    }
    return clamp((iso - va) / denom, 0.0, 1.0);
}

// Interpolated crossing point on cell edge `edge` (0=bottom, 1=right, 2=top,
// 3=left) for the cell whose bottom-left corner is `base`; mirrors the
// reference `edge_point`. `v` holds the four corner values in
// [bottom-left, bottom-right, top-right, top-left] order. The fall-through arm
// is the left edge (edge == 3); callers only ever pass 0..=3.
fn edge_point(edge: u32, base: vec2<f32>, v: vec4<f32>, iso: f32) -> vec2<f32> {
    if (edge == 0u) {
        return vec2<f32>(base.x + lerp_param(v.x, v.y, iso), base.y);
    }
    if (edge == 1u) {
        return vec2<f32>(base.x + 1.0, base.y + lerp_param(v.y, v.z, iso));
    }
    if (edge == 2u) {
        return vec2<f32>(base.x + 1.0 - lerp_param(v.z, v.w, iso), base.y + 1.0);
    }
    return vec2<f32>(base.x, base.y + 1.0 - lerp_param(v.w, v.x, iso));
}

// Marching Squares edge-connection table; mirrors the reference
// `segment_edges`. Returns up to two edge pairs flattened as (e0, e1, e2, e3);
// each pair [e0, e1] becomes one segment, and a -1 marks "no segment". The two
// diagonal saddle cases (5 and 10) depend on whether the cell center is inside.
fn segment_edges(code: u32, center_inside: bool) -> array<i32, 4> {
    var r = array<i32, 4>(-1, -1, -1, -1);
    switch (code) {
        case 1u, 14u: {
            r = array<i32, 4>(3, 0, -1, -1);
        }
        case 2u, 13u: {
            r = array<i32, 4>(0, 1, -1, -1);
        }
        case 4u, 11u: {
            r = array<i32, 4>(1, 2, -1, -1);
        }
        case 7u, 8u: {
            r = array<i32, 4>(2, 3, -1, -1);
        }
        case 3u, 12u: {
            r = array<i32, 4>(3, 1, -1, -1);
        }
        case 6u, 9u: {
            r = array<i32, 4>(0, 2, -1, -1);
        }
        case 5u: {
            if (center_inside) {
                r = array<i32, 4>(0, 1, 2, 3);
            } else {
                r = array<i32, 4>(3, 0, 1, 2);
            }
        }
        case 10u: {
            if (center_inside) {
                r = array<i32, 4>(3, 0, 1, 2);
            } else {
                r = array<i32, 4>(0, 1, 2, 3);
            }
        }
        default: {
        }
    }
    return r;
}

// Euclidean length of a segment; mirrors the reference `Segment::length` over
// `Point2::distance`, the module's only sqrt.
fn seg_len(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = b - a;
    return sqrt(d.x * d.x + d.y * d.y);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let v = q.corners;
    let iso = q.iso;

    let code = case_index(v, iso);
    // Four-corner center average disambiguates the diagonal saddle cases.
    let center = (v.x + v.y + v.z + v.w) * 0.25;
    let center_inside = center >= iso;
    let edges = segment_edges(code, center_inside);

    var out: Result;
    out.seg0_a = vec2<f32>(0.0, 0.0);
    out.seg0_b = vec2<f32>(0.0, 0.0);
    out.seg1_a = vec2<f32>(0.0, 0.0);
    out.seg1_b = vec2<f32>(0.0, 0.0);
    out.len0 = 0.0;
    out.len1 = 0.0;
    out.code = code;

    var count: u32 = 0u;
    if (edges[0] >= 0 && edges[1] >= 0) {
        let a = edge_point(u32(edges[0]), q.base, v, iso);
        let b = edge_point(u32(edges[1]), q.base, v, iso);
        out.seg0_a = a;
        out.seg0_b = b;
        out.len0 = seg_len(a, b);
        count = count + 1u;
    }
    if (edges[2] >= 0 && edges[3] >= 0) {
        let a = edge_point(u32(edges[2]), q.base, v, iso);
        let b = edge_point(u32(edges[3]), q.base, v, iso);
        out.seg1_a = a;
        out.seg1_b = b;
        out.len1 = seg_len(a, b);
        count = count + 1u;
    }
    out.seg_count = count;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MARCHING_SQUARES_WGSL`].
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

/// `repr(C)` `std430` layout of one cell query, matching the `WGSL` `Query`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Four corner values `(v0, v1, v2, v3)`.
    corners: [f32; 4],
    /// Bottom-left cell coordinate `(cx, cy)`.
    base: [f32; 2],
    /// Iso threshold.
    iso: f32,
    /// Pad lane.
    pad: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First segment endpoint `a`.
    seg0_a: [f32; 2],
    /// First segment endpoint `b`.
    seg0_b: [f32; 2],
    /// Second segment endpoint `a` (zero when absent).
    seg1_a: [f32; 2],
    /// Second segment endpoint `b` (zero when absent).
    seg1_b: [f32; 2],
    /// Length of the first segment (zero when absent).
    len0: f32,
    /// Length of the second segment (zero when absent).
    len1: f32,
    /// `case_index` classification code in `0..=15`.
    code: u32,
    /// Number of segments produced (`0`, `1`, or `2`).
    seg_count: u32,
}

/// One per-cell query for the Marching Squares twin: the four corner scalar
/// values, the cell's bottom-left coordinate, and the iso threshold.
///
/// Corner values are in the `[bottom-left, bottom-right, top-right, top-left]`
/// order used by
/// [`case_index`](prism_render_architecture::particle::marching_squares::case_index),
/// i.e. `(v0, v1, v2, v3)`. `base` is the cell's bottom-left grid coordinate; a
/// single cell spans the unit square from there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchingSquaresQuery {
    /// Corner values `(v0, v1, v2, v3)` in `case_index` order.
    pub corners: [f32; 4],
    /// Bottom-left cell coordinate `(cx, cy)`.
    pub base: [f32; 2],
    /// Iso threshold.
    pub iso: f32,
}

/// One resolved answer for a single cell, mirroring the reference per-cell
/// classification and segments.
///
/// Only the leading `seg_count` entries of `segments` and `lengths` are
/// meaningful; the remainder are zero. Each segment is `[a, b]` with each
/// endpoint `[x, y]` in grid coordinates, as the reference `edge_point` places
/// them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarchingSquaresResult {
    /// `case_index` classification code in `0..=15`, matching
    /// [`case_index`](prism_render_architecture::particle::marching_squares::case_index).
    pub case_index: u32,
    /// Number of segments the cell contributes (`0`, `1`, or `2`).
    pub seg_count: u32,
    /// Up to two segments, each `[a, b]` with each endpoint `[x, y]`.
    pub segments: [[[f32; 2]; 2]; 2],
    /// Per-segment Euclidean length, matching
    /// [`Segment::length`](prism_render_architecture::particle::marching_squares::Segment::length).
    pub lengths: [f32; 2],
}

/// Encodes one [`MarchingSquaresQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MarchingSquaresQuery) -> GpuQuery {
    GpuQuery {
        corners: q.corners,
        base: q.base,
        iso: q.iso,
        pad: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MarchingSquaresResult`].
fn decode_result(raw: &GpuResult) -> MarchingSquaresResult {
    MarchingSquaresResult {
        case_index: raw.code,
        seg_count: raw.seg_count,
        segments: [[raw.seg0_a, raw.seg0_b], [raw.seg1_a, raw.seg1_b]],
        lengths: [raw.len0, raw.len1],
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

/// A compiled, reusable Marching Squares per-cell compute pipeline, twinning the
/// `CPU` golden
/// [`marching_squares`](prism_render_architecture::particle::marching_squares)
/// per-cell path.
pub struct GpuMarchingSquares {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMarchingSquares {
    /// Compiles the Marching Squares per-cell kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMarchingSquares {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_marching_squares"),
            source: ShaderSource::Wgsl(MARCHING_SQUARES_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_marching_squares_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_marching_squares_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_marching_squares_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMarchingSquares {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every cell in `queries` and returns one [`MarchingSquaresResult`]
    /// per input, in order.
    ///
    /// The case code and segment count equal the reference exactly for inputs
    /// clear of the `== iso` classification boundary; the crossing coordinates
    /// and segment lengths match to within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MarchingSquaresQuery],
    ) -> Vec<MarchingSquaresResult> {
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
            label: Some("prism_volumetric_marching_squares_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_marching_squares_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_marching_squares_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_marching_squares_bind_group"),
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
            label: Some("prism_volumetric_marching_squares_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_marching_squares_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_marching_squares_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell, flattened to a 1-D dispatch.
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
