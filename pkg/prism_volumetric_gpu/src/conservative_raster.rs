//! `wgpu` compute twin of the conservative triangle-rasterization edge /
//! coverage contract
//! ([`conservative_raster`](prism_render_architecture::particle::conservative_raster),
//! particle design §8.2, §16).
//!
//! The `CPU` golden
//! [`conservative_raster`](prism_render_architecture::particle::conservative_raster)
//! owns the small, verifiable screen-space geometry the decal-splat,
//! trail-ribbon and tile-assignment paths share: twice the signed triangle area
//! ([`triangle_area2`](prism_render_architecture::particle::conservative_raster::triangle_area2)),
//! the zero-area degeneracy test
//! ([`is_degenerate`](prism_render_architecture::particle::conservative_raster::is_degenerate)),
//! the three directed edge lines of a triangle normalized to a counter-clockwise
//! (`CCW`) interior
//! ([`edges_from_triangle`](prism_render_architecture::particle::conservative_raster::edges_from_triangle)),
//! the half-pixel outward dilation that turns a center test into a
//! touch-anything test
//! ([`dilate_edges`](prism_render_architecture::particle::conservative_raster::dilate_edges)),
//! and the per-pixel coverage predicate
//! ([`pixel_covered`](prism_render_architecture::particle::conservative_raster::pixel_covered))
//! built on the edge evaluation
//! ([`Edge::eval`](prism_render_architecture::particle::conservative_raster::Edge::eval)).
//! [`GpuConservativeRaster`] is the on-device twin: one thread resolves one
//! pixel query against a shared triangle, so a passing real-device parity test
//! is direct evidence the ported kernel classifies the same coverage and
//! computes the same edge values the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The host packs one triangle (three `[f32; 2]` corners) and one `half_pixel`
//! dilation radius, plus a batch of independent integer pixel queries `(x, y)`.
//! For each query the kernel reproduces the full reference pipeline: normalize
//! the triangle winding to `CCW`, build the three directed edges, dilate each
//! edge outward by `half_pixel * (|a| + |b|)`, evaluate the three dilated edges
//! at the pixel center `(x + 0.5, y + 0.5)`, and classify the pixel as inside
//! exactly when every dilated edge value is at least `-CMP_EPS`. A degenerate
//! (near-zero-area) triangle covers no pixel, matching the reference
//! short-circuit in
//! [`rasterize_coverage`](prism_render_architecture::particle::conservative_raster::rasterize_coverage):
//! the kernel reports `inside = 0` for every query against it while still
//! returning the (meaningless but defined) edge values.
//!
//! # Correctness model
//!
//! The inside flag is a discrete classification built from `f32` magnitude
//! comparisons and the integer degeneracy branch, so for triangles clear of the
//! degeneracy threshold and pixels clear of the dilated edge zero-lines the
//! `CPU` and `GPU` agree exactly, and the parity test asserts an exact `==` on
//! the `inside` `bool`. The three edge values thread through multiplies and adds
//! (and the dilation `|a| + |b|` term), so `CPU` and `GPU` are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate. The
//! parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each edge value.
//!
//! # Degenerate inputs
//!
//! A degenerate triangle has `|triangle_area2| < CMP_EPS`; the kernel detects it
//! with an integer flag and forces `inside = 0` for every query, mirroring the
//! empty coverage the reference reports. The edge values are still computed from
//! the (un-swapped) edges so the output buffer is always fully written.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`, an
//! `i32`-to-`f32` cast and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt`, no `round`
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. There is no loop: each thread performs a fixed, bounded sequence
//! of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::conservative_raster`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` conservative-raster kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`conservative_raster`](prism_render_architecture::particle::conservative_raster)
/// branch for branch; see the module documentation for the algorithm.
const CONSERVATIVE_RASTER_WGSL: &str = r#"
// Conservative-raster twin: one thread per pixel query reproduces the reference
// triangle-area sign, the degeneracy test, the CCW edge construction, the
// half-pixel edge dilation, the per-edge evaluation at the pixel center, and the
// all-edges-non-negative coverage predicate. It mirrors the CPU golden
// `particle::conservative_raster` branch for branch, uses only the portable
// core-WGSL subset (abs and + - * / plus an i32->f32 cast and unsigned index
// math), needs no sqrt and no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::conservative_raster；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a signed area or an edge value is treated as zero.
// Matches the reference `CMP_EPS`; the compare rule used instead of an f32 `==`.
// A pixel is covered by an edge when its edge value is at least `-CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Triangle corner 0, 1 and 2 in screen space, shared by every query.
    v0: vec2<f32>,
    v1: vec2<f32>,
    v2: vec2<f32>,
    // Half-pixel dilation radius: 0.5 for true conservative coverage, 0.0 for a
    // standard center-inside test.
    half_pixel: f32,
    // Number of pixel queries; threads past this short-circuit.
    count: u32,
}

struct Query {
    // Integer pixel column and row the thread classifies.
    px: i32,
    py: i32,
}

struct Result {
    // The three dilated edge values at the pixel center.
    edge: vec3<f32>,
    // 1 when the pixel is covered (and the triangle is non-degenerate), else 0.
    inside: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Builds the directed edge line (a, b, c) for the segment from `p0` to `p1`.
// The gradient (a, b) is the inward normal for a CCW triangle, so the edge value
// is non-negative to the left of `p0 -> p1`; mirrors the reference
// `edge_between`.
fn edge_between(p0: vec2<f32>, p1: vec2<f32>) -> vec3<f32> {
    let a = p0.y - p1.y;
    let b = p1.x - p0.x;
    let c = (p1.y - p0.y) * p0.x - (p1.x - p0.x) * p0.y;
    return vec3<f32>(a, b, c);
}

// Pushes a single edge outward along its normal by `half_pixel * (|a| + |b|)`;
// mirrors the reference `dilate_one`.
fn dilate_one(e: vec3<f32>, half_pixel: f32) -> vec3<f32> {
    return vec3<f32>(e.x, e.y, e.z + half_pixel * (abs(e.x) + abs(e.y)));
}

// Evaluates the signed edge value `a*x + b*y + c` at (x, y); mirrors the
// reference `Edge::eval`.
fn edge_eval(e: vec3<f32>, x: f32, y: f32) -> f32 {
    return e.x * x + e.y * y + e.z;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Twice the signed area encodes the winding; a near-zero magnitude is a
    // degenerate (collinear or coincident) triangle that covers no pixel.
    let area2 =
        (params.v1.x - params.v0.x) * (params.v2.y - params.v0.y)
        - (params.v1.y - params.v0.y) * (params.v2.x - params.v0.x);
    let degenerate = abs(area2) < CMP_EPS;

    // Normalize a clockwise input to CCW by swapping the last two corners, so
    // the coverage test is a single all-edges-non-negative check.
    var a0 = params.v0;
    var a1 = params.v1;
    var a2 = params.v2;
    if (area2 < 0.0) {
        a1 = params.v2;
        a2 = params.v1;
    }

    // Build and dilate the three directed edges.
    let e0 = dilate_one(edge_between(a0, a1), params.half_pixel);
    let e1 = dilate_one(edge_between(a1, a2), params.half_pixel);
    let e2 = dilate_one(edge_between(a2, a0), params.half_pixel);

    // Evaluate the dilated edges at the pixel center (px + 0.5, py + 0.5).
    let cx = f32(q.px) + 0.5;
    let cy = f32(q.py) + 0.5;
    let val0 = edge_eval(e0, cx, cy);
    let val1 = edge_eval(e1, cx, cy);
    let val2 = edge_eval(e2, cx, cy);

    // A pixel is covered when every dilated edge value is at least -CMP_EPS and
    // the triangle is non-degenerate.
    var inside: u32 = 0u;
    if (!degenerate && val0 >= -CMP_EPS && val1 >= -CMP_EPS && val2 >= -CMP_EPS) {
        inside = 1u;
    }

    var out: Result;
    out.edge = vec3<f32>(val0, val1, val2);
    out.inside = inside;
    results[idx] = out;
}
"#;

/// `repr(C)` uniform parameters for one dispatch: the shared triangle corners,
/// the `half_pixel` dilation radius and the query count, matching the `WGSL`
/// `Params` struct in [`CONSERVATIVE_RASTER_WGSL`]. The three `vec2` corners and
/// the trailing `f32` / `u32` pack into a `32`-byte, `std140`-aligned block with
/// no explicit padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Triangle corner `0`.
    v0: [f32; 2],
    /// Triangle corner `1`.
    v1: [f32; 2],
    /// Triangle corner `2`.
    v2: [f32; 2],
    /// Half-pixel dilation radius.
    half_pixel: f32,
    /// Number of valid pixel queries in the input and output buffers.
    count: u32,
}

/// `repr(C)` `std430` layout of one pixel query, matching the `WGSL` `Query`
/// struct: an integer pixel column and row.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Pixel column `x`.
    px: i32,
    /// Pixel row `y`.
    py: i32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the three dilated edge values and the packed `inside` flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The three dilated edge values at the pixel center.
    edge: [f32; 3],
    /// `1` when the pixel is covered, `0` otherwise.
    inside: u32,
}

/// One pixel query for the conservative-raster twin: the integer pixel
/// coordinates `(x, y)` classified against the shared triangle.
///
/// The triangle and the `half_pixel` dilation radius are supplied once per
/// batch to [`GpuConservativeRaster::evaluate`]; each query only carries its own
/// pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConservativeRasterQuery {
    /// Pixel column.
    pub x: i32,
    /// Pixel row.
    pub y: i32,
}

/// One resolved answer for a single pixel query, mirroring the reference
/// coverage classification and edge evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConservativeRasterResult {
    /// Whether the pixel is covered, matching the reference
    /// [`pixel_covered`](prism_render_architecture::particle::conservative_raster::pixel_covered)
    /// on the dilated edges (forced `false` for a degenerate triangle).
    pub inside: bool,
    /// The three dilated edge values at the pixel center `(x + 0.5, y + 0.5)`,
    /// matching the reference
    /// [`Edge::eval`](prism_render_architecture::particle::conservative_raster::Edge::eval)
    /// on the dilated edges.
    pub edges: [f32; 3],
}

/// Encodes the shared triangle, dilation radius and query count into the
/// `std140` [`GpuParams`] block.
fn encode_params(triangle: [[f32; 2]; 3], half_pixel: f32, count: u32) -> GpuParams {
    GpuParams {
        v0: triangle[0],
        v1: triangle[1],
        v2: triangle[2],
        half_pixel,
        count,
    }
}

/// Encodes one [`ConservativeRasterQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ConservativeRasterQuery) -> GpuQuery {
    GpuQuery { px: q.x, py: q.y }
}

/// Decodes one packed [`GpuResult`] into the public [`ConservativeRasterResult`],
/// turning the `inside` flag back into a `bool`.
fn decode_result(raw: &GpuResult) -> ConservativeRasterResult {
    ConservativeRasterResult {
        inside: raw.inside != 0,
        edges: raw.edge,
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

/// A compiled, reusable conservative-raster compute pipeline, twinning the
/// `CPU` golden
/// [`conservative_raster`](prism_render_architecture::particle::conservative_raster).
pub struct GpuConservativeRaster {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConservativeRaster {
    /// Compiles the conservative-raster kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConservativeRaster {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_conservative_raster"),
            source: ShaderSource::Wgsl(CONSERVATIVE_RASTER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_conservative_raster_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_conservative_raster_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_conservative_raster_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConservativeRaster {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every query in `queries` against the shared `triangle` dilated
    /// by `half_pixel`, returning one [`ConservativeRasterResult`] per input, in
    /// order.
    ///
    /// The `inside` flag equals the reference exactly for triangles clear of the
    /// degeneracy threshold and pixels clear of the dilated edge zero-lines; the
    /// edge values match to within the tolerance documented on this module. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        triangle: [[f32; 2]; 3],
        half_pixel: f32,
        queries: &[ConservativeRasterQuery],
    ) -> Vec<ConservativeRasterResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = encode_params(triangle, half_pixel, count as u32);
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_conservative_raster_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_conservative_raster_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_conservative_raster_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_conservative_raster_bind_group"),
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
            label: Some("prism_volumetric_conservative_raster_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_conservative_raster_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_conservative_raster_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel query, flattened to a 1-D dispatch.
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
