//! `wgpu` compute twin of the deterministic vis-buffer software rasterizer
//! ([`rasterize_triangle`](prism_render_architecture::virtual_geometry::software_raster::rasterize_triangle),
//! the `CPU` golden that fills a 64-bit visibility buffer with top-left-rule
//! coverage and reversed-Z depth compositing).
//!
//! # What is twinned
//!
//! The whole single-triangle raster: the signed-area winding test and optional
//! back-face cull, the last-two-vertex swap that re-orients a negative-area
//! triangle, the clamped integer bounding box, the three top-left edge flags,
//! and the per-pixel coverage, screen-linear barycentric depth, depth-key
//! encoding and nearest-depth composite. One `GPU` thread owns one whole
//! triangle ([`RasterTriangleVisQuery`]) and replays the exact scan the
//! reference does, writing the full `width * height` pixel grid.
//!
//! # Representation
//!
//! The reference packs each pixel into a `u64` (`depth_key << 32 | payload`)
//! and composites with a single `u64` `max`. `WGSL` has no `u64`, so the twin
//! stores each pixel as two `u32` words — `hi` is the depth key, `lo` is the
//! payload — and reproduces the `u64` ordering lexicographically: a candidate
//! beats the slot when its `hi` is larger, or the `hi` ties and its `lo` is
//! larger. That is bit-for-bit identical to the reference's `u64` comparison.
//! The depth key is the raw `bitcast<u32>(clamp(depth, 0, 1))`, matching
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth).
//!
//! The buffer dimensions are bounded by a fixed `MAX_DIM` of `64` (so at most
//! `4096` pixels); the host sizes the readback to the query's `width * height`.
//!
//! # What stays on the host
//!
//! The variable-length vertex projection, cluster dispatch and id packing of
//! the shipping path stay host-side; this twin pins only the fixed per-triangle
//! fill. The host builds the reference `VisBuffer` for the oracle and splits its
//! `u64` words into the same `(hi, lo)` pairs for the parity diff.
//!
//! # Correctness model
//!
//! Every covered-pixel quantity flows through the same `f32` arithmetic in the
//! same evaluation order as the reference, and the depth key, payload and
//! composite are integer or bit-pattern operations, so the `CPU` and `GPU`
//! agree bit for bit and the parity test asserts an exact `==` on every pixel.
//! Products are materialized before the barycentric depth sum so no fused
//! multiply-add can reassociate the twin away from the scalar reference.
//! Fixtures use integer vertex coordinates and depths strictly inside `(0, 1)`,
//! keeping every edge value exact and every pixel-center coverage decision on
//! the same side as the reference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `f32` arithmetic,
//! comparisons, `min`, `max`, `clamp`, `floor`, `ceil` and `bitcast` — with no
//! `u64`, no `sin`, `cos`, `exp`, `log`, `pow` or `sqrt`, no inverse
//! trigonometry and no `round`. No optional device feature is required, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。
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

/// Maximum side length, in pixels, of a vis-buffer the device twin rasterizes.
/// `width` and `height` must each be at most this value; the fixed-width result
/// buffer reserves `MAX_DIM * MAX_DIM` pixels and the host trims the readback to
/// the query's `width * height`.
const MAX_DIM: u32 = 64;

/// Maximum number of pixels in one vis-buffer, `MAX_DIM * MAX_DIM`. Kept in sync
/// with the `4096u` literal in the inlined `WGSL`.
const MAX_PIXELS: usize = (MAX_DIM as usize) * (MAX_DIM as usize);

/// Number of threads per workgroup. The raster is a serial per-triangle scan, so
/// one invocation owns one whole triangle and the dispatch is one workgroup per
/// query.
const WORKGROUP_SIZE: u32 = 1;

/// The portable core-`WGSL` single-triangle vis-buffer rasterizer, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `raster` mirrors the `CPU` golden
/// [`rasterize_triangle`](prism_render_architecture::virtual_geometry::software_raster::rasterize_triangle).
const RASTER_TRIANGLE_VIS_WGSL: &str = r#"
// Single-triangle vis-buffer raster twin: one thread rasterizes one whole
// triangle into a width*height pixel grid, reproducing the CPU golden
// `virtual_geometry::software_raster::rasterize_triangle` — signed-area winding
// and back-face cull, the last-two-vertex swap, the clamped bounding box, the
// three top-left edge flags, per-pixel top-left coverage, screen-linear
// barycentric depth, depth-key encode and nearest-depth composite. The u64
// vis word is split into two u32 (hi = depth key, lo = payload) and the u64
// max composite is reproduced lexicographically. No floating-point equality is
// used: a `== 0.0` test is written as `v <= 0.0 && v >= 0.0`, which is exactly
// the reference's zero test for finite values.
//
// Provenance: 孪生自本仓 prism_render_architecture::virtual_geometry::software_raster；无第三方
// 引擎源码或衍生代码。

const MAX_PIXELS: u32 = 4096u;

struct Params {
    // Number of triangles (queries) in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Vertex 0 position (x, y) and reversed-Z depth.
    v0x: f32,
    v0y: f32,
    v0d: f32,
    // Vertex 1.
    v1x: f32,
    v1y: f32,
    v1d: f32,
    // Vertex 2.
    v2x: f32,
    v2y: f32,
    v2d: f32,
    // Payload written verbatim into the low word of every covered pixel.
    payload: u32,
    // Non-zero to cull a back-facing (non-positive original area) triangle.
    cull_back: u32,
    // Vis-buffer dimensions; both <= MAX_DIM.
    width: u32,
    height: u32,
}

struct Pixel {
    // Depth key: bitcast<u32>(clamp(depth, 0, 1)).
    hi: u32,
    // Payload.
    lo: u32,
}

struct Result {
    // Row-major pixels, indexed y * width + x; only width*height are meaningful.
    pixels: array<Pixel, 4096>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Signed edge function of point p against the directed edge a -> b. The two
// products are materialized before the subtract so no fused multiply-add can
// reassociate this away from the scalar reference.
fn edge_fn(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    let m0 = (bx - ax) * (py - ay);
    let m1 = (by - ay) * (px - ax);
    return m0 - m1;
}

// Exact `v == 0.0` for finite values, written without an f32 equality operator.
fn is_zero_f(v: f32) -> bool {
    return v <= 0.0 && v >= 0.0;
}

// True when directed edge a -> b is a top or left edge (y-down, positive area):
// dy < 0, or a horizontal edge pointing in +x.
fn is_top_left(ax: f32, ay: f32, bx: f32, by: f32) -> bool {
    let dx = bx - ax;
    let dy = by - ay;
    return dy < 0.0 || (is_zero_f(dy) && dx > 0.0);
}

// Fill test for one edge value under the top-left rule.
fn covers(e: f32, tl: bool) -> bool {
    return e > 0.0 || (is_zero_f(e) && tl);
}

// Floors a coordinate and clamps it into [0, limit] as a start index.
fn clamp_floor(value: f32, limit: u32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return min(u32(floor(value)), limit);
}

// Ceils a coordinate to an exclusive end index clamped into [0, limit].
fn clamp_ceil(value: f32, limit: u32) -> u32 {
    if (value <= 0.0) {
        return 0u;
    }
    return min(u32(ceil(value)), limit);
}

@compute @workgroup_size(1)
fn raster(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.count) {
        return;
    }

    let width = queries[qi].width;
    let height = queries[qi].height;
    let used = width * height;

    // Clear the meaningful pixels to the farthest key with an empty payload.
    for (var i: u32 = 0u; i < used; i = i + 1u) {
        results[qi].pixels[i].hi = 0u;
        results[qi].pixels[i].lo = 0u;
    }

    var v0x: f32 = queries[qi].v0x;
    var v0y: f32 = queries[qi].v0y;
    var v0d: f32 = queries[qi].v0d;
    var v1x: f32 = queries[qi].v1x;
    var v1y: f32 = queries[qi].v1y;
    var v1d: f32 = queries[qi].v1d;
    var v2x: f32 = queries[qi].v2x;
    var v2y: f32 = queries[qi].v2y;
    var v2d: f32 = queries[qi].v2d;
    let payload = queries[qi].payload;
    let cull_back = queries[qi].cull_back;

    let raw_area = edge_fn(v0x, v0y, v1x, v1y, v2x, v2y);
    // Degenerate triangle: no coverage (pixels already cleared).
    if (is_zero_f(raw_area)) {
        return;
    }
    // Back-facing under the original winding.
    if (cull_back != 0u && raw_area < 0.0) {
        return;
    }
    // Re-orient a negative-area triangle by swapping the last two vertices.
    if (raw_area < 0.0) {
        let tx = v1x;
        let ty = v1y;
        let td = v1d;
        v1x = v2x;
        v1y = v2y;
        v1d = v2d;
        v2x = tx;
        v2y = ty;
        v2d = td;
    }
    let area = edge_fn(v0x, v0y, v1x, v1y, v2x, v2y);
    let inv_area = 1.0 / area;

    let min_x = min(min(v0x, v1x), v2x);
    let max_x = max(max(v0x, v1x), v2x);
    let min_y = min(min(v0y, v1y), v2y);
    let max_y = max(max(v0y, v1y), v2y);

    let x_start = clamp_floor(min_x, width);
    let x_end = clamp_ceil(max_x, width);
    let y_start = clamp_floor(min_y, height);
    let y_end = clamp_ceil(max_y, height);

    // Edge v1->v2 owns e0/bary0, v2->v0 owns e1/bary1, v0->v1 owns e2/bary2.
    let tl0 = is_top_left(v1x, v1y, v2x, v2y);
    let tl1 = is_top_left(v2x, v2y, v0x, v0y);
    let tl2 = is_top_left(v0x, v0y, v1x, v1y);

    for (var y: u32 = y_start; y < y_end; y = y + 1u) {
        for (var x: u32 = x_start; x < x_end; x = x + 1u) {
            let px = f32(x) + 0.5;
            let py = f32(y) + 0.5;
            let e0 = edge_fn(v1x, v1y, v2x, v2y, px, py);
            let e1 = edge_fn(v2x, v2y, v0x, v0y, px, py);
            let e2 = edge_fn(v0x, v0y, v1x, v1y, px, py);

            let inside = covers(e0, tl0) && covers(e1, tl1) && covers(e2, tl2);
            if (!inside) {
                continue;
            }

            let b0 = e0 * inv_area;
            let b1 = e1 * inv_area;
            let b2 = e2 * inv_area;
            // Materialize the products so the depth sum cannot be fused.
            let t0 = b0 * v0d;
            let t1 = b1 * v1d;
            let t2 = b2 * v2d;
            let depth = (t0 + t1) + t2;

            let hi = bitcast<u32>(clamp(depth, 0.0, 1.0));
            let lo = payload;

            let idx = y * width + x;
            let shi = results[qi].pixels[idx].hi;
            let slo = results[qi].pixels[idx].lo;
            // Lexicographic (hi, lo) compare reproduces the reference u64 max.
            let greater = hi > shi || (hi == shi && lo > slo);
            if (greater) {
                results[qi].pixels[idx].hi = hi;
                results[qi].pixels[idx].lo = lo;
            }
        }
    }
}
"#;

/// Uniform parameters for one dispatch: the number of triangles plus padding to
/// a 16-byte `std430` uniform.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of triangles (queries).
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one triangle query, matching the `WGSL` `Query`
/// struct field for field: nine vertex floats, the payload, the cull flag and
/// the two dimensions.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Vertex 0 x.
    v0x: f32,
    /// Vertex 0 y.
    v0y: f32,
    /// Vertex 0 depth.
    v0d: f32,
    /// Vertex 1 x.
    v1x: f32,
    /// Vertex 1 y.
    v1y: f32,
    /// Vertex 1 depth.
    v1d: f32,
    /// Vertex 2 x.
    v2x: f32,
    /// Vertex 2 y.
    v2y: f32,
    /// Vertex 2 depth.
    v2d: f32,
    /// Payload written to every covered pixel.
    payload: u32,
    /// Non-zero to cull a back-facing triangle.
    cull_back: u32,
    /// Vis-buffer width.
    width: u32,
    /// Vis-buffer height.
    height: u32,
}

/// `repr(C)` `std430` layout of one pixel, matching the `WGSL` `Pixel` struct:
/// the depth key then the payload.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPixel {
    /// Depth key: `bitcast<u32>(clamp(depth, 0, 1))`.
    hi: u32,
    /// Payload.
    lo: u32,
}

/// `repr(C)` `std430` layout of one rasterized vis-buffer, matching the `WGSL`
/// `Result` struct: a fixed-width pixel grid indexed `y * width + x`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Row-major pixels; only `width * height` are meaningful.
    pixels: [GpuPixel; MAX_PIXELS],
}

/// One triangle query: three `[x, y, depth]` vertices in y-down pixel space, the
/// payload, the back-face cull flag and the target vis-buffer dimensions.
///
/// Mirrors the inputs to the reference
/// [`rasterize_triangle`](prism_render_architecture::virtual_geometry::software_raster::rasterize_triangle);
/// `vertices[i]` is `[x, y, depth]`, matching
/// [`ScreenVertex`](prism_render_architecture::virtual_geometry::software_raster::ScreenVertex).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterTriangleVisQuery {
    /// The three vertices, each `[x, y, depth]` with reversed-Z depth in
    /// `[0, 1]`.
    pub vertices: [[f32; 3]; 3],
    /// Payload written verbatim into the low word of every covered pixel.
    pub payload: u32,
    /// `true` to cull a triangle whose original winding is back-facing.
    pub cull_back: bool,
    /// Vis-buffer width in pixels; at most `64`.
    pub width: u32,
    /// Vis-buffer height in pixels; at most `64`.
    pub height: u32,
}

/// One rasterized vis-buffer, mirroring the reference
/// [`VisBuffer`](prism_render_architecture::virtual_geometry::software_raster::VisBuffer).
///
/// `pixels` is row-major (`y * width + x`); each entry is `[hi, lo]` where `hi`
/// is the depth key and `lo` is the payload, the two halves of the reference's
/// packed `u64` word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterTriangleVisResult {
    /// Vis-buffer width in pixels.
    pub width: u32,
    /// Vis-buffer height in pixels.
    pub height: u32,
    /// Row-major pixels; each `[hi, lo]` is the split of the reference `u64`.
    pub pixels: Vec<[u32; 2]>,
}

/// A compiled, reusable single-triangle vis-buffer raster pipeline, twinning the
/// `CPU` golden
/// [`rasterize_triangle`](prism_render_architecture::virtual_geometry::software_raster::rasterize_triangle).
pub struct GpuRasterTriangleVis {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRasterTriangleVis {
    /// Compiles the raster kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRasterTriangleVis {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis"),
            source: ShaderSource::Wgsl(RASTER_TRIANGLE_VIS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("raster"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRasterTriangleVis {
            module,
            layout,
            pipeline,
        }
    }

    /// Rasterizes every triangle in `queries` and returns one
    /// [`RasterTriangleVisResult`] per input, in order.
    ///
    /// Every pixel equals the reference exactly (coverage and compositing are
    /// bit-pattern and integer operations and the depth arithmetic matches the
    /// reference's evaluation order). An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RasterTriangleVisQuery],
    ) -> Vec<RasterTriangleVisResult> {
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
            label: Some("prism_volumetric_raster_triangle_vis_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_bind_group"),
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
            label: Some("prism_volumetric_raster_triangle_vis_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_raster_triangle_vis_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_raster_triangle_vis_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triangle, flattened to a 1-D dispatch.
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

        raw.iter()
            .zip(queries.iter())
            .map(|(r, q)| decode_result(r, q.width, q.height))
            .collect()
    }
}

/// Encodes one [`RasterTriangleVisQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RasterTriangleVisQuery) -> GpuQuery {
    GpuQuery {
        v0x: q.vertices[0][0],
        v0y: q.vertices[0][1],
        v0d: q.vertices[0][2],
        v1x: q.vertices[1][0],
        v1y: q.vertices[1][1],
        v1d: q.vertices[1][2],
        v2x: q.vertices[2][0],
        v2y: q.vertices[2][1],
        v2d: q.vertices[2][2],
        payload: q.payload,
        cull_back: u32::from(q.cull_back),
        width: q.width,
        height: q.height,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RasterTriangleVisResult`],
/// trimming the fixed pixel grid to the `width * height` meaningful pixels.
fn decode_result(raw: &GpuResult, width: u32, height: u32) -> RasterTriangleVisResult {
    let used = (width as usize) * (height as usize);
    let pixels = raw.pixels[..used].iter().map(|p| [p.hi, p.lo]).collect();
    RasterTriangleVisResult {
        width,
        height,
        pixels,
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
