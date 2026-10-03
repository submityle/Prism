//! `wgpu` compute twin of the deterministic indexed-cluster vis-buffer software
//! rasterizer
//! ([`rasterize_cluster`](prism_render_architecture::virtual_geometry::software_raster::rasterize_cluster),
//! the `CPU` golden that composites a whole cluster of indexed triangles into a
//! 64-bit visibility buffer with top-left-rule coverage and reversed-Z depth).
//!
//! # What is twinned
//!
//! The full per-cluster raster: for each triangle in the cluster the signed-area
//! winding test and optional back-face cull, the last-two-vertex swap that
//! re-orients a negative-area triangle, the clamped integer bounding box, the
//! three top-left edge flags, the per-pixel coverage, screen-linear barycentric
//! depth, depth-key encode and nearest-depth composite, plus the index skip for
//! an out-of-range triangle and the `(cluster_id << 7) | triangle_id` payload
//! packing from
//! [`pack_cluster_triangle`](prism_render_architecture::virtual_geometry::software_raster::pack_cluster_triangle).
//!
//! # Parallelization
//!
//! The reference composites triangles serially, but a `u64` `max` composite is
//! order-independent: the surviving pixel is the maximum packed word over every
//! covering triangle. The twin therefore runs one `GPU` thread per output pixel
//! ([`RasterClusterVisQuery`] is one cluster), looping over all cluster
//! triangles and keeping the lexicographic maximum `(hi, lo)` pair. Because the
//! triangle id lives in the low payload bits, a depth tie resolves to the
//! larger triangle id — exactly the triangle the serial reference composites
//! last — so the parallel result is identical to the serial scan.
//!
//! # Representation
//!
//! The reference packs each pixel into a `u64` (`depth_key << 32 | payload`) and
//! composites with a single `u64` `max`. `WGSL` has no `u64`, so the twin stores
//! each pixel as two `u32` words — `hi` is the depth key, `lo` is the payload —
//! and reproduces the `u64` ordering lexicographically: a candidate beats the
//! slot when its `hi` is larger, or the `hi` ties and its `lo` is larger. That
//! is bit-for-bit identical to the reference's `u64` comparison. The depth key
//! is the raw `bitcast<u32>(clamp(depth, 0, 1))`, matching
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth).
//!
//! The buffer dimensions are bounded (`width`, `height` at most `16`, so at most
//! `256` pixels); the cluster holds at most `64` vertices and `128` triangles,
//! matching the reference
//! [`MAX_CLUSTER_TRIANGLES`](prism_render_architecture::virtual_geometry::software_raster::MAX_CLUSTER_TRIANGLES).
//! The host sizes the readback to the query's `width * height`.
//!
//! # What stays on the host
//!
//! The variable-length vertex projection and the cluster-assignment pass of the
//! shipping path stay host-side; this twin pins only the fixed-capacity
//! per-cluster fill. The host builds the reference `VisBuffer` for the oracle and
//! splits its `u64` words into the same `(hi, lo)` pairs for the parity diff.
//!
//! # Correctness model
//!
//! Every covered-pixel quantity flows through the same `f32` arithmetic in the
//! same evaluation order as the reference; the payload and the composite order
//! are integer or bit-pattern operations. Products are materialized before the
//! barycentric depth sum so no fused multiply-add can reassociate the twin away
//! from the scalar reference. The depth key is a `bitcast` of a continuous
//! barycentric sum, which a device `f32` divide / multiply-add chain can land a
//! unit in the last place from, so the parity test compares the decoded depth
//! within tolerance while asserting an exact `==` on every payload id.
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

/// Maximum vis-buffer width, in pixels, the cluster twin rasterizes.
const MAX_W: u32 = 16;

/// Maximum vis-buffer height, in pixels, the cluster twin rasterizes.
const MAX_H: u32 = 16;

/// Maximum number of pixels in one vis-buffer, `MAX_W * MAX_H`. Kept in sync
/// with the `256u` literal and `array<Pixel, 256>` in the inlined `WGSL`.
const MAX_PIXELS: usize = (MAX_W as usize) * (MAX_H as usize);

/// Maximum number of vertices one cluster may carry. Kept in sync with the
/// `64u` literal and the `array<f32, 192>` vertex store in the inlined `WGSL`.
const MAX_VERTS: usize = 64;

/// Maximum number of triangles one cluster may carry, mirroring the reference
/// [`MAX_CLUSTER_TRIANGLES`](prism_render_architecture::virtual_geometry::software_raster::MAX_CLUSTER_TRIANGLES).
/// Kept in sync with the `128u` literal and the `array<u32, 384>` index store in
/// the inlined `WGSL`.
const MAX_TRIS: usize = 128;

/// Length of the flattened vertex store, `MAX_VERTS * 3` (`x`, `y`, `depth`).
const VERT_WORDS: usize = MAX_VERTS * 3;

/// Length of the flattened triangle index store, `MAX_TRIS * 3`.
const TRI_WORDS: usize = MAX_TRIS * 3;

/// Number of threads per workgroup. The twin runs one thread per output pixel.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` indexed-cluster vis-buffer rasterizer, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `raster` mirrors the `CPU` golden
/// [`rasterize_cluster`](prism_render_architecture::virtual_geometry::software_raster::rasterize_cluster).
const RASTER_CLUSTER_VIS_WGSL: &str = r#"
// Indexed-cluster vis-buffer raster twin: one thread owns one output pixel and
// loops over every triangle in the cluster, reproducing the CPU golden
// `virtual_geometry::software_raster::rasterize_cluster` — the per-triangle
// signed-area winding and back-face cull, the last-two-vertex swap, the clamped
// bounding box, the three top-left edge flags, per-pixel top-left coverage,
// screen-linear barycentric depth, depth-key encode, the out-of-range index
// skip and the (cluster_id << 7) | triangle_id payload. The serial u64 max
// composite is order-independent, so a per-pixel lexicographic maximum over all
// covering triangles reproduces it exactly. The u64 vis word is split into two
// u32 (hi = depth key, lo = payload). No floating-point equality is used: a
// `== 0.0` test is written as `v <= 0.0 && v >= 0.0`, exactly the reference's
// zero test for finite values.
//
// Provenance: 孪生自本仓 prism_render_architecture::virtual_geometry::software_raster；无第三方
// 引擎源码或衍生代码。

const MAX_PIXELS: u32 = 256u;
const MAX_VERTS: u32 = 64u;
const MAX_TRIS: u32 = 128u;
const CLUSTER_TRIANGLE_BITS: u32 = 7u;
const TRIANGLE_ID_MASK: u32 = 127u;

struct Params {
    // Number of clusters (queries) in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Flattened vertex store, 3 words (x, y, depth) per vertex.
    verts: array<f32, 192>,
    // Flattened triangle index store, 3 indices per triangle.
    tris: array<u32, 384>,
    // Number of valid vertices; an index at or past this skips the triangle.
    vert_count: u32,
    // Number of triangles to rasterize (already capped to MAX_TRIS on host).
    tri_count: u32,
    // Cluster id packed into the high payload bits.
    cluster_id: u32,
    // Non-zero to cull back-facing (non-positive original area) triangles.
    cull_back: u32,
    // Vis-buffer dimensions; both <= 16.
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
}

struct Pixel {
    // Depth key: bitcast<u32>(clamp(depth, 0, 1)).
    hi: u32,
    // Payload: (cluster_id << 7) | triangle_id.
    lo: u32,
}

struct Result {
    // Row-major pixels, indexed y * width + x; only width*height are meaningful.
    pixels: array<Pixel, 256>,
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

// True when directed edge a -> b is a top or left edge (y-down, positive area).
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

@compute @workgroup_size(64)
fn raster(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total = params.count * MAX_PIXELS;
    if (tid >= total) {
        return;
    }
    let qi = tid / MAX_PIXELS;
    let pi = tid % MAX_PIXELS;

    let width = queries[qi].width;
    let height = queries[qi].height;
    let used = width * height;
    if (pi >= used) {
        // Outside the meaningful grid for this query; leave a cleared pixel.
        results[qi].pixels[pi].hi = 0u;
        results[qi].pixels[pi].lo = 0u;
        return;
    }

    let x = pi % width;
    let y = pi / width;
    let px = f32(x) + 0.5;
    let py = f32(y) + 0.5;

    let tri_count = queries[qi].tri_count;
    let vert_count = queries[qi].vert_count;
    let cluster_id = queries[qi].cluster_id;
    let cull_back = queries[qi].cull_back;

    var best_hi: u32 = 0u;
    var best_lo: u32 = 0u;

    for (var t: u32 = 0u; t < tri_count; t = t + 1u) {
        let i0 = queries[qi].tris[t * 3u + 0u];
        let i1 = queries[qi].tris[t * 3u + 1u];
        let i2 = queries[qi].tris[t * 3u + 2u];
        // Out-of-range index: skip this triangle, exactly like the reference.
        if (i0 >= vert_count || i1 >= vert_count || i2 >= vert_count) {
            continue;
        }

        var v0x: f32 = queries[qi].verts[i0 * 3u + 0u];
        var v0y: f32 = queries[qi].verts[i0 * 3u + 1u];
        var v0d: f32 = queries[qi].verts[i0 * 3u + 2u];
        var v1x: f32 = queries[qi].verts[i1 * 3u + 0u];
        var v1y: f32 = queries[qi].verts[i1 * 3u + 1u];
        var v1d: f32 = queries[qi].verts[i1 * 3u + 2u];
        var v2x: f32 = queries[qi].verts[i2 * 3u + 0u];
        var v2y: f32 = queries[qi].verts[i2 * 3u + 1u];
        var v2d: f32 = queries[qi].verts[i2 * 3u + 2u];

        let raw_area = edge_fn(v0x, v0y, v1x, v1y, v2x, v2y);
        // Degenerate triangle: no coverage.
        if (is_zero_f(raw_area)) {
            continue;
        }
        // Back-facing under the original winding.
        if (cull_back != 0u && raw_area < 0.0) {
            continue;
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
        // This pixel lies outside the triangle's clamped scan window.
        if (x < x_start || x >= x_end || y < y_start || y >= y_end) {
            continue;
        }

        // Edge v1->v2 owns e0/bary0, v2->v0 owns e1/bary1, v0->v1 owns e2/bary2.
        let tl0 = is_top_left(v1x, v1y, v2x, v2y);
        let tl1 = is_top_left(v2x, v2y, v0x, v0y);
        let tl2 = is_top_left(v0x, v0y, v1x, v1y);

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
        let d0 = b0 * v0d;
        let d1 = b1 * v1d;
        let d2 = b2 * v2d;
        let depth = (d0 + d1) + d2;

        let hi = bitcast<u32>(clamp(depth, 0.0, 1.0));
        let lo = (cluster_id << CLUSTER_TRIANGLE_BITS) | (t & TRIANGLE_ID_MASK);

        // Lexicographic (hi, lo) compare reproduces the reference u64 max.
        let greater = hi > best_hi || (hi == best_hi && lo > best_lo);
        if (greater) {
            best_hi = hi;
            best_lo = lo;
        }
    }

    results[qi].pixels[pi].hi = best_hi;
    results[qi].pixels[pi].lo = best_lo;
}
"#;

/// Uniform parameters for one dispatch: the number of clusters plus padding to a
/// 16-byte `std430` uniform.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of clusters (queries).
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one cluster query, matching the `WGSL` `Query`
/// struct field for field: the flattened vertex and triangle stores followed by
/// the counts, cluster id, cull flag and dimensions.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Flattened vertices, three words (`x`, `y`, `depth`) per vertex.
    verts: [f32; VERT_WORDS],
    /// Flattened triangle indices, three per triangle.
    tris: [u32; TRI_WORDS],
    /// Number of valid vertices.
    vert_count: u32,
    /// Number of triangles to rasterize.
    tri_count: u32,
    /// Cluster id packed into the high payload bits.
    cluster_id: u32,
    /// Non-zero to cull back-facing triangles.
    cull_back: u32,
    /// Vis-buffer width.
    width: u32,
    /// Vis-buffer height.
    height: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one pixel, matching the `WGSL` `Pixel` struct:
/// the depth key then the payload.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPixel {
    /// Depth key: `bitcast<u32>(clamp(depth, 0, 1))`.
    hi: u32,
    /// Payload: `(cluster_id << 7) | triangle_id`.
    lo: u32,
}

/// `repr(C)` `std430` layout of one rasterized vis-buffer, matching the `WGSL`
/// `Result` struct: a fixed-capacity pixel grid indexed `y * width + x`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Row-major pixels; only `width * height` are meaningful.
    pixels: [GpuPixel; MAX_PIXELS],
}

/// One cluster query: a list of `[x, y, depth]` vertices in y-down pixel space,
/// a list of triangle index triples into those vertices, the cluster id, the
/// back-face cull flag and the target vis-buffer dimensions.
///
/// Mirrors the inputs to the reference
/// [`rasterize_cluster`](prism_render_architecture::virtual_geometry::software_raster::rasterize_cluster);
/// each `vertices[i]` is `[x, y, depth]`, matching
/// [`ScreenVertex`](prism_render_architecture::virtual_geometry::software_raster::ScreenVertex),
/// and each `triangles[t]` is `[i0, i1, i2]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterClusterVisQuery {
    /// Cluster vertices, each `[x, y, depth]` with reversed-Z depth in `[0, 1]`;
    /// at most `64`.
    pub vertices: Vec<[f32; 3]>,
    /// Triangle index triples into `vertices`; at most `128` are rasterized.
    pub triangles: Vec<[u32; 3]>,
    /// Cluster id packed into the high bits of every covered pixel's payload.
    pub cluster_id: u32,
    /// `true` to cull triangles whose original winding is back-facing.
    pub cull_back: bool,
    /// Vis-buffer width in pixels; at most `16`.
    pub width: u32,
    /// Vis-buffer height in pixels; at most `16`.
    pub height: u32,
}

/// One rasterized vis-buffer, mirroring the reference
/// [`VisBuffer`](prism_render_architecture::virtual_geometry::software_raster::VisBuffer).
///
/// `pixels` is row-major (`y * width + x`); each entry is `[hi, lo]` where `hi`
/// is the depth key and `lo` is the payload, the two halves of the reference's
/// packed `u64` word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterClusterVisResult {
    /// Vis-buffer width in pixels.
    pub width: u32,
    /// Vis-buffer height in pixels.
    pub height: u32,
    /// Row-major pixels; each `[hi, lo]` is the split of the reference `u64`.
    pub pixels: Vec<[u32; 2]>,
}

/// A compiled, reusable indexed-cluster vis-buffer raster pipeline, twinning the
/// `CPU` golden
/// [`rasterize_cluster`](prism_render_architecture::virtual_geometry::software_raster::rasterize_cluster).
pub struct GpuRasterClusterVis {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRasterClusterVis {
    /// Compiles the cluster raster kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRasterClusterVis {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis"),
            source: ShaderSource::Wgsl(RASTER_CLUSTER_VIS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("raster"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRasterClusterVis {
            module,
            layout,
            pipeline,
        }
    }

    /// Rasterizes every cluster in `queries` and returns one
    /// [`RasterClusterVisResult`] per input, in order.
    ///
    /// An empty batch short-circuits before any dispatch and returns an empty
    /// vector, because a zero-sized storage buffer is not allowed.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RasterClusterVisQuery],
    ) -> Vec<RasterClusterVisResult> {
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
            label: Some("prism_volumetric_raster_cluster_vis_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_bind_group"),
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
            label: Some("prism_volumetric_raster_cluster_vis_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_raster_cluster_vis_encoder"),
        });
        {
            let threads = (count as u32) * (MAX_PIXELS as u32);
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_raster_cluster_vis_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output pixel across all clusters, flattened to 1-D.
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

/// Encodes one [`RasterClusterVisQuery`] into its `std430` [`GpuQuery`] slot,
/// flattening the vertices and triangle indices and capping the triangle count
/// to [`MAX_TRIS`] exactly as the reference does.
fn encode_query(q: &RasterClusterVisQuery) -> GpuQuery {
    let mut verts = [0.0_f32; VERT_WORDS];
    let vert_count = q.vertices.len().min(MAX_VERTS);
    for (slot, v) in verts
        .chunks_exact_mut(3)
        .zip(q.vertices.iter().take(vert_count))
    {
        slot[0] = v[0];
        slot[1] = v[1];
        slot[2] = v[2];
    }

    let mut tris = [0_u32; TRI_WORDS];
    let tri_count = q.triangles.len().min(MAX_TRIS);
    for (slot, t) in tris
        .chunks_exact_mut(3)
        .zip(q.triangles.iter().take(tri_count))
    {
        slot[0] = t[0];
        slot[1] = t[1];
        slot[2] = t[2];
    }

    GpuQuery {
        verts,
        tris,
        vert_count: vert_count as u32,
        tri_count: tri_count as u32,
        cluster_id: q.cluster_id,
        cull_back: u32::from(q.cull_back),
        width: q.width,
        height: q.height,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RasterClusterVisResult`],
/// trimming the fixed pixel grid to the `width * height` meaningful pixels.
fn decode_result(raw: &GpuResult, width: u32, height: u32) -> RasterClusterVisResult {
    let used = (width as usize) * (height as usize);
    let pixels = raw.pixels[..used].iter().map(|p| [p.hi, p.lo]).collect();
    RasterClusterVisResult {
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
