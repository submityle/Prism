//! `wgpu` compute twin of the simple-polygon scanline-fill contract
//! ([`scanline_polygon_fill`](prism_render_architecture::particle::scanline_polygon_fill),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`scanline_polygon_fill`](prism_render_architecture::particle::scanline_polygon_fill)
//! turns a simple polygon ring (an ordered, *open* slice of 2D vertices) into
//! the ordered list of horizontal interior fill spans that cover it:
//! [`scanline_fill`](prism_render_architecture::particle::scanline_polygon_fill::scanline_fill)
//! builds an Edge Table of every non-horizontal ring edge, sweeps integer
//! scanlines sampled at their centres `yc = y + 0.5`, maintains an Active Edge
//! Table under the half-open `[y_lower, y_upper)` rule, sorts the per-row `x`
//! crossings and pairs them under the even-odd parity rule into one
//! [`Span`](prism_render_architecture::particle::scanline_polygon_fill::Span)
//! per interval, and
//! [`filled_pixel_count`](prism_render_architecture::particle::scanline_polygon_fill::filled_pixel_count)
//! sums every span's
//! [`width`](prism_render_architecture::particle::scanline_polygon_fill::Span::width).
//! [`GpuScanlinePolygonFill`] is the on-device twin: one thread fills one
//! polygon, each ring living in its own fixed-length vertex slot, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! same spans, in the same order, and the same total pixel count the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent polygons, each packed into one fixed-length slot
//! of up to [`MAX_POLYGON_VERTS`] vertices, the kernel reproduces the full
//! [`scanline_fill`](prism_render_architecture::particle::scanline_polygon_fill::scanline_fill)
//! span list and the
//! [`filled_pixel_count`](prism_render_architecture::particle::scanline_polygon_fill::filled_pixel_count)
//! reduction into one [`GpuScanlineFill`] record: the span count, the ordered
//! [`GpuSpan`] runs (each a `y`, `x_start`, `x_end` triple mirroring the
//! reference
//! [`Span`](prism_render_architecture::particle::scanline_polygon_fill::Span)),
//! and the summed interior pixel count. The ring is *open*: the closing edge
//! from the last vertex back to the first is implied and never repeated,
//! exactly as in the reference.
//!
//! # Correctness model
//!
//! Every twinned answer is integer-exact. The span coordinates are
//! `(x + 0.5).floor()` / `(x - 0.5).floor()` pixel snaps cast to `i32`, the span
//! count is a `u32` tally, and the pixel count is a `u32` sum of integer widths,
//! so the parity test asserts an exact `==` on the span count, on every span's
//! `y` / `x_start` / `x_end`, and on the total pixel count. There is no
//! continuous quantity to compare under tolerance: the only `f32` arithmetic is
//! the per-edge crossing `x_lower + (yc - y_lower) * inv_slope`, and the
//! fixtures are chosen so every crossing is a dyadic value well clear of a
//! half-integer pixel boundary, so a legal fused multiply-add perturbation
//! cannot flip the floored integer the span records.
//!
//! # Degenerate inputs
//!
//! A ring with fewer than three vertices, or one whose every edge is horizontal
//! (a collinear horizontal ring), builds an empty Edge Table and yields no
//! spans and a zero pixel count, matching the reference short circuits. A
//! zero-width (collinear vertical) ring builds edges but every per-row crossing
//! coincides, so each run is zero-width and dropped, again yielding no spans. An
//! empty polygon batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized; a ring with more than
//! [`MAX_POLYGON_VERTS`] vertices is clamped to that many on upload, and a
//! polygon whose vertical span exceeds the device scanline bound or that emits
//! more than [`MAX_SPANS`] runs is truncated, so callers must keep rings within
//! the fixed slots the twin exercises.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `floor`, `+ - * /`, one `/` per edge for the inverse slope, and unsigned /
//! signed index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`,
//! no inverse trigonometry, no `smoothstep`, no `round`, no `sqrt` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Every loop is bounded — the Edge Table build and the per-row crossing
//! gather by the vertex count (capped at [`MAX_POLYGON_VERTS`]), the insertion
//! sort by the active-edge count, and the scanline sweep by a fixed device
//! bound — so the kernel provably terminates with no runaway loop.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::scanline_polygon_fill`；无第三方引擎源码或衍生代码。
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

/// Maximum number of vertices one polygon slot holds on device.
///
/// Each polygon is uploaded into a fixed-length `std430` slot of this many
/// `vec2<f32>` lanes, so a ring with more vertices than this is clamped on
/// upload. `16` comfortably covers the convex and concave fixtures the twin
/// exercises while keeping the per-element slot small.
///
/// Provenance: `MAX_POLYGON_VERTS` chosen for this twin; mirrors no reference
/// constant.
pub const MAX_POLYGON_VERTS: usize = 16;

/// Maximum number of fill spans one polygon result slot holds on device.
///
/// Each result reserves a fixed-length `std430` array of this many [`GpuSpan`]
/// runs, so a polygon that would emit more spans is truncated to this many on
/// device. `64` comfortably covers the concave fixtures the twin exercises.
///
/// Provenance: `MAX_SPANS` chosen for this twin; mirrors no reference constant.
pub const MAX_SPANS: usize = 64;

/// The portable core-`WGSL` scanline-fill kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`scanline_polygon_fill`](prism_render_architecture::particle::scanline_polygon_fill)
/// branch for branch; see the module documentation for the algorithm.
const SCANLINE_POLYGON_FILL_WGSL: &str = r#"
// Scanline-fill twin: one thread fills one simple polygon ring and reproduces
// the CPU golden `particle::scanline_polygon_fill` answers branch for branch --
// the ordered list of horizontal interior spans (one `y`, `x_start`, `x_end`
// run per even-odd crossing pair) and the summed interior pixel count.
// `polygons` holds one fixed-length ring per thread; `results` receives one
// record per ring. Non-horizontal edges are entered into a hand-rolled Edge
// Table, integer scanlines are swept at their centres `yc = y + 0.5`, an edge
// is active on a row under the half-open `[y_lower, y_upper)` rule, the per-row
// crossings are insertion-sorted left to right and paired under the even-odd
// rule, and each non-empty pair becomes one span. The kernel uses only the
// portable core-WGSL subset (min/max/abs/floor, + - * / and one / per edge, and
// signed / unsigned index math), needs no transcendental call and no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. Every loop is bound
// by the vertex count, the active-edge count or a fixed scanline bound, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::scanline_polygon_fill；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a y-extent, a slope denominator or a scanline / vertex y
// difference is treated as zero, matching the reference `CMP_EPS`; the compare
// rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// Fixed number of vertex lanes in each polygon slot; mirrors the host
// `MAX_POLYGON_VERTS`.
const MAX_VERTS: u32 = 16u;

// Fixed number of span runs in each result slot; mirrors the host `MAX_SPANS`.
const MAX_SPANS: u32 = 64u;

// Upper bound on the number of integer scanlines one polygon's vertical span may
// cover; the sweep breaks once this many rows have been visited so the kernel
// provably terminates. Comfortably covers the fixtures the twin exercises.
const MAX_SCANLINES: i32 = 256;

struct Params {
    // Number of polygons in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Polygon {
    // Fixed-length ring: the first `vertex_count` lanes are live vertices and the
    // closing edge back to lane 0 is implied, never stored.
    verts: array<vec2<f32>, 16>,
    // Number of live vertices in `verts` (0..=16).
    vertex_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Span {
    // The integer scanline (pixel row) this run lies on.
    y: i32,
    // The leftmost filled pixel column, inclusive.
    x_start: i32,
    // The rightmost filled pixel column, inclusive.
    x_end: i32,
}

struct Result {
    // Number of fill spans emitted (may exceed MAX_SPANS if truncated on store).
    span_count: u32,
    // Total interior pixels covered: sum of every span width.
    filled_pixel_count: u32,
    // The ordered spans; only the first `span_count` lanes are meaningful.
    spans: array<Span, 64>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> polygons: array<Polygon>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    var n = polygons[idx].vertex_count;
    if (n > MAX_VERTS) {
        n = MAX_VERTS;
    }

    // Edge Table: every non-horizontal ring edge, normalized so y_lower <=
    // y_upper. A horizontal edge carries no x-per-y crossing and is skipped. A
    // ring needs three vertices to enclose area, so fewer leaves the table empty.
    var e_y_lower: array<f32, 16>;
    var e_y_upper: array<f32, 16>;
    var e_x_lower: array<f32, 16>;
    var e_inv_slope: array<f32, 16>;
    var edge_count: u32 = 0u;
    if (n >= 3u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let a = polygons[idx].verts[i];
            let b = polygons[idx].verts[(i + 1u) % n];
            let dy = b.y - a.y;
            if (abs(dy) > CMP_EPS) {
                var lower = a;
                var upper = b;
                if (!(a.y < b.y)) {
                    lower = b;
                    upper = a;
                }
                e_y_lower[edge_count] = lower.y;
                e_y_upper[edge_count] = upper.y;
                e_x_lower[edge_count] = lower.x;
                e_inv_slope[edge_count] = (upper.x - lower.x) / (upper.y - lower.y);
                edge_count = edge_count + 1u;
            }
        }
    }

    var span_count: u32 = 0u;
    var filled: u32 = 0u;
    var out: Result;

    if (edge_count > 0u) {
        // Vertical extent of the ring over all live vertices; scanline centres
        // outside it fill nothing.
        var y_min = polygons[idx].verts[0].y;
        var y_max = polygons[idx].verts[0].y;
        for (var i: u32 = 1u; i < n; i = i + 1u) {
            let vy = polygons[idx].verts[i].y;
            y_min = min(y_min, vy);
            y_max = max(y_max, vy);
        }

        // A padded integer scanline range: rows whose centre falls outside the
        // active edges simply produce no crossings and emit nothing.
        let y_start = i32(floor(y_min - 1.0));
        let y_end = i32(floor(y_max + 1.0));

        for (var y: i32 = y_start; y <= y_end; y = y + 1) {
            if (y - y_start >= MAX_SCANLINES) {
                break;
            }
            let yc = f32(y) + 0.5;

            // Gather this row's x crossings from the edges active under the
            // half-open [y_lower, y_upper) rule: an edge opens once its lower end
            // is at or below yc and retires once its upper end is at or below yc,
            // so a shared local-maximum vertex is excluded exactly once.
            var xs: array<f32, 16>;
            var xs_count: u32 = 0u;
            for (var e: u32 = 0u; e < edge_count; e = e + 1u) {
                if (e_y_lower[e] <= yc + CMP_EPS && e_y_upper[e] - yc > CMP_EPS) {
                    xs[xs_count] = e_x_lower[e] + (yc - e_y_lower[e]) * e_inv_slope[e];
                    xs_count = xs_count + 1u;
                }
            }

            // Insertion-sort the crossings left to right (stable, ascending).
            for (var i: u32 = 1u; i < xs_count; i = i + 1u) {
                let key = xs[i];
                var j: i32 = i32(i) - 1;
                loop {
                    if (j < 0) {
                        break;
                    }
                    if (!(xs[u32(j)] > key)) {
                        break;
                    }
                    xs[u32(j + 1)] = xs[u32(j)];
                    j = j - 1;
                }
                xs[u32(j + 1)] = key;
            }

            // Even-odd parity: interior lies between consecutive crossing pairs.
            var pair: u32 = 0u;
            loop {
                if (pair + 1u >= xs_count) {
                    break;
                }
                let x_left = xs[pair];
                let x_right = xs[pair + 1u];
                // A pixel column is filled when its centre lies inside the run.
                let x_start = i32(floor(x_left + 0.5));
                let x_end = i32(floor(x_right - 0.5));
                if (x_start <= x_end) {
                    if (span_count < MAX_SPANS) {
                        out.spans[span_count] = Span(y, x_start, x_end);
                    }
                    span_count = span_count + 1u;
                    filled = filled + u32(x_end - x_start + 1);
                }
                pair = pair + 2u;
            }
        }
    }

    out.span_count = span_count;
    out.filled_pixel_count = filled;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the polygon count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SCANLINE_POLYGON_FILL_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid polygons in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one polygon slot, matching the `WGSL` `Polygon`
/// struct. The `verts` lane array is `32` floats, i.e. [`MAX_POLYGON_VERTS`]
/// `vec2<f32>` lanes laid out as `[x0, y0, x1, y1, ...]`, so its `128` bytes map
/// onto the device `array<vec2<f32>, 16>` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPolygon {
    /// [`MAX_POLYGON_VERTS`] interleaved `vec2<f32>` vertex lanes.
    verts: [f32; 32],
    /// Number of live vertices in `verts`.
    vertex_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One horizontal run of filled interior pixels on a single scanline, mirroring
/// the `CPU` golden
/// [`Span`](prism_render_architecture::particle::scanline_polygon_fill::Span).
///
/// The run covers the inclusive pixel columns `x_start ..= x_end` on row `y`,
/// and is only ever emitted when non-empty, so `x_start <= x_end` always holds.
/// This is both the public span type and the `std430` element matching the
/// `WGSL` `Span` struct, so its `12` bytes map onto the device
/// `array<Span, 64>` byte-for-byte.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::scanline_polygon_fill`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSpan {
    /// The integer scanline (pixel row) this run lies on, matching
    /// [`Span::y`](prism_render_architecture::particle::scanline_polygon_fill::Span::y).
    pub y: i32,
    /// The leftmost filled pixel column, inclusive, matching
    /// [`Span::x_start`](prism_render_architecture::particle::scanline_polygon_fill::Span::x_start).
    pub x_start: i32,
    /// The rightmost filled pixel column, inclusive, matching
    /// [`Span::x_end`](prism_render_architecture::particle::scanline_polygon_fill::Span::x_end).
    pub x_end: i32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Number of fill spans emitted.
    span_count: u32,
    /// Total interior pixels covered: sum of every span width.
    filled_pixel_count: u32,
    /// The ordered spans; only the first `span_count` lanes are meaningful.
    spans: [GpuSpan; MAX_SPANS],
}

/// Every answer the twin reports for a single polygon, mirroring the `CPU`
/// golden
/// [`scanline_polygon_fill`](prism_render_architecture::particle::scanline_polygon_fill).
///
/// `span_count` equals `spans.len()`; it is reported explicitly so a caller can
/// detect a device-side truncation (a polygon that would emit more than
/// [`MAX_SPANS`] runs) by comparing it against the returned span count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuScanlineFill {
    /// Number of fill spans emitted, matching the length of the `CPU` golden
    /// [`scanline_fill`](prism_render_architecture::particle::scanline_polygon_fill::scanline_fill)
    /// vector.
    pub span_count: u32,
    /// The ordered interior fill spans, matching the `CPU` golden
    /// [`scanline_fill`](prism_render_architecture::particle::scanline_polygon_fill::scanline_fill)
    /// vector span for span.
    pub spans: Vec<GpuSpan>,
    /// Total interior pixels covered, matching the `CPU` golden
    /// [`filled_pixel_count`](prism_render_architecture::particle::scanline_polygon_fill::filled_pixel_count).
    pub filled_pixel_count: u32,
}

/// Encodes one open polygon ring into its fixed-length `std430` [`GpuPolygon`]
/// slot, clamping to [`MAX_POLYGON_VERTS`] live vertices.
fn encode_polygon(polygon: &[[f32; 2]]) -> GpuPolygon {
    let count = polygon.len().min(MAX_POLYGON_VERTS);
    let mut verts = [0.0_f32; 32];
    for (i, &p) in polygon.iter().take(count).enumerate() {
        verts[2 * i] = p[0];
        verts[2 * i + 1] = p[1];
    }
    GpuPolygon {
        verts,
        vertex_count: count as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuScanlineFill`], copying
/// out only the live span lanes (clamped to [`MAX_SPANS`] on a truncation).
fn decode_result(raw: &GpuResult) -> GpuScanlineFill {
    let live = (raw.span_count as usize).min(MAX_SPANS);
    GpuScanlineFill {
        span_count: raw.span_count,
        spans: raw.spans[..live].to_vec(),
        filled_pixel_count: raw.filled_pixel_count,
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

/// A compiled, reusable scanline-fill compute pipeline, twinning the `CPU`
/// golden
/// [`scanline_polygon_fill`](prism_render_architecture::particle::scanline_polygon_fill).
pub struct GpuScanlinePolygonFill {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuScanlinePolygonFill {
    /// Compiles the scanline-fill kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuScanlinePolygonFill {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill"),
            source: ShaderSource::Wgsl(SCANLINE_POLYGON_FILL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuScanlinePolygonFill {
            module,
            layout,
            pipeline,
        }
    }

    /// Fills every polygon in `polygons` and returns one [`GpuScanlineFill`] per
    /// input, in order.
    ///
    /// Each ring is passed as an *open* slice of 2D vertices; the closing edge is
    /// implied. The span count, every span's coordinates and the total pixel
    /// count equal the reference exactly for the fixtures the twin exercises. An
    /// empty `polygons` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized. A ring longer than
    /// [`MAX_POLYGON_VERTS`] is clamped to that many vertices on upload, and a
    /// polygon emitting more than [`MAX_SPANS`] runs is truncated on device.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, polygons: &[&[[f32; 2]]]) -> Vec<GpuScanlineFill> {
        let count = polygons.len();
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
            label: Some("prism_volumetric_scanline_polygon_fill_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuPolygon> = polygons.iter().map(|&p| encode_polygon(p)).collect();
        let polygons_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_polygons"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: polygons_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_scanline_polygon_fill_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_scanline_polygon_fill_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per polygon, flattened to a 1-D dispatch.
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
