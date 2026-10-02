//! `wgpu` compute twin of the 2D simple-polygon *metrics* contract
//! ([`polygon_area_2d`](prism_render_architecture::particle::polygon_area_2d),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`polygon_area_2d`](prism_render_architecture::particle::polygon_area_2d)
//! owns the small, verifiable scalar facts several particle stages read off an
//! ordered ring of 2D vertices: the signed shoelace
//! [`signed_area`](prism_render_architecture::particle::polygon_area_2d::signed_area),
//! its absolute
//! [`area`](prism_render_architecture::particle::polygon_area_2d::area), the
//! closed
//! [`perimeter`](prism_render_architecture::particle::polygon_area_2d::perimeter),
//! the area-weighted
//! [`centroid`](prism_render_architecture::particle::polygon_area_2d::centroid),
//! the winding sense
//! ([`winding_is_ccw`](prism_render_architecture::particle::polygon_area_2d::winding_is_ccw)),
//! the convexity predicate
//! ([`is_convex`](prism_render_architecture::particle::polygon_area_2d::is_convex)),
//! and the axis-aligned
//! [`bounding_box`](prism_render_architecture::particle::polygon_area_2d::bounding_box).
//! [`GpuPolygonArea2d`] is the on-device twin: one thread measures one polygon,
//! each ring living in its own fixed-length vertex slot, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same metrics and classifies the same degenerate cases the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-ring metric the reference computes is reproduced for a batch of
//! independent polygons, each packed into one [`GpuPolygonMetrics`] record: the
//! signed and absolute shoelace area, the closed perimeter, the area-weighted
//! centroid (with the reference's degenerate vertex-mean fallback), the `CCW`
//! winding flag, the convexity flag, and the axis-aligned bounding box. The ring
//! is *open*: the closing edge from the last vertex back to the first is implied
//! and never repeated, exactly as in the reference. The per-edge outward normals
//! [`edge_normals_outward`](prism_render_architecture::particle::polygon_area_2d::edge_normals_outward)
//! are a `Vec`-valued, per-edge contract rather than a per-ring scalar reduction
//! and are left to a dedicated twin; this kernel mirrors only the seven scalar /
//! vector metrics above.
//!
//! # Correctness model
//!
//! The winding and convexity flags are discrete classifications built from `f32`
//! magnitude comparisons against [`CMP_EPS`], so for rings clear of the
//! degeneracy threshold the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on both flags. The area, perimeter, centroid and
//! bounding box thread through multiplies, adds, a `sqrt` and one guarded
//! division, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous quantity,
//! tight enough to catch a genuinely wrong port (a dropped term, a swapped
//! coordinate, a wrong fallback) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A ring with fewer than three vertices encloses no area, so its signed area is
//! `0.0`, its winding and convexity flags are both `0`, matching the reference
//! short circuits. The centroid of a degenerate (near-zero signed area within
//! [`CMP_EPS`]) ring falls back to the plain vertex mean rather than dividing by
//! six times a near-zero area; an empty ring yields the origin and a single
//! vertex yields itself, exactly as the reference does. An empty bounding box is
//! the origin pair. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized; a ring with more than
//! [`MAX_POLYGON_VERTS`] vertices is clamped to that many on upload, so callers
//! must keep rings within the fixed slot.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Each
//! per-vertex loop is bounded by the ring's vertex count, itself capped at
//! [`MAX_POLYGON_VERTS`], so the kernel provably terminates with no runaway loop.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::polygon_area_2d`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` polygon-metrics kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`polygon_area_2d`](prism_render_architecture::particle::polygon_area_2d)
/// branch for branch; see the module documentation for the algorithm.
const POLYGON_AREA_2D_WGSL: &str = r#"
// Polygon-metrics twin: one thread measures one polygon ring and reproduces the
// CPU golden `particle::polygon_area_2d` answers branch for branch -- the signed
// and absolute shoelace area, the closed perimeter, the area-weighted centroid
// (with the degenerate vertex-mean fallback), the CCW winding flag, the
// convexity flag, and the axis-aligned bounding box. `polygons` holds one
// fixed-length ring per thread; `results` receives one record per ring. The
// kernel uses only the portable core-WGSL subset (min/max/abs/sqrt, + - * /, and
// unsigned index math), needs no transcendental call and no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12. Each per-vertex loop is bounded
// by the vertex count, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::polygon_area_2d；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a signed area is treated as zero, matching the reference
// `CMP_EPS`; the compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// Fixed number of vertex lanes in each polygon slot; mirrors the host
// `MAX_POLYGON_VERTS`.
const MAX_VERTS: u32 = 16u;

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

struct Result {
    // Signed shoelace area: positive for a CCW ring, negative for CW, 0 when
    // fewer than three vertices.
    signed_area: f32,
    // Absolute (unsigned) area, i.e. abs(signed_area).
    area: f32,
    // Closed perimeter: summed edge length including the implied closing edge.
    perimeter: f32,
    // CCW winding flag: 1 when signed_area > CMP_EPS, 0 otherwise.
    winding_ccw: u32,
    // Area-weighted centroid, with the degenerate vertex-mean fallback.
    centroid: vec2<f32>,
    // Axis-aligned bounding box minimum corner.
    bbox_min: vec2<f32>,
    // Axis-aligned bounding box maximum corner.
    bbox_max: vec2<f32>,
    // Convexity flag: 1 when every turn shares one sign, 0 otherwise.
    is_convex: u32,
    // Padding to a 16-byte, 8-byte-aligned result slot.
    pad0: u32,
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

    // Signed shoelace area: a ring needs three vertices to enclose area, so the
    // accumulator stays zero (and the signed area with it) for fewer.
    var acc: f32 = 0.0;
    if (n >= 3u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let p = polygons[idx].verts[i];
            let q = polygons[idx].verts[(i + 1u) % n];
            acc = acc + (p.x * q.y - q.x * p.y);
        }
    }
    let signed = 0.5 * acc;
    let abs_area = abs(signed);

    // Closed perimeter: fewer than two vertices bound no edge.
    var perim: f32 = 0.0;
    if (n >= 2u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let p = polygons[idx].verts[i];
            let q = polygons[idx].verts[(i + 1u) % n];
            let d = q - p;
            perim = perim + sqrt(d.x * d.x + d.y * d.y);
        }
    }

    // CCW winding: strictly positive signed area within the degeneracy band.
    var winding_ccw: u32 = 0u;
    if (signed > CMP_EPS) {
        winding_ccw = 1u;
    }

    // Area-weighted centroid, mirroring the reference fallbacks: an empty ring
    // is the origin, a single vertex is itself, and a degenerate (near-zero
    // area) ring falls back to the plain vertex mean rather than dividing by six
    // times a near-zero area.
    var centroid: vec2<f32> = vec2<f32>(0.0, 0.0);
    if (n == 1u) {
        centroid = polygons[idx].verts[0];
    } else if (n >= 2u) {
        if (abs_area <= CMP_EPS) {
            var sum: vec2<f32> = vec2<f32>(0.0, 0.0);
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                sum = sum + polygons[idx].verts[i];
            }
            centroid = sum / f32(n);
        } else {
            var cx: f32 = 0.0;
            var cy: f32 = 0.0;
            for (var i: u32 = 0u; i < n; i = i + 1u) {
                let p = polygons[idx].verts[i];
                let q = polygons[idx].verts[(i + 1u) % n];
                let w = p.x * q.y - q.x * p.y;
                cx = cx + (p.x + q.x) * w;
                cy = cy + (p.y + q.y) * w;
            }
            let denom = 6.0 * signed;
            centroid = vec2<f32>(cx / denom, cy / denom);
        }
    }

    // Convexity: every turn between consecutive edges shares one orientation;
    // collinear turns within CMP_EPS are skipped. Fewer than three vertices do
    // not form a polygon. A reflex turn flips `ok` false, which never resets.
    var is_convex: u32 = 0u;
    if (n >= 3u) {
        var sign: i32 = 0;
        var ok: bool = true;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let prev = polygons[idx].verts[(i + n - 1u) % n];
            let cur = polygons[idx].verts[i];
            let next = polygons[idx].verts[(i + 1u) % n];
            let a = cur - prev;
            let b = next - cur;
            let turn = a.x * b.y - a.y * b.x;
            if (turn > CMP_EPS) {
                if (sign < 0) {
                    ok = false;
                }
                sign = 1;
            } else if (turn < -CMP_EPS) {
                if (sign > 0) {
                    ok = false;
                }
                sign = -1;
            }
        }
        if (ok) {
            is_convex = 1u;
        }
    }

    // Axis-aligned bounding box: an empty ring has no extent and yields the
    // origin pair; otherwise reduce componentwise over the live vertices.
    var bbox_min: vec2<f32> = vec2<f32>(0.0, 0.0);
    var bbox_max: vec2<f32> = vec2<f32>(0.0, 0.0);
    if (n >= 1u) {
        bbox_min = polygons[idx].verts[0];
        bbox_max = polygons[idx].verts[0];
        for (var i: u32 = 1u; i < n; i = i + 1u) {
            let p = polygons[idx].verts[i];
            bbox_min = min(bbox_min, p);
            bbox_max = max(bbox_max, p);
        }
    }

    var out: Result;
    out.signed_area = signed;
    out.area = abs_area;
    out.perimeter = perim;
    out.winding_ccw = winding_ccw;
    out.centroid = centroid;
    out.bbox_min = bbox_min;
    out.bbox_max = bbox_max;
    out.is_convex = is_convex;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the polygon count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`POLYGON_AREA_2D_WGSL`].
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
/// struct. The `verts` lane array is `32` floats, i.e. `MAX_POLYGON_VERTS`
/// `vec2<f32>` lanes laid out as `[x0, y0, x1, y1, ...]`, so its `128` bytes map
/// onto the device `array<vec2<f32>, 16>` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPolygon {
    /// `MAX_POLYGON_VERTS` interleaved `vec2<f32>` vertex lanes.
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

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed shoelace area.
    signed_area: f32,
    /// Absolute (unsigned) area.
    area: f32,
    /// Closed perimeter.
    perimeter: f32,
    /// `CCW` winding flag (`1` when `CCW`, `0` otherwise).
    winding_ccw: u32,
    /// Area-weighted centroid.
    centroid: [f32; 2],
    /// Bounding-box minimum corner.
    bbox_min: [f32; 2],
    /// Bounding-box maximum corner.
    bbox_max: [f32; 2],
    /// Convexity flag (`1` when convex, `0` otherwise).
    is_convex: u32,
    /// Padding lane.
    pad0: u32,
}

/// Every per-ring metric the twin reports for a single polygon, mirroring the
/// `CPU` golden
/// [`polygon_area_2d`](prism_render_architecture::particle::polygon_area_2d).
///
/// The two orientation predicates are returned as `u32` flags (`0` or `1`) so
/// the on-device bool maps back without an `f32` comparison:
/// [`GpuPolygonMetrics::winding_ccw`] mirrors
/// [`winding_is_ccw`](prism_render_architecture::particle::polygon_area_2d::winding_is_ccw)
/// and [`GpuPolygonMetrics::is_convex`] mirrors
/// [`is_convex`](prism_render_architecture::particle::polygon_area_2d::is_convex).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPolygonMetrics {
    /// Signed shoelace area, matching
    /// [`signed_area`](prism_render_architecture::particle::polygon_area_2d::signed_area).
    pub signed_area: f32,
    /// Absolute area, matching
    /// [`area`](prism_render_architecture::particle::polygon_area_2d::area).
    pub area: f32,
    /// Closed perimeter, matching
    /// [`perimeter`](prism_render_architecture::particle::polygon_area_2d::perimeter).
    pub perimeter: f32,
    /// Area-weighted centroid, matching
    /// [`centroid`](prism_render_architecture::particle::polygon_area_2d::centroid).
    pub centroid: [f32; 2],
    /// Bounding-box minimum corner, matching the first tuple element of
    /// [`bounding_box`](prism_render_architecture::particle::polygon_area_2d::bounding_box).
    pub bbox_min: [f32; 2],
    /// Bounding-box maximum corner, matching the second tuple element of
    /// [`bounding_box`](prism_render_architecture::particle::polygon_area_2d::bounding_box).
    pub bbox_max: [f32; 2],
    /// `CCW` winding flag (`1` when `CCW`, `0` otherwise).
    pub winding_ccw: u32,
    /// Convexity flag (`1` when convex, `0` otherwise).
    pub is_convex: u32,
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

/// Decodes one packed [`GpuResult`] into the public [`GpuPolygonMetrics`].
fn decode_result(raw: &GpuResult) -> GpuPolygonMetrics {
    GpuPolygonMetrics {
        signed_area: raw.signed_area,
        area: raw.area,
        perimeter: raw.perimeter,
        centroid: raw.centroid,
        bbox_min: raw.bbox_min,
        bbox_max: raw.bbox_max,
        winding_ccw: raw.winding_ccw,
        is_convex: raw.is_convex,
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

/// A compiled, reusable polygon-metrics compute pipeline, twinning the `CPU`
/// golden
/// [`polygon_area_2d`](prism_render_architecture::particle::polygon_area_2d).
pub struct GpuPolygonArea2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPolygonArea2d {
    /// Compiles the polygon-metrics kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPolygonArea2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_polygon_area_2d"),
            source: ShaderSource::Wgsl(POLYGON_AREA_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPolygonArea2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Measures every polygon in `polygons` and returns one [`GpuPolygonMetrics`]
    /// per input, in order.
    ///
    /// Each ring is passed as an *open* slice of 2D vertices; the closing edge is
    /// implied. The winding and convexity flags equal the reference exactly for
    /// rings clear of the degeneracy threshold; the area, perimeter, centroid and
    /// bounding box match to within the tolerance documented on this module. An
    /// empty `polygons` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized. A ring longer than
    /// [`MAX_POLYGON_VERTS`] is clamped to that many vertices on upload.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, polygons: &[&[[f32; 2]]]) -> Vec<GpuPolygonMetrics> {
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
            label: Some("prism_volumetric_polygon_area_2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuPolygon> = polygons.iter().map(|&p| encode_polygon(p)).collect();
        let polygons_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_polygons"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_bind_group"),
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
            label: Some("prism_volumetric_polygon_area_2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_polygon_area_2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_polygon_area_2d_pass"),
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
