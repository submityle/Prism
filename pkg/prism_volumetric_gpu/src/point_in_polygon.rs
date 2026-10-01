//! `wgpu` compute twin of the 2D point-in-polygon containment and signed
//! boundary-distance contract
//! ([`point_in_polygon`](prism_render_architecture::particle::point_in_polygon),
//! particle design §8.2, §12-§13).
//!
//! The `CPU` golden
//! [`point_in_polygon`](prism_render_architecture::particle::point_in_polygon)
//! owns the small, verifiable spatial math several particle stages share: the
//! even-odd ray-crossing containment rule
//! ([`point_in_polygon_crossing`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_crossing)),
//! the signed [`winding_number`](prism_render_architecture::particle::point_in_polygon::winding_number)
//! and the non-zero winding containment rule it feeds
//! ([`point_in_polygon_winding`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_winding)),
//! and the signed distance to the polygon boundary
//! ([`signed_distance_to_polygon`](prism_render_architecture::particle::point_in_polygon::signed_distance_to_polygon),
//! built on the point-to-segment primitive
//! [`distance_to_edge`](prism_render_architecture::particle::point_in_polygon::distance_to_edge)).
//! [`GpuPointInPolygon`] is the on-device twin: the polygon vertices live in one
//! shared storage buffer, each thread tests one query point against the whole
//! ring, and one kernel reproduces all three answers so a passing real-device
//! parity test is direct evidence the ported kernel solves the same geometry and
//! classifies the same degenerate cases the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! query points sharing one polygon: the even-odd crossing flag, the integer
//! winding number (and the non-zero winding containment derived from it), and
//! the signed boundary distance (negative inside, positive outside). The
//! even-odd rule and the winding rule disagree on self-intersecting rings — the
//! overlapping core of a star is outside under crossing yet inside under
//! winding — and the twin preserves that disagreement because it mirrors both
//! loops branch for branch.
//!
//! # Correctness model
//!
//! The crossing flag and the winding number are discrete classifications built
//! from `f32` comparisons and the exact [`is_left`] side test, so for inputs
//! clear of the boundary the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on both the containment flag and the winding integer.
//! The signed distance threads through a `sqrt` and a division, so `CPU` and
//! `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the signed
//! distance, tight enough to catch a genuinely wrong port (a dropped edge, a
//! swapped sign, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! A point placed exactly on an edge or on a vertex is a measure-zero boundary
//! case where the ray-crossing division and the side test straddle their
//! comparison thresholds; whether such a point counts as inside is left
//! unspecified, exactly as in the reference, and the parity fixtures keep their
//! containment probes clear of the boundary (the signed distance there is near
//! zero and still compared under tolerance).
//!
//! # Degenerate inputs
//!
//! A polygon with fewer than three vertices encloses no area: the crossing flag
//! is `false` and the winding number is `0`, matching the reference short
//! circuits. A polygon with fewer than two vertices has no edge and its signed
//! distance is [`f32::INFINITY`] (written on-device with a bit-cast, since
//! `WGSL` has no infinity literal). An empty query batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized; an empty
//! polygon is uploaded as a single placeholder vertex the kernel never reads
//! because `vertex_count` is pinned to `0`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `clamp`,
//! `dot`, `sqrt`, `+ - * /`, unsigned index arithmetic and a `bitcast` for the
//! infinity sentinel — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The three per-vertex loops are each bounded by
//! the vertex count, so the kernel provably terminates with no runaway loop.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::point_in_polygon`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` point-in-polygon kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`point_in_polygon`](prism_render_architecture::particle::point_in_polygon)
/// loops branch for branch; see the module documentation for the algorithm.
const POINT_IN_POLYGON_WGSL: &str = r#"
// Point-in-polygon twin: one thread per query point tests a shared polygon ring
// and reproduces the CPU golden `particle::point_in_polygon` answers branch for
// branch -- the even-odd crossing flag, the signed winding number, and the
// signed boundary distance. `polygon` is the shared vertex array; `queries`
// holds one test point per thread; `results` receives one record per query. The
// kernel uses only the portable core-WGSL subset (min/clamp/dot/sqrt, + - * /,
// unsigned index math and a bitcast for the infinity sentinel), needs no
// transcendental call and no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12. Each per-vertex loop is bounded by the vertex count, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::point_in_polygon；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a squared edge length is treated as degenerate, matching
// the reference `CMP_EPS`; the compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// IEEE-754 bit pattern of +inf; WGSL has no infinity literal, so the empty-edge
// signed distance is produced with a bitcast of this constant.
const F32_INF_BITS: u32 = 0x7f800000u;

struct Params {
    // Number of polygon vertices in the shared `polygon` array. Fewer than two
    // means no edge (infinite signed distance); fewer than three means no area
    // (never inside). Zero means an empty polygon (the `polygon` buffer still
    // holds one placeholder vertex that is never read).
    vertex_count: u32,
    // Number of valid query points; threads past this short-circuit.
    query_count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Even-odd crossing containment flag: 1 when inside, 0 otherwise.
    inside: u32,
    // Signed winding number of the polygon around the query point.
    winding: i32,
    // Signed distance to the polygon boundary: negative inside (non-zero
    // winding rule), positive outside, +inf when the polygon has no edge.
    signed_distance: f32,
    // Padding to a 16-byte, 4-byte-aligned result slot.
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> polygon: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Twice the signed area of triangle a, b, p (the 2D cross product
// (b - a) x (p - a)); positive when p is left of the directed edge a -> b,
// matching the reference `is_left`.
fn is_left(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (p.x - a.x) * (b.y - a.y);
}

// Euclidean distance from p to the segment a -> b, with the projection
// parameter clamped to [0, 1]; a degenerate segment (squared length at or below
// CMP_EPS) collapses to its start a. Mirrors the reference `distance_to_edge`.
fn distance_to_edge(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let denom = dot(ab, ab);
    var t: f32 = 0.0;
    if (denom > CMP_EPS) {
        t = clamp(dot(ap, ab) / denom, 0.0, 1.0);
    }
    let closest = a + ab * t;
    let d = p - closest;
    return sqrt(dot(d, d));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let p = queries[idx];
    let n = params.vertex_count;

    // Even-odd ray crossing: count edges a horizontal ray from p crosses, using
    // the previous vertex as in the reference. A ring needs three vertices to
    // enclose area. `&&` short-circuits so the division only runs when the edge
    // straddles p's level and the denominator is non-zero.
    var inside: bool = false;
    if (n >= 3u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let vi = polygon[i];
            let vj = polygon[(i + n - 1u) % n];
            if (((vi.y > p.y) != (vj.y > p.y))
                && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)) {
                inside = !inside;
            }
        }
    }

    // Signed winding number: only edges crossing p's horizontal level
    // contribute, and `is_left` decides each crossing's sign, so no atan2 is
    // needed. Mirrors the reference `winding_number`.
    var wn: i32 = 0;
    if (n >= 3u) {
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let a = polygon[i];
            let b = polygon[(i + 1u) % n];
            if (a.y <= p.y) {
                if (b.y > p.y && is_left(a, b, p) > 0.0) {
                    wn = wn + 1;
                }
            } else if (b.y <= p.y && is_left(a, b, p) < 0.0) {
                wn = wn - 1;
            }
        }
    }

    // Signed boundary distance: nearest-edge magnitude with the non-zero winding
    // rule choosing the sign. A polygon with fewer than two vertices has no edge
    // and returns +inf, matching the reference `signed_distance_to_polygon`.
    var signed_distance: f32 = bitcast<f32>(F32_INF_BITS);
    if (n >= 2u) {
        var min_d: f32 = bitcast<f32>(F32_INF_BITS);
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let a = polygon[i];
            let b = polygon[(i + 1u) % n];
            min_d = min(min_d, distance_to_edge(p, a, b));
        }
        if (wn != 0) {
            signed_distance = -min_d;
        } else {
            signed_distance = min_d;
        }
    }

    var out: Result;
    if (inside) {
        out.inside = 1u;
    } else {
        out.inside = 0u;
    }
    out.winding = wn;
    out.signed_distance = signed_distance;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the shared vertex count and the query
/// count plus two pad words to fill a `16`-byte, `std140`-aligned uniform struct
/// matching `Params` in [`POINT_IN_POLYGON_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of polygon vertices in the shared storage array.
    vertex_count: u32,
    /// Number of valid query points in the input and output buffers.
    query_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result: the containment flag, the winding
/// integer, the signed distance and one pad lane — `16` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Even-odd crossing containment flag (`1` inside, `0` outside).
    inside: u32,
    /// Signed winding number of the polygon around the query point.
    winding: i32,
    /// Signed distance to the polygon boundary.
    signed_distance: f32,
    /// Padding lane.
    pad0: f32,
}

/// One resolved answer for a single query point, mirroring every value the
/// reference reports across its three twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointInPolygonResult {
    /// Even-odd ray-crossing containment, matching
    /// [`point_in_polygon_crossing`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_crossing).
    pub inside_crossing: bool,
    /// Signed winding number, matching
    /// [`winding_number`](prism_render_architecture::particle::point_in_polygon::winding_number).
    /// Non-zero means inside under the winding rule
    /// ([`point_in_polygon_winding`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_winding)).
    pub winding: i32,
    /// Signed distance to the polygon boundary, matching
    /// [`signed_distance_to_polygon`](prism_render_architecture::particle::point_in_polygon::signed_distance_to_polygon).
    pub signed_distance: f32,
}

impl PointInPolygonResult {
    /// Non-zero winding containment, matching
    /// [`point_in_polygon_winding`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_winding).
    #[must_use]
    pub const fn inside_winding(&self) -> bool {
        self.winding != 0
    }
}

/// A compiled, reusable point-in-polygon compute pipeline, twinning the `CPU`
/// golden
/// [`point_in_polygon`](prism_render_architecture::particle::point_in_polygon).
pub struct GpuPointInPolygon {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPointInPolygon {
    /// Compiles the point-in-polygon kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPointInPolygon {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_point_in_polygon"),
            source: ShaderSource::Wgsl(POINT_IN_POLYGON_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_point_in_polygon_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_point_in_polygon_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_point_in_polygon_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPointInPolygon {
            module,
            layout,
            pipeline,
        }
    }

    /// Tests every point in `points` against the shared `polygon` and returns
    /// one [`PointInPolygonResult`] per input, in order.
    ///
    /// The containment flag and winding integer equal the reference exactly for
    /// points clear of the boundary; the signed distance matches to within the
    /// tolerance documented on this module. An empty `points` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized. An empty `polygon` is uploaded as a single placeholder vertex
    /// the kernel never reads, so every point reports outside, winding `0` and
    /// an infinite signed distance.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        polygon: &[[f32; 2]],
        points: &[[f32; 2]],
    ) -> Vec<PointInPolygonResult> {
        let query_count = points.len();
        if query_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            vertex_count: polygon.len() as u32,
            query_count: query_count as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_in_polygon_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        // A storage buffer cannot be zero-sized; for an empty polygon upload one
        // placeholder vertex that the kernel never reads because `vertex_count`
        // is 0.
        let placeholder = [[0.0_f32, 0.0_f32]];
        let polygon_contents: &[[f32; 2]] = if polygon.is_empty() {
            &placeholder
        } else {
            polygon
        };
        let polygon_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_in_polygon_polygon"),
            contents: bytemuck::cast_slice(polygon_contents),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_in_polygon_queries"),
            contents: bytemuck::cast_slice(points),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (query_count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_point_in_polygon_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_point_in_polygon_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: polygon_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_point_in_polygon_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_point_in_polygon_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_point_in_polygon_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query point, flattened to a 1-D dispatch.
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

    /// Even-odd crossing containment for every point, mirroring
    /// [`point_in_polygon_crossing`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_crossing).
    #[must_use]
    pub fn crossing(
        &self,
        ctx: &GpuContext,
        polygon: &[[f32; 2]],
        points: &[[f32; 2]],
    ) -> Vec<bool> {
        self.evaluate(ctx, polygon, points)
            .into_iter()
            .map(|r| r.inside_crossing)
            .collect()
    }

    /// Signed winding number for every point, mirroring
    /// [`winding_number`](prism_render_architecture::particle::point_in_polygon::winding_number).
    #[must_use]
    pub fn winding(&self, ctx: &GpuContext, polygon: &[[f32; 2]], points: &[[f32; 2]]) -> Vec<i32> {
        self.evaluate(ctx, polygon, points)
            .into_iter()
            .map(|r| r.winding)
            .collect()
    }

    /// Non-zero winding containment for every point, mirroring
    /// [`point_in_polygon_winding`](prism_render_architecture::particle::point_in_polygon::point_in_polygon_winding).
    #[must_use]
    pub fn winding_inside(
        &self,
        ctx: &GpuContext,
        polygon: &[[f32; 2]],
        points: &[[f32; 2]],
    ) -> Vec<bool> {
        self.evaluate(ctx, polygon, points)
            .into_iter()
            .map(|r| r.inside_winding())
            .collect()
    }

    /// Signed boundary distance for every point, mirroring
    /// [`signed_distance_to_polygon`](prism_render_architecture::particle::point_in_polygon::signed_distance_to_polygon).
    #[must_use]
    pub fn signed_distance(
        &self,
        ctx: &GpuContext,
        polygon: &[[f32; 2]],
        points: &[[f32; 2]],
    ) -> Vec<f32> {
        self.evaluate(ctx, polygon, points)
            .into_iter()
            .map(|r| r.signed_distance)
            .collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PointInPolygonResult`].
fn decode_result(raw: &GpuResult) -> PointInPolygonResult {
    PointInPolygonResult {
        inside_crossing: raw.inside != 0,
        winding: raw.winding,
        signed_distance: raw.signed_distance,
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
