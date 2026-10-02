#![forbid(unsafe_code)]
//! `wgpu` compute twin of the 2D `Sutherland-Hodgman` convex-window polygon
//! clip golden
//! ([`sutherland_hodgman_2d`](prism_render_architecture::particle::sutherland_hodgman_2d),
//! particle design §12, §13).
//!
//! The `CPU` golden
//! [`clip_polygon`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon)
//! owns the small, verifiable contract several particle stages share: it crops a
//! 2D subject polygon to the intersection of a convex `CCW` (counter-clockwise)
//! clip window by clipping the subject against one window edge at a time
//! ([`clip_edge`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_edge)).
//! [`GpuSutherlandHodgman2d`] is the on-device twin: one thread clips one
//! `(subject, clip)` pair, so a passing real-device parity test is direct
//! evidence the ported kernel folds the same four in/out cases, the same
//! crossing-parameter interpolation and the same duplicate-vertex collapse the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each pair of fixed-capacity polygons (a subject of up to [`MAX_SUBJECT`]
//! vertices and a convex window of up to [`MAX_CLIP`] vertices) the kernel
//! reproduces the whole reference solve: it walks the current ring once per
//! window edge, classifies each endpoint inside/outside the half-plane to the
//! left of the directed edge via a 2D cross product (a point within
//! [`CMP_EPS`] of the edge line counts as inside), and emits the four cases —
//! inside-to-inside keeps the endpoint, inside-to-outside emits the boundary
//! crossing, outside-to-inside emits the crossing then the endpoint, and
//! outside-to-outside emits nothing. The crossing uses the parameter
//! `t = sc / (sc - sn)` clamped into `0..=1`, with a parallel-edge guard when
//! the denominator's magnitude does not exceed [`CMP_EPS`]. Two fixed-length
//! local arrays ping-pong, feeding each window edge's output into the next.
//! A final pass collapses consecutive and wraparound duplicate vertices and
//! culls a ring that degenerates below three vertices.
//!
//! # Layout
//!
//! Each query uploads the subject and clip rings as fixed `array<vec2<f32>>`
//! slots plus the two valid vertex counts; each result carries the surviving
//! vertex count and a fixed `array<vec2<f32>>` of up to [`MAX_OUT`] ring
//! vertices (unused slots zeroed). Because `Sutherland-Hodgman` clipping of an
//! `n`-vertex ring against one half-plane yields at most `n + 1` vertices and
//! the window contributes at most [`MAX_CLIP`] edges, [`MAX_OUT`] is set to
//! [`MAX_SUBJECT`] `+` [`MAX_CLIP`], the tight bound on both the final and every
//! intermediate ring.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable walk of cross products and linear
//! interpolations, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts an *exact* match
//! on the surviving vertex count and the per-vertex order yet a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the vertex coordinates, with
//! fixtures kept clear of the inside/parallel boundary so the discrete vertex
//! count never flips under a legal perturbation.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp` and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt` or
//! optional device feature, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sutherland_hodgman_2d`；
//! classic `Sutherland-Hodgman` convex-window polygon clip plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon;
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

/// Maximum subject-polygon vertex count carried per query. A quad, triangle or
/// small emitter ring fits a single lane; the host passes the valid count.
///
/// Provenance: fixed-capacity upper bound for the twin of
/// `prism_render_architecture::particle::sutherland_hodgman_2d`.
pub const MAX_SUBJECT: usize = 8;

/// Maximum convex-window vertex count carried per query. The host passes the
/// valid count; trailing slots are ignored.
///
/// Provenance: fixed-capacity upper bound for the twin of
/// `prism_render_architecture::particle::sutherland_hodgman_2d`.
pub const MAX_CLIP: usize = 8;

/// Maximum surviving vertex count a clip can produce, [`MAX_SUBJECT`] `+`
/// [`MAX_CLIP`]: clipping an `n`-vertex ring against one half-plane yields at
/// most `n + 1` vertices and the window contributes at most [`MAX_CLIP`] edges,
/// so this bounds every intermediate and the final ring.
///
/// Provenance: fixed-capacity upper bound for the twin of
/// `prism_render_architecture::particle::sutherland_hodgman_2d`.
pub const MAX_OUT: usize = MAX_SUBJECT + MAX_CLIP;

/// The portable core-`WGSL` `Sutherland-Hodgman` convex-window clip kernel,
/// embedded inline so the twin ships as a single source file. The single entry
/// point `solve` mirrors the `CPU` golden
/// [`clip_polygon`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon)
/// edge for edge; see the module documentation for the algorithm.
const SUTHERLAND_HODGMAN_2D_WGSL: &str = r#"
// Sutherland-Hodgman convex-window polygon clip twin: one thread clips one
// (subject, clip) pair. For each directed window edge the kernel walks the
// current ring once, classifies each endpoint inside/outside the half-plane to
// its left and emits the four in/out cases, feeding each edge's output into the
// next through two fixed-length local arrays. A final pass collapses duplicate
// vertices and culls a ring that falls below three vertices. It mirrors the CPU
// golden `particle::sutherland_hodgman_2d` edge for edge, uses only the portable
// core-WGSL subset (abs/min/max/clamp, + - * /), needs no transcendental call
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. Each thread sweeps at most MAX_CLIP edges over a ring of at most MAX_OUT
// vertices, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::sutherland_hodgman_2d；无第三方
// 引擎源码或衍生代码。

const MAX_SUBJECT: u32 = 8u;
const MAX_CLIP: u32 = 8u;
const MAX_OUT: u32 = 16u;

// Magnitude below which a cross product or a coordinate difference is treated as
// zero, matching the reference `CMP_EPS`: a point whose signed side value does
// not fall below -CMP_EPS counts as inside, and a crossing denominator whose
// magnitude does not exceed CMP_EPS marks a parallel edge. The compare rule used
// instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of valid queries; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Subject-polygon vertices; only the first `subject_count` are read.
    subject: array<vec2<f32>, 8>,
    // Convex-window vertices, CCW; only the first `clip_count` are read.
    clip: array<vec2<f32>, 8>,
    // Valid subject vertex count.
    subject_count: u32,
    // Valid clip-window vertex count.
    clip_count: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Surviving ring vertex count (0 on an empty clip).
    out_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Surviving ring vertices in order; unused slots are zeroed.
    verts: array<vec2<f32>, 16>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Signed side of point `p` relative to the directed window edge `a -> b`: the 2D
// cross product (b - a) x (p - a). Positive to the left (inside a CCW window),
// negative to the right, near zero on the edge line. Mirrors the reference
// `edge_side`.
fn edge_side(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let apx = p.x - a.x;
    let apy = p.y - a.y;
    return abx * apy - aby * apx;
}

// Linear interpolation a + (b - a) * t between two 2D points; mirrors `lerp2`.
fn lerp2(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
    return vec2<f32>(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
}

// True when two points coincide within CMP_EPS on both axes; mirrors
// `points_equal`.
fn points_equal(a: vec2<f32>, b: vec2<f32>) -> bool {
    return abs(a.x - b.x) <= CMP_EPS && abs(a.y - b.y) <= CMP_EPS;
}

// Crossing parameter where a subject edge with start/end side values sc, sn
// meets the window edge line. `valid` is false when the edge is parallel (the
// denominator magnitude does not exceed CMP_EPS); otherwise t = sc / (sc - sn)
// clamped into 0..=1. Mirrors the reference `crossing_param`.
struct Cross {
    valid: bool,
    t: f32,
}

fn crossing_param(sc: f32, sn: f32) -> Cross {
    var c: Cross;
    let denom = sc - sn;
    if (abs(denom) <= CMP_EPS) {
        c.valid = false;
        c.t = 0.0;
        return c;
    }
    c.valid = true;
    c.t = clamp(sc / denom, 0.0, 1.0);
    return c;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    var res: Result;
    res.out_count = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;
    for (var i = 0u; i < MAX_OUT; i = i + 1u) {
        res.verts[i] = vec2<f32>(0.0, 0.0);
    }

    let ns = queries[idx].subject_count;
    let nc = queries[idx].clip_count;

    // A polygon with fewer than three vertices has no face; the reference
    // returns empty for either degenerate input.
    if (ns < 3u || nc < 3u) {
        results[idx] = res;
        return;
    }

    // Two fixed-length rings ping-pong: `poly` is the current ring, `tmp`
    // receives one window edge's output before it is copied back.
    var poly: array<vec2<f32>, 16>;
    var tmp: array<vec2<f32>, 16>;
    var poly_n = ns;
    for (var i = 0u; i < ns; i = i + 1u) {
        poly[i] = queries[idx].subject[i];
    }

    for (var e = 0u; e < nc; e = e + 1u) {
        // Matches the reference `if out.is_empty() break` before each edge.
        if (poly_n == 0u) {
            break;
        }
        // `clip_edge` returns empty for a ring of fewer than two vertices.
        if (poly_n < 2u) {
            poly_n = 0u;
            break;
        }

        let a = queries[idx].clip[e];
        let b = queries[idx].clip[(e + 1u) % nc];

        var tmp_n = 0u;
        for (var i = 0u; i < poly_n; i = i + 1u) {
            let cur = poly[i];
            let nxt = poly[(i + 1u) % poly_n];
            let sc = edge_side(a, b, cur);
            let sn = edge_side(a, b, nxt);
            let cur_in = sc >= -CMP_EPS;
            let nxt_in = sn >= -CMP_EPS;
            if (cur_in && nxt_in) {
                tmp[tmp_n] = nxt;
                tmp_n = tmp_n + 1u;
            } else if (cur_in) {
                let cr = crossing_param(sc, sn);
                if (cr.valid) {
                    tmp[tmp_n] = lerp2(cur, nxt, cr.t);
                    tmp_n = tmp_n + 1u;
                }
            } else if (nxt_in) {
                let cr = crossing_param(sc, sn);
                if (cr.valid) {
                    tmp[tmp_n] = lerp2(cur, nxt, cr.t);
                    tmp_n = tmp_n + 1u;
                }
                tmp[tmp_n] = nxt;
                tmp_n = tmp_n + 1u;
            }
        }

        for (var i = 0u; i < tmp_n; i = i + 1u) {
            poly[i] = tmp[i];
        }
        poly_n = tmp_n;
    }

    // dedup_ring: collapse consecutive and wraparound duplicates, mirroring the
    // reference. Rings of fewer than two vertices pass through unchanged.
    var ded: array<vec2<f32>, 16>;
    var ded_n = 0u;
    if (poly_n < 2u) {
        for (var i = 0u; i < poly_n; i = i + 1u) {
            ded[i] = poly[i];
        }
        ded_n = poly_n;
    } else {
        for (var i = 0u; i < poly_n; i = i + 1u) {
            let p = poly[i];
            if (ded_n > 0u && points_equal(ded[ded_n - 1u], p)) {
                continue;
            }
            ded[ded_n] = p;
            ded_n = ded_n + 1u;
        }
        if (ded_n >= 2u && points_equal(ded[0], ded[ded_n - 1u])) {
            ded_n = ded_n - 1u;
        }
    }

    // Cull a ring that degenerates below three vertices (a sliver or point).
    if (ded_n < 3u) {
        results[idx] = res;
        return;
    }

    res.out_count = ded_n;
    for (var i = 0u; i < ded_n; i = i + 1u) {
        res.verts[i] = ded[i];
    }
    results[idx] = res;
}
"#;

/// One clip query: a subject polygon and the convex `CCW` window to crop it to.
///
/// Only the first [`MAX_SUBJECT`] / [`MAX_CLIP`] vertices are read on device;
/// the host passes the valid counts through the [`Vec`] lengths. Holds `f32`
/// geometry, so it derives only [`PartialEq`] (no `Eq`/`Hash`).
///
/// Provenance: twin query of
/// `prism_render_architecture::particle::sutherland_hodgman_2d`.
#[derive(Clone, Debug, PartialEq)]
pub struct SutherlandHodgman2dQuery {
    /// The subject polygon ring (convex or concave); at most [`MAX_SUBJECT`]
    /// vertices.
    pub subject: Vec<[f32; 2]>,
    /// The convex `CCW` clip window; at most [`MAX_CLIP`] vertices.
    pub clip: Vec<[f32; 2]>,
}

/// The resolved clip for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `verts` is the surviving vertex ring in order (empty when either polygon has
/// fewer than three vertices, when the subject lies wholly outside the window,
/// or when the clip degenerates the ring below three vertices). Holds `f32`
/// geometry, so it derives only [`PartialEq`] (no `Eq`/`Hash`).
///
/// Provenance: twin result of
/// `prism_render_architecture::particle::sutherland_hodgman_2d`.
#[derive(Clone, Debug, PartialEq)]
pub struct SutherlandHodgman2dResult {
    /// The surviving clipped ring in order; empty on a fully-removed subject.
    pub verts: Vec<[f32; 2]>,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`SUTHERLAND_HODGMAN_2D_WGSL`]: the query count and three pad
/// words, `16` bytes.
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

/// One query as uploaded. `repr(C)` `std430` layout matching `Query` in the
/// shader: the subject and clip rings as fixed `vec2` slots, the two valid
/// counts and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Subject vertices; only the first `subject_count` are read.
    subject: [[f32; 2]; MAX_SUBJECT],
    /// Clip-window vertices; only the first `clip_count` are read.
    clip: [[f32; 2]; MAX_CLIP],
    /// Valid subject vertex count.
    subject_count: u32,
    /// Valid clip-window vertex count.
    clip_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One result as read back. `repr(C)` `std430` layout matching `Result` in the
/// shader: the surviving vertex count, three pad words and the fixed `vec2` ring
/// slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surviving ring vertex count.
    out_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Surviving ring vertices in order; unused slots are zeroed.
    verts: [[f32; 2]; MAX_OUT],
}

/// Encodes one [`SutherlandHodgman2dQuery`] into its `std430` [`GpuQuery`] slot,
/// zero-filling trailing vertex slots and clamping the counts to the fixed
/// capacities.
fn encode_query(q: &SutherlandHodgman2dQuery) -> GpuQuery {
    let mut subject = [[0.0f32; 2]; MAX_SUBJECT];
    let mut clip = [[0.0f32; 2]; MAX_CLIP];
    let ns = q.subject.len().min(MAX_SUBJECT);
    let nc = q.clip.len().min(MAX_CLIP);
    subject[..ns].copy_from_slice(&q.subject[..ns]);
    clip[..nc].copy_from_slice(&q.clip[..nc]);
    GpuQuery {
        subject,
        clip,
        subject_count: ns as u32,
        clip_count: nc as u32,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SutherlandHodgman2dResult`], copying only the `out_count` live vertices.
fn decode_result(raw: &GpuResult) -> SutherlandHodgman2dResult {
    let n = (raw.out_count as usize).min(MAX_OUT);
    let verts = raw.verts[..n].to_vec();
    SutherlandHodgman2dResult { verts }
}

/// The `CPU` golden clip for one query, dispatching to the reference
/// [`clip_polygon`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon)
/// so callers (and the parity test) can pin the twin vertex for vertex.
#[must_use]
pub fn cpu_reference(query: &SutherlandHodgman2dQuery) -> Vec<[f32; 2]> {
    clip_polygon(&query.subject, &query.clip)
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

/// A compiled, reusable 2D `Sutherland-Hodgman` convex-window clip pipeline,
/// twinning the `CPU` golden
/// [`clip_polygon`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon).
pub struct GpuSutherlandHodgman2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSutherlandHodgman2d {
    /// Compiles the clip kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSutherlandHodgman2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d"),
            source: ShaderSource::Wgsl(SUTHERLAND_HODGMAN_2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSutherlandHodgman2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Clips every query in `queries`, returning one
    /// [`SutherlandHodgman2dResult`] per query in input order.
    ///
    /// The returned ring for query `q` mirrors the reference
    /// [`clip_polygon`](prism_render_architecture::particle::sutherland_hodgman_2d::clip_polygon)
    /// evaluated on `q.subject` and `q.clip`. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return before any dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[SutherlandHodgman2dQuery],
    ) -> Vec<SutherlandHodgman2dResult> {
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
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sutherland_hodgman_2d_bind_group"),
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
            label: Some("prism_volumetric_sutherland_hodgman_2d_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sutherland_hodgman_2d_pass"),
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
