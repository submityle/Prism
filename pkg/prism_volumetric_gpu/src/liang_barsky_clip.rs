//! `wgpu` compute twin of the `Liang-Barsky` parametric line-segment clipping
//! contract
//! ([`liang_barsky_clip`](prism_render_architecture::particle::liang_barsky_clip),
//! particle design §12-§13).
//!
//! The `CPU` golden
//! [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip) owns
//! the small, verifiable reference the ribbon renderer and trail resampler
//! share: it trims one 2D segment to an axis-aligned window *and* reports the
//! scalar entry/exit parameters `t0`/`t1` along the original span, which the
//! outcode-based [`super::cohen_sutherland_clip`] twin deliberately never
//! exposes. [`GpuLiangBarskyClip`] is the on-device twin: one thread clips one
//! segment against one rectangle, reproducing the reference branch for branch so
//! a passing real-device parity test is direct evidence the ported kernel
//! accepts, rejects and trims the same geometry — and reports the same `t`
//! parameters — the reference does, not merely that it compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference reports is reproduced: the "survives"
//! boolean (the discriminant of
//! [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip)'s
//! `Option`), the two clipped endpoint coordinates kept in the original travel
//! direction, and the entry/exit parameters `t0`/`t1`. The reference's regimes
//! are mirrored branch for branch: a segment fully inside the window (returned
//! unchanged with `t0 = 0`, `t1 = 1`), a segment rejected because the surviving
//! parameter interval is empty, a segment parallel to and outside an edge
//! (rejected on the `p_k == 0`, `q_k < 0` test), a segment partially clipped
//! against one or more window edges, and a degenerate zero-length segment that
//! survives only when its single point is inside.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`
//! and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, `sqrt` or
//! `smoothstep`, no `u64` and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of comparisons (the per-edge
//! parallel/sign tests) and multiplies/adds/divides (the parameter updates), so
//! `CPU` and `GPU` evaluate the same closed form in the same order. The survival
//! boolean is a pure sign/epsilon decision over values conditioned clear of a
//! tie, so it matches exactly and the parity test asserts `==` on it. The
//! clipped `f32` coordinates and the `t` parameters are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few units in the last place, so the parity test
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on them.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`liang_barsky_clip`](prism_render_architecture::particle::liang_barsky_clip);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::liang_barsky_clip::{ClipRect, ClipResult, Segment2};
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

/// Magnitude below which the projected direction `p_k` against an edge is
/// treated as zero (segment parallel to that edge), mirroring
/// [`CLIP_EPS`](prism_render_architecture::particle::liang_barsky_clip::CLIP_EPS).
/// Used instead of `==` on `f32`: a direction component within this band counts
/// as parallel so the division `q_k / p_k` is never evaluated on a near-zero
/// denominator.
///
/// Provenance: twinned from this repository's `liang_barsky_clip`.
pub const CLIP_EPS: f32 = 1.0e-6;

/// The portable core-`WGSL` `Liang-Barsky` clip kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip) branch
/// for branch; see the module documentation for the algorithm.
const LIANG_BARSKY_CLIP_WGSL: &str = r#"
// Liang-Barsky clip twin: one thread clips one 2D segment against one
// axis-aligned rectangle. It mirrors the CPU golden
// `particle::liang_barsky_clip::clip` branch for branch, uses only the portable
// core-WGSL subset (min/max/abs and + - * /), needs no sqrt and no
// transcendental, and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::liang_barsky_clip; no third-party engine source or derived code.

// Magnitude below which the projected direction p_k against an edge is treated
// as zero (segment parallel to that edge), matching the reference `CLIP_EPS`.
// Used instead of an exact == / != on an f32, and it guards the q_k / p_k
// division: the divide only runs when abs(p_k) exceeds this band.
const CLIP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of clip queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Window lower bounds (min.x, min.y), as supplied before normalization.
    rect_min: vec2<f32>,
    // Window upper bounds (max.x, max.y), as supplied before normalization.
    rect_max: vec2<f32>,
    // Segment start endpoint a (parameter t = 0).
    a: vec2<f32>,
    // Segment end endpoint b (parameter t = 1).
    b: vec2<f32>,
}

struct Result {
    // 1 when the clipped segment survives (clip returned Some), else 0.
    hit: u32,
    // Padding word so the following f32 pair keeps a stable std430 offset.
    pad0: u32,
    // Entry parameter t0 along the original segment, in [0, 1].
    t0: f32,
    // Exit parameter t1 along the original segment, in [0, 1].
    t1: f32,
    // Clipped start point, equal to point_at(t0); the raw a when hit == 0.
    p0: vec2<f32>,
    // Clipped end point, equal to point_at(t1); the raw b when hit == 0.
    p1: vec2<f32>,
}

// The surviving sub-segment plus its presence flag, the WGSL analogue of the
// reference `Option<ClipResult>`.
struct ClipOut {
    hit: u32,
    t0: f32,
    t1: f32,
    p0: vec2<f32>,
    p1: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Evaluates the segment at parameter t, i.e. a + t * (b - a). Matches the
// reference `Segment2::point_at`.
fn point_at(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
    return a + t * (b - a);
}

// Clips segment a -> b to the (normalized) rectangular window, returning the
// surviving sub-segment with its entry/exit parameters and presence flag. The
// four edges (left, right, bottom, top) fold into max/min updates of the entry
// and exit parameters. Matches the reference `clip`.
fn clip_segment(rmin: vec2<f32>, rmax: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> ClipOut {
    var out: ClipOut;
    out.hit = 0u;
    out.t0 = 0.0;
    out.t1 = 1.0;
    out.p0 = a;
    out.p1 = b;

    let dx = b.x - a.x;
    let dy = b.y - a.y;

    // Per-edge (p_k, q_k) in the order left, right, bottom, top. The point stays
    // inside edge k exactly when t * p_k <= q_k.
    let p = array<f32, 4>(-dx, dx, -dy, dy);
    let q = array<f32, 4>(
        a.x - rmin.x,
        rmax.x - a.x,
        a.y - rmin.y,
        rmax.y - a.y,
    );

    var t_enter = 0.0;
    var t_exit = 1.0;

    for (var k = 0u; k < 4u; k = k + 1u) {
        let pk = p[k];
        let qk = q[k];
        if (abs(pk) <= CLIP_EPS) {
            // Parallel to this edge: reject only when the start point already
            // sits outside the edge's slab.
            if (qk < 0.0) {
                out.hit = 0u;
                return out;
            }
        } else {
            // abs(pk) > CLIP_EPS guards the division against a near-zero
            // denominator.
            let t = qk / pk;
            if (pk < 0.0) {
                // Crossing inward: tighten the entry parameter.
                t_enter = max(t_enter, t);
            } else {
                // Crossing outward: tighten the exit parameter.
                t_exit = min(t_exit, t);
            }
        }
    }

    if (t_enter > t_exit) {
        out.hit = 0u;
        return out;
    }

    out.hit = 1u;
    out.t0 = t_enter;
    out.t1 = t_exit;
    out.p0 = point_at(a, b, t_enter);
    out.p1 = point_at(a, b, t_exit);
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let query = queries[idx];
    // Normalize the window so min <= max on both axes before clipping, matching
    // the reference `ClipRect::normalized`.
    let rmin = min(query.rect_min, query.rect_max);
    let rmax = max(query.rect_min, query.rect_max);

    let clip = clip_segment(rmin, rmax, query.a, query.b);

    var out: Result;
    out.hit = clip.hit;
    out.pad0 = 0u;
    out.t0 = clip.t0;
    out.t1 = clip.t1;
    out.p0 = clip.p0;
    out.p1 = clip.p1;
    results[idx] = out;
}
"#;

/// One segment-clip query: the axis-aligned clip window `rect` and the segment
/// `seg`, the same inputs the reference
/// [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip)
/// consumes.
///
/// Provenance: twinned from this repository's `liang_barsky_clip`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiangBarskyQuery {
    /// The axis-aligned clip window, taken as supplied; the kernel normalizes it
    /// internally exactly as the `CPU` reference does, so an inverted rectangle
    /// is clipped against its swapped-corner equivalent rather than rejected.
    pub rect: ClipRect,
    /// The 2D segment to clip, from `seg.a` (`t = 0`) to `seg.b` (`t = 1`).
    pub seg: Segment2,
}

impl LiangBarskyQuery {
    /// Builds a query from the clip window and the segment.
    ///
    /// Provenance: twinned from this repository's `liang_barsky_clip`.
    #[must_use]
    pub const fn new(rect: ClipRect, seg: Segment2) -> LiangBarskyQuery {
        LiangBarskyQuery { rect, seg }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip)
/// reports.
///
/// Provenance: twinned from this repository's `liang_barsky_clip`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiangBarskyResult {
    /// Whether the clipped segment survives, matching `clip(..).is_some()`.
    pub hit: bool,
    /// The clipped start point derived from the input `seg.a`; equals the raw
    /// `seg.a` when `hit` is `false`.
    pub p0: [f32; 2],
    /// The clipped end point derived from the input `seg.b`; equals the raw
    /// `seg.b` when `hit` is `false`.
    pub p1: [f32; 2],
    /// The entry parameter along the original segment, in `[0, 1]`; `0` when
    /// `hit` is `false`.
    pub t0: f32,
    /// The exit parameter along the original segment, in `[0, 1]`; `1` when
    /// `hit` is `false`.
    pub t1: f32,
}

impl LiangBarskyResult {
    /// Rebuilds the reference
    /// [`clip`](prism_render_architecture::particle::liang_barsky_clip::clip)
    /// return value: `Some(ClipResult)` when the segment survives, `None`
    /// otherwise.
    ///
    /// Provenance: twinned from this repository's `liang_barsky_clip`.
    #[must_use]
    pub fn clipped(&self) -> Option<ClipResult> {
        if self.hit {
            Some(ClipResult {
                p0: self.p0,
                p1: self.p1,
                t0: self.t0,
                t1: self.t1,
            })
        } else {
            None
        }
    }
}

/// `repr(C)` `std430` image of one packed query: four `vec2<f32>` slots
/// `(rect_min, rect_max, a, b)` — `32` bytes, each `vec2` on its `8`-byte-aligned
/// slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Window lower bounds `(min.x, min.y)`.
    rect_min: [f32; 2],
    /// Window upper bounds `(max.x, max.y)`.
    rect_max: [f32; 2],
    /// Segment start endpoint.
    a: [f32; 2],
    /// Segment end endpoint.
    b: [f32; 2],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &LiangBarskyQuery) -> GpuQuery {
        GpuQuery {
            rect_min: query.rect.min,
            rect_max: query.rect.max,
            a: query.seg.a,
            b: query.seg.b,
        }
    }
}

/// `repr(C)` `std430` image of one result: two `u32`/`f32` words `(hit, pad0)`,
/// the parameter pair `(t0, t1)`, then two `vec2<f32>` endpoint slots
/// `(p0, p1)` — `32` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the clipped segment survives, else `0`.
    hit: u32,
    /// Padding word keeping the following `f32` pair at a stable offset.
    pad0: u32,
    /// Entry parameter `t0`.
    t0: f32,
    /// Exit parameter `t1`.
    t1: f32,
    /// Clipped start point derived from `a`.
    p0: [f32; 2],
    /// Clipped end point derived from `b`.
    p1: [f32; 2],
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

/// A compiled, reusable `Liang-Barsky` clip compute pipeline.
///
/// Provenance: twinned from this repository's `liang_barsky_clip`.
pub struct GpuLiangBarskyClip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLiangBarskyClip {
    /// Compiles the clip kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: twinned from this repository's `liang_barsky_clip`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLiangBarskyClip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip"),
            source: ShaderSource::Wgsl(LIANG_BARSKY_CLIP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLiangBarskyClip {
            module,
            layout,
            pipeline,
        }
    }

    /// Clips every query on-device and returns one [`LiangBarskyResult`] per
    /// input, in order.
    ///
    /// Each result matches the reference answers: the survival boolean exactly,
    /// the clipped coordinates and the `t` parameters to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: twinned from this repository's `liang_barsky_clip`.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[LiangBarskyQuery]) -> Vec<LiangBarskyResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_output"),
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
            label: Some("prism_volumetric_liang_barsky_clip_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_bind_group"),
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
            label: Some("prism_volumetric_liang_barsky_clip_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_liang_barsky_clip_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_liang_barsky_clip_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per segment, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`LiangBarskyResult`].
fn decode_result(raw: &GpuResult) -> LiangBarskyResult {
    LiangBarskyResult {
        hit: raw.hit != 0,
        p0: raw.p0,
        p1: raw.p1,
        t0: raw.t0,
        t1: raw.t1,
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
