//! `wgpu` compute twin of the `Cohen-Sutherland` line-segment clipping contract
//! ([`cohen_sutherland_clip`](prism_render_architecture::particle::cohen_sutherland_clip),
//! particle design §12-§13).
//!
//! The `CPU` golden
//! [`cohen_sutherland_clip`](prism_render_architecture::particle::cohen_sutherland_clip)
//! owns the small, verifiable reference the trail-ribbon clipper, the 2D
//! broadphase and the authoring overlay all share: the 4-bit *outcode* predicate
//! ([`outcode`](prism_render_architecture::particle::cohen_sutherland_clip::ClipRect::outcode))
//! and the segment-against-rectangle clip
//! ([`clip_segment`](prism_render_architecture::particle::cohen_sutherland_clip::clip_segment)).
//! [`GpuCohenSutherlandClip`] is the on-device twin: one thread clips one 2D
//! segment against one axis-aligned rectangle, reproducing the reference branch
//! for branch, so a passing real-device parity test is direct evidence the
//! ported kernel accepts, rejects and trims the same geometry — and the same
//! degenerate cases — the reference does, not merely that it compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference reports is reproduced: the "survives"
//! boolean (the discriminant of `clip_segment`'s `Option`), the two clipped
//! endpoint coordinates kept in the original travel direction, and the raw
//! [`outcode`](prism_render_architecture::particle::cohen_sutherland_clip::ClipRect::outcode)
//! of each input endpoint as an exact classification code. The reference's
//! regimes are mirrored branch for branch: a segment fully inside the window
//! (returned unchanged), a segment trivially rejected because both endpoints
//! share an outside half-plane, a segment partially clipped against one, two or
//! three window edges, and a degenerate zero-length segment that survives only
//! when its single point is inside.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, bitwise
//! `and`/`or`, unsigned comparison and `+ - * /` — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, `sqrt` or `smoothstep`, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of comparisons (the
//! outcodes) and multiplies/adds/divides (the edge crossings), so `CPU` and
//! `GPU` evaluate the same closed form in the same order. The survival boolean
//! and both outcodes are pure sign/epsilon decisions over values conditioned
//! clear of a tie, so they match exactly and the parity test asserts `==` on
//! them. The clipped `f32` coordinates are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, so the parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on them.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`cohen_sutherland_clip`](prism_render_architecture::particle::cohen_sutherland_clip);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::cohen_sutherland_clip::ClipRect;
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

/// Outcode value for a point inside every window half-plane (no bits set),
/// mirroring
/// [`OUTCODE_INSIDE`](prism_render_architecture::particle::cohen_sutherland_clip::OUTCODE_INSIDE).
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
pub const OUTCODE_INSIDE: u32 = 0;

/// Outcode bit set when a point lies to the left of the window (`x < xmin`),
/// mirroring
/// [`OUTCODE_LEFT`](prism_render_architecture::particle::cohen_sutherland_clip::OUTCODE_LEFT).
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
pub const OUTCODE_LEFT: u32 = 0b0001;

/// Outcode bit set when a point lies to the right of the window (`x > xmax`),
/// mirroring
/// [`OUTCODE_RIGHT`](prism_render_architecture::particle::cohen_sutherland_clip::OUTCODE_RIGHT).
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
pub const OUTCODE_RIGHT: u32 = 0b0010;

/// Outcode bit set when a point lies below the window (`y < ymin`), mirroring
/// [`OUTCODE_BOTTOM`](prism_render_architecture::particle::cohen_sutherland_clip::OUTCODE_BOTTOM).
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
pub const OUTCODE_BOTTOM: u32 = 0b0100;

/// Outcode bit set when a point lies above the window (`y > ymax`), mirroring
/// [`OUTCODE_TOP`](prism_render_architecture::particle::cohen_sutherland_clip::OUTCODE_TOP).
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
pub const OUTCODE_TOP: u32 = 0b1000;

/// The portable core-`WGSL` `Cohen-Sutherland` clip kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`cohen_sutherland_clip`](prism_render_architecture::particle::cohen_sutherland_clip)
/// branch for branch; see the module documentation for the algorithm.
const COHEN_SUTHERLAND_CLIP_WGSL: &str = r#"
// Cohen-Sutherland clip twin: one thread clips one 2D segment against one
// axis-aligned rectangle. It mirrors the CPU golden
// `particle::cohen_sutherland_clip` branch for branch, uses only the portable
// core-WGSL subset (min/max, bitwise and/or, unsigned compare and + - * /),
// needs no sqrt and no transcendental, and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::cohen_sutherland_clip; no third-party engine source or derived code.

// Magnitude below which a coordinate difference against a window edge is
// treated as zero, matching the reference `CMP_EPS`: a point within this band
// of an edge counts as inside that edge rather than outside it. Used instead of
// an exact == / != on an f32.
const CMP_EPS: f32 = 1.0e-6;

const OUTCODE_INSIDE: u32 = 0u;
const OUTCODE_LEFT: u32 = 1u;
const OUTCODE_RIGHT: u32 = 2u;
const OUTCODE_BOTTOM: u32 = 4u;
const OUTCODE_TOP: u32 = 8u;

struct Params {
    // Number of clip queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Window lower bounds (xmin, ymin).
    rect_min: vec2<f32>,
    // Window upper bounds (xmax, ymax).
    rect_max: vec2<f32>,
    // Segment start endpoint a.
    a: vec2<f32>,
    // Segment end endpoint b.
    b: vec2<f32>,
}

struct Result {
    // 1 when the clipped segment survives (clip_segment returned Some), else 0.
    hit: u32,
    // Raw Cohen-Sutherland outcode of the input endpoint a.
    outcode_a: u32,
    // Raw Cohen-Sutherland outcode of the input endpoint b.
    outcode_b: u32,
    // Padding word so the following vec2 lands on its 8-byte-aligned slot.
    pad0: u32,
    // Clipped endpoint derived from a; the raw a when hit == 0.
    clip_a: vec2<f32>,
    // Clipped endpoint derived from b; the raw b when hit == 0.
    clip_b: vec2<f32>,
}

// The surviving sub-segment plus its presence flag, the WGSL analogue of the
// reference `Option<([f32; 2], [f32; 2])>`.
struct ClipOut {
    hit: u32,
    pa: vec2<f32>,
    pb: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Computes the 4-bit Cohen-Sutherland outcode of point p against the window.
// A coordinate within CMP_EPS of an edge is treated as inside that edge.
// Matches the reference `ClipRect::outcode`.
fn outcode(rmin: vec2<f32>, rmax: vec2<f32>, p: vec2<f32>) -> u32 {
    var code = OUTCODE_INSIDE;
    if (p.x < rmin.x - CMP_EPS) {
        code = code | OUTCODE_LEFT;
    } else if (p.x > rmax.x + CMP_EPS) {
        code = code | OUTCODE_RIGHT;
    }
    if (p.y < rmin.y - CMP_EPS) {
        code = code | OUTCODE_BOTTOM;
    } else if (p.y > rmax.y + CMP_EPS) {
        code = code | OUTCODE_TOP;
    }
    return code;
}

// Intersects segment a -> b with the single window edge named by the highest
// relevant bit of out_code, using a linear parameter so the math stays affine
// (+ - * /). The paired coordinate difference in the denominator is non-zero by
// construction. Matches the reference `edge_intersection`.
fn edge_intersection(
    rmin: vec2<f32>,
    rmax: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    out_code: u32,
) -> vec2<f32> {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    if ((out_code & OUTCODE_TOP) != 0u) {
        return vec2<f32>(a.x + dx * (rmax.y - a.y) / dy, rmax.y);
    } else if ((out_code & OUTCODE_BOTTOM) != 0u) {
        return vec2<f32>(a.x + dx * (rmin.y - a.y) / dy, rmin.y);
    } else if ((out_code & OUTCODE_RIGHT) != 0u) {
        return vec2<f32>(rmax.x, a.y + dy * (rmax.x - a.x) / dx);
    }
    // The remaining case is the left edge; a non-inside out_code always has at
    // least one of the four bits set.
    return vec2<f32>(rmin.x, a.y + dy * (rmin.x - a.x) / dx);
}

// Clips segment a_in -> b_in to the rectangular window, returning the surviving
// sub-segment with its presence flag. Each step clears at least one outcode
// bit, so the loop resolves within four edge clips; the fixed bound keeps the
// control flow uniform. Matches the reference `clip_segment`.
fn clip_segment(
    rmin: vec2<f32>,
    rmax: vec2<f32>,
    a_in: vec2<f32>,
    b_in: vec2<f32>,
) -> ClipOut {
    var out: ClipOut;
    out.hit = 0u;
    out.pa = a_in;
    out.pb = b_in;

    var pa = a_in;
    var pb = b_in;
    var code_a = outcode(rmin, rmax, pa);
    var code_b = outcode(rmin, rmax, pb);

    for (var iter = 0u; iter < 8u; iter = iter + 1u) {
        if ((code_a | code_b) == OUTCODE_INSIDE) {
            // Both endpoints inside every half-plane: accept the segment.
            out.hit = 1u;
            out.pa = pa;
            out.pb = pb;
            return out;
        }
        if ((code_a & code_b) != OUTCODE_INSIDE) {
            // Both endpoints share an outside half-plane: reject the segment.
            out.hit = 0u;
            return out;
        }

        // At least one endpoint is outside and they do not share a region;
        // clip the outside endpoint against one offending edge.
        var out_code = code_b;
        if (code_a != OUTCODE_INSIDE) {
            out_code = code_a;
        }
        let clipped = edge_intersection(rmin, rmax, pa, pb, out_code);
        if (out_code == code_a) {
            pa = clipped;
            code_a = outcode(rmin, rmax, pa);
        } else {
            pb = clipped;
            code_b = outcode(rmin, rmax, pb);
        }
    }

    // Unreachable in practice: the loop always resolves within four clips.
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let rmin = q.rect_min;
    let rmax = q.rect_max;

    let clip = clip_segment(rmin, rmax, q.a, q.b);

    var out: Result;
    out.hit = clip.hit;
    out.outcode_a = outcode(rmin, rmax, q.a);
    out.outcode_b = outcode(rmin, rmax, q.b);
    out.pad0 = 0u;
    out.clip_a = clip.pa;
    out.clip_b = clip.pb;
    results[idx] = out;
}
"#;

/// One segment-clip query: the axis-aligned clip window `rect` and the segment
/// `a -> b`, the same inputs the reference `clip_segment` and `outcode` consume.
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipSegmentQuery {
    /// The axis-aligned clip window, taken as supplied (callers that might pass
    /// an inverted rectangle should `normalized` it first, exactly as on the
    /// `CPU` reference).
    pub rect: ClipRect,
    /// Segment start endpoint `a`.
    pub a: [f32; 2],
    /// Segment end endpoint `b`.
    pub b: [f32; 2],
}

impl ClipSegmentQuery {
    /// Builds a query from the clip window and the two segment endpoints.
    ///
    /// Provenance: twinned from this repository's `cohen_sutherland_clip`.
    #[must_use]
    pub const fn new(rect: ClipRect, a: [f32; 2], b: [f32; 2]) -> ClipSegmentQuery {
        ClipSegmentQuery { rect, a, b }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across `clip_segment` and `outcode`.
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipSegmentResult {
    /// Whether the clipped segment survives, matching `clip_segment(..).is_some()`.
    pub hit: bool,
    /// The clipped endpoint derived from the input `a`; equals the raw `a` when
    /// `hit` is `false`.
    pub a: [f32; 2],
    /// The clipped endpoint derived from the input `b`; equals the raw `b` when
    /// `hit` is `false`.
    pub b: [f32; 2],
    /// The raw `Cohen-Sutherland` outcode of the input endpoint `a`, one of the
    /// `OUTCODE_*` bit combinations.
    pub outcode_a: u32,
    /// The raw `Cohen-Sutherland` outcode of the input endpoint `b`, one of the
    /// `OUTCODE_*` bit combinations.
    pub outcode_b: u32,
}

impl ClipSegmentResult {
    /// Rebuilds the reference `clip_segment` return value: `Some((a, b))` when
    /// the segment survives, `None` otherwise.
    ///
    /// Provenance: twinned from this repository's `cohen_sutherland_clip`.
    #[must_use]
    pub fn clipped(&self) -> Option<([f32; 2], [f32; 2])> {
        if self.hit {
            Some((self.a, self.b))
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
    /// Window lower bounds `(xmin, ymin)`.
    rect_min: [f32; 2],
    /// Window upper bounds `(xmax, ymax)`.
    rect_max: [f32; 2],
    /// Segment start endpoint.
    a: [f32; 2],
    /// Segment end endpoint.
    b: [f32; 2],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &ClipSegmentQuery) -> GpuQuery {
        GpuQuery {
            rect_min: [query.rect.xmin, query.rect.ymin],
            rect_max: [query.rect.xmax, query.rect.ymax],
            a: query.a,
            b: query.b,
        }
    }
}

/// `repr(C)` `std430` image of one result: four `u32` words
/// `(hit, outcode_a, outcode_b, pad0)` followed by two `vec2<f32>` endpoint
/// slots `(clip_a, clip_b)` — `32` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the clipped segment survives, else `0`.
    hit: u32,
    /// Raw outcode of the input endpoint `a`.
    outcode_a: u32,
    /// Raw outcode of the input endpoint `b`.
    outcode_b: u32,
    /// Padding word aligning the following `vec2` to its `8`-byte slot.
    pad0: u32,
    /// Clipped endpoint derived from `a`.
    clip_a: [f32; 2],
    /// Clipped endpoint derived from `b`.
    clip_b: [f32; 2],
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

/// A compiled, reusable `Cohen-Sutherland` clip compute pipeline.
///
/// Provenance: twinned from this repository's `cohen_sutherland_clip`.
pub struct GpuCohenSutherlandClip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCohenSutherlandClip {
    /// Compiles the clip kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: twinned from this repository's `cohen_sutherland_clip`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCohenSutherlandClip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip"),
            source: ShaderSource::Wgsl(COHEN_SUTHERLAND_CLIP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCohenSutherlandClip {
            module,
            layout,
            pipeline,
        }
    }

    /// Clips every query on-device and returns one [`ClipSegmentResult`] per
    /// input, in order.
    ///
    /// Each result matches the reference answers: the survival boolean and both
    /// outcodes exactly, the clipped coordinates to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: twinned from this repository's `cohen_sutherland_clip`.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ClipSegmentQuery]) -> Vec<ClipSegmentResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_output"),
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
            label: Some("prism_volumetric_cohen_sutherland_clip_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_bind_group"),
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
            label: Some("prism_volumetric_cohen_sutherland_clip_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cohen_sutherland_clip_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cohen_sutherland_clip_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`ClipSegmentResult`].
fn decode_result(raw: &GpuResult) -> ClipSegmentResult {
    ClipSegmentResult {
        hit: raw.hit != 0,
        a: raw.clip_a,
        b: raw.clip_b,
        outcode_a: raw.outcode_a,
        outcode_b: raw.outcode_b,
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
