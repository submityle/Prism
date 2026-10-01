//! `wgpu` compute twin of the half-space (plane) segment-clipping contract
//! ([`plane_clip`](prism_render_architecture::particle::plane_clip)).
//!
//! The `CPU` golden
//! [`plane_clip`](prism_render_architecture::particle::plane_clip) owns the
//! small, verifiable reference the frustum culling, near/far slicing and volume
//! clipping passes share for one segment against one oriented plane: the signed
//! distance
//! [`Plane::signed_distance`](prism_render_architecture::particle::plane_clip::Plane::signed_distance)
//! (`n · p + d`), the half-space classification
//! [`classify`](prism_render_architecture::particle::plane_clip::classify) into
//! [`Side`](prism_render_architecture::particle::plane_clip::Side), the crossing
//! parameter
//! [`intersect_param`](prism_render_architecture::particle::plane_clip::intersect_param)
//! (`t = da / (da - db)`, kept only when `t` lands in `[0, 1]` and the segment
//! is not parallel), and the segment clip
//! [`clip_segment`](prism_render_architecture::particle::plane_clip::clip_segment)
//! that keeps the inside (`n · p + d >= 0`) half-space. [`GpuPlaneClip`] is the
//! on-device twin: one thread clips one 3D segment against one plane,
//! reproducing the reference branch for branch, so a passing real-device parity
//! test is direct evidence the ported kernel classifies, intersects and trims
//! the same geometry — and the same degenerate cases — the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference reports is reproduced for one shared
//! geometry `(normal, d, a, b, eps)`: the signed distance of each endpoint, the
//! [`Side`](prism_render_architecture::particle::plane_clip::Side)
//! classification code of each endpoint under the symmetric `eps` band, the
//! `intersect_param` hit flag with its crossing `t`, the `clip_segment` survival
//! flag and the two clipped endpoints kept in the original `a -> b` order. The
//! reference's regimes are mirrored branch for branch: a segment fully inside
//! the half-space (returned unchanged), a segment fully outside (rejected), a
//! segment that straddles the plane with either endpoint trimmed to the boundary
//! intersection, and a segment parallel to the plane (the `intersect_param`
//! denominator guard fires).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `dot`,
//! `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no `sqrt` (the
//! solve is a single ratio, never a length) and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! divide, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! The classification codes and the two hit flags are integer / sign / epsilon
//! decisions over values conditioned clear of a tie, so they match exactly and
//! the parity test asserts `==` on them. The continuous `f32` fields (the signed
//! distances, the crossing `t` and the clipped endpoints) are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The parity
//! test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`)
//! on them, tight enough to catch a genuinely wrong port yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`plane_clip`](prism_render_architecture::particle::plane_clip); no
//! third-party engine source or derived code.

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

/// Classification code for a point strictly inside the kept half-space
/// (`n · p + d > eps`), mirroring
/// [`Side::Inside`](prism_render_architecture::particle::plane_clip::Side::Inside).
///
/// Provenance: twinned from this repository's `plane_clip`.
pub const SIDE_INSIDE: u32 = 0;

/// Classification code for a point strictly outside the kept half-space
/// (`n · p + d < -eps`), mirroring
/// [`Side::Outside`](prism_render_architecture::particle::plane_clip::Side::Outside).
///
/// Provenance: twinned from this repository's `plane_clip`.
pub const SIDE_OUTSIDE: u32 = 1;

/// Classification code for a point within `eps` of the plane, mirroring
/// [`Side::On`](prism_render_architecture::particle::plane_clip::Side::On).
///
/// Provenance: twinned from this repository's `plane_clip`.
pub const SIDE_ON: u32 = 2;

/// The portable core-`WGSL` plane segment-clip kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`plane_clip`](prism_render_architecture::particle::plane_clip) branch for
/// branch; see the module documentation for the algorithm.
const PLANE_CLIP_WGSL: &str = r#"
// Plane segment-clip twin: one thread clips one 3D segment against one oriented
// plane. It reproduces the signed distance of each endpoint, the half-space
// classification code of each endpoint, the intersect_param crossing and the
// clip_segment survivor, mirroring the CPU golden `particle::plane_clip` branch
// for branch. It uses only the portable core-WGSL subset (abs, dot and
// + - * /), needs no sqrt (the solve is a single ratio) and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::plane_clip; no
// third-party engine source or derived code.

// Epsilon guarding the intersect_param denominator against a parallel segment,
// matching the reference `CMP_EPS`. The classification band uses the per-query
// `eps` instead. No exact == / != on an f32 appears.
const CMP_EPS: f32 = 1.0e-6;

const SIDE_INSIDE: u32 = 0u;
const SIDE_OUTSIDE: u32 = 1u;
const SIDE_ON: u32 = 2u;

struct Params {
    // Number of clip queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Plane normal n in `n · p + d`; the plane constant d sits in the w lane.
    normal: vec3<f32>,
    d: f32,
    // Segment start endpoint a; the classify band half-width eps sits in w.
    a: vec3<f32>,
    eps: f32,
    // Segment end endpoint b; a pad lane follows.
    b: vec3<f32>,
    pad0: f32,
}

struct Result {
    // Side code of endpoint a, side code of endpoint b, the intersect_param hit
    // flag and the clip_segment survival flag.
    classify_a: u32,
    classify_b: u32,
    intersect_hit: u32,
    clip_hit: u32,
    // Signed distance of a, signed distance of b, the crossing parameter t and
    // one pad lane: four scalars filling one vec4 slot.
    signed_a: f32,
    signed_b: f32,
    intersect_t: f32,
    pad0: f32,
    // Clipped endpoint derived from a; the raw a when clip_hit == 0. A pad lane
    // follows.
    clip_a: vec3<f32>,
    pad1: f32,
    // Clipped endpoint derived from b; the raw b when clip_hit == 0. A pad lane
    // follows.
    clip_b: vec3<f32>,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Evaluates `n · p + d`, the signed distance scaled by |n|. Matches the
// reference `Plane::signed_distance`.
fn signed_distance(normal: vec3<f32>, d: f32, p: vec3<f32>) -> f32 {
    return dot(normal, p) + d;
}

// Classifies a point against the plane with a symmetric eps boundary band,
// matching the reference `classify`. Inside when the signed distance exceeds
// eps, outside when it is below -eps, otherwise on the boundary.
fn classify(normal: vec3<f32>, d: f32, p: vec3<f32>, eps: f32) -> u32 {
    let dist = signed_distance(normal, d, p);
    if (dist > eps) {
        return SIDE_INSIDE;
    } else if (dist < -eps) {
        return SIDE_OUTSIDE;
    }
    return SIDE_ON;
}

// Linearly interpolates two points: a + (b - a) * t. Matches the reference
// `lerp3`.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let normal = q.normal;
    let d = q.d;
    let a = q.a;
    let b = q.b;

    let da = signed_distance(normal, d, a);
    let db = signed_distance(normal, d, b);

    // intersect_param: the crossing t = da / (da - db), kept only when the
    // segment is not parallel (|denom| >= CMP_EPS) and t lands in [0, 1].
    let denom = da - db;
    var intersect_hit: u32 = 0u;
    var intersect_t: f32 = 0.0;
    if (abs(denom) >= CMP_EPS) {
        let t = da / denom;
        if (t >= 0.0 && t <= 1.0) {
            intersect_hit = 1u;
            intersect_t = t;
        }
    }

    // clip_segment: keep the inside (n · p + d >= 0) half-space, replacing the
    // outside endpoint with the boundary intersection and preserving a -> b
    // order. On a full miss the raw endpoints are echoed back.
    let a_inside = da >= 0.0;
    let b_inside = db >= 0.0;
    var clip_hit: u32 = 0u;
    var clip_a: vec3<f32> = a;
    var clip_b: vec3<f32> = b;
    if (a_inside && b_inside) {
        clip_hit = 1u;
        clip_a = a;
        clip_b = b;
    } else if (a_inside && !b_inside) {
        let t = da / (da - db);
        clip_hit = 1u;
        clip_a = a;
        clip_b = lerp3(a, b, t);
    } else if (!a_inside && b_inside) {
        let t = da / (da - db);
        clip_hit = 1u;
        clip_a = lerp3(a, b, t);
        clip_b = b;
    }

    var out: Result;
    out.classify_a = classify(normal, d, a, q.eps);
    out.classify_b = classify(normal, d, b, q.eps);
    out.intersect_hit = intersect_hit;
    out.clip_hit = clip_hit;
    out.signed_a = da;
    out.signed_b = db;
    out.intersect_t = intersect_t;
    out.pad0 = 0.0;
    out.clip_a = clip_a;
    out.pad1 = 0.0;
    out.clip_b = clip_b;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One plane segment-clip query: a shared geometry fed to all four reference
/// solves. The plane is `normal · p + d` with the inside half-space
/// `normal · p + d >= 0`; the segment runs `a -> b`; `eps` is the symmetric
/// classification band half-width passed to the reference `classify`.
///
/// Provenance: twinned from this repository's `plane_clip`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneClipQuery {
    /// The plane normal `n` in `n · p + d` (need not be unit length).
    pub normal: [f32; 3],
    /// The plane constant `d` in `n · p + d`.
    pub d: f32,
    /// The segment start endpoint `a`.
    pub a: [f32; 3],
    /// The segment end endpoint `b`.
    pub b: [f32; 3],
    /// The symmetric classification band half-width passed to `classify`.
    pub eps: f32,
}

impl PlaneClipQuery {
    /// Builds a query from the shared geometry.
    ///
    /// Provenance: twinned from this repository's `plane_clip`.
    #[must_use]
    pub const fn new(
        normal: [f32; 3],
        d: f32,
        a: [f32; 3],
        b: [f32; 3],
        eps: f32,
    ) -> PlaneClipQuery {
        PlaneClipQuery {
            normal,
            d,
            a,
            b,
            eps,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across `signed_distance`, `classify`, `intersect_param` and
/// `clip_segment`.
///
/// Provenance: twinned from this repository's `plane_clip`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneClipResult {
    /// The `Side` classification code of endpoint `a`, one of `SIDE_INSIDE`,
    /// `SIDE_OUTSIDE` or `SIDE_ON`.
    pub classify_a: u32,
    /// The `Side` classification code of endpoint `b`.
    pub classify_b: u32,
    /// The signed distance `n · a + d`, matching `Plane::signed_distance`.
    pub signed_distance_a: f32,
    /// The signed distance `n · b + d`.
    pub signed_distance_b: f32,
    /// Whether `intersect_param` returns `Some`, matching a non-parallel segment
    /// whose crossing `t` lands in `[0, 1]`.
    pub intersect_hit: bool,
    /// The crossing parameter `t`; meaningful only when `intersect_hit`.
    pub intersect_t: f32,
    /// Whether `clip_segment` returns `Some` (any part survives inside the
    /// half-space).
    pub clip_hit: bool,
    /// The clipped endpoint derived from the input `a`; equals the raw `a` when
    /// `clip_hit` is `false`.
    pub clip_a: [f32; 3],
    /// The clipped endpoint derived from the input `b`; equals the raw `b` when
    /// `clip_hit` is `false`.
    pub clip_b: [f32; 3],
}

impl PlaneClipResult {
    /// Rebuilds the reference `clip_segment` return value: `Some((a, b))` when
    /// the segment survives, `None` otherwise.
    ///
    /// Provenance: twinned from this repository's `plane_clip`.
    #[must_use]
    pub fn clipped(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.clip_hit {
            Some((self.clip_a, self.clip_b))
        } else {
            None
        }
    }

    /// Rebuilds the reference `intersect_param` return value: `Some(t)` when the
    /// segment crosses the plane within `[0, 1]`, `None` otherwise.
    ///
    /// Provenance: twinned from this repository's `plane_clip`.
    #[must_use]
    pub fn intersect(&self) -> Option<f32> {
        if self.intersect_hit {
            Some(self.intersect_t)
        } else {
            None
        }
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots holding
/// `(normal.xyz, d)`, `(a.xyz, eps)` and `(b.xyz, pad)` — `48` bytes, each
/// `vec3` on its `16`-byte-aligned slot exactly as the `WGSL` `Query` struct
/// reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Plane normal `n`.
    normal: [f32; 3],
    /// Plane constant `d`, packed into the normal slot's `w` lane.
    d: f32,
    /// Segment start endpoint.
    a: [f32; 3],
    /// Classification band half-width, packed into the `a` slot's `w` lane.
    eps: f32,
    /// Segment end endpoint.
    b: [f32; 3],
    /// Padding lane after the end endpoint.
    pad0: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &PlaneClipQuery) -> GpuQuery {
        GpuQuery {
            normal: query.normal,
            d: query.d,
            a: query.a,
            eps: query.eps,
            b: query.b,
            pad0: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-`u32` slot
/// `(classify_a, classify_b, intersect_hit, clip_hit)`, a four-`f32` slot
/// `(signed_a, signed_b, intersect_t, pad)`, then two `vec4` slots for the
/// clipped endpoints — `64` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `Side` code of endpoint `a`.
    classify_a: u32,
    /// `Side` code of endpoint `b`.
    classify_b: u32,
    /// `intersect_param` hit flag.
    intersect_hit: u32,
    /// `clip_segment` survival flag.
    clip_hit: u32,
    /// Signed distance of `a`.
    signed_a: f32,
    /// Signed distance of `b`.
    signed_b: f32,
    /// Crossing parameter `t`.
    intersect_t: f32,
    /// Padding word.
    pad0: f32,
    /// Clipped endpoint derived from `a`.
    clip_a: [f32; 3],
    /// Padding lane after the clipped `a`.
    pad1: f32,
    /// Clipped endpoint derived from `b`.
    clip_b: [f32; 3],
    /// Padding lane after the clipped `b`.
    pad2: f32,
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

/// A compiled, reusable plane segment-clip compute pipeline.
///
/// Provenance: twinned from this repository's `plane_clip`.
pub struct GpuPlaneClip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPlaneClip {
    /// Compiles the plane segment-clip kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: twinned from this repository's `plane_clip`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPlaneClip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_plane_clip"),
            source: ShaderSource::Wgsl(PLANE_CLIP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_plane_clip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_plane_clip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_plane_clip_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPlaneClip {
            module,
            layout,
            pipeline,
        }
    }

    /// Clips every query on-device and returns one [`PlaneClipResult`] per input,
    /// in order.
    ///
    /// Each result matches the reference answers: the classification codes and
    /// both hit flags exactly, the signed distances, crossing parameter and
    /// clipped endpoints to within the tolerance documented on this module. An
    /// empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: twinned from this repository's `plane_clip`.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[PlaneClipQuery]) -> Vec<PlaneClipResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_plane_clip_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_plane_clip_output"),
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
            label: Some("prism_volumetric_plane_clip_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_plane_clip_bind_group"),
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
            label: Some("prism_volumetric_plane_clip_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_plane_clip_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_plane_clip_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`PlaneClipResult`].
fn decode_result(raw: &GpuResult) -> PlaneClipResult {
    PlaneClipResult {
        classify_a: raw.classify_a,
        classify_b: raw.classify_b,
        signed_distance_a: raw.signed_a,
        signed_distance_b: raw.signed_b,
        intersect_hit: raw.intersect_hit != 0,
        intersect_t: raw.intersect_t,
        clip_hit: raw.clip_hit != 0,
        clip_a: raw.clip_a,
        clip_b: raw.clip_b,
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
