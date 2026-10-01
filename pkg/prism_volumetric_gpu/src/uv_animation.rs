//! `wgpu` compute twin of the continuous `UV`-coordinate animation contract
//! ([`uv_animation`](prism_render_architecture::particle::uv_animation),
//! particle design §15).
//!
//! The `CPU` golden
//! [`uv_animation`](prism_render_architecture::particle::uv_animation) owns the
//! pure maths that maps an incoming texture coordinate to an outgoing one for a
//! `Sprite`/material sampler: the atomic translate
//! ([`uv_scroll`](prism_render_architecture::particle::uv_animation::uv_scroll)),
//! the pivoted scale
//! ([`uv_tile`](prism_render_architecture::particle::uv_animation::uv_tile)),
//! the pivoted rotation by a caller-supplied `(cos_rot, sin_rot)` pair
//! ([`uv_rotate`](prism_render_architecture::particle::uv_animation::uv_rotate)),
//! the time-driven linear panner
//! ([`scroll_offset`](prism_render_architecture::particle::uv_animation::scroll_offset)),
//! the half-open fold back into `[0, 1)`
//! ([`wrap01`](prism_render_architecture::particle::uv_animation::wrap01) and its
//! componentwise sibling
//! [`wrap_uv`](prism_render_architecture::particle::uv_animation::wrap_uv)), the
//! flowmap-style twin-offset blend
//! ([`dual_panner`](prism_render_architecture::particle::uv_animation::dual_panner)),
//! and the composed transform
//! ([`UvTransform::apply`](prism_render_architecture::particle::uv_animation::UvTransform::apply)).
//! [`GpuUvAnimation`] is the on-device twin: one thread per `UV` reproduces every
//! answer, so a passing real-device parity test is direct evidence the ported
//! kernel evaluates the same coordinate warp the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Every per-element answer the reference computes is reproduced for a batch of
//! independent `UV`s: the scrolled, tiled and rotated coordinates, the
//! time-driven scroll offset, the wrapped coordinate (which exercises both
//! components of `wrap01`), the two dual-panner offsets with their clamped blend
//! weight, and the fully composed `apply`. The rotation reads a caller-supplied
//! `(cos_rot, sin_rot)` pair exactly as the reference does; the kernel computes
//! no trigonometry itself.
//!
//! # Determinism
//!
//! The reference performs no transcendental maths: its only floating-point
//! primitive beyond ordinary arithmetic is `floor`, used by `wrap01` to fold a
//! coordinate via `x - floor(x)`. The kernel mirrors that exactly — `floor`
//! plus `+ - * /` and one `clamp` for the blend weight — so no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow` or `sqrt` appears and no optional device feature
//! is required. `wrap01` snaps a folded value within [`EPS`] of `1.0` back to
//! `0.0` to stay strictly half-open; the kernel reproduces that same compare,
//! and the parity fixtures stay clear of the snap threshold so both devices take
//! the same branch.
//!
//! # Correctness model
//!
//! Each element is a fixed, non-reorderable sequence of adds, multiplies and one
//! `floor`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the continuous coordinates,
//! tight enough to catch a genuinely wrong port (a dropped term, a swapped
//! coefficient, a wrong pivot) yet loose enough to admit legal fused
//! multiply-add contraction. The blend weight is clamped identically on both
//! sides.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::uv_animation::{
    dual_panner, scroll_offset, uv_rotate, uv_scroll, uv_tile, wrap_uv, UvTransform,
};
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

/// The portable core-`WGSL` `UV`-animation kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`uv_animation`](prism_render_architecture::particle::uv_animation) branch for
/// branch; see the module documentation for the algorithm.
const UV_ANIMATION_WGSL: &str = r#"
// UV-animation twin: one thread per UV reproduces the scroll, tile, rotate,
// time-driven scroll offset, wrap, dual-panner and composed apply the CPU golden
// `particle::uv_animation` computes. It mirrors the reference branch for branch,
// uses only the portable core-WGSL subset (floor/clamp and + - * /), computes no
// trigonometry (the rotation reads a caller-supplied cos/sin pair) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. There is no
// loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::uv_animation；无第三方
// 引擎源码或衍生代码。

// Snap threshold for the half-open wrap: a folded value within EPS of 1.0 is
// pulled back to 0.0 so wrap01 stays strictly half-open. Matches the reference
// `EPS`; the compare rule used instead of an f32 `==`.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of UV elements in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // uv.xy and the uv_scroll offset.xy.
    uv_offset: vec4<f32>,
    // tiling.xy and pivot.xy, shared by uv_tile, uv_rotate and apply.
    tiling_pivot: vec4<f32>,
    // cos_rot, sin_rot and the scroll_offset velocity.xy.
    rot_velocity: vec4<f32>,
    // time, the apply scroll.xy and the dual-panner blend weight.
    time_scroll_blend: vec4<f32>,
    // dual-panner offset_a.xy and offset_b.xy.
    panner_offsets: vec4<f32>,
    // The coordinate fed to wrap_uv (its two lanes exercise wrap01) plus two pad
    // lanes.
    wrap_input: vec4<f32>,
}

struct Result {
    // uv_scroll result.xy and uv_tile result.xy.
    scrolled_tiled: vec4<f32>,
    // uv_rotate result.xy and scroll_offset result.xy.
    rotated_scroll_off: vec4<f32>,
    // wrap_uv result.xy and dual-panner first offset.xy.
    wrapped_dual_a: vec4<f32>,
    // dual-panner second offset.xy and UvTransform::apply result.xy.
    dual_b_applied: vec4<f32>,
    // Clamped dual-panner blend weight plus three pad lanes.
    blend: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Translates `uv` by `offset`; mirrors the reference `uv_scroll`.
fn scroll(uv: vec2<f32>, offset: vec2<f32>) -> vec2<f32> {
    return uv + offset;
}

// Scales `uv` by `tiling` about `pivot`; mirrors the reference `uv_tile`. The
// pivot is a fixed point.
fn tile(uv: vec2<f32>, tiling: vec2<f32>, pivot: vec2<f32>) -> vec2<f32> {
    return (uv - pivot) * tiling + pivot;
}

// Rotates `uv` about `pivot` using the caller-supplied cos/sin; mirrors the
// reference `uv_rotate`. No trigonometry is computed here.
fn rotate(uv: vec2<f32>, cos_rot: f32, sin_rot: f32, pivot: vec2<f32>) -> vec2<f32> {
    let lu = uv.x - pivot.x;
    let lv = uv.y - pivot.y;
    return vec2<f32>(
        cos_rot * lu - sin_rot * lv + pivot.x,
        sin_rot * lu + cos_rot * lv + pivot.y,
    );
}

// Folds a single coordinate back into the half-open unit range [0, 1) via
// `x - floor(x)`; mirrors the reference `wrap01`. A result within EPS of 1.0 (or
// above it) snaps to 0.0 so the range stays strictly half-open.
fn wrap01(x: f32) -> f32 {
    let f = x - floor(x);
    if (f >= 1.0 - EPS) {
        return 0.0;
    }
    return f;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let uv = q.uv_offset.xy;
    let offset = q.uv_offset.zw;
    let tiling = q.tiling_pivot.xy;
    let pivot = q.tiling_pivot.zw;
    let cos_rot = q.rot_velocity.x;
    let sin_rot = q.rot_velocity.y;
    let velocity = q.rot_velocity.zw;
    let time = q.time_scroll_blend.x;
    let apply_scroll = q.time_scroll_blend.yz;
    let blend = q.time_scroll_blend.w;
    let offset_a = q.panner_offsets.xy;
    let offset_b = q.panner_offsets.zw;
    let wrap_in = q.wrap_input.xy;

    // Atomic steps.
    let scrolled = scroll(uv, offset);
    let tiled = tile(uv, tiling, pivot);
    let rotated = rotate(uv, cos_rot, sin_rot, pivot);
    let scroll_off = velocity * time;

    // Half-open fold on both lanes.
    let wrapped = vec2<f32>(wrap01(wrap_in.x), wrap01(wrap_in.y));

    // Flowmap dual panner: two scrolled copies plus a clamped blend weight.
    let dual_a = scroll(uv, offset_a);
    let dual_b = scroll(uv, offset_b);
    let clamped_blend = clamp(blend, 0.0, 1.0);

    // Composed transform: subtract pivot, scale, rotate, add pivot back, then
    // add the global scroll. The pivot is a fixed point of the tile+rotate stage.
    let su = (uv.x - pivot.x) * tiling.x;
    let sv = (uv.y - pivot.y) * tiling.y;
    let ru = cos_rot * su - sin_rot * sv;
    let rv = sin_rot * su + cos_rot * sv;
    let applied = vec2<f32>(
        ru + pivot.x + apply_scroll.x,
        rv + pivot.y + apply_scroll.y,
    );

    var out: Result;
    out.scrolled_tiled = vec4<f32>(scrolled.x, scrolled.y, tiled.x, tiled.y);
    out.rotated_scroll_off = vec4<f32>(rotated.x, rotated.y, scroll_off.x, scroll_off.y);
    out.wrapped_dual_a = vec4<f32>(wrapped.x, wrapped.y, dual_a.x, dual_a.y);
    out.dual_b_applied = vec4<f32>(dual_b.x, dual_b.y, applied.x, applied.y);
    out.blend = vec4<f32>(clamped_blend, 0.0, 0.0, 0.0);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the element count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`UV_ANIMATION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// six `vec4` slots packing every input the twinned functions consume.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `uv.xy` and the `uv_scroll` `offset.xy`.
    uv_offset: [f32; 4],
    /// `tiling.xy` and `pivot.xy`.
    tiling_pivot: [f32; 4],
    /// `cos_rot`, `sin_rot` and the `scroll_offset` `velocity.xy`.
    rot_velocity: [f32; 4],
    /// `time`, the `apply` `scroll.xy` and the dual-panner `blend` weight.
    time_scroll_blend: [f32; 4],
    /// Dual-panner `offset_a.xy` and `offset_b.xy`.
    panner_offsets: [f32; 4],
    /// The `wrap_uv` input `.xy` plus two pad lanes.
    wrap_input: [f32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// five `vec4` slots packing every answer the reference reports.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `uv_scroll` result `.xy` and `uv_tile` result `.xy`.
    scrolled_tiled: [f32; 4],
    /// `uv_rotate` result `.xy` and `scroll_offset` result `.xy`.
    rotated_scroll_off: [f32; 4],
    /// `wrap_uv` result `.xy` and dual-panner first offset `.xy`.
    wrapped_dual_a: [f32; 4],
    /// Dual-panner second offset `.xy` and `apply` result `.xy`.
    dual_b_applied: [f32; 4],
    /// Clamped dual-panner blend weight plus three pad lanes.
    blend: [f32; 4],
}

/// One query for the `UV`-animation twin: an input `UV` plus every parameter the
/// twinned functions consume.
///
/// The atomic steps, the dual panner and the composed `apply` are independent,
/// so a single query exercises every twinned function at once. The rotation and
/// `apply` share the same `(cos_rot, sin_rot)`, `tiling` and `pivot`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvAnimationQuery {
    /// The incoming texture coordinate the atomic steps and `apply` transform.
    pub uv: [f32; 2],
    /// Translation for `uv_scroll`.
    pub offset: [f32; 2],
    /// Per-axis scale for `uv_tile` and `apply`, applied about `pivot`.
    pub tiling: [f32; 2],
    /// The fixed point `tiling` and the rotation act about.
    pub pivot: [f32; 2],
    /// Cosine of the rotation angle, supplied by the caller.
    pub cos_rot: f32,
    /// Sine of the rotation angle, supplied by the caller.
    pub sin_rot: f32,
    /// Velocity for `scroll_offset`.
    pub velocity: [f32; 2],
    /// Time for `scroll_offset`.
    pub time: f32,
    /// Global scroll added last by `apply`.
    pub scroll: [f32; 2],
    /// First dual-panner offset.
    pub offset_a: [f32; 2],
    /// Second dual-panner offset.
    pub offset_b: [f32; 2],
    /// Dual-panner blend weight, clamped to `[0, 1]`.
    pub blend: f32,
    /// Coordinate fed to `wrap_uv`; its two lanes exercise `wrap01`.
    pub wrap_input: [f32; 2],
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvAnimationResult {
    /// `uv_scroll` result, matching
    /// [`uv_scroll`](prism_render_architecture::particle::uv_animation::uv_scroll).
    pub scrolled: [f32; 2],
    /// `uv_tile` result, matching
    /// [`uv_tile`](prism_render_architecture::particle::uv_animation::uv_tile).
    pub tiled: [f32; 2],
    /// `uv_rotate` result, matching
    /// [`uv_rotate`](prism_render_architecture::particle::uv_animation::uv_rotate).
    pub rotated: [f32; 2],
    /// `scroll_offset` result, matching
    /// [`scroll_offset`](prism_render_architecture::particle::uv_animation::scroll_offset).
    pub scroll_offset: [f32; 2],
    /// `wrap_uv` result, matching
    /// [`wrap_uv`](prism_render_architecture::particle::uv_animation::wrap_uv) (and
    /// so [`wrap01`](prism_render_architecture::particle::uv_animation::wrap01) on
    /// each lane).
    pub wrapped: [f32; 2],
    /// Dual-panner first offset, matching the first component of
    /// [`dual_panner`](prism_render_architecture::particle::uv_animation::dual_panner).
    pub dual_a: [f32; 2],
    /// Dual-panner second offset, matching the second component of
    /// [`dual_panner`](prism_render_architecture::particle::uv_animation::dual_panner).
    pub dual_b: [f32; 2],
    /// Clamped dual-panner blend weight, matching the third component of
    /// [`dual_panner`](prism_render_architecture::particle::uv_animation::dual_panner).
    pub dual_blend: f32,
    /// `UvTransform::apply` result, matching
    /// [`UvTransform::apply`](prism_render_architecture::particle::uv_animation::UvTransform::apply).
    pub applied: [f32; 2],
}

/// Evaluates the `CPU` golden for one query, calling every twinned function so a
/// parity test can pin the `GPU` result against the reference.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn golden(query: &UvAnimationQuery) -> UvAnimationResult {
    let transform = UvTransform::new(
        query.scroll,
        query.tiling,
        query.pivot,
        query.cos_rot,
        query.sin_rot,
    );
    let (dual_a, dual_b, dual_blend) =
        dual_panner(query.uv, query.offset_a, query.offset_b, query.blend);
    UvAnimationResult {
        scrolled: uv_scroll(query.uv, query.offset),
        tiled: uv_tile(query.uv, query.tiling, query.pivot),
        rotated: uv_rotate(query.uv, query.cos_rot, query.sin_rot, query.pivot),
        scroll_offset: scroll_offset(query.velocity, query.time),
        wrapped: wrap_uv(query.wrap_input),
        dual_a,
        dual_b,
        dual_blend,
        applied: transform.apply(query.uv),
    }
}

/// Encodes one [`UvAnimationQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &UvAnimationQuery) -> GpuQuery {
    GpuQuery {
        uv_offset: [q.uv[0], q.uv[1], q.offset[0], q.offset[1]],
        tiling_pivot: [q.tiling[0], q.tiling[1], q.pivot[0], q.pivot[1]],
        rot_velocity: [q.cos_rot, q.sin_rot, q.velocity[0], q.velocity[1]],
        time_scroll_blend: [q.time, q.scroll[0], q.scroll[1], q.blend],
        panner_offsets: [q.offset_a[0], q.offset_a[1], q.offset_b[0], q.offset_b[1]],
        wrap_input: [q.wrap_input[0], q.wrap_input[1], 0.0, 0.0],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`UvAnimationResult`].
fn decode_result(raw: &GpuResult) -> UvAnimationResult {
    UvAnimationResult {
        scrolled: [raw.scrolled_tiled[0], raw.scrolled_tiled[1]],
        tiled: [raw.scrolled_tiled[2], raw.scrolled_tiled[3]],
        rotated: [raw.rotated_scroll_off[0], raw.rotated_scroll_off[1]],
        scroll_offset: [raw.rotated_scroll_off[2], raw.rotated_scroll_off[3]],
        wrapped: [raw.wrapped_dual_a[0], raw.wrapped_dual_a[1]],
        dual_a: [raw.wrapped_dual_a[2], raw.wrapped_dual_a[3]],
        dual_b: [raw.dual_b_applied[0], raw.dual_b_applied[1]],
        dual_blend: raw.blend[0],
        applied: [raw.dual_b_applied[2], raw.dual_b_applied[3]],
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

/// A compiled, reusable `UV`-animation compute pipeline, twinning the `CPU`
/// golden
/// [`uv_animation`](prism_render_architecture::particle::uv_animation).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
pub struct GpuUvAnimation {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuUvAnimation {
    /// Compiles the `UV`-animation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuUvAnimation {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_uv_animation"),
            source: ShaderSource::Wgsl(UV_ANIMATION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_uv_animation_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_uv_animation_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_uv_animation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuUvAnimation {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`UvAnimationResult`] per
    /// input, in order.
    ///
    /// Each coordinate matches the reference to within the tolerance documented
    /// on this module; the clamped blend weight matches identically. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::uv_animation`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[UvAnimationQuery],
    ) -> Vec<UvAnimationResult> {
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
            label: Some("prism_volumetric_uv_animation_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_uv_animation_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_uv_animation_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_uv_animation_bind_group"),
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
            label: Some("prism_volumetric_uv_animation_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_uv_animation_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_uv_animation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per UV element, flattened to a 1-D dispatch.
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
