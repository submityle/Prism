//! `wgpu` compute twin of the per-channel color sanitizer inside the
//! reconstructor
//! ([`reconstruct`](prism_render_architecture::temporal_upscale::reconstruct)).
//!
//! The `CPU` golden `sanitize_color` keeps a stray `NaN`, `inf`, or negative
//! input (from an upstream pass) from poisoning the tone-map and the
//! accumulated history: each channel becomes `max(c, 0)` when it is finite and
//! `0` when it is non-finite. It is a stateless, closed-form, per-channel
//! predicate with no transcendental calls, no `64`-bit integers, and no loops,
//! so it ports to the device directly.
//!
//! The golden is a private helper, but it is reached deterministically through
//! the public entry point
//! [`resolve`](prism_render_architecture::temporal_upscale::reconstruct::resolve):
//! on the reset path (a disocclusion, missing history, or an empty
//! neighborhood) `resolve` returns `sanitize_color(current_color)` as the
//! output color with a fresh single-frame confidence. The parity oracle calls
//! `resolve` with `disoccluded = true` so the returned color is exactly
//! `sanitize_color(current_color)`, grounding `GPU == resolve == sanitize_color`.
//!
//! [`GpuTaauReconstructSanitizeColor`] is the on-device twin of that helper:
//! one thread sanitizes one color, reproducing the finite guard and the
//! negative clamp per channel. A passing real-device parity test is direct
//! evidence the ported kernel folds the exact same edge cases the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query holding a raw color `[r, g, b]` the kernel reproduces, per
//! channel `c`:
//! - `out = max(c, 0.0)` when `c` is finite;
//! - `out = 0.0` when `c` is non-finite (`NaN` or `+/- inf`).
//!
//! The finite test is `abs(c) <= f32::MAX`, which is `true` for every finite
//! value, `false` for `+/- inf` (whose magnitude exceeds the largest finite
//! `f32`), and `false` for `NaN` (which compares `false` against every bound).
//! A negative but finite channel clamps to `0`; a `-inf` channel also lands on
//! `0`, matching the golden exactly.
//!
//! # What stays on the host
//!
//! Nothing of this helper stays on the host; it is fully closed-form. The
//! sibling composite resolve
//! ([`resolve`](prism_render_architecture::temporal_upscale::reconstruct::resolve))
//! chains tone-map, `YCoCg`, neighborhood clip, and lock state and is twinned by
//! other modules, not here. An empty batch short-circuits on the host, since a
//! storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The kernel performs only a comparison, a `max`, and a select — no arithmetic
//! that could drift — so a passing fixture reproduces the reference branch
//! exactly. The parity test still asserts the shared continuous tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every channel, which here is
//! satisfied with zero difference on every fixture, including the non-finite
//! inputs that both sides fold to a finite `0`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `max`, and an
//! `f32` magnitude comparison — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `sqrt`, no `round`, and no `64`-bit
//! integers. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::reconstruct`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` color sanitizer kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `sanitize_color`; see the module documentation for the
/// algorithm.
const TAAU_RECONSTRUCT_SANITIZE_COLOR_WGSL: &str = r#"
// Color sanitizer twin: one thread replaces each non-finite channel with 0 and
// clamps negatives to 0, mirroring the CPU golden
// `temporal_upscale::reconstruct::sanitize_color` with only abs, max and an f32
// magnitude comparison. A channel is finite iff abs(c) <= f32::MAX; NaN and
// +/- inf both fail that test and fold to 0.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::reconstruct；无第三方
// 引擎源码或衍生代码。

// Largest finite f32 (f32::MAX). A finite channel has magnitude at most this;
// +/- inf exceeds it and NaN compares false against it.
const F32_MAX: f32 = 3.40282347e38;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Raw color channels before sanitizing.
    r: f32,
    g: f32,
    b: f32,
    pad0: u32,
}

struct Result {
    // Sanitized color channels (non-negative, finite).
    r: f32,
    g: f32,
    b: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Reproduces one channel of `sanitize_color`: a finite channel clamps negatives
// to 0, a non-finite channel folds to 0. The finite test uses abs(c) <= f32::MAX
// so NaN (compares false) and +/- inf (magnitude exceeds the bound) both fold.
fn sanitize_channel(c: f32) -> f32 {
    if (abs(c) <= F32_MAX) {
        return max(c, 0.0);
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.r = sanitize_channel(q.r);
    out.g = sanitize_channel(q.g);
    out.b = sanitize_channel(q.b);
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_RECONSTRUCT_SANITIZE_COLOR_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query: the three raw color channels plus one
/// pad word, a `16`-byte stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Raw red channel before sanitizing.
    r: f32,
    /// Raw green channel before sanitizing.
    g: f32,
    /// Raw blue channel before sanitizing.
    b: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the three sanitized color channels plus one pad word, a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Sanitized red channel.
    r: f32,
    /// Sanitized green channel.
    g: f32,
    /// Sanitized blue channel.
    b: f32,
    /// Padding word.
    pad0: u32,
}

/// One sanitize query for the twin: a raw linear color straight off an upstream
/// pass, before any finite guard or negative clamp.
///
/// The host enqueues one [`TaauReconstructSanitizeColorQuery`] per query,
/// mirroring the input of the reference `sanitize_color` (reached through the
/// public
/// [`resolve`](prism_render_architecture::temporal_upscale::reconstruct::resolve)
/// reset path).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauReconstructSanitizeColorQuery {
    /// Raw color channels `[r, g, b]` before sanitizing.
    pub color: [f32; 3],
}

impl TaauReconstructSanitizeColorQuery {
    /// Builds a query from a raw color `[r, g, b]`.
    #[must_use]
    pub const fn new(color: [f32; 3]) -> TaauReconstructSanitizeColorQuery {
        TaauReconstructSanitizeColorQuery { color }
    }
}

/// One sanitized color, mirroring the reference `sanitize_color` (reached
/// through the public
/// [`resolve`](prism_render_architecture::temporal_upscale::reconstruct::resolve)
/// reset path).
///
/// Every channel is non-negative and finite: a finite negative channel clamps
/// to `0` and a non-finite channel (`NaN` or `+/- inf`) folds to `0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauReconstructSanitizeColorResult {
    /// Sanitized color channels `[r, g, b]` (non-negative, finite).
    pub color: [f32; 3],
}

/// Encodes one [`TaauReconstructSanitizeColorQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &TaauReconstructSanitizeColorQuery) -> GpuQuery {
    GpuQuery {
        r: q.color[0],
        g: q.color[1],
        b: q.color[2],
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`TaauReconstructSanitizeColorResult`].
fn decode_result(raw: &GpuResult) -> TaauReconstructSanitizeColorResult {
    TaauReconstructSanitizeColorResult {
        color: [raw.r, raw.g, raw.b],
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

/// A compiled, reusable color sanitizer compute pipeline, twinning the `CPU`
/// golden `sanitize_color` from
/// [`reconstruct`](prism_render_architecture::temporal_upscale::reconstruct).
pub struct GpuTaauReconstructSanitizeColor {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauReconstructSanitizeColor {
    /// Compiles the sanitizer kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauReconstructSanitizeColor {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color"),
            source: ShaderSource::Wgsl(TAAU_RECONSTRUCT_SANITIZE_COLOR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauReconstructSanitizeColor {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`TaauReconstructSanitizeColorResult`] per input, in order.
    ///
    /// Each output matches the reference exactly: a finite negative channel
    /// clamps to `0` and a non-finite channel folds to `0`. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauReconstructSanitizeColorQuery],
    ) -> Vec<TaauReconstructSanitizeColorResult> {
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
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_bind_group"),
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
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_reconstruct_sanitize_color_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_reconstruct_sanitize_color_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
