//! `wgpu` compute twin of the `NaN`-safe tone-mapping clamp
//! ([`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)).
//!
//! The `CPU` golden
//! [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)
//! takes an untrusted paper-white / peak luminance pair in `nits` and clamps it
//! into the physically meaningful range of a selected
//! [`DisplayOutput`](prism_render_architecture::display::DisplayOutput). It is a
//! pure, fixed-width per-element transform: the peak is clamped to
//! `[MIN_NITS, output.peak_nits()]`, the paper-white is then clamped to
//! `[MIN_NITS, peak]`, and `NaN` on either input collapses to the lower bound.
//! There is no container, no sort and no state, so the whole function is a
//! clean device twin.
//!
//! [`GpuDisplayTonemapTarget`] is the on-device twin: one thread resolves one
//! query, reproducing the reference's two-stage `sanitize` exactly, so a
//! passing real-device parity test is direct evidence the ported kernel
//! performs the same `NaN`-guarded double clamp the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For one query the twin reads the raw `paper_white_nits` and `peak_nits` plus
//! an output selector, resolves the output's peak-`nits` ceiling (`SdrSrgb` →
//! `80`, `ScRgb` → `1000`, `Hdr10Pq` → `10000`, matching the reference
//! metadata table), and applies the two-stage `sanitize`: `peak` clamped to
//! `[MIN_NITS, ceiling]` then `paper_white` clamped to `[MIN_NITS, peak]`, with
//! `NaN` collapsing to the lower bound. The resolved `(paper_white, peak)` pair
//! obeys the reference invariant `paper_white <= peak`.
//!
//! # What stays on the host
//!
//! Nothing of the clamp itself: the whole function is per-element and
//! fixed-width. The host only owns the empty-batch short-circuit (a storage
//! buffer cannot be zero-sized) and the mapping of a
//! [`DisplayOutput`](prism_render_architecture::display::DisplayOutput) variant
//! to the kernel's output selector word.
//!
//! # Correctness model
//!
//! The kernel performs no floating-point arithmetic — only comparisons, a
//! `min`, and value copies — so the resolved luminances are produced by exactly
//! the same selection the reference `clamp`/`sanitize` makes. The reference
//! guards `NaN` with [`f32::is_nan`] before [`f32::clamp`]; `WGSL` has no
//! `NaN` predicate, so the twin uses the identity that `NaN` fails every
//! ordered comparison: `value >= lo` is `false` for `NaN`, leaving the result
//! at the lower bound `lo`, which is precisely the reference's `NaN` → `lo`
//! mapping. For finite inputs the gate plus a `min` reproduces
//! `clamp(value, lo, hi)` bit-for-bit (`value < lo` → `lo`, `value > hi` →
//! `hi`, in range → `value`). The parity test still compares with a tolerance
//! to honor the house f32 rule, but the two sides agree exactly on every
//! fixture clear of the clamp knots.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — comparisons, `min`
//! and unsigned index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `sqrt`, no inverse trigonometry, no `round` and no `ceil`, and no bare f32
//! equality. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::display::tonemap`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::display::DisplayOutput;
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

/// Output selector word for the `SdrSrgb` display output (ceiling `80` `nits`).
const OUTPUT_SDR_SRGB: u32 = 0;
/// Output selector word for the `ScRgb` display output (ceiling `1000` `nits`).
const OUTPUT_SC_RGB: u32 = 1;
/// Output selector word for the `Hdr10Pq` display output (ceiling `10000`
/// `nits`).
const OUTPUT_HDR10_PQ: u32 = 2;

/// The portable core-`WGSL` tone-map clamp kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)
/// two-stage `NaN`-safe clamp; see the module documentation for the algorithm.
const DISPLAY_TONEMAP_TARGET_WGSL: &str = r#"
// Tone-map clamp twin: one thread resolves one query, reproducing the CPU
// golden `display::ToneMapTarget::clamped_to` two-stage NaN-safe
// clamp with only comparisons, a min and value copies. The host owns the
// empty-batch short-circuit and the DisplayOutput->selector mapping; there is
// no container, sort or state on the device.
//
// Provenance: 孪生自本仓 prism_render_architecture::display::tonemap；无第三方
// 引擎源码或衍生代码。

const MIN_NITS: f32 = 1.0;

const OUTPUT_SDR_SRGB: u32 = 0u;
const OUTPUT_SC_RGB: u32 = 1u;
const OUTPUT_HDR10_PQ: u32 = 2u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Raw diffuse paper-white luminance in nits (may be NaN or out of range).
    paper_white_nits: f32,
    // Raw peak luminance in nits (may be NaN or out of range).
    peak_nits: f32,
    // Output selector: 0 = SdrSrgb, 1 = ScRgb, 2 = Hdr10Pq.
    output_code: u32,
    pad0: u32,
}

struct Result {
    // Clamped diffuse paper-white luminance in nits.
    paper_white_nits: f32,
    // Clamped peak luminance in nits.
    peak_nits: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps `value` into [lo, hi], mapping NaN to lo. NaN fails every ordered
// comparison, so `value >= lo` is false for NaN and the result stays at lo,
// exactly as the reference `sanitize` maps NaN to lo. For finite inputs the
// gate plus `min` reproduces clamp(value, lo, hi): below lo yields lo, above
// hi yields hi, in range yields value. Uses no bare f32 equality.
fn sanitize(value: f32, lo: f32, hi: f32) -> f32 {
    var out: f32 = lo;
    if (value >= lo) {
        out = min(value, hi);
    }
    return out;
}

// Resolves the output's peak-nits ceiling, mirroring the reference metadata
// table (SdrSrgb -> 80, ScRgb -> 1000, Hdr10Pq -> 10000).
fn ceiling_for(code: u32) -> f32 {
    var ceiling: f32 = 80.0;
    if (code == OUTPUT_SC_RGB) {
        ceiling = 1000.0;
    }
    if (code == OUTPUT_HDR10_PQ) {
        ceiling = 10000.0;
    }
    return ceiling;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Two-stage clamp: peak first into [MIN_NITS, ceiling], then paper-white
    // into [MIN_NITS, peak], so the invariant paper_white <= peak holds.
    let ceiling = ceiling_for(q.output_code);
    let peak = sanitize(q.peak_nits, MIN_NITS, ceiling);
    let paper_white = sanitize(q.paper_white_nits, MIN_NITS, peak);

    var out: Result;
    out.paper_white_nits = paper_white;
    out.peak_nits = peak;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`DISPLAY_TONEMAP_TARGET_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the raw paper-white and peak
/// luminances plus an output selector word and one pad word to a `16`-byte
/// stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Raw diffuse paper-white luminance in `nits`.
    paper_white_nits: f32,
    /// Raw peak luminance in `nits`.
    peak_nits: f32,
    /// Output selector: `0` = `SdrSrgb`, `1` = `ScRgb`, `2` = `Hdr10Pq`.
    output_code: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the clamped paper-white and peak luminances plus two pad words to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped diffuse paper-white luminance in `nits`.
    paper_white_nits: f32,
    /// Clamped peak luminance in `nits`.
    peak_nits: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One tone-map clamp query: the raw `paper_white_nits` and `peak_nits` and the
/// target [`DisplayOutput`](prism_render_architecture::display::DisplayOutput)
/// whose peak-`nits` ceiling bounds the clamp.
///
/// Mirrors the inputs of the `CPU` golden
/// [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to):
/// the pair is an untrusted
/// [`ToneMapTarget`](prism_render_architecture::display::ToneMapTarget)
/// and the selected output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayTonemapTargetQuery {
    /// Raw diffuse paper-white luminance in `nits` (may be `NaN` or out of
    /// range).
    pub paper_white_nits: f32,
    /// Raw peak luminance in `nits` (may be `NaN` or out of range).
    pub peak_nits: f32,
    /// The target display output whose peak-`nits` ceiling bounds the clamp.
    pub output: DisplayOutput,
}

impl DisplayTonemapTargetQuery {
    /// Builds a query from a raw paper-white / peak pair and a target `output`.
    #[must_use]
    pub const fn new(
        paper_white_nits: f32,
        peak_nits: f32,
        output: DisplayOutput,
    ) -> DisplayTonemapTargetQuery {
        DisplayTonemapTargetQuery {
            paper_white_nits,
            peak_nits,
            output,
        }
    }
}

/// One resolved tone-map clamp, mirroring the `(paper_white, peak)` pair the
/// `CPU` golden
/// [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)
/// returns.
///
/// The reference invariant `paper_white_nits <= peak_nits` always holds, and
/// both values lie in `[MIN_NITS, output.peak_nits()]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayTonemapTargetResult {
    /// Clamped diffuse paper-white luminance in `nits`.
    pub paper_white_nits: f32,
    /// Clamped peak luminance in `nits`.
    pub peak_nits: f32,
}

/// Maps a [`DisplayOutput`](prism_render_architecture::display::DisplayOutput)
/// to the kernel's output selector word.
fn output_code(output: DisplayOutput) -> u32 {
    match output {
        DisplayOutput::SdrSrgb => OUTPUT_SDR_SRGB,
        DisplayOutput::ScRgb => OUTPUT_SC_RGB,
        DisplayOutput::Hdr10Pq => OUTPUT_HDR10_PQ,
    }
}

/// Encodes one [`DisplayTonemapTargetQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &DisplayTonemapTargetQuery) -> GpuQuery {
    GpuQuery {
        paper_white_nits: q.paper_white_nits,
        peak_nits: q.peak_nits,
        output_code: output_code(q.output),
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`DisplayTonemapTargetResult`].
fn decode_result(raw: &GpuResult) -> DisplayTonemapTargetResult {
    DisplayTonemapTargetResult {
        paper_white_nits: raw.paper_white_nits,
        peak_nits: raw.peak_nits,
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

/// A compiled, reusable tone-map clamp compute pipeline, twinning the `CPU`
/// golden
/// [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to).
pub struct GpuDisplayTonemapTarget {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDisplayTonemapTarget {
    /// Compiles the tone-map clamp kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDisplayTonemapTarget {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_display_tonemap_target"),
            source: ShaderSource::Wgsl(DISPLAY_TONEMAP_TARGET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDisplayTonemapTarget {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`DisplayTonemapTargetResult`] per input, in order.
    ///
    /// Each resolved pair equals the reference
    /// [`ToneMapTarget::clamped_to`](prism_render_architecture::display::ToneMapTarget::clamped_to)
    /// result for the same inputs. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DisplayTonemapTargetQuery],
    ) -> Vec<DisplayTonemapTargetResult> {
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
            label: Some("prism_volumetric_display_tonemap_target_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_bind_group"),
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
            label: Some("prism_volumetric_display_tonemap_target_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_display_tonemap_target_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_display_tonemap_target_pass"),
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
