//! `wgpu` compute twin of the temporal-upscale color-space primitives
//! ([`color`](prism_render_architecture::temporal_upscale::color)).
//!
//! A temporal upsampler decides how much accumulated history to keep by
//! comparing it against the current frame in a luma/chroma space, and averages
//! `HDR` samples through a firefly-suppressing weight and an exactly-invertible
//! tone-map. The `CPU` golden
//! [`color`](prism_render_architecture::temporal_upscale::color) owns those
//! deterministic, transcendental-free primitives:
//! [`luminance`](prism_render_architecture::temporal_upscale::color::luminance),
//! [`rgb_to_ycocg`](prism_render_architecture::temporal_upscale::color::rgb_to_ycocg),
//! [`ycocg_to_rgb`](prism_render_architecture::temporal_upscale::color::ycocg_to_rgb),
//! [`tonemap`](prism_render_architecture::temporal_upscale::color::tonemap),
//! [`untonemap`](prism_render_architecture::temporal_upscale::color::untonemap)
//! and
//! [`tonemap_weight`](prism_render_architecture::temporal_upscale::color::tonemap_weight).
//!
//! [`GpuTaauColorTransform`] is the on-device twin that runs one thread per
//! pixel and reproduces every one of those closed forms step for step, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same color transforms the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one input linear `RGB` color and one weighting luma the kernel emits:
//!
//! * the Rec. 709 relative luminance `0.2126 R + 0.7152 G + 0.0722 B`;
//! * the `YCoCg` lifting transform `Y = R/4 + G/2 + B/4`, `Co = R/2 - B/2`,
//!   `Cg = -R/4 + G/2 - B/4`;
//! * the Karis tone-map `c / (1 + max(m, 0))` where `m` is the largest channel;
//! * the firefly weight `1 / (1 + luma)`, with a non-finite or non-positive
//!   luma resolving to the full weight `1`;
//! * the `YCoCg -> RGB` round trip `ycocg_to_rgb(rgb_to_ycocg(rgb))`; and
//! * the tone-map round trip `untonemap(tonemap(rgb))`, whose inverse divides by
//!   `(1 - m)` guarded to the smallest positive normal.
//!
//! Every coefficient is copied verbatim from the golden, so the only source of
//! `CPU`/`GPU` divergence is a last-place rounding difference in a shared
//! `+ - * /` sequence.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `max` and
//! reciprocal division — with no `sin`, `cos`, `exp`, `log`, `pow`, inverse
//! trigonometry, `round`, `smoothstep` or `sqrt`, and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! The non-finite guard in the weight is expressed without an `f32` equality:
//! the full-weight fallback is the default and the reciprocal branch is taken
//! only when `luma > 0`, which is `false` for both a non-positive luma and a
//! `NaN` (every ordered comparison with `NaN` is `false`), matching the golden
//! `luma.is_nan() || luma <= 0.0` fallback exactly.
//!
//! # Correctness model
//!
//! Every operation is a shared `+ - * /` / `max` sequence with no reordering,
//! so `CPU` and `GPU` agree to within a last-place rounding slack. The parity
//! test asserts `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every continuous
//! output and additionally checks that both round trips recover the input.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::color`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` color-transform kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `transform` mirrors
/// the `CPU` golden
/// [`color`](prism_render_architecture::temporal_upscale::color) closed forms;
/// see the module documentation for the algorithms.
const TAAU_COLOR_TRANSFORM_WGSL: &str = r#"
// Temporal-upscale color-transform twin: one thread computes one pixel's
// luminance, YCoCg transform, Karis tone-map, firefly weight and the two
// inverse round trips, mirroring the CPU golden
// `temporal_upscale::color` with only + - * /, max and reciprocal division.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::color；无第三方
// 引擎源码或衍生代码。

// Smallest positive normal f32, matching the golden `f32::MIN_POSITIVE` guard
// on the tone-map inverse denominator.
const MIN_POSITIVE: f32 = 1.17549435e-38;

struct Params {
    // Number of pixels in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Linear RGB color channels.
    r: f32,
    g: f32,
    b: f32,
    // Luma fed to the firefly weight; kept independent so a caller can probe
    // the zero / large / degenerate branches directly.
    weight_luma: f32,
}

struct Result {
    luminance: f32,
    // YCoCg transform of the input RGB.
    y: f32,
    co: f32,
    cg: f32,
    // Karis tone-map of the input RGB.
    tr: f32,
    tg: f32,
    tb: f32,
    // Firefly weight of weight_luma.
    weight: f32,
    // ycocg_to_rgb(rgb_to_ycocg(rgb)) round trip.
    rr: f32,
    rg: f32,
    rb: f32,
    // untonemap(tonemap(rgb)) round trip.
    hr: f32,
    hg: f32,
    hb: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Rec. 709 relative luminance, mirroring the golden `luminance`.
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

// Largest of three channels, written as nested max to match the golden
// `max_channel`.
fn max_channel(r: f32, g: f32, b: f32) -> f32 {
    return max(max(r, g), b);
}

@compute @workgroup_size(64)
fn transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let r = q.r;
    let g = q.g;
    let b = q.b;

    var out: Result;
    out.luminance = luminance(r, g, b);

    // rgb_to_ycocg lifting form.
    let y = 0.25 * r + 0.5 * g + 0.25 * b;
    let co = 0.5 * r - 0.5 * b;
    let cg = -0.25 * r + 0.5 * g - 0.25 * b;
    out.y = y;
    out.co = co;
    out.cg = cg;

    // Karis tone-map c / (1 + max(m, 0)).
    let m_tone = max(max_channel(r, g, b), 0.0);
    let inv_tone = 1.0 / (1.0 + m_tone);
    out.tr = r * inv_tone;
    out.tg = g * inv_tone;
    out.tb = b * inv_tone;

    // Firefly weight: full weight by default, reciprocal only for luma > 0 so a
    // non-positive luma and a NaN both resolve to 1 without an f32 equality.
    var weight: f32 = 1.0;
    if (q.weight_luma > 0.0) {
        weight = 1.0 / (1.0 + q.weight_luma);
    }
    out.weight = weight;

    // ycocg_to_rgb round trip of the computed YCoCg.
    out.rr = y + co - cg;
    out.rg = y + cg;
    out.rb = y - co - cg;

    // untonemap(tonemap(rgb)) round trip; the inverse divides by (1 - m)
    // guarded to the smallest positive normal, matching the golden.
    let tr = out.tr;
    let tg = out.tg;
    let tb = out.tb;
    let m_inv = max(max_channel(tr, tg, tb), 0.0);
    let denom = max(1.0 - m_inv, MIN_POSITIVE);
    let inv_un = 1.0 / denom;
    out.hr = tr * inv_un;
    out.hg = tg * inv_un;
    out.hb = tb * inv_un;

    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pixel count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_COLOR_TRANSFORM_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pixels in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one pixel query: the three linear `RGB`
/// channels and the firefly-weight luma, a `16`-byte stride matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Red channel.
    r: f32,
    /// Green channel.
    g: f32,
    /// Blue channel.
    b: f32,
    /// Luma fed to the firefly weight.
    weight_luma: f32,
}

/// `repr(C)` `std430` layout of one pixel result, matching the `WGSL` `Result`
/// struct: the luminance, the `YCoCg` triple, the tone-mapped triple, the
/// firefly weight, the two round-trip triples and two pad words to a `64`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Rec. 709 luminance.
    luminance: f32,
    /// `YCoCg` luma channel.
    y: f32,
    /// `YCoCg` orange-chroma channel.
    co: f32,
    /// `YCoCg` green-chroma channel.
    cg: f32,
    /// Tone-mapped red.
    tr: f32,
    /// Tone-mapped green.
    tg: f32,
    /// Tone-mapped blue.
    tb: f32,
    /// Firefly weight.
    weight: f32,
    /// Round-trip red through `YCoCg`.
    rr: f32,
    /// Round-trip green through `YCoCg`.
    rg: f32,
    /// Round-trip blue through `YCoCg`.
    rb: f32,
    /// Round-trip red through the tone-map.
    hr: f32,
    /// Round-trip green through the tone-map.
    hg: f32,
    /// Round-trip blue through the tone-map.
    hb: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One pixel query for the color-transform twin: a linear `RGB` color and the
/// luma fed to the firefly weight.
///
/// The weight luma is carried independently of the color so a caller can probe
/// the zero, large and degenerate (`NaN` / negative) branches of
/// [`tonemap_weight`](prism_render_architecture::temporal_upscale::color::tonemap_weight)
/// directly.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::color`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauColorTransformQuery {
    /// Linear `RGB` color channels.
    pub rgb: [f32; 3],
    /// Luma fed to the firefly weight.
    pub weight_luma: f32,
}

impl TaauColorTransformQuery {
    /// Builds a query from a color and a weighting luma.
    #[must_use]
    pub const fn new(rgb: [f32; 3], weight_luma: f32) -> TaauColorTransformQuery {
        TaauColorTransformQuery { rgb, weight_luma }
    }
}

/// One pixel's bundle of color transforms, mirroring the `CPU` golden
/// [`color`](prism_render_architecture::temporal_upscale::color) primitives.
///
/// `ycocg` is
/// [`rgb_to_ycocg`](prism_render_architecture::temporal_upscale::color::rgb_to_ycocg),
/// `tonemap` is
/// [`tonemap`](prism_render_architecture::temporal_upscale::color::tonemap),
/// `rgb_round_trip` is
/// [`ycocg_to_rgb`](prism_render_architecture::temporal_upscale::color::ycocg_to_rgb)
/// applied to `ycocg`, and `hdr_round_trip` is
/// [`untonemap`](prism_render_architecture::temporal_upscale::color::untonemap)
/// applied to `tonemap`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::color`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauColorTransformResult {
    /// Rec. 709 relative luminance of the input color.
    pub luminance: f32,
    /// `YCoCg` transform of the input color.
    pub ycocg: [f32; 3],
    /// Karis tone-map of the input color.
    pub tonemap: [f32; 3],
    /// Firefly weight of the query's `weight_luma`.
    pub tonemap_weight: f32,
    /// `ycocg_to_rgb(rgb_to_ycocg(rgb))` round trip of the input color.
    pub rgb_round_trip: [f32; 3],
    /// `untonemap(tonemap(rgb))` round trip of the input color.
    pub hdr_round_trip: [f32; 3],
}

/// Encodes one [`TaauColorTransformQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TaauColorTransformQuery) -> GpuQuery {
    GpuQuery {
        r: q.rgb[0],
        g: q.rgb[1],
        b: q.rgb[2],
        weight_luma: q.weight_luma,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauColorTransformResult`].
fn decode_result(raw: &GpuResult) -> TaauColorTransformResult {
    TaauColorTransformResult {
        luminance: raw.luminance,
        ycocg: [raw.y, raw.co, raw.cg],
        tonemap: [raw.tr, raw.tg, raw.tb],
        tonemap_weight: raw.weight,
        rgb_round_trip: [raw.rr, raw.rg, raw.rb],
        hdr_round_trip: [raw.hr, raw.hg, raw.hb],
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

/// A compiled, reusable color-transform compute pipeline, twinning the `CPU`
/// golden
/// [`color`](prism_render_architecture::temporal_upscale::color) primitives.
pub struct GpuTaauColorTransform {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauColorTransform {
    /// Compiles the color-transform kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauColorTransform {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_color_transform"),
            source: ShaderSource::Wgsl(TAAU_COLOR_TRANSFORM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_color_transform_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_color_transform_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_color_transform_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("transform"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauColorTransform {
            module,
            layout,
            pipeline,
        }
    }

    /// Transforms every pixel in `queries` and returns one
    /// [`TaauColorTransformResult`] per input, in order.
    ///
    /// Each field equals the matching `CPU` golden
    /// [`color`](prism_render_architecture::temporal_upscale::color) primitive
    /// to within a last-place rounding slack. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauColorTransformQuery],
    ) -> Vec<TaauColorTransformResult> {
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
            label: Some("prism_volumetric_taau_color_transform_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_color_transform_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_color_transform_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_color_transform_bind_group"),
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
            label: Some("prism_volumetric_taau_color_transform_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_color_transform_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_color_transform_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, flattened to a 1-D dispatch.
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
