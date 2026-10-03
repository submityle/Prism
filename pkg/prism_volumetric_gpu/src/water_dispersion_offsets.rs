//! `wgpu` compute twin of the spectral-dispersion refraction primitives inside
//! the water optics contract
//! ([`dispersion`](prism_render_architecture::water::dispersion), water design
//! §5 refraction routing).
//!
//! The `CPU` golden models normal chromatic dispersion with the classic
//! `Cauchy` two-term index law and turns the per-channel indices into the
//! ordered screen-space refraction offsets the refraction pass samples the
//! scene colour with. This twin reproduces every one of those pure numeric
//! functions on the device, one thread per sample, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same indices,
//! transmitted sines, colour spread and offsets the reference does.
//!
//! # What is twinned
//!
//! For one sample the kernel reproduces, in closed form:
//!
//! - [`cauchy_ior`](prism_render_architecture::water::dispersion::cauchy_ior):
//!   `n(lambda) = a + b / lambda^2`, with a degenerate non-positive wavelength
//!   clamped to a tiny positive value.
//! - [`spectral_iors`](prism_render_architecture::water::dispersion::spectral_iors):
//!   the same law sampled at the three reference wavelengths (red `0.700`,
//!   green `0.546`, blue `0.440` micrometres), giving `r < g < b` for positive
//!   dispersion.
//! - [`channel_transmitted_sine`](prism_render_architecture::water::dispersion::channel_transmitted_sine):
//!   `Snell`'s law `sin(theta_t) = clamp(sin(theta_i), 0, 1) / n`, re-clamped to
//!   `0..=1`.
//! - [`dispersion_spread`](prism_render_architecture::water::dispersion::dispersion_spread):
//!   `max(sin_t(red) - sin_t(blue), 0)`, the non-negative width of the colour
//!   fringe.
//! - [`dispersion_offsets`](prism_render_architecture::water::dispersion::dispersion_offsets):
//!   each channel's transmitted sine scaled by `max(strength, 0)`.
//!
//! The [`RgbIor`](prism_render_architecture::water::dispersion::RgbIor) the last
//! two functions read is flattened to three `f32` fields in the query.
//!
//! # What stays on the host
//!
//! The reference exposes only these fixed-width pure functions; there is no
//! variable-length aggregate to port. The host builds the per-sample queries
//! (one per refraction sample) and reads the offsets back, enqueuing at least
//! one sample so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every output threads through only subtracts, multiplies, a guarded divide,
//! `max` and `clamp`, so the `CPU` and `GPU` are not bit-exact but agree to a
//! few units in the last place. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, relative floor `1e-6`) on each
//! continuous field, tight enough to catch a genuinely wrong port (a dropped
//! clamp, a swapped channel, a wrong radicand) yet loose enough to admit a legal
//! last-place difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `-`, `*`, `/`,
//! `max` and `clamp` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry, no `sqrt`, no `round` and no `ceil`. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::dispersion`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` spectral-dispersion kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`dispersion`](prism_render_architecture::water::dispersion) per-sample pure
/// functions; see the module documentation for the algorithm.
const WATER_DISPERSION_OFFSETS_WGSL: &str = r#"
// Spectral dispersion twin: one thread resolves one refraction sample,
// mirroring the CPU golden `water::dispersion` closed forms. It evaluates the
// Cauchy index law, the three reference-wavelength indices, the Snell
// transmitted sine, the red-minus-blue colour spread and the per-channel
// screen-space offsets, with only +, -, *, /, max and clamp; no floating-point
// ==, no sqrt, no transcendental, no u64.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::dispersion；无第三方
// 引擎源码或衍生代码。

// Shared epsilon matching `water::EPS`, guarding the wavelength and index
// divisors from a non-positive input.
const EPS: f32 = 1.0e-6;

// Reference RGB sample wavelengths in micrometres: red, green, blue.
const WAVELENGTH_R: f32 = 0.700;
const WAVELENGTH_G: f32 = 0.546;
const WAVELENGTH_B: f32 = 0.440;

struct Params {
    // Number of refraction samples in the storage arrays; threads past return.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Cauchy baseline index a (about 1.324 for water).
    cauchy_a: f32,
    // Cauchy dispersion coefficient b in micrometre-squared.
    cauchy_b: f32,
    // Standalone wavelength in micrometres for the cauchy_ior probe.
    wavelength_um: f32,
    // Sine of the incidence angle in air for the Snell transmitted sine.
    sin_incidence: f32,
    // Screen-space refraction gain folded with water thickness.
    strength: f32,
    // Standalone channel index for the channel_transmitted_sine probe.
    channel_ior: f32,
    // Red-channel index for the spread/offset functions (flattened RgbIor.r).
    ior_r: f32,
    // Green-channel index for the offset function (flattened RgbIor.g).
    ior_g: f32,
    // Blue-channel index for the spread/offset functions (flattened RgbIor.b).
    ior_b: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // cauchy_ior(a, b, wavelength_um).
    cauchy: f32,
    // spectral_iors(a, b).r.
    spectral_r: f32,
    // spectral_iors(a, b).g.
    spectral_g: f32,
    // spectral_iors(a, b).b.
    spectral_b: f32,
    // channel_transmitted_sine(sin_incidence, channel_ior).
    channel_sine: f32,
    // dispersion_spread({ior_r, ior_g, ior_b}, sin_incidence).
    spread: f32,
    // dispersion_offsets(...).r.
    offset_r: f32,
    // dispersion_offsets(...).g.
    offset_g: f32,
    // dispersion_offsets(...).b.
    offset_b: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Cauchy refractive-index law n(lambda) = a + b / lambda^2, guarding a tiny
// non-positive wavelength to EPS to avoid dividing by zero.
fn cauchy_ior(a: f32, b: f32, wavelength_um: f32) -> f32 {
    let lambda = max(wavelength_um, EPS);
    return a + b / (lambda * lambda);
}

// Snell transmitted sine: clamp(sin_incidence, 0, 1) / n, re-clamped to 0..=1,
// with the index guarded away from zero.
fn channel_transmitted_sine(sin_incidence: f32, ior: f32) -> f32 {
    let n = max(ior, EPS);
    let s = clamp(sin_incidence, 0.0, 1.0);
    return clamp(s / n, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Standalone Cauchy probe and the three reference-wavelength indices.
    let cauchy = cauchy_ior(q.cauchy_a, q.cauchy_b, q.wavelength_um);
    let spectral_r = cauchy_ior(q.cauchy_a, q.cauchy_b, WAVELENGTH_R);
    let spectral_g = cauchy_ior(q.cauchy_a, q.cauchy_b, WAVELENGTH_G);
    let spectral_b = cauchy_ior(q.cauchy_a, q.cauchy_b, WAVELENGTH_B);

    // Standalone Snell transmitted sine for one channel.
    let channel_sine = channel_transmitted_sine(q.sin_incidence, q.channel_ior);

    // Colour spread between red and blue refracted rays, clamped non-negative.
    let s = clamp(q.sin_incidence, 0.0, 1.0);
    let sine_r = channel_transmitted_sine(s, q.ior_r);
    let sine_g = channel_transmitted_sine(s, q.ior_g);
    let sine_b = channel_transmitted_sine(s, q.ior_b);
    let spread = max(sine_r - sine_b, 0.0);

    // Per-channel screen-space offsets scaled by the non-negative gain.
    let gain = max(q.strength, 0.0);
    let offset_r = sine_r * gain;
    let offset_g = sine_g * gain;
    let offset_b = sine_b * gain;

    var out: Result;
    out.cauchy = cauchy;
    out.spectral_r = spectral_r;
    out.spectral_g = spectral_g;
    out.spectral_b = spectral_b;
    out.channel_sine = channel_sine;
    out.spread = spread;
    out.offset_r = offset_r;
    out.offset_g = offset_g;
    out.offset_b = offset_b;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_DISPERSION_OFFSETS_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid samples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one dispersion sample: the nine planner inputs
/// plus three pad words to a `48`-byte, `16`-aligned stride matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `Cauchy` baseline index `a`.
    cauchy_a: f32,
    /// `Cauchy` dispersion coefficient `b`.
    cauchy_b: f32,
    /// Standalone probe wavelength in micrometres.
    wavelength_um: f32,
    /// Sine of the incidence angle in air.
    sin_incidence: f32,
    /// Screen-space refraction gain.
    strength: f32,
    /// Standalone channel index.
    channel_ior: f32,
    /// Flattened `RgbIor.r` index.
    ior_r: f32,
    /// Flattened `RgbIor.g` index.
    ior_g: f32,
    /// Flattened `RgbIor.b` index.
    ior_b: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one resolved sample: the nine continuous
/// outputs plus three pad words to a `48`-byte, `16`-aligned stride matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `cauchy_ior(a, b, wavelength_um)`.
    cauchy: f32,
    /// Red reference-wavelength index.
    spectral_r: f32,
    /// Green reference-wavelength index.
    spectral_g: f32,
    /// Blue reference-wavelength index.
    spectral_b: f32,
    /// Standalone `Snell` transmitted sine.
    channel_sine: f32,
    /// Red-minus-blue colour spread.
    spread: f32,
    /// Red-channel screen-space offset.
    offset_r: f32,
    /// Green-channel screen-space offset.
    offset_g: f32,
    /// Blue-channel screen-space offset.
    offset_b: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One dispersion-sample query for the twin: the `Cauchy` law parameters, a
/// standalone wavelength and channel index to probe, the incidence sine, the
/// refraction strength, and the flattened
/// [`RgbIor`](prism_render_architecture::water::dispersion::RgbIor) the
/// spread/offset functions read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterDispersionOffsetsQuery {
    /// `Cauchy` baseline index `a` (about `1.324` for water).
    pub cauchy_a: f32,
    /// `Cauchy` dispersion coefficient `b` in micrometre-squared.
    pub cauchy_b: f32,
    /// Standalone wavelength in micrometres for the `cauchy_ior` probe.
    pub wavelength_um: f32,
    /// Sine of the incidence angle in air; clamped to `0..=1`.
    pub sin_incidence: f32,
    /// Screen-space refraction gain folded with water thickness.
    pub strength: f32,
    /// Standalone channel index for the `channel_transmitted_sine` probe.
    pub channel_ior: f32,
    /// Red-channel index (flattened `RgbIor.r`).
    pub ior_r: f32,
    /// Green-channel index (flattened `RgbIor.g`).
    pub ior_g: f32,
    /// Blue-channel index (flattened `RgbIor.b`).
    pub ior_b: f32,
}

/// One resolved dispersion sample, mirroring the golden
/// [`dispersion`](prism_render_architecture::water::dispersion) per-function
/// outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterDispersionOffsetsResult {
    /// `cauchy_ior(a, b, wavelength_um)`.
    pub cauchy: f32,
    /// Red reference-wavelength index from `spectral_iors`.
    pub spectral_r: f32,
    /// Green reference-wavelength index from `spectral_iors`.
    pub spectral_g: f32,
    /// Blue reference-wavelength index from `spectral_iors`.
    pub spectral_b: f32,
    /// Standalone `channel_transmitted_sine(sin_incidence, channel_ior)`.
    pub channel_sine: f32,
    /// Red-minus-blue colour spread from `dispersion_spread`.
    pub spread: f32,
    /// Red-channel screen-space offset from `dispersion_offsets`.
    pub offset_r: f32,
    /// Green-channel screen-space offset from `dispersion_offsets`.
    pub offset_g: f32,
    /// Blue-channel screen-space offset from `dispersion_offsets`.
    pub offset_b: f32,
}

/// Encodes one [`WaterDispersionOffsetsQuery`] into its `std430` [`GpuQuery`].
fn encode_query(q: &WaterDispersionOffsetsQuery) -> GpuQuery {
    GpuQuery {
        cauchy_a: q.cauchy_a,
        cauchy_b: q.cauchy_b,
        wavelength_um: q.wavelength_um,
        sin_incidence: q.sin_incidence,
        strength: q.strength,
        channel_ior: q.channel_ior,
        ior_r: q.ior_r,
        ior_g: q.ior_g,
        ior_b: q.ior_b,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterDispersionOffsetsResult`].
fn decode_result(raw: &GpuResult) -> WaterDispersionOffsetsResult {
    WaterDispersionOffsetsResult {
        cauchy: raw.cauchy,
        spectral_r: raw.spectral_r,
        spectral_g: raw.spectral_g,
        spectral_b: raw.spectral_b,
        channel_sine: raw.channel_sine,
        spread: raw.spread,
        offset_r: raw.offset_r,
        offset_g: raw.offset_g,
        offset_b: raw.offset_b,
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

/// A compiled, reusable spectral-dispersion compute pipeline, twinning the `CPU`
/// golden [`dispersion`](prism_render_architecture::water::dispersion) numeric
/// functions.
pub struct GpuWaterDispersionOffsets {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterDispersionOffsets {
    /// Compiles the dispersion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterDispersionOffsets {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets"),
            source: ShaderSource::Wgsl(WATER_DISPERSION_OFFSETS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterDispersionOffsets {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every refraction sample and returns one
    /// [`WaterDispersionOffsetsResult`] per input, in order.
    ///
    /// Each result equals the reference's per-sample output within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterDispersionOffsetsQuery],
    ) -> Vec<WaterDispersionOffsetsResult> {
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
            label: Some("prism_volumetric_water_dispersion_offsets_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_bind_group"),
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
            label: Some("prism_volumetric_water_dispersion_offsets_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_dispersion_offsets_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_dispersion_offsets_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per refraction sample, flattened to a 1-D dispatch.
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
