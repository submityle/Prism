//! `wgpu` compute twin of the spectral melanin absorption sampler
//! ([`spectral_absorption`](prism_render_architecture::hair::spectral_absorption)).
//!
//! A spectral hair renderer needs the full `lambda -> sigma_a` absorption curve
//! rather than the three RGB-primary coefficients the RGB pigment model carries.
//! The `CPU` golden samples two fixed per-pigment spectra
//! ([`EUMELANIN_SPECTRUM`](prism_render_architecture::hair::spectral_absorption::EUMELANIN_SPECTRUM),
//! [`PHEOMELANIN_SPECTRUM`](prism_render_architecture::hair::spectral_absorption::PHEOMELANIN_SPECTRUM))
//! at an arbitrary wavelength by clamped linear interpolation, then forms the
//! per-fibre coefficient as a concentration-weighted sum. It is a deterministic,
//! material-independent table lookup with no transcendental math, no `64`-bit
//! integers, and no loops over variable-length data, so it ports to the device
//! directly.
//!
//! [`GpuHairSpectralAbsorption`] is the on-device twin: one thread samples one
//! wavelength for one fibre, reproducing
//! [`eumelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::eumelanin_sigma_a_at),
//! [`pheomelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::pheomelanin_sigma_a_at)
//! and
//! [`melanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::melanin_sigma_a_at).
//! A passing real-device parity test is direct evidence the ported kernel
//! reproduces the clamp, the interpolation, and the concentration weighting the
//! reference performs, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query holding pigment concentrations `(eumelanin, pheomelanin)` and a
//! wavelength `lambda` in `nanometres`, the kernel reproduces:
//! - the per-unit eumelanin coefficient `eumelanin_sigma_a_at(lambda)`;
//! - the per-unit pheomelanin coefficient `pheomelanin_sigma_a_at(lambda)`; and
//! - the combined coefficient
//!   `clamp(eumelanin) * eumelanin_sigma_a_at(lambda) + clamp(pheomelanin) *
//!   pheomelanin_sigma_a_at(lambda)`.
//!
//! The wavelength is first sanitized: a finite `lambda` is clamped into the
//! sampled band `[380, 730]` `nanometres`, while a non-finite `lambda` folds to
//! the violet endpoint, matching the golden `sanitized_wavelength`. Each
//! concentration is clamped by the golden `clamp_concentration`: a finite
//! positive value passes through, while a non-positive or non-finite value folds
//! to `0`. The finite test is `abs(x) <= f32::MAX`, which is `true` for every
//! finite value and `false` for `NaN` (which compares `false` against every
//! bound) and `+/- inf` (whose magnitude exceeds the largest finite `f32`).
//!
//! # What stays on the host
//!
//! The reference `spectrum_sample_map` resamples a variable-length slice of
//! wavelengths; that length-dependent fan-out stays on the host, which simply
//! enqueues one query per wavelength. The `Hero-wavelength` stratified sampler
//! and its `std430` packing helpers are classification/layout utilities, not
//! part of this per-sample kernel. An empty batch short-circuits on the host,
//! since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The kernel performs only clamps, a `floor`, a divide by a fixed non-zero
//! step, and a linear blend of two tabulated values, so a correct port
//! reproduces the reference curve to within floating-point rounding. The parity
//! test asserts the shared continuous tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on every output. Fixtures keep the interpolation
//! parameter away from the exact sample-grid boundaries so a `floor` landing a
//! unit in the last place either side still agrees within tolerance.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `floor`, `+`, `-`, `*`, `/`, an `f32` magnitude comparison and `u32` table
//! indexing — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt`, no `round`, and no `64`-bit integers. No optional
//! device feature is required, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::spectral_absorption`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` spectral absorption sampler, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `melanin_sigma_a_at` (and the two per-pigment samplers it
/// builds on); see the module documentation for the algorithm.
const HAIR_SPECTRAL_ABSORPTION_WGSL: &str = r#"
// Spectral melanin absorption twin: one thread samples the two fixed per-pigment
// spectra at one wavelength by clamped linear interpolation, then forms the
// concentration-weighted combined coefficient, mirroring the CPU golden
// `hair::spectral_absorption::{eumelanin_sigma_a_at, pheomelanin_sigma_a_at,
// melanin_sigma_a_at}` with only abs, clamp, floor, a fixed-step divide and a
// linear blend of tabulated values.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::spectral_absorption；无第三方
// 引擎源码或衍生代码。

// Largest finite f32 (f32::MAX). A finite value has magnitude at most this;
// +/- inf exceeds it and NaN compares false against it.
const F32_MAX: f32 = 3.40282347e38;

// Sampled visible band and sample count, matching the golden SPECTRUM_* consts.
const SPECTRUM_SAMPLES: u32 = 15u;
const SPECTRUM_MIN_NM: f32 = 380.0;
const SPECTRUM_MAX_NM: f32 = 730.0;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Eumelanin concentration (per-unit), sanitized device-side.
    eumelanin: f32,
    // Pheomelanin concentration (per-unit), sanitized device-side.
    pheomelanin: f32,
    // Wavelength in nanometres, sanitized/clamped device-side.
    wavelength_nm: f32,
    pad0: u32,
}

struct Result {
    // Per-unit eumelanin absorption at the wavelength.
    eumelanin_sigma: f32,
    // Per-unit pheomelanin absorption at the wavelength.
    pheomelanin_sigma: f32,
    // Combined, concentration-weighted absorption coefficient.
    combined_sigma: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns one tabulated per-unit coefficient. `pigment` selects the spectrum
// (0 = eumelanin, 1 = pheomelanin); `i` is the sample index in [0, 14]. The
// tables are copied verbatim from the golden EUMELANIN_SPECTRUM /
// PHEOMELANIN_SPECTRUM.
fn table(pigment: u32, i: u32) -> f32 {
    var eu = array<f32, 15>(
        1.95, 1.75, 1.56, 1.40, 1.18, 0.92, 0.70, 0.60,
        0.52, 0.46, 0.419, 0.39, 0.37, 0.35, 0.33
    );
    var ph = array<f32, 15>(
        1.30, 1.20, 1.12, 1.06, 0.85, 0.60, 0.40, 0.33,
        0.27, 0.22, 0.187, 0.17, 0.155, 0.14, 0.13
    );
    if (pigment == 0u) {
        return eu[i];
    }
    return ph[i];
}

// Reproduces the golden `sanitized_wavelength`: a finite wavelength clamps into
// the sampled band; a non-finite one folds to the violet endpoint.
fn sanitized_wavelength(wavelength_nm: f32) -> f32 {
    if (abs(wavelength_nm) <= F32_MAX) {
        return clamp(wavelength_nm, SPECTRUM_MIN_NM, SPECTRUM_MAX_NM);
    }
    return SPECTRUM_MIN_NM;
}

// Reproduces the golden `clamp_concentration`: a finite positive value passes
// through; a non-positive or non-finite value folds to 0. The short-circuit &&
// means NaN (abs(c) <= F32_MAX is false) never reaches the c > 0 test.
fn clamp_concentration(c: f32) -> f32 {
    if (abs(c) <= F32_MAX && c > 0.0) {
        return c;
    }
    return 0.0;
}

// Reproduces the golden `sample_spectrum`: clamped linear interpolation of a
// uniformly sampled spectrum, with the top endpoint returned directly when the
// interpolation parameter lands on or past the last sample.
fn sample_at(wavelength_nm: f32, pigment: u32) -> f32 {
    let clamped = sanitized_wavelength(wavelength_nm);
    let span = SPECTRUM_MAX_NM - SPECTRUM_MIN_NM;
    let step = span / (f32(SPECTRUM_SAMPLES) - 1.0);
    let position = (clamped - SPECTRUM_MIN_NM) / step;
    let lower_f = floor(position);
    let lower = u32(lower_f);
    if (lower >= SPECTRUM_SAMPLES - 1u) {
        return table(pigment, SPECTRUM_SAMPLES - 1u);
    }
    let frac = position - lower_f;
    return table(pigment, lower) * (1.0 - frac) + table(pigment, lower + 1u) * frac;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let eu_sigma = sample_at(q.wavelength_nm, 0u);
    let ph_sigma = sample_at(q.wavelength_nm, 1u);
    let eu_c = clamp_concentration(q.eumelanin);
    let ph_c = clamp_concentration(q.pheomelanin);

    var out: Result;
    out.eumelanin_sigma = eu_sigma;
    out.pheomelanin_sigma = ph_sigma;
    out.combined_sigma = eu_c * eu_sigma + ph_c * ph_sigma;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`HAIR_SPECTRAL_ABSORPTION_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the two pigment concentrations and
/// the wavelength plus one pad word, a `16`-byte stride matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Eumelanin concentration (per-unit) before clamping.
    eumelanin: f32,
    /// Pheomelanin concentration (per-unit) before clamping.
    pheomelanin: f32,
    /// Wavelength in `nanometres` before sanitizing.
    wavelength_nm: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the two per-pigment coefficients and the combined coefficient plus one pad
/// word, a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Per-unit eumelanin absorption at the wavelength.
    eumelanin_sigma: f32,
    /// Per-unit pheomelanin absorption at the wavelength.
    pheomelanin_sigma: f32,
    /// Combined, concentration-weighted absorption coefficient.
    combined_sigma: f32,
    /// Padding word.
    pad0: u32,
}

/// One spectral absorption query for the twin: a fibre's pigment concentrations
/// and the wavelength at which to sample its absorption.
///
/// The host enqueues one [`HairSpectralAbsorptionQuery`] per wavelength,
/// mirroring the inputs of the reference
/// [`melanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::melanin_sigma_a_at).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairSpectralAbsorptionQuery {
    /// Eumelanin concentration (per-unit); clamped to a finite, non-negative
    /// value device-side.
    pub eumelanin: f32,
    /// Pheomelanin concentration (per-unit); clamped to a finite, non-negative
    /// value device-side.
    pub pheomelanin: f32,
    /// Wavelength in `nanometres`; clamped into the sampled band device-side.
    pub wavelength_nm: f32,
}

impl HairSpectralAbsorptionQuery {
    /// Builds a query from pigment concentrations and a wavelength in
    /// `nanometres`.
    #[must_use]
    pub const fn new(
        eumelanin: f32,
        pheomelanin: f32,
        wavelength_nm: f32,
    ) -> HairSpectralAbsorptionQuery {
        HairSpectralAbsorptionQuery {
            eumelanin,
            pheomelanin,
            wavelength_nm,
        }
    }
}

/// One sampled absorption triple, mirroring the reference
/// [`eumelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::eumelanin_sigma_a_at),
/// [`pheomelanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::pheomelanin_sigma_a_at)
/// and
/// [`melanin_sigma_a_at`](prism_render_architecture::hair::spectral_absorption::melanin_sigma_a_at).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairSpectralAbsorptionResult {
    /// Per-unit eumelanin absorption at the wavelength.
    pub eumelanin_sigma: f32,
    /// Per-unit pheomelanin absorption at the wavelength.
    pub pheomelanin_sigma: f32,
    /// Combined, concentration-weighted absorption coefficient.
    pub combined_sigma: f32,
}

/// Encodes one [`HairSpectralAbsorptionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &HairSpectralAbsorptionQuery) -> GpuQuery {
    GpuQuery {
        eumelanin: q.eumelanin,
        pheomelanin: q.pheomelanin,
        wavelength_nm: q.wavelength_nm,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`HairSpectralAbsorptionResult`].
fn decode_result(raw: &GpuResult) -> HairSpectralAbsorptionResult {
    HairSpectralAbsorptionResult {
        eumelanin_sigma: raw.eumelanin_sigma,
        pheomelanin_sigma: raw.pheomelanin_sigma,
        combined_sigma: raw.combined_sigma,
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

/// A compiled, reusable spectral absorption compute pipeline, twinning the `CPU`
/// golden spectral melanin sampler from
/// [`spectral_absorption`](prism_render_architecture::hair::spectral_absorption).
pub struct GpuHairSpectralAbsorption {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairSpectralAbsorption {
    /// Compiles the spectral absorption kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairSpectralAbsorption {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption"),
            source: ShaderSource::Wgsl(HAIR_SPECTRAL_ABSORPTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairSpectralAbsorption {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`HairSpectralAbsorptionResult`] per input, in order.
    ///
    /// Each output matches the reference: the wavelength is clamped into the
    /// sampled band, each concentration is clamped to a finite non-negative
    /// value, and the combined coefficient is the concentration-weighted sum of
    /// the two interpolated per-pigment coefficients. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairSpectralAbsorptionQuery],
    ) -> Vec<HairSpectralAbsorptionResult> {
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
            label: Some("prism_volumetric_hair_spectral_absorption_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_bind_group"),
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
            label: Some("prism_volumetric_hair_spectral_absorption_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_spectral_absorption_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_spectral_absorption_pass"),
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
