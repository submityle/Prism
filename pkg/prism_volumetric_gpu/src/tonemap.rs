//! `wgpu` compute twin of the `HDR`-to-`LDR` `tonemap` contract
//! ([`tonemap`](prism_render_architecture::particle::tonemap), particle design
//! §16, §21).
//!
//! The `CPU` golden
//! [`tonemap`](prism_render_architecture::particle::tonemap) owns the
//! deterministic maths the particle post/compositing stack shares: a linear
//! exposure scale
//! ([`apply_exposure`](prism_render_architecture::particle::tonemap::apply_exposure),
//! [`exposure_from_stops`](prism_render_architecture::particle::tonemap::exposure_from_stops)),
//! three tone curves
//! ([`reinhard`](prism_render_architecture::particle::tonemap::reinhard),
//! [`reinhard_extended`](prism_render_architecture::particle::tonemap::reinhard_extended),
//! [`aces_film`](prism_render_architecture::particle::tonemap::aces_film)), the
//! square-root display-`gamma` encode and its inverse
//! ([`linear_to_srgb_approx`](prism_render_architecture::particle::tonemap::linear_to_srgb_approx),
//! [`srgb_to_linear`](prism_render_architecture::particle::tonemap::srgb_to_linear)),
//! and the full per-channel pipeline
//! ([`TonemapParams::map`](prism_render_architecture::particle::tonemap::TonemapParams::map)).
//! [`GpuTonemap`] is the on-device twin: one thread per query reproduces the
//! `exposure` -> operator -> encode chain and side-covers every scalar operator,
//! so a passing real-device parity test is direct evidence the ported kernel
//! evaluates the same rational-polynomial maths and selects the same operator
//! branch the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries one linear `HDR` `rgb` triple plus the scalar inputs the
//! operators consume (an `HDR` `scalar`, an `encoded` channel in `[0, 1]`, an
//! `exposure` multiplier, a `white` point, an operator code and an integer
//! `stops` count). The kernel returns, per query: the full
//! [`TonemapParams::map`](prism_render_architecture::particle::tonemap::TonemapParams::map)
//! output, the bare [`apply_exposure`](prism_render_architecture::particle::tonemap::apply_exposure)
//! product, the [`exposure_from_stops`](prism_render_architecture::particle::tonemap::exposure_from_stops)
//! factor, each tone curve evaluated at `scalar`, the `gamma` encode of
//! `scalar`, the decode of `encoded`, and the selected
//! [`TonemapOperator::apply`](prism_render_architecture::particle::tonemap::TonemapOperator::apply).
//! The host-only `std430` packing helpers the reference exposes (`to_std430`,
//! `pack_slice`, `gpu_storage_bytes`) are not kernel math and are left to the
//! `CPU` reference.
//!
//! # Correctness model
//!
//! Every curve is a rational polynomial or a square-root `gamma`
//! approximation, so evaluation touches only `+ - * /`, [`f32::sqrt`] and
//! integer shifts — no transcendental and no `round`/`ceil`. `CPU` and `GPU`
//! evaluate the same closed form in the same associativity, but they are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts `abs_diff <= 1e-4` or `rel_diff <= 1e-3`
//! on every continuous quantity, tight enough to catch a genuinely wrong port
//! (a dropped term, a swapped coefficient, a wrong operator branch) yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The reference guards each division: a denominator within `MIN_DENOM` of zero
//! yields `0.0`, a `white` point is floored to `MIN_WHITE` before squaring, and
//! `exposure_from_stops` clamps its magnitude to `MAX_STOPS` before the shift.
//! The kernel mirrors all three guards with the same `< MIN_DENOM` test (not an
//! `f32` `==`) so both devices stay on the same side of every crack. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`/`max`/`clamp`,
//! `+ - * /`, unsigned/signed index math, integer shifts and [`f32::sqrt`] —
//! with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, inverse trigonometry,
//! `smoothstep`, `round` or `cbrt`, and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop over a runtime
//! count: each thread runs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
//! `Reinhard`, extended `Reinhard` and Narkowicz `ACES` tone curves plus a
//! `wgpu` compute dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::tonemap::{
    aces_film, apply_exposure, exposure_from_stops, linear_to_srgb_approx, reinhard,
    reinhard_extended, srgb_to_linear, TonemapOperator, TonemapParams,
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

/// The portable core-`WGSL` `tonemap` kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden
/// [`tonemap`](prism_render_architecture::particle::tonemap) branch for branch;
/// see the module documentation for the algorithm.
const TONEMAP_WGSL: &str = r#"
// HDR -> LDR tonemap twin: one thread per query reproduces the exposure ->
// operator -> gamma-encode pipeline and side-covers every scalar operator the
// CPU golden particle::tonemap exposes. It mirrors the reference branch for
// branch, uses only the portable core-WGSL subset (min/max/clamp, + - * /,
// integer shifts and sqrt), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop,
// so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::tonemap; no
// third-party engine source or derived code.

// Denominators with magnitude below this are treated as zero, matching the
// reference MIN_DENOM so evaluation never divides by (near) zero.
const MIN_DENOM: f32 = 0.000001;

// A white point is floored to at least this before squaring, matching the
// reference MIN_WHITE so a degenerate zero white cannot collapse the extended
// Reinhard denominator.
const MIN_WHITE: f32 = 0.0001;

// The largest exposure stop magnitude honored by exposure_from_stops, matching
// the reference MAX_STOPS so the integer shift stays in range.
const MAX_STOPS: i32 = 31;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Linear HDR RGB input for the full map pipeline and apply_exposure.
    rgb: vec3<f32>,
    // HDR scalar fed to the per-channel tone curves and the gamma encode.
    scalar: f32,
    // Linear exposure multiplier applied before the tone curve.
    exposure: f32,
    // White luminance control for the extended Reinhard operator.
    white: f32,
    // Already-encoded channel in [0, 1] fed to srgb_to_linear.
    encoded: f32,
    pad0: f32,
    // Numeric tone-curve operator code (0 Reinhard, 1 extended, 2 ACES).
    op_code: u32,
    // Integer photographic stops fed to exposure_from_stops.
    stops: i32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Full map pipeline output: exposure -> operator -> gamma encode per channel.
    mapped: vec3<f32>,
    pad0: f32,
    // Bare apply_exposure product rgb * exposure.
    exposed: vec3<f32>,
    pad1: f32,
    // exposure_from_stops(stops).
    exp_from_stops: f32,
    // reinhard(scalar).
    reinhard_s: f32,
    // reinhard_extended(scalar, white).
    reinhard_ext: f32,
    // aces_film(scalar).
    aces_s: f32,
    // linear_to_srgb_approx(scalar).
    encode_s: f32,
    // srgb_to_linear(encoded).
    decode_s: f32,
    // Selected operator applied to scalar with the white point.
    op_apply: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// A numerically guarded division mirroring the reference `safe_div`: returns
// 0.0 when the denominator is within MIN_DENOM of zero, otherwise the true
// quotient, keeping every curve finite for pathological inputs.
fn safe_div(numerator: f32, denominator: f32) -> f32 {
    if (abs(denominator) < MIN_DENOM) {
        return 0.0;
    }
    return numerator / denominator;
}

// The multiplicative exposure factor for an integer number of photographic
// stops (2^stops), computed with an integer bit shift rather than a forbidden
// pow, mirroring the reference `exposure_from_stops`. The magnitude is clamped
// to MAX_STOPS so the shift stays in range.
fn exposure_from_stops(stops: i32) -> f32 {
    let magnitude = min(abs(stops), MAX_STOPS);
    let shift = u32(magnitude);
    let factor_bits = 1u << shift;
    let factor = f32(factor_bits);
    if (stops >= 0) {
        return factor;
    }
    return safe_div(1.0, factor);
}

// The classic Reinhard tone curve x / (1 + x), mirroring the reference
// `reinhard`.
fn reinhard(x: f32) -> f32 {
    return safe_div(x, 1.0 + x);
}

// The extended Reinhard tone curve x * (1 + x / white^2) / (1 + x), mirroring
// the reference `reinhard_extended`. The white point is floored to MIN_WHITE
// before squaring so the denominator never collapses.
fn reinhard_extended(x: f32, white: f32) -> f32 {
    let w = max(white, MIN_WHITE);
    let white_sq = w * w;
    let numerator = x * (1.0 + safe_div(x, white_sq));
    return safe_div(numerator, 1.0 + x);
}

// The Narkowicz ACES filmic approximation, mirroring the reference
// `aces_film`: clamp((x * (2.51*x + 0.03)) / (x * (2.43*x + 0.59) + 0.14), 0, 1).
fn aces_film(x: f32) -> f32 {
    let numerator = x * (2.51 * x + 0.03);
    let denominator = x * (2.43 * x + 0.59) + 0.14;
    return clamp(safe_div(numerator, denominator), 0.0, 1.0);
}

// Encodes a linear channel into display gamma with the square-root sRGB
// approximation, mirroring the reference `linear_to_srgb_approx`. The input is
// clamped into [0, 1] first.
fn linear_to_srgb_approx(linear: f32) -> f32 {
    return sqrt(clamp(linear, 0.0, 1.0));
}

// The exact inverse of linear_to_srgb_approx: squares an encoded channel back
// into linear space, mirroring the reference `srgb_to_linear`.
fn srgb_to_linear(encoded: f32) -> f32 {
    let e = clamp(encoded, 0.0, 1.0);
    return e * e;
}

// Applies the operator selected by `code` to one already-exposed channel,
// threading `white` through only when the operator uses it, mirroring the
// reference `TonemapOperator::apply` match (code 0 is the Reinhard default).
fn apply_operator(code: u32, x: f32, white: f32) -> f32 {
    if (code == 1u) {
        return reinhard_extended(x, white);
    }
    if (code == 2u) {
        return aces_film(x);
    }
    return reinhard(x);
}

// Runs the full map pipeline on a linear HDR RGB triple, mirroring the
// reference `TonemapParams::map`: multiply by exposure, apply the selected
// operator per channel clamped into [0, 1], then encode to display gamma.
fn map_rgb(rgb: vec3<f32>, exposure: f32, white: f32, code: u32) -> vec3<f32> {
    let exposed = rgb * exposure;
    var out: vec3<f32>;
    out.x = linear_to_srgb_approx(clamp(apply_operator(code, exposed.x, white), 0.0, 1.0));
    out.y = linear_to_srgb_approx(clamp(apply_operator(code, exposed.y, white), 0.0, 1.0));
    out.z = linear_to_srgb_approx(clamp(apply_operator(code, exposed.z, white), 0.0, 1.0));
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.mapped = map_rgb(q.rgb, q.exposure, q.white, q.op_code);
    out.exposed = q.rgb * q.exposure;
    out.exp_from_stops = exposure_from_stops(q.stops);
    out.reinhard_s = reinhard(q.scalar);
    out.reinhard_ext = reinhard_extended(q.scalar, q.white);
    out.aces_s = aces_film(q.scalar);
    out.encode_s = linear_to_srgb_approx(q.scalar);
    out.decode_s = srgb_to_linear(q.encoded);
    out.op_apply = apply_operator(q.op_code, q.scalar, q.white);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One `tonemap` query bundling every input the reference operators consume for
/// a single evaluation: the linear `HDR` `rgb` triple plus the scalar inputs
/// the per-channel curves, the `gamma` encode/decode and `exposure_from_stops`
/// read.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TonemapQuery {
    /// Linear `HDR` `RGB` input for the full map pipeline and `apply_exposure`.
    pub rgb: [f32; 3],
    /// `HDR` scalar fed to the per-channel tone curves and the `gamma` encode.
    pub scalar: f32,
    /// Already-encoded channel in `[0, 1]` fed to `srgb_to_linear`.
    pub encoded: f32,
    /// Linear exposure multiplier applied before the tone curve.
    pub exposure: f32,
    /// White luminance control for the extended `Reinhard` operator.
    pub white: f32,
    /// Which tone curve the map pipeline and `operator_apply` select.
    pub operator: TonemapOperator,
    /// Integer photographic stops fed to `exposure_from_stops`.
    pub stops: i32,
}

impl TonemapQuery {
    /// Builds a query from the `RGB` triple, the scalar inputs, the exposure and
    /// white controls, the selected operator and the stop count.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        rgb: [f32; 3],
        scalar: f32,
        encoded: f32,
        exposure: f32,
        white: f32,
        operator: TonemapOperator,
        stops: i32,
    ) -> TonemapQuery {
        TonemapQuery {
            rgb,
            scalar,
            encoded,
            exposure,
            white,
            operator,
            stops,
        }
    }
}

/// The resolved answer for one query, mirroring every operator the reference
/// exposes: the full pipeline output, the bare exposure product, and each
/// scalar operator evaluated independently.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TonemapResult {
    /// Full map pipeline output, matching
    /// [`TonemapParams::map`](prism_render_architecture::particle::tonemap::TonemapParams::map).
    pub mapped: [f32; 3],
    /// Bare exposure product, matching
    /// [`apply_exposure`](prism_render_architecture::particle::tonemap::apply_exposure).
    pub exposed: [f32; 3],
    /// Exposure factor for the integer stop count, matching
    /// [`exposure_from_stops`](prism_render_architecture::particle::tonemap::exposure_from_stops).
    pub exposure_from_stops: f32,
    /// Plain `Reinhard` tone curve at `scalar`, matching
    /// [`reinhard`](prism_render_architecture::particle::tonemap::reinhard).
    pub reinhard: f32,
    /// Extended `Reinhard` tone curve at `scalar`, matching
    /// [`reinhard_extended`](prism_render_architecture::particle::tonemap::reinhard_extended).
    pub reinhard_extended: f32,
    /// Narkowicz `ACES` tone curve at `scalar`, matching
    /// [`aces_film`](prism_render_architecture::particle::tonemap::aces_film).
    pub aces: f32,
    /// `gamma` encode of `scalar`, matching
    /// [`linear_to_srgb_approx`](prism_render_architecture::particle::tonemap::linear_to_srgb_approx).
    pub encode: f32,
    /// `gamma` decode of `encoded`, matching
    /// [`srgb_to_linear`](prism_render_architecture::particle::tonemap::srgb_to_linear).
    pub decode: f32,
    /// Selected operator applied to `scalar`, matching
    /// [`TonemapOperator::apply`](prism_render_architecture::particle::tonemap::TonemapOperator::apply).
    pub operator_apply: f32,
}

/// Evaluates the `CPU` golden for one query, producing every operator the
/// on-device twin reproduces.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &TonemapQuery) -> TonemapResult {
    let params = TonemapParams::new(query.exposure, query.white, query.operator);
    TonemapResult {
        mapped: params.map(query.rgb),
        exposed: apply_exposure(query.rgb, query.exposure),
        exposure_from_stops: exposure_from_stops(query.stops),
        reinhard: reinhard(query.scalar),
        reinhard_extended: reinhard_extended(query.scalar, query.white),
        aces: aces_film(query.scalar),
        encode: linear_to_srgb_approx(query.scalar),
        decode: srgb_to_linear(query.encoded),
        operator_apply: query.operator.apply(query.scalar, query.white),
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots holding the
/// `vec3` `rgb` with the scalar inputs on its padding lanes and the operator
/// code plus stop count, exactly as the `WGSL` `Query` struct reads it — `48`
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear `HDR` `RGB` input.
    rgb: [f32; 3],
    /// `HDR` scalar for the per-channel curves and the encode.
    scalar: f32,
    /// Linear exposure multiplier.
    exposure: f32,
    /// White luminance control.
    white: f32,
    /// Already-encoded channel in `[0, 1]`.
    encoded: f32,
    /// Padding lane completing the second `vec4` slot.
    pad0: f32,
    /// Numeric operator code.
    op_code: u32,
    /// Integer photographic stops.
    stops: i32,
    /// Padding word.
    pad1: u32,
    /// Padding word completing the third `vec4` slot.
    pad2: u32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &TonemapQuery) -> GpuQuery {
        GpuQuery {
            rgb: query.rgb,
            scalar: query.scalar,
            exposure: query.exposure,
            white: query.white,
            encoded: query.encoded,
            pad0: 0.0,
            op_code: query.operator.code(),
            stops: query.stops,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four `vec4` slots holding the two
/// `vec3` triples on their aligned slots and the seven scalar operator outputs,
/// matching the `WGSL` `Result` struct — `64` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Full map pipeline output.
    mapped: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Bare exposure product.
    exposed: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// `exposure_from_stops` factor.
    exp_from_stops: f32,
    /// Plain `Reinhard` at the scalar.
    reinhard_s: f32,
    /// Extended `Reinhard` at the scalar.
    reinhard_ext: f32,
    /// `ACES` at the scalar.
    aces_s: f32,
    /// `gamma` encode of the scalar.
    encode_s: f32,
    /// `gamma` decode of the encoded channel.
    decode_s: f32,
    /// Selected operator applied to the scalar.
    op_apply: f32,
    /// Padding lane completing the fourth `vec4` slot.
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

/// Decodes one packed [`GpuResult`] into the public [`TonemapResult`].
fn decode_result(raw: &GpuResult) -> TonemapResult {
    TonemapResult {
        mapped: raw.mapped,
        exposed: raw.exposed,
        exposure_from_stops: raw.exp_from_stops,
        reinhard: raw.reinhard_s,
        reinhard_extended: raw.reinhard_ext,
        aces: raw.aces_s,
        encode: raw.encode_s,
        decode: raw.decode_s,
        operator_apply: raw.op_apply,
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

/// A compiled, reusable `tonemap` compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
/// no third-party engine source or derived code.
pub struct GpuTonemap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTonemap {
    /// Compiles the `tonemap` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTonemap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tonemap"),
            source: ShaderSource::Wgsl(TONEMAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tonemap_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tonemap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tonemap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTonemap {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`TonemapResult`] per input,
    /// in order.
    ///
    /// Each result equals the reference operators to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::tonemap`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[TonemapQuery]) -> Vec<TonemapResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tonemap_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tonemap_output"),
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
            label: Some("prism_volumetric_tonemap_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tonemap_bind_group"),
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
            label: Some("prism_volumetric_tonemap_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_tonemap_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tonemap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
