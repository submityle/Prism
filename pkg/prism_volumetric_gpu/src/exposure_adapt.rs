//! `wgpu` compute twin of the temporal auto-exposure / eye-adaptation contract
//! ([`exposure_adapt`](prism_render_architecture::particle::exposure_adapt),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`exposure_adapt`](prism_render_architecture::particle::exposure_adapt) owns
//! the determinism-locked maths of how the `HDR` shading front end eases a
//! running exposure toward a frame's metered target over time: it meters a
//! *target* `EV` from an average `luminance` against a middle-grey `key`
//! ([`ExposureAdaptConfig::target_exposure`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::target_exposure)),
//! advances the running exposure one frame-rate-independent step toward that
//! target
//! ([`ExposureAdaptConfig::adapt_step`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::adapt_step)
//! and its metering convenience
//! [`ExposureAdaptConfig::adapt_toward_luminance`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::adapt_toward_luminance)),
//! and turns the converged `EV` back into the linear exposure scale a shader
//! multiplies into radiance
//! ([`ev_to_exposure_scale`](prism_render_architecture::particle::exposure_adapt::ev_to_exposure_scale)).
//! [`GpuExposureAdapt`] is the on-device twin: one thread drives one eye, so a
//! passing real-device parity test is direct evidence the ported kernel
//! converges exactly as the reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent eyes each thread reproduces four values the
//! reference computes: the metered `target_ev`
//! ([`ExposureAdaptConfig::target_exposure`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::target_exposure)),
//! the frame-rate-independent blend factor `rate`
//! ([`rate_factor`](prism_render_architecture::particle::exposure_adapt::rate_factor)
//! on the direction-appropriate speed), the converged `next_ev`
//! ([`ExposureAdaptConfig::adapt_toward_luminance`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::adapt_toward_luminance)),
//! and the linear `exposure_scale`
//! ([`ev_to_exposure_scale`](prism_render_architecture::particle::exposure_adapt::ev_to_exposure_scale)
//! of `next_ev`). The reference `adapt_step` is covered through `next_ev`, since
//! `adapt_toward_luminance` is exactly `adapt_step` on the metered target.
//!
//! # Determinism without transcendentals
//!
//! The reference forbids `exp`/`ln`/`powf`, so it reconstructs the `log2` that
//! places a `luminance` ratio on the `EV` axis and the `exp2` that turns an `EV`
//! back into a linear scale from the `IEEE`-754 field layout of `f32`: the
//! biased exponent field supplies the integer octave exactly and a quadratic
//! closes the fractional octave. The temporal blend uses the rational stand-in
//! `rate = 1 - 1 / (1 + speed * dt)` for the exponential relaxation weight. This
//! twin reproduces that exact bit-level recipe in `WGSL` — the same exponent
//! shift, the same mantissa fraction, the same two quadratics — using only
//! `floor`, `abs`, `min`, `max`, add / subtract / multiply, a guarded divide and
//! `bitcast`, and **never** calls the built-in `exp2`/`log2`. No lookup table
//! and no input-dependent iteration count, so the kernel reproduces the `CPU`
//! result on `Metal`, `Vulkan` and `DX12` alike.
//!
//! # Correctness model
//!
//! Every twinned value threads through multiplies, adds, one guarded divide and
//! a `bitcast`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each
//! continuous quantity, tight enough to catch a genuinely wrong port (a dropped
//! term, a wrong clamp, a swapped speed) yet loose enough to admit legal fused
//! multiply-add contraction. Fixtures stay clear of the brighten / darken
//! direction threshold and the clamp edges so both paths select the same branch.
//!
//! # Degenerate inputs
//!
//! A non-positive or subnormal `luminance` ratio pins the metered `EV` to the
//! sentinel [`NEG_EV_FLOOR`](prism_render_architecture::particle::exposure_adapt)
//! floor (so the clamp drives it to the darkest end) instead of emitting
//! `-inf`/`NaN`; both the average `luminance` and the `key` are floored at a
//! small positive epsilon before the divide. An `EV` beyond the representable
//! exponent range saturates the linear scale to `0.0` or [`f32::MAX`] rather
//! than a subnormal or infinity. A zero-length frame (`dt = 0`) yields a zero
//! blend factor, so the exposure does not move. An empty batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::exposure_adapt`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig;
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

/// The portable core-`WGSL` eye-adaptation kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`exposure_adapt`](prism_render_architecture::particle::exposure_adapt)
/// branch for branch; see the module documentation for the algorithm.
const EXPOSURE_ADAPT_WGSL: &str = r#"
// Eye-adaptation twin: one thread per eye reproduces the metered target EV, the
// frame-rate-independent blend factor, the temporally converged next EV, and the
// linear exposure scale. It mirrors the CPU golden
// `particle::exposure_adapt` branch for branch, reconstructs log2/exp2 from the
// IEEE-754 f32 field layout with the reference's own quadratics (never calling
// the built-in exp2/log2), and uses only the portable core-WGSL subset
// (floor/abs/min/max, + - * /, a guarded divide and bitcast), so it runs
// unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::exposure_adapt；无第三方
// 引擎源码或衍生代码。

// Denominators (average luminance, key) with magnitude below this are floored so
// no divide hits zero. Matches the reference `MIN_DENOM`.
const MIN_DENOM: f32 = 1.0e-6;

// Quadratic correction coefficient for the log2 mantissa term:
// log2(1 + f) ~= f + LOG2_MANTISSA_K * f * (1 - f). Matches the reference.
const LOG2_MANTISSA_K: f32 = 0.3465736;

// Linear coefficient of the quadratic exp2 fractional fit
// 2^f ~= 1 + EXP2_FRAC_C1 * f + EXP2_FRAC_C2 * f * f. Matches the reference.
const EXP2_FRAC_C1: f32 = 0.6568542;

// Quadratic coefficient of the exp2 fractional fit. Matches the reference.
const EXP2_FRAC_C2: f32 = 0.3431458;

// 2^23, the width of the f32 mantissa field, as a float divisor.
const MANTISSA_SCALE: f32 = 8388608.0;

// Sentinel EV returned for non-positive / subnormal ratios, far below any
// physical HDR exposure so the clamp pins the target to the darkest end.
const NEG_EV_FLOOR: f32 = -1000.0;

struct Params {
    // Number of eyes in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Darkest and brightest allowed exposure, in EV.
    min_ev: f32,
    max_ev: f32,
    // Convergence speeds while brightening (up) and darkening (down).
    up_speed: f32,
    down_speed: f32,
    // Artist EV bias added to the metered target before clamping.
    exposure_compensation: f32,
    // Running exposure EV at the start of the step.
    current_ev: f32,
    // Average frame luminance and the middle-grey key it meters against.
    avg_luminance: f32,
    key: f32,
    // Frame time, in the same units as the speeds.
    dt: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Metered target EV (target_exposure).
    target_ev: f32,
    // Frame-rate-independent blend factor used for the step (rate_factor).
    rate: f32,
    // Temporally converged EV after one step (adapt_toward_luminance).
    next_ev: f32,
    // Linear exposure scale of next_ev (ev_to_exposure_scale).
    exposure_scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps `value` into [lo, hi] without a min > max panic: lo wins low, hi wins
// high, a degenerate lo > hi collapses to hi. Mirrors the reference `clamp`.
fn clamp_range(value: f32, lo: f32, hi: f32) -> f32 {
    return min(max(value, lo), hi);
}

// Integer-exponent log2 approximation built from the f32 bit pattern. For a
// positive normal x = (1 + f) * 2^e the biased exponent gives e exactly and the
// mantissa fraction f is closed with the reference quadratic. Non-positive or
// subnormal inputs return NEG_EV_FLOOR. No built-in log2 is called.
fn approx_log2(x: f32) -> f32 {
    if (x <= 0.0) {
        return NEG_EV_FLOOR;
    }
    let bits = bitcast<u32>(x);
    let exp_field = i32((bits >> 23u) & 0xffu);
    if (exp_field == 0) {
        // Subnormal: below the smallest normal exposure ratio we ever place.
        return NEG_EV_FLOOR;
    }
    let mantissa = bits & 0x007fffffu;
    let frac = f32(mantissa) / MANTISSA_SCALE;
    let log_mant = frac + LOG2_MANTISSA_K * frac * (1.0 - frac);
    let exponent = f32(exp_field - 127);
    return exponent + log_mant;
}

// Integer-exponent exp2 (2^x), the inverse of approx_log2. The integer part is
// written straight into the exponent field (exact) and the fractional octave is
// closed with the reference quadratic. Out-of-range magnitudes saturate to 0.0 /
// f32::MAX. No built-in exp2 is called; the power of two is a bit shift.
fn ev_to_exposure_scale(ev: f32) -> f32 {
    let fl = floor(ev);
    let frac = ev - fl;
    let mantissa = 1.0 + EXP2_FRAC_C1 * frac + EXP2_FRAC_C2 * frac * frac;
    let exponent = i32(fl);
    if (exponent > 127) {
        // Bit pattern of f32::MAX (0x7f7fffff): saturate instead of +inf.
        return bitcast<f32>(0x7f7fffffu);
    }
    if (exponent < -126) {
        return 0.0;
    }
    let field = u32(exponent + 127);
    let scale = bitcast<f32>(field << 23u);
    return mantissa * scale;
}

// Frame-rate-independent temporal blend factor in [0, 1]:
// rate = 1 - 1 / (1 + speed * dt), with dt and speed floored at 0. Mirrors the
// reference `rate_factor`.
fn rate_factor(dt: f32, speed: f32) -> f32 {
    let d = max(dt, 0.0);
    let s = max(speed, 0.0);
    let product = s * d;
    let raw = 1.0 - 1.0 / (1.0 + product);
    return clamp_range(raw, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Ordered exposure window so a mis-ordered min_ev > max_ev never makes the
    // clamp behave oddly. Mirrors the reference `ev_window`.
    let lo = min(q.min_ev, q.max_ev);
    let hi = max(q.min_ev, q.max_ev);

    // target_exposure: EV of the key / average ratio, floored denominators, plus
    // the artist compensation, clamped into the window.
    let avg = max(q.avg_luminance, MIN_DENOM);
    let key = max(q.key, MIN_DENOM);
    let metered = approx_log2(key / avg) + q.exposure_compensation;
    let target_ev = clamp_range(metered, lo, hi);

    // adapt_step: pick the direction-appropriate speed, build the blend factor,
    // take one convex step toward the target, and clamp the result.
    var speed: f32 = q.down_speed;
    if (target_ev > q.current_ev) {
        speed = q.up_speed;
    }
    let rate = rate_factor(q.dt, speed);
    let next_raw = q.current_ev + (target_ev - q.current_ev) * rate;
    let next_ev = clamp_range(next_raw, lo, hi);

    // EV -> linear exposure scale a shader multiplies into radiance.
    let exposure_scale = ev_to_exposure_scale(next_ev);

    var out: Result;
    out.target_ev = target_ev;
    out.rate = rate;
    out.next_ev = next_ev;
    out.exposure_scale = exposure_scale;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the eye count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`EXPOSURE_ADAPT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid eyes in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one eye, matching the `WGSL` `Query` struct.
/// Twelve scalars (nine inputs plus three pad lanes) span three `vec4` slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Darkest allowed exposure, in `EV`.
    min_ev: f32,
    /// Brightest allowed exposure, in `EV`.
    max_ev: f32,
    /// Brightening convergence speed.
    up_speed: f32,
    /// Darkening convergence speed.
    down_speed: f32,
    /// Artist `EV` bias added to the metered target before clamping.
    exposure_compensation: f32,
    /// Running exposure `EV` at the start of the step.
    current_ev: f32,
    /// Average frame `luminance`.
    avg_luminance: f32,
    /// Middle-grey `key` the average meters against.
    key: f32,
    /// Frame time.
    dt: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Metered target `EV`.
    target_ev: f32,
    /// Frame-rate-independent blend factor used for the step.
    rate: f32,
    /// Temporally converged `EV` after one step.
    next_ev: f32,
    /// Linear exposure scale of `next_ev`.
    exposure_scale: f32,
}

/// One eye for the exposure-adaptation twin: a configuration block plus the
/// per-frame running exposure, metered `luminance`, middle-grey `key` and frame
/// time.
///
/// The `config` reuses the `CPU` golden
/// [`ExposureAdaptConfig`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig)
/// so the twin binds exactly the fields the reference packs; no new config type
/// is introduced here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureAdaptQuery {
    /// Exposure window, convergence speeds and compensation for this eye.
    pub config: ExposureAdaptConfig,
    /// Running exposure `EV` at the start of the step.
    pub current_ev: f32,
    /// Average frame `luminance` handed in from the histogram stage.
    pub avg_luminance: f32,
    /// Middle-grey `key` the average meters against.
    pub key: f32,
    /// Frame time, in the same units as the configuration speeds.
    pub dt: f32,
}

/// One resolved answer for a single eye, mirroring every value the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureAdaptResult {
    /// Metered target `EV`, matching
    /// [`ExposureAdaptConfig::target_exposure`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::target_exposure).
    pub target_ev: f32,
    /// Frame-rate-independent blend factor, matching
    /// [`rate_factor`](prism_render_architecture::particle::exposure_adapt::rate_factor)
    /// on the direction-appropriate speed.
    pub rate: f32,
    /// Temporally converged `EV` after one step, matching
    /// [`ExposureAdaptConfig::adapt_toward_luminance`](prism_render_architecture::particle::exposure_adapt::ExposureAdaptConfig::adapt_toward_luminance).
    pub next_ev: f32,
    /// Linear exposure scale of `next_ev`, matching
    /// [`ev_to_exposure_scale`](prism_render_architecture::particle::exposure_adapt::ev_to_exposure_scale).
    pub exposure_scale: f32,
}

/// Encodes one [`ExposureAdaptQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ExposureAdaptQuery) -> GpuQuery {
    GpuQuery {
        min_ev: q.config.min_ev,
        max_ev: q.config.max_ev,
        up_speed: q.config.up_speed,
        down_speed: q.config.down_speed,
        exposure_compensation: q.config.exposure_compensation,
        current_ev: q.current_ev,
        avg_luminance: q.avg_luminance,
        key: q.key,
        dt: q.dt,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ExposureAdaptResult`].
fn decode_result(raw: &GpuResult) -> ExposureAdaptResult {
    ExposureAdaptResult {
        target_ev: raw.target_ev,
        rate: raw.rate,
        next_ev: raw.next_ev,
        exposure_scale: raw.exposure_scale,
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

/// A compiled, reusable eye-adaptation compute pipeline, twinning the `CPU`
/// golden
/// [`exposure_adapt`](prism_render_architecture::particle::exposure_adapt).
pub struct GpuExposureAdapt {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuExposureAdapt {
    /// Compiles the eye-adaptation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuExposureAdapt {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_exposure_adapt"),
            source: ShaderSource::Wgsl(EXPOSURE_ADAPT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_exposure_adapt_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_exposure_adapt_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_exposure_adapt_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuExposureAdapt {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every eye in `queries` and returns one [`ExposureAdaptResult`]
    /// per input, in order.
    ///
    /// Each value matches the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ExposureAdaptQuery],
    ) -> Vec<ExposureAdaptResult> {
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
            label: Some("prism_volumetric_exposure_adapt_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_exposure_adapt_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_exposure_adapt_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_exposure_adapt_bind_group"),
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
            label: Some("prism_volumetric_exposure_adapt_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_exposure_adapt_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_exposure_adapt_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per eye, flattened to a 1-D dispatch.
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
