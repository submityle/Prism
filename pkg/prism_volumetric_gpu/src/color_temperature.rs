//! `wgpu` compute twin of the white-balance / colour-temperature linear-space
//! `RGB` gain contract
//! ([`color_temperature`](prism_render_architecture::particle::color_temperature),
//! particle design §5, §30).
//!
//! The `CPU` golden
//! [`color_temperature`](prism_render_architecture::particle::color_temperature)
//! turns a [`WhiteBalance`](prism_render_architecture::particle::color_temperature::WhiteBalance)
//! setting (a correlated colour temperature in `kelvin` plus a green-magenta
//! `tint`) into a per-channel linear-space `RGB` gain and multiplies that gain
//! onto a pixel. The temperature follows the planckian locus through a
//! **piecewise rational-polynomial** fit evaluated by Horner's method, with no
//! `sin`/`cos`/`exp`/`log`/`pow` anywhere, so the reference stays
//! bit-reproducible against a `GPU` evaluator. [`GpuColorTemperature`] is the
//! on-device twin: one thread resolves one query (its gain and the applied
//! pixel), so a passing real-device parity test is direct evidence the ported
//! kernel evaluates the same locus fit the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each independent query the kernel reproduces the resolved
//! [`rgb_gain`](prism_render_architecture::particle::color_temperature::WhiteBalance::rgb_gain)
//! triple and the
//! [`apply`](prism_render_architecture::particle::color_temperature::WhiteBalance::apply)
//! result (the gain multiplied onto the supplied linear-space `RGB` pixel). The
//! gain itself composes the green-normalised black-body split along the
//! planckian locus with the green-magenta `tint` trade, exactly as the
//! reference does.
//!
//! # Correctness model
//!
//! Every channel threads through the same multiplies, adds and one guarded
//! rational division the reference uses, so `CPU` and `GPU` are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity, tight enough to catch a genuinely wrong port (a
//! dropped term, a swapped coefficient, a missing clamp) yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! The temperature is clamped into `[KELVIN_MIN, KELVIN_MAX]` and the `tint`
//! into `[TINT_MIN, TINT_MAX]` before evaluation, so out-of-range requests
//! resolve to the boundary gain the reference reports. Below `BLUE_ZERO_KELVIN`
//! the blue channel is held at zero (a pure red-orange black body emits no
//! blue), and both the red and blue channels and the magenta multiplier are
//! floored at zero, so no gain is ever negative. The fitted denominators never
//! vanish on `[KELVIN_MIN, KELVIN_MAX]`, so the rational divides never produce a
//! `NaN`. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, no inverse trigonometry, no `sqrt` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::color_temperature`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` colour-temperature kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `resolve` mirrors
/// the `CPU` golden
/// [`color_temperature`](prism_render_architecture::particle::color_temperature)
/// branch for branch; see the module documentation for the algorithm.
const COLOR_TEMPERATURE_WGSL: &str = r#"
// Colour-temperature twin: one thread per query reproduces the green-normalised
// black-body RGB gain (a piecewise rational-polynomial fit of the planckian
// locus) and the gain applied to a linear-space pixel. It mirrors the CPU
// golden `particle::color_temperature` branch for branch, uses only the
// portable core-WGSL subset (clamp/max and + - * / plus unsigned index math),
// calls no sin/cos/exp/log/pow and no sqrt and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::color_temperature；无第三方
// 引擎源码或衍生代码。

// Fit domain and locus knees, matching the reference constants.
const KELVIN_MIN: f32 = 1000.0;
const KELVIN_MAX: f32 = 15000.0;
const BLUE_ZERO_KELVIN: f32 = 1900.0;
const WARM_COOL_SPLIT_KELVIN: f32 = 6600.0;
const TINT_MIN: f32 = -1.0;
const TINT_MAX: f32 = 1.0;
const TINT_GREEN_GAIN: f32 = 0.4;
const TINT_MAGENTA_GAIN: f32 = 0.2;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Linear-space RGB pixel the resolved gain is multiplied onto.
    rgb: vec3<f32>,
    // Target correlated colour temperature in kelvin.
    temp_kelvin: f32,
    // Green-magenta tint in [TINT_MIN, TINT_MAX].
    tint: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Resolved linear-space RGB gain, with a trailing pad lane.
    gain: vec3<f32>,
    pad_g: f32,
    // Gain applied to the input pixel, with a trailing pad lane.
    applied: vec3<f32>,
    pad_a: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Red-channel rational fit over the whole domain (ascending-power coefficients
// evaluated by Horner's method), mirroring the reference `R_NUM` / `R_DEN`.
fn red_gain(x: f32) -> f32 {
    let num = ((-0.0351264f * x + 0.398053f) * x + -1.02807f) * x + -3.09790f;
    let den = ((-0.0449934f * x + 0.601709f) * x + -2.55882f) * x + 1.0f;
    return max(num / den, 0.0f);
}

// Blue-channel warm-branch rational fit (BLUE_ZERO_KELVIN..=WARM_COOL_SPLIT),
// mirroring the reference `BW_NUM` / `BW_DEN`.
fn blue_warm_gain(x: f32) -> f32 {
    let num = ((-0.111423f * x + 1.14996f) * x + -3.12041f) * x + 2.54166f;
    let den = (((8.75e-05f * x + -0.0941453f) * x + 0.870344f) * x + -1.81911f) * x + 1.0f;
    return max(num / den, 0.0f);
}

// Blue-channel cool-branch rational fit (WARM_COOL_SPLIT..=KELVIN_MAX),
// mirroring the reference `BC_NUM` / `BC_DEN`.
fn blue_cool_gain(x: f32) -> f32 {
    let num = -0.158482f * x + 0.911769f;
    let den = (((9.9e-06f * x + -0.000512f) * x + 0.0105773f) * x + -0.222001f) * x + 1.0f;
    return max(num / den, 0.0f);
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Clamp the temperature into the fit domain, then evaluate the
    // green-normalised black-body split in kilokelvin.
    let kelvin = clamp(q.temp_kelvin, KELVIN_MIN, KELVIN_MAX);
    let x = kelvin / 1000.0f;
    let red = red_gain(x);
    var blue: f32 = 0.0f;
    if (kelvin < BLUE_ZERO_KELVIN) {
        blue = 0.0f;
    } else if (kelvin <= WARM_COOL_SPLIT_KELVIN) {
        blue = blue_warm_gain(x);
    } else {
        blue = blue_cool_gain(x);
    }

    // Green-magenta tint axis: green rises with tint, magenta (red+blue) falls.
    let tint = clamp(q.tint, TINT_MIN, TINT_MAX);
    let green_mul = 1.0f + tint * TINT_GREEN_GAIN;
    let magenta_mul = max(1.0f - tint * TINT_MAGENTA_GAIN, 0.0f);
    let gain = vec3<f32>(red * magenta_mul, 1.0f * green_mul, blue * magenta_mul);

    let applied = vec3<f32>(q.rgb.x * gain.x, q.rgb.y * gain.y, q.rgb.z * gain.z);

    var out: Result;
    out.gain = gain;
    out.pad_g = 0.0f;
    out.applied = applied;
    out.pad_a = 0.0f;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`COLOR_TEMPERATURE_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The `rgb` lane plus `temp_kelvin` fill one `vec4` slot; the `tint` plus three
/// pad words fill the next.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear-space `RGB` pixel the gain is applied to.
    rgb: [f32; 3],
    /// Target colour temperature in `kelvin`.
    temp_kelvin: f32,
    /// Green-magenta tint.
    tint: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Each `vec3` carries a trailing pad word so it stays `16`-byte
/// aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Resolved linear-space `RGB` gain.
    gain: [f32; 3],
    /// Pad lane after `gain`.
    pad_g: f32,
    /// Gain applied to the input pixel.
    applied: [f32; 3],
    /// Pad lane after `applied`.
    pad_a: f32,
}

/// One query for the colour-temperature twin: a white-balance setting (a
/// temperature and a tint) together with the linear-space `RGB` pixel the
/// resolved gain is multiplied onto.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTemperatureQuery {
    /// Target correlated colour temperature in `kelvin`; values outside the fit
    /// domain clamp before evaluation.
    pub temp_kelvin: f32,
    /// Green-magenta tint; positive lifts green and suppresses magenta, and
    /// values outside `[-1, 1]` clamp before evaluation.
    pub tint: f32,
    /// Linear-space `RGB` pixel the resolved gain is applied to.
    pub rgb: [f32; 3],
}

/// One resolved answer for a single query, mirroring the reference
/// [`rgb_gain`](prism_render_architecture::particle::color_temperature::WhiteBalance::rgb_gain)
/// and
/// [`apply`](prism_render_architecture::particle::color_temperature::WhiteBalance::apply).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorTemperatureResult {
    /// Resolved linear-space `RGB` gain, matching
    /// [`rgb_gain`](prism_render_architecture::particle::color_temperature::WhiteBalance::rgb_gain).
    pub gain: [f32; 3],
    /// Gain applied to the input pixel, matching
    /// [`apply`](prism_render_architecture::particle::color_temperature::WhiteBalance::apply).
    pub applied: [f32; 3],
}

/// Encodes one [`ColorTemperatureQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ColorTemperatureQuery) -> GpuQuery {
    GpuQuery {
        rgb: q.rgb,
        temp_kelvin: q.temp_kelvin,
        tint: q.tint,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ColorTemperatureResult`].
fn decode_result(raw: &GpuResult) -> ColorTemperatureResult {
    ColorTemperatureResult {
        gain: raw.gain,
        applied: raw.applied,
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

/// A compiled, reusable colour-temperature compute pipeline, twinning the `CPU`
/// golden
/// [`color_temperature`](prism_render_architecture::particle::color_temperature).
pub struct GpuColorTemperature {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuColorTemperature {
    /// Compiles the colour-temperature kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuColorTemperature {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_color_temperature"),
            source: ShaderSource::Wgsl(COLOR_TEMPERATURE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_color_temperature_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_color_temperature_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_color_temperature_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuColorTemperature {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`ColorTemperatureResult`] per input, in order.
    ///
    /// Each gain and applied pixel matches the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ColorTemperatureQuery],
    ) -> Vec<ColorTemperatureResult> {
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
            label: Some("prism_volumetric_color_temperature_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_temperature_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_color_temperature_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_color_temperature_bind_group"),
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
            label: Some("prism_volumetric_color_temperature_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_color_temperature_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_color_temperature_pass"),
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
