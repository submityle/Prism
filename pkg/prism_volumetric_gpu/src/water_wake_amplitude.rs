//! `wgpu` compute twin of the stateless scalar wake primitives in the
//! ship-hull `Kelvin` wake planner
//! ([`wake`](prism_render_architecture::water::wake)).
//!
//! The `CPU` golden exposes three closed-form scalars that feed the larger,
//! variable-length wake-source planner: the deep-water transverse wavelength
//! ([`transverse_wavelength`](prism_render_architecture::water::wake::transverse_wavelength)),
//! the length-based `Froude` number
//! ([`froude_length`](prism_render_architecture::water::wake::froude_length)),
//! and the per-stamp emission amplitude
//! ([`emission_amplitude`](prism_render_architecture::water::wake::emission_amplitude)).
//! Each is a fixed-width arithmetic map over a hull's motion and a
//! [`WakeConfig`](prism_render_architecture::water::wake::WakeConfig), so each
//! has an exact device analogue.
//!
//! [`GpuWaterWakeAmplitude`] is the on-device twin of all three. One thread
//! evaluates one query, reproducing the reference's exact closed form, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same scalars the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one query the kernel reproduces, with only `+ - * /`, `max` and `sqrt`:
//!
//! - `transverse_wavelength`: `g = max(gravity, 0)`; `0` when `g <= EPS`,
//!   otherwise `TWO_PI * max(speed, 0)^2 / g`.
//! - `froude_length`: `denom = max(max(gravity, 0) * max(length_m, 0), 0)`;
//!   `0` when `denom <= EPS`, otherwise `max(speed, 0) / sqrt(denom)`.
//! - `emission_amplitude`: `0` when the config is invalid or
//!   `hull_speed < min_speed`, otherwise
//!   `max(base_amplitude, 0) * speed_factor * draft_scale` where
//!   `drive = max(hull_speed - min_speed, 0)`,
//!   `speed_factor = drive / (drive + reference_speed)` and
//!   `draft_scale = 1 + max(draft_m, 0)`.
//!
//! The validity predicate mirrors
//! [`WakeConfig::is_valid`](prism_render_architecture::water::wake::WakeConfig::is_valid):
//! `gravity > EPS`, `source_spacing_m > EPS`, `trail_length_m > EPS`,
//! `reference_speed > EPS`, `base_amplitude >= 0`, `min_speed >= 0` and
//! `max_sources > 0`.
//!
//! # What stays on the host
//!
//! The surrounding wake-source planner
//! ([`plan_wake_sources`](prism_render_architecture::water::wake)) is a
//! variable-length aggregate — it walks source stations, mirrors divergent
//! arms and caps per-train counts — so it stays on the host; the device sees
//! only the per-query scalar math. An empty `queries` batch short-circuits on
//! the host and issues no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! The three outputs thread through a `sqrt` and guarded divisions, so `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts a
//! tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-4`) on each output, tight
//! enough to catch a genuinely wrong port (a dropped `max`, a swapped field, a
//! wrong constant) yet loose enough to admit a legal last-place `sqrt` or
//! divide difference. Fixtures that straddle the `EPS` gate are supplied as
//! deterministic on-branch cases so the zero paths are covered without pinning
//! parity on a tie.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `sqrt`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round`, no `ceil` and no
//! `floor`. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::wake`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` wake-scalar kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden scalars
/// [`transverse_wavelength`](prism_render_architecture::water::wake::transverse_wavelength),
/// [`froude_length`](prism_render_architecture::water::wake::froude_length) and
/// [`emission_amplitude`](prism_render_architecture::water::wake::emission_amplitude);
/// see the module documentation for the algorithm.
const WATER_WAKE_AMPLITUDE_WGSL: &str = r#"
// Wake scalar twin: one thread reproduces the three stateless scalar wake
// primitives of the CPU golden `water::wake` with only max/sqrt and + - * /.
// It owns no source-train planner: that variable-length aggregate stays host-
// side.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::wake；无第三方引擎
// 源码或衍生代码。

const EPS: f32 = 1.0e-6;
const TWO_PI: f32 = 6.2831855;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // transverse_wavelength / froude_length driving speed and shared gravity.
    speed: f32,
    gravity: f32,
    // froude_length characteristic length.
    length_m: f32,
    // emission_amplitude hull speed.
    hull_speed: f32,
    // WakeConfig scalars read by emission_amplitude and is_valid.
    min_speed: f32,
    reference_speed: f32,
    base_amplitude: f32,
    draft_m: f32,
    source_spacing_m: f32,
    trail_length_m: f32,
    max_sources: u32,
    pad0: u32,
}

struct Result {
    transverse_wavelength: f32,
    froude_length: f32,
    emission_amplitude: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // transverse_wavelength: TWO_PI * v^2 / g, guarded on a non-positive g.
    let g = max(q.gravity, 0.0);
    var wavelength: f32 = 0.0;
    if (g > EPS) {
        let v = max(q.speed, 0.0);
        wavelength = TWO_PI * v * v / g;
    }

    // froude_length: speed / sqrt(g*length), guarded on a tiny denominator.
    let denom = max(max(q.gravity, 0.0) * max(q.length_m, 0.0), 0.0);
    var froude: f32 = 0.0;
    if (denom > EPS) {
        froude = max(q.speed, 0.0) / sqrt(denom);
    }

    // WakeConfig::is_valid: every field finite-positive where required.
    let valid = (q.gravity > EPS)
        && (q.source_spacing_m > EPS)
        && (q.trail_length_m > EPS)
        && (q.reference_speed > EPS)
        && (q.base_amplitude >= 0.0)
        && (q.min_speed >= 0.0)
        && (q.max_sources > 0u);

    // emission_amplitude: zero when invalid or below the cut-in speed.
    var amplitude: f32 = 0.0;
    if (valid && (q.hull_speed >= q.min_speed)) {
        let drive = max(q.hull_speed - q.min_speed, 0.0);
        // reference_speed > EPS under validity, so the denominator is positive.
        let speed_factor = drive / (drive + q.reference_speed);
        let draft_scale = 1.0 + max(q.draft_m, 0.0);
        amplitude = max(q.base_amplitude, 0.0) * speed_factor * draft_scale;
    }

    var out: Result;
    out.transverse_wavelength = wavelength;
    out.froude_length = froude;
    out.emission_amplitude = amplitude;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_WAKE_AMPLITUDE_WGSL`].
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

/// `repr(C)` `std430` layout of one wake-scalar query, matching the `WGSL`
/// `Query` struct: the eleven `f32` inputs plus the `max_sources` count and one
/// pad word to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Driving speed shared by `transverse_wavelength` and `froude_length`.
    speed: f32,
    /// Gravitational acceleration shared by both wavelength laws.
    gravity: f32,
    /// Characteristic length for `froude_length`.
    length_m: f32,
    /// Hull speed for `emission_amplitude`.
    hull_speed: f32,
    /// Cut-in speed below which the hull leaves no wake.
    min_speed: f32,
    /// Speed at which the emission amplitude reaches half its asymptote.
    reference_speed: f32,
    /// Amplitude scale at full drive and unit draft.
    base_amplitude: f32,
    /// Hull draft; deeper hulls displace more water.
    draft_m: f32,
    /// Along-track spacing between emitted source stamps.
    source_spacing_m: f32,
    /// How far behind the hull the wake persists.
    trail_length_m: f32,
    /// Hard cap on the number of stamps per train.
    max_sources: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one wake-scalar result, matching the `WGSL`
/// `Result` struct: the three scalar outputs plus one pad word to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Deep-water transverse wavelength.
    transverse_wavelength: f32,
    /// Length-based `Froude` number.
    froude_length: f32,
    /// Per-stamp emission amplitude.
    emission_amplitude: f32,
    /// Padding word.
    pad0: f32,
}

/// One wake-scalar query: the union of inputs the three twinned golden scalars
/// read.
///
/// `speed` and `gravity` drive
/// [`transverse_wavelength`](prism_render_architecture::water::wake::transverse_wavelength);
/// `speed`, `length_m` and `gravity` drive
/// [`froude_length`](prism_render_architecture::water::wake::froude_length);
/// the remaining fields are the hull and
/// [`WakeConfig`](prism_render_architecture::water::wake::WakeConfig) scalars
/// read by
/// [`emission_amplitude`](prism_render_architecture::water::wake::emission_amplitude)
/// and its validity predicate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterWakeAmplitudeQuery {
    /// Driving speed for the wavelength and `Froude` laws.
    pub speed: f32,
    /// Gravitational acceleration shared by both wavelength laws.
    pub gravity: f32,
    /// Characteristic length for `froude_length`.
    pub length_m: f32,
    /// Hull speed for `emission_amplitude`.
    pub hull_speed: f32,
    /// Cut-in speed below which the hull leaves no wake.
    pub min_speed: f32,
    /// Speed at which the emission amplitude reaches half its asymptote.
    pub reference_speed: f32,
    /// Amplitude scale at full drive and unit draft.
    pub base_amplitude: f32,
    /// Hull draft; deeper hulls displace more water.
    pub draft_m: f32,
    /// Along-track spacing between emitted source stamps.
    pub source_spacing_m: f32,
    /// How far behind the hull the wake persists.
    pub trail_length_m: f32,
    /// Hard cap on the number of stamps per train.
    pub max_sources: u32,
}

/// One resolved wake-scalar triple, mirroring the three `CPU` golden scalars.
///
/// `transverse_wavelength` is the deep-water transverse wavelength,
/// `froude_length` is the length-based `Froude` number, and
/// `emission_amplitude` is the per-stamp emission amplitude; each equals the
/// matching golden free function within the tolerance documented on this
/// module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterWakeAmplitudeResult {
    /// Deep-water transverse wavelength.
    pub transverse_wavelength: f32,
    /// Length-based `Froude` number.
    pub froude_length: f32,
    /// Per-stamp emission amplitude.
    pub emission_amplitude: f32,
}

/// Encodes one [`WaterWakeAmplitudeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterWakeAmplitudeQuery) -> GpuQuery {
    GpuQuery {
        speed: q.speed,
        gravity: q.gravity,
        length_m: q.length_m,
        hull_speed: q.hull_speed,
        min_speed: q.min_speed,
        reference_speed: q.reference_speed,
        base_amplitude: q.base_amplitude,
        draft_m: q.draft_m,
        source_spacing_m: q.source_spacing_m,
        trail_length_m: q.trail_length_m,
        max_sources: q.max_sources,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterWakeAmplitudeResult`].
fn decode_result(raw: &GpuResult) -> WaterWakeAmplitudeResult {
    WaterWakeAmplitudeResult {
        transverse_wavelength: raw.transverse_wavelength,
        froude_length: raw.froude_length,
        emission_amplitude: raw.emission_amplitude,
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

/// A compiled, reusable wake-scalar compute pipeline, twinning the three
/// stateless scalar primitives of the `CPU` golden
/// [`wake`](prism_render_architecture::water::wake).
pub struct GpuWaterWakeAmplitude {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterWakeAmplitude {
    /// Compiles the wake-scalar kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterWakeAmplitude {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude"),
            source: ShaderSource::Wgsl(WATER_WAKE_AMPLITUDE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterWakeAmplitude {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`WaterWakeAmplitudeResult`] per input, in order.
    ///
    /// Each output equals the matching `CPU` golden scalar within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterWakeAmplitudeQuery],
    ) -> Vec<WaterWakeAmplitudeResult> {
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
            label: Some("prism_volumetric_water_wake_amplitude_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_bind_group"),
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
            label: Some("prism_volumetric_water_wake_amplitude_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_wake_amplitude_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_wake_amplitude_pass"),
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
