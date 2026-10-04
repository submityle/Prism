//! `wgpu` compute twin of Haff's homogeneous-cooling law for an inelastic
//! granular gas, from the `CPU` golden `prism_physics_core::collider::haff_cooling`'s
//! `HaffCooling` constructor and depth/time getters.
//!
//! An unforced granular gas of inelastic hard spheres loses kinetic energy at
//! every collision, so its granular temperature `T` decays algebraically. The
//! initial cooling rate is set by the collision frequency `ω0` and the normal
//! restitution `e`, and the temperature follows Haff's law:
//!
//! ```text
//! ζ0 = (1 − e²) · ω0 / 3,     τ = 2 / ζ0,
//! T(t) = T0 / (1 + t/τ)²,     ω(t) = ω0 / (1 + t/τ).
//! ```
//!
//! This module ports the constructor's validity gate and the five getters onto
//! the device: one thread resolves one query, so a passing real-device parity
//! test is direct evidence the ported kernel computes the same quantities the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel evaluates, in the golden operator order:
//!
//! * `cooling_rate = ζ0 = (1 − e²) · ω0 / 3`.
//! * `cooling_time = τ = 2 / ζ0`.
//! * `temperature_at(t) = T0 / (1 + t/τ)²`, gated to `0` for a non-finite or
//!   negative `t`.
//! * `collision_frequency_at(t) = ω0 / (1 + t/τ)`, same `t` gate.
//! * `time_to_fraction(f) = τ · (f^{-1/2} − 1)`, gated for `f` outside
//!   `(0, 1]`.
//! * `half_life = τ · (√2 − 1)`.
//!
//! # Validity model
//!
//! The master validity flag mirrors the golden `HaffCooling::new`: it is set
//! only when `T0 ≥ 0`, `ω0 ≥ 0`, `e ∈ [0, 1)` are all finite and the derived
//! `ζ0` and `τ` are finite and positive. A perfectly elastic grain (`e = 1`)
//! or a collisionless gas (`ω0 = 0`) drives `ζ0` to `0`, so the master flag is
//! cleared and every scalar output is zeroed. When the master flag is set, the
//! two per-time getters carry their own gate (`t` finite and non-negative) and
//! the fraction getter its own gate (`f ∈ (0, 1]`); a closed sub-gate zeroes
//! only that getter's value and clears its own validity word.
//!
//! # Correctness model
//!
//! The golden is pure `f32`; this kernel is pure `f32` too, and the host oracle
//! re-derives the same closed form in `f32`. The golden's `half_life` uses the
//! `f32` constant `√2`; the kernel computes `sqrt(2.0)` with the built-in,
//! which agrees to well within tolerance. Continuous scalars are compared with
//! an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the sweep
//! keeps `e ≤ 0.95`, `ω0 ≥ 0.1` and `f ≥ 0.01` away from the knees where `ζ0`
//! or `1/√f` blow up. The four validity words are compared exactly.
//!
//! # Degenerate inputs
//!
//! A cleared master flag yields all-zero scalars and all-zero sub-validity
//! words. The two divisors (`ζ0` into `τ`, and `τ` into the `1 + t/τ` ratio) and
//! the fraction divisor are each guarded by a `select` that substitutes `1`
//! when the corresponding gate is closed, so no `inf`/`NaN` survives into a
//! discarded branch. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `abs`, `sqrt`, `select`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round` and no `f32` remainder, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the ordered
//! compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`) rather
//! than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::haff_cooling`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Haff cooling-law kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `HaffCooling` constructor and getters; see the module
/// documentation for the closed form.
const HAFF_COOLING_LAW_WGSL: &str = r#"
// Haff cooling-law twin: one thread per query evaluates the cooling rate,
// cooling time, temperature and collision frequency at a time t, the time to a
// temperature fraction f, and the half-life, from the initial temperature,
// collision frequency and restitution. It uses only the portable core-WGSL
// subset (abs, sqrt, select, + - * / plus unsigned index math), has no loop and
// no branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) fed to select; no bare
// f32 equality anywhere. Every divisor is select-guarded so no inf/NaN survives
// into a discarded degenerate branch.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Initial granular temperature T0.
    initial_temperature: f32,
    // Initial collision frequency omega0.
    initial_collision_frequency: f32,
    // Normal restitution e in [0, 1).
    restitution: f32,
    // Elapsed time t for the temperature/frequency getters.
    elapsed_time: f32,
    // Target temperature fraction f in (0, 1] for the time-to-fraction getter.
    fraction: f32,
    // Padding to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Initial cooling rate zeta0.
    cooling_rate: f32,
    // Characteristic cooling time tau.
    cooling_time: f32,
    // Temperature T(t).
    temperature_at: f32,
    // Collision frequency omega(t).
    collision_frequency_at: f32,
    // Time to reach the temperature fraction f.
    time_to_fraction: f32,
    // Half-life tau*(sqrt(2)-1).
    half_life: f32,
    // Master validity (constructor succeeded).
    valid: u32,
    // Temperature getter sub-validity.
    temperature_valid: u32,
    // Collision-frequency getter sub-validity.
    collision_frequency_valid: u32,
    // Time-to-fraction getter sub-validity.
    time_to_fraction_valid: u32,
    // Padding to a 16-byte-friendly stride.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let temp0 = q.initial_temperature;
    let freq0 = q.initial_collision_frequency;
    let rest = q.restitution;

    // Constructor validity gate, mirroring HaffCooling::new in order.
    let temp0_ok = (abs(temp0) < FINITE_LIMIT) && (temp0 >= 0.0);
    let freq0_ok = (abs(freq0) < FINITE_LIMIT) && (freq0 >= 0.0);
    let rest_ok = (abs(rest) < FINITE_LIMIT) && (rest >= 0.0) && (rest < 1.0);

    let inelasticity = 1.0 - rest * rest;
    let zeta0 = inelasticity * freq0 / 3.0;
    let zeta0_ok = (abs(zeta0) < FINITE_LIMIT) && (zeta0 > 0.0);
    let master_pre = temp0_ok && freq0_ok && rest_ok && zeta0_ok;

    // Guard the cooling-rate divisor before forming tau.
    let zeta_den = select(1.0, zeta0, master_pre);
    let tau = 2.0 / zeta_den;
    let tau_ok = (abs(tau) < FINITE_LIMIT) && (tau > 0.0);
    let master_ok = master_pre && tau_ok;

    // Guard the cooling-time divisor used by every (1 + t/tau) ratio.
    let tau_den = select(1.0, tau, master_ok);

    let cooling_rate = select(0.0, zeta0, master_ok);
    let cooling_time = select(0.0, tau, master_ok);
    let half_life = select(0.0, tau * (sqrt(2.0) - 1.0), master_ok);

    // Temperature and collision-frequency getters share the time gate.
    let elapsed = q.elapsed_time;
    let time_ok = master_ok && (abs(elapsed) < FINITE_LIMIT) && (elapsed >= 0.0);
    let ratio = 1.0 + elapsed / tau_den;
    let ratio_den = select(1.0, ratio, time_ok);
    let temperature_at = select(0.0, temp0 / (ratio_den * ratio_den), time_ok);
    let collision_frequency_at = select(0.0, freq0 / ratio_den, time_ok);

    // Time-to-fraction getter: f in (0, 1].
    let frac = q.fraction;
    let frac_ok = master_ok && (abs(frac) < FINITE_LIMIT) && (frac > 0.0) && (frac <= 1.0);
    let frac_den = select(1.0, frac, frac_ok);
    let inv_sqrt = 1.0 / sqrt(frac_den);
    let time_to_fraction = select(0.0, tau * (inv_sqrt - 1.0), frac_ok);

    var out: Result;
    out.cooling_rate = cooling_rate;
    out.cooling_time = cooling_time;
    out.temperature_at = temperature_at;
    out.collision_frequency_at = collision_frequency_at;
    out.time_to_fraction = time_to_fraction;
    out.half_life = half_life;
    out.valid = select(0u, 1u, master_ok);
    out.temperature_valid = select(0u, 1u, time_ok);
    out.collision_frequency_valid = select(0u, 1u, time_ok);
    out.time_to_fraction_valid = select(0u, 1u, frac_ok);
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the three constructor inputs, the elapsed time, the target fraction and
/// three padding words — `8` `f32` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    initial_temperature: f32,
    initial_collision_frequency: f32,
    restitution: f32,
    elapsed_time: f32,
    fraction: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: six `f32` scalars, four `u32` validity words and two padding words —
/// `12` words (`48` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    cooling_rate: f32,
    cooling_time: f32,
    temperature_at: f32,
    collision_frequency_at: f32,
    time_to_fraction: f32,
    half_life: f32,
    valid: u32,
    temperature_valid: u32,
    collision_frequency_valid: u32,
    time_to_fraction_valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One Haff cooling-law query: the initial temperature, initial collision
/// frequency and restitution that build the law, plus the elapsed time and
/// target temperature fraction at which to evaluate the getters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaffCoolingLawQuery {
    /// Initial granular temperature `T0` (`≥ 0`).
    pub initial_temperature: f32,
    /// Initial collision frequency `omega0` (`≥ 0`).
    pub initial_collision_frequency: f32,
    /// Normal restitution `e` in `[0, 1)`.
    pub restitution: f32,
    /// Elapsed time `t` for the temperature and collision-frequency getters.
    pub time: f32,
    /// Target temperature fraction `f` in `(0, 1]` for the time-to-fraction
    /// getter.
    pub fraction: f32,
}

impl HaffCoolingLawQuery {
    /// Builds a query from the three constructor inputs, the elapsed time and
    /// the target fraction.
    #[must_use]
    pub fn new(
        initial_temperature: f32,
        initial_collision_frequency: f32,
        restitution: f32,
        time: f32,
        fraction: f32,
    ) -> HaffCoolingLawQuery {
        HaffCoolingLawQuery {
            initial_temperature,
            initial_collision_frequency,
            restitution,
            time,
            fraction,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference cooling law.
///
/// When `valid` is `false` the constructor's gate rejected the inputs and every
/// scalar is zero. When `valid` is `true` the cooling rate, cooling time and
/// half-life are always meaningful, while `temperature_valid`,
/// `collision_frequency_valid` and `time_to_fraction_valid` report whether each
/// getter's own gate accepted its argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaffCoolingLawResult {
    /// Initial cooling rate `zeta0 = (1 − e²) · omega0 / 3`.
    pub cooling_rate: f32,
    /// Characteristic cooling time `tau = 2 / zeta0`.
    pub cooling_time: f32,
    /// Granular temperature `T(t)`, zero when `temperature_valid` is `false`.
    pub temperature_at: f32,
    /// Collision frequency `omega(t)`, zero when `collision_frequency_valid` is
    /// `false`.
    pub collision_frequency_at: f32,
    /// Time to reach the temperature fraction `f`, zero when
    /// `time_to_fraction_valid` is `false`.
    pub time_to_fraction: f32,
    /// Half-life `tau · (√2 − 1)`.
    pub half_life: f32,
    /// Master validity: the constructor accepted the inputs.
    pub valid: bool,
    /// Whether the temperature getter's time gate accepted `t`.
    pub temperature_valid: bool,
    /// Whether the collision-frequency getter's time gate accepted `t`.
    pub collision_frequency_valid: bool,
    /// Whether the time-to-fraction getter's gate accepted `f`.
    pub time_to_fraction_valid: bool,
}

/// Encodes one [`HaffCoolingLawQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HaffCoolingLawQuery) -> GpuQuery {
    GpuQuery {
        initial_temperature: q.initial_temperature,
        initial_collision_frequency: q.initial_collision_frequency,
        restitution: q.restitution,
        elapsed_time: q.time,
        fraction: q.fraction,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HaffCoolingLawResult`].
fn decode_result(raw: &GpuResult) -> HaffCoolingLawResult {
    HaffCoolingLawResult {
        cooling_rate: raw.cooling_rate,
        cooling_time: raw.cooling_time,
        temperature_at: raw.temperature_at,
        collision_frequency_at: raw.collision_frequency_at,
        time_to_fraction: raw.time_to_fraction,
        half_life: raw.half_life,
        valid: raw.valid != 0,
        temperature_valid: raw.temperature_valid != 0,
        collision_frequency_valid: raw.collision_frequency_valid != 0,
        time_to_fraction_valid: raw.time_to_fraction_valid != 0,
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

/// A compiled, reusable Haff cooling-law compute pipeline, twinning the `CPU`
/// golden `HaffCooling` constructor and getters.
pub struct GpuHaffCoolingLaw {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHaffCoolingLaw {
    /// Compiles the Haff cooling-law kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHaffCoolingLaw {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_haff_cooling_law"),
            source: ShaderSource::Wgsl(HAFF_COOLING_LAW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHaffCoolingLaw {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`HaffCoolingLawResult`]
    /// per input, in order.
    ///
    /// Each continuous scalar matches the reference to the module's tolerance
    /// and each validity word matches exactly. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HaffCoolingLawQuery],
    ) -> Vec<HaffCoolingLawResult> {
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
            label: Some("prism_volumetric_haff_cooling_law_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_bind_group"),
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
            label: Some("prism_volumetric_haff_cooling_law_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_haff_cooling_law_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_haff_cooling_law_pass"),
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
