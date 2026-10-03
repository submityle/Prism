//! `wgpu` compute twin of the flow-aware foam decay scalars
//! ([`foam`](prism_render_architecture::water::foam)).
//!
//! Dynamic water foam is a scalar coverage field that dissipates over time.
//! Two pure, stateless scalars drive that dissipation per cell: a flow-aware
//! decay rate that rises from a slow residual in still water up to a full rate
//! in churning water, and an exponential decay step that applies that rate over
//! a timestep. Both are built from clamps, a guarded divide, signed
//! comparisons and the shared hand-rolled `exp_approx` (the `water` subsystem
//! forbids `f32::exp`), so they port to the device directly.
//!
//! [`GpuWaterFoamDecay`] is the on-device twin: one thread resolves one query,
//! reproducing
//! [`foam_decay_rate`](prism_render_architecture::water::foam::foam_decay_rate)
//! and
//! [`decay_foam`](prism_render_architecture::water::foam::decay_foam). A passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! reference arithmetic, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query carrying a local `flow_speed`, the three `FoamConfig` decay
//! fields `base_decay`, `persistence_floor` and `reference_speed`, plus a
//! `density`, a `rate` and a `dt`, the kernel reproduces:
//! - the flow-aware decay rate `base_decay * (floor + (1 - floor) * t)`, where
//!   `floor` is `persistence_floor` clamped to `0..=1` and `t` is `flow_speed /
//!   reference` clamped to `0..=1` with `reference` guarded to at least `EPS`;
//!   and
//! - the exponential decay step `max(density, 0) * exp_approx(-rate * dt)`.
//!
//! The decay `rate` fed to the exponential step is an independent query input,
//! mirroring the two reference functions being twinned separately rather than
//! chaining the computed rate into the step.
//!
//! # What stays on the host
//!
//! The grid-shaped `FoamConfig` fields (`nx`, `nz`, `dx`) and every
//! slice-walking routine in the reference (`sample_bilinear`,
//! `advect_foam_field`, `step_foam`, `inject_foam`, `reactive_mask_into`) stay
//! on the host: they index variable-length buffers rather than performing a
//! stateless per-element map. An empty batch short-circuits on the host, since
//! a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The kernel performs clamps, a guarded divide by a non-zero reference speed,
//! signed comparisons and the shared squaring `exp_approx`, so a correct port
//! reproduces the reference to within floating-point rounding. The parity test
//! asserts the shared continuous tolerance (`abs_diff <= 1e-4` or `rel_diff <=
//! 1e-3`) on both continuous outputs. Fixtures keep `reference_speed` well above
//! `EPS` and bound `rate * dt` so the squaring approximation stays accurate.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`, `+`,
//! `-`, `*`, `/`, a signed comparison and a bounded `i32` loop that squares a
//! running base — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt`, and no `64`-bit integers or floats. The exponential
//! is the same `(1 + x / 4096)^4096` twelve-squaring limit identity the golden
//! `exp_approx` uses. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::foam`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` foam-decay kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden `foam_decay_rate` and `decay_foam`; see the module documentation for
/// the algorithm.
const WATER_FOAM_DECAY_WGSL: &str = r#"
// Foam-decay twin: one thread resolves one query's flow-aware decay rate and
// its exponential decay step, mirroring the CPU golden
// `water::foam::{foam_decay_rate, decay_foam}` with only clamps, a guarded
// divide, a signed comparison and the shared squaring `exp_approx`.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::foam；无第三方引擎源码
// 或衍生代码。

// Shared `water` module epsilon guarding the degenerate reference speed, copied
// from the golden `water::EPS`.
const EPS: f32 = 1e-6;

// Hand-rolled exp for the water subsystem: the `(1 + x / 4096)^4096` limit
// identity evaluated by twelve squarings, byte-for-byte matching the golden
// `water::exp_approx` (pure add/multiply, non-negative base clamp).
fn exp_approx(x: f32) -> f32 {
    var base: f32 = 1.0 + x / 4096.0;
    if (base < 0.0) {
        base = 0.0;
    }
    for (var i: i32 = 0; i < 12; i = i + 1) {
        base = base * base;
    }
    return base;
}

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Local flow speed feeding the flow-aware decay rate, in meters/second.
    flow_speed: f32,
    // Baseline decay rate at or above the reference speed, per second.
    base_decay: f32,
    // Fraction of base_decay that still applies in still water, in 0..=1.
    persistence_floor: f32,
    // Flow speed at which decay reaches the full base_decay, in meters/second.
    reference_speed: f32,
    // Current foam density to decay.
    density: f32,
    // Decay rate fed to the exponential step (independent input).
    rate: f32,
    // Timestep of the decay step, in seconds.
    dt: f32,
    pad0: u32,
}

struct Result {
    // Flow-aware decay rate from foam_decay_rate.
    decay_rate: f32,
    // Decayed foam density from decay_foam.
    decayed_density: f32,
    pad0: u32,
    pad1: u32,
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

    // Flow-aware decay rate: still water uses the persistence floor, fast water
    // the full base rate, interpolated by the clamped speed ratio.
    let floor = clamp(q.persistence_floor, 0.0, 1.0);
    var reference: f32 = EPS;
    if (q.reference_speed > EPS) {
        reference = q.reference_speed;
    }
    let t = clamp(q.flow_speed / reference, 0.0, 1.0);
    let decay_rate = q.base_decay * (floor + (1.0 - floor) * t);

    // Exponential decay step over dt at the independent query rate; a negative
    // density folds to zero before decaying.
    var d: f32 = 0.0;
    if (q.density > 0.0) {
        d = q.density;
    }
    let decayed_density = d * exp_approx(-q.rate * q.dt);

    var out: Result;
    out.decay_rate = decay_rate;
    out.decayed_density = decayed_density;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words,
/// filling a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_FOAM_DECAY_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the flow speed, the three decay
/// config scalars, the density, the rate and the timestep plus one pad word, a
/// `32`-byte stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Local flow speed feeding the flow-aware decay rate.
    flow_speed: f32,
    /// Baseline decay rate at or above the reference speed.
    base_decay: f32,
    /// Fraction of `base_decay` that still applies in still water.
    persistence_floor: f32,
    /// Flow speed at which decay reaches the full `base_decay`.
    reference_speed: f32,
    /// Current foam density to decay.
    density: f32,
    /// Decay rate fed to the exponential step.
    rate: f32,
    /// Timestep of the decay step.
    dt: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the flow-aware decay rate and the decayed density plus two pad words,
/// a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Flow-aware decay rate from `foam_decay_rate`.
    decay_rate: f32,
    /// Decayed foam density from `decay_foam`.
    decayed_density: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One foam-decay query for the twin: a local flow speed, the three `FoamConfig`
/// decay-tuning scalars, and a density/rate/timestep triple for the exponential
/// decay step.
///
/// The grid fields of the reference
/// [`FoamConfig`](prism_render_architecture::water::foam::FoamConfig) (`nx`,
/// `nz`, `dx`) are unused by the two twinned scalars and stay on the host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFoamDecayQuery {
    /// Local flow speed feeding the flow-aware decay rate.
    pub flow_speed: f32,
    /// Baseline decay rate at or above `reference_speed`.
    pub base_decay: f32,
    /// Fraction of `base_decay` that still applies in still water, in `0..=1`.
    pub persistence_floor: f32,
    /// Flow speed at which decay reaches the full `base_decay`.
    pub reference_speed: f32,
    /// Current foam density to decay.
    pub density: f32,
    /// Decay rate fed to the exponential decay step.
    pub rate: f32,
    /// Timestep of the decay step, in seconds.
    pub dt: f32,
}

impl WaterFoamDecayQuery {
    /// Builds a query from a flow speed, the three decay-config scalars, and a
    /// density/rate/timestep triple.
    #[must_use]
    pub const fn new(
        flow_speed: f32,
        base_decay: f32,
        persistence_floor: f32,
        reference_speed: f32,
        density: f32,
        rate: f32,
        dt: f32,
    ) -> WaterFoamDecayQuery {
        WaterFoamDecayQuery {
            flow_speed,
            base_decay,
            persistence_floor,
            reference_speed,
            density,
            rate,
            dt,
        }
    }
}

/// One resolved foam-decay query, mirroring the reference
/// [`foam_decay_rate`](prism_render_architecture::water::foam::foam_decay_rate)
/// and
/// [`decay_foam`](prism_render_architecture::water::foam::decay_foam).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFoamDecayResult {
    /// Flow-aware decay rate (per second) from `foam_decay_rate`.
    pub decay_rate: f32,
    /// Decayed foam density from `decay_foam`.
    pub decayed_density: f32,
}

/// Encodes one [`WaterFoamDecayQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterFoamDecayQuery) -> GpuQuery {
    GpuQuery {
        flow_speed: q.flow_speed,
        base_decay: q.base_decay,
        persistence_floor: q.persistence_floor,
        reference_speed: q.reference_speed,
        density: q.density,
        rate: q.rate,
        dt: q.dt,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterFoamDecayResult`].
fn decode_result(raw: &GpuResult) -> WaterFoamDecayResult {
    WaterFoamDecayResult {
        decay_rate: raw.decay_rate,
        decayed_density: raw.decayed_density,
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

/// A compiled, reusable foam-decay compute pipeline, twinning the `CPU` golden
/// flow-aware foam decay from [`foam`](prism_render_architecture::water::foam).
pub struct GpuWaterFoamDecay {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFoamDecay {
    /// Compiles the foam-decay kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterFoamDecay {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_foam_decay"),
            source: ShaderSource::Wgsl(WATER_FOAM_DECAY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_decay_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_foam_decay_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_foam_decay_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFoamDecay {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`WaterFoamDecayResult`] per input, in order.
    ///
    /// Each output matches the reference: the flow-aware decay rate and the
    /// exponential decay step. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterFoamDecayQuery],
    ) -> Vec<WaterFoamDecayResult> {
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
            label: Some("prism_volumetric_water_foam_decay_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_foam_decay_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_foam_decay_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_foam_decay_bind_group"),
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
            label: Some("prism_volumetric_water_foam_decay_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_foam_decay_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_foam_decay_pass"),
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
