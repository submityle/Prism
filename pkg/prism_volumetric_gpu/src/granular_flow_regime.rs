//! `wgpu` compute twin of the granular-suspension flow-regime classifier, from
//! the `CPU` golden
//! `prism_physics_core::collider::granular_flow_regime`'s
//! `GranularFlowRegime::from_state` and its `bagnold_number` / `stokes_number`
//! / `regime` getters.
//!
//! When grains shear through a viscous interstitial fluid, the macroscopic
//! stress is a competition between fluid-viscous stress and grain-collision
//! (inertial) stress. Bagnold (1954) captures their ratio with the
//! dimensionless Bagnold number and classifies the flow accordingly:
//!
//! ```text
//! Ba = rho_s * d^2 * shear_rate / mu_f          (Bagnold number)
//! St = Ba / 18                                   (Stokes number)
//! ```
//!
//! with `rho_s` the grain material density, `d` the grain diameter,
//! `shear_rate` the shear rate `γ̇` and `mu_f` the interstitial fluid dynamic
//! viscosity. The classic Bagnold thresholds split the flow into three regimes:
//!
//! * `MacroViscous` (viscous-dominated): `Ba < 40`
//! * `Transitional`: `40 <= Ba <= 450`
//! * `GrainInertia` (collision-dominated): `Ba > 450`
//!
//! This module ports that single stateless closed form onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same dimensionless numbers and the
//! same regime label as the reference, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `from_state` plus the getters:
//!
//! * If any input is non-finite, or `rho_s <= 0`, `d <= 0`, `shear_rate < 0`
//!   or `mu_f <= 0`, the state is invalid (`valid = 0`, every output `0`).
//! * Otherwise `bagnold = rho_s * d * d * shear_rate / mu_f` in exactly the
//!   golden operator order, `stokes = bagnold / 18`, and `regime` is the
//!   `0 = MacroViscous`, `1 = Transitional`, `2 = GrainInertia` label from the
//!   Bagnold thresholds.
//!
//! # Correctness model
//!
//! The continuous arithmetic (three multiplies, one division and the Stokes
//! division) threads through operators a `GPU` may contract, so `CPU` and `GPU`
//! are not necessarily bit-exact; the valid `bagnold` and `stokes` scalars are
//! compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR =
//! 1e-6`). The discrete `regime` and `valid` words are compared exactly; the
//! parity sweep keeps inputs master-valid and off the `Ba = 40` / `Ba = 450`
//! knees so neither the validity nor the regime decision can be flipped by
//! round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite input or a positivity-gate failure yields `valid = 0` with
//! every output `0`. When valid the viscosity is strictly positive, so the
//! division is well defined; the kernel still feeds the divisor through a
//! `select` guard so the un-taken (invalid) branch never divides by zero. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`, and validity and the regime thresholds
//! with ordered compares fed to `select`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_flow_regime`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` flow-regime kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `GranularFlowRegime::from_state` plus getters; see the module
/// documentation for the closed form.
const GRANULAR_FLOW_REGIME_WGSL: &str = r#"
// Flow-regime twin: one thread per query reproduces the Bagnold/Stokes numbers
// and the three-way regime label. It uses only the portable core-WGSL subset
// (abs, + - * /, select plus unsigned index math), takes no optional feature,
// and has no loop and no branch, so it provably terminates. Finiteness is an
// ordered abs < 3.0e38 compare (rejecting infinities and NaN), validity and the
// Bagnold thresholds are ordered compares, all fed to select.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Grain material density rho_s.
    grain_density: f32,
    // Grain diameter d.
    grain_diameter: f32,
    // Shear rate gamma-dot.
    shear_rate: f32,
    // Interstitial fluid dynamic viscosity mu_f.
    fluid_viscosity: f32,
}

struct Result {
    // Bagnold number rho_s*d*d*shear/mu when valid, else 0.
    bagnold: f32,
    // Stokes number Ba/18 when valid, else 0.
    stokes: f32,
    // Regime label 0=MacroViscous, 1=Transitional, 2=GrainInertia; 0 if invalid.
    regime: u32,
    // 1 when the state passes the finiteness and positivity gate, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const MACRO_VISCOUS_UPPER: f32 = 40.0;
const TRANSITIONAL_UPPER: f32 = 450.0;
const STOKES_DIVISOR: f32 = 18.0;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let rho_s = q.grain_density;
    let diam = q.grain_diameter;
    let shear = q.shear_rate;
    let mu = q.fluid_viscosity;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false), plus the golden positivity gate:
    // rho_s > 0, d > 0, shear_rate >= 0, mu_f > 0. No bare f32 equality.
    let finite = (abs(rho_s) < FINITE_LIMIT)
        && (abs(diam) < FINITE_LIMIT)
        && (abs(shear) < FINITE_LIMIT)
        && (abs(mu) < FINITE_LIMIT);
    let positive = (rho_s > 0.0) && (diam > 0.0) && (shear >= 0.0) && (mu > 0.0);
    let ok = finite && positive;

    // Guard the divisor so the un-taken (invalid) branch never divides by zero;
    // when ok the viscosity is strictly positive.
    let denom = select(1.0, mu, ok);
    // Golden operator order: rho_s * d * d * shear_rate first, then divide by
    // mu_f; the Stokes number is Ba / 18.
    let ba = rho_s * diam * diam * shear / denom;
    let st = ba / STOKES_DIVISOR;

    // Regime by Bagnold thresholds, encoded 0/1/2 via ordered compares:
    // Ba < 40 -> MacroViscous(0), Ba > 450 -> GrainInertia(2), else
    // Transitional(1).
    let is_macro = ba < MACRO_VISCOUS_UPPER;
    let is_grain = ba > TRANSITIONAL_UPPER;
    let regime_code = select(select(1u, 0u, is_macro), 2u, is_grain);

    var out: Result;
    out.bagnold = select(0.0, ba, ok);
    out.stokes = select(0.0, st, ok);
    out.regime = select(0u, regime_code, ok);
    out.valid = select(0u, 1u, ok);
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
/// four `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    grain_density: f32,
    grain_diameter: f32,
    shear_rate: f32,
    fluid_viscosity: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two dimensionless numbers, the regime label and the validity
/// flag — `4` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    bagnold: f32,
    stokes: f32,
    regime: u32,
    valid: u32,
}

/// One flow-regime query: the grain-fluid state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularFlowRegimeQuery {
    /// Grain material density `rho_s` (`> 0`).
    pub grain_density: f32,
    /// Grain diameter `d` (`> 0`).
    pub grain_diameter: f32,
    /// Shear rate `γ̇` (`>= 0`).
    pub shear_rate: f32,
    /// Interstitial fluid dynamic viscosity `mu_f` (`> 0`).
    pub fluid_viscosity: f32,
}

impl GranularFlowRegimeQuery {
    /// Builds a query from the grain-fluid state, in the golden `from_state`
    /// argument order.
    #[must_use]
    pub fn new(
        grain_density: f32,
        grain_diameter: f32,
        shear_rate: f32,
        fluid_viscosity: f32,
    ) -> GranularFlowRegimeQuery {
        GranularFlowRegimeQuery {
            grain_density,
            grain_diameter,
            shear_rate,
            fluid_viscosity,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GranularFlowRegime` output for that state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularFlowRegimeResult {
    /// Bagnold number `rho_s * d² * γ̇ / mu_f` when valid, else `0`.
    pub bagnold: f32,
    /// Stokes number `Ba / 18` when valid, else `0`.
    pub stokes: f32,
    /// Regime label: `0 = MacroViscous`, `1 = Transitional`,
    /// `2 = GrainInertia`; `0` when invalid.
    pub regime: u32,
    /// `1` when the state is finite and passes the positivity gate, else `0`.
    pub valid: u32,
}

/// Encodes one [`GranularFlowRegimeQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GranularFlowRegimeQuery) -> GpuQuery {
    GpuQuery {
        grain_density: q.grain_density,
        grain_diameter: q.grain_diameter,
        shear_rate: q.shear_rate,
        fluid_viscosity: q.fluid_viscosity,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`GranularFlowRegimeResult`].
fn decode_result(raw: &GpuResult) -> GranularFlowRegimeResult {
    GranularFlowRegimeResult {
        bagnold: raw.bagnold,
        stokes: raw.stokes,
        regime: raw.regime,
        valid: raw.valid,
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

/// A compiled, reusable flow-regime compute pipeline, twinning the `CPU` golden
/// `GranularFlowRegime::from_state` and its getters.
pub struct GpuGranularFlowRegime {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGranularFlowRegime {
    /// Compiles the flow-regime kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGranularFlowRegime {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_granular_flow_regime"),
            source: ShaderSource::Wgsl(GRANULAR_FLOW_REGIME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGranularFlowRegime {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GranularFlowRegimeResult`] per input, in order.
    ///
    /// The `regime` and `valid` words match the reference exactly and the
    /// `bagnold` / `stokes` scalars to the module's tolerance. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GranularFlowRegimeQuery],
    ) -> Vec<GranularFlowRegimeResult> {
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
            label: Some("prism_volumetric_granular_flow_regime_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_bind_group"),
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
            label: Some("prism_volumetric_granular_flow_regime_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_granular_flow_regime_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_granular_flow_regime_pass"),
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
