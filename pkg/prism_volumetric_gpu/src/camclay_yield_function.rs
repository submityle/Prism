//! `wgpu` compute twin of the modified Cam-Clay trial-yield criterion, from the
//! `CPU` golden
//! `prism_physics_core::collider::tet_fem_camclay_plasticity`'s
//! `return_map_camclay`.
//!
//! The modified Cam-Clay yield surface is the ellipse
//! `Y(p, q) = q^2 / M^2 + p * (p - p_c)` in the mean-stress / deviatoric-stress
//! `(p, q)` plane: a trial state is elastic when `Y <= 0` and plastic (it has
//! yielded) when `Y > 0`. This module ports that single stateless trial
//! criterion onto the device: one thread resolves one `(p, q, M, p_c)` tuple,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same yield decision the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the trial-yield branch of
//! `return_map_camclay` for one explicit tuple of the trial pressure `p`, the
//! trial deviatoric stress `q`, the critical-state slope `M = slope_m` and the
//! pre-consolidation pressure `p_c = pre_consolidation`:
//!
//! * The model is valid only when all four inputs are finite, `M > 0` and
//!   `p_c > 0` (the reference's `CamClayModel::new` / `CamClayState::new`
//!   guards). An invalid tuple yields `valid = 0`, `yield_value = 0`,
//!   `yielded = 0`.
//! * Otherwise `yield_value = q * q / M^2 + p * (p - p_c)` and
//!   `yielded = yield_value > 0`.
//!
//! # Correctness model
//!
//! The reference computes the criterion in `f64` intermediates; the device is
//! `f32` only. The host oracle therefore replays the golden `f64` steps and
//! narrows to `f32`, and the valid `yield_value` scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `yielded` and `valid` flags are compared exactly; the parity sweep keeps the
//! trial criterion well away from the `Y = 0` knee so the yield decision cannot
//! be flipped by the `f32`/`f64` gap.
//!
//! # Degenerate inputs
//!
//! A non-finite input, `M <= 0` or `p_c <= 0` yields `valid = 0` with
//! `yield_value = 0` and `yielded = 0`. The `M^2` divisor is routed through a
//! `select` so the un-taken (invalid) branch never divides by zero, and the
//! value and yield flag are masked to `0` for an invalid tuple. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `sqrt` and no `f32` remainder, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_camclay_plasticity`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Cam-Clay trial-yield kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the trial-yield branch of the `CPU` golden `return_map_camclay`; see the
/// module documentation for the closed form.
const CAMCLAY_YIELD_FUNCTION_WGSL: &str = r#"
// Cam-Clay trial-yield twin: one thread per query reproduces
// Y = q*q/M^2 + p*(p - p_c) and the yielded = Y > 0 decision. It uses only the
// portable core-WGSL subset (abs, + - * /, select plus unsigned index math),
// takes no optional feature, and has no loop and no branch, so it provably
// terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) fed to select; there is no bare f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Trial mean stress p_tr.
    p: f32,
    // Trial deviatoric stress q_tr.
    q: f32,
    // Critical-state slope M (slope_m), must be > 0 for a valid model.
    slope_m: f32,
    // Pre-consolidation pressure p_c0, must be > 0 for a valid state.
    pre_consolidation: f32,
}

struct Result {
    // Yield value Y = q*q/M^2 + p*(p - p_c) when valid, else 0.
    yield_value: f32,
    // 1 when Y > 0 (plastic / yielded), else 0.
    yielded: u32,
    // 1 when all inputs are finite, M > 0 and p_c > 0, else 0.
    valid: u32,
    // Padding word to a 16-byte stride.
    pad0: u32,
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
    let qd = queries[idx];
    let p = qd.p;
    let q = qd.q;
    let m = qd.slope_m;
    let p_c = qd.pre_consolidation;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false), plus model/state positivity. No bare f32
    // equality anywhere.
    let finite = (abs(p) < FINITE_LIMIT)
        && (abs(q) < FINITE_LIMIT)
        && (abs(m) < FINITE_LIMIT)
        && (abs(p_c) < FINITE_LIMIT);
    let positive = (m > 0.0) && (p_c > 0.0);
    let ok = finite && positive;

    // Guard the M^2 divisor so the un-taken (invalid) branch never divides by
    // zero; when ok, M > 0 so M^2 is strictly positive.
    let m2 = m * m;
    let denom = select(1.0, m2, ok);
    let y = q * q / denom + p * (p - p_c);

    var out: Result;
    out.yield_value = select(0.0, y, ok);
    // yielded only when valid and Y strictly positive (ordered compare).
    out.yielded = select(0u, 1u, ok && (y > 0.0));
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
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
/// the four scalars `(p, q, slope_m, pre_consolidation)` — `4` `f32` words
/// (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    p: f32,
    q: f32,
    slope_m: f32,
    pre_consolidation: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the yield value, the yield flag, the validity flag and one padding
/// word — `4` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    yield_value: f32,
    yielded: u32,
    valid: u32,
    pad0: u32,
}

/// One Cam-Clay trial-yield query: the trial stresses and the model/state
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamClayYieldFunctionQuery {
    /// Trial mean stress `p_tr`.
    pub p: f32,
    /// Trial deviatoric stress `q_tr`.
    pub q: f32,
    /// Critical-state slope `M` (`slope_m`).
    pub slope_m: f32,
    /// Pre-consolidation pressure `p_c0`.
    pub pre_consolidation: f32,
}

impl CamClayYieldFunctionQuery {
    /// Builds a query from the trial stresses and model/state parameters.
    #[must_use]
    pub fn new(p: f32, q: f32, slope_m: f32, pre_consolidation: f32) -> CamClayYieldFunctionQuery {
        CamClayYieldFunctionQuery {
            p,
            q,
            slope_m,
            pre_consolidation,
        }
    }
}

/// One resolved answer for a single query, mirroring the trial-yield branch of
/// the reference `return_map_camclay` for that tuple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamClayYieldFunctionResult {
    /// The yield value `q * q / M^2 + p * (p - p_c)` when valid, else `0`.
    pub yield_value: f32,
    /// `true` when the trial state has yielded (`yield_value > 0`), else
    /// `false`.
    pub yielded: bool,
    /// `true` when all inputs are finite, `M > 0` and `p_c > 0`, else `false`.
    pub valid: bool,
}

/// Encodes one [`CamClayYieldFunctionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &CamClayYieldFunctionQuery) -> GpuQuery {
    GpuQuery {
        p: q.p,
        q: q.q,
        slope_m: q.slope_m,
        pre_consolidation: q.pre_consolidation,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CamClayYieldFunctionResult`].
fn decode_result(raw: &GpuResult) -> CamClayYieldFunctionResult {
    CamClayYieldFunctionResult {
        yield_value: raw.yield_value,
        yielded: raw.yielded != 0,
        valid: raw.valid != 0,
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

/// A compiled, reusable Cam-Clay trial-yield compute pipeline, twinning the
/// trial-yield branch of the `CPU` golden `return_map_camclay`.
pub struct GpuCamClayYieldFunction {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCamClayYieldFunction {
    /// Compiles the Cam-Clay trial-yield kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCamClayYieldFunction {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_camclay_yield_function"),
            source: ShaderSource::Wgsl(CAMCLAY_YIELD_FUNCTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCamClayYieldFunction {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CamClayYieldFunctionResult`] per input, in order.
    ///
    /// The `yielded` and `valid` flags match the reference exactly and the
    /// `yield_value` scalar to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CamClayYieldFunctionQuery],
    ) -> Vec<CamClayYieldFunctionResult> {
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
            label: Some("prism_volumetric_camclay_yield_function_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_bind_group"),
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
            label: Some("prism_volumetric_camclay_yield_function_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_camclay_yield_function_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_camclay_yield_function_pass"),
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
