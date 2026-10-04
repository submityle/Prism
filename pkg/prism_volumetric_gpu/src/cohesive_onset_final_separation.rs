//! `wgpu` compute twin of the cohesive-zone onset/final separation derivation,
//! from the `CPU` golden `prism_physics_core::collider::cohesive_zone`'s
//! `CohesiveModel::new`.
//!
//! A bilinear cohesive-zone law is parameterised by a penalty stiffness `K`, a
//! peak traction (strength) `σ_c`, a fracture energy `G_c` and a shear
//! mode-mixity weight `β`. From these the model derives two characteristic
//! separations:
//!
//! * the damage-onset separation `δ₀ = σ_c / K`, and
//! * the final (fully-separated) separation `δ_f = 2 · G_c / σ_c`.
//!
//! This module ports that single stateless derivation onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same separations the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the `CohesiveModel::new` derivation and
//! its validity gate:
//!
//! * If any of `K`, `σ_c`, `G_c`, `β` is non-finite, or `K <= 0`, `σ_c <= 0`,
//!   `G_c <= 0`, `β < 0`, the model is invalid (`valid = 0`, both separations
//!   `0`).
//! * Otherwise `onset = σ_c / K` and `final_sep = 2 · G_c / σ_c`. If the derived
//!   `final_sep <= onset` the model is still invalid (a bilinear law needs a
//!   strictly positive softening branch); otherwise it is valid.
//!
//! The shear weight `β` participates only in the validity gate (it must be
//! non-negative), never in the two derived separations.
//!
//! # Correctness model
//!
//! The continuous arithmetic (two divisions and a multiply) threads through
//! operators a `GPU` may contract, so `CPU` and `GPU` are not necessarily
//! bit-exact; each valid separation is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly; the parity test keeps the inputs well
//! inside the valid region (and `final_sep` comfortably above `onset`) so the
//! validity decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite input, a non-positive `K`/`σ_c`/`G_c`, a negative `β`, or a
//! derived `final_sep <= onset` all yield `valid = 0` with both separations `0`.
//! The two divisors (`K` and `σ_c`) are fed through a `select` guard so the
//! un-taken (invalid) branch never evaluates a division by zero. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and the positivity / ordering gates with ordered
//! comparisons; there is no `f32` equality anywhere. The result field is named
//! `final_sep` rather than `final`, which is a `WGSL` reserved word.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cohesive-zone separation kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `CohesiveModel::new`; see the module documentation
/// for the closed form.
const COHESIVE_ONSET_FINAL_SEPARATION_WGSL: &str = r#"
// Cohesive-zone separation twin: one thread per query reproduces the onset and
// final separations of CohesiveModel::new. It uses only the portable core-WGSL
// subset (abs, + - * /, select plus unsigned index math), takes no optional
// feature, and has no loop and no branch, so it provably terminates. Finiteness
// is an ordered abs < 3.0e38 compare (rejecting infinities and NaN) and the
// positivity / ordering gates are ordered compares, all fed to select. The
// final-separation field is named final_sep because `final` is a WGSL keyword.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Penalty / initial-slope stiffness K.
    stiffness: f32,
    // Peak traction (interface strength) sigma_c.
    strength: f32,
    // Fracture energy G_c.
    fracture_energy: f32,
    // Shear mode-mixity weight beta (validity only).
    shear_weight: f32,
}

struct Result {
    // Damage-onset separation sigma_c / K when valid, else 0.
    onset: f32,
    // Final separation 2 * G_c / sigma_c when valid, else 0.
    final_sep: f32,
    // 1 when the model passes every validity gate, else 0.
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
    let q = queries[idx];
    let k = q.stiffness;
    let sigma = q.strength;
    let energy = q.fracture_energy;
    let beta = q.shear_weight;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let finite = (abs(k) < FINITE_LIMIT)
        && (abs(sigma) < FINITE_LIMIT)
        && (abs(energy) < FINITE_LIMIT)
        && (abs(beta) < FINITE_LIMIT);
    let positive = (k > 0.0) && (sigma > 0.0) && (energy > 0.0) && (beta >= 0.0);
    let gate = finite && positive;

    // Guard the divisors so the un-taken (invalid) branch never divides by zero;
    // when gate holds both K and sigma_c are strictly positive.
    let safe_k = select(1.0, k, gate);
    let safe_sigma = select(1.0, sigma, gate);
    // Golden operator order: onset = sigma_c / K, final = 2 * G_c / sigma_c.
    let onset = sigma / safe_k;
    let final_sep = 2.0 * energy / safe_sigma;

    // A bilinear law needs a strictly positive softening branch.
    let ordered = final_sep > onset;
    let ok = gate && ordered;

    var out: Result;
    out.onset = select(0.0, onset, ok);
    out.final_sep = select(0.0, final_sep, ok);
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
/// the four cohesive parameters packed as `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    stiffness: f32,
    strength: f32,
    fracture_energy: f32,
    shear_weight: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two separations, the validity flag and a padding word — `4`
/// words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    onset: f32,
    final_sep: f32,
    valid: u32,
    pad0: u32,
}

/// One cohesive-zone query: the four model parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveOnsetFinalSeparationQuery {
    /// Penalty / initial-slope stiffness `K`.
    pub stiffness: f32,
    /// Peak traction (interface strength) `σ_c`.
    pub strength: f32,
    /// Fracture energy `G_c`.
    pub fracture_energy: f32,
    /// Shear mode-mixity weight `β` (participates only in the validity gate).
    pub shear_weight: f32,
}

impl CohesiveOnsetFinalSeparationQuery {
    /// Builds a query from the four cohesive-zone parameters.
    #[must_use]
    pub fn new(
        stiffness: f32,
        strength: f32,
        fracture_energy: f32,
        shear_weight: f32,
    ) -> CohesiveOnsetFinalSeparationQuery {
        CohesiveOnsetFinalSeparationQuery {
            stiffness,
            strength,
            fracture_energy,
            shear_weight,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CohesiveModel::new` derivation for that parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveOnsetFinalSeparationResult {
    /// The damage-onset separation `σ_c / K` when valid, else `0`.
    pub onset: f32,
    /// The final separation `2 · G_c / σ_c` when valid, else `0`.
    pub final_sep: f32,
    /// `true` when the model passes every validity gate, else `false`.
    pub valid: bool,
}

/// Encodes one [`CohesiveOnsetFinalSeparationQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &CohesiveOnsetFinalSeparationQuery) -> GpuQuery {
    GpuQuery {
        stiffness: q.stiffness,
        strength: q.strength,
        fracture_energy: q.fracture_energy,
        shear_weight: q.shear_weight,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CohesiveOnsetFinalSeparationResult`].
fn decode_result(raw: &GpuResult) -> CohesiveOnsetFinalSeparationResult {
    CohesiveOnsetFinalSeparationResult {
        onset: raw.onset,
        final_sep: raw.final_sep,
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

/// A compiled, reusable cohesive-zone separation compute pipeline, twinning the
/// `CPU` golden `CohesiveModel::new`.
pub struct GpuCohesiveOnsetFinalSeparation {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCohesiveOnsetFinalSeparation {
    /// Compiles the cohesive-zone separation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCohesiveOnsetFinalSeparation {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation"),
            source: ShaderSource::Wgsl(COHESIVE_ONSET_FINAL_SEPARATION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCohesiveOnsetFinalSeparation {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CohesiveOnsetFinalSeparationResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each separation to the
    /// module's tolerance. An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CohesiveOnsetFinalSeparationQuery],
    ) -> Vec<CohesiveOnsetFinalSeparationResult> {
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
            label: Some("prism_volumetric_cohesive_onset_final_separation_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_bind_group"),
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
            label: Some("prism_volumetric_cohesive_onset_final_separation_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cohesive_onset_final_separation_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cohesive_onset_final_separation_pass"),
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
