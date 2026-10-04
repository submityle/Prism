//! `wgpu` compute twin of the bilinear cohesive-zone secant-damage closed form,
//! from the `CPU` golden `prism_physics_core::collider::cohesive_zone`'s
//! `CohesiveModel::damage_at`.
//!
//! As a bonded interface (a crack face or weak internal surface) is pulled
//! apart, its cohesion decays with the largest effective separation ever
//! reached, `κ`. The secant damage `d(κ)` rises from `0` at the onset
//! separation `δ₀` to `1` at the final separation `δ_f` along the bilinear
//! (Alfano–Crisfield) envelope. This module ports that single stateless closed
//! form onto the device: one thread resolves one query, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same damage the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `damage_at` for one history variable
//! `κ` under a model's onset and final separations `(δ₀, δ_f)`:
//!
//! * `κ ≤ δ₀` yields `d = 0` (still in the reversible elastic rise).
//! * `κ ≥ δ_f` yields `d = 1` (fully decohered).
//! * Otherwise `d = δ_f·(κ − δ₀) / (κ·(δ_f − δ₀))`, clamped to `[0, 1]`, in the
//!   golden operator order.
//!
//! The caller is responsible for supplying a well-formed model, as the golden's
//! `CohesiveModel::new` guarantees `δ_f > δ₀ > 0`.
//!
//! # Correctness model
//!
//! The softening branch threads through a subtract, two multiplies, a divide
//! and a clamp, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The `damage` scalar is
//! compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`).
//! The parity test rejects random `κ` within `0.05·δ_f` of either knee (`δ₀`,
//! `δ_f`) so the branch decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! The two branch thresholds are ordered compares (`κ ≤ δ₀`, `κ ≥ δ_f`) fed to
//! `select`. The softening divisor `κ·(δ_f − δ₀)` is positive whenever that
//! middle branch is taken (there `κ > δ₀ > 0` and `δ_f − δ₀ > 0`), yet the
//! kernel still routes it through a `select` guard so the un-taken branches
//! never evaluate a division by zero. The final result is clamped with ordered
//! `min(max(x, 0), 1)`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Every branch uses an
//! ordered compare rather than a bare `x == x`; there is no `f32` equality
//! anywhere.
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

/// The portable core-`WGSL` secant-damage kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `CohesiveModel::damage_at`; see the module documentation for
/// the closed form.
const COHESIVE_ZONE_DAMAGE_WGSL: &str = r#"
// Secant-damage twin: one thread per query reproduces damage_at. It uses only
// the portable core-WGSL subset (min, max, + - * /, select plus unsigned index
// math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. The two thresholds are ordered compares (kappa <= onset,
// kappa >= final) fed to select; the softening divisor is select-guarded so the
// un-taken branches never divide by zero. There is no f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Onset separation delta0 (softening begins; d = 0 at and below it).
    onset_separation: f32,
    // Final separation deltaf (full decohesion; d = 1 at and above it).
    final_separation: f32,
    // History variable kappa (largest effective separation ever reached).
    kappa: f32,
    // Padding to a 16-byte std430 stride.
    pad0: f32,
}

struct Result {
    // Secant damage d(kappa) in [0, 1].
    damage: f32,
    // Padding to an 8-byte std430 stride.
    pad0: u32,
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
    let onset = q.onset_separation;
    let final_sep = q.final_separation;
    let kappa = q.kappa;

    // Ordered branch predicates; no bare f32 equality.
    let below = kappa <= onset;
    let above = kappa >= final_sep;
    let middle = (!below) && (!above);

    // Softening divisor kappa*(final - onset) is positive on the middle branch;
    // the select guard keeps the un-taken branches from dividing by zero.
    let den_raw = kappa * (final_sep - onset);
    let den = select(1.0, den_raw, middle);
    let num = final_sep * (kappa - onset);
    // Clamp with ordered min(max(x, 0), 1).
    let soft = min(max(num / den, 0.0), 1.0);

    // d = 0 below onset, 1 above final, soft in the middle.
    let damage = select(select(0.0, 1.0, above), soft, middle);

    var out: Result;
    out.damage = damage;
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
/// the onset and final separations, the history variable and a padding word —
/// `4` `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    onset_separation: f32,
    final_separation: f32,
    kappa: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the damage scalar and a padding word — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    damage: f32,
    pad0: u32,
}

/// One secant-damage query: the model's onset and final separations and the
/// monotone history variable `κ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveZoneDamageQuery {
    /// Onset separation `δ₀` where softening begins.
    pub onset_separation: f32,
    /// Final separation `δ_f` where cohesion is fully lost.
    pub final_separation: f32,
    /// History variable `κ`, the largest effective separation ever reached.
    pub kappa: f32,
}

impl CohesiveZoneDamageQuery {
    /// Builds a query from the onset and final separations and the history
    /// variable `κ`.
    #[must_use]
    pub fn new(
        onset_separation: f32,
        final_separation: f32,
        kappa: f32,
    ) -> CohesiveZoneDamageQuery {
        CohesiveZoneDamageQuery {
            onset_separation,
            final_separation,
            kappa,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CohesiveModel::damage_at` output for that configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveZoneDamageResult {
    /// Secant damage `d(κ) ∈ [0, 1]`.
    pub damage: f32,
}

/// Encodes one [`CohesiveZoneDamageQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &CohesiveZoneDamageQuery) -> GpuQuery {
    GpuQuery {
        onset_separation: q.onset_separation,
        final_separation: q.final_separation,
        kappa: q.kappa,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`CohesiveZoneDamageResult`].
fn decode_result(raw: &GpuResult) -> CohesiveZoneDamageResult {
    CohesiveZoneDamageResult { damage: raw.damage }
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

/// A compiled, reusable secant-damage compute pipeline, twinning the `CPU`
/// golden `CohesiveModel::damage_at`.
pub struct GpuCohesiveZoneDamage {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCohesiveZoneDamage {
    /// Compiles the secant-damage kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCohesiveZoneDamage {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage"),
            source: ShaderSource::Wgsl(COHESIVE_ZONE_DAMAGE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCohesiveZoneDamage {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CohesiveZoneDamageResult`] per input, in order.
    ///
    /// The `damage` scalar matches the reference to the module's tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CohesiveZoneDamageQuery],
    ) -> Vec<CohesiveZoneDamageResult> {
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
            label: Some("prism_volumetric_cohesive_zone_damage_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_bind_group"),
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
            label: Some("prism_volumetric_cohesive_zone_damage_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cohesive_zone_damage_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cohesive_zone_damage_pass"),
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
