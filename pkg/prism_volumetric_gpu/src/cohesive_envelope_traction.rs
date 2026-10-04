//! `wgpu` compute twin of the bilinear cohesive-envelope traction closed form,
//! from the `CPU` golden `prism_physics_core::collider::cohesive_zone`'s
//! `CohesiveZone::envelope_traction`.
//!
//! The monotonic-loading effective traction on a bilinear cohesive envelope is a
//! piecewise closed form of the effective separation `lambda`: zero below
//! contact, a linear elastic ramp `K * lambda` up to the onset separation
//! `delta0`, a linear softening branch `sigma_c * (delta_f - lambda) /
//! (delta_f - delta0)` between `delta0` and the final separation `delta_f`, and
//! zero once fully separated. This module ports that single stateless closed
//! form onto the device: one thread resolves one query, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same traction
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `envelope_traction` from the four
//! envelope parameters (`stiffness`, `strength`, `onset_separation`,
//! `final_separation`) and one `lambda`, in exactly the golden branch order:
//!
//! * `lambda <= 0` → `0`.
//! * `lambda <= delta0` → `K * lambda` (elastic ramp).
//! * `lambda >= delta_f` → `0` (fully separated).
//! * otherwise → `sigma_c * (delta_f - lambda) / (delta_f - delta0)` (softening).
//!
//! # Correctness model
//!
//! The continuous arithmetic (a multiply on the ramp, a subtract-multiply-divide
//! on the softening branch) threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` are not necessarily bit-exact; the `traction` scalar is
//! compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`).
//! The parity sweep uses rejection sampling to keep `lambda` well away from the
//! three branch knees (`0`, `delta0`, `delta_f`) so the piecewise selection
//! cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! The envelope is constructed with `delta_f > delta0 > 0`, `K > 0` and
//! `sigma_c > 0`, so the softening denominator `delta_f - delta0` is strictly
//! positive whenever the softening branch is taken. The kernel still feeds that
//! denominator through a `select` guard so the un-taken branches never evaluate
//! a division that could produce an infinity or `NaN`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `select`
//! and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The three branch thresholds use ordered
//! compares (`lambda <= 0`, `lambda <= delta0`, `lambda >= delta_f`) fed to a
//! `select` cascade rather than any bare `x == x`; there is no `f32` equality
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

/// The portable core-`WGSL` cohesive-envelope traction kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `CohesiveZone::envelope_traction`; see the module
/// documentation for the closed form.
const COHESIVE_ENVELOPE_TRACTION_WGSL: &str = r#"
// Cohesive-envelope traction twin: one thread per query reproduces
// envelope_traction. It uses only the portable core-WGSL subset (+ - * /,
// select plus unsigned index math), takes no optional feature, and has no loop
// and no branch, so it provably terminates. The three branch thresholds are
// ordered compares fed to a select cascade; the softening denominator is guarded
// by select so the un-taken branches never divide by zero. No bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Interface stiffness K (> 0 by construction).
    stiffness: f32,
    // Cohesive strength sigma_c (> 0 by construction).
    strength: f32,
    // Onset separation delta0 (> 0 by construction).
    onset_separation: f32,
    // Final separation delta_f (> delta0 by construction).
    final_separation: f32,
    // Effective separation lambda the envelope is evaluated at.
    lambda: f32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Envelope traction magnitude at lambda.
    traction: f32,
    // Padding word to an 8-byte stride.
    pad: u32,
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
    let stiffness = q.stiffness;
    let strength = q.strength;
    let delta0 = q.onset_separation;
    let delta_f = q.final_separation;
    let lambda = q.lambda;

    // Ordered branch selectors; every comparison with NaN is false, so no bare
    // f32 equality is needed.
    let below_contact = lambda <= 0.0;
    let in_elastic = lambda <= delta0;
    let fully_separated = lambda >= delta_f;
    // Softening branch iff above delta0 and strictly below delta_f.
    let in_soft = (!in_elastic) && (!fully_separated);

    // Elastic ramp K*lambda.
    let elastic = stiffness * lambda;

    // Softening sigma_c*(delta_f - lambda)/(delta_f - delta0). Guard the divisor
    // so the un-taken branches never divide by zero; when in_soft the span is
    // strictly positive by construction (delta_f > delta0).
    let span = delta_f - delta0;
    let denom = select(1.0, span, in_soft);
    let soft = strength * (delta_f - lambda) / denom;

    // Golden branch order collapsed into a select cascade:
    //   lambda <= 0        -> 0
    //   lambda <= delta0   -> elastic
    //   lambda >= delta_f  -> 0
    //   otherwise          -> soft
    var traction = soft;
    traction = select(traction, elastic, in_elastic);
    traction = select(traction, 0.0, fully_separated && (!in_elastic));
    traction = select(traction, 0.0, below_contact);

    var out: Result;
    out.traction = traction;
    out.pad = 0u;
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
/// the five envelope scalars padded to `8` `f32` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    stiffness: f32,
    strength: f32,
    onset_separation: f32,
    final_separation: f32,
    lambda: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the traction and a padding word — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    traction: f32,
    pad: u32,
}

/// One cohesive-envelope traction query: the four envelope parameters plus the
/// effective separation the envelope is evaluated at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveEnvelopeTractionQuery {
    /// Interface stiffness `K`.
    pub stiffness: f32,
    /// Cohesive strength `sigma_c`.
    pub strength: f32,
    /// Onset separation `delta0`.
    pub onset_separation: f32,
    /// Final separation `delta_f`.
    pub final_separation: f32,
    /// Effective separation `lambda`.
    pub lambda: f32,
}

impl CohesiveEnvelopeTractionQuery {
    /// Builds a query from the four envelope parameters and the separation.
    #[must_use]
    pub fn new(
        stiffness: f32,
        strength: f32,
        onset_separation: f32,
        final_separation: f32,
        lambda: f32,
    ) -> CohesiveEnvelopeTractionQuery {
        CohesiveEnvelopeTractionQuery {
            stiffness,
            strength,
            onset_separation,
            final_separation,
            lambda,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CohesiveZone::envelope_traction` output for that configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveEnvelopeTractionResult {
    /// The envelope traction magnitude at `lambda`.
    pub traction: f32,
}

/// Encodes one [`CohesiveEnvelopeTractionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &CohesiveEnvelopeTractionQuery) -> GpuQuery {
    GpuQuery {
        stiffness: q.stiffness,
        strength: q.strength,
        onset_separation: q.onset_separation,
        final_separation: q.final_separation,
        lambda: q.lambda,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CohesiveEnvelopeTractionResult`].
fn decode_result(raw: &GpuResult) -> CohesiveEnvelopeTractionResult {
    CohesiveEnvelopeTractionResult {
        traction: raw.traction,
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

/// A compiled, reusable cohesive-envelope traction compute pipeline, twinning
/// the `CPU` golden `CohesiveZone::envelope_traction`.
pub struct GpuCohesiveEnvelopeTraction {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCohesiveEnvelopeTraction {
    /// Compiles the cohesive-envelope traction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCohesiveEnvelopeTraction {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction"),
            source: ShaderSource::Wgsl(COHESIVE_ENVELOPE_TRACTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCohesiveEnvelopeTraction {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CohesiveEnvelopeTractionResult`] per input, in order.
    ///
    /// The `traction` scalar matches the reference to the module's tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CohesiveEnvelopeTractionQuery],
    ) -> Vec<CohesiveEnvelopeTractionResult> {
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
            label: Some("prism_volumetric_cohesive_envelope_traction_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_bind_group"),
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
            label: Some("prism_volumetric_cohesive_envelope_traction_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cohesive_envelope_traction_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cohesive_envelope_traction_pass"),
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
