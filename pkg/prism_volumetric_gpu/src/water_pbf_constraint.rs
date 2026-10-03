//! `wgpu` compute twin of the stateless Position-Based-Fluids density-solve
//! primitives
//! [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
//! and
//! [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda).
//!
//! Position-Based Fluids (Macklin & Müller) binds an incompressible liquid by a
//! single per-particle constraint: hold the `SPH`-estimated density at the rest
//! density `rho_0`. Each solver iteration forms the density constraint `C_i =
//! rho_i / rho_0 - 1` and the `XPBD` scaling factor `lambda_i = -C_i / (|grad
//! C_i|^2 / rho_0^2 + epsilon)`. Both are pure algebra — divide, multiply, add
//! and ordered comparison — with no floating-point transcendental, so the port
//! is faithful.
//!
//! [`GpuWaterPbfConstraint`] is the on-device twin of those two primitives. One
//! thread solves one query, reproducing the reference's rest-density guard, the
//! density constraint and the clamped `XPBD` scaling factor, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same scheduling the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces both reference outputs for each query:
//!
//! * `constraint`:
//!   [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
//!   — `0` when the rest density is at or below the shared `EPS`, else `density
//!   / rest_density - 1`.
//! * `lambda`:
//!   [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda)
//!   — `0` when the rest density is at or below `EPS`; otherwise the gradient
//!   denominator `(|grad_sum|^2 + max(grad_sq_sum, 0)) / rho_0^2 + max(epsilon,
//!   0)` is formed, and the result is `0` when that denominator is at or below
//!   `EPS`, else `-constraint / denominator`.
//!
//! The squared gradient length `|grad_sum|^2 = x*x + y*y + z*z` is the
//! reference `Vec3::length_squared`, reproduced inline with multiply and add so
//! no `sqrt` is needed.
//!
//! # What stays on the host
//!
//! Nothing in these two primitives is stateful: they are closed-form `f32`
//! maps. The surrounding `PBF` solve — the spatial-hash neighbour binning, the
//! `Poly6` density sum, the `Spiky` gradient gather, the artificial-pressure
//! correction and the position-update accumulation — is separate host code and
//! is never dispatched by this twin.
//!
//! # Correctness model
//!
//! The host and the device evaluate the identical rest-density guard, density
//! ratio and gradient-denominator algebra, so the outputs agree to within
//! floating-point tolerance and the parity test asserts both with an
//! absolute-or-relative closeness check. The two discrete branch decisions —
//! the rest-density guard and the denominator guard — are kept away from their
//! `EPS` crossings by the fixtures and reject-sampling, so a last-place
//! rounding difference cannot flip a branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, multiply,
//! divide, subtract, negate and ordered comparison — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, no inverse trigonometry, no `sqrt` and no `u64`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `PBF` density-constraint kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` goldens
/// [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
/// and
/// [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda);
/// see the module documentation for the algorithm.
const WATER_PBF_CONSTRAINT_WGSL: &str = r#"
// Position-Based-Fluids density-solve twin: one thread resolves one query into
// the density constraint (rho/rho_0 - 1, guarded at rest density <= EPS) and
// the XPBD scaling factor lambda (-C / (|grad_sum|^2 + max(grad_sq_sum, 0)) /
// rho_0^2 + max(epsilon, 0)), guarded at a near-zero denominator), mirroring
// the CPU goldens `water::pbf::{density_constraint, constraint_lambda}` with
// only max, multiply, divide, subtract, negate and ordered comparison. It owns
// no neighbour binning, kernel sums or position update; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pbf；无第三方引擎源码
// 或衍生代码。

// Shared near-zero guard constant, matching `prism_render_architecture::water::EPS`.
const EPS: f32 = 1e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // SPH-estimated density rho_i.
    density: f32,
    // Rest density rho_0; a value at or below EPS is the degenerate no-constraint case.
    rest_density: f32,
    // Accumulated constraint gradient grad_sum = sum_j grad W_ij (x, y, z).
    grad_sum_x: f32,
    grad_sum_y: f32,
    grad_sum_z: f32,
    // Accumulated squared gradient grad_sq_sum = sum_j |grad W_ij|^2.
    grad_sq_sum: f32,
    // XPBD relaxation/compliance term epsilon.
    epsilon: f32,
}

struct Result {
    // Density constraint C_i = rho_i / rho_0 - 1 (0 in the degenerate case).
    constraint: f32,
    // XPBD scaling factor lambda_i (0 in either degenerate case).
    lambda: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Density constraint C_i = rho_i / rho_0 - 1, with the degenerate rest density
// (<= EPS) returning 0. A verbatim port of the reference `density_constraint`.
fn density_constraint(density: f32, rest_density: f32) -> f32 {
    if (rest_density <= EPS) {
        return 0.0;
    }
    return density / rest_density - 1.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.constraint = density_constraint(q.density, q.rest_density);

    if (q.rest_density <= EPS) {
        // Degenerate rest density: no constraint and no correction.
        out.lambda = 0.0;
    } else {
        let c = out.constraint;
        let inv_rho2 = 1.0 / (q.rest_density * q.rest_density);
        let len_sq =
            q.grad_sum_x * q.grad_sum_x
            + q.grad_sum_y * q.grad_sum_y
            + q.grad_sum_z * q.grad_sum_z;
        let denom = (len_sq + max(q.grad_sq_sum, 0.0)) * inv_rho2 + max(q.epsilon, 0.0);
        if (denom <= EPS) {
            // Near-zero denominator: the compliance term fully relaxes lambda.
            out.lambda = 0.0;
        } else {
            out.lambda = -c / denom;
        }
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_PBF_CONSTRAINT_WGSL`].
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
/// the density, the rest density, the three gradient-sum components, the
/// squared-gradient sum and the relaxation epsilon.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `SPH`-estimated density (the golden `density`).
    density: f32,
    /// Rest density (the golden `rest_density`).
    rest_density: f32,
    /// `X` component of the gradient sum (the golden `grad_sum.x`).
    grad_sum_x: f32,
    /// `Y` component of the gradient sum (the golden `grad_sum.y`).
    grad_sum_y: f32,
    /// `Z` component of the gradient sum (the golden `grad_sum.z`).
    grad_sum_z: f32,
    /// Squared-gradient sum (the golden `grad_sq_sum`).
    grad_sq_sum: f32,
    /// Relaxation/compliance term (the golden `epsilon`).
    epsilon: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the density constraint and the `XPBD` scaling factor.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Density constraint `C_i` (the golden `density_constraint`).
    constraint: f32,
    /// `XPBD` scaling factor `lambda_i` (the golden `constraint_lambda`).
    lambda: f32,
}

/// One `PBF` density-solve query: the density, the rest density, the gradient
/// sum, the squared-gradient sum and the relaxation epsilon, mirroring the
/// arguments the goldens
/// [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
/// and
/// [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda)
/// read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfConstraintQuery {
    /// `SPH`-estimated density `rho_i` (the golden `density`).
    pub density: f32,
    /// Rest density `rho_0` (the golden `rest_density`); a value at or below
    /// the shared `EPS` is the degenerate no-constraint case.
    pub rest_density: f32,
    /// `X` component of the gradient sum (the golden `grad_sum.x`).
    pub grad_sum_x: f32,
    /// `Y` component of the gradient sum (the golden `grad_sum.y`).
    pub grad_sum_y: f32,
    /// `Z` component of the gradient sum (the golden `grad_sum.z`).
    pub grad_sum_z: f32,
    /// Squared-gradient sum (the golden `grad_sq_sum`); floored to zero.
    pub grad_sq_sum: f32,
    /// Relaxation/compliance term (the golden `epsilon`); floored to zero.
    pub epsilon: f32,
}

impl WaterPbfConstraintQuery {
    /// Builds a query from the density, the rest density, the three gradient-sum
    /// components, the squared-gradient sum and the relaxation epsilon.
    #[must_use]
    pub const fn new(
        density: f32,
        rest_density: f32,
        grad_sum_x: f32,
        grad_sum_y: f32,
        grad_sum_z: f32,
        grad_sq_sum: f32,
        epsilon: f32,
    ) -> WaterPbfConstraintQuery {
        WaterPbfConstraintQuery {
            density,
            rest_density,
            grad_sum_x,
            grad_sum_y,
            grad_sum_z,
            grad_sq_sum,
            epsilon,
        }
    }
}

/// One resolved `PBF` density-solve response, mirroring the pair of golden
/// outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfConstraintResult {
    /// Density constraint `C_i` (the golden `density_constraint`).
    pub constraint: f32,
    /// `XPBD` scaling factor `lambda_i` (the golden `constraint_lambda`).
    pub lambda: f32,
}

/// Encodes one [`WaterPbfConstraintQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterPbfConstraintQuery) -> GpuQuery {
    GpuQuery {
        density: q.density,
        rest_density: q.rest_density,
        grad_sum_x: q.grad_sum_x,
        grad_sum_y: q.grad_sum_y,
        grad_sum_z: q.grad_sum_z,
        grad_sq_sum: q.grad_sq_sum,
        epsilon: q.epsilon,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterPbfConstraintResult`].
fn decode_result(raw: &GpuResult) -> WaterPbfConstraintResult {
    WaterPbfConstraintResult {
        constraint: raw.constraint,
        lambda: raw.lambda,
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

/// A compiled, reusable `PBF` density-constraint compute pipeline, twinning the
/// stateless `f32` primitives of the `CPU` goldens
/// [`density_constraint`](prism_render_architecture::water::pbf::density_constraint)
/// and
/// [`constraint_lambda`](prism_render_architecture::water::pbf::constraint_lambda).
pub struct GpuWaterPbfConstraint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterPbfConstraint {
    /// Compiles the `PBF` density-constraint kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterPbfConstraint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint"),
            source: ShaderSource::Wgsl(WATER_PBF_CONSTRAINT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterPbfConstraint {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`WaterPbfConstraintResult`] per input, in order.
    ///
    /// Both the constraint and the scaling factor equal the reference to within
    /// floating-point tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterPbfConstraintQuery],
    ) -> Vec<WaterPbfConstraintResult> {
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
            label: Some("prism_volumetric_water_pbf_constraint_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_bind_group"),
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
            label: Some("prism_volumetric_water_pbf_constraint_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_pbf_constraint_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_pbf_constraint_pass"),
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
