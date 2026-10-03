//! `wgpu` compute twin of the `PBF` density-solve scheduler
//! ([`plan_solve`](prism_render_architecture::water::pbf::plan_solve)).
//!
//! Position-based fluids run an `XPBD` density solve once per sub-step. Before
//! the solve, [`plan_solve`](prism_render_architecture::water::pbf::plan_solve)
//! turns the tuning parameters into a tiny deterministic schedule: how many
//! constraint-projection iterations to run, and whether the artificial-pressure
//! term participates. Both answers are pure, closed-form functions of two
//! parameter fields, so the schedule ports cleanly to the device, and a passing
//! real-device parity run is direct evidence the ported kernel folds the same
//! arithmetic the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one parameter set. It reproduces
//! [`plan_solve`](prism_render_architecture::water::pbf::plan_solve) exactly:
//! `iterations = max(solver_iterations, 1)` and
//! `artificial_pressure = artificial_pressure_k > EPS`, with `EPS = 1.0e-6`
//! matching [`prism_render_architecture::water`]. The iteration count is a
//! `u32`; the artificial-pressure flag is reported as a `u32` (`1` for enabled,
//! `0` for disabled). Both are pinned with exact equality.
//!
//! # What stays on the host
//!
//! The rest of [`PbfParams`](prism_render_architecture::water::pbf::PbfParams) —
//! the rest density, particle mass, smoothing radius, relaxation epsilon,
//! artificial-pressure exponent and reference fraction — feeds the per-particle
//! constraint solve, not the schedule, so this twin never sees it. The actual
//! density solve, the neighbour gather, and the position correction remain host
//! or other-kernel responsibilities.
//!
//! # Correctness model
//!
//! Both outputs are integers: the iteration count comes from an integer `max`,
//! and the flag from a single ordered `>` comparison against `EPS`. There is no
//! continuous output, so the parity test asserts exact `==` on each. The
//! comparison is strict, so a strength exactly equal to `EPS` disables the
//! term; fixtures straddle that boundary explicitly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — an integer `max`, one
//! float comparison, and a `select` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no `round`, no
//! `cbrt`, and no `sqrt`. Each thread performs a bounded, branch-free sequence,
//! so the kernel provably terminates. No optional device feature is required,
//! so it runs unmodified across backends.
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

/// Number of threads per workgroup. The schedule is a single-thread-per-element
/// kernel, so one thread per workgroup keeps the dispatch trivial.
const WORKGROUP_SIZE: u32 = 1;

/// The inlined `WGSL` twin of
/// [`plan_solve`](prism_render_architecture::water::pbf::plan_solve): one thread
/// per parameter set, computing the iteration count with an integer `max` and
/// the artificial-pressure flag with one ordered comparison and a `select`.
const WATER_PBF_PLAN_WGSL: &str = r#"
// Twin of water::pbf::plan_solve. One thread per parameter set:
//   iterations          = max(solver_iterations, 1)
//   artificial_pressure = (artificial_pressure_k > EPS) ? 1 : 0
// Only an integer max, one float comparison, and a select; no transcendental
// and no sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pbf；无第三方引擎源码或衍生代码。

// Rest threshold matching water::EPS; the strength must strictly exceed it.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of parameter sets in the storage arrays; threads past this stop.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Number of constraint-projection iterations requested.
    solver_iterations: u32,
    // Artificial-pressure strength k.
    artificial_pressure_k: f32,
}

struct Result {
    // Clamped iteration count (at least one).
    iterations: u32,
    // Whether artificial pressure participates (1 enabled, 0 disabled).
    artificial_pressure: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(1)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    results[idx].iterations = max(q.solver_iterations, 1u);
    results[idx].artificial_pressure = select(0u, 1u, q.artificial_pressure_k > EPS);
}
"#;

/// Uniform parameters for one dispatch: the count plus three pad words to fill
/// a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_PBF_PLAN_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid parameter sets in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one plan query, matching the `WGSL` `Query`
/// struct: the two schedule-relevant parameter fields (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Number of constraint-projection iterations requested.
    solver_iterations: u32,
    /// Artificial-pressure strength `k`.
    artificial_pressure_k: f32,
}

/// `repr(C)` `std430` layout of one plan result, matching the `WGSL` `Result`
/// struct: the iteration count and the artificial-pressure flag (alignment
/// `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped iteration count (at least one).
    iterations: u32,
    /// Whether artificial pressure participates (`1` enabled, `0` disabled).
    artificial_pressure: u32,
}

/// One `PBF` plan query to run on the device, carrying the two fields of
/// [`PbfParams`](prism_render_architecture::water::pbf::PbfParams) that
/// [`plan_solve`](prism_render_architecture::water::pbf::plan_solve) reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPbfPlanQuery {
    /// Number of constraint-projection iterations requested.
    pub solver_iterations: u32,
    /// Artificial-pressure strength `k`.
    pub artificial_pressure_k: f32,
}

/// One `PBF` plan result, mirroring the golden
/// [`PbfSolvePlan`](prism_render_architecture::water::pbf::PbfSolvePlan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaterPbfPlanResult {
    /// Number of constraint-projection iterations to run (at least one).
    pub iterations: u32,
    /// Whether artificial pressure participates (`1` enabled, `0` disabled).
    pub artificial_pressure: u32,
}

/// Encodes one [`WaterPbfPlanQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterPbfPlanQuery) -> GpuQuery {
    GpuQuery {
        solver_iterations: q.solver_iterations,
        artificial_pressure_k: q.artificial_pressure_k,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterPbfPlanResult`].
fn decode_result(raw: &GpuResult) -> WaterPbfPlanResult {
    WaterPbfPlanResult {
        iterations: raw.iterations,
        artificial_pressure: raw.artificial_pressure,
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

/// A compiled, reusable `PBF` plan compute pipeline, twinning the `CPU` golden
/// [`plan_solve`](prism_render_architecture::water::pbf::plan_solve).
pub struct GpuWaterPbfPlan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterPbfPlan {
    /// Compiles the `PBF` plan kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterPbfPlan {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_pbf_plan"),
            source: ShaderSource::Wgsl(WATER_PBF_PLAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterPbfPlan {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one [`WaterPbfPlanResult`] per
    /// input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterPbfPlanQuery],
    ) -> Vec<WaterPbfPlanResult> {
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
            label: Some("prism_volumetric_water_pbf_plan_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_bind_group"),
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
            label: Some("prism_volumetric_water_pbf_plan_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_pbf_plan_encoder"),
        });
        {
            // One thread per parameter set.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_pbf_plan_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
