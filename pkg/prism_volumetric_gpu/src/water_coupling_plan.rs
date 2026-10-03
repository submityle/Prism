//! `wgpu` compute twin of the two-way coupling sub-step scheduler
//! ([`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling)).
//!
//! Two-way fluid/rigid coupling runs a small fixed number of sub-steps per
//! frame so the fluid and rigid-body integrators do not jitter against each
//! other, and batches field read-backs into a bounded size rather than reading
//! the whole `GPU` field every frame. The schedule is a pure, closed-form
//! function of the frame's relative speed, time step, cell size, and the
//! per-frame caps: a `Courant`-like crossing count sets the sub-step demand,
//! which is floored at one and clamped to a cap, while the read-back batch is
//! the query count capped at a maximum. All of it is integer `min`/`max` plus
//! one guarded divide and a single `f32`-to-`u32` truncation, so it ports
//! cleanly to the device, and a passing real-device parity run is direct
//! evidence the ported kernel folds the same arithmetic the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one query. It reproduces
//! [`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling)
//! exactly: `cap = max(max_substeps, 1)`; `cell = max(dx, EPS)`;
//! `crossings = max(max_rel_speed, 0) * max(dt, 0) / cell`;
//! `needed = 1 + (crossings as u32)` (non-negative truncation toward zero);
//! `substeps = min(needed, cap)`; and
//! `readback_batch = min(query_count, max_readback)`. Both outputs are `u32`
//! and are pinned with exact equality.
//!
//! # What stays on the host
//!
//! The buoyancy/drag/added-mass force math, the source-writeback weighting, the
//! query batching itself, and the actual field read-back are host (or other
//! kernel) responsibilities. The device sees only the six scalar scheduling
//! inputs per query.
//!
//! # Correctness model
//!
//! The only floating-point work is `max(max_rel_speed, 0) * max(dt, 0) / cell`
//! followed by a truncation to `u32`; every other step is integer `min`/`max`
//! and addition. Both outputs are therefore integers compared with exact `==`.
//! Callers must keep `crossings` in the safe, non-saturating `u32` range and
//! away from integer boundaries so the `CPU` `as u32` and the device
//! `u32(floor(...))` truncate to the same integer.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `*`, `/`, `floor`,
//! integer `min`/`max`, and an `f32`-to-`u32` conversion — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`,
//! no `round`, no `cbrt`, and no `sqrt`. Each thread performs a bounded,
//! branch-free sequence, so the kernel provably terminates. No optional device
//! feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coupling`；无第三方引擎源码或衍生代码。
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

/// The inlined `WGSL` twin of
/// [`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling):
/// one thread per query, computing the sub-step count and read-back batch with
/// integer `min`/`max`, one guarded divide, and a single truncation.
const WATER_COUPLING_PLAN_WGSL: &str = r#"
// Twin of water::coupling::plan_coupling. One thread per query:
//   cap        = max(max_substeps, 1)
//   cell       = max(dx, EPS)
//   crossings  = max(max_rel_speed, 0) * max(dt, 0) / cell
//   needed     = 1 + u32(floor(crossings))
//   substeps   = min(needed, cap)
//   readback   = min(query_count, max_readback)
// Only integer min/max, one divide, one f32->u32 truncation; no transcendental
// and no sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::coupling；无第三方引擎源码或衍生代码。

// Rest threshold matching water::EPS; the cell size never falls below it.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Number of field queries this frame.
    query_count: u32,
    // Fastest body speed relative to the fluid.
    max_rel_speed: f32,
    // Frame time step.
    dt: f32,
    // Grid cell size.
    dx: f32,
    // Upper bound on sub-steps.
    max_substeps: u32,
    // Upper bound on the read-back batch.
    max_readback: u32,
}

struct Result {
    // Chosen fixed-point sub-step count.
    substeps: u32,
    // Chosen read-back batch size.
    readback_batch: u32,
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
    let cap = max(q.max_substeps, 1u);
    let cell = max(q.dx, EPS);
    let crossings = max(q.max_rel_speed, 0.0) * max(q.dt, 0.0) / cell;
    let needed = 1u + u32(floor(crossings));
    results[idx].substeps = min(needed, cap);
    results[idx].readback_batch = min(q.query_count, q.max_readback);
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_COUPLING_PLAN_WGSL`].
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

/// `repr(C)` `std430` layout of one coupling-plan query, matching the `WGSL`
/// `Query` struct: six scalar scheduling inputs (all alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Number of field queries this frame.
    query_count: u32,
    /// Fastest body speed relative to the fluid.
    max_rel_speed: f32,
    /// Frame time step.
    dt: f32,
    /// Grid cell size.
    dx: f32,
    /// Upper bound on sub-steps.
    max_substeps: u32,
    /// Upper bound on the read-back batch.
    max_readback: u32,
}

/// `repr(C)` `std430` layout of one coupling-plan result, matching the `WGSL`
/// `Result` struct: the two `u32` schedule outputs (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Chosen fixed-point sub-step count.
    substeps: u32,
    /// Chosen read-back batch size.
    readback_batch: u32,
}

/// One coupling-plan query to run on the device, mirroring the inputs of
/// [`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCouplingPlanQuery {
    /// Number of field queries this frame.
    pub query_count: u32,
    /// Fastest body speed relative to the fluid.
    pub max_rel_speed: f32,
    /// Frame time step.
    pub dt: f32,
    /// Grid cell size.
    pub dx: f32,
    /// Upper bound on sub-steps.
    pub max_substeps: u32,
    /// Upper bound on the read-back batch.
    pub max_readback: u32,
}

/// One coupling-plan result, mirroring the golden
/// [`CouplingPlan`](prism_render_architecture::water::coupling::CouplingPlan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaterCouplingPlanResult {
    /// Chosen fixed-point sub-step count.
    pub substeps: u32,
    /// Chosen read-back batch size.
    pub readback_batch: u32,
}

/// Encodes one [`WaterCouplingPlanQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterCouplingPlanQuery) -> GpuQuery {
    GpuQuery {
        query_count: q.query_count,
        max_rel_speed: q.max_rel_speed,
        dt: q.dt,
        dx: q.dx,
        max_substeps: q.max_substeps,
        max_readback: q.max_readback,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterCouplingPlanResult`].
fn decode_result(raw: &GpuResult) -> WaterCouplingPlanResult {
    WaterCouplingPlanResult {
        substeps: raw.substeps,
        readback_batch: raw.readback_batch,
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

/// A compiled, reusable coupling-plan compute pipeline, twinning the `CPU`
/// golden
/// [`plan_coupling`](prism_render_architecture::water::coupling::plan_coupling).
pub struct GpuWaterCouplingPlan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCouplingPlan {
    /// Compiles the coupling-plan kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCouplingPlan {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_coupling_plan"),
            source: ShaderSource::Wgsl(WATER_COUPLING_PLAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCouplingPlan {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one
    /// [`WaterCouplingPlanResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterCouplingPlanQuery],
    ) -> Vec<WaterCouplingPlanResult> {
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
            label: Some("prism_volumetric_water_coupling_plan_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_bind_group"),
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
            label: Some("prism_volumetric_water_coupling_plan_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_coupling_plan_encoder"),
        });
        {
            // One thread per query.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_coupling_plan_pass"),
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
