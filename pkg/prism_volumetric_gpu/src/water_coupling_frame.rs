//! `wgpu` compute twin of the per-frame two-way coupling assembly
//! ([`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame)).
//!
//! Two-way fluid/rigid coupling assembles, once per frame, the `Archimedes`
//! buoyancy, the quadratic form-drag, the added-mass reaction, the fraction of
//! body momentum stamped back into the fluid, and the sub-step/read-back
//! schedule. Each piece is a pure, closed-form function of a static
//! body/fluid `profile` and the current frame `inputs`, so the whole frame plan
//! is a deterministic composition that ports cleanly to the device. A passing
//! real-device parity run is direct evidence the ported kernel folds the same
//! arithmetic the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one `(profile, inputs)` pair. It reproduces
//! [`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame)
//! exactly by composing the five
//! [`coupling`](prism_render_architecture::water::coupling) primitives:
//!
//! - the schedule `plan_coupling`: `cap = max(max_substeps, 1)`,
//!   `cell = max(cell_size, EPS)`,
//!   `crossings = max(max_rel_speed, 0) * max(frame_dt, 0) / cell`,
//!   `substeps = min(1 + (crossings as u32), cap)`, and
//!   `readback_batch = min(query_count, max_readback)`;
//! - `buoyancy = max(fluid_density, 0) * max(submerged_volume, 0) * max(GRAVITY, 0)`;
//! - `drag = 0.5 * max(drag_coeff, 0) * max(fluid_density, 0) * max(cross_section, 0) * v * v`
//!   with `v = max(rel_speed, 0)`;
//! - `added_mass = max(added_mass_coeff, 0) * max(fluid_density, 0) * max(submerged_volume, 0)`;
//! - `writeback_fraction`: with `total = max(total_volume, 0)`, returns `0` when
//!   `total <= EPS`, else `clamp(max(submerged_volume, 0) / total, 0, 1)`.
//!
//! `GRAVITY` is `9.81` and `EPS` is `1.0e-6`, matching
//! [`prism_render_architecture::water`]. The two schedule outputs are `u32`
//! (pinned with exact equality); the four forces/fractions are `f32` (pinned
//! within tolerance).
//!
//! # What stays on the host
//!
//! The variable-length query batching and the actual bounded `GPU` field
//! read-back are host responsibilities. The device sees only the thirteen
//! scalar `profile`/`inputs` fields per frame.
//!
//! # Correctness model
//!
//! The schedule outputs are integers built from one divide and a truncation, so
//! they are compared with exact `==`; callers keep `crossings` in the safe,
//! non-saturating range and away from integer boundaries (or let the sub-step
//! `cap` bind the result) so the `CPU` `as u32` and device `u32(floor(..))`
//! agree. The forces thread through multiplies, a divide, and `clamp`, so the
//! `CPU` and `GPU` are not bit-exact and the parity test asserts an absolute or
//! relative tolerance on each.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `*`, `/`, `floor`,
//! `clamp`, integer and float `min`/`max`, and an `f32`-to-`u32` conversion —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `smoothstep`, no `round`, no `cbrt`, and no `sqrt`. Each thread performs a
//! bounded, branch-light sequence, so the kernel provably terminates. No
//! optional device feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coupling_frame`；无第三方引擎源码或衍生代码。
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
/// [`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame):
/// one thread per frame, composing the schedule and the buoyancy/drag/added-mass
/// forces plus the write-back fraction with integer and float `min`/`max`, one
/// guarded divide for the schedule, one guarded divide for the fraction, a
/// `clamp`, and a single truncation.
const WATER_COUPLING_FRAME_WGSL: &str = r#"
// Twin of water::coupling_frame::plan_coupling_frame. One thread per frame:
//   schedule (plan_coupling):
//     cap        = max(max_substeps, 1)
//     cell       = max(cell_size, EPS)
//     crossings  = max(max_rel_speed, 0) * max(frame_dt, 0) / cell
//     substeps   = min(1 + u32(floor(crossings)), cap)
//     readback   = min(query_count, max_readback)
//   buoyancy  = max(fluid_density,0) * max(submerged_volume,0) * max(GRAVITY,0)
//   drag      = 0.5 * max(drag_coeff,0) * max(fluid_density,0)
//                   * max(cross_section,0) * v * v,  v = max(rel_speed,0)
//   added     = max(added_mass_coeff,0) * max(fluid_density,0)
//                   * max(submerged_volume,0)
//   writeback = (total<=EPS) ? 0 : clamp(max(submerged_volume,0)/total, 0, 1),
//               total = max(total_volume, 0)
// Only integer/float min/max, two divides, one clamp, one f32->u32 truncation;
// no transcendental and no sqrt.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::coupling_frame；无第三方引擎源码或衍生代码。

// Rest threshold matching water::EPS; cell size and total volume floor on it.
const EPS: f32 = 1.0e-6;
// Standard gravity matching water::GRAVITY.
const GRAVITY: f32 = 9.81;

struct Params {
    // Number of frames in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Density of the surrounding fluid.
    fluid_density: f32,
    // Quadratic form-drag coefficient.
    drag_coeff: f32,
    // Added-mass coefficient for the body shape.
    added_mass_coeff: f32,
    // Upper bound on coupling sub-steps per frame.
    max_substeps: u32,
    // Upper bound on the read-back batch.
    max_readback: u32,
    // Number of field queries this frame.
    query_count: u32,
    // Fastest relative body/fluid speed, for scheduling.
    max_rel_speed: f32,
    // Frame time step.
    frame_dt: f32,
    // Fluid grid cell size.
    cell_size: f32,
    // Body volume below the surface.
    submerged_volume: f32,
    // Total body volume.
    total_volume: f32,
    // Cross-sectional area presented to the flow.
    cross_section: f32,
    // Relative body/fluid speed used for drag.
    rel_speed: f32,
}

struct Result {
    // Chosen fixed-point sub-step count.
    substeps: u32,
    // Chosen read-back batch size.
    readback_batch: u32,
    // Archimedes buoyancy force magnitude.
    buoyancy: f32,
    // Quadratic drag force magnitude.
    drag: f32,
    // Added-mass reaction magnitude.
    added_mass: f32,
    // Fraction of body momentum written back into the fluid.
    writeback_fraction: f32,
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

    // Schedule (plan_coupling).
    let cap = max(q.max_substeps, 1u);
    let cell = max(q.cell_size, EPS);
    let crossings = max(q.max_rel_speed, 0.0) * max(q.frame_dt, 0.0) / cell;
    let needed = 1u + u32(floor(crossings));
    results[idx].substeps = min(needed, cap);
    results[idx].readback_batch = min(q.query_count, q.max_readback);

    // Forces.
    let density = max(q.fluid_density, 0.0);
    let submerged = max(q.submerged_volume, 0.0);
    results[idx].buoyancy = density * submerged * max(GRAVITY, 0.0);

    let v = max(q.rel_speed, 0.0);
    results[idx].drag =
        0.5 * max(q.drag_coeff, 0.0) * density * max(q.cross_section, 0.0) * v * v;

    results[idx].added_mass = max(q.added_mass_coeff, 0.0) * density * submerged;

    // Write-back fraction with a guarded divide.
    let total = max(q.total_volume, 0.0);
    if (total <= EPS) {
        results[idx].writeback_fraction = 0.0;
    } else {
        results[idx].writeback_fraction = clamp(submerged / total, 0.0, 1.0);
    }
}
"#;

/// Uniform parameters for one dispatch: the frame count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_COUPLING_FRAME_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid frames in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one coupling-frame query, matching the `WGSL`
/// `Query` struct: five `profile` fields then eight `inputs` fields, all
/// alignment `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Density of the surrounding fluid.
    fluid_density: f32,
    /// Quadratic form-drag coefficient.
    drag_coeff: f32,
    /// Added-mass coefficient for the body shape.
    added_mass_coeff: f32,
    /// Upper bound on coupling sub-steps per frame.
    max_substeps: u32,
    /// Upper bound on the read-back batch.
    max_readback: u32,
    /// Number of field queries this frame.
    query_count: u32,
    /// Fastest relative body/fluid speed, for scheduling.
    max_rel_speed: f32,
    /// Frame time step.
    frame_dt: f32,
    /// Fluid grid cell size.
    cell_size: f32,
    /// Body volume below the surface.
    submerged_volume: f32,
    /// Total body volume.
    total_volume: f32,
    /// Cross-sectional area presented to the flow.
    cross_section: f32,
    /// Relative body/fluid speed used for drag.
    rel_speed: f32,
}

/// `repr(C)` `std430` layout of one coupling-frame result, matching the `WGSL`
/// `Result` struct: the two `u32` schedule outputs then the four `f32`
/// forces/fractions (alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Chosen fixed-point sub-step count.
    substeps: u32,
    /// Chosen read-back batch size.
    readback_batch: u32,
    /// `Archimedes` buoyancy force magnitude.
    buoyancy: f32,
    /// Quadratic drag force magnitude.
    drag: f32,
    /// Added-mass reaction magnitude.
    added_mass: f32,
    /// Fraction of body momentum written back into the fluid.
    writeback_fraction: f32,
}

/// One coupling-frame query to run on the device, flattening the `profile` and
/// `inputs` of
/// [`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCouplingFrameQuery {
    /// Density of the surrounding fluid.
    pub fluid_density: f32,
    /// Quadratic form-drag coefficient.
    pub drag_coeff: f32,
    /// Added-mass coefficient for the body shape.
    pub added_mass_coeff: f32,
    /// Upper bound on coupling sub-steps per frame.
    pub max_substeps: u32,
    /// Upper bound on the read-back batch.
    pub max_readback: u32,
    /// Number of field queries this frame.
    pub query_count: u32,
    /// Fastest relative body/fluid speed, for scheduling.
    pub max_rel_speed: f32,
    /// Frame time step.
    pub frame_dt: f32,
    /// Fluid grid cell size.
    pub cell_size: f32,
    /// Body volume below the surface.
    pub submerged_volume: f32,
    /// Total body volume.
    pub total_volume: f32,
    /// Cross-sectional area presented to the flow.
    pub cross_section: f32,
    /// Relative body/fluid speed used for drag.
    pub rel_speed: f32,
}

/// One coupling-frame result, mirroring the golden
/// [`CouplingFramePlan`](prism_render_architecture::water::coupling_frame::CouplingFramePlan).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCouplingFrameResult {
    /// Chosen fixed-point sub-step count.
    pub substeps: u32,
    /// Chosen read-back batch size.
    pub readback_batch: u32,
    /// `Archimedes` buoyancy force magnitude, directed upward.
    pub buoyancy: f32,
    /// Quadratic drag force magnitude, opposing relative motion.
    pub drag: f32,
    /// Added-mass reaction magnitude from the entrained fluid.
    pub added_mass: f32,
    /// Fraction of body momentum written back into the fluid, in `0..=1`.
    pub writeback_fraction: f32,
}

/// Encodes one [`WaterCouplingFrameQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterCouplingFrameQuery) -> GpuQuery {
    GpuQuery {
        fluid_density: q.fluid_density,
        drag_coeff: q.drag_coeff,
        added_mass_coeff: q.added_mass_coeff,
        max_substeps: q.max_substeps,
        max_readback: q.max_readback,
        query_count: q.query_count,
        max_rel_speed: q.max_rel_speed,
        frame_dt: q.frame_dt,
        cell_size: q.cell_size,
        submerged_volume: q.submerged_volume,
        total_volume: q.total_volume,
        cross_section: q.cross_section,
        rel_speed: q.rel_speed,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterCouplingFrameResult`].
fn decode_result(raw: &GpuResult) -> WaterCouplingFrameResult {
    WaterCouplingFrameResult {
        substeps: raw.substeps,
        readback_batch: raw.readback_batch,
        buoyancy: raw.buoyancy,
        drag: raw.drag,
        added_mass: raw.added_mass,
        writeback_fraction: raw.writeback_fraction,
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

/// A compiled, reusable coupling-frame compute pipeline, twinning the `CPU`
/// golden
/// [`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame).
pub struct GpuWaterCouplingFrame {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCouplingFrame {
    /// Compiles the coupling-frame kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCouplingFrame {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_coupling_frame"),
            source: ShaderSource::Wgsl(WATER_COUPLING_FRAME_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCouplingFrame {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every frame in `queries` and returns one
    /// [`WaterCouplingFrameResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterCouplingFrameQuery],
    ) -> Vec<WaterCouplingFrameResult> {
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
            label: Some("prism_volumetric_water_coupling_frame_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_bind_group"),
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
            label: Some("prism_volumetric_water_coupling_frame_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_coupling_frame_encoder"),
        });
        {
            // One thread per frame.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_coupling_frame_pass"),
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
