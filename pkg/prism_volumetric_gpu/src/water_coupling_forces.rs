//! `wgpu` compute twin of the pure two-way fluid/rigid coupling force math
//! ([`coupling`](prism_render_architecture::water::coupling)).
//!
//! The `CPU` golden module
//! [`coupling`](prism_render_architecture::water::coupling) owns the
//! deterministic force and scheduling math of the two-way fluid/rigid exchange.
//! This twin reproduces three of its stateless, scalar primitives on device:
//! the quadratic hydrodynamic
//! [`drag_force`](prism_render_architecture::water::coupling::drag_force), the
//! [`added_mass`](prism_render_architecture::water::coupling::added_mass)
//! reaction, and the
//! [`source_writeback_fraction`](prism_render_architecture::water::coupling::source_writeback_fraction)
//! submersion weight. One thread computes one body's three coupling scalars.
//!
//! The buoyancy primitive
//! [`buoyancy_force`](prism_render_architecture::water::coupling::buoyancy_force)
//! and the variable-length `plan_coupling` schedule are intentionally out of
//! scope here; this module twins only the three force/fraction closed forms
//! listed above.
//!
//! # What is twinned
//!
//! For one body the kernel reproduces, each guarded with the same `max(0)`
//! flooring the reference uses:
//! - `drag_force(drag_coeff, fluid_density, area, rel_speed)`:
//!   `v = max(rel_speed, 0)`, then
//!   `0.5 * max(drag_coeff, 0) * max(fluid_density, 0) * max(area, 0) * v * v`.
//! - `added_mass(added_mass_coeff, fluid_density, displaced_volume)`:
//!   `max(added_mass_coeff, 0) * max(fluid_density, 0) * max(displaced_volume, 0)`.
//! - `source_writeback_fraction(submerged_volume, total_volume)`: with
//!   `total = max(total_volume, 0)`, returns `0` when `total <= EPS`
//!   (`EPS = 1e-6`), otherwise `clamp(max(submerged_volume, 0) / total, 0, 1)`.
//!
//! # What stays on the host
//!
//! Nothing of these three closed forms stays host-side; every step is portable
//! scalar arithmetic. The host only owns the empty-batch short-circuit (a
//! storage buffer cannot be zero-sized) and the packing of the public
//! [`WaterCouplingForcesQuery`] into its `std430` slot.
//!
//! # Correctness model
//!
//! The drag and added-mass outputs thread through plain multiplies and the
//! `max(0)` floors; the writeback fraction adds a divide, a `clamp`, and a
//! `total <= EPS` short-circuit. The `CPU` and `GPU` are not bit-exact across a
//! divide, so each continuous output is asserted within a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a wrong
//! port yet loose enough to admit a legal last-place difference. Fixtures keep
//! the writeback fraction a clear margin away from both clamp edges and from the
//! `EPS` total-volume threshold so the two agree on every branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `clamp`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, `sqrt`, no inverse trigonometry, no `round` or `ceil`,
//! and no `u64`/`u16`/`i64`/`f64`. No optional device feature is required, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` coupling-force kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`coupling`](prism_render_architecture::water::coupling) closed forms; see the
/// module documentation for the algorithm.
const WATER_COUPLING_FORCES_WGSL: &str = r#"
// Two-way coupling force twin: one thread computes a body's quadratic drag, its
// added-mass reaction, and its submersion writeback fraction, mirroring the CPU
// golden `water::coupling` closed forms with only max/clamp and + - * /. It owns
// no scheduling and no variable-length work.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::coupling；无第三方引擎
// 源码或衍生代码。

const EPS: f32 = 1.0e-6;

struct Params {
    // Number of bodies in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    drag_coeff: f32,
    fluid_density: f32,
    area: f32,
    rel_speed: f32,
    added_mass_coeff: f32,
    displaced_volume: f32,
    submerged_volume: f32,
    total_volume: f32,
}

struct Result {
    drag: f32,
    added_mass: f32,
    writeback_fraction: f32,
    pad0: f32,
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

    // Quadratic form drag: 0.5 * Cd * rho * A * v^2 with every input floored to
    // zero, matching the reference `drag_force`.
    let v = max(q.rel_speed, 0.0);
    let drag = 0.5
        * max(q.drag_coeff, 0.0)
        * max(q.fluid_density, 0.0)
        * max(q.area, 0.0)
        * v
        * v;

    // Added-mass reaction: Ca * rho * V_displaced, each input floored to zero.
    let added_mass = max(q.added_mass_coeff, 0.0)
        * max(q.fluid_density, 0.0)
        * max(q.displaced_volume, 0.0);

    // Writeback fraction: clamp(submerged / total, 0, 1), zero for a degenerate
    // (total <= EPS) total volume.
    let total = max(q.total_volume, 0.0);
    var writeback: f32 = 0.0;
    if (total > EPS) {
        writeback = clamp(max(q.submerged_volume, 0.0) / total, 0.0, 1.0);
    }

    var out: Result;
    out.drag = drag;
    out.added_mass = added_mass;
    out.writeback_fraction = writeback;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the body count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`WATER_COUPLING_FORCES_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid bodies in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one coupling query: eight scalars to a `32`-byte
/// stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Drag coefficient `Cd`.
    drag_coeff: f32,
    /// Fluid density `rho`.
    fluid_density: f32,
    /// Reference frontal area `A`.
    area: f32,
    /// Body speed relative to the fluid.
    rel_speed: f32,
    /// Added-mass coefficient `Ca`.
    added_mass_coeff: f32,
    /// Volume of fluid displaced by the body.
    displaced_volume: f32,
    /// Submerged body volume.
    submerged_volume: f32,
    /// Total body volume.
    total_volume: f32,
}

/// `repr(C)` `std430` layout of one coupling result, matching the `WGSL` `Result`
/// struct: three force/fraction scalars and one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Quadratic drag magnitude.
    drag: f32,
    /// Added-mass reaction magnitude.
    added_mass: f32,
    /// Submersion writeback fraction in `0..=1`.
    writeback_fraction: f32,
    /// Padding word.
    pad0: f32,
}

/// One coupling query: the eight scalars feeding the three twinned primitives of
/// the reference [`coupling`](prism_render_architecture::water::coupling) module.
///
/// `drag_coeff`, `fluid_density`, `area` and `rel_speed` feed
/// [`drag_force`](prism_render_architecture::water::coupling::drag_force);
/// `added_mass_coeff`, `fluid_density` and `displaced_volume` feed
/// [`added_mass`](prism_render_architecture::water::coupling::added_mass); and
/// `submerged_volume` with `total_volume` feed
/// [`source_writeback_fraction`](prism_render_architecture::water::coupling::source_writeback_fraction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCouplingForcesQuery {
    /// Drag coefficient `Cd`.
    pub drag_coeff: f32,
    /// Fluid density `rho`.
    pub fluid_density: f32,
    /// Reference frontal area `A`.
    pub area: f32,
    /// Body speed relative to the fluid.
    pub rel_speed: f32,
    /// Added-mass coefficient `Ca`.
    pub added_mass_coeff: f32,
    /// Volume of fluid displaced by the body.
    pub displaced_volume: f32,
    /// Submerged body volume.
    pub submerged_volume: f32,
    /// Total body volume.
    pub total_volume: f32,
}

/// One resolved coupling result: the three force/fraction scalars the reference
/// [`coupling`](prism_render_architecture::water::coupling) primitives produce.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCouplingForcesResult {
    /// Quadratic hydrodynamic drag magnitude.
    pub drag: f32,
    /// Added-mass reaction magnitude.
    pub added_mass: f32,
    /// Submersion writeback fraction in `0..=1`.
    pub writeback_fraction: f32,
}

/// Encodes one [`WaterCouplingForcesQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterCouplingForcesQuery) -> GpuQuery {
    GpuQuery {
        drag_coeff: q.drag_coeff,
        fluid_density: q.fluid_density,
        area: q.area,
        rel_speed: q.rel_speed,
        added_mass_coeff: q.added_mass_coeff,
        displaced_volume: q.displaced_volume,
        submerged_volume: q.submerged_volume,
        total_volume: q.total_volume,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterCouplingForcesResult`].
fn decode_result(raw: &GpuResult) -> WaterCouplingForcesResult {
    WaterCouplingForcesResult {
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

/// A compiled, reusable coupling-force compute pipeline, twinning the three
/// stateless primitives of the `CPU` golden
/// [`coupling`](prism_render_architecture::water::coupling) module.
pub struct GpuWaterCouplingForces {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCouplingForces {
    /// Compiles the coupling-force kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCouplingForces {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_coupling_forces"),
            source: ShaderSource::Wgsl(WATER_COUPLING_FORCES_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCouplingForces {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every body in `queries` and returns one
    /// [`WaterCouplingForcesResult`] per input, in order.
    ///
    /// Each continuous output matches the reference within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterCouplingForcesQuery],
    ) -> Vec<WaterCouplingForcesResult> {
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
            label: Some("prism_volumetric_water_coupling_forces_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_bind_group"),
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
            label: Some("prism_volumetric_water_coupling_forces_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_coupling_forces_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_coupling_forces_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per body, flattened to a 1-D dispatch.
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
