//! `wgpu` compute twin of the stateless shallow-water interaction-injection
//! primitives
//! [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
//! [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity),
//! together with their shared cell addressing
//! [`SweConfig::index`](prism_render_architecture::water::swe::SweConfig::index).
//!
//! Shallow-water bodies (rivers, ponds, flooded terrain) are a height field `h`
//! over a regular grid with a depth-averaged velocity field `(u, v)`.
//! Interaction sources — a character wading, rain, a boat hull — inject a
//! localized depth or velocity perturbation at one cell between steps. Injection
//! is pure address arithmetic plus a single floating-point add, clamped to
//! in-range cells so a stale source index never writes out of bounds, so the
//! port is faithful with no floating-point transcendental.
//!
//! [`GpuWaterSweInject`] is the on-device twin of those primitives. One thread
//! resolves one injection query, reproducing the row-major index, its two range
//! guards and the additive perturbation, so a passing real-device parity test is
//! direct evidence the ported kernel computes the same injection the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the injection outcome for each query:
//!
//! * `in_range`: the combined range predicate — `x < nx` and `z < nz` (the
//!   [`SweConfig::index`](prism_render_architecture::water::swe::SweConfig::index)
//!   bounds) and the resolved flat index `z * nx + x` below `cell_count` (the
//!   `state.h.len()` / `state.u.len()` / `state.v.len()` guard shared by
//!   [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
//!   [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity)),
//!   encoded as a `u32` so the `bool` survives readback.
//! * `idx`: the row-major flat index `z * nx + x` when `in_range`, else `0`.
//! * `new_depth` / `new_u` / `new_v`: the perturbed cell values `old + delta`
//!   when `in_range`, else the untouched `old` values — mirroring that an
//!   out-of-range injection leaves the state unchanged.
//!
//! # What stays on the host
//!
//! The surrounding `SWE` solver — the `CFL` timestep selection, the
//! conservative continuity flux, the momentum update and the full
//! grid-walking state — stays on the host. This twin models a single cell's
//! injection, with `cell_count` the common length of the host's `h`/`u`/`v`
//! vectors; the host owns the variable-length state those vectors live in.
//!
//! # Correctness model
//!
//! The host and the device evaluate the identical integer range predicate and
//! the identical additive perturbation, so the three discrete outputs
//! (`in_range`, `idx`) match exactly and the perturbed values — a single `f32`
//! add — match to within floating-point tolerance (and, being the same add,
//! exactly in practice). The one branch decision is a pure integer comparison,
//! so there is no floating-point crossing to flip.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! comparison and multiply-add and a single `f32` add — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, no inverse trigonometry, no `sqrt` and no `u64`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` shallow-water injection kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` goldens
/// [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
/// [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity)
/// over the shared
/// [`SweConfig::index`](prism_render_architecture::water::swe::SweConfig::index);
/// see the module documentation for the algorithm.
const WATER_SWE_INJECT_WGSL: &str = r#"
// Shallow-water interaction-injection twin: one thread resolves one query into
// the row-major flat index z*nx+x, its range predicate (x<nx && z<nz &&
// idx<cell_count), and the additive depth/velocity perturbation (old+delta on a
// hit, old otherwise), mirroring the CPU goldens
// `water::swe::{inject_depth, inject_velocity}` over `SweConfig::index` with
// only unsigned comparison, integer multiply-add and a single f32 add. It owns
// no CFL step, flux or grid state; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::swe；无第三方引擎源码
// 或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Grid extent along x and z (the SweConfig::index bounds).
    nx: u32,
    nz: u32,
    // Target cell coordinates.
    x: u32,
    z: u32,
    // Common length of the host h/u/v vectors; a resolved idx must stay below it.
    cell_count: u32,
    // Current cell values before injection.
    old_depth: f32,
    old_u: f32,
    old_v: f32,
    // Perturbations added on a hit.
    delta_depth: f32,
    delta_u: f32,
    delta_v: f32,
}

struct Result {
    // 1 when the cell is in range and modified, else 0.
    in_range: u32,
    // Row-major flat index z*nx+x on a hit, else 0.
    idx: u32,
    // Perturbed cell values on a hit, else the untouched old values.
    new_depth: f32,
    new_u: f32,
    new_v: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.count) {
        return;
    }
    let q = queries[qi];

    var out: Result;
    // Reproduce SweConfig::index: in bounds only when both coordinates fit.
    let in_bounds = (q.x < q.nx) && (q.z < q.nz);
    let flat = q.z * q.nx + q.x;
    // The shared inject guard also requires the flat index below cell_count.
    let hit = in_bounds && (flat < q.cell_count);

    if (hit) {
        out.in_range = 1u;
        out.idx = flat;
        out.new_depth = q.old_depth + q.delta_depth;
        out.new_u = q.old_u + q.delta_u;
        out.new_v = q.old_v + q.delta_v;
    } else {
        out.in_range = 0u;
        out.idx = 0u;
        out.new_depth = q.old_depth;
        out.new_u = q.old_u;
        out.new_v = q.old_v;
    }

    results[qi] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SWE_INJECT_WGSL`].
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
/// the grid extents, the cell coordinates, the common cell count and the
/// current/perturbation values.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Grid extent along x (the golden `cfg.nx`).
    nx: u32,
    /// Grid extent along z (the golden `cfg.nz`).
    nz: u32,
    /// Target cell x coordinate.
    x: u32,
    /// Target cell z coordinate.
    z: u32,
    /// Common length of the host `h`/`u`/`v` vectors.
    cell_count: u32,
    /// Current depth at the cell (the golden `state.h[idx]`).
    old_depth: f32,
    /// Current x-velocity at the cell (the golden `state.u[idx]`).
    old_u: f32,
    /// Current z-velocity at the cell (the golden `state.v[idx]`).
    old_v: f32,
    /// Depth perturbation (the golden `delta_depth`).
    delta_depth: f32,
    /// x-velocity perturbation (the golden `delta_u`).
    delta_u: f32,
    /// z-velocity perturbation (the golden `delta_v`).
    delta_v: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the range flag, the resolved index and the three perturbed values.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `1` when the cell was in range and modified, else `0`.
    in_range: u32,
    /// Row-major flat index on a hit, else `0`.
    idx: u32,
    /// Depth after injection (the golden `state.h[idx]`).
    new_depth: f32,
    /// x-velocity after injection (the golden `state.u[idx]`).
    new_u: f32,
    /// z-velocity after injection (the golden `state.v[idx]`).
    new_v: f32,
}

/// One shallow-water injection query: the grid extents, the target cell, the
/// common cell count and the current and perturbation values, mirroring the
/// arguments the goldens
/// [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
/// [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity)
/// read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweInjectQuery {
    /// Grid extent along x (the golden `cfg.nx`).
    pub nx: u32,
    /// Grid extent along z (the golden `cfg.nz`).
    pub nz: u32,
    /// Target cell x coordinate.
    pub x: u32,
    /// Target cell z coordinate.
    pub z: u32,
    /// Common length of the host `h`/`u`/`v` vectors; a resolved index must
    /// stay below it to hit.
    pub cell_count: u32,
    /// Current depth at the cell (the golden `state.h[idx]`).
    pub old_depth: f32,
    /// Current x-velocity at the cell (the golden `state.u[idx]`).
    pub old_u: f32,
    /// Current z-velocity at the cell (the golden `state.v[idx]`).
    pub old_v: f32,
    /// Depth perturbation (the golden `delta_depth`).
    pub delta_depth: f32,
    /// x-velocity perturbation (the golden `delta_u`).
    pub delta_u: f32,
    /// z-velocity perturbation (the golden `delta_v`).
    pub delta_v: f32,
}

impl WaterSweInjectQuery {
    /// Builds a query from the grid extents, the target cell, the common cell
    /// count and the current and perturbation values.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query flattens the goldens' grid, cell and perturbation inputs into one std430 record"
    )]
    pub const fn new(
        nx: u32,
        nz: u32,
        x: u32,
        z: u32,
        cell_count: u32,
        old_depth: f32,
        old_u: f32,
        old_v: f32,
        delta_depth: f32,
        delta_u: f32,
        delta_v: f32,
    ) -> WaterSweInjectQuery {
        WaterSweInjectQuery {
            nx,
            nz,
            x,
            z,
            cell_count,
            old_depth,
            old_u,
            old_v,
            delta_depth,
            delta_u,
            delta_v,
        }
    }
}

/// One resolved shallow-water injection response, mirroring the golden
/// injection outcome with the `bool` range flag encoded as a `u32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweInjectResult {
    /// `1` when the cell was in range and modified (the goldens returned
    /// `true`), else `0`.
    pub in_range: u32,
    /// Row-major flat index `z * nx + x` on a hit, else `0`.
    pub idx: u32,
    /// Depth after injection (the golden `state.h[idx]`), or the untouched
    /// depth when out of range.
    pub new_depth: f32,
    /// x-velocity after injection (the golden `state.u[idx]`), or the untouched
    /// value when out of range.
    pub new_u: f32,
    /// z-velocity after injection (the golden `state.v[idx]`), or the untouched
    /// value when out of range.
    pub new_v: f32,
}

/// Encodes one [`WaterSweInjectQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterSweInjectQuery) -> GpuQuery {
    GpuQuery {
        nx: q.nx,
        nz: q.nz,
        x: q.x,
        z: q.z,
        cell_count: q.cell_count,
        old_depth: q.old_depth,
        old_u: q.old_u,
        old_v: q.old_v,
        delta_depth: q.delta_depth,
        delta_u: q.delta_u,
        delta_v: q.delta_v,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterSweInjectResult`].
fn decode_result(raw: &GpuResult) -> WaterSweInjectResult {
    WaterSweInjectResult {
        in_range: raw.in_range,
        idx: raw.idx,
        new_depth: raw.new_depth,
        new_u: raw.new_u,
        new_v: raw.new_v,
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

/// A compiled, reusable shallow-water injection compute pipeline, twinning the
/// stateless primitives of the `CPU` goldens
/// [`inject_depth`](prism_render_architecture::water::swe::inject_depth) and
/// [`inject_velocity`](prism_render_architecture::water::swe::inject_velocity).
pub struct GpuWaterSweInject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSweInject {
    /// Compiles the shallow-water injection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSweInject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_swe_inject"),
            source: ShaderSource::Wgsl(WATER_SWE_INJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_inject_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_inject_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_swe_inject_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSweInject {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every injection query in `queries` and returns one
    /// [`WaterSweInjectResult`] per input, in order.
    ///
    /// The range flag and the resolved index match the reference exactly and
    /// the perturbed values match to within floating-point tolerance. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterSweInjectQuery],
    ) -> Vec<WaterSweInjectResult> {
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
            label: Some("prism_volumetric_water_swe_inject_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_swe_inject_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_swe_inject_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_swe_inject_bind_group"),
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
            label: Some("prism_volumetric_water_swe_inject_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_swe_inject_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_swe_inject_pass"),
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
