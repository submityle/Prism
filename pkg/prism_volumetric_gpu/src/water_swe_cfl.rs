//! `wgpu` compute twin of the shallow-water `CFL` wave-speed primitives
//! ([`swe`](prism_render_architecture::water::swe)).
//!
//! Explicit shallow-water stepping is only stable under the
//! Courant-Friedrichs-Lewy (`CFL`) condition, which bounds the timestep by the
//! fastest signal speed on the grid. The `CPU` golden module derives that speed
//! from the depth and depth-averaged velocity of each cell. This twin ports its
//! three stateless numeric kernels so a surface pass can evaluate the stability
//! query on-device:
//!
//! * [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed)
//!   — the local signal speed `|(u, v)| + sqrt(g * h)`, with negative depth
//!   treated as dry (zero) so the square root stays real.
//! * [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed)
//!   — the largest `cell_wave_speed` over the active cells of one query, the
//!   speed the `CFL` condition must resolve. The host-side golden walks a
//!   variable-length grid; this twin fixes an upper bound of
//!   [`MAX_CELLS`] cells per query plus an explicit `count`.
//! * [`is_cfl_stable`](prism_render_architecture::water::swe::is_cfl_stable)
//!   — the predicate `dt * max_speed <= cfl_number * dx + EPS`, evaluated on the
//!   `max_speed` the kernel just computed.
//!
//! [`GpuWaterSweCfl`] is the on-device twin: one thread reduces one query's
//! cells to the maximum wave speed, records the first cell's speed, and decides
//! the `CFL` predicate, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same stability answer the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each [`WaterSweCflQuery`] carries the per-body physical constants
//! (`gravity`, `dt`, `dx`, `cfl_number`), the active cell `count`, and up to
//! [`MAX_CELLS`] cells. The kernel returns one [`WaterSweCflResult`] holding the
//! reduced `max_speed`, the first cell's speed, and the `CFL` stability flag.
//!
//! # What stays on the host
//!
//! The variable-length grid stepping
//! ([`step`](prism_render_architecture::water::swe::step)), the conserved
//! [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume),
//! the sentinel [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep),
//! and the interaction-injection mutators stay on the host. The empty-batch
//! short-circuit also stays on the host, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! The two speeds are continuous and thread through a `sqrt` and a max
//! reduction, so for fixtures clear of the stability boundary the `CPU` and
//! `GPU` agree to a tight tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`).
//! The stability flag is a discrete decision, asserted exactly; fixtures keep
//! `dt * max_speed` well clear of `cfl_number * dx` so a last-place `sqrt`
//! difference cannot flip the verdict.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `sqrt`,
//! a bounded loop and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! inverse trigonometry, no `round`, and no bare f32 equality. No optional
//! device feature is required, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
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

/// Fixed upper bound on the number of cells one [`WaterSweCflQuery`] can carry.
/// The host-side golden
/// [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed)
/// walks a variable-length grid; the twin caps each query at this many cells so
/// the storage layout stays a fixed stride.
pub const MAX_CELLS: usize = 32;

/// The portable core-`WGSL` shallow-water `CFL` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` reduces
/// one query's cells to the maximum wave speed and decides the `CFL` predicate,
/// twinning the `CPU` golden
/// [`swe`](prism_render_architecture::water::swe) closed forms; see the module
/// documentation for the algorithm.
const WATER_SWE_CFL_WGSL: &str = r#"
// wgpu compute twin of prism_render_architecture::water::swe: the stateless
// CFL wave-speed kernels (cell_wave_speed, max_wave_speed, is_cfl_stable), with
// only min, max, sqrt, a bounded loop and + - * /. The variable-length grid
// stepping and the interaction mutators stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::swe；无第三方引擎
// 源码或衍生代码。

// Shared non-negative epsilon, matching water::EPS.
const EPS: f32 = 1e-6;
// Fixed upper bound on cells per query, matching MAX_CELLS on the host.
const MAX_CELLS: u32 = 32u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Gravitational acceleration for the celerity term.
    gravity: f32,
    // Candidate explicit timestep under test.
    dt: f32,
    // Cell size in meters.
    dx: f32,
    // CFL number (Courant limit).
    cfl_number: f32,
    // Active cell count, clamped to MAX_CELLS.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Flattened cells: three consecutive f32 per cell (h, u, v), MAX_CELLS of
    // them. A flat f32 array keeps the std430 stride unambiguous at 4 bytes.
    cells: array<f32, 96>,
}

struct Result {
    // Largest wave speed over the active cells.
    max_speed: f32,
    // Wave speed of cell 0 (zero when the query has no cells).
    first_cell_speed: f32,
    // CFL stability flag, 1 when stable else 0.
    cfl_stable: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Local signal speed |(u, v)| + sqrt(g * h), negative depth treated as dry.
fn cell_wave_speed(depth: f32, u: f32, v: f32, gravity: f32) -> f32 {
    let flow = sqrt(u * u + v * v);
    var clamped_depth: f32 = 0.0;
    if (depth > 0.0) {
        clamped_depth = depth;
    }
    return flow + sqrt(gravity * clamped_depth);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let n = min(q.count, MAX_CELLS);
    var max_speed: f32 = 0.0;
    var first_cell_speed: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= n) {
            break;
        }
        let base = i * 3u;
        let speed = cell_wave_speed(q.cells[base], q.cells[base + 1u], q.cells[base + 2u], q.gravity);
        if (i == 0u) {
            first_cell_speed = speed;
        }
        if (speed > max_speed) {
            max_speed = speed;
        }
        i = i + 1u;
    }

    var cfl_stable: u32 = 0u;
    if (q.dt * max_speed <= q.cfl_number * q.dx + EPS) {
        cfl_stable = 1u;
    }

    var out: Result;
    out.max_speed = max_speed;
    out.first_cell_speed = first_cell_speed;
    out.cfl_stable = cfl_stable;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words,
/// filling a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SWE_CFL_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// four physical `f32` constants, the active cell `count` with three pad words,
/// then the flattened `96`-entry cell array (`h`, `u`, `v` per cell).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Gravitational acceleration `gravity`.
    gravity: f32,
    /// Candidate timestep `dt`.
    dt: f32,
    /// Cell size `dx`.
    dx: f32,
    /// Courant limit `cfl_number`.
    cfl_number: f32,
    /// Active cell `count`.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Flattened cells, three `f32` per cell: `[h0, u0, v0, h1, u1, v1, ...]`.
    cells: [f32; 96],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two scalar speeds, the stability flag, and one pad word to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Largest wave speed over the active cells.
    max_speed: f32,
    /// Wave speed of cell `0`.
    first_cell_speed: f32,
    /// `CFL` stability flag, `1` when stable.
    cfl_stable: u32,
    /// Padding word.
    pad0: u32,
}

/// One shallow-water cell: depth and the two depth-averaged velocity
/// components, the per-cell drivers of
/// [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterSweCflCell {
    /// Water depth `h` (meters); negative is treated as dry.
    pub h: f32,
    /// x-velocity `u` (meters per second).
    pub u: f32,
    /// z-velocity `v` (meters per second).
    pub v: f32,
}

/// One shallow-water `CFL` query: the physical constants, the active cell
/// `count`, and up to [`MAX_CELLS`] cells.
///
/// Mirrors the inputs the matching `CPU` golden
/// [`swe`](prism_render_architecture::water::swe) functions read: `gravity`
/// feeds [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
/// and `dt`, `dx`, `cfl_number` feed
/// [`is_cfl_stable`](prism_render_architecture::water::swe::is_cfl_stable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweCflQuery {
    /// Gravitational acceleration `gravity`.
    pub gravity: f32,
    /// Candidate explicit timestep `dt`.
    pub dt: f32,
    /// Cell size `dx` in meters.
    pub dx: f32,
    /// Courant limit `cfl_number`.
    pub cfl_number: f32,
    /// Number of active cells (clamped to [`MAX_CELLS`]).
    pub count: u32,
    /// The cells; only the first `count` (capped at [`MAX_CELLS`]) are read.
    pub cells: [WaterSweCflCell; MAX_CELLS],
}

/// One shallow-water `CFL` result, mirroring the values the matching `CPU`
/// golden returns.
///
/// Holds the reduced
/// [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
/// the first cell's
/// [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
/// and the
/// [`is_cfl_stable`](prism_render_architecture::water::swe::is_cfl_stable)
/// verdict encoded as `0` or `1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSweCflResult {
    /// Largest wave speed over the active cells.
    pub max_speed: f32,
    /// Wave speed of cell `0` (zero when the query has no cells).
    pub first_cell_speed: f32,
    /// `CFL` stability flag, `1` when stable else `0`.
    pub cfl_stable: u32,
}

/// Encodes one [`WaterSweCflQuery`] into its `std430` [`GpuQuery`] slot,
/// clamping the active cell count to [`MAX_CELLS`] and flattening the cells.
fn encode_query(q: &WaterSweCflQuery) -> GpuQuery {
    let mut cells = [0.0_f32; 96];
    let n = (q.count as usize).min(MAX_CELLS);
    let mut i = 0;
    while i < n {
        let base = i * 3;
        cells[base] = q.cells[i].h;
        cells[base + 1] = q.cells[i].u;
        cells[base + 2] = q.cells[i].v;
        i += 1;
    }
    GpuQuery {
        gravity: q.gravity,
        dt: q.dt,
        dx: q.dx,
        cfl_number: q.cfl_number,
        count: n as u32,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        cells,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterSweCflResult`].
fn decode_result(raw: &GpuResult) -> WaterSweCflResult {
    WaterSweCflResult {
        max_speed: raw.max_speed,
        first_cell_speed: raw.first_cell_speed,
        cfl_stable: raw.cfl_stable,
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

/// A compiled, reusable shallow-water `CFL` compute pipeline, twinning the three
/// stateless wave-speed kernels of the `CPU` golden
/// [`swe`](prism_render_architecture::water::swe).
pub struct GpuWaterSweCfl {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSweCfl {
    /// Compiles the shallow-water `CFL` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSweCfl {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_swe_cfl"),
            source: ShaderSource::Wgsl(WATER_SWE_CFL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSweCfl {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`WaterSweCflResult`]
    /// per input, in order.
    ///
    /// Each result equals the matching `CPU` golden
    /// [`swe`](prism_render_architecture::water::swe) outcome within the
    /// tolerance documented on this module. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterSweCflQuery],
    ) -> Vec<WaterSweCflResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_bind_group"),
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
            label: Some("prism_volumetric_water_swe_cfl_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_swe_cfl_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_swe_cfl_pass"),
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
