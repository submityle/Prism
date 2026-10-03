//! `wgpu` compute twin of the shallow-water-equations (`SWE`) wave-speed and
//! `CFL`-timestep selection
//! ([`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
//! [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
//! and [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep)).
//!
//! Explicit shallow-water surface stepping is only stable under the
//! Courant-Friedrichs-Lewy (`CFL`) condition, so each frame first estimates the
//! largest signal speed on the grid and then derives the largest stable
//! timestep from it. The per-cell signal speed is the flow speed plus the
//! gravity-wave celerity, `|(u, v)| + sqrt(g * max(h, 0))`; the grid maximum is
//! a plain reduction, and the timestep is `cfl * dx / max_speed` with a large
//! sentinel when the body is fully at rest. All three steps are pure
//! multiply/add/divide plus a single `sqrt`, so they port cleanly to the
//! device, and a passing real-device parity run is direct evidence the ported
//! kernel folds the same arithmetic the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! One thread serves one query. Given a cell count `n` and per-cell depth `h`,
//! x-velocity `u`, and z-velocity `v`, the thread computes
//! [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed)
//! for every cell (`flow = sqrt(u*u + v*v)`, a non-positive depth folded to
//! zero so the celerity square root stays real, then `flow + sqrt(g*h)`),
//! reduces them to the grid maximum
//! ([`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed)),
//! and applies
//! [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep):
//! `dt = f32::MAX` when `max_speed <= EPS`, otherwise `cfl * dx / max_speed`.
//! The per-cell speeds, the grid maximum, and the timestep are all returned and
//! all pinned by the parity suite.
//!
//! # What stays on the host
//!
//! The variable-length state layout, the grid-size bookkeeping (`SweConfig`),
//! the continuity/momentum stepping, the interaction injection, and the
//! `n = min(h.len, u.len, v.len)` length reconciliation are host
//! responsibilities. The host truncates and zero-pads each cell array to
//! [`MAX_CELLS`] before dispatch, so the device sees only fixed-capacity arrays
//! and a single scalar cell count.
//!
//! # Correctness model
//!
//! Each per-cell speed is a flow magnitude plus a gravity-celerity term, both
//! built from `+`, `-`, `*`, and one `sqrt`; the maximum is an ordered
//! reduction and the timestep a single guarded divide. The `CPU` and `GPU`
//! therefore agree on the continuous quantities to within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`. The at-rest sentinel `f32::MAX` is reproduced bit-for-bit
//! on the device (constructed by bit pattern) and compared with exact equality.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+`, `-`, `*`, `/`,
//! `sqrt`, ordered comparisons, and bounded loops — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no
//! `round`, and no `cbrt`. Each thread performs a bounded sequence of
//! arithmetic, so the kernel provably terminates. No optional device feature is
//! required, so it runs unmodified across backends.
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

/// Fixed upper bound on the number of cells a single query may carry. The host
/// truncates and zero-pads each per-cell array to this capacity before
/// dispatch.
pub const MAX_CELLS: usize = 256;

/// The inlined `WGSL` twin of
/// [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
/// [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
/// and [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep):
/// one thread per query, computing per-cell signal speeds, their maximum, and
/// the `CFL` timestep with only `+`, `-`, `*`, `/`, and `sqrt`.
const WATER_SWE_WAVESPEED_WGSL: &str = r#"
// Twin of water::swe::{cell_wave_speed, max_wave_speed, cfl_timestep}. One
// thread per query folds the per-cell signal speed |(u,v)| + sqrt(g*max(h,0))
// over the grid, reduces to the maximum, and derives the CFL timestep
// cfl*dx/max_speed (with an f32::MAX sentinel at rest). No transcendental other
// than sqrt; only multiply/add/divide and ordered comparisons.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::swe；无第三方引擎源码或衍生代码。

// Fixed capacity matching the Rust side (MAX_CELLS).
const MAX_CELLS: u32 = 256u;

// Rest threshold matching water::EPS; below it the body imposes no CFL limit.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Fixed-capacity per-cell depth, zero-padded beyond cell_count.
    h: array<f32, 256>,
    // Fixed-capacity per-cell x-velocity.
    u: array<f32, 256>,
    // Fixed-capacity per-cell z-velocity.
    v: array<f32, 256>,
    // Number of valid cells (<= MAX_CELLS).
    cell_count: u32,
    // Gravitational acceleration for the celerity term.
    gravity: f32,
    // Cell size in meters for the CFL timestep.
    dx: f32,
    // CFL number for the timestep.
    cfl: f32,
}

struct Result {
    // Fixed-capacity per-cell signal speed; cells beyond cell_count stay zero.
    speeds: array<f32, 256>,
    // Grid-maximum signal speed.
    max_speed: f32,
    // CFL-stable timestep, or f32::MAX at rest.
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Per-cell signal speed: flow magnitude plus gravity-wave celerity. A
// non-positive depth is folded to zero so the celerity square root stays real.
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

    let n = queries[idx].cell_count;
    let gravity = queries[idx].gravity;

    var max_speed: f32 = 0.0;
    var i: u32 = 0u;
    for (; i < n; i = i + 1u) {
        let s = cell_wave_speed(queries[idx].h[i], queries[idx].u[i], queries[idx].v[i], gravity);
        results[idx].speeds[i] = s;
        if (s > max_speed) {
            max_speed = s;
        }
    }
    // Zero the unused tail so stale storage never leaks into a readback.
    for (; i < MAX_CELLS; i = i + 1u) {
        results[idx].speeds[i] = 0.0;
    }

    results[idx].max_speed = max_speed;

    // cfl_timestep: a body at rest imposes no constraint, so return the
    // f32::MAX sentinel (bit pattern 0x7f7fffff) exactly.
    if (max_speed <= EPS) {
        results[idx].dt = bitcast<f32>(0x7f7fffffu);
    } else {
        results[idx].dt = queries[idx].cfl * queries[idx].dx / max_speed;
    }
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_SWE_WAVESPEED_WGSL`].
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

/// `repr(C)` `std430` layout of one wave-speed query, matching the `WGSL`
/// `Query` struct: three fixed-capacity per-cell arrays followed by the cell
/// count and the three scalar parameters (all alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Fixed-capacity per-cell depth, zero-padded beyond `cell_count`.
    h: [f32; MAX_CELLS],
    /// Fixed-capacity per-cell x-velocity.
    u: [f32; MAX_CELLS],
    /// Fixed-capacity per-cell z-velocity.
    v: [f32; MAX_CELLS],
    /// Number of valid cells.
    cell_count: u32,
    /// Gravitational acceleration for the celerity term.
    gravity: f32,
    /// Cell size in meters for the `CFL` timestep.
    dx: f32,
    /// `CFL` number for the timestep.
    cfl: f32,
}

/// `repr(C)` `std430` layout of one wave-speed result, matching the `WGSL`
/// `Result` struct: the fixed-capacity per-cell speeds followed by the grid
/// maximum and the timestep (all alignment `4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Fixed-capacity per-cell signal speed; cells beyond `cell_count` stay
    /// zero.
    speeds: [f32; MAX_CELLS],
    /// Grid-maximum signal speed.
    max_speed: f32,
    /// `CFL`-stable timestep, or `f32::MAX` at rest.
    dt: f32,
}

/// One wave-speed query to run on the device, mirroring the inputs of
/// [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed)
/// and [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep).
///
/// The host supplies the per-cell depth and velocity fields (reconciled to the
/// shortest common length and zero-padded to [`MAX_CELLS`] on encode), the
/// gravity used for celerity, and the cell size `dx` and `CFL` number used for
/// the timestep.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterSweWavespeedQuery {
    /// Per-cell water depth.
    pub h: Vec<f32>,
    /// Per-cell x-velocity.
    pub u: Vec<f32>,
    /// Per-cell z-velocity.
    pub v: Vec<f32>,
    /// Gravitational acceleration.
    pub gravity: f32,
    /// Cell size in meters.
    pub dx: f32,
    /// `CFL` number.
    pub cfl: f32,
}

/// One wave-speed result, mirroring the golden quantities. `speeds` has length
/// equal to the reconciled cell count `n = min(h.len, u.len, v.len)`.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterSweWavespeedResult {
    /// Per-cell signal speed, length `n`.
    pub speeds: Vec<f32>,
    /// Grid-maximum signal speed.
    pub max_speed: f32,
    /// `CFL`-stable timestep, or `f32::MAX` at rest.
    pub dt: f32,
}

/// Reconciled cell count for a query: the shortest common length of the three
/// per-cell fields, clamped to [`MAX_CELLS`], matching the golden
/// [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed)
/// bound `n = min(h.len, u.len, v.len)`.
fn cell_count(q: &WaterSweWavespeedQuery) -> usize {
    q.h.len().min(q.u.len()).min(q.v.len()).min(MAX_CELLS)
}

/// Encodes one [`WaterSweWavespeedQuery`] into its `std430` [`GpuQuery`] slot,
/// zero-filling each per-cell array and copying the input up to the reconciled
/// cell count.
fn encode_query(q: &WaterSweWavespeedQuery) -> GpuQuery {
    let n = cell_count(q);
    let mut h = [0.0f32; MAX_CELLS];
    let mut u = [0.0f32; MAX_CELLS];
    let mut v = [0.0f32; MAX_CELLS];
    h[..n].copy_from_slice(&q.h[..n]);
    u[..n].copy_from_slice(&q.u[..n]);
    v[..n].copy_from_slice(&q.v[..n]);
    GpuQuery {
        h,
        u,
        v,
        cell_count: n as u32,
        gravity: q.gravity,
        dx: q.dx,
        cfl: q.cfl,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterSweWavespeedResult`], keeping only the first `n` per-cell speeds.
fn decode_result(n: usize, raw: &GpuResult) -> WaterSweWavespeedResult {
    WaterSweWavespeedResult {
        speeds: raw.speeds[..n].to_vec(),
        max_speed: raw.max_speed,
        dt: raw.dt,
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

/// A compiled, reusable shallow-water wave-speed compute pipeline, twinning the
/// `CPU` golden
/// [`cell_wave_speed`](prism_render_architecture::water::swe::cell_wave_speed),
/// [`max_wave_speed`](prism_render_architecture::water::swe::max_wave_speed),
/// and [`cfl_timestep`](prism_render_architecture::water::swe::cfl_timestep).
pub struct GpuWaterSweWavespeed {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSweWavespeed {
    /// Compiles the wave-speed kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSweWavespeed {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed"),
            source: ShaderSource::Wgsl(WATER_SWE_WAVESPEED_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSweWavespeed {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries` and returns one [`WaterSweWavespeedResult`]
    /// per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterSweWavespeedQuery],
    ) -> Vec<WaterSweWavespeedResult> {
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
            label: Some("prism_volumetric_water_swe_wavespeed_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_bind_group"),
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
            label: Some("prism_volumetric_water_swe_wavespeed_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_swe_wavespeed_encoder"),
        });
        {
            // One thread per query.
            let threads = count as u32;
            let groups = threads.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_swe_wavespeed_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(q, r)| decode_result(cell_count(q), r))
            .collect()
    }
}
