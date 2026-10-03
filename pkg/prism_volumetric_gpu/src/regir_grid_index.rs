//! `wgpu` compute twin of the stateless grid-index geometry inside the
//! world-space `ReGIR` light grid contract
//! ([`regir`](prism_render_architecture::lighting::regir)).
//!
//! The `CPU` golden
//! [`RegirConfig`](prism_render_architecture::lighting::regir::RegirConfig)
//! describes a coarse world-space grid and owns the pure index-geometry that
//! maps between world positions, integer cell coordinates, linear cell indices,
//! and cell centers. Those maps are stateless closed forms over a single cell
//! and are twinned here:
//!
//! - [`cell_coords`](prism_render_architecture::lighting::regir::RegirConfig::cell_coords)
//!   / [`cell_index_of`](prism_render_architecture::lighting::regir::RegirConfig::cell_index_of):
//!   a world `pos` to its linear cell index, or "outside the grid".
//! - [`coords_of`](prism_render_architecture::lighting::regir::RegirConfig::coords_of):
//!   a linear index back to integer cell coordinates.
//! - [`cell_center`](prism_render_architecture::lighting::regir::RegirConfig::cell_center):
//!   integer coordinates to the world-space cell center.
//!
//! [`GpuRegirGridIndex`] is the on-device twin of those three maps. One thread
//! solves one query, computing all three directions at once, reproducing the
//! reference's exact arithmetic, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same cell geometry the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For the position-to-index direction the kernel reproduces the reference's
//! out-of-grid rejection: a non-positive or `NaN` `cell_size` is "outside"; per
//! axis, a `local = (pos - grid_min) / cell_size` that is negative or `NaN` is
//! "outside"; the truncating `local as u32` (reproduced as `u32(local)`, which
//! for the already-`>= 0` `local` equals its floor) that reaches or exceeds the
//! axis extent is "outside"; otherwise the row-major
//! `(z * dy + y) * dx + x`. For the index-to-coordinate direction it reproduces
//! the integer decomposition `x = linear % dx`, `y = (linear / dx) % dy`,
//! `z = linear / (dx * dy)`. For the coordinate-to-center direction it
//! reproduces `grid_min + (coords + 0.5) * cell_size` per axis.
//!
//! # What stays on the host
//!
//! The grid's stateful presampling — the per-frame reservoir
//! [`rebuild`](prism_render_architecture::lighting::regir::RegirGrid::rebuild)
//! and the `RIS` candidate draw
//! [`candidate_at`](prism_render_architecture::lighting::regir::RegirGrid::candidate_at)
//! — is sequential Monte Carlo state with no fixed-width per-query device
//! analogue, so it stays on the host and is never dispatched. Only the
//! stateless index geometry is twinned. The host guarantees every axis extent
//! `dims[axis] >= 1`, so the device-side integer `%` and `/` by `dx`, `dy`, and
//! `dx * dy` never divide by zero.
//!
//! # Correctness model
//!
//! The discrete answers — the `in_grid` flag, the linear index, and the decoded
//! coordinates — are built from ordered comparisons and integer `%`/`/`, so for
//! fixtures chosen clear of a cell boundary (a `local` near an integer, where a
//! `GPU` divide and a `CPU` divide could truncate to adjacent cells) the `CPU`
//! and `GPU` agree exactly and the parity test asserts an exact `==` on each.
//! The continuous cell center threads through a multiply and an add, so `CPU`
//! and `GPU` are not bit-exact; it is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`/`max`, `+ - *
//! /`, integer `%`, the `u32(f32)` truncation of a non-negative value, and
//! unsigned index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `sqrt`, no `floor` and no `round`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::regir`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `ReGIR` grid-index kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`RegirConfig`](prism_render_architecture::lighting::regir::RegirConfig)
/// index-geometry closed forms; see the module documentation for the algorithm.
const REGIR_GRID_INDEX_WGSL: &str = r#"
// ReGIR grid-index twin: one thread computes all three stateless index maps for
// one query — a world position to its linear cell index (with the reference's
// out-of-grid rejection), a linear index back to integer coordinates, and
// integer coordinates to a world-space cell center — mirroring the CPU golden
// `lighting::regir::RegirConfig` with only + - * /, integer %, and the u32()
// truncation of a non-negative value. It owns no reservoir rebuild and no
// candidate draw; that stateful Monte Carlo work stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::regir；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // World-space corner of cell (0, 0, 0); w lane is unused padding.
    grid_min: vec4<f32>,
    // World position to map to a cell index; w lane is unused padding.
    pos: vec4<f32>,
    // Cells along each axis (each >= 1, host-guaranteed); w lane unused.
    dims: vec4<u32>,
    // Integer coordinates to map to a cell center; w lane unused.
    coords_in: vec4<u32>,
    // Edge length of a cubic cell in world units.
    cell_size: f32,
    // Linear cell index to decompose into coordinates.
    linear_in: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // World-space center of `coords_in`; w lane is unused padding.
    center: vec4<f32>,
    // Integer coordinates decoded from `linear_in`; w lane unused.
    coords_out: vec4<u32>,
    // Linear cell index of `pos` (valid only when in_grid is 1).
    linear_out: u32,
    // 1 when `pos` lies inside the grid, else 0.
    in_grid: u32,
    pad0: u32,
    pad1: u32,
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

    // --- (A) pos -> cell index -------------------------------------------
    // A non-positive or NaN cell_size is outside: `!(cs > 0.0)` captures both
    // `cs <= 0` and `cs` NaN (NaN > 0.0 is false), mirroring the golden
    // `cell_size <= 0.0 || cell_size.is_nan()` branch.
    var in_grid: u32 = 1u;
    var linear: u32 = 0u;
    let cs = q.cell_size;
    if (!(cs > 0.0)) {
        in_grid = 0u;
    } else {
        let gmin = q.grid_min.xyz;
        let p = q.pos.xyz;
        let lx = (p.x - gmin.x) / cs;
        let ly = (p.y - gmin.y) / cs;
        let lz = (p.z - gmin.z) / cs;
        // A negative or NaN local coordinate is outside: `!(l >= 0.0)` captures
        // both `l < 0` and `l` NaN, mirroring the golden `local < 0.0 ||
        // local.is_nan()` branch.
        if (!(lx >= 0.0) || !(ly >= 0.0) || !(lz >= 0.0)) {
            in_grid = 0u;
        } else {
            // `local as u32`: truncation toward zero, which for the already
            // `>= 0` local equals its floor. No `floor` builtin is used.
            let ix = u32(lx);
            let iy = u32(ly);
            let iz = u32(lz);
            if (ix >= q.dims.x || iy >= q.dims.y || iz >= q.dims.z) {
                in_grid = 0u;
            } else {
                // Row-major linear index (x fastest): (z*dy + y)*dx + x.
                linear = (iz * q.dims.y + iy) * q.dims.x + ix;
            }
        }
    }

    // --- (B) linear -> coords --------------------------------------------
    // Integer decomposition; dx, dy and dx*dy are non-zero because the host
    // guarantees every axis extent >= 1.
    let dx = q.dims.x;
    let dy = q.dims.y;
    let lin = q.linear_in;
    let cx = lin % dx;
    let cy = (lin / dx) % dy;
    let cz = lin / (dx * dy);

    // --- (C) coords -> center --------------------------------------------
    let gmin = q.grid_min.xyz;
    let cf = vec3<f32>(f32(q.coords_in.x), f32(q.coords_in.y), f32(q.coords_in.z));
    let center = gmin + (cf + vec3<f32>(0.5, 0.5, 0.5)) * cs;

    var out: Result;
    out.center = vec4<f32>(center, 0.0);
    out.coords_out = vec4<u32>(cx, cy, cz, 0u);
    out.linear_out = linear;
    out.in_grid = in_grid;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`REGIR_GRID_INDEX_WGSL`].
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

/// `repr(C)` `std430` layout of one grid-index query, matching the `WGSL`
/// `Query` struct: four `16`-byte lanes (grid origin, position, extent,
/// coordinates) followed by the `cell_size`, the linear index to decode, and
/// two pad words to an `80`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// World-space corner of cell `(0, 0, 0)`, padded to a `16`-byte lane.
    grid_min: [f32; 4],
    /// World position to map, padded to a `16`-byte lane.
    pos: [f32; 4],
    /// Cells along each axis, padded to a `16`-byte lane.
    dims: [u32; 4],
    /// Integer coordinates to map to a center, padded to a `16`-byte lane.
    coords_in: [u32; 4],
    /// Edge length of a cubic cell in world units.
    cell_size: f32,
    /// Linear cell index to decompose.
    linear_in: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one grid-index result, matching the `WGSL`
/// `Result` struct: a `16`-byte center lane, a `16`-byte coordinate lane, the
/// linear index, the `in_grid` flag, and two pad words to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// World-space center of the query's `coords_in`, padded to a `16`-byte
    /// lane.
    center: [f32; 4],
    /// Integer coordinates decoded from the query's `linear_in`, padded to a
    /// `16`-byte lane.
    coords_out: [u32; 4],
    /// Linear cell index of the query's `pos` (valid only when `in_grid` is
    /// `1`).
    linear_out: u32,
    /// `1` when `pos` lies inside the grid, `0` otherwise.
    in_grid: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One grid-index query, carrying the grid configuration plus the three
/// independent inputs the twin resolves at once: a world `pos` to map to a cell
/// index, a `linear` index to decode into coordinates, and `coords` to map to a
/// cell center.
///
/// The host owns the surrounding stateful grid — the reservoir rebuild and the
/// candidate draw — and enqueues one [`RegirGridIndexQuery`] per lookup,
/// mirroring the inputs the reference
/// [`RegirConfig`](prism_render_architecture::lighting::regir::RegirConfig)
/// index-geometry methods take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegirGridIndexQuery {
    /// World-space corner of cell `(0, 0, 0)` (the golden `grid_min`).
    pub grid_min: [f32; 3],
    /// Edge length of a cubic cell in world units (the golden `cell_size`).
    pub cell_size: f32,
    /// Number of cells along each axis (the golden `dims`; each `>= 1`).
    pub dims: [u32; 3],
    /// World position to map to a cell index (direction `pos -> index`).
    pub pos: [f32; 3],
    /// Linear cell index to decode into coordinates (direction
    /// `index -> coords`).
    pub linear: u32,
    /// Integer coordinates to map to a cell center (direction
    /// `coords -> center`).
    pub coords: [u32; 3],
}

impl RegirGridIndexQuery {
    /// Builds a query from the grid configuration and the three direction
    /// inputs.
    #[must_use]
    pub const fn new(
        grid_min: [f32; 3],
        cell_size: f32,
        dims: [u32; 3],
        pos: [f32; 3],
        linear: u32,
        coords: [u32; 3],
    ) -> RegirGridIndexQuery {
        RegirGridIndexQuery {
            grid_min,
            cell_size,
            dims,
            pos,
            linear,
            coords,
        }
    }
}

/// One resolved grid-index query, mirroring the three reference maps.
///
/// `linear` and `in_grid` are the position-to-index result (`in_grid` is
/// `false` when `pos` lies outside the grid, matching the reference `None`);
/// `coords` is the index-to-coordinate decode of the query's `linear`; `center`
/// is the coordinate-to-center map of the query's `coords`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegirGridIndexResult {
    /// Linear cell index of the query's `pos` (meaningful only when `in_grid`).
    pub linear: u32,
    /// Whether the query's `pos` lies inside the grid.
    pub in_grid: bool,
    /// Integer coordinates decoded from the query's `linear`.
    pub coords: [u32; 3],
    /// World-space center of the query's `coords`.
    pub center: [f32; 3],
}

/// Encodes one [`RegirGridIndexQuery`] into its `std430` [`GpuQuery`] slot,
/// padding each triple to a `16`-byte lane.
fn encode_query(q: &RegirGridIndexQuery) -> GpuQuery {
    GpuQuery {
        grid_min: [q.grid_min[0], q.grid_min[1], q.grid_min[2], 0.0],
        pos: [q.pos[0], q.pos[1], q.pos[2], 0.0],
        dims: [q.dims[0], q.dims[1], q.dims[2], 0],
        coords_in: [q.coords[0], q.coords[1], q.coords[2], 0],
        cell_size: q.cell_size,
        linear_in: q.linear,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RegirGridIndexResult`],
/// turning the `in_grid` word back into a [`bool`].
fn decode_result(raw: &GpuResult) -> RegirGridIndexResult {
    RegirGridIndexResult {
        linear: raw.linear_out,
        in_grid: raw.in_grid != 0,
        coords: [raw.coords_out[0], raw.coords_out[1], raw.coords_out[2]],
        center: [raw.center[0], raw.center[1], raw.center[2]],
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

/// A compiled, reusable `ReGIR` grid-index compute pipeline, twinning the
/// stateless index geometry of the `CPU` golden
/// [`regir`](prism_render_architecture::lighting::regir).
pub struct GpuRegirGridIndex {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRegirGridIndex {
    /// Compiles the `ReGIR` grid-index kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRegirGridIndex {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_regir_grid_index"),
            source: ShaderSource::Wgsl(REGIR_GRID_INDEX_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_regir_grid_index_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_regir_grid_index_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_regir_grid_index_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRegirGridIndex {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`RegirGridIndexResult`]
    /// per input, in order.
    ///
    /// The `in_grid` flag, linear index and decoded coordinates equal the
    /// reference exactly for queries clear of a cell boundary; the cell center
    /// matches to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RegirGridIndexQuery],
    ) -> Vec<RegirGridIndexResult> {
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
            label: Some("prism_volumetric_regir_grid_index_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_regir_grid_index_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_regir_grid_index_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_regir_grid_index_bind_group"),
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
            label: Some("prism_volumetric_regir_grid_index_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_regir_grid_index_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_regir_grid_index_pass"),
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
