//! `wgpu` compute twin of the uniform spatial-hash neighbour gather inside the
//! Position-Based Fluids solve ([`pbf`](prism_render_architecture::water::pbf)).
//!
//! The `CPU` golden [`pbf`](prism_render_architecture::water::pbf) lays each
//! particle into a uniform grid whose cell size is the smoothing radius, so a
//! particle's neighbourhood is exactly its own cell plus the 26 adjacent cells.
//! [`gather_neighbors`](prism_render_architecture::water::pbf::gather_neighbors)
//! scans that `3x3x3` block in a fixed `z`-outer, `y`-middle, `x`-inner order,
//! walks each cell's bucket in stored (ascending index) order, excludes the
//! query particle, and keeps every particle whose squared distance is within
//! the cell size squared. The result is deterministic cell-major, index-sorted
//! order, independent of any device thread schedule.
//!
//! [`GpuWaterPbfNeighbors`] is the on-device twin of exactly that scan. One
//! invocation gathers one particle's neighbours: it reproduces the reference's
//! [`cell_coord`](prism_render_architecture::water::pbf::PbfGrid::cell_coord)
//! truncation, the `saturating_sub(1)` lower bound and the `(c + 1).min(n - 1)`
//! upper bound of the neighbourhood, the row-major
//! [`flat_index`](prism_render_architecture::water::pbf::PbfGrid::flat_index),
//! and the same traversal and squared-distance test, pushing survivors in the
//! identical order. A passing real-device parity test is direct evidence the
//! ported kernel gathers the same neighbour list the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For one query particle the kernel reproduces, bit-for-bit on the integer
//! outputs:
//!
//! - the grid cell of the particle via the reference `cell_coord` (`local`
//!   offset, non-negativity guard, truncating divide by `cell_size`, in-range
//!   guard), returning an empty list when the particle is out of range or the
//!   grid is degenerate;
//! - the clamped `3x3x3` neighbourhood (`saturating_sub(1)` low, `.min(n - 1)`
//!   high) and the row-major `flat_index` `cz * nx * ny + cy * nx + cx`;
//! - the `z`-outer, `y`-middle, `x`-inner scan with each bucket in stored
//!   order, the `j == particle` self-exclusion and the
//!   `length_squared <= cell_size^2` keep test;
//! - the collected neighbour indices in the reference's exact order plus their
//!   count.
//!
//! # What stays on the host
//!
//! The variable-length bucket storage
//! ([`PbfBins`](prism_render_architecture::water::pbf::PbfBins)) and the binning
//! pass ([`bin_particles`](prism_render_architecture::water::pbf::bin_particles))
//! are container work the host owns; the device never sees a `Vec<Vec<u32>>`.
//! The host flattens the buckets into a compressed-sparse-row pair — a
//! `cell_offsets` prefix array of length `cell_count + 1` and a `cell_items`
//! array concatenating the buckets in row-major cell order, each bucket in
//! stored order — and down-feeds the positions as a flat triple array, so the
//! device sees fixed-stride storage. An empty query batch short-circuits with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! Every twinned output is an integer (a neighbour index or a count), built
//! from truncating casts, unsigned comparisons and index arithmetic, so for
//! fixtures chosen clear of a cell boundary (a position near a cell face) and
//! clear of the keep boundary (a pair distance near `cell_size`) the `CPU` and
//! `GPU` agree exactly and the parity test asserts an exact `==` on the whole
//! list, order and count included. The single `f32` comparison is the
//! `length_squared <= cell_size^2` keep test; fixtures and the randomized sweep
//! keep every candidate pair's squared distance well clear of `cell_size^2` so
//! `CPU` and `GPU` cannot straddle it.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `select`, truncating `u32(f32)` casts, `+ - * /` and unsigned index
//! arithmetic over bounded loops — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, no inverse trigonometry, no `smoothstep`, no `round`, no `ceil` and
//! no `sqrt`: the distance test compares squared magnitudes. No optional device
//! feature is required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! The loops are bounded by the `3x3x3` neighbourhood and the per-cell bucket
//! lengths, so the kernel provably terminates.
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

/// Number of invocations per workgroup. A single query particle maps to a
/// single invocation that walks the whole `3x3x3` neighbourhood in order, so
/// the twin dispatches one workgroup of one thread per query.
const WORKGROUP_SIZE: u32 = 1;

/// Upper bound on the number of neighbour indices the device record carries.
/// The reference returns an unbounded `Vec<u32>`; the device writes the first
/// `MAX_NEIGHBORS` survivors into a fixed slot array alongside the full count.
/// Fixtures keep the gathered count at or below this bound so the parity list
/// compare is exact.
pub const MAX_NEIGHBORS: usize = 256;

/// The portable core-`WGSL` neighbour-gather twin, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`gather_neighbors`](prism_render_architecture::water::pbf::gather_neighbors)
/// scan; see the module documentation for the algorithm.
const WATER_PBF_NEIGHBORS_WGSL: &str = r#"
// PBF neighbour-gather twin: one invocation gathers one particle's neighbours
// by scanning its cell and the 26 adjacent cells in z-outer, y-middle, x-inner
// order, each bucket in stored order, excluding the particle itself and keeping
// any neighbour whose squared distance is within cell_size^2. Mirrors the CPU
// golden `water::pbf::gather_neighbors` exactly with only min/max/select and
// + - * / over bounded loops; no sqrt, no transcendental. The variable-length
// bucket storage and the binning pass stay host-side (down-fed as CSR).
//
// Provenance: 孪生自本仓 prism_render_architecture::water::pbf；
// 无第三方引擎源码或衍生代码。

const MAX_NEIGHBORS: u32 = 256u;
// Degeneracy floor on the cell size, matching the golden EPS = 1e-6.
const EPS: f32 = 0.000001;

struct Params {
    // Minimum corner of the domain.
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    // Cell edge length (the smoothing radius); the squared keep radius.
    cell_size: f32,
    // Grid cell counts along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Index of the query particle into the positions array.
    particle: u32,
    // Number of positions down-fed (bounds the particle index).
    position_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Number of neighbours gathered (may exceed MAX_NEIGHBORS; only the first
    // MAX_NEIGHBORS are written to the slot array).
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Gathered neighbour indices in cell-major, index-sorted order.
    neighbors: array<u32, 256>,
}

@group(0) @binding(0) var<uniform> params: Params;
// Positions flattened to triples: position i is (p[3*i], p[3*i+1], p[3*i+2]).
@group(0) @binding(1) var<storage, read> positions: array<f32>;
// CSR prefix: bucket `flat` spans cell_offsets[flat] .. cell_offsets[flat+1].
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
// CSR items: concatenated buckets in row-major cell order, each in stored order.
@group(0) @binding(3) var<storage, read> cell_items: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: array<Result>;

// saturating_sub(c, 1): 0 when c == 0, else c - 1. The c - 1u in the false
// branch is only selected when c != 0, so no unsigned wrap is ever observed.
fn sat_sub1(c: u32) -> u32 {
    return select(c - 1u, 0u, c == 0u);
}

@compute @workgroup_size(1)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 1u) {
        return;
    }

    var out: Result;
    out.count = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    let particle = params.particle;
    // Out-of-range particle yields no neighbours (golden positions.get()? ).
    if (particle >= params.position_count) {
        results[0] = out;
        return;
    }
    // Degenerate cell size yields no cell coordinate -> no neighbours.
    if (params.cell_size <= EPS) {
        results[0] = out;
        return;
    }

    let px = positions[3u * particle + 0u];
    let py = positions[3u * particle + 1u];
    let pz = positions[3u * particle + 2u];
    let lx = px - params.origin_x;
    let ly = py - params.origin_y;
    let lz = pz - params.origin_z;
    // Non-negativity guard, matching the golden cell_coord.
    if (lx < 0.0 || ly < 0.0 || lz < 0.0) {
        results[0] = out;
        return;
    }
    // Truncating divide, matching the golden `(local / cell_size) as u32`.
    let cx = u32(lx / params.cell_size);
    let cy = u32(ly / params.cell_size);
    let cz = u32(lz / params.cell_size);
    if (cx >= params.nx || cy >= params.ny || cz >= params.nz) {
        results[0] = out;
        return;
    }

    let radius_sq = params.cell_size * params.cell_size;
    let nx = params.nx;
    let ny = params.ny;

    let z_lo = sat_sub1(cz);
    let z_hi = min(cz + 1u, sat_sub1(params.nz));
    let y_lo = sat_sub1(cy);
    let y_hi = min(cy + 1u, sat_sub1(params.ny));
    let x_lo = sat_sub1(cx);
    let x_hi = min(cx + 1u, sat_sub1(params.nx));

    var count = 0u;
    var z = z_lo;
    loop {
        if (z > z_hi) {
            break;
        }
        var y = y_lo;
        loop {
            if (y > y_hi) {
                break;
            }
            var x = x_lo;
            loop {
                if (x > x_hi) {
                    break;
                }
                let flat = z * nx * ny + y * nx + x;
                let start = cell_offsets[flat];
                let end = cell_offsets[flat + 1u];
                var k = start;
                loop {
                    if (k >= end) {
                        break;
                    }
                    let j = cell_items[k];
                    if (j != particle) {
                        let qx = positions[3u * j + 0u];
                        let qy = positions[3u * j + 1u];
                        let qz = positions[3u * j + 2u];
                        let dx = px - qx;
                        let dy = py - qy;
                        let dz = pz - qz;
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 <= radius_sq) {
                            if (count < MAX_NEIGHBORS) {
                                out.neighbors[count] = j;
                            }
                            count = count + 1u;
                        }
                    }
                    k = k + 1u;
                }
                x = x + 1u;
            }
            y = y + 1u;
        }
        z = z + 1u;
    }

    out.count = count;
    results[0] = out;
}
"#;

/// Uniform parameters for one query dispatch, matching the `WGSL` `Params`
/// struct: the grid origin and cell size, the grid cell counts, the query
/// particle index and the position count, plus three pad words to fill a
/// `std140`-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Domain origin `x`.
    origin_x: f32,
    /// Domain origin `y`.
    origin_y: f32,
    /// Domain origin `z`.
    origin_z: f32,
    /// Cell edge length (the smoothing radius).
    cell_size: f32,
    /// Grid cell count along `x`.
    nx: u32,
    /// Grid cell count along `y`.
    ny: u32,
    /// Grid cell count along `z`.
    nz: u32,
    /// Query particle index.
    particle: u32,
    /// Number of positions down-fed.
    position_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of the single query result, matching the `WGSL`
/// `Result` struct: the gathered count, three pad words to a `16`-byte offset,
/// then the fixed `MAX_NEIGHBORS` neighbour-index slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Number of neighbours gathered.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Gathered neighbour indices; only the first `count` are valid.
    neighbors: [u32; MAX_NEIGHBORS],
}

/// One neighbour-gather query for the `PBF` spatial-hash twin: the grid, the
/// flattened positions, the compressed-sparse-row buckets and the query
/// particle index.
///
/// [`gather_neighbors`](prism_render_architecture::water::pbf::gather_neighbors)
/// reads the grid
/// ([`origin`](Self::origin), [`cell_size`](Self::cell_size),
/// [`nx`](Self::nx), [`ny`](Self::ny), [`nz`](Self::nz)), the
/// [`positions`](Self::positions) and the buckets flattened here into
/// [`cell_offsets`](Self::cell_offsets) and [`cell_items`](Self::cell_items),
/// then gathers the neighbours of [`particle`](Self::particle).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterPbfNeighborsQuery {
    /// Minimum corner of the domain in world space.
    pub origin: [f32; 3],
    /// Cell edge length (the smoothing radius).
    pub cell_size: f32,
    /// Grid cell count along `x`.
    pub nx: u32,
    /// Grid cell count along `y`.
    pub ny: u32,
    /// Grid cell count along `z`.
    pub nz: u32,
    /// Particle positions, one triple per particle.
    pub positions: Vec<[f32; 3]>,
    /// Compressed-sparse-row prefix of length `cell_count + 1`; bucket `flat`
    /// spans `cell_offsets[flat]..cell_offsets[flat + 1]`.
    pub cell_offsets: Vec<u32>,
    /// Concatenated buckets in row-major cell order, each in stored order.
    pub cell_items: Vec<u32>,
    /// Index of the particle whose neighbours are gathered.
    pub particle: u32,
}

impl WaterPbfNeighborsQuery {
    /// Builds a query from the grid, positions, compressed-sparse-row buckets
    /// and the query particle index.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query mirrors the golden grid, positions, bins and particle inputs verbatim"
    )]
    pub fn new(
        origin: [f32; 3],
        cell_size: f32,
        nx: u32,
        ny: u32,
        nz: u32,
        positions: Vec<[f32; 3]>,
        cell_offsets: Vec<u32>,
        cell_items: Vec<u32>,
        particle: u32,
    ) -> WaterPbfNeighborsQuery {
        WaterPbfNeighborsQuery {
            origin,
            cell_size,
            nx,
            ny,
            nz,
            positions,
            cell_offsets,
            cell_items,
            particle,
        }
    }
}

/// One gathered neighbour list of the `PBF` spatial-hash twin, mirroring the
/// golden
/// [`gather_neighbors`](prism_render_architecture::water::pbf::gather_neighbors)
/// return value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaterPbfNeighborsResult {
    /// Gathered neighbour indices in cell-major, index-sorted order.
    pub neighbors: Vec<u32>,
}

/// Decodes one packed [`GpuResult`] into the public [`WaterPbfNeighborsResult`].
/// Only the first `count` neighbour slots (capped at [`MAX_NEIGHBORS`]) are
/// kept, matching the reference's variable-length list.
fn decode_result(raw: &GpuResult) -> WaterPbfNeighborsResult {
    let count = (raw.count as usize).min(MAX_NEIGHBORS);
    WaterPbfNeighborsResult {
        neighbors: raw.neighbors[..count].to_vec(),
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

/// A compiled, reusable `PBF` neighbour-gather compute pipeline, twinning the
/// spatial-hash scan of the `CPU` golden
/// [`pbf`](prism_render_architecture::water::pbf).
pub struct GpuWaterPbfNeighbors {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterPbfNeighbors {
    /// Compiles the neighbour-gather kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterPbfNeighbors {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors"),
            source: ShaderSource::Wgsl(WATER_PBF_NEIGHBORS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterPbfNeighbors {
            module,
            layout,
            pipeline,
        }
    }

    /// Gathers the neighbours of every query in `queries` and returns one
    /// [`WaterPbfNeighborsResult`] per input, in order.
    ///
    /// Each query is dispatched as a single invocation that walks its own
    /// grid, positions and buckets, so the neighbour list and its order equal
    /// the reference exactly for fixtures clear of a cell or keep boundary. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterPbfNeighborsQuery],
    ) -> Vec<WaterPbfNeighborsResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        queries.iter().map(|q| self.evaluate_one(ctx, q)).collect()
    }

    /// Dispatches one query and reads back its gathered neighbour list.
    fn evaluate_one(
        &self,
        ctx: &GpuContext,
        query: &WaterPbfNeighborsQuery,
    ) -> WaterPbfNeighborsResult {
        let device = ctx.device();

        let params = GpuParams {
            origin_x: query.origin[0],
            origin_y: query.origin[1],
            origin_z: query.origin[2],
            cell_size: query.cell_size,
            nx: query.nx,
            ny: query.ny,
            nz: query.nz,
            particle: query.particle,
            position_count: query.positions.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Flatten positions to a triple array; a storage buffer cannot be
        // zero-sized, so an empty list carries a single dummy triple that the
        // kernel never reads (the particle index is out of range).
        let mut pos_flat: Vec<f32> = Vec::with_capacity(query.positions.len() * 3 + 1);
        for p in &query.positions {
            pos_flat.push(p[0]);
            pos_flat.push(p[1]);
            pos_flat.push(p[2]);
        }
        if pos_flat.is_empty() {
            pos_flat.push(0.0);
        }
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_positions"),
            contents: bytemuck::cast_slice(&pos_flat),
            usage: BufferUsages::STORAGE,
        });

        // CSR prefix of length cell_count + 1; pad a degenerate empty prefix to
        // a well-formed two-entry (single empty cell) prefix.
        let mut offsets = query.cell_offsets.clone();
        if offsets.len() < 2 {
            offsets.clear();
            offsets.push(0);
            offsets.push(0);
        }
        let offsets_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_offsets"),
            contents: bytemuck::cast_slice(&offsets),
            usage: BufferUsages::STORAGE,
        });

        // CSR items; a bucket is never indexed past its [start, end) span, so a
        // dummy entry for the empty-grid case is harmless.
        let mut items = query.cell_items.clone();
        if items.is_empty() {
            items.push(0);
        }
        let items_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_items"),
            contents: bytemuck::cast_slice(&items),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = size_of::<GpuResult>() as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: offsets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: items_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_pbf_neighbors_encoder"),
        });
        {
            let groups = 1u32.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_pbf_neighbors_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One invocation gathers the single query particle's neighbours.
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view)[0];
        drop(view);
        stage.unmap();

        decode_result(&raw)
    }
}
