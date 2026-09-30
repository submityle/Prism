//! `wgpu` compute twin of Prism's uniform-grid self-collision neighbor gather
//! ([`self_collision_grid`](prism_render_architecture::hair::self_collision_grid)).
//!
//! The `CPU` golden reifies the strand self-collision spatial hash as a
//! reusable [`UniformGrid`](prism_render_architecture::hair::self_collision_grid::UniformGrid)
//! and flattens it into a `GPU`-uploadable compressed-sparse-row form
//! ([`GridCsr`](prism_render_architecture::hair::self_collision_grid::GridCsr)):
//! a lexicographically sorted `cell_keys` table, a `cell_starts` prefix-sum
//! offset array, and a flat `indices` pool concatenating every cell's ascending
//! particle indices. Its
//! [`neighbors`](prism_render_architecture::hair::self_collision_grid::UniformGrid::neighbors)
//! query is the spatial lookup any `hair_self_collision.wesl` pass needs: the
//! ascending union of every particle index in the 27 cells around a query
//! cell. This crate is the on-device twin: one thread per particle
//! binary-searches the sorted key table for each of its 27 neighbor cells,
//! copies the hit buckets into its own disjoint output slice, and
//! insertion-sorts that slice ascending, so a passing real-device parity test
//! is direct evidence the ported query gathers the same neighbor set, in the
//! same order, as the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuSelfCollisionGrid::eval`] takes the particle array and `cell_size` and
//! returns one ascending neighbor-index list per particle, matching
//! [`neighbors`](prism_render_architecture::hair::self_collision_grid::UniformGrid::neighbors)
//! of the golden grid queried at that particle's cell. The host builds the grid
//! and its `CSR` with the golden, hashes each particle's query cell with the
//! golden [`grid_cell_of`](prism_render_architecture::hair::self_collision_grid::grid_cell_of)
//! (so no float cast happens on device), pre-counts each particle's neighbor
//! population to lay out an exclusive-prefix output range, and each thread fills
//! its slice.
//!
//! # Portability
//!
//! The kernel only compares integer cell keys and copies `u32` indices — no
//! transcendental, no optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12. This is the first twin to introduce a device-side binary
//! search plus insertion sort; both stay in the portable core-`WGSL` subset.
//!
//! # Correctness model
//!
//! The kernel performs no floating-point arithmetic on the payload: it routes
//! integer indices, so the parity test asserts bit-identical neighbor lists,
//! not a tolerance. Each thread owns a disjoint output slice, and every
//! particle sits in exactly one cell, so the 27-cell union carries no
//! duplicates and the per-particle insertion sort reproduces the reference's
//! `sort_unstable` order exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard uniform-grid spatial hash neighbor gather plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::dynamics::StrandParticle;
use prism_render_architecture::hair::self_collision_grid::{build_csr, grid_cell_of, UniformGrid};

use crate::context::GpuContext;

/// Gather parameters uploaded to the kernel. `16`-byte scalar-packed `repr(C)`
/// matching `HairSelfCollisionGridParams` in `shaders/self_collision_grid.wesl`
/// (already a multiple of 16 for the uniform block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    cell_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One compacted per-particle output slice descriptor uploaded to the kernel.
/// `8`-byte `repr(C)` matching `HairGridRange` in
/// `shaders/self_collision_grid.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRange {
    start: u32,
    count: u32,
}

/// A compiled, reusable uniform-grid self-collision neighbor-gather pipeline.
pub struct GpuSelfCollisionGrid {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSelfCollisionGrid {
    /// Compiles the neighbor-gather kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSelfCollisionGrid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_self_collision_grid"),
            source: ShaderSource::Wgsl(include_str!("../shaders/self_collision_grid.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_self_collision_grid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_self_collision_grid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_self_collision_grid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSelfCollisionGrid {
            module,
            layout,
            pipeline,
        }
    }

    /// Gathers each particle's 27-cell neighborhood on device, returning one
    /// ascending neighbor-index list per particle.
    ///
    /// List `p` equals
    /// [`neighbors`](prism_render_architecture::hair::self_collision_grid::UniformGrid::neighbors)
    /// of the golden grid (built from `particles`/`cell_size`) queried at
    /// particle `p`'s cell — same members, same ascending order, bit-for-bit
    /// (the kernel only copies `u32` indices). A degenerate `cell_size` yields
    /// an empty grid, so every list is empty. An empty particle batch returns no
    /// lists without a dispatch (storage buffers cannot be zero-sized).
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[StrandParticle],
        cell_size: f32,
    ) -> Vec<Vec<u32>> {
        let particle_count = particles.len();
        if particle_count == 0 {
            return Vec::new();
        }

        // Host prep with the golden: build the grid + CSR, hash each particle's
        // query cell, and pre-count its neighbor population for the layout.
        let grid = UniformGrid::build(particles, cell_size);
        let csr = build_csr(&grid);

        let mut query_cells: Vec<i32> = Vec::with_capacity(particle_count * 3);
        let mut out_ranges: Vec<GpuRange> = Vec::with_capacity(particle_count);
        let mut scratch: Vec<u32> = Vec::new();
        let mut running = 0u32;
        for particle in particles {
            let cell = grid_cell_of(particle.position, cell_size);
            query_cells.push(cell.0);
            query_cells.push(cell.1);
            query_cells.push(cell.2);
            grid.neighbors(cell, &mut scratch);
            let count = scratch.len() as u32;
            out_ranges.push(GpuRange {
                start: running,
                count,
            });
            running += count;
        }
        let total = running as usize;

        // Flatten cell keys into `array<i32>` (three per key) to avoid a
        // `vec3<i32>` alignment mismatch across the host/device boundary.
        let mut cell_keys: Vec<i32> = Vec::with_capacity(csr.cell_keys.len() * 3);
        for key in &csr.cell_keys {
            cell_keys.push(key[0]);
            cell_keys.push(key[1]);
            cell_keys.push(key[2]);
        }
        let mut cell_starts = csr.cell_starts.clone();
        let mut indices = csr.indices.clone();

        // Storage buffers cannot be zero-sized. `cell_starts` always carries at
        // least the `[0]` sentinel; pad the key and index pools when the grid is
        // empty, and the output pool when no particle has neighbors. `cell_count`
        // and the `count == 0` ranges keep the kernel off every dummy.
        let cell_count = csr.cell_count();
        if cell_keys.is_empty() {
            cell_keys.push(0);
        }
        if cell_starts.is_empty() {
            cell_starts.push(0);
        }
        if indices.is_empty() {
            indices.push(0);
        }
        let mut out_len = total;
        if out_len == 0 {
            out_len = 1;
        }

        let uniform = Params {
            particle_count: particle_count as u32,
            cell_count: cell_count as u32,
            pad0: 0,
            pad1: 0,
        };

        let device = ctx.device();
        let out_bytes = (out_len * size_of::<u32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_grid_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let cell_keys_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_grid_cell_keys"),
            contents: bytemuck::cast_slice(&cell_keys),
            usage: BufferUsages::STORAGE,
        });
        let cell_starts_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_grid_cell_starts"),
            contents: bytemuck::cast_slice(&cell_starts),
            usage: BufferUsages::STORAGE,
        });
        let indices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_grid_indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::STORAGE,
        });
        let query_cells_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_grid_query_cells"),
            contents: bytemuck::cast_slice(&query_cells),
            usage: BufferUsages::STORAGE,
        });
        let ranges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_grid_ranges"),
            contents: bytemuck::cast_slice(&out_ranges),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_grid_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_grid_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_self_collision_grid_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: cell_keys_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: cell_starts_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: indices_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: query_cells_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: ranges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_self_collision_grid_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_self_collision_grid_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (particle_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out_flat = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        // Slice the flat pool back into one neighbor list per particle (empty
        // lists stay empty; the dummy pad, if any, is never sliced out).
        let mut neighbor_lists: Vec<Vec<u32>> = Vec::with_capacity(particle_count);
        for range in &out_ranges {
            let start = range.start as usize;
            let count = range.count as usize;
            neighbor_lists.push(out_flat[start..start + count].to_vec());
        }
        neighbor_lists
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
