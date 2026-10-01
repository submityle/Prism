//! `wgpu` compute twin of Prism's hair self-collision voxel velocity sampler
//! ([`DensityField::sample_avg_velocity`](prism_render_architecture::hair::self_collision_voxel::DensityField::sample_avg_velocity)).
//!
//! For each world-space query position the kernel returns the local
//! mass-weighted average velocity of the splatted self-collision field. The
//! field stores a per-cell accumulated splat mass `cells[i]` and a per-cell
//! mass-weighted velocity *sum* `velocity[i]` (not yet divided by density). With
//! the continuous cell coordinate `g = (p - origin)/cell_size - 0.5`, base cell
//! `b = floor(g)` and fraction `f = g - b`, both the density and the velocity
//! sum are gathered with the `2x2x2` trilinear weights
//!
//! ```text
//!     rho(p) = sum_corners (wx*wy*wz) * cells[idx]
//!     V(p)   = sum_corners (wx*wy*wz) * velocity[idx],  wk in {1-fk, fk}
//! ```
//!
//! with out-of-grid corners contributing `0`, and the average velocity is
//! `V(p) / rho(p)` when `rho(p) > EPS`, else the zero vector (no nearby mass).
//!
//! # Why one thread per query
//!
//! Each query's average velocity depends only on its own position and the shared
//! read-only density + velocity field, so this is embarrassingly parallel: one
//! thread owns one position, gathers from the shared `cells` and `velocity`
//! buffers and writes its own output row. No thread writes the field, so there
//! is no aliasing.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairVoxelAvgVelocity::eval`] takes a sanitized
//! [`VoxelGrid`](prism_render_architecture::hair::self_collision_voxel::VoxelGrid),
//! the field's per-cell density slice and per-cell velocity-sum slice, and a
//! batch of query positions, and returns one average-velocity vector per query,
//! in input order.
//!
//! # Correctness model
//!
//! The reference sanitizes its grid (forcing `cell_size > 0`, every `dims` axis
//! `>= 1`, `origin` finite) and treats non-finite query components as `0`. This
//! twin supplies the already-sanitized grid (via
//! [`VoxelGrid::sanitized`](prism_render_architecture::hair::self_collision_voxel::VoxelGrid::sanitized))
//! and finite positions to the device and asserts against the same reference, so
//! the kernel keeps only the in-bounds corner test and omits the non-finite
//! guards. The golden is a closed-form gather-and-divide (a handful of
//! add/sub/mul/divide/floor) with no chained recurrence, so the only `CPU` vs
//! `GPU` divergence is legal fused-multiply-add contraction and correctly
//! rounded division; parity is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`. Query positions stay clear of integer cell faces so the
//! base cell chosen is identical on both sides, and the density-gated zero
//! branch yields a bit-exact zero vector where `rho <= EPS`.
//!
//! # Portability
//!
//! The kernel uses only add/sub/mul/divide/floor/select in the portable
//! core-`WGSL` subset — no `exp`, `pow`, `log` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard trilinear voxel sampling plus a `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::self_collision_voxel::{DensityField, Vec3, VoxelGrid};

use crate::context::GpuContext;

/// Uniform parameters for one voxel average-velocity dispatch. Layout matches
/// `Params` in `shaders/voxel_avg_velocity.wesl`: the sanitized grid origin and
/// cell size in one `16`-byte slot, then the grid dims and the query count in a
/// second.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    cell_size: f32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    query_count: u32,
}

/// Compiled per-query voxel average-velocity compute twin: the shader module
/// (kept alive so its pipeline stays valid), the bind-group layout and the
/// pipeline.
pub struct GpuHairVoxelAvgVelocity {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairVoxelAvgVelocity {
    /// Compiles the per-query voxel average-velocity kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairVoxelAvgVelocity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_voxel_avg_velocity"),
            source: ShaderSource::Wgsl(include_str!("../shaders/voxel_avg_velocity.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairVoxelAvgVelocity {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples the field's average velocity at every query position under the
    /// shared `grid`, returning one vector per query, in input order.
    ///
    /// The vector for query `i` equals
    /// `field.sample_avg_velocity(grid, queries[i])`, to within the
    /// single-evaluation tolerance documented on this module (`abs_diff < 1e-4`
    /// or `rel_diff < 1e-3`). The grid is sanitized once on the host so the
    /// device sees the same clamped grid the reference uses; `cells` must be the
    /// field's own per-cell density slice and `vel_sum` its per-cell
    /// velocity-sum slice (one `Vec3` per cell, flattened to three `f32`), both
    /// of length
    /// [`VoxelGrid::cell_count`](prism_render_architecture::hair::self_collision_voxel::VoxelGrid::cell_count).
    /// The empty batch is handled without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        grid: VoxelGrid,
        cells: &[f32],
        vel_sum: &[Vec3],
        queries: &[Vec3],
    ) -> Vec<Vec3> {
        if queries.is_empty() {
            return Vec::new();
        }

        let g = grid.sanitized();
        let device = ctx.device();
        let gpu_params = GpuParams {
            origin_x: g.origin.x,
            origin_y: g.origin.y,
            origin_z: g.origin.z,
            cell_size: g.cell_size,
            dim_x: g.dims[0],
            dim_y: g.dims[1],
            dim_z: g.dims[2],
            query_count: queries.len() as u32,
        };

        // Flatten the per-cell velocity sums to three `f32` per cell (x, y, z).
        let mut flat_vel: Vec<f32> = Vec::with_capacity(vel_sum.len() * 3);
        for v in vel_sum {
            flat_vel.push(v.x);
            flat_vel.push(v.y);
            flat_vel.push(v.z);
        }

        // Flatten the query positions to three `f32` per query (x, y, z).
        let mut flat_queries: Vec<f32> = Vec::with_capacity(queries.len() * 3);
        for q in queries {
            flat_queries.push(q.x);
            flat_queries.push(q.y);
            flat_queries.push(q.z);
        }

        let out_bytes = (queries.len() as u64) * 4 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let cells_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_cells"),
            contents: bytemuck::cast_slice(cells),
            usage: BufferUsages::STORAGE,
        });
        let vel_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_vel"),
            contents: bytemuck::cast_slice(&flat_vel),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_queries"),
            contents: bytemuck::cast_slice(&flat_queries),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: cells_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: vel_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_voxel_avg_velocity_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_voxel_avg_velocity_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        debug_assert_eq!(flat.len(), queries.len() * 4);

        let mut out: Vec<Vec3> = Vec::with_capacity(queries.len());
        for i in 0..queries.len() {
            let b = i * 4;
            out.push(Vec3::new(flat[b], flat[b + 1], flat[b + 2]));
        }
        out
    }
}

/// The `CPU` golden average velocity, re-exported so the parity test can assert
/// the device twin against the identical reference it mirrors.
///
/// Runs
/// [`DensityField::sample_avg_velocity`](prism_render_architecture::hair::self_collision_voxel::DensityField::sample_avg_velocity)
/// for the position `pos` under `grid`.
#[must_use]
pub fn reference_voxel_avg_velocity(field: &DensityField, grid: VoxelGrid, pos: Vec3) -> Vec3 {
    field.sample_avg_velocity(grid, pos)
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
