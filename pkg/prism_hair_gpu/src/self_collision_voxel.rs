//! `wgpu` compute twin of Prism's voxel-density strand self-collision resolve
//! ([`resolve_self_collision`](prism_render_architecture::hair::self_collision_voxel::resolve_self_collision)).
//!
//! A dense groom has far too many strands for an all-pairs strand-vs-strand
//! test, so the production answer (AMD `TressFX`, and the volumetric hair work
//! of `Petrovic` et al. 2005) treats hair as a *density field*: splat every
//! particle into a low-resolution voxel grid, then let the field drive three
//! cheap grid-local forces — density-gradient repulsion, two-sided volume
//! preservation, and a hair-hair friction velocity blend — that together
//! approximate large-scale self-collision in `O(n)`.
//!
//! That pass has two phases. **Phase 1** *splats* each particle's mass and
//! mass-weighted velocity into the field with trilinear scatter-add — a
//! many-writers-one-cell reduction whose on-device form needs float atomics (or
//! a separate binning dispatch), so it stays with the `CPU` golden
//! [`splat_density`](prism_render_architecture::hair::self_collision_voxel::splat_density)
//! and the finished read-only field is uploaded here. **Phase 2** is
//! embarrassingly parallel: each particle samples the field at its *original*
//! position and applies the three forces. Because the field is built once from
//! the input snapshot, phase 2 is a pure function of the inputs regardless of
//! thread order, so this twin reproduces the golden resolve body exactly, one
//! thread per particle — a passing real-device parity test is direct evidence
//! the ported kernel computes the same displaced positions and blended
//! velocities as the reference, not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairSelfCollisionVoxel::eval`] takes the particle array, the
//! [`VoxelGrid`] and the [`SelfCollisionParams`], builds the density field
//! host-side with the golden [`splat_density`], and returns one
//! [`HairPoint`] per input (same length, carrying the input mass through) with
//! its resolved position and velocity. Non-finite positions are left untouched
//! and an empty array returns an empty vector, matching the golden's guards.
//!
//! # Portability
//!
//! The kernel uses only `floor`, `sqrt`, `min`/`max`, `clamp`, `dot`, `select`
//! and multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The resolve contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form trilinear gather, central-difference gradient and friction
//! blend. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, and the gradient divides by a cell separation, so
//! the parity test asserts a per-component tolerance rather than exact equality.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard voxel-density hair self-collision (`TressFX` 4 /
//! `Petrovic` et al. 2005 volumetric hair) plus `wgpu` compute dispatch; no
//! Unreal Engine source or derived code.

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

use prism_render_architecture::hair::self_collision_voxel::{
    splat_density, HairPoint, SelfCollisionParams, Vec3, VoxelGrid,
};

use crate::context::GpuContext;

/// Uniform parameters for one voxel self-collision resolve dispatch. Layout
/// matches `Params` in `shaders/self_collision_voxel.wesl`: the particle count
/// and the sanitised param/grid scalars, padded to four `16`-byte rows.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    rest_density: f32,
    repulsion: f32,
    friction: f32,
    target_volume_gain: f32,
    cell_size: f32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    pad3: f32,
}

/// A compiled, reusable voxel self-collision resolve pipeline.
pub struct GpuHairSelfCollisionVoxel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairSelfCollisionVoxel {
    /// Compiles the voxel self-collision resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairSelfCollisionVoxel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_self_collision_voxel"),
            source: ShaderSource::Wgsl(include_str!("../shaders/self_collision_voxel.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_self_collision_voxel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_self_collision_voxel_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_self_collision_voxel_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairSelfCollisionVoxel {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves voxel self-collision for every particle, returning one
    /// [`HairPoint`] per input in order (same length as `points`, with each
    /// input mass carried through unchanged).
    ///
    /// The result equals
    /// [`resolve_self_collision`](prism_render_architecture::hair::self_collision_voxel::resolve_self_collision)
    /// applied to a clone of `points` with the same `grid` and `params`, to
    /// within the fused-multiply-add tolerance documented on this module. The
    /// density field is built host-side with the golden [`splat_density`]
    /// (phase 1), so this call only runs the parallel gather-and-resolve
    /// (phase 2). An empty array returns an empty vector.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        points: &[HairPoint],
        grid: VoxelGrid,
        params: SelfCollisionParams,
    ) -> Vec<HairPoint> {
        if points.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();
        let grid = grid.sanitized();
        let params = params.sanitized();
        // Phase 1 (host): build the read-only density + velocity field exactly
        // as the golden resolve does, so the kernel gathers from the same field.
        let field = splat_density(points, grid);

        let uniforms = Params {
            particle_count: points.len() as u32,
            rest_density: params.rest_density,
            repulsion: params.repulsion,
            friction: params.friction,
            target_volume_gain: params.target_volume_gain,
            cell_size: grid.cell_size,
            dim_x: grid.dims[0],
            dim_y: grid.dims[1],
            dim_z: grid.dims[2],
            pad0: 0,
            pad1: 0,
            pad2: 0,
            origin_x: grid.origin.x,
            origin_y: grid.origin.y,
            origin_z: grid.origin.z,
            pad3: 0.0,
        };

        // Flatten per-particle positions / velocities (stride 3).
        let mut positions: Vec<f32> = Vec::with_capacity(points.len() * 3);
        let mut velocities: Vec<f32> = Vec::with_capacity(points.len() * 3);
        for p in points {
            positions.push(p.position.x);
            positions.push(p.position.y);
            positions.push(p.position.z);
            velocities.push(p.velocity.x);
            velocities.push(p.velocity.y);
            velocities.push(p.velocity.z);
        }

        // Flatten the splatted field: scalar density (stride 1) and the
        // mass-weighted velocity sum (stride 3). Both have `cell_count` entries,
        // which is `>= 1` on the sanitised grid, so neither buffer is empty.
        let cells: Vec<f32> = field.cells.clone();
        let mut field_vel: Vec<f32> = Vec::with_capacity(field.velocity.len() * 3);
        for v in &field.velocity {
            field_vel.push(v.x);
            field_vel.push(v.y);
            field_vel.push(v.z);
        }

        let out_bytes = (positions.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_voxel_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_voxel_positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: BufferUsages::STORAGE,
        });
        let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_voxel_velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: BufferUsages::STORAGE,
        });
        let cells_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_voxel_cells"),
            contents: bytemuck::cast_slice(&cells),
            usage: BufferUsages::STORAGE,
        });
        let field_vel_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_voxel_field_vel"),
            contents: bytemuck::cast_slice(&field_vel),
            usage: BufferUsages::STORAGE,
        });
        let out_pos_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_voxel_out_pos"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_vel_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_voxel_out_vel"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pos_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_voxel_pos_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vel_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_voxel_vel_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_self_collision_voxel_bind_group"),
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
                    resource: velocities_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: cells_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: field_vel_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: out_pos_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: out_vel_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_self_collision_voxel_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_self_collision_voxel_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (points.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_pos_buf, 0, &pos_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&out_vel_buf, 0, &vel_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        pos_stage.slice(..).map_async(MapMode::Read, |_| {});
        vel_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let pos_view = pos_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out_pos = bytemuck::cast_slice::<u8, f32>(&pos_view).to_vec();
        drop(pos_view);
        pos_stage.unmap();

        let vel_view = vel_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out_vel = bytemuck::cast_slice::<u8, f32>(&vel_view).to_vec();
        drop(vel_view);
        vel_stage.unmap();

        points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let b = i * 3;
                HairPoint {
                    position: Vec3::new(out_pos[b], out_pos[b + 1], out_pos[b + 2]),
                    velocity: Vec3::new(out_vel[b], out_vel[b + 1], out_vel[b + 2]),
                    mass: p.mass,
                }
            })
            .collect()
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
