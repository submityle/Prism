//! `wgpu` compute twin of Prism's Jacobi strand self-collision accumulation
//! ([`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections)).
//!
//! The Gauss-Seidel resolver applies each self-collision push in index order,
//! so a later particle already sees an earlier one's moved position — a
//! sequential reference that does not map to a `GPU` kernel. On the `GPU` every
//! particle is corrected in parallel from the *same* read-only snapshot, which
//! is a Jacobi iteration. The `CPU` golden for that parallel pass is
//! [`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections);
//! this crate is the on-device twin that walks the same per-particle
//! accumulation, one thread per particle, so a passing real-device parity test
//! is direct evidence the ported kernel computes the same corrections as the
//! reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuSelfCollisionJacobi::eval`] takes the particle array, a prebuilt
//! [`UniformGrid`] and the [`SelfCollisionParams`], and returns each particle's
//! accumulated correction. Neighbor gathering (the 27-cell grid query, sorted
//! ascending) is done host-side and uploaded as a per-particle
//! compressed-sparse-row slice, so the kernel walks each particle's neighbors
//! in the same ascending order the reference sums them (device-side spatial
//! hashing is a separate dispatch). The pass writes corrections only; applying
//! them to positions stays with the caller, matching the golden's split of
//! [`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections)
//! and `apply_corrections`.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `min`, `max`, `dot` and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The accumulation contains no transcendental call, so `CPU` and `GPU`
//! evaluate the same closed-form geometry summed in the same ascending
//! neighbor order. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits. The
//! parity test therefore asserts a per-component tolerance rather than exact
//! equality.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard uniform-grid self-collision push (`TressFX` 4 style)
//! plus Jacobi accumulation and `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
use prism_render_architecture::hair::self_collision::SelfCollisionParams;
use prism_render_architecture::hair::self_collision_grid::{grid_cell_of, UniformGrid};

use crate::context::GpuContext;

/// Uniform parameters for one Jacobi accumulation dispatch. Layout matches
/// `Params` in `shaders/self_collision_jacobi.wesl`: the particle count then the
/// three precomputed scalar terms, packed into one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    min_sep: f32,
    min_sep_sq: f32,
    stiffness: f32,
}

/// A compiled, reusable Jacobi self-collision accumulation pipeline.
pub struct GpuSelfCollisionJacobi {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSelfCollisionJacobi {
    /// Compiles the Jacobi self-collision kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSelfCollisionJacobi {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_self_collision_jacobi"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/self_collision_jacobi.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_self_collision_jacobi_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_self_collision_jacobi_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_self_collision_jacobi_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSelfCollisionJacobi {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates every particle's Jacobi self-collision correction, returning
    /// one [`Vec3`] per particle in input order (same length as `particles`).
    ///
    /// The result for particle `i` equals
    /// [`accumulate_jacobi_corrections`](prism_render_architecture::hair::self_collision_jacobi::accumulate_jacobi_corrections)
    /// applied to the same array, grid and params, to within the
    /// fused-multiply-add tolerance documented on this module. An empty array
    /// returns an empty vector, and degenerate params (non-positive radius,
    /// stiffness or cell size, or a non-finite cell size) return all-zero
    /// corrections without a dispatch — matching the golden's no-op guard while
    /// keeping storage buffers non-zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[StrandParticle],
        grid: &UniformGrid,
        params: SelfCollisionParams,
    ) -> Vec<Vec3> {
        if particles.is_empty() {
            return Vec::new();
        }
        // The golden clears `out` to zero then returns before any push when the
        // parameters are degenerate; reproduce that all-zero result without a
        // dispatch so no zero-sized storage buffer arises.
        if params.particle_radius <= 0.0
            || params.stiffness <= 0.0
            || params.cell_size <= 0.0
            || !params.cell_size.is_finite()
        {
            return alloc_zero(particles.len());
        }

        let device = ctx.device();
        let min_sep = 2.0 * params.particle_radius;
        let min_sep_sq = min_sep * min_sep;
        let stiffness = params.stiffness.clamp(0.0, 1.0);
        let cell_size = grid.cell_size();
        let uniforms = Params {
            particle_count: particles.len() as u32,
            min_sep,
            min_sep_sq,
            stiffness,
        };

        // Flatten positions and inverse masses (stride 3 / stride 1).
        let mut positions: Vec<f32> = Vec::with_capacity(particles.len() * 3);
        let mut inv_mass: Vec<f32> = Vec::with_capacity(particles.len());
        for particle in particles {
            positions.push(particle.position.x);
            positions.push(particle.position.y);
            positions.push(particle.position.z);
            inv_mass.push(particle.inverse_mass);
        }

        // Build the per-particle neighbor CSR exactly as the golden gathers it:
        // the 27-cell grid query around each particle's own cell, sorted
        // ascending. Non-finite or pinned particles still get a slice built here
        // (harmless); the kernel short-circuits them before it is walked.
        let mut starts: Vec<u32> = Vec::with_capacity(particles.len() + 1);
        let mut indices: Vec<u32> = Vec::new();
        let mut neighbors: Vec<u32> = Vec::new();
        starts.push(0);
        for particle in particles {
            grid.neighbors(grid_cell_of(particle.position, cell_size), &mut neighbors);
            indices.extend_from_slice(&neighbors);
            starts.push(indices.len() as u32);
        }
        // A storage buffer cannot be zero-sized: if no particle had a neighbor
        // (every particle non-finite, so the grid is empty), pad with one index
        // the kernel never reads (every slice is empty).
        if indices.is_empty() {
            indices.push(0);
        }

        let out_bytes = (positions.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_jacobi_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_jacobi_positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: BufferUsages::STORAGE,
        });
        let inv_mass_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_jacobi_inv_mass"),
            contents: bytemuck::cast_slice(&inv_mass),
            usage: BufferUsages::STORAGE,
        });
        let starts_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_jacobi_starts"),
            contents: bytemuck::cast_slice(&starts),
            usage: BufferUsages::STORAGE,
        });
        let indices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_self_collision_jacobi_indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_jacobi_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_self_collision_jacobi_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_self_collision_jacobi_bind_group"),
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
                    resource: inv_mass_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: starts_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: indices_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_self_collision_jacobi_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_self_collision_jacobi_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (particles.len() as u32).div_ceil(64);
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        flat.chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect()
    }
}

/// A vector of `n` zero corrections, matching the golden's cleared-`out`
/// early return for degenerate parameters.
fn alloc_zero(n: usize) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(n);
    out.resize(n, Vec3::ZERO);
    out
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
