//! `wgpu` compute twin of Prism's groom motion-energy reduction
//! ([`groom_motion_energy`](prism_render_architecture::hair::sleep::groom_motion_energy)),
//! the groom-global scalar that drives the hysteretic sleep gate.
//!
//! The sleep gate puts a groom that has come to rest to sleep so it stops
//! consuming the shared deformation budget. Its input is one groom-global
//! scalar: the sum over every strand particle of its squared implicit velocity,
//! where velocity is stored implicitly as `position - prev_position` (a Verlet
//! pair). The golden walks the flat particle array left to right, accumulating
//! `(position - prev_position)` dotted with itself. This kernel folds the
//! identical quantity on-device: a single `256`-wide workgroup grid-strides
//! across the particles, derives each particle's squared diff inline, and
//! tree-reduces the partials to one scalar.
//!
//! # A distinct sibling of the shared-lane reduction
//!
//! The [`analysis_reduce`](crate::analysis_reduce) twin reduces a *precomputed*
//! per-element lane of a finished metric buffer. This kernel instead *fuses*
//! the per-particle compute (`pos - prev`, then the self-`dot`) with the sum
//! reduction in one pass straight from raw particle positions — a different
//! input (positions, not a metric lane) and a different first step.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairMotionEnergy::energy`] takes the groom's flat particle slice and
//! returns the single folded scalar. Only each particle's `position` and
//! `prev_position` are uploaded (two `vec4`, xyz used); the inverse mass does
//! not affect the energy and is not sent. An empty groom returns the additive
//! identity `0` without a dispatch — storage buffers cannot be zero-sized.
//!
//! # Correctness model
//!
//! Floating-point addition is commutative but not associative, so the tree's
//! pairwise order differs from the golden's left-to-right walk by a few
//! low-mantissa `ULP`; the parity test asserts a tolerance (`abs_diff < 1e-4`
//! or `rel_diff < 1e-3`) rather than bit-for-bit. The per-particle squared diff
//! is itself exact up to fma fusion.
//!
//! # Portability
//!
//! The kernel uses only subtraction, multiply/add and workgroup shared memory
//! in the portable core-`WGSL` subset — no `exp`, `pow`, atomics or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction over a
//! Verlet velocity proxy; no Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dynamics::StrandParticle;
use prism_render_architecture::hair::sleep::groom_motion_energy;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

extern crate alloc;

/// Uniform parameters for one motion-energy dispatch. `16`-byte scalar-packed
/// `repr(C)` matching `Params` in `shaders/motion_energy.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable groom motion-energy reduction pipeline.
pub struct GpuHairMotionEnergy {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairMotionEnergy {
    /// Compiles the motion-energy reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus workgroup
    /// shared memory, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairMotionEnergy {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_motion_energy"),
            source: ShaderSource::Wgsl(include_str!("../shaders/motion_energy.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_motion_energy_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_motion_energy_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_motion_energy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairMotionEnergy {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds the groom's whole particle slice into its scalar motion energy.
    ///
    /// The result equals the `CPU` golden
    /// [`groom_motion_energy`](prism_render_architecture::hair::sleep::groom_motion_energy)
    /// of the same particles within an fma / reassociation tolerance (the only
    /// departures are the tree-reduction's pairwise sum order and
    /// possibly-fused multiply-adds). An empty groom returns the additive
    /// identity `0` without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn energy(&self, ctx: &GpuContext, particles: &[StrandParticle]) -> f32 {
        // Empty groom: nothing to fold; return the identity without touching
        // the device (storage buffers cannot be zero-sized).
        if particles.is_empty() {
            return 0.0;
        }

        let uniform = Params {
            particle_count: particles.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Two vec4 per particle: position (xyz), then prev_position (xyz).
        let mut gpu_particles: Vec<[f32; 4]> = Vec::with_capacity(particles.len() * 2);
        for p in particles {
            gpu_particles.push([p.position.x, p.position.y, p.position.z, 0.0]);
            gpu_particles.push([p.prev_position.x, p.prev_position.y, p.prev_position.z, 0.0]);
        }

        let device = ctx.device();
        let out_bytes = size_of::<f32>() as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_motion_energy_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let particles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_motion_energy_particles"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_motion_energy_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_motion_energy_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_motion_energy_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: particles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_motion_energy_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_motion_energy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup cooperatively reduces the whole buffer.
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        read_f32_scalar(&out_stage)
    }
}

/// Runs the golden motion-energy reduction directly; a thin re-export so the
/// parity test can name one reference path.
#[must_use]
pub fn reference_motion_energy(particles: &[StrandParticle]) -> f32 {
    groom_motion_energy(particles)
}

/// Reads the single mapped `f32` back from a staging buffer, then unmaps.
fn read_f32_scalar(stage: &wgpu::Buffer) -> f32 {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let value = bytemuck::cast_slice::<u8, f32>(&view)[0];
    drop(view);
    stage.unmap();
    value
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
