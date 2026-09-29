//! Real-device `wgpu` compute implementation of the `FLIP`/`APIC` particle-grid
//! transfer.
//!
//! [`GpuFluidSolver`] compiles `shaders/fluid_transfer.wgsl` once and exposes
//! [`GpuFluidSolver::transfer`], which uploads the particles and a fresh set of
//! staggered-field accumulators, then encodes the transfer as a chain of
//! compute passes in a single submission: `p2g_scatter` (fixed-point atomic
//! splat) -> `normalize` (momentum / weight per face) -> `g2p` (trilinear
//! gather with the `PIC`/`FLIP` blend). Separate passes give the implicit memory
//! barrier so the scattered accumulators are visible to normalisation and the
//! normalised field is visible to the gather, reproducing the sequential `CPU`
//! golden pipeline.
//!
//! # Scope
//!
//! This is the transfer stage only: the pressure projection, gravity, solid
//! handling, and advection that complete a full fluid step are separate
//! milestones. The saved (`FLIP`-increment) field is therefore uploaded as
//! zero here, so a pure-`PIC` blend is an exact particle-grid-particle
//! round-trip and is what the real-device parity test checks.
//!
//! # Provenance
//!
//! Trilinear `P2G`/`G2P` with the `PIC`/`FLIP` blend (Zhu and Bridson 2005;
//! Bridson) and fixed-point atomic scatter (standard `GPU` technique). No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::fluid::config::FluidError;
use crate::fluid::grid::GridDims;
use crate::fluid::particle::FluidParticles;

use super::layout::{buffer_entry, entry};

/// Uniform parameter block. Layout matches `Params` in
/// `shaders/fluid_transfer.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `xyz` = grid origin, `w` = cell size.
    origin_dx: [f32; 4],
    /// `x`/`y`/`z` = cell counts, `w` = particle count.
    dims: [u32; 4],
    /// `x`/`y`/`z` = concatenated face bases, `w` = total face count.
    bases: [u32; 4],
    /// `x` = `FLIP` blend, `yzw` = padding.
    blend: [f32; 4],
}

/// A compiled, reusable `GPU` fluid-transfer pipeline set.
pub struct GpuFluidSolver {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    p2g_scatter: ComputePipeline,
    normalize: ComputePipeline,
    g2p: ComputePipeline,
}

impl GpuFluidSolver {
    /// Compiles the transfer kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fluid_transfer"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fluid_transfer.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fluid_transfer_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fluid_transfer_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let p2g_scatter = make(
            device,
            &module,
            "p2g_scatter",
            "prism_fluid_p2g_scatter",
            &pipeline_layout,
        );
        let normalize = make(
            device,
            &module,
            "normalize",
            "prism_fluid_normalize",
            &pipeline_layout,
        );
        let g2p = make(device, &module, "g2p", "prism_fluid_g2p", &pipeline_layout);
        GpuFluidSolver {
            module,
            layout,
            p2g_scatter,
            normalize,
            g2p,
        }
    }

    /// Transfers particle velocities to the grid and back on device, writing the
    /// reconstructed velocities into `particles` in place.
    ///
    /// `blend` is the `FLIP` fraction in `[0, 1]`; with the zero saved field of
    /// this transfer-only stage, `blend = 0` is a pure-`PIC` round-trip.
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::EmptyGrid`] when `dims` has a zero axis and
    /// [`FluidError::InconsistentParticles`] when the particle columns differ in
    /// length. Does nothing (returns `Ok`) when there are no particles.
    pub fn transfer(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        particles: &mut FluidParticles,
        blend: f32,
    ) -> Result<(), FluidError> {
        if !dims.is_valid() {
            return Err(FluidError::EmptyGrid);
        }
        if !particles.is_consistent() {
            return Err(FluidError::InconsistentParticles);
        }
        if particles.is_empty() {
            return Ok(());
        }

        let plan = self.upload(ctx, dims, particles, blend);
        let staging = self.encode_and_run(ctx, &plan);
        let out = buffer::read_back::<[f32; 4]>(ctx, &staging);
        let velocities = particles.velocities_mut();
        for (dst, src) in velocities.iter_mut().zip(out.iter()) {
            *dst = glam::Vec3::new(src[0], src[1], src[2]);
        }
        Ok(())
    }

    /// Uploads every buffer and builds the bind group for one transfer.
    fn upload(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        particles: &FluidParticles,
        blend: f32,
    ) -> TransferPlan {
        let device = ctx.device();
        let particle_count = particles.len() as u32;
        let face_total = dims.face_total() as u32;

        let params = Params {
            origin_dx: [dims.origin.x, dims.origin.y, dims.origin.z, dims.dx],
            dims: [dims.nx, dims.ny, dims.nz, particle_count],
            bases: [
                dims.u_offset() as u32,
                dims.v_offset() as u32,
                dims.w_offset() as u32,
                face_total,
            ],
            blend: [blend.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
        };

        let positions: Vec<[f32; 4]> = particles.positions().iter().map(vec3_to_vec4).collect();
        let velocities: Vec<[f32; 4]> = particles.velocities().iter().map(vec3_to_vec4).collect();
        let out_init = vec![[0.0f32; 4]; particles.len()];
        let saved = vec![0.0f32; face_total.max(1) as usize];

        let params_buf = buffer::uniform(device, "fluid_params", &params);
        let positions_buf = buffer::storage_read(device, "fluid_positions", &positions);
        let velocities_in_buf = buffer::storage_read(device, "fluid_velocities_in", &velocities);
        let velocities_out_buf = buffer::storage_rw_init(device, "fluid_velocities_out", &out_init);
        let momentum_buf =
            buffer::storage_rw_zeroed(device, "fluid_momentum", u64::from(face_total) * 4);
        let weight_buf =
            buffer::storage_rw_zeroed(device, "fluid_weight", u64::from(face_total) * 4);
        let velocity_buf =
            buffer::storage_rw_zeroed(device, "fluid_velocity", u64::from(face_total) * 4);
        let saved_buf = buffer::storage_read(device, "fluid_saved", &saved);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_transfer_bind_group"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_in_buf),
                entry(3, &velocities_out_buf),
                entry(4, &momentum_buf),
                entry(5, &weight_buf),
                entry(6, &velocity_buf),
                entry(7, &saved_buf),
            ],
        });

        TransferPlan {
            bind,
            velocities_out_buf,
            velocities_bytes: (particles.len() * 16) as u64,
            particle_count,
            face_total,
        }
    }

    /// Records the three transfer passes into one encoder, submits, and returns
    /// the staging buffer the reconstructed velocities were copied into.
    fn encode_and_run(&self, ctx: &GpuContext, plan: &TransferPlan) -> Buffer {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fluid_transfer_encoder"),
        });
        let particle_groups = plan.particle_count.div_ceil(64).max(1);
        let face_groups = plan.face_total.div_ceil(64).max(1);
        self.pass(&mut encoder, &self.p2g_scatter, plan, particle_groups);
        self.pass(&mut encoder, &self.normalize, plan, face_groups);
        self.pass(&mut encoder, &self.g2p, plan, particle_groups);

        let stage = buffer::staging(device, "fluid_velocity_stage", plan.velocities_bytes);
        buffer::copy(
            &mut encoder,
            &plan.velocities_out_buf,
            &stage,
            plan.velocities_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        stage
    }

    /// Records one compute pass dispatching `groups` workgroups.
    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ComputePipeline,
        plan: &TransferPlan,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_fluid_transfer_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &plan.bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// The uploaded buffers and dispatch dimensions for one transfer.
struct TransferPlan {
    bind: BindGroup,
    velocities_out_buf: Buffer,
    velocities_bytes: u64,
    particle_count: u32,
    face_total: u32,
}

/// Compiles one compute pipeline for `entry` under `layout`.
fn make(
    device: &wgpu::Device,
    module: &ShaderModule,
    entry_point: &str,
    label: &str,
    layout: &PipelineLayout,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Packs a [`glam::Vec3`] into a padded `vec4` upload element.
fn vec3_to_vec4(v: &glam::Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}
