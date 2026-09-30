//! Real-device `wgpu` implementation of the marker-particle advection that
//! closes the full fluid step.
//!
//! [`GpuAdvect`] compiles `shaders/fluid_advect.wgsl` once and exposes
//! [`GpuAdvect::advect`], which moves every marker through the staggered `MAC`
//! velocity field by one second-order Runge–Kutta midpoint step in a single
//! compute dispatch of one thread per marker. It reproduces the arithmetic of
//! the [`super::super::cpu::advect`] golden twin; because the only operations
//! are the two trilinear gathers already shared with the transfer, the device
//! result matches the twin within a tight tolerance.
//!
//! # Provenance
//!
//! Second-order Runge–Kutta advection of markers through a sampled velocity
//! field is a standard semi-Lagrangian technique (Bridson; Zhu and Bridson
//! 2005). No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::fluid::config::FluidError;
use crate::fluid::grid::GridDims;

use super::layout::{buffer_entry, entry};

/// Uniform parameter block. Layout matches `Params` in
/// `shaders/fluid_advect.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `xyz` = grid origin, `w` = cell size.
    origin_dx: [f32; 4],
    /// `x`/`y`/`z` = cell counts, `w` = particle count.
    dims: [u32; 4],
    /// `x`/`y`/`z` = concatenated face bases, `w` = total face count.
    bases: [u32; 4],
    /// `x` = time step `dt`, `yzw` = padding.
    step: [f32; 4],
}

/// A compiled, reusable `GPU` advection pipeline.
pub struct GpuAdvect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    advect: ComputePipeline,
}

impl GpuAdvect {
    /// Compiles the advection kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAdvect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fluid_advect"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fluid_advect.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fluid_advect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fluid_advect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let advect = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_fluid_advect_step"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("advect"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAdvect {
            module,
            layout,
            advect,
        }
    }

    /// Advects `positions` through the concatenated `[u | v | w]` `velocity`
    /// field by one midpoint step of size `dt`, writing the result in place.
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::EmptyGrid`] when `dims` is degenerate and
    /// [`FluidError::InvalidConfig`] when `velocity` does not have
    /// [`GridDims::face_total`] entries. Does nothing (returns `Ok`) when
    /// `positions` is empty.
    pub fn advect(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        velocity: &[f32],
        positions: &mut [Vec3],
        dt: f32,
    ) -> Result<(), FluidError> {
        if !dims.is_valid() {
            return Err(FluidError::EmptyGrid);
        }
        if velocity.len() != dims.face_total() {
            return Err(FluidError::InvalidConfig(
                "velocity length must equal the concatenated face count",
            ));
        }
        if positions.is_empty() {
            return Ok(());
        }

        let plan = self.upload(ctx, dims, velocity, positions, dt);
        let stage = self.encode_and_run(ctx, &plan);
        let out = buffer::read_back::<[f32; 4]>(ctx, &stage);
        for (dst, src) in positions.iter_mut().zip(out.iter()) {
            *dst = Vec3::new(src[0], src[1], src[2]);
        }
        Ok(())
    }

    /// Uploads the field, positions, and parameters and builds the bind group.
    fn upload(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        velocity: &[f32],
        positions: &[Vec3],
        dt: f32,
    ) -> Plan {
        let device = ctx.device();
        let count = positions.len() as u32;
        let face_total = dims.face_total() as u32;
        let params = Params {
            origin_dx: [dims.origin.x, dims.origin.y, dims.origin.z, dims.dx],
            dims: [dims.nx, dims.ny, dims.nz, count],
            bases: [
                dims.u_offset() as u32,
                dims.v_offset() as u32,
                dims.w_offset() as u32,
                face_total,
            ],
            step: [dt, 0.0, 0.0, 0.0],
        };
        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let out_init = vec![[0.0f32; 4]; positions.len()];

        let params_buf = buffer::uniform(device, "fluid_advect_params", &params);
        let positions_in = buffer::storage_read(device, "fluid_advect_positions_in", &packed);
        let positions_out =
            buffer::storage_rw_init(device, "fluid_advect_positions_out", &out_init);
        let velocity_buf = buffer::storage_read(device, "fluid_advect_velocity", velocity);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_advect_bind_group"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_in),
                entry(2, &positions_out),
                entry(3, &velocity_buf),
            ],
        });

        Plan {
            bind,
            positions_out,
            positions_bytes: (positions.len() * 16) as u64,
            groups: count.div_ceil(64).max(1),
        }
    }

    /// Records the advection pass, copies the results to a staging buffer,
    /// submits, and returns the staging buffer.
    fn encode_and_run(&self, ctx: &GpuContext, plan: &Plan) -> Buffer {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fluid_advect_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_fluid_advect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.advect);
            pass.set_bind_group(0, &plan.bind, &[]);
            pass.dispatch_workgroups(plan.groups, 1, 1);
        }
        let stage = buffer::staging(device, "fluid_advect_stage", plan.positions_bytes);
        buffer::copy(
            &mut encoder,
            &plan.positions_out,
            &stage,
            plan.positions_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        stage
    }
}

/// The uploaded buffers, bind group, and dispatch size for one advection.
struct Plan {
    bind: BindGroup,
    positions_out: Buffer,
    positions_bytes: u64,
    groups: u32,
}
