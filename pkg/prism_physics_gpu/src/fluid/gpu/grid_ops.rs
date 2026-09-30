//! Real-device `wgpu` implementation of the per-face grid operators that run
//! between the particle-to-grid scatter and the pressure projection: the
//! body-force integration and the no-through-flow solid boundary condition.
//!
//! [`GpuGridOps`] compiles `shaders/fluid_grid_ops.wgsl` once and exposes
//! [`GpuGridOps::apply`], which uploads the cell classification and the
//! concatenated `[u | v | w]` face velocities, then encodes the operators as a
//! chain of compute passes in a single submission: one `add_gravity` dispatch
//! over every face, then one `enforce_u` / `enforce_v` / `enforce_w` dispatch
//! per staggered axis. The order matches the reference step
//! ([`prism_physics_core`](prism_physics_core::fluid)): gravity is integrated
//! first, then the solid faces are forced back to zero.
//!
//! Every write is either a single add of a host-computed constant or a store of
//! zero, so the projected field matches the [`super::super::cpu::grid_ops`]
//! golden twin exactly, which the real-device parity test asserts.
//!
//! # Provenance
//!
//! Explicit body-force integration on a staggered `MAC` grid and the solid
//! no-through-flow boundary are standard `CFD` constructs (Harlow and Welch
//! 1965; Bridson, *Fluid Simulation for Computer Graphics*). No Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::fluid::config::FluidError;
use crate::fluid::grid::{CellType, GridDims};

use super::layout::{buffer_entry, entry};

/// Uniform parameter block. Layout matches `Params` in
/// `shaders/fluid_grid_ops.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `x`/`y`/`z` = cell counts, `w` = cell count.
    dims: [u32; 4],
    /// `x`/`y`/`z` = concatenated face bases, `w` = total face count.
    bases: [u32; 4],
    /// `x`/`y`/`z` = per-axis face increment `gravity * dt`, `w` = unused.
    dv: [f32; 4],
}

/// A compiled, reusable `GPU` grid-operator pipeline set.
pub struct GpuGridOps {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    add_gravity: ComputePipeline,
    enforce_u: ComputePipeline,
    enforce_v: ComputePipeline,
    enforce_w: ComputePipeline,
}

impl GpuGridOps {
    /// Compiles the grid-operator kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGridOps {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fluid_grid_ops"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fluid_grid_ops.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fluid_grid_ops_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fluid_grid_ops_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let add_gravity = make(
            device,
            &module,
            "add_gravity",
            "prism_fluid_add_gravity",
            &pipeline_layout,
        );
        let enforce_u = make(
            device,
            &module,
            "enforce_u",
            "prism_fluid_enforce_u",
            &pipeline_layout,
        );
        let enforce_v = make(
            device,
            &module,
            "enforce_v",
            "prism_fluid_enforce_v",
            &pipeline_layout,
        );
        let enforce_w = make(
            device,
            &module,
            "enforce_w",
            "prism_fluid_enforce_w",
            &pipeline_layout,
        );
        GpuGridOps {
            module,
            layout,
            add_gravity,
            enforce_u,
            enforce_v,
            enforce_w,
        }
    }

    /// Integrates the body force `gravity * dt` into every face and then zeroes
    /// the faces bordering solid cells, in place, reading back the result into
    /// `velocity`.
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::EmptyGrid`] when `dims` has a zero axis and
    /// [`FluidError::InvalidConfig`] when a slice length does not match `dims`.
    pub fn apply(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        cell_types: &[CellType],
        velocity: &mut [f32],
        gravity: Vec3,
        dt: f32,
    ) -> Result<(), FluidError> {
        if !dims.is_valid() {
            return Err(FluidError::EmptyGrid);
        }
        if cell_types.len() != dims.cell_count() {
            return Err(FluidError::InvalidConfig(
                "cell_types length must equal the cell count",
            ));
        }
        if velocity.len() != dims.face_total() {
            return Err(FluidError::InvalidConfig(
                "velocity length must equal the total face count",
            ));
        }

        let plan = self.upload(ctx, dims, cell_types, velocity, gravity, dt);
        let stage = self.encode_and_run(ctx, dims, &plan);
        let projected = buffer::read_back::<f32>(ctx, &stage);
        velocity.copy_from_slice(&projected[..velocity.len()]);
        Ok(())
    }

    /// Uploads every buffer and builds the bind group for one application.
    fn upload(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        cell_types: &[CellType],
        velocity: &[f32],
        gravity: Vec3,
        dt: f32,
    ) -> GridOpsPlan {
        let device = ctx.device();
        let dv = gravity * dt;
        let params = Params {
            dims: [dims.nx, dims.ny, dims.nz, dims.cell_count() as u32],
            bases: [
                dims.u_offset() as u32,
                dims.v_offset() as u32,
                dims.w_offset() as u32,
                dims.face_total() as u32,
            ],
            dv: [dv.x, dv.y, dv.z, 0.0],
        };

        let codes: Vec<u32> = cell_types.iter().map(|t| t.code()).collect();
        let params_buf = buffer::uniform(device, "fluid_grid_ops_params", &params);
        let cell_buf = buffer::storage_read(device, "fluid_grid_ops_cells", &codes);
        let velocity_buf = buffer::storage_rw_init(device, "fluid_grid_ops_velocity", velocity);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_grid_ops_bind_group"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &cell_buf),
                entry(2, &velocity_buf),
            ],
        });

        GridOpsPlan {
            bind,
            velocity_buf,
            velocity_bytes: (dims.face_total() * 4) as u64,
        }
    }

    /// Records the gravity add and the three solid-face enforcements into one
    /// encoder, submits, and returns the velocity staging buffer.
    fn encode_and_run(&self, ctx: &GpuContext, dims: GridDims, plan: &GridOpsPlan) -> Buffer {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fluid_grid_ops_encoder"),
        });
        self.pass(
            &mut encoder,
            &self.add_gravity,
            plan,
            (dims.face_total() as u32).div_ceil(64).max(1),
        );
        self.pass(
            &mut encoder,
            &self.enforce_u,
            plan,
            (dims.u_count() as u32).div_ceil(64).max(1),
        );
        self.pass(
            &mut encoder,
            &self.enforce_v,
            plan,
            (dims.v_count() as u32).div_ceil(64).max(1),
        );
        self.pass(
            &mut encoder,
            &self.enforce_w,
            plan,
            (dims.w_count() as u32).div_ceil(64).max(1),
        );

        let stage = buffer::staging(device, "fluid_grid_ops_stage", plan.velocity_bytes);
        buffer::copy(
            &mut encoder,
            &plan.velocity_buf,
            &stage,
            plan.velocity_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        stage
    }

    /// Records one compute pass dispatching `groups` workgroups.
    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ComputePipeline,
        plan: &GridOpsPlan,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_fluid_grid_ops_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &plan.bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// The uploaded buffers and readback size for one application.
struct GridOpsPlan {
    bind: BindGroup,
    velocity_buf: Buffer,
    velocity_bytes: u64,
}

/// Compiles one compute pipeline for `entry_point` under `layout`.
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
