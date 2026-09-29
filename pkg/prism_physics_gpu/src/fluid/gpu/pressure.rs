//! Real-device `wgpu` compute implementation of the red-black `SOR` pressure
//! projection.
//!
//! [`GpuPressureSolver`] compiles `shaders/fluid_pressure.wgsl` once and exposes
//! [`GpuPressureSolver::project`], which uploads the cell classification, the
//! concatenated `[u | v | w]` face velocities, and a zeroed pressure field,
//! then encodes the solve as a chain of compute passes in a single submission:
//! for each iteration a `sor_red` dispatch then a `sor_black` dispatch, and
//! finally one `subtract_u` / `subtract_v` / `subtract_w` dispatch per axis.
//!
//! Separate passes give the implicit memory barrier that makes the red sweep's
//! pressure writes visible to the black sweep, so the device reproduces the
//! sequential-per-colour order of the [`super::super::cpu::pressure`] golden
//! twin. The projected velocities (and the pressure field) are read back into
//! the caller's slice.
//!
//! # Provenance
//!
//! The `MAC` pressure-projection scheme, the solid / free-surface boundary
//! handling, and the red-black `SOR` Poisson solve follow Bridson, *Fluid
//! Simulation for Computer Graphics*, and Foster and Fedkiw 2001. No Unreal
//! Engine source or derived code.

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
use crate::fluid::cpu::pressure::PressureConfig;
use crate::fluid::grid::{CellType, GridDims};

use super::layout::{buffer_entry, entry};

/// Uniform parameter block. Layout matches `Params` in
/// `shaders/fluid_pressure.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `x`/`y`/`z` = cell counts, `w` = cell count.
    dims: [u32; 4],
    /// `x`/`y`/`z` = concatenated face bases, `w` = total face count.
    bases: [u32; 4],
    /// `x` = right-hand-side scale, `y` = over-relaxation, `z` = subtraction
    /// scale, `w` = cell size.
    coeff: [f32; 4],
}

/// A compiled, reusable `GPU` pressure-projection pipeline set.
pub struct GpuPressureSolver {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    sor_red: ComputePipeline,
    sor_black: ComputePipeline,
    subtract_u: ComputePipeline,
    subtract_v: ComputePipeline,
    subtract_w: ComputePipeline,
}

impl GpuPressureSolver {
    /// Compiles the pressure-projection kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPressureSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fluid_pressure"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fluid_pressure.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fluid_pressure_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fluid_pressure_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sor_red = make(
            device,
            &module,
            "sor_red",
            "prism_fluid_sor_red",
            &pipeline_layout,
        );
        let sor_black = make(
            device,
            &module,
            "sor_black",
            "prism_fluid_sor_black",
            &pipeline_layout,
        );
        let subtract_u = make(
            device,
            &module,
            "subtract_u",
            "prism_fluid_subtract_u",
            &pipeline_layout,
        );
        let subtract_v = make(
            device,
            &module,
            "subtract_v",
            "prism_fluid_subtract_v",
            &pipeline_layout,
        );
        let subtract_w = make(
            device,
            &module,
            "subtract_w",
            "prism_fluid_subtract_w",
            &pipeline_layout,
        );
        GpuPressureSolver {
            module,
            layout,
            sor_red,
            sor_black,
            subtract_u,
            subtract_v,
            subtract_w,
        }
    }

    /// Projects the concatenated `[u | v | w]` velocity field in place on
    /// device, returning the row-major pressure field.
    ///
    /// `cell_types` is the row-major cell classification (length
    /// [`GridDims::cell_count`]); `velocity` is the concatenated face array
    /// (length [`GridDims::face_total`]).
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::EmptyGrid`] when `dims` has a zero axis and
    /// [`FluidError::InvalidConfig`] when a slice length does not match `dims`.
    pub fn project(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        cell_types: &[CellType],
        velocity: &mut [f32],
        cfg: PressureConfig,
    ) -> Result<Vec<f32>, FluidError> {
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

        let plan = self.upload(ctx, dims, cell_types, velocity, cfg);
        let (velocity_stage, pressure_stage) = self.encode_and_run(ctx, &plan, cfg.iterations);

        let projected = buffer::read_back::<f32>(ctx, &velocity_stage);
        velocity.copy_from_slice(&projected[..velocity.len()]);
        let pressure = buffer::read_back::<f32>(ctx, &pressure_stage);
        Ok(pressure[..dims.cell_count()].to_vec())
    }

    /// Uploads every buffer and builds the bind group for one solve.
    fn upload(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        cell_types: &[CellType],
        velocity: &[f32],
        cfg: PressureConfig,
    ) -> PressurePlan {
        let device = ctx.device();
        let cell_count = dims.cell_count();
        let face_total = dims.face_total();
        let rhs_scale = cfg.density * dims.dx * dims.dx / cfg.dt;
        let omega = cfg.effective_over_relaxation();
        let sub_scale = cfg.dt / (cfg.density * dims.dx);

        let params = Params {
            dims: [dims.nx, dims.ny, dims.nz, cell_count as u32],
            bases: [
                dims.u_offset() as u32,
                dims.v_offset() as u32,
                dims.w_offset() as u32,
                face_total as u32,
            ],
            coeff: [rhs_scale, omega, sub_scale, dims.dx],
        };

        let codes: Vec<u32> = cell_types.iter().map(|t| t.code()).collect();
        let pressure_init = vec![0.0f32; cell_count];

        let params_buf = buffer::uniform(device, "fluid_pressure_params", &params);
        let cell_buf = buffer::storage_read(device, "fluid_pressure_cells", &codes);
        let velocity_buf = buffer::storage_rw_init(device, "fluid_pressure_velocity", velocity);
        let pressure_buf =
            buffer::storage_rw_init(device, "fluid_pressure_pressure", &pressure_init);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_pressure_bind_group"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &cell_buf),
                entry(2, &velocity_buf),
                entry(3, &pressure_buf),
            ],
        });

        PressurePlan {
            bind,
            velocity_buf,
            pressure_buf,
            velocity_bytes: (face_total * 4) as u64,
            pressure_bytes: (cell_count * 4) as u64,
            cell_count: cell_count as u32,
            u_count: dims.u_count() as u32,
            v_count: dims.v_count() as u32,
            w_count: dims.w_count() as u32,
        }
    }

    /// Records the iterated red-black sweeps and the gradient subtraction into
    /// one encoder, submits, and returns the velocity and pressure staging
    /// buffers.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &PressurePlan,
        iterations: u32,
    ) -> (Buffer, Buffer) {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fluid_pressure_encoder"),
        });
        let cell_groups = plan.cell_count.div_ceil(64).max(1);
        for _ in 0..iterations {
            self.pass(&mut encoder, &self.sor_red, plan, cell_groups);
            self.pass(&mut encoder, &self.sor_black, plan, cell_groups);
        }
        self.pass(
            &mut encoder,
            &self.subtract_u,
            plan,
            plan.u_count.div_ceil(64).max(1),
        );
        self.pass(
            &mut encoder,
            &self.subtract_v,
            plan,
            plan.v_count.div_ceil(64).max(1),
        );
        self.pass(
            &mut encoder,
            &self.subtract_w,
            plan,
            plan.w_count.div_ceil(64).max(1),
        );

        let velocity_stage =
            buffer::staging(device, "fluid_pressure_velocity_stage", plan.velocity_bytes);
        buffer::copy(
            &mut encoder,
            &plan.velocity_buf,
            &velocity_stage,
            plan.velocity_bytes,
        );
        let pressure_stage =
            buffer::staging(device, "fluid_pressure_pressure_stage", plan.pressure_bytes);
        buffer::copy(
            &mut encoder,
            &plan.pressure_buf,
            &pressure_stage,
            plan.pressure_bytes,
        );

        ctx.queue().submit([encoder.finish()]);
        (velocity_stage, pressure_stage)
    }

    /// Records one compute pass dispatching `groups` workgroups.
    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ComputePipeline,
        plan: &PressurePlan,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_fluid_pressure_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &plan.bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// The uploaded buffers and dispatch dimensions for one solve.
struct PressurePlan {
    bind: BindGroup,
    velocity_buf: Buffer,
    pressure_buf: Buffer,
    velocity_bytes: u64,
    pressure_bytes: u64,
    cell_count: u32,
    u_count: u32,
    v_count: u32,
    w_count: u32,
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
