//! Real-device `wgpu` implementation of the free-surface velocity
//! extrapolation that runs after the pressure projection.
//!
//! [`GpuExtrapolate`] compiles `shaders/fluid_extrapolate.wgsl` once and exposes
//! [`GpuExtrapolate::extrapolate_axis`], which grows the known velocity band
//! outward by one nearest-neighbour-averaging Jacobi sweep per compute
//! dispatch. Because each sweep reads a snapshot and writes fresh state, the
//! solver ping-pongs two `(field, known)` buffer pairs: sweep `s` reads pair
//! `A` and writes pair `B` when `s` is even and the reverse when odd, so the
//! final field lands in `A` after an even sweep count and `B` after an odd one.
//!
//! This reproduces the Jacobi structure of the [`super::super::cpu::extrapolate`]
//! golden twin; only the single division per filled face is floating point, so
//! the real-device parity test checks the fields within a tight tolerance.
//!
//! # Provenance
//!
//! Iterative velocity extrapolation from the known band into the air region is
//! a standard free-surface technique (Bridson, *Fluid Simulation for Computer
//! Graphics*; Zhu and Bridson 2005). No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::fluid::config::FluidError;
use crate::fluid::cpu::extrapolate::AxisDims;

use super::layout::{buffer_entry, entry};

/// Uniform parameter block. Layout matches `Params` in
/// `shaders/fluid_extrapolate.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// `x`/`y`/`z` = axis sample counts, `w` = total sample count.
    dims: [u32; 4],
}

/// A compiled, reusable `GPU` extrapolation pipeline.
pub struct GpuExtrapolate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    sweep: ComputePipeline,
}

impl GpuExtrapolate {
    /// Compiles the extrapolation kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuExtrapolate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fluid_extrapolate"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fluid_extrapolate.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fluid_extrapolate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fluid_extrapolate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let sweep = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_fluid_extrapolate_sweep"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sweep"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuExtrapolate {
            module,
            layout,
            sweep,
        }
    }

    /// Extrapolates one staggered face field in place over `iterations` sweeps.
    ///
    /// `weights` seeds the known band (positive weight means known); the result
    /// is read back into `field`.
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::InvalidConfig`] when `field` or `weights` does not
    /// match [`AxisDims::count`].
    pub fn extrapolate_axis(
        &self,
        ctx: &GpuContext,
        dims: AxisDims,
        field: &mut [f32],
        weights: &[f32],
        iterations: u32,
    ) -> Result<(), FluidError> {
        let count = dims.count();
        if field.len() != count {
            return Err(FluidError::InvalidConfig(
                "field length must equal the axis sample count",
            ));
        }
        if weights.len() != count {
            return Err(FluidError::InvalidConfig(
                "weights length must equal the axis sample count",
            ));
        }
        if count == 0 {
            return Ok(());
        }

        let plan = self.upload(ctx, dims, field, weights);
        let stage = self.encode_and_run(ctx, &plan, iterations);
        let result = buffer::read_back::<f32>(ctx, &stage);
        field.copy_from_slice(&result[..field.len()]);
        Ok(())
    }

    /// Uploads the ping-pong buffers and builds the two alternating bind groups.
    fn upload(&self, ctx: &GpuContext, dims: AxisDims, field: &[f32], weights: &[f32]) -> Plan {
        let device = ctx.device();
        let count = dims.count();
        let params = Params {
            dims: [dims.dx, dims.dy, dims.dz, count as u32],
        };
        let known: Vec<u32> = weights.iter().map(|&w| u32::from(w > 0.0)).collect();
        let bytes_f = (count * 4) as u64;

        let params_buf = buffer::uniform(device, "fluid_extrap_params", &params);
        let field_a = buffer::storage_rw_init(device, "fluid_extrap_field_a", field);
        let field_b = buffer::storage_rw_zeroed(device, "fluid_extrap_field_b", bytes_f);
        let known_a = buffer::storage_rw_init(device, "fluid_extrap_known_a", &known);
        let known_b = buffer::storage_rw_zeroed(device, "fluid_extrap_known_b", bytes_f);

        let bind_ab = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_extrap_bind_ab"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &field_a),
                entry(2, &known_a),
                entry(3, &field_b),
                entry(4, &known_b),
            ],
        });
        let bind_ba = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_extrap_bind_ba"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &field_b),
                entry(2, &known_b),
                entry(3, &field_a),
                entry(4, &known_a),
            ],
        });

        Plan {
            bind_ab,
            bind_ba,
            field_a,
            field_b,
            bytes_f,
            groups: (count as u32).div_ceil(64).max(1),
        }
    }

    /// Records `iterations` alternating sweeps, copies the buffer holding the
    /// final field to a staging buffer, submits, and returns the staging buffer.
    fn encode_and_run(&self, ctx: &GpuContext, plan: &Plan, iterations: u32) -> Buffer {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fluid_extrap_encoder"),
        });
        for s in 0..iterations {
            let bind = if s.is_multiple_of(2) {
                &plan.bind_ab
            } else {
                &plan.bind_ba
            };
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_fluid_extrap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sweep);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(plan.groups, 1, 1);
            drop(pass);
        }
        // After an even sweep count the final field is in A, otherwise in B.
        let final_field = if iterations.is_multiple_of(2) {
            &plan.field_a
        } else {
            &plan.field_b
        };
        let stage = buffer::staging(device, "fluid_extrap_stage", plan.bytes_f);
        buffer::copy(&mut encoder, final_field, &stage, plan.bytes_f);
        ctx.queue().submit([encoder.finish()]);
        stage
    }
}

/// The uploaded ping-pong buffers, bind groups, and dispatch size for one run.
struct Plan {
    bind_ab: BindGroup,
    bind_ba: BindGroup,
    field_a: Buffer,
    field_b: Buffer,
    bytes_f: u64,
    groups: u32,
}
