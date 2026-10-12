//! The real-device `wgpu` pipeline that fills an indirect-dispatch argument
//! block from a device-side count.
//!
//! [`GpuDispatchArgs`] compiles `shaders/indirect_args.wgsl` once and offers two
//! surfaces:
//!
//! - [`GpuDispatchArgs::record`] appends the one-invocation args build to a
//!   caller's command encoder. This is the production path: it fuses into the
//!   same submission as the stage that produced the count and the stage that
//!   consumes the args, so a whole chain of kernels runs without the host ever
//!   reading a count back. The written buffer carries [`wgpu::BufferUsages::INDIRECT`],
//!   so it binds straight into `dispatch_workgroups_indirect`.
//! - [`GpuDispatchArgs::compute`] runs the build on its own and reads the block
//!   back. It exists so a parity test can compare the device result against the
//!   [`cpu_dispatch_args`](super::args::cpu_dispatch_args) golden twin; a
//!   GPU-driven pipeline never needs this readback.
//!
//! # Provenance
//!
//! Standard `wgpu` compute-pipeline construction and indirect dispatch. No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

use super::args::DispatchArgs;

/// Bytes in a [`DispatchArgs`] block: four `u32` words.
const ARGS_BYTES: u64 = 16;

/// A compiled, reusable pipeline that turns a device-side count into a
/// [`DispatchArgs`] indirect-dispatch block.
pub struct GpuDispatchArgs {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// Binds the params uniform, the read-only counts, and the writable args.
    layout: BindGroupLayout,
    /// The single-invocation args-building kernel.
    build: ComputePipeline,
}

/// Uniform parameters shared with `Params` in `shaders/indirect_args.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Lanes per workgroup the consumer stage will launch.
    workgroup_size: u32,
    /// Index of the count word to read from the counts buffer.
    count_offset: u32,
    /// Per-dimension workgroup cap before the grid tiles into `y`.
    max_dim: u32,
    /// Padding to a 16-byte uniform boundary.
    pad: u32,
}

impl GpuDispatchArgs {
    /// Compiles the args-building kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDispatchArgs {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_indirect_args"),
            source: ShaderSource::Wgsl(include_str!("../shaders/indirect_args.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_indirect_args_layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_indirect_args_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_indirect_args_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("build_args"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDispatchArgs {
            module,
            layout,
            build,
        }
    }

    /// Allocates an indirect-dispatch argument buffer sized for one
    /// [`DispatchArgs`] block.
    ///
    /// The buffer carries [`BufferUsages::INDIRECT`] so it binds directly into
    /// `dispatch_workgroups_indirect`, plus `STORAGE` so this kernel can write
    /// it and a consumer can read its published row width, and `COPY_SRC` so a
    /// parity test can read it back.
    #[must_use]
    pub fn args_buffer(device: &Device, label: &str) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: ARGS_BYTES,
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The device's per-dimension workgroup cap, the natural `max_dim` for a
    /// launch on `ctx`.
    #[must_use]
    pub fn device_max_dim(ctx: &GpuContext) -> u32 {
        ctx.device().limits().max_compute_workgroups_per_dimension
    }

    /// Appends the args build to `encoder`, reading `counts[count_offset]` and
    /// writing the grid for `workgroup_size`-wide groups into `args`.
    ///
    /// This records into the caller's encoder so the build fuses with the
    /// surrounding pipeline: the stage that wrote `counts`, this build, and the
    /// stage that will `dispatch_workgroups_indirect(args, 0)` all submit
    /// together with no intervening host round-trip. `args` must have been
    /// created by [`GpuDispatchArgs::args_buffer`].
    pub fn record(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        counts: &Buffer,
        count_offset: u32,
        workgroup_size: u32,
        max_dim: u32,
        args: &Buffer,
    ) {
        let params = Params {
            workgroup_size,
            count_offset,
            max_dim: max_dim.max(1),
            pad: 0,
        };
        let params_buf = buffer::uniform(device, "prism_indirect_args_params", &params);
        let bind = self.bind(device, &params_buf, counts, args);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_indirect_args_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.build);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Builds and reads back the grid for `counts[count_offset]`.
    ///
    /// A convenience for parity testing against
    /// [`cpu_dispatch_args`](super::args::cpu_dispatch_args); a GPU-driven
    /// pipeline uses [`GpuDispatchArgs::record`] and never pays this readback.
    #[must_use]
    pub fn compute(
        &self,
        ctx: &GpuContext,
        counts: &[u32],
        count_offset: u32,
        workgroup_size: u32,
        max_dim: u32,
    ) -> DispatchArgs {
        let device = ctx.device();
        let counts_buf = buffer::storage_read(device, "prism_indirect_args_counts", counts);
        let args = GpuDispatchArgs::args_buffer(device, "prism_indirect_args_out");
        let stage = buffer::staging(device, "prism_indirect_args_stage", ARGS_BYTES);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_indirect_args_encoder"),
        });
        self.record(
            device,
            &mut encoder,
            &counts_buf,
            count_offset,
            workgroup_size,
            max_dim,
            &args,
        );
        buffer::copy(&mut encoder, &args, &stage, ARGS_BYTES);
        ctx.queue().submit([encoder.finish()]);

        let words = buffer::read_back::<u32>(ctx, &stage);
        DispatchArgs {
            x: words[0],
            y: words[1],
            z: words[2],
            groups_x: words[3],
        }
    }

    /// Wires the params uniform, the counts input, and the args output into a
    /// bind group for the build kernel.
    fn bind(&self, device: &Device, params: &Buffer, counts: &Buffer, args: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_indirect_args_bind"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: counts.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: args.as_entire_binding(),
                },
            ],
        })
    }
}

/// A compute-visible uniform-buffer binding layout entry for `binding`.
fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compute-visible storage-buffer binding layout entry for `binding`.
fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
