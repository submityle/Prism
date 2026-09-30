//! Real-device `wgpu` compute implementation of the `CFL` velocity reduction.
//!
//! [`GpuCflReduce`] compiles `shaders/cfl_reduce.wgsl` once and exposes
//! [`GpuCflReduce::max_speed`], which reduces a velocity field to its largest
//! speed on the `GPU`, and [`GpuCflReduce::suggest_dt`], which feeds that speed
//! through [`CflConfig::suggest_dt`]. The reduction matches the
//! [`cpu_max_speed`](super::cpu::cpu_max_speed) golden twin: the fold is an
//! order-independent maximum over squared speeds and only the final square root
//! carries floating-point rounding, so parity holds within a tight tolerance.
//!
//! Provenance: the `CFL` condition is a classical, openly published stability
//! criterion and shared-memory tree reduction is a standard `GPU` technique. No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::config::CflConfig;
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 256;

/// Uniform parameters shared with `Params` in `shaders/cfl_reduce.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of velocity entries.
    n: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

/// A compiled, reusable `GPU` `CFL` reduction pipeline.
pub struct GpuCflReduce {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, the velocity field, and the output.
    layout: BindGroupLayout,
    /// The reduction kernel: one invocation per velocity entry.
    reduce: ComputePipeline,
}

impl GpuCflReduce {
    /// Compiles the reduction kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCflReduce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cfl_reduce"),
            source: ShaderSource::Wgsl(include_str!("../shaders/cfl_reduce.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cfl_reduce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cfl_reduce_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let reduce = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cfl_reduce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reduce"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCflReduce {
            module,
            layout,
            reduce,
        }
    }

    /// Reduces `velocities` to the largest speed present, or `0.0` for an empty
    /// field.
    ///
    /// The kernel folds the squared speeds with an order-independent maximum and
    /// this method takes the single closing square root, matching the
    /// [`cpu_max_speed`](super::cpu::cpu_max_speed) twin within a tight
    /// tolerance.
    #[must_use]
    pub fn max_speed(&self, ctx: &GpuContext, velocities: &[Vec3]) -> f32 {
        if velocities.is_empty() {
            return 0.0;
        }

        let device = ctx.device();
        let n = velocities.len();

        let params = Params {
            n: u32::try_from(n).unwrap_or(u32::MAX),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = buffer::uniform(device, "prism_cfl_reduce_params", &params);

        let packed: Vec<[f32; 4]> = velocities.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        let vel_buf = buffer::storage_read(device, "prism_cfl_reduce_vel", &packed);

        // A single u32 slot, zero-initialised: bit pattern 0 is +0.0, the
        // identity for a maximum of non-negative squared speeds.
        let out_buf =
            buffer::storage_rw_zeroed(device, "prism_cfl_reduce_out", size_of::<u32>() as u64);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cfl_reduce_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &vel_buf),
                entry(2, &out_buf),
            ],
        });

        let out_stage = buffer::staging(
            device,
            "prism_cfl_reduce_out_stage",
            size_of::<u32>() as u64,
        );

        let groups = u32::try_from(n.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cfl_reduce_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cfl_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reduce);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &out_buf, &out_stage, size_of::<u32>() as u64);
        ctx.queue().submit([encoder.finish()]);

        let bits = buffer::read_back::<u32>(ctx, &out_stage);
        let max_sq = f32::from_bits(bits[0]);
        max_sq.max(0.0).sqrt()
    }

    /// Suggests a `CFL`-stable time step for the scene described by
    /// `velocities`, reducing on the `GPU` and clamping through `config`.
    #[must_use]
    pub fn suggest_dt(&self, ctx: &GpuContext, velocities: &[Vec3], config: &CflConfig) -> f32 {
        config.suggest_dt(self.max_speed(ctx, velocities))
    }
}
