//! `wgpu` compute twin of the temporal variance box clip
//! ([`variance_clip`](prism_render_architecture::volumetric::temporal::variance_clip)).
//!
//! The temporal upsampler (design section 10) rectifies a reprojected history
//! sample against the current-frame statistics to reject ghosting while keeping
//! more history than a hard neighbourhood clamp:
//!
//! ```text
//! sd     = max(std_dev, 0)
//! g      = max(gamma, 0)
//! half   = sd * g
//! result = clamp(history_sample, mean - half, mean + half)
//! ```
//!
//! `std_dev` and `gamma` are floored at zero, so the clip window is always
//! well-formed (its lower bound never exceeds its upper bound) and the result
//! lies inside `[mean - half, mean + half]`. The `CPU` golden
//! [`variance_clip`](prism_render_architecture::volumetric::temporal::variance_clip)
//! owns that math; [`GpuVarianceClip`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The `clamp` is expanded to the *same* branch form the CPU `math::clamp` uses
//! (below the low bound → low, above the high bound → high, else pass through),
//! so `CPU` and `GPU` evaluate the same closed-form algebra. The only slack is a
//! legal multiply-add contraction of a few `ULP`, so the parity test asserts a
//! tight tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also
//! assert every result lands inside the (floored) clip window, so a degenerate
//! kernel could not pass.
//!
//! # Portability
//!
//! The kernel is comparisons and multiply/add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard TAA variance box clip plus `wgpu` compute dispatch; no
//! Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One variance-clip query: the reprojected history sample plus the current
/// frame's neighbourhood mean, standard deviation and the gamma window scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarianceClipQuery {
    /// The reprojected history sample to rectify.
    pub history_sample: f32,
    /// The current-frame neighbourhood mean.
    pub mean: f32,
    /// The current-frame neighbourhood standard deviation (floored at zero).
    pub std_dev: f32,
    /// The clip-window scale in standard deviations (floored at zero).
    pub gamma: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/variance_clip.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    history_sample: f32,
    mean: f32,
    std_dev: f32,
    gamma: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/variance_clip.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable variance-clip pipeline.
pub struct GpuVarianceClip {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVarianceClip {
    /// Compiles the variance-clip kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVarianceClip {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_variance_clip"),
            source: ShaderSource::Wgsl(include_str!("../shaders/variance_clip.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_variance_clip_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_variance_clip_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_variance_clip_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("variance_clip_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVarianceClip {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the variance clip for every query in `queries`, returning one
    /// value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`variance_clip`](prism_render_architecture::volumetric::temporal::variance_clip)`(q.history_sample, q.mean, q.std_dev, q.gamma)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[VarianceClipQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                history_sample: q.history_sample,
                mean: q.mean,
                std_dev: q.std_dev,
                gamma: q.gamma,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_variance_clip_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_variance_clip_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_variance_clip_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_variance_clip_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_variance_clip_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_variance_clip_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_variance_clip_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), queries.len());
        values
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
