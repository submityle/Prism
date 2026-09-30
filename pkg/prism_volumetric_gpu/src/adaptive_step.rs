//! `wgpu` compute twin of the raymarch adaptive-step kernel
//! ([`adaptive_step`](prism_render_architecture::volumetric::raymarch::adaptive_step)).
//!
//! Volumetric clouds march with a variable stride so empty space is skipped in
//! large jumps and dense medium is sampled finely (design section 6). Outside
//! the cloud (not `in_cloud`, or density below
//! [`RaymarchConfig::density_threshold`](prism_render_architecture::volumetric::raymarch::RaymarchConfig::density_threshold))
//! the largest stride `max_step` is returned to skip empty space. Inside the
//! cloud the step interpolates from `base_step` toward `min_step` as the
//! saturated density rises, so the step is monotone non-increasing in density,
//! and the result is always clamped into `[min_step, max_step]`.
//!
//! The `CPU` golden
//! [`adaptive_step`](prism_render_architecture::volumetric::raymarch::adaptive_step)
//! owns that logic; [`GpuAdaptiveStep`] is the on-device twin that runs one
//! thread per query.
//!
//! # Correctness model
//!
//! The math is pure `clamp`/`lerp`/`saturate` with no transcendental, so this
//! kernel mirrors the `CPU` branch-form `clamp` and `lerp` exactly; the parity
//! test asserts agreement within a tight floating-point tolerance. Clamping the
//! step into `[min_step, max_step]` and interpolating with a saturated weight
//! keeps the stride bounded and monotone non-increasing in density.
//!
//! # Portability
//!
//! The kernel is the portable core-`WGSL` subset — branch-form `clamp`, `lerp`
//! and comparisons, no optional device feature — so it runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard density-adaptive raymarch step selection plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::raymarch::RaymarchConfig;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One step query: the sample density, whether the sample is in-cloud and the
/// raymarch config whose bounds drive the stride.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveStepQuery {
    /// The density at the current sample.
    pub current_density: f32,
    /// Whether the sample is treated as inside the cloud.
    pub in_cloud: bool,
    /// The raymarch config whose `base_step` / `max_step` / `min_step` /
    /// `density_threshold` drive the stride.
    pub cfg: RaymarchConfig,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/adaptive_step.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    current_density: f32,
    in_cloud: u32,
    base_step: f32,
    max_step: f32,
    min_step: f32,
    density_threshold: f32,
    pad0: u32,
    pad1: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/adaptive_step.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable adaptive-step pipeline.
pub struct GpuAdaptiveStep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAdaptiveStep {
    /// Compiles the adaptive-step kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAdaptiveStep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_adaptive_step"),
            source: ShaderSource::Wgsl(include_str!("../shaders/adaptive_step.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_adaptive_step_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_adaptive_step_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_adaptive_step_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("adaptive_step_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAdaptiveStep {
            module,
            layout,
            pipeline,
        }
    }

    /// Chooses the stride for every query in `queries`, returning one step per
    /// query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`adaptive_step`](prism_render_architecture::volumetric::raymarch::adaptive_step)`(q.current_density, q.in_cloud, q.cfg)`
    /// within a tight floating-point tolerance. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AdaptiveStepQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                current_density: q.current_density,
                in_cloud: u32::from(q.in_cloud),
                base_step: q.cfg.base_step,
                max_step: q.cfg.max_step,
                min_step: q.cfg.min_step,
                density_threshold: q.cfg.density_threshold,
                pad0: 0,
                pad1: 0,
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
            label: Some("prism_volumetric_adaptive_step_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_adaptive_step_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_adaptive_step_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_adaptive_step_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_adaptive_step_bind_group"),
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
            label: Some("prism_volumetric_adaptive_step_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_adaptive_step_pass"),
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(out.len(), queries.len());
        out
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
