//! `wgpu` compute twin of the raymarch early-termination predicate
//! ([`should_early_terminate`](prism_render_architecture::volumetric::raymarch::should_early_terminate)).
//!
//! As a volumetric ray marches, its accumulated `transmittance` decays
//! monotonically toward zero; once it falls below the config cutoff the
//! remaining medium contributes negligibly and the walk stops early to save
//! work (design section 6). This kernel evaluates the predicate per query:
//! `transmittance < transmittance_cutoff`.
//!
//! The `CPU` golden
//! [`should_early_terminate`](prism_render_architecture::volumetric::raymarch::should_early_terminate)
//! owns that logic; [`GpuShouldEarlyTerminate`] is the on-device twin that runs
//! one thread per query.
//!
//! # Correctness model
//!
//! The predicate is a single float comparison, so the decision is exact and the
//! parity test asserts every decision matches the `CPU` golden bit for bit,
//! including the boundary where `transmittance == transmittance_cutoff` (strict
//! `<`, so the boundary does not terminate).
//!
//! # Portability
//!
//! The kernel is a single comparison in the portable core-`WGSL` subset — no
//! `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard transmittance-cutoff early termination plus `wgpu`
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

/// One termination query: the accumulated transmittance and the raymarch config
/// whose cutoff decides when to stop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShouldEarlyTerminateQuery {
    /// The current accumulated transmittance (fraction surviving to the eye).
    pub transmittance: f32,
    /// The raymarch config whose `transmittance_cutoff` is the stop threshold.
    pub cfg: RaymarchConfig,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/should_early_terminate.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    transmittance: f32,
    transmittance_cutoff: f32,
    pad0: u32,
    pad1: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/should_early_terminate.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable early-termination pipeline.
pub struct GpuShouldEarlyTerminate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuShouldEarlyTerminate {
    /// Compiles the early-termination kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuShouldEarlyTerminate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_should_early_terminate"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/should_early_terminate.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_should_early_terminate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_should_early_terminate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_should_early_terminate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("should_early_terminate_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuShouldEarlyTerminate {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the termination predicate for every query in `queries`,
    /// returning one boolean per query in input order (`true` = stop marching).
    ///
    /// The returned decision for query `q` equals
    /// [`should_early_terminate`](prism_render_architecture::volumetric::raymarch::should_early_terminate)`(q.transmittance, q.cfg)`
    /// exactly. An empty `queries` slice yields an empty result — storage
    /// buffers cannot be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ShouldEarlyTerminateQuery]) -> Vec<bool> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                transmittance: q.transmittance,
                transmittance_cutoff: q.cfg.transmittance_cutoff,
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

        let out_bytes = (queries.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_should_early_terminate_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_should_early_terminate_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_should_early_terminate_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_should_early_terminate_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_should_early_terminate_bind_group"),
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
            label: Some("prism_volumetric_should_early_terminate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_should_early_terminate_pass"),
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
        let flags = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(flags.len(), queries.len());
        flags.into_iter().map(|f| f != 0).collect()
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
