//! `wgpu` compute twin of the temporal history invalidation rule
//! ([`should_fallback`](prism_render_architecture::volumetric::temporal::should_fallback)).
//!
//! The temporal reprojection (design section 10) discards a reprojected history
//! sample — falling back to the freshly `raymarch`ed current frame — when the
//! history was flagged invalid (for example a reprojection that lands
//! off-screen) or when the parallax disocclusion measure exceeds a threshold:
//!
//! ```text
//! result = (!history_valid) || (disocclusion > max_disocclusion)
//! ```
//!
//! This deterministic invalidation rule stops disoccluded or high-parallax
//! regions from dragging stale cloud colour. The `CPU` golden
//! [`should_fallback`](prism_render_architecture::volumetric::temporal::should_fallback)
//! owns that logic; [`GpuShouldFallback`] is the on-device twin that runs one
//! thread per query and reproduces the same decision.
//!
//! # Correctness model
//!
//! The kernel evaluates the same boolean expression as the `CPU` (a comparison
//! and a logical or), so the decision is exact — there is no floating-point
//! slack to tolerate. The parity test asserts every decision matches bit for
//! bit, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is a comparison and a boolean or in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard TAA disocclusion / validity fallback rule plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
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

/// One fallback query: the history validity flag plus the parallax disocclusion
/// measure and its threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShouldFallbackQuery {
    /// Whether the reprojected history sample is valid (for example on-screen).
    pub history_valid: bool,
    /// The parallax disocclusion measure for this pixel.
    pub disocclusion: f32,
    /// The disocclusion threshold above which history is discarded.
    pub max_disocclusion: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/should_fallback.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    history_valid: u32,
    disocclusion: f32,
    max_disocclusion: f32,
    pad: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/should_fallback.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable should-fallback pipeline.
pub struct GpuShouldFallback {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuShouldFallback {
    /// Compiles the should-fallback kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuShouldFallback {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_should_fallback"),
            source: ShaderSource::Wgsl(include_str!("../shaders/should_fallback.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_should_fallback_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_should_fallback_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_should_fallback_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("should_fallback_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuShouldFallback {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the fallback decision for every query in `queries`, returning
    /// one boolean per query in input order (`true` = discard history and fall
    /// back to the current frame).
    ///
    /// The returned decision for query `q` equals
    /// [`should_fallback`](prism_render_architecture::volumetric::temporal::should_fallback)`(q.history_valid, q.disocclusion, q.max_disocclusion)`
    /// exactly. An empty `queries` slice yields an empty result — storage
    /// buffers cannot be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ShouldFallbackQuery]) -> Vec<bool> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                history_valid: u32::from(q.history_valid),
                disocclusion: q.disocclusion,
                max_disocclusion: q.max_disocclusion,
                pad: 0,
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
            label: Some("prism_volumetric_should_fallback_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_should_fallback_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_should_fallback_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_should_fallback_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_should_fallback_bind_group"),
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
            label: Some("prism_volumetric_should_fallback_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_should_fallback_pass"),
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
