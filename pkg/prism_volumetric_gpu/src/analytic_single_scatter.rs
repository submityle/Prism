//! `wgpu` compute twin of the analytic single-scatter kernel
//! ([`analytic_single_scatter`](prism_render_architecture::volumetric::reference::analytic_single_scatter)).
//!
//! This is the closed-form single-scattered radiance of a homogeneous medium
//! lit by a constant light radiance — the reference value the delta / ratio
//! tracking estimators must converge to (design section 9c reference). It
//! evaluates
//! `sigma_s * phase * light_radiance * (1 - exp(-sigma_t * distance)) / sigma_t`
//! with `sigma_t` floored at `EPS` (so the path integral never divides by
//! zero), `distance` clamped to zero, and `sigma_s`, `phase` and
//! `light_radiance` clamped to `>= 0`, so the result is always non-negative.
//!
//! The `CPU` golden
//! [`analytic_single_scatter`](prism_render_architecture::volumetric::reference::analytic_single_scatter)
//! owns that logic; [`GpuAnalyticSingleScatter`] is the on-device twin that runs
//! one thread per query.
//!
//! # Correctness model
//!
//! The `CPU` golden uses a hand-rolled `exp_approx` (a polynomial, no float
//! intrinsic, for cross-target determinism); this kernel mirrors that same
//! polynomial `exp` so the on-device result matches bit-close, rather than
//! calling the `WGSL` `exp` builtin. The parity test asserts the radiance
//! within a tight tolerance and checks that it stays non-negative, grows
//! monotonically with distance and vanishes when any factor is zero.
//!
//! # Portability
//!
//! The kernel is arithmetic plus the polynomial `exp` in the portable
//! core-`WGSL` subset — no `exp` builtin or optional device feature — so it runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard radiative-transfer single-scatter integral plus `wgpu`
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

/// One single-scatter query: the medium coefficients, the phase-function value,
/// the incident light radiance and the path distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSingleScatterQuery {
    /// The medium extinction coefficient (floored at `EPS` on device).
    pub sigma_t: f32,
    /// The medium scattering coefficient (clamped to `>= 0`).
    pub sigma_s: f32,
    /// The phase-function value toward the eye (clamped to `>= 0`).
    pub phase: f32,
    /// The incident light radiance (clamped to `>= 0`).
    pub light_radiance: f32,
    /// The path distance through the medium (clamped to `>= 0`).
    pub distance: f32,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/analytic_single_scatter.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sigma_t: f32,
    sigma_s: f32,
    phase: f32,
    light_radiance: f32,
    distance: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/analytic_single_scatter.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable analytic single-scatter pipeline.
pub struct GpuAnalyticSingleScatter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnalyticSingleScatter {
    /// Compiles the analytic single-scatter kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnalyticSingleScatter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/analytic_single_scatter.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("analytic_single_scatter_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnalyticSingleScatter {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one single-scatter radiance
    /// per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`analytic_single_scatter`](prism_render_architecture::volumetric::reference::analytic_single_scatter)`(q.sigma_t, q.sigma_s, q.phase, q.light_radiance, q.distance)`
    /// within a tight floating-point tolerance. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AnalyticSingleScatterQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sigma_t: q.sigma_t,
                sigma_s: q.sigma_s,
                phase: q.phase,
                light_radiance: q.light_radiance,
                distance: q.distance,
                pad0: 0,
                pad1: 0,
                pad2: 0,
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
            label: Some("prism_volumetric_analytic_single_scatter_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_analytic_single_scatter_bind_group"),
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
            label: Some("prism_volumetric_analytic_single_scatter_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_analytic_single_scatter_pass"),
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
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());
        raw
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
