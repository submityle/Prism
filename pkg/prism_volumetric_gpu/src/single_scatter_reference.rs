//! `wgpu` compute twin of the Monte-Carlo single-scatter radiance oracle
//! ([`single_scatter_reference`](prism_render_architecture::volumetric::reference::single_scatter_reference))
//! for a **homogeneous** medium with a **constant** phase value.
//!
//! The `CPU` golden takes an arbitrary phase closure `phase_fn` so anisotropic
//! phases can be probed along the ray. Its on-device twin restricts that
//! closure to the constant `phase`, which is exactly the regime the parity
//! test drives the golden with, so the two sides run the identical estimator.
//! Each of `samples` walks re-seeds a deterministic Weyl+hash `RNG` from
//! `sample_seed(seed, i)`, draws one free-flight scatter distance
//! `t = -ln(1 - u) / sigma_t` from the extinction `pdf`, and — when `t` lands
//! inside the segment — contributes `sigma_s * phase * light_radiance /
//! sigma_t` (the transmittance and `pdf` cancel). The mean over the walks is an
//! unbiased single-scatter radiance that converges to
//! [`analytic_single_scatter`](prism_render_architecture::volumetric::reference::analytic_single_scatter)
//! for a constant phase. [`GpuSingleScatterReference`] runs one thread per
//! query and returns that mean.
//!
//! # Correctness model
//!
//! The `RNG`, `ln_approx` and arithmetic mirror the golden exactly: `WGSL`
//! `u32` wraps like `wrapping_*`, and `ln_approx` is the same
//! bit-reconstruction + `atanh` series. Floating-point division is
//! spec-allowed to differ by a few `ULP` across devices, so the parity test
//! uses an absolute tolerance rather than bit equality. The test also checks
//! non-negativity, the zero-radiance identity when any factor or the distance
//! is non-positive, and convergence to the analytic single-scatter value as
//! `samples` grows.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — no optional device
//! feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard free-flight importance-sampled single-scatter
//! estimator; no Unreal Engine source or derived code.
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

/// One single-scatter query: the homogeneous medium coefficients, the constant
/// phase value, the incident light radiance, the path distance and the
/// deterministic seed / sample budget for the estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleScatterReferenceQuery {
    /// The medium extinction coefficient (floored at `EPS` on device).
    pub sigma_t: f32,
    /// The medium scattering coefficient (clamped to `>= 0`).
    pub sigma_s: f32,
    /// The incident light radiance (clamped to `>= 0`).
    pub light_radiance: f32,
    /// The constant phase-function value toward the eye (clamped to `>= 0`).
    pub phase: f32,
    /// The path distance through the medium (clamped to `>= 0`).
    pub distance: f32,
    /// The base `RNG` seed; each sample decorrelates from it.
    pub seed: u32,
    /// The number of independent free-flight walks to average.
    pub samples: u32,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/single_scatter_reference.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sigma_t: f32,
    sigma_s: f32,
    light_radiance: f32,
    phase: f32,
    distance: f32,
    seed: u32,
    samples: u32,
    pad0: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/single_scatter_reference.wesl`: the query count plus three pad
/// words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable single-scatter reference pipeline.
pub struct GpuSingleScatterReference {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSingleScatterReference {
    /// Compiles the single-scatter reference kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSingleScatterReference {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_shader"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/single_scatter_reference.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("single_scatter_reference_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSingleScatterReference {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one radiance per query in
    /// input order.
    ///
    /// The returned value for query `q` equals
    /// [`single_scatter_reference`](prism_render_architecture::volumetric::reference::single_scatter_reference)`(q.sigma_t, q.sigma_s, q.light_radiance, |_| q.phase, q.distance, q.seed, q.samples)`
    /// up to floating-point division tolerance. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SingleScatterReferenceQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sigma_t: q.sigma_t,
                sigma_s: q.sigma_s,
                light_radiance: q.light_radiance,
                phase: q.phase,
                distance: q.distance,
                seed: q.seed,
                samples: q.samples,
                pad0: 0,
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
            label: Some("prism_volumetric_single_scatter_reference_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_single_scatter_reference_bind_group"),
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
            label: Some("prism_volumetric_single_scatter_reference_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_single_scatter_reference_pass"),
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
