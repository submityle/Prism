//! `wgpu` compute twin of the contrail diffusion kernel
//! ([`contrail_kernel`](prism_render_architecture::volumetric::fog::contrail_kernel)).
//!
//! The unified volumetric fog (design section 9f) injects the line-shaped
//! condensation trails aircraft leave at altitude as a Gaussian cross-section
//! that widens with age. [`contrail_kernel`] returns the unit-area kernel value
//! at a cross-section `offset`: with
//! `sigma = max(0.5 * max(width, 0) + max(diffusion, 0) * max(age, 0), EPS)`
//! the value is `1 / (sigma * sqrt(2*pi)) * e^{-0.5 * (offset/sigma)^2}`.
//! Because the analytic normalization constant is applied, integrating the
//! kernel across the full cross-section yields `1`, so injecting a contrail
//! conserves its total mass no matter how far it has diffused. The `CPU` golden
//! [`contrail_kernel`](prism_render_architecture::volumetric::fog::contrail_kernel)
//! owns that math; [`GpuContrailKernel`] is the on-device twin that runs one
//! thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The exponential is evaluated with the *same* hand-rolled `exp_approx` the
//! reference uses — base-two range reduction with a fractional seven-term
//! polynomial times an integer power assembled from the `f32` exponent field —
//! not the device-native `exp`. The normalization constant uses the native
//! `sqrt` both sides already share. Mirroring the polynomial keeps the twin
//! bit-close to the reference, so the parity test asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert the kernel
//! is non-negative, peaks at `offset = 0`, is symmetric in `offset`, and that a
//! freshly formed trail is narrower (taller peak) than an aged one, so a
//! degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops, `sqrt` and multiply/add in
//! the portable core-`WGSL` subset — no `exp`, `pow` or optional device
//! feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard normalized-Gaussian condensation kernel plus `wgpu`
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

/// One contrail-kernel query: the cross-section offset plus the trail's age,
/// authored width and diffusion coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrailKernelQuery {
    /// Cross-section offset from the trail centerline, in world units.
    pub offset: f32,
    /// Age in seconds since the trail formed; drives diffusion widening.
    pub age: f32,
    /// Authored base cross-section width in world units.
    pub width: f32,
    /// Diffusion coefficient scaling how fast the trail spreads with age.
    pub diffusion: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/contrail_kernel.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    offset: f32,
    age: f32,
    width: f32,
    diffusion: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/contrail_kernel.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable contrail-kernel pipeline.
pub struct GpuContrailKernel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuContrailKernel {
    /// Compiles the contrail-kernel shader on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuContrailKernel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_contrail_kernel"),
            source: ShaderSource::Wgsl(include_str!("../shaders/contrail_kernel.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_contrail_kernel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_contrail_kernel_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_contrail_kernel_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("contrail_kernel_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuContrailKernel {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the contrail kernel for every query in `queries`, returning
    /// one value per query in input order.
    ///
    /// The returned value for query `q` equals
    /// [`contrail_kernel`](prism_render_architecture::volumetric::fog::contrail_kernel)`(q.offset, Contrail { age: q.age, width: q.width, diffusion: q.diffusion })`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ContrailKernelQuery]) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                offset: q.offset,
                age: q.age,
                width: q.width,
                diffusion: q.diffusion,
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
            label: Some("prism_volumetric_contrail_kernel_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_contrail_kernel_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_contrail_kernel_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_contrail_kernel_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_contrail_kernel_bind_group"),
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
            label: Some("prism_volumetric_contrail_kernel_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_contrail_kernel_pass"),
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
