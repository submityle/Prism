//! `wgpu` compute twin of Prism's forward-scatter crossing accumulation
//! ([`accumulate_forward_scatter`](prism_render_architecture::hair::dual_scattering::accumulate_forward_scatter)).
//!
//! Dual scattering needs the coverage-weighted count `n` of strands crossed in
//! front of a receiver so the shading side can raise it into `a_f^n` and
//! `n * beta_f^2`. The golden, per light ray/texel, sums each sample's opacity
//! (clamped to `0..=1`) over every sample whose depth is at or in front of the
//! receiver depth. The reduction is ordering-independent and never panics; an
//! empty ray yields `0`. With an infinitely deep receiver it collapses to
//! [`total_crossings`](prism_render_architecture::hair::dual_scattering::total_crossings),
//! the saturation count for a fully buried receiver, which this twin also
//! exercises.
//!
//! This is the additive counterpart of the [`deep_opacity`](crate::deep_opacity)
//! transmittance product: deep opacity multiplies `1 - alpha` into a surviving
//! fraction, whereas this twin sums coverage into a crossing count. Both batch a
//! ragged set of per-ray samples via the shared flatten-plus-ranges layout.
//!
//! # What the kernel evaluates
//!
//! [`GpuForwardScatterAccumulate::eval`] takes a batch of rays (each its strand
//! samples and its receiver depth) and returns one `f32` crossing count per ray
//! in input order. The host concatenates every ray's `(depth, opacity)` pairs
//! into one pool and uploads each ray's `(sample_start, sample_count)` run plus
//! its receiver depth; the kernel owns the per-ray depth-gated clamped sum. The
//! sample fields are host passthrough: the kernel reproduces the golden's
//! defensive `opacity.clamp(0, 1)` and compares the raw depth, so a hand-built
//! sample with a stray out-of-range value is sanitised identically to the
//! reference.
//!
//! # Portability
//!
//! The kernel uses only compare, `clamp` and add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The reduction is a closed-form clamped sum with no transcendental call, so
//! `CPU` and `GPU` evaluate the same arithmetic. They are **not** bit-exact: a
//! `GPU` may reassociate the running sum or fuse a multiply-add, perturbing the
//! low mantissa bits by a few `ULP`. The parity test asserts a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) per crossing count.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard coverage-weighted depth-gated crossing sum plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::deep_transmittance::TransmittanceSample;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One ray's inputs to the forward-scatter accumulation: the strand `samples`
/// on that light ray/texel and the `receiver_depth` that gates the sum.
///
/// Mirrors the golden arguments of
/// [`accumulate_forward_scatter`](prism_render_architecture::hair::dual_scattering::accumulate_forward_scatter):
/// a sample contributes its clamped opacity iff its depth is at or in front of
/// `receiver_depth`. Pass `f32::INFINITY` to recover
/// [`total_crossings`](prism_render_architecture::hair::dual_scattering::total_crossings).
#[derive(Clone, Copy, Debug)]
pub struct ForwardScatterRay<'a> {
    /// Strand samples on this ray; may arrive in any order (the sum is
    /// ordering-independent).
    pub samples: &'a [TransmittanceSample],
    /// Light-space receiver depth; samples at or in front of it are counted.
    pub receiver_depth: f32,
}

/// Uniform ray count uploaded to the kernel. A single `u32` padded to the
/// `16`-byte uniform block, matching `Params` in
/// `shaders/forward_scatter_accumulate.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    ray_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One ray's sample slice descriptor plus its receiver depth: `(sample_start,
/// sample_count)` index the shared `samples` pool. `16`-byte `repr(C)` matching
/// the shader's `Ray` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRay {
    sample_start: u32,
    sample_count: u32,
    receiver_depth: f32,
    pad: u32,
}

/// A compiled, reusable forward-scatter accumulation pipeline.
pub struct GpuForwardScatterAccumulate {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuForwardScatterAccumulate {
    /// Compiles the accumulation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuForwardScatterAccumulate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/forward_scatter_accumulate.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuForwardScatterAccumulate {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates each ray's coverage-weighted crossing count, returning one
    /// `f32` per ray in input order.
    ///
    /// The result equals
    /// [`accumulate_forward_scatter`](prism_render_architecture::hair::dual_scattering::accumulate_forward_scatter)
    /// per ray to within the fused-multiply-add tolerance documented on this
    /// module. The host flattens every ray's samples into a shared pool (the
    /// kernel does no slicing); sample depth and opacity are host passthrough and
    /// the kernel reproduces the golden's defensive opacity clamp. Empty input,
    /// or a batch where every ray is empty, returns the all-zero counts without a
    /// dispatch (storage buffers cannot be zero-sized).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, rays: &[ForwardScatterRay<'_>]) -> Vec<f32> {
        let ray_count = rays.len();
        if ray_count == 0 {
            return Vec::new();
        }

        // Flatten every ray's `(depth, opacity)` pairs into one pool and record
        // each ray's slice plus its receiver depth. The sample fields are passed
        // through verbatim (the kernel clamps opacity, matching the golden).
        let mut samples: Vec<f32> = Vec::new();
        let mut gpu_rays: Vec<GpuRay> = Vec::with_capacity(ray_count);
        for ray in rays {
            let sample_start = (samples.len() / 2) as u32;
            for s in ray.samples {
                samples.push(s.depth);
                samples.push(s.opacity);
            }
            let sample_count = (samples.len() / 2) as u32 - sample_start;
            gpu_rays.push(GpuRay {
                sample_start,
                sample_count,
                receiver_depth: ray.receiver_depth,
                pad: 0,
            });
        }

        // Every ray empty -> no samples; mirror the golden's zero counts without
        // touching the device (storage buffers cannot be zero-sized).
        if samples.is_empty() {
            return vec![0.0_f32; ray_count];
        }

        let uniform = Params {
            ray_count: ray_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let device = ctx.device();
        let out_bytes = (ray_count * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_samples"),
            contents: bytemuck::cast_slice(&samples),
            usage: BufferUsages::STORAGE,
        });
        let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_rays"),
            contents: bytemuck::cast_slice(&gpu_rays),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: rays_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_forward_scatter_accumulate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_forward_scatter_accumulate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (ray_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        read_f32(&out_stage)
    }
}

/// Reads a mapped staging buffer back into an owned `f32` vector, then unmaps.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
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
