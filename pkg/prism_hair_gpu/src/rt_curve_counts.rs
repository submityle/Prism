//! `wgpu` compute twin of Prism's per-strand `Linear Swept Spheres` segment
//! count
//! ([`lss_segment_counts`](prism_render_architecture::hair::rt_curve::lss_segment_counts)).
//!
//! Before a ray-tracing acceleration structure builds its hair curve `BLAS`, the
//! builder needs each strand's segment contribution so it can bucket primitives
//! and size its budget. This twin computes those per-strand counts batch-wide on
//! the device, one thread per strand.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRtCurveCounts::eval`] takes a batch of strands and returns, in strand
//! order, the number of segments each contributes: a strand of `n` vertices
//! yields `max(0, n - 1)` segments. The strand index is the invocation id
//! (`@compute @workgroup_size(64)`, a one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the strand count early-return.
//!
//! # Correctness model
//!
//! The kernel reads only each strand's vertex count, so it is pure integer
//! arithmetic — no floating point, no rounding, no fma, no transcendental step.
//! The twin is therefore *bit-exact* with the scalar reference and the parity
//! test asserts integer equality. The guarded subtraction mirrors the golden's
//! `max(0, len - 1)` on `u32` (a strand of fewer than two vertices yields `0`).
//!
//! # Portability
//!
//! The kernel uses only `select` and integer subtraction in the portable
//! core-`WGSL` subset — no optional device feature — so the twin runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: hardware ray-traced curve / LSS strand primitive `BLAS` contract
//! (`RTX` `DXR` / `OptiX` LSS) plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::rt_curve::{lss_segment_counts, CurveVertex};

use crate::context::GpuContext;

/// Uniform parameters for one segment-count dispatch. Layout matches `Params`
/// in `shaders/rt_curve_counts.wesl`: the strand count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    strand_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-strand LSS segment-count pipeline.
pub struct GpuHairRtCurveCounts {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRtCurveCounts {
    /// Compiles the per-strand segment-count kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRtCurveCounts {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rt_curve_counts"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rt_curve_counts.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rt_curve_counts_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rt_curve_counts_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rt_curve_counts_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRtCurveCounts {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes the per-strand LSS segment count on the device.
    ///
    /// Returns, in strand order, `max(0, len - 1)` for each strand, matching the
    /// `CPU` golden
    /// [`lss_segment_counts`](prism_render_architecture::hair::rt_curve::lss_segment_counts)
    /// exactly (pure integer arithmetic — no fma, no rounding). An empty batch
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[&[CurveVertex]]) -> Vec<u32> {
        let strand_count = strands.len();
        if strand_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            strand_count: strand_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Only each strand's vertex count drives the kernel.
        let lengths: Vec<u32> = strands.iter().map(|s| s.len() as u32).collect();

        let out_bytes = (strand_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_counts_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let lengths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_counts_lengths"),
            contents: bytemuck::cast_slice(&lengths),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_counts_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_counts_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rt_curve_counts_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lengths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_rt_curve_counts_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rt_curve_counts_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strand_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden per-strand segment count, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_lss_segment_counts(strands: &[&[CurveVertex]]) -> Vec<usize> {
    lss_segment_counts(strands)
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
