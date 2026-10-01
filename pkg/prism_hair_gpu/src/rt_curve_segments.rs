//! `wgpu` compute twin of Prism's strand-polyline to `Linear Swept Spheres`
//! segment conversion
//! ([`strand_to_lss`](prism_render_architecture::hair::rt_curve::strand_to_lss)).
//!
//! Before a ray-tracing acceleration structure can register a hair strand as
//! curve / `Linear Swept Spheres` (LSS) primitives, each strand polyline must be
//! split into its per-segment endpoints and radii. This twin performs that split
//! batch-wide on the device, one thread per output segment.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRtCurveSegments::eval`] takes a strand of
//! [`CurveVertex`](prism_render_architecture::hair::rt_curve::CurveVertex) and
//! returns one [`LssSegment`] per adjacent vertex pair in input order: `n`
//! vertices yield `n - 1` segments, and fewer than two vertices yield none. The
//! segment index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! segment count early-return.
//!
//! # Correctness model
//!
//! The kernel does no arithmetic beyond the per-component sanitiser — it copies
//! each sanitised position and radius into an endpoint pair — so there is **no**
//! multiply-add the `GPU` could contract into an fma, and the twin is *bit-exact*
//! with the scalar reference: the parity test compares raw bit patterns, not a
//! tolerance. The sanitiser mirrors the golden's
//! [`CurveVertex::sanitized`](prism_render_architecture::hair::rt_curve::CurveVertex::sanitized)
//! bit-faithfully (`v == v && abs(v) < +inf` keeps finite coordinates and forces
//! the rest to `0`; a negative or non-finite radius collapses to `0`), so the
//! host uploads the raw authored vertices unchanged.
//!
//! # Portability
//!
//! The kernel uses only `select` and comparisons in the portable core-`WGSL`
//! subset — no `sqrt`, `exp`, `pow` or optional device feature — so the twin
//! runs unmodified on Metal, Vulkan and DX12.
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

use prism_render_architecture::hair::rt_curve::{strand_to_lss, CurveVertex, LssSegment};

use crate::context::GpuContext;

/// Uniform parameters for one strand-to-segments dispatch. Layout matches
/// `Params` in `shaders/rt_curve_segments.wesl`: the segment count padded to one
/// `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    segment_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable strand-polyline to LSS-segment pipeline.
pub struct GpuHairRtCurveSegments {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRtCurveSegments {
    /// Compiles the strand-to-segments kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRtCurveSegments {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rt_curve_segments"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rt_curve_segments.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rt_curve_segments_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rt_curve_segments_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rt_curve_segments_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRtCurveSegments {
            module,
            layout,
            pipeline,
        }
    }

    /// Splits one strand polyline into its LSS segments on the device.
    ///
    /// Returns one [`LssSegment`] per adjacent vertex pair, matching the `CPU`
    /// golden
    /// [`strand_to_lss`](prism_render_architecture::hair::rt_curve::strand_to_lss)
    /// *bit-for-bit* (there is no fma to contract), with non-finite coordinates
    /// and negative/non-finite radii sanitised to the same values as the
    /// reference. A strand with fewer than two vertices yields an empty vector
    /// without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, vertices: &[CurveVertex]) -> Vec<LssSegment> {
        if vertices.len() < 2 {
            return Vec::new();
        }
        let segment_count = vertices.len() - 1;

        let device = ctx.device();

        let uniforms = Params {
            segment_count: segment_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Flatten each vertex to 4 f32: position.xyz, radius. The raw authored
        // values are uploaded unchanged; the shader sanitises them bit-faithfully
        // to the golden.
        let mut flat: Vec<f32> = Vec::with_capacity(vertices.len() * 4);
        for v in vertices {
            flat.push(v.position[0]);
            flat.push(v.position[1]);
            flat.push(v.position[2]);
            flat.push(v.radius);
        }

        // Output is 8 f32 (a.xyz, radius_a, b.xyz, radius_b) per segment.
        let out_len = segment_count * 8;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_segments_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let vertices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_segments_vertices"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_segments_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_segments_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rt_curve_segments_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: vertices_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_rt_curve_segments_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rt_curve_segments_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (segment_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out.chunks_exact(8)
            .map(|c| LssSegment::new([c[0], c[1], c[2]], [c[4], c[5], c[6]], c[3], c[7]))
            .collect()
    }
}

/// The `CPU` golden strand-to-segments split, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_strand_to_lss(vertices: &[CurveVertex]) -> Vec<LssSegment> {
    strand_to_lss(vertices)
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
