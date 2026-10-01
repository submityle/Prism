//! `wgpu` compute twin of Prism's Linear-Swept-Spheres per-segment bounding box
//! ([`lss_segment_aabb`](prism_render_architecture::hair::rt_curve::lss_segment_aabb)).
//!
//! The newest ray-tracing hardware (`RTX` via `DXR` / `OptiX`) can express a
//! hair strand segment as a `Linear Swept Spheres` (LSS) / curve primitive the
//! acceleration structure traverses directly, so hair self-shadowing and
//! in-reflection real strands no longer need a proxy mesh. Before the driver
//! builds that `BLAS` it needs each segment's conservative `axis-aligned`
//! bounding box (`AABB`); this twin computes that box batch-wide on the device,
//! one thread per segment.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRtCurveAabb::eval`] takes a batch of [`LssSegment`] and returns one
//! [`Aabb`] per segment in input order — the array-in/array-out form a `BLAS`
//! builder consumes. The segment index is the invocation id
//! (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the segment count early-return.
//!
//! # Correctness model
//!
//! A swept-sphere endpoint box needs no square root — each axis is the centre
//! plus or minus its (non-negative) radius — and the segment box is the per-axis
//! `min`/`max` union of the two endpoint boxes. There is **no** multiply-add the
//! `GPU` could contract into an fma, so the kernel is *bit-exact* with the
//! scalar reference: the parity test compares raw bit patterns, not a tolerance.
//! The per-component sanitiser mirrors the golden's
//! [`LssSegment::sanitized`](prism_render_architecture::hair::rt_curve::LssSegment::sanitized)
//! bit-faithfully (`v == v && abs(v) < +inf` keeps finite coordinates and
//! forces the rest to `0`; a negative or non-finite radius collapses to `0`), so
//! the host uploads the raw authored segments unchanged.
//!
//! # Portability
//!
//! The kernel uses only `min`, `max`, `select` and single add/sub in the
//! portable core-`WGSL` subset — no `sqrt`, `exp`, `pow` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
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

use prism_render_architecture::hair::rt_curve::{lss_segment_aabb, Aabb, LssSegment};

use crate::context::GpuContext;

/// Uniform parameters for one segment-`AABB` dispatch. Layout matches `Params`
/// in `shaders/rt_curve_aabb.wesl`: the segment count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    segment_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-segment LSS bounding-box pipeline.
pub struct GpuHairRtCurveAabb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRtCurveAabb {
    /// Compiles the per-segment LSS bounding-box kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRtCurveAabb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rt_curve_aabb"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rt_curve_aabb.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rt_curve_aabb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rt_curve_aabb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rt_curve_aabb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRtCurveAabb {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes the conservative `AABB` of each LSS segment, returning one
    /// [`Aabb`] per input in order.
    ///
    /// The box for segment `i` equals the `CPU` golden
    /// [`lss_segment_aabb`](prism_render_architecture::hair::rt_curve::lss_segment_aabb)
    /// of `segments[i]` *bit-for-bit* (there is no fma to contract), with
    /// non-finite coordinates and negative/non-finite radii sanitised to the
    /// same values as the reference. An empty batch yields an empty vector
    /// without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, segments: &[LssSegment]) -> Vec<Aabb> {
        let segment_count = segments.len();
        if segment_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            segment_count: segment_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Flatten each segment to 8 f32: a.xyz, radius_a, b.xyz, radius_b. The
        // raw authored values are uploaded unchanged; the shader sanitises them
        // bit-faithfully to the golden.
        let mut flat: Vec<f32> = Vec::with_capacity(segment_count * 8);
        for s in segments {
            flat.push(s.a[0]);
            flat.push(s.a[1]);
            flat.push(s.a[2]);
            flat.push(s.radius_a);
            flat.push(s.b[0]);
            flat.push(s.b[1]);
            flat.push(s.b[2]);
            flat.push(s.radius_b);
        }

        // Output is 6 f32 (min.xyz, max.xyz) per segment.
        let out_len = segment_count * 6;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_aabb_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let segments_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_aabb_segments"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_aabb_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_aabb_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rt_curve_aabb_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: segments_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_rt_curve_aabb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rt_curve_aabb_pass"),
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

        out.chunks_exact(6)
            .map(|c| Aabb::new([c[0], c[1], c[2]], [c[3], c[4], c[5]]))
            .collect()
    }
}

/// The `CPU` golden box for one LSS segment, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_segment_aabb(seg: &LssSegment) -> Aabb {
    lss_segment_aabb(seg)
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
