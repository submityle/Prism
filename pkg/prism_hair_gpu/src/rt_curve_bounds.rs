//! `wgpu` compute twin of Prism's Linear-Swept-Spheres batch bounding-box union
//! ([`segments_aabb`](prism_render_architecture::hair::rt_curve::segments_aabb)).
//!
//! The newest ray-tracing hardware (`RTX` via `DXR` / `OptiX`) can express a
//! hair strand segment as a `Linear Swept Spheres` (LSS) / curve primitive the
//! acceleration structure traverses directly. Where [`GpuHairRtCurveAabb`] emits
//! one conservative box per segment, a `BLAS` builder also needs the single
//! groom-global box that encloses a whole strand (or strand set) at once — the
//! root bound it registers before refining per primitive. This twin folds a
//! batch of [`LssSegment`] into that one enclosing [`Aabb`] on the device.
//!
//! # Reduction shape
//!
//! Unlike the per-segment map twins, this is a shared-memory **tree reduction**
//! (the same shape as [`GpuHairAnalysisReduce`](crate::analysis_reduce)): a
//! single `256`-wide workgroup cooperatively unions the whole segment array.
//! Each invocation grid-strides across the segments accumulating a private
//! partial box from the union identity (the empty box `min = +inf`,
//! `max = -inf`), the partials are staged into workgroup memory, and a
//! logarithmic tree fold unions them down to lane `0`, which writes the single
//! output box.
//!
//! # Correctness model
//!
//! The union of a set of boxes is a per-axis `min` of the mins and `max` of the
//! maxes. `min`/`max` select an existing operand with no arithmetic and are
//! both commutative *and* associative, so the tree's pairwise fold order yields
//! the exact same bits as the golden's left-to-right
//! [`Aabb::union`](prism_render_architecture::hair::rt_curve::Aabb::union) walk:
//! the twin is **bit-exact** with the scalar reference and the parity test
//! compares raw bit patterns, not a tolerance. Each endpoint box is centre plus
//! or minus its (non-negative) radius — single add/sub, **no** multiply the
//! `GPU` could contract into an fma — and the per-component sanitiser mirrors
//! the golden's
//! [`LssSegment::sanitized`](prism_render_architecture::hair::rt_curve::LssSegment::sanitized)
//! bit-faithfully, so the host uploads the raw authored segments unchanged.
//!
//! # Portability
//!
//! The kernel uses only comparisons, `min`, `max`, `select`, single add/sub and
//! workgroup shared memory in the portable core-`WGSL` subset — no `sqrt`,
//! `exp`, `pow`, atomics or optional device feature — so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: hardware ray-traced curve / LSS strand primitive `BLAS` contract
//! (`RTX` `DXR` / `OptiX` LSS) plus a standard single-workgroup shared-memory
//! tree reduction and `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

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

use prism_render_architecture::hair::rt_curve::{segments_aabb, Aabb, LssSegment};

use crate::context::GpuContext;

/// Uniform parameters for one batch-union dispatch. Layout matches `Params` in
/// `shaders/rt_curve_bounds.wesl`: the segment count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    segment_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable LSS batch bounding-box union pipeline.
pub struct GpuHairRtCurveBounds {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRtCurveBounds {
    /// Compiles the batch bounding-box union kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRtCurveBounds {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_rt_curve_bounds"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rt_curve_bounds.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_rt_curve_bounds_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_rt_curve_bounds_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_rt_curve_bounds_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRtCurveBounds {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds a batch of LSS `segments` into the single conservative [`Aabb`]
    /// that encloses them all.
    ///
    /// The result equals
    /// [`segments_aabb`](prism_render_architecture::hair::rt_curve::segments_aabb)
    /// *bit-for-bit* (union is order-independent `min`/`max`, with no fma to
    /// contract), with non-finite coordinates and negative/non-finite radii
    /// sanitised to the same values as the reference. An empty batch yields the
    /// empty box ([`Aabb::EMPTY`]) without a dispatch — storage buffers cannot
    /// be zero-sized — mirroring the golden's empty-set behaviour.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, segments: &[LssSegment]) -> Aabb {
        let segment_count = segments.len();
        if segment_count == 0 {
            return Aabb::EMPTY;
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

        // Output is a single box: 6 f32 (min.xyz, max.xyz).
        let out_bytes = 6u64 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_bounds_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let segments_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_rt_curve_bounds_segments"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_bounds_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_rt_curve_bounds_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_rt_curve_bounds_bind_group"),
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
            label: Some("prism_hair_rt_curve_bounds_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_rt_curve_bounds_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup cooperatively unions the whole batch.
            pass.dispatch_workgroups(1, 1, 1);
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

        Aabb::new([out[0], out[1], out[2]], [out[3], out[4], out[5]])
    }
}

/// The `CPU` golden enclosing box for a batch of LSS segments, re-exported so
/// the parity test can assert the device twin against the identical reference
/// it mirrors.
#[must_use]
pub fn reference_segments_aabb(segs: &[LssSegment]) -> Aabb {
    segments_aabb(segs)
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
