//! Host orchestration of the GPU finite-segment closest-point kernel
//! (§24.1 twin: nearest-point-on-segment for GPU support / skinning / cloth).
//!
//! [`GpuSegment`] element-wise projects a batch of query points onto a batch of
//! finite segments **on a real device**, mirroring the CPU queries
//! [`Segment3::closest_t`](prism_math::geom::segment::Segment3::closest_t),
//! [`closest_point`](prism_math::geom::segment::Segment3::closest_point), and
//! [`distance_squared`](prism_math::geom::segment::Segment3::distance_squared).
//! Each element receives the segment parameter `t`, the closest point, and the
//! squared distance in one dispatch. The query math is **not** duplicated here:
//! the kernel is composed at runtime by prefixing the single-sourced fragment
//! [`WGSL_SEGMENT`](prism_math::shader_mirror::WGSL_SEGMENT) ahead of a thin
//! compute wrapper, so the device query cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Tolerance / discrete parity
//!
//! The kernel does a projection divide plus a dot product, and Metal compiles
//! WGSL under fast-math, so `t`, the closest point, and the squared distance
//! agree with the CPU only within a small absolute+relative epsilon. The parity
//! suite uses non-degenerate segments (a comfortable margin above the
//! `1.0e-20` squared-length guard) so the clamped parameter and the
//! interior/endpoint classification are unambiguous.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_SEGMENT;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Threads per workgroup; a standard 1D batch tiling.
const WORKGROUP: u32 = 64;

/// Compute wrapper for the batched segment closest-point query. Each item is
/// three `vec4`: `start`, `end`, and the query `point` (`.xyz` used).
const WRAP_SEGMENT: &str = "\
struct Params { count: u32 };\n\
struct SegIn { start: vec4<f32>, end: vec4<f32>, point: vec4<f32> };\n\
struct QueryOut { closest: vec4<f32>, dist: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> items: array<SegIn>;\n\
@group(0) @binding(2) var<storage, read_write> out: array<QueryOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = items[i];\n\
    let t = prism_segment_closest_t(s.start.xyz, s.end.xyz, s.point.xyz);\n\
    let cp = prism_segment_closest_point(s.start.xyz, s.end.xyz, s.point.xyz);\n\
    let ds = prism_segment_distance_squared(s.start.xyz, s.end.xyz, s.point.xyz);\n\
    var o: QueryOut;\n\
    o.closest = vec4<f32>(cp, t);\n\
    o.dist = vec4<f32>(ds, 0.0, 0.0, 0.0);\n\
    out[i] = o;\n\
}\n";

/// A single segment/point query: the segment endpoints (`start.xyz`, `end.xyz`)
/// and the query point (`point.xyz`). The `w` lanes are padding to keep the
/// `std430` stride at 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuSegmentQuery {
    /// Segment start endpoint in `[x, y, z, _]`.
    pub start: [f32; 4],
    /// Segment end endpoint in `[x, y, z, _]`.
    pub end: [f32; 4],
    /// Query point in `[x, y, z, _]`.
    pub point: [f32; 4],
}

impl GpuSegmentQuery {
    /// Builds a query from the segment endpoints and the query point.
    #[must_use]
    pub fn new(start: [f32; 3], end: [f32; 3], point: [f32; 3]) -> GpuSegmentQuery {
        GpuSegmentQuery {
            start: [start[0], start[1], start[2], 0.0],
            end: [end[0], end[1], end[2], 0.0],
            point: [point[0], point[1], point[2], 0.0],
        }
    }
}

/// Raw `std430` query record read back from the device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QueryRaw {
    closest: [f32; 4],
    dist: [f32; 4],
}

/// A decoded closest-point result, mirroring the CPU `Segment3` query trio:
/// the clamped parameter `t`, the closest point, and the squared distance.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GpuSegmentHit {
    /// Segment parameter `t` in `[0, 1]` of the closest point.
    pub t: f32,
    /// World-space point on the segment closest to the query point.
    pub closest_point: [f32; 3],
    /// Squared distance from the query point to the segment.
    pub distance_squared: f32,
}

impl From<QueryRaw> for GpuSegmentHit {
    fn from(raw: QueryRaw) -> GpuSegmentHit {
        GpuSegmentHit {
            t: raw.closest[3],
            closest_point: [raw.closest[0], raw.closest[1], raw.closest[2]],
            distance_squared: raw.dist[0],
        }
    }
}

/// Uniform block: the batch element count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    count: u32,
    _pad: [u32; 3],
}

/// A real-device batched segment closest-point kernel, composed from the
/// single-sourced [`WGSL_SEGMENT`] fragment.
pub struct GpuSegment {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSegment {
    /// Builds the segment closest-point pipeline from the single-sourced
    /// [`WGSL_SEGMENT`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSegment {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_segment_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_segment_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_SEGMENT);
        source.push('\n');
        source.push_str(WRAP_SEGMENT);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_segment"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_segment"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSegment { layout, pipeline }
    }

    /// Element-wise projects `queries[i].point` onto the segment
    /// `(queries[i].start, queries[i].end)`, mirroring
    /// [`prism_math::geom::segment::Segment3`]'s `closest_t` / `closest_point` /
    /// `distance_squared` in one dispatch.
    #[must_use]
    pub fn query(&self, ctx: &GpuContext, queries: &[GpuSegmentQuery]) -> Vec<GpuSegmentHit> {
        let n = queries.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_segment_params",
            &CountParams {
                count: crate::dispatch::checked_count(n),
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_segment_in", queries);
        let out_bytes = (n * size_of::<QueryRaw>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_segment_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_segment_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_segment_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<QueryRaw>(ctx, &stage)
            .into_iter()
            .map(GpuSegmentHit::from)
            .collect()
    }

    /// Builds the three-entry bind group (count uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_segment_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Records a 1D batch dispatch covering `n` elements at [`WORKGROUP`] threads
/// per group.
fn dispatch(
    enc: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    n: usize,
) {
    let groups = crate::dispatch::checked_workgroup_count(n, WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_segment_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// Device-free gate: the composed kernel (fragment + wrapper) must be a valid
/// WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernel_parses() {
        let src = alloc::format!("{}\n{}", super::WGSL_SEGMENT, super::WRAP_SEGMENT);
        if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
            panic!("segment failed to parse: {err}\n--- source ---\n{src}");
        }
    }
}
