//! Host orchestration of the GPU axis-aligned box query kernels (§24.1 twin:
//! BVH refit/merge and box-distance queries on device).
//!
//! [`GpuAabbOps`] batch-evaluates the box helpers **on a real device**,
//! mirroring the CPU [`prism_math::Aabb3`] methods: the center, the closest
//! point, the squared distance to a point, and the SAH `merge` of two boxes. The
//! math is **not** duplicated here: each kernel is composed at runtime by
//! prefixing the single-sourced fragment
//! [`WGSL_AABB_OPS`](prism_math::shader_mirror::WGSL_AABB_OPS) ahead of a thin
//! compute wrapper, so the device query cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Parity
//!
//! The ops are component-wise min/max/add/mul and a dot product, so the device
//! matches the CPU to rounding (compared within a small absolute+relative
//! epsilon). It skips gracefully when no adapter is available; the device-free
//! WGSL parse gate and a real `GPU` cover the rest.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::Pod;
use prism_math::shader_mirror::WGSL_AABB_OPS;
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

/// Center wrapper: one box in, `vec4(center.xyz, 0)` out.
const WRAP_CENTER: &str = "\
struct Params { count: u32 };\n\
struct AabbIn { lo: vec4<f32>, hi: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<AabbIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let b = src[i];\n\
    dst[i] = vec4<f32>(prism_aabb_center(b.lo.xyz, b.hi.xyz), 0.0);\n\
}\n";

/// Closest-point wrapper: one box + one point in, `vec4(closest.xyz, 0)` out.
const WRAP_CLOSEST: &str = "\
struct Params { count: u32 };\n\
struct QueryIn { lo: vec4<f32>, hi: vec4<f32>, p: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<QueryIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let q = src[i];\n\
    dst[i] = vec4<f32>(prism_aabb_closest_point(q.lo.xyz, q.hi.xyz, q.p.xyz), 0.0);\n\
}\n";

/// Squared-distance wrapper: one box + one point in, `f32` out.
const WRAP_DIST2: &str = "\
struct Params { count: u32 };\n\
struct QueryIn { lo: vec4<f32>, hi: vec4<f32>, p: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<QueryIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let q = src[i];\n\
    dst[i] = prism_aabb_distance_squared(q.lo.xyz, q.hi.xyz, q.p.xyz);\n\
}\n";

/// Merge wrapper: two boxes in, the merged box out.
const WRAP_MERGE: &str = "\
struct Params { count: u32 };\n\
struct PairIn { a_lo: vec4<f32>, a_hi: vec4<f32>, b_lo: vec4<f32>, b_hi: vec4<f32> };\n\
struct AabbOut { lo: vec4<f32>, hi: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<PairIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<AabbOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    var o: AabbOut;\n\
    o.lo = vec4<f32>(prism_aabb_merge_lo(p.a_lo.xyz, p.b_lo.xyz), 0.0);\n\
    o.hi = vec4<f32>(prism_aabb_merge_hi(p.a_hi.xyz, p.b_hi.xyz), 0.0);\n\
    dst[i] = o;\n\
}\n";

/// Uniform block: count only, padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CountParams {
    count: u32,
    _pad: [u32; 3],
}

/// A box input element: `[lo.xyz, _]` then `[hi.xyz, _]` (32 bytes).
pub type Aabb4 = [[f32; 4]; 2];

/// A box + query point input element (48 bytes).
pub type AabbPoint = [[f32; 4]; 3];

/// A pair of boxes input element (64 bytes).
pub type AabbPair = [[f32; 4]; 4];

/// Packs a box from `lo`/`hi` corners.
#[must_use]
pub fn aabb4(lo: [f32; 3], hi: [f32; 3]) -> Aabb4 {
    [[lo[0], lo[1], lo[2], 0.0], [hi[0], hi[1], hi[2], 0.0]]
}

/// Packs a box + query point.
#[must_use]
pub fn aabb_point(lo: [f32; 3], hi: [f32; 3], p: [f32; 3]) -> AabbPoint {
    [
        [lo[0], lo[1], lo[2], 0.0],
        [hi[0], hi[1], hi[2], 0.0],
        [p[0], p[1], p[2], 0.0],
    ]
}

/// Packs a pair of boxes.
#[must_use]
pub fn aabb_pair(a: Aabb4, b: Aabb4) -> AabbPair {
    [a[0], a[1], b[0], b[1]]
}

/// Real-device batched AABB query kernels, composed from the single-sourced
/// [`WGSL_AABB_OPS`] fragment.
pub struct GpuAabbOps {
    layout: BindGroupLayout,
    center: ComputePipeline,
    closest: ComputePipeline,
    dist2: ComputePipeline,
    merge: ComputePipeline,
}

impl GpuAabbOps {
    /// Builds the four AABB query pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAabbOps {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_aabb_ops_layout"),
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
            label: Some("prism_math_aabb_ops_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_AABB_OPS);
            source.push('\n');
            source.push_str(wrapper);
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source: ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        GpuAabbOps {
            center: build("prism_math_aabb_center", WRAP_CENTER),
            closest: build("prism_math_aabb_closest", WRAP_CLOSEST),
            dist2: build("prism_math_aabb_dist2", WRAP_DIST2),
            merge: build("prism_math_aabb_merge", WRAP_MERGE),
            layout,
        }
    }

    /// Batch box centers, mirroring [`prism_math::Aabb3::center`].
    #[must_use]
    pub fn center(&self, ctx: &GpuContext, boxes: &[Aabb4]) -> Vec<[f32; 3]> {
        self.run::<Aabb4, [f32; 4]>(ctx, &self.center, boxes)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }

    /// Batch closest points, mirroring [`prism_math::Aabb3::closest_point`].
    #[must_use]
    pub fn closest_point(&self, ctx: &GpuContext, queries: &[AabbPoint]) -> Vec<[f32; 3]> {
        self.run::<AabbPoint, [f32; 4]>(ctx, &self.closest, queries)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }

    /// Batch squared distances, mirroring [`prism_math::Aabb3::distance_squared`].
    #[must_use]
    pub fn distance_squared(&self, ctx: &GpuContext, queries: &[AabbPoint]) -> Vec<f32> {
        self.run::<AabbPoint, f32>(ctx, &self.dist2, queries)
    }

    /// Batch SAH merges, mirroring [`prism_math::Aabb3::merge`]. Each result is
    /// `[lo, hi]`.
    #[must_use]
    pub fn merge(&self, ctx: &GpuContext, pairs: &[AabbPair]) -> Vec<Aabb4> {
        self.run::<AabbPair, Aabb4>(ctx, &self.merge, pairs)
    }

    /// Uploads an input batch, dispatches `pipeline`, and reads `U` back.
    fn run<T: Pod, U: Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[T],
    ) -> Vec<U> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_aabb_ops_params",
            &CountParams {
                count: n as u32,
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_aabb_ops_in", items);
        let out_bytes = (n * size_of::<U>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_aabb_ops_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_aabb_ops_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_aabb_ops_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<U>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_aabb_ops_bind_group"),
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
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_aabb_ops_pass"),
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

/// Device-free gate: every composed kernel (fragment + wrapper) must be a valid
/// WGSL module on the pinned `naga`, so a wrapper bug fails CI without a GPU.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernels_parse() {
        for (label, wrapper) in [
            ("center", super::WRAP_CENTER),
            ("closest", super::WRAP_CLOSEST),
            ("dist2", super::WRAP_DIST2),
            ("merge", super::WRAP_MERGE),
        ] {
            let src = alloc::format!("{}\n{}", super::WGSL_AABB_OPS, wrapper);
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("aabb_ops `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
