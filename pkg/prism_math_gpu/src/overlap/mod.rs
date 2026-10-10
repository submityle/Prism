//! Host orchestration of the GPU broad-phase primitive overlap kernels
//! (§24.1/§24.7 twin: the GPU side of collision / culling broad-phase
//! prefiltering for indirect dispatch).
//!
//! [`GpuOverlap`] batch-evaluates the classic broad-phase overlap predicates
//! **on a real device** — AABB vs AABB, sphere vs sphere, and sphere vs AABB —
//! mirroring the CPU booleans [`prism_math::intersect::aabb_aabb`] /
//! [`sphere_sphere`](prism_math::intersect::sphere_sphere) /
//! [`sphere_aabb`](prism_math::intersect::sphere_aabb). This is the compute
//! prefilter a GPU-driven pipeline runs before a narrow phase (contact
//! generation / exact intersection). The predicate WGSL is **not** duplicated
//! here: each kernel is composed at runtime by prefixing the single-sourced
//! fragment [`WGSL_OVERLAP`](prism_math::shader_mirror::WGSL_OVERLAP) ahead of a
//! thin compute wrapper, so the device predicate cannot silently drift from the
//! CPU reference — it is literally the same text.
//!
//! # Discrete-result parity (with a conservative broad-phase caveat)
//!
//! Each predicate returns `1u` on overlap and `0u` on disjoint. The arithmetic
//! is pure comparisons and dot products (no transcendental, no normalize), and
//! Metal compiles WGSL under fast-math, so a pair whose separation lies within
//! fast-math rounding of exact tangency could flip one way. That is the
//! standard broad-phase conservative tolerance real engines accept; the parity
//! suite therefore exercises pairs with a comfortable margin from touching,
//! where the discrete boolean agrees with the CPU reference exactly.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_OVERLAP;
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

/// Compute wrapper for batched AABB-vs-AABB tests. Each pair is four `vec4`:
/// `a_min`, `a_max`, `b_min`, `b_max` (`.xyz` used).
const WRAP_AABB_AABB: &str = "\
struct Params { count: u32 };\n\
struct Pair { a_min: vec4<f32>, a_max: vec4<f32>, b_min: vec4<f32>, b_max: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;\n\
@group(0) @binding(2) var<storage, read_write> result: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = pairs[i];\n\
    result[i] = prism_overlap_aabb_aabb(p.a_min.xyz, p.a_max.xyz, p.b_min.xyz, p.b_max.xyz);\n\
}\n";

/// Compute wrapper for batched sphere-vs-sphere tests. Each pair is two `vec4`:
/// `a` and `b`, each `[cx, cy, cz, radius]`.
const WRAP_SPHERE_SPHERE: &str = "\
struct Params { count: u32 };\n\
struct Pair { a: vec4<f32>, b: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;\n\
@group(0) @binding(2) var<storage, read_write> result: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = pairs[i];\n\
    result[i] = prism_overlap_sphere_sphere(p.a.xyz, p.a.w, p.b.xyz, p.b.w);\n\
}\n";

/// Compute wrapper for batched sphere-vs-AABB tests. Each pair is three `vec4`:
/// `sphere` (`[cx, cy, cz, radius]`), `b_min`, `b_max` (`.xyz` used).
const WRAP_SPHERE_AABB: &str = "\
struct Params { count: u32 };\n\
struct Pair { sphere: vec4<f32>, b_min: vec4<f32>, b_max: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;\n\
@group(0) @binding(2) var<storage, read_write> result: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = pairs[i];\n\
    result[i] = prism_overlap_sphere_aabb(p.sphere.xyz, p.sphere.w, p.b_min.xyz, p.b_max.xyz);\n\
}\n";

/// Uniform block: the batch element count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    _pad: [u32; 3],
}

/// A real-device batched broad-phase overlap kernel trio (AABB/AABB,
/// sphere/sphere, sphere/AABB), each composed from the single-sourced
/// [`WGSL_OVERLAP`] fragment.
pub struct GpuOverlap {
    layout: BindGroupLayout,
    aabb_aabb: ComputePipeline,
    sphere_sphere: ComputePipeline,
    sphere_aabb: ComputePipeline,
}

impl GpuOverlap {
    /// Builds the three overlap pipelines, each composed from the
    /// single-sourced [`WGSL_OVERLAP`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOverlap {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_overlap_layout"),
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
            label: Some("prism_math_overlap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_OVERLAP);
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
        GpuOverlap {
            aabb_aabb: build("prism_math_overlap_aabb_aabb", WRAP_AABB_AABB),
            sphere_sphere: build("prism_math_overlap_sphere_sphere", WRAP_SPHERE_SPHERE),
            sphere_aabb: build("prism_math_overlap_sphere_aabb", WRAP_SPHERE_AABB),
            layout,
        }
    }

    /// Batch-tests AABB-vs-AABB overlap. Each pair is four `vec4`
    /// (`a_min`, `a_max`, `b_min`, `b_max`), mirroring
    /// [`prism_math::intersect::aabb_aabb`]. `true` means the boxes overlap.
    #[must_use]
    pub fn aabb_aabb(&self, ctx: &GpuContext, pairs: &[[[f32; 4]; 4]]) -> Vec<bool> {
        self.run(ctx, &self.aabb_aabb, pairs)
    }

    /// Batch-tests sphere-vs-sphere overlap. Each pair is two `vec4`
    /// (`a`, `b`, each `[cx, cy, cz, radius]`), mirroring
    /// [`prism_math::intersect::sphere_sphere`]. `true` means they overlap.
    #[must_use]
    pub fn sphere_sphere(&self, ctx: &GpuContext, pairs: &[[[f32; 4]; 2]]) -> Vec<bool> {
        self.run(ctx, &self.sphere_sphere, pairs)
    }

    /// Batch-tests sphere-vs-AABB overlap. Each pair is three `vec4`
    /// (`sphere` `[cx, cy, cz, radius]`, `b_min`, `b_max`), mirroring
    /// [`prism_math::intersect::sphere_aabb`]. `true` means they overlap.
    #[must_use]
    pub fn sphere_aabb(&self, ctx: &GpuContext, pairs: &[[[f32; 4]; 3]]) -> Vec<bool> {
        self.run(ctx, &self.sphere_aabb, pairs)
    }

    /// Uploads a `T` input batch, dispatches `pipeline`, and reads back one
    /// boolean (`1u`/`0u`) overlap result per element.
    fn run<T: Pod>(&self, ctx: &GpuContext, pipeline: &ComputePipeline, items: &[T]) -> Vec<bool> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_overlap_params",
            &Params {
                count: n as u32,
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_overlap_in", items);
        let out_bytes = (n * size_of::<u32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_overlap_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_overlap_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_overlap_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<u32>(ctx, &stage)
            .into_iter()
            .map(|v| v != 0)
            .collect()
    }

    /// Builds the three-entry bind group (count uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_overlap_bind_group"),
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
        label: Some("prism_math_overlap_pass"),
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
