//! Host orchestration of the GPU-driven view-frustum culling kernels
//! (§24.1 twin: the GPU side of visibility determination for indirect draw).
//!
//! [`GpuFrustumCull`] batch-classifies bounding spheres and axis-aligned boxes
//! against a six-plane [`prism_math::geom::Frustum`] **on a real device** — the
//! canonical GPU-driven-rendering cull that feeds indirect draw / instance
//! compaction — mirroring the CPU classifiers
//! [`prism_math::intersect::frustum_sphere`] / [`frustum_aabb`]. The plane test
//! WGSL is **not** duplicated here: each kernel is composed at runtime by
//! prefixing the single-sourced fragment
//! [`WGSL_FRUSTUM_CULL`](prism_math::shader_mirror::WGSL_FRUSTUM_CULL) ahead of
//! a thin compute wrapper, so the device cull math cannot silently drift from
//! the CPU reference — it is literally the same text.
//!
//! # Discrete-result parity (with a conservative-culling caveat)
//!
//! The classifier returns the [`Containment`](prism_math::intersect::Containment)
//! discriminant as a `u32` (`0 = Outside`, `1 = Intersecting`, `2 = Inside`).
//! The arithmetic is float dot products, and Metal compiles WGSL under
//! fast-math, so a volume whose boundary lies within fast-math rounding of a
//! frustum plane could be classified one level differently than the CPU. That
//! is exactly the conservative-culling tolerance real engines accept; the
//! parity suite therefore exercises geometry with a comfortable margin from
//! every plane, where the discrete classification agrees exactly.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_FRUSTUM_CULL;
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

/// `Containment::Outside` discriminant mirrored on the device.
pub const OUTSIDE: u32 = 0;
/// `Containment::Intersecting` discriminant mirrored on the device.
pub const INTERSECTING: u32 = 1;
/// `Containment::Inside` discriminant mirrored on the device.
pub const INSIDE: u32 = 2;

/// Compute wrapper for batched sphere culling.
const WRAP_SPHERE: &str = "\
struct Frustum { planes: array<vec4<f32>, 6>, count: u32 };\n\
@group(0) @binding(0) var<uniform> fr: Frustum;\n\
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> result: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= fr.count) { return; }\n\
    let s = spheres[i];\n\
    result[i] = prism_frustum_classify_sphere(fr.planes, s.xyz, s.w);\n\
}\n";

/// Compute wrapper for batched AABB culling. Each box is two `vec4`: the center
/// (`.xyz`) and the positive half-extents (`.xyz`).
const WRAP_AABB: &str = "\
struct Frustum { planes: array<vec4<f32>, 6>, count: u32 };\n\
struct AabbIn { center: vec4<f32>, extent: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> fr: Frustum;\n\
@group(0) @binding(1) var<storage, read> boxes: array<AabbIn>;\n\
@group(0) @binding(2) var<storage, read_write> result: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= fr.count) { return; }\n\
    let b = boxes[i];\n\
    result[i] = prism_frustum_classify_aabb(fr.planes, b.center.xyz, b.extent.xyz);\n\
}\n";

/// Uniform block: the six frustum planes (`normal.xyz, d`) plus the batch count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrustumParams {
    planes: [[f32; 4]; 6],
    count: u32,
    _pad: [u32; 3],
}

/// A real-device batched frustum-cull kernel pair (sphere + AABB), each
/// composed from the single-sourced [`WGSL_FRUSTUM_CULL`] fragment.
pub struct GpuFrustumCull {
    layout: BindGroupLayout,
    sphere: ComputePipeline,
    aabb: ComputePipeline,
}

impl GpuFrustumCull {
    /// Builds the two cull pipelines (sphere & AABB), each composed from the
    /// single-sourced [`WGSL_FRUSTUM_CULL`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFrustumCull {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_frustum_cull_layout"),
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
            label: Some("prism_math_frustum_cull_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_FRUSTUM_CULL);
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
        GpuFrustumCull {
            sphere: build("prism_math_frustum_cull_sphere", WRAP_SPHERE),
            aabb: build("prism_math_frustum_cull_aabb", WRAP_AABB),
            layout,
        }
    }

    /// Batch-classifies bounding spheres (`[cx, cy, cz, radius]`) against the
    /// `planes` (each `[nx, ny, nz, d]`, inward-facing unit normal), mirroring
    /// [`prism_math::intersect::frustum_sphere`]. Each output is the
    /// `Containment` discriminant ([`OUTSIDE`] / [`INTERSECTING`] / [`INSIDE`]).
    #[must_use]
    pub fn cull_spheres(
        &self,
        ctx: &GpuContext,
        planes: &[[f32; 4]; 6],
        spheres: &[[f32; 4]],
    ) -> Vec<u32> {
        self.run(ctx, &self.sphere, planes, spheres)
    }

    /// Batch-classifies AABBs (each two `vec4`: `[cx, cy, cz, _]` center and
    /// `[ex, ey, ez, _]` positive half-extents) against the `planes`, mirroring
    /// [`prism_math::intersect::frustum_aabb`].
    #[must_use]
    pub fn cull_aabbs(
        &self,
        ctx: &GpuContext,
        planes: &[[f32; 4]; 6],
        boxes: &[[[f32; 4]; 2]],
    ) -> Vec<u32> {
        self.run(ctx, &self.aabb, planes, boxes)
    }

    /// Uploads the frustum uniform + a `T` input batch, dispatches `pipeline`,
    /// and reads back one `u32` classification per element.
    fn run<T: Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        planes: &[[f32; 4]; 6],
        items: &[T],
    ) -> Vec<u32> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_frustum_cull_params",
            &FrustumParams {
                planes: *planes,
                count: n as u32,
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_frustum_cull_in", items);
        let out_bytes = (n * size_of::<u32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_frustum_cull_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_frustum_cull_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_frustum_cull_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<u32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (frustum uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_frustum_cull_bind_group"),
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
        label: Some("prism_math_frustum_cull_pass"),
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
