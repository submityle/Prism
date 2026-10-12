//! Host orchestration of the GPU bounding-sphere query kernels (§24.1 twin:
//! sphere containment and smallest-enclosing-sphere refit on device).
//!
//! [`GpuSphere`] batch-evaluates the sphere helpers **on a real device**,
//! mirroring the CPU [`prism_math::BoundingSphere`] `contains_point` and
//! `merge`. The math is **not** duplicated here: each kernel is composed at
//! runtime by prefixing the single-sourced fragment
//! [`WGSL_SPHERE`](prism_math::shader_mirror::WGSL_SPHERE) ahead of a thin
//! compute wrapper, so the device query cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Tolerance / parity
//!
//! `contains_point` is a pure squared-distance comparison (exact away from the
//! boundary); `merge` also computes a `length` and a divide, so it is compared
//! within a small absolute+relative epsilon. It skips gracefully when no adapter
//! is available.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use prism_math::shader_mirror::WGSL_SPHERE;
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

/// Containment wrapper: a sphere (`[cx,cy,cz,r]`) + point per element, `u32`
/// out.
const WRAP_CONTAINS: &str = "\
struct QueryIn { sphere: vec4<f32>, point: vec4<f32> };\n\
@group(0) @binding(0) var<storage, read> src: array<QueryIn>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&src) || i >= arrayLength(&dst)) { return; }\n\
    let q = src[i];\n\
    dst[i] = prism_sphere_contains_point(q.sphere.xyz, q.sphere.w, q.point.xyz);\n\
}\n";

/// Merge wrapper: two spheres in, the enclosing `vec4(center, radius)` out.
const WRAP_MERGE: &str = "\
struct PairIn { a: vec4<f32>, b: vec4<f32> };\n\
@group(0) @binding(0) var<storage, read> src: array<PairIn>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&src) || i >= arrayLength(&dst)) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_sphere_merge(p.a.xyz, p.a.w, p.b.xyz, p.b.w);\n\
}\n";

/// A sphere + query point input element: `[cx,cy,cz,r]` then `[px,py,pz,_]`.
pub type SpherePoint = [[f32; 4]; 2];

/// A pair of spheres input element: two `[cx,cy,cz,r]` `vec4`s.
pub type SpherePair = [[f32; 4]; 2];

/// Real-device batched bounding-sphere kernels, composed from the single-sourced
/// [`WGSL_SPHERE`] fragment.
pub struct GpuSphere {
    layout: BindGroupLayout,
    contains: ComputePipeline,
    merge: ComputePipeline,
}

impl GpuSphere {
    /// Builds the containment and merge pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphere {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_sphere_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_sphere_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_SPHERE);
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
        GpuSphere {
            contains: build("prism_math_sphere_contains", WRAP_CONTAINS),
            merge: build("prism_math_sphere_merge", WRAP_MERGE),
            layout,
        }
    }

    /// Batch point-in-sphere tests, mirroring
    /// [`prism_math::BoundingSphere::contains_point`].
    #[must_use]
    pub fn contains_point(&self, ctx: &GpuContext, items: &[SpherePoint]) -> Vec<bool> {
        self.run::<SpherePoint, u32>(ctx, &self.contains, items)
            .into_iter()
            .map(|v| v != 0)
            .collect()
    }

    /// Batch smallest-enclosing-sphere merges, mirroring
    /// [`prism_math::BoundingSphere::merge`]. Each result is
    /// `[cx, cy, cz, radius]`.
    #[must_use]
    pub fn merge(&self, ctx: &GpuContext, pairs: &[SpherePair]) -> Vec<[f32; 4]> {
        self.run::<SpherePair, [f32; 4]>(ctx, &self.merge, pairs)
    }

    /// Uploads an input batch, dispatches `pipeline`, and reads `U` back.
    fn run<T: bytemuck::Pod, U: bytemuck::Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[T],
    ) -> Vec<U> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let _ = crate::dispatch::checked_count(n);
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_sphere_in", items);
        let out_bytes = (n * size_of::<U>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_sphere_out", out_bytes);
        let bind_group = self.bind(device, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_sphere_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_sphere_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<U>(ctx, &stage)
    }

    /// Builds the two-entry bind group (input, output).
    fn bind(&self, device: &Device, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_sphere_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
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
        label: Some("prism_math_sphere_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// Device-free gate: every composed kernel (fragment + wrapper) must be a valid
/// WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernels_parse() {
        for (label, wrapper) in [
            ("contains", super::WRAP_CONTAINS),
            ("merge", super::WRAP_MERGE),
        ] {
            let src = alloc::format!("{}\n{}", super::WGSL_SPHERE, wrapper);
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("sphere `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
