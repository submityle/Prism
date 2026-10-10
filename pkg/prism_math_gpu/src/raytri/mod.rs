//! Host orchestration of the ray-triangle (Möller-Trumbore) compute kernel
//! (§24.1 twin).
//!
//! [`GpuRayTri`] intersects a batch of ray/triangle pairs on a real device,
//! mirroring the CPU references
//! [`ray_triangle`](prism_math::intersect::ray_triangle) /
//! [`ray_triangle_bary`](prism_math::intersect::ray_triangle_bary). The WGSL
//! intersection routine is **not** duplicated here: the kernel source is
//! composed at runtime by prefixing the single-sourced fragment
//! [`WGSL_RAYTRI`](prism_math::shader_mirror::WGSL_RAYTRI) ahead of a thin
//! compute wrapper, so the device math cannot silently drift from the CPU
//! reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! The barycentric divides, cross/dot products, and the normal `normalize`
//! are evaluated under Metal fast-math, which rounds differently from the
//! CPU's separately-rounded operations. The parity tests therefore assert the
//! discrete hit flag matches exactly (for geometry comfortably away from an
//! edge / grazing ray) and `t`/`u`/`v`/point/normal agree within a small
//! absolute+relative tolerance.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_RAYTRI;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::raycast::GpuRay;

/// Compute workgroup length (the kernel is `@workgroup_size(64, 1, 1)`).
pub const WORKGROUP: u32 = 64;

/// A single triangle `(a, b, c)`; each vertex is `[x, y, z, _]` padded to a
/// `vec4<f32>` storage element.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuTri {
    /// Vertex `a` in `[x, y, z, _]`.
    pub a: [f32; 4],
    /// Vertex `b` in `[x, y, z, _]`.
    pub b: [f32; 4],
    /// Vertex `c` in `[x, y, z, _]`.
    pub c: [f32; 4],
}

impl GpuTri {
    /// Builds a triangle from its three vertices.
    #[must_use]
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> GpuTri {
        GpuTri {
            a: [a[0], a[1], a[2], 0.0],
            b: [b[0], b[1], b[2], 0.0],
            c: [c[0], c[1], c[2], 0.0],
        }
    }
}

/// Raw 32-byte output record matching `TriHitRaw` in the kernel: a header
/// `vec4` holding `(hit, t, u, v)` and a `vec4` normal (`xyz` + pad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TriHitRaw {
    header: [f32; 4],
    normal: [f32; 4],
}

/// Decoded ray-triangle intersection result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTriHit {
    /// Whether the ray hit the triangle.
    pub hit: bool,
    /// Ray parameter at the hit (`origin + t * dir`).
    pub t: f32,
    /// Barycentric weight of vertex `b`.
    pub u: f32,
    /// Barycentric weight of vertex `c`.
    pub v: f32,
    /// Geometric normal oriented against the ray.
    pub normal: [f32; 3],
}

impl GpuTriHit {
    fn decode(raw: TriHitRaw) -> GpuTriHit {
        GpuTriHit {
            hit: raw.header[0] != 0.0,
            t: raw.header[1],
            u: raw.header[2],
            v: raw.header[3],
            normal: [raw.normal[0], raw.normal[1], raw.normal[2]],
        }
    }
}

/// Uniform block carrying the batch length.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// The compute wrapper appended after the single-sourced intersection
/// fragment. It references `prism_ray_triangle`, which is *only* defined by
/// [`WGSL_RAYTRI`]; the two are concatenated in [`GpuRayTri::new`].
const COMPUTE_WRAPPER: &str = "\
struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 };\n\
struct RayIn { origin: vec4<f32>, dir: vec4<f32> };\n\
struct TriIn { a: vec4<f32>, b: vec4<f32>, c: vec4<f32> };\n\
struct TriHitRaw { header: vec4<f32>, normal: vec4<f32> };\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> rays: array<RayIn>;\n\
@group(0) @binding(2) var<storage, read> tris: array<TriIn>;\n\
@group(0) @binding(3) var<storage, read_write> hits: array<TriHitRaw>;\n\
\n\
@compute @workgroup_size(64, 1, 1)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) {\n\
        return;\n\
    }\n\
    let r = rays[i];\n\
    let tri = tris[i];\n\
    let h = prism_ray_triangle(r.origin.xyz, r.dir.xyz, tri.a.xyz, tri.b.xyz, tri.c.xyz);\n\
    var rec: TriHitRaw;\n\
    rec.header = vec4<f32>(h.hit, h.t, h.u, h.v);\n\
    rec.normal = vec4<f32>(h.normal, 0.0);\n\
    hits[i] = rec;\n\
}\n";

/// Compiled ray-triangle pipeline and its bind-group layout.
pub struct GpuRayTri {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuRayTri {
    /// Compiles the ray-triangle kernel on `ctx`'s device, composing the shader
    /// from the single-sourced [`WGSL_RAYTRI`] fragment plus the compute
    /// wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayTri {
        let device = ctx.device();
        let mut source = String::with_capacity(WGSL_RAYTRI.len() + COMPUTE_WRAPPER.len() + 1);
        source.push_str(WGSL_RAYTRI);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_raytri"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_raytri_layout"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_raytri_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_raytri_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRayTri { pipeline, layout }
    }

    /// Intersects each `(rays[i], tris[i])` pair on the device, returning the
    /// decoded hits in order.
    ///
    /// An empty input returns an empty `Vec` without dispatching.
    ///
    /// # Panics
    ///
    /// Panics if `rays` and `tris` do not have the same length.
    #[must_use]
    pub fn cast(&self, ctx: &GpuContext, rays: &[GpuRay], tris: &[GpuTri]) -> Vec<GpuTriHit> {
        assert_eq!(
            rays.len(),
            tris.len(),
            "ray-triangle inputs must be equal length"
        );
        let count = rays.len();
        if count == 0 {
            return Vec::new();
        }

        let device = ctx.device();
        let params = Params {
            count: u32::try_from(count).expect("pair count fits in u32"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (count * size_of::<TriHitRaw>()) as u64;
        let params_buf = buffer::uniform(device, "prism_math_raytri_params", &params);
        let rays_buf = buffer::storage_read(device, "prism_math_raytri_rays", rays);
        let tris_buf = buffer::storage_read(device, "prism_math_raytri_tris", tris);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_raytri_out", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_raytri_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: rays_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: tris_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_raytri_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_raytri_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.count.div_ceil(WORKGROUP), 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_raytri_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let raw = buffer::read_back::<TriHitRaw>(ctx, &stage);
        raw.iter()
            .take(count)
            .map(|&r| GpuTriHit::decode(r))
            .collect()
    }
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
