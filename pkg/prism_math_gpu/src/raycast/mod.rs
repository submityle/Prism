//! Host orchestration of the GPU ray/primitive intersection kernels
//! (§24.1 twin: the GPU side of picking / spatial queries / batched casts).
//!
//! [`GpuRayCast`] element-wise intersects a batch of rays against a batch of
//! spheres or axis-aligned boxes **on a real device**, mirroring the CPU
//! queries [`prism_math::intersect::ray_sphere`] /
//! [`prism_math::intersect::ray_aabb`]. The intersection math is **not**
//! duplicated here: each kernel is composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_RAYCAST`](prism_math::shader_mirror::WGSL_RAYCAST) ahead of a thin
//! compute wrapper, so the device query cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Tolerance / discrete parity
//!
//! The kernel does floating-point arithmetic (the sphere quadratic and the
//! AABB slab divides), and Metal compiles WGSL under fast-math, so the ray
//! parameter `t` and the hit point agree with the CPU only within a small
//! absolute+relative epsilon. The discrete hit flag and the axis-aligned AABB
//! normal are exact for geometry with a comfortable margin from a
//! tangent/grazing boundary; the parity suite exercises exactly such geometry.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_RAYCAST;
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

/// A single ray: origin (`.xyz` of `origin`) and direction (`.xyz` of `dir`).
/// `w` lanes are padding to keep the std430 stride at 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuRay {
    /// Ray origin in `[x, y, z, _]`.
    pub origin: [f32; 4],
    /// Ray direction in `[x, y, z, _]` (need not be unit length).
    pub dir: [f32; 4],
}

impl GpuRay {
    /// Builds a ray from an origin and (possibly non-unit) direction.
    #[must_use]
    pub fn new(origin: [f32; 3], dir: [f32; 3]) -> GpuRay {
        GpuRay {
            origin: [origin[0], origin[1], origin[2], 0.0],
            dir: [dir[0], dir[1], dir[2], 0.0],
        }
    }
}

/// An axis-aligned box: minimum (`lo.xyz`) and maximum (`hi.xyz`) corners.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuAabb {
    /// Minimum corner in `[x, y, z, _]`.
    pub lo: [f32; 4],
    /// Maximum corner in `[x, y, z, _]`.
    pub hi: [f32; 4],
}

impl GpuAabb {
    /// Builds a box from its minimum and maximum corners.
    #[must_use]
    pub fn new(lo: [f32; 3], hi: [f32; 3]) -> GpuAabb {
        GpuAabb {
            lo: [lo[0], lo[1], lo[2], 0.0],
            hi: [hi[0], hi[1], hi[2], 0.0],
        }
    }
}

/// Raw std430 hit record read back from the device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HitRaw {
    header: [f32; 4],
    point: [f32; 4],
    normal: [f32; 4],
}

/// A decoded ray intersection result, mirroring
/// [`prism_math::intersect::RayHit`] wrapped in an `Option`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GpuRayHit {
    /// Whether the ray hit the primitive.
    pub hit: bool,
    /// Ray parameter `t` of the hit (valid only when `hit`).
    pub t: f32,
    /// World-space hit point (valid only when `hit`).
    pub point: [f32; 3],
    /// Unit surface normal oriented against the ray (valid only when `hit`).
    pub normal: [f32; 3],
}

impl From<HitRaw> for GpuRayHit {
    fn from(raw: HitRaw) -> GpuRayHit {
        GpuRayHit {
            hit: raw.header[0] != 0.0,
            t: raw.header[1],
            point: [raw.point[0], raw.point[1], raw.point[2]],
            normal: [raw.normal[0], raw.normal[1], raw.normal[2]],
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

/// Compute wrapper for batched ray/sphere intersection. Each sphere is one
/// `vec4` (`.xyz` center, `.w` radius).
const WRAP_SPHERE: &str = "\
struct Params { count: u32 };\n\
struct RayIn { origin: vec4<f32>, dir: vec4<f32> };\n\
struct HitOut { header: vec4<f32>, point: vec4<f32>, normal: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> rays: array<RayIn>;\n\
@group(0) @binding(2) var<storage, read> spheres: array<vec4<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> hits: array<HitOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let r = rays[i];\n\
    let s = spheres[i];\n\
    let h = prism_ray_sphere(r.origin.xyz, r.dir.xyz, s.xyz, s.w);\n\
    var o: HitOut;\n\
    o.header = vec4<f32>(h.hit, h.t, 0.0, 0.0);\n\
    o.point = vec4<f32>(h.point, 0.0);\n\
    o.normal = vec4<f32>(h.normal, 0.0);\n\
    hits[i] = o;\n\
}\n";

/// Compute wrapper for batched ray/AABB intersection. Each box is two `vec4`
/// (`lo.xyz` min corner, `hi.xyz` max corner).
const WRAP_AABB: &str = "\
struct Params { count: u32 };\n\
struct RayIn { origin: vec4<f32>, dir: vec4<f32> };\n\
struct AabbIn { lo: vec4<f32>, hi: vec4<f32> };\n\
struct HitOut { header: vec4<f32>, point: vec4<f32>, normal: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> rays: array<RayIn>;\n\
@group(0) @binding(2) var<storage, read> boxes: array<AabbIn>;\n\
@group(0) @binding(3) var<storage, read_write> hits: array<HitOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let r = rays[i];\n\
    let b = boxes[i];\n\
    let h = prism_ray_aabb(r.origin.xyz, r.dir.xyz, b.lo.xyz, b.hi.xyz);\n\
    var o: HitOut;\n\
    o.header = vec4<f32>(h.hit, h.t, 0.0, 0.0);\n\
    o.point = vec4<f32>(h.point, 0.0);\n\
    o.normal = vec4<f32>(h.normal, 0.0);\n\
    hits[i] = o;\n\
}\n";

/// A real-device batched ray-intersection kernel pair (sphere + AABB), each
/// composed from the single-sourced [`WGSL_RAYCAST`] fragment.
pub struct GpuRayCast {
    layout: BindGroupLayout,
    sphere: ComputePipeline,
    aabb: ComputePipeline,
}

impl GpuRayCast {
    /// Builds the two ray-query pipelines (sphere & AABB), each composed from
    /// the single-sourced [`WGSL_RAYCAST`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRayCast {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_raycast_layout"),
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
            label: Some("prism_math_raycast_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_RAYCAST);
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
        GpuRayCast {
            sphere: build("prism_math_raycast_sphere", WRAP_SPHERE),
            aabb: build("prism_math_raycast_aabb", WRAP_AABB),
            layout,
        }
    }

    /// Element-wise intersects `rays[i]` against `spheres[i]` (each `vec4`:
    /// `[cx, cy, cz, radius]`), mirroring [`prism_math::intersect::ray_sphere`].
    ///
    /// # Panics
    /// Panics if `rays.len() != spheres.len()`.
    #[must_use]
    pub fn cast_spheres(
        &self,
        ctx: &GpuContext,
        rays: &[GpuRay],
        spheres: &[[f32; 4]],
    ) -> Vec<GpuRayHit> {
        assert_eq!(rays.len(), spheres.len(), "rays and spheres must align");
        self.run(ctx, &self.sphere, rays, spheres)
    }

    /// Element-wise intersects `rays[i]` against `boxes[i]`, mirroring
    /// [`prism_math::intersect::ray_aabb`].
    ///
    /// # Panics
    /// Panics if `rays.len() != boxes.len()`.
    #[must_use]
    pub fn cast_aabbs(
        &self,
        ctx: &GpuContext,
        rays: &[GpuRay],
        boxes: &[GpuAabb],
    ) -> Vec<GpuRayHit> {
        assert_eq!(rays.len(), boxes.len(), "rays and boxes must align");
        self.run(ctx, &self.aabb, rays, boxes)
    }

    /// Uploads the count uniform + the ray batch + a `T` primitive batch,
    /// dispatches `pipeline`, and reads back one decoded hit per element.
    fn run<T: Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        rays: &[GpuRay],
        prims: &[T],
    ) -> Vec<GpuRayHit> {
        let n = rays.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_raycast_params",
            &CountParams {
                count: n as u32,
                _pad: [0; 3],
            },
        );
        let ray_buf = buffer::storage_read(device, "prism_math_raycast_rays", rays);
        let prim_buf = buffer::storage_read(device, "prism_math_raycast_prims", prims);
        let out_bytes = (n * size_of::<HitRaw>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_raycast_out", out_bytes);
        let bind_group = self.bind(device, &params, &ray_buf, &prim_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_raycast_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_raycast_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<HitRaw>(ctx, &stage)
            .into_iter()
            .map(GpuRayHit::from)
            .collect()
    }

    /// Builds the four-entry bind group (count uniform, rays, primitives, out).
    fn bind(
        &self,
        device: &Device,
        params: &Buffer,
        rays: &Buffer,
        prims: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_raycast_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: rays.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: prims.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Records a 1D batch dispatch covering `n` elements at [`WORKGROUP`] threads
/// per group.
fn dispatch(enc: &mut CommandEncoder, pipeline: &ComputePipeline, bind_group: &BindGroup, n: usize) {
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_raycast_pass"),
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
