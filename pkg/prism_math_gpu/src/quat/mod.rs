//! Host orchestration of the quaternion-rotate compute kernel (§24.1 twin).
//!
//! [`GpuQuatRotate`] rotates a batch of `vec3` by a single unit quaternion on a
//! real device, mirroring the CPU reference
//! [`quat_rotate_vec3`](prism_math::shader_mirror::quat_rotate_vec3). The WGSL
//! rotation function is **not** duplicated here: the kernel source is composed
//! at runtime by prefixing the single-sourced fragment
//! [`WGSL_QUAT_ROTATE`](prism_math::shader_mirror::WGSL_QUAT_ROTATE) ahead of a
//! thin compute wrapper, so the device math cannot silently drift from the CPU
//! reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! Unlike the pure copy/compare kernels in the sibling twins, this kernel does
//! floating-point arithmetic (two cross products and two fused add/scales).
//! Metal compiles WGSL under fast-math, so the shader compiler is free to
//! contract `a + b * c` into a single `fma` and to reassociate, which rounds
//! differently from the CPU's separately-rounded multiply-then-add. The parity
//! tests therefore assert agreement within a small absolute+relative tolerance
//! (the §24.1 contract is explicitly a *tolerance* round-trip), not bit-exact
//! equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_QUAT_ROTATE;
use prism_math::Quat;
use prism_math::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Compute workgroup length (the kernel is `@workgroup_size(64, 1, 1)`).
pub const WORKGROUP: u32 = 64;

/// The compute wrapper appended after the single-sourced rotation fragment.
///
/// It references `prism_quat_rotate`, which is *only* defined by
/// [`WGSL_QUAT_ROTATE`]; the two are concatenated in [`GpuQuatRotate::new`].
const COMPUTE_WRAPPER: &str = "\
struct Params {\n\
    quat: vec4<f32>,\n\
    count: u32,\n\
    pad0: u32,\n\
    pad1: u32,\n\
    pad2: u32,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(64, 1, 1)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) {\n\
        return;\n\
    }\n\
    let v = src[i].xyz;\n\
    let r = prism_quat_rotate(params.quat, v);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// Uniform block shared with `Params` in the composed kernel. `quat` is `xyzw`
/// matching [`prism_math::shader_mirror::pack_quat`]; the three pads bring the
/// struct to the 32-byte, 16-byte-aligned size WGSL assigns it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    quat: [f32; 4],
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `Pod` mirror of a `vec4<f32>` storage element (16-byte stride, `xyz` used).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl GpuVec4 {
    fn from_vec3(v: Vec3) -> GpuVec4 {
        GpuVec4 {
            x: v.x,
            y: v.y,
            z: v.z,
            w: 0.0,
        }
    }

    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

/// Compiled quaternion-rotate pipeline and its bind-group layout.
pub struct GpuQuatRotate {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuQuatRotate {
    /// Compiles the quaternion-rotate kernel on `ctx`'s device, composing the
    /// shader from the single-sourced [`WGSL_QUAT_ROTATE`] fragment plus the
    /// compute wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuatRotate {
        let device = ctx.device();
        let mut source = String::with_capacity(WGSL_QUAT_ROTATE.len() + COMPUTE_WRAPPER.len() + 1);
        source.push_str(WGSL_QUAT_ROTATE);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_quat_rotate"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_quat_rotate_layout"),
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
            label: Some("prism_math_quat_rotate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_quat_rotate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuatRotate { pipeline, layout }
    }

    /// Rotates every vector in `vectors` by the unit quaternion `q` on the
    /// device, returning the rotated vectors in the same order.
    ///
    /// An empty input returns an empty `Vec` without dispatching.
    #[must_use]
    pub fn rotate(&self, ctx: &GpuContext, q: Quat, vectors: &[Vec3]) -> Vec<Vec3> {
        let count = vectors.len();
        if count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let src: Vec<GpuVec4> = vectors.iter().map(|&v| GpuVec4::from_vec3(v)).collect();
        let params = Params {
            quat: [q.x, q.y, q.z, q.w],
            count: u32::try_from(count).expect("vector count fits in u32"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (count * size_of::<GpuVec4>()) as u64;
        let params_buf = buffer::uniform(device, "prism_math_quat_rotate_params", &params);
        let src_buf = buffer::storage_read(device, "prism_math_quat_rotate_src", &src);
        let dst_buf = buffer::storage_rw_zeroed(device, "prism_math_quat_rotate_dst", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_quat_rotate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_quat_rotate_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_quat_rotate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.count.div_ceil(WORKGROUP), 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_quat_rotate_stage", out_bytes);
        buffer::copy(&mut enc, &dst_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let out_words = buffer::read_back::<GpuVec4>(ctx, &stage);
        out_words.iter().take(count).map(|&w| w.to_vec3()).collect()
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
