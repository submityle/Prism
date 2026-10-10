//! Host orchestration of the quaternion-interpolation compute kernel
//! (`slerp` / `nlerp`, §24.1 twin).
//!
//! [`GpuQuatInterp`] blends batches of quaternion pairs on a real device,
//! mirroring the CPU references [`prism_math::Quat::slerp`] and
//! [`prism_math::Quat::nlerp`]. The WGSL interpolation functions are **not**
//! duplicated here: the kernel source is composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_QUAT_INTERP`](prism_math::shader_mirror::WGSL_QUAT_INTERP) ahead of a
//! thin compute wrapper, so the device math cannot silently drift from the CPU
//! reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! Both kernels evaluate `normalize` (and `slerp` additionally `acos`/`sin`)
//! under Metal fast-math, which rounds differently from the CPU's `libm`
//! transcendentals and separately-rounded multiply-adds. The parity tests
//! therefore assert agreement within a small absolute+relative tolerance (the
//! §24.1 contract is explicitly a *tolerance* round-trip), not bit-exact
//! equality. Quaternion sign is canonicalized before comparison because
//! `q` and `-q` represent the same rotation.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_QUAT_INTERP;
use prism_math::Quat;
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

/// `mode` selector value for spherical interpolation.
const MODE_SLERP: u32 = 0;
/// `mode` selector value for normalized-linear interpolation.
const MODE_NLERP: u32 = 1;

/// The compute wrapper appended after the single-sourced interpolation
/// fragment. It references `prism_quat_slerp` / `prism_quat_nlerp`, which are
/// *only* defined by [`WGSL_QUAT_INTERP`]; the two are concatenated in
/// [`GpuQuatInterp::new`].
const COMPUTE_WRAPPER: &str = "\
struct Params {\n\
    count: u32,\n\
    mode: u32,\n\
    pad0: u32,\n\
    pad1: u32,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> a_in: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read> b_in: array<vec4<f32>>;\n\
@group(0) @binding(3) var<storage, read> t_in: array<f32>;\n\
@group(0) @binding(4) var<storage, read_write> out_q: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(64, 1, 1)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) {\n\
        return;\n\
    }\n\
    let a = a_in[i];\n\
    let b = b_in[i];\n\
    let t = t_in[i];\n\
    var r: vec4<f32>;\n\
    if (params.mode == 0u) {\n\
        r = prism_quat_slerp(a, b, t);\n\
    } else {\n\
        r = prism_quat_nlerp(a, b, t);\n\
    }\n\
    out_q[i] = r;\n\
}\n";

/// Uniform block shared with `Params` in the composed kernel. The three/one
/// pads bring the struct to the 16-byte-aligned size WGSL assigns it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    mode: u32,
    pad0: u32,
    pad1: u32,
}

/// `Pod` mirror of a `vec4<f32>` storage element (16-byte stride, `xyzw`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl GpuQuat {
    fn from_quat(q: Quat) -> GpuQuat {
        GpuQuat {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }

    fn to_quat(self) -> Quat {
        Quat::from_xyzw(self.x, self.y, self.z, self.w)
    }
}

/// Compiled quaternion-interpolation pipeline and its bind-group layout.
///
/// A single pipeline serves both `slerp` and `nlerp`; the mode is chosen by a
/// uniform at dispatch time.
pub struct GpuQuatInterp {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuQuatInterp {
    /// Compiles the quaternion-interpolation kernel on `ctx`'s device,
    /// composing the shader from the single-sourced [`WGSL_QUAT_INTERP`]
    /// fragment plus the compute wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuatInterp {
        let device = ctx.device();
        let mut source = String::with_capacity(WGSL_QUAT_INTERP.len() + COMPUTE_WRAPPER.len() + 1);
        source.push_str(WGSL_QUAT_INTERP);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_quat_interp"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_quat_interp_layout"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    4,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_quat_interp_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_quat_interp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuatInterp { pipeline, layout }
    }

    /// Spherically interpolates each `(a[i], b[i])` pair by `t[i]` on the
    /// device, mirroring [`Quat::slerp`](prism_math::Quat::slerp).
    #[must_use]
    pub fn slerp(&self, ctx: &GpuContext, a: &[Quat], b: &[Quat], t: &[f32]) -> Vec<Quat> {
        self.dispatch(ctx, MODE_SLERP, a, b, t)
    }

    /// Normalized-linearly interpolates each `(a[i], b[i])` pair by `t[i]` on
    /// the device, mirroring [`Quat::nlerp`](prism_math::Quat::nlerp).
    #[must_use]
    pub fn nlerp(&self, ctx: &GpuContext, a: &[Quat], b: &[Quat], t: &[f32]) -> Vec<Quat> {
        self.dispatch(ctx, MODE_NLERP, a, b, t)
    }

    /// Shared dispatch for both interpolation modes.
    ///
    /// # Panics
    ///
    /// Panics if `a`, `b`, and `t` do not all have the same length.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        mode: u32,
        a: &[Quat],
        b: &[Quat],
        t: &[f32],
    ) -> Vec<Quat> {
        assert!(
            a.len() == b.len() && b.len() == t.len(),
            "slerp/nlerp inputs must be equal length"
        );
        let count = a.len();
        if count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let a_pod: Vec<GpuQuat> = a.iter().map(|&q| GpuQuat::from_quat(q)).collect();
        let b_pod: Vec<GpuQuat> = b.iter().map(|&q| GpuQuat::from_quat(q)).collect();
        let params = Params {
            count: u32::try_from(count).expect("quat count fits in u32"),
            mode,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (count * size_of::<GpuQuat>()) as u64;
        let params_buf = buffer::uniform(device, "prism_math_quat_interp_params", &params);
        let a_buf = buffer::storage_read(device, "prism_math_quat_interp_a", &a_pod);
        let b_buf = buffer::storage_read(device, "prism_math_quat_interp_b", &b_pod);
        let t_buf = buffer::storage_read(device, "prism_math_quat_interp_t", t);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_quat_interp_out", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_quat_interp_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: t_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_quat_interp_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_quat_interp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.count.div_ceil(WORKGROUP), 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_quat_interp_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let out_words = buffer::read_back::<GpuQuat>(ctx, &stage);
        out_words.iter().take(count).map(|&q| q.to_quat()).collect()
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
