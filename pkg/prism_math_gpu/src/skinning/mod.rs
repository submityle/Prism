//! Host orchestration of the dual-quaternion skinning compute kernel (§24.1
//! twin).
//!
//! [`GpuDualQuatSkin`] runs a 4-influence dual-quaternion linear blend (`DLB`)
//! vertex transform **on a real device** — the GPU side of skeletal skinning —
//! and reads the transformed point back, mirroring the CPU path
//! [`prism_math::DualQuat::blend_weighted`] followed by
//! [`prism_math::DualQuat::transform_point3`]. The blend/normalize/transform
//! WGSL is **not** duplicated here: the kernel is composed at runtime by
//! prefixing the single-sourced fragments
//! [`WGSL_QUAT_ROTATE`](prism_math::shader_mirror::WGSL_QUAT_ROTATE) and
//! [`WGSL_DUAL_QUAT_SKIN`](prism_math::shader_mirror::WGSL_DUAL_QUAT_SKIN) ahead
//! of a thin compute wrapper, so the device skinning math cannot silently drift
//! from the CPU reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! The blend renormalizes (reciprocal square root) and composes Hamilton
//! products. Metal compiles WGSL under fast-math, so the shader compiler may use
//! a lower-precision `rsqrt` and contract/reassociate the multiply-adds,
//! rounding differently from the CPU. The parity tests therefore assert
//! agreement within a small absolute+relative tolerance (the §24.1 contract is
//! explicitly a *tolerance* round-trip), not bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::{WGSL_DUAL_QUAT_SKIN, WGSL_QUAT_ROTATE};
use prism_math::DualQuat;
use prism_math::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// The maximum bone influences the kernel blends per vertex: the standard AAA
/// skinning fan-in. A zero weight contributes nothing (matching the CPU
/// `w == 0` skip), so narrower fan-ins pad with zero weights.
pub const MAX_INFLUENCES: usize = 4;

/// One `(dual quaternion, weight)` skinning influence.
#[derive(Clone, Copy, Debug)]
pub struct Influence {
    /// The bone's rigid transform as a (not necessarily unit after blend) dual
    /// quaternion.
    pub transform: DualQuat,
    /// The vertex's blend weight for this bone.
    pub weight: f32,
}

/// The compute wrapper appended after the single-sourced fragments.
///
/// It unpacks the uniform block, calls `prism_dq_skin4` (defined only by
/// [`WGSL_DUAL_QUAT_SKIN`], which itself calls `prism_quat_rotate` from
/// [`WGSL_QUAT_ROTATE`]), and writes the transformed point to `dst`.
const COMPUTE_WRAPPER: &str = "\
struct Params {\n\
    r0: vec4<f32>,\n\
    d0: vec4<f32>,\n\
    r1: vec4<f32>,\n\
    d1: vec4<f32>,\n\
    r2: vec4<f32>,\n\
    d2: vec4<f32>,\n\
    r3: vec4<f32>,\n\
    d3: vec4<f32>,\n\
    weights: vec4<f32>,\n\
    point: vec4<f32>,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(1, 1, 1)\n\
fn main() {\n\
    let p = prism_dq_skin4(\n\
        params.r0, params.d0,\n\
        params.r1, params.d1,\n\
        params.r2, params.d2,\n\
        params.r3, params.d3,\n\
        params.weights, params.point.xyz);\n\
    dst[0] = vec4<f32>(p, 1.0);\n\
}\n";

/// Uniform block shared with `Params` in the composed kernel: four
/// `(real, dual)` bone pairs, the four blend weights, and the point to skin.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pairs: [[f32; 4]; 8],
    weights: [f32; 4],
    point: [f32; 4],
}

/// `Pod` mirror of the `vec4<f32>` storage element written back (16-byte
/// stride): the skinned point in `xyz`, `w = 1`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPoint {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

/// A real-device 4-influence dual-quaternion skinning vertex transform (§24.1).
pub struct GpuDualQuatSkin {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuDualQuatSkin {
    /// Builds the compute pipeline, composing the kernel from the single-sourced
    /// [`WGSL_QUAT_ROTATE`] + [`WGSL_DUAL_QUAT_SKIN`] fragments plus the thin
    /// [`COMPUTE_WRAPPER`].
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDualQuatSkin {
        let device = ctx.device();

        let mut source = String::new();
        source.push_str(WGSL_QUAT_ROTATE);
        source.push('\n');
        source.push_str(WGSL_DUAL_QUAT_SKIN);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_dq_skin"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_dq_skin_layout"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_dq_skin_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_dq_skin_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDualQuatSkin { pipeline, layout }
    }

    /// Blends up to [`MAX_INFLUENCES`] weighted bone dual quaternions on the
    /// device and returns the skinned position of `point`, mirroring
    /// [`DualQuat::blend_weighted`] + [`DualQuat::transform_point3`]. Fewer than
    /// four influences are padded with zero-weight identity bones.
    #[must_use]
    pub fn skin_point(&self, ctx: &GpuContext, influences: &[Influence], point: Vec3) -> Vec3 {
        let device = ctx.device();

        let mut pairs = [[0.0f32; 4]; 8];
        let mut weights = [0.0f32; 4];
        for (i, inf) in influences.iter().take(MAX_INFLUENCES).enumerate() {
            let real = inf.transform.real;
            let dual = inf.transform.dual;
            pairs[i * 2] = [real.x, real.y, real.z, real.w];
            pairs[i * 2 + 1] = [dual.x, dual.y, dual.z, dual.w];
            weights[i] = inf.weight;
        }
        let params = Params {
            pairs,
            weights,
            point: [point.x, point.y, point.z, 1.0],
        };

        let out_bytes = size_of::<GpuPoint>() as u64;
        let params_buf = buffer::uniform(device, "prism_math_dq_skin_params", &params);
        let dst_buf = buffer::storage_rw_zeroed(device, "prism_math_dq_skin_dst", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_dq_skin_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_dq_skin_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_dq_skin_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_dq_skin_stage", out_bytes);
        buffer::copy(&mut enc, &dst_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let out = buffer::read_back::<GpuPoint>(ctx, &stage);
        Vec3::new(out[0].x, out[0].y, out[0].z)
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
