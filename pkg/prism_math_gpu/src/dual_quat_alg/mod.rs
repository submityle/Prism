//! Host orchestration of the GPU dual-quaternion algebra kernel (§24.1 twin:
//! rigid-motion composition / blending on device).
//!
//! [`GpuDualQuatAlg`] batch-evaluates dual-quaternion operations **on a real
//! device**, mirroring the CPU [`DualQuat`](prism_math::DualQuat) operators and
//! methods: the composition product (`a * b`, apply `b` then `a`), the
//! conjugate, the inverse (equal to the conjugate for unit dual quaternions),
//! and normalization (restoring `|real| = 1`, `dot(real, dual) = 0`). The math
//! is **not** duplicated here: the kernel is composed at runtime by prefixing the
//! single-sourced fragments
//! [`WGSL_QUAT_ALG`](prism_math::shader_mirror::WGSL_QUAT_ALG) (for the
//! quaternion product) and
//! [`WGSL_DUAL_QUAT_ALG`](prism_math::shader_mirror::WGSL_DUAL_QUAT_ALG) ahead
//! of a thin compute wrapper, so the device operation cannot silently drift from
//! the CPU reference — it is literally the same text.
//!
//! # Tolerance / parity
//!
//! Conjugate and inverse are exact sign flips; `mul` and `normalize` are float
//! arithmetic (quaternion products / a `sqrt` + divide) and Metal compiles WGSL
//! under fast-math, so they agree with the CPU only within a small
//! absolute+relative epsilon. The parity suite covers both exact and float ops.
//! It skips gracefully when no adapter is available so it still passes on a
//! device-less CI image while running the full dispatch on a real `GPU`.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::{WGSL_DUAL_QUAT_ALG, WGSL_QUAT_ALG};
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

/// Compute wrapper applying the selected dual-quaternion op to each input pair.
/// Each element is four `vec4`: `a`'s real/dual then `b`'s real/dual (the unary
/// ops ignore `b`).
const WRAP: &str = "\
struct Params { count: u32, op: u32 };\n\
struct DqIn {\n\
    a_real: vec4<f32>,\n\
    a_dual: vec4<f32>,\n\
    b_real: vec4<f32>,\n\
    b_dual: vec4<f32>,\n\
};\n\
struct DqOut { real: vec4<f32>, dual: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<DqIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<DqOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    let a = PrismDualQuat(s.a_real, s.a_dual);\n\
    let b = PrismDualQuat(s.b_real, s.b_dual);\n\
    let r = prism_dq_op(params.op, a, b);\n\
    dst[i] = DqOut(r.real, r.dual);\n\
}\n";

/// Dual-quaternion-op selector, matching the `prism_dq_op` dispatcher op codes
/// in [`WGSL_DUAL_QUAT_ALG`] and the CPU [`DualQuat`](prism_math::DualQuat)
/// methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DualQuatOp {
    /// Composition `a * b` (apply `b` then `a`), matching `impl Mul`.
    Mul = 0,
    /// Quaternion-conjugate both parts.
    Conjugate = 1,
    /// Inverse (equal to the conjugate for unit dual quaternions).
    Inverse = 2,
    /// Normalize to the unit invariants.
    Normalize = 3,
}

/// Uniform block carrying the valid element count and the op code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// A real-device batched dual-quaternion algebra kernel, composed from the
/// single-sourced [`WGSL_QUAT_ALG`] + [`WGSL_DUAL_QUAT_ALG`] fragments.
pub struct GpuDualQuatAlg {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDualQuatAlg {
    /// Builds the dual-quaternion-algebra pipeline.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDualQuatAlg {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_dq_alg_layout"),
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
            label: Some("prism_math_dq_alg_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_QUAT_ALG);
        source.push('\n');
        source.push_str(WGSL_DUAL_QUAT_ALG);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_dq_alg"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_dq_alg"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDualQuatAlg { layout, pipeline }
    }

    /// Batch-applies `op` to each `[a_real, a_dual, b_real, b_dual]` pair,
    /// mirroring the matching CPU [`prism_math::DualQuat`] operation. The unary
    /// ops ignore the `b` lanes; pass any placeholder.
    #[must_use]
    pub fn map(
        &self,
        ctx: &GpuContext,
        op: DualQuatOp,
        pairs: &[[[f32; 4]; 4]],
    ) -> Vec<[[f32; 4]; 2]> {
        let n = pairs.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_dq_alg_in", pairs);
        let out_bytes = (n * size_of::<[[f32; 4]; 2]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_dq_alg_out", out_bytes);
        let params = buffer::uniform(
            device,
            "prism_math_dq_alg_params",
            &Params {
                count: n as u32,
                op: op as u32,
                _pad: [0; 2],
            },
        );
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_dq_alg_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_dq_alg_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[[f32; 4]; 2]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_dq_alg_bind_group"),
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
        label: Some("prism_math_dq_alg_pass"),
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

/// Device-free gate: the composed kernel (quaternion + dual-quaternion
/// fragments + wrapper) must be a valid WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernel_parses() {
        let src = alloc::format!(
            "{}\n{}\n{}",
            super::WGSL_QUAT_ALG,
            super::WGSL_DUAL_QUAT_ALG,
            super::WRAP
        );
        if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
            panic!("dual_quat_alg failed to parse: {err}\n--- source ---\n{src}");
        }
    }
}
