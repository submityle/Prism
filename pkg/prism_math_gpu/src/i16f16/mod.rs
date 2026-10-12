//! Host orchestration of the GPU Q16.16 fixed-point kernels (§24.7 twin: the
//! deterministic fixed-point path on device).
//!
//! [`GpuI16F16`] batch-evaluates the saturating Q16.16 ops **on a real device**,
//! mirroring the CPU
//! [`I16F16`](prism_math::fixed::i16f16::I16F16) methods on the raw `i32` bits:
//! `add`/`sub`/`neg`/`abs` (i32 saturating), `min`/`max`, the rounding
//! saturating `mul`, the truncating saturating `div`, and the `to_int`/
//! `from_int` conversions. Integer fixed-point is the deterministic path, so the
//! device result is **bit-identical** to the CPU. The math is **not** duplicated
//! here: the kernel is composed at runtime from the shared
//! [`WGSL_U64`](prism_math::shader_mirror::WGSL_U64) emulation plus the
//! single-sourced
//! [`WGSL_I16F16`](prism_math::shader_mirror::WGSL_I16F16) fragment ahead of a
//! thin compute wrapper, so the device op cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Exact parity
//!
//! Every op is exact integer arithmetic (the `mul` widens to a 64-bit product
//! then rounds; the `div` is a 48-bit long division), so the parity suite
//! compares with **strict equality**. The suite skips gracefully when no adapter
//! is available; the device-free WGSL parse gate and a real `GPU` cover the rest.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use prism_math::shader_mirror::{WGSL_I16F16, WGSL_U64};
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

/// Compute wrapper applying the selected Q16.16 op to each `[a, b]` pair.
const WRAP: &str = "\
struct Params { count: u32, op: u32, _p0: u32, _p1: u32 };\n\
struct FxIn { a: i32, b: i32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<FxIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<i32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    var r = s.a;\n\
    if (params.op == 0u) { r = prism_i16f16_add(s.a, s.b); }\n\
    else if (params.op == 1u) { r = prism_i16f16_sub(s.a, s.b); }\n\
    else if (params.op == 2u) { r = prism_i16f16_mul(s.a, s.b); }\n\
    else if (params.op == 3u) { r = prism_i16f16_min(s.a, s.b); }\n\
    else if (params.op == 4u) { r = prism_i16f16_max(s.a, s.b); }\n\
    else if (params.op == 5u) { r = prism_i16f16_abs(s.a); }\n\
    else if (params.op == 6u) { r = prism_i16f16_neg(s.a); }\n\
    else if (params.op == 7u) { r = prism_i16f16_div(s.a, s.b); }\n\
    else if (params.op == 8u) { r = prism_i16f16_to_int(s.a); }\n\
    else { r = prism_i16f16_from_int(s.a); }\n\
    dst[i] = r;\n\
}\n";

/// Q16.16 op selector, matching the wrapper's op codes and the CPU
/// [`I16F16`](prism_math::fixed::i16f16::I16F16) methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum I16F16Op {
    /// Saturating addition.
    Add = 0,
    /// Saturating subtraction.
    Sub = 1,
    /// Saturating, round-to-nearest multiplication.
    Mul = 2,
    /// Minimum.
    Min = 3,
    /// Maximum.
    Max = 4,
    /// Saturating absolute value.
    Abs = 5,
    /// Saturating negation.
    Neg = 6,
    /// Saturating, truncating division (`a / b`; divide-by-zero saturates).
    Div = 7,
    /// Truncate toward zero to a whole integer (like Rust's `raw / 65536`, not
    /// an arithmetic shift, which would floor negative fractions).
    ToInt = 8,
    /// Build from a whole integer (`int << 16`, saturating). `a` is a plain
    /// integer, not Q16.16 bits.
    FromInt = 9,
}

/// Uniform block: count + op, padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// Real-device batched Q16.16 kernel, composed from the shared
/// [`WGSL_U64`](prism_math::shader_mirror::WGSL_U64) emulation plus the
/// single-sourced
/// [`WGSL_I16F16`](prism_math::shader_mirror::WGSL_I16F16) fragment.
pub struct GpuI16F16 {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuI16F16 {
    /// Builds the Q16.16 pipeline.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuI16F16 {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_i16f16_layout"),
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
            label: Some("prism_math_i16f16_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_U64);
        source.push('\n');
        source.push_str(WGSL_I16F16);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_i16f16"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_i16f16"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuI16F16 { layout, pipeline }
    }

    /// Batch-applies `op` to each `[a, b]` raw-bit pair, mirroring the matching
    /// CPU [`prism_math::fixed::i16f16::I16F16`] operation. The unary ops
    /// (`Abs`/`Neg`) use only `a`.
    #[must_use]
    pub fn apply(&self, ctx: &GpuContext, op: I16F16Op, pairs: &[[i32; 2]]) -> Vec<i32> {
        let n = pairs.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_i16f16_params",
            &Params {
                count: n as u32,
                op: op as u32,
                _pad: [0; 2],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_i16f16_in", pairs);
        let out_bytes = (n * size_of::<i32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_i16f16_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_i16f16_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_i16f16_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<i32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_i16f16_bind_group"),
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
        label: Some("prism_math_i16f16_pass"),
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

/// Device-free gate: the composed kernel (shared u64 emulation + fragment +
/// wrapper) must be a valid WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernel_parses() {
        let src = alloc::format!(
            "{}\n{}\n{}",
            super::WGSL_U64,
            super::WGSL_I16F16,
            super::WRAP
        );
        if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
            panic!("i16f16 failed to parse: {err}\n--- source ---\n{src}");
        }
    }
}
