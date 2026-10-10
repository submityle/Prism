//! Host orchestration of the order-3 spherical-harmonic evaluation compute
//! kernel (§24.1 twin).
//!
//! [`GpuSh3Eval`] reconstructs an order-3 (16-coefficient) real spherical
//! harmonic in a direction **on a real device** — the GPU side of diffuse-GI
//! probe reconstruction — and reads the scalar back, mirroring the CPU path
//! [`prism_math::Sh3::eval`]. The basis/accumulate WGSL is **not** duplicated
//! here: the kernel is composed at runtime by prefixing the single-sourced
//! fragment [`WGSL_SH3_EVAL`](prism_math::shader_mirror::WGSL_SH3_EVAL) ahead of
//! a thin compute wrapper, so the device evaluation cannot silently drift from
//! the CPU reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! The evaluation is a sum of 16 polynomial-weighted multiply-adds. Metal
//! compiles WGSL under fast-math, so the shader compiler may contract and
//! reassociate the multiply-adds, rounding differently from the CPU's ordered
//! `sum`. The parity tests therefore assert agreement within a small
//! absolute+relative tolerance (the §24.1 contract is explicitly a *tolerance*
//! round-trip), not bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_SH3_EVAL;
use prism_math::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// The spherical-harmonic order the kernel reconstructs: bands 0..=3, hence 16
/// coefficients. This matches [`prism_math::Sh3`], the AAA diffuse-irradiance
/// probe order.
pub const SH3_COEFFS: usize = 16;

/// The compute wrapper appended after the single-sourced fragment.
///
/// It unpacks the uniform block (four `vec4` chunks packing the 16 coefficients
/// plus the direction), rebuilds the flat `array<f32, 16>`, calls
/// `prism_sh3_eval` (defined only by [`WGSL_SH3_EVAL`]), and writes the scalar
/// result to `dst[0].x`.
const COMPUTE_WRAPPER: &str = "\
struct Params {\n\
    c0: vec4<f32>,\n\
    c1: vec4<f32>,\n\
    c2: vec4<f32>,\n\
    c3: vec4<f32>,\n\
    dir: vec4<f32>,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(1, 1, 1)\n\
fn main() {\n\
    let coeffs = array<f32, 16>(\n\
        params.c0.x, params.c0.y, params.c0.z, params.c0.w,\n\
        params.c1.x, params.c1.y, params.c1.z, params.c1.w,\n\
        params.c2.x, params.c2.y, params.c2.z, params.c2.w,\n\
        params.c3.x, params.c3.y, params.c3.z, params.c3.w);\n\
    let r = prism_sh3_eval(coeffs, params.dir.xyz);\n\
    dst[0] = vec4<f32>(r, 0.0, 0.0, 0.0);\n\
}\n";

/// Uniform block shared with `Params` in the composed kernel: the 16
/// coefficients packed into four `vec4` chunks and the evaluation direction.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    coeffs: [[f32; 4]; 4],
    dir: [f32; 4],
}

/// `Pod` mirror of the `vec4<f32>` storage element written back (16-byte
/// stride): the reconstructed scalar in `x`, the rest zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuScalar {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

/// A real-device order-3 spherical-harmonic evaluation (§24.1).
pub struct GpuSh3Eval {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuSh3Eval {
    /// Builds the compute pipeline, composing the kernel from the single-sourced
    /// [`WGSL_SH3_EVAL`] fragment plus the thin [`COMPUTE_WRAPPER`].
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSh3Eval {
        let device = ctx.device();

        let mut source = String::new();
        source.push_str(WGSL_SH3_EVAL);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_sh3_eval"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_sh3_eval_layout"),
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
            label: Some("prism_math_sh3_eval_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_sh3_eval_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSh3Eval { pipeline, layout }
    }

    /// Reconstructs the SH3 scalar value in direction `dir` on the device,
    /// mirroring [`prism_math::Sh3::eval`]. `coeffs` are the 16 band
    /// coefficients in the same `(l, m)` order as [`prism_math::Sh3`]; `dir`
    /// should be unit length (the basis polynomials assume it).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, coeffs: &[f32; SH3_COEFFS], dir: Vec3) -> f32 {
        let device = ctx.device();

        let mut packed = [[0.0f32; 4]; 4];
        for (chunk, src) in packed.iter_mut().zip(coeffs.chunks_exact(4)) {
            chunk.copy_from_slice(src);
        }
        let params = Params {
            coeffs: packed,
            dir: [dir.x, dir.y, dir.z, 0.0],
        };

        let out_bytes = size_of::<GpuScalar>() as u64;
        let params_buf = buffer::uniform(device, "prism_math_sh3_eval_params", &params);
        let dst_buf = buffer::storage_rw_zeroed(device, "prism_math_sh3_eval_dst", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_sh3_eval_bind_group"),
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
            label: Some("prism_math_sh3_eval_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_sh3_eval_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_sh3_eval_stage", out_bytes);
        buffer::copy(&mut enc, &dst_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let out = buffer::read_back::<GpuScalar>(ctx, &stage);
        out[0].x
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
