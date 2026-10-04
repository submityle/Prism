//! Host orchestration of the sRGB electro-optical transfer-function compute
//! kernels (§24.1 twin; the gamma encode/decode path renderers and UI depend
//! on for correct linear-light blending).
//!
//! [`GpuSrgbTransfer`] batch-decodes non-linear sRGB components to linear light
//! and encodes the inverse **on a real device**, mirroring the CPU exact
//! piecewise path [`prism_math::color::transfer::srgb_to_linear`] /
//! [`linear_to_srgb`]. The curve math is **not** duplicated here: each kernel
//! is composed at runtime by prefixing the single-sourced fragment
//! [`WGSL_SRGB`](prism_math::shader_mirror::WGSL_SRGB) ahead of a thin compute
//! wrapper, so the device helpers (`prism_srgb_to_linear` /
//! `prism_linear_to_srgb`) cannot silently drift from the CPU reference.
//!
//! # Parity contract (honest boundary)
//!
//! Both sides take the same branch for a given input because the breakpoint
//! literals are identical. The linear segment is a bare multiply/divide and
//! matches to tolerance. The power segment calls the WGSL `pow` builtin, which
//! Metal compiles under fast-math, whereas the CPU reference uses deterministic
//! `libm::powf`; the §24.1 contract on that segment is therefore a small
//! absolute+relative tolerance (`1e-5`) rather than a bit contract. The
//! piecewise curve is continuous at the breakpoint, so inputs that straddle it
//! stay within tolerance regardless of branch selection.
//!
//! [`linear_to_srgb`]: prism_math::color::transfer::linear_to_srgb
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_SRGB;
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

/// Compute wrapper that decodes each non-linear sRGB component to linear light.
const WRAP_DECODE: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_srgb_to_linear(src[i]);\n\
}\n";

/// Compute wrapper that encodes each linear-light component to non-linear sRGB.
const WRAP_ENCODE: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_linear_to_srgb(src[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the sRGB transfer-function path.
///
/// Build it once per device with [`GpuSrgbTransfer::new`]; the two pipelines
/// (decode, encode) and the shared bind-group layout are created up front and
/// reused across batches.
pub struct GpuSrgbTransfer {
    decode: ComputePipeline,
    encode: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuSrgbTransfer {
    /// Compiles the decode/encode pipelines on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_SRGB`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_srgb_layout"),
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
            label: Some("prism_math_srgb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_SRGB);
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
        GpuSrgbTransfer {
            decode: build("prism_math_srgb_decode", WRAP_DECODE),
            encode: build("prism_math_srgb_encode", WRAP_ENCODE),
            layout,
        }
    }

    /// Batch-decodes each non-linear sRGB component to linear light on the
    /// device, mirroring [`prism_math::color::transfer::srgb_to_linear`].
    #[must_use]
    pub fn srgb_to_linear(&self, ctx: &GpuContext, src: &[f32]) -> Vec<f32> {
        self.run(ctx, &self.decode, src)
    }

    /// Batch-encodes each linear-light component to non-linear sRGB on the
    /// device, mirroring [`prism_math::color::transfer::linear_to_srgb`].
    #[must_use]
    pub fn linear_to_srgb(&self, ctx: &GpuContext, src: &[f32]) -> Vec<f32> {
        self.run(ctx, &self.encode, src)
    }

    /// Shared upload/dispatch/read-back for either direction.
    fn run(&self, ctx: &GpuContext, pipeline: &ComputePipeline, src: &[f32]) -> Vec<f32> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_srgb_in", src);
        let out_bytes = size_of_val(src) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_srgb_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_srgb_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_srgb_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<f32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by both kernels.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_srgb_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_srgb_bind_group"),
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
fn dispatch(enc: &mut CommandEncoder, pipeline: &ComputePipeline, bind_group: &BindGroup, n: usize) {
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_srgb_pass"),
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
