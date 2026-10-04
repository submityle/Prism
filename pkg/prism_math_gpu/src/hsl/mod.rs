//! Host orchestration of the non-linear sRGB <-> HSL / HSV cylindrical color
//! conversion compute kernels (§24.1 twin; the color-picker / tint path used by
//! editor tooling and procedural palette generation).
//!
//! [`GpuHsl`] batch-converts packed `vec4<f32>` colors between non-linear sRGB
//! and the two cylindrical models **on a real device**, mirroring the CPU
//! reference [`prism_math::color::Hsla`] / [`prism_math::color::Hsva`]. The
//! conversion math is **not** duplicated here: each kernel is composed at
//! runtime by prefixing the single-sourced fragment
//! [`WGSL_HSL`](prism_math::shader_mirror::WGSL_HSL) ahead of a thin compute
//! wrapper, so the device hue decomposition cannot silently drift from the CPU
//! reference. The fourth lane (alpha) is carried through unchanged.
//!
//! # Parity contract (honest boundary)
//!
//! Both models are defined over non-linear sRGB, so the kernels take and return
//! the stored sRGB quad directly with no gamma step. The arithmetic is ordinary
//! FMA plus a `floor`-based Euclidean remainder; Metal compiles WGSL under
//! fast-math, so the shader may round the last ULP differently from the CPU.
//! Hue is additionally ill-conditioned near gray (chroma -> 0), where a tiny
//! operand difference can swing the reported angle. The parity tests therefore
//! lean on the sRGB -> model -> sRGB round-trip as the strong invariant and use
//! a loose per-axis tolerance for the forward/inverse checks, rather than
//! asserting bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_HSL;
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

/// Compute wrapper that converts each non-linear sRGB `vec4<f32>` to HSL.
const WRAP_HSL_FROM: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_hsl_from_srgb(src[i]);\n\
}\n";

/// Compute wrapper that converts each HSL `vec4<f32>` back to non-linear sRGB.
const WRAP_HSL_TO: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_hsl_to_srgb(src[i]);\n\
}\n";

/// Compute wrapper that converts each non-linear sRGB `vec4<f32>` to HSV.
const WRAP_HSV_FROM: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_hsv_from_srgb(src[i]);\n\
}\n";

/// Compute wrapper that converts each HSV `vec4<f32>` back to non-linear sRGB.
const WRAP_HSV_TO: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_hsv_to_srgb(src[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the non-linear sRGB <-> HSL / HSV conversion path.
///
/// Build it once per device with [`GpuHsl::new`]; the four pipelines (HSL and
/// HSV, each forward and inverse) and the shared bind-group layout are created
/// up front and reused across batches.
pub struct GpuHsl {
    hsl_from: ComputePipeline,
    hsl_to: ComputePipeline,
    hsv_from: ComputePipeline,
    hsv_to: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuHsl {
    /// Compiles the four pipelines on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_HSL`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_hsl_layout"),
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
            label: Some("prism_math_hsl_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_HSL);
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
        GpuHsl {
            hsl_from: build("prism_math_hsl_hsl_from", WRAP_HSL_FROM),
            hsl_to: build("prism_math_hsl_hsl_to", WRAP_HSL_TO),
            hsv_from: build("prism_math_hsl_hsv_from", WRAP_HSV_FROM),
            hsv_to: build("prism_math_hsl_hsv_to", WRAP_HSV_TO),
            layout,
        }
    }

    /// Batch-converts each non-linear sRGB `[r, g, b, alpha]` to the HSL quad
    /// `[hue, saturation, lightness, alpha]` on the device, mirroring
    /// [`prism_math::color::Hsla::from_srgb`].
    #[must_use]
    pub fn srgb_to_hsl(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.hsl_from, src)
    }

    /// Batch-converts each HSL quad `[hue, saturation, lightness, alpha]` back
    /// to non-linear sRGB `[r, g, b, alpha]` on the device, mirroring
    /// [`prism_math::color::Hsla::to_srgb`].
    #[must_use]
    pub fn hsl_to_srgb(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.hsl_to, src)
    }

    /// Batch-converts each non-linear sRGB `[r, g, b, alpha]` to the HSV quad
    /// `[hue, saturation, value, alpha]` on the device, mirroring
    /// [`prism_math::color::Hsva::from_srgb`].
    #[must_use]
    pub fn srgb_to_hsv(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.hsv_from, src)
    }

    /// Batch-converts each HSV quad `[hue, saturation, value, alpha]` back to
    /// non-linear sRGB `[r, g, b, alpha]` on the device, mirroring
    /// [`prism_math::color::Hsva::to_srgb`].
    #[must_use]
    pub fn hsv_to_srgb(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.hsv_to, src)
    }

    /// Shared upload/dispatch/read-back for any of the four conversion kernels.
    fn run(&self, ctx: &GpuContext, pipeline: &ComputePipeline, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_hsl_in", src);
        let out_bytes = size_of_val(src) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_hsl_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_hsl_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_hsl_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by all four kernels.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_hsl_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_hsl_bind_group"),
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
        label: Some("prism_math_hsl_pass"),
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
