//! Host orchestration of the linear-sRGB <-> `OkLab` perceptual color
//! conversion compute kernels (§24.1 twin; the GPU side of color grading and
//! perceptually-uniform gradient mixing).
//!
//! [`GpuOklab`] batch-converts packed `vec4<f32>` colors between linear-light
//! sRGB and `OkLab` **on a real device**, mirroring the CPU reference
//! [`prism_math::color::oklab::Oklaba`]. The conversion math is **not**
//! duplicated here: each kernel is composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_OKLAB`](prism_math::shader_mirror::WGSL_OKLAB) ahead of a thin compute
//! wrapper, so the device matrices cannot silently drift from the CPU
//! reference. The fourth lane (alpha) is carried through unchanged.
//!
//! # Parity contract (honest boundary)
//!
//! The matrix multiplies and the cube in the inverse direction are ordinary FMA
//! arithmetic. The forward direction needs a cube root, which WGSL lacks as a
//! built-in, so `prism_cbrt` is composed as `sign(x)·pow(|x|, 1/3)`; the CPU
//! reference uses `libm::cbrt`. The two therefore differ by a small `pow`
//! rounding error, a documented honest boundary, so the parity tests use an
//! absolute+relative tolerance rather than bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_OKLAB;
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

/// Compute wrapper that converts each linear-sRGB `vec4<f32>` to `OkLab`.
const WRAP_TO_OKLAB: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_linear_to_oklab(src[i]);\n\
}\n";

/// Compute wrapper that converts each `OkLab` `vec4<f32>` to linear-sRGB.
const WRAP_TO_LINEAR: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_oklab_to_linear(src[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the linear-sRGB <-> `OkLab` conversion path.
///
/// Build it once per device with [`GpuOklab::new`]; the two pipelines (forward
/// and inverse) and the shared bind-group layout are created up front and
/// reused across batches.
pub struct GpuOklab {
    to_oklab: ComputePipeline,
    to_linear: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuOklab {
    /// Compiles the two pipelines on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_OKLAB`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_oklab_layout"),
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
            label: Some("prism_math_oklab_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_OKLAB);
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
        GpuOklab {
            to_oklab: build("prism_math_oklab_to_oklab", WRAP_TO_OKLAB),
            to_linear: build("prism_math_oklab_to_linear", WRAP_TO_LINEAR),
            layout,
        }
    }

    /// Batch-converts each `[r, g, b, alpha]` linear-sRGB color to the `OkLab`
    /// quad `[l, a, b, alpha]` on the device, mirroring
    /// [`prism_math::color::oklab::Oklaba::from_linear`].
    #[must_use]
    pub fn linear_to_oklab(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.to_oklab, src)
    }

    /// Batch-converts each `OkLab` quad `[l, a, b, alpha]` back to the
    /// linear-sRGB color `[r, g, b, alpha]` on the device, mirroring
    /// [`prism_math::color::oklab::Oklaba::to_linear`].
    #[must_use]
    pub fn oklab_to_linear(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.to_linear, src)
    }

    /// Shared upload/dispatch/read-back for either conversion kernel.
    fn run(&self, ctx: &GpuContext, pipeline: &ComputePipeline, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_oklab_in", src);
        let out_bytes = size_of_val(src) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_oklab_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_oklab_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_oklab_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by both kernels.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_oklab_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_oklab_bind_group"),
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
        label: Some("prism_math_oklab_pass"),
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
