//! Host orchestration of the linear-sRGB <-> CIE 1931 XYZ (D65) conversion
//! compute kernels (§24.1 twin; the GPU side of the device-independent color
//! hub used for white-point adaptation and cross-gamut bridging).
//!
//! [`GpuXyz`] batch-converts packed `vec4<f32>` colors between linear-light
//! sRGB and CIE XYZ **on a real device**, mirroring the CPU reference
//! [`prism_math::color::LinearRgba::to_xyz`] /
//! [`prism_math::color::LinearRgba::from_xyz`]. The conversion math is **not**
//! duplicated here: each kernel is composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_XYZ`](prism_math::shader_mirror::WGSL_XYZ) ahead of a thin compute
//! wrapper, so the device matrices cannot silently drift from the CPU
//! reference. The fourth lane (alpha) is carried through unchanged.
//!
//! # Parity contract (honest boundary)
//!
//! Both directions are a single 3x3 matrix multiply, ordinary FMA arithmetic
//! with no transcendental. Metal compiles WGSL under fast-math, so the shader
//! compiler may contract and reassociate the multiply-adds and round the last
//! ULP differently from the CPU. The parity tests therefore use a tight
//! absolute+relative tolerance that only absorbs that last-ULP rounding rather
//! than asserting bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_XYZ;
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

/// Compute wrapper that converts each linear-sRGB `vec4<f32>` to CIE XYZ.
const WRAP_TO_XYZ: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_linear_to_xyz(src[i]);\n\
}\n";

/// Compute wrapper that converts each CIE XYZ `vec4<f32>` back to linear-sRGB.
const WRAP_TO_LINEAR: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_xyz_to_linear(src[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the linear-sRGB <-> CIE 1931 XYZ (D65) conversion path.
///
/// Build it once per device with [`GpuXyz::new`]; the two pipelines (forward
/// and inverse) and the shared bind-group layout are created up front and
/// reused across batches.
pub struct GpuXyz {
    to_xyz: ComputePipeline,
    to_linear: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuXyz {
    /// Compiles the two pipelines on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_XYZ`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_xyz_layout"),
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
            label: Some("prism_math_xyz_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_XYZ);
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
        GpuXyz {
            to_xyz: build("prism_math_xyz_to_xyz", WRAP_TO_XYZ),
            to_linear: build("prism_math_xyz_to_linear", WRAP_TO_LINEAR),
            layout,
        }
    }

    /// Batch-converts each `[r, g, b, alpha]` linear-sRGB color to the CIE XYZ
    /// quad `[x, y, z, alpha]` on the device, mirroring
    /// [`prism_math::color::LinearRgba::to_xyz`].
    #[must_use]
    pub fn linear_to_xyz(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.to_xyz, src)
    }

    /// Batch-converts each CIE XYZ quad `[x, y, z, alpha]` back to the
    /// linear-sRGB color `[r, g, b, alpha]` on the device, mirroring
    /// [`prism_math::color::LinearRgba::from_xyz`].
    #[must_use]
    pub fn xyz_to_linear(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.to_linear, src)
    }

    /// Shared upload/dispatch/read-back for either conversion kernel.
    fn run(&self, ctx: &GpuContext, pipeline: &ComputePipeline, src: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_xyz_in", src);
        let out_bytes = size_of_val(src) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_xyz_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_xyz_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_xyz_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by both kernels.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_xyz_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_xyz_bind_group"),
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
        label: Some("prism_math_xyz_pass"),
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
