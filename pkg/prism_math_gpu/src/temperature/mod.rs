//! Host orchestration of the correlated-color-temperature -> linear-sRGB
//! compute kernel (§24.1 twin; the white-balance / sky-and-sun lighting path
//! that drives a blackbody Kelvin slider straight into linear-light color).
//!
//! [`GpuTemperature`] batch-converts a run of Kelvin scalars into packed
//! `vec4<f32>` linear-sRGB colors **on a real device**, mirroring the CPU
//! reference [`prism_math::color::LinearRgba::from_temperature`]. The Planckian
//! locus math is **not** duplicated here: the kernel is composed at runtime by
//! prefixing the single-sourced fragment
//! [`WGSL_TEMPERATURE`](prism_math::shader_mirror::WGSL_TEMPERATURE) ahead of a
//! thin compute wrapper, so the device spline cannot silently drift from the
//! CPU reference.
//!
//! # Parity contract (honest boundary)
//!
//! The conversion is a clamp, a piecewise cubic, two divides, and the XYZ ->
//! linear matrix: ordinary FMA arithmetic with no transcendental. The spline
//! branch cutoffs are on the exact clamped Kelvin input, so the GPU and CPU
//! always pick the same segment. Metal compiles WGSL under fast-math, so the
//! shader may round the last ULP differently; parity is therefore a tight
//! absolute+relative tolerance, not bit-exact equality. Out-of-gamut negative
//! components are clamped to `0` on both sides and the output alpha is a
//! constant `1.0`.
//!
//! This transform is one-way (a temperature cannot be uniquely recovered from a
//! color), so there is a single forward kernel and no inverse.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_TEMPERATURE;
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

/// Compute wrapper that maps each Kelvin scalar to a linear-sRGB `vec4<f32>`.
const WRAP_TO_LINEAR: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_temperature_to_linear(src[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the correlated-color-temperature -> linear-sRGB path.
///
/// Build it once per device with [`GpuTemperature::new`]; the single forward
/// pipeline and the shared bind-group layout are created up front and reused
/// across batches.
pub struct GpuTemperature {
    to_linear: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuTemperature {
    /// Compiles the forward pipeline on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_TEMPERATURE`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_temperature_layout"),
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
            label: Some("prism_math_temperature_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_TEMPERATURE);
        source.push('\n');
        source.push_str(WRAP_TO_LINEAR);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_temperature_to_linear"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let to_linear = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_temperature_to_linear"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTemperature { to_linear, layout }
    }

    /// Batch-converts each Kelvin scalar to a unit-luminance linear-sRGB quad
    /// `[r, g, b, 1.0]` on the device, mirroring
    /// [`prism_math::color::LinearRgba::from_temperature`].
    #[must_use]
    pub fn temperature_to_linear(&self, ctx: &GpuContext, kelvin: &[f32]) -> Vec<[f32; 4]> {
        let n = kelvin.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_temperature_in", kelvin);
        let out_bytes = (n * size_of::<[f32; 4]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_temperature_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_temperature_encoder"),
        });
        dispatch(&mut enc, &self.to_linear, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_temperature_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output).
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_temperature_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_temperature_bind_group"),
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
        label: Some("prism_math_temperature_pass"),
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
