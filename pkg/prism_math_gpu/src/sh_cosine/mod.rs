//! Host orchestration of the GPU SH clamped-cosine convolution kernel (§24.6
//! twin: diffuse-irradiance probe conversion on device).
//!
//! [`GpuShCosine`] batch-convolves radiance spherical-harmonic coefficient sets
//! into **diffuse-irradiance** sets **on a real device**, mirroring the CPU
//! [`prism_math::Sh2::convolve_cosine`] / [`prism_math::Sh3::convolve_cosine`].
//! Evaluating the result in a normal `n` gives the irradiance `E(n)`. The math
//! is **not** duplicated here: the kernel is composed at runtime by prefixing
//! the single-sourced fragment
//! [`WGSL_SH_COSINE`](prism_math::shader_mirror::WGSL_SH_COSINE) ahead of a thin
//! compute wrapper, so the device convolution cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Parity
//!
//! Each coefficient is a single multiply by the per-band factor `π` / `2π/3` /
//! `π/4` / `0` (band 3 vanishes, matching the CPU). A correctly-rounded single
//! multiply is exact, so the parity suite compares at a tight tolerance that
//! still rejects an operands/factor/band-split drift. It skips gracefully when
//! no adapter is available so it still passes on a device-less CI image while
//! running the full dispatch on a real `GPU`.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_SH_COSINE;
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

/// Coefficient count of the order-2 generator [`prism_math::Sh2`].
pub const SH2_COEFFS: usize = 9;

/// Coefficient count of the order-3 generator [`prism_math::Sh3`].
pub const SH3_COEFFS: usize = 16;

/// Compute wrapper convolving a flat batch of coefficient sets. `params.stride`
/// is the per-set coefficient count (9 for order 2, 16 for order 3).
const WRAP: &str = "\
struct Params { count: u32, stride: u32, _pad0: u32, _pad1: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<f32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let base = i * params.stride;\n\
    for (var j = 0u; j < params.stride; j = j + 1u) {\n\
        dst[base + j] = prism_sh_convolve_cosine(src[base + j], j);\n\
    }\n\
}\n";

/// Uniform block carrying the set count and the per-set stride, padded to the
/// 16-byte minimum WebGPU uniform binding size.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    stride: u32,
    _pad: [u32; 2],
}

/// A real-device batched SH cosine-convolution kernel, composed from the
/// single-sourced [`WGSL_SH_COSINE`] fragment.
pub struct GpuShCosine {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuShCosine {
    /// Builds the cosine-convolution pipeline from the single-sourced
    /// [`WGSL_SH_COSINE`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuShCosine {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_sh_cosine_layout"),
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
            label: Some("prism_math_sh_cosine_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_SH_COSINE);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_sh_cosine"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_sh_cosine"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuShCosine { layout, pipeline }
    }

    /// Convolves a batch of order-2 coefficient sets with the clamped-cosine
    /// kernel, mirroring [`prism_math::Sh2::convolve_cosine`] element-wise.
    #[must_use]
    pub fn convolve2(
        &self,
        ctx: &GpuContext,
        sets: &[[f32; SH2_COEFFS]],
    ) -> Vec<[f32; SH2_COEFFS]> {
        let flat = bytemuck::cast_slice::<[f32; SH2_COEFFS], f32>(sets);
        let out = self.run(ctx, flat, sets.len(), SH2_COEFFS as u32);
        out.chunks_exact(SH2_COEFFS)
            .map(|chunk| {
                let mut set = [0.0f32; SH2_COEFFS];
                set.copy_from_slice(chunk);
                set
            })
            .collect()
    }

    /// Convolves a batch of order-3 coefficient sets with the clamped-cosine
    /// kernel, mirroring [`prism_math::Sh3::convolve_cosine`] element-wise
    /// (band 3 vanishes).
    #[must_use]
    pub fn convolve3(
        &self,
        ctx: &GpuContext,
        sets: &[[f32; SH3_COEFFS]],
    ) -> Vec<[f32; SH3_COEFFS]> {
        let flat = bytemuck::cast_slice::<[f32; SH3_COEFFS], f32>(sets);
        let out = self.run(ctx, flat, sets.len(), SH3_COEFFS as u32);
        out.chunks_exact(SH3_COEFFS)
            .map(|chunk| {
                let mut set = [0.0f32; SH3_COEFFS];
                set.copy_from_slice(chunk);
                set
            })
            .collect()
    }

    /// Uploads a flat coefficient batch, dispatches one thread per set, and
    /// reads the convolved flat batch back.
    fn run(&self, ctx: &GpuContext, flat: &[f32], count: usize, stride: u32) -> Vec<f32> {
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_sh_cosine_params",
            &Params {
                count: crate::dispatch::checked_count(count),
                stride,
                _pad: [0; 2],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_sh_cosine_in", flat);
        let out_bytes = size_of_val(flat) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_sh_cosine_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_sh_cosine_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, count);
        let stage = buffer::staging(device, "prism_math_sh_cosine_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<f32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_sh_cosine_bind_group"),
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

/// Records a 1D batch dispatch covering `count` sets at [`WORKGROUP`] threads
/// per group.
fn dispatch(
    enc: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    count: usize,
) {
    let groups = crate::dispatch::checked_workgroup_count(count, WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_sh_cosine_pass"),
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

/// Device-free gate: the composed kernel (fragment + wrapper) must be a valid
/// WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernel_parses() {
        let src = alloc::format!("{}\n{}", super::WGSL_SH_COSINE, super::WRAP);
        if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
            panic!("sh_cosine failed to parse: {err}\n--- source ---\n{src}");
        }
    }
}
