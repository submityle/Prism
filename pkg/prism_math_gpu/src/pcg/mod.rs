//! Host orchestration of the GPU `PCG32` (XSH-RR) generator kernel
//! (§24.7 twin: the same deterministic stream on device and host).
//!
//! [`GpuPcg32`] element-wise reproduces the first 32-bit draw of
//! [`prism_math::rng::Pcg32::with_stream`] **on a real device**: each input
//! carries a `(seed, stream)` pair, and the kernel runs the PCG seeding ritual
//! followed by the XSH-RR output permutation. The math is **not** duplicated
//! here: the kernel is composed at runtime by prefixing the shared
//! [`WGSL_U64`](prism_math::shader_mirror::WGSL_U64) emulation plus the
//! single-sourced generator fragment
//! [`WGSL_PCG32`](prism_math::shader_mirror::WGSL_PCG32) ahead of a thin compute
//! wrapper, so the device draw cannot silently drift from the CPU reference —
//! it is literally the same text.
//!
//! # Exact parity
//!
//! `PCG32` is 64-bit LCG state with a 32-bit output. WGSL has no `u64`, so the
//! kernel carries the state as a `(lo, hi)` `u32` lane pair and emulates the
//! 64-bit multiply with 32×32 partial products. This is **exact** integer
//! arithmetic — no floating point, no fast-math — so the device output is
//! bit-identical to the CPU for every `(seed, stream)`. The parity suite
//! compares with strict equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::{WGSL_PCG32, WGSL_U64};
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

/// Compute wrapper: one `(seed, stream)` pair per element, one `u32` out.
const WRAP: &str = "\
struct Params { count: u32 };\n\
struct SeedIn { seed_lo: u32, seed_hi: u32, stream_lo: u32, stream_hi: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> seeds: array<SeedIn>;\n\
@group(0) @binding(2) var<storage, read_write> out: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = seeds[i];\n\
    out[i] = prism_pcg32_first(s.seed_lo, s.seed_hi, s.stream_lo, s.stream_hi);\n\
}\n";

/// Uniform block: the batch element count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    count: u32,
    _pad: [u32; 3],
}

/// A `PCG32` seed/stream selector, packed as little-endian `u32` lane pairs to
/// match the WGSL `SeedIn` storage layout exactly (16 bytes, no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PcgSeed {
    /// Low 32 bits of the seed.
    pub seed_lo: u32,
    /// High 32 bits of the seed.
    pub seed_hi: u32,
    /// Low 32 bits of the stream selector.
    pub stream_lo: u32,
    /// High 32 bits of the stream selector.
    pub stream_hi: u32,
}

impl PcgSeed {
    /// Builds a seed selector from 64-bit `seed` and `stream` values, matching
    /// the CPU [`Pcg32::with_stream`](prism_math::rng::Pcg32::with_stream).
    #[must_use]
    pub fn new(seed: u64, stream: u64) -> PcgSeed {
        PcgSeed {
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            stream_lo: stream as u32,
            stream_hi: (stream >> 32) as u32,
        }
    }
}

/// A real-device batched `PCG32` first-draw kernel, composed from the
/// single-sourced [`WGSL_PCG32`] fragment.
pub struct GpuPcg32 {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPcg32 {
    /// Builds the PCG32 pipeline from the single-sourced [`WGSL_PCG32`]
    /// fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPcg32 {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_pcg_layout"),
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
            label: Some("prism_math_pcg_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_U64);
        source.push('\n');
        source.push_str(WGSL_PCG32);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_pcg"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_pcg"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPcg32 { layout, pipeline }
    }

    /// Element-wise reproduces the first 32-bit draw of
    /// [`prism_math::rng::Pcg32::with_stream`] for each seed selector,
    /// bit-identical to the CPU.
    #[must_use]
    pub fn first_u32(&self, ctx: &GpuContext, seeds: &[PcgSeed]) -> Vec<u32> {
        let n = seeds.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_pcg_params",
            &CountParams {
                count: n as u32,
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_pcg_in", seeds);
        let out_bytes = (n * size_of::<u32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_pcg_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_pcg_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_pcg_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<u32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, seeds, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_pcg_bind_group"),
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
        label: Some("prism_math_pcg_pass"),
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

/// Device-free gate: the composed kernel (shared u64 emulation + generator
/// fragment + wrapper) must be a valid WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernel_parses() {
        let src = alloc::format!(
            "{}\n{}\n{}",
            super::WGSL_U64,
            super::WGSL_PCG32,
            super::WRAP
        );
        if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
            panic!("pcg failed to parse: {err}\n--- source ---\n{src}");
        }
    }
}
