//! Host orchestration of the GPU 64-bit generator kernels `SplitMix64` and
//! `xoshiro256**` (§24.7 twin: the same deterministic stream on device and
//! host).
//!
//! [`GpuRng64`] element-wise reproduces the first `u64` draw of
//! [`prism_math::rng::SplitMix64::new`] and
//! [`prism_math::rng::Xoshiro256StarStar::new`] **on a real device**: each input
//! carries a `u64` seed, and the kernel runs the generator's seeding + first
//! step. The math is **not** duplicated here: the kernels are composed at
//! runtime by prefixing the shared
//! [`WGSL_U64`](prism_math::shader_mirror::WGSL_U64) emulation plus the
//! single-sourced generator fragments
//! [`WGSL_SPLITMIX64`](prism_math::shader_mirror::WGSL_SPLITMIX64) /
//! [`WGSL_XOSHIRO256SS`](prism_math::shader_mirror::WGSL_XOSHIRO256SS) ahead of
//! thin compute wrappers, so the device draw cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Exact parity
//!
//! Both generators are 64-bit integer state. WGSL has no `u64`, so the kernels
//! carry the state as `(lo, hi)` `u32` lane pairs and emulate the 64-bit
//! arithmetic (multiply, add, shift, rotate) with 32×32 partial products. This
//! is **exact** integer arithmetic — no floating point, no fast-math — so the
//! device draws are bit-identical to the CPU. The parity suite compares with
//! strict equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::{WGSL_SPLITMIX64, WGSL_U64, WGSL_XOSHIRO256SS};
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

/// Shared input/output declarations for the 64-bit generator wrappers: one
/// `u64` seed in, one `u64` draw out, both as `(lo, hi)` lane pairs.
const IO: &str = "\
struct Params { count: u32 };\n\
struct SeedIn { lo: u32, hi: u32 };\n\
struct U64Out { lo: u32, hi: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> seeds: array<SeedIn>;\n\
@group(0) @binding(2) var<storage, read_write> out: array<U64Out>;\n";

/// `SplitMix64` wrapper: one seed in, `prism_splitmix64_first` out.
const WRAP_SPLITMIX: &str = "\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = seeds[i];\n\
    let r = prism_splitmix64_first(s.lo, s.hi);\n\
    out[i] = U64Out(r.x, r.y);\n\
}\n";

/// `xoshiro256**` wrapper: one seed in, `prism_xoshiro256ss_first` out.
const WRAP_XOSHIRO: &str = "\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = seeds[i];\n\
    let r = prism_xoshiro256ss_first(s.lo, s.hi);\n\
    out[i] = U64Out(r.x, r.y);\n\
}\n";

/// Uniform block: the batch element count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    count: u32,
    _pad: [u32; 3],
}

/// A `u64` seed packed as little-endian `u32` lane pairs to match the WGSL
/// `SeedIn` storage layout exactly (8 bytes, no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Seed64 {
    /// Low 32 bits of the seed.
    pub lo: u32,
    /// High 32 bits of the seed.
    pub hi: u32,
}

impl Seed64 {
    /// Builds a seed from a 64-bit value.
    #[must_use]
    pub fn new(seed: u64) -> Seed64 {
        Seed64 {
            lo: seed as u32,
            hi: (seed >> 32) as u32,
        }
    }
}

/// Raw `std430` `u64` draw read back from the device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct U64Out {
    lo: u32,
    hi: u32,
}

impl U64Out {
    /// Recombines the lane pair into a `u64`.
    fn value(self) -> u64 {
        u64::from(self.lo) | (u64::from(self.hi) << 32)
    }
}

/// A real-device batched 64-bit generator pair (`SplitMix64` + `xoshiro256**`),
/// each composed from the shared [`WGSL_U64`] emulation and its generator
/// fragment.
pub struct GpuRng64 {
    layout: BindGroupLayout,
    splitmix: ComputePipeline,
    xoshiro: ComputePipeline,
}

impl GpuRng64 {
    /// Builds the two generator pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRng64 {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_rng64_layout"),
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
            label: Some("prism_math_rng64_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, fragments: &[&str], wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            for fragment in fragments {
                source.push_str(fragment);
                source.push('\n');
            }
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
        let splitmix = build(
            "prism_math_splitmix64",
            &[WGSL_U64, WGSL_SPLITMIX64, IO],
            WRAP_SPLITMIX,
        );
        let xoshiro = build(
            "prism_math_xoshiro256ss",
            &[WGSL_U64, WGSL_SPLITMIX64, WGSL_XOSHIRO256SS, IO],
            WRAP_XOSHIRO,
        );
        GpuRng64 {
            layout,
            splitmix,
            xoshiro,
        }
    }

    /// Element-wise reproduces the first `u64` draw of
    /// [`prism_math::rng::SplitMix64::new`], bit-identical to the CPU.
    #[must_use]
    pub fn splitmix64_first(&self, ctx: &GpuContext, seeds: &[Seed64]) -> Vec<u64> {
        self.run(ctx, &self.splitmix, seeds)
    }

    /// Element-wise reproduces the first `u64` draw of
    /// [`prism_math::rng::Xoshiro256StarStar::new`], bit-identical to the CPU.
    #[must_use]
    pub fn xoshiro256ss_first(&self, ctx: &GpuContext, seeds: &[Seed64]) -> Vec<u64> {
        self.run(ctx, &self.xoshiro, seeds)
    }

    /// Uploads the count uniform + seed batch, dispatches `pipeline`, and reads
    /// back one `u64` draw per element.
    fn run(&self, ctx: &GpuContext, pipeline: &ComputePipeline, seeds: &[Seed64]) -> Vec<u64> {
        let n = seeds.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_rng64_params",
            &CountParams {
                count: crate::dispatch::checked_count(n),
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_rng64_in", seeds);
        let out_bytes = (n * size_of::<U64Out>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_rng64_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_rng64_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_rng64_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<U64Out>(ctx, &stage)
            .into_iter()
            .map(U64Out::value)
            .collect()
    }

    /// Builds the three-entry bind group (count uniform, seeds, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_rng64_bind_group"),
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
    let groups = crate::dispatch::checked_workgroup_count(n, WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_rng64_pass"),
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

/// Device-free gate: every composed kernel (shared u64 emulation + generator
/// fragment(s) + I/O + wrapper) must be a valid WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernels_parse() {
        let splitmix = alloc::format!(
            "{}\n{}\n{}\n{}",
            super::WGSL_U64,
            super::WGSL_SPLITMIX64,
            super::IO,
            super::WRAP_SPLITMIX
        );
        let xoshiro = alloc::format!(
            "{}\n{}\n{}\n{}\n{}",
            super::WGSL_U64,
            super::WGSL_SPLITMIX64,
            super::WGSL_XOSHIRO256SS,
            super::IO,
            super::WRAP_XOSHIRO
        );
        for (label, src) in [("splitmix", splitmix), ("xoshiro", xoshiro)] {
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("rng64 `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
