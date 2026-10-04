//! Host orchestration of the 8-bit-per-channel vertex-attribute pack/unpack
//! compute kernels (§24.1 twin; the `unorm8x4` / `snorm8x4` bandwidth path for
//! vertex colors, normals, and tangents).
//!
//! [`GpuPack8`] batch-folds `vec4<f32>` lanes into one `u32` of four 8-bit
//! channels **on a real device** — the canonical compact vertex-attribute
//! encoding — and widens them back, mirroring the CPU codec
//! [`prism_math::pack8`]. The quantizer math is **not** duplicated here: each
//! kernel is composed at runtime by prefixing the single-sourced fragment
//! [`WGSL_PACK8`](prism_math::shader_mirror::WGSL_PACK8) ahead of a thin compute
//! wrapper, so the device helpers (`pack4x8unorm` / `pack4x8snorm` and the
//! `unpack` inverses) cannot silently drift from the CPU reference.
//!
//! # Parity contract (honest boundary)
//!
//! WGSL defines the quantizers as `⌊0.5 + N·clamp(c)⌋`, the same rounding the
//! CPU reference uses, so for exactly-representable quantized inputs the packed
//! bytes match [`prism_math::pack8`] **byte-for-byte**; the parity tests assert
//! that on byte-exact sweeps. Arbitrary inputs may differ by at most one code
//! at a rounding tie (implementations may round halves differently), a
//! documented honest boundary checked with a one-code tolerance. The widening
//! (`unpack`) direction is exact on both sides.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_PACK8;
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

/// Compute wrapper that packs each `vec4<f32>` into one `unorm8x4` `u32`.
const WRAP_PACK_UNORM: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_pack_unorm4x8(src[i]);\n\
}\n";

/// Compute wrapper that packs each `vec4<f32>` into one `snorm8x4` `u32`.
const WRAP_PACK_SNORM: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_pack_snorm4x8(src[i]);\n\
}\n";

/// Compute wrapper that unpacks each `unorm8x4` `u32` into a `vec4<f32>`.
const WRAP_UNPACK_UNORM: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_unpack_unorm4x8(src[i]);\n\
}\n";

/// Compute wrapper that unpacks each `snorm8x4` `u32` into a `vec4<f32>`.
const WRAP_UNPACK_SNORM: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    dst[i] = prism_unpack_snorm4x8(src[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the `unorm8x4` / `snorm8x4` pack/unpack path.
///
/// Build it once per device with [`GpuPack8::new`]; the four pipelines (unorm
/// pack/unpack, snorm pack/unpack) and the shared bind-group layout are created
/// up front and reused across batches.
pub struct GpuPack8 {
    pack_unorm: ComputePipeline,
    pack_snorm: ComputePipeline,
    unpack_unorm: ComputePipeline,
    unpack_snorm: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuPack8 {
    /// Compiles the four pipelines on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_PACK8`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_pack8_layout"),
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
            label: Some("prism_math_pack8_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_PACK8);
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
        GpuPack8 {
            pack_unorm: build("prism_math_pack8_pack_unorm", WRAP_PACK_UNORM),
            pack_snorm: build("prism_math_pack8_pack_snorm", WRAP_PACK_SNORM),
            unpack_unorm: build("prism_math_pack8_unpack_unorm", WRAP_UNPACK_UNORM),
            unpack_snorm: build("prism_math_pack8_unpack_snorm", WRAP_UNPACK_SNORM),
            layout,
        }
    }

    /// Batch-packs each `[0, 1]` four-vector into one `unorm8x4` `u32` on the
    /// device, mirroring [`prism_math::pack8::pack_unorm4x8`].
    #[must_use]
    pub fn pack_unorm4x8(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<u32> {
        self.run(ctx, &self.pack_unorm, src)
    }

    /// Batch-packs each `[-1, 1]` four-vector into one `snorm8x4` `u32` on the
    /// device, mirroring [`prism_math::pack8::pack_snorm4x8`].
    #[must_use]
    pub fn pack_snorm4x8(&self, ctx: &GpuContext, src: &[[f32; 4]]) -> Vec<u32> {
        self.run(ctx, &self.pack_snorm, src)
    }

    /// Batch-unpacks each `unorm8x4` `u32` into a `[0, 1]` four-vector on the
    /// device, mirroring [`prism_math::pack8::unpack_unorm4x8`].
    #[must_use]
    pub fn unpack_unorm4x8(&self, ctx: &GpuContext, src: &[u32]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.unpack_unorm, src)
    }

    /// Batch-unpacks each `snorm8x4` `u32` into a `[-1, 1]` four-vector on the
    /// device, mirroring [`prism_math::pack8::unpack_snorm4x8`].
    #[must_use]
    pub fn unpack_snorm4x8(&self, ctx: &GpuContext, src: &[u32]) -> Vec<[f32; 4]> {
        self.run(ctx, &self.unpack_snorm, src)
    }

    /// Shared upload/dispatch/read-back for any of the four 1:1 kernels.
    fn run<In: Pod, Out: Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        src: &[In],
    ) -> Vec<Out> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_pack8_in", src);
        let out_bytes = (n * size_of::<Out>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_pack8_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_pack8_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_pack8_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<Out>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by all four kernels.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_pack8_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_pack8_bind_group"),
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
        label: Some("prism_math_pack8_pass"),
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
