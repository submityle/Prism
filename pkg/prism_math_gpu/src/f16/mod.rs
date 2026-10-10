//! Host orchestration of the half-precision (`binary16`) pack/unpack compute
//! kernels (§24.1 twin / §24.3 f16 bandwidth path).
//!
//! [`GpuF16Pack`] batch-folds pairs of `f32` lanes into one `u32` of two
//! packed `binary16` values **on a real device** — the canonical vertex-stream
//! / lightmap / G-buffer bandwidth primitive — and widens them back, mirroring
//! the CPU path [`prism_math::f16::F16::from_f32`] / [`to_f32`]. The pack math
//! is **not** duplicated here: each kernel is composed at runtime by prefixing
//! the single-sourced fragment
//! [`WGSL_F16`](prism_math::shader_mirror::WGSL_F16) ahead of a thin compute
//! wrapper, so the device helper (`pack2x16float` / `unpack2x16float`) cannot
//! silently drift from the CPU reference.
//!
//! # Parity contract (honest boundary)
//!
//! WGSL defines `pack2x16float` as round-to-nearest-even, exactly like the CPU
//! reference, so for finite values inside the `f16` **normal** range
//! (`|x|` in `[2^-14, 65504]`) the packed 16 bits match
//! [`F16::from_f32`](prism_math::f16::F16::from_f32) **bit-for-bit**. Two cases
//! are deliberately not asserted bit-exact: **subnormals** (`|x| < 2^-14`),
//! which Metal and other GPUs may flush to zero, and **overflow**
//! (`|x| > 65504`) plus `NaN`, which the WGSL spec leaves implementation-defined
//! for `pack2x16float`. The parity test therefore drives normal-range values
//! through pack (bit-exact) and the full `u16` space through unpack
//! (`binary16 -> f32` is exact on both sides).
//!
//! [`to_f32`]: prism_math::f16::F16::to_f32
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_F16;
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

/// Compute wrapper that packs each `vec2<f32>` lane pair into one `u32`.
const WRAP_PACK: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> pairs: array<vec2<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = pairs[i];\n\
    keys[i] = prism_f16_pack2(p.x, p.y);\n\
}\n";

/// Compute wrapper that unpacks each `u32` into a `vec2<f32>` lane pair.
const WRAP_UNPACK: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> keys: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> pairs: array<vec2<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    pairs[i] = prism_f16_unpack2(keys[i]);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the `binary16` pack/unpack path.
///
/// Build it once per device with [`GpuF16Pack::new`]; the two pipelines (pack,
/// unpack) and the shared bind-group layout are created up front and reused
/// across batches.
pub struct GpuF16Pack {
    pack: ComputePipeline,
    unpack: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuF16Pack {
    /// Compiles the pack/unpack pipelines on `device`, embedding the
    /// single-sourced [`WGSL_F16`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_f16_layout"),
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
            label: Some("prism_math_f16_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_F16);
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
        GpuF16Pack {
            pack: build("prism_math_f16_pack", WRAP_PACK),
            unpack: build("prism_math_f16_unpack", WRAP_UNPACK),
            layout,
        }
    }

    /// Batch-packs each `[a, b]` lane pair into one `u32` holding two
    /// `binary16` values (low 16 bits = `a`, high 16 bits = `b`) on the device,
    /// mirroring [`prism_math::f16::F16::from_f32`].
    #[must_use]
    pub fn pack2(&self, ctx: &GpuContext, pairs: &[[f32; 2]]) -> Vec<u32> {
        let n = pairs.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_f16_in", pairs);
        let out_bytes = (n * size_of::<u32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_f16_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_f16_pack_encoder"),
        });
        dispatch(&mut enc, &self.pack, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_f16_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<u32>(ctx, &stage)
    }

    /// Batch-unpacks each `u32` into its `[a, b]` lane pair on the device,
    /// mirroring [`prism_math::f16::F16::to_f32`].
    #[must_use]
    pub fn unpack2(&self, ctx: &GpuContext, keys: &[u32]) -> Vec<[f32; 2]> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_f16_in", keys);
        let out_bytes = (n * size_of::<[f32; 2]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_f16_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_f16_unpack_encoder"),
        });
        dispatch(&mut enc, &self.unpack, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_f16_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 2]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by both kernels.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_f16_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_f16_bind_group"),
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
        label: Some("prism_math_f16_pass"),
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
