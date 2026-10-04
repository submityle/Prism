//! Host orchestration of the Morton (Z-order) spatial-key compute kernels
//! (§24.1 twin / §24.7 spatial encoding).
//!
//! [`GpuMorton`] batch-encodes integer lattice points into Morton (Z-order)
//! sort keys **on a real device** — the canonical GPU radix-sort / BVH-build
//! primitive — and decodes them back, mirroring the CPU path
//! [`prism_math::spatial::morton_encode2`] / [`morton_encode3`] and their
//! inverses. The bit-interleave WGSL is **not** duplicated here: each kernel is
//! composed at runtime by prefixing the single-sourced fragment
//! [`WGSL_MORTON`](prism_math::shader_mirror::WGSL_MORTON) ahead of a thin
//! compute wrapper, so the device key math cannot silently drift from the CPU
//! reference — they are literally the same text.
//!
//! # Exact parity (not a tolerance)
//!
//! Unlike the floating-point twins, Morton keys are pure integer bit
//! manipulation, so the §24.1 round-trip here is **bit-exact**, not a
//! tolerance. WGSL has no 64-bit integer type, so the kernels run at the
//! GPU-representable key widths — 16 bits per axis for the 2D encoder (32-bit
//! key) and 10 bits per axis for the 3D encoder (30-bit key) — which are
//! exactly the widths used for on-device sort keys. Because interleaving is
//! bit-local, for inputs masked to those axis widths the GPU key equals the
//! low bits of the wider CPU encoder bit-for-bit.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_MORTON;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Axis bit width consumed by the 2D GPU encoder (full 32-bit key).
pub const BITS_2D: u32 = 16;
/// Inclusive maximum per-axis coordinate accepted by [`GpuMorton::encode2`].
pub const MAX_COORD_2D: u32 = (1 << BITS_2D) - 1;
/// Axis bit width consumed by the 3D GPU encoder (30-bit key).
pub const BITS_3D: u32 = 10;
/// Inclusive maximum per-axis coordinate accepted by [`GpuMorton::encode3`].
pub const MAX_COORD_3D: u32 = (1 << BITS_3D) - 1;

/// Threads per workgroup; a standard 1D batch tiling.
const WORKGROUP: u32 = 64;

/// Compute wrapper for 2D Morton encoding.
const WRAP_ENCODE2: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> coords: array<vec4<u32>>;\n\
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let c = coords[i];\n\
    keys[i] = prism_morton_encode2(c.x, c.y);\n\
}\n";

/// Compute wrapper for 2D Morton decoding.
const WRAP_DECODE2: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> keys: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> coords: array<vec4<u32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let d = prism_morton_decode2(keys[i]);\n\
    coords[i] = vec4<u32>(d.x, d.y, 0u, 0u);\n\
}\n";

/// Compute wrapper for 3D Morton encoding.
const WRAP_ENCODE3: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> coords: array<vec4<u32>>;\n\
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let c = coords[i];\n\
    keys[i] = prism_morton_encode3(c.x, c.y, c.z);\n\
}\n";

/// Compute wrapper for 3D Morton decoding.
const WRAP_DECODE3: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> keys: array<u32>;\n\
@group(0) @binding(2) var<storage, read_write> coords: array<vec4<u32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let d = prism_morton_decode3(keys[i]);\n\
    coords[i] = vec4<u32>(d.x, d.y, d.z, 0u);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// A real-device batch Morton (Z-order) encoder/decoder (§24.1 / §24.7).
pub struct GpuMorton {
    layout: BindGroupLayout,
    encode2: ComputePipeline,
    decode2: ComputePipeline,
    encode3: ComputePipeline,
    decode3: ComputePipeline,
}

impl GpuMorton {
    /// Builds the four Morton pipelines (2D/3D encode & decode), each composed
    /// from the single-sourced [`WGSL_MORTON`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMorton {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_morton_layout"),
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
            label: Some("prism_math_morton_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_MORTON);
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
        GpuMorton {
            encode2: build("prism_math_morton_encode2", WRAP_ENCODE2),
            decode2: build("prism_math_morton_decode2", WRAP_DECODE2),
            encode3: build("prism_math_morton_encode3", WRAP_ENCODE3),
            decode3: build("prism_math_morton_decode3", WRAP_DECODE3),
            layout,
        }
    }

    /// Batch-encodes `(x, y)` lattice points into 2D Morton keys on the device,
    /// mirroring [`prism_math::spatial::morton_encode2`] at 16 bits per axis.
    #[must_use]
    pub fn encode2(&self, ctx: &GpuContext, coords: &[[u32; 2]]) -> Vec<u32> {
        let padded: Vec<[u32; 4]> = coords.iter().map(|c| [c[0], c[1], 0, 0]).collect();
        self.run_encode(ctx, &self.encode2, &padded)
    }

    /// Batch-decodes 2D Morton keys back into `(x, y)` lattice points, mirroring
    /// [`prism_math::spatial::morton_decode2`].
    #[must_use]
    pub fn decode2(&self, ctx: &GpuContext, keys: &[u32]) -> Vec<[u32; 2]> {
        self.run_decode(ctx, &self.decode2, keys)
            .into_iter()
            .map(|c| [c[0], c[1]])
            .collect()
    }

    /// Batch-encodes `(x, y, z)` lattice points into 3D Morton keys on the
    /// device, mirroring [`prism_math::spatial::morton_encode3`] at 10 bits per
    /// axis.
    #[must_use]
    pub fn encode3(&self, ctx: &GpuContext, coords: &[[u32; 3]]) -> Vec<u32> {
        let padded: Vec<[u32; 4]> = coords.iter().map(|c| [c[0], c[1], c[2], 0]).collect();
        self.run_encode(ctx, &self.encode3, &padded)
    }

    /// Batch-decodes 3D Morton keys back into `(x, y, z)` lattice points,
    /// mirroring [`prism_math::spatial::morton_decode3`].
    #[must_use]
    pub fn decode3(&self, ctx: &GpuContext, keys: &[u32]) -> Vec<[u32; 3]> {
        self.run_decode(ctx, &self.decode3, keys)
            .into_iter()
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }

    /// Uploads a `vec4<u32>` coordinate batch, dispatches `pipeline`, and reads
    /// back one `u32` key per input.
    fn run_encode(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        coords: &[[u32; 4]],
    ) -> Vec<u32> {
        let n = coords.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_morton_in", coords);
        let out_bytes = (n * size_of::<u32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_morton_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_morton_encode_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_morton_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<u32>(ctx, &stage)
    }

    /// Uploads a `u32` key batch, dispatches `pipeline`, and reads back one
    /// `vec4<u32>` coordinate per input.
    fn run_decode(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        keys: &[u32],
    ) -> Vec<[u32; 4]> {
        let n = keys.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_morton_in", keys);
        let out_bytes = (n * size_of::<[u32; 4]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_morton_out", out_bytes);
        let bind_group = self.bind(device, n, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_morton_decode_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_morton_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[u32; 4]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (count uniform, input, output) shared
    /// by every Morton kernel.
    fn bind(&self, device: &Device, count: usize, input: &Buffer, output: &Buffer) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_morton_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_morton_bind_group"),
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
        label: Some("prism_math_morton_pass"),
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
