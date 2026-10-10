//! Host orchestration of the octahedral normal-codec kernels (§24.1 twin: the
//! GPU side of compact `GBuffer` normal storage).
//!
//! [`GpuOctahedral`] batch-encodes unit directions to octahedral coordinates,
//! snorm-packs them to a single `u32`, and decodes them back **on a real
//! device** — the standard deferred-renderer normal (de)compression path —
//! mirroring the CPU codec [`prism_math::octahedral::encode`] / [`decode`] /
//! [`pack_snorm`] / [`unpack_snorm`]. The projection/fold math is **not**
//! duplicated here: each kernel is composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_OCTAHEDRAL`](prism_math::shader_mirror::WGSL_OCTAHEDRAL) ahead of a
//! thin compute wrapper, so the device codec cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Tolerance, not bit-exactness
//!
//! The encode step divides by the L1 norm, and Metal compiles WGSL under
//! fast-math, so a coordinate can round a last ULP differently than the CPU.
//! The full-precision [`encode`](GpuOctahedral::encode) /
//! [`decode`](GpuOctahedral::decode) parity is therefore an absolute+relative
//! tolerance round-trip, and the snorm-[`pack`](GpuOctahedral::pack) parity is
//! defined on the reconstructed direction (plus a +/-1 tolerance on each 16-bit
//! quantized code to admit a fast-math round crossing a quantization boundary).
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::Pod;
use prism_math::shader_mirror::WGSL_OCTAHEDRAL;
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

/// Encode wrapper: `vec4` normals (xyz used) -> `vec2` octahedral coordinates.
const WRAP_ENCODE: &str = "\
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&dst)) { return; }\n\
    dst[i] = prism_oct_encode(src[i].xyz);\n\
}\n";

/// Decode wrapper: `vec2` octahedral coordinates -> `vec4` unit directions
/// (`.w = 0`).
const WRAP_DECODE: &str = "\
@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&dst)) { return; }\n\
    dst[i] = vec4<f32>(prism_oct_decode(src[i]), 0.0);\n\
}\n";

/// Pack wrapper: `vec4` normals (xyz used) -> snorm-packed `u32`.
const WRAP_PACK: &str = "\
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&dst)) { return; }\n\
    dst[i] = prism_oct_pack_snorm(src[i].xyz);\n\
}\n";

/// Unpack wrapper: snorm-packed `u32` -> `vec4` unit directions (`.w = 0`).
const WRAP_UNPACK: &str = "\
@group(0) @binding(0) var<storage, read> src: array<u32>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&dst)) { return; }\n\
    dst[i] = vec4<f32>(prism_oct_unpack_snorm(src[i]), 0.0);\n\
}\n";

/// A real-device octahedral normal codec (encode / decode / pack / unpack),
/// each kernel composed from the single-sourced [`WGSL_OCTAHEDRAL`] fragment.
pub struct GpuOctahedral {
    layout: BindGroupLayout,
    encode: ComputePipeline,
    decode: ComputePipeline,
    pack: ComputePipeline,
    unpack: ComputePipeline,
}

impl GpuOctahedral {
    /// Builds the four codec pipelines, each composed from the single-sourced
    /// [`WGSL_OCTAHEDRAL`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOctahedral {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_octahedral_layout"),
            entries: &[
                buffer_layout(0, BufferBindingType::Storage { read_only: true }),
                buffer_layout(1, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_octahedral_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_OCTAHEDRAL);
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
        GpuOctahedral {
            encode: build("prism_math_octahedral_encode", WRAP_ENCODE),
            decode: build("prism_math_octahedral_decode", WRAP_DECODE),
            pack: build("prism_math_octahedral_pack", WRAP_PACK),
            unpack: build("prism_math_octahedral_unpack", WRAP_UNPACK),
            layout,
        }
    }

    /// Batch-encodes directions (`[nx, ny, nz, _]`, need not be normalized) to
    /// octahedral coordinates `[ex, ey]`, mirroring
    /// [`prism_math::octahedral::encode`].
    #[must_use]
    pub fn encode(&self, ctx: &GpuContext, normals: &[[f32; 4]]) -> Vec<[f32; 2]> {
        self.run::<[f32; 4], [f32; 2]>(ctx, &self.encode, normals)
    }

    /// Batch-decodes octahedral coordinates `[ex, ey]` back to unit directions
    /// `[nx, ny, nz, 0]`, mirroring [`prism_math::octahedral::decode`].
    #[must_use]
    pub fn decode(&self, ctx: &GpuContext, coords: &[[f32; 2]]) -> Vec<[f32; 4]> {
        self.run::<[f32; 2], [f32; 4]>(ctx, &self.decode, coords)
    }

    /// Batch snorm-packs directions to one `u32` each (`x` in the low 16 bits,
    /// `y` in the high 16), mirroring [`prism_math::octahedral::pack_snorm`].
    #[must_use]
    pub fn pack(&self, ctx: &GpuContext, normals: &[[f32; 4]]) -> Vec<u32> {
        self.run::<[f32; 4], u32>(ctx, &self.pack, normals)
    }

    /// Batch-unpacks snorm `u32` codes back to unit directions `[nx, ny, nz, 0]`,
    /// mirroring [`prism_math::octahedral::unpack_snorm`].
    #[must_use]
    pub fn unpack(&self, ctx: &GpuContext, bits: &[u32]) -> Vec<[f32; 4]> {
        self.run::<u32, [f32; 4]>(ctx, &self.unpack, bits)
    }

    /// Uploads an `In` batch, dispatches `pipeline`, and reads back one `Out`
    /// per element.
    fn run<In: Pod, Out: Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[In],
    ) -> Vec<Out> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_octahedral_in", items);
        let out_bytes = (n * size_of::<Out>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_octahedral_out", out_bytes);
        let bind_group = self.bind(device, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_octahedral_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_octahedral_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<Out>(ctx, &stage)
    }

    /// Builds the two-entry bind group (input, output).
    fn bind(&self, device: &Device, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_octahedral_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
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
        label: Some("prism_math_octahedral_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
