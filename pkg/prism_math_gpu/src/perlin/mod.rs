//! Host orchestration of the Perlin gradient-noise compute kernels (§24.1
//! twin; the terrain / particle / procedural-texture sampling path that drives
//! classic "improved" Perlin noise straight onto the GPU).
//!
//! [`GpuPerlin`] batch-samples a run of 2D or 3D coordinates **on a real
//! device**, mirroring the CPU reference [`prism_math::noise::Perlin::get2`]
//! and [`get3`](prism_math::noise::Perlin::get3). The noise math is **not**
//! duplicated here: both kernels are composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_PERLIN`](prism_math::shader_mirror::WGSL_PERLIN) ahead of a thin
//! compute wrapper, so the device sampler cannot silently drift from the CPU
//! reference.
//!
//! # Permutation table (uploaded, not rebuilt)
//!
//! A Perlin field is defined by a seeded 512-entry permutation table. Rather
//! than re-deriving the Fisher-Yates shuffle on the device (which would risk a
//! divergent integer path), each sampling call uploads the exact table from
//! [`prism_math::noise::Perlin::permutation_table`] as a `[u32; 512]` storage
//! buffer. Every integer hash lookup is therefore **bit-exact**, so the GPU
//! and CPU always dot the identical gradient at every lattice corner.
//!
//! # Parity contract (honest boundary)
//!
//! The integer index path is bit-exact (see above); only the quintic fade, the
//! gradient dot products, and the lerps are floating point. Metal compiles
//! WGSL under fast-math, so the shader may round the last ULP differently;
//! parity is therefore a tight absolute+relative tolerance, not bit-exact
//! equality. Noise sampling is one-way (there is no inverse), so there is a
//! forward get2 kernel and a forward get3 kernel and nothing else.
//!
//! Standard `wgpu` compute orchestration. No neural, learned, or data-driven
//! components. No Unreal Engine or Unity source or derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::noise::Perlin;
use prism_math::shader_mirror::WGSL_PERLIN;
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

/// Compute wrapper that samples 2D Perlin noise for each `vec2<f32>` input.
const WRAP_GET2: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(2) var<storage, read> src: array<vec2<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_perlin_get2(p.x, p.y);\n\
}\n";

/// Compute wrapper that samples 3D Perlin noise for each `vec4<f32>` input.
///
/// The input is padded to `vec4<f32>` (the `.w` lane is ignored) so the host
/// can upload `[f32; 4]` with a 16-byte stride that matches the WGSL array
/// element layout; `array<vec3<f32>>` would mismatch the packed `[f32; 3]`.
const WRAP_GET3: &str = "\
struct Params { count: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(2) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let p = src[i];\n\
    dst[i] = prism_perlin_get3(p.x, p.y, p.z);\n\
}\n";

/// Uniform block carrying the valid element count for the batch bounds check.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Count {
    count: u32,
    _pad: [u32; 3],
}

/// Real-device twin of the Perlin gradient-noise sampling path.
///
/// Build it once per device with [`GpuPerlin::new`]; both forward pipelines and
/// the shared four-entry bind-group layout are created up front and reused
/// across batches and seeds. The seeded permutation table is uploaded per
/// sampling call, so one `GpuPerlin` serves every seed on its device.
pub struct GpuPerlin {
    get2: ComputePipeline,
    get3: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuPerlin {
    /// Compiles both forward pipelines on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_PERLIN`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_perlin_layout"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_perlin_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let get2 = compile(device, &pipeline_layout, WRAP_GET2, "prism_math_perlin_get2");
        let get3 = compile(device, &pipeline_layout, WRAP_GET3, "prism_math_perlin_get3");
        GpuPerlin {
            get2,
            get3,
            layout,
        }
    }

    /// Batch-samples 2D Perlin noise at each `[x, y]` on the device for the
    /// field seeded by `seed`, mirroring [`Perlin::get2`].
    #[must_use]
    pub fn get2(&self, ctx: &GpuContext, seed: u64, points: &[[f32; 2]]) -> Vec<f32> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let perm = perm_u32(seed);
        let perm_buf = buffer::storage_read(device, "prism_math_perlin_perm", &perm);
        let in_buf = buffer::storage_read(device, "prism_math_perlin_in2", points);
        self.run(ctx, &self.get2, &perm_buf, &in_buf, n)
    }

    /// Batch-samples 3D Perlin noise at each `[x, y, z]` on the device for the
    /// field seeded by `seed`, mirroring [`Perlin::get3`].
    #[must_use]
    pub fn get3(&self, ctx: &GpuContext, seed: u64, points: &[[f32; 3]]) -> Vec<f32> {
        let n = points.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let perm = perm_u32(seed);
        let perm_buf = buffer::storage_read(device, "prism_math_perlin_perm", &perm);
        // Pad to `[f32; 4]` so the upload stride matches `array<vec4<f32>>`.
        let padded: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let in_buf = buffer::storage_read(device, "prism_math_perlin_in3", &padded);
        self.run(ctx, &self.get3, &perm_buf, &in_buf, n)
    }

    /// Shared dispatch/readback for either kernel: allocates the `f32` output,
    /// binds the four buffers, dispatches, and reads the result back.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        perm_buf: &Buffer,
        in_buf: &Buffer,
        n: usize,
    ) -> Vec<f32> {
        let device = ctx.device();
        let out_bytes = (n * size_of::<f32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_perlin_out", out_bytes);
        let bind_group = self.bind(device, n, perm_buf, in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_perlin_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_perlin_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<f32>(ctx, &stage)
    }

    /// Builds the four-entry bind group (count uniform, perm table, input,
    /// output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        perm: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_perlin_params",
            &Count {
                count: count as u32,
                _pad: [0; 3],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_perlin_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: perm.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Widens the seeded 512-entry `u8` permutation table to the `[u32; 512]`
/// storage layout the kernel binds (WGSL storage arrays index `u32`).
fn perm_u32(seed: u64) -> [u32; 512] {
    let table = Perlin::new(seed).permutation_table();
    let mut out = [0u32; 512];
    for (dst, &src) in out.iter_mut().zip(table.iter()) {
        *dst = u32::from(src);
    }
    out
}

/// Compiles one forward pipeline from the shared `WGSL_PERLIN` fragment plus a
/// kernel-specific wrapper.
fn compile(
    device: &Device,
    pipeline_layout: &wgpu::PipelineLayout,
    wrapper: &str,
    label: &str,
) -> ComputePipeline {
    let mut source = String::new();
    source.push_str(WGSL_PERLIN);
    source.push('\n');
    source.push_str(wrapper);
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Records a 1D batch dispatch covering `n` elements at [`WORKGROUP`] threads
/// per group.
fn dispatch(enc: &mut CommandEncoder, pipeline: &ComputePipeline, bind_group: &BindGroup, n: usize) {
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_perlin_pass"),
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
