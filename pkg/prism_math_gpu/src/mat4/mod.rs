//! Host orchestration of the GPU 4x4 affine-transform kernels (§24.1 twin:
//! instance / skinning / normal transforms on device).
//!
//! [`GpuMat4`] batch-transforms points and vectors by a 4x4 matrix **on a real
//! device**, mirroring the CPU
//! [`prism_math::Mat4::transform_point3`] (implicit `w = 1`, perspective divide)
//! and [`prism_math::Mat4::transform_vector3`] (implicit `w = 0`, no
//! translation). The matrix is passed as its four **columns** (the CPU `Mat4` is
//! column-major). The math is **not** duplicated here: the kernel is composed at
//! runtime by prefixing the single-sourced fragment
//! [`WGSL_TRANSFORM`](prism_math::shader_mirror::WGSL_TRANSFORM) ahead of a thin
//! compute wrapper, so the device transform cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Tolerance / parity
//!
//! Both ops are multiply-add chains (the point op adds a `w` divide); Metal
//! fast-math may contract/reassociate, so agreement is a small
//! absolute+relative epsilon. It skips gracefully when no adapter is available.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use prism_math::shader_mirror::WGSL_TRANSFORM;
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

/// Point-transform wrapper: a 4x4 column set in a uniform, vectors in a storage
/// batch, `arrayLength` guard, one transformed `vec4` out.
const WRAP_POINT: &str = "\
struct Params { m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, m3: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&src) || i >= arrayLength(&dst)) { return; }\n\
    let r = prism_mat4_transform_point(params.m0, params.m1, params.m2, params.m3, src[i].xyz);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// Vector-transform wrapper: like [`WRAP_POINT`] but with the translation column
/// omitted (implicit `w = 0`).
const WRAP_VECTOR: &str = "\
struct Params { m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, m3: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&src) || i >= arrayLength(&dst)) { return; }\n\
    let r = prism_mat4_transform_vector(params.m0, params.m1, params.m2, params.m3, src[i].xyz);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// Uniform block: the four matrix columns (column-major), 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat4Cols {
    m0: [f32; 4],
    m1: [f32; 4],
    m2: [f32; 4],
    m3: [f32; 4],
}

/// Real-device batched 4x4 transform kernels, composed from the single-sourced
/// [`WGSL_TRANSFORM`] fragment.
pub struct GpuMat4 {
    layout: BindGroupLayout,
    point: ComputePipeline,
    vector: ComputePipeline,
}

impl GpuMat4 {
    /// Builds the point and vector transform pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMat4 {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_mat4_layout"),
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
            label: Some("prism_math_mat4_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_TRANSFORM);
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
        GpuMat4 {
            point: build("prism_math_mat4_point", WRAP_POINT),
            vector: build("prism_math_mat4_vector", WRAP_VECTOR),
            layout,
        }
    }

    /// Batch-transforms points (implicit `w = 1`, perspective divide), mirroring
    /// [`prism_math::Mat4::transform_point3`]. `m` is the four column-major
    /// columns.
    #[must_use]
    pub fn transform_points(
        &self,
        ctx: &GpuContext,
        m: &[[f32; 4]; 4],
        points: &[[f32; 3]],
    ) -> Vec<[f32; 3]> {
        self.run(ctx, &self.point, m, points)
    }

    /// Batch-transforms directions (implicit `w = 0`, no translation), mirroring
    /// [`prism_math::Mat4::transform_vector3`].
    #[must_use]
    pub fn transform_vectors(
        &self,
        ctx: &GpuContext,
        m: &[[f32; 4]; 4],
        vectors: &[[f32; 3]],
    ) -> Vec<[f32; 3]> {
        self.run(ctx, &self.vector, m, vectors)
    }

    /// Uploads the matrix columns + input batch, dispatches `pipeline`, and
    /// reads the transformed `xyz` back.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        m: &[[f32; 4]; 4],
        items: &[[f32; 3]],
    ) -> Vec<[f32; 3]> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let _ = crate::dispatch::checked_count(n);
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_mat4_params",
            &Mat4Cols {
                m0: m[0],
                m1: m[1],
                m2: m[2],
                m3: m[3],
            },
        );
        let packed: Vec<[f32; 4]> = items.iter().map(|v| [v[0], v[1], v[2], 0.0]).collect();
        let in_buf = buffer::storage_read(device, "prism_math_mat4_in", &packed);
        let out_bytes = (n * size_of::<[f32; 4]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_mat4_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_mat4_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_mat4_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect()
    }

    /// Builds the three-entry bind group (matrix uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_mat4_bind_group"),
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
        label: Some("prism_math_mat4_pass"),
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

/// Device-free gate: every composed kernel (fragment + wrapper) must be a valid
/// WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernels_parse() {
        for (label, wrapper) in [("point", super::WRAP_POINT), ("vector", super::WRAP_VECTOR)] {
            let src = alloc::format!("{}\n{}", super::WGSL_TRANSFORM, wrapper);
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("mat4 `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
