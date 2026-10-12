//! Host orchestration of the GPU scale/rotation/translation (`TRS`) affine
//! transform kernels (§24.1 twin: TRS instance / bone transforms on device).
//!
//! [`GpuAffine3`] batch-transforms points and vectors by a per-element
//! `(scale, quaternion, translation)` triple **on a real device**, mirroring the
//! CPU [`prism_math::Affine3::from_scale_rotation_translation`] followed by
//! `transform_point3` / `transform_vector3`. The math is **not** duplicated
//! here: the kernel is composed at runtime by prefixing the single-sourced
//! fragment [`WGSL_AFFINE`](prism_math::shader_mirror::WGSL_AFFINE) ahead of a
//! thin compute wrapper, so the device transform cannot silently drift from the
//! CPU reference — it is literally the same text.
//!
//! # Tolerance / parity
//!
//! Both ops are the CPU quaternion→matrix formula plus a `matrix3·v` chain;
//! Metal fast-math may contract/reassociate, so agreement is a small
//! absolute+relative epsilon. It skips gracefully when no adapter is available.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use prism_math::shader_mirror::WGSL_AFFINE;
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

/// Shared declarations: one `(scale, quat, translation, point)` item.
const IO: &str = "\
struct AffineIn {\n\
    scale: vec4<f32>,\n\
    quat: vec4<f32>,\n\
    translation: vec4<f32>,\n\
    point: vec4<f32>,\n\
};\n\
@group(0) @binding(0) var<storage, read> src: array<AffineIn>;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n";

/// Point wrapper: `matrix3·p + t`.
const WRAP_POINT: &str = "\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&src)) { return; }\n\
    let a = src[i];\n\
    let r = prism_affine_transform_point(a.quat, a.scale.xyz, a.translation.xyz, a.point.xyz);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// Vector wrapper: `matrix3·v` (no translation).
const WRAP_VECTOR: &str = "\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= arrayLength(&src)) { return; }\n\
    let a = src[i];\n\
    let r = prism_affine_transform_vector(a.quat, a.scale.xyz, a.point.xyz);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// A per-element TRS item: `scale.xyz`, `quat` (xyzw), `translation.xyz`, and
/// the point/vector to transform in `point.xyz`. 64 bytes, matching `AffineIn`.
pub type AffineItem = [[f32; 4]; 4];

/// Packs a TRS item with the value to transform.
#[must_use]
pub fn affine_item(
    scale: [f32; 3],
    quat: [f32; 4],
    translation: [f32; 3],
    point: [f32; 3],
) -> AffineItem {
    [
        [scale[0], scale[1], scale[2], 0.0],
        quat,
        [translation[0], translation[1], translation[2], 0.0],
        [point[0], point[1], point[2], 0.0],
    ]
}

/// Real-device batched TRS affine kernels, composed from the single-sourced
/// [`WGSL_AFFINE`] fragment.
pub struct GpuAffine3 {
    layout: BindGroupLayout,
    point: ComputePipeline,
    vector: ComputePipeline,
}

impl GpuAffine3 {
    /// Builds the point and vector TRS pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAffine3 {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_affine_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_affine_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_AFFINE);
            source.push('\n');
            source.push_str(IO);
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
        GpuAffine3 {
            point: build("prism_math_affine_point", WRAP_POINT),
            vector: build("prism_math_affine_vector", WRAP_VECTOR),
            layout,
        }
    }

    /// Batch-transforms points by each TRS item, mirroring
    /// [`prism_math::Affine3::transform_point3`].
    #[must_use]
    pub fn transform_points(&self, ctx: &GpuContext, items: &[AffineItem]) -> Vec<[f32; 3]> {
        self.run(ctx, &self.point, items)
    }

    /// Batch-transforms directions by each TRS item (no translation), mirroring
    /// [`prism_math::Affine3::transform_vector3`].
    #[must_use]
    pub fn transform_vectors(&self, ctx: &GpuContext, items: &[AffineItem]) -> Vec<[f32; 3]> {
        self.run(ctx, &self.vector, items)
    }

    /// Uploads a batch, dispatches `pipeline`, and reads the transformed `xyz`
    /// back.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[AffineItem],
    ) -> Vec<[f32; 3]> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_affine_in", items);
        let out_bytes = (n * size_of::<[f32; 4]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_affine_out", out_bytes);
        let bind_group = self.bind(device, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_affine_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_affine_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect()
    }

    /// Builds the two-entry bind group (input, output).
    fn bind(&self, device: &Device, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_affine_bind_group"),
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
        label: Some("prism_math_affine_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage bind-group-layout entry visible to the compute stage.
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
            let src = alloc::format!("{}\n{}\n{}", super::WGSL_AFFINE, super::IO, wrapper);
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("affine `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
