//! Host orchestration of the batched spherical-harmonic evaluation kernels
//! (§24.6 twin: GPU-side GI probe reconstruction at scale).
//!
//! [`GpuShBatch`] evaluates a batch of order-2 (9-coefficient) or order-3
//! (16-coefficient) real spherical harmonics, each in its own direction, in a
//! single dispatch **on a real device**, mirroring the CPU
//! [`prism_math::Sh2::eval`] / [`prism_math::Sh3::eval`]. This complements the
//! single-query [`crate::GpuSh3Eval`] with the through-put path a real GI pass
//! needs (many probes × directions). The basis/accumulate WGSL is **not**
//! duplicated here: the kernels are composed at runtime from the single-sourced
//! [`WGSL_SH3_EVAL`](prism_math::shader_mirror::WGSL_SH3_EVAL) fragment (whose
//! first 9 bands are exactly the order-2 basis), so the device evaluation cannot
//! silently drift from the CPU reference.
//!
//! # Parity, not bit-exactness
//!
//! Each evaluation is a sum of 9 or 16 polynomial-weighted multiply-adds. Metal
//! compiles WGSL under fast-math, so the shader compiler may contract and
//! reassociate the multiply-adds, rounding differently from the CPU's ordered
//! `sum`. Agreement is asserted within a small absolute+relative tolerance. It
//! skips gracefully when no adapter is available; the device-free WGSL parse gate
//! and a real `GPU` cover the rest.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_SH3_EVAL;
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

/// Packed-query helpers: rebuild the flat basis array from the four `vec4`
/// coefficient lanes and evaluate. `prism_sh3_eval`/`prism_sh3_basis` come from
/// [`WGSL_SH3_EVAL`].
const HELPERS: &str = "\
fn prism_sh3_eval_packed(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, dir: vec3<f32>) -> f32 {\n\
    let c = array<f32, 16>(\n\
        c0.x, c0.y, c0.z, c0.w,\n\
        c1.x, c1.y, c1.z, c1.w,\n\
        c2.x, c2.y, c2.z, c2.w,\n\
        c3.x, c3.y, c3.z, c3.w\n\
    );\n\
    return prism_sh3_eval(c, dir);\n\
}\n\
\n\
fn prism_sh2_eval_packed(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, dir: vec3<f32>) -> f32 {\n\
    let b = prism_sh3_basis(dir);\n\
    var acc = 0.0;\n\
    acc = acc + c0.x * b[0] + c0.y * b[1] + c0.z * b[2] + c0.w * b[3];\n\
    acc = acc + c1.x * b[4] + c1.y * b[5] + c1.z * b[6] + c1.w * b[7];\n\
    acc = acc + c2.x * b[8];\n\
    return acc;\n\
}\n";

/// Wrapper for the order-3 batch: one scalar per query.
const WRAP_SH3: &str = "\
struct Params { count: u32 };\n\
struct ShQuery {\n\
    c0: vec4<f32>,\n\
    c1: vec4<f32>,\n\
    c2: vec4<f32>,\n\
    c3: vec4<f32>,\n\
    dir: vec4<f32>,\n\
};\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<ShQuery>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let q = src[i];\n\
    dst[i] = prism_sh3_eval_packed(q.c0, q.c1, q.c2, q.c3, q.dir.xyz);\n\
}\n";

/// Wrapper for the order-2 batch: one scalar per query (the `c3` lane is
/// ignored).
const WRAP_SH2: &str = "\
struct Params { count: u32 };\n\
struct ShQuery {\n\
    c0: vec4<f32>,\n\
    c1: vec4<f32>,\n\
    c2: vec4<f32>,\n\
    c3: vec4<f32>,\n\
    dir: vec4<f32>,\n\
};\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<ShQuery>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let q = src[i];\n\
    dst[i] = prism_sh2_eval_packed(q.c0, q.c1, q.c2, q.dir.xyz);\n\
}\n";

/// Uniform block: count only, padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    count: u32,
    _pad: [u32; 3],
}

/// A packed SH query: up to 16 coefficients in four `vec4` lanes plus a query
/// direction; the layout matches the WGSL `ShQuery` (80 bytes, no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuShQuery {
    /// Coefficients 0..4.
    pub c0: [f32; 4],
    /// Coefficients 4..8.
    pub c1: [f32; 4],
    /// Coefficients 8..12.
    pub c2: [f32; 4],
    /// Coefficients 12..16 (ignored by the order-2 kernel).
    pub c3: [f32; 4],
    /// Query direction in `[x, y, z, _]` (should be unit length).
    pub dir: [f32; 4],
}

impl GpuShQuery {
    /// Packs 16 order-3 coefficients and a direction.
    #[must_use]
    pub fn from_coeffs16(coeffs: &[f32; 16], dir: [f32; 3]) -> GpuShQuery {
        GpuShQuery {
            c0: [coeffs[0], coeffs[1], coeffs[2], coeffs[3]],
            c1: [coeffs[4], coeffs[5], coeffs[6], coeffs[7]],
            c2: [coeffs[8], coeffs[9], coeffs[10], coeffs[11]],
            c3: [coeffs[12], coeffs[13], coeffs[14], coeffs[15]],
            dir: [dir[0], dir[1], dir[2], 0.0],
        }
    }

    /// Packs 9 order-2 coefficients and a direction (the unused `c3` lane is
    /// zeroed).
    #[must_use]
    pub fn from_coeffs9(coeffs: &[f32; 9], dir: [f32; 3]) -> GpuShQuery {
        let mut q = GpuShQuery::from_coeffs16(
            &[
                coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6],
                coeffs[7], coeffs[8], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            dir,
        );
        q.c3 = [0.0; 4];
        q
    }
}

/// A real-device batched SH evaluation kernel pair (order 2 + order 3),
/// composed from the single-sourced
/// [`WGSL_SH3_EVAL`](prism_math::shader_mirror::WGSL_SH3_EVAL) fragment.
pub struct GpuShBatch {
    layout: BindGroupLayout,
    sh2: ComputePipeline,
    sh3: ComputePipeline,
}

impl GpuShBatch {
    /// Builds the order-2 and order-3 batch pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuShBatch {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_sh_batch_layout"),
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
            label: Some("prism_math_sh_batch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_SH3_EVAL);
            source.push('\n');
            source.push_str(HELPERS);
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
        GpuShBatch {
            sh2: build("prism_math_sh_batch_2", WRAP_SH2),
            sh3: build("prism_math_sh_batch_3", WRAP_SH3),
            layout,
        }
    }

    /// Batch-evaluates order-2 SH, mirroring [`prism_math::Sh2::eval`]
    /// element-wise (each query uses its first 9 coefficients).
    #[must_use]
    pub fn eval2(&self, ctx: &GpuContext, queries: &[GpuShQuery]) -> Vec<f32> {
        self.run(ctx, &self.sh2, queries)
    }

    /// Batch-evaluates order-3 SH, mirroring [`prism_math::Sh3::eval`]
    /// element-wise.
    #[must_use]
    pub fn eval3(&self, ctx: &GpuContext, queries: &[GpuShQuery]) -> Vec<f32> {
        self.run(ctx, &self.sh3, queries)
    }

    /// Uploads a query batch, dispatches `pipeline`, and reads one scalar per
    /// element back.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        queries: &[GpuShQuery],
    ) -> Vec<f32> {
        let n = queries.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let params = buffer::uniform(
            device,
            "prism_math_sh_batch_params",
            &CountParams {
                count: n as u32,
                _pad: [0; 3],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_sh_batch_in", queries);
        let out_bytes = (n * size_of::<f32>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_sh_batch_out", out_bytes);
        let bind_group = self.bind(device, &params, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_sh_batch_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_sh_batch_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<f32>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_sh_batch_bind_group"),
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
        label: Some("prism_math_sh_batch_pass"),
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

/// Device-free gate: every composed kernel (fragment + helpers + wrapper) must
/// be a valid WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernels_parse() {
        for (label, wrapper) in [("sh2", super::WRAP_SH2), ("sh3", super::WRAP_SH3)] {
            let src = alloc::format!("{}\n{}\n{}", super::WGSL_SH3_EVAL, super::HELPERS, wrapper);
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("sh_batch `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
