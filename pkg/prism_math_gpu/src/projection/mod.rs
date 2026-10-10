//! Host orchestration of the projection-matrix compute kernel (§24.1 twin).
//!
//! [`GpuProjection`] builds a camera projection matrix **on a real device**
//! from its scalar parameters and reads the four columns back, mirroring the
//! CPU constructors [`prism_math::projection::perspective_rh`],
//! [`prism_math::projection::perspective_reverse_z_rh`], and
//! [`prism_math::projection::orthographic_rh`]. The WGSL builder functions are
//! **not** duplicated here: the kernel source is composed at runtime by
//! prefixing the single-sourced fragment
//! [`WGSL_PROJECTION_RH`](prism_math::shader_mirror::WGSL_PROJECTION_RH) ahead
//! of a thin compute wrapper, so the device formulas cannot silently drift from
//! the CPU reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! The builders divide, negate, and multiply floats. Metal compiles WGSL under
//! fast-math, so the shader compiler may contract `a * b + c` into one `fma`
//! and reassociate, rounding differently from the CPU's separate operations.
//! The parity tests therefore assert agreement within a small
//! absolute+relative tolerance (the §24.1 contract is explicitly a *tolerance*
//! round-trip), not bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_PROJECTION_RH;
use prism_math::Mat4;
use prism_math::Vec4;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Which right-handed projection the kernel should build.
///
/// The discriminants match the `mode` branch selector in the composed kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ProjectionKind {
    /// [`prism_math::projection::perspective_rh`] — depth `[0, 1]`.
    PerspectiveRh = 0,
    /// [`prism_math::projection::perspective_reverse_z_rh`] — `near -> 1`,
    /// `far -> 0`.
    PerspectiveReverseZRh = 1,
    /// [`prism_math::projection::orthographic_rh`] — depth `[0, 1]`.
    OrthographicRh = 2,
}

/// The compute wrapper appended after the single-sourced builder fragment.
///
/// It calls `prism_perspective_rh` / `prism_perspective_reverse_z_rh` /
/// `prism_orthographic_rh`, which are *only* defined by [`WGSL_PROJECTION_RH`];
/// the two are concatenated in [`GpuProjection::new`]. A single invocation
/// writes the four matrix columns to `dst`.
const COMPUTE_WRAPPER: &str = "\
struct Params {\n\
    p0: f32,\n\
    p1: f32,\n\
    p2: f32,\n\
    p3: f32,\n\
    z_near: f32,\n\
    z_far: f32,\n\
    mode: u32,\n\
    pad: u32,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(1, 1, 1)\n\
fn main() {\n\
    var m: mat4x4<f32>;\n\
    if (params.mode == 0u) {\n\
        m = prism_perspective_rh(params.p0, params.p1, params.z_near, params.z_far);\n\
    } else if (params.mode == 1u) {\n\
        m = prism_perspective_reverse_z_rh(params.p0, params.p1, params.z_near, params.z_far);\n\
    } else {\n\
        m = prism_orthographic_rh(params.p0, params.p1, params.p2, params.p3, params.z_near, params.z_far);\n\
    }\n\
    dst[0] = m[0];\n\
    dst[1] = m[1];\n\
    dst[2] = m[2];\n\
    dst[3] = m[3];\n\
}\n";

/// Uniform block shared with `Params` in the composed kernel.
///
/// For the perspective modes `p0 = fovy_radians`, `p1 = aspect`, and `p2`/`p3`
/// are unused. For the orthographic mode `p0 = left`, `p1 = right`,
/// `p2 = bottom`, `p3 = top`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    p0: f32,
    p1: f32,
    p2: f32,
    p3: f32,
    z_near: f32,
    z_far: f32,
    mode: u32,
    pad: u32,
}

/// `Pod` mirror of a `vec4<f32>` storage element (16-byte stride): one matrix
/// column.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuColumn {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl GpuColumn {
    fn to_vec4(self) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, self.w)
    }
}

/// The number of `vec4<f32>` columns a `mat4x4<f32>` readback occupies.
const COLUMNS: usize = 4;

/// A real-device builder for the §24.1 right-handed projection matrices.
pub struct GpuProjection {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuProjection {
    /// Builds the compute pipeline, composing the kernel from the single-sourced
    /// [`WGSL_PROJECTION_RH`] fragment plus the thin [`COMPUTE_WRAPPER`].
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuProjection {
        let device = ctx.device();

        let mut source = String::new();
        source.push_str(WGSL_PROJECTION_RH);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_projection"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_projection_layout"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_projection_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_projection_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuProjection { pipeline, layout }
    }

    /// Builds a right-handed perspective matrix on the device (depth `[0, 1]`).
    #[must_use]
    pub fn perspective_rh(
        &self,
        ctx: &GpuContext,
        fovy_radians: f32,
        aspect: f32,
        z_near: f32,
        z_far: f32,
    ) -> Mat4 {
        self.build(
            ctx,
            Params {
                p0: fovy_radians,
                p1: aspect,
                p2: 0.0,
                p3: 0.0,
                z_near,
                z_far,
                mode: ProjectionKind::PerspectiveRh as u32,
                pad: 0,
            },
        )
    }

    /// Builds a right-handed reverse-Z perspective matrix on the device
    /// (`near -> 1`, `far -> 0`).
    #[must_use]
    pub fn perspective_reverse_z_rh(
        &self,
        ctx: &GpuContext,
        fovy_radians: f32,
        aspect: f32,
        z_near: f32,
        z_far: f32,
    ) -> Mat4 {
        self.build(
            ctx,
            Params {
                p0: fovy_radians,
                p1: aspect,
                p2: 0.0,
                p3: 0.0,
                z_near,
                z_far,
                mode: ProjectionKind::PerspectiveReverseZRh as u32,
                pad: 0,
            },
        )
    }

    /// Builds a right-handed orthographic matrix on the device (depth `[0, 1]`).
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the 6-param CPU ortho constructor plus ctx"
    )]
    pub fn orthographic_rh(
        &self,
        ctx: &GpuContext,
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        z_near: f32,
        z_far: f32,
    ) -> Mat4 {
        self.build(
            ctx,
            Params {
                p0: left,
                p1: right,
                p2: bottom,
                p3: top,
                z_near,
                z_far,
                mode: ProjectionKind::OrthographicRh as u32,
                pad: 0,
            },
        )
    }

    /// Dispatches the single-invocation builder for `params` and assembles the
    /// four read-back columns into a [`Mat4`].
    fn build(&self, ctx: &GpuContext, params: Params) -> Mat4 {
        let device = ctx.device();

        let out_bytes = (COLUMNS * size_of::<GpuColumn>()) as u64;
        let params_buf = buffer::uniform(device, "prism_math_projection_params", &params);
        let dst_buf = buffer::storage_rw_zeroed(device, "prism_math_projection_dst", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_projection_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_projection_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_projection_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_projection_stage", out_bytes);
        buffer::copy(&mut enc, &dst_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let cols = buffer::read_back::<GpuColumn>(ctx, &stage);
        Mat4::from_cols(
            cols[0].to_vec4(),
            cols[1].to_vec4(),
            cols[2].to_vec4(),
            cols[3].to_vec4(),
        )
    }
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
