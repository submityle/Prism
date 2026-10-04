//! Host orchestration of the view (look-at) matrix compute kernel (§24.1 twin).
//!
//! [`GpuView`] builds a right-handed camera view matrix **on a real device**
//! from `eye`/`target`(or `dir`)/`up` and reads the four columns back,
//! mirroring the CPU constructors [`prism_math::projection::look_at_rh`] and
//! [`prism_math::projection::look_to_rh`]. The WGSL builder functions are
//! **not** duplicated here: the kernel source is composed at runtime by
//! prefixing the single-sourced fragment
//! [`WGSL_LOOK_AT_RH`](prism_math::shader_mirror::WGSL_LOOK_AT_RH) ahead of a
//! thin compute wrapper, so the device basis derivation cannot silently drift
//! from the CPU reference — they are literally the same text.
//!
//! # Parity, not bit-exactness
//!
//! The basis derivation normalizes (reciprocal square root) and takes cross
//! products. Metal compiles WGSL under fast-math, so the shader compiler may
//! use a lower-precision `rsqrt` and contract/reassociate the dot and cross
//! terms, rounding differently from the CPU. The parity tests therefore assert
//! agreement within a small absolute+relative tolerance (the §24.1 contract is
//! explicitly a *tolerance* round-trip), not bit-exact equality.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;

use bytemuck::{Pod, Zeroable};
use prism_math::Mat4;
use prism_math::Vec3;
use prism_math::Vec4;
use prism_math::shader_mirror::WGSL_LOOK_AT_RH;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Which right-handed view matrix the kernel should build.
///
/// The discriminants match the `mode` branch selector in the composed kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ViewKind {
    /// [`prism_math::projection::look_at_rh`] — `b` is the look-at `target`.
    LookAtRh = 0,
    /// [`prism_math::projection::look_to_rh`] — `b` is the forward `dir`.
    LookToRh = 1,
}

/// The compute wrapper appended after the single-sourced builder fragment.
///
/// It calls `prism_look_at_rh` / `prism_look_to_rh`, which are *only* defined by
/// [`WGSL_LOOK_AT_RH`]; the two are concatenated in [`GpuView::new`]. A single
/// invocation writes the four matrix columns to `dst`.
const COMPUTE_WRAPPER: &str = "\
struct Params {\n\
    eye: vec4<f32>,\n\
    b: vec4<f32>,\n\
    up: vec4<f32>,\n\
    mode: vec4<u32>,\n\
};\n\
\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;\n\
\n\
@compute @workgroup_size(1, 1, 1)\n\
fn main() {\n\
    var m: mat4x4<f32>;\n\
    if (params.mode.x == 0u) {\n\
        m = prism_look_at_rh(params.eye.xyz, params.b.xyz, params.up.xyz);\n\
    } else {\n\
        m = prism_look_to_rh(params.eye.xyz, params.b.xyz, params.up.xyz);\n\
    }\n\
    dst[0] = m[0];\n\
    dst[1] = m[1];\n\
    dst[2] = m[2];\n\
    dst[3] = m[3];\n\
}\n";

/// Uniform block shared with `Params` in the composed kernel. `eye.xyz` is the
/// camera position, `b.xyz` is the look-at target (mode 0) or forward dir
/// (mode 1), `up.xyz` is the up hint, and `mode.x` selects the branch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    eye: [f32; 4],
    b: [f32; 4],
    up: [f32; 4],
    mode: [u32; 4],
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

/// A real-device builder for the §24.1 right-handed view (look-at) matrices.
pub struct GpuView {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuView {
    /// Builds the compute pipeline, composing the kernel from the single-sourced
    /// [`WGSL_LOOK_AT_RH`] fragment plus the thin [`COMPUTE_WRAPPER`].
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuView {
        let device = ctx.device();

        let mut source = String::new();
        source.push_str(WGSL_LOOK_AT_RH);
        source.push('\n');
        source.push_str(COMPUTE_WRAPPER);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_view"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_view_layout"),
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
            label: Some("prism_math_view_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_view_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuView { pipeline, layout }
    }

    /// Builds a right-handed look-at view matrix on the device (camera at `eye`
    /// looking toward `target`).
    #[must_use]
    pub fn look_at_rh(&self, ctx: &GpuContext, eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        self.build(ctx, eye, target, up, ViewKind::LookAtRh)
    }

    /// Builds a right-handed look-to view matrix on the device using an explicit
    /// forward `dir`.
    #[must_use]
    pub fn look_to_rh(&self, ctx: &GpuContext, eye: Vec3, dir: Vec3, up: Vec3) -> Mat4 {
        self.build(ctx, eye, dir, up, ViewKind::LookToRh)
    }

    /// Dispatches the single-invocation builder and assembles the four
    /// read-back columns into a [`Mat4`].
    fn build(&self, ctx: &GpuContext, eye: Vec3, b: Vec3, up: Vec3, kind: ViewKind) -> Mat4 {
        let device = ctx.device();

        let params = Params {
            eye: [eye.x, eye.y, eye.z, 0.0],
            b: [b.x, b.y, b.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            mode: [kind as u32, 0, 0, 0],
        };

        let out_bytes = (COLUMNS * size_of::<GpuColumn>()) as u64;
        let params_buf = buffer::uniform(device, "prism_math_view_params", &params);
        let dst_buf = buffer::storage_rw_zeroed(device, "prism_math_view_dst", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_view_bind_group"),
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
            label: Some("prism_math_view_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_math_view_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let stage = buffer::staging(device, "prism_math_view_stage", out_bytes);
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
