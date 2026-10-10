//! Host orchestration of the cubic-spline compute kernel (§24.1 twin;
//! animation, camera-path, and procedural-geometry systems consume these
//! spline evaluations on the GPU side).
//!
//! [`GpuSpline`] batch-evaluates one spline function over a slice of
//! control-point tuples **on a real device**, mirroring the CPU reference
//! family [`prism_math::curve::spline`]. The curve math is **not** duplicated
//! here: a single pipeline is composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_SPLINE`](prism_math::shader_mirror::WGSL_SPLINE) ahead of a thin
//! compute wrapper that calls the `prism_spline(op, a, b, c, d, t)` dispatcher,
//! so the device helpers cannot silently drift from the CPU reference. The
//! function is selected per dispatch by the [`Spline`] op code carried in the
//! uniform.
//!
//! The control values are `vec3<f32>` (the AAA use case being 3D position and
//! velocity curves). Each sample carries the four control vectors `a..d` in the
//! order the matching CPU function takes them, plus the segment parameter `t`.
//!
//! # Parity contract (honest boundary)
//!
//! Every spline evaluator is bare multiply/add arithmetic with no
//! transcendental calls, so the GPU result matches the CPU path to a tight
//! FMA tolerance (`1e-5`), rejecting any genuine operand-order, layout, or
//! algorithm drift while tolerating last-ULP fast-math rounding.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_SPLINE;
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

/// Spline-function selector, matching the `prism_spline` dispatcher op codes in
/// [`WGSL_SPLINE`] and the CPU function set in [`prism_math::curve::spline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Spline {
    /// Cubic Hermite position: control order `(p0, m0, p1, m1)`.
    Hermite = 0,
    /// Cubic Hermite tangent (derivative w.r.t. `t`).
    HermiteTangent = 1,
    /// Uniform Catmull-Rom position: control order `(p0, p1, p2, p3)`.
    CatmullRom = 2,
    /// Uniform Catmull-Rom tangent.
    CatmullRomTangent = 3,
    /// Cubic Bézier position: control order `(p0, p1, p2, p3)`.
    BezierCubic = 4,
    /// Cubic Bézier tangent.
    BezierCubicTangent = 5,
}

/// One spline evaluation input: the four `vec3` control values and the segment
/// parameter `t`.
///
/// Each vector is padded to a `vec4` so the struct matches the WGSL
/// `std140`-style 16-byte alignment of the device-side `SplineSample`; only the
/// `xyz` lanes of the control vectors and the `x` lane of `t` are read.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplineSample {
    /// First control vector (`p0` for every function).
    pub a: [f32; 4],
    /// Second control vector (`m0` for Hermite, `p1` otherwise).
    pub b: [f32; 4],
    /// Third control vector (`p1` for Hermite, `p2` otherwise).
    pub c: [f32; 4],
    /// Fourth control vector (`m1` for Hermite, `p3` otherwise).
    pub d: [f32; 4],
    /// Segment parameter in `.x`; the remaining lanes are padding.
    pub t: [f32; 4],
}

impl SplineSample {
    /// Builds a sample from four `vec3` control values (`[x, y, z]`) and the
    /// segment parameter `t`, zero-padding the `w` lanes.
    #[must_use]
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3], t: f32) -> Self {
        SplineSample {
            a: [a[0], a[1], a[2], 0.0],
            b: [b[0], b[1], b[2], 0.0],
            c: [c[0], c[1], c[2], 0.0],
            d: [d[0], d[1], d[2], 0.0],
            t: [t, 0.0, 0.0, 0.0],
        }
    }
}

/// Compute wrapper that applies the selected spline op to each input sample.
const WRAP: &str = "\
struct SplineSample { a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, t: vec4<f32> };\n\
struct Params { count: u32, op: u32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<SplineSample>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    let r = prism_spline(params.op, s.a.xyz, s.b.xyz, s.c.xyz, s.d.xyz, s.t.x);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// Uniform block carrying the valid element count and the spline op code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// Real-device twin of the cubic-spline evaluator family.
///
/// Build it once per device with [`GpuSpline::new`]; the single pipeline and
/// bind-group layout are created up front and reused across batches and ops.
pub struct GpuSpline {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuSpline {
    /// Compiles the spline pipeline on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_SPLINE`] fragment verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_spline_layout"),
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
            label: Some("prism_math_spline_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_SPLINE);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_spline"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_spline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpline { pipeline, layout }
    }

    /// Batch-evaluates `spline` for each sample in `src` on the device,
    /// mirroring the matching CPU function in [`prism_math::curve::spline`].
    ///
    /// The returned vectors carry the result in the `xyz` lanes with a zero
    /// `w`, one entry per input sample.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, spline: Spline, src: &[SplineSample]) -> Vec<[f32; 4]> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_spline_in", src);
        let out_bytes = (n * size_of::<[f32; 4]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_spline_out", out_bytes);
        let bind_group = self.bind(device, n, spline, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_spline_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_spline_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        spline: Spline,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_spline_params",
            &Params {
                count: count as u32,
                op: spline as u32,
                _pad: [0; 2],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_spline_bind_group"),
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
        label: Some("prism_math_spline_pass"),
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
