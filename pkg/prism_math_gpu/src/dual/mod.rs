//! Host orchestration of the GPU forward-mode automatic-differentiation kernel
//! (§24.5 twin: value + exact first derivative for GPU-side analytic queries).
//!
//! [`GpuDual`] batch-evaluates a unary function and its **exact** analytic
//! derivative through the dual-number carrier, **on a real device**, mirroring
//! the CPU [`Dual`](prism_math::dual::Dual) op set selectable by [`DualOp`].
//! Each input is seeded as the independent variable (`Dual::variable`) and the
//! selected op is applied, returning the value `f(x)` and derivative `f'(x)`.
//! This is the GPU side of analytic SDF normals, curve tangents, and physics /
//! IK Jacobian columns — strictly forward mode, first order, no reverse pass.
//! The value/derivative math is **not** duplicated here: the kernel is composed
//! at runtime by prefixing the single-sourced fragment
//! [`WGSL_DUAL`](prism_math::shader_mirror::WGSL_DUAL) ahead of a thin compute
//! wrapper, so the device result cannot silently drift from the CPU reference —
//! it is literally the same text.
//!
//! # Tolerance / parity
//!
//! The polynomial ops (`recip`/`squared`) and the `sqrt` builtin are compared
//! to the CPU `libm` reference at a tight tolerance; the transcendental ops
//! (`exp`/`ln`/`sin`/`cos`/`tan`/`powf`) evaluate device builtins and so agree
//! only within a looser relative epsilon. Metal compiles WGSL under fast-math,
//! so even the tight ops carry a small absolute+relative slack. The parity suite
//! exercises exactly the domain of each op (positive arguments for `sqrt`/`ln`/
//! `powf`, non-zero derivative signs for `abs`).
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_DUAL;
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

/// Compute wrapper that applies the selected dual op to each input scalar.
/// `src[i].x` is the seeded variable `x`; `src[i].p` is the exponent used only
/// by the `powf` op (ignored otherwise).
const WRAP: &str = "\
struct Params { count: u32, op: u32 };\n\
struct DualIn { x: f32, p: f32 };\n\
struct DualOut { re: f32, du: f32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<DualIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<DualOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let r = prism_dual_apply(params.op, src[i].x, src[i].p);\n\
    dst[i] = DualOut(r.re, r.du);\n\
}\n";

/// Dual-number op selector, matching the `prism_dual_apply` dispatcher op codes
/// in [`WGSL_DUAL`] and the CPU [`Dual`](prism_math::dual::Dual) method ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DualOp {
    /// Reciprocal `1/x`, derivative `-1/x^2`.
    Recip = 0,
    /// Square root, derivative `1/(2*sqrt x)`.
    Sqrt = 1,
    /// Square `x^2`, derivative `2x`.
    Squared = 2,
    /// Power `x^p` (constant exponent from [`DualSample::p`]), derivative
    /// `p*x^(p-1)`. Requires `x > 0` for a non-integer `p`.
    Powf = 3,
    /// Natural exponential, derivative `e^x`.
    Exp = 4,
    /// Natural logarithm, derivative `1/x`. Requires `x > 0`.
    Ln = 5,
    /// Sine, derivative `cos x`.
    Sin = 6,
    /// Cosine, derivative `-sin x`.
    Cos = 7,
    /// Tangent, derivative `1 + tan^2 x`.
    Tan = 8,
    /// Absolute value, derivative `sign(x)` (zero at the origin).
    Abs = 9,
}

/// Uniform block carrying the valid element count and the dual op code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// A single dual-op input: the seeded variable `x` and the `powf` exponent `p`.
/// The struct is 8 bytes with no padding, matching the WGSL `DualIn` storage
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct DualSample {
    /// The seeded independent variable `x`.
    pub x: f32,
    /// The `powf` exponent (ignored by every other op).
    pub p: f32,
}

impl DualSample {
    /// Seeds a dual variable at `x` with no exponent (for every op but `powf`).
    #[must_use]
    pub fn new(x: f32) -> DualSample {
        DualSample { x, p: 0.0 }
    }

    /// Seeds a dual variable at `x` with the `powf` exponent `p`.
    #[must_use]
    pub fn with_exponent(x: f32, p: f32) -> DualSample {
        DualSample { x, p }
    }
}

/// Raw `std430` dual record read back from the device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DualOut {
    re: f32,
    du: f32,
}

/// A decoded dual result: the value `f(x)` and its exact first derivative
/// `f'(x)`, mirroring [`prism_math::dual::Dual`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GpuDualResult {
    /// Value component `f(x)`.
    pub re: f32,
    /// Derivative component `f'(x)`.
    pub du: f32,
}

impl From<DualOut> for GpuDualResult {
    fn from(raw: DualOut) -> GpuDualResult {
        GpuDualResult {
            re: raw.re,
            du: raw.du,
        }
    }
}

/// Real-device twin of the scalar forward-mode dual-number op set.
///
/// Build it once per device with [`GpuDual::new`]; the single pipeline and
/// bind-group layout are created up front and reused across batches and ops.
pub struct GpuDual {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDual {
    /// Builds the dual-number pipeline, composed from the single-sourced
    /// [`WGSL_DUAL`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDual {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_dual_layout"),
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
            label: Some("prism_math_dual_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_DUAL);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_dual"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_dual"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDual { layout, pipeline }
    }

    /// Batch-applies `op` to each seeded input, returning the value and exact
    /// first derivative per element, mirroring the matching CPU
    /// [`prism_math::dual::Dual`] method on a `Dual::variable(x)` seed.
    #[must_use]
    pub fn apply(&self, ctx: &GpuContext, op: DualOp, src: &[DualSample]) -> Vec<GpuDualResult> {
        let n = src.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_dual_in", src);
        let out_bytes = (n * size_of::<DualOut>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_dual_out", out_bytes);
        let bind_group = self.bind(device, n, op, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_dual_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_dual_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<DualOut>(ctx, &stage)
            .into_iter()
            .map(GpuDualResult::from)
            .collect()
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        op: DualOp,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_dual_params",
            &Params {
                count: count as u32,
                op: op as u32,
                _pad: [0; 2],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_dual_bind_group"),
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
        label: Some("prism_math_dual_pass"),
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

/// Device-free gate: the composed kernel (fragment + wrapper) must be a valid
/// WGSL module on the pinned `naga`.
#[cfg(test)]
mod parse_gate {
    #[test]
    fn composed_kernel_parses() {
        let src = alloc::format!("{}\n{}", super::WGSL_DUAL, super::WRAP);
        if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
            panic!("dual failed to parse: {err}\n--- source ---\n{src}");
        }
    }
}
