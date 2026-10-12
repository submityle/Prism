//! Host orchestration of the GPU conservatively-rounded interval-arithmetic
//! kernels (§24.3 twin: "never miss" enclosures for GPU culling / CCD).
//!
//! [`GpuInterval`] batch-evaluates the interval enclosure ops **on a real
//! device**, mirroring the CPU [`Interval`](prism_math::interval::Interval)
//! arithmetic: `add`/`sub`/`mul`/`div` (lower bound rounded down, upper bound
//! rounded up through the ULP-step helpers) plus `abs`/`sqrt`. Every result
//! provably encloses the exact real result, so a device culling / broad-phase
//! decision built on it can never reject a visible primitive. The arithmetic is
//! **not** duplicated here: the kernels are composed at runtime by prefixing the
//! single-sourced fragment
//! [`WGSL_INTERVAL`](prism_math::shader_mirror::WGSL_INTERVAL) ahead of thin
//! compute wrappers, so the device enclosure cannot silently drift from the CPU
//! reference — it is literally the same text.
//!
//! # Parity
//!
//! Each op is a few single-precision operations followed by a deterministic
//! one-ULP `bitcast` step. In the ideal the device result is bit-exact to the
//! CPU, but Metal compiles WGSL under fast-math, so the raw bounds can differ by
//! a rounding step and the `next_up`/`next_down` step then lands one ULP away.
//! The parity suite therefore compares the bounds at a small absolute+relative
//! epsilon (and treats the whole-line `[-inf, +inf]` enclosure specially),
//! which still rejects a genuine operand-order or rounding-direction drift.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_INTERVAL;
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

/// Compute wrapper for the batched binary interval ops. Each item is four
/// scalars: operand `a`'s `[lo, hi]` then operand `b`'s `[lo, hi]`.
const WRAP_BINARY: &str = "\
struct Params { count: u32, op: u32 };\n\
struct BinIn { a_lo: f32, a_hi: f32, b_lo: f32, b_hi: f32 };\n\
struct IntervalOut { lo: f32, hi: f32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<BinIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<IntervalOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let a = PrismInterval(src[i].a_lo, src[i].a_hi);\n\
    let b = PrismInterval(src[i].b_lo, src[i].b_hi);\n\
    let r = prism_interval_binary(params.op, a, b);\n\
    dst[i] = IntervalOut(r.lo, r.hi);\n\
}\n";

/// Compute wrapper for the batched unary interval ops. Each item is one
/// operand's `[lo, hi]`.
const WRAP_UNARY: &str = "\
struct Params { count: u32, op: u32 };\n\
struct UnIn { lo: f32, hi: f32 };\n\
struct IntervalOut { lo: f32, hi: f32 };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<UnIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<IntervalOut>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let a = PrismInterval(src[i].lo, src[i].hi);\n\
    let r = prism_interval_unary(params.op, a);\n\
    dst[i] = IntervalOut(r.lo, r.hi);\n\
}\n";

/// Binary interval-op selector, matching the `prism_interval_binary` dispatcher
/// op codes in [`WGSL_INTERVAL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum IntervalBinaryOp {
    /// Sum enclosure `[a.lo + b.lo, a.hi + b.hi]`.
    Add = 0,
    /// Difference enclosure `[a.lo - b.hi, a.hi - b.lo]`.
    Sub = 1,
    /// Product enclosure over the four corner products.
    Mul = 2,
    /// Quotient enclosure; the whole line when the divisor straddles zero.
    Div = 3,
}

/// Unary interval-op selector, matching the `prism_interval_unary` dispatcher
/// op codes in [`WGSL_INTERVAL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum IntervalUnaryOp {
    /// Absolute-value enclosure.
    Abs = 0,
    /// Square-root enclosure, clamping negative parts to zero.
    Sqrt = 1,
}

/// Uniform block carrying the valid element count and the interval op code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// A binary interval operand pair: `a = [a_lo, a_hi]`, `b = [b_lo, b_hi]`. The
/// struct is 16 bytes with no padding, matching the WGSL `BinIn` storage stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct IntervalPair {
    /// Lower bound of the first operand.
    pub a_lo: f32,
    /// Upper bound of the first operand.
    pub a_hi: f32,
    /// Lower bound of the second operand.
    pub b_lo: f32,
    /// Upper bound of the second operand.
    pub b_hi: f32,
}

impl IntervalPair {
    /// Builds a pair from two `[lo, hi]` operands.
    #[must_use]
    pub fn new(a: [f32; 2], b: [f32; 2]) -> IntervalPair {
        IntervalPair {
            a_lo: a[0],
            a_hi: a[1],
            b_lo: b[0],
            b_hi: b[1],
        }
    }
}

/// Raw `std430` interval record read back from the device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IntervalOut {
    lo: f32,
    hi: f32,
}

/// A decoded interval enclosure, mirroring the CPU
/// [`Interval`](prism_math::interval::Interval) bounds.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IntervalBounds {
    /// Lower bound `lo`.
    pub lo: f32,
    /// Upper bound `hi`.
    pub hi: f32,
}

impl From<IntervalOut> for IntervalBounds {
    fn from(raw: IntervalOut) -> IntervalBounds {
        IntervalBounds {
            lo: raw.lo,
            hi: raw.hi,
        }
    }
}

/// Real-device twin of the conservatively-rounded interval arithmetic.
///
/// Build it once per device with [`GpuInterval::new`]; the two pipelines and
/// bind-group layout are created up front and reused across batches and ops.
pub struct GpuInterval {
    layout: BindGroupLayout,
    binary: ComputePipeline,
    unary: ComputePipeline,
}

impl GpuInterval {
    /// Builds the binary and unary interval pipelines, each composed from the
    /// single-sourced [`WGSL_INTERVAL`] fragment plus its thin wrapper.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuInterval {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_interval_layout"),
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
            label: Some("prism_math_interval_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_INTERVAL);
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
        GpuInterval {
            binary: build("prism_math_interval_binary", WRAP_BINARY),
            unary: build("prism_math_interval_unary", WRAP_UNARY),
            layout,
        }
    }

    /// Batch-evaluates a binary interval op over `pairs`, mirroring the CPU
    /// [`Interval`](prism_math::interval::Interval) operators.
    #[must_use]
    pub fn binary(
        &self,
        ctx: &GpuContext,
        op: IntervalBinaryOp,
        pairs: &[IntervalPair],
    ) -> Vec<IntervalBounds> {
        self.run(ctx, &self.binary, op as u32, pairs)
    }

    /// Batch-evaluates a unary interval op over `items`, each `[lo, hi]`,
    /// mirroring the CPU [`Interval`](prism_math::interval::Interval) methods.
    #[must_use]
    pub fn unary(
        &self,
        ctx: &GpuContext,
        op: IntervalUnaryOp,
        items: &[[f32; 2]],
    ) -> Vec<IntervalBounds> {
        self.run(ctx, &self.unary, op as u32, items)
    }

    /// Uploads the count/op uniform + a `T` input batch, dispatches `pipeline`,
    /// and reads back one interval bound pair per element.
    fn run<T: Pod>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        op: u32,
        items: &[T],
    ) -> Vec<IntervalBounds> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let in_buf = buffer::storage_read(device, "prism_math_interval_in", items);
        let out_bytes = (n * size_of::<IntervalOut>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_interval_out", out_bytes);
        let bind_group = self.bind(device, n, op, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_interval_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_interval_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<IntervalOut>(ctx, &stage)
            .into_iter()
            .map(IntervalBounds::from)
            .collect()
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        op: u32,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_interval_params",
            &Params {
                count: count as u32,
                op,
                _pad: [0; 2],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_interval_bind_group"),
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
        label: Some("prism_math_interval_pass"),
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
        for (label, wrapper) in [("binary", super::WRAP_BINARY), ("unary", super::WRAP_UNARY)] {
            let src = alloc::format!("{}\n{}", super::WGSL_INTERVAL, wrapper);
            if let Err(err) = wgpu::naga::front::wgsl::parse_str(&src) {
                panic!("interval `{label}` failed to parse: {err}\n--- source ---\n{src}");
            }
        }
    }
}
