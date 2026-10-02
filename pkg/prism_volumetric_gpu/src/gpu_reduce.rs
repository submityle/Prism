//! `wgpu` compute twin of the particle-subsystem *tiled scalar reduction*
//! primitive
//! ([`gpu_reduce`](prism_render_architecture::particle::gpu_reduce), particle
//! design §11 counters, §13 cull thresholds, §28 significance metrics).
//!
//! A *scalar reduction* collapses a per-particle channel (a speed, an age, a
//! significance score, a spawn-counter total) into one value under an
//! associative fold — `min`, `max` or a wrapping / widened `sum`. The `CPU`
//! golden [`gpu_reduce`](prism_render_architecture::particle::gpu_reduce) owns
//! the math as a two-level tiled scheme: the input is split into
//! `workgroup`-sized blocks, each block is folded left to right into one
//! *partial*
//! ([`reduce_all`](prism_render_architecture::particle::gpu_reduce::reduce_all)),
//! and the partials are folded again
//! ([`reduce_partials`](prism_render_architecture::particle::gpu_reduce::reduce_partials))
//! until a single scalar remains
//! ([`ReduceConfig::reduce`](prism_render_architecture::particle::gpu_reduce::ReduceConfig::reduce)).
//!
//! [`GpuReduce`] is the on-device twin. One thread per block folds its own
//! `workgroup`-sized chunk sequentially into a partial on the device; the host
//! then folds the partials with the identical chunk logic the golden uses, so
//! the whole two-level reduction is reproduced. This mirrors the
//! "host aggregates the device partials" pattern the stream-compaction twin
//! [`GpuCompact`](crate::gpu_compact::GpuCompact) uses for its coarse scan.
//!
//! # What is twinned
//!
//! The kernel reproduces the golden per-block first pass: block `b`'s device
//! partial equals
//! [`reduce_all`](prism_render_architecture::particle::gpu_reduce::reduce_all)
//! over block `b`'s elements, and the host second-level fold reproduces
//! [`ReduceConfig::reduce`](prism_render_architecture::particle::gpu_reduce::ReduceConfig::reduce).
//! Both `u32` and `f32` channels are supported through one classified kernel.
//!
//! # Correctness model
//!
//! The `u32` folds (`min`, `max`, and a wrapping `+`) are pure integer algebra:
//! the device `u32` add wraps exactly as the golden `wrapping_add`, and `min` /
//! `max` are associative, so regrouping the serial fold into blocks changes
//! nothing. The `GPU` result therefore equals the serial
//! [`reduce_all`](prism_render_architecture::particle::gpu_reduce::reduce_all)
//! bit for bit and the parity test asserts an exact `==` with no tolerance.
//!
//! The `f32` folds are compared with an epsilon, never for exact equality,
//! because floating-point addition is not associative. The golden widens each
//! block into an `f64` accumulator, which `WGSL` cannot express (it has no
//! `f64`), so the device folds each block in `f32` instead; the host then folds
//! the partials in `f64` exactly as the golden does. The only divergence is the
//! per-block `f32`-vs-`f64` summation rounding, which the bounded fixtures keep
//! well inside the `abs_diff <= 1e-4 || rel_diff <= 1e-3` tolerance. The `f32`
//! `min` / `max` folds introduce no rounding at all, so they agree exactly, but
//! are still compared with the epsilon per the crate's float convention.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned and `f32`
//! `+`, the `min` / `max` built-ins, unsigned comparisons, index arithmetic and
//! `bitcast` (to read `f32` channels and to build the `f32` infinities that
//! seed `min` / `max`). There is no `sqrt`, no transcendental call, no `f64`
//! and no `u64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! only loop walks a single block, bounded by the block width, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
const WORKGROUP_SIZE: u32 = 64;

/// Lowest clamp of the per-block fold width, mirroring the golden
/// [`ReduceConfig`](prism_render_architecture::particle::gpu_reduce::ReduceConfig)
/// clamp so a degenerate zero can never cause a divide-by-zero.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
const MIN_WIDTH: u32 = 1;

/// Highest clamp of the per-block fold width, matching the golden
/// [`ReduceConfig`](prism_render_architecture::particle::gpu_reduce::ReduceConfig)
/// `[1, 1024]` range.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
const MAX_WIDTH: u32 = 1024;

/// The tiled first-pass reduction kernel, mirroring the `CPU` golden
/// [`gpu_reduce`](prism_render_architecture::particle::gpu_reduce). The single
/// entry point `solve` folds one block per thread, embedded inline so the twin
/// ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
const GPU_REDUCE_WGSL: &str = r#"
// gpu_reduce twin: one thread per block reproduces the CPU golden
// `particle::gpu_reduce` first pass. Each thread folds its own workgroup-sized
// block left to right into a single partial under the classified op (0=min,
// 1=max, 2=sum) and scalar kind (0=u32, 1=f32). The host folds the partials
// into the final scalar with the identical chunk logic. The u32 add wraps
// exactly as the golden `wrapping_add`, agreeing bit for bit; the f32 folds
// agree within the crate's float tolerance.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_reduce;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 32-byte uniform block: the element count, the block
// width, the op code, the scalar-kind code, the partial count, and three pad
// words, matching the host `Params`.
struct Params {
    count: u32,
    width: u32,
    op: u32,
    kind: u32,
    partial_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_bits: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;

const OP_MIN: u32 = 0u;
const OP_MAX: u32 = 1u;
const OP_SUM: u32 = 2u;
const KIND_U32: u32 = 0u;

// Positive and negative f32 infinities, built by bitcast since WGSL has no
// infinity literal. These seed the f32 min / max folds, matching the golden
// `+inf` / `-inf` identities.
fn pos_inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block = gid.x;
    if (block >= params.partial_count) {
        return;
    }
    let width = max(params.width, 1u);
    let start = block * width;
    // Clamp the block end to the element count so the final ragged block folds
    // only its real elements. The host guarantees count > 0 and start < count
    // for every valid block, so the block is never empty.
    var end = start + width;
    if (end > params.count) {
        end = params.count;
    }

    if (params.kind == KIND_U32) {
        var acc_u: u32;
        if (params.op == OP_MIN) {
            acc_u = 0xffffffffu;
        } else {
            // Max and Sum share the identity 0.
            acc_u = 0u;
        }
        for (var k: u32 = start; k < end; k = k + 1u) {
            let v = input_bits[k];
            if (params.op == OP_MIN) {
                acc_u = min(acc_u, v);
            } else if (params.op == OP_MAX) {
                acc_u = max(acc_u, v);
            } else {
                // Unsigned add wraps on overflow, matching golden wrapping_add.
                acc_u = acc_u + v;
            }
        }
        partials[block] = acc_u;
    } else {
        var acc_f: f32;
        if (params.op == OP_MIN) {
            acc_f = pos_inf();
        } else if (params.op == OP_MAX) {
            acc_f = neg_inf();
        } else {
            acc_f = 0.0;
        }
        for (var k: u32 = start; k < end; k = k + 1u) {
            let v = bitcast<f32>(input_bits[k]);
            if (params.op == OP_MIN) {
                acc_f = min(acc_f, v);
            } else if (params.op == OP_MAX) {
                acc_f = max(acc_f, v);
            } else {
                acc_f = acc_f + v;
            }
        }
        partials[block] = bitcast<u32>(acc_f);
    }
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_REDUCE_WGSL`]: the element `count`, the block `width`, the
/// `op` code, the scalar-kind code, the `partial_count` and three pad words —
/// `32` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of input elements.
    count: u32,
    /// Elements folded per block (the clamped `workgroup_size`).
    width: u32,
    /// Fold op code: `0` is `Min`, `1` is `Max`, `2` is `Sum`.
    op: u32,
    /// Scalar-kind code: `0` is `u32`, `1` is `f32`.
    kind: u32,
    /// Number of blocks (valid threads), `div_ceil(count, width)`.
    partial_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// The associative fold a reduction applies across its elements, twinning the
/// golden
/// [`ReduceOp`](prism_render_architecture::particle::gpu_reduce::ReduceOp).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuReduceOp {
    /// Smallest element (identity: the type's maximum / `+inf`).
    Min,
    /// Largest element (identity: the type's minimum / `-inf`).
    Max,
    /// Wrapping (`u32`) or `f32` sum (identity: zero).
    Sum,
}

impl GpuReduceOp {
    /// Op code for `Min`, matching the `WGSL` `OP_MIN`.
    const CODE_MIN: u32 = 0;
    /// Op code for `Max`, matching the `WGSL` `OP_MAX`.
    const CODE_MAX: u32 = 1;
    /// Op code for `Sum`, matching the `WGSL` `OP_SUM`.
    const CODE_SUM: u32 = 2;

    /// Returns the `u32` classification code the kernel branches on.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
    #[must_use]
    fn code(self) -> u32 {
        match self {
            GpuReduceOp::Min => GpuReduceOp::CODE_MIN,
            GpuReduceOp::Max => GpuReduceOp::CODE_MAX,
            GpuReduceOp::Sum => GpuReduceOp::CODE_SUM,
        }
    }
}

/// Scalar-kind code for a `u32` channel, matching the `WGSL` `KIND_U32`.
const KIND_U32: u32 = 0;
/// Scalar-kind code for an `f32` channel, matching the `WGSL` `KIND_F32`.
const KIND_F32: u32 = 1;

/// One reduction request: the fold `op`, the `workgroup_size` each simulated
/// block folds, and the typed `input` channel.
///
/// The `workgroup_size` is clamped to the valid `[1, 1024]` range on dispatch,
/// exactly as the golden
/// [`ReduceConfig`](prism_render_architecture::particle::gpu_reduce::ReduceConfig)
/// clamps it, so a degenerate zero can never cause a divide-by-zero.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuReduceQuery {
    /// The associative fold to apply.
    pub op: GpuReduceOp,
    /// Elements folded per block (per simulated `workgroup`).
    pub workgroup_size: u32,
    /// The typed input channel to reduce.
    pub input: GpuReduceInput,
}

/// The typed input channel a reduction folds: either a `u32` or an `f32`
/// per-element array.
///
/// A single classified kernel handles both; the variant selects the `WGSL`
/// scalar-kind code and whether the device folds in `u32` or `f32`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
#[derive(Clone, Debug, PartialEq)]
pub enum GpuReduceInput {
    /// A `u32` channel, folded with exact integer algebra.
    U32(Vec<u32>),
    /// An `f32` channel, folded within the crate's float tolerance.
    F32(Vec<f32>),
}

impl GpuReduceInput {
    /// Number of input elements.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
    #[must_use]
    fn element_count(&self) -> usize {
        match self {
            GpuReduceInput::U32(values) => values.len(),
            GpuReduceInput::F32(values) => values.len(),
        }
    }

    /// The scalar-kind code the kernel branches on.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
    #[must_use]
    fn kind(&self) -> u32 {
        match self {
            GpuReduceInput::U32(_) => KIND_U32,
            GpuReduceInput::F32(_) => KIND_F32,
        }
    }

    /// The input reinterpreted as a raw `u32` bit buffer for upload. `u32`
    /// channels pass through; `f32` channels are bit-reinterpreted so the
    /// kernel can `bitcast` them back.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
    #[must_use]
    fn raw_bits(&self) -> Vec<u32> {
        match self {
            GpuReduceInput::U32(values) => values.clone(),
            GpuReduceInput::F32(values) => values.iter().map(|&v| v.to_bits()).collect(),
        }
    }
}

/// The device-computed answer of one reduction, carrying both the per-block
/// partials (the first-pass output) and the final folded scalar.
///
/// The variant matches the [`GpuReduceInput`] variant of the request. The
/// `partials` vector is empty when the input was empty (the host short-circuits
/// without a dispatch), in which case `reduced` is the `op` identity.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
#[derive(Clone, Debug, PartialEq)]
pub enum GpuReduceResult {
    /// A `u32` reduction: the device partials and the final folded value.
    U32 {
        /// Per-block first-pass partials, in block order.
        partials: Vec<u32>,
        /// The final folded scalar.
        reduced: u32,
    },
    /// An `f32` reduction: the device partials and the final folded value.
    F32 {
        /// Per-block first-pass partials, in block order.
        partials: Vec<f32>,
        /// The final folded scalar.
        reduced: f32,
    },
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
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

/// Folds two `u32` accumulators under `op`, mirroring the golden `u32`
/// `ReduceScalar::fold`: `min`, `max`, or a `wrapping_add`.
fn fold_u32(op: GpuReduceOp, a: u32, b: u32) -> u32 {
    match op {
        GpuReduceOp::Min => a.min(b),
        GpuReduceOp::Max => a.max(b),
        GpuReduceOp::Sum => a.wrapping_add(b),
    }
}

/// The `u32` fold identity, mirroring the golden `u32`
/// `ReduceScalar::identity`: `u32::MAX` for `Min`, `0` for `Max` and `Sum`.
fn identity_u32(op: GpuReduceOp) -> u32 {
    match op {
        GpuReduceOp::Min => u32::MAX,
        GpuReduceOp::Max | GpuReduceOp::Sum => 0,
    }
}

/// Folds two `f64` accumulators under `op`, mirroring the golden `f32`
/// `ReduceScalar::fold` which widens into `f64`.
fn fold_f64(op: GpuReduceOp, a: f64, b: f64) -> f64 {
    match op {
        GpuReduceOp::Min => a.min(b),
        GpuReduceOp::Max => a.max(b),
        GpuReduceOp::Sum => a + b,
    }
}

/// The `f32` fold identity, mirroring the golden `f32`
/// `ReduceScalar::identity`: `+inf` for `Min`, `-inf` for `Max`, `0.0` for
/// `Sum`.
fn identity_f64(op: GpuReduceOp) -> f64 {
    match op {
        GpuReduceOp::Min => f64::INFINITY,
        GpuReduceOp::Max => f64::NEG_INFINITY,
        GpuReduceOp::Sum => 0.0,
    }
}

/// Folds the `u32` device partials into the final scalar, replicating the
/// golden second-level fold: repeatedly chunk the partials by `fold_width` and
/// fold each chunk until a single value remains. For `u32` every op is exactly
/// associative, so the result equals the serial reference.
fn fold_partials_u32(op: GpuReduceOp, partials: &[u32], fold_width: usize) -> u32 {
    let mut current: Vec<u32> = partials.to_vec();
    while current.len() > 1 {
        current = current
            .chunks(fold_width)
            .map(|chunk| {
                chunk
                    .iter()
                    .fold(identity_u32(op), |acc, &v| fold_u32(op, acc, v))
            })
            .collect();
    }
    match current.first() {
        Some(&value) => value,
        None => identity_u32(op),
    }
}

/// Folds the `f32` device partials into the final scalar in `f64` space,
/// replicating the golden second-level fold (`reduce_partials` widens into
/// `f64`). The chunking matches the golden `fold_width`, so the only divergence
/// from the serial reference is the device's first-pass `f32` rounding.
fn fold_partials_f32(op: GpuReduceOp, partials: &[f32], fold_width: usize) -> f32 {
    let mut current: Vec<f32> = partials.to_vec();
    while current.len() > 1 {
        current = current
            .chunks(fold_width)
            .map(|chunk| {
                let acc = chunk
                    .iter()
                    .fold(identity_f64(op), |acc, &v| fold_f64(op, acc, f64::from(v)));
                acc as f32
            })
            .collect();
    }
    match current.first() {
        Some(&value) => value,
        None => identity_f64(op) as f32,
    }
}

/// A compiled, reusable tiled-reduction compute pipeline, twinning the `CPU`
/// golden [`gpu_reduce`](prism_render_architecture::particle::gpu_reduce).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
pub struct GpuReduce {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuReduce {
    /// Compiles the tiled-reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuReduce {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_reduce_module"),
            source: ShaderSource::Wgsl(GPU_REDUCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_reduce_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_reduce_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_reduce_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuReduce {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the full tiled reduction on the device and returns both the
    /// per-block partials and the final folded scalar, in a variant matching
    /// the request's [`GpuReduceInput`].
    ///
    /// An empty `input` issues **no dispatch** — a storage buffer may not be
    /// zero-sized — and returns empty partials with the `op` identity as the
    /// reduced value, exactly as the golden
    /// [`reduce_all`](prism_render_architecture::particle::gpu_reduce::reduce_all)
    /// yields on an empty slice.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_reduce`。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, query: &GpuReduceQuery) -> GpuReduceResult {
        let count = query.input.element_count();
        let op = query.op;
        let width = query.workgroup_size.clamp(MIN_WIDTH, MAX_WIDTH);
        let fold_width = width.max(2) as usize;

        if count == 0 {
            return match query.input {
                GpuReduceInput::U32(_) => GpuReduceResult::U32 {
                    partials: Vec::new(),
                    reduced: identity_u32(op),
                },
                GpuReduceInput::F32(_) => GpuReduceResult::F32 {
                    partials: Vec::new(),
                    reduced: identity_f64(op) as f32,
                },
            };
        }

        let partial_bits = self.dispatch(ctx, query, count, width);

        match query.input {
            GpuReduceInput::U32(_) => {
                let partials = partial_bits;
                let reduced = fold_partials_u32(op, &partials, fold_width);
                GpuReduceResult::U32 { partials, reduced }
            }
            GpuReduceInput::F32(_) => {
                let partials: Vec<f32> = partial_bits.iter().map(|&b| f32::from_bits(b)).collect();
                let reduced = fold_partials_f32(op, &partials, fold_width);
                GpuReduceResult::F32 { partials, reduced }
            }
        }
    }

    /// Issues the one first-pass dispatch and reads back the raw `u32` partial
    /// bits, one per block. Callers must guarantee `count > 0`; the public
    /// [`GpuReduce::evaluate`] short-circuits the empty case before calling in.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        query: &GpuReduceQuery,
        count: usize,
        width: u32,
    ) -> Vec<u32> {
        let device = ctx.device();

        let partial_count = (count as u32).div_ceil(width);
        let params = Params {
            count: count as u32,
            width,
            op: query.op.code(),
            kind: query.input.kind(),
            partial_count,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_reduce_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let input_bits = query.input.raw_bits();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_reduce_input"),
            contents: bytemuck::cast_slice(&input_bits),
            usage: BufferUsages::STORAGE,
        });

        let partial_bytes = (partial_count as u64) * (size_of::<u32>() as u64);
        let partials_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_reduce_partials"),
            size: partial_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let partials_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_reduce_partials_stage"),
            size: partial_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_reduce_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: partials_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_reduce_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_reduce_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per block, flattened to a 1-D dispatch.
            let groups = partial_count.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partials_buf, 0, &partials_stage, 0, partial_bytes);
        ctx.queue().submit([encoder.finish()]);

        partials_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let view = partials_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let partials = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        partials_stage.unmap();

        debug_assert_eq!(partials.len(), partial_count as usize);
        partials
    }
}
