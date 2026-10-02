//! `wgpu` compute twin of the fixed-function `stencil` test/operation state
//! machine
//! ([`stencil_op`](prism_render_architecture::particle::stencil_op), particle
//! design §9 render pass, §16 shading router).
//!
//! The `CPU` golden
//! [`stencil_op`](prism_render_architecture::particle::stencil_op) owns the
//! pure, device-free half of the hardware `stencil` unit: the eight comparison
//! predicates
//! ([`CompareFunc::test`](prism_render_architecture::particle::stencil_op::CompareFunc::test)),
//! the eight buffer operations
//! ([`StencilOp::apply`](prism_render_architecture::particle::stencil_op::StencilOp::apply)),
//! the masked write-back
//! ([`write_masked`](prism_render_architecture::particle::stencil_op::write_masked)),
//! the per-face operation select
//! ([`StencilFace::selected_op`](prism_render_architecture::particle::stencil_op::StencilFace::selected_op))
//! and the full per-face resolve
//! ([`StencilFace::resolve`](prism_render_architecture::particle::stencil_op::StencilFace::resolve),
//! which [`StencilState::resolve`](prism_render_architecture::particle::stencil_op::StencilState::resolve)
//! delegates to after picking the front or back face).
//! [`GpuStencilOp`] is the on-device twin: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel runs
//! the same integer state machine the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the compare predicate result
//! ([`CompareFunc::test`](prism_render_architecture::particle::stencil_op::CompareFunc::test)
//! of the read-masked reference and buffer values), the selected operation code
//! ([`StencilFace::selected_op`](prism_render_architecture::particle::stencil_op::StencilFace::selected_op)
//! for the supplied `compare_passed` / `depth_passed` outcome), the resolved
//! new buffer value
//! ([`StencilFace::resolve`](prism_render_architecture::particle::stencil_op::StencilFace::resolve),
//! which recomputes its own compare pass then applies and write-masks the
//! operation), and a standalone
//! ([`write_masked`](prism_render_architecture::particle::stencil_op::write_masked))
//! of the reference value into the buffer value under the write mask. The eight
//! [`StencilOp`](prism_render_architecture::particle::stencil_op::StencilOp)
//! operations and eight
//! [`CompareFunc`](prism_render_architecture::particle::stencil_op::CompareFunc)
//! predicates are dispatched by their stable `to_u32` codes, which the host
//! packs into each query.
//!
//! # Correctness model
//!
//! The entire contract is exact `u32` / `bool` integer bit-twiddling — bitwise
//! `and` / `or` / `not`, masked compares, saturating and wrapping increment and
//! decrement — with no `f32` and no rounding anywhere, so the `CPU` reference
//! and the `GPU` kernel agree bit for bit. The parity test therefore asserts an
//! exact `==` on every output word with no tolerance: any mismatch is a genuine
//! port bug.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — bitwise operators,
//! unsigned compares, `min`, `select` and a `switch` on the operation and
//! predicate codes — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no `sqrt`
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. There is no loop: each thread performs a fixed, bounded sequence
//! of integer arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`stencil_op`](prism_render_architecture::particle::stencil_op); no
//! third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::stencil_op::{CompareFunc, StencilOp};
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
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `stencil` state-machine kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`stencil_op`](prism_render_architecture::particle::stencil_op) branch for
/// branch; see the module documentation for the algorithm.
const STENCIL_OP_WGSL: &str = r#"
// Fixed-function stencil state-machine twin: one thread per query reproduces the
// compare predicate, the selected operation code, the resolved new buffer value
// and a standalone write-masked blend. It mirrors the CPU golden
// particle::stencil_op branch for branch, uses only the portable core-WGSL
// subset (bitwise operators, unsigned compares, min/select and a switch on the
// enum codes) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12. The whole contract is exact u32/bool integer algebra, so the
// CPU and GPU agree bit for bit. There is no loop, so the kernel provably
// terminates.
//
// Provenance: twinned from this repository's particle::stencil_op; no
// third-party engine source or derived code.

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // CompareFunc code (0..=7) and the three StencilOp codes selected by the
    // compare/depth outcome: fail, depth-fail and pass.
    compare_func: u32,
    fail_op: u32,
    depth_fail_op: u32,
    pass_op: u32,
    // Reference value, current buffer value, read mask, write mask and the
    // wrap/clamp ceiling.
    ref_val: u32,
    stencil_val: u32,
    read_mask: u32,
    write_mask: u32,
    max_val: u32,
    // Supplied compare/depth outcomes for the standalone selected-op query (as
    // u32 booleans); a pad word fills the slot.
    compare_passed: u32,
    depth_passed: u32,
    pad0: u32,
}

struct Result {
    // CompareFunc::test as a u32 boolean, the selected StencilOp code, the
    // resolved new buffer value and the standalone write-masked blend.
    test_result: u32,
    selected_op: u32,
    resolved: u32,
    write_masked_val: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// CompareFunc::test: mask both operands with read_mask, then apply the predicate
// selected by the stable code. Mirrors the reference match arm for arm.
fn compare_test(func: u32, ref_val: u32, stencil_val: u32, read_mask: u32) -> bool {
    let masked_ref = ref_val & read_mask;
    let masked_val = stencil_val & read_mask;
    switch (func) {
        case 0u: { return false; }                 // Never
        case 1u: { return masked_ref < masked_val; }   // Less
        case 2u: { return masked_ref == masked_val; }  // Equal
        case 3u: { return masked_ref <= masked_val; }  // LessEqual
        case 4u: { return masked_ref > masked_val; }    // Greater
        case 5u: { return masked_ref != masked_val; }  // NotEqual
        case 6u: { return masked_ref >= masked_val; }  // GreaterEqual
        default: { return true; }                   // Always (7u)
    }
}

// StencilOp::apply: run the operation selected by the stable code against the
// current value, using ref_val for Replace and max_val as the wrap/clamp
// ceiling. Mirrors the reference saturating/wrapping integer arithmetic.
fn stencil_apply(op: u32, current: u32, ref_val: u32, max_val: u32) -> u32 {
    switch (op) {
        case 0u: { return current; }   // Keep
        case 1u: { return 0u; }        // Zero
        case 2u: { return ref_val; }   // Replace
        case 3u: {                     // IncrementClamp: saturating_add(1).min(max)
            let incremented = select(current + 1u, 0xFFFFFFFFu, current == 0xFFFFFFFFu);
            return min(incremented, max_val);
        }
        case 4u: {                     // DecrementClamp: saturating_sub(1)
            return select(current - 1u, 0u, current == 0u);
        }
        case 5u: { return ~current; }  // Invert
        case 6u: {                     // IncrementWrap: wrap to 0 past max
            if (current >= max_val) {
                return 0u;
            }
            return current + 1u;
        }
        default: {                     // DecrementWrap (7u): wrap to max from 0
            if (current == 0u) {
                return max_val;
            }
            return current - 1u;
        }
    }
}

// StencilFace::selected_op: fail op when the compare fails, pass op when both
// compare and depth pass, depth-fail op otherwise.
fn selected_op(fail_op: u32, depth_fail_op: u32, pass_op: u32, compare_passed: bool, depth_passed: bool) -> u32 {
    if (!compare_passed) {
        return fail_op;
    }
    if (depth_passed) {
        return pass_op;
    }
    return depth_fail_op;
}

// write_masked: keep the surviving bits of old outside the write mask and take
// the incoming bits of new_val inside it.
fn write_masked(old: u32, new_val: u32, write_mask: u32) -> u32 {
    return (old & ~write_mask) | (new_val & write_mask);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // CompareFunc::test on the read-masked reference and buffer values.
    let test_pass = compare_test(q.compare_func, q.ref_val, q.stencil_val, q.read_mask);

    // StencilFace::selected_op for the supplied (host) compare/depth outcome.
    let sel = selected_op(
        q.fail_op,
        q.depth_fail_op,
        q.pass_op,
        q.compare_passed != 0u,
        q.depth_passed != 0u,
    );

    // StencilFace::resolve: recompute the compare pass internally, select the op
    // from that pass and the supplied depth outcome, apply it, then write back
    // under the write mask.
    let resolve_op = selected_op(
        q.fail_op,
        q.depth_fail_op,
        q.pass_op,
        test_pass,
        q.depth_passed != 0u,
    );
    let new_val = stencil_apply(resolve_op, q.stencil_val, q.ref_val, q.max_val);
    let resolved = write_masked(q.stencil_val, new_val, q.write_mask);

    // Standalone write_masked of the reference value into the buffer value.
    let wm = write_masked(q.stencil_val, q.ref_val, q.write_mask);

    var out: Result;
    out.test_result = select(0u, 1u, test_pass);
    out.selected_op = sel;
    out.resolved = resolved;
    out.write_masked_val = wm;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`STENCIL_OP_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// twelve `u32` words (eleven live fields plus one pad), all `4`-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `CompareFunc` code (`0..=7`).
    compare_func: u32,
    /// `StencilOp` code applied when the compare fails.
    fail_op: u32,
    /// `StencilOp` code applied when the compare passes but depth fails.
    depth_fail_op: u32,
    /// `StencilOp` code applied when both compare and depth pass.
    pass_op: u32,
    /// Reference value compared against the buffer and used by `Replace`.
    ref_val: u32,
    /// Current buffer value.
    stencil_val: u32,
    /// Mask applied to both operands before the compare.
    read_mask: u32,
    /// Mask restricting which bits the resolve writes back.
    write_mask: u32,
    /// Wrap/clamp ceiling for the increment and decrement operations.
    max_val: u32,
    /// Supplied compare outcome for the standalone selected-op query.
    compare_passed: u32,
    /// Supplied depth outcome for the selected-op and resolve queries.
    depth_passed: u32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// four `4`-byte `u32` words, a `16`-byte array element.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `CompareFunc::test` result as a `u32` boolean.
    test_result: u32,
    /// Selected `StencilOp` code from `StencilFace::selected_op`.
    selected_op: u32,
    /// Resolved new buffer value from `StencilFace::resolve`.
    resolved: u32,
    /// Standalone `write_masked` of the reference into the buffer value.
    write_masked_val: u32,
}

/// One `stencil` query: a programmed compare predicate and its three selectable
/// operations, the reference, buffer, mask and ceiling words, and the supplied
/// compare/depth outcome the selected-op query consumes.
///
/// The compare predicate and the three operations are supplied as the golden
/// [`CompareFunc`](prism_render_architecture::particle::stencil_op::CompareFunc)
/// and [`StencilOp`](prism_render_architecture::particle::stencil_op::StencilOp)
/// enums; the host encodes them into the device buffer through their stable
/// `to_u32` codes.
///
/// Provenance: twinned from this repository's
/// [`stencil_op`](prism_render_architecture::particle::stencil_op); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilOpQuery {
    /// Comparison predicate applied by
    /// [`CompareFunc::test`](prism_render_architecture::particle::stencil_op::CompareFunc::test).
    pub compare: CompareFunc,
    /// Operation applied when the compare fails.
    pub fail_op: StencilOp,
    /// Operation applied when the compare passes but the depth test fails.
    pub depth_fail_op: StencilOp,
    /// Operation applied when both the compare and depth test pass.
    pub pass_op: StencilOp,
    /// Reference value compared against the buffer and used by
    /// [`StencilOp::Replace`](prism_render_architecture::particle::stencil_op::StencilOp::Replace).
    pub ref_val: u32,
    /// Current buffer value.
    pub stencil_val: u32,
    /// Mask applied to both operands before the compare.
    pub read_mask: u32,
    /// Mask restricting which bits the resolve writes back.
    pub write_mask: u32,
    /// Wrap/clamp ceiling for the increment and decrement operations.
    pub max_val: u32,
    /// Supplied compare outcome fed to the standalone selected-op query.
    pub compare_passed: bool,
    /// Supplied depth outcome fed to the selected-op and resolve queries.
    pub depth_passed: bool,
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions. Every field is exact.
///
/// Provenance: twinned from this repository's
/// [`stencil_op`](prism_render_architecture::particle::stencil_op); no
/// third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilOpResult {
    /// Compare predicate result, matching
    /// [`CompareFunc::test`](prism_render_architecture::particle::stencil_op::CompareFunc::test).
    pub test: bool,
    /// Selected operation code, matching
    /// [`StencilFace::selected_op`](prism_render_architecture::particle::stencil_op::StencilFace::selected_op)
    /// followed by
    /// [`StencilOp::to_u32`](prism_render_architecture::particle::stencil_op::StencilOp::to_u32).
    pub selected_op: u32,
    /// Resolved new buffer value, matching
    /// [`StencilFace::resolve`](prism_render_architecture::particle::stencil_op::StencilFace::resolve).
    pub resolved: u32,
    /// Standalone write-mask blend of the reference into the buffer value,
    /// matching
    /// [`write_masked`](prism_render_architecture::particle::stencil_op::write_masked).
    pub write_masked: u32,
}

/// Encodes one [`StencilOpQuery`] into its `std430` [`GpuQuery`] slot, packing
/// the compare and operation enums through their stable `to_u32` codes.
fn encode_query(q: &StencilOpQuery) -> GpuQuery {
    GpuQuery {
        compare_func: q.compare.to_u32(),
        fail_op: q.fail_op.to_u32(),
        depth_fail_op: q.depth_fail_op.to_u32(),
        pass_op: q.pass_op.to_u32(),
        ref_val: q.ref_val,
        stencil_val: q.stencil_val,
        read_mask: q.read_mask,
        write_mask: q.write_mask,
        max_val: q.max_val,
        compare_passed: u32::from(q.compare_passed),
        depth_passed: u32::from(q.depth_passed),
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`StencilOpResult`], turning
/// the `test` word back into a `bool`.
fn decode_result(raw: &GpuResult) -> StencilOpResult {
    StencilOpResult {
        test: raw.test_result != 0,
        selected_op: raw.selected_op,
        resolved: raw.resolved,
        write_masked: raw.write_masked_val,
    }
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

/// A compiled, reusable `stencil` state-machine compute pipeline, twinning the
/// `CPU` golden
/// [`stencil_op`](prism_render_architecture::particle::stencil_op).
pub struct GpuStencilOp {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStencilOp {
    /// Compiles the `stencil` state-machine kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStencilOp {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_stencil_op"),
            source: ShaderSource::Wgsl(STENCIL_OP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_stencil_op_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_stencil_op_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_stencil_op_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStencilOp {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`StencilOpResult`] per
    /// input, in order.
    ///
    /// Every output word equals the reference exactly, since the whole contract
    /// is exact `u32` / `bool` integer algebra. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[StencilOpQuery]) -> Vec<StencilOpResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_stencil_op_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_stencil_op_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_stencil_op_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_stencil_op_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_stencil_op_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_stencil_op_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_stencil_op_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
