//! `wgpu` compute twin of the device-free indirect draw / dispatch argument
//! packing
//! ([`indirect_draw`](prism_render_architecture::particle::indirect_draw),
//! particle design §9, §15).
//!
//! The `CPU` golden
//! [`indirect_draw`](prism_render_architecture::particle::indirect_draw) owns
//! the little-endian `u32` words that `wgpu` / `WebGPU` `draw_indirect`,
//! `draw_indexed_indirect`, and `dispatch_workgroups_indirect` consume, packed
//! from a resolved alive count and the per-renderer geometry contract:
//! [`DrawIndirectArgs::sprite`](prism_render_architecture::particle::indirect_draw::DrawIndirectArgs::sprite)
//! for billboards,
//! [`DrawIndexedIndirectArgs::mesh`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs::mesh)
//! /
//! [`ribbon`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs::ribbon)
//! /
//! [`beam`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs::beam)
//! for indexed geometry, and
//! [`DispatchIndirectArgs::new`](prism_render_architecture::particle::indirect_draw::DispatchIndirectArgs::new)
//! /
//! [`linear_1d`](prism_render_architecture::particle::indirect_draw::DispatchIndirectArgs::linear_1d)
//! for compute launches. [`GpuIndirectDraw`] is the on-device twin: one thread
//! resolves one [`GpuIndirectDrawQuery`] — a `kind`-tagged bundle of up to five
//! `u32` inputs — into one [`GpuIndirectDrawResult`] carrying the packed words
//! and their count, so a passing real-device parity test is direct evidence the
//! ported kernel reproduces the exact field layout, the saturating
//! segment-to-index product, and the guarded `ceil`-division the reference
//! publishes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `kind` discriminant and five parameter words. The
//! kernel reproduces the per-record `u32` words of the golden packers, one
//! layout per `kind`:
//! - `KIND_SPRITE`: `[SPRITE_QUAD_VERTEX_COUNT, alive, 0, first_instance]`
//!   (`4` words), mirroring
//!   [`DrawIndirectArgs::sprite`](prism_render_architecture::particle::indirect_draw::DrawIndirectArgs::sprite).
//! - `KIND_MESH`: `[index_count, alive, 0, 0, first_instance]` (`5` words),
//!   mirroring
//!   [`DrawIndexedIndirectArgs::mesh`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs::mesh).
//! - `KIND_RIBBON` / `KIND_BEAM`: `[sat_mul(segments, INDICES_PER_SEGMENT),
//!   chain, 0, 0, first_instance]` (`5` words), mirroring the shared
//!   segment-to-index expansion of
//!   [`DrawIndexedIndirectArgs::ribbon`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs::ribbon)
//!   and
//!   [`beam`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs::beam).
//! - `KIND_DISPATCH`: `[x, y, z]` (`3` words), mirroring
//!   [`DispatchIndirectArgs::new`](prism_render_architecture::particle::indirect_draw::DispatchIndirectArgs::new).
//! - `KIND_DISPATCH_LINEAR`: `[clamp(ceil_div(alive, workgroup_size),
//!   max_workgroups), 1, 1]` (`3` words), mirroring
//!   [`DispatchIndirectArgs::linear_1d`](prism_render_architecture::particle::indirect_draw::DispatchIndirectArgs::linear_1d).
//! - `KIND_INDEXED_RAW`: the five raw indexed words `[index_count,
//!   instance_count, first_index, base_vertex_bits, first_instance]`,
//!   mirroring a general
//!   [`DrawIndexedIndirectArgs`](prism_render_architecture::particle::indirect_draw::DrawIndexedIndirectArgs)
//!   whose signed `base_vertex` is carried as its raw two's-complement word.
//!
//! The `GPU` kernel deliberately stops at the per-record `u32` words. The
//! golden
//! [`first_instance_prefix`](prism_render_architecture::particle::indirect_draw::first_instance_prefix)
//! is an exclusive prefix scan whose running total the golden saturates in a
//! wider accumulator; that cross-record aggregate is **not** twinned here and
//! is left to a later prefix-scan pass. Each thread touches only its own
//! record.
//!
//! # Correctness model
//!
//! Every twinned value is a `u32` word: a copied field, the constant quad
//! vertex count, a saturating multiply, or an overflow-safe guarded
//! `ceil`-division. `CPU` and `GPU` therefore compute identical bit patterns,
//! and the parity test asserts an exact `==` on every word and on the word
//! count with no tolerance. Fixtures keep every count well inside `u32` except
//! for the deliberate saturation probes, which pin the saturating multiply and
//! the clamped launch size.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `/`, `%`,
//! `+`, `-`, `*`, unsigned compares and index arithmetic. There is no `sqrt`,
//! no transcendental call, no `u64` and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Each thread performs a fixed,
//! bounded sequence of integer work, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`；无第三方引擎源码或衍生代码。
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
const WORKGROUP_SIZE: u32 = 64;

/// Maximum number of packed words any single layout produces. The widest layout
/// is the five-word indexed draw; shorter layouts leave the trailing words
/// zero and report their own `word_count`.
const MAX_WORDS: usize = 5;

/// The indirect-draw argument kernel, mirroring the `CPU` golden
/// [`indirect_draw`](prism_render_architecture::particle::indirect_draw) packers
/// record for record. The single entry point `solve` resolves one query per
/// thread into the packed `u32` words for its `kind`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
const INDIRECT_DRAW_WGSL: &str = r#"
// indirect_draw twin: one thread per query reproduces the CPU golden
// `particle::indirect_draw`. A query is a kind-tagged bundle of up to five u32
// parameter words; the kernel packs the little-endian u32 words wgpu/WebGPU
// draw_indirect / draw_indexed_indirect / dispatch_workgroups_indirect consume,
// one layout per kind, and reports how many words the layout uses. Unused
// trailing words are left zero. The cross-record first_instance prefix scan the
// golden exposes is NOT computed here; each thread touches only its own record.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::indirect_draw；无第三方
// 引擎源码或衍生代码。

// Kind discriminants, matching the host `GpuIndirectDrawQuery::KIND_*`.
const KIND_SPRITE: u32 = 0u;
const KIND_MESH: u32 = 1u;
const KIND_RIBBON: u32 = 2u;
const KIND_BEAM: u32 = 3u;
const KIND_DISPATCH: u32 = 4u;
const KIND_DISPATCH_LINEAR: u32 = 5u;
const KIND_INDEXED_RAW: u32 = 6u;

// Fixed geometry constants, matching the golden `SPRITE_QUAD_VERTEX_COUNT` and
// `INDICES_PER_SEGMENT`.
const SPRITE_QUAD_VERTEX_COUNT: u32 = 6u;
const INDICES_PER_SEGMENT: u32 = 6u;

// Draw parameters. 16-byte uniform block: the valid query count plus three pad
// words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 24-byte std430 stride of six scalar words, matching the host
// `GpuIndirectDrawQuery`: the kind discriminant plus five parameter words whose
// meaning depends on the kind.
struct DrawQuery {
    kind: u32,
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
}

// One record. 32-byte std430 stride of eight scalar words, matching the host
// `IndirectRecord`: the five packed words, the word count and two pad words.
struct DrawRecord {
    w0: u32,
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    word_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<DrawQuery>;
@group(0) @binding(2) var<storage, read_write> records: array<DrawRecord>;

// Saturating u32 multiply, matching Rust `u32::saturating_mul`.
fn sat_mul_u32(a: u32, b: u32) -> u32 {
    if (a == 0u || b == 0u) {
        return 0u;
    }
    let u32_max = 0xffffffffu;
    if (a > u32_max / b) {
        return u32_max;
    }
    return a * b;
}

// Divide-by-zero-guarded ceil-division with a saturating numerator addition,
// matching the golden `ceil_div_saturating`: a zero divisor yields 0, and the
// round-up never wraps near u32::MAX.
fn ceil_div_sat(numerator: u32, divisor: u32) -> u32 {
    if (divisor == 0u) {
        return 0u;
    }
    let dm1 = divisor - 1u;
    let u32_max = 0xffffffffu;
    var s: u32;
    if (numerator > u32_max - dm1) {
        s = u32_max;
    } else {
        s = numerator + dm1;
    }
    return s / divisor;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var w0: u32 = 0u;
    var w1: u32 = 0u;
    var w2: u32 = 0u;
    var w3: u32 = 0u;
    var w4: u32 = 0u;
    var n: u32 = 0u;

    if (q.kind == KIND_SPRITE) {
        // [vertex_count, instance_count, first_vertex, first_instance].
        w0 = SPRITE_QUAD_VERTEX_COUNT;
        w1 = q.p0;
        w2 = 0u;
        w3 = q.p1;
        n = 4u;
    } else if (q.kind == KIND_MESH) {
        // [index_count, instance_count, first_index, base_vertex, first_instance].
        w0 = q.p1;
        w1 = q.p0;
        w2 = 0u;
        w3 = 0u;
        w4 = q.p2;
        n = 5u;
    } else if (q.kind == KIND_RIBBON || q.kind == KIND_BEAM) {
        // Ribbon and beam share the segment-to-index expansion exactly.
        w0 = sat_mul_u32(q.p1, INDICES_PER_SEGMENT);
        w1 = q.p0;
        w2 = 0u;
        w3 = 0u;
        w4 = q.p2;
        n = 5u;
    } else if (q.kind == KIND_DISPATCH) {
        // [x, y, z].
        w0 = q.p0;
        w1 = q.p1;
        w2 = q.p2;
        n = 3u;
    } else if (q.kind == KIND_DISPATCH_LINEAR) {
        // x = clamp(ceil(alive / workgroup_size), max_workgroups); y = z = 1.
        let needed = ceil_div_sat(q.p0, q.p1);
        var x: u32 = needed;
        if (needed > q.p2) {
            x = q.p2;
        }
        w0 = x;
        w1 = 1u;
        w2 = 1u;
        n = 3u;
    } else if (q.kind == KIND_INDEXED_RAW) {
        // Verbatim [index_count, instance_count, first_index, base_vertex_bits,
        // first_instance].
        w0 = q.p0;
        w1 = q.p1;
        w2 = q.p2;
        w3 = q.p3;
        w4 = q.p4;
        n = 5u;
    }

    var out: DrawRecord;
    out.w0 = w0;
    out.w1 = w1;
    out.w2 = w2;
    out.w3 = w3;
    out.w4 = w4;
    out.word_count = n;
    out.pad0 = 0u;
    out.pad1 = 0u;
    records[idx] = out;
}
"#;

/// Uniform parameters for one evaluation. `repr(C)` `std430` layout matching
/// `Params` in [`INDIRECT_DRAW_WGSL`]: the valid query count plus three pad
/// words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One raw device record read back from the kernel. `repr(C)` `std430` layout
/// matching the `WGSL` `DrawRecord`: the five packed `u32` words, the
/// `word_count`, and two pad words — `32` bytes with no interior padding. This
/// stays private; the public [`GpuIndirectDrawResult`] wraps it and exposes the
/// meaningful prefix.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndirectRecord {
    /// Packed word `0`.
    w0: u32,
    /// Packed word `1`.
    w1: u32,
    /// Packed word `2`.
    w2: u32,
    /// Packed word `3`.
    w3: u32,
    /// Packed word `4`.
    w4: u32,
    /// Number of meaningful words this layout produced (`3`, `4` or `5`).
    word_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One query for the indirect-draw twin: a `kind` discriminant tagging the
/// layout to pack, plus five parameter words whose meaning depends on `kind`.
///
/// The parameter words are interpreted per `kind`:
/// - [`KIND_SPRITE`](GpuIndirectDrawQuery::KIND_SPRITE): `p0` = `alive_count`,
///   `p1` = `first_instance`.
/// - [`KIND_MESH`](GpuIndirectDrawQuery::KIND_MESH): `p0` = `alive_count`, `p1`
///   = `index_count`, `p2` = `first_instance`.
/// - [`KIND_RIBBON`](GpuIndirectDrawQuery::KIND_RIBBON) /
///   [`KIND_BEAM`](GpuIndirectDrawQuery::KIND_BEAM): `p0` = `chain_count`, `p1`
///   = `segments_per_chain`, `p2` = `first_instance`.
/// - [`KIND_DISPATCH`](GpuIndirectDrawQuery::KIND_DISPATCH): `p0` = `x`, `p1` =
///   `y`, `p2` = `z`.
/// - [`KIND_DISPATCH_LINEAR`](GpuIndirectDrawQuery::KIND_DISPATCH_LINEAR): `p0`
///   = `alive_count`, `p1` = `workgroup_size`, `p2` = `max_workgroups`.
/// - [`KIND_INDEXED_RAW`](GpuIndirectDrawQuery::KIND_INDEXED_RAW): `p0` =
///   `index_count`, `p1` = `instance_count`, `p2` = `first_index`, `p3` =
///   `base_vertex` raw bits, `p4` = `first_instance`.
///
/// The `repr(C)` layout — six `u32` words, `24` bytes with no padding — matches
/// the `WGSL` `DrawQuery` struct exactly, so it is uploaded to the device
/// without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuIndirectDrawQuery {
    /// Layout discriminant, one of the `KIND_*` associated constants.
    pub kind: u32,
    /// Parameter word `0`.
    pub p0: u32,
    /// Parameter word `1`.
    pub p1: u32,
    /// Parameter word `2`.
    pub p2: u32,
    /// Parameter word `3`.
    pub p3: u32,
    /// Parameter word `4`.
    pub p4: u32,
}

impl GpuIndirectDrawQuery {
    /// Non-indexed `Sprite` layout: a fixed quad instanced once per alive
    /// particle.
    pub const KIND_SPRITE: u32 = 0;
    /// Indexed `Mesh` layout: one indexed mesh instance per alive particle.
    pub const KIND_MESH: u32 = 1;
    /// Indexed `Ribbon` layout: one strip instance per chain, segments expanded
    /// to indices.
    pub const KIND_RIBBON: u32 = 2;
    /// Indexed `Beam` layout: identical segment-to-index expansion as the
    /// ribbon.
    pub const KIND_BEAM: u32 = 3;
    /// Explicit compute dispatch layout: the three workgroup counts verbatim.
    pub const KIND_DISPATCH: u32 = 4;
    /// Linear 1-D compute dispatch layout: a clamped `ceil`-division launch
    /// size with `y = z = 1`.
    pub const KIND_DISPATCH_LINEAR: u32 = 5;
    /// Verbatim five-word indexed layout, carrying the signed `base_vertex` as
    /// its raw word.
    pub const KIND_INDEXED_RAW: u32 = 6;

    /// A [`KIND_SPRITE`](Self::KIND_SPRITE) query for `alive_count` particles
    /// offset by `first_instance`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn sprite(alive_count: u32, first_instance: u32) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_SPRITE,
            p0: alive_count,
            p1: first_instance,
            p2: 0,
            p3: 0,
            p4: 0,
        }
    }

    /// A [`KIND_MESH`](Self::KIND_MESH) query for `alive_count` instances of a
    /// mesh with `index_count` indices, offset by `first_instance`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn mesh(
        alive_count: u32,
        index_count: u32,
        first_instance: u32,
    ) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_MESH,
            p0: alive_count,
            p1: index_count,
            p2: first_instance,
            p3: 0,
            p4: 0,
        }
    }

    /// A [`KIND_RIBBON`](Self::KIND_RIBBON) query for `chain_count` chains of
    /// `segments_per_chain` segments, offset by `first_instance`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn ribbon(
        chain_count: u32,
        segments_per_chain: u32,
        first_instance: u32,
    ) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_RIBBON,
            p0: chain_count,
            p1: segments_per_chain,
            p2: first_instance,
            p3: 0,
            p4: 0,
        }
    }

    /// A [`KIND_BEAM`](Self::KIND_BEAM) query for `chain_count` chains of
    /// `segments_per_chain` segments, offset by `first_instance`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn beam(
        chain_count: u32,
        segments_per_chain: u32,
        first_instance: u32,
    ) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_BEAM,
            p0: chain_count,
            p1: segments_per_chain,
            p2: first_instance,
            p3: 0,
            p4: 0,
        }
    }

    /// A [`KIND_DISPATCH`](Self::KIND_DISPATCH) query packing the explicit
    /// workgroup counts `x`, `y`, `z`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn dispatch(x: u32, y: u32, z: u32) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_DISPATCH,
            p0: x,
            p1: y,
            p2: z,
            p3: 0,
            p4: 0,
        }
    }

    /// A [`KIND_DISPATCH_LINEAR`](Self::KIND_DISPATCH_LINEAR) query for a 1-D
    /// launch covering `alive_count` invocations at `workgroup_size` per group,
    /// clamped to `max_workgroups`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn dispatch_linear(
        alive_count: u32,
        workgroup_size: u32,
        max_workgroups: u32,
    ) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_DISPATCH_LINEAR,
            p0: alive_count,
            p1: workgroup_size,
            p2: max_workgroups,
            p3: 0,
            p4: 0,
        }
    }

    /// A [`KIND_INDEXED_RAW`](Self::KIND_INDEXED_RAW) query carrying the five
    /// indexed words verbatim, with `base_vertex_bits` the raw two's-complement
    /// word of a signed `base_vertex`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub const fn indexed_raw(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex_bits: u32,
        first_instance: u32,
    ) -> GpuIndirectDrawQuery {
        GpuIndirectDrawQuery {
            kind: Self::KIND_INDEXED_RAW,
            p0: index_count,
            p1: instance_count,
            p2: first_index,
            p3: base_vertex_bits,
            p4: first_instance,
        }
    }
}

/// One resolved answer for a single [`GpuIndirectDrawQuery`]: the packed `u32`
/// words and how many of them are meaningful for the query's layout.
///
/// `words` always holds `MAX_WORDS` entries; only the first `word_count`
/// (`3`, `4` or `5`) are the indirect-buffer words for the query's `kind`, and
/// the trailing entries are zero. These are exactly the little-endian words the
/// golden packers emit, so parity is an exact `==` over the meaningful prefix.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuIndirectDrawResult {
    /// The packed indirect-buffer words; only the first `word_count` are
    /// meaningful, the rest are zero.
    pub words: [u32; MAX_WORDS],
    /// Number of meaningful words in `words` (`3`, `4` or `5`).
    pub word_count: u32,
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

/// A compiled, reusable indirect-draw argument compute pipeline, twinning the
/// `CPU` golden
/// [`indirect_draw`](prism_render_architecture::particle::indirect_draw).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
pub struct GpuIndirectDraw {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIndirectDraw {
    /// Compiles the indirect-draw argument kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIndirectDraw {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_indirect_draw_module"),
            source: ShaderSource::Wgsl(INDIRECT_DRAW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_indirect_draw_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_indirect_draw_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_indirect_draw_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIndirectDraw {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuIndirectDrawResult`] per input, in order.
    ///
    /// Each result's meaningful word prefix equals the matching golden packer's
    /// `as_words` output exactly, because the whole on-device path is unsigned
    /// integer word assembly. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::indirect_draw`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuIndirectDrawQuery],
    ) -> Vec<GpuIndirectDrawResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_indirect_draw_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_indirect_draw_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<IndirectRecord>() as u64);
        let records_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_indirect_draw_records"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let records_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_indirect_draw_records_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_indirect_draw_bind_group"),
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
                    resource: records_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_indirect_draw_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_indirect_draw_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&records_buf, 0, &records_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        records_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = records_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let records = bytemuck::cast_slice::<u8, IndirectRecord>(&view).to_vec();
        drop(view);
        records_stage.unmap();
        debug_assert_eq!(records.len(), queries.len());

        records
            .into_iter()
            .map(|r| GpuIndirectDrawResult {
                words: [r.w0, r.w1, r.w2, r.w3, r.w4],
                word_count: r.word_count,
            })
            .collect()
    }
}
