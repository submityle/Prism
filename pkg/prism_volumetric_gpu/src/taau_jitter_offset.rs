//! `wgpu` compute twin of the deterministic sub-pixel jitter primitives inside
//! the temporal-upscale jitter contract
//! ([`jitter`](prism_render_architecture::temporal_upscale::jitter)).
//!
//! The `CPU` golden
//! [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
//! reflects the base-`b` digits of an index about the radix point with pure
//! integer division/modulo and a running reciprocal-power multiply, and
//! [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase)
//! builds a Halton `(2, 3)` sub-pixel offset from it (index `phase + 1`,
//! recentered to `[-0.5, 0.5)`). Both are closed-form, loop-bounded numeric
//! kernels with no transcendental calls, so they port to the device directly.
//!
//! [`GpuTaauJitterOffset`] is the on-device twin of that pair: one thread
//! resolves one query, reproducing the reference's integer digit loop and the
//! `2`-D Halton offset. A passing real-device parity test is direct evidence the
//! ported kernel computes the same low-discrepancy offsets the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query the kernel reproduces:
//! - `rinv_x = radical_inverse(base_x, index)` and
//!   `rinv_y = radical_inverse(base_y, index)`, the raw radical inverses in
//!   `[0, 1)`;
//! - `offset_x = radical_inverse(base_x, phase + 1) - 0.5` and
//!   `offset_y = radical_inverse(base_y, phase + 1) - 0.5`, the recentered
//!   Halton offset
//!   [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase)
//!   returns, with the `index = phase + 1` skip of the un-jittered center.
//!
//! The `phase + 1` uses a wrapping add exactly like the reference, so a `phase`
//! of `u32::MAX` folds to `index 0` and yields the origin radical inverse.
//!
//! # What stays on the host
//!
//! The frame-to-phase mapping
//! ([`phase_for_frame`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::phase_for_frame)
//! and
//! [`offset_for_frame`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_frame))
//! takes a `64`-bit absolute frame number, which the core-`WGSL` subset cannot
//! represent, so it stays on the host; the host reduces a frame to a `u32`
//! `phase` before enqueuing. An empty batch short-circuits on the host, since a
//! storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The radical inverse threads a `u32` digit through an integer modulo/divide
//! loop and accumulates `f32` products, so the `CPU` and `GPU` are not
//! bit-exact: a `GPU` reciprocal and fused multiply-add may land a few units in
//! the last place from the scalar reference. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous
//! output, tight enough to catch a genuinely wrong port (a dropped `- 0.5`, a
//! swapped base, a wrong digit order) yet loose enough to admit a legal
//! last-place difference. The digit loop itself is exact (integer modulo and
//! divide), so for every fixture the two implementations walk the identical
//! sequence of digits.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `u32` modulo/divide,
//! `f32` `+`, `*`, and a single reciprocal `/` — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt`, no `round`, and no
//! `64`-bit integers. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. The digit loop is bounded: each
//! iteration divides the `u32` index by a base of at least `2`, so it drains to
//! zero in at most `32` steps and the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::temporal_upscale::jitter`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` jitter radical-inverse and Halton-offset kernel,
/// embedded inline so the twin ships as a single source file. The single entry
/// point `solve` mirrors the `CPU` golden
/// [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
/// and
/// [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase);
/// see the module documentation for the algorithm.
const TAAU_JITTER_OFFSET_WGSL: &str = r#"
// Deterministic jitter twin: one thread computes the base-b radical inverse of
// an index and the recentered Halton (2,3) offset of a phase, mirroring the CPU
// golden `temporal_upscale::jitter` with only u32 modulo/divide and f32 +,*,/.
// It owns no 64-bit frame mapping; that stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::temporal_upscale::jitter；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Halton base for the horizontal axis (>= 2 for a meaningful expansion).
    base_x: u32,
    // Halton base for the vertical axis (>= 2 for a meaningful expansion).
    base_y: u32,
    // Index whose raw radical inverse is reported directly.
    index: u32,
    // Phase whose recentered Halton offset is reported (index = phase + 1).
    phase: u32,
}

struct Result {
    // radical_inverse(base_x, index) in [0, 1).
    rinv_x: f32,
    // radical_inverse(base_y, index) in [0, 1).
    rinv_y: f32,
    // Recentered Halton offset x: radical_inverse(base_x, phase + 1) - 0.5.
    offset_x: f32,
    // Recentered Halton offset y: radical_inverse(base_y, phase + 1) - 0.5.
    offset_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Reproduces `radical_inverse`: a base below 2 has no positional expansion and
// returns 0; otherwise the base-b digits of the index are read
// most-significant-first after reflection, weighted by successive reciprocal
// powers. Pure integer modulo/divide plus an f32 accumulate; no transcendental.
fn radical_inverse(base: u32, index_in: u32) -> f32 {
    if (base < 2u) {
        return 0.0;
    }
    let inv_base = 1.0 / f32(base);
    var inv_weight = inv_base;
    var result = 0.0;
    var index = index_in;
    loop {
        if (index > 0u) {
            let digit = index % base;
            result = result + f32(digit) * inv_weight;
            index = index / base;
            inv_weight = inv_weight * inv_base;
        } else {
            break;
        }
    }
    return result;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // The Halton index skips the un-jittered center with phase + 1; a wrapping
    // add folds phase u32::MAX to index 0, matching the reference wrapping_add.
    let phase_index = q.phase + 1u;

    var out: Result;
    out.rinv_x = radical_inverse(q.base_x, q.index);
    out.rinv_y = radical_inverse(q.base_y, q.index);
    out.offset_x = radical_inverse(q.base_x, phase_index) - 0.5;
    out.offset_y = radical_inverse(q.base_y, phase_index) - 0.5;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TAAU_JITTER_OFFSET_WGSL`].
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

/// `repr(C)` `std430` layout of one query: two Halton bases, a raw `index`, and
/// a `phase`, a `16`-byte stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Halton base for the horizontal axis.
    base_x: u32,
    /// Halton base for the vertical axis.
    base_y: u32,
    /// Index whose raw radical inverse is reported.
    index: u32,
    /// Phase whose recentered Halton offset is reported.
    phase: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// two raw radical inverses and the two recentered Halton offsets, a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `radical_inverse(base_x, index)`.
    rinv_x: f32,
    /// `radical_inverse(base_y, index)`.
    rinv_y: f32,
    /// Recentered Halton offset `x`.
    offset_x: f32,
    /// Recentered Halton offset `y`.
    offset_y: f32,
}

/// One jitter query for the twin: the two Halton bases, a raw `index` for the
/// direct radical inverse, and a `phase` for the recentered Halton offset.
///
/// The host owns the `64`-bit frame-to-phase mapping and enqueues one
/// [`TaauJitterOffsetQuery`] per query, mirroring the inputs of the reference
/// [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
/// and
/// [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaauJitterOffsetQuery {
    /// Halton base for the horizontal axis.
    pub base_x: u32,
    /// Halton base for the vertical axis.
    pub base_y: u32,
    /// Index whose raw radical inverse is reported in `rinv_x`/`rinv_y`.
    pub index: u32,
    /// Phase whose recentered Halton offset is reported in `offset_x`/`offset_y`.
    pub phase: u32,
}

impl TaauJitterOffsetQuery {
    /// Builds a query with explicit bases, raw `index`, and `phase`.
    #[must_use]
    pub const fn new(base_x: u32, base_y: u32, index: u32, phase: u32) -> TaauJitterOffsetQuery {
        TaauJitterOffsetQuery {
            base_x,
            base_y,
            index,
            phase,
        }
    }
}

/// One resolved jitter query, mirroring the reference
/// [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
/// and
/// [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase).
///
/// `rinv_x`/`rinv_y` are the raw radical inverses of `index` in `[0, 1)`, and
/// `offset_x`/`offset_y` are the Halton offset for `phase` recentered to
/// `[-0.5, 0.5)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaauJitterOffsetResult {
    /// `radical_inverse(base_x, index)`.
    pub rinv_x: f32,
    /// `radical_inverse(base_y, index)`.
    pub rinv_y: f32,
    /// Recentered Halton offset `x` for `phase`.
    pub offset_x: f32,
    /// Recentered Halton offset `y` for `phase`.
    pub offset_y: f32,
}

/// Encodes one [`TaauJitterOffsetQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TaauJitterOffsetQuery) -> GpuQuery {
    GpuQuery {
        base_x: q.base_x,
        base_y: q.base_y,
        index: q.index,
        phase: q.phase,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TaauJitterOffsetResult`].
fn decode_result(raw: &GpuResult) -> TaauJitterOffsetResult {
    TaauJitterOffsetResult {
        rinv_x: raw.rinv_x,
        rinv_y: raw.rinv_y,
        offset_x: raw.offset_x,
        offset_y: raw.offset_y,
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

/// A compiled, reusable jitter radical-inverse and Halton-offset compute
/// pipeline, twinning the `CPU` golden
/// [`radical_inverse`](prism_render_architecture::temporal_upscale::jitter::radical_inverse)
/// and
/// [`offset_for_phase`](prism_render_architecture::temporal_upscale::jitter::JitterSequence::offset_for_phase).
pub struct GpuTaauJitterOffset {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTaauJitterOffset {
    /// Compiles the jitter kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTaauJitterOffset {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset"),
            source: ShaderSource::Wgsl(TAAU_JITTER_OFFSET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTaauJitterOffset {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`TaauJitterOffsetResult`] per input, in order.
    ///
    /// Each output matches the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TaauJitterOffsetQuery],
    ) -> Vec<TaauJitterOffsetResult> {
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
            label: Some("prism_volumetric_taau_jitter_offset_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_bind_group"),
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
            label: Some("prism_volumetric_taau_jitter_offset_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_taau_jitter_offset_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_taau_jitter_offset_pass"),
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
