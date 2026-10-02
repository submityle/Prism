//! `wgpu` compute twin of the device-free cross-pass bind-group sizing
//! primitives
//! ([`pipeline_layout`](prism_render_architecture::particle::pipeline_layout),
//! particle design §9).
//!
//! The `CPU` golden
//! [`pipeline_layout`](prism_render_architecture::particle::pipeline_layout)
//! maps every one of the ten ordered
//! [`FramePass`](prism_render_architecture::particle::frame_pipeline::FramePass)
//! steps to the `@group(0)` buffer contract it dispatches over, derives each
//! pass's sizing extent from a single whole-frame
//! [`ParticleFrameExtent`](prism_render_architecture::particle::pipeline_layout::ParticleFrameExtent),
//! and reports two closed-form integer facts per pass: the `@group(0)` binding
//! count
//! ([`pass_binding_count`](prism_render_architecture::particle::pipeline_layout::pass_binding_count))
//! and the clamped total declared byte size
//! ([`pass_declared_bytes`](prism_render_architecture::particle::pipeline_layout::pass_declared_bytes)).
//! [`GpuPipelineLayout`] is the on-device twin: one thread resolves one
//! [`GpuPipelineLayoutQuery`] (one [`FramePass`](prism_render_architecture::particle::frame_pipeline::FramePass)
//! classification code plus the whole-frame extent) into one
//! [`GpuPipelineLayoutResult`], so a passing real-device parity test is direct
//! evidence the ported kernel reproduces the same binding counts and the same
//! clamped byte sizes the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `pass_code` (the pass's zero-based pipeline
//! `order_index`, `0`..`9`), the twelve whole-frame extent scalars, and two
//! probe strides used for the stride-consistency flag. The kernel reproduces:
//!
//! - `pass_binding_count` as the fixed per-pass `@group(0)` binding tally.
//! - `pass_declared_bytes` by projecting the whole-frame extent into the pass's
//!   sub-extent (the golden `emitter_extent` / `spawn_extent` / `sim_extent` /
//!   `event_extent` / `sort_cull_extent` / `draw_extent` field extraction) and
//!   summing `stride * max(count, 1u)` over every buffer the pass binds — the
//!   same clamp-to-one-element rule a non-empty `WebGPU` storage binding needs.
//! - the handoff stride-consistency predicate
//!   ([`Handoff::is_consistent`](prism_render_architecture::particle::pipeline_layout::Handoff::is_consistent))
//!   as the discrete `1`/`0` flag `producer_stride == consumer_stride` yields.
//!
//! The whole-frame roll-ups
//! ([`frame_binding_count`](prism_render_architecture::particle::pipeline_layout::frame_binding_count)
//! and
//! [`frame_upper_bound_bytes`](prism_render_architecture::particle::pipeline_layout::frame_upper_bound_bytes))
//! are the saturating sum of the ten per-pass device answers; the convenience
//! methods [`GpuPipelineLayout::frame_binding_count`] and
//! [`GpuPipelineLayout::frame_upper_bound_bytes`] dispatch the ten-pass batch
//! and fold the device results. The host-side description and validation of the
//! cross-pass handoffs (`describe_handoffs` and `validate_handoffs`) are not
//! twinned: they build and walk a `Vec` of host records rather than a
//! per-element `GPU` computation.
//!
//! # Correctness model
//!
//! Every value is a `u32` count, a `u32` byte size or a discrete `1`/`0` flag:
//! the byte size is pure unsigned multiply-add-and-max with no rounding, the
//! binding count is a fixed classification and the consistency flag is an
//! unsigned equality. `CPU` and `GPU` therefore compute identical bit patterns,
//! and the parity test asserts an exact `==` on every field with no tolerance.
//! Fixtures keep every element count, every per-buffer product and every
//! per-pass sum well below `2^31`, so the device `u32` arithmetic never wraps
//! where the golden `saturating_mul` / `saturating_add` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*`, `+`, the
//! `max` built-in, an unsigned compare, `select` and a `switch`. There is no
//! `sqrt`, no divide, no transcendental call, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of integer work, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`；无第三方引擎源码或衍生代码。
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

/// The ten `FramePass` pipeline `order_index` codes, in pipeline order, that the
/// frame roll-up helpers dispatch over.
///
/// The codes mirror
/// [`FramePass::order_index`](prism_render_architecture::particle::frame_pipeline::FramePass::order_index):
/// `EmitterUpdate` is `0` and `RenderDraw` is `9`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
const FRAME_PASS_CODES: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// The cross-pass bind-group sizing kernel, mirroring the `CPU` golden
/// [`pipeline_layout`](prism_render_architecture::particle::pipeline_layout)
/// pass for pass. The single entry point `solve` resolves one query per thread,
/// embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
const PIPELINE_LAYOUT_WGSL: &str = r#"
// pipeline_layout twin: one thread per query reproduces the CPU golden
// `particle::pipeline_layout`. For a FramePass code (0..9) and a whole-frame
// extent it reports the pass's `@group(0)` binding count, the clamped total
// declared byte size, and a stride-consistency flag. Byte sizing is
// `stride * max(count, 1u)` summed over every buffer the pass binds — the
// clamp-to-one-element rule a non-empty WebGPU storage binding needs. Pure u32
// arithmetic: multiplies, adds, `max`, an unsigned compare and a switch. There
// is no sqrt, no divide, no transcendental call and no u64, so the kernel runs
// unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel
// provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::pipeline_layout；无第三方
// 引擎源码或衍生代码。

// std430 strides, mirroring the golden per-pass byte-layout modules.
const U32_STRIDE: u32 = 4u;
const VEC2_STRIDE: u32 = 8u;
const VEC4_STRIDE: u32 = 16u;
const EMITTER_PARAMS_STRIDE: u32 = 64u;    // 4 * VEC4
const EMITTER_STATE_STRIDE: u32 = 32u;     // 2 * VEC4
const SPAWN_PARAMS_STRIDE: u32 = 32u;      // 2 * VEC4
const SIM_UNIFORM_SIZE: u32 = 64u;
const CONSTRAINT_STRIDE: u32 = 32u;
const EVENT_RECORD_STRIDE: u32 = 32u;      // VEC4 + 4 * U32
const CULL_PARAMS_STRIDE: u32 = 128u;      // (6 + 1 + 1) * VEC4
const INSTANCE_STRIDE: u32 = 64u;          // 3 * VEC4 + 2 * VEC2
const DRAW_INDEXED_INDIRECT_STRIDE: u32 = 20u; // 4 * U32 + U32

// Fixed element-count constants, independent of the extent.
const SPAWN_REQUEST_COUNTER_COUNT: u32 = 1u;
const SPAWN_COUNTER_COUNT: u32 = 4u;

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 60-byte std430 stride of fifteen scalar words, matching the host
// `GpuPipelineLayoutQuery`: the pass code, two probe strides and the twelve
// whole-frame extent scalars in `ParticleFrameExtent` declaration order.
struct LayoutQuery {
    pass_code: u32,
    a_stride: u32,
    b_stride: u32,
    emitter_count: u32,
    particle_capacity: u32,
    spawn_count: u32,
    alive_count: u32,
    visible_count: u32,
    grid_cell_count: u32,
    constraint_count: u32,
    event_source_capacity: u32,
    event_channel_count: u32,
    event_scattered_capacity: u32,
    radix_buckets: u32,
    workgroup_count: u32,
}

// One result. 12-byte std430 stride of three scalar words, matching the host
// `GpuPipelineLayoutResult`: the binding count, the clamped total byte size and
// the consistency flag.
struct LayoutResult {
    binding_count: u32,
    declared_bytes: u32,
    consistent: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<LayoutQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<LayoutResult>;

// Clamp-to-one storage byte size: `stride * max(count, 1u)`, mirroring the
// golden `storage_bytes`.
fn sb(stride: u32, count: u32) -> u32 {
    return stride * max(count, 1u);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var binding_count: u32 = 0u;
    var declared_bytes: u32 = 0u;

    switch q.pass_code {
        // Step 1 — EmitterUpdate (emitter_extent).
        case 0u: {
            binding_count = 4u;
            declared_bytes =
                sb(EMITTER_PARAMS_STRIDE, q.emitter_count)
                + sb(EMITTER_STATE_STRIDE, q.emitter_count)
                + sb(VEC4_STRIDE, q.spawn_count)
                + sb(U32_STRIDE, SPAWN_REQUEST_COUNTER_COUNT);
        }
        // Step 2 — Spawn (spawn_extent).
        case 1u: {
            binding_count = 6u;
            declared_bytes =
                sb(U32_STRIDE, q.particle_capacity)
                + sb(U32_STRIDE, SPAWN_COUNTER_COUNT)
                + sb(U32_STRIDE, q.spawn_count)
                + sb(SPAWN_PARAMS_STRIDE, q.emitter_count)
                + sb(VEC4_STRIDE, q.particle_capacity)
                + sb(VEC4_STRIDE, q.particle_capacity);
        }
        // Step 3 — SimulationStages (sim_extent).
        case 2u: {
            binding_count = 8u;
            declared_bytes =
                sb(VEC4_STRIDE, q.particle_capacity)
                + sb(VEC4_STRIDE, q.particle_capacity)
                + sb(VEC4_STRIDE, q.particle_capacity)
                + sb(SIM_UNIFORM_SIZE, 1u)
                + sb(U32_STRIDE, q.grid_cell_count)
                + sb(VEC2_STRIDE, q.particle_capacity)
                + sb(CONSTRAINT_STRIDE, q.constraint_count)
                + sb(U32_STRIDE, q.constraint_count);
        }
        // Step 4 — EventScatter (event_extent).
        case 3u: {
            binding_count = 4u;
            declared_bytes =
                sb(U32_STRIDE, q.event_channel_count)
                + sb(EVENT_RECORD_STRIDE, q.event_source_capacity)
                + sb(EVENT_RECORD_STRIDE, q.event_scattered_capacity)
                + sb(U32_STRIDE, q.event_channel_count);
        }
        // Step 5 — Compaction (sort_cull_extent, particle_count = alive_count).
        case 4u: {
            binding_count = 3u;
            declared_bytes =
                sb(U32_STRIDE, q.alive_count)
                + sb(U32_STRIDE, q.alive_count)
                + sb(U32_STRIDE, q.alive_count);
        }
        // Step 6 — Bounds (sort_cull_extent).
        case 5u: {
            binding_count = 2u;
            let eff_wg = max(q.workgroup_count, 1u);
            declared_bytes =
                sb(VEC4_STRIDE, q.alive_count)
                + sb(VEC4_STRIDE, eff_wg * 2u);
        }
        // Step 7 — Cull (sort_cull_extent, candidate_count = alive_count).
        case 6u: {
            binding_count = 4u;
            declared_bytes =
                sb(U32_STRIDE, q.alive_count)
                + sb(CULL_PARAMS_STRIDE, 1u)
                + sb(U32_STRIDE, q.alive_count)
                + sb(VEC2_STRIDE, 1u);
        }
        // Step 8 — Sort (sort_cull_extent).
        case 7u: {
            binding_count = 4u;
            let hist = max(q.radix_buckets, 1u) * max(q.workgroup_count, 1u);
            declared_bytes =
                sb(U32_STRIDE, q.alive_count)
                + sb(U32_STRIDE, q.alive_count)
                + sb(U32_STRIDE, hist)
                + sb(U32_STRIDE, hist);
        }
        // Step 9 — FillDrawArgs (draw_extent, SortedIndices sized by capacity).
        case 8u: {
            binding_count = 3u;
            declared_bytes =
                sb(U32_STRIDE, 1u)
                + sb(U32_STRIDE, q.particle_capacity)
                + sb(DRAW_INDEXED_INDIRECT_STRIDE, 1u);
        }
        // Step 10 — RenderDraw (draw_extent).
        case 9u: {
            binding_count = 3u;
            declared_bytes =
                sb(INSTANCE_STRIDE, q.particle_capacity)
                + sb(U32_STRIDE, q.particle_capacity)
                + sb(DRAW_INDEXED_INDIRECT_STRIDE, 1u);
        }
        default: {
            binding_count = 0u;
            declared_bytes = 0u;
        }
    }

    var out: LayoutResult;
    out.binding_count = binding_count;
    out.declared_bytes = declared_bytes;
    // is_consistent: producer_stride == consumer_stride, mirroring the golden
    // `Handoff::is_consistent`.
    out.consistent = select(0u, 1u, q.a_stride == q.b_stride);

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`PIPELINE_LAYOUT_WGSL`]: the valid query count plus three pad
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

/// One query for the cross-pass sizing twin: a
/// [`FramePass`](prism_render_architecture::particle::frame_pipeline::FramePass)
/// `pass_code`, two probe strides for the consistency flag and the twelve
/// whole-frame extent scalars.
///
/// The `pass_code` is the pass's zero-based pipeline `order_index`
/// ([`FramePass::order_index`](prism_render_architecture::particle::frame_pipeline::FramePass::order_index)):
/// `0` for `EmitterUpdate` through `9` for `RenderDraw`. The twelve extent
/// fields mirror
/// [`ParticleFrameExtent`](prism_render_architecture::particle::pipeline_layout::ParticleFrameExtent)
/// in declaration order; `a_stride` and `b_stride` feed the handoff
/// stride-consistency flag. The `repr(C)` layout — fifteen `u32` words, `60`
/// bytes with no padding — matches the `WGSL` `LayoutQuery` struct exactly, so
/// it is uploaded to the device without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuPipelineLayoutQuery {
    /// The pass's zero-based pipeline `order_index`, `0`..`9`.
    pub pass_code: u32,
    /// First probe stride for the stride-consistency flag.
    pub a_stride: u32,
    /// Second probe stride for the stride-consistency flag.
    pub b_stride: u32,
    /// Number of emitters advanced this frame.
    pub emitter_count: u32,
    /// Pooled particle capacity (the per-particle pool length).
    pub particle_capacity: u32,
    /// Particles spawned this frame (the spawn-request domain).
    pub spawn_count: u32,
    /// Live particles entering compaction, bounds, cull and sort.
    pub alive_count: u32,
    /// Particles surviving cull that become the draw instance count.
    pub visible_count: u32,
    /// Spatial-hash grid cell count (the cell-offset table length).
    pub grid_cell_count: u32,
    /// `XPBD` constraint count in the batch.
    pub constraint_count: u32,
    /// Capacity of the raw source-event pool.
    pub event_source_capacity: u32,
    /// Number of event append channels.
    pub event_channel_count: u32,
    /// Capacity of the compacted scattered-event pool.
    pub event_scattered_capacity: u32,
    /// `radix` histogram bucket count for the sort.
    pub radix_buckets: u32,
    /// Number of workgroups a reduction / sort pass dispatches.
    pub workgroup_count: u32,
}

/// One resolved answer for a single [`GpuPipelineLayoutQuery`], mirroring the
/// golden `pass_binding_count`, `pass_declared_bytes` and `Handoff::is_consistent`
/// outputs.
///
/// The `repr(C)` layout — three `u32` words, `12` bytes — matches the `WGSL`
/// `LayoutResult` struct exactly, so device results are read back without a
/// separate decode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuPipelineLayoutResult {
    /// The pass's `@group(0)` binding count, matching the golden
    /// [`pass_binding_count`](prism_render_architecture::particle::pipeline_layout::pass_binding_count).
    pub binding_count: u32,
    /// The pass's clamped total declared byte size, matching the golden
    /// [`pass_declared_bytes`](prism_render_architecture::particle::pipeline_layout::pass_declared_bytes).
    pub declared_bytes: u32,
    /// Stride-consistency flag, `1` when the two probe strides are equal and
    /// `0` otherwise, matching the golden
    /// [`Handoff::is_consistent`](prism_render_architecture::particle::pipeline_layout::Handoff::is_consistent).
    pub consistent: u32,
}

impl GpuPipelineLayoutQuery {
    /// Builds the ten-pass whole-frame query batch for one extent, in pipeline
    /// order.
    ///
    /// The `extent` array holds the twelve
    /// [`ParticleFrameExtent`](prism_render_architecture::particle::pipeline_layout::ParticleFrameExtent)
    /// scalars in declaration order (`emitter_count`, `particle_capacity`,
    /// `spawn_count`, `alive_count`, `visible_count`, `grid_cell_count`,
    /// `constraint_count`, `event_source_capacity`, `event_channel_count`,
    /// `event_scattered_capacity`, `radix_buckets`, `workgroup_count`). The
    /// probe strides are set equal so the consistency flag stays `1` and never
    /// affects the frame roll-ups.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
    #[must_use]
    pub fn frame_batch(extent: [u32; 12]) -> [GpuPipelineLayoutQuery; 10] {
        let mut batch = [GpuPipelineLayoutQuery {
            pass_code: 0,
            a_stride: 0,
            b_stride: 0,
            emitter_count: extent[0],
            particle_capacity: extent[1],
            spawn_count: extent[2],
            alive_count: extent[3],
            visible_count: extent[4],
            grid_cell_count: extent[5],
            constraint_count: extent[6],
            event_source_capacity: extent[7],
            event_channel_count: extent[8],
            event_scattered_capacity: extent[9],
            radix_buckets: extent[10],
            workgroup_count: extent[11],
        }; 10];
        let mut i = 0;
        while i < FRAME_PASS_CODES.len() {
            batch[i].pass_code = FRAME_PASS_CODES[i];
            i += 1;
        }
        batch
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

/// A compiled, reusable cross-pass sizing compute pipeline, twinning the `CPU`
/// golden
/// [`pipeline_layout`](prism_render_architecture::particle::pipeline_layout).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
pub struct GpuPipelineLayout {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPipelineLayout {
    /// Compiles the cross-pass sizing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPipelineLayout {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_pipeline_layout"),
            source: ShaderSource::Wgsl(PIPELINE_LAYOUT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_pipeline_layout_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_pipeline_layout_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_pipeline_layout_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPipelineLayout {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuPipelineLayoutResult`] per input, in order.
    ///
    /// Each result equals the matching golden triple exactly — `binding_count`
    /// mirrors
    /// [`pass_binding_count`](prism_render_architecture::particle::pipeline_layout::pass_binding_count),
    /// `declared_bytes` mirrors
    /// [`pass_declared_bytes`](prism_render_architecture::particle::pipeline_layout::pass_declared_bytes)
    /// and `consistent` mirrors
    /// [`Handoff::is_consistent`](prism_render_architecture::particle::pipeline_layout::Handoff::is_consistent)
    /// — because the whole path is integer bit algebra. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuPipelineLayoutQuery],
    ) -> Vec<GpuPipelineLayoutResult> {
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
            label: Some("prism_volumetric_pipeline_layout_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_pipeline_layout_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuPipelineLayoutResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_pipeline_layout_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_pipeline_layout_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_pipeline_layout_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_pipeline_layout_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_pipeline_layout_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuPipelineLayoutResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }

    /// Dispatches the ten-pass batch for `extent` and returns the saturating sum
    /// of every pass's device `binding_count`, twinning the golden
    /// [`frame_binding_count`](prism_render_architecture::particle::pipeline_layout::frame_binding_count).
    ///
    /// The extent does not affect binding counts, but it is threaded through so
    /// the device path is identical to [`GpuPipelineLayout::frame_upper_bound_bytes`].
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
    #[must_use]
    pub fn frame_binding_count(&self, ctx: &GpuContext, extent: [u32; 12]) -> u32 {
        let batch = GpuPipelineLayoutQuery::frame_batch(extent);
        self.evaluate(ctx, &batch)
            .into_iter()
            .map(|r| r.binding_count)
            .fold(0u32, u32::saturating_add)
    }

    /// Dispatches the ten-pass batch for `extent` and returns the saturating sum
    /// of every pass's device `declared_bytes`, twinning the golden
    /// [`frame_upper_bound_bytes`](prism_render_architecture::particle::pipeline_layout::frame_upper_bound_bytes).
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::pipeline_layout`。
    #[must_use]
    pub fn frame_upper_bound_bytes(&self, ctx: &GpuContext, extent: [u32; 12]) -> u32 {
        let batch = GpuPipelineLayoutQuery::frame_batch(extent);
        self.evaluate(ctx, &batch)
            .into_iter()
            .map(|r| r.declared_bytes)
            .fold(0u32, u32::saturating_add)
    }
}
