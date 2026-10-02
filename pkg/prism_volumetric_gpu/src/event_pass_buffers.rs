//! `wgpu` compute twin of the device-free `std430` event-scatter bind-group
//! contract
//! ([`event_pass_buffers`](prism_render_architecture::particle::event_pass_buffers),
//! particle design §9: `Event Scatter`).
//!
//! The `CPU` golden
//! [`event_pass_buffers`](prism_render_architecture::particle::event_pass_buffers)
//! owns one storage-buffer enum
//! ([`EventScatterBuffer`](prism_render_architecture::particle::event_pass_buffers::EventScatterBuffer),
//! four variants) and reports, for each buffer, its `@binding` index, element
//! stride, access mode, writability flag, element count and clamped total byte
//! size against a
//! [`ParticleEventExtent`](prism_render_architecture::particle::event_pass_buffers::ParticleEventExtent).
//! [`GpuEventPassBuffers`] is the on-device twin: one thread resolves one
//! [`GpuEventPassBufferQuery`] — a `variant_code` plus an extent — into one
//! [`GpuEventPassBufferResult`] carrying all six answers, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! same `ABI` the reference publishes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `variant_code` (the index into
//! [`EventScatterBuffer::ALL`](prism_render_architecture::particle::event_pass_buffers::EventScatterBuffer),
//! which is also the `@binding` index) and the three extent counts
//! `source_event_capacity`, `channel_count` and `scattered_capacity`. The
//! kernel reproduces the golden `match` tables for `binding`, `stride`,
//! `access` (as the `0`/`1` code
//! [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
//! uses), `is_output`, `element_count` and `byte_size` (the clamp-to-one
//! `stride * max(count, 1)` rule). The counter and offset tables are
//! channel-sized, the raw record pool spans `source_event_capacity`, and the
//! compacted output spans `scattered_capacity`; no buffer is a fixed
//! single-element block, so an empty extent drives every `element_count` to
//! `0` while `byte_size` still clamps up to one record.
//!
//! # Correctness model
//!
//! Every value is a `u32`, a classification code or a `bool`-flavoured flag:
//! the byte size is pure unsigned multiply-and-max with no rounding, and the
//! binding, stride, access code and output flag are discrete classifications.
//! `CPU` and `GPU` therefore compute identical bit patterns, and the parity
//! test asserts an exact `==` on every field with no tolerance. Fixtures keep
//! `stride * capacity` well below `2^31`, so the device `u32` multiply never
//! wraps where the golden `saturating_mul` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*`, the
//! `max` built-in, unsigned comparisons and index arithmetic. There is no
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
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`；无第三方引擎源码或衍生代码。
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

/// The event-scatter bind-group kernel, mirroring the `CPU` golden
/// [`event_pass_buffers`](prism_render_architecture::particle::event_pass_buffers)
/// field for field. The single entry point `solve` resolves one query per
/// thread, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
const EVENT_PASS_BUFFERS_WGSL: &str = r#"
// event_pass_buffers twin: one thread per query reproduces the CPU golden
// `particle::event_pass_buffers`. A query is a `variant_code` (the index into
// `EventScatterBuffer::ALL`, which is also the @binding index) plus the three
// extent counts; the kernel reproduces the golden `match` tables for binding,
// stride, access, is_output, element_count and byte_size. `byte_size` is
// `stride * max(element_count, 1u)` — the clamp-to-one-element rule a non-empty
// WebGPU storage binding needs — and `is_output` is the discrete 1u/0u flag the
// golden `access().is_writable()` yields (1u for the ReadWrite variants,
// access_code 1u). Pure u32 arithmetic: comparisons, one multiply and one
// `max`. There is no sqrt, no divide, no transcendental call and no u64, so the
// kernel runs unmodified on Metal, Vulkan and DX12. There is no loop, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::event_pass_buffers；无第三方
// 引擎源码或衍生代码。

// Variant codes in EventScatterBuffer::ALL / @binding order.
const SLOT_EVENT_COUNTERS: u32 = 0u;
const SLOT_SOURCE_EVENTS: u32 = 1u;
const SLOT_SCATTERED_EVENTS: u32 = 2u;
const SLOT_CHANNEL_OFFSETS: u32 = 3u;

// std430 strides mirrored from the golden `gpu_layout` and
// `event_pass_buffers`: a scalar u32 for the atomic counters and the prefix-sum
// offset table, and the shared 32-byte packed event record for the source and
// scattered pools.
const U32_STRIDE: u32 = 4u;
const EVENT_RECORD_STRIDE: u32 = 32u;

// Access codes in the golden enum's declaration order: Read then ReadWrite.
const ACCESS_READ: u32 = 0u;
const ACCESS_READ_WRITE: u32 = 1u;

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 16-byte std430 stride of four scalar words, matching the host
// `GpuEventPassBufferQuery`: the variant code and the three `ParticleEventExtent`
// field counts.
struct EventQuery {
    variant_code: u32,
    source_event_capacity: u32,
    channel_count: u32,
    scattered_capacity: u32,
}

// One result. 24-byte std430 stride of six scalar words, matching the host
// `GpuEventPassBufferResult`: binding, stride, access code, output flag,
// element count and clamped total byte size.
struct EventResult {
    binding: u32,
    stride: u32,
    access_code: u32,
    is_output: u32,
    element_count: u32,
    byte_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<EventQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<EventResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let slot = q.variant_code;

    // stride: the atomic counters (slot 0) and the prefix-sum offset table
    // (slot 3) are 4-byte scalars; the source and scattered pools (slots 1, 2)
    // share the 32-byte packed event record.
    var stride: u32 = U32_STRIDE;
    if (slot == SLOT_SOURCE_EVENTS || slot == SLOT_SCATTERED_EVENTS) {
        stride = EVENT_RECORD_STRIDE;
    }

    // access_code: the append counters (slot 0) and the compacted output
    // (slot 2) are mutated in place; the raw source records (slot 1) and the
    // channel offsets (slot 3) are read-only inputs.
    var access_code: u32 = ACCESS_READ;
    if (slot == SLOT_EVENT_COUNTERS || slot == SLOT_SCATTERED_EVENTS) {
        access_code = ACCESS_READ_WRITE;
    }
    // is_output: the pass writes iff the access is ReadWrite.
    let is_output = access_code;

    // element_count: the counter and offset tables (slots 0, 3) have one entry
    // per channel; the source pool (slot 1) spans source_event_capacity and the
    // scattered pool (slot 2) spans scattered_capacity. Nothing is a fixed
    // single-element block, so an empty extent drives this to 0.
    var element_count: u32 = q.channel_count;
    if (slot == SLOT_SOURCE_EVENTS) {
        element_count = q.source_event_capacity;
    } else if (slot == SLOT_SCATTERED_EVENTS) {
        element_count = q.scattered_capacity;
    }

    var out: EventResult;
    // binding is the index into EventScatterBuffer::ALL, i.e. the variant code.
    out.binding = slot;
    out.stride = stride;
    out.access_code = access_code;
    out.is_output = is_output;
    out.element_count = element_count;

    // byte_size: a non-empty storage binding reserves at least one element, so
    // the count is clamped up to one before the multiply.
    out.byte_size = stride * max(element_count, 1u);

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`EVENT_PASS_BUFFERS_WGSL`]: the valid query count plus three
/// pad words — `16` bytes with no interior padding.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
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

/// One query for the event-scatter bind-group twin: a `variant_code` and the
/// three `ParticleEventExtent` field counts.
///
/// `variant_code` is the index into
/// [`EventScatterBuffer::ALL`](prism_render_architecture::particle::event_pass_buffers::EventScatterBuffer::ALL)
/// in binding order `0..4` (which equals the `@binding` index). The three
/// counts mirror the fields of
/// [`ParticleEventExtent`](prism_render_architecture::particle::event_pass_buffers::ParticleEventExtent):
/// `source_event_capacity`, `channel_count` and `scattered_capacity`. The
/// `repr(C)` layout — four `u32` words, `16` bytes with no padding — matches
/// the `WGSL` `EventQuery` struct exactly, so it is uploaded to the device
/// without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuEventPassBufferQuery {
    /// Index into `EventScatterBuffer::ALL`; also the `@binding` index in `0..4`.
    pub variant_code: u32,
    /// Capacity of the raw `SourceEvents` record pool
    /// (`ParticleEventExtent::source_event_capacity`).
    pub source_event_capacity: u32,
    /// Number of append channels
    /// (`ParticleEventExtent::channel_count`).
    pub channel_count: u32,
    /// Capacity of the compacted `ScatteredEvents` output pool
    /// (`ParticleEventExtent::scattered_capacity`).
    pub scattered_capacity: u32,
}

/// One resolved answer for a single [`GpuEventPassBufferQuery`], mirroring the
/// golden `binding`, `stride`, `access`, `is_output`, `element_count` and
/// `byte_size` outputs.
///
/// The `repr(C)` layout — six `u32` words, `24` bytes — matches the `WGSL`
/// `EventResult` struct exactly, so device results are read back without a
/// separate decode step. `access_code` encodes a
/// [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
/// variant (`0` for `Read`, `1` for `ReadWrite`) and `is_output` is the
/// discrete `1`/`0` writability flag the golden `access().is_writable()`
/// yields.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuEventPassBufferResult {
    /// The `@group(0)` binding index of the buffer.
    pub binding: u32,
    /// Byte stride of one element, matching the `WESL` scalar / record layout.
    pub stride: u32,
    /// Access-mode code: `0` for `Read`, `1` for `ReadWrite`.
    pub access_code: u32,
    /// Output flag, `1` when the pass writes this buffer and `0` otherwise.
    pub is_output: u32,
    /// Element count for the pool: channel-sized for the counter and offset
    /// tables, and capacity-sized for the source and scattered record pools.
    pub element_count: u32,
    /// Total byte size `stride * max(element_count, 1)`, matching the golden
    /// clamp-to-one `byte_size`.
    pub byte_size: u32,
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

/// A compiled, reusable event-scatter bind-group compute pipeline, twinning the
/// `CPU` golden
/// [`event_pass_buffers`](prism_render_architecture::particle::event_pass_buffers).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
pub struct GpuEventPassBuffers {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEventPassBuffers {
    /// Compiles the event-scatter bind-group kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEventPassBuffers {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_module"),
            source: ShaderSource::Wgsl(EVENT_PASS_BUFFERS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEventPassBuffers {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuEventPassBufferResult`] per input, in order.
    ///
    /// Each result equals the matching golden tuple exactly — `binding`,
    /// `stride`, `access`, `is_output`, `element_count` and `byte_size` all
    /// mirror the `CPU`
    /// [`event_pass_buffers`](prism_render_architecture::particle::event_pass_buffers)
    /// reference — because the whole path is integer bit algebra. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::event_pass_buffers`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuEventPassBufferQuery],
    ) -> Vec<GpuEventPassBufferResult> {
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
            label: Some("prism_volumetric_event_pass_buffers_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuEventPassBufferResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_event_pass_buffers_bind_group"),
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
            label: Some("prism_volumetric_event_pass_buffers_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_event_pass_buffers_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuEventPassBufferResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
