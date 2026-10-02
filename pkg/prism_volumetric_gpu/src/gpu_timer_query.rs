//! `wgpu` compute twin of the `GPU` timestamp-query-pool *slot layout* contract
//! ([`gpu_timer_query`](prism_render_architecture::particle::gpu_timer_query),
//! particle design §28).
//!
//! The `CPU` golden
//! [`TimestampQueryPool`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool)
//! owns the physical layout of a `GPU` timestamp query set: how many query
//! slots the pool holds, how many complete begin/end pairs (measured spans) fit,
//! which slot holds the begin and end timestamp of a given pair, and how many
//! bytes the resolve buffer needs. Every one of those answers is pure, bounded
//! integer arithmetic — division, multiplication, a range test and an
//! `Option` — so the layout is the bit-exact lock-step currency between the
//! reference path and the render graph that binds the real `wgpu` `QuerySet`.
//! [`GpuTimerQueryLayout`] is the on-device twin: one thread evaluates one
//! layout query, routed through an operation selector so one dispatch can mix
//! every slot-layout question in a single batch.
//!
//! A passing real-device parity test is therefore direct evidence the ported
//! kernel reproduces the exact slot index arithmetic and the exact
//! out-of-range rejection the reference computes, not merely that its shader
//! compiles.
//!
//! # What is twinned
//!
//! The twin mirrors the five pure-integer slot-layout functions line for line:
//! [`TimestampQueryPool::query_count`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::query_count),
//! [`TimestampQueryPool::pair_capacity`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::pair_capacity),
//! [`TimestampQueryPool::begin_query_index`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::begin_query_index),
//! [`TimestampQueryPool::end_query_index`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::end_query_index)
//! and
//! [`TimestampQueryPool::resolve_buffer_bytes`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::resolve_buffer_bytes).
//! The two indexing functions return an
//! [`Option`](core::option::Option) on the host, so the kernel emits a `(value,
//! valid)` pair and the host maps `valid == 1` back to `Some(value)` and
//! `valid == 0` back to `None`, the same `Option`-to-`u32`-flag round trip the
//! sibling ray/`AABB` twin uses.
//!
//! # What is not twinned
//!
//! Everything that is not pure, non-overflowing `u32` arithmetic is excluded on
//! purpose, because `WGSL` has only `i32`, `u32`, `f32` and `bool` and this
//! crate forbids `f32` equality:
//!
//! - the elapsed-time decoders
//!   [`TimestampPeriod::elapsed_nanos`](prism_render_architecture::particle::gpu_timer_query::TimestampPeriod::elapsed_nanos)
//!   and
//!   [`TimestampPeriod::elapsed_millis`](prism_render_architecture::particle::gpu_timer_query::TimestampPeriod::elapsed_millis)
//!   multiply an `f32` nanoseconds-per-`tick` by a `u64` `tick` interval, which
//!   needs both an `f32` tolerance comparison and a `64`-bit integer the shader
//!   lacks;
//! - the statistics reducer
//!   [`TimerStats::from_samples`](prism_render_architecture::particle::gpu_timer_query::TimerStats::from_samples)
//!   is an `f32` min/max/mean fold, again an `f32` comparison;
//! - [`TimestampQueryPool::all_pairs`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::all_pairs)
//!   materializes a variable-length [`Vec`](alloc::vec::Vec), which has no fixed
//!   on-device result stride; its per-pair arithmetic is already covered lane
//!   for lane by [`GpuTimerQueryOp::BeginQueryIndex`] and
//!   [`GpuTimerQueryOp::EndQueryIndex`].
//!
//! # Restricted subset for the resolve-buffer size
//!
//! The golden
//! [`TimestampQueryPool::resolve_buffer_bytes`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::resolve_buffer_bytes)
//! returns a [`usize`] equal to
//! [`TIMESTAMP_BYTES`](prism_render_architecture::particle::gpu_timer_query::TIMESTAMP_BYTES)
//! times the capacity, a product that can exceed `u32::MAX` for a large
//! capacity. The kernel computes it in `u32`, so the twin is only faithful for
//! the restricted subset where the byte product fits a `u32`, that is capacity
//! at most [`MAX_RESOLVE_BYTES_CAPACITY`]. The host rejects any larger capacity
//! for that one operation with a `valid == 0` flag rather than reporting a
//! wrapped size; the fixtures keep every capacity well inside the subset so a
//! real timestamp pool (dozens to thousands of slots) is always covered.
//!
//! # Correctness model
//!
//! Every twinned operation is exact, non-overflowing integer arithmetic: the
//! pair stride is [`QUERIES_PER_PAIR`](prism_render_architecture::particle::gpu_timer_query::QUERIES_PER_PAIR),
//! and because a valid `pair` is strictly below `capacity / 2` the products
//! `pair * 2` and `pair * 2 + 1` never exceed the capacity, so no `u32` wrap can
//! occur. The parity test therefore asserts a strict `==` on both the value word
//! and the validity flag with no tolerance: any mismatch is a genuine port
//! defect.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer `*`, integer
//! `/`, a `u32` comparison and a `switch` on the operation code — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, no inverse trigonometry and no
//! `64`-bit integer, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! There is no loop: each thread performs a fixed, bounded sequence of integer
//! operations, so the kernel provably terminates.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized. A zero capacity holds no pairs, so every
//! indexing query reports `valid == 0`; an odd capacity drops its trailing slot
//! exactly as the reference does.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::gpu_timer_query` 的纯 `u32` 槽位布局
//! 子集；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::gpu_timer_query::{
    TimestampQueryPool, QUERIES_PER_PAIR, TIMESTAMP_BYTES,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

/// Discrete validity code written by the kernel for a layout query that yields a
/// value: matches the host `== 1` decode in [`decode_result`]. An out-of-range
/// indexing query (or an out-of-subset resolve-size query) writes `0` instead.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
const CODE_VALID: u32 = 1;

/// The largest pool capacity whose resolve-buffer byte product
/// ([`TIMESTAMP_BYTES`](prism_render_architecture::particle::gpu_timer_query::TIMESTAMP_BYTES)
/// times capacity) still fits a `u32`. For a larger capacity the golden
/// [`usize`] product would overflow a `u32`, so the twin reports
/// [`GpuTimerQueryOp::ResolveBufferBytes`] as invalid rather than wrapping.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
pub const MAX_RESOLVE_BYTES_CAPACITY: u32 = u32::MAX / TIMESTAMP_BYTES as u32;

/// The portable core-`WGSL` timestamp-query-pool slot-layout kernel, embedded
/// inline so the twin ships as a single source file. Mirrors the pure-`u32`
/// golden slot arithmetic.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
const GPU_TIMER_QUERY_WGSL: &str = r#"
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Query {
    capacity: u32,
    pair: u32,
    op: u32,
    pad0: u32,
};

struct Result {
    value: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const QUERIES_PER_PAIR: u32 = 2u;
const TIMESTAMP_BYTES: u32 = 8u;
const MAX_RESOLVE_CAPACITY: u32 = 536870911u;

// pair_capacity: integer division by the pair stride drops any odd trailing
// slot, matching the reference `capacity / QUERIES_PER_PAIR`.
fn pair_capacity(capacity: u32) -> u32 {
    return capacity / QUERIES_PER_PAIR;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    var value: u32 = 0u;
    var valid: u32 = 0u;

    switch (q.op) {
        // query_count: the total slot count is the capacity itself.
        case 0u: {
            value = q.capacity;
            valid = 1u;
        }
        // pair_capacity: complete begin/end pairs that fit.
        case 1u: {
            value = pair_capacity(q.capacity);
            valid = 1u;
        }
        // begin_query_index: Some(pair * 2) while pair is in range, else None.
        case 2u: {
            if (q.pair < pair_capacity(q.capacity)) {
                value = q.pair * QUERIES_PER_PAIR;
                valid = 1u;
            }
        }
        // end_query_index: Some(pair * 2 + 1) while pair is in range, else None.
        case 3u: {
            if (q.pair < pair_capacity(q.capacity)) {
                value = q.pair * QUERIES_PER_PAIR + 1u;
                valid = 1u;
            }
        }
        // resolve_buffer_bytes: TIMESTAMP_BYTES per slot, valid only within the
        // restricted subset where the u32 product cannot overflow.
        case 4u: {
            if (q.capacity <= MAX_RESOLVE_CAPACITY) {
                value = TIMESTAMP_BYTES * q.capacity;
                valid = 1u;
            }
        }
        default: {
            value = 0u;
            valid = 0u;
        }
    }

    var res: Result;
    res.value = value;
    res.valid = valid;
    res.pad0 = 0u;
    res.pad1 = 0u;
    results[idx] = res;
}
"#;

/// The slot-layout question an individual layout query selects.
///
/// Each variant names one of the five pure-`u32` golden slot-layout functions
/// mirrored by this twin; the stable [`GpuTimerQueryOp::to_code`] mapping is the
/// operation code the kernel's `switch` dispatches on. [`GpuTimerQueryOp::QueryCount`],
/// [`GpuTimerQueryOp::PairCapacity`] and [`GpuTimerQueryOp::ResolveBufferBytes`]
/// read only [`GpuTimerQueryLayoutQuery::capacity`]; [`GpuTimerQueryOp::BeginQueryIndex`]
/// and [`GpuTimerQueryOp::EndQueryIndex`] additionally read
/// [`GpuTimerQueryLayoutQuery::pair`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuTimerQueryOp {
    /// Total slot count (`query_count`).
    QueryCount,
    /// Complete begin/end pair count (`pair_capacity`).
    PairCapacity,
    /// Begin-timestamp slot index of a pair (`begin_query_index`).
    BeginQueryIndex,
    /// End-timestamp slot index of a pair (`end_query_index`).
    EndQueryIndex,
    /// Resolve-buffer byte size (`resolve_buffer_bytes`).
    ResolveBufferBytes,
}

impl GpuTimerQueryOp {
    /// The stable `u32` code the kernel `switch` dispatches on.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
    /// 无第三方引擎源码或衍生代码。
    #[must_use]
    const fn to_code(self) -> u32 {
        match self {
            GpuTimerQueryOp::QueryCount => 0,
            GpuTimerQueryOp::PairCapacity => 1,
            GpuTimerQueryOp::BeginQueryIndex => 2,
            GpuTimerQueryOp::EndQueryIndex => 3,
            GpuTimerQueryOp::ResolveBufferBytes => 4,
        }
    }
}

/// One timestamp-query-pool slot-layout query: the pool capacity, the pair
/// index and the slot-layout question to evaluate.
///
/// `capacity` is the number of individual query slots in the pool, matching the
/// argument to
/// [`TimestampQueryPool::new`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::new).
/// `pair` is the begin/end pair index, read only by
/// [`GpuTimerQueryOp::BeginQueryIndex`] and [`GpuTimerQueryOp::EndQueryIndex`];
/// the other operations ignore it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuTimerQueryLayoutQuery {
    /// Number of individual query slots in the pool.
    pub capacity: u32,
    /// Begin/end pair index (used only by the indexing operations).
    pub pair: u32,
    /// Which golden slot-layout function to evaluate.
    pub op: GpuTimerQueryOp,
}

/// The resolved slot-layout answer for one query, the host-side mirror of the
/// kernel's `Result` lane.
///
/// `value` carries the computed slot index, count or byte size; `present`
/// decodes the kernel validity flag, so it is `false` exactly when the golden
/// returned [`None`](core::option::Option::None) (an out-of-range indexing
/// query) or when a resolve-size query fell outside the restricted `u32` subset.
/// For the always-defined operations ([`GpuTimerQueryOp::QueryCount`],
/// [`GpuTimerQueryOp::PairCapacity`]) `present` is always `true`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuTimerQueryLayoutResult {
    /// The computed value; meaningful only when `present`.
    pub value: u32,
    /// Whether the value is defined (`valid == 1` on the device).
    pub present: bool,
}

impl GpuTimerQueryLayoutResult {
    /// Views the result as an [`Option`](core::option::Option): `Some(value)`
    /// when `present`, otherwise `None`. Mirrors the golden `Option<u32>` return
    /// of the two indexing functions.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
    /// 无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn as_option(self) -> Option<u32> {
        if self.present {
            Some(self.value)
        } else {
            None
        }
    }
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_TIMER_QUERY_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `16`-byte `std430` stride matching `Query` in the
/// shader: the capacity, the pair index, the operation code and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Pool capacity in slots.
    capacity: u32,
    /// Begin/end pair index.
    pair: u32,
    /// Operation code from [`GpuTimerQueryOp::to_code`].
    op: u32,
    /// Padding word.
    pad0: u32,
}

impl GpuQuery {
    /// Packs a [`GpuTimerQueryLayoutQuery`] into the `std430` upload layout.
    fn from_query(query: &GpuTimerQueryLayoutQuery) -> GpuQuery {
        GpuQuery {
            capacity: query.capacity,
            pair: query.pair,
            op: query.op.to_code(),
            pad0: 0,
        }
    }
}

/// One result as read back. `16`-byte `std430` stride matching `Result` in the
/// shader: the value word, the validity flag and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The computed value.
    value: u32,
    /// Validity flag (`1` = defined).
    valid: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Maps one kernel `Result` lane back to the host [`GpuTimerQueryLayoutResult`].
fn decode_result(raw: &GpuResult) -> GpuTimerQueryLayoutResult {
    GpuTimerQueryLayoutResult {
        value: raw.value,
        present: raw.valid == CODE_VALID,
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

/// A compiled, reusable timestamp-query-pool slot-layout pipeline.
pub struct GpuTimerQueryLayout {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTimerQueryLayout {
    /// Compiles the slot-layout kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
    /// 无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTimerQueryLayout {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_timer_query"),
            source: ShaderSource::Wgsl(GPU_TIMER_QUERY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTimerQueryLayout {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one
    /// [`GpuTimerQueryLayoutResult`] per query in input order.
    ///
    /// The returned result for query `q` mirrors the golden slot-layout function
    /// `q.op` names, evaluated on
    /// [`TimestampQueryPool::new`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool::new)`(q.capacity)`.
    /// An empty `queries` slice yields an empty vector — storage buffers cannot
    /// be zero-sized, so it is handled by an early return before any dispatch.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
    /// 无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn run(
        &self,
        ctx: &GpuContext,
        queries: &[GpuTimerQueryLayoutQuery],
    ) -> Vec<GpuTimerQueryLayoutResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_timer_query_bind_group"),
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
            label: Some("prism_volumetric_gpu_timer_query_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_timer_query_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}

/// The `CPU` golden slot-layout answer for one query, dispatching to the
/// reference
/// [`TimestampQueryPool`](prism_render_architecture::particle::gpu_timer_query::TimestampQueryPool)
/// so callers (and the parity test) can pin the twin lane for lane.
///
/// Returns `(value, present)`: the always-defined operations
/// ([`GpuTimerQueryOp::QueryCount`], [`GpuTimerQueryOp::PairCapacity`]) report
/// `present == true`; the indexing operations decode their `Option`; the
/// resolve-size operation reports `present == false` when the capacity leaves
/// the restricted `u32` subset ([`MAX_RESOLVE_BYTES_CAPACITY`]).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_timer_query`；
/// 无第三方引擎源码或衍生代码。
#[must_use]
pub fn cpu_reference(query: &GpuTimerQueryLayoutQuery) -> (u32, bool) {
    let pool = TimestampQueryPool::new(query.capacity);
    match query.op {
        GpuTimerQueryOp::QueryCount => (pool.query_count(), true),
        GpuTimerQueryOp::PairCapacity => (pool.pair_capacity(), true),
        GpuTimerQueryOp::BeginQueryIndex => match pool.begin_query_index(query.pair) {
            Some(value) => (value, true),
            None => (0, false),
        },
        GpuTimerQueryOp::EndQueryIndex => match pool.end_query_index(query.pair) {
            Some(value) => (value, true),
            None => (0, false),
        },
        GpuTimerQueryOp::ResolveBufferBytes => {
            if query.capacity <= MAX_RESOLVE_BYTES_CAPACITY {
                let bytes = pool.resolve_buffer_bytes();
                (u32::try_from(bytes).unwrap_or(0), true)
            } else {
                (0, false)
            }
        }
    }
}

/// Compile-time cross-checks that the inlined `WGSL` constants mirror the golden
/// values; a drift here would silently desynchronize the twin from the
/// reference.
const _: () = {
    assert!(QUERIES_PER_PAIR == 2);
    assert!(TIMESTAMP_BYTES == 8);
    assert!(MAX_RESOLVE_BYTES_CAPACITY == 536_870_911);
};
