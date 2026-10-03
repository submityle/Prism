//! `wgpu` compute twin of the per-frame deformation scheduling arbiter
//! ([`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations),
//! the `CPU` decision layer that picks which deformation jobs run this frame
//! within a fixed `GPU` budget).
//!
//! The `CPU` golden
//! [`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations)
//! is a deterministic, fully serial priority-greedy arbiter: it orders the
//! requests by descending priority then ascending handle, admits the single
//! highest-priority request unconditionally (forward progress), then admits
//! each remaining request only while the vertex budget allows, deferring the
//! rest. A `BLAS` refit slot is granted to an admitted job that wants one while
//! refit slots remain. Every decision is integer and boolean, so the twin
//! reproduces it exactly rather than within a tolerance.
//!
//! [`GpuDeformSchedulePlan`] is the on-device twin of that serial algorithm.
//! This is *not* a data-parallel kernel: one thread owns one whole scheduling
//! problem (one [`DeformSchedulePlanQuery`]) and replays the arbiter serially —
//! a bounded stable insertion sort of the request indices by `(priority desc,
//! handle asc)` followed by the greedy admit/defer scan. A passing real-device
//! parity run is therefore direct evidence the ported kernel makes the same
//! admit, defer and refit decisions the reference does.
//!
//! # What is twinned
//!
//! The full serial arbiter for a bounded request set: the stable priority /
//! handle ordering, the unconditional admission of the top request, the
//! saturating vertex-budget charge and defer decision, and the separate refit
//! slot accounting. The device echoes, per original request slot, whether it
//! was admitted, whether its refit was granted, and its dispatch-order index
//! within the schedule, plus the aggregate `vertices_used`, `refits_used`,
//! `scheduled_count` and `deferred_count`.
//!
//! # What stays on the host
//!
//! The variable-length `Vec` outputs of the reference
//! ([`DeformationPlan::scheduled`](prism_render_architecture::deformation::schedule::DeformationPlan)
//! and its `deferred` list, the per-kind filters) stay host-side; the device
//! reports fixed-width per-slot decisions the host folds back into those lists.
//! The request count is capped at [`MAX_REQUESTS`]; the host zero-pads shorter
//! batches into the fixed slot, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every quantity is an integer or a boolean — the sort keys, the saturating
//! vertex sum, the budget comparisons, the refit slot counter and the schedule
//! order index — so the `CPU` and `GPU` agree bit for bit and the parity test
//! asserts an exact `==` on every field. Handles are unique within a query, so
//! `(priority desc, handle asc)` is a total order and the stable insertion sort
//! reproduces the reference's stable sort deterministically.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned and signed
//! integer arithmetic, comparisons and a wrap-detected saturating add — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, no floating-point at all, no
//! inverse trigonometry and no `round`. No optional device feature is required,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Each thread performs
//! a fixed, bounded sort and scan over at most [`MAX_REQUESTS`] entries, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::deformation::schedule`；无第三方引擎源码或衍生代码。
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

/// Maximum number of deformation requests the device twin schedules in one
/// query. The host zero-pads shorter batches into this fixed-width slot.
pub const MAX_REQUESTS: usize = 32;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels; here one thread owns
/// one whole scheduling problem.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` serial deformation-scheduling kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations)
/// serial arbiter; see the module documentation for the algorithm.
const DEFORM_SCHEDULE_PLAN_WGSL: &str = r#"
// Serial deformation-scheduling twin: one thread replays the whole arbiter for
// one query. It stable-insertion-sorts the request indices by (priority desc,
// handle asc), then greedily admits within the vertex budget while granting
// refit slots, mirroring the CPU golden
// `deformation::schedule::plan_deformations`. It owns no variable-length Vec;
// the host folds the fixed-width per-slot decisions back into lists.
//
// Provenance: 孪生自本仓 prism_render_architecture::deformation::schedule；无第三方
// 引擎源码或衍生代码。

const MAX_REQUESTS: u32 = 32u;
const U32_MAX: u32 = 4294967295u;

struct Params {
    // Number of scheduling problems (queries) in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Req {
    // Deformation cache handle; unique within a query, used as the tie-break.
    handle: u32,
    // Owning subsystem discriminant; passed through, not used for scheduling.
    kind: u32,
    // Vertices charged against the vertex budget.
    vertex_count: u32,
    // Higher runs first; ties break by ascending handle.
    priority: u32,
    // 1 when the deformed geometry wants a BLAS refit this frame.
    needs_blas_refit: u32,
}

struct Query {
    // Number of valid requests in this query (<= MAX_REQUESTS).
    count: u32,
    // Vertex budget per frame.
    vertices_per_frame: u32,
    // BLAS refit slots per frame.
    blas_refits_per_frame: u32,
    pad0: u32,
    reqs: array<Req, 32>,
}

struct Slot {
    // 1 when this request was admitted this frame.
    admitted: u32,
    // 1 when a refit slot was granted to this admitted request.
    refit_granted: u32,
    // Dispatch-order index within the schedule, or -1 when deferred.
    schedule_order: i32,
    pad0: u32,
}

struct Result {
    // Total vertices charged this frame.
    vertices_used: u32,
    // BLAS refit slots consumed this frame.
    refits_used: u32,
    // Number of admitted jobs.
    scheduled_count: u32,
    // Number of deferred jobs.
    deferred_count: u32,
    slots: array<Slot, 32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let qi = gid.x;
    if (qi >= params.count) {
        return;
    }
    let n = queries[qi].count;
    let vpf = queries[qi].vertices_per_frame;
    let brpf = queries[qi].blas_refits_per_frame;

    // Identity permutation of the valid request indices.
    var idxs: array<u32, 32>;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        idxs[i] = i;
    }

    // Stable insertion sort by (priority desc, handle asc). An element shifts
    // only when it is strictly ordered before its predecessor, so equal keys
    // keep their original order, matching the reference's stable sort.
    for (var i: u32 = 1u; i < n; i = i + 1u) {
        let key = idxs[i];
        let kp = queries[qi].reqs[key].priority;
        let kh = queries[qi].reqs[key].handle;
        var j: i32 = i32(i) - 1;
        loop {
            if (j < 0) {
                break;
            }
            let cur = idxs[u32(j)];
            let cp = queries[qi].reqs[cur].priority;
            let ch = queries[qi].reqs[cur].handle;
            var key_before: bool = false;
            if (kp > cp) {
                key_before = true;
            } else if (kp == cp) {
                if (kh < ch) {
                    key_before = true;
                }
            }
            if (!key_before) {
                break;
            }
            idxs[u32(j + 1)] = cur;
            j = j - 1;
        }
        idxs[u32(j + 1)] = key;
    }

    // Reset every slot to a deterministic deferred state before the scan.
    for (var i: u32 = 0u; i < MAX_REQUESTS; i = i + 1u) {
        results[qi].slots[i].admitted = 0u;
        results[qi].slots[i].refit_granted = 0u;
        results[qi].slots[i].schedule_order = -1;
        results[qi].slots[i].pad0 = 0u;
    }

    var vertices_used: u32 = 0u;
    var refits_used: u32 = 0u;
    var sched_count: u32 = 0u;
    var def_count: u32 = 0u;

    for (var s: u32 = 0u; s < n; s = s + 1u) {
        let ri = idxs[s];
        let first = (sched_count == 0u);
        let vc = queries[qi].reqs[ri].vertex_count;
        // Saturating add: a wrap (sum < addend base) pins to u32::MAX, matching
        // the reference's `saturating_add`.
        var next_v: u32 = vertices_used + vc;
        if (next_v < vertices_used) {
            next_v = U32_MAX;
        }
        // Every request after the first must fit the vertex budget; the first
        // (highest-priority) request is admitted unconditionally.
        if (!first && next_v > vpf) {
            def_count = def_count + 1u;
            continue;
        }
        vertices_used = next_v;
        var granted: u32 = 0u;
        if (queries[qi].reqs[ri].needs_blas_refit != 0u) {
            if (refits_used < brpf) {
                granted = 1u;
                refits_used = refits_used + 1u;
            }
        }
        results[qi].slots[ri].admitted = 1u;
        results[qi].slots[ri].refit_granted = granted;
        results[qi].slots[ri].schedule_order = i32(sched_count);
        sched_count = sched_count + 1u;
    }

    results[qi].vertices_used = vertices_used;
    results[qi].refits_used = refits_used;
    results[qi].scheduled_count = sched_count;
    results[qi].deferred_count = def_count;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`DEFORM_SCHEDULE_PLAN_WGSL`].
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

/// `repr(C)` `std430` layout of one deformation request, matching the `WGSL`
/// `Req` struct: five `u32` words at a `20`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuReq {
    /// Deformation cache handle (unique within a query).
    handle: u32,
    /// Owning subsystem discriminant, passed through.
    kind: u32,
    /// Vertices charged against the budget.
    vertex_count: u32,
    /// Scheduling priority (higher first).
    priority: u32,
    /// `1` when the request wants a `BLAS` refit.
    needs_blas_refit: u32,
}

/// `repr(C)` `std430` layout of one scheduling query: the count, the two budget
/// words, a pad word, then the fixed-width request array, matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Number of valid requests in this query.
    count: u32,
    /// Vertex budget per frame.
    vertices_per_frame: u32,
    /// `BLAS` refit slots per frame.
    blas_refits_per_frame: u32,
    /// Padding word.
    pad0: u32,
    /// The requests, zero-padded to [`MAX_REQUESTS`].
    reqs: [GpuReq; MAX_REQUESTS],
}

/// `repr(C)` `std430` layout of one per-request decision, matching the `WGSL`
/// `Slot` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSlot {
    /// `1` when admitted this frame.
    admitted: u32,
    /// `1` when a refit slot was granted.
    refit_granted: u32,
    /// Dispatch-order index within the schedule, or `-1` when deferred.
    schedule_order: i32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one scheduling result, matching the `WGSL`
/// `Result` struct: four aggregate counters then the per-request decisions.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Total vertices charged this frame.
    vertices_used: u32,
    /// `BLAS` refit slots consumed this frame.
    refits_used: u32,
    /// Number of admitted jobs.
    scheduled_count: u32,
    /// Number of deferred jobs.
    deferred_count: u32,
    /// Per-request decisions, indexed by original request slot.
    slots: [GpuSlot; MAX_REQUESTS],
}

/// One deformation request mirroring the reference
/// [`DeformationRequest`](prism_render_architecture::deformation::schedule::DeformationRequest):
/// a handle, a subsystem `kind` discriminant, a vertex count, a priority, and a
/// refit flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeformSchedulePlanRequest {
    /// Deformation cache handle; unique within a query.
    pub handle: u32,
    /// Owning subsystem discriminant (the reference `DeformationKind` as `u32`).
    pub kind: u32,
    /// Vertices this job will process, charged against the vertex budget.
    pub vertex_count: u32,
    /// Scheduling priority; higher runs first, ties break by ascending handle.
    pub priority: u32,
    /// `true` when the deformed geometry needs a `BLAS` refit.
    pub needs_blas_refit: bool,
}

/// One scheduling query: the deformation requests plus the per-frame budget.
///
/// The host enqueues one [`DeformSchedulePlanQuery`] per frame's worth of
/// requests, mirroring the inputs to the reference
/// [`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeformSchedulePlanQuery {
    /// The requests to schedule; at most [`MAX_REQUESTS`].
    pub requests: Vec<DeformSchedulePlanRequest>,
    /// Vertex budget per frame.
    pub vertices_per_frame: u32,
    /// `BLAS` refit slots per frame.
    pub blas_refits_per_frame: u32,
}

/// One per-request scheduling decision, indexed by the original request slot.
///
/// Mirrors the reference
/// [`ScheduledDeformation`](prism_render_architecture::deformation::schedule::ScheduledDeformation)
/// decision folded back onto the request that produced it: whether it was
/// admitted, whether its refit was granted, and its dispatch-order index within
/// the schedule (`-1` when deferred).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeformSchedulePlanSlot {
    /// `true` when this request was admitted this frame.
    pub admitted: bool,
    /// `true` when a `BLAS` refit slot was granted to this admitted request.
    pub refit_granted: bool,
    /// Dispatch-order index within the schedule, or `-1` when deferred.
    pub schedule_order: i32,
}

/// One resolved scheduling plan, mirroring the reference
/// [`DeformationPlan`](prism_render_architecture::deformation::schedule::DeformationPlan).
///
/// `slots` has one entry per input request, in the original request order; the
/// four counters mirror the reference plan's aggregate totals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeformSchedulePlanResult {
    /// Per-request decisions, indexed by original request slot.
    pub slots: Vec<DeformSchedulePlanSlot>,
    /// Total vertices charged against the budget this frame.
    pub vertices_used: u32,
    /// `BLAS` refit slots consumed this frame.
    pub refits_used: u32,
    /// Number of admitted jobs.
    pub scheduled_count: u32,
    /// Number of deferred jobs.
    pub deferred_count: u32,
}

/// Encodes one [`DeformSchedulePlanQuery`] into its `std430` [`GpuQuery`] slot,
/// zero-padding the request array to [`MAX_REQUESTS`].
fn encode_query(q: &DeformSchedulePlanQuery) -> GpuQuery {
    let mut reqs = [GpuReq {
        handle: 0,
        kind: 0,
        vertex_count: 0,
        priority: 0,
        needs_blas_refit: 0,
    }; MAX_REQUESTS];
    for (slot, r) in reqs.iter_mut().zip(q.requests.iter()) {
        *slot = GpuReq {
            handle: r.handle,
            kind: r.kind,
            vertex_count: r.vertex_count,
            priority: r.priority,
            needs_blas_refit: u32::from(r.needs_blas_refit),
        };
    }
    GpuQuery {
        count: q.requests.len() as u32,
        vertices_per_frame: q.vertices_per_frame,
        blas_refits_per_frame: q.blas_refits_per_frame,
        pad0: 0,
        reqs,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`DeformSchedulePlanResult`],
/// trimming the fixed slot array to the `count` meaningful requests.
fn decode_result(raw: &GpuResult, count: usize) -> DeformSchedulePlanResult {
    let slots = raw.slots[..count]
        .iter()
        .map(|s| DeformSchedulePlanSlot {
            admitted: s.admitted != 0,
            refit_granted: s.refit_granted != 0,
            schedule_order: s.schedule_order,
        })
        .collect();
    DeformSchedulePlanResult {
        slots,
        vertices_used: raw.vertices_used,
        refits_used: raw.refits_used,
        scheduled_count: raw.scheduled_count,
        deferred_count: raw.deferred_count,
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

/// A compiled, reusable serial deformation-scheduling compute pipeline, twinning
/// the `CPU` golden
/// [`plan_deformations`](prism_render_architecture::deformation::schedule::plan_deformations).
pub struct GpuDeformSchedulePlan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDeformSchedulePlan {
    /// Compiles the scheduling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDeformSchedulePlan {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan"),
            source: ShaderSource::Wgsl(DEFORM_SCHEDULE_PLAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDeformSchedulePlan {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every scheduling problem in `queries` and returns one
    /// [`DeformSchedulePlanResult`] per input, in order.
    ///
    /// Every field equals the reference exactly (all decisions are integer or
    /// boolean). An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[DeformSchedulePlanQuery],
    ) -> Vec<DeformSchedulePlanResult> {
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
            label: Some("prism_volumetric_deform_schedule_plan_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_bind_group"),
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
            label: Some("prism_volumetric_deform_schedule_plan_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_deform_schedule_plan_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_deform_schedule_plan_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per scheduling problem, flattened to a 1-D dispatch.
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

        raw.iter()
            .zip(queries.iter())
            .map(|(r, q)| decode_result(r, q.requests.len()))
            .collect()
    }
}
