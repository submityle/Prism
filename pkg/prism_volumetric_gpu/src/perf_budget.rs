//! `wgpu` compute twin of the frame-time budget numeric contract
//! ([`perf_budget`](prism_render_architecture::particle::perf_budget), particle
//! design §32 *Performance budget*).
//!
//! The `CPU` golden
//! [`perf_budget`](prism_render_architecture::particle::perf_budget) answers,
//! among other things, how a measured per-stage frame-time
//! [`FrameTimeSample`](prism_render_architecture::particle::perf_budget::FrameTimeSample)
//! compares to a per-stage
//! [`FrameTimeBudget`](prism_render_architecture::particle::perf_budget::FrameTimeBudget):
//! the overspend of each of the
//! [`FrameStage::COUNT`](prism_render_architecture::particle::perf_budget::FrameStage::COUNT)
//! stages, the measured and overspent totals, whether the frame stayed within
//! its total target, whether any single stage ran over, the overspend as a
//! guarded fraction of the target, and a microsecond-to-millisecond conversion.
//! Every one of those pieces is a pure, deterministic, transcendental-free
//! function of its inputs — only `+ - * /`, `max`, and a single guarded divide —
//! so the same inputs always produce the same verdict and the `CPU` reference
//! and this `GPU` twin agree. [`GpuPerfBudget`] is the on-device twin: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel evaluates the same closed form the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query numeric primitive is reproduced through a tagged
//! [`PerfBudgetQuery`]: one variant per reference routine, dispatched by an
//! integer `op` code the kernel branches on. The twinned routines are the
//! per-stage/total overspend evaluation
//! ([`FrameTimeBudget::evaluate`](prism_render_architecture::particle::perf_budget::FrameTimeBudget::evaluate)),
//! the per-stage target sum
//! ([`FrameTimeBudget::stage_sum_ms`](prism_render_architecture::particle::perf_budget::FrameTimeBudget::stage_sum_ms)),
//! the per-stage measured sum
//! ([`FrameTimeSample::total_ms`](prism_render_architecture::particle::perf_budget::FrameTimeSample::total_ms)),
//! the within-total predicate
//! ([`FrameTimeReport::is_within_total`](prism_render_architecture::particle::perf_budget::FrameTimeReport::is_within_total)),
//! the any-stage-over predicate
//! ([`FrameTimeReport::any_stage_over`](prism_render_architecture::particle::perf_budget::FrameTimeReport::any_stage_over)),
//! the guarded overspend ratio
//! ([`FrameTimeReport::overspend_ratio`](prism_render_architecture::particle::perf_budget::FrameTimeReport::overspend_ratio)),
//! and the microsecond-to-millisecond conversion
//! ([`micros_to_ms`](prism_render_architecture::particle::perf_budget::micros_to_ms)).
//!
//! # Correctness model
//!
//! Each op evaluates the same arithmetic the reference does in the same order,
//! so `CPU` and `GPU` walk the identical branch and evaluate the identical
//! closed form. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore allows `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every continuous lane; the
//! boolean verdicts are carried as exact `u32` flags and compared exactly.
//!
//! # Degenerate inputs
//!
//! Every guard matches the reference: an overspend is clamped at `0.0` rather
//! than reporting a negative "underspend", a (near) zero total target yields a
//! `0.0` ratio rather than dividing by zero, and the within-total and
//! any-stage-over predicates use the same `BUDGET_EPS` tolerance so a value
//! exactly on a ceiling counts as within budget. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max` and
//! `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no inverse
//! trigonometry, no builtin `smoothstep`, no `round`, and no `cbrt`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Every stage fold is a fixed
//! six-iteration walk, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! # Honest host boundary
//!
//! The `VRAM` estimators
//! ([`estimate_vram_bytes`](prism_render_architecture::particle::perf_budget::estimate_vram_bytes)
//! and
//! [`estimate_attribute_vram_bytes`](prism_render_architecture::particle::perf_budget::estimate_attribute_vram_bytes))
//! are deliberately *not* twinned: they accumulate a saturating `u64` byte
//! count, and the portable `WGSL` subset has no `64`-bit integer. The stateful
//! budget ledger and tracker
//! ([`BudgetTracker`](prism_render_architecture::particle::perf_budget::BudgetTracker)
//! and its `charge_*`/`reset`/`report` methods), the pressure arbitration
//! ([`arbitrate`](prism_render_architecture::particle::perf_budget::arbitrate)),
//! and the acceptance-threshold aggregation
//! ([`aggregate_acceptance`](prism_render_architecture::particle::perf_budget::aggregate_acceptance))
//! are host-side mutable state and classification rather than per-element kernel
//! math, so they stay on the host as well.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
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
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
const WORKGROUP_SIZE: u32 = 64;

// Number of accountable frame stages, mirroring `FrameStage::COUNT`; used as the
// fixed array length in both the host layout and the device struct. Private, so
// no provenance line is required.
const STAGE_COUNT: usize = 6;

const PERF_BUDGET_WGSL: &str = r#"
// Frame-time budget twin: one thread per query reproduces one reference numeric
// routine selected by `op`. It mirrors the CPU golden
// prism_render_architecture::particle::perf_budget term for term, uses only the
// portable core-WGSL subset (max and + - * /), needs no transcendental call, and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
// Every stage fold is a fixed six-iteration walk, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::perf_budget；
// 无第三方引擎源码或衍生代码。

// Millisecond/error comparison guard, matching BUDGET_EPS.
const BUDGET_EPS: f32 = 1.0e-6;
// Number of accountable frame stages, matching FrameStage::COUNT.
const STAGE_COUNT: u32 = 6u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Per-stage measured milliseconds (evaluate / total_ms ops).
    stage_ms: array<f32, 6>,
    // Per-stage target milliseconds (evaluate / stage_sum ops).
    stage_targets_ms: array<f32, 6>,
    // Per-stage overspend milliseconds (any_stage_over op).
    stage_overspend_ms: array<f32, 6>,
    // Scalar inputs; a field a given op does not name is ignored.
    total_target_ms: f32,
    total_overspend_ms: f32,
    micros: u32,
    // Op classification code (0..=6).
    op: u32,
}

struct Result {
    // Per-stage overspend milliseconds (evaluate op).
    stage_overspend_ms: array<f32, 6>,
    // Measured and overspent totals (evaluate op).
    total_measured_ms: f32,
    total_overspend_ms: f32,
    // Shared scalar lane (stage_sum / total_ms / overspend_ratio / micros_to_ms).
    scalar: f32,
    // Shared boolean lane (is_within_total / any_stage_over) as a 0/1 flag.
    flag: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.stage_overspend_ms = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    out.total_measured_ms = 0.0;
    out.total_overspend_ms = 0.0;
    out.scalar = 0.0;
    out.flag = 0u;

    if (q.op == 0u) {
        // evaluate: per-stage overspend plus measured and overspent totals.
        var total_measured: f32 = 0.0;
        for (var i: u32 = 0u; i < STAGE_COUNT; i = i + 1u) {
            let measured = q.stage_ms[i];
            let goal = q.stage_targets_ms[i];
            out.stage_overspend_ms[i] = max(measured - goal, 0.0);
            total_measured = total_measured + measured;
        }
        out.total_measured_ms = total_measured;
        out.total_overspend_ms = max(total_measured - q.total_target_ms, 0.0);
    } else if (q.op == 1u) {
        // stage_sum_ms: sum of the per-stage targets.
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < STAGE_COUNT; i = i + 1u) {
            sum = sum + q.stage_targets_ms[i];
        }
        out.scalar = sum;
    } else if (q.op == 2u) {
        // total_ms: sum of the per-stage measured times.
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < STAGE_COUNT; i = i + 1u) {
            sum = sum + q.stage_ms[i];
        }
        out.scalar = sum;
    } else if (q.op == 3u) {
        // is_within_total: total overspend within the EPS guard.
        if (q.total_overspend_ms <= BUDGET_EPS) {
            out.flag = 1u;
        } else {
            out.flag = 0u;
        }
    } else if (q.op == 4u) {
        // any_stage_over: any per-stage overspend beyond the EPS guard.
        var any_over: u32 = 0u;
        for (var i: u32 = 0u; i < STAGE_COUNT; i = i + 1u) {
            if (q.stage_overspend_ms[i] > BUDGET_EPS) {
                any_over = 1u;
            }
        }
        out.flag = any_over;
    } else if (q.op == 5u) {
        // overspend_ratio: guarded divide, 0.0 for a (near) zero target.
        if (q.total_target_ms <= BUDGET_EPS) {
            out.scalar = 0.0;
        } else {
            out.scalar = q.total_overspend_ms / q.total_target_ms;
        }
    } else {
        // micros_to_ms: integer microseconds to fractional milliseconds.
        out.scalar = f32(q.micros) / 1000.0;
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`PERF_BUDGET_WGSL`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every lane is `4`-byte aligned (`f32`/`u32` only) with no interior padding, so
/// the host and device agree byte for byte.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Per-stage measured milliseconds.
    stage_ms: [f32; STAGE_COUNT],
    /// Per-stage target milliseconds.
    stage_targets_ms: [f32; STAGE_COUNT],
    /// Per-stage overspend milliseconds (input for the any-stage-over op).
    stage_overspend_ms: [f32; STAGE_COUNT],
    /// Total target milliseconds.
    total_target_ms: f32,
    /// Total overspend milliseconds (input for within-total / ratio ops).
    total_overspend_ms: f32,
    /// Microsecond count for the micros-to-ms op.
    micros: u32,
    /// Op classification code (`0..=6`).
    op: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The `scalar` lane carries the single-value ops and the `flag` lane
/// carries the boolean verdicts as a `0`/`1` `u32`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Per-stage overspend milliseconds (evaluate op).
    stage_overspend_ms: [f32; STAGE_COUNT],
    /// Measured total milliseconds (evaluate op).
    total_measured_ms: f32,
    /// Overspent total milliseconds (evaluate op).
    total_overspend_ms: f32,
    /// Shared scalar output lane.
    scalar: f32,
    /// Shared boolean output lane (`0` or `1`).
    flag: u32,
}

/// One tagged query selecting which reference routine the kernel evaluates.
///
/// There is one variant per twinned `CPU` golden routine; a field a variant does
/// not name is ignored. The discriminant order matches the `u32` op codes the
/// kernel branches on.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PerfBudgetQuery {
    /// Evaluates a per-stage sample against a budget, matching
    /// [`FrameTimeBudget::evaluate`](prism_render_architecture::particle::perf_budget::FrameTimeBudget::evaluate).
    Evaluate {
        /// Per-stage measured milliseconds.
        stage_ms: [f32; STAGE_COUNT],
        /// Per-stage target milliseconds.
        stage_targets_ms: [f32; STAGE_COUNT],
        /// Total target milliseconds.
        total_target_ms: f32,
    },
    /// Sums the per-stage targets, matching
    /// [`FrameTimeBudget::stage_sum_ms`](prism_render_architecture::particle::perf_budget::FrameTimeBudget::stage_sum_ms).
    StageTargetSum {
        /// Per-stage target milliseconds.
        stage_targets_ms: [f32; STAGE_COUNT],
    },
    /// Sums the per-stage measured times, matching
    /// [`FrameTimeSample::total_ms`](prism_render_architecture::particle::perf_budget::FrameTimeSample::total_ms).
    SampleTotalMs {
        /// Per-stage measured milliseconds.
        stage_ms: [f32; STAGE_COUNT],
    },
    /// Whether the frame total stayed within budget, matching
    /// [`FrameTimeReport::is_within_total`](prism_render_architecture::particle::perf_budget::FrameTimeReport::is_within_total).
    IsWithinTotal {
        /// Total overspend milliseconds.
        total_overspend_ms: f32,
    },
    /// Whether any single stage ran over, matching
    /// [`FrameTimeReport::any_stage_over`](prism_render_architecture::particle::perf_budget::FrameTimeReport::any_stage_over).
    AnyStageOver {
        /// Per-stage overspend milliseconds.
        stage_overspend_ms: [f32; STAGE_COUNT],
    },
    /// The guarded overspend ratio, matching
    /// [`FrameTimeReport::overspend_ratio`](prism_render_architecture::particle::perf_budget::FrameTimeReport::overspend_ratio).
    OverspendRatio {
        /// Total overspend milliseconds.
        total_overspend_ms: f32,
        /// Total target milliseconds.
        total_target_ms: f32,
    },
    /// Microseconds to milliseconds, matching
    /// [`micros_to_ms`](prism_render_architecture::particle::perf_budget::micros_to_ms).
    MicrosToMs {
        /// Microsecond count.
        micros: u32,
    },
}

impl PerfBudgetQuery {
    /// The `u32` op code the kernel branches on for this query.
    const fn code(&self) -> u32 {
        match self {
            PerfBudgetQuery::Evaluate { .. } => 0,
            PerfBudgetQuery::StageTargetSum { .. } => 1,
            PerfBudgetQuery::SampleTotalMs { .. } => 2,
            PerfBudgetQuery::IsWithinTotal { .. } => 3,
            PerfBudgetQuery::AnyStageOver { .. } => 4,
            PerfBudgetQuery::OverspendRatio { .. } => 5,
            PerfBudgetQuery::MicrosToMs { .. } => 6,
        }
    }
}

/// One decoded result, matching the variant of the [`PerfBudgetQuery`] that
/// produced it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PerfBudgetResult {
    /// The per-stage and total overspend verdict of an evaluate query.
    Evaluate {
        /// Per-stage overspend milliseconds.
        stage_overspend_ms: [f32; STAGE_COUNT],
        /// Measured total milliseconds.
        total_measured_ms: f32,
        /// Overspent total milliseconds.
        total_overspend_ms: f32,
    },
    /// The summed per-stage targets, in milliseconds.
    StageTargetSum {
        /// Summed target milliseconds.
        sum_ms: f32,
    },
    /// The summed per-stage measured times, in milliseconds.
    SampleTotalMs {
        /// Summed measured milliseconds.
        sum_ms: f32,
    },
    /// Whether the frame total stayed within budget.
    IsWithinTotal {
        /// `true` when the total overspend is within the `BUDGET_EPS` guard.
        within: bool,
    },
    /// Whether any single stage ran over its own target.
    AnyStageOver {
        /// `true` when some per-stage overspend exceeds the `BUDGET_EPS` guard.
        any_over: bool,
    },
    /// The guarded overspend ratio.
    OverspendRatio {
        /// Overspend as a fraction of the target (`0.0` for a near-zero target).
        ratio: f32,
    },
    /// The millisecond conversion of a microsecond count.
    MicrosToMs {
        /// Converted milliseconds.
        ms: f32,
    },
}

/// Encodes a [`PerfBudgetQuery`] into its flat `std430` device form, zeroing
/// every lane the variant does not populate.
fn encode_query(query: &PerfBudgetQuery) -> GpuQuery {
    let mut gpu = GpuQuery {
        stage_ms: [0.0; STAGE_COUNT],
        stage_targets_ms: [0.0; STAGE_COUNT],
        stage_overspend_ms: [0.0; STAGE_COUNT],
        total_target_ms: 0.0,
        total_overspend_ms: 0.0,
        micros: 0,
        op: query.code(),
    };
    match *query {
        PerfBudgetQuery::Evaluate {
            stage_ms,
            stage_targets_ms,
            total_target_ms,
        } => {
            gpu.stage_ms = stage_ms;
            gpu.stage_targets_ms = stage_targets_ms;
            gpu.total_target_ms = total_target_ms;
        }
        PerfBudgetQuery::StageTargetSum { stage_targets_ms } => {
            gpu.stage_targets_ms = stage_targets_ms;
        }
        PerfBudgetQuery::SampleTotalMs { stage_ms } => {
            gpu.stage_ms = stage_ms;
        }
        PerfBudgetQuery::IsWithinTotal { total_overspend_ms } => {
            gpu.total_overspend_ms = total_overspend_ms;
        }
        PerfBudgetQuery::AnyStageOver { stage_overspend_ms } => {
            gpu.stage_overspend_ms = stage_overspend_ms;
        }
        PerfBudgetQuery::OverspendRatio {
            total_overspend_ms,
            total_target_ms,
        } => {
            gpu.total_overspend_ms = total_overspend_ms;
            gpu.total_target_ms = total_target_ms;
        }
        PerfBudgetQuery::MicrosToMs { micros } => {
            gpu.micros = micros;
        }
    }
    gpu
}

/// Decodes one raw device [`GpuResult`] into the [`PerfBudgetResult`] matching
/// the query that produced it.
fn decode_result(query: &PerfBudgetQuery, raw: &GpuResult) -> PerfBudgetResult {
    match query {
        PerfBudgetQuery::Evaluate { .. } => PerfBudgetResult::Evaluate {
            stage_overspend_ms: raw.stage_overspend_ms,
            total_measured_ms: raw.total_measured_ms,
            total_overspend_ms: raw.total_overspend_ms,
        },
        PerfBudgetQuery::StageTargetSum { .. } => {
            PerfBudgetResult::StageTargetSum { sum_ms: raw.scalar }
        }
        PerfBudgetQuery::SampleTotalMs { .. } => {
            PerfBudgetResult::SampleTotalMs { sum_ms: raw.scalar }
        }
        PerfBudgetQuery::IsWithinTotal { .. } => PerfBudgetResult::IsWithinTotal {
            within: raw.flag != 0,
        },
        PerfBudgetQuery::AnyStageOver { .. } => PerfBudgetResult::AnyStageOver {
            any_over: raw.flag != 0,
        },
        PerfBudgetQuery::OverspendRatio { .. } => {
            PerfBudgetResult::OverspendRatio { ratio: raw.scalar }
        }
        PerfBudgetQuery::MicrosToMs { .. } => PerfBudgetResult::MicrosToMs { ms: raw.scalar },
    }
}

/// Builds a storage/uniform bind-group-layout entry at `binding`.
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

/// A compiled, reusable frame-time-budget compute pipeline, twinning the `CPU`
/// golden [`perf_budget`](prism_render_architecture::particle::perf_budget).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
pub struct GpuPerfBudget {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPerfBudget {
    /// Compiles the frame-time-budget kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPerfBudget {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_perf_budget"),
            source: ShaderSource::Wgsl(PERF_BUDGET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_perf_budget_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_perf_budget_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_perf_budget_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPerfBudget {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`PerfBudgetResult`]
    /// per input, in order.
    ///
    /// Each result matches the `CPU` golden within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::perf_budget`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[PerfBudgetQuery]) -> Vec<PerfBudgetResult> {
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
            label: Some("prism_volumetric_perf_budget_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_perf_budget_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_perf_budget_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_perf_budget_bind_group"),
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
            label: Some("prism_volumetric_perf_budget_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_perf_budget_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_perf_budget_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
