//! `wgpu` compute twin of the particle statistics debug-overlay layout golden
//! ([`stats_overlay`](prism_render_architecture::particle::stats_overlay),
//! particle design §9 pipeline stats, §13 culling counters, §30 in-engine
//! `HUD`).
//!
//! The `CPU` golden
//! [`stats_overlay`](prism_render_architecture::particle::stats_overlay) owns
//! the deterministic, device-free half that turns a reduced per-frame counter
//! record into a drawable overlay *layout*: the linear histogram bucket a
//! spark-line widget assigns, the clamped grid arithmetic (`cell_count`,
//! `grid_width_px`, `grid_height_px`, `row_index`, `col_index`) and the
//! severity color table.
//!
//! [`GpuStatsOverlay`] is the on-device twin: one thread per
//! [`StatsOverlayQuery`] dispatches on an operation tag and reproduces the
//! module's pure numeric surface branch for branch — the
//! [`histogram_bucket`](prism_render_architecture::particle::stats_overlay::histogram_bucket)
//! divide-and-`floor` with every degenerate range folded to bucket zero, the
//! saturating `u32` products and divides/remainders of
//! [`OverlayLayout`](prism_render_architecture::particle::stats_overlay::OverlayLayout),
//! and the
//! [`OverlayRow::severity_rgba`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity_rgba)
//! classification-code color lookup — so a passing real-device parity test is
//! direct evidence the ported kernel evaluates the same layout the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query mirrors one golden operation:
//!
//! - [`StatsOverlayQuery::HistogramBucket`] →
//!   [`histogram_bucket`](prism_render_architecture::particle::stats_overlay::histogram_bucket).
//! - [`StatsOverlayQuery::CellCount`] / [`StatsOverlayQuery::GridWidth`] /
//!   [`StatsOverlayQuery::GridHeight`] → the three saturating `u32` products
//!   [`OverlayLayout::cell_count`](prism_render_architecture::particle::stats_overlay::OverlayLayout::cell_count),
//!   [`OverlayLayout::grid_width_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_width_px)
//!   and
//!   [`OverlayLayout::grid_height_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_height_px).
//! - [`StatsOverlayQuery::RowIndex`] / [`StatsOverlayQuery::ColIndex`] → the
//!   reverse lookups
//!   [`OverlayLayout::row_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::row_index)
//!   and
//!   [`OverlayLayout::col_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::col_index).
//! - [`StatsOverlayQuery::SeverityRgba`] → the color table of
//!   [`OverlayRow::severity_rgba`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity_rgba).
//!
//! The twin consumes the *already clamped* column and row counts: the golden
//! [`OverlayLayout::new`](prism_render_architecture::particle::stats_overlay::OverlayLayout::new)
//! clamps both up to one on the host, and the twin's grid arithmetic assumes
//! that clamp so the divides and remainders never see a zero divisor. The
//! severity classification itself compares `u64` counters against `u64`
//! thresholds; since the twin has no `u64`, the host reduces each row to a
//! classification code (`0` nominal, `1` warn, `2` critical) and the twin
//! reproduces only the code→`RGBA` mapping.
//!
//! # What stays on the host
//!
//! The quad-batch byte budget
//! [`OverlayLayout::vertex_bytes`](prism_render_architecture::particle::stats_overlay::OverlayLayout::vertex_bytes)
//! widens to `u64` and stays host-side, as the twin carries no `u64`. The
//! variable-length
//! [`build_rows`](prism_render_architecture::particle::stats_overlay::build_rows)
//! builder allocates a `Vec` and the
//! [`OverlayStatField::label`](prism_render_architecture::particle::stats_overlay::OverlayStatField::label)
//! lookup returns a `&'static str`; both are host-only. The `u64` threshold
//! comparison inside
//! [`OverlayRow::severity`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity)
//! and the
//! [`OverlayLayout::new`](prism_render_architecture::particle::stats_overlay::OverlayLayout::new)
//! constructor also stay on the host.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `min`,
//! `max`, integer `+ - * /` and remainder — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `smoothstep` builtin, `round` or optional device feature, and
//! no `u64`/`u16`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! The histogram bucket is a single divide, a multiply and a `floor`, and the
//! grid arithmetic is exact integer work; the `GPU` and `CPU` evaluate the same
//! closed form. The integer results (`bucket`, `cell_count`, indices) are
//! exact, so the parity test asserts bit-equality on them; only the `RGBA`
//! color channels are `f32`, and they are the exact literals `0.0`/`1.0`, so
//! the test compares them with the documented tolerance to honor the
//! `f32`-never-`==` rule.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`；
//! deterministic debug-overlay layout plus `wgpu` compute dispatch；无需外部
//! 数学库，无第三方引擎源码或衍生代码。

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

/// Operation tag: [`histogram_bucket`](prism_render_architecture::particle::stats_overlay::histogram_bucket).
const OP_HISTOGRAM_BUCKET: u32 = 0;
/// Operation tag: [`OverlayLayout::cell_count`](prism_render_architecture::particle::stats_overlay::OverlayLayout::cell_count).
const OP_CELL_COUNT: u32 = 1;
/// Operation tag: [`OverlayLayout::grid_width_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_width_px).
const OP_GRID_WIDTH: u32 = 2;
/// Operation tag: [`OverlayLayout::grid_height_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_height_px).
const OP_GRID_HEIGHT: u32 = 3;
/// Operation tag: [`OverlayLayout::row_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::row_index).
const OP_ROW_INDEX: u32 = 4;
/// Operation tag: [`OverlayLayout::col_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::col_index).
const OP_COL_INDEX: u32 = 5;
/// Operation tag: [`OverlayRow::severity_rgba`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity_rgba).
const OP_SEVERITY_RGBA: u32 = 6;

/// The portable core-`WGSL` statistics-overlay numeric kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// dispatches on a per-query operation tag and mirrors the `CPU` golden
/// [`stats_overlay`](prism_render_architecture::particle::stats_overlay)
/// numeric surface; see the module documentation for the formulae.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`。
const STATS_OVERLAY_WGSL: &str = r#"
// stats_overlay twin: one thread per query dispatches on an operation tag and
// reproduces the CPU golden `particle::stats_overlay` numeric surface — the
// histogram_bucket divide-and-floor with every degenerate range folded to
// bucket zero, the saturating u32 grid products, the cell-index divide and
// remainder reverse lookups and the severity classification-code color table.
// It mirrors the reference branch for branch, uses only the portable core-WGSL
// subset (floor/min/max, integer + - * / and remainder), takes no optional
// feature, uses no u64/u16, and runs unmodified on Metal, Vulkan and DX12. Each
// thread runs a fixed, bounded sequence, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::stats_overlay; no
// external math library, no third-party engine source or derived code.

const OP_HISTOGRAM_BUCKET: u32 = 0u;
const OP_CELL_COUNT: u32 = 1u;
const OP_GRID_WIDTH: u32 = 2u;
const OP_GRID_HEIGHT: u32 = 3u;
const OP_ROW_INDEX: u32 = 4u;
const OP_COL_INDEX: u32 = 5u;
const OP_SEVERITY_RGBA: u32 = 6u;
const KIND_INT: u32 = 0u;
const KIND_RGBA: u32 = 1u;

const U32_MAX: u32 = 4294967295u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which golden operation this slot evaluates.
    op: u32,
    // First u32 operand: columns (cell_count/grid_width), rows
    // (grid_height), or cell (row_index/col_index).
    u0: u32,
    // Second u32 operand: rows (cell_count), cell_width_px (grid_width),
    // cell_height_px (grid_height), or columns (row_index/col_index).
    u1: u32,
    // Histogram bucket count.
    bucket_count: u32,
    // Severity classification code (0 nominal, 1 warn, 2 critical).
    severity_code: u32,
    // Histogram sample value and inclusive range endpoints.
    value: f32,
    min_v: f32,
    max_v: f32,
}

struct Result {
    // Result-kind tag selecting how the host decodes the lanes.
    kind: u32,
    // Integer answer lane (bucket / count / dimension / index).
    lane: u32,
    // RGBA color lanes (severity_rgba).
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Saturating u32 multiply, mirroring the reference `u32::saturating_mul`: WGSL
// integer multiply wraps, so an overflow is detected by dividing the (wrapped)
// product back out and comparing, clamping to u32::MAX on overflow.
fn sat_mul(a: u32, b: u32) -> u32 {
    let product = a * b;
    if (a != 0u && product / a != b) {
        return U32_MAX;
    }
    return product;
}

// Linear histogram bucket in [0, bucket_count - 1], mirroring the reference
// `histogram_bucket`: an empty range (min_v >= max_v) or zero bucket count
// collapses to zero, values at or below min_v land in bucket zero and values
// at or above max_v land in the last bucket, and the interior is
// floor(normalized * bucket_count) clamped to the last bucket.
fn histogram_bucket(value: f32, min_v: f32, max_v: f32, bucket_count: u32) -> u32 {
    if (bucket_count == 0u || min_v >= max_v) {
        return 0u;
    }
    let last = bucket_count - 1u;
    if (value <= min_v) {
        return 0u;
    }
    if (value >= max_v) {
        return last;
    }
    let normalized = (value - min_v) / (max_v - min_v);
    let scaled = normalized * f32(bucket_count);
    let bucket = u32(floor(scaled));
    return min(bucket, last);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.kind = KIND_INT;
    out.lane = 0u;
    out.r = 0.0;
    out.g = 0.0;
    out.b = 0.0;
    out.a = 0.0;

    switch q.op {
        case OP_HISTOGRAM_BUCKET: {
            out.kind = KIND_INT;
            out.lane = histogram_bucket(q.value, q.min_v, q.max_v, q.bucket_count);
        }
        case OP_CELL_COUNT, OP_GRID_WIDTH, OP_GRID_HEIGHT: {
            out.kind = KIND_INT;
            out.lane = sat_mul(q.u0, q.u1);
        }
        case OP_ROW_INDEX: {
            out.kind = KIND_INT;
            out.lane = q.u0 / q.u1;
        }
        case OP_COL_INDEX: {
            out.kind = KIND_INT;
            out.lane = q.u0 % q.u1;
        }
        case OP_SEVERITY_RGBA: {
            out.kind = KIND_RGBA;
            // 0 nominal => green, 1 warn => yellow, 2 critical => red; all
            // fully opaque. The default arm reproduces the nominal color so an
            // out-of-range code never leaves the lanes undefined.
            switch q.severity_code {
                case 2u: {
                    out.r = 1.0;
                    out.g = 0.0;
                    out.b = 0.0;
                    out.a = 1.0;
                }
                case 1u: {
                    out.r = 1.0;
                    out.g = 1.0;
                    out.b = 0.0;
                    out.a = 1.0;
                }
                default: {
                    out.r = 0.0;
                    out.g = 1.0;
                    out.b = 0.0;
                    out.a = 1.0;
                }
            }
        }
        default: {
        }
    }
    results[idx] = out;
}
"#;

/// One numeric query against the statistics-overlay twin.
///
/// Each variant mirrors one golden operation. The grid variants carry the
/// *already clamped* `columns` and `rows` the golden
/// [`OverlayLayout::new`](prism_render_architecture::particle::stats_overlay::OverlayLayout::new)
/// produces (both at least one), so the twin and the reference see identical
/// inputs and the divides never see a zero divisor. The `severity_code` is the
/// host-reduced classification (`0` nominal, `1` warn, `2` critical) since the
/// twin carries no `u64` to compare counters against thresholds.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatsOverlayQuery {
    /// Linear histogram bucket assignment
    /// ([`histogram_bucket`](prism_render_architecture::particle::stats_overlay::histogram_bucket)).
    HistogramBucket {
        /// Sample value to bucket.
        value: f32,
        /// Inclusive lower range endpoint.
        min: f32,
        /// Inclusive upper range endpoint.
        max: f32,
        /// Number of buckets the range is split into.
        bucket_count: u32,
    },
    /// Saturating total cell count `columns * rows`
    /// ([`OverlayLayout::cell_count`](prism_render_architecture::particle::stats_overlay::OverlayLayout::cell_count)).
    CellCount {
        /// Clamped column count (at least one).
        columns: u32,
        /// Clamped row count (at least one).
        rows: u32,
    },
    /// Saturating total grid width `columns * cell_width_px`
    /// ([`OverlayLayout::grid_width_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_width_px)).
    GridWidth {
        /// Clamped column count (at least one).
        columns: u32,
        /// Width of a single cell in pixels.
        cell_width_px: u32,
    },
    /// Saturating total grid height `rows * cell_height_px`
    /// ([`OverlayLayout::grid_height_px`](prism_render_architecture::particle::stats_overlay::OverlayLayout::grid_height_px)).
    GridHeight {
        /// Clamped row count (at least one).
        rows: u32,
        /// Height of a single cell in pixels.
        cell_height_px: u32,
    },
    /// Row index of a linear cell index `cell / columns`
    /// ([`OverlayLayout::row_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::row_index)).
    RowIndex {
        /// Linear cell index.
        cell: u32,
        /// Clamped column count (at least one).
        columns: u32,
    },
    /// Column index of a linear cell index `cell % columns`
    /// ([`OverlayLayout::col_index`](prism_render_architecture::particle::stats_overlay::OverlayLayout::col_index)).
    ColIndex {
        /// Linear cell index.
        cell: u32,
        /// Clamped column count (at least one).
        columns: u32,
    },
    /// Severity color-table lookup
    /// ([`OverlayRow::severity_rgba`](prism_render_architecture::particle::stats_overlay::OverlayRow::severity_rgba)).
    SeverityRgba {
        /// Host-reduced classification code (`0` nominal, `1` warn, `2`
        /// critical).
        severity_code: u32,
    },
}

/// One resolved answer, mirroring whichever golden operation the query
/// selected.
///
/// `Int` carries the integer answers (`histogram_bucket`, `cell_count`,
/// `grid_width_px`, `grid_height_px`, `row_index`, `col_index`); `Rgba` carries
/// the four normalized color channels of `severity_rgba`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatsOverlayResult {
    /// An integer answer (`histogram_bucket` / `cell_count` / `grid_width_px` /
    /// `grid_height_px` / `row_index` / `col_index`).
    Int(u32),
    /// A normalized `RGBA` color (`severity_rgba`).
    Rgba([f32; 4]),
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// All members are `4`-byte scalars, so the struct packs with no interior
/// padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting which golden operation this slot evaluates.
    op: u32,
    /// First `u32` operand (`columns` / `rows` / `cell`).
    u0: u32,
    /// Second `u32` operand (`rows` / `cell_width_px` / `cell_height_px` /
    /// `columns`).
    u1: u32,
    /// Histogram bucket count.
    bucket_count: u32,
    /// Severity classification code.
    severity_code: u32,
    /// Histogram sample value.
    value: f32,
    /// Inclusive lower range endpoint.
    min_v: f32,
    /// Inclusive upper range endpoint.
    max_v: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Result-kind tag.
    kind: u32,
    /// Integer answer lane.
    lane: u32,
    /// Red color channel.
    r: f32,
    /// Green color channel.
    g: f32,
    /// Blue color channel.
    b: f32,
    /// Alpha color channel.
    a: f32,
}

/// A fully zeroed query slot, filled per variant by [`encode_query`].
fn empty_query() -> GpuQuery {
    GpuQuery {
        op: 0,
        u0: 0,
        u1: 0,
        bucket_count: 0,
        severity_code: 0,
        value: 0.0,
        min_v: 0.0,
        max_v: 0.0,
    }
}

/// Encodes one [`StatsOverlayQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(query: &StatsOverlayQuery) -> GpuQuery {
    let mut q = empty_query();
    match query {
        StatsOverlayQuery::HistogramBucket {
            value,
            min,
            max,
            bucket_count,
        } => {
            q.op = OP_HISTOGRAM_BUCKET;
            q.value = *value;
            q.min_v = *min;
            q.max_v = *max;
            q.bucket_count = *bucket_count;
        }
        StatsOverlayQuery::CellCount { columns, rows } => {
            q.op = OP_CELL_COUNT;
            q.u0 = *columns;
            q.u1 = *rows;
        }
        StatsOverlayQuery::GridWidth {
            columns,
            cell_width_px,
        } => {
            q.op = OP_GRID_WIDTH;
            q.u0 = *columns;
            q.u1 = *cell_width_px;
        }
        StatsOverlayQuery::GridHeight {
            rows,
            cell_height_px,
        } => {
            q.op = OP_GRID_HEIGHT;
            q.u0 = *rows;
            q.u1 = *cell_height_px;
        }
        StatsOverlayQuery::RowIndex { cell, columns } => {
            q.op = OP_ROW_INDEX;
            q.u0 = *cell;
            q.u1 = *columns;
        }
        StatsOverlayQuery::ColIndex { cell, columns } => {
            q.op = OP_COL_INDEX;
            q.u0 = *cell;
            q.u1 = *columns;
        }
        StatsOverlayQuery::SeverityRgba { severity_code } => {
            q.op = OP_SEVERITY_RGBA;
            q.severity_code = *severity_code;
        }
    }
    q
}

/// Decodes one [`GpuResult`] back into a [`StatsOverlayResult`], selecting the
/// variant from the query that produced it.
fn decode_result(query: &StatsOverlayQuery, raw: &GpuResult) -> StatsOverlayResult {
    match query {
        StatsOverlayQuery::HistogramBucket { .. }
        | StatsOverlayQuery::CellCount { .. }
        | StatsOverlayQuery::GridWidth { .. }
        | StatsOverlayQuery::GridHeight { .. }
        | StatsOverlayQuery::RowIndex { .. }
        | StatsOverlayQuery::ColIndex { .. } => StatsOverlayResult::Int(raw.lane),
        StatsOverlayQuery::SeverityRgba { .. } => {
            StatsOverlayResult::Rgba([raw.r, raw.g, raw.b, raw.a])
        }
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

/// A compiled, reusable statistics-overlay compute pipeline, twinning the `CPU`
/// golden
/// [`stats_overlay`](prism_render_architecture::particle::stats_overlay).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`。
pub struct GpuStatsOverlay {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStatsOverlay {
    /// Compiles the statistics-overlay kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStatsOverlay {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_stats_overlay"),
            source: ShaderSource::Wgsl(STATS_OVERLAY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_stats_overlay_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_stats_overlay_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_stats_overlay_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStatsOverlay {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`StatsOverlayResult`]
    /// per input, in order.
    ///
    /// The integer results match the reference bit for bit; the `RGBA` color
    /// channels match within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::stats_overlay`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[StatsOverlayQuery],
    ) -> Vec<StatsOverlayResult> {
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
            label: Some("prism_volumetric_stats_overlay_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_stats_overlay_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_stats_overlay_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_stats_overlay_bind_group"),
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
            label: Some("prism_volumetric_stats_overlay_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_stats_overlay_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_stats_overlay_pass"),
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
