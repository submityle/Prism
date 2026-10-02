#![forbid(unsafe_code)]
//! `wgpu` compute twin of the integer `Bresenham` line-rasterization golden
//! ([`bresenham_line`](prism_render_architecture::particle::bresenham_line),
//! particle design §12-§13, §29).
//!
//! The `CPU` golden
//! [`rasterize`](prism_render_architecture::particle::bresenham_line::rasterize)
//! turns a pair of integer endpoints into the ordered list of grid cells a
//! straight segment crosses (endpoints included), and
//! [`step_count`](prism_render_architecture::particle::bresenham_line::step_count)
//! reports how many cells that walk contains (`max(|dx|, |dy|) + 1`).
//! [`GpuBresenhamLine`] is the on-device twin: one thread walks one line, so a
//! passing real-device parity test is direct evidence the ported kernel folds
//! the same error-term walk, the same dominant-axis selection and the same
//! reversal convention the reference does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the whole reference walk: it computes
//! the cell count `max(|dx|, |dy|) + 1`, walks `Bresenham`'s single-accumulator
//! error algorithm across all eight octants (`err = dx - dy`, stepping the
//! dominant axis every iteration and the minor axis only when the doubled error
//! crosses zero), and emits the ordered grid cells. To match the reference's
//! reversal convention — `rasterize(a, b)` is the exact reverse of
//! `rasterize(b, a)` — the walk is always computed from the lexicographically
//! smaller endpoint and the output order is flipped when the caller passed the
//! larger endpoint first.
//!
//! # Honest scope: a coordinate-restricted subset
//!
//! The golden widens its arithmetic to `i64` so a segment spanning the full
//! `i32` range can never overflow the error term. `WGSL` has no `i64` (only
//! `i32`, `u32`, `f32`, `bool`), so the full `i32` coordinate range is **not
//! twinnable**. This twin therefore covers only the restricted subset where
//! every coordinate magnitude is at most [`MAX_COORD`]: then `|dx|` and `|dy|`
//! are at most `60000`, so the intermediate `2 * err` and `dx - dy` terms stay
//! well inside `i32`, and a pure-`i32` walk reproduces the golden exactly. The
//! parity fixtures all live inside this box.
//!
//! # Layout
//!
//! Each query uploads the two integer endpoints. Each result carries the cell
//! count plus a fixed-capacity [`MAX_CELLS`] buffer of `(x, y)` cells; the
//! kernel writes the count and the first `count` cells, and the host reads back
//! `cells[..count]`. [`MAX_CELLS`] (`65536`) covers the longest line in the
//! restricted box (`60001` cells), so the walk provably terminates and never
//! overruns the output.
//!
//! # Correctness model
//!
//! The walk is pure integer arithmetic — `+`, `-`, `*2`, comparison and integer
//! sign — with no floating point anywhere, so `CPU` and `GPU` agree bit for bit.
//! The parity test asserts an *exact* `==` on the emitted count and on every
//! `(x, y)` cell, in order.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `max`,
//! `select`, `+ - *` and integer comparison — with no transcendental call and
//! no optional device feature, so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bresenham_line`；
//! classic integer `Bresenham` line rasterization plus `wgpu` compute dispatch;
//! 纯整数、无超越数学、无需外部数学库；no third-party engine source or derived code.

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Largest coordinate magnitude the twin accepts. `WGSL` has no `i64`, so the
/// full `i32` range the golden supports is not twinnable; restricting each of
/// `x0`, `y0`, `x1` and `y1` to `[-MAX_COORD, MAX_COORD]` keeps `|dx|` and
/// `|dy|` at most `60000`, so the intermediate `2 * err` and `dx - dy` terms
/// stay well inside `i32` and the pure-`i32` walk matches the golden exactly.
///
/// Provenance: twin precondition for
/// `prism_render_architecture::particle::bresenham_line`.
pub const MAX_COORD: i32 = 30000;

/// Fixed output capacity per lane: the maximum number of cells one line may
/// emit. The longest line inside the restricted box spans `2 * MAX_COORD`
/// cells, i.e. `60001` cells, so `65536` is a comfortable upper bound and the
/// kernel's loop — bounded by the computed count, itself at most this — provably
/// terminates and never overruns the result slots.
///
/// Provenance: fixed-capacity upper bound for the twin of
/// `prism_render_architecture::particle::bresenham_line`.
pub const MAX_CELLS: usize = 65536;

/// The portable core-`WGSL` integer `Bresenham` line kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`rasterize`](prism_render_architecture::particle::bresenham_line::rasterize)
/// cell for cell; see the module documentation for the algorithm.
const BRESENHAM_LINE_WGSL: &str = r#"
// Integer Bresenham line twin: one thread walks one segment and writes the
// ordered grid cells it crosses (endpoints included) plus the cell count. The
// walk is computed from the lexicographically smaller endpoint and the output
// order is flipped when the caller passed the larger endpoint first, so a line
// and its reverse produce mirror-image cell lists, exactly as the CPU golden
// `particle::bresenham_line`. It uses only the portable core-WGSL subset
// (abs/max/select, + - * and integer comparison), needs no transcendental call
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. Coordinates are restricted to |coord| <= 30000 so 2*err and dx-dy stay
// inside i32 (WGSL has no i64); the count bounds the loop so it terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::bresenham_line；纯整数、
// 无需外部数学库；no third-party engine source or derived code.

const MAX_CELLS: u32 = 65536u;

struct Params {
    // Number of valid queries; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Segment endpoints: (x0, y0) -> (x1, y1).
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

struct Result {
    // Number of emitted cells, max(|dx|, |dy|) + 1.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Emitted cells in order; only the first count entries are meaningful.
    cells: array<vec2<i32>, MAX_CELLS>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The +1/0/-1 step from `a` toward `b` along one axis, matching the reference
// `axis_step`. A zero step means the two endpoints share that coordinate.
fn axis_step(a: i32, b: i32) -> i32 {
    if (a < b) {
        return 1;
    }
    if (a > b) {
        return -1;
    }
    return 0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    let span_x = abs(q.x1 - q.x0);
    let span_y = abs(q.y1 - q.y0);
    let n = u32(max(span_x, span_y)) + 1u;

    results[idx].count = n;
    results[idx].pad0 = 0u;
    results[idx].pad1 = 0u;
    results[idx].pad2 = 0u;

    // Walk from the lexicographically smaller endpoint; when (x0, y0) is the
    // larger endpoint the output order is reversed so the forward and reverse
    // walks are exact mirrors, matching the golden.
    let forward = (q.x0 < q.x1) || (q.x0 == q.x1 && q.y0 <= q.y1);
    var fx: i32;
    var fy: i32;
    var tx: i32;
    var ty: i32;
    if (forward) {
        fx = q.x0;
        fy = q.y0;
        tx = q.x1;
        ty = q.y1;
    } else {
        fx = q.x1;
        fy = q.y1;
        tx = q.x0;
        ty = q.y0;
    }

    let dx = abs(tx - fx);
    let dy = abs(ty - fy);
    let sx = axis_step(fx, tx);
    let sy = axis_step(fy, ty);

    var x = fx;
    var y = fy;
    var err = dx - dy;
    var i = 0u;
    loop {
        // The count bounds the loop, so termination is provable even if a port
        // bug perturbed the error walk.
        if (i >= n) {
            break;
        }
        let out_index = select(i, n - 1u - i, !forward);
        results[idx].cells[out_index] = vec2<i32>(x, y);
        i = i + 1u;

        let e2 = 2 * err;
        if (e2 > -dy) {
            err = err - dy;
            x = x + sx;
        }
        if (e2 < dx) {
            err = err + dx;
            y = y + sy;
        }
    }
}
"#;

/// One rasterization query: the two integer endpoints of a straight segment.
///
/// Both endpoints must lie inside the restricted box (`|coord| <= MAX_COORD`)
/// so the pure-`i32` kernel reproduces the `i64` golden exactly. Holds only
/// integers, so it derives [`Eq`].
///
/// Provenance: twin query of
/// `prism_render_architecture::particle::bresenham_line`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBresenhamQuery {
    /// Start `x` coordinate.
    pub x0: i32,
    /// Start `y` coordinate.
    pub y0: i32,
    /// End `x` coordinate.
    pub x1: i32,
    /// End `y` coordinate.
    pub y1: i32,
}

/// The resolved walk for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `step_count` is the number of grid cells the segment crosses
/// (`max(|dx|, |dy|) + 1`) and equals `cells.len()`. `cells` is the ordered
/// list of `(x, y)` grid cells, endpoints included, in the caller's direction.
/// Holds only integers, so it derives [`Eq`].
///
/// Provenance: twin result of
/// `prism_render_architecture::particle::bresenham_line`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuBresenhamResult {
    /// Number of emitted cells, `max(|dx|, |dy|) + 1`.
    pub step_count: u32,
    /// The ordered grid cells the segment crosses, endpoints included.
    pub cells: Vec<(i32, i32)>,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`BRESENHAM_LINE_WGSL`]: the query count and three pad words,
/// `16` bytes.
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

/// One query as uploaded. `repr(C)` `std430` layout matching `Query` in the
/// shader: the four `i32` endpoint coordinates, `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Start `x` coordinate.
    x0: i32,
    /// Start `y` coordinate.
    y0: i32,
    /// End `x` coordinate.
    x1: i32,
    /// End `y` coordinate.
    y1: i32,
}

/// One result as read back. `repr(C)` `std430` layout matching `Result` in the
/// shader: the emitted cell count, three pad words (so the `vec2<i32>` array
/// starts `8`-byte aligned) and the fixed `(x, y)` cell slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Number of emitted cells.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Emitted cells in order; only the first `count` entries are meaningful.
    cells: [[i32; 2]; MAX_CELLS],
}

/// Encodes one [`GpuBresenhamQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GpuBresenhamQuery) -> GpuQuery {
    GpuQuery {
        x0: q.x0,
        y0: q.y0,
        x1: q.x1,
        y1: q.y1,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuBresenhamResult`],
/// keeping only the first `count` cells (clamped to [`MAX_CELLS`]).
fn decode_result(raw: &GpuResult) -> GpuBresenhamResult {
    let n = (raw.count as usize).min(MAX_CELLS);
    let mut cells = Vec::with_capacity(n);
    for cell in raw.cells.iter().take(n) {
        cells.push((cell[0], cell[1]));
    }
    GpuBresenhamResult {
        step_count: n as u32,
        cells,
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

/// A compiled, reusable integer `Bresenham` line-rasterization pipeline.
pub struct GpuBresenhamLine {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBresenhamLine {
    /// Compiles the `Bresenham` line kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBresenhamLine {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bresenham_line"),
            source: ShaderSource::Wgsl(BRESENHAM_LINE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bresenham_line_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bresenham_line_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bresenham_line_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBresenhamLine {
            module,
            layout,
            pipeline,
        }
    }

    /// Rasterizes every query in `queries`, returning one [`GpuBresenhamResult`]
    /// per query in input order.
    ///
    /// The returned result for query `q` mirrors the `CPU` golden
    /// [`rasterize`](prism_render_architecture::particle::bresenham_line::rasterize)
    /// evaluated on `q`'s endpoints, with `step_count` equal to
    /// [`step_count`](prism_render_architecture::particle::bresenham_line::step_count).
    /// Every endpoint must lie inside the restricted box
    /// (`|coord| <= MAX_COORD`). An empty `queries` slice yields an empty result
    /// — storage buffers cannot be zero-sized, so it is handled by an early
    /// return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuBresenhamQuery]) -> Vec<GpuBresenhamResult> {
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
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bresenham_line_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bresenham_line_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bresenham_line_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bresenham_line_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bresenham_line_bind_group"),
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
            label: Some("prism_volumetric_bresenham_line_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bresenham_line_pass"),
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
