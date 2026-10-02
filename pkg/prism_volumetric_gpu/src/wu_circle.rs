//! `wgpu` compute twin of the per-column coverage primitive inside the Xiaolin
//! Wu anti-aliased circle rasterization contract
//! ([`wu_circle`](prism_render_architecture::particle::wu_circle), particle
//! design §12-§13, §16, §29).
//!
//! The `CPU` golden
//! [`wu_circle`](prism_render_architecture::particle::wu_circle) turns a center
//! and radius into a deterministic, de-duplicated, sorted list of
//! `(x, y, coverage)` samples
//! ([`rasterize`](prism_render_architecture::particle::wu_circle::rasterize)).
//! That public entry point is a *variable-length aggregate*: it walks the first
//! octant, mirrors every column through the ring's eight symmetries, merges
//! coincident lattice cells by keeping the maximum coverage, and returns the
//! survivors sorted in a `BTreeMap`. The dedup/sort/symmetry aggregate, the
//! degenerate `r.is_nan() || r < 0` empty result and the degenerate
//! `r <= RADIUS_EPS` single center pixel all stay on the host: they are
//! variable-length container work with no fixed-width device analogue.
//!
//! [`GpuWuCircle`] is the on-device twin of the *numeric core* that aggregate is
//! built from: the Wu anti-aliased coverage of one first-octant integer column.
//! One thread solves one column, reproducing the reference's exact closed form,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same sub-pixel coverage split the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For an integer column `x` and radius `r` the reference computes the ideal
//! ring height `y = sqrt(r^2 - x^2)` (guarding a tiny negative radicand to
//! zero), splits it across the two straddling pixel rows — the lower row
//! `floor(y)` receives `1 - frac` and the row above it receives
//! `frac = y - floor(y)` — and stops a column walk once it passes the
//! 45-degree diagonal (`2 * x^2 > r^2`). The twin reproduces each of those
//! quantities per column: the lower-row `y` coordinate `floor(y)`, the lower
//! coverage `cov_lo = 1 - frac`, the upper coverage `cov_hi = frac`, and the
//! `within_diagonal` flag that marks whether the column is still inside the
//! first octant. The two coverages a single column emits therefore sum to one,
//! exactly as the reference's lower/upper emit pair does.
//!
//! # What stays on the host
//!
//! The eight-way symmetry emit
//! ([`emit_octant_symmetry`](prism_render_architecture::particle::wu_circle)),
//! the maximum-coverage merge of coincident cells, the ascending `BTreeMap`
//! sort, the `COVERAGE_EPS` drop of near-zero samples, the `NaN`/negative empty
//! result and the `RADIUS_EPS` single-center-pixel case are all variable-length
//! container work that the host owns; the device never sees them. The host
//! enqueues only the normal first-octant columns of a drawable ring
//! (`r > RADIUS_EPS`), so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The discrete answers — the column `x`, the lower-row `y`, and the
//! `within_diagonal` flag — are built from `floor`-based rounding and magnitude
//! comparisons, so for fixtures chosen clear of a coverage tie (a ring height
//! `y` near an integer, or a column near the exact diagonal) the `CPU` and
//! `GPU` agree exactly and the parity test asserts an exact `==` on each. The
//! continuous `cov_lo` and `cov_hi` thread through a subtract and a `sqrt`, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` may land a few units in the
//! last place from the scalar reference. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the coverages, tight
//! enough to catch a genuinely wrong port (a dropped `1 -`, a swapped row, a
//! wrong radicand) yet loose enough to admit a legal last-place `sqrt`
//! difference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `sqrt`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no `ceil`: the
//! integer part is `floor(v)` and the fractional part is `v - floor(v)`. No
//! optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::wu_circle`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Wu anti-aliased circle per-column kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`wu_circle`](prism_render_architecture::particle::wu_circle) per-column
/// closed form; see the module documentation for the algorithm.
const WU_CIRCLE_WGSL: &str = r#"
// Wu anti-aliased circle per-column twin: one thread computes one first-octant
// integer column's ring height split into a lower-row and upper-row coverage,
// mirroring the CPU golden `particle::wu_circle` closed form with only
// floor/sqrt and + - * /. It owns no symmetry emit, no coincident-cell merge
// and no sort; those variable-length aggregates stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::wu_circle；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of columns in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The ring radius r (always > RADIUS_EPS; degenerate radii stay host-side).
    radius: f32,
    // The first-octant integer column x this thread evaluates.
    column: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Lower-row coverage 1 - frac for the pixel (x, floor(y)).
    cov_lo: f32,
    // Upper-row coverage frac for the pixel (x, floor(y) + 1).
    cov_hi: f32,
    // The column x echoed back as a signed coordinate.
    x: i32,
    // The lower pixel row floor(y).
    y_lo: i32,
    // 1 when the column is inside the first octant (2*x^2 <= r^2), else 0.
    within_diagonal: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
    let r = q.radius;
    let r_sq = r * r;
    let xf = f32(q.column);

    // The column walk stops once it passes the 45-degree diagonal; the flag
    // mirrors the reference loop's `2 * x^2 > r^2` break condition.
    var within: u32 = 0u;
    if (2.0 * xf * xf <= r_sq) {
        within = 1u;
    }

    // Ideal ring height y = sqrt(r^2 - x^2), guarding a tiny negative radicand
    // from floating-point error to zero, matching the reference.
    let inside_raw = r_sq - xf * xf;
    var inside = inside_raw;
    if (inside_raw <= 0.0) {
        inside = 0.0;
    }
    let y = sqrt(inside);

    // Integer and fractional parts spelled with floor so no banned round/ceil
    // is used: ipart(y) = floor(y), fpart(y) = y - floor(y).
    let y_lo = floor(y);
    let frac = y - floor(y);

    var out: Result;
    // Lower row keeps the majority weight; the row above takes the rest, so the
    // two coverages a column emits sum to one.
    out.cov_lo = 1.0 - frac;
    out.cov_hi = frac;
    out.x = i32(q.column);
    out.y_lo = i32(y_lo);
    out.within_diagonal = within;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the column count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WU_CIRCLE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid columns in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one column query: a radius and a column index
/// plus two pad words to a `16`-byte stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The ring radius `r`.
    radius: f32,
    /// The first-octant integer column `x`.
    column: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one column result, matching the `WGSL` `Result`
/// struct: two coverages, two coordinates, a flag, and three pad words to a
/// `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Lower-row coverage `1 - frac`.
    cov_lo: f32,
    /// Upper-row coverage `frac`.
    cov_hi: f32,
    /// Column `x` echoed back as a signed coordinate.
    x: i32,
    /// Lower pixel row `floor(y)`.
    y_lo: i32,
    /// `1` when the column is inside the first octant, `0` otherwise.
    within_diagonal: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One per-column query for the Wu anti-aliased circle twin: the ring `radius`
/// and the first-octant integer `column` to evaluate.
///
/// The host owns the surrounding aggregate — the symmetry emit, the
/// coincident-cell merge, the sort, and the degenerate radius cases — and
/// enqueues one [`WuCircleQuery`] per normal first-octant column of a drawable
/// ring, matching the reference
/// [`rasterize`](prism_render_architecture::particle::wu_circle::rasterize)
/// column walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WuCircleQuery {
    /// The ring radius `r`; the host only enqueues drawable rings
    /// (`r > RADIUS_EPS`).
    pub radius: f32,
    /// The first-octant integer column `x` to evaluate.
    pub column: u32,
}

impl WuCircleQuery {
    /// Builds a query for `column` of the ring with radius `radius`.
    #[must_use]
    pub const fn new(radius: f32, column: u32) -> WuCircleQuery {
        WuCircleQuery { radius, column }
    }
}

/// One resolved column of the Wu anti-aliased circle, mirroring the lower/upper
/// coverage split the reference
/// [`rasterize`](prism_render_architecture::particle::wu_circle::rasterize)
/// emits for a first-octant column.
///
/// The lower pixel `(x, y_lo)` receives `cov_lo = 1 - frac` and the pixel above
/// it `(x, y_lo + 1)` receives `cov_hi = frac`, so the two coverages sum to one.
/// `within_diagonal` is `true` while the column is still inside the first octant
/// (`2 * x^2 <= r^2`), mirroring the reference column walk's break condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WuCircleResult {
    /// The column `x` echoed back as a signed coordinate.
    pub x: i32,
    /// The lower pixel row `floor(y)`.
    pub y_lo: i32,
    /// Lower-row coverage `1 - frac` for the pixel `(x, y_lo)`.
    pub cov_lo: f32,
    /// Upper-row coverage `frac` for the pixel `(x, y_lo + 1)`.
    pub cov_hi: f32,
    /// `true` when the column is inside the first octant (`2 * x^2 <= r^2`).
    pub within_diagonal: bool,
}

/// Encodes one [`WuCircleQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WuCircleQuery) -> GpuQuery {
    GpuQuery {
        radius: q.radius,
        column: q.column,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WuCircleResult`], turning
/// the `within_diagonal` word back into a [`bool`].
fn decode_result(raw: &GpuResult) -> WuCircleResult {
    WuCircleResult {
        x: raw.x,
        y_lo: raw.y_lo,
        cov_lo: raw.cov_lo,
        cov_hi: raw.cov_hi,
        within_diagonal: raw.within_diagonal != 0,
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

/// A compiled, reusable Wu anti-aliased circle per-column compute pipeline,
/// twinning the numeric core of the `CPU` golden
/// [`wu_circle`](prism_render_architecture::particle::wu_circle).
pub struct GpuWuCircle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWuCircle {
    /// Compiles the Wu-circle per-column kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWuCircle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_wu_circle"),
            source: ShaderSource::Wgsl(WU_CIRCLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_wu_circle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_wu_circle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_wu_circle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWuCircle {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every column in `queries` and returns one [`WuCircleResult`] per
    /// input, in order.
    ///
    /// The column `x`, the lower row `y_lo` and the `within_diagonal` flag equal
    /// the reference exactly for columns clear of a ring-height or diagonal tie;
    /// the two coverages match to within the tolerance documented on this
    /// module. An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[WuCircleQuery]) -> Vec<WuCircleResult> {
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
            label: Some("prism_volumetric_wu_circle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_wu_circle_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_wu_circle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_wu_circle_bind_group"),
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
            label: Some("prism_volumetric_wu_circle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_wu_circle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_wu_circle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per column, flattened to a 1-D dispatch.
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
