//! `wgpu` compute twin of the analytic sub-pixel hair line-coverage kernel,
//! mirroring this repository's `prism_render_architecture::hair::line_coverage`
//! module.
//!
//! A single hair fibre projected to screen is almost always thinner than one
//! pixel, so a hard point-sampled rasteriser shimmers. The golden module fixes
//! this analytically: it models the pixel as a round coverage kernel of radius
//! `0.5` and the fibre as a capsule of half-width `width * 0.5`, then returns
//! the monotone trapezoidal ramp `clamp(width*0.5 + 0.5 - d, 0, 1)`, where `d`
//! is the point-to-segment distance from the pixel centre. This twin
//! reproduces, for one query per thread, the three closed forms of that path:
//!
//! - `sanitized_width`: the stroke width clamped to a finite, non-negative
//!   value (negative or non-finite becomes `0`).
//! - `point_segment_distance`: the shortest distance from the pixel centre to
//!   the segment `[a, b]`, with the projection parameter clamped to `[0, 1]`
//!   and a degenerate (near-zero-length) segment collapsing to the point
//!   distance `|p - a|`.
//! - `pixel_coverage`: the clamped trapezoidal coverage ramp above.
//!
//! The array-batched `coverage_map` is intentionally not twinned; the kernel is
//! already one-thread-per-pixel, so a whole tile is one dispatch of per-pixel
//! queries.
//!
//! # Finite-width guard
//!
//! The golden `sanitized_width` is `if width.is_finite() && width > 0.0 { width
//! } else { 0.0 }`. `WGSL` has no `isFinite`, so the kernel tests the `f32`
//! exponent field directly: a finite value has an exponent not equal to all
//! ones, so `NaN` and the infinities are rejected to `0`, and the ordered
//! `width > 0.0` compare drops negatives (and `NaN`, which fails every ordered
//! compare). This matches the golden result for every width without a bare
//! floating-point equality test.
//!
//! # Degenerate segment
//!
//! A segment whose squared length is below [`EPS`] is treated as the single
//! point `a`, so the distance is `|p - a|` and the kernel never divides by a
//! zero length or returns `NaN`. The predicate is the ordered compare
//! `abs(len_sq) < EPS`, mirroring the golden `len_sq.abs() < EPS`.
//!
//! # Precision model
//!
//! The golden path evaluates in `f32`; this twin and its host oracle both
//! evaluate the same closed form in `f32`, each query a fixed, non-reorderable
//! sequence of multiplies, adds, absolute values, one `clamp` and one `sqrt`,
//! so `CPU` and `GPU` compute the same arithmetic in the same order. They are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on every continuous output and an exact `==` on
//! `valid`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `sqrt`, `select`, `bitcast` and `+ - * /` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, `round` or optional device feature, and no `u64`,
//! `i64` or `f64`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! only non-rational operation is the distance `sqrt`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::line_coverage`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly
/// default shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The reference epsilon used to detect a degenerate (zero-length) segment,
/// matching the golden module's `EPS`.
pub const EPS: f32 = 1.0e-6;

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `line_coverage` mirrors the
/// finite-width guard, the point-to-segment distance and the trapezoidal
/// coverage ramp of the `CPU` golden module; see the module documentation for
/// the algorithm.
const HAIR_LINE_COVERAGE_WGSL: &str = r#"
// Hair line-coverage twin: one thread per query evaluates the sanitized stroke
// width, the point-to-segment distance from the pixel centre and the clamped
// trapezoidal coverage ramp clamp(width*0.5 + 0.5 - d, 0, 1). It uses only the
// portable core-WGSL subset (abs/clamp/sqrt/select/bitcast and + - * /) with no
// u64/i64/f64 and no transcendental, so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::line_coverage；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the screen segment endpoints a and b, the pixel centre and the
// stroke width (plus a pad lane to round the struct to 32 bytes).
struct Query {
    a: vec2<f32>,
    b: vec2<f32>,
    center: vec2<f32>,
    width: f32,
    pad0: f32,
}

// One result: the coverage ramp, the point-to-segment distance, the valid flag
// and a pad word to round to 16 bytes.
struct Res {
    coverage: f32,
    dist: f32,
    valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// width.is_finite() && width > 0.0, else 0.0. A finite f32 has an exponent
// field (bits 23..30) that is not all ones, so NaN and the infinities are
// rejected; the ordered width > 0.0 compare drops negatives and NaN. This
// avoids any bare floating-point equality test.
fn sanitized_width(w: f32) -> f32 {
    let bits = bitcast<u32>(w);
    let exp = (bits >> 23u) & 0xffu;
    let finite = exp != 0xffu;
    return select(0.0, w, finite && (w > 0.0));
}

// Shortest distance from p to the segment [a, b]; a degenerate segment (squared
// length below EPS) collapses to the point distance |p - a|.
fn point_segment_distance(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let abx = bx - ax;
    let aby = by - ay;
    let len_sq = abx * abx + aby * aby;
    let apx = px - ax;
    let apy = py - ay;
    if (abs(len_sq) < 1e-6) {
        return sqrt(apx * apx + apy * apy);
    }
    let t = clamp((apx * abx + apy * aby) / len_sq, 0.0, 1.0);
    let cx = ax + t * abx;
    let cy = ay + t * aby;
    let dx = px - cx;
    let dy = py - cy;
    return sqrt(dx * dx + dy * dy);
}

@compute @workgroup_size(64)
fn line_coverage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let half_width = sanitized_width(q.width) * 0.5;
    let d = point_segment_distance(
        q.center.x, q.center.y, q.a.x, q.a.y, q.b.x, q.b.y,
    );
    let cov = clamp(half_width + 0.5 - d, 0.0, 1.0);

    var out: Res;
    out.coverage = cov;
    out.dist = d;
    out.valid = 1u;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// One hair line-coverage query: a screen-space segment (two pixel-coordinate
/// endpoints and a stroke width) plus the pixel centre to shade.
///
/// `width` need not be finite or positive; the kernel sanitizes it before use.
/// Derives only [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32`
/// geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairLineCoverageQuery {
    /// Segment start in pixel coordinates.
    pub a: [f32; 2],
    /// Segment end in pixel coordinates.
    pub b: [f32; 2],
    /// Stroke width in pixels; negative / non-finite is sanitized to `0`.
    pub width: f32,
    /// The pixel centre to compute coverage for.
    pub pixel_center: [f32; 2],
}

impl HairLineCoverageQuery {
    /// Builds a query from a segment, a stroke width and a pixel centre.
    #[must_use]
    pub const fn new(
        a: [f32; 2],
        b: [f32; 2],
        width: f32,
        pixel_center: [f32; 2],
    ) -> HairLineCoverageQuery {
        HairLineCoverageQuery {
            a,
            b,
            width,
            pixel_center,
        }
    }
}

/// The hair line-coverage outputs for one query, the host-side mirror of the
/// kernel's `Res` lane.
///
/// `coverage` is the clamped trapezoidal ramp in `[0, 1]`, `distance` the
/// point-to-segment distance from the pixel centre and `valid` the always-`1`
/// flag. Derives only [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds
/// `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairLineCoverageResult {
    /// Analytic coverage ramp in `[0, 1]`.
    pub coverage: f32,
    /// Point-to-segment distance from the pixel centre.
    pub distance: f32,
    /// Always `1`: there is no degenerate rejection.
    pub valid: u32,
}

/// `repr(C)` `std430` layout of one packed query: the two endpoints, the pixel
/// centre, the width and one pad lane — `32` bytes, exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Segment start.
    a: [f32; 2],
    /// Segment end.
    b: [f32; 2],
    /// Pixel centre.
    center: [f32; 2],
    /// Stroke width.
    width: f32,
    /// Padding lane.
    pad0: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &HairLineCoverageQuery) -> GpuQuery {
        GpuQuery {
            a: query.a,
            b: query.b,
            center: query.pixel_center,
            width: query.width,
            pad0: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the coverage, the distance, the
/// `valid` flag and one pad word in the same order as the `WGSL` `Res` struct —
/// `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Coverage ramp.
    coverage: f32,
    /// Point-to-segment distance.
    dist: f32,
    /// Always-`1` valid flag.
    valid: u32,
    /// Padding word.
    pad0: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// round the uniform block out to `16` bytes.
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

/// Decodes one packed `GpuResult` into the public [`HairLineCoverageResult`].
fn decode_result(raw: &GpuResult) -> HairLineCoverageResult {
    HairLineCoverageResult {
        coverage: raw.coverage,
        distance: raw.dist,
        valid: raw.valid,
    }
}

/// Builds one storage/uniform buffer bind-group-layout entry.
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

/// On-device twin of the analytic hair line-coverage kernel.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuHairLineCoverage::new`] and reuse it across
/// [`GpuHairLineCoverage::evaluate`] calls.
pub struct GpuHairLineCoverage {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `line_coverage` entry point.
    pipeline: ComputePipeline,
}

impl GpuHairLineCoverage {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairLineCoverage {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_shader"),
            source: ShaderSource::Wgsl(HAIR_LINE_COVERAGE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("line_coverage"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairLineCoverage {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every hair line-coverage query on-device and returns one
    /// [`HairLineCoverageResult`] per input, in order.
    ///
    /// Each result equals the reference closed form to within the tolerance
    /// documented on this module. An empty input returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairLineCoverageQuery],
    ) -> Vec<HairLineCoverageResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_line_coverage_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_line_coverage_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
