//! `wgpu` compute twin of the Xiaolin Wu anti-aliased line rasterization
//! contract
//! ([`wu_line`](prism_render_architecture::particle::wu_line), particle design
//! §12-§13, §29).
//!
//! The `CPU` golden
//! [`wu_line`](prism_render_architecture::particle::wu_line) turns a pair of
//! floating-point endpoints into an ordered list of `(x, y, coverage)` samples,
//! where `coverage` in `[0, 1]` is the fraction of each pixel a thin bright
//! streak paints
//! ([`rasterize`](prism_render_architecture::particle::wu_line::rasterize)).
//! [`GpuWuLineRasterizer`] is the on-device twin: one thread rasterizes one
//! segment, reproducing the reference's whole control flow, so a passing
//! real-device parity test is direct evidence the ported kernel paints the same
//! anti-aliased pixels and drops the same near-zero coverage the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every branch the reference walks is mirrored: the steep-axis canonicalization
//! (`|dy| > |dx|` swaps the major and minor axes, and the emitted coordinates
//! are swapped back into the caller's frame), the left-to-right ordering swap,
//! the degenerate `gradient = 1.0` fallback for a coincident-endpoint segment,
//! the two endpoint columns attenuated by their sub-pixel `x-gap`, the interior
//! major-axis walk that splits each step across two pixels whose coverages sum
//! to one, the single-column merge when both endpoints land in the same major
//! column, and the `COVERAGE_EPS` drop of near-zero samples. The hand-rolled
//! `ipart`, `fpart`, `rfpart`, `round_nearest` and `emit` helpers are
//! reproduced verbatim in the kernel.
//!
//! # Variable-length output
//!
//! A line emits a data-dependent number of samples, so each thread writes into a
//! fixed-capacity slot [`GpuWuLine`] holding a `count` and a `pixels` array of
//! [`GpuPixel`] of length [`MAX_SAMPLES`]; the host trims each slot to its
//! reported `count`. [`MAX_SAMPLES`] is `512`, a comfortable upper bound: the
//! longest line this twin rasterizes, `(0, 0)` to `(100, 40)`, emits about `202`
//! samples. No recursion and no dynamic allocation appear in the kernel; a small
//! fixed scratch of four lanes backs the same-column merge.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `abs`,
//! `clamp`, `min`, `max`, `+ - * /` and integer arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no `round` and no `ceil`: rounding to the
//! nearest integer is spelled `floor(v + 0.5)`, the integer part is `floor(v)`,
//! and the fractional part is `v - floor(v)`. No optional device feature is
//! required, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! The discrete answers — the sample `count` and each sample's integer `(x, y)`
//! — are built from axis swaps, `floor`-based rounding and integer stepping, so
//! for the well-conditioned fixtures (chosen clear of coverage ties) the `CPU`
//! and `GPU` agree exactly and the parity test asserts an exact `==` on both the
//! count and every coordinate. The continuous `coverage` threads through
//! multiplies and adds, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on `coverage`,
//! tight enough to catch a genuinely wrong port (a dropped branch, a swapped
//! axis, a wrong gap term) yet loose enough to admit legal fused multiply-add
//! contraction. Because the emitted order is deterministic but can differ in
//! presentation, both sides are compared as a set sorted by `(x, y)`.
//!
//! # Degenerate inputs
//!
//! A coincident-endpoint segment yields `gradient = 1.0` and an empty interior
//! loop, so the kernel reports whatever the two endpoint columns paint (possibly
//! a single merged column) exactly as the reference does. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`wu_line`](prism_render_architecture::particle::wu_line); no third-party
//! engine source or derived code.

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

/// Fixed capacity of one line's sample slot. The longest fixture line,
/// `(0, 0)` to `(100, 40)`, emits about `202` samples, so `512` is a safe upper
/// bound with ample headroom.
pub const MAX_SAMPLES: usize = 512;

/// The portable core-`WGSL` Xiaolin Wu line kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`rasterize`](prism_render_architecture::particle::wu_line::rasterize) branch
/// for branch; see the module documentation for the algorithm.
const WU_LINE_WGSL: &str = r#"
// Xiaolin Wu anti-aliased line twin: one thread rasterizes one segment into a
// fixed-capacity slot of (x, y, coverage) samples, mirroring the CPU golden
// `rasterize` branch for branch with only floor/abs/clamp and +-*/.

// Coverage at or below this magnitude is treated as "nothing painted" and the
// sample is dropped, matching the reference COVERAGE_EPS.
const COVERAGE_EPS: f32 = 1.0e-6;

// Span (after axis canonicalization) at or below this magnitude marks a
// degenerate, coincident-endpoint segment; the gradient is then 1.0, matching
// the reference DEGENERATE_EPS.
const DEGENERATE_EPS: f32 = 1.0e-6;

// Fixed capacity of one line's sample slot. Must match the host MAX_SAMPLES.
const MAX_SAMPLES: u32 = 512u;

struct Params {
    // Number of lines in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Line {
    // Endpoints packed as (x0, y0, x1, y1).
    endpoints: vec4<f32>,
}

struct Pixel {
    x: i32,
    y: i32,
    coverage: f32,
}

struct WuLine {
    count: u32,
    pixels: array<Pixel, 512>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lines: array<Line>;
@group(0) @binding(2) var<storage, read_write> results: array<WuLine>;

// Integer part of v as an f32 (its floor).
fn ipart(v: f32) -> f32 {
    return floor(v);
}

// Fractional part of v in [0, 1), computed as v - floor(v) so it is correct for
// negative inputs.
fn fpart(v: f32) -> f32 {
    return v - floor(v);
}

// Reverse fractional part, 1 - fpart(v), in (0, 1].
fn rfpart(v: f32) -> f32 {
    return 1.0 - (v - floor(v));
}

// Rounds v to the nearest integer (ties toward +inf), spelled with floor so no
// banned round/ceil is used.
fn round_nearest(v: f32) -> f32 {
    return floor(v + 0.5);
}

// Pushes one sample into results[idx], mapping the internal (major, minor) walk
// frame back into the caller's (x, y) frame and dropping near-zero coverage.
fn emit(idx: u32, n: ptr<function, u32>, steep: bool, major: i32, minor: i32, coverage: f32) {
    let c = clamp(coverage, 0.0, 1.0);
    if (c <= COVERAGE_EPS) {
        return;
    }
    var px: Pixel;
    if (steep) {
        px.x = minor;
        px.y = major;
    } else {
        px.x = major;
        px.y = minor;
    }
    px.coverage = c;
    let k = *n;
    if (k >= MAX_SAMPLES) {
        return;
    }
    results[idx].pixels[k] = px;
    *n = k + 1u;
}

// Accumulates one (minor, coverage) contribution into the four-lane same-column
// merge scratch, summing coverage when the minor coordinate repeats.
fn acc_add(
    minors: ptr<function, array<i32, 4>>,
    covs: ptr<function, array<f32, 4>>,
    m: ptr<function, u32>,
    minor: i32,
    coverage: f32,
) {
    let used = *m;
    for (var i: u32 = 0u; i < used; i = i + 1u) {
        if ((*minors)[i] == minor) {
            (*covs)[i] = (*covs)[i] + coverage;
            return;
        }
    }
    (*minors)[used] = minor;
    (*covs)[used] = coverage;
    *m = used + 1u;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let e = lines[idx].endpoints;
    var ax = e.x;
    var ay = e.y;
    var bx = e.z;
    var by = e.w;

    // Walk the longer axis: swap x/y when the line is steeper than 45 degrees.
    let steep = abs(by - ay) > abs(bx - ax);
    if (steep) {
        let t0 = ax;
        ax = ay;
        ay = t0;
        let t1 = bx;
        bx = by;
        by = t1;
    }
    // Always march left-to-right along the (now dominant) x axis.
    if (ax > bx) {
        let t0 = ax;
        ax = bx;
        bx = t0;
        let t1 = ay;
        ay = by;
        by = t1;
    }

    let dx = bx - ax;
    let dy = by - ay;
    // After canonicalization |dy| <= |dx|; only a coincident-endpoint segment
    // has a ~zero run, whose gradient is defined as 1.0 to keep the math finite.
    var gradient: f32;
    if (abs(dx) <= DEGENERATE_EPS) {
        gradient = 1.0;
    } else {
        gradient = dy / dx;
    }

    // First endpoint.
    let xend0 = round_nearest(ax);
    let yend0 = ay + gradient * (xend0 - ax);
    let xgap0 = rfpart(ax + 0.5);
    let xpxl1 = i32(xend0);
    let ypxl1 = i32(ipart(yend0));
    let cov0_near = rfpart(yend0) * xgap0;
    let cov0_far = fpart(yend0) * xgap0;

    // Second endpoint.
    let xend1 = round_nearest(bx);
    let yend1 = by + gradient * (xend1 - bx);
    let xgap1 = fpart(bx + 0.5);
    let xpxl2 = i32(xend1);
    let ypxl2 = i32(ipart(yend1));
    let cov1_near = rfpart(yend1) * xgap1;
    let cov1_far = fpart(yend1) * xgap1;

    var n: u32 = 0u;

    if (xpxl1 == xpxl2) {
        // The segment is shorter than a pixel along the major axis: both
        // endpoints land in the same column. Merge their (up to four) pixel
        // contributions by minor coordinate so no coordinate repeats.
        var minors: array<i32, 4>;
        var covs: array<f32, 4>;
        var m: u32 = 0u;
        acc_add(&minors, &covs, &m, ypxl1, cov0_near);
        acc_add(&minors, &covs, &m, ypxl1 + 1, cov0_far);
        acc_add(&minors, &covs, &m, ypxl2, cov1_near);
        acc_add(&minors, &covs, &m, ypxl2 + 1, cov1_far);
        for (var i: u32 = 0u; i < m; i = i + 1u) {
            emit(idx, &n, steep, xpxl1, minors[i], covs[i]);
        }
        results[idx].count = n;
        return;
    }

    // First endpoint column.
    emit(idx, &n, steep, xpxl1, ypxl1, cov0_near);
    emit(idx, &n, steep, xpxl1, ypxl1 + 1, cov0_far);

    // Interior: one major step per column, two anti-aliased pixels each.
    var intery = yend0 + gradient;
    var x = xpxl1 + 1;
    loop {
        if (x >= xpxl2) {
            break;
        }
        let base = i32(ipart(intery));
        emit(idx, &n, steep, x, base, rfpart(intery));
        emit(idx, &n, steep, x, base + 1, fpart(intery));
        intery = intery + gradient;
        x = x + 1;
    }

    // Second endpoint column.
    emit(idx, &n, steep, xpxl2, ypxl2, cov1_near);
    emit(idx, &n, steep, xpxl2, ypxl2 + 1, cov1_far);

    results[idx].count = n;
}
"#;

/// One line-rasterization query: the two floating-point endpoints the reference
/// [`rasterize`](prism_render_architecture::particle::wu_line::rasterize)
/// consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WuLineQuery {
    /// The `x` coordinate of the first endpoint.
    pub x0: f32,
    /// The `y` coordinate of the first endpoint.
    pub y0: f32,
    /// The `x` coordinate of the second endpoint.
    pub x1: f32,
    /// The `y` coordinate of the second endpoint.
    pub y1: f32,
}

impl WuLineQuery {
    /// Builds a query from the two endpoints `(x0, y0)` and `(x1, y1)`.
    #[must_use]
    pub const fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> WuLineQuery {
        WuLineQuery { x0, y0, x1, y1 }
    }
}

/// One anti-aliased pixel sample: an integer `(x, y)` grid cell and the fraction
/// of it the line paints, matching one `(x, y, coverage)` tuple the reference
/// [`rasterize`](prism_render_architecture::particle::wu_line::rasterize)
/// returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WuPixel {
    /// The `x` grid coordinate of the painted pixel.
    pub x: i32,
    /// The `y` grid coordinate of the painted pixel.
    pub y: i32,
    /// The fraction of the pixel the line paints, in `[0, 1]`.
    pub coverage: f32,
}

/// The rasterized samples for one input line, trimmed to the thread's reported
/// `count`, mirroring the ordered sample list the reference
/// [`rasterize`](prism_render_architecture::particle::wu_line::rasterize)
/// returns.
#[derive(Clone, Debug, PartialEq)]
pub struct WuLineResult {
    /// The emitted samples, in device-emission order.
    pub pixels: Vec<WuPixel>,
}

/// `repr(C)` `std430` layout of one line query: a single `vec4` holding
/// `(x0, y0, x1, y1)` — `16` bytes, exactly as the `WGSL` `Line` struct reads
/// it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLine {
    /// Endpoints packed as `(x0, y0, x1, y1)`.
    endpoints: [f32; 4],
}

impl GpuLine {
    /// Packs one query into its `std430` image.
    fn new(query: &WuLineQuery) -> GpuLine {
        GpuLine {
            endpoints: [query.x0, query.y0, query.x1, query.y1],
        }
    }
}

/// `repr(C)` `std430` layout of one sample: three `4`-byte scalars `(x, y,
/// coverage)` — `12` bytes with no padding, matching the `WGSL` `Pixel` struct
/// and its `12`-byte array stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPixel {
    /// The `x` grid coordinate of the painted pixel.
    x: i32,
    /// The `y` grid coordinate of the painted pixel.
    y: i32,
    /// The fraction of the pixel the line paints.
    coverage: f32,
}

/// `repr(C)` `std430` layout of one line's output slot: a `count` word followed
/// by a fixed [`MAX_SAMPLES`]-entry array of [`GpuPixel`] — `4 + 512 * 12 =
/// 6148` bytes with no padding, matching the `WGSL` `WuLine` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuWuLine {
    /// Number of valid samples written into `pixels`.
    count: u32,
    /// Fixed-capacity sample storage; only the first `count` entries are valid.
    pixels: [GpuPixel; MAX_SAMPLES],
}

/// Uniform parameters for one dispatch: the line count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of lines in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable Xiaolin Wu line-rasterization compute pipeline.
pub struct GpuWuLineRasterizer {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWuLineRasterizer {
    /// Compiles the Xiaolin Wu line kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWuLineRasterizer {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_wu_line"),
            source: ShaderSource::Wgsl(WU_LINE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_wu_line_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_wu_line_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_wu_line_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWuLineRasterizer {
            module,
            layout,
            pipeline,
        }
    }

    /// Rasterizes every line on-device and returns one [`WuLineResult`] per
    /// input, in order.
    ///
    /// Each result holds the same anti-aliased `(x, y, coverage)` samples the
    /// reference
    /// [`rasterize`](prism_render_architecture::particle::wu_line::rasterize)
    /// returns, with integer coordinates matched exactly and coverage matched to
    /// within the tolerance documented on this module. An empty input returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, lines: &[WuLineQuery]) -> Vec<WuLineResult> {
        if lines.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = lines.len();

        let packed: Vec<GpuLine> = lines.iter().map(GpuLine::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_wu_line_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuWuLine>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_wu_line_output"),
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
            label: Some("prism_volumetric_wu_line_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_wu_line_bind_group"),
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
            label: Some("prism_volumetric_wu_line_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_wu_line_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_wu_line_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per line, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuWuLine>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuWuLine`] slot into the public [`WuLineResult`],
/// trimming the fixed array to the thread's reported `count`.
fn decode_result(raw: &GpuWuLine) -> WuLineResult {
    let count = (raw.count as usize).min(MAX_SAMPLES);
    let pixels = raw.pixels[..count]
        .iter()
        .map(|p| WuPixel {
            x: p.x,
            y: p.y,
            coverage: p.coverage,
        })
        .collect();
    WuLineResult { pixels }
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
