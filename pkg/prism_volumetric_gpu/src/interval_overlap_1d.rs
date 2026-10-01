//! `wgpu` compute twin of the one-dimensional closed-interval algebra
//! ([`interval_overlap_1d`](prism_render_architecture::particle::interval_overlap_1d),
//! particle design §10, §14).
//!
//! Many particle subsystems reduce to reasoning about a single axis: a particle
//! lifetime window `[spawn, death]`, one axis of an `AABB`, or the overlap test
//! between two scheduled emitter bursts. The `CPU` golden
//! [`interval_overlap_1d`](prism_render_architecture::particle::interval_overlap_1d)
//! owns that per-axis algebra over the closed interval
//! [`Interval`](prism_render_architecture::particle::interval_overlap_1d::Interval)
//! `[min, max]`; [`GpuIntervalOverlap1d`] is the on-device twin that runs one
//! thread per query and reproduces every per-element predicate and set
//! operation the reference reports, so a passing real-device parity test is
//! direct evidence the ported kernel classifies the same emptiness, folds the
//! same `±inf` sentinels and returns the same bounds the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reproduces the twelve per-element answers the reference computes
//! on one interval `a`, one interval `b` and the per-query scalars `value`,
//! `amount` and `delta`:
//! [`is_empty`](prism_render_architecture::particle::interval_overlap_1d::Interval::is_empty),
//! [`contains`](prism_render_architecture::particle::interval_overlap_1d::Interval::contains),
//! [`contains_interval`](prism_render_architecture::particle::interval_overlap_1d::Interval::contains_interval),
//! [`overlaps`](prism_render_architecture::particle::interval_overlap_1d::Interval::overlaps),
//! [`length`](prism_render_architecture::particle::interval_overlap_1d::Interval::length),
//! [`center`](prism_render_architecture::particle::interval_overlap_1d::Interval::center),
//! [`clamp_value`](prism_render_architecture::particle::interval_overlap_1d::Interval::clamp_value),
//! [`intersect`](prism_render_architecture::particle::interval_overlap_1d::Interval::intersect),
//! [`hull`](prism_render_architecture::particle::interval_overlap_1d::Interval::hull),
//! [`gap`](prism_render_architecture::particle::interval_overlap_1d::Interval::gap),
//! [`expand`](prism_render_architecture::particle::interval_overlap_1d::Interval::expand)
//! and
//! [`translate`](prism_render_architecture::particle::interval_overlap_1d::Interval::translate).
//! The empty sentinel `[+inf, -inf]` and the all-covering `[-inf, +inf]` are
//! carried through verbatim: the kernel never normalizes the stored bounds, so
//! `is_empty` stays a single `min > max` ordering test and `intersect` /
//! `expand` / `translate` return the same `[+inf, -inf]` sentinel the reference
//! does.
//!
//! The reduction twins
//! [`merge_sorted`](prism_render_architecture::particle::interval_overlap_1d::merge_sorted)
//! and
//! [`total_covered_length`](prism_render_architecture::particle::interval_overlap_1d::total_covered_length)
//! depend on a sort and a prefix reduction (a multi-pass host orchestration),
//! so they are deliberately *not* ported to `WGSL`; they stay host-side on the
//! golden reference.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `select`, `+ - * /` and unsigned bit arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no `sqrt` and no optional device feature,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The two infinite
//! sentinels are reconstructed with `bitcast<f32>` of the `IEEE-754` `±inf` bit
//! patterns rather than a non-portable infinity literal.
//!
//! # Correctness model
//!
//! Every answer is a fixed, non-reorderable sequence of orderings, `min` / `max`
//! / `clamp` selections and at most one add, subtract and halving, so `CPU` and
//! `GPU` evaluate the same closed form in the same order. The boolean and
//! emptiness verdicts are driven purely by `f32` orderings on values that each
//! fixture keeps clear of a tie, so they match *exactly* and are read back as
//! `u32` `0` / `1`. The continuous bounds are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the finite
//! `f32` fields while comparing the infinite and `NaN` sentinels by exact
//! classification.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! [`interval_overlap_1d`](prism_render_architecture::particle::interval_overlap_1d)；
//! 无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::interval_overlap_1d::Interval;
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

/// Discrete code written by the kernel for a true boolean verdict; the host
/// decodes a lane as `true` with an exact `== 1` integer compare, never an
/// `f32` equality.
const CODE_TRUE: u32 = 1;

/// The portable core-`WGSL` interval-algebra kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`interval_overlap_1d`](prism_render_architecture::particle::interval_overlap_1d)
/// predicate for predicate and operation for operation; see the module
/// documentation for the algebra.
const INTERVAL_OVERLAP_1D_WGSL: &str = r#"
// 1D closed-interval algebra twin: one thread per query reproduces the twelve
// per-element predicates and set operations the reference computes on a closed
// interval [min, max]. It mirrors the CPU golden interval_overlap_1d guard for
// guard; the empty sentinel [+inf, -inf] and the all-covering [-inf, +inf] are
// carried through unnormalized so is_empty stays one ordering test.

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Interval a bounds, carried verbatim (an inverted pair encodes empty).
    a_min: f32,
    a_max: f32,
    // Interval b bounds, carried verbatim.
    b_min: f32,
    b_max: f32,
    // Scalar probed by contains and clamp_value.
    value: f32,
    // Signed grow amount for expand (negative shrinks and may collapse).
    amount: f32,
    // Signed shift for translate.
    delta: f32,
    pad0: f32,
}

struct Result {
    // Boolean predicates as 0u / 1u.
    is_empty: u32,
    contains: u32,
    contains_interval: u32,
    overlaps: u32,
    // Continuous scalar readings.
    length: f32,
    center: f32,
    clamp_value: f32,
    gap: f32,
    // Set-operation interval bounds.
    intersect_min: f32,
    intersect_max: f32,
    hull_min: f32,
    hull_max: f32,
    expand_min: f32,
    expand_max: f32,
    translate_min: f32,
    translate_max: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// A closed interval working value, kept separate from the packed Query layout.
struct Iv {
    lo: f32,
    hi: f32,
}

// The IEEE-754 positive infinity bit pattern; WGSL has no portable inf literal.
fn pos_inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// The IEEE-754 negative infinity bit pattern.
fn neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

// The canonical empty sentinel [+inf, -inf], matching Interval::empty.
fn iv_empty() -> Iv {
    return Iv(pos_inf(), neg_inf());
}

// Canonical constructor: swaps inverted bounds, matching Interval::new.
fn iv_new(lo: f32, hi: f32) -> Iv {
    if (lo > hi) {
        return Iv(hi, lo);
    }
    return Iv(lo, hi);
}

// True exactly when min > max, which only the empty sentinel satisfies.
fn iv_is_empty(a: Iv) -> bool {
    return a.lo > a.hi;
}

// True when the scalar v lies in the closed interval; an empty interval holds
// nothing, which falls out of the min <= max invariant.
fn iv_contains(a: Iv, v: f32) -> bool {
    return a.lo <= v && v <= a.hi;
}

// True when b is wholly contained in a; an empty b is a subset of everything.
fn iv_contains_interval(a: Iv, b: Iv) -> bool {
    if (iv_is_empty(b)) {
        return true;
    }
    if (iv_is_empty(a)) {
        return false;
    }
    return a.lo <= b.lo && b.hi <= a.hi;
}

// True when the two intervals share at least one point; touching at a single
// endpoint counts as overlap, and either being empty yields false.
fn iv_overlaps(a: Iv, b: Iv) -> bool {
    if (iv_is_empty(a) || iv_is_empty(b)) {
        return false;
    }
    return a.lo <= b.hi && b.lo <= a.hi;
}

// The length max - min, clamped non-negative; the empty interval reports 0.
fn iv_length(a: Iv) -> f32 {
    if (iv_is_empty(a)) {
        return 0.0;
    }
    return max(a.hi - a.lo, 0.0);
}

// The midpoint (min + max) * 0.5, computed unconditionally as the reference does
// (an empty or all-covering interval therefore yields a NaN midpoint).
fn iv_center(a: Iv) -> f32 {
    return (a.lo + a.hi) * 0.5;
}

// v clamped into the interval; an empty interval returns its min sentinel so the
// clamp is never evaluated on inverted bounds, matching Interval::clamp_value.
fn iv_clamp_value(a: Iv, v: f32) -> f32 {
    if (iv_is_empty(a)) {
        return a.lo;
    }
    return clamp(v, a.lo, a.hi);
}

// The intersection; disjoint or empty operands return the empty sentinel.
fn iv_intersect(a: Iv, b: Iv) -> Iv {
    if (iv_is_empty(a) || iv_is_empty(b)) {
        return iv_empty();
    }
    let lo = max(a.lo, b.lo);
    let hi = min(a.hi, b.hi);
    if (lo > hi) {
        return iv_empty();
    }
    return Iv(lo, hi);
}

// The smallest interval enclosing both operands; an empty operand contributes
// nothing.
fn iv_hull(a: Iv, b: Iv) -> Iv {
    if (iv_is_empty(a)) {
        return b;
    }
    if (iv_is_empty(b)) {
        return a;
    }
    return Iv(min(a.lo, b.lo), max(a.hi, b.hi));
}

// The empty gap between the two intervals; overlapping, touching or empty
// operands report 0.
fn iv_gap(a: Iv, b: Iv) -> f32 {
    if (iv_is_empty(a) || iv_is_empty(b)) {
        return 0.0;
    }
    if (a.hi < b.lo) {
        return b.lo - a.hi;
    }
    if (b.hi < a.lo) {
        return a.lo - b.hi;
    }
    return 0.0;
}

// The interval grown by amount on both sides; a negative amount can over-shrink,
// in which case new() collapses the inverted bounds to their midpoint rather
// than letting the interval flip. The empty interval stays empty.
fn iv_expand(a: Iv, amount: f32) -> Iv {
    if (iv_is_empty(a)) {
        return iv_empty();
    }
    let lo = a.lo - amount;
    let hi = a.hi + amount;
    if (lo > hi) {
        let mid = (lo + hi) * 0.5;
        return iv_new(mid, mid);
    }
    return iv_new(lo, hi);
}

// The interval shifted by delta; the empty interval stays empty.
fn iv_translate(a: Iv, delta: f32) -> Iv {
    if (iv_is_empty(a)) {
        return iv_empty();
    }
    return Iv(a.lo + delta, a.hi + delta);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    // Bounds are loaded verbatim, never re-normalized, so an inverted pair stays
    // empty exactly as the reference stores it.
    let a = Iv(q.a_min, q.a_max);
    let b = Iv(q.b_min, q.b_max);

    let inter = iv_intersect(a, b);
    let hl = iv_hull(a, b);
    let ex = iv_expand(a, q.amount);
    let tr = iv_translate(a, q.delta);

    var out: Result;
    out.is_empty = select(0u, 1u, iv_is_empty(a));
    out.contains = select(0u, 1u, iv_contains(a, q.value));
    out.contains_interval = select(0u, 1u, iv_contains_interval(a, b));
    out.overlaps = select(0u, 1u, iv_overlaps(a, b));
    out.length = iv_length(a);
    out.center = iv_center(a);
    out.clamp_value = iv_clamp_value(a, q.value);
    out.gap = iv_gap(a, b);
    out.intersect_min = inter.lo;
    out.intersect_max = inter.hi;
    out.hull_min = hl.lo;
    out.hull_max = hl.hi;
    out.expand_min = ex.lo;
    out.expand_max = ex.hi;
    out.translate_min = tr.lo;
    out.translate_max = tr.hi;
    results[idx] = out;
}
"#;

/// One interval-algebra query: two intervals plus the per-query scalars the
/// twinned functions consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalOverlapQuery {
    /// The primary interval `a`, the receiver of every unary query and the first
    /// operand of each binary one.
    pub a: Interval,
    /// The secondary interval `b`, the second operand of `contains_interval`,
    /// `overlaps`, `intersect`, `hull` and `gap`.
    pub b: Interval,
    /// Scalar probed by `contains` and clamped by `clamp_value`.
    pub value: f32,
    /// Signed grow amount for `expand`; a negative value shrinks `a`.
    pub amount: f32,
    /// Signed shift for `translate`.
    pub delta: f32,
}

impl IntervalOverlapQuery {
    /// Builds a query from two intervals and the probe scalar, grow amount and
    /// shift the twinned functions consume.
    #[must_use]
    pub const fn new(
        a: Interval,
        b: Interval,
        value: f32,
        amount: f32,
        delta: f32,
    ) -> IntervalOverlapQuery {
        IntervalOverlapQuery {
            a,
            b,
            value,
            amount,
            delta,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its twelve per-element functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalOverlapResult {
    /// Whether `a` is empty, matching `Interval::is_empty`.
    pub is_empty: bool,
    /// Whether `a` contains `value`, matching `Interval::contains`.
    pub contains: bool,
    /// Whether `a` wholly contains `b`, matching `Interval::contains_interval`.
    pub contains_interval: bool,
    /// Whether `a` and `b` share a point, matching `Interval::overlaps`.
    pub overlaps: bool,
    /// The length of `a`, matching `Interval::length`.
    pub length: f32,
    /// The midpoint of `a`, matching `Interval::center`.
    pub center: f32,
    /// `value` clamped into `a`, matching `Interval::clamp_value`.
    pub clamp_value: f32,
    /// The gap between `a` and `b`, matching `Interval::gap`.
    pub gap: f32,
    /// The intersection of `a` and `b`, matching `Interval::intersect`.
    pub intersect: Interval,
    /// The hull of `a` and `b`, matching `Interval::hull`.
    pub hull: Interval,
    /// `a` grown by `amount`, matching `Interval::expand`.
    pub expand: Interval,
    /// `a` shifted by `delta`, matching `Interval::translate`.
    pub translate: Interval,
}

/// `repr(C)` `std430` layout of one packed query: the four interval bounds then
/// the `value`, `amount`, `delta` scalars and one pad word — `32` bytes, each
/// scalar on its natural `4`-byte slot exactly as the `WGSL` `Query` struct
/// reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Interval `a` lower bound.
    a_min: f32,
    /// Interval `a` upper bound.
    a_max: f32,
    /// Interval `b` lower bound.
    b_min: f32,
    /// Interval `b` upper bound.
    b_max: f32,
    /// Scalar probed by `contains` and `clamp_value`.
    value: f32,
    /// Signed grow amount for `expand`.
    amount: f32,
    /// Signed shift for `translate`.
    delta: f32,
    /// Padding word rounding the struct to a `16`-byte-aligned `32` bytes.
    pad0: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image, carrying the interval bounds
    /// verbatim so an empty sentinel stays inverted on the device.
    fn new(query: &IntervalOverlapQuery) -> GpuQuery {
        GpuQuery {
            a_min: query.a.min,
            a_max: query.a.max,
            b_min: query.b.min,
            b_max: query.b.max,
            value: query.value,
            amount: query.amount,
            delta: query.delta,
            pad0: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four `u32` predicate flags, four
/// continuous scalars and eight interval-bound scalars — `64` bytes matching the
/// `WGSL` `Result` struct field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `is_empty` flag as `0` / `1`.
    is_empty: u32,
    /// `contains` flag as `0` / `1`.
    contains: u32,
    /// `contains_interval` flag as `0` / `1`.
    contains_interval: u32,
    /// `overlaps` flag as `0` / `1`.
    overlaps: u32,
    /// `length` of `a`.
    length: f32,
    /// `center` of `a`.
    center: f32,
    /// `clamp_value` of `value` into `a`.
    clamp_value: f32,
    /// `gap` between `a` and `b`.
    gap: f32,
    /// `intersect` lower bound.
    intersect_min: f32,
    /// `intersect` upper bound.
    intersect_max: f32,
    /// `hull` lower bound.
    hull_min: f32,
    /// `hull` upper bound.
    hull_max: f32,
    /// `expand` lower bound.
    expand_min: f32,
    /// `expand` upper bound.
    expand_max: f32,
    /// `translate` lower bound.
    translate_min: f32,
    /// `translate` upper bound.
    translate_max: f32,
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

/// A compiled, reusable interval-algebra compute pipeline.
pub struct GpuIntervalOverlap1d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIntervalOverlap1d {
    /// Compiles the interval-algebra kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIntervalOverlap1d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d"),
            source: ShaderSource::Wgsl(INTERVAL_OVERLAP_1D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIntervalOverlap1d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`IntervalOverlapResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers across the twelve twinned
    /// `Interval` functions: the four predicates match exactly, the finite
    /// continuous fields to within the tolerance documented on this module, and
    /// the `±inf` / `NaN` sentinels by exact classification. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[IntervalOverlapQuery],
    ) -> Vec<IntervalOverlapResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_output"),
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
            label: Some("prism_volumetric_interval_overlap_1d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_bind_group"),
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
            label: Some("prism_volumetric_interval_overlap_1d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_interval_overlap_1d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_interval_overlap_1d_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}

/// Decodes one packed [`GpuResult`] into the public [`IntervalOverlapResult`],
/// turning the `u32` flags back into `bool` with an exact `== CODE_TRUE` compare
/// and rebuilding each [`Interval`] from its bound pair.
fn decode_result(raw: &GpuResult) -> IntervalOverlapResult {
    IntervalOverlapResult {
        is_empty: raw.is_empty == CODE_TRUE,
        contains: raw.contains == CODE_TRUE,
        contains_interval: raw.contains_interval == CODE_TRUE,
        overlaps: raw.overlaps == CODE_TRUE,
        length: raw.length,
        center: raw.center,
        clamp_value: raw.clamp_value,
        gap: raw.gap,
        intersect: Interval {
            min: raw.intersect_min,
            max: raw.intersect_max,
        },
        hull: Interval {
            min: raw.hull_min,
            max: raw.hull_max,
        },
        expand: Interval {
            min: raw.expand_min,
            max: raw.expand_max,
        },
        translate: Interval {
            min: raw.translate_min,
            max: raw.translate_max,
        },
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
