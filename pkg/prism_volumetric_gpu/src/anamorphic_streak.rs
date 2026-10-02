//! `wgpu` compute twin of the anamorphic lens-streak highlight-extraction and
//! directional-smear contract
//! ([`anamorphic_streak`](prism_render_architecture::particle::anamorphic_streak),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`anamorphic_streak`](prism_render_architecture::particle::anamorphic_streak)
//! owns the small, verifiable maths an anamorphic bloom streak needs: the
//! `Rec. 709` perceptual luminance of a linear `RGB` triple
//! ([`luminance`](prism_render_architecture::particle::anamorphic_streak::luminance)),
//! the parameter normalization a streak is built from
//! ([`StreakParams::new`](prism_render_architecture::particle::anamorphic_streak::StreakParams::new)),
//! the soft-`threshold` highlight weight
//! ([`StreakParams::threshold_weight`](prism_render_architecture::particle::anamorphic_streak::StreakParams::threshold_weight)),
//! the one-dimensional tap offsets whose spacing doubles each level
//! ([`StreakParams::streak_tap_offsets`](prism_render_architecture::particle::anamorphic_streak::StreakParams::streak_tap_offsets)),
//! and the geometric weighted accumulation of the sampled taps
//! ([`StreakParams::accumulate_streak`](prism_render_architecture::particle::anamorphic_streak::StreakParams::accumulate_streak)).
//! [`GpuAnamorphicStreak`] is the on-device twin: one thread solves one query,
//! so a passing real-device parity test is direct evidence the ported kernel
//! normalizes the same direction, extracts the same highlight weight, walks the
//! same tap chain and accumulates the same streak color the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for a batch of
//! independent queries: the unit-normalized `direction` (with the degenerate
//! zero-direction `+x` fallback) and the forced-at-least-one `tap_count` from
//! [`StreakParams::new`](prism_render_architecture::particle::anamorphic_streak::StreakParams::new),
//! the soft-`threshold` weight for a supplied luminance, the tap offsets of the
//! one-dimensional chain, the accumulated streak color for a supplied tap set,
//! and the standalone `Rec. 709` luminance of a supplied color. Because the
//! `tap_count` upper bound is a host-known constant, the kernel reports the
//! offsets into a fixed-length array (`MAX_TAPS` slots, the first `tap_count`
//! valid) and accumulates a fixed-length tap array (the first `accum_count`
//! valid).
//!
//! # Correctness model
//!
//! The `tap_count` and the array lengths are discrete classifications built from
//! integer arithmetic, so `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on them. The direction, the weight, the offsets, the
//! accumulation and the luminance thread through multiplies, adds, one guarded
//! division and one `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance (`abs_diff <=
//! 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every continuous
//! quantity.
//!
//! # Degenerate inputs
//!
//! A zero-length `direction` would divide by a near-zero magnitude; the kernel
//! checks the squared length against [`MIN_DENOM`] and falls back to the `+x`
//! axis instead, matching the reference. A zero `knee` collapses the soft band
//! to a hard cutoff at the `threshold` (step from `0` to `1`), reproduced by the
//! same `span <= MIN_DENOM` guard. An empty query batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /`, unsigned index arithmetic and one `sqrt` for the
//! unit-normalization — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no
//! inverse trigonometry, no `smoothstep` intrinsic (the soft band is the
//! hand-expanded `3t^2 - 2t^3` polynomial `t*t*(3-2t)`) and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The two
//! bounded loops run a fixed `MAX_TAPS` iterations, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::anamorphic_streak`；无第三方引擎源码或衍生代码。
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

/// Fixed upper bound on the number of taps the kernel reports and accumulates.
/// The reference `tap_count` is host-known, so the twin uses a fixed-length
/// array sized to the largest `tap_count` the parity suite exercises; the first
/// `tap_count` (or `accum_count`) slots are valid and the rest are zero.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::anamorphic_streak`；无第三方引擎源码或衍生代码。
pub const MAX_TAPS: usize = 8;

/// The portable core-`WGSL` anamorphic-streak kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`anamorphic_streak`](prism_render_architecture::particle::anamorphic_streak)
/// branch for branch; see the module documentation for the algorithm.
const ANAMORPHIC_STREAK_WGSL: &str = r#"
// Anamorphic-streak twin: one thread per query reproduces the Rec. 709
// luminance, the StreakParams::new normalization (unit direction with the zero
// fallback, clamped scalars, forced-at-least-one tap_count), the soft-threshold
// weight, the doubling tap-offset chain, and the geometric streak accumulation.
// It mirrors the CPU golden `particle::anamorphic_streak` branch for branch,
// uses only the portable core-WGSL subset (min/max/clamp/abs, + - * / and one
// sqrt for the normalization, plus unsigned index math), uses no smoothstep
// intrinsic (the soft band is the hand-expanded 3t^2 - 2t^3 polynomial
// `t*t*(3-2t)`) and no transcendental call, and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12. The two loops run a fixed MAX_TAPS
// iterations, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::anamorphic_streak；无第三方
// 引擎源码或衍生代码。

// Magnitude below which a squared direction length or a soft-band span is
// treated as zero. Matches the reference `MIN_DENOM`; the compare rule used
// instead of an f32 `==`.
const MIN_DENOM: f32 = 1.0e-6;

// Fixed tap-array upper bound; matches the host `MAX_TAPS`.
const MAX_TAPS: u32 = 8u;

// Rec. 709 luminance weights.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Raw (pre-normalization) lens axis fed to StreakParams::new.
    direction: vec2<f32>,
    // Raw non-negative scalars (clamped by the twin like StreakParams::new).
    threshold: f32,
    knee: f32,
    intensity: f32,
    stretch: f32,
    // Luminance fed to threshold_weight.
    lum: f32,
    // Raw tap_count (forced to at least one) and the count of taps to
    // accumulate (independent of tap_count, like the golden's slice length).
    tap_count: u32,
    accum_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Per-streak chromatic tint (vec4 aligned).
    tint: vec3<f32>,
    pad_tint: f32,
    // Color fed to the standalone luminance function (vec4 aligned).
    color: vec3<f32>,
    pad_color: f32,
    // Fixed-length RGB taps for accumulate (rgb in xyz; w unused).
    taps: array<vec4<f32>, 8>,
}

struct Result {
    // Unit-normalized direction.
    direction: vec2<f32>,
    // Forced-at-least-one tap_count.
    tap_count: u32,
    // Soft-threshold weight for `lum`.
    threshold_weight: f32,
    // Accumulated streak color.
    accumulate: vec3<f32>,
    // Rec. 709 luminance of `color`.
    luminance: f32,
    // Tap offsets (xy per slot; the first tap_count valid, rest zero).
    tap_offsets: array<vec4<f32>, 8>,
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

    // StreakParams::new: unit-normalize the direction, falling back to the +x
    // axis for a degenerate zero direction.
    let len_sq = q.direction.x * q.direction.x + q.direction.y * q.direction.y;
    var dir: vec2<f32> = vec2<f32>(1.0, 0.0);
    if (len_sq > MIN_DENOM) {
        let inv = 1.0 / sqrt(len_sq);
        dir = vec2<f32>(q.direction.x * inv, q.direction.y * inv);
    }
    // Clamp the non-negative scalars and force at least one tap, like `new`.
    let tap_count = max(q.tap_count, 1u);
    let stretch = max(q.stretch, 0.0);
    let threshold = max(q.threshold, 0.0);
    let knee = max(q.knee, 0.0);
    let intensity = max(q.intensity, 0.0);

    // luminance: hand-rolled Rec. 709 dot product.
    let lum_out = q.color.x * LUMA_R + q.color.y * LUMA_G + q.color.z * LUMA_B;

    // threshold_weight: a zero `knee` degenerates to a hard cutoff at the
    // threshold; otherwise the hand-expanded smoothstep polynomial 3t^2 - 2t^3.
    let lo = threshold - knee;
    let span = 2.0 * knee;
    var weight: f32 = 0.0;
    if (span <= MIN_DENOM) {
        if (q.lum >= threshold) {
            weight = 1.0;
        } else {
            weight = 0.0;
        }
    } else {
        let t = clamp((q.lum - lo) / span, 0.0, 1.0);
        weight = t * t * (3.0 - 2.0 * t);
    }

    // streak_tap_offsets: tap i sits at direction * stretch * 2^i; spacing
    // doubles each level. Slots past tap_count stay zero.
    var out: Result;
    var spacing = stretch;
    for (var i: u32 = 0u; i < MAX_TAPS; i = i + 1u) {
        if (i < tap_count) {
            out.tap_offsets[i] = vec4<f32>(dir.x * spacing, dir.y * spacing, 0.0, 0.0);
            spacing = spacing * 2.0;
        } else {
            out.tap_offsets[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    }

    // accumulate_streak: geometric weighted sum (each tap half the previous)
    // scaled by the tint and the intensity. An empty tap set stays black.
    var acc: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var tap_weight: f32 = 1.0;
    for (var i: u32 = 0u; i < MAX_TAPS; i = i + 1u) {
        if (i < q.accum_count) {
            acc = acc + q.taps[i].xyz * tap_weight;
            tap_weight = tap_weight * 0.5;
        }
    }
    let accum = vec3<f32>(
        acc.x * q.tint.x * intensity,
        acc.y * q.tint.y * intensity,
        acc.z * q.tint.z * intensity,
    );

    out.direction = dir;
    out.tap_count = tap_count;
    out.threshold_weight = weight;
    out.accumulate = accum;
    out.luminance = lum_out;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`ANAMORPHIC_STREAK_WGSL`].
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
/// Each `vec3` lane carries a trailing pad word and the integer tail is padded
/// so the `tint` `vec3` lands on a `16`-byte boundary on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Raw lens axis.
    direction: [f32; 2],
    /// Raw `threshold`.
    threshold: f32,
    /// Raw `knee`.
    knee: f32,
    /// Raw `intensity`.
    intensity: f32,
    /// Raw base tap `stretch`.
    stretch: f32,
    /// Luminance fed to `threshold_weight`.
    lum: f32,
    /// Raw tap count.
    tap_count: u32,
    /// Number of taps to accumulate.
    accum_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Per-streak `tint`.
    tint: [f32; 3],
    /// Pad lane after `tint`.
    pad_tint: f32,
    /// Color fed to `luminance`.
    color: [f32; 3],
    /// Pad lane after `color`.
    pad_color: f32,
    /// Fixed-length `RGB` taps (`rgb` in `xyz`).
    taps: [[f32; 4]; MAX_TAPS],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Unit-normalized `direction`.
    direction: [f32; 2],
    /// Forced-at-least-one `tap_count`.
    tap_count: u32,
    /// Soft-`threshold` weight.
    threshold_weight: f32,
    /// Accumulated streak color.
    accumulate: [f32; 3],
    /// `Rec. 709` luminance.
    luminance: f32,
    /// Tap offsets (`xy` per slot).
    tap_offsets: [[f32; 4]; MAX_TAPS],
}

/// One query for the anamorphic-streak twin: the raw constructor inputs plus the
/// standalone-function inputs, so a single query exercises every twinned
/// function at once.
///
/// `direction`, `tap_count`, `stretch`, `tint`, `threshold`, `knee` and
/// `intensity` are the raw
/// [`StreakParams::new`](prism_render_architecture::particle::anamorphic_streak::StreakParams::new)
/// inputs (the twin applies the same normalization and clamping). `lum` is the
/// luminance fed to
/// [`StreakParams::threshold_weight`](prism_render_architecture::particle::anamorphic_streak::StreakParams::threshold_weight);
/// `color` is the triple fed to
/// [`luminance`](prism_render_architecture::particle::anamorphic_streak::luminance);
/// and the first `accum_count` entries of `taps` are the samples fed to
/// [`StreakParams::accumulate_streak`](prism_render_architecture::particle::anamorphic_streak::StreakParams::accumulate_streak).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnamorphicStreakQuery {
    /// Raw lens axis for `StreakParams::new`.
    pub direction: [f32; 2],
    /// Raw tap count (forced to at least one).
    pub tap_count: u32,
    /// Raw base tap `stretch`.
    pub stretch: f32,
    /// Per-streak chromatic `tint`.
    pub tint: [f32; 3],
    /// Raw soft-`threshold` center.
    pub threshold: f32,
    /// Raw soft-band half-width `knee`.
    pub knee: f32,
    /// Raw global `intensity` gain.
    pub intensity: f32,
    /// Luminance fed to `threshold_weight`.
    pub lum: f32,
    /// Color fed to the standalone `luminance` function.
    pub color: [f32; 3],
    /// Fixed-length `RGB` taps; the first `accum_count` are accumulated.
    pub taps: [[f32; 3]; MAX_TAPS],
    /// Number of valid taps to accumulate (`0..=MAX_TAPS`).
    pub accum_count: u32,
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnamorphicStreakResult {
    /// Unit-normalized `direction`, matching the field
    /// [`StreakParams::new`](prism_render_architecture::particle::anamorphic_streak::StreakParams::new)
    /// stores.
    pub direction: [f32; 2],
    /// Forced-at-least-one `tap_count`, matching
    /// [`StreakParams::new`](prism_render_architecture::particle::anamorphic_streak::StreakParams::new).
    pub tap_count: u32,
    /// Soft-`threshold` weight for `lum`, matching
    /// [`StreakParams::threshold_weight`](prism_render_architecture::particle::anamorphic_streak::StreakParams::threshold_weight).
    pub threshold_weight: f32,
    /// Tap offsets; the first `tap_count` are valid and match
    /// [`StreakParams::streak_tap_offsets`](prism_render_architecture::particle::anamorphic_streak::StreakParams::streak_tap_offsets),
    /// the rest are zero.
    pub tap_offsets: [[f32; 2]; MAX_TAPS],
    /// Accumulated streak color, matching
    /// [`StreakParams::accumulate_streak`](prism_render_architecture::particle::anamorphic_streak::StreakParams::accumulate_streak).
    pub accumulate: [f32; 3],
    /// `Rec. 709` luminance of `color`, matching
    /// [`luminance`](prism_render_architecture::particle::anamorphic_streak::luminance).
    pub luminance: f32,
}

/// Encodes one [`AnamorphicStreakQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &AnamorphicStreakQuery) -> GpuQuery {
    let mut taps = [[0.0f32; 4]; MAX_TAPS];
    for (slot, tap) in taps.iter_mut().zip(q.taps.iter()) {
        *slot = [tap[0], tap[1], tap[2], 0.0];
    }
    GpuQuery {
        direction: q.direction,
        threshold: q.threshold,
        knee: q.knee,
        intensity: q.intensity,
        stretch: q.stretch,
        lum: q.lum,
        tap_count: q.tap_count,
        accum_count: q.accum_count,
        pad0: 0,
        pad1: 0,
        pad2: 0,
        tint: q.tint,
        pad_tint: 0.0,
        color: q.color,
        pad_color: 0.0,
        taps,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`AnamorphicStreakResult`],
/// dropping the padding lane of each tap offset.
fn decode_result(raw: &GpuResult) -> AnamorphicStreakResult {
    let mut tap_offsets = [[0.0f32; 2]; MAX_TAPS];
    for (slot, off) in tap_offsets.iter_mut().zip(raw.tap_offsets.iter()) {
        *slot = [off[0], off[1]];
    }
    AnamorphicStreakResult {
        direction: raw.direction,
        tap_count: raw.tap_count,
        threshold_weight: raw.threshold_weight,
        tap_offsets,
        accumulate: raw.accumulate,
        luminance: raw.luminance,
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

/// A compiled, reusable anamorphic-streak compute pipeline, twinning the `CPU`
/// golden
/// [`anamorphic_streak`](prism_render_architecture::particle::anamorphic_streak).
pub struct GpuAnamorphicStreak {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAnamorphicStreak {
    /// Compiles the anamorphic-streak kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAnamorphicStreak {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_anamorphic_streak"),
            source: ShaderSource::Wgsl(ANAMORPHIC_STREAK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAnamorphicStreak {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`AnamorphicStreakResult`]
    /// per input, in order.
    ///
    /// The `tap_count` and the discrete array lengths equal the reference
    /// exactly; the direction, weight, offsets, accumulation and luminance match
    /// to within the tolerance documented on this module. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[AnamorphicStreakQuery],
    ) -> Vec<AnamorphicStreakResult> {
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
            label: Some("prism_volumetric_anamorphic_streak_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_bind_group"),
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
            label: Some("prism_volumetric_anamorphic_streak_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_anamorphic_streak_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_anamorphic_streak_pass"),
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

        raw.iter().map(decode_result).collect()
    }
}
