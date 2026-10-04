//! `wgpu` compute twin of the hair reactive-mask pixel reactivity and
//! blue-noise dither from the `CPU` golden
//! `prism_render_architecture::hair::reactive_mask`.
//!
//! Thin hair fibres are the worst case for temporal anti-aliasing: a single
//! strand covers only a fraction of a pixel, so its sub-pixel coverage flickers
//! between frames and the history buffer smears those flickers into ghosts. The
//! reference policy emits, per pixel, a `reactivity` in `[0, 1]` that leans on
//! the current frame where ghosting is most likely — low coverage, high
//! screen-space velocity, large depth change — and a stochastic
//! `blue-noise`-style dither threshold with a hard draw decision for the
//! thinnest fibres. This module ports those stateless, no-`RNG` closed forms
//! onto the device: one thread resolves one pixel query.
//!
//! [`GpuHairReactiveMask`] is the on-device twin, so a passing real-device
//! parity test is direct evidence the ported kernel takes the same sanitise /
//! saturation-ramp / integer-hash / threshold branch the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces three closed forms. `pixel_reactivity`
//! sanitises the weights and inputs, forms the `1 - coverage` term and the two
//! rational saturation ramps `v / (v + k)` (with `VELOCITY_SATURATION = 2` and
//! `DEPTH_SATURATION = 0.1`), takes the weighted sum and clamps it to
//! `[0, max_reactivity]`. `dither_threshold` mixes `(x, y, frame)` with large
//! odd constants through an integer xorshift / multiply finaliser and
//! normalises by `2^32` to land in `[0, 1)`. `dither_alpha` sanitises a
//! coverage and a threshold and returns the hard `coverage >= threshold` draw
//! decision. There is no loop: each thread performs a fixed, bounded sequence,
//! so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The reactivity path threads through multiplies, adds and guarded divisions,
//! so `CPU` and `GPU` are not necessarily bit-exact (a `GPU` may fuse a
//! multiply-add). The dither threshold is pure integer arithmetic followed by a
//! single `u32 -> f32` round-to-nearest-even conversion, so both sides agree to
//! the last bit. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous channel; the discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! Non-finite weights or magnitudes collapse to `0`, a non-finite coverage is
//! treated as fully covered (`1`) for reactivity and as `0` for the dither
//! draw, and a non-finite threshold is treated as `1`. These are detected with
//! an exponent-bit test on the `u32` reinterpretation of the float, matching
//! the reference `is_finite` branch without any floating-point equality. Every
//! query produces a finite, in-range result, so `valid` is always `1`; the
//! field is kept so the result occupies one `16`-byte `std430` slot. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `select`, `clamp`,
//! `abs`, `+ - * /`, `bitcast`, `u32` wrapping arithmetic and ordered `f32`
//! comparisons — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no
//! `f32` modulo and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::reactive_mask`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` hair reactive-mask kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `pixel_reactivity`, `dither_threshold` and `dither_alpha`
/// branch for branch; see the module documentation for the algorithm.
const HAIR_REACTIVE_MASK_WGSL: &str = r#"
// Hair reactive-mask twin: one thread per query reproduces pixel_reactivity,
// dither_threshold and dither_alpha. It mirrors the CPU golden branch for
// branch, uses only the portable core-WGSL subset (select, clamp, abs, bitcast,
// u32 wrapping arithmetic and ordered f32 comparisons), takes no optional
// feature, and has no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::reactive_mask；无第三方引擎源码
// 或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Weighting / clamping parameters for pixel_reactivity.
    coverage_weight: f32,
    velocity_weight: f32,
    depth_weight: f32,
    max_reactivity: f32,
    // Per-pixel reactivity inputs.
    coverage: f32,
    screen_velocity: f32,
    depth_delta: f32,
    // Sub-pixel coverage fed to the dither draw decision.
    dither_coverage: f32,
    // Pixel coordinate and frame index for the dither threshold hash.
    x: u32,
    y: u32,
    frame: u32,
    pad0: u32,
}

struct Result {
    // Reactive-mask value in [0, max_reactivity].
    reactivity: f32,
    // Blue-noise dither threshold in [0, 1).
    threshold: f32,
    // Hard dither draw decision in {0, 1}.
    dither_alpha: f32,
    // 1 when the query produced a finite, in-range result (always, here).
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Saturation constant (pixels per frame) for the velocity ramp v / (v + k).
const VELOCITY_SATURATION: f32 = 2.0;
// Saturation constant for the depth-delta ramp d / (d + k).
const DEPTH_SATURATION: f32 = 0.1;
// IEEE-754 single-precision exponent mask; an all-ones exponent is inf/nan.
const EXP_MASK: u32 = 0x7F800000u;
// 2^32, the normaliser that keeps the dither threshold strictly below one.
const TWO_POW_32: f32 = 4294967296.0;

// True when x is finite: a non-all-ones exponent field, matching the reference
// is_finite branch without any floating-point equality test.
fn is_finite_f32(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & EXP_MASK) != EXP_MASK;
}

// A finite, non-negative weight: non-finite or non-positive collapses to zero.
fn sanitize_weight(w: f32) -> f32 {
    return select(0.0, w, is_finite_f32(w) && (w > 0.0));
}

// Sub-pixel coverage clamped to [0, 1]; non-finite is treated as fully covered.
fn sanitize_coverage(c: f32) -> f32 {
    return select(1.0, clamp(c, 0.0, 1.0), is_finite_f32(c));
}

// A finite, non-negative magnitude: non-finite -> 0, otherwise the abs value.
fn sanitize_magnitude(v: f32) -> f32 {
    return select(0.0, abs(v), is_finite_f32(v));
}

// A rational saturation ramp v / (v + k) with v >= 0 and k > 0, so no divide by
// zero and the result lies in [0, 1).
fn saturation_ramp(v: f32, k: f32) -> f32 {
    return v / (v + k);
}

// The reactive-mask value for one pixel: a weighted sum of the three monotone
// terms, clamped to [0, max_reactivity], with all inputs sanitised first.
fn pixel_reactivity(q: Query) -> f32 {
    let coverage_weight = sanitize_weight(q.coverage_weight);
    let velocity_weight = sanitize_weight(q.velocity_weight);
    let depth_weight = sanitize_weight(q.depth_weight);
    let max_reactivity = select(1.0, clamp(q.max_reactivity, 0.0, 1.0), is_finite_f32(q.max_reactivity));

    let coverage = sanitize_coverage(q.coverage);
    let velocity = sanitize_magnitude(q.screen_velocity);
    let depth = sanitize_magnitude(q.depth_delta);

    let coverage_term = 1.0 - coverage;
    let velocity_term = saturation_ramp(velocity, VELOCITY_SATURATION);
    let depth_term = saturation_ramp(depth, DEPTH_SATURATION);

    let raw = coverage_weight * coverage_term
        + velocity_weight * velocity_term
        + depth_weight * depth_term;

    return clamp(raw, 0.0, max_reactivity);
}

// A deterministic blue-noise-style threshold in [0, 1): the three coordinates
// are mixed with large odd constants and passed through an integer xorshift /
// multiply finaliser, then normalised by 2^32. u32 arithmetic wraps modulo
// 2^32, matching the reference wrapping_mul / wrapping_add.
fn dither_threshold(x: u32, y: u32, frame: u32) -> f32 {
    var h = x * 0x9E3779B1u;
    h = h ^ (y * 0x85EBCA77u);
    h = h ^ (frame * 0xC2B2AE3Du);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    return f32(h) / TWO_POW_32;
}

// A hard sub-pixel draw decision: 1 when the sanitised coverage is at least the
// sanitised threshold, else 0. Non-finite coverage -> 0 (no draw), non-finite
// threshold -> 1 (only full coverage draws).
fn dither_alpha(coverage: f32, threshold: f32) -> f32 {
    let c = select(0.0, clamp(coverage, 0.0, 1.0), is_finite_f32(coverage));
    let t = select(1.0, clamp(threshold, 0.0, 1.0), is_finite_f32(threshold));
    return select(0.0, 1.0, c >= t);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.reactivity = pixel_reactivity(q);
    out.threshold = dither_threshold(q.x, q.y, q.frame);
    out.dither_alpha = dither_alpha(q.dither_coverage, out.threshold);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`HAIR_REACTIVE_MASK_WGSL`].
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
/// Eleven `4`-byte scalars plus a trailing pad word fill exactly three
/// `16`-byte slots, so the host stride matches the shader stride for batches of
/// two or more elements.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    coverage_weight: f32,
    velocity_weight: f32,
    depth_weight: f32,
    max_reactivity: f32,
    coverage: f32,
    screen_velocity: f32,
    depth_delta: f32,
    dither_coverage: f32,
    x: u32,
    y: u32,
    frame: u32,
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three `f32` and one `u32` fill one `16`-byte slot exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    reactivity: f32,
    threshold: f32,
    dither_alpha: f32,
    valid: u32,
}

/// One query for the hair reactive-mask twin: the reactivity weighting /
/// clamping parameters, the per-pixel reactivity inputs, a separate coverage
/// for the dither draw, and the pixel coordinate plus frame index for the
/// dither threshold hash.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairReactiveMaskQuery {
    /// Weight on the `1 - coverage` reactivity term.
    pub coverage_weight: f32,
    /// Weight on the screen-`velocity` saturation term.
    pub velocity_weight: f32,
    /// Weight on the `depth_delta` saturation term.
    pub depth_weight: f32,
    /// Upper clamp on the final reactivity, itself clamped to `[0, 1]`.
    pub max_reactivity: f32,
    /// Sub-pixel coverage in `[0, 1]`; non-finite is treated as fully covered.
    pub coverage: f32,
    /// Screen-space motion in pixels per frame; the magnitude is used.
    pub screen_velocity: f32,
    /// Relative depth change; the magnitude is used.
    pub depth_delta: f32,
    /// Sub-pixel coverage fed to the dither draw decision.
    pub dither_coverage: f32,
    /// Pixel `x` coordinate for the dither threshold hash.
    pub x: u32,
    /// Pixel `y` coordinate for the dither threshold hash.
    pub y: u32,
    /// Frame index that rotates the dither pattern in time.
    pub frame: u32,
}

impl HairReactiveMaskQuery {
    /// Builds a query from the reactivity parameters, the per-pixel inputs, the
    /// dither coverage and the `(x, y, frame)` coordinate.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a flat query mirrors the std430 layout uploaded to the kernel"
    )]
    pub fn new(
        coverage_weight: f32,
        velocity_weight: f32,
        depth_weight: f32,
        max_reactivity: f32,
        coverage: f32,
        screen_velocity: f32,
        depth_delta: f32,
        dither_coverage: f32,
        x: u32,
        y: u32,
        frame: u32,
    ) -> HairReactiveMaskQuery {
        HairReactiveMaskQuery {
            coverage_weight,
            velocity_weight,
            depth_weight,
            max_reactivity,
            coverage,
            screen_velocity,
            depth_delta,
            dither_coverage,
            x,
            y,
            frame,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `pixel_reactivity`, `dither_threshold` and `dither_alpha` outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairReactiveMaskResult {
    /// The reactive-mask value in `[0, max_reactivity]`.
    pub reactivity: f32,
    /// The blue-noise dither threshold in `[0, 1)`.
    pub threshold: f32,
    /// The hard dither draw decision in `{0, 1}`.
    pub dither_alpha: f32,
    /// `1` when the query produced a finite, in-range result (always, here).
    pub valid: u32,
}

/// Encodes one [`HairReactiveMaskQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HairReactiveMaskQuery) -> GpuQuery {
    GpuQuery {
        coverage_weight: q.coverage_weight,
        velocity_weight: q.velocity_weight,
        depth_weight: q.depth_weight,
        max_reactivity: q.max_reactivity,
        coverage: q.coverage,
        screen_velocity: q.screen_velocity,
        depth_delta: q.depth_delta,
        dither_coverage: q.dither_coverage,
        x: q.x,
        y: q.y,
        frame: q.frame,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HairReactiveMaskResult`].
fn decode_result(raw: &GpuResult) -> HairReactiveMaskResult {
    HairReactiveMaskResult {
        reactivity: raw.reactivity,
        threshold: raw.threshold,
        dither_alpha: raw.dither_alpha,
        valid: raw.valid,
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

/// A compiled, reusable hair reactive-mask compute pipeline, twinning the `CPU`
/// golden `pixel_reactivity`, `dither_threshold` and `dither_alpha`.
pub struct GpuHairReactiveMask {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairReactiveMask {
    /// Compiles the hair reactive-mask kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairReactiveMask {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask"),
            source: ShaderSource::Wgsl(HAIR_REACTIVE_MASK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairReactiveMask {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HairReactiveMaskResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairReactiveMaskQuery],
    ) -> Vec<HairReactiveMaskResult> {
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
            label: Some("prism_volumetric_hair_reactive_mask_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_bind_group"),
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
            label: Some("prism_volumetric_hair_reactive_mask_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_reactive_mask_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_reactive_mask_pass"),
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
