//! `wgpu` compute twin of the screen-space per-particle motion-blur `tap`
//! kernel
//! ([`motion_blur`](prism_render_architecture::particle::motion_blur), design
//! §21, "逐粒子屏幕空间运动模糊").
//!
//! A fast particle that moves several pixels per frame reads as a crisp,
//! strobing sprite unless it is smeared along its screen-space motion. The
//! `CPU` golden
//! [`motion_blur`](prism_render_architecture::particle::motion_blur) turns one
//! *already-computed* screen-space motion vector (pixels per frame) plus a
//! shutter model into a symmetric multi-`tap` sampling kernel —
//! [`MotionBlurParams::build_taps`](prism_render_architecture::particle::motion_blur::MotionBlurParams::build_taps)
//! produces the along-vector `tap` offsets and their normalized weights, and
//! [`accumulate_taps`](prism_render_architecture::particle::motion_blur::accumulate_taps)
//! forms the weighted-average colour a draw kernel writes back. [`GpuMotionBlur`]
//! is the on-device twin that runs one thread per particle and reproduces that
//! whole pipeline: it rebuilds the `tap` offsets and weights and accumulates the
//! supplied per-`tap` scene colours, so a passing real-device parity test is
//! direct evidence the ported kernel smears along the same vector, with the same
//! `tap` count, the same weights and the same accumulation order the reference
//! does, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! The full reference pipeline is reproduced per particle: (1) the
//! direction/length decomposition and the shutter-scaled, `max_blur_px`-clamped
//! streak half-length
//! ([`span_length`](prism_render_architecture::particle::motion_blur::MotionBlurParams::span_length)),
//! (2) the self-contained `SplitMix32` integer bit-mixer turning a per-particle
//! seed into a sub-`tap` jitter offset in normalized `tap` space
//! ([`jitter_offset`](prism_render_architecture::particle::motion_blur::MotionBlurParams::jitter_offset)),
//! (3) the evenly spaced, jittered and clamped `tap` coordinates with their
//! `dir * (t * half)` offsets and rational
//! [`tap_weight`](prism_render_architecture::particle::motion_blur::tap_weight)
//! `1 / (1 + 3 t^2)` profile, (4) the sum-to-one weight normalization, and
//! (5) the weighted-average colour accumulation
//! ([`accumulate_taps`](prism_render_architecture::particle::motion_blur::accumulate_taps))
//! bounded by the shorter of the supplied colour run and the `tap` count. The
//! jitter hash is a real feature of the pass, not a general-purpose codec, and
//! it is reproduced bit for bit (integer avalanche plus an exact `2^-24` scale),
//! so the jittered `tap` coordinates match the reference exactly.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference exactly: the same `hash_u32` `SplitMix32`
//! avalanche, the same high-24-bit `/ 2^24` unit scale, the same
//! `(unit - 0.5) * jitter_strength * spacing` jitter, the same
//! `-1 + 2 i / (n - 1)` `tap` coordinates clamped to `[-1, 1]` after jitter, the
//! same `normalize_or_zero` direction (zero below [`EPS_LEN`]), the same
//! `clamp(speed * shutter, 0, max_blur_px)` span, the same `dir * (t * half)`
//! offsets, the same `1 / (1 + 3 t^2)` weight, the same divide-by-sum
//! normalization, and the same weighted-average accumulation in `tap` order
//! bounded by `min(tap_count, colour_count)`. A single `tap`, a zero-length
//! motion vector or a zero streak collapses to a centred, zero-offset direct
//! passthrough exactly as the reference does, and an empty colour run yields a
//! fully transparent black sample.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `sqrt`, `+ - * /` and unsigned integer arithmetic / bit shifts —
//! with no `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it
//! runs unmodified on Metal, Vulkan and DX12. The only transcendental-adjacent
//! call is `sqrt` for the vector length (the reference uses it too); every
//! divide is guarded — the direction reciprocal is reached only after the
//! length clears [`EPS_LEN`], the weight denominator `1 + 3 t^2` is at least
//! `1`, and both normalization divides are reached only after their sum is
//! proven strictly positive.
//!
//! # Correctness model
//!
//! The jitter hash is exact integer arithmetic plus an exact power-of-two
//! scale, so the per-particle jitter offset and the jittered `tap` coordinates
//! are bit-identical on `CPU` and `GPU`; the `tap` count and the per-`tap`
//! weight *profile input* `t` therefore match exactly. The downstream offsets,
//! weights and accumulated colour fold those coordinates through short
//! closed-form expressions (one `sqrt`, a handful of divides and
//! multiply-adds); they are not guaranteed bit-exact because a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test asserts an
//! absolute/relative tolerance tight enough to catch a genuinely wrong port
//! (a swapped `tap` order, a dropped jitter, a wrong weight profile, a missing
//! normalization or a flipped accumulation bound) yet loose enough to admit
//! that legal fused multiply-add contraction, and additionally pins the
//! degenerate zero-motion and empty-colour cases bit for bit, where the offsets
//! collapse to an exact zero vector and the accumulated colour to an exact
//! transparent black.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard reconstruction-filter per-particle motion blur (Unreal,
//! Unity `VFX Graph`, `Frostbite` at the algorithm level) plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::motion_blur::{MotionBlurParams, Rgba, Vec2};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` motion-blur kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` `tap` build hash-for-hash
/// and the colour accumulation step-for-step; see the module documentation for
/// the algorithm.
///
/// The `Params` struct lays the four packed [`MotionBlurParams`] scalars out in
/// field order (three `f32` then the `u32` `tap_count`, a single `vec4`-sized
/// `std430` slot), followed by the per-dispatch particle count and padding to a
/// `32`-byte, `16`-byte-aligned uniform block.
const MOTION_BLUR_WGSL: &str = r#"
// Motion-blur twin: one thread per particle rebuilds the along-vector tap
// offsets and normalized weights and accumulates the supplied per-tap scene
// colours into a weighted-average RGBA. It mirrors the CPU golden
// `particle::motion_blur` (`MotionBlurParams::build_taps` + `accumulate_taps`),
// reproduces the SplitMix32 jitter hash bit for bit, uses only the portable
// core-WGSL subset (min/max/clamp/sqrt and + - * / plus integer bit ops), and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard reconstruction-filter per-particle motion blur; no
// Unreal Engine source or derived code.

struct Params {
    // Maximum blur-streak length in pixels; the streak is clamped to this.
    max_blur_px: f32,
    // Per-particle tap jitter strength in [0, 1], a fraction of one tap spacing.
    jitter_strength: f32,
    // Shutter fraction in [0, 1]: how much of the frame interval is open.
    shutter: f32,
    // Number of taps along the streak, at least one (host-clamped).
    tap_count: u32,
    // Particle count, one thread each. Reuses the first pad word of the shared
    // std430 params layout.
    particle_count: u32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One particle. 32-byte std430 stride matching the host `GpuQuery`.
struct Query {
    // Screen-space motion vector (pixels/frame), horizontal component.
    motion_x: f32,
    // Screen-space motion vector (pixels/frame), vertical component.
    motion_y: f32,
    // Per-particle seed for the sub-tap jitter hash.
    particle_seed: u32,
    // Start index of this particle's colour run in the flat colour buffer, in
    // RGBA units (four f32 each).
    color_offset: u32,
    // Number of RGBA colours supplied for this particle.
    color_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
// Flat per-tap scene colours, four f32 (RGBA) per colour, run-indexed per query.
@group(0) @binding(2) var<storage, read> colors: array<f32>;
// Flat per-tap offsets, two f32 (x, y) per tap, `tap_count` taps per particle.
@group(0) @binding(3) var<storage, read_write> offsets: array<f32>;
// Flat per-tap normalized weights, `tap_count` per particle.
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
// Flat accumulated RGBA, four f32 per particle.
@group(0) @binding(5) var<storage, read_write> accum: array<f32>;

// Smallest vector length treated as non-zero when normalizing, matching the
// reference `EPS_LEN`.
const EPS_LEN: f32 = 1e-12;
// Falloff constant of the rational tap-weight profile, matching the reference
// `WEIGHT_FALLOFF`.
const WEIGHT_FALLOFF: f32 = 3.0;
// Reciprocal of 2^24, scaling a 24-bit integer exactly into [0, 1), matching
// the reference `INV_2POW24`. A compile-time constant divide, not a run-time
// transcendental.
const INV_2POW24: f32 = 1.0 / 16777216.0;

// Self-contained SplitMix32-style integer bit-mixer, matching the reference
// `hash_u32`: a seeded add then two xor-shift / odd-multiply rounds and a final
// xor-shift. WGSL u32 add and multiply wrap on overflow, exactly like the
// reference `wrapping_add` / `wrapping_mul`.
fn hash_u32(seed: u32) -> u32 {
    var x = seed + 0x9e3779b9u;
    x = (x ^ (x >> 16u)) * 0x85ebca6bu;
    x = (x ^ (x >> 13u)) * 0xc2b2ae35u;
    return x ^ (x >> 16u);
}

// Maps a 32-bit hash word to a uniform f32 in [0, 1) using only the high 24
// bits, matching the reference `unit_f32_from_bits`. The 24-bit payload is
// exactly representable, so the cast plus the exact 2^-24 scale is
// bit-reproducible.
fn unit_f32_from_bits(bits: u32) -> f32 {
    let payload = bits >> 8u;
    return f32(payload) * INV_2POW24;
}

// Rational tap weight for a normalized coordinate t in [-1, 1], matching the
// reference `tap_weight`. The denominator is at least 1.0, so the reciprocal
// never divides by a vanishing value.
fn tap_weight(t: f32) -> f32 {
    return 1.0 / (1.0 + WEIGHT_FALLOFF * t * t);
}

@compute @workgroup_size(64)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle = gid.x;
    if (particle >= params.particle_count) {
        return;
    }
    let query = queries[particle];
    // Host clamps `tap_count` to at least one, so `n >= 1` here.
    let n = params.tap_count;
    let out_base = particle * n;

    // Direction / length decomposition, matching `normalize_or_zero`: a vector
    // shorter than EPS_LEN has no well-defined direction and yields the zero
    // vector, so every tap offset collapses to zero (a centred passthrough).
    let mx = query.motion_x;
    let my = query.motion_y;
    let len = sqrt(mx * mx + my * my);
    var dir_x = 0.0;
    var dir_y = 0.0;
    if (len > EPS_LEN) {
        let inv_len = 1.0 / len;
        dir_x = mx * inv_len;
        dir_y = my * inv_len;
    }

    // Shutter-scaled, max_blur_px-clamped streak half-length, matching
    // `span_length(length) * 0.5`. A negative speed cannot arise (the length is
    // non-negative), but the max(.,0) mirrors the reference exactly.
    let span = clamp(max(len, 0.0) * params.shutter, 0.0, params.max_blur_px);
    let half = span * 0.5;

    // Uniform normalized tap spacing in [-1, 1] space, zero for a single tap,
    // matching `tap_spacing`.
    var spacing = 0.0;
    if (n > 1u) {
        spacing = 2.0 / f32(n - 1u);
    }
    // Deterministic per-particle jitter in normalized tap space, matching
    // `jitter_offset`: a symmetric fraction of one tap spacing driven by the
    // integer hash.
    let unit = unit_f32_from_bits(hash_u32(query.particle_seed));
    let jitter = (unit - 0.5) * params.jitter_strength * spacing;

    // First pass: evenly spaced, jittered, clamped tap coordinates; their
    // `dir * (t * half)` offsets and raw rational weights. Accumulate the raw
    // weight sum for normalization, matching `build_taps`.
    var wsum = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        var coord = 0.0;
        if (n > 1u) {
            coord = -1.0 + 2.0 * f32(i) / f32(n - 1u);
        }
        let t = clamp(coord + jitter, -1.0, 1.0);
        let scaled = t * half;
        offsets[(out_base + i) * 2u] = dir_x * scaled;
        offsets[(out_base + i) * 2u + 1u] = dir_y * scaled;
        let w = tap_weight(t);
        weights[out_base + i] = w;
        wsum = wsum + w;
    }

    // Normalize the weights to sum to one, matching `normalize_weights`. The
    // rational profile is strictly positive and n >= 1, so the sum is always
    // positive; the guard mirrors the reference's non-positive-sum early-out.
    if (wsum > 0.0) {
        let inv_sum = 1.0 / wsum;
        for (var i = 0u; i < n; i = i + 1u) {
            weights[out_base + i] = weights[out_base + i] * inv_sum;
        }
    }

    // Weighted-average colour accumulation, matching `accumulate_taps`: the sum
    // is bounded by the shorter of the supplied colour run and the tap count,
    // and the per-channel weighted sum is divided by the accumulated weight.
    var taps = n;
    if (query.color_count < taps) {
        taps = query.color_count;
    }
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var awsum = 0.0;
    for (var k = 0u; k < taps; k = k + 1u) {
        let w = weights[out_base + k];
        awsum = awsum + w;
        let cbase = (query.color_offset + k) * 4u;
        acc0 = acc0 + colors[cbase] * w;
        acc1 = acc1 + colors[cbase + 1u] * w;
        acc2 = acc2 + colors[cbase + 2u] * w;
        acc3 = acc3 + colors[cbase + 3u] * w;
    }
    let abase = particle * 4u;
    if (awsum > 0.0) {
        let inv_w = 1.0 / awsum;
        accum[abase] = acc0 * inv_w;
        accum[abase + 1u] = acc1 * inv_w;
        accum[abase + 2u] = acc2 * inv_w;
        accum[abase + 3u] = acc3 * inv_w;
    } else {
        // An empty colour run (or non-positive weight sum) is transparent black.
        accum[abase] = 0.0;
        accum[abase + 1u] = 0.0;
        accum[abase + 2u] = 0.0;
        accum[abase + 3u] = 0.0;
    }
}
"#;

/// One particle's motion-blur query: the twin of a single
/// [`build_taps`](prism_render_architecture::particle::motion_blur::MotionBlurParams::build_taps)
/// plus
/// [`accumulate_taps`](prism_render_architecture::particle::motion_blur::accumulate_taps)
/// pair.
///
/// `tap_colors` are the scene colours sampled at each `tap`'s offset, in `tap`
/// order. The reference accumulator zips its weights against this slice, so the
/// effective accumulation length is `min(params.tap_count, tap_colors.len())`;
/// a slice shorter than the `tap` count simply stops the accumulation early,
/// exactly as the reference does, while the full `tap_count` offsets and
/// weights are always produced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlurQuery<'colors> {
    /// The *given* screen-space motion vector (pixels per frame); the twin only
    /// consumes it and never recomputes it.
    pub motion_px: Vec2,
    /// Per-particle seed for the sub-`tap` jitter hash.
    pub particle_seed: u32,
    /// Scene colours sampled at each `tap`'s offset, in `tap` order.
    pub tap_colors: &'colors [Rgba],
}

/// The reconstructed `tap` kernel plus accumulated colour for one particle, the
/// device-side twin of the reference
/// [`TapKernel`](prism_render_architecture::particle::motion_blur::TapKernel)
/// and its
/// [`accumulate_taps`](prism_render_architecture::particle::motion_blur::accumulate_taps)
/// result.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotionBlurResult {
    /// Per-`tap` screen-space offsets from the particle centre, in pixels,
    /// matching `TapKernel::offsets`.
    pub offsets: Vec<Vec2>,
    /// Per-`tap` normalized weights, summing to one, matching
    /// `TapKernel::weights`.
    pub weights: Vec<f32>,
    /// The weighted-average accumulated colour, matching `accumulate_taps`.
    pub accumulated: Rgba,
}

/// Uniform parameters for one motion-blur dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`MOTION_BLUR_WGSL`]: the four [`MotionBlurParams`]
/// fields in declaration order, then the particle count (reusing the first pad
/// word) and three pad words — `32` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    max_blur_px: f32,
    jitter_strength: f32,
    shutter: f32,
    tap_count: u32,
    particle_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One particle as uploaded. `32`-byte `std430` stride matching `Query` in the
/// shader: the two motion components, the jitter seed, the colour-run offset and
/// count, then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    motion_x: f32,
    motion_y: f32,
    particle_seed: u32,
    color_offset: u32,
    color_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-particle motion-blur pipeline.
pub struct GpuMotionBlur {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionBlur {
    /// Compiles the motion-blur `tap` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionBlur {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_blur"),
            source: ShaderSource::Wgsl(MOTION_BLUR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_blur_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_blur_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_blur_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("blur"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionBlur {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the `tap` kernel and accumulates the supplied per-`tap` colours for
    /// every particle in `queries` under the shared `params`, returning one
    /// [`MotionBlurResult`] per particle in input order.
    ///
    /// For particle `q` the returned `offsets` and `weights` equal
    /// [`params.build_taps(q.motion_px, q.particle_seed)`](prism_render_architecture::particle::motion_blur::MotionBlurParams::build_taps)
    /// and `accumulated` equals
    /// [`accumulate_taps(q.tap_colors, &kernel.weights)`](prism_render_architecture::particle::motion_blur::accumulate_taps)
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: &MotionBlurParams,
        queries: &[MotionBlurQuery<'_>],
    ) -> Vec<MotionBlurResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let taps = params.effective_taps();
        let tap_count = u32::try_from(taps).unwrap_or(1).max(1);
        let particle_count = queries.len();

        // Flatten the per-particle colour runs into one contiguous RGBA buffer,
        // recording each particle's run offset (in RGBA units) and length so the
        // kernel reads the same colours the reference accumulator zips against.
        let mut flat_colors: Vec<f32> = Vec::new();
        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| {
                let color_offset = u32::try_from(flat_colors.len() / 4).unwrap_or(0);
                for color in q.tap_colors {
                    flat_colors.extend_from_slice(color);
                }
                GpuQuery {
                    motion_x: q.motion_px.x,
                    motion_y: q.motion_px.y,
                    particle_seed: q.particle_seed,
                    color_offset,
                    color_count: u32::try_from(q.tap_colors.len()).unwrap_or(0),
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                }
            })
            .collect();

        // Storage buffers cannot be zero-sized; a batch whose particles supply
        // no colours at all still needs one element to bind.
        if flat_colors.is_empty() {
            flat_colors.push(0.0);
        }

        let gpu_params = GpuParams {
            max_blur_px: params.max_blur_px,
            jitter_strength: params.jitter_strength,
            shutter: params.shutter,
            tap_count,
            particle_count: u32::try_from(particle_count).unwrap_or(u32::MAX),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Output element counts. `particle_count >= 1` (empty is handled above)
        // and `taps >= 1`, so none of these is zero.
        let offsets_elems = particle_count * taps * 2;
        let weights_elems = particle_count * taps;
        let accum_elems = particle_count * 4;
        let f32_size = size_of::<f32>() as u64;
        let offsets_bytes = offsets_elems as u64 * f32_size;
        let weights_bytes = weights_elems as u64 * f32_size;
        let accum_bytes = accum_elems as u64 * f32_size;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_blur_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_blur_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let colors_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_blur_colors"),
            contents: bytemuck::cast_slice(&flat_colors),
            usage: BufferUsages::STORAGE,
        });
        let offsets_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_blur_offsets"),
            size: offsets_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let weights_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_blur_weights"),
            size: weights_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let accum_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_blur_accum"),
            size: accum_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let offsets_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_blur_offsets_stage"),
            size: offsets_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weights_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_blur_weights_stage"),
            size: weights_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let accum_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_blur_accum_stage"),
            size: accum_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_blur_bind_group"),
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
                    resource: colors_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: offsets_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: weights_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: accum_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_blur_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_blur_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per particle, in workgroups of 64 (the kernel's size).
            let groups = u32::try_from(particle_count)
                .unwrap_or(u32::MAX)
                .div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&offsets_buf, 0, &offsets_stage, 0, offsets_bytes);
        encoder.copy_buffer_to_buffer(&weights_buf, 0, &weights_stage, 0, weights_bytes);
        encoder.copy_buffer_to_buffer(&accum_buf, 0, &accum_stage, 0, accum_bytes);
        ctx.queue().submit([encoder.finish()]);

        offsets_stage.slice(..).map_async(MapMode::Read, |_| {});
        weights_stage.slice(..).map_async(MapMode::Read, |_| {});
        accum_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let offsets_flat = read_f32(&offsets_stage);
        let weights_flat = read_f32(&weights_stage);
        let accum_flat = read_f32(&accum_stage);

        debug_assert_eq!(offsets_flat.len(), offsets_elems);
        debug_assert_eq!(weights_flat.len(), weights_elems);
        debug_assert_eq!(accum_flat.len(), accum_elems);
        debug_assert_eq!(MOTION_BLUR_PARAMS_STRIDE, size_of::<GpuParams>());

        (0..particle_count)
            .map(|p| {
                let base = p * taps;
                let offsets = (0..taps)
                    .map(|i| {
                        let o = (base + i) * 2;
                        Vec2::new(offsets_flat[o], offsets_flat[o + 1])
                    })
                    .collect();
                let weights = (0..taps).map(|i| weights_flat[base + i]).collect();
                let a = p * 4;
                let accumulated = [
                    accum_flat[a],
                    accum_flat[a + 1],
                    accum_flat[a + 2],
                    accum_flat[a + 3],
                ];
                MotionBlurResult {
                    offsets,
                    weights,
                    accumulated,
                }
            })
            .collect()
    }
}

/// Byte stride of the uploaded [`GpuParams`] block: two `vec4`-sized `std430`
/// slots. Used only to assert the host and shader layouts agree.
const MOTION_BLUR_PARAMS_STRIDE: usize = 32;

/// Maps a readback staging buffer and copies its contents out as `f32`s.
///
/// The caller must have polled the device so the mapping is ready; the mapped
/// range is dropped and the buffer unmapped before returning.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
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
