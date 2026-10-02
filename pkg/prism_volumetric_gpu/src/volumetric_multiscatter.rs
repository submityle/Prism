//! `wgpu` compute twin of the octave-summed volumetric *multiple-scattering*
//! energy-compensation contract
//! ([`volumetric_multiscatter`](prism_render_architecture::particle::volumetric_multiscatter),
//! particle design §20).
//!
//! The `CPU` golden
//! [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
//! folds a fixed, bounded number of scattering orders (octaves) into one
//! per-channel (`RGB`) response: order `i` carries a running-product throughput
//! `(albedo · octave_decay)^i` and a phase lobe whose anisotropy is the base
//! [`PhaseParams`](prism_render_architecture::particle::shading::PhaseParams)
//! scaled by `anisotropy_falloff^i`, plus an isotropic ambient floor
//! `ambient_lift · (1 / 4π) · Σ throughput`. Each octave evaluates the
//! `sqrt`-only Henyey-Greenstein double-lobe phase
//! [`double_lobe_phase`](prism_render_architecture::particle::volumetrics::double_lobe_phase).
//! [`GpuVolumetricMultiScatter`] is the on-device twin: one thread folds one
//! query's octave loop, so a passing real-device parity test is direct evidence
//! the ported kernel runs the same octave recurrence, the same `^1.5`
//! denominator and the same degenerate fallbacks the reference does — not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Each per-query answer the reference computes is reproduced: the three `RGB`
//! channels of [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
//! (equivalently [`MultiScatterParams::response`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response)
//! when the query carries directions) and the scalar
//! [`MultiScatterParams::luminance_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::luminance_cos)
//! as the fourth output lane. The kernel mirrors the reference branch for
//! branch: it [`sanitized`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::sanitized)-clamps
//! every field (octaves to `1..=64`, `albedo`/`anisotropy_falloff`/`octave_decay`
//! into `0..=1`, `ambient_lift` to non-negative), runs the dynamic octave loop
//! as running products (`×` only, no `powf`), and — for a direction query —
//! collapses a zero-length input to `cos_theta = 0` via the same
//! `normalize_or_zero` guard instead of producing `NaN`.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`, `min`, `max`, `clamp` and multiply/add in the
//! portable core-`WGSL` subset — no `sin`, `cos`, `exp`, `log`, `pow`, `tan`,
//! inverse trig, `smoothstep`, rounding or `cbrt`, and no optional device
//! feature — so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The `^1.5`
//! Henyey-Greenstein denominator is formed as `d · sqrt(d)` with `d` floored at
//! `EPS`, exactly as the reference spells it.
//!
//! # Correctness model
//!
//! The response contains no transcendental call (the reference restricts itself
//! to `sqrt` for exactly this reason), so `CPU` and `GPU` evaluate the same
//! closed-form algebra over the same fixed octave count — the integer loop bound
//! is bit-exact because the clamp is unsigned integer arithmetic. They are
//! **not** bit-exact in the floats: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a `1e-6` relative floor),
//! tight enough to catch a genuinely wrong port (a dropped octave, a swapped
//! lobe, a missing ambient term, a wrong falloff) yet loose enough to admit
//! legal fused-multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`volumetric_multiscatter`](prism_render_architecture::particle::volumetric_multiscatter)
//! (public octave-sum multiple-scattering energy compensation); no third-party
//! engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::shading::PhaseParams;
use prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams;
use prism_render_architecture::particle::Vec3;
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

/// The portable core-`WGSL` octave-summed multiple-scattering kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `multiscatter_main` mirrors the `CPU` golden
/// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
/// branch for branch; see the module documentation for the algorithm.
const VOLUMETRIC_MULTISCATTER_WGSL: &str = r#"
// Octave-summed volumetric multiple-scattering twin: one thread folds one
// query's dynamic octave loop into the per-channel RGB response plus its mean
// luminance. It mirrors the CPU golden
// `particle::volumetric_multiscatter::MultiScatterParams::response_cos` branch
// for branch, inlines the sqrt-only Henyey-Greenstein double-lobe phase from
// `particle::volumetrics::double_lobe_phase`, uses only the portable core-WGSL
// subset (clamp/min/max/sqrt and + - * /) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::volumetric_multiscatter; no third-party engine source or derived
// code.

// `4*PI`, the exact f32 literal the reference uses for the phase normalization.
const FOUR_PI: f32 = 12.566371;
// Phase-denominator floor, matching `particle::volumetrics::EPS`.
const EPS: f32 = 1e-6;
// Squared-length floor for normalize-or-zero, matching
// `particle::EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1e-12;
// Isotropic phase `1 / (4*PI)`, formed by division exactly like the reference
// `ISOTROPIC_PHASE` so the ambient floor matches.
const ISOTROPIC_PHASE: f32 = 1.0 / FOUR_PI;
// Upper bound on the folded octave count, matching the reference `MAX_OCTAVES`.
const MAX_OCTAVES: u32 = 64u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Flat scalar layout (every member is 4 bytes, so the struct aligns to 4 and
// no vec3 16-byte padding intrudes), matching the repr(C) `GpuQuery`.
struct Query {
    // Per-channel single-scatter albedo (RGB).
    albedo_x: f32,
    albedo_y: f32,
    albedo_z: f32,
    // Base single-scatter phase: forward anisotropy, back-lobe weight, back
    // anisotropy (reused verbatim for order 0).
    phase_g: f32,
    back_lobe_weight: f32,
    back_g: f32,
    // Per-octave anisotropy bandwidth factor and extra throughput decay.
    anisotropy_falloff: f32,
    octave_decay: f32,
    // Isotropic ambient-floor strength.
    ambient_lift: f32,
    // Scattering cosine used directly when `use_direction` is zero.
    cos_theta: f32,
    // Incoming / outgoing propagation directions, used when `use_direction`
    // is non-zero (collapsed to cos_theta = 0 on a zero-length input).
    inc_x: f32,
    inc_y: f32,
    inc_z: f32,
    out_x: f32,
    out_y: f32,
    out_z: f32,
    // Number of scattering orders to fold (clamped to 1..=MAX_OCTAVES).
    octaves: u32,
    // Non-zero: derive cos_theta from the directions; zero: use `cos_theta`.
    use_direction: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Per-channel RGB multiple-scattering response.
    r: f32,
    g: f32,
    b: f32,
    // Mean of the three channels (the reference `luminance_cos`).
    lum: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Single-lobe Henyey-Greenstein phase, the sqrt-only `^1.5` form the reference
// `particle::volumetrics::henyey_greenstein` uses: `(1 - g^2) / (4*PI * d^1.5)`
// with `d = max(base, EPS)` and `d^1.5 = d * sqrt(d)`.
fn hg_phase(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let base = 1.0 + g2 - 2.0 * g * cos_theta;
    var d = EPS;
    if (base > EPS) {
        d = base;
    }
    let d15 = d * sqrt(d);
    return (1.0 - g2) / (FOUR_PI * d15);
}

// Double-lobe (front + back) phase: `(1 - w) * front + w * back`, with the
// weight clamped to 0..=1 exactly as `particle::volumetrics::double_lobe_phase`.
fn double_lobe(g: f32, back_lobe_weight: f32, back_g: f32, cos_theta: f32) -> f32 {
    var w = 0.0;
    if (back_lobe_weight > 1.0) {
        w = 1.0;
    } else if (back_lobe_weight > 0.0) {
        w = back_lobe_weight;
    }
    let front = hg_phase(g, cos_theta);
    let back = hg_phase(back_g, cos_theta);
    return (1.0 - w) * front + w * back;
}

// Unit vector along (x, y, z), or the zero vector when the input is
// numerically zero, matching `particle::Vec3::normalize_or_zero`.
fn normalize_or_zero(x: f32, y: f32, z: f32) -> vec3<f32> {
    let len_sq = x * x + y * y + z * z;
    if (len_sq > EPS_LEN_SQ) {
        let inv = 1.0 / sqrt(len_sq);
        return vec3<f32>(x * inv, y * inv, z * inv);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn multiscatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Sanitize every field into its valid range (reference `sanitized`).
    let octaves = clamp(q.octaves, 1u, MAX_OCTAVES);
    let albedo_x = clamp(q.albedo_x, 0.0, 1.0);
    let albedo_y = clamp(q.albedo_y, 0.0, 1.0);
    let albedo_z = clamp(q.albedo_z, 0.0, 1.0);
    let anisotropy_falloff = clamp(q.anisotropy_falloff, 0.0, 1.0);
    let octave_decay = clamp(q.octave_decay, 0.0, 1.0);
    let ambient_lift = max(q.ambient_lift, 0.0);

    // Scattering cosine: taken directly, or derived from the (normalized)
    // incoming / outgoing directions, which collapse a zero-length input to 0.
    var cos_theta = q.cos_theta;
    if (q.use_direction != 0u) {
        let n_in = normalize_or_zero(q.inc_x, q.inc_y, q.inc_z);
        let n_out = normalize_or_zero(q.out_x, q.out_y, q.out_z);
        cos_theta = dot(n_in, n_out);
    }

    // Per-octave throughput ratio: albedo tinted by the extra decay.
    let decay_x = albedo_x * octave_decay;
    let decay_y = albedo_y * octave_decay;
    let decay_z = albedo_z * octave_decay;

    // Order-0 throughput is unit; order-0 anisotropy scale is 1 (base lobe).
    var tput_x = 1.0;
    var tput_y = 1.0;
    var tput_z = 1.0;
    var anisotropy_scale = 1.0;
    var dir_x = 0.0;
    var dir_y = 0.0;
    var dir_z = 0.0;
    var sum_x = 0.0;
    var sum_y = 0.0;
    var sum_z = 0.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        let octave_phase = double_lobe(
            q.phase_g * anisotropy_scale,
            q.back_lobe_weight,
            q.back_g * anisotropy_scale,
            cos_theta,
        );
        dir_x = dir_x + tput_x * octave_phase;
        dir_y = dir_y + tput_y * octave_phase;
        dir_z = dir_z + tput_z * octave_phase;
        sum_x = sum_x + tput_x;
        sum_y = sum_y + tput_y;
        sum_z = sum_z + tput_z;
        // Advance the running products for the next octave (x only).
        tput_x = tput_x * decay_x;
        tput_y = tput_y * decay_y;
        tput_z = tput_z * decay_z;
        anisotropy_scale = anisotropy_scale * anisotropy_falloff;
    }

    let ambient = ambient_lift * ISOTROPIC_PHASE;
    var out: Result;
    out.r = dir_x + sum_x * ambient;
    out.g = dir_y + sum_y * ambient;
    out.b = dir_z + sum_z * ambient;
    out.lum = (out.r + out.g + out.b) / 3.0;
    results[idx] = out;
}
"#;

/// One octave-summed multiple-scattering query: the full
/// [`MultiScatterParams`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams)
/// field set plus the scattering input.
///
/// When [`Self::use_direction`] is `false` the kernel evaluates
/// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
/// at [`Self::cos_theta`]; when it is `true` the kernel evaluates
/// [`MultiScatterParams::response`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response)
/// by deriving the cosine from [`Self::incoming`] and [`Self::outgoing`]. Build
/// one with [`MultiScatterQuery::from_cos`] or
/// [`MultiScatterQuery::from_directions`] to copy the fields straight from a
/// reference `MultiScatterParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiScatterQuery {
    /// Per-channel single-scatter albedo (`RGB`).
    pub albedo: [f32; 3],
    /// Base single-scatter forward anisotropy `g`.
    pub phase_g: f32,
    /// Back-lobe blend weight.
    pub back_lobe_weight: f32,
    /// Back-lobe anisotropy `back_g`.
    pub back_g: f32,
    /// Per-octave anisotropy bandwidth factor.
    pub anisotropy_falloff: f32,
    /// Extra per-octave throughput decay.
    pub octave_decay: f32,
    /// Isotropic ambient-floor strength.
    pub ambient_lift: f32,
    /// Number of scattering orders to fold.
    pub octaves: u32,
    /// Scattering cosine used when [`Self::use_direction`] is `false`.
    pub cos_theta: f32,
    /// Incoming propagation direction, used when [`Self::use_direction`].
    pub incoming: [f32; 3],
    /// Outgoing propagation direction, used when [`Self::use_direction`].
    pub outgoing: [f32; 3],
    /// Selects the direction path (`true`) over the raw-cosine path (`false`).
    pub use_direction: bool,
}

impl MultiScatterQuery {
    /// Builds a `cos_theta` query from a reference `params` and a scattering
    /// cosine, mirroring
    /// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos).
    #[must_use]
    pub fn from_cos(params: &MultiScatterParams, cos_theta: f32) -> MultiScatterQuery {
        MultiScatterQuery {
            albedo: [params.albedo.x, params.albedo.y, params.albedo.z],
            phase_g: params.phase.g,
            back_lobe_weight: params.phase.back_lobe_weight,
            back_g: params.phase.back_g,
            anisotropy_falloff: params.anisotropy_falloff,
            octave_decay: params.octave_decay,
            ambient_lift: params.ambient_lift,
            octaves: params.octaves,
            cos_theta,
            incoming: [0.0, 0.0, 0.0],
            outgoing: [0.0, 0.0, 0.0],
            use_direction: false,
        }
    }

    /// Builds a direction query from a reference `params` and the `incoming` /
    /// `outgoing` directions, mirroring
    /// [`MultiScatterParams::response`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response).
    #[must_use]
    pub fn from_directions(
        params: &MultiScatterParams,
        incoming: Vec3,
        outgoing: Vec3,
    ) -> MultiScatterQuery {
        MultiScatterQuery {
            albedo: [params.albedo.x, params.albedo.y, params.albedo.z],
            phase_g: params.phase.g,
            back_lobe_weight: params.phase.back_lobe_weight,
            back_g: params.phase.back_g,
            anisotropy_falloff: params.anisotropy_falloff,
            octave_decay: params.octave_decay,
            ambient_lift: params.ambient_lift,
            octaves: params.octaves,
            cos_theta: 0.0,
            incoming: [incoming.x, incoming.y, incoming.z],
            outgoing: [outgoing.x, outgoing.y, outgoing.z],
            use_direction: true,
        }
    }

    /// Returns the [`PhaseParams`] packed into this query, matching the
    /// reference `MultiScatterParams::phase` field.
    #[must_use]
    pub fn phase(&self) -> PhaseParams {
        PhaseParams {
            g: self.phase_g,
            back_lobe_weight: self.back_lobe_weight,
            back_g: self.back_g,
        }
    }
}

/// The resolved answer for one query: the per-channel `RGB` response and its
/// mean luminance, matching the reference
/// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
/// and
/// [`MultiScatterParams::luminance_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::luminance_cos).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumetricMultiScatterResponse {
    /// The per-channel `RGB` multiple-scattering response.
    pub rgb: [f32; 3],
    /// The mean of the three channels.
    pub luminance: f32,
}

/// `repr(C)` `std430` layout of one packed query: twenty `4`-byte scalar lanes
/// (`80` bytes), each `vec3` flattened to three `f32` so the struct keeps a
/// `4`-byte alignment and no `16`-byte `vec3` padding intrudes, exactly as the
/// `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `albedo.x`.
    albedo_x: f32,
    /// `albedo.y`.
    albedo_y: f32,
    /// `albedo.z`.
    albedo_z: f32,
    /// Forward anisotropy `g`.
    phase_g: f32,
    /// Back-lobe blend weight.
    back_lobe_weight: f32,
    /// Back-lobe anisotropy.
    back_g: f32,
    /// Per-octave anisotropy bandwidth factor.
    anisotropy_falloff: f32,
    /// Extra per-octave throughput decay.
    octave_decay: f32,
    /// Isotropic ambient-floor strength.
    ambient_lift: f32,
    /// Raw scattering cosine.
    cos_theta: f32,
    /// `incoming.x`.
    inc_x: f32,
    /// `incoming.y`.
    inc_y: f32,
    /// `incoming.z`.
    inc_z: f32,
    /// `outgoing.x`.
    out_x: f32,
    /// `outgoing.y`.
    out_y: f32,
    /// `outgoing.z`.
    out_z: f32,
    /// Number of scattering orders to fold.
    octaves: u32,
    /// Direction-path selector (`0` = raw cosine, `1` = directions).
    use_direction: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &MultiScatterQuery) -> GpuQuery {
        GpuQuery {
            albedo_x: query.albedo[0],
            albedo_y: query.albedo[1],
            albedo_z: query.albedo[2],
            phase_g: query.phase_g,
            back_lobe_weight: query.back_lobe_weight,
            back_g: query.back_g,
            anisotropy_falloff: query.anisotropy_falloff,
            octave_decay: query.octave_decay,
            ambient_lift: query.ambient_lift,
            cos_theta: query.cos_theta,
            inc_x: query.incoming[0],
            inc_y: query.incoming[1],
            inc_z: query.incoming[2],
            out_x: query.outgoing[0],
            out_y: query.outgoing[1],
            out_z: query.outgoing[2],
            octaves: query.octaves,
            use_direction: u32::from(query.use_direction),
            pad0: 0,
            pad1: 0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: four `f32` lanes (`16` bytes)
/// matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `RGB` red channel.
    r: f32,
    /// `RGB` green channel.
    g: f32,
    /// `RGB` blue channel.
    b: f32,
    /// Mean-of-channels luminance.
    lum: f32,
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

/// A compiled, reusable octave-summed multiple-scattering compute pipeline.
pub struct GpuVolumetricMultiScatter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVolumetricMultiScatter {
    /// Compiles the multiple-scattering kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVolumetricMultiScatter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_multiscatter"),
            source: ShaderSource::Wgsl(VOLUMETRIC_MULTISCATTER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_multiscatter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_multiscatter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_multiscatter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("multiscatter_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVolumetricMultiScatter {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the multiple-scattering response for every query on-device and
    /// returns one [`VolumetricMultiScatterResponse`] per input, in order.
    ///
    /// Each result equals the reference
    /// [`MultiScatterParams::response_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response_cos)
    /// (or
    /// [`MultiScatterParams::response`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::response)
    /// for a direction query) to within the tolerance documented on this
    /// module, with the fourth lane reproducing
    /// [`MultiScatterParams::luminance_cos`](prism_render_architecture::particle::volumetric_multiscatter::MultiScatterParams::luminance_cos).
    /// An empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[MultiScatterQuery],
    ) -> Vec<VolumetricMultiScatterResponse> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_multiscatter_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_multiscatter_output"),
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
            label: Some("prism_volumetric_multiscatter_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_multiscatter_bind_group"),
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
            label: Some("prism_volumetric_multiscatter_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_multiscatter_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_multiscatter_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`VolumetricMultiScatterResponse`].
fn decode_result(raw: &GpuResult) -> VolumetricMultiScatterResponse {
    VolumetricMultiScatterResponse {
        rgb: [raw.r, raw.g, raw.b],
        luminance: raw.lum,
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
