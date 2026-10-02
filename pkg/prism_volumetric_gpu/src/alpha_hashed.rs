//! `wgpu` compute twin of the stochastic *hashed alpha testing* golden
//! ([`alpha_hashed`](prism_render_architecture::particle::alpha_hashed),
//! particle design §17).
//!
//! The `CPU` golden
//! ([`alpha_hashed`](prism_render_architecture::particle::alpha_hashed))
//! replaces a fixed `alpha < 0.5` cutoff with a *per-location* threshold drawn
//! from a stable spatial hash: a fragment survives when `alpha >= threshold`,
//! and because the threshold is uniformly distributed in `[0, 1)` the surviving
//! fraction equals the fragment `alpha`, reading as stochastic transparency
//! that resolves cleanly under `MSAA` and `TAA`. The threshold is anchored to
//! object/world coordinates, discretized at two bracketing power-of-two `LOD`
//! levels and linearly blended, so it stays put on the surface and does not pop
//! as the sprite changes size.
//!
//! [`GpuAlphaHashed`] is the on-device twin: one thread per
//! [`AlphaHashedQuery`] (an `anchor`, a `ddx` / `ddy` derivative pair, the
//! fragment `alpha`, the three `AlphaHashParams` fields and a raw lattice probe)
//! reproduces the same arithmetic branch for branch and writes one
//! [`AlphaHashedResult`] holding the blended `threshold`, the anisotropic
//! `lod_scale`, the keep/discard decision, the expected `coverage` and an
//! independent `hash_probe`.
//!
//! # The `log2` / `exp2` are bit-tricks, not transcendentals
//!
//! The golden never calls `log2` / `exp2` / `powf`. Instead
//! [`approx_log2`](prism_render_architecture::particle::alpha_hashed) reads the
//! `IEEE754` `f32` bit pattern — the biased exponent field gives the integer
//! octave and the mantissa in `[1, 2)` supplies the linear fraction — and
//! [`exp2_pow`](prism_render_architecture::particle::alpha_hashed) rebuilds
//! `2^level` by writing the biased exponent field directly. This twin
//! reproduces both **bit-identically** in `WGSL` using `bitcast<u32>` /
//! `bitcast<f32>` and the same exponent-field assembly; the kernel calls no
//! `WGSL` `log2`, `exp2` or `pow` builtin anywhere. The integer avalanche
//! [`mix`](prism_render_architecture::particle::alpha_hashed) and
//! [`hash3`](prism_render_architecture::particle::alpha_hashed) are reproduced
//! with the same `u32` xor-shifts and odd multiplies (`WGSL` `u32` arithmetic
//! wraps modulo `2^32`, matching `wrapping_mul`), so the hash path is bit-exact.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `+ - * /`, unsigned/signed bit ops, one `sqrt`, `select`
//! and `bitcast` — with no `sin`, `cos`, `exp`, `exp2`, `log`, `log2`, `pow`,
//! `tan` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! The integer hash path (`mix`, `hash3`, the exponent-field `exp2_pow` and the
//! `floor`-based `quantize_coord`) is reproduced bit-for-bit, so the integer
//! lattice cell, the chosen `LOD` level and the keep/discard decision agree
//! exactly. The continuous fields (`threshold`, `lod_scale`, `coverage`,
//! `hash_probe`) are a fixed, non-reorderable sequence of multiplies, adds,
//! divides and one `sqrt`: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on those fields while the keep/discard `u32` flag is
//! compared for exact equality. Because the `floor`-driven level and lattice
//! selection are discontinuous at octave and cell boundaries, the parity
//! fixtures are rejection-sampled away from those cracks (see the parity test).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓
//! `prism_render_architecture::particle::alpha_hashed`；`Wyman`-style hashed
//! alpha-test threshold with a bit-pattern `log2` / `exp2` and `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::alpha_hashed::{
    alpha_hash_threshold, anisotropic_lod_scale, coverage_from_alpha, hash3, hashed_alpha_test,
    AlphaHashParams,
};
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

/// The portable core-`WGSL` hashed-alpha kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`alpha_hashed`](prism_render_architecture::particle::alpha_hashed) branch
/// for branch, including the bit-pattern `log2` / `exp2` and the integer
/// avalanche; see the module documentation for the algorithm.
const ALPHA_HASHED_WGSL: &str = r#"
// Hashed-alpha twin: one thread per query reproduces the Wyman-style stochastic
// alpha-test threshold. It mirrors the CPU golden particle::alpha_hashed branch
// for branch, uses only the portable core-WGSL subset (min/max/clamp/floor,
// + - * /, signed/unsigned bit ops, one sqrt, select and bitcast) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12. The log2,
// exp2 and the integer avalanche are verbatim replicas of the golden bit
// tricks, not log2/exp2/pow builtins.
//
// Provenance: twinned from this repository's particle::alpha_hashed; no
// third-party engine source or derived code.

// Comparison epsilon guarding every f32 denominator and clamp floor, matching
// the reference CMP_EPS so no f32 == / != is needed and no divide yields a NaN.
const CMP_EPS: f32 = 1.0e-6;

// Reciprocal of 2^24, normalizing a 24-bit hash mantissa into [0, 1). Matches
// the golden INV_2POW24.
const INV_2POW24: f32 = 1.0 / 16777216.0;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // World/object-space anchor the hash is pinned to.
    ax: f32,
    ay: f32,
    az: f32,
    // Screen-space anchor derivative along x.
    dx0: f32,
    dx1: f32,
    dx2: f32,
    // Screen-space anchor derivative along y.
    dy0: f32,
    dy1: f32,
    dy2: f32,
    // Fragment alpha under test.
    alpha: f32,
    // Spatial granularity of the hash noise.
    hash_scale: f32,
    // Lower clamp on the produced threshold.
    min_threshold: f32,
    // Seed decorrelating one dither pattern from another.
    seed: u32,
    // Raw lattice coordinates for the independent hash3 self-check.
    probe_x: i32,
    probe_y: i32,
    probe_z: i32,
}

struct Result {
    // Blended, clamped hashed threshold in [min_threshold, 1].
    threshold: f32,
    // Anisotropic LOD scale, max(len(ddx), len(ddy)).
    lod_scale: f32,
    // Keep (1) or discard (0) decision for alpha >= threshold.
    keep: u32,
    // Expected surviving coverage, clamp01(alpha).
    coverage: f32,
    // Independent hash3(probe, seed) self-check in [0, 1).
    hash_probe: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// One xor-shift / odd-multiply avalanche round, a verbatim replica of the
// golden mix. WGSL u32 arithmetic wraps modulo 2^32, matching wrapping_mul.
fn mix_hash(h: u32, value: u32) -> u32 {
    var x = h ^ (value * 0x9e3779b9u);
    x = x ^ (x >> 15u);
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    return x ^ (x >> 16u);
}

// Deterministic 3D spatial hash in [0, 1), a verbatim replica of the golden
// hash3: fold the three signed lattice indices (two's-complement reinterpret
// via bitcast) and the seed through the avalanche, then normalize the top 24
// bits with INV_2POW24. No transcendental is called.
fn hash3(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    var h = seed ^ 0x811c9dc5u;
    h = mix_hash(h, bitcast<u32>(x));
    h = mix_hash(h, bitcast<u32>(y));
    h = mix_hash(h, bitcast<u32>(z));
    let mantissa = h >> 8u;
    return f32(mantissa) * INV_2POW24;
}

// Rebuilds 2^level as an f32 by writing the biased exponent field, a verbatim
// replica of the golden exp2_pow. level is clamped to the normal-f32 exponent
// range so the shift can never construct a subnormal or an infinity. No exp2 /
// pow builtin is called.
fn exp2_pow(level: i32) -> f32 {
    let clamped = clamp(level, -126, 127);
    let biased = u32(clamped + 127);
    return bitcast<f32>(biased << 23u);
}

// Piecewise-linear log2(x) read from the f32 exponent and mantissa bits, a
// verbatim replica of the golden approx_log2: the exponent field gives the
// integer octave and the mantissa in [1, 2) supplies the linear fraction. Exact
// at every power of two; no log2 builtin is called. The caller floors the
// argument to CMP_EPS, so a non-positive input never reaches here.
fn approx_log2(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let exponent = i32((bits >> 23u) & 0xffu) - 127;
    let mantissa_bits = (bits & 0x007fffffu) | 0x3f800000u;
    let mantissa = bitcast<f32>(mantissa_bits);
    return f32(exponent) + (mantissa - 1.0);
}

// Quantizes an anchor component to an integer lattice index at a given scale
// via floor(coord * scale), a verbatim replica of the golden quantize_coord.
fn quantize_coord(coord: f32, scale: f32) -> i32 {
    return i32(floor(coord * scale));
}

// Anisotropic LOD scale: the larger of the two derivative lengths, a verbatim
// replica of the golden anisotropic_lod_scale. Uses only sqrt.
fn anisotropic_lod_scale(ddx: vec3<f32>, ddy: vec3<f32>) -> f32 {
    let len_x = sqrt(ddx.x * ddx.x + ddx.y * ddx.y + ddx.z * ddx.z);
    let len_y = sqrt(ddy.x * ddy.x + ddy.y * ddy.y + ddy.z * ddy.z);
    return max(len_x, len_y);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let anchor = vec3<f32>(q.ax, q.ay, q.az);
    let ddx = vec3<f32>(q.dx0, q.dx1, q.dx2);
    let ddy = vec3<f32>(q.dy0, q.dy1, q.dy2);

    let lod_scale = anisotropic_lod_scale(ddx, ddy);

    // alpha_hash_threshold, branch for branch with the golden.
    let scale = max(q.hash_scale, CMP_EPS);
    let footprint = max(lod_scale, CMP_EPS);
    // Pixel scale: how many hashed cells span the anchor footprint. Guarded so
    // the following log2 always sees a strictly positive argument.
    let pix_scale = max(1.0 / (scale * footprint), CMP_EPS);

    let level = approx_log2(pix_scale);
    let coarse_level = floor(level);
    let frac_level = level - coarse_level;
    let coarse = i32(coarse_level);

    let scale_lo = exp2_pow(coarse) * scale;
    let scale_hi = exp2_pow(coarse + 1) * scale;

    let h0 = hash3(
        quantize_coord(anchor.x, scale_lo),
        quantize_coord(anchor.y, scale_lo),
        quantize_coord(anchor.z, scale_lo),
        q.seed
    );
    let h1 = hash3(
        quantize_coord(anchor.x, scale_hi),
        quantize_coord(anchor.y, scale_hi),
        quantize_coord(anchor.z, scale_hi),
        q.seed
    );

    // two_level_lerp: a + (b - a) * t.
    let blended = h0 + (h1 - h0) * frac_level;
    let threshold_floor = clamp(q.min_threshold, 0.0, 1.0);
    let threshold = clamp(blended, threshold_floor, 1.0);

    // hashed_alpha_test keeps the fragment when alpha >= threshold (>=, so the
    // boundary tie is kept), coverage_from_alpha is clamp01(alpha).
    let keep = select(0u, 1u, q.alpha >= threshold);
    let coverage = clamp(q.alpha, 0.0, 1.0);

    // Independent integer-lattice hash3 self-check.
    let hash_probe = hash3(q.probe_x, q.probe_y, q.probe_z, q.seed);

    var out: Result;
    out.threshold = threshold;
    out.lod_scale = lod_scale;
    out.keep = keep;
    out.coverage = coverage;
    out.hash_probe = hash_probe;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One hashed-alpha query: the surface `anchor`, its screen-space `ddx` / `ddy`
/// derivatives, the fragment `alpha`, the three `AlphaHashParams` fields
/// (`hash_scale`, `min_threshold`, `seed`) and a raw integer lattice `probe` for
/// an independent `hash3` self-check — the same inputs the reference
/// [`alpha_hash_threshold`](prism_render_architecture::particle::alpha_hashed::alpha_hash_threshold),
/// [`anisotropic_lod_scale`](prism_render_architecture::particle::alpha_hashed::anisotropic_lod_scale),
/// [`hashed_alpha_test`](prism_render_architecture::particle::alpha_hashed::hashed_alpha_test),
/// [`coverage_from_alpha`](prism_render_architecture::particle::alpha_hashed::coverage_from_alpha)
/// and [`hash3`](prism_render_architecture::particle::alpha_hashed::hash3)
/// consume.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
/// source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaHashedQuery {
    /// Stable world/object-space anchor the hash is pinned to.
    pub anchor: [f32; 3],
    /// Anchor derivative with respect to screen x.
    pub ddx: [f32; 3],
    /// Anchor derivative with respect to screen y.
    pub ddy: [f32; 3],
    /// Fragment alpha under test.
    pub alpha: f32,
    /// Spatial granularity of the hash noise (`AlphaHashParams::hash_scale`).
    pub hash_scale: f32,
    /// Lower clamp on the produced threshold (`AlphaHashParams::min_threshold`).
    pub min_threshold: f32,
    /// Seed decorrelating one dither pattern from another
    /// (`AlphaHashParams::seed`).
    pub seed: u32,
    /// Raw integer lattice coordinates for the independent `hash3` self-check.
    pub probe: [i32; 3],
}

impl AlphaHashedQuery {
    /// Builds a query from the anchor, the two derivatives, the fragment
    /// `alpha`, the three `AlphaHashParams` fields and the raw lattice `probe`.
    ///
    /// Provenance: 孪生自本仓
    /// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
    /// source or derived code.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the golden anchor, derivative pair, alpha, the three AlphaHashParams fields and the hash3 probe as one query record"
    )]
    pub const fn new(
        anchor: [f32; 3],
        ddx: [f32; 3],
        ddy: [f32; 3],
        alpha: f32,
        hash_scale: f32,
        min_threshold: f32,
        seed: u32,
        probe: [i32; 3],
    ) -> AlphaHashedQuery {
        AlphaHashedQuery {
            anchor,
            ddx,
            ddy,
            alpha,
            hash_scale,
            min_threshold,
            seed,
            probe,
        }
    }
}

/// The resolved hashed-alpha answer for one query: the blended `threshold`, the
/// anisotropic `lod_scale`, the keep/discard decision, the expected `coverage`
/// and the independent `hash_probe` the reference exposes.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
/// source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaHashedResult {
    /// Blended, clamped hashed threshold in `[min_threshold, 1]`.
    pub threshold: f32,
    /// Anisotropic `LOD` scale, `max(len(ddx), len(ddy))`.
    pub lod_scale: f32,
    /// Keep (`true`) or discard (`false`) decision for `alpha >= threshold`.
    pub keep: bool,
    /// Expected surviving coverage, `clamp01(alpha)`.
    pub coverage: f32,
    /// Independent `hash3(probe, seed)` self-check in `[0, 1)`.
    pub hash_probe: f32,
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`anisotropic_lod_scale`](prism_render_architecture::particle::alpha_hashed::anisotropic_lod_scale),
/// [`alpha_hash_threshold`](prism_render_architecture::particle::alpha_hashed::alpha_hash_threshold),
/// [`hashed_alpha_test`](prism_render_architecture::particle::alpha_hashed::hashed_alpha_test),
/// [`coverage_from_alpha`](prism_render_architecture::particle::alpha_hashed::coverage_from_alpha)
/// and [`hash3`](prism_render_architecture::particle::alpha_hashed::hash3) so the
/// host side and the device twin are checked against the same source of truth.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
/// source or derived code.
#[must_use]
pub fn golden(query: &AlphaHashedQuery) -> AlphaHashedResult {
    let params = AlphaHashParams::new(query.hash_scale, query.min_threshold, query.seed);
    let lod_scale = anisotropic_lod_scale(query.ddx, query.ddy);
    let threshold = alpha_hash_threshold(query.anchor, lod_scale, &params);
    let keep = hashed_alpha_test(query.alpha, threshold);
    let coverage = coverage_from_alpha(query.alpha);
    let hash_probe = hash3(query.probe[0], query.probe[1], query.probe[2], query.seed);
    AlphaHashedResult {
        threshold,
        lod_scale,
        keep,
        coverage,
        hash_probe,
    }
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding the
/// anchor, the two derivatives, the fragment `alpha`, the three param fields and
/// the raw lattice probe — `64` bytes matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `anchor.x`.
    ax: f32,
    /// `anchor.y`.
    ay: f32,
    /// `anchor.z`.
    az: f32,
    /// `ddx.x`.
    dx0: f32,
    /// `ddx.y`.
    dx1: f32,
    /// `ddx.z`.
    dx2: f32,
    /// `ddy.x`.
    dy0: f32,
    /// `ddy.y`.
    dy1: f32,
    /// `ddy.z`.
    dy2: f32,
    /// Fragment alpha under test.
    alpha: f32,
    /// Spatial granularity of the hash noise.
    hash_scale: f32,
    /// Lower clamp on the produced threshold.
    min_threshold: f32,
    /// Seed decorrelating the dither pattern.
    seed: u32,
    /// Raw lattice coordinate `x` for the `hash3` self-check.
    probe_x: i32,
    /// Raw lattice coordinate `y` for the `hash3` self-check.
    probe_y: i32,
    /// Raw lattice coordinate `z` for the `hash3` self-check.
    probe_z: i32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &AlphaHashedQuery) -> GpuQuery {
        GpuQuery {
            ax: query.anchor[0],
            ay: query.anchor[1],
            az: query.anchor[2],
            dx0: query.ddx[0],
            dx1: query.ddx[1],
            dx2: query.ddx[2],
            dy0: query.ddy[0],
            dy1: query.ddy[1],
            dy2: query.ddy[2],
            alpha: query.alpha,
            hash_scale: query.hash_scale,
            min_threshold: query.min_threshold,
            seed: query.seed,
            probe_x: query.probe[0],
            probe_y: query.probe[1],
            probe_z: query.probe[2],
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(threshold, lod_scale, keep, coverage)` and `(hash_probe, pad, pad, pad)` —
/// `32` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Blended, clamped hashed threshold.
    threshold: f32,
    /// Anisotropic `LOD` scale.
    lod_scale: f32,
    /// Keep (`1`) or discard (`0`) flag.
    keep: u32,
    /// Expected surviving coverage.
    coverage: f32,
    /// Independent `hash3` self-check.
    hash_probe: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
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

/// Decodes one packed [`GpuResult`] into the public [`AlphaHashedResult`].
fn decode_result(raw: &GpuResult) -> AlphaHashedResult {
    AlphaHashedResult {
        threshold: raw.threshold,
        lod_scale: raw.lod_scale,
        keep: raw.keep != 0,
        coverage: raw.coverage,
        hash_probe: raw.hash_probe,
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

/// A compiled, reusable hashed-alpha compute pipeline.
///
/// Provenance: 孪生自本仓
/// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
/// source or derived code.
pub struct GpuAlphaHashed {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAlphaHashed {
    /// Compiles the hashed-alpha kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓
    /// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
    /// source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAlphaHashed {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_alpha_hashed"),
            source: ShaderSource::Wgsl(ALPHA_HASHED_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_alpha_hashed_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_alpha_hashed_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_alpha_hashed_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAlphaHashed {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`AlphaHashedResult`] per
    /// input, in order.
    ///
    /// Each continuous field equals the reference
    /// [`alpha_hashed`](prism_render_architecture::particle::alpha_hashed)
    /// answers to within the tolerance documented on this module, and the
    /// keep/discard flag matches exactly. An empty input returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓
    /// `prism_render_architecture::particle::alpha_hashed`；no third-party engine
    /// source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AlphaHashedQuery]) -> Vec<AlphaHashedResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_alpha_hashed_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_alpha_hashed_output"),
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
            label: Some("prism_volumetric_alpha_hashed_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_alpha_hashed_bind_group"),
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
            label: Some("prism_volumetric_alpha_hashed_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_alpha_hashed_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_alpha_hashed_pass"),
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
