//! `wgpu` compute twin of the life-driven *alpha erosion* (dissolve) golden
//! ([`alpha_erosion`](prism_render_architecture::particle::alpha_erosion),
//! particle design §17).
//!
//! The `CPU` golden
//! [`alpha_erosion`](prism_render_architecture::particle::alpha_erosion) fades a
//! particle out not with a flat alpha ramp but by *eroding* it against a
//! per-particle noise field: a rising threshold sweeps across the noise so the
//! sprite dissolves in irregular holes, with an emissive rim glowing along the
//! moving dissolve edge. The pipeline is: (1) a normalized age `t` in `0..=1`
//! drives an erosion `threshold` via
//! [`threshold_over_age`](prism_render_architecture::particle::alpha_erosion::threshold_over_age);
//! (2) a per-particle noise value `n` in `0..=1` comes from a self-contained
//! integer-hash pseudo-noise
//! ([`hash_noise01`](prism_render_architecture::particle::alpha_erosion::hash_noise01));
//! (3)
//! [`erosion_alpha`](prism_render_architecture::particle::alpha_erosion::erosion_alpha)
//! compares `n` against the threshold through a `smoothstep` soft edge; and (4)
//! [`edge_glow_factor`](prism_render_architecture::particle::alpha_erosion::edge_glow_factor)
//! lights a band-shaped rim right at the dissolve boundary.
//! [`ErosionParams::evaluate`](prism_render_architecture::particle::alpha_erosion::ErosionParams::evaluate)
//! wires these into an
//! [`ErosionSample`](prism_render_architecture::particle::alpha_erosion::ErosionSample).
//!
//! [`GpuAlphaErosion`] is the on-device twin: one thread per query reproduces
//! the same hash chain and the same dissolve algebra branch for branch, so a
//! passing real-device parity test is direct evidence the ported kernel mixes
//! the same bits and shades the same dissolve the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced: the raw
//! avalanche-mixed `u32` word (the private `hash_u32` mixer, reproduced
//! bit-for-bit in the portable `WGSL` `u32` lane), its `0..=1` normalization
//! [`hash_noise01`](prism_render_architecture::particle::alpha_erosion::hash_noise01),
//! and the full
//! [`ErosionSample`](prism_render_architecture::particle::alpha_erosion::ErosionSample)
//! (dissolve `alpha` and the rim-glow `glow_rgb`) that
//! [`ErosionParams::evaluate`](prism_render_architecture::particle::alpha_erosion::ErosionParams::evaluate)
//! produces for a supplied age `age` and per-particle noise `n`.
//!
//! # Correctness model
//!
//! The integer mix is exact: `WGSL` `u32` xor-shift and wrapping multiply match
//! the reference `wrapping_mul` and shift bit-for-bit, so the parity test
//! asserts an exact `==` on the hashed word. The hard-step branches
//! (`edge_width` below `MIN_EDGE`, or a `smoothstep` span below `MIN_EDGE`) are
//! discrete classifications, so for inputs clear of the tie the `CPU` and `GPU`
//! take the same branch. The continuous outputs (`noise01`, `alpha`,
//! `glow_rgb`) thread through multiplies, adds and one guarded division, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on
//! every continuous `f32`, tight enough to catch a genuinely wrong port (a
//! dropped term, a wrong constant, a mis-expanded `smoothstep`) yet loose enough
//! to admit legal fused multiply-add contraction.
//!
//! # Smoothstep expansion
//!
//! `WGSL` has a built-in `smoothstep`, but to stay bit-for-branch faithful to
//! the reference the kernel does not call it: it reproduces the reference's own
//! Hermite expansion `t * t * (3 - 2 * t)` over a `clamp`-ed, guarded `t`,
//! including the reference's degenerate collapse to a hard step when the edge
//! span is below `MIN_EDGE`.
//!
//! # Degenerate inputs
//!
//! A non-positive `edge_width` (below `MIN_EDGE`) collapses the soft edge to a
//! hard cutoff `n <= threshold ? 0 : 1` and yields no glow, exactly as the
//! reference does, so no division by a near-zero span can produce a `NaN`. The
//! age and the threshold are `clamp`-ed into `0..=1`, so an out-of-range age or
//! an inverted threshold pair stays well defined. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `+ - * /`, `u32` xor-shift and wrapping multiply, and a `u32`-to-`f32`
//! convert — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `sqrt`, no built-in `smoothstep` call and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
//! self-contained integer avalanche hash plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::alpha_erosion::{hash_noise01, ErosionParams};
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

/// The portable core-`WGSL` alpha-erosion kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`alpha_erosion`](prism_render_architecture::particle::alpha_erosion) branch
/// for branch; see the module documentation for the algorithm.
const ALPHA_EROSION_WGSL: &str = r#"
// Alpha-erosion (dissolve) twin: one thread per query reproduces the private
// hash_u32 avalanche mixer (bit-for-bit in the u32 lane), its hash_noise01
// normalization, and the full ErosionSample (dissolve alpha and rim glow). It
// mirrors the CPU golden particle::alpha_erosion branch for branch, uses only
// the portable core-WGSL subset (clamp/min/max and + - * / plus u32 xor-shift
// and wrapping multiply and a u32->f32 convert), needs no sqrt, no transcendental
// call and no built-in smoothstep, and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. There is no loop, so the kernel provably
// terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::alpha_erosion；无第三方
// 引擎源码或衍生代码。

// Minimum edge width (and generic denominator guard) below which a soft edge
// collapses to a hard step, so no division by zero can produce a NaN. Matches
// the reference MIN_EDGE; the compare rule used instead of an f32 ==.
const MIN_EDGE: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // vec4 slot 0: linear-RGB rim-glow colour and the glow intensity multiplier.
    glow_rgb: vec3<f32>,
    glow_intensity: f32,
    // vec4 slot 1: smoothstep edge width, the birth/death thresholds and a pad.
    edge_width: f32,
    threshold_start: f32,
    threshold_end: f32,
    pad0: f32,
    // vec4 slot 2: normalized age, per-particle noise, the hash seed and a pad.
    age: f32,
    n: f32,
    seed: u32,
    pad1: u32,
}

struct Result {
    // vec4 slot 0: the scaled rim-glow colour and the dissolve alpha.
    glow_rgb: vec3<f32>,
    alpha: f32,
    // vec4 slot 1: the normalized hash noise, the raw hashed word and two pads.
    noise01: f32,
    hashed: u32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamps a scalar into the 0..=1 range without an f32 equality compare;
// mirrors the reference clamp01.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep from low to high evaluated at x, reproducing the
// reference's own t * t * (3 - 2 * t) expansion rather than the WGSL built-in.
// A degenerate (near-equal) interval collapses to a hard step at high rather
// than dividing by zero, matching the reference smoothstep.
fn soft_step(low: f32, high: f32, x: f32) -> f32 {
    let span = high - low;
    if (span < MIN_EDGE) {
        if (x < high) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp01((x - low) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Integer avalanche hash mixing a u32 seed into a well-distributed u32: a
// sequence of xor-shifts and odd-constant wrapping multiplies, reproducing the
// reference private hash_u32 bit-for-bit (WGSL u32 arithmetic wraps modulo
// 2^32 and >> is a logical shift, matching the reference wrapping_mul and >>).
fn hash_u32(seed: u32) -> u32 {
    var x: u32 = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Erosion threshold at normalized age t, biased from start to end. Both t and
// the result clamp into 0..=1; mirrors the reference threshold_over_age.
fn threshold_over_age(t: f32, low: f32, high: f32) -> f32 {
    let tt = clamp01(t);
    return clamp01(low + (high - low) * tt);
}

// Dissolve alpha for noise n against threshold with a smoothstep edge. A
// non-positive edge_width degenerates to a hard cutoff at threshold; mirrors
// the reference erosion_alpha.
fn erosion_alpha(n: f32, threshold: f32, edge_width: f32) -> f32 {
    if (edge_width < MIN_EDGE) {
        if (n <= threshold) {
            return 0.0;
        }
        return 1.0;
    }
    return soft_step(threshold, threshold + edge_width, n);
}

// Rim-glow weight in 0..=1 for the dissolve edge window of n: a band formed by
// subtracting a rising smoothstep from an earlier rising smoothstep, zero at
// both ends of the window and peaking at its centre. A non-positive edge_width
// yields no glow; mirrors the reference edge_glow_factor.
fn edge_glow_factor(n: f32, threshold: f32, edge_width: f32) -> f32 {
    if (edge_width < MIN_EDGE) {
        return 0.0;
    }
    let u = clamp01((n - threshold) / edge_width);
    let rising = soft_step(0.0, 0.5, u);
    let falling = soft_step(0.5, 1.0, u);
    return clamp01(rising - falling);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Dissolve chain: advance the threshold by age, erode the alpha against the
    // noise, and light the rim glow scaled by intensity into the glow colour.
    let threshold = threshold_over_age(q.age, q.threshold_start, q.threshold_end);
    let alpha = erosion_alpha(q.n, threshold, q.edge_width);
    let glow = edge_glow_factor(q.n, threshold, q.edge_width) * q.glow_intensity;

    // Hash chain: mix the seed to a well-distributed word and normalize it.
    let hashed = hash_u32(q.seed);
    let noise01 = f32(hashed) / f32(0xffffffffu);

    var out: Result;
    out.glow_rgb = vec3<f32>(
        q.glow_rgb.x * glow,
        q.glow_rgb.y * glow,
        q.glow_rgb.z * glow,
    );
    out.alpha = alpha;
    out.noise01 = noise01;
    out.hashed = hashed;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct matching `Params` in
/// [`ALPHA_EROSION_WGSL`].
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

/// `repr(C)` `std430` layout of one query: three `vec4` slots holding
/// `(glow_rgb, glow_intensity)`, `(edge_width, threshold_start, threshold_end,
/// pad)` and `(age, n, seed, pad)` — `48` bytes, each `vec3` on its
/// `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it. The
/// first two slots mirror the reference
/// [`ErosionParams::to_std430`](prism_render_architecture::particle::alpha_erosion::ErosionParams::to_std430)
/// packing.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Linear-`RGB` rim-glow colour.
    glow_rgb: [f32; 3],
    /// Rim-glow intensity multiplier, packed in the fourth lane of slot `0`.
    glow_intensity: f32,
    /// Width of the `smoothstep` dissolve edge in noise units.
    edge_width: f32,
    /// Erosion threshold at birth (normalized age `0`).
    threshold_start: f32,
    /// Erosion threshold at death (normalized age `1`).
    threshold_end: f32,
    /// Padding lane after the thresholds.
    pad0: f32,
    /// Normalized particle age in `0..=1`.
    age: f32,
    /// Per-particle noise value `n` in `0..=1`.
    n: f32,
    /// Integer seed fed to the avalanche hash.
    seed: u32,
    /// Padding word after the seed.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(glow_rgb, alpha)` and `(noise01, hashed, pad, pad)` — `32` bytes matching
/// the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Scaled linear-`RGB` rim-glow contribution.
    glow_rgb: [f32; 3],
    /// Dissolve alpha, packed in the fourth lane of slot `0`.
    alpha: f32,
    /// Per-particle hash noise in `0..=1`.
    noise01: f32,
    /// Raw avalanche-mixed `u32` word.
    hashed: u32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

/// One alpha-erosion query: the hash seed plus the age, the per-particle noise
/// and the erosion parameters the reference
/// [`ErosionParams::evaluate`](prism_render_architecture::particle::alpha_erosion::ErosionParams::evaluate)
/// consumes.
///
/// A single query exercises both twinned halves at once: the `seed`-driven hash
/// chain and the `(age, n)`-driven dissolve shading.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaErosionQuery {
    /// Integer seed fed to the avalanche hash for the per-particle noise.
    pub seed: u32,
    /// Normalized particle age in `0..=1` driving the erosion threshold.
    pub age: f32,
    /// Per-particle noise value `n` in `0..=1` eroded against the threshold.
    pub n: f32,
    /// Width of the `smoothstep` dissolve edge in noise units.
    pub edge_width: f32,
    /// Linear-`RGB` colour of the dissolve rim glow.
    pub glow_color: [f32; 3],
    /// Scalar multiplier applied to the rim-glow band weight.
    pub glow_intensity: f32,
    /// Erosion threshold at birth (normalized age `0`).
    pub threshold_start: f32,
    /// Erosion threshold at death (normalized age `1`).
    pub threshold_end: f32,
}

impl AlphaErosionQuery {
    /// Builds a query from the hash seed, the age, the per-particle noise and
    /// the erosion parameters.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
    /// no third-party engine source or derived code.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the reference ErosionParams fields plus the per-query seed, age and noise"
    )]
    pub const fn new(
        seed: u32,
        age: f32,
        n: f32,
        edge_width: f32,
        glow_color: [f32; 3],
        glow_intensity: f32,
        threshold_start: f32,
        threshold_end: f32,
    ) -> AlphaErosionQuery {
        AlphaErosionQuery {
            seed,
            age,
            n,
            edge_width,
            glow_color,
            glow_intensity,
            threshold_start,
            threshold_end,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports: the dissolve
/// [`ErosionSample`](prism_render_architecture::particle::alpha_erosion::ErosionSample)
/// and the hash chain.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaErosionResult {
    /// Dissolve alpha in `0..=1`, matching
    /// [`ErosionSample::alpha`](prism_render_architecture::particle::alpha_erosion::ErosionSample::alpha).
    pub alpha: f32,
    /// Emissive rim colour contribution in linear `RGB`, matching
    /// [`ErosionSample::glow_rgb`](prism_render_architecture::particle::alpha_erosion::ErosionSample::glow_rgb).
    pub glow_rgb: [f32; 3],
    /// Per-particle hash noise in `0..=1`, matching
    /// [`hash_noise01`](prism_render_architecture::particle::alpha_erosion::hash_noise01).
    pub noise01: f32,
    /// Raw avalanche-mixed `u32` word (the reference private `hash_u32` output),
    /// compared bit-exactly in the parity test.
    pub hashed: u32,
}

/// Reproduces the reference private `hash_u32` avalanche mixer on the host so
/// the device twin's raw `u32` word can be pinned bit-for-bit.
///
/// The reference `hash_u32` is private, so it is reproduced here from the
/// documented sequence; it is anchored to the public golden through
/// [`hash_noise01`], since `host_hash_u32(seed) as f32 / u32::MAX as f32` is
/// exactly `hash_noise01(seed)` by the reference's own construction.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
/// no third-party engine source or derived code.
#[must_use]
fn host_hash_u32(seed: u32) -> u32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`ErosionParams::evaluate`](prism_render_architecture::particle::alpha_erosion::ErosionParams::evaluate)
/// and
/// [`hash_noise01`](prism_render_architecture::particle::alpha_erosion::hash_noise01)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &AlphaErosionQuery) -> AlphaErosionResult {
    let params = ErosionParams::new(
        query.edge_width,
        query.glow_color,
        query.glow_intensity,
        query.threshold_start,
        query.threshold_end,
    );
    let sample = params.evaluate(query.age, query.n);
    AlphaErosionResult {
        alpha: sample.alpha,
        glow_rgb: sample.glow_rgb,
        noise01: hash_noise01(query.seed),
        hashed: host_hash_u32(query.seed),
    }
}

/// Encodes one [`AlphaErosionQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(query: &AlphaErosionQuery) -> GpuQuery {
    GpuQuery {
        glow_rgb: query.glow_color,
        glow_intensity: query.glow_intensity,
        edge_width: query.edge_width,
        threshold_start: query.threshold_start,
        threshold_end: query.threshold_end,
        pad0: 0.0,
        age: query.age,
        n: query.n,
        seed: query.seed,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`AlphaErosionResult`].
fn decode_result(raw: &GpuResult) -> AlphaErosionResult {
    AlphaErosionResult {
        alpha: raw.alpha,
        glow_rgb: raw.glow_rgb,
        noise01: raw.noise01,
        hashed: raw.hashed,
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

/// A compiled, reusable alpha-erosion compute pipeline, twinning the `CPU`
/// golden
/// [`alpha_erosion`](prism_render_architecture::particle::alpha_erosion).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
/// no third-party engine source or derived code.
pub struct GpuAlphaErosion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAlphaErosion {
    /// Compiles the alpha-erosion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAlphaErosion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_alpha_erosion"),
            source: ShaderSource::Wgsl(ALPHA_EROSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_alpha_erosion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_alpha_erosion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_alpha_erosion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAlphaErosion {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`AlphaErosionResult`] per
    /// input, in order.
    ///
    /// The hashed `u32` word equals the reference exactly; the continuous
    /// `noise01`, `alpha` and `glow_rgb` match the reference to within the
    /// tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::alpha_erosion`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[AlphaErosionQuery]) -> Vec<AlphaErosionResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_alpha_erosion_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_alpha_erosion_output"),
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
            label: Some("prism_volumetric_alpha_erosion_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_alpha_erosion_bind_group"),
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
            label: Some("prism_volumetric_alpha_erosion_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_alpha_erosion_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_alpha_erosion_pass"),
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
