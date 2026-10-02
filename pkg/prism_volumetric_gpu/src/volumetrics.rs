//! `wgpu` compute twin of the volumetric evaluation / baking golden
//! ([`volumetrics`](prism_render_architecture::particle::volumetrics), particle
//! design §20, the phase / six-way / deep-opacity numeric core).
//!
//! The `CPU` golden
//! [`volumetrics`](prism_render_architecture::particle::volumetrics) owns the
//! arithmetic that bakes and evaluates the §20 volumetric contracts declared in
//! [`shading`](prism_render_architecture::particle::shading): the single-lobe
//! Henyey-Greenstein phase
//! ([`henyey_greenstein`](prism_render_architecture::particle::volumetrics::henyey_greenstein)),
//! its double-lobe blend
//! ([`double_lobe_phase`](prism_render_architecture::particle::volumetrics::double_lobe_phase)),
//! the `NPR` cel-banded response
//! ([`phase_response_banded`](prism_render_architecture::particle::volumetrics::phase_response_banded)),
//! the six-way luminance deposit of a single light sample
//! ([`SixWayAccumulator::add_light`](prism_render_architecture::particle::volumetrics::SixWayAccumulator::add_light)),
//! the directional read-back of a baked rig
//! ([`six_way_response`](prism_render_architecture::particle::shading::six_way_response)),
//! and the per-bracket interpolation of a layered `deep opacity map`
//! ([`sample_deep_transmittance`](prism_render_architecture::particle::volumetrics::sample_deep_transmittance)).
//! Every one of those answers is a closed-form sequence of `+ - * /`, `abs`,
//! `min`, `max`, `clamp`, `floor` and one guarded `sqrt`, so the whole numeric
//! core ports to a portable core-`WGSL` kernel with no transcendental call —
//! the `HG` denominator `(1 + g² − 2·g·cosθ)^1.5` factors as `d · sqrt(d)`.
//!
//! [`GpuVolumetrics`] is the on-device twin: one thread per [`VolumetricsQuery`]
//! reproduces the matching golden answer, so a passing real-device parity test
//! is direct evidence the ported kernel evaluates the same phase polynomials,
//! folds the same six-way deposit and snaps the same cel bands the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Per query, exactly the golden routines:
//! [`henyey_greenstein`](prism_render_architecture::particle::volumetrics::henyey_greenstein),
//! [`double_lobe_phase`](prism_render_architecture::particle::volumetrics::double_lobe_phase),
//! [`phase_response_banded`](prism_render_architecture::particle::volumetrics::phase_response_banded)
//! (which snaps through
//! [`quantize_cel_bands`](prism_render_architecture::particle::shading::quantize_cel_bands)),
//! [`six_way_response`](prism_render_architecture::particle::shading::six_way_response),
//! the single-sample
//! [`SixWayAccumulator::add_light`](prism_render_architecture::particle::volumetrics::SixWayAccumulator::add_light)
//! deposit baked into a
//! [`SixWayLuminance`](prism_render_architecture::particle::shading::SixWayLuminance),
//! and the per-bracket linear interpolation inside
//! [`sample_deep_transmittance`](prism_render_architecture::particle::volumetrics::sample_deep_transmittance).
//!
//! # What is not twinned
//!
//! The host-only, list-and-grid parts of the golden are deliberately left on the
//! `CPU`: the
//! [`FroxelGrid`](prism_render_architecture::particle::volumetrics::FroxelGrid)
//! and
//! [`FroxelDensityField`](prism_render_architecture::particle::volumetrics::FroxelDensityField)
//! `usize` cell indexing and bounds, the multi-sample accumulation loop of
//! [`SixWayAccumulator`](prism_render_architecture::particle::volumetrics::SixWayAccumulator)
//! (the host reduces many samples; the twin deposits one), and the
//! bracket-search traversal of
//! [`sample_deep_transmittance`](prism_render_architecture::particle::volumetrics::sample_deep_transmittance)
//! over the variable-length layer slice (the host finds the bracketing window;
//! the twin runs only the per-bracket lerp). Those are index arithmetic and a
//! variable-length search, not the per-element math a one-thread-per-element
//! kernel is for.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `floor`, `+ - * /` and one guarded `sqrt` — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep` builtin
//! and no optional device feature, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. The `^1.5` power is the golden's own `d · sqrt(d)`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! guarded divide / `sqrt`, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! channels, tight enough to catch a genuinely wrong port yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::volumetrics`
//! 与 `prism_render_architecture::particle::shading`；纯整数/无超越数学，无需外部
//! 数学库；无第三方引擎源码或衍生代码。

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::shading::{
    quantize_cel_bands, six_way_response, PhaseParams, SixWayLuminance,
};
use prism_render_architecture::particle::volumetrics::{
    double_lobe_phase, henyey_greenstein, phase_response_banded, SixWayAccumulator,
};
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

/// Op code for the single-lobe Henyey-Greenstein phase.
const OP_HENYEY_GREENSTEIN: u32 = 0;
/// Op code for the double-lobe (front + back) phase blend.
const OP_DOUBLE_LOBE_PHASE: u32 = 1;
/// Op code for the `NPR` cel-banded phase response.
const OP_PHASE_RESPONSE_BANDED: u32 = 2;
/// Op code for the six-way directional read-back.
const OP_SIX_WAY_RESPONSE: u32 = 3;
/// Op code for the single-sample six-way luminance deposit.
const OP_SIX_WAY_DEPOSIT: u32 = 4;
/// Op code for the per-bracket deep-opacity transmittance lerp.
const OP_DEEP_TRANSMITTANCE_LERP: u32 = 5;

/// The portable core-`WGSL` volumetric kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` dispatches on an op
/// code to mirror each golden
/// [`volumetrics`](prism_render_architecture::particle::volumetrics) routine for
/// routine; see the module documentation for the algorithm.
const VOLUMETRICS_WGSL: &str = r#"
// Volumetrics twin: one thread per query reproduces the Henyey-Greenstein phase
// (single and double lobe), the NPR cel-banded response, the six-way deposit and
// directional read-back and the deep-opacity per-bracket lerp. It mirrors the
// CPU golden volumetrics / shading routines for routine.
//
// Portability: only the core subset (abs, min, max, clamp, floor, + - * / and
// one guarded sqrt) is used; no sin/cos/exp/log/pow/tan, no inverse
// trigonometry, no smoothstep builtin and no optional device feature, so it
// runs unmodified on Metal, Vulkan and DX12. The ^1.5 power is the golden's own
// d * sqrt(d).
//
// Provenance: twinned from this repository's particle::volumetrics and
// particle::shading; no third-party engine source or derived code.

// 4*pi, the Henyey-Greenstein normalization constant, as the same f32 literal
// the golden FOUR_PI uses (4.0 * PI rounded to f32).
const FOUR_PI: f32 = 12.566371;

// General-purpose magnitude floor guarding divisions and near-zero spans,
// matching the reference EPS.
const EPS: f32 = 1.0e-6;

// Squared-length floor for the normalize-or-zero guard, matching the reference
// EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1.0e-12;

const OP_HENYEY_GREENSTEIN: u32 = 0u;
const OP_DOUBLE_LOBE_PHASE: u32 = 1u;
const OP_PHASE_RESPONSE_BANDED: u32 = 2u;
const OP_SIX_WAY_RESPONSE: u32 = 3u;
const OP_SIX_WAY_DEPOSIT: u32 = 4u;
const OP_DEEP_TRANSMITTANCE_LERP: u32 = 5u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Op code selecting the routine.
    op: u32,
    // Cel-band count for the banded response op.
    bands: u32,
    pad0: u32,
    pad1: u32,
    // Scalar parameters, meaning per op (g/cos_theta/back params/peak,
    // light_dir xyz + intensity, deep-opacity lo/hi/depth).
    s: array<f32, 8>,
    // Six baked luminance buckets (right, left, up, down, front, back) for the
    // directional read-back op; two trailing pad lanes.
    lum: array<f32, 8>,
}

struct Result {
    // Single-scalar output (phase value, banded response, six-way response,
    // transmittance).
    scalar: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    // Six-bucket output (right, left, up, down, front, back) for the deposit op;
    // two trailing pad lanes.
    six: array<f32, 8>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector along v, or the zero vector when v is numerically zero, so
// normalization never yields NaN. Mirrors the reference normalize_or_zero.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Single-lobe Henyey-Greenstein phase, mirroring the reference
// henyey_greenstein. The ^1.5 power is d * sqrt(d); the base is floored at EPS
// so the grazing case stays finite instead of dividing by zero.
fn henyey_greenstein(g: f32, cos_theta: f32) -> f32 {
    let g2 = g * g;
    let base = 1.0 + g2 - 2.0 * g * cos_theta;
    var d = EPS;
    if (base > EPS) {
        d = base;
    }
    let d15 = d * sqrt(d);
    return (1.0 - g2) / (FOUR_PI * d15);
}

// Clamp of the back-lobe weight into [0, 1] using the golden's exact if-chain
// (so a NaN collapses to 0.0 like the reference), mirroring double_lobe_phase.
fn clamp_back_weight(w: f32) -> f32 {
    if (w > 1.0) {
        return 1.0;
    }
    if (w > 0.0) {
        return w;
    }
    return 0.0;
}

// Double-lobe (front + back) phase blend, mirroring the reference
// double_lobe_phase.
fn double_lobe_phase(g: f32, back_lobe_weight: f32, back_g: f32, cos_theta: f32) -> f32 {
    let w = clamp_back_weight(back_lobe_weight);
    let front = henyey_greenstein(g, cos_theta);
    let back = henyey_greenstein(back_g, cos_theta);
    return (1.0 - w) * front + w * back;
}

// Cel-band quantization, mirroring the reference quantize_cel_bands. The clamp
// uses the golden's if-chain so a NaN collapses to 0.0 (the low band); bands <= 1
// returns the clamped value unchanged.
fn quantize_cel_bands(response: f32, bands: u32) -> f32 {
    var clamped = 0.0;
    if (response > 1.0) {
        clamped = 1.0;
    } else if (response > 0.0) {
        clamped = response;
    } else {
        clamped = 0.0;
    }
    if (bands <= 1u) {
        return clamped;
    }
    let steps = f32(bands);
    var idx = floor(clamped * steps);
    let top = steps - 1.0;
    if (idx > top) {
        idx = top;
    }
    return idx / top;
}

// NPR cel-banded phase response, mirroring the reference phase_response_banded.
// A peak at or below EPS is treated as unit normalization.
fn phase_response_banded(
    g: f32,
    back_lobe_weight: f32,
    back_g: f32,
    cos_theta: f32,
    peak: f32,
    bands: u32,
) -> f32 {
    let response = double_lobe_phase(g, back_lobe_weight, back_g, cos_theta);
    var normalized = response;
    if (peak > EPS) {
        normalized = response / peak;
    }
    return quantize_cel_bands(normalized, bands);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let qd = queries[idx];
    var r: Result;
    r.scalar = 0.0;
    r.pad0 = 0.0;
    r.pad1 = 0.0;
    r.pad2 = 0.0;
    r.six = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    let op = qd.op;

    if (op == OP_HENYEY_GREENSTEIN) {
        r.scalar = henyey_greenstein(qd.s[0], qd.s[1]);
    } else if (op == OP_DOUBLE_LOBE_PHASE) {
        r.scalar = double_lobe_phase(qd.s[0], qd.s[1], qd.s[2], qd.s[3]);
    } else if (op == OP_PHASE_RESPONSE_BANDED) {
        r.scalar = phase_response_banded(
            qd.s[0],
            qd.s[1],
            qd.s[2],
            qd.s[3],
            qd.s[4],
            qd.bands,
        );
    } else if (op == OP_SIX_WAY_RESPONSE) {
        let d = normalize_or_zero(vec3<f32>(qd.s[0], qd.s[1], qd.s[2]));
        var x = 0.0;
        if (d.x >= 0.0) {
            x = d.x * qd.lum[0];
        } else {
            x = -d.x * qd.lum[1];
        }
        var y = 0.0;
        if (d.y >= 0.0) {
            y = d.y * qd.lum[2];
        } else {
            y = -d.y * qd.lum[3];
        }
        var z = 0.0;
        if (d.z >= 0.0) {
            z = d.z * qd.lum[4];
        } else {
            z = -d.z * qd.lum[5];
        }
        r.scalar = x + y + z;
    } else if (op == OP_SIX_WAY_DEPOSIT) {
        var energy = 0.0;
        if (qd.s[3] > 0.0) {
            energy = qd.s[3];
        }
        let d = normalize_or_zero(vec3<f32>(qd.s[0], qd.s[1], qd.s[2]));
        var right = 0.0;
        var left = 0.0;
        if (d.x >= 0.0) {
            right = d.x * energy;
        } else {
            left = -d.x * energy;
        }
        var up = 0.0;
        var down = 0.0;
        if (d.y >= 0.0) {
            up = d.y * energy;
        } else {
            down = -d.y * energy;
        }
        var front = 0.0;
        var back = 0.0;
        if (d.z >= 0.0) {
            front = d.z * energy;
        } else {
            back = -d.z * energy;
        }
        r.six = array<f32, 8>(right, left, up, down, front, back, 0.0, 0.0);
    } else {
        // OP_DEEP_TRANSMITTANCE_LERP: per-bracket lerp between two stored layers.
        let lo_depth = qd.s[0];
        let lo_t = qd.s[1];
        let hi_depth = qd.s[2];
        let hi_t = qd.s[3];
        let depth = qd.s[4];
        let span = hi_depth - lo_depth;
        if (span > EPS) {
            let t = (depth - lo_depth) / span;
            r.scalar = lo_t + t * (hi_t - lo_t);
        } else {
            r.scalar = hi_t;
        }
    }

    results[idx] = r;
}
"#;

/// One volumetric query: the routine and its inputs.
///
/// Each variant mirrors one golden
/// [`volumetrics`](prism_render_architecture::particle::volumetrics) (or
/// [`shading`](prism_render_architecture::particle::shading)) routine. The
/// six-way variants carry a single light sample or a baked rig, since the twin
/// runs only the per-sample deposit or the directional read-back the host feeds
/// it; the deep-opacity variant carries one host-selected bracket.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::volumetrics`
/// 与 `prism_render_architecture::particle::shading`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VolumetricsQuery {
    /// Evaluate the single-lobe Henyey-Greenstein phase
    /// ([`henyey_greenstein`](prism_render_architecture::particle::volumetrics::henyey_greenstein)).
    HenyeyGreenstein {
        /// Anisotropy `g` in `(-1, 1)` (forward for `g > 0`).
        g: f32,
        /// Cosine of the angle between the incoming and outgoing directions.
        cos_theta: f32,
    },
    /// Evaluate the double-lobe (front + back) phase blend
    /// ([`double_lobe_phase`](prism_render_architecture::particle::volumetrics::double_lobe_phase)).
    DoubleLobePhase {
        /// Primary forward-lobe anisotropy `g`.
        g: f32,
        /// Back-lobe weight in `0..=1` (clamped internally).
        back_lobe_weight: f32,
        /// Back-lobe anisotropy (typically negative).
        back_g: f32,
        /// Cosine of the scattering angle.
        cos_theta: f32,
    },
    /// Evaluate the `NPR` cel-banded phase response
    /// ([`phase_response_banded`](prism_render_architecture::particle::volumetrics::phase_response_banded)).
    PhaseResponseBanded {
        /// Primary forward-lobe anisotropy `g`.
        g: f32,
        /// Back-lobe weight in `0..=1` (clamped internally).
        back_lobe_weight: f32,
        /// Back-lobe anisotropy (typically negative).
        back_g: f32,
        /// Cosine of the scattering angle.
        cos_theta: f32,
        /// Normalization peak; at or below `EPS` means unit normalization.
        peak: f32,
        /// Number of discrete cel bands (`<= 1` disables banding).
        bands: u32,
    },
    /// Read the baked six-way rig for a light direction
    /// ([`six_way_response`](prism_render_architecture::particle::shading::six_way_response)).
    SixWayResponse {
        /// Baked luminance buckets `[right, left, up, down, front, back]`.
        luminance: [f32; 6],
        /// Light direction (need not be normalized; a zero direction yields `0`).
        light_dir: [f32; 3],
    },
    /// Deposit a single light sample into the six axis buckets
    /// ([`SixWayAccumulator::add_light`](prism_render_architecture::particle::volumetrics::SixWayAccumulator::add_light)).
    SixWayDeposit {
        /// Light direction (need not be normalized; a zero direction is ignored).
        light_dir: [f32; 3],
        /// Light intensity; a negative value is clamped to zero.
        intensity: f32,
    },
    /// Interpolate transmittance within one host-selected deep-opacity bracket
    /// ([`sample_deep_transmittance`](prism_render_architecture::particle::volumetrics::sample_deep_transmittance)).
    DeepTransmittanceLerp {
        /// Depth of the near (lower) bracket layer.
        lo_depth: f32,
        /// Transmittance of the near (lower) bracket layer.
        lo_transmittance: f32,
        /// Depth of the far (upper) bracket layer.
        hi_depth: f32,
        /// Transmittance of the far (upper) bracket layer.
        hi_transmittance: f32,
        /// Query depth inside the bracket.
        depth: f32,
    },
}

/// The resolved answer for one [`VolumetricsQuery`], one variant per query kind.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::volumetrics`
/// 与 `prism_render_architecture::particle::shading`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VolumetricsResult {
    /// The single-lobe phase value, matching `henyey_greenstein`.
    HenyeyGreenstein(f32),
    /// The double-lobe phase value, matching `double_lobe_phase`.
    DoubleLobePhase(f32),
    /// The cel-banded response, matching `phase_response_banded`.
    PhaseResponseBanded(f32),
    /// The directional six-way response, matching `six_way_response`.
    SixWayResponse(f32),
    /// The deposited axis buckets `[right, left, up, down, front, back]`,
    /// matching one `SixWayAccumulator::add_light` on a fresh accumulator.
    SixWayDeposit([f32; 6]),
    /// The interpolated bracket transmittance, matching the inner lerp of
    /// `sample_deep_transmittance`.
    DeepTransmittanceLerp(f32),
}

/// `repr(C)` `std430` layout of one packed query: two `u32` control words
/// (`op` and `bands`) plus two pads, an eight-lane `f32` scalar block and an
/// eight-lane luminance block — `80` bytes, matching the `WGSL` `Query` struct
/// lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Op code selecting the routine.
    op: u32,
    /// Cel-band count for the banded response op.
    bands: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Scalar parameters, meaning per op.
    s: [f32; 8],
    /// Six baked luminance buckets plus two pad lanes.
    lum: [f32; 8],
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &VolumetricsQuery) -> GpuQuery {
        let mut g = GpuQuery::zeroed();
        match query {
            VolumetricsQuery::HenyeyGreenstein { g: gg, cos_theta } => {
                g.op = OP_HENYEY_GREENSTEIN;
                g.s[0] = *gg;
                g.s[1] = *cos_theta;
            }
            VolumetricsQuery::DoubleLobePhase {
                g: gg,
                back_lobe_weight,
                back_g,
                cos_theta,
            } => {
                g.op = OP_DOUBLE_LOBE_PHASE;
                g.s[0] = *gg;
                g.s[1] = *back_lobe_weight;
                g.s[2] = *back_g;
                g.s[3] = *cos_theta;
            }
            VolumetricsQuery::PhaseResponseBanded {
                g: gg,
                back_lobe_weight,
                back_g,
                cos_theta,
                peak,
                bands,
            } => {
                g.op = OP_PHASE_RESPONSE_BANDED;
                g.s[0] = *gg;
                g.s[1] = *back_lobe_weight;
                g.s[2] = *back_g;
                g.s[3] = *cos_theta;
                g.s[4] = *peak;
                g.bands = *bands;
            }
            VolumetricsQuery::SixWayResponse {
                luminance,
                light_dir,
            } => {
                g.op = OP_SIX_WAY_RESPONSE;
                g.s[0] = light_dir[0];
                g.s[1] = light_dir[1];
                g.s[2] = light_dir[2];
                g.lum[0] = luminance[0];
                g.lum[1] = luminance[1];
                g.lum[2] = luminance[2];
                g.lum[3] = luminance[3];
                g.lum[4] = luminance[4];
                g.lum[5] = luminance[5];
            }
            VolumetricsQuery::SixWayDeposit {
                light_dir,
                intensity,
            } => {
                g.op = OP_SIX_WAY_DEPOSIT;
                g.s[0] = light_dir[0];
                g.s[1] = light_dir[1];
                g.s[2] = light_dir[2];
                g.s[3] = *intensity;
            }
            VolumetricsQuery::DeepTransmittanceLerp {
                lo_depth,
                lo_transmittance,
                hi_depth,
                hi_transmittance,
                depth,
            } => {
                g.op = OP_DEEP_TRANSMITTANCE_LERP;
                g.s[0] = *lo_depth;
                g.s[1] = *lo_transmittance;
                g.s[2] = *hi_depth;
                g.s[3] = *hi_transmittance;
                g.s[4] = *depth;
            }
        }
        g
    }
}

/// `repr(C)` `std430` layout of one result: one scalar output plus three pads
/// and an eight-lane six-bucket block — `48` bytes, matching the `WGSL`
/// `Result` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Single-scalar output.
    scalar: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
    /// Six-bucket output plus two pad lanes.
    six: [f32; 8],
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

/// Decodes one packed [`GpuResult`] into the public [`VolumetricsResult`]
/// matching the originating `query`'s variant.
fn decode_result(query: &VolumetricsQuery, raw: &GpuResult) -> VolumetricsResult {
    match query {
        VolumetricsQuery::HenyeyGreenstein { .. } => {
            VolumetricsResult::HenyeyGreenstein(raw.scalar)
        }
        VolumetricsQuery::DoubleLobePhase { .. } => VolumetricsResult::DoubleLobePhase(raw.scalar),
        VolumetricsQuery::PhaseResponseBanded { .. } => {
            VolumetricsResult::PhaseResponseBanded(raw.scalar)
        }
        VolumetricsQuery::SixWayResponse { .. } => VolumetricsResult::SixWayResponse(raw.scalar),
        VolumetricsQuery::SixWayDeposit { .. } => VolumetricsResult::SixWayDeposit([
            raw.six[0], raw.six[1], raw.six[2], raw.six[3], raw.six[4], raw.six[5],
        ]),
        VolumetricsQuery::DeepTransmittanceLerp { .. } => {
            VolumetricsResult::DeepTransmittanceLerp(raw.scalar)
        }
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

/// A compiled, reusable volumetric compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::volumetrics`
/// 与 `prism_render_architecture::particle::shading`；无第三方引擎源码或衍生代码。
pub struct GpuVolumetrics {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVolumetrics {
    /// Compiles the volumetric kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVolumetrics {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_volumetrics"),
            source: ShaderSource::Wgsl(VOLUMETRICS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_volumetrics_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_volumetrics_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_volumetrics_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVolumetrics {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`VolumetricsResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answer for the query's variant to within
    /// the tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[VolumetricsQuery]) -> Vec<VolumetricsResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_volumetrics_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_volumetrics_output"),
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
            label: Some("prism_volumetric_volumetrics_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_volumetrics_bind_group"),
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
            label: Some("prism_volumetric_volumetrics_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_volumetrics_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_volumetrics_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}

/// Compile-time proof that the public re-exports stay reachable: the golden
/// routines this twin mirrors are named here so a rename upstream fails the
/// build rather than silently drifting.
#[doc(hidden)]
fn _golden_anchor() {
    let _ = henyey_greenstein;
    let _ = double_lobe_phase;
    let _ = phase_response_banded;
    let _ = six_way_response;
    let _ = quantize_cel_bands;
    let _: fn() -> SixWayAccumulator = SixWayAccumulator::new;
    let _ = PhaseParams::isotropic;
    let _ = Vec3::ZERO;
    let _: fn(SixWayLuminance) -> f32 = |l| l.right;
}
