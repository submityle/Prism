//! `wgpu` compute twin of the audio-spectrum sampler maths
//! ([`audio_spectrum`](prism_render_architecture::particle::audio_spectrum),
//! particle design §8.3).
//!
//! The `CPU` golden
//! [`audio_spectrum`](prism_render_architecture::particle::audio_spectrum) owns
//! the small, verifiable scalar arithmetic an audio-reactive effect needs once
//! an upstream `FFT` has already produced a magnitude/energy spectrum: bin
//! geometry
//! ([`Spectrum::bin_width_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_width_hz),
//! [`Spectrum::nyquist_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::nyquist_hz),
//! [`Spectrum::hz_to_bin`](prism_render_architecture::particle::audio_spectrum::Spectrum::hz_to_bin)),
//! energy aggregation
//! ([`Spectrum::bin_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_energy),
//! [`Spectrum::band_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::band_energy),
//! [`Spectrum::low_mid_high`](prism_render_architecture::particle::audio_spectrum::Spectrum::low_mid_high)),
//! the linear attack/release envelope follower
//! ([`EnvelopeState::advance`](prism_render_architecture::particle::audio_spectrum::EnvelopeState::advance)),
//! the spectral-flux onset detector step
//! ([`OnsetDetector::update`](prism_render_architecture::particle::audio_spectrum::OnsetDetector::update)),
//! and the linear parameter remaps
//! ([`map_range`](prism_render_architecture::particle::audio_spectrum::map_range)
//! and
//! [`normalize`](prism_render_architecture::particle::audio_spectrum::normalize)).
//! [`GpuAudioSpectrum`] is the on-device twin: one thread answers one
//! [`AudioSpectrumQuery`], so a passing real-device parity test is direct
//! evidence the ported kernel computes the same bin widths, located bins,
//! summed band energies, smoothed envelopes, onset verdicts and remapped
//! scalars the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! A single batched kernel evaluates a tagged union of independent queries,
//! one per thread, mirroring the reference tap for tap:
//!
//! - [`AudioSpectrumQuery::BinWidthHz`] reproduces
//!   [`Spectrum::bin_width_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_width_hz):
//!   `sample_rate / fft_size`, guarded so a zero `fft_size` yields `0`.
//! - [`AudioSpectrumQuery::NyquistHz`] reproduces
//!   [`Spectrum::nyquist_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::nyquist_hz):
//!   half the sample rate.
//! - [`AudioSpectrumQuery::HzToBin`] reproduces
//!   [`Spectrum::hz_to_bin`](prism_render_architecture::particle::audio_spectrum::Spectrum::hz_to_bin):
//!   `floor(hz / bin_width)` clamped into `[0, bin_count - 1]`.
//! - [`AudioSpectrumQuery::BinEnergy`] reproduces
//!   [`Spectrum::bin_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_energy):
//!   the packed bin value, or `0` when the index is out of range.
//! - [`AudioSpectrumQuery::BandEnergy`] reproduces
//!   [`Spectrum::band_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::band_energy)
//!   given host-located endpoint bins: the inclusive sum `bins[lo..=hi]`.
//! - [`AudioSpectrumQuery::LowMidHigh`] reproduces
//!   [`Spectrum::low_mid_high`](prism_render_architecture::particle::audio_spectrum::Spectrum::low_mid_high):
//!   the three crossover-split band sums packed into a `vec3`.
//! - [`AudioSpectrumQuery::EnvelopeAdvance`] reproduces
//!   [`EnvelopeState::advance`](prism_render_architecture::particle::audio_spectrum::EnvelopeState::advance):
//!   the linear one-pole step `value += (target - value) * coeff` with
//!   `coeff = clamp(dt / time_constant, 0, 1)`.
//! - [`AudioSpectrumQuery::OnsetUpdate`] reproduces
//!   [`OnsetDetector::update`](prism_render_architecture::particle::audio_spectrum::OnsetDetector::update):
//!   the half-wave-rectified flux, the sensitivity-scaled threshold verdict and
//!   the one-pole baseline advance.
//! - [`AudioSpectrumQuery::MapRange`] reproduces
//!   [`map_range`](prism_render_architecture::particle::audio_spectrum::map_range):
//!   the clamped linear remap, with a degenerate input span returning `out_lo`.
//! - [`AudioSpectrumQuery::Normalize`] reproduces
//!   [`normalize`](prism_render_architecture::particle::audio_spectrum::normalize):
//!   `clamp(energy / reference, 0, 1)`, guarded against a zero reference.
//!
//! # What stays on the host (not twinned)
//!
//! The `FFT` itself is transcendental (trigonometric twiddle factors) and is an
//! upstream `DSP` concern, never performed here. The variable-length slice
//! walks that iterate a `&[f32]` of unknown length stay on the host: the
//! `Hz`-to-index aggregation that
//! [`Spectrum::band_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::band_energy)
//! performs over an arbitrary bin count, the arbitrary-edge
//! [`Spectrum::bands`](prism_render_architecture::particle::audio_spectrum::Spectrum::bands)
//! split, and the `&[f32]` capacity itself. The twin packs a bounded bin array
//! ([`BIN_CAPACITY`] lanes) and receives host-located endpoint bins for
//! [`AudioSpectrumQuery::BandEnergy`], so the device performs only the fixed,
//! bounded arithmetic.
//!
//! # Correctness model
//!
//! The continuous answers (bin width, energies, envelope, onset flux/trigger,
//! remapped scalars) thread through multiplies, adds and guarded divides, so
//! `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits. The parity test
//! therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on
//! every continuous quantity. The discrete answers (the located bin index and
//! the onset `bool` verdict) are integer / `bool` and are compared exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::audio_spectrum`；无第三方引擎源码或衍生代码。
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

/// Maximum number of packed spectrum bins one query carries.
///
/// The host asserts every fixture's bin count stays within this bound; the
/// device twin sums only within it so the kernel's work is fixed and bounded.
pub const BIN_CAPACITY: usize = 32;

/// Operation tag selecting [`AudioSpectrumQuery::BinWidthHz`] on device.
const OP_BIN_WIDTH_HZ: u32 = 0;
/// Operation tag selecting [`AudioSpectrumQuery::NyquistHz`] on device.
const OP_NYQUIST_HZ: u32 = 1;
/// Operation tag selecting [`AudioSpectrumQuery::HzToBin`] on device.
const OP_HZ_TO_BIN: u32 = 2;
/// Operation tag selecting [`AudioSpectrumQuery::BinEnergy`] on device.
const OP_BIN_ENERGY: u32 = 3;
/// Operation tag selecting [`AudioSpectrumQuery::BandEnergy`] on device.
const OP_BAND_ENERGY: u32 = 4;
/// Operation tag selecting [`AudioSpectrumQuery::LowMidHigh`] on device.
const OP_LOW_MID_HIGH: u32 = 5;
/// Operation tag selecting [`AudioSpectrumQuery::EnvelopeAdvance`] on device.
const OP_ENVELOPE_ADVANCE: u32 = 6;
/// Operation tag selecting [`AudioSpectrumQuery::OnsetUpdate`] on device.
const OP_ONSET_UPDATE: u32 = 7;
/// Operation tag selecting [`AudioSpectrumQuery::MapRange`] on device.
const OP_MAP_RANGE: u32 = 8;
/// Operation tag selecting [`AudioSpectrumQuery::Normalize`] on device.
const OP_NORMALIZE: u32 = 9;

/// The portable core-`WGSL` audio-spectrum kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`audio_spectrum`](prism_render_architecture::particle::audio_spectrum)
/// function for function; see the module documentation for the algorithm.
const AUDIO_SPECTRUM_WGSL: &str = r#"
// Audio-spectrum twin: one thread per query reproduces the bin geometry
// (bin_width_hz, nyquist_hz, hz_to_bin), the energy aggregation (bin_energy,
// band_energy, low_mid_high), the linear attack/release envelope follower
// (EnvelopeState::advance), the spectral-flux onset step
// (OnsetDetector::update) and the linear remaps (map_range, normalize). It
// mirrors the CPU golden particle::audio_spectrum function for function, uses
// only the portable core-WGSL subset (floor/abs/clamp and + - * / plus unsigned
// index math), needs no transcendental call and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12. Every loop is bounded by the
// packed bin capacity, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::audio_spectrum；无第三方
// 引擎源码或衍生代码。

// Divisor magnitude below which a denominator is treated as zero, matching the
// reference EPS; the guard used instead of an f32 `==` on denominators.
const EPS: f32 = 1.0e-9;

// Low/mid crossover in Hz, matching the reference LOW_MID_CROSSOVER_HZ.
const LOW_MID_CROSSOVER_HZ: f32 = 250.0;
// Mid/high crossover in Hz, matching the reference MID_HIGH_CROSSOVER_HZ.
const MID_HIGH_CROSSOVER_HZ: f32 = 4000.0;

// Packed bin capacity, matching the host BIN_CAPACITY.
const BIN_CAPACITY: u32 = 32u;

// Operation tags, matching the host op codes.
const OP_BIN_WIDTH_HZ: u32 = 0u;
const OP_NYQUIST_HZ: u32 = 1u;
const OP_HZ_TO_BIN: u32 = 2u;
const OP_BIN_ENERGY: u32 = 3u;
const OP_BAND_ENERGY: u32 = 4u;
const OP_LOW_MID_HIGH: u32 = 5u;
const OP_ENVELOPE_ADVANCE: u32 = 6u;
const OP_ONSET_UPDATE: u32 = 7u;
const OP_MAP_RANGE: u32 = 8u;
const OP_NORMALIZE: u32 = 9u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which twinned function this thread evaluates.
    op: u32,
    // Number of valid packed bins (<= BIN_CAPACITY), the twin's bin_count.
    bin_count: u32,
    // Single bin index for the bin_energy op.
    bin_index: u32,
    // Host-located inclusive lower endpoint bin for the band_energy op.
    lo_bin: u32,
    // Host-located inclusive upper endpoint bin for the band_energy op.
    hi_bin: u32,
    // Audio sample rate in Hz for the geometry and low_mid_high ops.
    sample_rate_hz: f32,
    // FFT window size (as f32) for the geometry and low_mid_high ops.
    fft_size: f32,
    // Frequency in Hz for the hz_to_bin op.
    hz: f32,
    // Current envelope value for the envelope op.
    envelope_value: f32,
    // Envelope target (named `needle` because `target` is a WGSL reserved word).
    needle: f32,
    // Envelope attack time constant in seconds.
    attack_seconds: f32,
    // Envelope release time constant in seconds.
    release_seconds: f32,
    // Frame delta time in seconds for the envelope and onset ops.
    dt: f32,
    // Incoming band energy for the onset op, and the input for normalize.
    energy: f32,
    // Running-average baseline for the onset op.
    running_avg: f32,
    // Onset threshold.
    threshold: f32,
    // Onset sensitivity multiplier.
    sensitivity: f32,
    // Onset running-average smoothing time constant in seconds.
    smoothing_seconds: f32,
    // Input value for the map_range op.
    map_x: f32,
    // map_range input-range lower bound.
    in_lo: f32,
    // map_range input-range upper bound.
    in_hi: f32,
    // map_range output-range lower bound.
    out_lo: f32,
    // map_range output-range upper bound.
    out_hi: f32,
    // Reference energy for the normalize op.
    reference: f32,
    // Packed per-bin energies; only the first bin_count lanes are valid.
    bins: array<f32, 32>,
}

struct Result {
    // Operation tag echoed back so the host decodes the right union variant.
    op: u32,
    // Located bin index for the hz_to_bin op.
    bin_index: u32,
    // Onset verdict (0 / 1) for the onset op.
    onset_flag: u32,
    pad0: u32,
    // Scalar result for the width/nyquist/energy/envelope/map/normalize ops.
    scalar: f32,
    // Half-wave-rectified flux for the onset op.
    flux: f32,
    // Threshold-exceedance trigger for the onset op.
    trigger: f32,
    // Advanced running-average baseline for the onset op.
    running_avg: f32,
    // Low/mid/high band energies for the low_mid_high op. A pad lane follows.
    vec: vec3<f32>,
    pad_vec: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamp into the unit interval, matching the reference `clamp(x, 0, 1)`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Frequency span of one bin, sample_rate / fft_size, guarded against a zero
// window so a malformed view yields 0 (scalar divide guarded by EPS).
fn bin_width_hz(sample_rate_hz: f32, fft_size: f32) -> f32 {
    if (fft_size > EPS) {
        return sample_rate_hz / fft_size;
    }
    return 0.0;
}

// Locate the bin for a frequency with floor and clamp into [0, count - 1],
// matching the reference `Spectrum::hz_to_bin`.
fn hz_to_bin(hz: f32, width: f32, count: u32) -> u32 {
    if (count == 0u) {
        return 0u;
    }
    if (width <= EPS) {
        return 0u;
    }
    let located = floor(hz / width);
    if (located <= 0.0) {
        return 0u;
    }
    let countf = f32(count);
    if (located >= countf) {
        return count - 1u;
    }
    return u32(located);
}

// Inclusive sum of bins[lo..=hi], guarded by the valid bin count and the packed
// capacity, matching the reference `Spectrum::band_energy` summation.
fn sum_bins(q: Query, lo: u32, hi: u32) -> f32 {
    var acc = 0.0;
    var i = lo;
    loop {
        if (i > hi) {
            break;
        }
        if (i < q.bin_count && i < BIN_CAPACITY) {
            acc = acc + q.bins[i];
        }
        i = i + 1u;
    }
    return acc;
}

// Summed energy of every bin whose centre falls in [lo_hz, hi_hz], ordering the
// endpoints and locating them with floor, matching `Spectrum::band_energy`.
fn band_energy_hz(q: Query, lo_hz: f32, hi_hz: f32) -> f32 {
    let width = bin_width_hz(q.sample_rate_hz, q.fft_size);
    var a = lo_hz;
    var b = hi_hz;
    if (lo_hz > hi_hz) {
        a = hi_hz;
        b = lo_hz;
    }
    let lo = hz_to_bin(a, width, q.bin_count);
    let hi = hz_to_bin(b, width, q.bin_count);
    return sum_bins(q, lo, hi);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op;

    var out: Result;
    out.op = op;
    out.bin_index = 0u;
    out.onset_flag = 0u;
    out.pad0 = 0u;
    out.scalar = 0.0;
    out.flux = 0.0;
    out.trigger = 0.0;
    out.running_avg = 0.0;
    out.vec = vec3<f32>(0.0, 0.0, 0.0);
    out.pad_vec = 0.0;

    if (op == OP_BIN_WIDTH_HZ) {
        out.scalar = bin_width_hz(q.sample_rate_hz, q.fft_size);
    } else if (op == OP_NYQUIST_HZ) {
        out.scalar = q.sample_rate_hz * 0.5;
    } else if (op == OP_HZ_TO_BIN) {
        let width = bin_width_hz(q.sample_rate_hz, q.fft_size);
        out.bin_index = hz_to_bin(q.hz, width, q.bin_count);
    } else if (op == OP_BIN_ENERGY) {
        if (q.bin_index < q.bin_count && q.bin_index < BIN_CAPACITY) {
            out.scalar = q.bins[q.bin_index];
        }
    } else if (op == OP_BAND_ENERGY) {
        out.scalar = sum_bins(q, q.lo_bin, q.hi_bin);
    } else if (op == OP_LOW_MID_HIGH) {
        let low = band_energy_hz(q, 0.0, LOW_MID_CROSSOVER_HZ);
        let mid = band_energy_hz(q, LOW_MID_CROSSOVER_HZ, MID_HIGH_CROSSOVER_HZ);
        let nyquist = q.sample_rate_hz * 0.5;
        let high = band_energy_hz(q, MID_HIGH_CROSSOVER_HZ, nyquist);
        out.vec = vec3<f32>(low, mid, high);
    } else if (op == OP_ENVELOPE_ADVANCE) {
        let rising = q.needle > q.envelope_value;
        var time_constant = q.release_seconds;
        if (rising) {
            time_constant = q.attack_seconds;
        }
        var coeff = 1.0;
        if (time_constant > EPS) {
            coeff = clamp01(q.dt / time_constant);
        }
        out.scalar = q.envelope_value + (q.needle - q.envelope_value) * coeff;
    } else if (op == OP_ONSET_UPDATE) {
        let diff = q.energy - q.running_avg;
        var flux = 0.0;
        if (diff > 0.0) {
            flux = diff;
        }
        let scaled = flux * q.sensitivity;
        var trigger = 0.0;
        if (scaled > q.threshold) {
            out.onset_flag = 1u;
            trigger = scaled - q.threshold;
        }
        var coeff = 1.0;
        if (q.smoothing_seconds > EPS) {
            coeff = clamp01(q.dt / q.smoothing_seconds);
        }
        out.flux = flux;
        out.trigger = trigger;
        out.running_avg = q.running_avg + (q.energy - q.running_avg) * coeff;
    } else if (op == OP_MAP_RANGE) {
        let span = q.in_hi - q.in_lo;
        if (abs(span) <= EPS) {
            out.scalar = q.out_lo;
        } else {
            let t = clamp01((q.map_x - q.in_lo) / span);
            out.scalar = q.out_lo + (q.out_hi - q.out_lo) * t;
        }
    } else {
        // OP_NORMALIZE: clamp(energy / reference, 0, 1), guarded reference.
        if (abs(q.reference) <= EPS) {
            out.scalar = 0.0;
        } else {
            out.scalar = clamp01(q.energy / q.reference);
        }
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`AUDIO_SPECTRUM_WGSL`].
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
/// All members are `4`-byte scalars (or the packed bin array), laid out in the
/// same order on both sides so the host and device agree byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting the twinned function.
    op: u32,
    /// Number of valid packed bins (the twin's `bin_count`).
    bin_count: u32,
    /// Single bin index for the `bin_energy` op.
    bin_index: u32,
    /// Host-located inclusive lower endpoint bin for the `band_energy` op.
    lo_bin: u32,
    /// Host-located inclusive upper endpoint bin for the `band_energy` op.
    hi_bin: u32,
    /// Audio sample rate in `Hz`.
    sample_rate_hz: f32,
    /// `FFT` window size, stored as `f32`.
    fft_size: f32,
    /// Frequency in `Hz` for the `hz_to_bin` op.
    hz: f32,
    /// Current envelope value.
    envelope_value: f32,
    /// Envelope target.
    needle: f32,
    /// Envelope attack time constant in seconds.
    attack_seconds: f32,
    /// Envelope release time constant in seconds.
    release_seconds: f32,
    /// Frame delta time in seconds.
    dt: f32,
    /// Incoming energy (onset) or input value (normalize).
    energy: f32,
    /// Running-average baseline for the onset op.
    running_avg: f32,
    /// Onset threshold.
    threshold: f32,
    /// Onset sensitivity multiplier.
    sensitivity: f32,
    /// Onset running-average smoothing time constant in seconds.
    smoothing_seconds: f32,
    /// Input value for the `map_range` op.
    map_x: f32,
    /// `map_range` input-range lower bound.
    in_lo: f32,
    /// `map_range` input-range upper bound.
    in_hi: f32,
    /// `map_range` output-range lower bound.
    out_lo: f32,
    /// `map_range` output-range upper bound.
    out_hi: f32,
    /// Reference energy for the `normalize` op.
    reference: f32,
    /// Packed per-bin energies; only the first `bin_count` lanes are valid.
    bins: [f32; BIN_CAPACITY],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The `vec` member sits at a `16`-byte offset to honour the `vec3`
/// alignment, with a trailing pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Operation tag echoed back for decoding.
    op: u32,
    /// Located bin index for the `hz_to_bin` op.
    bin_index: u32,
    /// Onset verdict (`0` / `1`) for the onset op.
    onset_flag: u32,
    /// Padding word.
    pad0: u32,
    /// Scalar result for the width/nyquist/energy/envelope/map/normalize ops.
    scalar: f32,
    /// Half-wave-rectified flux for the onset op.
    flux: f32,
    /// Threshold-exceedance trigger for the onset op.
    trigger: f32,
    /// Advanced running-average baseline for the onset op.
    running_avg: f32,
    /// Low/mid/high band energies for the `low_mid_high` op.
    vec: [f32; 3],
    /// Padding lane keeping `vec` `16`-byte aligned.
    pad_vec: f32,
}

/// One query for the audio-spectrum twin: a tagged union selecting which of the
/// twinned reference functions this element evaluates.
///
/// The spectrum-bearing variants carry a bounded bin slice (`<= BIN_CAPACITY`
/// lanes) plus the parameters the reference needs; the scalar variants carry
/// the raw reference arguments.
#[derive(Clone, Debug, PartialEq)]
pub enum AudioSpectrumQuery {
    /// Bin width in `Hz`, twinning
    /// [`Spectrum::bin_width_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_width_hz).
    BinWidthHz {
        /// Audio sample rate in `Hz`.
        sample_rate_hz: f32,
        /// `FFT` window size.
        fft_size: u32,
    },
    /// Nyquist frequency in `Hz`, twinning
    /// [`Spectrum::nyquist_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::nyquist_hz).
    NyquistHz {
        /// Audio sample rate in `Hz`.
        sample_rate_hz: f32,
    },
    /// Frequency-to-bin location, twinning
    /// [`Spectrum::hz_to_bin`](prism_render_architecture::particle::audio_spectrum::Spectrum::hz_to_bin).
    HzToBin {
        /// Audio sample rate in `Hz`.
        sample_rate_hz: f32,
        /// `FFT` window size.
        fft_size: u32,
        /// Number of bins in the spectrum.
        bin_count: u32,
        /// Frequency in `Hz` to locate.
        hz: f32,
    },
    /// Single-bin energy, twinning
    /// [`Spectrum::bin_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_energy).
    BinEnergy {
        /// Packed per-bin energies (`<= BIN_CAPACITY` lanes).
        bins: Vec<f32>,
        /// Bin index to read.
        index: u32,
    },
    /// Inclusive band sum over host-located endpoints, twinning
    /// [`Spectrum::band_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::band_energy).
    BandEnergy {
        /// Packed per-bin energies (`<= BIN_CAPACITY` lanes).
        bins: Vec<f32>,
        /// Host-located inclusive lower endpoint bin.
        lo_bin: u32,
        /// Host-located inclusive upper endpoint bin.
        hi_bin: u32,
    },
    /// Three-band crossover split, twinning
    /// [`Spectrum::low_mid_high`](prism_render_architecture::particle::audio_spectrum::Spectrum::low_mid_high).
    LowMidHigh {
        /// Packed per-bin energies (`<= BIN_CAPACITY` lanes).
        bins: Vec<f32>,
        /// Audio sample rate in `Hz`.
        sample_rate_hz: f32,
        /// `FFT` window size.
        fft_size: u32,
    },
    /// One linear envelope step, twinning
    /// [`EnvelopeState::advance`](prism_render_architecture::particle::audio_spectrum::EnvelopeState::advance).
    EnvelopeAdvance {
        /// Current smoothed envelope value.
        value: f32,
        /// Target the envelope moves toward.
        target: f32,
        /// Attack time constant in seconds.
        attack_seconds: f32,
        /// Release time constant in seconds.
        release_seconds: f32,
        /// Frame delta time in seconds.
        dt: f32,
    },
    /// One onset-detector step, twinning
    /// [`OnsetDetector::update`](prism_render_architecture::particle::audio_spectrum::OnsetDetector::update).
    OnsetUpdate {
        /// Incoming band energy.
        energy: f32,
        /// Current running-average baseline.
        running_avg: f32,
        /// Onset threshold.
        threshold: f32,
        /// Onset sensitivity multiplier.
        sensitivity: f32,
        /// Running-average smoothing time constant in seconds.
        smoothing_seconds: f32,
        /// Frame delta time in seconds.
        dt: f32,
    },
    /// Clamped linear remap, twinning
    /// [`map_range`](prism_render_architecture::particle::audio_spectrum::map_range).
    MapRange {
        /// Input value.
        x: f32,
        /// Input-range lower bound.
        in_lo: f32,
        /// Input-range upper bound.
        in_hi: f32,
        /// Output-range lower bound.
        out_lo: f32,
        /// Output-range upper bound.
        out_hi: f32,
    },
    /// Reference normalization, twinning
    /// [`normalize`](prism_render_architecture::particle::audio_spectrum::normalize).
    Normalize {
        /// Input energy.
        energy: f32,
        /// Reference energy.
        reference: f32,
    },
}

/// One resolved answer for a single [`AudioSpectrumQuery`], mirroring the value
/// the reference reports for the corresponding function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AudioSpectrumResult {
    /// Bin width in `Hz`, matching
    /// [`Spectrum::bin_width_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_width_hz).
    BinWidthHz {
        /// Frequency span of one bin.
        hz_per_bin: f32,
    },
    /// Nyquist frequency in `Hz`, matching
    /// [`Spectrum::nyquist_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::nyquist_hz).
    NyquistHz {
        /// Half the sample rate.
        hz: f32,
    },
    /// Located bin index, matching
    /// [`Spectrum::hz_to_bin`](prism_render_architecture::particle::audio_spectrum::Spectrum::hz_to_bin).
    HzToBin {
        /// The located bin.
        bin: u32,
    },
    /// Single-bin energy, matching
    /// [`Spectrum::bin_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_energy).
    BinEnergy {
        /// The bin's energy, or `0` when out of range.
        energy: f32,
    },
    /// Inclusive band sum, matching
    /// [`Spectrum::band_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::band_energy).
    BandEnergy {
        /// Summed band energy.
        energy: f32,
    },
    /// Three-band split, matching
    /// [`Spectrum::low_mid_high`](prism_render_architecture::particle::audio_spectrum::Spectrum::low_mid_high).
    LowMidHigh {
        /// Low-band energy.
        low: f32,
        /// Mid-band energy.
        mid: f32,
        /// High-band energy.
        high: f32,
    },
    /// Advanced envelope value, matching
    /// [`EnvelopeState::advance`](prism_render_architecture::particle::audio_spectrum::EnvelopeState::advance).
    EnvelopeAdvance {
        /// The new smoothed value.
        value: f32,
    },
    /// Onset decision, matching
    /// [`OnsetDetector::update`](prism_render_architecture::particle::audio_spectrum::OnsetDetector::update).
    OnsetUpdate {
        /// Whether this step crossed the onset threshold.
        is_onset: bool,
        /// Half-wave-rectified spectral flux.
        flux: f32,
        /// Threshold-exceedance trigger.
        trigger: f32,
        /// Advanced running-average baseline.
        running_avg: f32,
    },
    /// Remapped scalar, matching
    /// [`map_range`](prism_render_architecture::particle::audio_spectrum::map_range).
    MapRange {
        /// The remapped value.
        value: f32,
    },
    /// Normalized scalar, matching
    /// [`normalize`](prism_render_architecture::particle::audio_spectrum::normalize).
    Normalize {
        /// The normalized value in `[0, 1]`.
        value: f32,
    },
}

/// Copies a bounded bin slice into the packed query lanes, recording the valid
/// bin count. Only the first [`BIN_CAPACITY`] lanes are populated.
fn pack_bins(slot: &mut GpuQuery, bins: &[f32]) {
    let n = bins.len().min(BIN_CAPACITY);
    slot.bins[..n].copy_from_slice(&bins[..n]);
    slot.bin_count = bins.len() as u32;
}

/// Encodes one [`AudioSpectrumQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &AudioSpectrumQuery) -> GpuQuery {
    let mut slot = GpuQuery::zeroed();
    match q {
        AudioSpectrumQuery::BinWidthHz {
            sample_rate_hz,
            fft_size,
        } => {
            slot.op = OP_BIN_WIDTH_HZ;
            slot.sample_rate_hz = *sample_rate_hz;
            slot.fft_size = *fft_size as f32;
        }
        AudioSpectrumQuery::NyquistHz { sample_rate_hz } => {
            slot.op = OP_NYQUIST_HZ;
            slot.sample_rate_hz = *sample_rate_hz;
        }
        AudioSpectrumQuery::HzToBin {
            sample_rate_hz,
            fft_size,
            bin_count,
            hz,
        } => {
            slot.op = OP_HZ_TO_BIN;
            slot.sample_rate_hz = *sample_rate_hz;
            slot.fft_size = *fft_size as f32;
            slot.bin_count = *bin_count;
            slot.hz = *hz;
        }
        AudioSpectrumQuery::BinEnergy { bins, index } => {
            slot.op = OP_BIN_ENERGY;
            pack_bins(&mut slot, bins);
            slot.bin_index = *index;
        }
        AudioSpectrumQuery::BandEnergy {
            bins,
            lo_bin,
            hi_bin,
        } => {
            slot.op = OP_BAND_ENERGY;
            pack_bins(&mut slot, bins);
            slot.lo_bin = *lo_bin;
            slot.hi_bin = *hi_bin;
        }
        AudioSpectrumQuery::LowMidHigh {
            bins,
            sample_rate_hz,
            fft_size,
        } => {
            slot.op = OP_LOW_MID_HIGH;
            pack_bins(&mut slot, bins);
            slot.sample_rate_hz = *sample_rate_hz;
            slot.fft_size = *fft_size as f32;
        }
        AudioSpectrumQuery::EnvelopeAdvance {
            value,
            target,
            attack_seconds,
            release_seconds,
            dt,
        } => {
            slot.op = OP_ENVELOPE_ADVANCE;
            slot.envelope_value = *value;
            slot.needle = *target;
            slot.attack_seconds = *attack_seconds;
            slot.release_seconds = *release_seconds;
            slot.dt = *dt;
        }
        AudioSpectrumQuery::OnsetUpdate {
            energy,
            running_avg,
            threshold,
            sensitivity,
            smoothing_seconds,
            dt,
        } => {
            slot.op = OP_ONSET_UPDATE;
            slot.energy = *energy;
            slot.running_avg = *running_avg;
            slot.threshold = *threshold;
            slot.sensitivity = *sensitivity;
            slot.smoothing_seconds = *smoothing_seconds;
            slot.dt = *dt;
        }
        AudioSpectrumQuery::MapRange {
            x,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
        } => {
            slot.op = OP_MAP_RANGE;
            slot.map_x = *x;
            slot.in_lo = *in_lo;
            slot.in_hi = *in_hi;
            slot.out_lo = *out_lo;
            slot.out_hi = *out_hi;
        }
        AudioSpectrumQuery::Normalize { energy, reference } => {
            slot.op = OP_NORMALIZE;
            slot.energy = *energy;
            slot.reference = *reference;
        }
    }
    slot
}

/// Decodes one packed [`GpuResult`] into the public [`AudioSpectrumResult`],
/// selecting the union variant from the echoed operation tag.
fn decode_result(raw: &GpuResult) -> AudioSpectrumResult {
    match raw.op {
        OP_BIN_WIDTH_HZ => AudioSpectrumResult::BinWidthHz {
            hz_per_bin: raw.scalar,
        },
        OP_NYQUIST_HZ => AudioSpectrumResult::NyquistHz { hz: raw.scalar },
        OP_HZ_TO_BIN => AudioSpectrumResult::HzToBin { bin: raw.bin_index },
        OP_BIN_ENERGY => AudioSpectrumResult::BinEnergy { energy: raw.scalar },
        OP_BAND_ENERGY => AudioSpectrumResult::BandEnergy { energy: raw.scalar },
        OP_LOW_MID_HIGH => AudioSpectrumResult::LowMidHigh {
            low: raw.vec[0],
            mid: raw.vec[1],
            high: raw.vec[2],
        },
        OP_ENVELOPE_ADVANCE => AudioSpectrumResult::EnvelopeAdvance { value: raw.scalar },
        OP_ONSET_UPDATE => AudioSpectrumResult::OnsetUpdate {
            is_onset: raw.onset_flag != 0,
            flux: raw.flux,
            trigger: raw.trigger,
            running_avg: raw.running_avg,
        },
        OP_MAP_RANGE => AudioSpectrumResult::MapRange { value: raw.scalar },
        _ => AudioSpectrumResult::Normalize { value: raw.scalar },
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

/// A compiled, reusable audio-spectrum compute pipeline, twinning the `CPU`
/// golden
/// [`audio_spectrum`](prism_render_architecture::particle::audio_spectrum).
pub struct GpuAudioSpectrum {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAudioSpectrum {
    /// Compiles the audio-spectrum kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAudioSpectrum {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_audio_spectrum"),
            source: ShaderSource::Wgsl(AUDIO_SPECTRUM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_audio_spectrum_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_audio_spectrum_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_audio_spectrum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAudioSpectrum {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`AudioSpectrumResult`]
    /// per input, in order.
    ///
    /// The continuous answers match the reference to within the tolerance
    /// documented on this module; the discrete answers match exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[AudioSpectrumQuery],
    ) -> Vec<AudioSpectrumResult> {
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
            label: Some("prism_volumetric_audio_spectrum_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_audio_spectrum_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_audio_spectrum_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_audio_spectrum_bind_group"),
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
            label: Some("prism_volumetric_audio_spectrum_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_audio_spectrum_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_audio_spectrum_pass"),
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
