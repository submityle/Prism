//! Spectral-shape descriptors (centroid, spread, flatness, rolloff, flux, ...).
//!
//! Where the [`SpectrumAnalyzer`](crate::nodes::analysis::spectrum::SpectrumAnalyzer)
//! answers *at which frequencies* a signal's energy lives by handing back a
//! full magnitude spectrum, this module condenses one such spectrum into a
//! handful of scalar descriptors that summarise its *shape*. These are the
//! low-level timbre features of music-information-retrieval (`MIR`) and the
//! control signals a timbre-aware mixer, an auto-`EQ`, or a visualiser reads:
//! a single number for brightness (centroid), one for bandwidth (spread), one
//! for noisiness (flatness), and so on. The module renders no pixels and makes
//! no mixing decisions; it produces the numbers those consumers act on.
//!
//! # Model
//!
//! Given a single-sided magnitude spectrum `M[k]` (`k = 0..K`, as produced by
//! the [`SpectrumAnalyzer`]) and the sample rate, the analysis-length is
//! recovered as `size = (K - 1) * 2` and the centre frequency of bin `k` is
//! `f[k] = k * sample_rate / size` in `Hz`. Writing the (power-normalised)
//! probability mass as `p[k] = M[k] / Sum(M)`, the descriptors are the classic
//! textbook / `MPEG-7` definitions:
//!
//! - **centroid** `C = Sum(f[k] * M[k]) / Sum(M[k])` -- the magnitude-weighted
//!   mean frequency; the perceptual "brightness" of the frame.
//! - **spread** `sqrt(Sum((f[k] - C)^2 * M[k]) / Sum(M[k]))` -- the standard
//!   deviation of frequency about the centroid; the frame's bandwidth.
//! - **skewness** `Sum((f - C)^3 * M) / Sum(M) / spread^3` -- the asymmetry of
//!   the distribution about the centroid.
//! - **kurtosis** `Sum((f - C)^4 * M) / Sum(M) / spread^4` -- the peakedness of
//!   the distribution. This is the raw fourth standardised moment, not the
//!   excess kurtosis (3 is *not* subtracted); a Gaussian reads about 3.
//! - **flatness** `geometric_mean(P) / arithmetic_mean(P)` over the power
//!   spectrum `P[k] = M[k]^2`, in `0..=1` -- 1 for an ideally flat (white)
//!   spectrum and tending to 0 for a single tone; the Wiener entropy / tonality
//!   measure.
//! - **crest** `max(P) / mean(P)` -- the spectral crest factor; large for a
//!   peaky (tonal) spectrum, near 1 for a flat (noisy) one.
//! - **rolloff** the lowest frequency below which a configurable fraction
//!   ([`SpectralFeatures::rolloff_fraction`], typically `0.85`) of the total
//!   magnitude lies; a robust high-frequency-content measure.
//! - **flux** `sqrt(Sum((M_t[k] - M_{t-1}[k])^2))` -- the `L2` distance between
//!   consecutive magnitude spectra; large at note onsets and transients, zero
//!   in steady state. The very first analysed frame is compared against an
//!   all-zero previous frame.
//! - **slope** the least-squares slope of `M[k]` regressed on `f[k]`:
//!   `Sum((f - fbar)(M - Mbar)) / Sum((f - fbar)^2)` in magnitude-per-`Hz`;
//!   negative for the usual high-frequency roll-off.
//! - **decrease** the `MPEG-7` spectral decrease
//!   `Sum_{k>=1} (M[k] - M[0]) / k / Sum_{k>=1} M[k]` -- a perceptually
//!   motivated measure of the amount of decrease of the spectrum.
//!
//! A silent frame (total magnitude below a tiny floor) yields an all-zero
//! [`SpectralFeatureSet`] so no descriptor can divide by zero.
//!
//! # Real-time contract
//!
//! [`SpectralFeatures`] owns exactly one heap buffer (the previous-frame
//! magnitudes, used by the flux term); it is sized once on the first
//! [`analyze`](SpectralFeatures::analyze) call for a given spectrum length and
//! reused thereafter, so the steady-state hot path performs no allocation,
//! takes no locks, and cannot panic. [`SpectralFeaturesNode`] additionally owns
//! a [`SpectrumAnalyzer`], whose own buffers are allocated once at
//! construction. All transcendental math routes through [`bevy_math::ops`], so
//! every descriptor is bit-reproducible across platforms. Non-finite or empty
//! inputs are treated as a silent frame.
//!
//! # Provenance
//!
//! The spectral-shape descriptors implemented here are the standard
//! music-information-retrieval and `MPEG-7` Audio low-level descriptors
//! (spectral centroid, spread, skewness, kurtosis, flatness, crest, rolloff,
//! flux, slope, and decrease), each defined by an elementary closed-form
//! statistic over a magnitude spectrum and described in every spectral-feature
//! reference. This module reuses only this crate's own [`Sample`] scalar, its
//! graph traits, and the public API of its sibling [`SpectrumAnalyzer`]; it
//! copies no external code. It is pure classic DSP with no AI or ML and
//! contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code**; it is implemented purely
//! from those publicly documented formulae.
//!
//! # Relationship
//!
//! This module is a thin statistical layer *on top of* the
//! [`SpectrumAnalyzer`]: [`SpectralFeaturesNode`] drives an embedded analyzer
//! and, each time it completes a transform, reduces the resulting magnitude
//! bins to a [`SpectralFeatureSet`]. It therefore complements the
//! [`spectrum`](crate::nodes::analysis::spectrum) analyzer (which exposes the
//! raw bins) and the scalar meters -- the
//! [`LoudnessMeter`](crate::nodes::analysis::loudness::LoudnessMeter) (how
//! loud), the [`CorrelationMeter`](crate::nodes::analysis::correlation::CorrelationMeter)
//! (how wide), and the
//! [`PitchDetector`](crate::nodes::analysis::pitch_detector::PitchDetector)
//! (which pitch) -- by describing the *timbre* of the frame. It shares no code
//! with any of them beyond the public [`SpectrumAnalyzer`] API it consumes.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::analysis::spectrum::{SpectrumAnalyzer, Window};

/// Default fraction of total magnitude used for the spectral-rolloff frequency.
pub const DEFAULT_ROLLOFF_FRACTION: Sample = 0.85;

/// Smallest magnitude sum treated as a non-silent frame.
///
/// Frames whose total magnitude is below this floor are reported as all-zero
/// descriptors so that no ratio can divide by zero.
const SILENCE_FLOOR: f64 = 1e-12;

/// Tiny power floor added before the logarithm in the flatness geometric mean.
///
/// This keeps the geometric mean finite when some power bins are exactly zero
/// (a pure tone) without materially biasing a genuinely flat spectrum.
const POWER_FLOOR: f64 = 1e-20;

/// A single frame of spectral-shape descriptors.
///
/// Every field is finite for every input: a silent or empty frame reports all
/// zeros. Frequencies are in `Hz`; the dimensionless ratios (flatness, crest,
/// skewness, kurtosis) carry no unit; slope is magnitude-per-`Hz`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpectralFeatureSet {
    /// Magnitude-weighted mean frequency (brightness), in `Hz`.
    pub centroid: Sample,
    /// Standard deviation of frequency about the centroid (bandwidth), in `Hz`.
    pub spread: Sample,
    /// Third standardised moment of the spectrum about the centroid.
    pub skewness: Sample,
    /// Fourth standardised (raw, not excess) moment about the centroid.
    pub kurtosis: Sample,
    /// Spectral flatness (Wiener entropy) in `0..=1`; 1 is maximally flat.
    pub flatness: Sample,
    /// Spectral crest factor `max(power) / mean(power)`; >= 1.
    pub crest: Sample,
    /// Spectral-rolloff frequency, in `Hz`.
    pub rolloff: Sample,
    /// `L2` spectral flux relative to the previous frame.
    pub flux: Sample,
    /// Least-squares spectral slope, in magnitude-per-`Hz`.
    pub slope: Sample,
    /// `MPEG-7` spectral decrease (dimensionless).
    pub decrease: Sample,
}

/// Reduces single-sided magnitude spectra to [`SpectralFeatureSet`] frames.
///
/// The analyzer is stateful only in the one buffer it needs for the spectral
/// flux term (the previous frame's magnitudes); every other descriptor is a
/// pure function of the current spectrum. Construct it with a rolloff fraction
/// in `0..=1` and call [`analyze`](Self::analyze) once per completed spectrum.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::analysis::spectral_features::SpectralFeatures;
///
/// // A single-sided spectrum (K = size/2 + 1 bins). A lone peak at bin 2 of a
/// // size-8 transform (sample rate 8 Hz) sits at 2 * 8 / 8 = 2 Hz.
/// let mut features = SpectralFeatures::new(0.85);
/// let mags = [0.0, 0.0, 1.0, 0.0, 0.0];
/// let set = features.analyze(&mags, 8);
/// assert!((set.centroid - 2.0).abs() < 1e-5);
/// ```
#[derive(Clone, Debug)]
pub struct SpectralFeatures {
    prev_mag: Vec<Sample>,
    rolloff_fraction: Sample,
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self::new(DEFAULT_ROLLOFF_FRACTION)
    }
}

impl SpectralFeatures {
    /// Builds an analyzer with the given rolloff fraction, clamped to `0..=1`.
    ///
    /// A non-finite fraction falls back to [`DEFAULT_ROLLOFF_FRACTION`].
    #[must_use]
    pub fn new(rolloff_fraction: Sample) -> Self {
        Self {
            prev_mag: Vec::new(),
            rolloff_fraction: sanitise_fraction(rolloff_fraction),
        }
    }

    /// Returns the configured rolloff fraction (in `0..=1`).
    #[inline]
    #[must_use]
    pub fn rolloff_fraction(&self) -> Sample {
        self.rolloff_fraction
    }

    /// Sets the rolloff fraction, clamped to `0..=1`.
    ///
    /// A non-finite value is ignored (the previous fraction is kept).
    pub fn set_rolloff_fraction(&mut self, fraction: Sample) {
        if fraction.is_finite() {
            self.rolloff_fraction = fraction.clamp(0.0, 1.0);
        }
    }

    /// Clears the stored previous frame used by the flux term.
    pub fn reset(&mut self) {
        self.prev_mag.clear();
    }

    /// Reduces one single-sided magnitude spectrum to a [`SpectralFeatureSet`].
    ///
    /// `magnitudes` is the single-sided spectrum (length `K = size/2 + 1`), as
    /// returned by [`SpectrumAnalyzer::magnitudes`]; `sample_rate` is the
    /// stream's sample rate in `Hz`. A spectrum shorter than two bins, or one
    /// whose total magnitude is below the silence floor, yields all zeros.
    ///
    /// The spectral-flux field compares `magnitudes` against the spectrum
    /// passed to the previous call (an all-zero frame on the first call).
    #[must_use]
    pub fn analyze(&mut self, magnitudes: &[Sample], sample_rate: u32) -> SpectralFeatureSet {
        let bins = magnitudes.len();
        if bins < 2 {
            self.remember(magnitudes);
            return SpectralFeatureSet::default();
        }

        // Recover the analysis length from the single-sided bin count.
        let size = (bins - 1) * 2;
        let hz_per_bin = f64::from(sample_rate) / size as f64;

        // First pass: magnitude sum, frequency-weighted sum, and power stats.
        let mut sum_m = 0.0f64;
        let mut sum_fm = 0.0f64;
        let mut sum_p = 0.0f64;
        let mut max_p = 0.0f64;
        let mut sum_ln_p = 0.0f64;
        for (k, &mag) in magnitudes.iter().enumerate() {
            let m = f64::from(mag);
            let m = if m.is_finite() { m.abs() } else { 0.0 };
            let f = k as f64 * hz_per_bin;
            let power = m * m;
            sum_m += m;
            sum_fm += f * m;
            sum_p += power;
            if power > max_p {
                max_p = power;
            }
            // bevy_math::ops operates on f32; accumulate the result in f64.
            sum_ln_p += f64::from(ops::ln((power + POWER_FLOOR) as Sample));
        }

        if sum_m < SILENCE_FLOOR {
            self.remember(magnitudes);
            return SpectralFeatureSet::default();
        }

        let centroid = sum_fm / sum_m;

        // Second pass: central moments about the centroid, and regression sums.
        let mean_f = {
            // Mean bin frequency: f runs 0, hz_per_bin, ..., (bins-1)*hz_per_bin.
            (bins - 1) as f64 * hz_per_bin / 2.0
        };
        let mean_m = sum_m / bins as f64;
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        let mut m4 = 0.0f64;
        let mut cov_fm = 0.0f64;
        let mut var_f = 0.0f64;
        for (k, &mag) in magnitudes.iter().enumerate() {
            let m = f64::from(mag);
            let m = if m.is_finite() { m.abs() } else { 0.0 };
            let f = k as f64 * hz_per_bin;
            let d = f - centroid;
            let d2 = d * d;
            m2 += d2 * m;
            m3 += d2 * d * m;
            m4 += d2 * d2 * m;
            let df = f - mean_f;
            cov_fm += df * (m - mean_m);
            var_f += df * df;
        }

        let variance = m2 / sum_m;
        let spread = ops_sqrt(variance);
        let spread3 = spread * spread * spread;
        let spread4 = spread3 * spread;
        let skewness = if spread3 > SILENCE_FLOOR {
            m3 / sum_m / spread3
        } else {
            0.0
        };
        let kurtosis = if spread4 > SILENCE_FLOOR {
            m4 / sum_m / spread4
        } else {
            0.0
        };

        // Flatness: geometric mean over arithmetic mean of the power spectrum.
        let geo_mean = ops_exp(sum_ln_p / bins as f64);
        let arith_mean = sum_p / bins as f64;
        let flatness = if arith_mean > POWER_FLOOR {
            (geo_mean / arith_mean).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Crest: peak power over mean power.
        let crest = if arith_mean > POWER_FLOOR {
            max_p / arith_mean
        } else {
            0.0
        };

        // Rolloff: lowest frequency capturing the configured magnitude fraction.
        let target = sum_m * f64::from(self.rolloff_fraction);
        let mut running = 0.0f64;
        let mut rolloff_bin = bins - 1;
        for (k, &mag) in magnitudes.iter().enumerate() {
            let m = f64::from(mag);
            let m = if m.is_finite() { m.abs() } else { 0.0 };
            running += m;
            if running >= target {
                rolloff_bin = k;
                break;
            }
        }
        let rolloff = rolloff_bin as f64 * hz_per_bin;

        // Slope: least-squares regression of magnitude on frequency.
        let slope = if var_f > SILENCE_FLOOR { cov_fm / var_f } else { 0.0 };

        // MPEG-7 spectral decrease (bins 1..K relative to bin 0).
        let m0 = {
            let v = f64::from(magnitudes[0]);
            if v.is_finite() { v.abs() } else { 0.0 }
        };
        let mut dec_num = 0.0f64;
        let mut dec_den = 0.0f64;
        for (k, &mag) in magnitudes.iter().enumerate().skip(1) {
            let m = f64::from(mag);
            let m = if m.is_finite() { m.abs() } else { 0.0 };
            dec_num += (m - m0) / k as f64;
            dec_den += m;
        }
        let decrease = if dec_den > SILENCE_FLOOR { dec_num / dec_den } else { 0.0 };

        // Flux: L2 distance from the previous frame's magnitudes.
        let flux = self.spectral_flux(magnitudes);
        self.remember(magnitudes);

        SpectralFeatureSet {
            centroid: centroid as Sample,
            spread: spread as Sample,
            skewness: skewness as Sample,
            kurtosis: kurtosis as Sample,
            flatness: flatness as Sample,
            crest: crest as Sample,
            rolloff: rolloff as Sample,
            flux: flux as Sample,
            slope: slope as Sample,
            decrease: decrease as Sample,
        }
    }

    /// Computes the `L2` flux between `current` and the stored previous frame.
    fn spectral_flux(&self, current: &[Sample]) -> f64 {
        let mut acc = 0.0f64;
        for (k, &mag) in current.iter().enumerate() {
            let m = f64::from(mag);
            let m = if m.is_finite() { m.abs() } else { 0.0 };
            let prev = self.prev_mag.get(k).copied().unwrap_or(0.0);
            let p = f64::from(prev);
            let diff = m - p;
            acc += diff * diff;
        }
        ops_sqrt(acc)
    }

    /// Stores `current` (sanitised) as the previous frame for the next flux.
    fn remember(&mut self, current: &[Sample]) {
        self.prev_mag.clear();
        self.prev_mag.reserve(current.len());
        for &mag in current {
            let v = if mag.is_finite() { mag.abs() } else { 0.0 };
            self.prev_mag.push(v);
        }
    }
}

/// Clamps a rolloff fraction to `0..=1`, mapping non-finite to the default.
#[inline]
fn sanitise_fraction(fraction: Sample) -> Sample {
    if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        DEFAULT_ROLLOFF_FRACTION
    }
}

/// `sqrt` of a non-negative `f64`, routed through [`bevy_math::ops`] in `f32`.
#[inline]
fn ops_sqrt(x: f64) -> f64 {
    f64::from(ops::sqrt(x.max(0.0) as Sample))
}

/// `exp` of an `f64`, routed through [`bevy_math::ops`] in `f32`.
#[inline]
fn ops_exp(x: f64) -> f64 {
    f64::from(ops::exp(x as Sample))
}

/// A pass-through metering node that reports [`SpectralFeatureSet`] frames.
///
/// The node owns a [`SpectrumAnalyzer`] and a [`SpectralFeatures`] reducer. It
/// copies its input to its output unchanged (it is a measurement tap) and feeds
/// the first channel into the analyzer. Each time the analyzer completes a
/// transform, the node reduces the fresh magnitude bins to a
/// [`SpectralFeatureSet`] available from [`latest`](Self::latest).
///
/// # Examples
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::analysis::spectral_features::SpectralFeaturesNode;
/// use prism_audio_core::nodes::analysis::spectrum::Window;
///
/// let sr = 48_000u32;
/// let mut node = SpectralFeaturesNode::new(1024, 512, Window::Hann, 0.85);
/// // A bright 8 kHz tone has a higher centroid than a dark 200 Hz tone.
/// let w = core::f32::consts::TAU * 8_000.0 / sr as f32;
/// let mut n = 0u64;
/// for _ in 0..8 {
///     let mut input = AudioBuffer::new(ChannelLayout::Mono, 512);
///     for s in input.channel_mut(0).iter_mut() {
///         *s = (w * n as f32).sin();
///         n += 1;
///     }
///     let mut output = AudioBuffer::new(ChannelLayout::Mono, 512);
///     let inputs = [input];
///     let mut outputs = [output];
///     let ctx = RenderContext { sample_rate: sr, frames: 512, playhead: 0 };
///     let mut io = ProcessIo::new(&inputs, &mut outputs);
///     node.process(&ctx, &mut io);
/// }
/// assert!(node.frames_analyzed() > 0);
/// assert!(node.latest().centroid > 4_000.0);
/// ```
#[derive(Clone, Debug)]
pub struct SpectralFeaturesNode {
    analyzer: SpectrumAnalyzer,
    features: SpectralFeatures,
    latest: SpectralFeatureSet,
    frames_analyzed: u64,
    last_frame: u64,
}

impl SpectralFeaturesNode {
    /// Builds a node around a fresh analyzer and feature reducer.
    ///
    /// `requested_size`, `hop`, and `window` configure the embedded
    /// [`SpectrumAnalyzer`]; `rolloff_fraction` is clamped to `0..=1`.
    #[must_use]
    pub fn new(requested_size: usize, hop: usize, window: Window, rolloff_fraction: Sample) -> Self {
        Self {
            analyzer: SpectrumAnalyzer::new(requested_size, hop, window),
            features: SpectralFeatures::new(rolloff_fraction),
            latest: SpectralFeatureSet::default(),
            frames_analyzed: 0,
            last_frame: 0,
        }
    }

    /// Returns the most recently computed descriptors (zeros until the first).
    #[inline]
    #[must_use]
    pub fn latest(&self) -> SpectralFeatureSet {
        self.latest
    }

    /// Returns the number of frames reduced to descriptors so far.
    #[inline]
    #[must_use]
    pub fn frames_analyzed(&self) -> u64 {
        self.frames_analyzed
    }

    /// Immutable access to the embedded spectrum analyzer.
    #[inline]
    #[must_use]
    pub fn analyzer(&self) -> &SpectrumAnalyzer {
        &self.analyzer
    }

    /// Mutable access to the embedded spectrum analyzer (to retune or reset).
    #[inline]
    pub fn analyzer_mut(&mut self) -> &mut SpectrumAnalyzer {
        &mut self.analyzer
    }

    /// Immutable access to the embedded feature reducer.
    #[inline]
    #[must_use]
    pub fn features(&self) -> &SpectralFeatures {
        &self.features
    }

    /// Mutable access to the embedded feature reducer (to retune or reset).
    #[inline]
    pub fn features_mut(&mut self) -> &mut SpectralFeatures {
        &mut self.features
    }
}

impl AudioNode for SpectralFeaturesNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let frames = output.active_frames();

        // Pass the signal through unchanged (this is a measurement tap).
        let copy_channels = out_channels.min(input.channels());
        for ch in 0..copy_channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }

        // Feed the first channel into the analyzer.
        if input.channels() >= 1 {
            let mono = input.channel(0);
            for &x in &mono[..frames] {
                self.analyzer.feed_sample(x);
            }
        }

        // If the analyzer completed one or more transforms this block, reduce
        // the most recent magnitude spectrum to a fresh descriptor set.
        let computed = self.analyzer.frames_computed();
        if computed > self.last_frame {
            self.last_frame = computed;
            self.latest = self.features.analyze(self.analyzer.magnitudes(), ctx.sample_rate);
            self.frames_analyzed += 1;
        }
    }

    fn reset(&mut self) {
        self.analyzer.reset();
        self.features.reset();
        self.latest = SpectralFeatureSet::default();
        self.frames_analyzed = 0;
        self.last_frame = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    /// Builds a flat single-sided magnitude spectrum of `bins` equal values.
    fn flat_spectrum(bins: usize, value: Sample) -> Vec<Sample> {
        vec![value; bins]
    }

    /// Builds a single-sided spectrum with one unit peak at `bin`.
    fn peak_spectrum(bins: usize, bin: usize) -> Vec<Sample> {
        let mut m = vec![0.0; bins];
        m[bin] = 1.0;
        m
    }

    #[test]
    fn centroid_matches_single_peak_frequency() {
        // size = (5 - 1) * 2 = 8; with SR = 8 Hz, hz_per_bin = 1 Hz.
        let mut f = SpectralFeatures::new(0.85);
        let set = f.analyze(&peak_spectrum(5, 3), 8);
        assert!((set.centroid - 3.0).abs() < 1e-4, "centroid {}", set.centroid);
        // A lone peak has zero spread about its own frequency.
        assert!(set.spread < 1e-3, "spread {}", set.spread);
    }

    #[test]
    fn low_peak_has_lower_centroid_than_high_peak() {
        let mut f = SpectralFeatures::new(0.85);
        let low = f.analyze(&peak_spectrum(9, 1), SR).centroid;
        f.reset();
        let high = f.analyze(&peak_spectrum(9, 7), SR).centroid;
        assert!(high > low, "high {high} should exceed low {low}");
    }

    #[test]
    fn flatness_near_one_for_flat_spectrum() {
        let mut f = SpectralFeatures::new(0.85);
        let set = f.analyze(&flat_spectrum(64, 0.5), SR);
        assert!(set.flatness > 0.95, "flatness {}", set.flatness);
        // A flat spectrum has a crest factor near unity.
        assert!((set.crest - 1.0).abs() < 0.1, "crest {}", set.crest);
    }

    #[test]
    fn flatness_low_for_single_tone() {
        let mut f = SpectralFeatures::new(0.85);
        let set = f.analyze(&peak_spectrum(64, 10), SR);
        assert!(set.flatness < 0.05, "flatness {}", set.flatness);
        // A tonal spectrum has a large crest factor.
        assert!(set.crest > 10.0, "crest {}", set.crest);
    }

    #[test]
    fn flatness_is_bounded() {
        let mut f = SpectralFeatures::new(0.85);
        for &v in &[0.01f32, 0.5, 2.0, 100.0] {
            let set = f.analyze(&flat_spectrum(32, v), SR);
            assert!((0.0..=1.0).contains(&set.flatness), "flatness {}", set.flatness);
        }
    }

    #[test]
    fn rolloff_is_monotonic_in_fraction() {
        let mags = flat_spectrum(16, 1.0);
        let mut lo = SpectralFeatures::new(0.25);
        let mut hi = SpectralFeatures::new(0.90);
        let r_lo = lo.analyze(&mags, SR).rolloff;
        let r_hi = hi.analyze(&mags, SR).rolloff;
        assert!(r_hi >= r_lo, "rolloff hi {r_hi} < lo {r_lo}");
    }

    #[test]
    fn flux_is_zero_in_steady_state() {
        let mut f = SpectralFeatures::new(0.85);
        let mags = flat_spectrum(32, 0.7);
        let _ = f.analyze(&mags, SR); // primes prev frame
        let set = f.analyze(&mags, SR);
        assert!(set.flux < 1e-4, "steady-state flux {}", set.flux);
    }

    #[test]
    fn flux_is_large_on_change() {
        let mut f = SpectralFeatures::new(0.85);
        let _ = f.analyze(&flat_spectrum(32, 0.0), SR);
        let set = f.analyze(&peak_spectrum(32, 5), SR);
        assert!(set.flux > 0.5, "onset flux {}", set.flux);
    }

    #[test]
    fn slope_is_negative_for_decreasing_spectrum() {
        // Magnitudes that fall off with frequency should have a negative slope.
        let mags: Vec<Sample> = (0..16).map(|k| 16.0 - k as Sample).collect();
        let mut f = SpectralFeatures::new(0.85);
        let set = f.analyze(&mags, SR);
        assert!(set.slope < 0.0, "slope {}", set.slope);
        // The MPEG-7 decrease is also negative for a falling spectrum.
        assert!(set.decrease < 0.0, "decrease {}", set.decrease);
    }

    #[test]
    fn kurtosis_and_spread_are_sane() {
        // A broad spectrum has larger spread than a narrow one.
        let narrow: Vec<Sample> = (0..64).map(|k| if (28..=36).contains(&k) { 1.0 } else { 0.0 }).collect();
        let broad = flat_spectrum(64, 1.0);
        let mut f = SpectralFeatures::new(0.85);
        let s_narrow = f.analyze(&narrow, SR).spread;
        f.reset();
        let s_broad = f.analyze(&broad, SR).spread;
        assert!(s_broad > s_narrow, "broad {s_broad} <= narrow {s_narrow}");
        f.reset();
        let set = f.analyze(&narrow, SR);
        assert!(set.kurtosis.is_finite());
        assert!(set.skewness.is_finite());
    }

    #[test]
    fn silent_frame_is_all_zero() {
        let mut f = SpectralFeatures::new(0.85);
        let set = f.analyze(&flat_spectrum(32, 0.0), SR);
        assert_eq!(set, SpectralFeatureSet::default());
    }

    #[test]
    fn short_spectrum_is_all_zero() {
        let mut f = SpectralFeatures::new(0.85);
        assert_eq!(f.analyze(&[1.0], SR), SpectralFeatureSet::default());
        assert_eq!(f.analyze(&[], SR), SpectralFeatureSet::default());
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut f = SpectralFeatures::new(0.85);
        let mags = [1.0, Sample::NAN, Sample::INFINITY, 0.5, 0.25];
        let set = f.analyze(&mags, SR);
        assert!(set.centroid.is_finite());
        assert!(set.spread.is_finite());
        assert!(set.flatness.is_finite());
        assert!(set.flux.is_finite());
        assert!(set.slope.is_finite());
        assert!(set.decrease.is_finite());
    }

    #[test]
    fn rolloff_fraction_is_clamped() {
        assert_eq!(SpectralFeatures::new(2.0).rolloff_fraction(), 1.0);
        assert_eq!(SpectralFeatures::new(-1.0).rolloff_fraction(), 0.0);
        assert_eq!(
            SpectralFeatures::new(Sample::NAN).rolloff_fraction(),
            DEFAULT_ROLLOFF_FRACTION
        );
        let mut f = SpectralFeatures::new(0.5);
        f.set_rolloff_fraction(5.0);
        assert_eq!(f.rolloff_fraction(), 1.0);
        f.set_rolloff_fraction(Sample::NAN);
        assert_eq!(f.rolloff_fraction(), 1.0);
    }

    #[test]
    fn default_is_default_fraction() {
        assert_eq!(
            SpectralFeatures::default().rolloff_fraction(),
            DEFAULT_ROLLOFF_FRACTION
        );
    }

    #[test]
    fn feature_set_default_is_zero() {
        let d = SpectralFeatureSet::default();
        assert_eq!(d.centroid, 0.0);
        assert_eq!(d.flatness, 0.0);
        assert_eq!(d.flux, 0.0);
    }

    /// Feeds `blocks` blocks of a sine at `freq_hz` through the node.
    fn drive_node(node: &mut SpectralFeaturesNode, freq_hz: Sample, blocks: usize) {
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        let mut n = 0u64;
        for _ in 0..blocks {
            let mut input = AudioBuffer::new(ChannelLayout::Mono, 512);
            for s in input.channel_mut(0).iter_mut() {
                *s = ops::sin(w * n as Sample);
                n += 1;
            }
            let output = AudioBuffer::new(ChannelLayout::Mono, 512);
            let inputs = [input];
            let mut outputs = [output];
            let ctx = RenderContext { sample_rate: SR, frames: 512, playhead: 0 };
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
    }

    #[test]
    fn node_passes_signal_through_unchanged() {
        let mut node = SpectralFeaturesNode::new(1024, 512, Window::Hann, 0.85);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 512);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = (i as Sample * 0.001).sin();
        }
        let expected: Vec<Sample> = input.channel(0).to_vec();
        let output = AudioBuffer::new(ChannelLayout::Mono, 512);
        let inputs = [input];
        let mut outputs = [output];
        let ctx = RenderContext { sample_rate: SR, frames: 512, playhead: 0 };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        assert_eq!(outputs[0].channel(0), expected.as_slice());
    }

    #[test]
    fn node_updates_latest_and_counts_frames() {
        let mut node = SpectralFeaturesNode::new(1024, 512, Window::Hann, 0.85);
        assert_eq!(node.frames_analyzed(), 0);
        drive_node(&mut node, 8_000.0, 8);
        assert!(node.frames_analyzed() > 0);
        let bright = node.latest().centroid;
        assert!(bright > 4_000.0, "bright centroid {bright}");

        let mut dark = SpectralFeaturesNode::new(1024, 512, Window::Hann, 0.85);
        drive_node(&mut dark, 200.0, 8);
        assert!(bright > dark.latest().centroid, "bright should exceed dark");
    }

    #[test]
    fn node_reset_clears_state() {
        let mut node = SpectralFeaturesNode::new(1024, 512, Window::Hann, 0.85);
        drive_node(&mut node, 1_000.0, 8);
        assert!(node.frames_analyzed() > 0);
        node.reset();
        assert_eq!(node.frames_analyzed(), 0);
        assert_eq!(node.latest(), SpectralFeatureSet::default());
        assert_eq!(node.analyzer().frames_computed(), 0);
    }

    #[test]
    fn node_exposes_inner_handles() {
        let mut node = SpectralFeaturesNode::new(512, 256, Window::Hann, 0.85);
        assert_eq!(node.analyzer().hop(), 256);
        assert_eq!(node.features().rolloff_fraction(), 0.85);
        node.features_mut().set_rolloff_fraction(0.5);
        assert_eq!(node.features().rolloff_fraction(), 0.5);
        node.analyzer_mut().reset();
    }
}
