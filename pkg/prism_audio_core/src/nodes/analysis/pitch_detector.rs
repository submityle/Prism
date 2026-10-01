//! Monophonic fundamental-frequency (`f0`) estimator using the `YIN`
//! difference function.
//!
//! Pitch detection answers "what note is this?" for a monophonic source: a
//! sung vowel, a bowed string, a bass line. This module implements the `YIN`
//! algorithm (de Cheveigne and Kawahara), a time-domain autocorrelation
//! variant that is robust to the octave errors that plague a raw
//! autocorrelation peak-pick. Like the other analysis taps it is a read-only
//! meter: the audio flows through [`PitchDetectorNode`] unchanged while the
//! detector observes it.
//!
//! # Model
//!
//! For a frame `x` of length `W + tau_max` the detector evaluates the squared
//! difference function over candidate lag `tau`,
//!
//! ```text
//! d(tau) = sum_{j=0}^{W-1} (x[j] - x[j + tau])^2,
//! ```
//!
//! which dips toward zero whenever `tau` equals the signal period. A raw
//! difference minimum is biased toward `tau = 0`, so `YIN` divides by the
//! running mean to form the cumulative-mean-normalised difference
//!
//! ```text
//! d'(0) = 1,   d'(tau) = d(tau) / ((1 / tau) * sum_{j=1}^{tau} d(j)).
//! ```
//!
//! The estimator walks `tau` from `tau_min` (set by the maximum frequency)
//! upward and takes the first lag whose `d'` falls below an absolute threshold
//! and sits at a local minimum; if none qualifies it falls back to the global
//! minimum of `d'`. A parabolic interpolation of the three `d'` values around
//! the chosen lag refines it to sub-sample resolution, and the pitch is
//! `f0 = sample_rate / tau`. The periodicity confidence is `1 - d'(tau)`.
//!
//! # Real-time contract
//!
//! The sample ring and every scratch buffer (the chronological frame copy and
//! the two `tau_max + 1`-length difference arrays) are allocated once in
//! [`PitchDetector::new`] / [`PitchDetectorNode::new`]. The hot path
//! ([`PitchDetector::feed_sample`], [`PitchDetectorNode::process`]) performs no
//! allocation, takes no locks, and cannot panic; a full `YIN` pass runs every
//! `hop` fed samples and costs `O(tau_max * W)` arithmetic. Non-finite inputs
//! are treated as silence so the ring can never be poisoned by a `NaN`.
//!
//! # Relationship
//!
//! This detector complements [`spectrum`](super::spectrum): the spectrum
//! analyzer reports where energy sits across all frequencies, while this tap
//! condenses a monophonic signal to a single `f0` with a confidence score.
//! Both pass the signal through untouched and preallocate their working
//! buffers, and neither alters the dynamics shaped by
//! [`dynamics`](crate::nodes::dynamics). It shares the [`Sample`] scalar and
//! the ring-plus-hop idiom with the other analysis meters.
//!
//! # Provenance
//!
//! The `YIN` fundamental-frequency estimator is a publicly published,
//! classic-DSP method (de Cheveigne and Kawahara, 2002). This module contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code**; it is implemented purely from
//! that published description. It is pure classic DSP with no AI or ML.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;

/// Default `YIN` absolute threshold for accepting a periodicity dip.
pub const DEFAULT_YIN_THRESHOLD: Sample = 0.15;
/// Default lowest detectable fundamental in hertz.
pub const DEFAULT_MIN_HZ: Sample = 50.0;
/// Default highest detectable fundamental in hertz.
pub const DEFAULT_MAX_HZ: Sample = 2_000.0;
/// Default hop: run a fresh estimate roughly every `256` samples.
pub const DEFAULT_HOP: usize = 256;

/// Smallest admissible integration lag in samples.
const MIN_TAU: usize = 2;
/// Frame energy below this (per-sample mean square) is treated as silence.
const SILENCE_ENERGY: f64 = 1.0e-10;

/// A single fundamental-frequency estimate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PitchEstimate {
    /// Estimated fundamental frequency in hertz. Zero when unvoiced.
    pub frequency_hz: Sample,
    /// Periodicity confidence in `0..=1` (`1 - d'` at the chosen lag).
    pub confidence: Sample,
    /// Whether the frame was judged voiced (a periodicity dip was accepted).
    pub is_voiced: bool,
}

/// A monophonic `YIN` pitch detector owning a fixed ring and scratch buffers.
///
/// The detector emits a fresh [`PitchEstimate`] every `hop` fed samples; read
/// the most recent one with [`latest`](Self::latest).
#[derive(Debug, Clone)]
pub struct PitchDetector {
    sample_rate: u32,
    tau_min: usize,
    tau_max: usize,
    window: usize,
    hop: usize,
    threshold: Sample,
    ring: Vec<Sample>,
    frame: Vec<Sample>,
    diff: Vec<f64>,
    cmnd: Vec<f64>,
    write: usize,
    since_last: usize,
    latest: PitchEstimate,
    frames_computed: u64,
}

impl PitchDetector {
    /// Builds a detector for the frequency range `[min_hz, max_hz]` that runs a
    /// fresh estimate every `hop` samples.
    ///
    /// The range is clamped into the open band `(0, nyquist)` with `min_hz`
    /// below `max_hz`; `hop` is clamped to at least one sample. All working
    /// storage is allocated here.
    #[must_use]
    pub fn new(sample_rate: u32, min_hz: Sample, max_hz: Sample, hop: usize) -> Self {
        let sr = sample_rate.max(1);
        let sr_f = sr as Sample;
        let nyquist = sr_f * 0.5;
        // Order and clamp the requested band into the open Nyquist interval.
        let (lo_hz, hi_hz) = if min_hz <= max_hz {
            (min_hz, max_hz)
        } else {
            (max_hz, min_hz)
        };
        let hi_hz = hi_hz.clamp(1.0, nyquist * 0.999).max(2.0);
        let lo_hz = lo_hz.clamp(1.0, hi_hz * 0.5).max(1.0);

        let tau_min = clamp_lag(ops::round(sr_f / hi_hz), MIN_TAU, usize::MAX / 4);
        let raw_max = clamp_lag(ops::round(sr_f / lo_hz), MIN_TAU + 2, usize::MAX / 4);
        let tau_max = raw_max.max(tau_min + 2);
        let window = tau_max;
        let ring_len = window + tau_max;
        let hop = hop.max(1);

        Self {
            sample_rate: sr,
            tau_min,
            tau_max,
            window,
            hop,
            threshold: DEFAULT_YIN_THRESHOLD,
            ring: vec![0.0; ring_len],
            frame: vec![0.0; ring_len],
            diff: vec![0.0; tau_max + 1],
            cmnd: vec![0.0; tau_max + 1],
            write: 0,
            since_last: 0,
            latest: PitchEstimate::default(),
            frames_computed: 0,
        }
    }

    /// The sample rate the detector was built for.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The hop between successive estimates, in samples.
    #[inline]
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// The lowest detectable fundamental in hertz for the configured range.
    #[inline]
    #[must_use]
    pub fn min_frequency_hz(&self) -> Sample {
        self.sample_rate as Sample / self.tau_max as Sample
    }

    /// The highest detectable fundamental in hertz for the configured range.
    #[inline]
    #[must_use]
    pub fn max_frequency_hz(&self) -> Sample {
        self.sample_rate as Sample / self.tau_min as Sample
    }

    /// The `YIN` absolute threshold used to accept a periodicity dip.
    #[inline]
    #[must_use]
    pub fn threshold(&self) -> Sample {
        self.threshold
    }

    /// Sets the `YIN` absolute threshold, clamped to `(0, 1)`.
    #[inline]
    pub fn set_threshold(&mut self, threshold: Sample) {
        self.threshold = threshold.clamp(1.0e-3, 0.999);
    }

    /// The most recent estimate, or the default (unvoiced) estimate before the
    /// first full frame has been computed.
    #[inline]
    #[must_use]
    pub fn latest(&self) -> PitchEstimate {
        self.latest
    }

    /// The number of complete `YIN` passes run so far.
    #[inline]
    #[must_use]
    pub fn frames_computed(&self) -> u64 {
        self.frames_computed
    }

    /// Clears the ring, scratch state, and latest estimate.
    pub fn reset(&mut self) {
        for v in &mut self.ring {
            *v = 0.0;
        }
        self.write = 0;
        self.since_last = 0;
        self.latest = PitchEstimate::default();
        self.frames_computed = 0;
    }

    /// Pushes one sample into the ring, running a `YIN` pass every `hop`
    /// samples. Non-finite inputs are treated as silence.
    pub fn feed_sample(&mut self, x: Sample) {
        let value = if x.is_finite() { x } else { 0.0 };
        self.ring[self.write] = value;
        self.write += 1;
        if self.write == self.ring.len() {
            self.write = 0;
        }
        self.since_last += 1;
        if self.since_last >= self.hop {
            self.since_last = 0;
            self.compute_frame();
        }
    }

    /// Loads the most recent frame oldest-first and runs the full `YIN` pass.
    fn compute_frame(&mut self) {
        let ring_len = self.ring.len();

        // Load the most recent `ring_len` samples oldest-first.
        let mut idx = self.write;
        for slot in self.frame.iter_mut() {
            *slot = self.ring[idx];
            idx += 1;
            if idx == ring_len {
                idx = 0;
            }
        }

        self.frames_computed += 1;

        // Reject silent frames outright.
        let mut energy = 0.0_f64;
        for &s in &self.frame[..self.window] {
            let v = f64::from(s);
            energy += v * v;
        }
        if energy / self.window as f64 <= SILENCE_ENERGY {
            self.latest = PitchEstimate::default();
            return;
        }

        self.latest = run_yin(
            &self.frame,
            self.window,
            self.tau_min,
            self.tau_max,
            self.threshold,
            self.sample_rate as Sample,
            &mut self.diff,
            &mut self.cmnd,
        );
    }
}

/// Clamps a rounded lag estimate into `[lo, hi]` as a sample count.
fn clamp_lag(rounded: Sample, lo: usize, hi: usize) -> usize {
    if !rounded.is_finite() || rounded < lo as Sample {
        return lo;
    }
    // `rounded` is finite and at least `lo`, so this cast is well defined.
    let value = ops::round(rounded) as usize;
    value.clamp(lo, hi)
}

/// Runs one `YIN` pass over `frame`, writing scratch into `diff` / `cmnd`.
#[expect(
    clippy::too_many_arguments,
    reason = "a free YIN pass takes the frame, its window, both lag bounds, the threshold, the rate, and two preallocated scratch buffers"
)]
fn run_yin(
    frame: &[Sample],
    window: usize,
    tau_min: usize,
    tau_max: usize,
    threshold: Sample,
    sample_rate: Sample,
    diff: &mut [f64],
    cmnd: &mut [f64],
) -> PitchEstimate {
    // Squared difference function d(tau) over all admissible lags.
    diff[0] = 0.0;
    for tau in 1..=tau_max {
        let d: f64 = frame[..window]
            .iter()
            .zip(&frame[tau..tau + window])
            .map(|(&a, &b)| {
                let delta = f64::from(a) - f64::from(b);
                delta * delta
            })
            .sum();
        diff[tau] = d;
    }

    // Cumulative-mean-normalised difference d'(tau).
    cmnd[0] = 1.0;
    let mut running = 0.0_f64;
    for tau in 1..=tau_max {
        running += diff[tau];
        cmnd[tau] = if running > 0.0 {
            diff[tau] * tau as f64 / running
        } else {
            1.0
        };
    }

    // Absolute-threshold search: first dip below threshold at a local minimum.
    let thresh = f64::from(threshold);
    let mut chosen = None;
    let mut tau = tau_min;
    while tau <= tau_max {
        if cmnd[tau] < thresh {
            let mut t = tau;
            while t < tau_max && cmnd[t + 1] < cmnd[t] {
                t += 1;
            }
            chosen = Some(t);
            break;
        }
        tau += 1;
    }

    // Fall back to the global minimum of d' across the admissible band.
    let chosen = chosen.unwrap_or_else(|| {
        let mut best = tau_min;
        let mut best_val = cmnd[tau_min];
        for (t, &val) in cmnd.iter().enumerate().take(tau_max + 1).skip(tau_min + 1) {
            if val < best_val {
                best_val = val;
                best = t;
            }
        }
        best
    });

    // Parabolic interpolation of the three d' values around the chosen lag.
    let refined = parabolic_lag(cmnd, chosen, tau_min, tau_max);
    let frequency_hz = if refined > 0.0 {
        sample_rate / refined
    } else {
        0.0
    };
    let dip = cmnd[chosen];
    // `dip` is a finite non-negative ratio, so `<=` is a clean voiced guard.
    let is_voiced = dip <= thresh;
    let confidence = (1.0 - dip).clamp(0.0, 1.0) as Sample;

    PitchEstimate {
        frequency_hz: if is_voiced { frequency_hz } else { 0.0 },
        confidence,
        is_voiced,
    }
}

/// Refines the integer lag `chosen` to sub-sample resolution by fitting a
/// parabola to the three surrounding `d'` values.
fn parabolic_lag(cmnd: &[f64], chosen: usize, tau_min: usize, tau_max: usize) -> Sample {
    if chosen <= tau_min || chosen >= tau_max {
        return chosen as Sample;
    }
    let s0 = cmnd[chosen - 1];
    let s1 = cmnd[chosen];
    let s2 = cmnd[chosen + 1];
    let denom = s0 + s2 - 2.0 * s1;
    if denom == 0.0 {
        return chosen as Sample;
    }
    let shift = 0.5 * (s0 - s2) / denom;
    // Keep the refinement within one sample of the integer lag.
    let shift = shift.clamp(-1.0, 1.0);
    chosen as Sample + shift as Sample
}

/// A pass-through node that taps its input into a [`PitchDetector`].
///
/// The first input channel feeds the detector; every channel is copied to the
/// output unchanged. Read the running estimate through
/// [`detector`](Self::detector).
#[derive(Debug, Clone)]
pub struct PitchDetectorNode {
    detector: PitchDetector,
}

impl PitchDetectorNode {
    /// Builds a node wrapping a [`PitchDetector`] for the given range and hop.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
    /// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
    /// use prism_audio_core::nodes::analysis::pitch_detector::PitchDetectorNode;
    ///
    /// let sr = 48_000u32;
    /// let mut node = PitchDetectorNode::new(sr, 50.0, 2_000.0, 256);
    /// // A 440 Hz sine over several blocks should read close to 440 Hz.
    /// let freq = 440.0f32;
    /// let w = core::f32::consts::TAU * freq / sr as f32;
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
    /// let est = node.detector().latest();
    /// assert!(est.is_voiced);
    /// assert!((est.frequency_hz - 440.0).abs() < 10.0);
    /// ```
    #[must_use]
    pub fn new(sample_rate: u32, min_hz: Sample, max_hz: Sample, hop: usize) -> Self {
        Self {
            detector: PitchDetector::new(sample_rate, min_hz, max_hz, hop),
        }
    }

    /// Immutable access to the underlying detector (to read estimates).
    #[inline]
    #[must_use]
    pub fn detector(&self) -> &PitchDetector {
        &self.detector
    }

    /// Mutable access to the underlying detector (to retune or reset).
    #[inline]
    pub fn detector_mut(&mut self) -> &mut PitchDetector {
        &mut self.detector
    }
}

impl AudioNode for PitchDetectorNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let frames = output.active_frames();

        // Pass the signal through unchanged.
        let copy_channels = out_channels.min(input.channels());
        for ch in 0..copy_channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }

        // Feed the first channel into the detector.
        if input.channels() >= 1 {
            let mono = input.channel(0);
            for &x in &mono[..frames] {
                self.detector.feed_sample(x);
            }
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    /// Feeds `frames` samples of a sine at `freq_hz` and returns the estimate.
    fn detect_sine(freq_hz: Sample, frames: usize, hop: usize) -> PitchEstimate {
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, hop);
        let w = core::f32::consts::TAU * freq_hz / SR as Sample;
        for n in 0..frames {
            det.feed_sample(ops::sin(w * n as Sample));
        }
        det.latest()
    }

    #[test]
    fn detects_440_hz_sine() {
        let est = detect_sine(440.0, 8_192, 256);
        assert!(est.is_voiced, "440 Hz sine should be voiced");
        assert!(
            (est.frequency_hz - 440.0).abs() < 5.0,
            "440 Hz read as {}",
            est.frequency_hz
        );
        assert!(est.confidence > 0.8, "confidence {}", est.confidence);
    }

    #[test]
    fn detects_low_and_high_tones() {
        let low = detect_sine(80.0, 16_384, 512);
        assert!(low.is_voiced);
        assert!((low.frequency_hz - 80.0).abs() < 3.0, "low {}", low.frequency_hz);

        let high = detect_sine(1_500.0, 8_192, 256);
        assert!(high.is_voiced);
        assert!(
            (high.frequency_hz - 1_500.0).abs() < 25.0,
            "high {}",
            high.frequency_hz
        );
    }

    #[test]
    fn silence_is_unvoiced() {
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        for _ in 0..4_096 {
            det.feed_sample(0.0);
        }
        let est = det.latest();
        assert!(!est.is_voiced);
        assert_eq!(est.frequency_hz, 0.0);
    }

    #[test]
    fn white_noise_has_low_confidence() {
        // A crude deterministic xorshift noise source.
        let mut state = 0x1234_5678u32;
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        for _ in 0..8_192 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let x = (state as Sample / u32::MAX as Sample) * 2.0 - 1.0;
            det.feed_sample(x);
        }
        let est = det.latest();
        // Noise may or may not trip the threshold, but it is never a crisp tone.
        assert!(est.confidence < 0.95, "noise confidence {}", est.confidence);
    }

    #[test]
    fn complex_tone_locks_to_fundamental() {
        // Fundamental at 220 Hz plus a strong second and third harmonic should
        // still resolve the 220 Hz period rather than an octave above it.
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        let f0 = 220.0;
        let w = core::f32::consts::TAU * f0 / SR as Sample;
        for n in 0..8_192 {
            let t = n as Sample;
            let x = ops::sin(w * t) + 0.5 * ops::sin(2.0 * w * t) + 0.3 * ops::sin(3.0 * w * t);
            det.feed_sample(x * 0.5);
        }
        let est = det.latest();
        assert!(est.is_voiced);
        assert!((est.frequency_hz - 220.0).abs() < 4.0, "f0 {}", est.frequency_hz);
    }

    #[test]
    fn range_is_clamped_and_ordered() {
        // Swapped bounds are reordered and clamped into the open Nyquist band.
        let det = PitchDetector::new(SR, 2_000.0, 50.0, 256);
        assert!(det.min_frequency_hz() < det.max_frequency_hz());
        assert!(det.max_frequency_hz() < SR as Sample * 0.5);
        assert!(det.min_frequency_hz() > 0.0);
    }

    #[test]
    fn hop_is_at_least_one() {
        let det = PitchDetector::new(SR, 50.0, 2_000.0, 0);
        assert_eq!(det.hop(), 1);
    }

    #[test]
    fn threshold_is_clamped() {
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        det.set_threshold(5.0);
        assert!(det.threshold() < 1.0);
        det.set_threshold(-1.0);
        assert!(det.threshold() > 0.0);
    }

    #[test]
    fn frames_computed_tracks_hops() {
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        for _ in 0..1_024 {
            det.feed_sample(0.0);
        }
        assert_eq!(det.frames_computed(), 4);
    }

    #[test]
    fn reset_clears_estimate() {
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        let w = core::f32::consts::TAU * 440.0 / SR as Sample;
        for n in 0..4_096 {
            det.feed_sample(ops::sin(w * n as Sample));
        }
        assert!(det.latest().is_voiced);
        det.reset();
        assert!(!det.latest().is_voiced);
        assert_eq!(det.frames_computed(), 0);
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut det = PitchDetector::new(SR, 50.0, 2_000.0, 256);
        let w = core::f32::consts::TAU * 440.0 / SR as Sample;
        for n in 0..4_096 {
            let x = if n % 500 == 0 { Sample::NAN } else { ops::sin(w * n as Sample) };
            det.feed_sample(x);
        }
        let est = det.latest();
        assert!(est.frequency_hz.is_finite());
        assert!(est.confidence.is_finite());
    }

    #[test]
    fn node_passes_signal_through_unchanged() {
        let mut node = PitchDetectorNode::new(SR, 50.0, 2_000.0, 256);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 128);
        for ch in 0..input.channels() {
            let data = input.channel_mut(ch);
            for (i, s) in data.iter_mut().enumerate() {
                *s = (i as Sample) * 0.01 - 0.5;
            }
        }
        let expected = [input.channel(0).to_vec(), input.channel(1).to_vec()];
        let output = AudioBuffer::new(ChannelLayout::Stereo, 128);
        let inputs = [input];
        let mut outputs = [output];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 128,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let [out] = outputs;
        assert_eq!(out.channel(0), expected[0].as_slice());
        assert_eq!(out.channel(1), expected[1].as_slice());
    }

    #[test]
    fn default_estimate_is_unvoiced() {
        let est = PitchEstimate::default();
        assert!(!est.is_voiced);
        assert_eq!(est.frequency_hz, 0.0);
        assert_eq!(est.confidence, 0.0);
    }

    #[test]
    fn parabolic_refinement_stays_near_integer_lag() {
        // The refined lag never jumps more than one sample from its integer bin.
        let cmnd = [1.0_f64, 0.9, 0.2, 0.05, 0.3, 0.8];
        let refined = parabolic_lag(&cmnd, 3, 2, 5);
        assert!((refined - 3.0).abs() <= 1.0, "refined {refined}");
    }
}
