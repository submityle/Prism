//! Windowed radix-2 FFT magnitude-spectrum analyzer for metering and display.
//!
//! A spectrum analyzer is the frequency-domain counterpart of the time-domain
//! meters in this submodule: instead of reporting how loud a signal is or how
//! wide its stereo image sits, it reports *where* the energy lives across
//! frequency. This module is the real-time-safe front end that turns a running
//! audio stream into a sequence of single-sided magnitude spectra -- exactly
//! the data a spectrogram, a real-time analyzer display, or a tuning tool
//! consumes. It does not render pixels; it produces the magnitude bins a
//! renderer plots.
//!
//! # Model
//!
//! The analyzer buffers incoming samples in a fixed ring and, every `hop`
//! samples, lifts the most recent `size`-sample window into the frequency
//! domain with a decimation-in-time (DIT) radix-2 Cooley-Tukey FFT. The
//! transform length `size` is always a power of two (rounded up from the
//! requested length at construction) so the classic butterfly recursion
//! applies directly. Before the transform the window is multiplied by one of
//! the standard tapers ([`Window`]) to trade main-lobe width against
//! side-lobe rejection, and the single-sided magnitude is normalised so that a
//! pure tone sitting exactly on an analysis bin reads back its own linear
//! amplitude.
//!
//! The normalisation uses the coherent gain of the chosen window (the sum of
//! its coefficients). A real sinusoid of amplitude `A` placed on an interior
//! bin produces a transform magnitude of `A / 2` times the window sum, so the
//! interior bins are scaled by `2 / window_sum`; the direct-current (bin 0) and
//! Nyquist bins have no mirror partner and are scaled by `1 / window_sum`.
//!
//! # Real-time contract
//!
//! Every buffer -- the sample ring, the real and imaginary scratch arrays, the
//! precomputed twiddle-factor tables, the bit-reversal permutation table, the
//! window coefficients, and the magnitude output -- is allocated once in
//! [`SpectrumAnalyzer::new`] / [`SpectrumNode::new`]. The per-sample hot path
//! ([`SpectrumAnalyzer::feed_sample`], [`SpectrumNode::process`]) performs no
//! allocation, takes no locks, and cannot panic: non-finite inputs are treated
//! as silence so the ring and the transform can never be poisoned by a `NaN`.
//! Reading the spectrum back ([`SpectrumAnalyzer::magnitudes`]) borrows the
//! internal buffer and copies nothing. All transcendental math routes through
//! [`bevy_math::ops`], so the transform is bit-reproducible across platforms.
//!
//! # Provenance
//!
//! The radix-2 Cooley-Tukey FFT and the Hann, Hamming, and Blackman window
//! tapers are textbook classical signal-processing constructions described in
//! every digital-signal-processing reference; the single-sided magnitude
//! normalisation is the elementary coherent-gain correction. This module reuses
//! only this crate's own [`Sample`] scalar and graph traits. It is pure classic
//! DSP with no AI or ML and contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from those publicly documented algorithms.
//!
//! # Relationship
//!
//! This analyzer is the spectral complement to the scalar
//! [`LoudnessMeter`](crate::nodes::analysis::loudness::LoudnessMeter) and
//! [`CorrelationMeter`](crate::nodes::analysis::correlation::CorrelationMeter)
//! and to the vectorscope
//! [`Goniometer`](crate::nodes::analysis::goniometer::Goniometer): the loudness
//! meter answers *how loud*, the correlation meter and goniometer answer *how
//! wide*, and this module answers *at which frequencies*. It shares no code
//! with any of them. The window tapers it defines are the same family of
//! analysis windows used throughout classical spectral estimation.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;

/// Smallest permitted transform length.
pub const MIN_FFT_SIZE: usize = 2;

/// Default transform length (a good balance of resolution and latency).
pub const DEFAULT_FFT_SIZE: usize = 2048;

/// Default hop size (half-overlapped frames).
pub const DEFAULT_HOP: usize = 1024;

/// Analysis window tapers applied before the transform.
///
/// Each taper trades main-lobe width (frequency resolution) against side-lobe
/// rejection (spectral leakage). [`Window::Rectangular`] has the narrowest main
/// lobe and the worst leakage; [`Window::Blackman`] has the widest main lobe and
/// the best leakage rejection, with Hann and Hamming in between.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Window {
    /// No taper (unity weight). Narrowest main lobe, worst spectral leakage.
    Rectangular,
    /// Raised-cosine Hann taper. A good general-purpose balance.
    #[default]
    Hann,
    /// Hamming taper. Lower nearest side lobe than Hann.
    Hamming,
    /// Blackman taper. Widest main lobe, strongest side-lobe rejection.
    Blackman,
}

impl Window {
    /// Returns the window coefficient for sample `n` of a `size`-point window.
    ///
    /// A window of a single point (or fewer) is treated as unity so the
    /// coefficient is always finite.
    #[must_use]
    pub fn coefficient(self, n: usize, size: usize) -> Sample {
        if size <= 1 {
            return 1.0;
        }
        let denom = (size - 1) as Sample;
        let phase = TAU * n as Sample / denom;
        match self {
            Window::Rectangular => 1.0,
            Window::Hann => 0.5 - 0.5 * ops::cos(phase),
            Window::Hamming => 0.54 - 0.46 * ops::cos(phase),
            Window::Blackman => 0.42 - 0.5 * ops::cos(phase) + 0.08 * ops::cos(2.0 * phase),
        }
    }

    /// Returns the analytic coherent gain (mean coefficient) of the taper.
    ///
    /// This is the asymptotic ratio of the window sum to its length and is used
    /// as a reference for amplitude calibration.
    #[must_use]
    pub fn coherent_gain(self) -> Sample {
        match self {
            Window::Rectangular => 1.0,
            Window::Hann => 0.5,
            Window::Hamming => 0.54,
            Window::Blackman => 0.42,
        }
    }
}

/// Reverses the lowest `bits` bits of `value` (for the FFT index permutation).
fn reverse_low_bits(mut value: usize, bits: u32) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    result
}

/// Rounds `requested` up to the next power of two, never below [`MIN_FFT_SIZE`].
fn power_of_two_at_least(requested: usize) -> usize {
    let mut size = MIN_FFT_SIZE;
    while size < requested {
        size <<= 1;
    }
    size
}

/// Converts a sample stream into a sequence of single-sided magnitude spectra.
///
/// The analyzer owns a fixed-capacity ring and a complete set of preallocated
/// scratch buffers; it emits a new magnitude frame every `hop` fed samples.
///
/// ```
/// use prism_audio_core::nodes::analysis::spectrum::{SpectrumAnalyzer, Window};
/// use core::f32::consts::TAU;
///
/// // A 16-point rectangular analysis, one frame per full window.
/// let mut analyzer = SpectrumAnalyzer::new(16, 16, Window::Rectangular);
/// for n in 0..16 {
///     let phase = TAU * 2.0 * n as f32 / 16.0;
///     analyzer.feed_sample(phase.cos());
/// }
/// assert_eq!(analyzer.frames_computed(), 1);
/// // A unit cosine on bin 2 reads back its amplitude of 1.0.
/// let magnitudes = analyzer.magnitudes();
/// assert!((magnitudes[2] - 1.0).abs() < 0.05);
/// ```
#[derive(Clone, Debug)]
pub struct SpectrumAnalyzer {
    size: usize,
    hop: usize,
    window: Window,
    win: Vec<Sample>,
    tw_re: Vec<Sample>,
    tw_im: Vec<Sample>,
    rev: Vec<usize>,
    ring: Vec<Sample>,
    re: Vec<Sample>,
    im: Vec<Sample>,
    mag: Vec<Sample>,
    write: usize,
    since_last: usize,
    inv_window_sum: Sample,
    two_inv_window_sum: Sample,
    frames_computed: u64,
}

impl SpectrumAnalyzer {
    /// Builds an analyzer with the requested transform length and hop.
    ///
    /// `requested_size` is rounded up to the next power of two (never below
    /// [`MIN_FFT_SIZE`]); `hop` is clamped to at least one sample. All working
    /// buffers are allocated here so the hot path never allocates.
    #[must_use]
    pub fn new(requested_size: usize, hop: usize, window: Window) -> Self {
        let size = power_of_two_at_least(requested_size);
        let hop = hop.max(1);
        let half = size / 2;

        let win: Vec<Sample> = (0..size).map(|n| window.coefficient(n, size)).collect();
        let mut acc = 0.0f64;
        for &w in &win {
            acc += f64::from(w);
        }
        let mut window_sum = acc as Sample;
        if window_sum <= 0.0 {
            window_sum = 1.0;
        }
        let inv_window_sum = 1.0 / window_sum;
        let two_inv_window_sum = 2.0 / window_sum;

        let tw_re: Vec<Sample> = (0..half)
            .map(|j| ops::cos(-TAU * j as Sample / size as Sample))
            .collect();
        let tw_im: Vec<Sample> = (0..half)
            .map(|j| ops::sin(-TAU * j as Sample / size as Sample))
            .collect();

        let bits = size.trailing_zeros();
        let rev: Vec<usize> = (0..size).map(|i| reverse_low_bits(i, bits)).collect();

        Self {
            size,
            hop,
            window,
            win,
            tw_re,
            tw_im,
            rev,
            ring: vec![0.0; size],
            re: vec![0.0; size],
            im: vec![0.0; size],
            mag: vec![0.0; half + 1],
            write: 0,
            since_last: 0,
            inv_window_sum,
            two_inv_window_sum,
            frames_computed: 0,
        }
    }

    /// Transform length in samples (a power of two).
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Hop size in samples between successive magnitude frames.
    #[must_use]
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// The analysis window taper in use.
    #[must_use]
    pub fn window(&self) -> Window {
        self.window
    }

    /// The most recent single-sided magnitude spectrum.
    ///
    /// The slice has `size / 2 + 1` bins: bin 0 is direct current and the last
    /// bin is the Nyquist frequency. Before the first frame is computed every
    /// bin is zero.
    #[must_use]
    pub fn magnitudes(&self) -> &[Sample] {
        &self.mag
    }

    /// Centre frequency in hertz of magnitude bin `bin` at `sample_rate`.
    #[must_use]
    pub fn bin_frequency(&self, bin: usize, sample_rate: u32) -> Sample {
        bin as Sample * sample_rate as Sample / self.size as Sample
    }

    /// Number of magnitude frames computed since the last [`Self::reset`].
    #[must_use]
    pub fn frames_computed(&self) -> u64 {
        self.frames_computed
    }

    /// Clears the ring, the output spectrum, and the frame counter.
    pub fn reset(&mut self) {
        for v in &mut self.ring {
            *v = 0.0;
        }
        for v in &mut self.mag {
            *v = 0.0;
        }
        self.write = 0;
        self.since_last = 0;
        self.frames_computed = 0;
    }

    /// Pushes one sample into the ring, computing a frame every `hop` samples.
    ///
    /// Non-finite inputs are treated as silence.
    pub fn feed_sample(&mut self, x: Sample) {
        let value = if x.is_finite() { x } else { 0.0 };
        self.ring[self.write] = value;
        self.write += 1;
        if self.write == self.size {
            self.write = 0;
        }
        self.since_last += 1;
        if self.since_last >= self.hop {
            self.since_last = 0;
            self.compute_frame();
        }
    }

    /// Windows the most recent frame, runs the FFT, and writes the magnitudes.
    fn compute_frame(&mut self) {
        let size = self.size;

        // Load the most recent `size` samples oldest-first and apply the window.
        let mut idx = self.write;
        for i in 0..size {
            let s = self.ring[idx];
            self.re[i] = s * self.win[i];
            self.im[i] = 0.0;
            idx += 1;
            if idx == size {
                idx = 0;
            }
        }

        // Bit-reversal permutation into transform order.
        for i in 0..size {
            let j = self.rev[i];
            if j > i {
                self.re.swap(i, j);
                self.im.swap(i, j);
            }
        }

        // Iterative decimation-in-time radix-2 butterflies.
        let mut len = 2;
        while len <= size {
            let half = len / 2;
            let step = size / len;
            let mut base = 0;
            while base < size {
                for k in 0..half {
                    let tw = k * step;
                    let wr = self.tw_re[tw];
                    let wi = self.tw_im[tw];
                    let a = base + k;
                    let b = base + k + half;
                    let tr = wr * self.re[b] - wi * self.im[b];
                    let ti = wr * self.im[b] + wi * self.re[b];
                    self.re[b] = self.re[a] - tr;
                    self.im[b] = self.im[a] - ti;
                    self.re[a] += tr;
                    self.im[a] += ti;
                }
                base += len;
            }
            len <<= 1;
        }

        // Single-sided magnitude with coherent-gain correction.
        let half = size / 2;
        self.mag[0] =
            ops::sqrt(self.re[0] * self.re[0] + self.im[0] * self.im[0]) * self.inv_window_sum;
        for bin in 1..half {
            let power = self.re[bin] * self.re[bin] + self.im[bin] * self.im[bin];
            self.mag[bin] = ops::sqrt(power) * self.two_inv_window_sum;
        }
        self.mag[half] = ops::sqrt(self.re[half] * self.re[half] + self.im[half] * self.im[half])
            * self.inv_window_sum;

        self.frames_computed += 1;
    }
}

/// Audio graph node that passes its input through unchanged while feeding the
/// first channel into an internal [`SpectrumAnalyzer`].
#[derive(Clone, Debug)]
pub struct SpectrumNode {
    analyzer: SpectrumAnalyzer,
}

impl SpectrumNode {
    /// Builds a pass-through analysis node around a fresh [`SpectrumAnalyzer`].
    #[must_use]
    pub fn new(requested_size: usize, hop: usize, window: Window) -> Self {
        Self {
            analyzer: SpectrumAnalyzer::new(requested_size, hop, window),
        }
    }

    /// Borrows the underlying analyzer (to read the spectrum).
    #[must_use]
    pub fn analyzer(&self) -> &SpectrumAnalyzer {
        &self.analyzer
    }

    /// Mutably borrows the underlying analyzer (to retune or reset).
    pub fn analyzer_mut(&mut self) -> &mut SpectrumAnalyzer {
        &mut self.analyzer
    }
}

impl AudioNode for SpectrumNode {
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

        // Feed the first channel into the analyzer.
        if input.channels() >= 1 {
            let mono = input.channel(0);
            for &x in &mono[..frames] {
                self.analyzer.feed_sample(x);
            }
        }
    }

    fn reset(&mut self) {
        self.analyzer.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    #[test]
    fn requested_size_rounds_up_to_power_of_two() {
        assert_eq!(SpectrumAnalyzer::new(1000, 500, Window::Hann).size(), 1024);
        assert_eq!(SpectrumAnalyzer::new(1024, 512, Window::Hann).size(), 1024);
        assert_eq!(SpectrumAnalyzer::new(1025, 512, Window::Hann).size(), 2048);
    }

    #[test]
    fn min_fft_size_enforced() {
        assert_eq!(SpectrumAnalyzer::new(0, 1, Window::Hann).size(), MIN_FFT_SIZE);
        assert_eq!(SpectrumAnalyzer::new(1, 1, Window::Hann).size(), MIN_FFT_SIZE);
    }

    #[test]
    fn hop_is_at_least_one() {
        assert_eq!(SpectrumAnalyzer::new(16, 0, Window::Hann).hop(), 1);
    }

    #[test]
    fn magnitudes_length_is_half_size_plus_one() {
        let analyzer = SpectrumAnalyzer::new(64, 32, Window::Hann);
        assert_eq!(analyzer.magnitudes().len(), 64 / 2 + 1);
    }

    #[test]
    fn dc_input_concentrates_in_bin_zero() {
        let mut analyzer = SpectrumAnalyzer::new(16, 16, Window::Rectangular);
        for _ in 0..16 {
            analyzer.feed_sample(1.0);
        }
        let mag = analyzer.magnitudes();
        assert!((mag[0] - 1.0).abs() < 1e-4);
        for &m in &mag[1..] {
            assert!(m < 1e-4);
        }
    }

    #[test]
    fn sine_at_exact_bin_reads_back_amplitude() {
        let size = 16;
        let mut analyzer = SpectrumAnalyzer::new(size, size, Window::Rectangular);
        for n in 0..size {
            let phase = TAU * 2.0 * n as Sample / size as Sample;
            analyzer.feed_sample(ops::cos(phase));
        }
        let mag = analyzer.magnitudes();
        assert!((mag[2] - 1.0).abs() < 0.05);
        // Neighbouring bins see almost no leakage on an exact bin.
        assert!(mag[1] < 0.05);
        assert!(mag[3] < 0.05);
    }

    #[test]
    fn nyquist_bin_reads_back_amplitude() {
        let size = 16;
        let mut analyzer = SpectrumAnalyzer::new(size, size, Window::Rectangular);
        for n in 0..size {
            let s = if n % 2 == 0 { 1.0 } else { -1.0 };
            analyzer.feed_sample(s);
        }
        let mag = analyzer.magnitudes();
        assert!((mag[size / 2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn bin_frequency_formula() {
        let analyzer = SpectrumAnalyzer::new(1024, 512, Window::Hann);
        // Bin k maps to k * sr / size.
        assert!((analyzer.bin_frequency(0, 48_000) - 0.0).abs() < 1e-6);
        assert!((analyzer.bin_frequency(512, 48_000) - 24_000.0).abs() < 1e-3);
        assert!((analyzer.bin_frequency(1, 48_000) - (48_000.0 / 1024.0)).abs() < 1e-3);
    }

    #[test]
    fn window_coherent_gain_values() {
        assert!((Window::Rectangular.coherent_gain() - 1.0).abs() < 1e-6);
        assert!((Window::Hann.coherent_gain() - 0.5).abs() < 1e-6);
        assert!((Window::Hamming.coherent_gain() - 0.54).abs() < 1e-6);
        assert!((Window::Blackman.coherent_gain() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn hann_window_endpoints_are_zero() {
        let size = 32;
        assert!(Window::Hann.coefficient(0, size).abs() < 1e-6);
        assert!(Window::Hann.coefficient(size - 1, size).abs() < 1e-6);
        // The centre of a Hann window peaks near unity.
        assert!(Window::Hann.coefficient(size / 2, size) > 0.9);
    }

    #[test]
    fn frames_computed_increments_with_hop() {
        let mut analyzer = SpectrumAnalyzer::new(16, 8, Window::Hann);
        assert_eq!(analyzer.frames_computed(), 0);
        for _ in 0..32 {
            analyzer.feed_sample(0.5);
        }
        // 32 samples at a hop of 8 yields four frames.
        assert_eq!(analyzer.frames_computed(), 4);
    }

    #[test]
    fn node_passes_audio_through_unchanged() {
        let mut node = SpectrumNode::new(16, 16, Window::Hann);
        let frames = 16;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        for ch in 0..input.channels() {
            let data = input.channel_mut(ch);
            for (n, v) in data.iter_mut().enumerate() {
                *v = 0.1 * n as Sample - 0.3 * ch as Sample;
            }
        }
        let expected: Vec<Vec<Sample>> = (0..input.channels())
            .map(|ch| input.channel(ch).to_vec())
            .collect();

        let out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        let inputs = [input];
        let mut outputs = [out];
        let c = ctx(frames);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        let [o] = outputs;

        for ch in 0..o.channels() {
            assert_eq!(o.channel(ch), expected[ch].as_slice());
        }
        assert_eq!(node.analyzer().frames_computed(), 1);
    }

    #[test]
    fn reset_clears_state() {
        let mut analyzer = SpectrumAnalyzer::new(16, 8, Window::Hann);
        for _ in 0..32 {
            analyzer.feed_sample(0.7);
        }
        assert!(analyzer.frames_computed() > 0);
        analyzer.reset();
        assert_eq!(analyzer.frames_computed(), 0);
        for &m in analyzer.magnitudes() {
            assert_eq!(m, 0.0);
        }
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut analyzer = SpectrumAnalyzer::new(16, 16, Window::Rectangular);
        for n in 0..16 {
            let x = if n == 3 { Sample::NAN } else { 1.0 };
            analyzer.feed_sample(x);
        }
        for &m in analyzer.magnitudes() {
            assert!(m.is_finite());
        }
    }

    #[test]
    fn multichannel_feeds_channel_zero() {
        let mut node = SpectrumNode::new(16, 16, Window::Rectangular);
        let frames = 16;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        {
            let left = input.channel_mut(0);
            for v in left.iter_mut() {
                *v = 1.0;
            }
        }
        {
            let right = input.channel_mut(1);
            for v in right.iter_mut() {
                *v = -0.5;
            }
        }
        let out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        let inputs = [input];
        let mut outputs = [out];
        let c = ctx(frames);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        // Channel 0 was constant 1.0, so bin 0 recovers 1.0.
        let mag = node.analyzer().magnitudes();
        assert!((mag[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = SpectrumNode::new(16, 16, Window::Hann);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 4);
        input.set_active_frames(0);
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 4);
        out.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [out];
        let c = ctx(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&c, &mut io);
        let [o] = outputs;
        for ch in 0..o.channels() {
            for &v in o.channel(ch) {
                assert_eq!(v, 0.0);
            }
        }
        assert_eq!(node.analyzer().frames_computed(), 0);
    }
}
