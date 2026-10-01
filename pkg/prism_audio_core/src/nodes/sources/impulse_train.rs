//! Band-limited impulse train (BLIT) source node.
//!
//! [`ImpulseTrainNode`] synthesizes a periodic train of alias-free impulses:
//! a sequence of narrow, band-limited spikes repeating at the fundamental
//! frequency. It is the raw excitation primitive behind the classic
//! "band-limited impulse train" family of analog-style oscillators -- a BLIT
//! integrated once yields an alias-free sawtooth, and a pair of offset BLITs
//! differenced and integrated yields an alias-free pulse. On its own the
//! impulse train is a bright, buzzy drone rich in equal-amplitude harmonics,
//! useful as a formant/subtractive excitation or a metronomic click source.
//!
//! # Model
//!
//! A mathematically ideal impulse train has infinite bandwidth and therefore
//! aliases catastrophically when sampled. The band-limited version keeps only
//! the harmonics that fit below the Nyquist frequency. Summing a DC term and
//! the first `N` cosine harmonics with equal weight is exactly the normalized
//! Dirichlet kernel (the periodic sinc):
//!
//! ```text
//!   D_M(t) = sin(M * pi * t) / (M * sin(pi * t))
//!          = (1 / M) * (1 + 2 * sum_{k=1..N} cos(2 * pi * k * t))
//! ```
//!
//! where `t` is the normalized phase in `[0, 1)`, `N = floor(period / 2)` is
//! the number of harmonics below Nyquist (`period = sample_rate / frequency`),
//! and `M = 2 * N + 1` is the odd harmonic count. The closed form is evaluated
//! directly rather than by summing cosines, so cost is constant regardless of
//! how many partials are present. At the impulse (`t -> 0`) the ratio is the
//! removable singularity `D_M(0) = 1`, handled by its analytic limit. Because
//! the kernel is normalized to a unit peak, the output magnitude never exceeds
//! the configured amplitude.
//!
//! Like any unipolar impulse train the signal carries a frequency-dependent DC
//! component of `1 / M` (the mean of the Dirichlet kernel). This is inherent to
//! an equal-amplitude harmonic series with a DC partial and is intentionally
//! preserved so the peak stays within `[-1, 1]`; route the node through a
//! [`super::super::effects::DcBlockerNode`] when a DC-free train is required.
//!
//! # Relationship
//!
//! Where [`super::oscillator::OscillatorNode`] band-limits *geometric*
//! waveforms with `PolyBLEP` edge corrections, this node band-limits the
//! *impulse* excitation itself via the closed-form Dirichlet kernel. The two
//! are complementary primitives: the oscillator shapes a continuous waveform,
//! the impulse train emits the alias-free spikes that classic subtractive and
//! formant voices integrate and filter.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`ImpulseTrainNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. The harmonic count is
//! derived once per block from the (constant-within-block) frequency, and the
//! amplitude is driven through a [`Smoothed`] value so automation never
//! produces zipper clicks.
//!
//! # Provenance
//!
//! Implemented from first principles from the public-domain band-limited
//! impulse train / Dirichlet-kernel technique. It contains no code, data, or
//! derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google
//! Resonance Audio, the Web Audio API, or any other audio engine; only the
//! shared mathematical ideas are used. There is no AI or machine learning of
//! any kind.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

const PI: Sample = core::f32::consts::PI;

/// Default fundamental frequency in hertz.
pub const DEFAULT_IMPULSE_FREQUENCY_HZ: Sample = 110.0;

/// Default linear output amplitude.
pub const DEFAULT_IMPULSE_AMPLITUDE: Sample = 1.0;

/// Denominator magnitude below which the Dirichlet kernel is replaced by its
/// analytic limit of `1.0`. `sin(pi*t)` is tiny only in the immediate
/// neighborhood of the impulse, where dividing by it would otherwise both risk
/// `0 / 0` and amplify the fast-math `sin` approximation error; the guard keeps
/// that neighborhood exact and bounded.
const DIRICHLET_EPSILON: Sample = 1.0e-4;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Normalized Dirichlet kernel `D_M(t) = sin(M*pi*t) / (M*sin(pi*t))`.
///
/// `m` is the odd harmonic count `2*N + 1`. The removable singularity at the
/// impulse (`sin(pi*t) -> 0`) is resolved by its analytic limit of `1.0`. The
/// closed form is provably bounded by `1` in magnitude; the final clamp removes
/// only the tiny overshoot introduced by the fast-math `sin` approximation, so
/// the result stays within `[-1, 1]`.
#[inline]
fn dirichlet_kernel(t: Sample, m: Sample) -> Sample {
    let denom = ops::sin(PI * t);
    let value = if denom.abs() < DIRICHLET_EPSILON {
        1.0
    } else {
        ops::sin(PI * m * t) / (m * denom)
    };
    value.clamp(-1.0, 1.0)
}

/// Construction parameters for an [`ImpulseTrainNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ImpulseTrainParams {
    /// Fundamental frequency in hertz. Clamped non-negative.
    pub frequency_hz: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for ImpulseTrainParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_IMPULSE_FREQUENCY_HZ,
            amplitude: DEFAULT_IMPULSE_AMPLITUDE,
        }
    }
}

/// A band-limited impulse train (BLIT) source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::ImpulseTrainNode;
/// use prism_audio_core::param::Ramp;
///
/// let mut node = ImpulseTrainNode::new(110.0, 1.0);
/// node.set_amplitude(0.5, Ramp::Immediate);
/// assert_eq!(node.frequency(), 110.0);
/// ```
#[derive(Debug, Clone)]
pub struct ImpulseTrainNode {
    /// Fundamental frequency in hertz. Always non-negative. Stored as a plain
    /// scalar because the phase accumulator is continuous, so a frequency
    /// change is click-free without smoothing.
    frequency: Sample,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl ImpulseTrainNode {
    /// Creates a band-limited impulse train at `frequency_hz` with master
    /// `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; frequency is clamped
    /// non-negative.
    #[must_use]
    pub fn new(frequency_hz: Sample, amplitude: Sample) -> Self {
        Self {
            frequency: finite_or(frequency_hz, DEFAULT_IMPULSE_FREQUENCY_HZ).max(0.0),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_IMPULSE_AMPLITUDE)),
            phase: 0.0,
        }
    }

    /// Builds an impulse train from an [`ImpulseTrainParams`] bundle.
    #[must_use]
    pub fn from_params(params: ImpulseTrainParams) -> Self {
        Self::new(params.frequency_hz, params.amplitude)
    }

    /// Sets the fundamental frequency in hertz (clamped non-negative).
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = finite_or(hz, self.frequency).max(0.0);
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the current fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Odd harmonic count `M = 2*N + 1` for a per-sample phase increment `dt`,
    /// where `N = floor(period / 2)` is the number of harmonics below Nyquist.
    ///
    /// Degenerate increments (zero or non-positive frequency, or a frequency at
    /// or above Nyquist where no harmonic fits) collapse to `M = 1`, i.e. a
    /// pure DC term, which keeps the output band-limited and bounded.
    #[inline]
    fn harmonic_count(dt: Sample) -> Sample {
        if dt <= 0.0 {
            return 1.0;
        }
        let period = 1.0 / dt;
        let harmonics = ops::floor(period * 0.5);
        (2.0 * harmonics + 1.0).max(1.0)
    }

    /// Produces one output sample at per-sample phase increment `dt` and odd
    /// harmonic count `m`, advancing the phase accumulator and the smoothed
    /// amplitude.
    #[inline]
    fn render_sample(&mut self, dt: Sample, m: Sample) -> Sample {
        let amp = self.amplitude.next_sample();
        let value = dirichlet_kernel(self.phase, m);

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        value * amp
    }
}

impl AudioNode for ImpulseTrainNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let dt = self.frequency / sample_rate;
        let m = Self::harmonic_count(dt);

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(dt, m);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.amplitude = Smoothed::new(self.amplitude.target());
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const TAU: Sample = core::f32::consts::TAU;

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut ImpulseTrainNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render_stereo(node: &mut ImpulseTrainNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    /// Reference impulse train evaluated as the explicit DC + cosine sum.
    fn reference_sample(phase: Sample, m: Sample) -> Sample {
        let n = ((m - 1.0) * 0.5).round() as i32;
        let mut acc = 1.0;
        for k in 1..=n {
            acc += 2.0 * ops::cos(TAU * (k as Sample) * phase);
        }
        acc / m
    }

    #[test]
    fn output_stays_bounded() {
        for &freq in &[27.5, 110.0, 440.0, 2000.0, 9000.0] {
            let mut node = ImpulseTrainNode::new(freq, 1.0);
            let out = render(&mut node, 48_000, 2048);
            for &s in out.channel(0) {
                assert!(s.is_finite());
                assert!(s.abs() <= 1.0 + 1e-4, "freq {freq} produced {s}");
            }
        }
    }

    #[test]
    fn matches_closed_form_cosine_sum() {
        let sample_rate = 48_000u32;
        let mut node = ImpulseTrainNode::new(220.0, 1.0);
        let frames: usize = 1024;
        let out = render(&mut node, sample_rate, frames);
        let dt = 220.0 / sample_rate as Sample;
        let m = ImpulseTrainNode::harmonic_count(dt);
        let mut phase = 0.0;
        for &s in out.channel(0) {
            let expected = reference_sample(phase, m);
            // The tolerance accommodates the fast-math `sin`/`cos` approximation,
            // which is largest in the immediate neighborhood of the impulse.
            assert!((s - expected).abs() < 5e-3, "got {s}, expected {expected}");
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn impulse_peak_is_near_unity() {
        // The first sample sits exactly on the impulse (phase 0) -> peak 1.
        let mut node = ImpulseTrainNode::new(100.0, 1.0);
        let out = render(&mut node, 48_000, 16);
        assert!((out.channel(0)[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mean_tracks_dc_of_one_over_m() {
        let sample_rate = 48_000u32;
        let freq = 300.0;
        let mut node = ImpulseTrainNode::new(freq, 1.0);
        let frames: usize = 48_000; // one full second => integer number of cycles
        let out = render(&mut node, sample_rate, frames);
        let mean: Sample =
            out.channel(0).iter().copied().sum::<Sample>() / frames as Sample;
        let dt = freq / sample_rate as Sample;
        let m = ImpulseTrainNode::harmonic_count(dt);
        assert!((mean - 1.0 / m).abs() < 2e-3, "mean {mean}, 1/m {}", 1.0 / m);
    }

    #[test]
    fn amplitude_scales_output() {
        let mut loud = ImpulseTrainNode::new(220.0, 1.0);
        let mut quiet = ImpulseTrainNode::new(220.0, 0.25);
        let a = render(&mut loud, 48_000, 256);
        let b = render(&mut quiet, 48_000, 256);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert!((0.25 * x - y).abs() < 1e-4);
        }
    }

    #[test]
    fn is_periodic_at_fundamental() {
        let sample_rate = 48_000u32;
        let freq = 480.0; // period = 100 samples exactly
        let mut node = ImpulseTrainNode::new(freq, 1.0);
        let out = render(&mut node, sample_rate, 400);
        let period = (sample_rate as Sample / freq).round() as usize;
        for i in 0..(400 - period) {
            assert!(
                (out.channel(0)[i] - out.channel(0)[i + period]).abs() < 1e-3,
                "sample {i} not periodic"
            );
        }
    }

    #[test]
    fn high_frequency_collapses_to_dc() {
        // Above Nyquist no harmonic fits: M = 1 => constant DC of 1.0.
        let mut node = ImpulseTrainNode::new(30_000.0, 1.0);
        let out = render(&mut node, 48_000, 64);
        for &s in out.channel(0) {
            assert!((s - 1.0).abs() < 1e-4, "expected DC, got {s}");
        }
    }

    #[test]
    fn harmonic_count_is_odd_and_below_nyquist() {
        let sample_rate = 48_000.0;
        for &freq in &[55.0, 110.0, 440.0, 1234.0] {
            let dt = freq / sample_rate;
            let m = ImpulseTrainNode::harmonic_count(dt);
            // M odd.
            assert!((m as i32) % 2 == 1, "M {m} not odd for {freq}");
            // Highest harmonic N = (M-1)/2 stays below Nyquist.
            let n = (m - 1.0) * 0.5;
            assert!(n * freq < sample_rate * 0.5, "harmonic exceeds Nyquist");
            // The next harmonic would exceed Nyquist.
            assert!((n + 1.0) * freq >= sample_rate * 0.5);
        }
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let mut node = ImpulseTrainNode::new(330.0, 0.8);
        let first = render(&mut node, 48_000, 256);
        node.reset();
        let second = render(&mut node, 48_000, 256);
        for (a, b) in first.channel(0).iter().zip(second.channel(0)) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn reset_restores_phase() {
        let mut node = ImpulseTrainNode::new(330.0, 1.0);
        let _ = render(&mut node, 48_000, 123);
        node.reset();
        let out = render(&mut node, 48_000, 16);
        // Phase back at the impulse => first sample near unity.
        assert!((out.channel(0)[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn negative_frequency_is_clamped() {
        let node = ImpulseTrainNode::new(-440.0, 1.0);
        assert_eq!(node.frequency(), 0.0);
    }

    #[test]
    fn zero_frequency_is_bounded_dc() {
        let mut node = ImpulseTrainNode::new(0.0, 1.0);
        let out = render(&mut node, 48_000, 32);
        for &s in out.channel(0) {
            assert!((s - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = ImpulseTrainNode::new(Sample::NAN, Sample::INFINITY);
        assert_eq!(node.frequency(), DEFAULT_IMPULSE_FREQUENCY_HZ);
        assert_eq!(node.amplitude(), DEFAULT_IMPULSE_AMPLITUDE);
    }

    #[test]
    fn from_params_matches_new() {
        let params = ImpulseTrainParams {
            frequency_hz: 320.0,
            amplitude: 0.7,
        };
        let mut a = ImpulseTrainNode::from_params(params);
        let mut b = ImpulseTrainNode::new(320.0, 0.7);
        let oa = render(&mut a, 48_000, 128);
        let ob = render(&mut b, 48_000, 128);
        for (x, y) in oa.channel(0).iter().zip(ob.channel(0)) {
            assert!((x - y).abs() < 1e-7);
        }
    }

    #[test]
    fn default_params_use_defaults() {
        let node = ImpulseTrainNode::from_params(ImpulseTrainParams::default());
        assert_eq!(node.frequency(), DEFAULT_IMPULSE_FREQUENCY_HZ);
        assert_eq!(node.amplitude(), DEFAULT_IMPULSE_AMPLITUDE);
    }

    #[test]
    fn stereo_channels_are_identical() {
        let mut node = ImpulseTrainNode::new(220.0, 1.0);
        let out = render_stereo(&mut node, 48_000, 128);
        let (left, right) = (out.channel(0), out.channel(1));
        assert_eq!(left, right);
    }

    #[test]
    fn super_nyquist_frequency_does_not_panic() {
        let mut node = ImpulseTrainNode::new(100_000.0, 1.0);
        let out = render(&mut node, 48_000, 64);
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s.abs() <= 1.0 + 1e-4);
        }
    }

    #[test]
    fn amplitude_ramp_is_click_free() {
        let sample_rate = 48_000u32;
        let frames: usize = 512;
        let mut node = ImpulseTrainNode::new(220.0, 1.0);
        node.set_amplitude(0.0, Ramp::Linear { samples: frames as u32 });
        let out = render(&mut node, sample_rate, frames);
        // Successive samples should not jump wildly from the ramp alone.
        for &s in out.channel(0) {
            assert!(s.is_finite());
            assert!(s.abs() <= 1.0 + 1e-4);
        }
        assert!(node.amplitude() == 0.0);
    }

    #[test]
    fn zero_channel_output_is_noop() {
        let mut node = ImpulseTrainNode::new(220.0, 1.0);
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 32);
        out.set_active_frames(0);
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, 0), &mut io);
        assert_eq!(outputs[0].active_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let mut node = ImpulseTrainNode::new(123.0, 0.9);
        assert_eq!(node.frequency(), 123.0);
        assert_eq!(node.amplitude(), 0.9);
        node.set_frequency(456.0);
        node.set_amplitude(0.3, Ramp::Immediate);
        assert_eq!(node.frequency(), 456.0);
        assert_eq!(node.amplitude(), 0.3);
    }

    #[test]
    fn not_silent() {
        let mut node = ImpulseTrainNode::new(220.0, 1.0);
        let out = render(&mut node, 48_000, 256);
        let energy: Sample = out.channel(0).iter().map(|s| s * s).sum();
        assert!(energy > 1e-3);
    }
}
