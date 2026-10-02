//! Discrete-summation-formula (DSF) oscillator source node.
//!
//! [`DsfOscillatorNode`] synthesizes a harmonic tone whose partial amplitudes
//! decay geometrically, evaluated in closed form by the discrete summation
//! formula (DSF). A single `brightness` control slides the geometric ratio from
//! `0` (a mathematically pure sine) toward a bright, saw-like spectrum, giving
//! a continuous, inherently band-limited "sine to buzz" morph at constant cost
//! regardless of how many partials are present.
//!
//! # Model
//!
//! The target spectrum is a harmonic series with `N + 1` partials whose linear
//! amplitudes form a geometric sequence `1, a, a^2, ...`:
//!
//! ```text
//!   x(t) = sum_{k=0}^{N} a^k * sin((k + 1) * phi),   phi = 2*pi * f0 * t
//! ```
//!
//! so partial `k` sits at `(k + 1) * f0` with amplitude `a^k` and the ratio
//! `a` sets the spectral tilt (brightness). Rather than summing the partials,
//! the geometric sine series is collapsed to its exact closed form (the DSF):
//!
//! ```text
//!   x = [ sin(phi)
//!         - a^(N+1) * sin((N + 2) * phi)
//!         + a^(N+2) * sin((N + 1) * phi) ]
//!       / (1 - 2*a*cos(phi) + a^2)
//! ```
//!
//! which follows from the geometric sine sum with the carrier and modulator
//! phases equal (`theta == beta == phi`), so the `sin(theta - beta)` term
//! vanishes. The denominator `1 - 2*a*cos(phi) + a^2 >= (1 - a)^2` stays
//! strictly positive for `a` in `[0, 1)`, so the division is always safe.
//!
//! `brightness` in `[0, 1]` maps to `a = brightness * MAX_RATIO` with
//! `MAX_RATIO = 0.95`, keeping the ratio bounded away from `1` so both the
//! denominator and the amplitude sum stay well conditioned. At `brightness = 0`
//! the ratio is `0`, every correction term vanishes, and the output collapses
//! to the single partial `sin(phi)`: a pure sine.
//!
//! The number of partials is chosen once per block from the fundamental so the
//! highest partial `(N + 1) * f0` always sits below the guarded Nyquist
//! frequency `sample_rate * NYQUIST_GUARD`; partials that would alias are never
//! generated, so the oscillator is band-limited by construction with no edge
//! correction. The result is normalized by the exact partial-amplitude sum
//! `sum_{k=0}^{N} a^k = (1 - a^(N+1)) / (1 - a)`, which bounds the magnitude by
//! `1` because the closed form equals a sum whose terms are bounded by that
//! same total; a final clamp removes only the tiny overshoot left by the
//! fast-math `sin` approximation.
//!
//! # Relationship
//!
//! The equal-amplitude special case `a = 1` of this series is exactly the
//! Dirichlet kernel emitted by [`super::impulse_train::ImpulseTrainNode`]; this
//! node instead keeps `a` strictly below `1` to expose a continuous brightness
//! tilt, trading the flat buzz of a BLIT for a tunable geometric rolloff. Where
//! [`super::additive_oscillator::AdditiveOscillatorNode`] sums an arbitrary,
//! independently addressable partial set at `O(N)` cost, this node evaluates a
//! strictly geometric series in `O(1)` closed form. And where
//! [`super::oscillator::OscillatorNode`] band-limits fixed geometric waveforms
//! with `PolyBLEP` edge corrections and
//! [`super::phase_distortion_oscillator::PhaseDistortionOscillatorNode`]
//! recolours a cosine by warping time, this node sculpts the spectrum directly
//! through the ratio `a`.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so [`DsfOscillatorNode::process`]
//! performs no allocation, no locking, and no panicking: it is a pure
//! per-sample state machine. The partial count is derived once per block from
//! the (constant-within-block) frequency, while `brightness` and `amplitude`
//! are driven through [`Smoothed`] values so automation never produces zipper
//! clicks. The single continuous phase accumulator makes frequency changes
//! click-free without smoothing. Reproducible across platforms via
//! [`bevy_math::ops`].
//!
//! # Provenance
//!
//! Implemented from first principles from the public discrete-summation-formula
//! technique for synthesizing complex harmonic spectra (closed-form geometric
//! sine series). It contains no code, data, or derivative of Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web
//! Audio API, the Synthesis Toolkit, or any other audio engine or toolkit; only
//! the shared mathematical ideas are used. There is no AI or machine learning
//! of any kind.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

const TAU: Sample = core::f32::consts::TAU;

/// Minimum fundamental frequency in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Default fundamental frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum fundamental frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 12_000.0;

/// Default brightness (geometric-ratio control) in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Maximum geometric ratio `a`, bounded away from `1` so the DSF denominator
/// and amplitude sum stay well conditioned.
pub const MAX_RATIO: Sample = 0.95;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fraction of the sample rate treated as the usable Nyquist ceiling.
pub const NYQUIST_GUARD: Sample = 0.49;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Non-negative fractional part `x - floor(x)` (a `no_std` stand-in for the
/// `f32::fract` method, which is unavailable without the standard library).
#[inline]
fn fract_nonneg(x: Sample) -> Sample {
    x - ops::floor(x)
}

/// Construction parameters for a [`DsfOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DsfOscillatorParams {
    /// Fundamental frequency in hertz. Clamped to `[MIN, MAX]`.
    pub frequency_hz: Sample,
    /// Brightness (geometric-ratio control) in `[0, 1]`.
    pub brightness: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for DsfOscillatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            brightness: DEFAULT_BRIGHTNESS,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl DsfOscillatorParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            brightness: finite_or(self.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A discrete-summation-formula oscillator source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::DsfOscillatorNode;
///
/// let mut node = DsfOscillatorNode::new(110.0, 0.0, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.brightness(), 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct DsfOscillatorNode {
    /// Fundamental frequency in hertz, clamped to `[MIN, MAX]`. Stored as a
    /// plain scalar because the phase accumulator is continuous, so a frequency
    /// change is click-free without smoothing.
    frequency_hz: Sample,
    /// Smoothed brightness (geometric-ratio control) in `[0, 1]`.
    brightness: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl DsfOscillatorNode {
    /// Creates a DSF oscillator at `frequency_hz` with `brightness` and linear
    /// `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; frequency is clamped to
    /// `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]` and brightness to `[0, 1]`.
    #[must_use]
    pub fn new(frequency_hz: Sample, brightness: Sample, amplitude: Sample) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            brightness: Smoothed::new(finite_or(brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase: 0.0,
        }
    }

    /// Builds a DSF oscillator from a [`DsfOscillatorParams`] bundle.
    #[must_use]
    pub fn from_params(params: DsfOscillatorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.brightness, p.amplitude)
    }

    /// Sets the fundamental frequency in hertz (clamped to `[MIN, MAX]`).
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample) {
        self.frequency_hz =
            finite_or(hz, self.frequency_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    }

    /// Sets a new target brightness in `[0, 1]`, gliding with `ramp`.
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample, ramp: Ramp) {
        self.brightness.set_target(
            finite_or(brightness, self.brightness.target()).clamp(0.0, 1.0),
            ramp,
        );
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
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the target brightness the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Highest partial index `N` (so `N + 1` partials are summed) such that the
    /// top partial `(N + 1) * f0` stays below the guarded Nyquist frequency.
    ///
    /// `f0` is assumed already clamped to the guard ceiling, so the ratio is at
    /// least `1` and `N >= 0`.
    #[inline]
    fn top_partial_index(f0: Sample, guard_hz: Sample) -> Sample {
        let partials = ops::floor(guard_hz / f0);
        (partials - 1.0).max(0.0)
    }

    /// Produces one output sample at normalized phase increment `dt` and top
    /// partial index `big_n`, advancing the phase accumulator and the smoothed
    /// controls.
    #[inline]
    fn render_sample(&mut self, dt: Sample, big_n: Sample) -> Sample {
        let brightness = self.brightness.next_sample();
        let amp = self.amplitude.next_sample();
        let a = (brightness * MAX_RATIO).clamp(0.0, MAX_RATIO);

        let t = self.phase;
        let phi = TAU * t;
        let sin_phi = ops::sin(phi);
        let cos_phi = ops::cos(phi);

        // Geometric-ratio powers at the truncation boundary.
        let a_np1 = ops::powf(a, big_n + 1.0);
        let a_np2 = a_np1 * a;

        // High-order partial phases reduced to `[0, 1)` before scaling by TAU so
        // the `sin` argument never grows large and lossy.
        let hi1 = fract_nonneg((big_n + 2.0) * t);
        let hi2 = fract_nonneg((big_n + 1.0) * t);

        let num = sin_phi - a_np1 * ops::sin(TAU * hi1) + a_np2 * ops::sin(TAU * hi2);
        // 1 - 2*a*cos(phi) + a^2 >= (1 - a)^2 > 0 for a in [0, MAX_RATIO].
        let denom = 1.0 - 2.0 * a * cos_phi + a * a;
        let series = num / denom;

        // Sum_{k=0}^{N} a^k = (1 - a^(N+1)) / (1 - a); >= 1, so normalization
        // bounds |value| by 1. The clamp removes only fast-math overshoot.
        let sum_amp = (1.0 - a_np1) / (1.0 - a);
        let value = (series / sum_amp).clamp(-1.0, 1.0);

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        value * amp
    }
}

impl AudioNode for DsfOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let guard_hz = sr * NYQUIST_GUARD;
        let f0 = self.frequency_hz.min(guard_hz);
        let dt = f0 / sr;
        let big_n = Self::top_partial_index(f0, guard_hz);

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(dt, big_n);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.brightness = Smoothed::new(self.brightness.target());
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

    const SR: u32 = 48_000;

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut DsfOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut DsfOscillatorNode,
        sample_rate: u32,
        frames: usize,
        layout: ChannelLayout,
    ) -> AudioBuffer {
        let mut out = AudioBuffer::new(layout, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().map(|s| s * s).sum()
    }

    /// Energy of a single Goertzel bin at `freq` over the channel.
    fn goertzel(buf: &AudioBuffer, sample_rate: u32, freq: Sample) -> Sample {
        let samples = buf.channel(0);
        let n = samples.len();
        if n == 0 {
            return 0.0;
        }
        let omega = TAU * freq / sample_rate as Sample;
        let coeff = 2.0 * ops::cos(omega);
        let mut s_prev = 0.0;
        let mut s_prev2 = 0.0;
        for &x in samples {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    }

    /// Sum of harmonic energy above the fundamental, used as a brightness proxy.
    fn upper_harmonic_energy(brightness: Sample) -> Sample {
        let mut node = DsfOscillatorNode::new(110.0, brightness, 0.8);
        let out = render(&mut node, SR, 8_192);
        let mut hi = 0.0;
        for h in 2..=40 {
            let f = 110.0 * h as Sample;
            if f >= SR as Sample * NYQUIST_GUARD {
                break;
            }
            hi += goertzel(&out, SR, f);
        }
        hi
    }

    #[test]
    fn renders_bounded_finite() {
        for &b in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut node = DsfOscillatorNode::new(110.0, b, 1.0);
            let out = render(&mut node, SR, 8_192);
            for &s in out.channel(0) {
                assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "b={b} s={s}");
            }
        }
    }

    #[test]
    fn brightness_zero_is_pure_sine() {
        // brightness 0 -> ratio 0 -> every correction term vanishes, so the
        // output is the single partial sin(2*pi*phase). Mirror the node's own
        // phase accumulator so the comparison is drift-free.
        let freq = 100.0;
        let mut node = DsfOscillatorNode::new(freq, 0.0, 1.0);
        let out = render(&mut node, SR, 2_048);
        let dt = freq / SR as Sample;
        let mut phase = 0.0_f32;
        for (n, &s) in out.channel(0).iter().enumerate() {
            let reference = ops::sin(TAU * phase);
            assert!((s - reference).abs() < 1e-6, "n={n} s={s} ref={reference}");
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn pure_sine_has_negligible_upper_harmonics() {
        let mut node = DsfOscillatorNode::new(220.0, 0.0, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 220.0);
        let h2 = goertzel(&out, SR, 440.0);
        assert!(h2 < fund * 1e-3, "fund={fund} h2={h2}");
    }

    #[test]
    fn brightness_increases_upper_harmonics() {
        let dull = upper_harmonic_energy(0.1);
        let bright = upper_harmonic_energy(0.9);
        assert!(bright > dull * 2.0, "dull={dull} bright={bright}");
    }

    #[test]
    fn second_harmonic_present_when_bright() {
        let mut node = DsfOscillatorNode::new(220.0, 0.8, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 220.0);
        let h2 = goertzel(&out, SR, 440.0);
        // The geometric series populates the harmonic above the fundamental.
        assert!(h2 > fund * 1e-2, "fund={fund} h2={h2}");
    }

    #[test]
    fn fundamental_locks_to_frequency() {
        let mut node = DsfOscillatorNode::new(220.0, 0.7, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 220.0);
        let off = goertzel(&out, SR, 330.0);
        assert!(fund > off, "fund={fund} off={off}");
    }

    #[test]
    fn spectrum_is_band_limited() {
        // At a high fundamental the partial count collapses but the output must
        // stay finite and bounded with no energy at an inter-harmonic bin above
        // the top partial.
        let mut node = DsfOscillatorNode::new(6_000.0, 1.0, 0.8);
        let out = render(&mut node, SR, 8_192);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 0.8 + 1e-3, "s={s}");
        }
        let fund = goertzel(&out, SR, 6_000.0);
        // 6000 Hz: guard = 23520, partials = floor(23520/6000) = 3, top = 18000.
        // A bin just below guard but not a harmonic must be essentially empty.
        let inter = goertzel(&out, SR, 23_000.0);
        assert!(inter < fund * 1e-2, "fund={fund} inter={inter}");
    }

    #[test]
    fn brightness_sweep_is_click_free() {
        let frames: usize = 4_096;
        let mut node = DsfOscillatorNode::new(55.0, 0.0, 0.8);
        node.set_brightness(
            0.7,
            Ramp::Linear {
                samples: frames as u32,
            },
        );
        let out = render(&mut node, SR, frames);
        for w in out.channel(0).windows(2) {
            assert!((w[1] - w[0]).abs() < 0.1, "step {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn deterministic() {
        let mut a = DsfOscillatorNode::new(130.0, 0.7, 0.8);
        let mut b = DsfOscillatorNode::new(130.0, 0.7, 0.8);
        let ra = render(&mut a, SR, 2_048);
        let rb = render(&mut b, SR, 2_048);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_replays_output() {
        let mut node = DsfOscillatorNode::new(130.0, 0.7, 0.8);
        let a = render(&mut node, SR, 1_024);
        node.reset();
        let b = render(&mut node, SR, 1_024);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_restores_phase() {
        let mut node = DsfOscillatorNode::new(130.0, 0.7, 0.8);
        let _ = render(&mut node, SR, 777);
        node.reset();
        let fresh = DsfOscillatorNode::new(130.0, 0.7, 0.8);
        assert_eq!(node.phase, fresh.phase);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = DsfOscillatorNode::new(110.0, 0.6, 0.0);
        let out = render(&mut node, SR, 1_024);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = DsfOscillatorNode::new(110.0, 0.6, 0.25);
        let mut loud = DsfOscillatorNode::new(110.0, 0.6, 0.5);
        let eq = energy(&render(&mut quiet, SR, 4_096));
        let el = energy(&render(&mut loud, SR, 4_096));
        let ratio = el / eq;
        assert!((ratio - 4.0).abs() < 0.1, "ratio={ratio}");
    }

    #[test]
    fn not_silent() {
        let mut node = DsfOscillatorNode::new(110.0, 0.6, 0.8);
        let out = render(&mut node, SR, 1_024);
        assert!(energy(&out) > 0.0);
    }

    #[test]
    fn mono_core_copied_to_stereo_and_quad() {
        for layout in [ChannelLayout::Stereo, ChannelLayout::Quad] {
            let mut node = DsfOscillatorNode::new(110.0, 0.6, 0.8);
            let out = render_layout(&mut node, SR, 512, layout);
            let ch0 = out.channel(0).to_vec();
            for ch in 1..out.channels() {
                assert_eq!(out.channel(ch), ch0.as_slice());
            }
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = DsfOscillatorNode::new(110.0, 0.6, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn latency_is_zero() {
        let node = DsfOscillatorNode::new(110.0, 0.6, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = DsfOscillatorNode::new(123.0, 0.4, 0.7);
        assert_eq!(node.frequency_hz(), 123.0);
        assert_eq!(node.brightness(), 0.4);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = DsfOscillatorParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!((0.0..=1.0).contains(&p.brightness));
        assert!(p.amplitude.is_finite());
    }

    #[test]
    fn from_params_matches_new() {
        let params = DsfOscillatorParams {
            frequency_hz: 123.0,
            brightness: 0.45,
            amplitude: 0.7,
        };
        let mut a = DsfOscillatorNode::from_params(params);
        let mut b = DsfOscillatorNode::new(123.0, 0.45, 0.7);
        let ra = render(&mut a, SR, 1_024);
        let rb = render(&mut b, SR, 1_024);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let low = DsfOscillatorNode::new(1.0, -1.0, 0.8);
        let high = DsfOscillatorNode::new(99_999.0, 9.0, 0.8);
        assert_eq!(low.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(low.brightness(), 0.0);
        assert_eq!(high.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(high.brightness(), 1.0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = DsfOscillatorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.brightness(), DEFAULT_BRIGHTNESS);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = DsfOscillatorNode::new(110.0, 0.5, 0.8);
        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 110.0);
        node.set_frequency_hz(99_999.0);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        node.set_brightness(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.brightness(), 0.5);
        node.set_brightness(5.0, Ramp::Immediate);
        assert_eq!(node.brightness(), 1.0);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn nyquist_guard_keeps_super_nyquist_bounded() {
        let mut node = DsfOscillatorNode::new(MAX_FREQUENCY_HZ, 1.0, 1.0);
        let out = render(&mut node, SR, 4_096);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn frequency_change_shifts_fundamental() {
        let mut low = DsfOscillatorNode::new(110.0, 0.7, 0.8);
        let mut high = DsfOscillatorNode::new(220.0, 0.7, 0.8);
        let el = render(&mut low, SR, 8_192);
        let eh = render(&mut high, SR, 8_192);
        assert!(goertzel(&el, SR, 110.0) > goertzel(&eh, SR, 110.0));
        assert!(goertzel(&eh, SR, 220.0) > goertzel(&el, SR, 220.0));
    }
}
