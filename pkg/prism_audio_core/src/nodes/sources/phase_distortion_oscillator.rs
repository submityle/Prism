//! Phase-distortion oscillator source node (non-resonant Casio-CZ style morph).
//!
//! [`PhaseDistortionOscillatorNode`] reproduces the classic "phase distortion"
//! synthesis timbre: a cosine carrier read through a time-warped phase so that
//! one half of the cycle is compressed and the other stretched. As the warp
//! deepens, the compressed half develops a steep edge and the spectrum fills in
//! with harmonics, morphing continuously from a pure sine toward a bright,
//! sawtooth-like tone. Sweeping the single `amount` control is the signature
//! phase-distortion gesture: a smooth, vocal brightening with no change in
//! perceived pitch.
//!
//! # Model
//!
//! The normalized phase `p` in `[0, 1)` is passed through a two-segment warp
//! around a break point `d` in `(0, 0.5]`:
//!
//! ```text
//! w(p) = p / (2 d)                 for p <  d
//! w(p) = 0.5 + (p - d)/(2 (1-d))   for p >= d
//! ```
//!
//! so `w` sweeps `0 -> 0.5` while `p` crosses `[0, d)` and `0.5 -> 1` while `p`
//! crosses `[d, 1)`. The output is the carrier `-cos(2*pi*w)`. The user-facing
//! `amount` in `[0, 1]` maps to `d = 0.5 - 0.49*amount`: at `amount = 0` the
//! break point is exactly `0.5`, `w == p`, and the output is a mathematically
//! pure sinusoid; as `amount -> 1` the break point shrinks toward `0.01`, the
//! first half-cycle is sharply compressed, and the harmonics rise.
//!
//! Because the warp break point `d` and the phase wrap both land where the
//! carrier's sine term is zero (`sin(pi) = sin(0) = 0`), the output is
//! continuous in both value and first derivative; only its curvature (second
//! derivative) is discontinuous. A curvature break aliases far more gently than
//! the step or slope breaks of a naive saw or triangle -- its images roll off at
//! roughly 18 dB/octave -- so the naive form is clean across the musical range.
//! The fundamental is additionally clamped below Nyquist to keep the harmonics
//! that do survive from folding back audibly.
//!
//! # Determinism
//!
//! The node is a per-sample state machine over a single phase accumulator and
//! two [`Smoothed`] controls. Given identical construction parameters, sample
//! rate, and buffer sizes it reproduces its output bit-for-bit, and [`reset`]
//! returns it to its exact initial state.
//!
//! [`reset`]: PhaseDistortionOscillatorNode::reset
//!
//! # Relationship
//!
//! Unlike the fixed-shape [`super::oscillator::OscillatorNode`] (which selects a
//! discrete waveform rather than morphing continuously), the detuned stack of
//! [`super::supersaw::SupersawNode`], the duty-cycle sweep of
//! [`super::pwm_oscillator::PwmOscillatorNode`] (which moves a *step* edge), or
//! the reset-driven [`super::hard_sync_oscillator::HardSyncOscillatorNode`],
//! this node keeps a single, discontinuity-free cosine carrier and brightens it
//! by *warping time* rather than by introducing a hard edge. It therefore needs
//! no band-limiting correction primitive at all.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`PhaseDistortionOscillatorNode::process`] performs no allocation, no
//! locking, and no panicking: it is a pure per-sample state machine. The warp
//! `amount` and amplitude are driven through [`Smoothed`] values so the morph
//! sweep and gain automation never produce zipper clicks.
//!
//! # Provenance
//!
//! Implemented from first principles from the public, long-documented phase-
//! distortion synthesis technique (a time-warped carrier phase). It contains no
//! code, data, or derivative of Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Google Resonance Audio, the Web Audio API, the Synthesis Toolkit
//! (STK), or any other audio engine or toolkit; only the shared mathematical
//! ideas are used. There is no AI or machine learning of any kind.

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

/// Default warp amount (half-way morph).
pub const DEFAULT_AMOUNT: Sample = 0.5;

/// Minimum break point. The warp break is clamped away from `0` so the
/// compressed half-cycle always has a finite, non-degenerate slope.
pub const MIN_BREAK_POINT: Sample = 0.01;

/// Default linear output amplitude. Below unity to leave a little headroom.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fraction of the sample rate above which the fundamental is clamped to stay
/// below Nyquist.
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

/// Maps a user-facing warp `amount` in `[0, 1]` to a break point in
/// `[MIN_BREAK_POINT, 0.5]`.
#[inline]
fn break_point(amount: Sample) -> Sample {
    (0.5 - 0.49 * amount.clamp(0.0, 1.0)).clamp(MIN_BREAK_POINT, 0.5)
}

/// Construction parameters for a [`PhaseDistortionOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PhaseDistortionOscillatorParams {
    /// Fundamental frequency in hertz. Clamped to
    /// `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]`.
    pub frequency_hz: Sample,
    /// Warp amount in `[0, 1]`: `0` is a pure sine, `1` is brightest.
    pub amount: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for PhaseDistortionOscillatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            amount: DEFAULT_AMOUNT,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl PhaseDistortionOscillatorParams {
    /// Returns a copy with every field sanitised into its valid domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            amount: finite_or(self.amount, DEFAULT_AMOUNT).clamp(0.0, 1.0),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A phase-distortion oscillator source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::PhaseDistortionOscillatorNode;
/// use prism_audio_core::param::Ramp;
///
/// let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.0, 0.8);
/// // Sweep the warp upward for the classic phase-distortion brightening.
/// node.set_amount(1.0, Ramp::Immediate);
/// assert_eq!(node.frequency_hz(), 110.0);
/// ```
#[derive(Debug, Clone)]
pub struct PhaseDistortionOscillatorNode {
    /// Fundamental frequency in hertz. Stored as a plain scalar because the
    /// phase accumulator is continuous, so a frequency change is click-free
    /// without smoothing.
    frequency_hz: Sample,
    /// Smoothed warp amount in `[0, 1]`.
    amount: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl PhaseDistortionOscillatorNode {
    /// Creates a phase-distortion oscillator at `frequency_hz` with warp
    /// `amount` and master `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `frequency_hz` is clamped to
    /// `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]` and `amount` to `[0, 1]`.
    #[must_use]
    pub fn new(frequency_hz: Sample, amount: Sample, amplitude: Sample) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            amount: Smoothed::new(finite_or(amount, DEFAULT_AMOUNT).clamp(0.0, 1.0)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase: 0.0,
        }
    }

    /// Builds a phase-distortion oscillator from a
    /// [`PhaseDistortionOscillatorParams`] bundle.
    #[must_use]
    pub fn from_params(params: PhaseDistortionOscillatorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.amount, p.amplitude)
    }

    /// Sets the fundamental frequency in hertz, clamped to
    /// `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]`.
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample) {
        self.frequency_hz =
            finite_or(hz, self.frequency_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    }

    /// Sets a new target warp amount, gliding with `ramp` so the morph sweep is
    /// click-free. The target is clamped to `[0, 1]`.
    #[inline]
    pub fn set_amount(&mut self, amount: Sample, ramp: Ramp) {
        let target = finite_or(amount, self.amount.target()).clamp(0.0, 1.0);
        self.amount.set_target(target, ramp);
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

    /// Returns the warp amount the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn amount(&self) -> Sample {
        self.amount.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample at per-sample phase increment `dt`, advancing
    /// the phase accumulator and the smoothed controls.
    #[inline]
    fn render_sample(&mut self, dt: Sample) -> Sample {
        let amount = self.amount.next_sample();
        let amp = self.amplitude.next_sample();
        let d = break_point(amount);

        let p = self.phase;
        let w = if p < d {
            p / (2.0 * d)
        } else {
            0.5 + (p - d) / (2.0 * (1.0 - d))
        };
        let value = -ops::cos(TAU * w);

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        value * amp
    }
}

impl AudioNode for PhaseDistortionOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let guard_hz = sr * NYQUIST_GUARD;
        let dt = self.frequency_hz.min(guard_hz) / sr;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(dt);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.amount = Smoothed::new(self.amount.target());
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

    fn render(
        node: &mut PhaseDistortionOscillatorNode,
        sample_rate: u32,
        frames: usize,
    ) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render_layout(
        node: &mut PhaseDistortionOscillatorNode,
        layout: ChannelLayout,
        sample_rate: u32,
        frames: usize,
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
    fn upper_harmonic_energy(amount: Sample) -> Sample {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, amount, 0.8);
        let out = render(&mut node, SR, 8_192);
        let mut hi = 0.0;
        for h in 2..=30 {
            let f = 110.0 * h as Sample;
            if f >= SR as Sample * 0.49 {
                break;
            }
            hi += goertzel(&out, SR, f);
        }
        hi
    }

    #[test]
    fn renders_bounded_finite() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.9, 1.0);
        let out = render(&mut node, SR, 8_192);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn amount_zero_is_pure_sine() {
        // At amount 0 the break point is exactly 0.5, w == p, so the output is a
        // mathematically pure sinusoid -cos(2*pi*phase).
        let freq = 100.0;
        let mut node = PhaseDistortionOscillatorNode::new(freq, 0.0, 1.0);
        let frames = 2_048;
        let out = render(&mut node, SR, frames);
        let dt = freq / SR as Sample;
        // Reproduce the node's own phase accumulator (not (n*dt).fract()) so the
        // test proves the warp is the identity, not that two phase recurrences
        // agree bit-for-bit after thousands of additions.
        let mut phase = 0.0_f32;
        for (n, &s) in out.channel(0).iter().enumerate() {
            let reference = -ops::cos(TAU * phase);
            assert!((s - reference).abs() < 1e-6, "n={n} s={s} ref={reference}");
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn brightness_increases_with_amount() {
        let dull = upper_harmonic_energy(0.1);
        let bright = upper_harmonic_energy(0.95);
        assert!(bright > dull * 2.0, "dull={dull} bright={bright}");
    }

    #[test]
    fn pure_sine_has_negligible_upper_harmonics() {
        // amount 0 is a single sinusoid, so harmonics above the fundamental must
        // carry essentially no energy relative to the fundamental.
        let mut node = PhaseDistortionOscillatorNode::new(220.0, 0.0, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 220.0);
        let h2 = goertzel(&out, SR, 440.0);
        assert!(h2 < fund * 1e-3, "fund={fund} h2={h2}");
    }

    #[test]
    fn fundamental_locks_to_frequency() {
        let mut node = PhaseDistortionOscillatorNode::new(220.0, 0.8, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 220.0);
        let off = goertzel(&out, SR, 330.0);
        assert!(fund > off, "fund={fund} off={off}");
    }

    #[test]
    fn first_derivative_is_continuous() {
        // A C1 waveform has no large sample-to-sample jumps: the per-sample
        // change is bounded by the (finite) slope times dt.
        let mut node = PhaseDistortionOscillatorNode::new(55.0, 0.6, 1.0);
        let out = render(&mut node, SR, 8_192);
        for w in out.channel(0).windows(2) {
            assert!((w[1] - w[0]).abs() < 0.05, "step {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn amount_sweep_is_click_free() {
        let frames: usize = 4_096;
        let mut node = PhaseDistortionOscillatorNode::new(55.0, 0.0, 0.8);
        node.set_amount(
            0.75,
            Ramp::Linear {
                samples: frames as u32,
            },
        );
        let out = render(&mut node, SR, frames);
        for w in out.channel(0).windows(2) {
            assert!((w[1] - w[0]).abs() < 0.05, "step {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn deterministic() {
        let mut a = PhaseDistortionOscillatorNode::new(130.0, 0.7, 0.8);
        let mut b = PhaseDistortionOscillatorNode::new(130.0, 0.7, 0.8);
        let ra = render(&mut a, SR, 2_048);
        let rb = render(&mut b, SR, 2_048);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_replays_output() {
        let mut node = PhaseDistortionOscillatorNode::new(130.0, 0.7, 0.8);
        let a = render(&mut node, SR, 1_024);
        node.reset();
        let b = render(&mut node, SR, 1_024);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_restores_phase() {
        let mut node = PhaseDistortionOscillatorNode::new(130.0, 0.7, 0.8);
        let _ = render(&mut node, SR, 97);
        node.reset();
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.0);
        let out = render(&mut node, SR, 1_024);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.25);
        let mut loud = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.5);
        let eq = energy(&render(&mut quiet, SR, 4_096));
        let el = energy(&render(&mut loud, SR, 4_096));
        assert!((el / eq - 4.0).abs() < 0.05, "ratio={}", el / eq);
    }

    #[test]
    fn mono_core_copied_to_stereo() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.8);
        let out = render_layout(&mut node, ChannelLayout::Stereo, SR, 512);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn mono_core_copied_to_quad() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.8);
        let out = render_layout(&mut node, ChannelLayout::Quad, SR, 512);
        for ch in 1..4 {
            assert_eq!(out.channel(0), out.channel(ch));
        }
    }

    #[test]
    fn not_silent() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.8);
        let out = render(&mut node, SR, 1_024);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.8);
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
        let node = PhaseDistortionOscillatorNode::new(110.0, 0.6, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_targets() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.2, 0.5);
        node.set_frequency_hz(321.0);
        node.set_amount(0.9, Ramp::Immediate);
        node.set_amplitude(0.7, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), 321.0);
        assert_eq!(node.amount(), 0.9);
        assert_eq!(node.amplitude(), 0.7);
    }

    #[test]
    fn default_params_in_domain() {
        let p = PhaseDistortionOscillatorParams::default();
        assert_eq!(p.frequency_hz, DEFAULT_FREQUENCY_HZ);
        assert_eq!(p.amount, DEFAULT_AMOUNT);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
        let s = p.sanitised();
        assert_eq!(s.frequency_hz, p.frequency_hz);
        assert_eq!(s.amount, p.amount);
        assert_eq!(s.amplitude, p.amplitude);
    }

    #[test]
    fn from_params_matches_new() {
        let params = PhaseDistortionOscillatorParams {
            frequency_hz: 123.0,
            amount: 0.45,
            amplitude: 0.7,
        };
        let mut a = PhaseDistortionOscillatorNode::from_params(params);
        let mut b = PhaseDistortionOscillatorNode::new(123.0, 0.45, 0.7);
        let ra = render(&mut a, SR, 1_024);
        let rb = render(&mut b, SR, 1_024);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let low = PhaseDistortionOscillatorNode::new(1.0, -1.0, 0.8);
        let high = PhaseDistortionOscillatorNode::new(99_999.0, 9.0, 0.8);
        assert_eq!(low.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(low.amount(), 0.0);
        assert_eq!(high.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(high.amount(), 1.0);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = PhaseDistortionOscillatorNode::new(
            Sample::NAN,
            Sample::INFINITY,
            Sample::NAN,
        );
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.amount(), DEFAULT_AMOUNT);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = PhaseDistortionOscillatorNode::new(110.0, 0.5, 0.8);
        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 110.0);
        node.set_amount(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.amount(), 0.5);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
        node.set_frequency_hz(1.0);
        assert_eq!(node.frequency_hz(), MIN_FREQUENCY_HZ);
        node.set_amount(9.0, Ramp::Immediate);
        assert_eq!(node.amount(), 1.0);
    }

    #[test]
    fn nyquist_guard_keeps_super_nyquist_bounded() {
        let mut node = PhaseDistortionOscillatorNode::new(MAX_FREQUENCY_HZ, 1.0, 1.0);
        let out = render(&mut node, SR, 2_048);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn frequency_change_shifts_fundamental() {
        let mut low = PhaseDistortionOscillatorNode::new(110.0, 0.5, 0.8);
        let mut high = PhaseDistortionOscillatorNode::new(220.0, 0.5, 0.8);
        let lo = render(&mut low, SR, 8_192);
        let hi = render(&mut high, SR, 8_192);
        let lo_at_220 = goertzel(&lo, SR, 220.0);
        let hi_at_220 = goertzel(&hi, SR, 220.0);
        assert!(hi_at_220 > lo_at_220, "lo={lo_at_220} hi={hi_at_220}");
    }
}
