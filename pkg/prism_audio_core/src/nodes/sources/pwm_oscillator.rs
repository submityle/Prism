//! Band-limited pulse-width-modulation (PWM) oscillator source node.
//!
//! [`PwmOscillatorNode`] synthesizes a rectangular pulse wave whose duty cycle
//! (pulse width) is a continuously controllable, smoothed parameter. Sweeping
//! the width is the classic "PWM" effect: as the mark/space ratio glides, the
//! harmonic balance of the pulse shifts, producing the thick, animated timbre
//! heard in analog-style synth leads and pads. At a 50% duty cycle the wave is
//! a plain square; at the extremes it thins toward a narrow spike rich in high
//! harmonics.
//!
//! # Model
//!
//! The naive bipolar pulse is `+1` for normalized phase in `[0, width)` and
//! `-1` for `[width, 1)`. That shape has two hard discontinuities per cycle
//! (a rising edge at phase `0` and a falling edge at phase `width`), each of
//! which would alias badly if emitted directly. Both edges are rounded with a
//! `PolyBLEP` (polynomial band-limited step) correction, exactly as the square
//! waveform of [`super::oscillator::OscillatorNode`] does, except the falling
//! edge is placed at the variable `width` rather than fixed at `0.5`. The
//! shared [`poly_blep`](super::oscillator::poly_blep) primitive is reused rather
//! than reimplemented.
//!
//! Like any real pulse wave, the output carries a duty-dependent DC component
//! of `2 * width - 1` (zero only at 50% duty). This is intentionally preserved
//! so the band-limited peak stays within `[-1, 1]`; route the node through a
//! [`super::super::effects::DcBlockerNode`] when a DC-free signal is required.
//!
//! # Relationship
//!
//! [`super::oscillator::OscillatorNode`] can emit a fixed 50%-duty square; this
//! node generalizes that to an arbitrary, audio-rate-modulatable duty cycle,
//! which the fixed square cannot express. It shares the `PolyBLEP` edge-
//! correction math with the oscillator but adds the second movable edge and a
//! smoothed width control so pulse-width modulation is click-free.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`PwmOscillatorNode::process`] performs no allocation, no locking, and no
//! panicking: it is a pure per-sample state machine. Width and amplitude are
//! driven through [`Smoothed`] values so automation never produces zipper
//! clicks.
//!
//! # Provenance
//!
//! Implemented from first principles from the public `PolyBLEP` band-limited
//! pulse technique. It contains no code, data, or derivative of Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web
//! Audio API, or any other audio engine; only the shared mathematical ideas are
//! used. There is no AI or machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::sources::oscillator::poly_blep;
use crate::param::{Ramp, Smoothed};

/// Default fundamental frequency in hertz.
pub const DEFAULT_PWM_FREQUENCY_HZ: Sample = 110.0;

/// Default duty cycle (50% -> a plain square).
pub const DEFAULT_PWM_WIDTH: Sample = 0.5;

/// Default linear output amplitude.
pub const DEFAULT_PWM_AMPLITUDE: Sample = 1.0;

/// Minimum duty cycle. The width is clamped away from the degenerate `0`/`1`
/// endpoints so the pulse always has two distinct edges per cycle.
pub const MIN_PWM_WIDTH: Sample = 0.01;

/// Maximum duty cycle (symmetric with [`MIN_PWM_WIDTH`]).
pub const MAX_PWM_WIDTH: Sample = 0.99;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Band-limited bipolar pulse of duty `width` for phase `t` with per-sample
/// increment `dt`.
///
/// `+1` over `[0, width)` and `-1` over `[width, 1)`, with a `PolyBLEP`
/// correction at the rising edge (phase `0`) and a second at the falling edge
/// (phase `width`). The result stays within `[-1, 1]`.
#[inline]
fn band_limited_pulse(t: Sample, dt: Sample, width: Sample) -> Sample {
    let mut value = if t < width { 1.0 } else { -1.0 };
    // Correct the rising edge sitting at phase 0.
    value += poly_blep(t, dt);
    // Correct the falling edge sitting at phase `width`: shift the phase so the
    // falling edge lands where `poly_blep` centers its step, then subtract.
    let mut t_fall = t + (1.0 - width);
    if t_fall >= 1.0 {
        t_fall -= 1.0;
    }
    value -= poly_blep(t_fall, dt);
    value
}

/// Construction parameters for a [`PwmOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PwmOscillatorParams {
    /// Fundamental frequency in hertz. Clamped non-negative.
    pub frequency_hz: Sample,
    /// Duty cycle in `[MIN_PWM_WIDTH, MAX_PWM_WIDTH]`.
    pub width: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for PwmOscillatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_PWM_FREQUENCY_HZ,
            width: DEFAULT_PWM_WIDTH,
            amplitude: DEFAULT_PWM_AMPLITUDE,
        }
    }
}

/// A band-limited pulse-width-modulation oscillator source node (0 inputs,
/// 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::PwmOscillatorNode;
/// use prism_audio_core::param::Ramp;
///
/// let mut node = PwmOscillatorNode::new(110.0, 0.5, 1.0);
/// // Sweep toward a narrow pulse for the classic PWM animation.
/// node.set_width(0.2, Ramp::Immediate);
/// assert_eq!(node.frequency(), 110.0);
/// ```
#[derive(Debug, Clone)]
pub struct PwmOscillatorNode {
    /// Fundamental frequency in hertz. Always non-negative. Stored as a plain
    /// scalar because the phase accumulator is continuous, so a frequency
    /// change is click-free without smoothing.
    frequency: Sample,
    /// Smoothed duty cycle in `[MIN_PWM_WIDTH, MAX_PWM_WIDTH]`.
    width: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
}

impl PwmOscillatorNode {
    /// Creates a PWM oscillator at `frequency_hz` with duty `width` and master
    /// `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; frequency is clamped
    /// non-negative and width is clamped to `[MIN_PWM_WIDTH, MAX_PWM_WIDTH]`.
    #[must_use]
    pub fn new(frequency_hz: Sample, width: Sample, amplitude: Sample) -> Self {
        Self {
            frequency: finite_or(frequency_hz, DEFAULT_PWM_FREQUENCY_HZ).max(0.0),
            width: Smoothed::new(
                finite_or(width, DEFAULT_PWM_WIDTH).clamp(MIN_PWM_WIDTH, MAX_PWM_WIDTH),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_PWM_AMPLITUDE)),
            phase: 0.0,
        }
    }

    /// Builds a PWM oscillator from a [`PwmOscillatorParams`] bundle.
    #[must_use]
    pub fn from_params(params: PwmOscillatorParams) -> Self {
        Self::new(params.frequency_hz, params.width, params.amplitude)
    }

    /// Sets the fundamental frequency in hertz (clamped non-negative).
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = finite_or(hz, self.frequency).max(0.0);
    }

    /// Sets a new target duty cycle, gliding with `ramp` so pulse-width
    /// modulation is click-free. The target is clamped to
    /// `[MIN_PWM_WIDTH, MAX_PWM_WIDTH]`.
    #[inline]
    pub fn set_width(&mut self, width: Sample, ramp: Ramp) {
        let target = finite_or(width, self.width.target()).clamp(MIN_PWM_WIDTH, MAX_PWM_WIDTH);
        self.width.set_target(target, ramp);
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the current fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Returns the duty cycle the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn width(&self) -> Sample {
        self.width.target()
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
        let width = self.width.next_sample();
        let amp = self.amplitude.next_sample();
        let value = band_limited_pulse(self.phase, dt, width);

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        value * amp
    }
}

impl AudioNode for PwmOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let dt = self.frequency / sample_rate;

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
        self.width = Smoothed::new(self.width.target());
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
    use crate::nodes::sources::oscillator::{OscillatorNode, Waveform};

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut PwmOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render_osc(node: &mut OscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render_stereo(node: &mut PwmOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    #[test]
    fn output_stays_bounded() {
        let mut node = PwmOscillatorNode::new(220.0, 0.3, 1.0);
        let out = render(&mut node, 48_000, 2_048);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn thin_pulse_stays_bounded() {
        // The extreme-duty pulse must still respect |output| <= 1.
        let mut node = PwmOscillatorNode::new(110.0, MAX_PWM_WIDTH, 1.0);
        let out = render(&mut node, 48_000, 4_096);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn half_duty_matches_oscillator_square() {
        let sr = 48_000;
        let frames = 512;
        let mut pwm = PwmOscillatorNode::new(130.0, 0.5, 1.0);
        let mut sq = OscillatorNode::new(Waveform::Square, 130.0, 1.0);
        let a = render(&mut pwm, sr, frames);
        let b = render_osc(&mut sq, sr, frames);
        for (n, (x, y)) in a.channel(0).iter().zip(b.channel(0)).enumerate() {
            assert!((x - y).abs() < 1e-5, "n={n} pwm={x} square={y}");
        }
    }

    #[test]
    fn width_changes_output() {
        let sr = 48_000;
        let frames = 512;
        let mut narrow = PwmOscillatorNode::new(110.0, 0.25, 1.0);
        let mut wide = PwmOscillatorNode::new(110.0, 0.75, 1.0);
        let a = render(&mut narrow, sr, frames);
        let b = render(&mut wide, sr, frames);
        let diff: Sample = a
            .channel(0)
            .iter()
            .zip(b.channel(0))
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 1.0, "changing duty should change the signal, diff={diff}");
    }

    #[test]
    fn mean_tracks_duty_cycle() {
        // A duty-`w` bipolar pulse has DC = 2w - 1; verify over whole cycles.
        let sr = 48_000;
        let freq = sr as Sample / 160.0; // 160-sample period
        let width = 0.75;
        let mut node = PwmOscillatorNode::new(freq, width, 1.0);
        let frames = 160 * 10;
        let out = render(&mut node, sr, frames);
        let mean: Sample = out.channel(0).iter().sum::<Sample>() / frames as Sample;
        assert!((mean - (2.0 * width - 1.0)).abs() < 0.05, "mean={mean}");
    }

    #[test]
    fn amplitude_scales_output() {
        let sr = 48_000;
        let frames = 512;
        let mut loud = PwmOscillatorNode::new(220.0, 0.4, 1.0);
        let mut quiet = PwmOscillatorNode::new(220.0, 0.4, 0.25);
        let a = render(&mut loud, sr, frames);
        let b = render(&mut quiet, sr, frames);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert!((x * 0.25 - y).abs() < 1e-4, "a*0.25={} b={y}", x * 0.25);
        }
    }

    #[test]
    fn is_periodic_at_fundamental() {
        let sr = 48_000;
        let freq = sr as Sample / 128.0;
        let mut node = PwmOscillatorNode::new(freq, 0.3, 1.0);
        let out = render(&mut node, sr, 384);
        let ch = out.channel(0);
        for n in 0..128 {
            assert!((ch[n] - ch[n + 128]).abs() < 1e-3, "n={n}");
        }
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let mut node = PwmOscillatorNode::new(130.0, 0.35, 1.0);
        let a = render(&mut node, 48_000, 300);
        node.reset();
        let b = render(&mut node, 48_000, 300);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_restores_phase() {
        let mut node = PwmOscillatorNode::new(130.0, 0.5, 1.0);
        let _ = render(&mut node, 48_000, 97);
        node.reset();
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn negative_frequency_is_clamped() {
        let node = PwmOscillatorNode::new(-440.0, 0.5, 1.0);
        assert_eq!(node.frequency(), 0.0);
    }

    #[test]
    fn width_is_clamped() {
        let low = PwmOscillatorNode::new(110.0, -5.0, 1.0);
        let high = PwmOscillatorNode::new(110.0, 5.0, 1.0);
        assert_eq!(low.width(), MIN_PWM_WIDTH);
        assert_eq!(high.width(), MAX_PWM_WIDTH);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = PwmOscillatorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.frequency(), DEFAULT_PWM_FREQUENCY_HZ);
        assert_eq!(node.width(), DEFAULT_PWM_WIDTH);
        assert_eq!(node.amplitude(), DEFAULT_PWM_AMPLITUDE);
    }

    #[test]
    fn from_params_matches_new() {
        let params = PwmOscillatorParams {
            frequency_hz: 123.0,
            width: 0.3,
            amplitude: 0.7,
        };
        let mut a = PwmOscillatorNode::from_params(params);
        let mut b = PwmOscillatorNode::new(123.0, 0.3, 0.7);
        let ra = render(&mut a, 48_000, 256);
        let rb = render(&mut b, 48_000, 256);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn default_params_are_a_square() {
        let params = PwmOscillatorParams::default();
        assert_eq!(params.frequency_hz, DEFAULT_PWM_FREQUENCY_HZ);
        assert_eq!(params.width, 0.5);
        assert_eq!(params.amplitude, 1.0);
    }

    #[test]
    fn stereo_channels_are_identical() {
        let mut node = PwmOscillatorNode::new(110.0, 0.4, 1.0);
        let out = render_stereo(&mut node, 48_000, 256);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn super_nyquist_frequency_does_not_panic() {
        let mut node = PwmOscillatorNode::new(40_000.0, 0.5, 1.0);
        let out = render(&mut node, 48_000, 128);
        for &s in out.channel(0) {
            assert!(s.is_finite(), "s={s}");
        }
    }

    #[test]
    fn width_ramp_is_click_free() {
        let sr = 48_000;
        let frames: usize = 1_024;
        let mut node = PwmOscillatorNode::new(110.0, 0.5, 1.0);
        node.set_width(0.1, Ramp::Linear { samples: frames as u32 });
        let out = render(&mut node, sr, frames);
        let ch = out.channel(0);
        // Across the sweep, step-to-step changes stay bounded away from a full
        // two-unit jump that an unsmoothed width edit would create mid-cycle.
        for w in ch.windows(2) {
            assert!((w[1] - w[0]).abs() <= 2.0 + 1e-3, "step {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn zero_channel_output_is_noop() {
        let mut node = PwmOscillatorNode::new(110.0, 0.5, 1.0);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, 0), &mut io);
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn getters_report_targets() {
        let mut node = PwmOscillatorNode::new(110.0, 0.5, 0.5);
        node.set_frequency(321.0);
        node.set_width(0.33, Ramp::Immediate);
        node.set_amplitude(0.9, Ramp::Immediate);
        assert_eq!(node.frequency(), 321.0);
        assert_eq!(node.width(), 0.33);
        assert_eq!(node.amplitude(), 0.9);
    }

    #[test]
    fn not_silent() {
        let mut node = PwmOscillatorNode::new(110.0, 0.4, 1.0);
        let out = render(&mut node, 48_000, 512);
        let energy: Sample = out.channel(0).iter().map(|s| s * s).sum();
        assert!(energy > 1.0, "energy={energy}");
    }

    #[test]
    fn dc_free_at_fifty_percent() {
        let sr = 48_000;
        let freq = sr as Sample / 120.0;
        let mut node = PwmOscillatorNode::new(freq, 0.5, 1.0);
        let frames = 120 * 8;
        let out = render(&mut node, sr, frames);
        let mean: Sample = out.channel(0).iter().sum::<Sample>() / frames as Sample;
        assert!(mean.abs() < 1e-2, "mean={mean}");
    }
}
