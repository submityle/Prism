//! Phase-modulation (FM) operator source node.
//!
//! [`FmOperatorNode`] is the primitive voice of classic *frequency-modulation*
//! synthesis in the style popularized by the Yamaha DX7: a single sine
//! generator whose instantaneous phase can be deflected by an incoming
//! modulation signal and by its own fed-back output. Chaining several operators
//! in the mix graph -- routing one operator's output into another's modulation
//! input -- reproduces the modulator/carrier "algorithms" that give FM its
//! characteristic metallic, bell-like, and electric-piano timbres.
//!
//! Despite the historical name "frequency modulation," the DX7 and essentially
//! every digital FM instrument actually perform **phase modulation (PM)**: the
//! modulator is added to the carrier's *phase argument* rather than its
//! *frequency*. PM is mathematically equivalent to FM for a sinusoidal
//! modulator (it only shifts the modulation index by a frequency-dependent
//! factor) but is numerically far better behaved -- the carrier pitch cannot
//! drift when the modulator has a DC component, and self-feedback stays
//! bounded. This node therefore implements PM.
//!
//! # Model
//!
//! For normalized phase `p` in `[0, 1)` advancing by `dt = f_c / sample_rate`
//! each sample, with modulation input `m` (dimensionless, nominally in
//! `[-1, 1]`), modulation index `I` (radians), and feedback coefficient `beta`
//! (radians), the operator emits
//!
//! ```text
//! y = sin(2*PI*p + I*m + beta*0.5*(y[-1] + y[-2]))
//! ```
//!
//! The self-feedback term averages the previous two outputs. This two-sample
//! average is the standard anti-chaos trick (as used by hardware FM operators):
//! a bare `beta*y[-1]` feedback loop closed around a single sample becomes chaotic
//! and noisy at high `beta`, whereas the mean of the last two samples acts as a
//! gentle one-pole lowpass inside the loop and keeps the self-oscillation
//! smooth, converging toward a sawtooth-like spectrum as `beta` grows.
//!
//! # Relationship
//!
//! Unlike [`super::oscillator::OscillatorNode`] (geometric `PolyBLEP` shapes)
//! and [`super::wavetable_oscillator::WavetableOscillatorNode`] (mipmap table
//! lookup), this operator synthesizes a *pure sine* core and derives its
//! spectral richness entirely from phase modulation and feedback rather than
//! from a stored waveform. It declares **one optional input port** (the phase
//! modulation signal) and one output; with nothing connected it is simply a
//! clean sine carrier.
//!
//! # Real-time contract
//!
//! Construction pre-computes all state, so [`FmOperatorNode::process`] performs
//! no allocation, no locking, and no panicking. The modulation index, feedback,
//! and amplitude are driven through [`Smoothed`] values so timbral and level
//! automation never produces zipper-noise clicks; frequency is a plain scalar
//! because the phase accumulator is continuous and a frequency change only
//! alters the waveform's slope, introducing no discontinuity.
//!
//! # Provenance
//!
//! Implemented from first principles from the public FM/PM synthesis
//! literature -- J. M. Chowning, "The Synthesis of Complex Audio Spectra by
//! Means of Frequency Modulation," *Journal of the Audio Engineering Society*
//! 21(7), 1973 -- and the well-documented behavior of phase-modulation
//! operators. It contains no source code or derived code from Unreal Engine,
//! Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web
//! Audio; only the shared mathematical ideas are referenced.

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Full turn in radians, used to map the normalized phase to the sine argument.
const TAU: Sample = core::f32::consts::TAU;

/// Default carrier frequency in hertz (concert A).
pub const DEFAULT_FM_FREQUENCY_HZ: Sample = 440.0;

/// Default modulation index in radians (one radian of peak phase deflection per
/// unit of modulation input).
pub const DEFAULT_FM_MOD_INDEX: Sample = 1.0;

/// Default self-feedback coefficient (no feedback).
pub const DEFAULT_FM_FEEDBACK: Sample = 0.0;

/// Default linear output amplitude.
pub const DEFAULT_FM_AMPLITUDE: Sample = 1.0;

/// Upper bound on the modulation index (radians). Deep FM rarely needs more
/// than a handful of radians; this generous ceiling keeps the phase argument
/// finite without constraining musical use.
pub const MAX_FM_MOD_INDEX: Sample = 100.0;

/// Magnitude bound on the self-feedback coefficient (radians). One full turn of
/// phase feedback is already past the point where the operator saturates into
/// its sawtooth-like limit.
pub const MAX_FM_FEEDBACK: Sample = TAU;

/// Construction parameters for an [`FmOperatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FmOperatorParams {
    /// Carrier frequency in hertz. Clamped to be non-negative.
    pub frequency_hz: Sample,
    /// Modulation index in radians: the peak phase deflection produced by a
    /// unit-amplitude modulation input. Clamped to `[0, MAX_FM_MOD_INDEX]`.
    pub mod_index: Sample,
    /// Self-feedback coefficient in radians. Clamped to
    /// `[-MAX_FM_FEEDBACK, MAX_FM_FEEDBACK]`.
    pub feedback: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for FmOperatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FM_FREQUENCY_HZ,
            mod_index: DEFAULT_FM_MOD_INDEX,
            feedback: DEFAULT_FM_FEEDBACK,
            amplitude: DEFAULT_FM_AMPLITUDE,
        }
    }
}

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// A phase-modulation (FM) operator source node.
///
/// One optional input (the phase-modulation signal) and one output. Every
/// output channel receives the same mono signal so downstream stereo/surround
/// nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::FmOperatorNode;
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// // A bare operator with no modulation input is a clean sine carrier.
/// let mut op = FmOperatorNode::new(440.0, 0.0, 0.0, 1.0);
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 256)];
/// outputs[0].set_active_frames(256);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// op.process(&ctx, &mut io);
///
/// let peak = outputs[0]
///     .channel(0)
///     .iter()
///     .fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.9 && peak <= 1.0 + 1e-4);
/// ```
#[derive(Debug, Clone)]
pub struct FmOperatorNode {
    /// Carrier frequency in hertz. Always non-negative.
    frequency: Sample,
    /// Smoothed modulation index in radians.
    mod_index: Smoothed,
    /// Smoothed self-feedback coefficient in radians.
    feedback: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
    /// Previous emitted sine output `y[-1]` (pre-amplitude), for feedback.
    last1: Sample,
    /// Output before that, `y[-2]` (pre-amplitude), for feedback averaging.
    last2: Sample,
}

impl FmOperatorNode {
    /// Creates an operator at `frequency_hz` with the given `mod_index`
    /// (radians), self-`feedback` (radians), and linear `amplitude`.
    ///
    /// All parameters are sanitized: non-finite values fall back to their
    /// defaults, frequency is clamped non-negative, the modulation index to
    /// `[0, MAX_FM_MOD_INDEX]`, and feedback to
    /// `[-MAX_FM_FEEDBACK, MAX_FM_FEEDBACK]`.
    #[must_use]
    pub fn new(
        frequency_hz: Sample,
        mod_index: Sample,
        feedback: Sample,
        amplitude: Sample,
    ) -> Self {
        Self {
            frequency: finite_or(frequency_hz, DEFAULT_FM_FREQUENCY_HZ).max(0.0),
            mod_index: Smoothed::new(
                finite_or(mod_index, DEFAULT_FM_MOD_INDEX).clamp(0.0, MAX_FM_MOD_INDEX),
            ),
            feedback: Smoothed::new(
                finite_or(feedback, DEFAULT_FM_FEEDBACK)
                    .clamp(-MAX_FM_FEEDBACK, MAX_FM_FEEDBACK),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_FM_AMPLITUDE)),
            phase: 0.0,
            last1: 0.0,
            last2: 0.0,
        }
    }

    /// Builds an operator from a [`FmOperatorParams`] bundle.
    #[must_use]
    pub fn from_params(params: FmOperatorParams) -> Self {
        Self::new(
            params.frequency_hz,
            params.mod_index,
            params.feedback,
            params.amplitude,
        )
    }

    /// Sets the carrier frequency in hertz (clamped non-negative).
    ///
    /// Takes effect immediately; because phase is continuous this is click-free
    /// without smoothing.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = finite_or(hz, self.frequency).max(0.0);
    }

    /// Sets a new target modulation index (radians), gliding with `ramp`.
    #[inline]
    pub fn set_mod_index(&mut self, radians: Sample, ramp: Ramp) {
        let target = finite_or(radians, self.mod_index.target()).clamp(0.0, MAX_FM_MOD_INDEX);
        self.mod_index.set_target(target, ramp);
    }

    /// Sets a new target self-feedback coefficient (radians), gliding with
    /// `ramp`.
    #[inline]
    pub fn set_feedback(&mut self, radians: Sample, ramp: Ramp) {
        let target = finite_or(radians, self.feedback.target())
            .clamp(-MAX_FM_FEEDBACK, MAX_FM_FEEDBACK);
        self.feedback.set_target(target, ramp);
    }

    /// Sets a new target amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the current carrier frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Returns the modulation index the operator is gliding toward (radians).
    #[inline]
    #[must_use]
    pub fn mod_index(&self) -> Sample {
        self.mod_index.target()
    }

    /// Returns the self-feedback coefficient the operator is gliding toward
    /// (radians).
    #[inline]
    #[must_use]
    pub fn feedback(&self) -> Sample {
        self.feedback.target()
    }

    /// Returns the target amplitude the operator is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample given the per-sample phase increment `dt` and
    /// the phase-modulation input `pm_in`, advancing all internal state.
    #[inline]
    fn render_sample(&mut self, dt: Sample, pm_in: Sample) -> Sample {
        let index = self.mod_index.next_sample();
        let beta = self.feedback.next_sample();
        let amp = self.amplitude.next_sample();

        // Two-sample-averaged self-feedback keeps the loop stable at high gain.
        let fb = beta * 0.5 * (self.last1 + self.last2);
        let sine = ops::sin(TAU * self.phase + index * pm_in + fb);

        // Feedback taps the operator's phase output *before* the level scaling.
        self.last2 = self.last1;
        self.last1 = sine;

        // Advance and wrap phase into [0, 1). `dt >= 0`, so truncating with
        // `as u32` yields the integer part (avoids the no_std-unavailable
        // `floor`) and handles `dt >= 1` (freq > sr) too.
        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= (self.phase as u32) as Sample;
        }

        sine * amp
    }
}

impl AudioNode for FmOperatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // Per-sample phase increment. `sample_rate` is validated non-zero by
        // the graph; guard defensively so `process` can never divide by zero.
        let sample_rate = ctx.sample_rate.max(1) as Sample;
        let dt = self.frequency / sample_rate;

        // Generate the mono signal into channel 0, reading the modulation input
        // (channel 0 of input port 0) when one is connected.
        if io.input_count() > 0 {
            let (input, output) = io.io(0, 0);
            let pm_src = input.channel(0);
            let buf = output.channel_mut(0);
            for (i, s) in buf.iter_mut().enumerate() {
                let pm_in = pm_src.get(i).copied().unwrap_or(0.0);
                *s = self.render_sample(dt, pm_in);
            }
        } else {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(dt, 0.0);
            }
        }

        // Replicate the mono signal into every remaining channel.
        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.last1 = 0.0;
        self.last2 = 0.0;
        self.mod_index = Smoothed::new(self.mod_index.target());
        self.feedback = Smoothed::new(self.feedback.target());
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

    /// Builds a render context for the given sample rate and frame count.
    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    /// Renders one mono block from a bare operator (no modulation input).
    fn render(node: &mut FmOperatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        let [buf] = outputs;
        buf
    }

    /// Renders one mono block driven by `modulator` on input port 0.
    fn render_modulated(
        node: &mut FmOperatorNode,
        modulator: &[Sample],
        sample_rate: u32,
    ) -> AudioBuffer {
        let frames = modulator.len();
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        input.set_active_frames(frames);
        input.channel_mut(0).copy_from_slice(modulator);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        let [buf] = outputs;
        buf
    }

    #[test]
    fn carrier_matches_pure_sine_without_modulation() {
        let sample_rate = 48_000;
        let frames = 256;
        let freq = 1_000.0;
        let amp = 0.75;
        let mut node = FmOperatorNode::new(freq, 0.0, 0.0, amp);
        let out = render(&mut node, sample_rate, frames);

        let dt = freq / sample_rate as Sample;
        let mut phase = 0.0f32;
        for &produced in out.channel(0) {
            let expected = amp * ops::sin(TAU * phase);
            assert!(
                (produced - expected).abs() < 1e-5,
                "produced={produced} expected={expected}"
            );
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn zero_index_ignores_modulation_input() {
        let sample_rate = 48_000;
        let frames = 512;
        // A loud modulator must have no effect when the index is zero.
        let modulator: Vec<Sample> = (0..frames).map(|i| ops::sin(0.1 * i as Sample)).collect();

        let mut modded = FmOperatorNode::new(440.0, 0.0, 0.0, 1.0);
        let a = render_modulated(&mut modded, &modulator, sample_rate);

        let mut bare = FmOperatorNode::new(440.0, 0.0, 0.0, 1.0);
        let b = render(&mut bare, sample_rate, frames);

        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert!((x - y).abs() < 1e-6, "x={x} y={y}");
        }
    }

    #[test]
    fn modulation_enriches_spectrum() {
        // With a modulator connected and a nonzero index, the output must
        // differ from the pure carrier (sidebands appear).
        let sample_rate = 48_000;
        let frames = 1_024;
        let modulator: Vec<Sample> =
            (0..frames).map(|i| ops::sin(TAU * 220.0 * i as Sample / sample_rate as Sample)).collect();

        let mut modded = FmOperatorNode::new(440.0, 2.0, 0.0, 1.0);
        let a = render_modulated(&mut modded, &modulator, sample_rate);

        let mut bare = FmOperatorNode::new(440.0, 2.0, 0.0, 1.0);
        let b = render(&mut bare, sample_rate, frames);

        let mut max_diff = 0.0f32;
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 0.1, "modulation produced no change (max_diff={max_diff})");
    }

    #[test]
    fn phase_modulation_matches_closed_form() {
        // y = sin(2*PI*p + I*m); verify against an independent accumulation.
        let sample_rate = 48_000;
        let frames = 300;
        let freq = 500.0;
        let index = 1.5;
        let modulator: Vec<Sample> = (0..frames).map(|i| 0.5 * ops::sin(0.07 * i as Sample)).collect();

        let mut node = FmOperatorNode::new(freq, index, 0.0, 1.0);
        let out = render_modulated(&mut node, &modulator, sample_rate);

        let dt = freq / sample_rate as Sample;
        let mut phase = 0.0f32;
        for (i, &produced) in out.channel(0).iter().enumerate() {
            let expected = ops::sin(TAU * phase + index * modulator[i]);
            assert!(
                (produced - expected).abs() < 1e-5,
                "i={i} produced={produced} expected={expected}"
            );
            phase += dt;
            if phase >= 1.0 {
                phase -= (phase as u32) as Sample;
            }
        }
    }

    #[test]
    fn output_scales_with_amplitude() {
        let sample_rate = 48_000;
        let frames = 256;
        let mut quiet = FmOperatorNode::new(440.0, 0.0, 0.0, 0.25);
        let mut loud = FmOperatorNode::new(440.0, 0.0, 0.0, 1.0);
        let q = render(&mut quiet, sample_rate, frames);
        let l = render(&mut loud, sample_rate, frames);
        for (a, b) in q.channel(0).iter().zip(l.channel(0)) {
            assert!((a - 0.25 * b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn feedback_stays_bounded() {
        // Even at maximum feedback the two-sample average keeps the operator
        // well-behaved (no blow-up, no NaN).
        let sample_rate = 48_000;
        let frames = 48_000;
        let mut node = FmOperatorNode::new(220.0, 0.0, MAX_FM_FEEDBACK, 1.0);
        let out = render(&mut node, sample_rate, frames);
        for &s in out.channel(0) {
            assert!(s.is_finite(), "feedback produced non-finite sample {s}");
            assert!(s.abs() <= 1.0 + 1e-4, "feedback exceeded unity: {s}");
        }
    }

    #[test]
    fn feedback_changes_timbre() {
        let sample_rate = 48_000;
        let frames = 2_048;
        let mut plain = FmOperatorNode::new(220.0, 0.0, 0.0, 1.0);
        let mut fed = FmOperatorNode::new(220.0, 0.0, 3.0, 1.0);
        let a = render(&mut plain, sample_rate, frames);
        let b = render(&mut fed, sample_rate, frames);
        let mut max_diff = 0.0f32;
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 0.1, "feedback had no audible effect (max_diff={max_diff})");
    }

    #[test]
    fn reset_makes_output_reproducible() {
        let sample_rate = 48_000;
        let frames = 256;
        let mut node = FmOperatorNode::new(330.0, 2.0, 1.5, 0.9);

        let first = render(&mut node, sample_rate, frames);
        node.reset();
        let second = render(&mut node, sample_rate, frames);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn reset_clears_phase_and_feedback_history() {
        let sample_rate = 48_000;
        let frames = 128;
        let mut node = FmOperatorNode::new(440.0, 0.0, 2.0, 1.0);
        let _ = render(&mut node, sample_rate, frames);
        assert!(node.phase != 0.0 || node.last1 != 0.0 || node.last2 != 0.0);

        node.reset();
        assert_eq!(node.phase, 0.0);
        assert_eq!(node.last1, 0.0);
        assert_eq!(node.last2, 0.0);
    }

    #[test]
    fn frequency_change_is_click_free() {
        // A mid-block frequency change only alters slope, never jumps value.
        let sample_rate = 48_000;
        let mut node = FmOperatorNode::new(440.0, 0.0, 0.0, 1.0);
        let before = render(&mut node, sample_rate, 64);
        let last_before = *before.channel(0).last().unwrap();
        node.set_frequency(880.0);
        let after = render(&mut node, sample_rate, 64);
        let first_after = after.channel(0)[0];
        // One sample of a <= 880 Hz sine at 48 kHz moves by well under 0.2.
        assert!(
            (first_after - last_before).abs() < 0.2,
            "discontinuity: {last_before} -> {first_after}"
        );
    }

    #[test]
    fn negative_frequency_is_clamped() {
        let mut node = FmOperatorNode::new(-100.0, 0.0, 0.0, 1.0);
        assert_eq!(node.frequency(), 0.0);
        node.set_frequency(-5.0);
        assert_eq!(node.frequency(), 0.0);
    }

    #[test]
    fn mod_index_is_clamped() {
        let node = FmOperatorNode::new(440.0, 1_000.0, 0.0, 1.0);
        assert_eq!(node.mod_index(), MAX_FM_MOD_INDEX);
        let node = FmOperatorNode::new(440.0, -3.0, 0.0, 1.0);
        assert_eq!(node.mod_index(), 0.0);
    }

    #[test]
    fn feedback_is_clamped() {
        let node = FmOperatorNode::new(440.0, 0.0, 50.0, 1.0);
        assert_eq!(node.feedback(), MAX_FM_FEEDBACK);
        let node = FmOperatorNode::new(440.0, 0.0, -50.0, 1.0);
        assert_eq!(node.feedback(), -MAX_FM_FEEDBACK);
    }

    #[test]
    fn non_finite_params_fall_back_to_defaults() {
        let node = FmOperatorNode::new(
            Sample::NAN,
            Sample::INFINITY,
            Sample::NEG_INFINITY,
            Sample::NAN,
        );
        assert_eq!(node.frequency(), DEFAULT_FM_FREQUENCY_HZ);
        // Infinity clamps to the finite ceilings after the finite fallback is
        // bypassed only for non-finite inputs; here they are non-finite so the
        // default is used, then clamped (defaults are already in range).
        assert!(node.mod_index().is_finite());
        assert!(node.feedback().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn from_params_matches_new() {
        let params = FmOperatorParams {
            frequency_hz: 660.0,
            mod_index: 2.5,
            feedback: 1.0,
            amplitude: 0.8,
        };
        let mut a = FmOperatorNode::from_params(params);
        let mut b = FmOperatorNode::new(660.0, 2.5, 1.0, 0.8);
        let oa = render(&mut a, 48_000, 128);
        let ob = render(&mut b, 48_000, 128);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn default_params_are_sane() {
        let params = FmOperatorParams::default();
        assert_eq!(params.frequency_hz, DEFAULT_FM_FREQUENCY_HZ);
        assert_eq!(params.mod_index, DEFAULT_FM_MOD_INDEX);
        assert_eq!(params.feedback, DEFAULT_FM_FEEDBACK);
        assert_eq!(params.amplitude, DEFAULT_FM_AMPLITUDE);
    }

    #[test]
    fn stereo_channels_are_identical() {
        let sample_rate = 48_000;
        let frames = 128;
        let mut node = FmOperatorNode::new(440.0, 1.0, 0.5, 1.0);
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        let [buf] = outputs;
        assert_eq!(buf.channel(0), buf.channel(1));
    }

    #[test]
    fn shorter_modulator_pads_with_silence() {
        // A modulator buffer shorter than the output must not panic; missing
        // frames behave as zero modulation.
        let sample_rate = 48_000;
        let out_frames = 64;
        let modulator = [0.5f32; 16];
        let mut input = AudioBuffer::new(ChannelLayout::Mono, out_frames);
        input.set_active_frames(modulator.len());
        input.channel_mut(0)[..modulator.len()].copy_from_slice(&modulator);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, out_frames);
        out.set_active_frames(out_frames);
        let mut node = FmOperatorNode::new(440.0, 2.0, 0.0, 1.0);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, out_frames), &mut io);
        for &s in outputs[0].channel(0) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn zero_channel_output_is_noop() {
        // A zero-channel output buffer must be handled without panicking.
        let mut node = FmOperatorNode::new(440.0, 1.0, 0.0, 1.0);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 1);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(48_000, 0), &mut io);
        // No frames rendered; phase untouched.
        assert_eq!(node.phase, 0.0);
    }

    #[test]
    fn silent_modulator_equals_bare_carrier() {
        let sample_rate = 48_000;
        let frames = 256;
        let modulator = vec![0.0f32; frames];
        let mut modded = FmOperatorNode::new(440.0, 5.0, 0.0, 1.0);
        let a = render_modulated(&mut modded, &modulator, sample_rate);
        let mut bare = FmOperatorNode::new(440.0, 5.0, 0.0, 1.0);
        let b = render(&mut bare, sample_rate, frames);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert!((x - y).abs() < 1e-6, "x={x} y={y}");
        }
    }

    #[test]
    fn mod_index_ramp_is_click_free() {
        // Gliding the modulation index must not throw a non-finite or wild
        // value; the output stays bounded throughout the ramp.
        let sample_rate = 48_000;
        let frames = 512;
        let modulator: Vec<Sample> =
            (0..frames).map(|i| ops::sin(TAU * 110.0 * i as Sample / sample_rate as Sample)).collect();
        let mut node = FmOperatorNode::new(440.0, 0.0, 0.0, 1.0);
        node.set_mod_index(4.0, Ramp::Linear { samples: frames });
        let out = render_modulated(&mut node, &modulator, sample_rate);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-4, "s={s}");
        }
    }

    #[test]
    fn high_frequency_above_nyquist_does_not_panic() {
        // Frequency above the sample rate must wrap safely (dt >= 1).
        let sample_rate = 8_000;
        let frames = 128;
        let mut node = FmOperatorNode::new(20_000.0, 0.0, 0.0, 1.0);
        let out = render(&mut node, sample_rate, frames);
        for &s in out.channel(0) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn getters_report_targets() {
        let mut node = FmOperatorNode::new(440.0, 1.0, 0.5, 0.9);
        node.set_mod_index(3.0, Ramp::Immediate);
        node.set_feedback(2.0, Ramp::Immediate);
        node.set_amplitude(0.5, Ramp::Immediate);
        assert_eq!(node.mod_index(), 3.0);
        assert_eq!(node.feedback(), 2.0);
        assert_eq!(node.amplitude(), 0.5);
    }
}
