//! Chaotic (logistic-map) oscillator source node.
//!
//! [`ChaoticOscillatorNode`] drives a pitched oscillator from the orbit of the
//! logistic map `x -> r * x * (1 - x)`. A normalized phase advances at the
//! `frequency` control; each time the phase completes one cycle the map is
//! iterated once, and the output linearly interpolates between the previous and
//! current iterate across the cycle. The `chaos` control selects the map rate
//! `r`, morphing the timbre from a steady period-two subharmonic tone toward
//! broadband, noise-like deterministic chaos.
//!
//! # Model
//!
//! A normalized phase `p` in `[0, 1)` advances by `frequency / sample_rate`
//! each sample. The map holds the current iterate `x` and the previous iterate
//! `x_prev`, both kept inside `[eps, 1 - eps]`. The bare output is the bipolar,
//! phase-interpolated iterate:
//!
//! ```text
//!   interp = x_prev + p * (x - x_prev)      (convex, so interp in [eps, 1-eps])
//!   out    = amplitude * (2 * interp - 1)
//! ```
//!
//! When the phase wraps past 1 the map advances one step:
//!
//! ```text
//!   r      = rate_min + chaos * (rate_max - rate_min)
//!   x_prev = x
//!   x      = clamp(r * x * (1 - x), eps, 1 - eps)
//! ```
//!
//! At a wrap `interp(p -> 1) = x`, and immediately afterwards `x_prev = x`, so
//! `interp(p = 0) = x`: the piecewise-linear output is continuous across every
//! cycle boundary and is therefore click-free by construction. Because
//! `interp` is a convex combination of two values in `[eps, 1 - eps]`, the bare
//! waveform stays strictly inside `[-1, 1]` and only `amplitude` scales it, so
//! no normalization is required.
//!
//! For `rate_min` in the period-two regime the orbit settles into a two-value
//! cycle, so the output repeats every two periods and reads as a steady tone an
//! octave below `frequency`. As `chaos` approaches one the rate reaches `4`,
//! the orbit becomes fully chaotic, and the spectrum spreads into broadband
//! noise whose coarse pitch still tracks `frequency`.
//!
//! # Relationship
//!
//! Unlike [`super::noise::NoiseNode`], which fills every sample with a
//! stochastic value drawn from a pseudo-random generator and has no pitch, this
//! node is fully deterministic and pitched: its roughness comes from
//! deterministic chaos, not randomness. Unlike the fixed periodic waveshape of
//! [`super::oscillator::OscillatorNode`], its per-cycle shape evolves, and the
//! `chaos` control sweeps continuously from a periodic subharmonic tone to a
//! broadband texture.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`ChaoticOscillatorNode::process`] performs no allocation, no locking, and
//! no panicking: it is a pure per-sample state machine. `chaos` and `amplitude`
//! are driven through [`Smoothed`] values, and `frequency` feeds a continuous
//! phase accumulator, so automation never produces zipper clicks. Two nodes
//! built with the same parameters produce bit-identical output, and
//! [`ChaoticOscillatorNode::reset`] restarts the exact same orbit.
//!
//! # Provenance
//!
//! Implemented from first principles from the classic public-domain chaotic /
//! iterated-map ("non-standard") synthesis idea: drive an oscillator from the
//! orbit of a nonlinear recurrence, interpolating successive iterates. The
//! logistic map `x -> r * x * (1 - x)` is a long-standing public-domain result
//! from nonlinear dynamics. This file contains no code, data, or derivative of
//! Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance
//! Audio, the Web Audio API, the Synthesis Toolkit, or any other audio engine
//! or toolkit; only the shared mathematical ideas are used. There is no AI or
//! machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Minimum oscillator frequency in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 1.0;

/// Default oscillator frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 110.0;

/// Maximum oscillator frequency in hertz.
pub const MAX_FREQUENCY_HZ: Sample = 4_000.0;

/// Minimum chaos amount (steady period-two subharmonic tone).
pub const MIN_CHAOS: Sample = 0.0;

/// Default chaos amount (fully chaotic broadband texture).
pub const DEFAULT_CHAOS: Sample = 1.0;

/// Maximum chaos amount (fully chaotic broadband texture).
pub const MAX_CHAOS: Sample = 1.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Lowest logistic-map rate, mapped from `chaos = 0` (solid period-two orbit).
const LOGISTIC_RATE_MIN: Sample = 3.2;

/// Highest logistic-map rate, mapped from `chaos = 1` (fully chaotic orbit).
const LOGISTIC_RATE_MAX: Sample = 4.0;

/// Deterministic initial map iterate (a generic interior point).
const INITIAL_STATE: Sample = 0.3;

/// Keeps the orbit away from the degenerate fixed points at `0` and `1`.
const STATE_EPSILON: Sample = 1.0e-6;

/// Fraction of the sample rate the per-sample phase increment is capped to.
const NYQUIST_GUARD: Sample = 0.49;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Construction parameters for a [`ChaoticOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ChaoticOscillatorParams {
    /// Oscillator frequency in hertz. Clamped to `[MIN, MAX]`.
    pub frequency_hz: Sample,
    /// Chaos amount in `[0, 1]` selecting the logistic-map rate.
    pub chaos: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for ChaoticOscillatorParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            chaos: DEFAULT_CHAOS,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl ChaoticOscillatorParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            chaos: finite_or(self.chaos, DEFAULT_CHAOS).clamp(MIN_CHAOS, MAX_CHAOS),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A chaotic (logistic-map) oscillator source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::ChaoticOscillatorNode;
///
/// let mut node = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
/// assert_eq!(node.frequency_hz(), 110.0);
/// assert_eq!(node.chaos(), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct ChaoticOscillatorNode {
    /// Oscillator frequency in hertz (feeds a continuous phase accumulator).
    frequency_hz: Sample,
    /// Smoothed chaos amount in `[0, 1]`.
    chaos: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized phase accumulator in `[0, 1)`.
    phase: Sample,
    /// Current logistic-map iterate, kept in `[eps, 1 - eps]`.
    state: Sample,
    /// Previous logistic-map iterate, kept in `[eps, 1 - eps]`.
    prev_state: Sample,
}

impl ChaoticOscillatorNode {
    /// Creates a chaotic oscillator at `frequency_hz` with the given `chaos`
    /// amount and linear `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `frequency_hz` is clamped to
    /// `[MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ]` and `chaos` to `[MIN_CHAOS,
    /// MAX_CHAOS]`.
    #[must_use]
    pub fn new(frequency_hz: Sample, chaos: Sample, amplitude: Sample) -> Self {
        Self {
            frequency_hz: finite_or(frequency_hz, DEFAULT_FREQUENCY_HZ)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            chaos: Smoothed::new(finite_or(chaos, DEFAULT_CHAOS).clamp(MIN_CHAOS, MAX_CHAOS)),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            phase: 0.0,
            state: INITIAL_STATE,
            prev_state: INITIAL_STATE,
        }
    }

    /// Builds a chaotic oscillator from a [`ChaoticOscillatorParams`] bundle.
    #[must_use]
    pub fn from_params(params: ChaoticOscillatorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.frequency_hz, p.chaos, p.amplitude)
    }

    /// Sets a new oscillator frequency in hertz (applied immediately; the
    /// continuous phase accumulator keeps the change click-free).
    #[inline]
    pub fn set_frequency_hz(&mut self, hz: Sample) {
        self.frequency_hz =
            finite_or(hz, self.frequency_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
    }

    /// Sets a new target chaos amount in `[0, 1]`, gliding with `ramp`.
    #[inline]
    pub fn set_chaos(&mut self, chaos: Sample, ramp: Ramp) {
        self.chaos.set_target(
            finite_or(chaos, self.chaos.target()).clamp(MIN_CHAOS, MAX_CHAOS),
            ramp,
        );
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the oscillator frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the target chaos amount the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn chaos(&self) -> Sample {
        self.chaos.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample, advancing the phase, the logistic map, and
    /// the smoothed controls. `inv_sr` is the reciprocal of the sample rate.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample) -> Sample {
        let chaos = self.chaos.next_sample();
        let amp = self.amplitude.next_sample();

        // Interpolate the current cycle before advancing; `phase` is the
        // fraction through the cycle so the output is piecewise-linear and
        // continuous across wraps.
        let interp = self.prev_state + self.phase * (self.state - self.prev_state);
        let out = (2.0 * interp - 1.0) * amp;

        let inc = (self.frequency_hz * inv_sr).clamp(0.0, NYQUIST_GUARD);
        self.phase += inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
            let rate = LOGISTIC_RATE_MIN + chaos * (LOGISTIC_RATE_MAX - LOGISTIC_RATE_MIN);
            self.prev_state = self.state;
            self.state =
                (rate * self.state * (1.0 - self.state)).clamp(STATE_EPSILON, 1.0 - STATE_EPSILON);
        }

        out
    }
}

impl AudioNode for ChaoticOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let inv_sr = 1.0 / sr;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample(inv_sr);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.state = INITIAL_STATE;
        self.prev_state = INITIAL_STATE;
        self.chaos = Smoothed::new(self.chaos.target());
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
    use bevy_math::ops;
    use core::f32::consts::TAU;

    const SR: u32 = 48_000;

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut ChaoticOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut ChaoticOscillatorNode,
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

    fn goertzel(buf: &AudioBuffer, sample_rate: u32, freq: Sample) -> Sample {
        let w = TAU * freq / sample_rate as Sample;
        let coeff = 2.0 * ops::cos(w);
        let mut s_prev = 0.0;
        let mut s_prev2 = 0.0;
        for &x in buf.channel(0) {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    }

    fn zero_crossings(buf: &AudioBuffer) -> usize {
        buf.channel(0)
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count()
    }

    fn max_adjacent_diff(buf: &AudioBuffer) -> Sample {
        buf.channel(0)
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, Sample::max)
    }

    #[test]
    fn renders_bounded_finite() {
        for &freq in &[1.0, 55.0, 440.0, 4_000.0] {
            for &chaos in &[0.0, 0.5, 1.0] {
                let mut node = ChaoticOscillatorNode::new(freq, chaos, 0.9);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "freq={freq} chaos={chaos} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn bare_waveform_stays_inside_unit() {
        let mut node = ChaoticOscillatorNode::new(220.0, 1.0, 1.0);
        let out = render(&mut node, SR, 16_384);
        for &s in out.channel(0) {
            assert!(s.abs() <= 1.0, "s={s}");
        }
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = ChaoticOscillatorNode::new(110.0, 1.0, 0.0);
        let out = render(&mut node, SR, 4_096);
        assert_eq!(energy(&out), 0.0);
    }

    #[test]
    fn not_silent() {
        let mut node = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let out = render(&mut node, SR, 8_192);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = ChaoticOscillatorNode::new(130.81, 0.7, 0.8);
        let mut b = ChaoticOscillatorNode::new(130.81, 0.7, 0.8);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = ChaoticOscillatorNode::new(196.0, 1.0, 0.8);
        let first = render(&mut node, SR, 4_096).channel(0).to_vec();
        node.reset();
        let second = render(&mut node, SR, 4_096).channel(0).to_vec();
        assert_eq!(first, second);
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = ChaoticOscillatorNode::new(110.0, 1.0, 0.25);
        let mut loud = ChaoticOscillatorNode::new(110.0, 1.0, 0.5);
        let e_quiet = energy(&render(&mut quiet, SR, 8_192));
        let e_loud = energy(&render(&mut loud, SR, 8_192));
        let ratio = e_loud / e_quiet;
        assert!((ratio - 4.0).abs() < 1e-2, "ratio={ratio}");
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let mut stereo = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let mut quad = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let m = render_layout(&mut mono, SR, 2_048, ChannelLayout::Mono);
        let s = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);
        let q = render_layout(&mut quad, SR, 2_048, ChannelLayout::Quad);
        assert_eq!(m.channel(0), s.channel(0));
        assert_eq!(s.channel(0), s.channel(1));
        assert_eq!(m.channel(0), q.channel(0));
        assert_eq!(q.channel(0), q.channel(3));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut idle = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        {
            let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
            out.set_active_frames(0);
            let inputs: [AudioBuffer; 0] = [];
            let mut outputs = [out];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            idle.process(&ctx(SR, 0), &mut io);
        }
        let after_idle = render(&mut idle, SR, 1_024).channel(0).to_vec();

        let mut fresh = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let fresh_out = render(&mut fresh, SR, 1_024).channel(0).to_vec();
        assert_eq!(after_idle, fresh_out);
    }

    #[test]
    fn latency_is_zero() {
        let node = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let node = ChaoticOscillatorNode::new(220.0, 0.6, 0.5);
        assert_eq!(node.frequency_hz(), 220.0);
        assert_eq!(node.chaos(), 0.6);
        assert_eq!(node.amplitude(), 0.5);
    }

    #[test]
    fn default_params_in_domain() {
        let p = ChaoticOscillatorParams::default();
        assert!(p.frequency_hz >= MIN_FREQUENCY_HZ && p.frequency_hz <= MAX_FREQUENCY_HZ);
        assert!(p.chaos >= MIN_CHAOS && p.chaos <= MAX_CHAOS);
        assert!(p.amplitude.is_finite());
    }

    #[test]
    fn from_params_matches_new() {
        let p = ChaoticOscillatorParams {
            frequency_hz: 164.81,
            chaos: 0.8,
            amplitude: 0.7,
        };
        let mut a = ChaoticOscillatorNode::from_params(p);
        let mut b = ChaoticOscillatorNode::new(164.81, 0.8, 0.7);
        assert_eq!(
            render(&mut a, SR, 2_048).channel(0),
            render(&mut b, SR, 2_048).channel(0)
        );
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = ChaoticOscillatorNode::new(50_000.0, 5.0, 0.8);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);
        assert_eq!(node.chaos(), MAX_CHAOS);

        let low = ChaoticOscillatorNode::new(0.001, -1.0, 0.8);
        assert_eq!(low.frequency_hz(), MIN_FREQUENCY_HZ);
        assert_eq!(low.chaos(), MIN_CHAOS);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = ChaoticOscillatorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.frequency_hz(), DEFAULT_FREQUENCY_HZ);
        assert_eq!(node.chaos(), DEFAULT_CHAOS);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = ChaoticOscillatorNode::new(110.0, 0.5, 0.8);

        node.set_frequency_hz(Sample::NAN);
        assert_eq!(node.frequency_hz(), 110.0);
        node.set_frequency_hz(99_999.0);
        assert_eq!(node.frequency_hz(), MAX_FREQUENCY_HZ);

        node.set_chaos(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.chaos(), 0.5);
        node.set_chaos(2.0, Ramp::Immediate);
        assert_eq!(node.chaos(), MAX_CHAOS);

        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
    }

    #[test]
    fn higher_frequency_raises_zero_crossing_rate() {
        let mut low = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let mut high = ChaoticOscillatorNode::new(440.0, 1.0, 0.8);
        let zc_low = zero_crossings(&render(&mut low, SR, 16_384));
        let zc_high = zero_crossings(&render(&mut high, SR, 16_384));
        assert!(zc_high > zc_low * 2, "zc_low={zc_low} zc_high={zc_high}");
    }

    #[test]
    fn higher_chaos_increases_broadband_content() {
        let mut tonal = ChaoticOscillatorNode::new(110.0, 0.0, 0.8);
        let mut noisy = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let zc_tonal = zero_crossings(&render(&mut tonal, SR, 16_384));
        let zc_noisy = zero_crossings(&render(&mut noisy, SR, 16_384));
        assert!(
            zc_noisy > zc_tonal,
            "zc_tonal={zc_tonal} zc_noisy={zc_noisy}"
        );
    }

    #[test]
    fn low_chaos_is_tonal_at_suboctave() {
        // At chaos = 0 the orbit settles into a period-two cycle, so the output
        // repeats every two periods and carries strong energy an octave below
        // `frequency`.
        let mut node = ChaoticOscillatorNode::new(220.0, 0.0, 0.8);
        let out = render(&mut node, SR, 16_384);
        let sub = goertzel(&out, SR, 110.0);
        let off = goertzel(&out, SR, 110.0 * 1.5);
        assert!(sub > off * 4.0, "sub={sub} off={off}");
    }

    #[test]
    fn chaos_change_takes_effect() {
        let mut node = ChaoticOscillatorNode::new(110.0, 0.0, 0.8);
        let before = zero_crossings(&render(&mut node, SR, 16_384));
        node.set_chaos(1.0, Ramp::Immediate);
        let after = zero_crossings(&render(&mut node, SR, 16_384));
        assert!(after > before, "before={before} after={after}");
    }

    #[test]
    fn frequency_change_is_click_free() {
        let mut node = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let _ = render(&mut node, SR, 4_096);
        node.set_frequency_hz(330.0);
        let out = render(&mut node, SR, 4_096);
        assert!(max_adjacent_diff(&out) < 0.1, "{}", max_adjacent_diff(&out));
    }

    #[test]
    fn chaos_change_is_click_free() {
        let mut node = ChaoticOscillatorNode::new(110.0, 0.0, 0.8);
        let _ = render(&mut node, SR, 4_096);
        node.set_chaos(1.0, Ramp::Immediate);
        let out = render(&mut node, SR, 4_096);
        assert!(max_adjacent_diff(&out) < 0.1, "{}", max_adjacent_diff(&out));
    }

    #[test]
    fn waveform_is_continuous() {
        // Piecewise-linear interpolation bounds the per-sample step by the
        // phase increment, so adjacent samples never jump.
        let mut node = ChaoticOscillatorNode::new(110.0, 1.0, 1.0);
        let out = render(&mut node, SR, 8_192);
        assert!(max_adjacent_diff(&out) < 0.05, "{}", max_adjacent_diff(&out));
    }

    #[test]
    fn reset_restores_phase_and_state() {
        let mut node = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let _ = render(&mut node, SR, 3_000);
        node.reset();
        let after_reset = render(&mut node, SR, 2_048).channel(0).to_vec();

        let mut fresh = ChaoticOscillatorNode::new(110.0, 1.0, 0.8);
        let fresh_out = render(&mut fresh, SR, 2_048).channel(0).to_vec();
        assert_eq!(after_reset, fresh_out);
    }

    #[test]
    fn extreme_frequency_stays_bounded() {
        // A tiny sample rate forces the Nyquist guard to cap the increment.
        let mut node = ChaoticOscillatorNode::new(4_000.0, 1.0, 0.9);
        let out = render(&mut node, 8_000, 4_096);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.0 + 1e-3, "s={s}");
        }
    }

    #[test]
    fn sanitised_leaves_valid_params_unchanged() {
        let p = ChaoticOscillatorParams {
            frequency_hz: 261.63,
            chaos: 0.42,
            amplitude: 0.6,
        };
        let q = p.sanitised();
        assert_eq!(p.frequency_hz, q.frequency_hz);
        assert_eq!(p.chaos, q.chaos);
        assert_eq!(p.amplitude, q.amplitude);
    }
}
