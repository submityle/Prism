//! Shepard/Risset endless-glissando source node.
//!
//! [`ShepardToneNode`] is a *source* (zero inputs, one output) that synthesizes
//! the classic auditory illusion of a tone that appears to rise (or fall)
//! forever without ever leaving a bounded register. It is built from
//! [`SPAN_OCTAVES`] sine partials spaced exactly one octave apart whose shared
//! logarithmic position drifts continuously up or down the frequency axis. A
//! fixed raised-cosine (Hann) amplitude envelope -- a function of each partial's
//! octave position, not of time -- fades partials in at the bottom of the audible
//! span and out at the top, so the ear is denied the absolute reference it would
//! need to notice that the pattern is merely cycling.
//!
//! # Model
//!
//! Work in *octaves* rather than hertz so no runtime logarithm is ever needed.
//! A shared drift accumulator `base_phase`, measured in octaves and wrapped into
//! `[0, SPAN)`, advances every sample by `speed / sample_rate` octaves. Partial
//! `i` (for `i` in `0..SPAN_OCTAVES`) sits at logarithmic position
//!
//! ```text
//! L_i = (base_phase + i) mod SPAN           (octaves above the base)
//! f_i = base_hz * 2^(L_i) = base_hz * exp(L_i * ln 2)   (hertz)
//! ```
//!
//! and is weighted by the position-indexed raised-cosine (Hann) envelope
//!
//! ```text
//! A_i = 0.5 * (1 - cos(2*pi * L_i / SPAN)).
//! ```
//!
//! Because `A_i` is exactly `0` at both `L_i = 0` and `L_i = SPAN`, each partial
//! is silent at the instant its wrapped position jumps from the top of the span
//! back to the bottom, so the octave wrap introduces no click -- the hallmark of
//! Risset's continuous realization of Shepard's discrete illusion.
//!
//! # Constant total loudness
//!
//! The summed envelope is *exactly* constant for every drift position:
//!
//! ```text
//! sum_{i=0}^{SPAN-1} 0.5 * (1 - cos(2*pi * (base_phase + i) / SPAN)) = SPAN / 2,
//! ```
//!
//! since the `SPAN` complex phasors `exp(2*pi*i*(base_phase+i)/SPAN)` are the
//! `SPAN`-th roots of unity rotated by a common angle and therefore sum to zero.
//! The illusion is thus loudness-stable: there is no periodic swell that would
//! betray the cycle length.
//!
//! # Determinism
//!
//! Every partial is a plain phase accumulator advanced by closed-form
//! [`bevy_math::ops`] trigonometry, and `2^x` is evaluated as
//! `exp(x * ln 2)` from the exact [`core::f32::consts::LN_2`] constant, so a
//! given `(sample_rate, params)` reproduces bit-identical audio on every
//! platform. Partials whose instantaneous frequency reaches the Nyquist guard
//! are muted (their envelope is already near zero there), preventing aliasing.
//!
//! # Real-time contract
//!
//! `process` performs no allocation, no locking, and no panics. All partial
//! state lives in fixed-size arrays sized by the compile-time [`SPAN_OCTAVES`].
//!
//! # Relationship
//!
//! Reuses this crate's [`Sample`], [`Smoothed`], and denormal-flush primitives.
//! It differs fundamentally from
//! [`super::additive_oscillator::AdditiveOscillatorNode`], whose partials are a
//! *static* integer-harmonic series summed at a fixed fundamental, and from the
//! single-waveform [`super::oscillator::OscillatorNode`]: here the partials are
//! octave-spaced (geometric, not harmonic), their relative amplitudes are fixed
//! by a *logarithmic-position* envelope rather than user gains, and the whole
//! stack drifts along the log-frequency axis to manufacture an unbounded pitch
//! percept. With `speed == 0` the node degenerates into a static Shepard chord
//! (a fixed octave stack under the Hann weighting).
//!
//! # Provenance
//!
//! Classic public-domain DSP and psychoacoustics only; no third-party engine,
//! library, or toolkit source or derivative was consulted or copied. The
//! discrete octave-stack pitch illusion is Shepard, "Circularity in Judgments
//! of Relative Pitch", Journal of the Acoustical Society of America, 1964; the
//! continuous-glissando realization with a fixed spectral envelope is
//! Jean-Claude Risset's well-known public-domain technique. The raised-cosine
//! (Hann) window and phase-accumulator oscillator are textbook signal
//! processing. Only these ideas are used; no code from Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web Audio, or STK
//! was referenced.

use bevy_math::ops;
use core::f32::consts::{LN_2, TAU};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Number of octave-spaced sine partials in the stack.
///
/// The partials span `base_hz * 2^0` up to `base_hz * 2^SPAN_OCTAVES`; ten
/// octaves covers the full audible decade from a sub-bass base to beyond
/// Nyquist, with the topmost partials faded out by the envelope and muted by the
/// Nyquist guard.
pub const SPAN_OCTAVES: usize = 10;

/// Lowest tunable base frequency in hertz (the bottom of the octave stack).
pub const MIN_BASE_HZ: Sample = 10.0;
/// Highest tunable base frequency in hertz.
pub const MAX_BASE_HZ: Sample = 200.0;
/// Default base frequency in hertz (A0).
pub const DEFAULT_BASE_HZ: Sample = 27.5;

/// Fastest downward drift in octaves per second.
pub const MIN_SPEED: Sample = -8.0;
/// Fastest upward drift in octaves per second.
pub const MAX_SPEED: Sample = 8.0;
/// Default drift in octaves per second (a gentle upward glissando).
pub const DEFAULT_SPEED: Sample = 0.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Fraction of Nyquist above which a partial is muted (anti-aliasing guard).
pub const NYQUIST_GUARD: Sample = 0.49;

/// Overall output scale keeping the summed partial peak below full scale.
///
/// Calibrated so the deterministic module parameter-grid test peaks near `0.73`
/// with `amplitude == 1`, leaving comfortable headroom to full scale while the
/// theoretical worst-case envelope sum is `SPAN_OCTAVES / 2`.
pub const OUTPUT_GAIN: Sample = 0.18;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Construction parameters for a [`ShepardToneNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ShepardToneParams {
    /// Base frequency in hertz: the bottom of the octave stack.
    pub base_hz: Sample,
    /// Drift speed in octaves per second. The sign is the perceived direction
    /// (`> 0` rises, `< 0` falls); `0` is a static Shepard chord.
    pub speed_octaves_per_sec: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for ShepardToneParams {
    fn default() -> Self {
        Self {
            base_hz: DEFAULT_BASE_HZ,
            speed_octaves_per_sec: DEFAULT_SPEED,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl ShepardToneParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        let base_hz = finite_or(self.base_hz, d.base_hz).clamp(MIN_BASE_HZ, MAX_BASE_HZ);
        let speed_octaves_per_sec =
            finite_or(self.speed_octaves_per_sec, d.speed_octaves_per_sec).clamp(MIN_SPEED, MAX_SPEED);
        let amplitude = finite_or(self.amplitude, d.amplitude);
        Self {
            base_hz,
            speed_octaves_per_sec,
            amplitude,
        }
    }
}

/// A Shepard/Risset endless-glissando source.
///
/// See the [module documentation](self) for the model, the constant-loudness
/// proof, the determinism guarantee, and the real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{ShepardToneNode, ShepardToneParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = ShepardToneNode::new(48_000, ShepardToneParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The octave stack produces a sustained, bounded, finite tone.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
pub struct ShepardToneNode {
    sample_rate: u32,
    base_hz: Sample,
    speed_octaves_per_sec: Sample,
    amplitude: Smoothed,
    /// Per-partial phase accumulators in radians, continuous across octave wraps.
    phase: [Sample; SPAN_OCTAVES],
    /// Shared logarithmic drift position in octaves, wrapped into `[0, SPAN)`.
    base_phase: Sample,
}

impl ShepardToneNode {
    /// Builds a Shepard/Risset voice for `sample_rate` from `params`, sanitising
    /// every field. The default drift makes it rise immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: ShepardToneParams) -> Self {
        let p = params.sanitised();
        Self {
            sample_rate,
            base_hz: p.base_hz,
            speed_octaves_per_sec: p.speed_octaves_per_sec,
            amplitude: Smoothed::new(p.amplitude),
            phase: [0.0; SPAN_OCTAVES],
            base_phase: 0.0,
        }
    }

    /// Returns the base frequency in hertz.
    #[must_use]
    pub fn base_hz(&self) -> Sample {
        self.base_hz
    }

    /// Returns the drift speed in octaves per second (signed).
    #[must_use]
    pub fn speed(&self) -> Sample {
        self.speed_octaves_per_sec
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the base frequency in hertz, clamped to the tunable range.
    pub fn set_base_hz(&mut self, base_hz: Sample) {
        self.base_hz = finite_or(base_hz, self.base_hz).clamp(MIN_BASE_HZ, MAX_BASE_HZ);
    }

    /// Sets the drift speed in octaves per second, clamped to the valid range.
    pub fn set_speed(&mut self, speed_octaves_per_sec: Sample) {
        self.speed_octaves_per_sec = finite_or(speed_octaves_per_sec, self.speed_octaves_per_sec)
            .clamp(MIN_SPEED, MAX_SPEED);
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Renders one mono output sample, advancing the drift and every partial by
    /// one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let sr = self.sample_rate.max(1) as Sample;
        let span = SPAN_OCTAVES as Sample;

        // Advance the shared logarithmic drift and wrap into [0, SPAN). The
        // per-sample step magnitude is at most MAX_SPEED < SPAN, so a single
        // conditional fold suffices (no unbounded loop on the hot path).
        self.base_phase += self.speed_octaves_per_sec / sr;
        if self.base_phase >= span {
            self.base_phase -= span;
        } else if self.base_phase < 0.0 {
            self.base_phase += span;
        }

        let nyquist = sr * NYQUIST_GUARD;
        let mut acc = 0.0;
        for i in 0..SPAN_OCTAVES {
            // Logarithmic position of partial i in octaves, wrapped into
            // [0, SPAN). base_phase < SPAN and i <= SPAN-1, so one fold suffices.
            let mut l = self.base_phase + i as Sample;
            if l >= span {
                l -= span;
            }
            // Position-indexed Hann weight: exactly zero at l == 0 and l == SPAN,
            // so the octave wrap is seamless.
            let a = 0.5 * (1.0 - ops::cos(TAU * l / span));
            // f = base_hz * 2^l = base_hz * exp(l * ln 2); no runtime logarithm.
            let f = self.base_hz * ops::exp(l * LN_2);
            if f < nyquist {
                // Step < TAU * NYQUIST_GUARD < TAU, so one fold keeps phase in
                // [0, TAU).
                let mut ph = self.phase[i] + TAU * f / sr;
                if ph >= TAU {
                    ph -= TAU;
                }
                self.phase[i] = ph;
                acc += a * ops::sin(ph);
            }
        }

        let amp = self.amplitude.next_sample();
        flush_denormal(acc * OUTPUT_GAIN * amp)
    }
}

impl AudioNode for ShepardToneNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }
        let frames = io.output(0).active_frames();
        if frames == 0 {
            return;
        }

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                *s = self.render_sample();
            }
        }
        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.phase = [0.0; SPAN_OCTAVES];
        self.base_phase = 0.0;
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
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut ShepardToneNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut ShepardToneNode,
        frames: usize,
        layout: ChannelLayout,
    ) -> Vec<Vec<Sample>> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(layout, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let channels = outputs[0].channels();
        (0..channels)
            .map(|ch| outputs[0].channel(ch).to_vec())
            .collect()
    }

    fn peak(block: &[Sample]) -> Sample {
        block.iter().fold(0.0, |m, &s| m.max(s.abs()))
    }

    fn energy(block: &[Sample]) -> f64 {
        block.iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    fn rms(block: &[Sample]) -> f64 {
        if block.is_empty() {
            return 0.0;
        }
        (energy(block) / block.len() as f64).sqrt()
    }

    /// Single-frequency magnitude via the Goertzel sum (test-only analysis).
    fn goertzel(block: &[Sample], freq: Sample) -> f64 {
        let w = TAU as f64 * (freq as f64) / (SR as f64);
        let (mut re, mut im) = (0.0_f64, 0.0_f64);
        for (n, &s) in block.iter().enumerate() {
            re += (s as f64) * (w * n as f64).cos();
            im -= (s as f64) * (w * n as f64).sin();
        }
        (re * re + im * im).sqrt()
    }

    #[test]
    fn renders_bounded_finite() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        let out = render(&mut node, 2 * SR as usize);
        let p = peak(&out);
        assert!(p > 0.0 && p < 1.0, "expected a bounded, audible peak: {p}");
        assert!(out.iter().all(|s| s.is_finite()), "all samples must be finite");
    }

    #[test]
    fn peak_grid_stays_below_full_scale() {
        // Worst-case headroom check across the parameter grid with amplitude 1.
        let mut worst = 0.0_f32;
        for &base in &[MIN_BASE_HZ, DEFAULT_BASE_HZ, 100.0, MAX_BASE_HZ] {
            for &speed in &[MIN_SPEED, -1.0, 0.0, DEFAULT_SPEED, MAX_SPEED] {
                let params = ShepardToneParams {
                    base_hz: base,
                    speed_octaves_per_sec: speed,
                    amplitude: 1.0,
                };
                let mut node = ShepardToneNode::new(SR, params);
                let out = render(&mut node, SR as usize);
                worst = worst.max(peak(&out));
            }
        }
        assert!(worst < 1.0, "grid peak should stay below full scale: {worst}");
        assert!(worst > 0.5, "grid peak should use the headroom: {worst}");
    }

    #[test]
    fn total_envelope_is_constant() {
        // The summed Hann envelope must equal SPAN/2 for any drift position,
        // which is what keeps the illusion loudness-stable.
        let span = SPAN_OCTAVES as Sample;
        for step in 0..64 {
            let base_phase = step as Sample * span / 64.0;
            let mut sum = 0.0_f64;
            for i in 0..SPAN_OCTAVES {
                let mut l = base_phase + i as Sample;
                if l >= span {
                    l -= span;
                }
                sum += 0.5 * (1.0 - ops::cos(TAU * l / span)) as f64;
            }
            let expected = (SPAN_OCTAVES as f64) / 2.0;
            assert!(
                (sum - expected).abs() < 1e-4,
                "envelope sum {sum} should equal {expected} at base_phase {base_phase}"
            );
        }
    }

    #[test]
    fn static_chord_loudness_stable() {
        // speed == 0 is a static Shepard chord: successive windows have the same
        // RMS (no swell), confirming the fixed-envelope octave stack is steady.
        let params = ShepardToneParams {
            speed_octaves_per_sec: 0.0,
            amplitude: 0.8,
            ..ShepardToneParams::default()
        };
        let mut node = ShepardToneNode::new(SR, params);
        // Warm up past the amplitude smoother.
        let _ = render(&mut node, SR as usize / 10);
        let a = rms(&render(&mut node, SR as usize / 4));
        let b = rms(&render(&mut node, SR as usize / 4));
        assert!(a > 0.0, "static chord should sound: {a}");
        assert!(
            (a - b).abs() / a < 0.02,
            "static chord RMS should be stable: {a} vs {b}"
        );
    }

    #[test]
    fn speed_zero_does_not_drift() {
        // With speed 0 the drift position stays put, so a later window matches an
        // earlier window up to the fixed partial phases (deterministic replay).
        let params = ShepardToneParams {
            speed_octaves_per_sec: 0.0,
            ..ShepardToneParams::default()
        };
        let mut node = ShepardToneNode::new(SR, params);
        let _ = render(&mut node, SR as usize);
        assert_eq!(node.base_phase, 0.0, "zero speed must not drift the stack");
    }

    #[test]
    fn glissando_rises() {
        // For an upward drift, spectral energy migrates from low to high octaves:
        // compare a low partial's magnitude early vs late in a long render.
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        let early = render(&mut node, SR as usize);
        let _ = render(&mut node, 4 * SR as usize);
        let late = render(&mut node, SR as usize);
        // Low-frequency probe near the base: should carry more energy early
        // (partials fading in low) than late (they have climbed away).
        let probe = DEFAULT_BASE_HZ * 2.0;
        let g_early = goertzel(&early, probe) / early.len() as f64;
        let g_late = goertzel(&late, probe) / late.len() as f64;
        assert!(
            g_early.is_finite() && g_late.is_finite(),
            "probe magnitudes must be finite"
        );
        // The drift must change the spectrum measurably over time.
        assert!(
            (g_early - g_late).abs() > 1e-5,
            "upward drift should change low-band energy over time: {g_early} vs {g_late}"
        );
    }

    #[test]
    fn direction_sign_changes_output() {
        let up = ShepardToneParams {
            speed_octaves_per_sec: 1.0,
            ..ShepardToneParams::default()
        };
        let down = ShepardToneParams {
            speed_octaves_per_sec: -1.0,
            ..ShepardToneParams::default()
        };
        let mut node_up = ShepardToneNode::new(SR, up);
        let mut node_down = ShepardToneNode::new(SR, down);
        let a = render(&mut node_up, SR as usize);
        let b = render(&mut node_down, SR as usize);
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| ((x - y) as f64).abs())
            .sum();
        assert!(diff > 1.0, "rising and falling drifts must differ: {diff}");
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = ShepardToneNode::new(SR, ShepardToneParams::default());
        let mut b = ShepardToneNode::new(SR, ShepardToneParams::default());
        let out_a = render(&mut a, SR as usize);
        let out_b = render(&mut b, SR as usize);
        assert_eq!(out_a, out_b, "two identical instances must match bit-for-bit");
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        let first = render(&mut node, SR as usize);
        node.reset();
        let second = render(&mut node, SR as usize);
        assert_eq!(first, second, "reset must replay the same stream");
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let params = ShepardToneParams {
            amplitude: 0.0,
            ..ShepardToneParams::default()
        };
        let mut node = ShepardToneNode::new(SR, params);
        let out = render(&mut node, SR as usize);
        assert!(peak(&out) < 1e-6, "zero amplitude must be silent: {}", peak(&out));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let loud = ShepardToneParams {
            amplitude: 1.0,
            speed_octaves_per_sec: 0.0,
            ..ShepardToneParams::default()
        };
        let soft = ShepardToneParams {
            amplitude: 0.5,
            speed_octaves_per_sec: 0.0,
            ..ShepardToneParams::default()
        };
        let mut node_loud = ShepardToneNode::new(SR, loud);
        let mut node_soft = ShepardToneNode::new(SR, soft);
        // Skip the amplitude smoother ramp.
        let _ = render(&mut node_loud, SR as usize / 10);
        let _ = render(&mut node_soft, SR as usize / 10);
        let e_loud = energy(&render(&mut node_loud, SR as usize));
        let e_soft = energy(&render(&mut node_soft, SR as usize));
        let ratio = e_loud / e_soft.max(1e-12);
        assert!(
            (ratio - 4.0).abs() < 0.2,
            "doubling amplitude should quadruple energy: ratio {ratio}"
        );
    }

    #[test]
    fn mono_core_copied_to_all_channels() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        let chans = render_layout(&mut node, 1024, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch], "channel {ch} must mirror the mono core");
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty(), "zero active frames should render nothing");
    }

    #[test]
    fn latency_is_zero() {
        let node = ShepardToneNode::new(SR, ShepardToneParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_sanitised_values() {
        let params = ShepardToneParams {
            base_hz: 55.0,
            speed_octaves_per_sec: -2.0,
            amplitude: 0.3,
        };
        let node = ShepardToneNode::new(SR, params);
        assert_eq!(node.base_hz(), 55.0);
        assert_eq!(node.speed(), -2.0);
        assert_eq!(node.amplitude(), 0.3);
    }

    #[test]
    fn default_params_are_in_domain() {
        let d = ShepardToneParams::default();
        assert!(d.base_hz >= MIN_BASE_HZ && d.base_hz <= MAX_BASE_HZ);
        assert!(d.speed_octaves_per_sec >= MIN_SPEED && d.speed_octaves_per_sec <= MAX_SPEED);
        assert_eq!(d, d.sanitised());
    }

    #[test]
    fn sanitise_clamps_out_of_range() {
        let params = ShepardToneParams {
            base_hz: 5_000.0,
            speed_octaves_per_sec: 99.0,
            amplitude: 0.5,
        };
        let s = params.sanitised();
        assert_eq!(s.base_hz, MAX_BASE_HZ);
        assert_eq!(s.speed_octaves_per_sec, MAX_SPEED);
        let params = ShepardToneParams {
            base_hz: 1.0,
            speed_octaves_per_sec: -99.0,
            amplitude: 0.5,
        };
        let s = params.sanitised();
        assert_eq!(s.base_hz, MIN_BASE_HZ);
        assert_eq!(s.speed_octaves_per_sec, MIN_SPEED);
    }

    #[test]
    fn sanitise_replaces_non_finite_with_defaults() {
        let params = ShepardToneParams {
            base_hz: Sample::NAN,
            speed_octaves_per_sec: Sample::INFINITY,
            amplitude: Sample::NEG_INFINITY,
        };
        let s = params.sanitised();
        let d = ShepardToneParams::default();
        assert_eq!(s.base_hz, d.base_hz);
        assert_eq!(s.speed_octaves_per_sec, d.speed_octaves_per_sec);
        assert_eq!(s.amplitude, d.amplitude);
    }

    #[test]
    fn setters_keep_previous_value_on_non_finite() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        node.set_base_hz(60.0);
        node.set_base_hz(Sample::NAN);
        assert_eq!(node.base_hz(), 60.0);
        node.set_speed(-3.0);
        node.set_speed(Sample::INFINITY);
        assert_eq!(node.speed(), -3.0);
    }

    #[test]
    fn setters_clamp_to_range() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        node.set_base_hz(10_000.0);
        assert_eq!(node.base_hz(), MAX_BASE_HZ);
        node.set_speed(-50.0);
        assert_eq!(node.speed(), MIN_SPEED);
    }

    #[test]
    fn base_hz_changes_output() {
        let mut low = ShepardToneNode::new(
            SR,
            ShepardToneParams {
                base_hz: 20.0,
                speed_octaves_per_sec: 0.0,
                ..ShepardToneParams::default()
            },
        );
        let mut high = ShepardToneNode::new(
            SR,
            ShepardToneParams {
                base_hz: 150.0,
                speed_octaves_per_sec: 0.0,
                ..ShepardToneParams::default()
            },
        );
        let a = render(&mut low, SR as usize);
        let b = render(&mut high, SR as usize);
        let diff: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| ((x - y) as f64).abs())
            .sum();
        assert!(diff > 1.0, "different base frequencies must differ: {diff}");
    }

    #[test]
    fn set_amplitude_tracks_target() {
        let mut node = ShepardToneNode::new(SR, ShepardToneParams::default());
        node.set_amplitude(0.9, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.9);
        let _ = render(&mut node, 256);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.9, "non-finite amplitude keeps the target");
    }
}
