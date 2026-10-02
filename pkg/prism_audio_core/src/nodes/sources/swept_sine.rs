//! Swept-sine (chirp) measurement source.
//!
//! [`SweptSineNode`] is a *source* (zero inputs, one output) that plays a single
//! sine whose frequency glides from a start frequency to an end frequency over a
//! fixed duration, then falls silent. It is the excitation half of the classic
//! swept-sine system-identification measurement: play the sweep through a system
//! under test, record the response, and deconvolve against the known sweep to
//! recover the system's impulse response and (for the exponential sweep)
//! separate the harmonic-distortion orders.
//!
//! # Model
//!
//! The output at sample `n` (while the sweep is running) is
//! `amplitude * window[n] * sin(phase[n])`, where the instantaneous frequency
//! `f[n]` follows one of two [`SweepMode`] laws:
//!
//! * [`SweepMode::Exponential`] -- a *logarithmic* sweep that spends equal time
//!   per octave: `f[n] = f_start * k^n` with the per-sample ratio
//!   `k = (f_end / f_start)^(1 / (N - 1))` over `N` sweep samples. This is the
//!   exponential sine sweep (ESS) used for impulse-response measurement because
//!   its harmonic-distortion products deconvolve into separable pre-sweep
//!   echoes;
//! * [`SweepMode::Linear`] -- a constant frequency-slope chirp,
//!   `f[n] = f_start + (f_end - f_start) * n / (N - 1)`, which spends equal time
//!   per hertz (flatter high-frequency energy, the classic radar/sonar chirp).
//!
//! The phase is accumulated one sample at a time, `phase += 2*pi*f[n] / sr`
//! (wrapped to stay bounded), and the frequency is advanced by a single multiply
//! (`f *= k`, exponential) or add (`f += step`, linear) per sample. Driving the
//! sweep by this per-sample recurrence keeps the frequency progression exact
//! without evaluating a logarithm on the hot path. The ends are tapered by a
//! raised-cosine (Hann) half-window of `fade` samples so the sweep starts and
//! stops without the broadband click an abrupt sine edge would inject.
//!
//! Once `N` samples have played the node is *inactive* and emits silence until
//! [`SweptSineNode::trigger`] (or [`SweptSineNode::reset`]) restarts it from the
//! start frequency.
//!
//! # Determinism
//!
//! The sweep is a closed-form deterministic recurrence (no noise), so two nodes
//! built with the same sample rate and parameters produce bit-identical output,
//! and [`SweptSineNode::reset`] replays the identical sweep.
//!
//! # Real-time contract
//!
//! `process` performs no allocation, locking, or panic on the hot path. The
//! sweep geometry (`k`/`step`, length, fade) is recomputed only on the cold path
//! when a parameter changes. Non-finite parameters are sanitised on the way in,
//! frequencies are clamped below the Nyquist guard, and outputs are flushed of
//! denormals. Latency is zero.
//!
//! # Relationship
//!
//! Reuses this crate's [`Sample`], [`Smoothed`], and denormal flush. It pairs
//! directly with the [`super::super::reverb::convolver::Convolver`] to perform a
//! swept-sine impulse-response measurement (play the sweep into a system, then
//! deconvolve the recorded response). It is unlike the steady, perpetual tones
//! of [`super::oscillator`] and [`super::additive_oscillator`] because its
//! frequency is time-varying and it is one-shot; unlike [`super::shepard_tone`],
//! whose octave-spaced partials glide perpetually to fake an ever-rising pitch,
//! this is a single real glissando across the band that ends; and unlike a
//! [`super::noise`] excitation it is deterministic and phase-coherent, which is
//! what makes the deconvolution clean.
//!
//! # Provenance
//!
//! Classic public-domain DSP only; no third-party engine, library, or toolkit
//! source or derivative was consulted or copied. The exponential (logarithmic)
//! sine sweep for simultaneous impulse-response and distortion measurement is
//! the public technique of A. Farina, "Simultaneous Measurement of Impulse
//! Response and Distortion with a Swept-Sine Technique" (AES 108th Convention,
//! 2000). The linear chirp is the classic radar/sonar matched-filter excitation
//! (P. M. Woodward; Klauder et al.). Raised-cosine (Hann) edge tapering and
//! per-sample phase accumulation are standard published DSP. No code from Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, Web
//! Audio, or STK was referenced.

use bevy_math::ops;
use core::f32::consts::{PI, TAU};

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{flush_denormal, Sample};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable sweep endpoint in hertz.
pub const MIN_FREQUENCY_HZ: Sample = 1.0;
/// Highest tunable sweep endpoint in hertz (further bounded by the Nyquist guard).
pub const MAX_FREQUENCY_HZ: Sample = 24_000.0;
/// Default sweep start frequency in hertz.
pub const DEFAULT_START_HZ: Sample = 20.0;
/// Default sweep end frequency in hertz.
pub const DEFAULT_END_HZ: Sample = 20_000.0;

/// Shortest sweep duration in seconds.
pub const MIN_DURATION_S: Sample = 0.01;
/// Longest sweep duration in seconds.
pub const MAX_DURATION_S: Sample = 60.0;
/// Default sweep duration in seconds.
pub const DEFAULT_DURATION_S: Sample = 2.0;

/// Shortest edge-fade time in seconds.
pub const MIN_FADE_S: Sample = 0.0;
/// Longest edge-fade time in seconds.
pub const MAX_FADE_S: Sample = 1.0;
/// Default edge-fade time in seconds (a short anti-click taper).
pub const DEFAULT_FADE_S: Sample = 0.01;

/// Default linear output amplitude (also the sweep peak).
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Fraction of Nyquist above which a sweep endpoint is clamped (anti-aliasing).
pub const NYQUIST_GUARD: Sample = 0.49;

/// Returns `value` when finite, otherwise `fallback`.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Frequency-progression law of a [`SweptSineNode`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SweepMode {
    /// Logarithmic sweep (equal time per octave); the measurement ESS.
    #[default]
    Exponential,
    /// Constant frequency-slope chirp (equal time per hertz).
    Linear,
}

/// Construction parameters for a [`SweptSineNode`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SweptSineParams {
    /// Sweep start frequency in hertz.
    pub start_hz: Sample,
    /// Sweep end frequency in hertz.
    pub end_hz: Sample,
    /// Sweep duration in seconds.
    pub duration_s: Sample,
    /// Raised-cosine edge-fade time in seconds (anti-click taper at both ends).
    pub fade_s: Sample,
    /// Frequency-progression law.
    pub mode: SweepMode,
    /// Linear output amplitude (sweep peak).
    pub amplitude: Sample,
}

impl Default for SweptSineParams {
    fn default() -> Self {
        Self {
            start_hz: DEFAULT_START_HZ,
            end_hz: DEFAULT_END_HZ,
            duration_s: DEFAULT_DURATION_S,
            fade_s: DEFAULT_FADE_S,
            mode: SweepMode::Exponential,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl SweptSineParams {
    /// Replaces non-finite fields with defaults and clamps every field to its
    /// valid range.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let d = Self::default();
        Self {
            start_hz: finite_or(self.start_hz, d.start_hz)
                .clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            end_hz: finite_or(self.end_hz, d.end_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ),
            duration_s: finite_or(self.duration_s, d.duration_s)
                .clamp(MIN_DURATION_S, MAX_DURATION_S),
            fade_s: finite_or(self.fade_s, d.fade_s).clamp(MIN_FADE_S, MAX_FADE_S),
            mode: self.mode,
            amplitude: finite_or(self.amplitude, d.amplitude),
        }
    }
}

/// A one-shot swept-sine (chirp) measurement source.
///
/// See the [module documentation](self) for the sweep laws, the measurement
/// workflow, the determinism guarantee, and the real-time contract.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{SweptSineNode, SweptSineParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = SweptSineNode::new(48_000, SweptSineParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The sweep is audible, bounded, and finite.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak < 1.0 && peak.is_finite());
/// ```
pub struct SweptSineNode {
    sample_rate: u32,
    start_hz: Sample,
    end_hz: Sample,
    duration_s: Sample,
    fade_s: Sample,
    mode: SweepMode,
    amplitude: Smoothed,
    // Sweep geometry (cold-path derived).
    total_samples: u32,
    fade_samples: u32,
    start_clamped: Sample,
    freq_mult: Sample,
    freq_step: Sample,
    // Running sweep state.
    pos: u32,
    phase: Sample,
    freq: Sample,
}

impl SweptSineNode {
    /// Builds a swept-sine source for `sample_rate` from `params`, sanitising
    /// every field, and arms it so it sweeps immediately.
    #[must_use]
    pub fn new(sample_rate: u32, params: SweptSineParams) -> Self {
        let p = params.sanitised();
        let mut node = Self {
            sample_rate,
            start_hz: p.start_hz,
            end_hz: p.end_hz,
            duration_s: p.duration_s,
            fade_s: p.fade_s,
            mode: p.mode,
            amplitude: Smoothed::new(p.amplitude),
            total_samples: 1,
            fade_samples: 0,
            start_clamped: p.start_hz,
            freq_mult: 1.0,
            freq_step: 0.0,
            pos: 0,
            phase: 0.0,
            freq: p.start_hz,
        };
        node.recompute();
        node.trigger();
        node
    }

    /// Returns the sweep start frequency in hertz.
    #[must_use]
    pub fn start_hz(&self) -> Sample {
        self.start_hz
    }

    /// Returns the sweep end frequency in hertz.
    #[must_use]
    pub fn end_hz(&self) -> Sample {
        self.end_hz
    }

    /// Returns the sweep duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> Sample {
        self.duration_s
    }

    /// Returns the edge-fade time in seconds.
    #[must_use]
    pub fn fade_s(&self) -> Sample {
        self.fade_s
    }

    /// Returns the frequency-progression law.
    #[must_use]
    pub fn mode(&self) -> SweepMode {
        self.mode
    }

    /// Returns the target linear output amplitude.
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Returns `true` while the sweep is still playing (not yet exhausted).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.pos < self.total_samples
    }

    /// Sets the sweep start frequency, clamped, and recomputes the geometry.
    pub fn set_start_hz(&mut self, start_hz: Sample) {
        self.start_hz =
            finite_or(start_hz, self.start_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
        self.recompute();
    }

    /// Sets the sweep end frequency, clamped, and recomputes the geometry.
    pub fn set_end_hz(&mut self, end_hz: Sample) {
        self.end_hz = finite_or(end_hz, self.end_hz).clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
        self.recompute();
    }

    /// Sets the sweep duration, clamped, and recomputes the geometry.
    pub fn set_duration(&mut self, duration_s: Sample) {
        self.duration_s =
            finite_or(duration_s, self.duration_s).clamp(MIN_DURATION_S, MAX_DURATION_S);
        self.recompute();
    }

    /// Sets the edge-fade time, clamped, and recomputes the geometry.
    pub fn set_fade(&mut self, fade_s: Sample) {
        self.fade_s = finite_or(fade_s, self.fade_s).clamp(MIN_FADE_S, MAX_FADE_S);
        self.recompute();
    }

    /// Sets the frequency-progression law and recomputes the geometry.
    pub fn set_mode(&mut self, mode: SweepMode) {
        self.mode = mode;
        self.recompute();
    }

    /// Sets the target output amplitude, gliding over `ramp`.
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Restarts the sweep from the start frequency.
    pub fn trigger(&mut self) {
        self.pos = 0;
        self.phase = 0.0;
        self.freq = self.start_clamped;
    }

    /// Recomputes the sweep geometry (length, fade, and the per-sample
    /// frequency recurrence) from the current scalar parameters. Never runs on
    /// the audio hot path.
    fn recompute(&mut self) {
        let sr = self.sample_rate.max(1) as Sample;
        let nyquist = sr * NYQUIST_GUARD;
        let hi = MAX_FREQUENCY_HZ.min(nyquist).max(MIN_FREQUENCY_HZ);
        let start = self.start_hz.clamp(MIN_FREQUENCY_HZ, hi);
        let end = self.end_hz.clamp(MIN_FREQUENCY_HZ, hi);
        self.start_clamped = start;

        let total = ops::round(self.duration_s * sr) as i64;
        self.total_samples = total.clamp(1, i64::from(u32::MAX)) as u32;

        let max_fade = self.total_samples / 2;
        let fade = ops::round(self.fade_s * sr) as i64;
        self.fade_samples = fade.clamp(0, i64::from(max_fade)) as u32;

        let span = (self.total_samples.max(2) - 1) as Sample;
        // Exponential: geometric per-sample ratio; Linear: arithmetic step.
        self.freq_mult = ops::powf(end / start, 1.0 / span);
        self.freq_step = (end - start) / span;

        // Keep the running frequency inside the (possibly new) clamp window.
        self.freq = self.freq.clamp(MIN_FREQUENCY_HZ, hi);
    }

    /// Raised-cosine edge weight for sample index `n` of the sweep.
    #[inline]
    fn window(&self, n: u32) -> Sample {
        let fade = self.fade_samples;
        if fade == 0 {
            return 1.0;
        }
        let total = self.total_samples;
        if n < fade {
            let x = (n + 1) as Sample / (fade as Sample + 1.0);
            0.5 - 0.5 * ops::cos(PI * x)
        } else if n >= total - fade {
            let from_end = total - 1 - n;
            let x = (from_end + 1) as Sample / (fade as Sample + 1.0);
            0.5 - 0.5 * ops::cos(PI * x)
        } else {
            1.0
        }
    }

    /// Renders one mono output sample, advancing the sweep by one step.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let amp = self.amplitude.next_sample();
        if self.pos >= self.total_samples {
            return 0.0;
        }

        let window = self.window(self.pos);
        let out = amp * window * ops::sin(self.phase);

        // Advance the phase (wrapped) and the instantaneous frequency.
        let sr = self.sample_rate.max(1) as Sample;
        self.phase += TAU * self.freq / sr;
        if self.phase >= TAU {
            self.phase -= TAU;
        }
        match self.mode {
            SweepMode::Exponential => self.freq *= self.freq_mult,
            SweepMode::Linear => self.freq += self.freq_step,
        }
        self.pos += 1;

        flush_denormal(out)
    }
}

impl AudioNode for SweptSineNode {
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
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.recompute();
        self.trigger();
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
    fn render(node: &mut SweptSineNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout`.
    fn render_layout(
        node: &mut SweptSineNode,
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

    /// Estimates the dominant frequency of `block` from its zero-crossing rate.
    fn freq_from_crossings(block: &[Sample]) -> f64 {
        let mut crossings = 0usize;
        for pair in block.windows(2) {
            if (pair[0] <= 0.0 && pair[1] > 0.0) || (pair[0] >= 0.0 && pair[1] < 0.0) {
                crossings += 1;
            }
        }
        // Two zero crossings per cycle.
        (crossings as f64 / 2.0) * (SR as f64) / (block.len() as f64)
    }

    fn short(duration_s: Sample, mode: SweepMode) -> SweptSineNode {
        SweptSineNode::new(
            SR,
            SweptSineParams {
                start_hz: 20.0,
                end_hz: 20_000.0,
                duration_s,
                fade_s: 0.005,
                mode,
                amplitude: 0.5,
            },
        )
    }

    #[test]
    fn renders_bounded_finite() {
        let mut node = SweptSineNode::new(SR, SweptSineParams::default());
        let out = render(&mut node, SR as usize);
        let p = peak(&out);
        assert!(p > 0.0 && p < 1.0, "default peak = {p}");
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn sweep_rises_through_the_band() {
        let mut node = short(0.4, SweepMode::Exponential);
        let out = render(&mut node, (0.4 * SR as f32) as usize);
        let n = out.len();
        let head = &out[..n / 8];
        let tail = &out[n - n / 8..];
        // Early on the low end dominates; late on the high end dominates.
        assert!(
            goertzel(head, 60.0) > goertzel(head, 8_000.0),
            "sweep should start low"
        );
        assert!(
            goertzel(tail, 12_000.0) > goertzel(tail, 60.0),
            "sweep should end high"
        );
    }

    #[test]
    fn exponential_is_lower_at_midpoint_than_linear() {
        let dur = 0.4;
        let mut expo = short(dur, SweepMode::Exponential);
        let mut lin = short(dur, SweepMode::Linear);
        let frames = (dur * SR as f32) as usize;
        let eo = render(&mut expo, frames);
        let lo = render(&mut lin, frames);
        let mid = frames / 2;
        let win = 1024;
        let ef = freq_from_crossings(&eo[mid - win / 2..mid + win / 2]);
        let lf = freq_from_crossings(&lo[mid - win / 2..mid + win / 2]);
        // Exponential spends equal time per octave, so its mid frequency is the
        // geometric mean (~632 Hz); linear's is the arithmetic mean (~10 kHz).
        assert!(
            ef < lf * 0.25,
            "exp mid = {ef} Hz should be far below linear mid = {lf} Hz"
        );
    }

    #[test]
    fn goes_silent_after_duration() {
        let dur = 0.1;
        let mut node = short(dur, SweepMode::Exponential);
        let total = (dur * SR as f32).round() as usize;
        let out = render(&mut node, total + 2_000);
        let tail = &out[total + 500..];
        assert_eq!(peak(tail), 0.0, "sweep must be silent after it ends");
        assert!(peak(&out[..total]) > 0.0, "sweep should have sounded");
    }

    #[test]
    fn is_active_tracks_the_sweep() {
        let dur = 0.05;
        let mut node = short(dur, SweepMode::Exponential);
        assert!(node.is_active());
        let total = (dur * SR as f32).round() as usize;
        let _ = render(&mut node, total + 100);
        assert!(!node.is_active(), "sweep should be exhausted");
    }

    #[test]
    fn trigger_restarts_the_sweep() {
        let dur = 0.05;
        let mut node = short(dur, SweepMode::Exponential);
        let total = (dur * SR as f32).round() as usize;
        let _ = render(&mut node, total + 100);
        assert!(!node.is_active());
        node.trigger();
        assert!(node.is_active());
        let out = render(&mut node, 512);
        assert!(peak(&out) > 0.0, "re-triggered sweep should sound again");
    }

    #[test]
    fn fade_tapers_the_onset() {
        let faded = {
            let mut node = SweptSineNode::new(
                SR,
                SweptSineParams {
                    fade_s: 0.02,
                    ..Default::default()
                },
            );
            render(&mut node, 64)
        };
        let abrupt = {
            let mut node = SweptSineNode::new(
                SR,
                SweptSineParams {
                    fade_s: 0.0,
                    ..Default::default()
                },
            );
            render(&mut node, 64)
        };
        assert!(
            peak(&faded) < peak(&abrupt),
            "fade should soften the onset: faded = {}, abrupt = {}",
            peak(&faded),
            peak(&abrupt)
        );
    }

    #[test]
    fn deterministic_across_instances() {
        let params = SweptSineParams::default();
        let mut a = SweptSineNode::new(SR, params);
        let mut b = SweptSineNode::new(SR, params);
        let oa = render(&mut a, SR as usize / 2);
        let ob = render(&mut b, SR as usize / 2);
        assert_eq!(oa, ob);
    }

    #[test]
    fn reset_replays_identical_sweep() {
        let mut node = SweptSineNode::new(SR, SweptSineParams::default());
        let first = render(&mut node, SR as usize / 2);
        node.reset();
        let second = render(&mut node, SR as usize / 2);
        assert_eq!(first, second);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = SweptSineNode::new(
            SR,
            SweptSineParams {
                amplitude: 0.0,
                ..Default::default()
            },
        );
        let out = render(&mut node, SR as usize / 2);
        assert_eq!(peak(&out), 0.0, "zero amplitude must be silent");
    }

    #[test]
    fn amplitude_squared_scales_energy() {
        let mut quiet = SweptSineNode::new(
            SR,
            SweptSineParams {
                amplitude: 0.25,
                ..Default::default()
            },
        );
        let quiet_e = energy(&render(&mut quiet, SR as usize / 2));

        let mut loud = SweptSineNode::new(
            SR,
            SweptSineParams {
                amplitude: 0.5,
                ..Default::default()
            },
        );
        let loud_e = energy(&render(&mut loud, SR as usize / 2));

        let ratio = loud_e / quiet_e;
        assert!(
            (ratio - 4.0).abs() < 0.05,
            "doubling amplitude should quadruple energy, ratio = {ratio}"
        );
    }

    #[test]
    fn down_sweep_is_bounded_and_high_first() {
        let mut node = SweptSineNode::new(
            SR,
            SweptSineParams {
                start_hz: 18_000.0,
                end_hz: 40.0,
                duration_s: 0.4,
                fade_s: 0.005,
                mode: SweepMode::Exponential,
                amplitude: 0.5,
            },
        );
        let out = render(&mut node, (0.4 * SR as f32) as usize);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) < 1.0);
        let n = out.len();
        assert!(
            goertzel(&out[..n / 8], 12_000.0) > goertzel(&out[..n / 8], 60.0),
            "down-sweep should start high"
        );
    }

    #[test]
    fn mono_core_copied_to_channels() {
        let mut node = SweptSineNode::new(SR, SweptSineParams::default());
        let chans = render_layout(&mut node, SR as usize / 4, ChannelLayout::Quad);
        assert_eq!(chans.len(), 4);
        for ch in 1..chans.len() {
            assert_eq!(chans[0], chans[ch], "channel {ch} should mirror the core");
        }
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = SweptSineNode::new(SR, SweptSineParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn latency_is_zero() {
        let node = SweptSineNode::new(SR, SweptSineParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_sanitised_state() {
        let node = SweptSineNode::new(
            SR,
            SweptSineParams {
                start_hz: 50.0,
                end_hz: 8_000.0,
                duration_s: 3.0,
                fade_s: 0.05,
                mode: SweepMode::Linear,
                amplitude: 0.3,
            },
        );
        assert_eq!(node.start_hz(), 50.0);
        assert_eq!(node.end_hz(), 8_000.0);
        assert_eq!(node.duration_s(), 3.0);
        assert_eq!(node.fade_s(), 0.05);
        assert_eq!(node.mode(), SweepMode::Linear);
        assert_eq!(node.amplitude(), 0.3);
    }

    #[test]
    fn default_params_in_domain() {
        let d = SweptSineParams::default();
        assert_eq!(d, d.sanitised());
        assert_eq!(d.sanitised(), d.sanitised().sanitised());
    }

    #[test]
    fn sanitise_clamps_and_repairs() {
        let p = SweptSineParams {
            start_hz: f32::NAN,
            end_hz: 1.0e9,
            duration_s: 1.0e9,
            fade_s: -1.0,
            mode: SweepMode::Linear,
            amplitude: f32::INFINITY,
        }
        .sanitised();
        assert_eq!(p.start_hz, DEFAULT_START_HZ);
        assert_eq!(p.end_hz, MAX_FREQUENCY_HZ);
        assert_eq!(p.duration_s, MAX_DURATION_S);
        assert_eq!(p.fade_s, MIN_FADE_S);
        assert_eq!(p.mode, SweepMode::Linear);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = SweptSineNode::new(SR, SweptSineParams::default());

        node.set_start_hz(1.0e9);
        assert_eq!(node.start_hz(), MAX_FREQUENCY_HZ);
        node.set_start_hz(f32::NAN);
        assert_eq!(node.start_hz(), MAX_FREQUENCY_HZ);

        node.set_end_hz(0.0);
        assert_eq!(node.end_hz(), MIN_FREQUENCY_HZ);

        node.set_duration(1.0e9);
        assert_eq!(node.duration_s(), MAX_DURATION_S);
        node.set_duration(f32::NAN);
        assert_eq!(node.duration_s(), MAX_DURATION_S);

        node.set_fade(-1.0);
        assert_eq!(node.fade_s(), MIN_FADE_S);

        node.set_mode(SweepMode::Linear);
        assert_eq!(node.mode(), SweepMode::Linear);

        node.set_amplitude(f32::NAN, Ramp::Immediate);
        assert!(node.amplitude().is_finite());
    }
}
