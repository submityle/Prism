//! Bowed-string digital-waveguide physical-modeling source node.
//!
//! [`BowedStringNode`] is a *source* (zero inputs, one output) that synthesizes
//! a sustained, self-oscillating bowed-string tone (violin, cello, viol). Where
//! [`super::karplus_strong::KarplusStrongNode`] injects a single pluck and then
//! only loses energy, a bowed string is *continuously driven*: the bow feeds
//! energy into the string every sample through a nonlinear friction
//! interaction, so the tone sustains for as long as the bow moves and its
//! loudness, pitch of attack, and timbre all respond to how the bow is played.
//!
//! # The waveguide
//!
//! The string is modeled as a pair of digital waveguide delay lines carrying
//! the left- and right-going transverse *velocity* waves. The bow contacts the
//! string at a point that divides it into a bridge-side segment and a nut-side
//! segment; the two delay lines hold those segments:
//!
//! ```text
//! bridge_in = bridge_line.read(bridge_delay)   (wave arriving from the bridge)
//! nut_in    = nut_line.read(nut_delay)          (wave arriving from the nut)
//! bridge_r  = -g * ((1-S)*bridge_in + S*z^-1)   (bridge reflection: loss + invert)
//! nut_r     = -nut_in                            (nut reflection: rigid invert)
//! v_string  = bridge_r + nut_r                   (string velocity at the bow)
//! ```
//!
//! Reflection at each termination inverts the velocity wave; the bridge is
//! lossy and slightly low-pass (a one-zero filter `(1-S) + S z^-1` scaled by a
//! loop gain `g`), modeling how a real string loses its high partials fastest.
//! `S` in `[0, 0.5]` is set from `brightness` exactly as in the Karplus-Strong
//! loop (`S = 0.5 * (1 - brightness)`), and its low-frequency phase delay of `S`
//! samples is folded into the tuning so the pitch stays accurate.
//!
//! # The bow (friction nonlinearity)
//!
//! The bow drives the string through the classic friction characteristic of the
//! `McIntyre`-Schumacher-Woodhouse (`MSW`) model. Let `v_bow` be the bow
//! velocity and `dv = v_bow - v_string` the relative (slipping) velocity. The
//! friction curve returns a reflection coefficient that is near `1` when the
//! string sticks to the bow (small `|dv|`) and falls off smoothly as the string
//! breaks away and slips:
//!
//! ```text
//! x  = (dv + offset) * slope
//! mu = clamp((|x| + 0.75)^(-4), 0, 1)
//! bow_out = dv * mu
//! ```
//!
//! `bow_out` is the velocity the bow adds to both outgoing waves. The `slope`
//! is set from `bow_force`: pressing harder narrows the sticking region, so the
//! stick-slip (Helmholtz) motion snaps more sharply and the tone brightens and
//! grows richer in harmonics. The injected waves close the loop:
//!
//! ```text
//! nut_line.write(bridge_r + bow_out)      (send toward the nut)
//! bridge_line.write(nut_r + bow_out)      (send toward the bridge)
//! output = bridge_in * amplitude          (radiated bridge velocity)
//! ```
//!
//! Because the friction coefficient saturates toward zero for large `|dv|`, the
//! bow cannot inject unbounded energy: the nonlinearity is self-limiting, which
//! (together with the sub-unity bridge loop gain) keeps the loop stable without
//! any hard clamp.
//!
//! # Pitch and timbre
//!
//! The fundamental is `f0 = sample_rate / D`, where `D = bridge_delay +
//! nut_delay + S` is the total loop delay. Given `D = sample_rate / f0`, the
//! node reserves `S` samples for the bridge filter and splits the remainder by
//! `bow_position` (`beta` in `[0.02, 0.5]`, the bow's fractional distance from
//! the bridge): `bridge_delay = (D - S) * beta` and `nut_delay = (D - S) *
//! (1 - beta)`. Both delay lines read at a fractional length by linear
//! interpolation, so the string tunes continuously. The bow position also
//! shapes the spectrum: harmonics with a node at the bow point are suppressed,
//! the physical origin of the characteristic bowed-string formant pattern.
//!
//! # Determinism
//!
//! The excitation is the deterministic bow motion, not noise, so the node holds
//! no random state: two [`BowedStringNode`]s built with the same sample rate and
//! parameters emit bit-identical streams on every platform via
//! [`bevy_math::ops`], and [`AudioNode::reset`] restarts the identical attack.
//!
//! # Relationship
//!
//! This is the *bowed* counterpart to the *plucked*
//! [`super::karplus_strong::KarplusStrongNode`]: both recirculate energy in a
//! tuned, loss-filtered delay loop with sub-sample fractional tuning, but the
//! plucked string is excited once and decays, whereas this node is excited
//! continuously by the bow friction nonlinearity and self-oscillates. Unlike
//! the geometric [`super::oscillator::OscillatorNode`] (a fixed waveform) or the
//! [`super::fof_source::FofSourceNode`] (a periodic formant-grain stream), the
//! bowed string has no prescribed waveform: its Helmholtz corner emerges from
//! the physics of the loop. It reuses only this crate's own [`Sample`] type,
//! [`Smoothed`] parameter smoother, and denormal-flushing primitive.
//!
//! # Real-time contract
//!
//! Both delay lines are pre-allocated in [`BowedStringNode::new`] for the lowest
//! supported pitch, so [`process`](crate::graph::AudioNode::process) performs no
//! allocation, takes no locks, and cannot panic: every recirculated sample is
//! denormal-flushed, the bridge loop gain is below unity, the friction
//! coefficient is clamped to `[0, 1]`, and non-finite parameters are rejected at
//! the setters. The bow velocity and output amplitude are driven through
//! [`Smoothed`] values so performance gestures never zipper.
//!
//! # Provenance
//!
//! The bowed-string friction model is that of `McIntyre`, Schumacher and
//! Woodhouse ("On the oscillations of musical instruments", *Journal of the
//! Acoustical Society of America*, 1983); the digital-waveguide realization and
//! the fractional-delay string loop follow the standard public treatment in
//! J. O. Smith's *Physical Audio Signal Processing* (public online text) and
//! J. O. Smith and S. A. Van Duyne's digital-waveguide string papers. The
//! one-zero bridge loss filter and `brightness` mapping are shared with the
//! Jaffe-Smith Karplus-Strong extensions. This module contains **no Unreal
//! Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, or
//! Web Audio source or derived code**, and nothing from any audio toolkit's
//! implementation; it is written purely from that publicly documented theory.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz. Bounds the pre-allocated delay lines
/// (`sample_rate / MIN_FREQUENCY_HZ` frames of maximum total delay).
pub const MIN_FREQUENCY_HZ: Sample = 40.0;

/// Default fundamental (pitch) frequency in hertz.
pub const DEFAULT_FREQUENCY_HZ: Sample = 220.0;

/// Minimum bow position (fraction of the string length from the bridge).
pub const MIN_BOW_POSITION: Sample = 0.02;

/// Maximum bow position. The string is symmetric about its midpoint, so bowing
/// positions beyond `0.5` mirror positions below it.
pub const MAX_BOW_POSITION: Sample = 0.5;

/// Default bow position (a typical violin bowing point near the bridge).
pub const DEFAULT_BOW_POSITION: Sample = 0.127;

/// Default normalized bow velocity in `[0, 1]`.
pub const DEFAULT_BOW_VELOCITY: Sample = 0.5;

/// Default normalized bow force (pressure) in `[0, 1]`.
pub const DEFAULT_BOW_FORCE: Sample = 0.5;

/// Default brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Internal scale mapping normalized bow velocity `[0, 1]` to physical units.
const MAX_BOW_VELOCITY: Sample = 0.3;

/// Friction-curve slope at the lightest bow force.
const BOW_SLOPE_MIN: Sample = 1.0;

/// Friction-curve slope at the heaviest bow force.
const BOW_SLOPE_MAX: Sample = 10.0;

/// Small offset keeping the friction curve asymmetric about zero so the string
/// reliably breaks into oscillation from rest.
const BOW_TABLE_OFFSET: Sample = 0.001;

/// Bridge reflection loop gain, just below unity so the passive loop is stable
/// while the bow supplies the sustaining energy.
const STRING_REFLECTION_GAIN: Sample = 0.95;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Clamps `frequency_hz` to `[MIN_FREQUENCY_HZ, sample_rate / 2]`, falling back
/// to [`MIN_FREQUENCY_HZ`] for non-finite input.
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist)
}

/// The nonlinear bow friction characteristic: a reflection coefficient in
/// `[0, 1]` that is near unity while the string sticks and decays as it slips.
#[inline]
fn bow_friction(relative_velocity: Sample, slope: Sample) -> Sample {
    let x = (relative_velocity + BOW_TABLE_OFFSET) * slope;
    ops::powf(x.abs() + 0.75, -4.0).clamp(0.0, 1.0)
}

/// A linear-interpolating waveguide delay line (a velocity-wave travelling
/// segment of the string).
#[derive(Debug, Clone)]
struct WaveguideDelay {
    buf: Vec<Sample>,
    write: usize,
}

impl WaveguideDelay {
    fn new(capacity: usize) -> Self {
        Self { buf: vec![0.0; capacity.max(2)], write: 0 }
    }

    /// Reads the wave delayed by `delay` samples with linear interpolation.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "delay is clamped to [1, len-1]; the integer part fits a usize exactly"
    )]
    fn read(&self, delay: Sample) -> Sample {
        let len = self.buf.len();
        let d = delay.clamp(1.0, (len - 1) as Sample);
        let di = d as usize;
        let frac = d - di as Sample;
        let i0 = (self.write + len - di) % len;
        let i1 = (self.write + len - di - 1) % len;
        self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac
    }

    /// Writes the next sample and advances the write head.
    #[inline]
    fn write_sample(&mut self, x: Sample) {
        self.buf[self.write] = flush_denormal(x);
        self.write += 1;
        if self.write == self.buf.len() {
            self.write = 0;
        }
    }

    /// Zeroes the line and resets the write head.
    fn clear(&mut self) {
        for v in &mut self.buf {
            *v = 0.0;
        }
        self.write = 0;
    }
}

/// Construction parameters for a [`BowedStringNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BowedStringParams {
    /// Fundamental (pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Normalized bow velocity in `[0, 1]` (loudness / drive).
    pub bow_velocity: Sample,
    /// Normalized bow force (pressure) in `[0, 1]` (timbre / stick-slip).
    pub bow_force: Sample,
    /// Bow position as a fraction of the string length from the bridge.
    pub bow_position: Sample,
    /// Brightness in `[0, 1]` (bridge filter high-frequency loss).
    pub brightness: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for BowedStringParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            bow_velocity: DEFAULT_BOW_VELOCITY,
            bow_force: DEFAULT_BOW_FORCE,
            bow_position: DEFAULT_BOW_POSITION,
            brightness: DEFAULT_BRIGHTNESS,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

/// A bowed-string digital-waveguide voice source node (0 inputs, 1 output).
///
/// The mono string signal is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{BowedStringNode, BowedStringParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = BowedStringNode::new(48_000, BowedStringParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The bow drives the string into sustained self-oscillation.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct BowedStringNode {
    /// Sample rate the delay lines were sized for.
    sample_rate: Sample,
    /// Fundamental frequency in hertz.
    frequency_hz: Sample,
    /// Bow position (`beta`) in `[MIN_BOW_POSITION, MAX_BOW_POSITION]`.
    bow_position: Sample,
    /// Bow force in `[0, 1]`.
    bow_force: Sample,
    /// Brightness in `[0, 1]`.
    brightness: Sample,
    /// Smoothed normalized bow velocity in `[0, 1]`.
    bow_velocity: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// Bridge-side (bow to bridge) waveguide segment.
    bridge_line: WaveguideDelay,
    /// Nut-side (bow to nut) waveguide segment.
    nut_line: WaveguideDelay,
    /// Previous bridge-filter input (`z^-1`) for the one-zero loss filter.
    bridge_filter_z1: Sample,

    /// Latched bridge-side fractional delay in samples.
    bridge_delay: Sample,
    /// Latched nut-side fractional delay in samples.
    nut_delay: Sample,
    /// Latched one-zero bridge-filter coefficient `S`.
    damping_s: Sample,
    /// Latched friction-curve slope.
    slope: Sample,
}

impl BowedStringNode {
    /// Builds a bowed string for `sample_rate` Hz from a parameter bundle.
    ///
    /// The delay lines are sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, `bow_position` to `[MIN_BOW_POSITION, MAX_BOW_POSITION]`, and
    /// `bow_velocity`/`bow_force`/`brightness` to `[0, 1]`.
    #[must_use]
    pub fn new(sample_rate: u32, params: BowedStringParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest total loop delay (frames at the lowest supported pitch) with
        // headroom for the bridge filter and interpolation taps.
        let max_delay_frames = ops::round(sr / MIN_FREQUENCY_HZ) as usize;
        let capacity = max_delay_frames + 4;

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let bow_position =
            finite_or(params.bow_position, DEFAULT_BOW_POSITION).clamp(MIN_BOW_POSITION, MAX_BOW_POSITION);
        let bow_force = finite_or(params.bow_force, DEFAULT_BOW_FORCE).clamp(0.0, 1.0);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let bow_velocity = finite_or(params.bow_velocity, DEFAULT_BOW_VELOCITY).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            frequency_hz,
            bow_position,
            bow_force,
            brightness,
            bow_velocity: Smoothed::new(bow_velocity),
            amplitude: Smoothed::new(amplitude),
            bridge_line: WaveguideDelay::new(capacity),
            nut_line: WaveguideDelay::new(capacity),
            bridge_filter_z1: 0.0,
            bridge_delay: 1.0,
            nut_delay: 1.0,
            damping_s: 0.25,
            slope: BOW_SLOPE_MIN,
        };
        node.recompute();
        node
    }

    /// Retunes the string to `frequency_hz` (clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`).
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        let requested = finite_or(frequency_hz, self.frequency_hz);
        self.frequency_hz = sanitize_frequency(requested, self.sample_rate);
        self.recompute();
    }

    /// Sets the bow position (`beta`) as a fraction of the string length from
    /// the bridge (clamped to `[MIN_BOW_POSITION, MAX_BOW_POSITION]`).
    #[inline]
    pub fn set_bow_position(&mut self, bow_position: Sample) {
        self.bow_position =
            finite_or(bow_position, self.bow_position).clamp(MIN_BOW_POSITION, MAX_BOW_POSITION);
        self.recompute();
    }

    /// Sets the bow force (pressure) in `[0, 1]`.
    #[inline]
    pub fn set_bow_force(&mut self, bow_force: Sample) {
        self.bow_force = finite_or(bow_force, self.bow_force).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the brightness in `[0, 1]`.
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the normalized bow velocity in `[0, 1]`, smoothing over `ramp`.
    #[inline]
    pub fn set_bow_velocity(&mut self, bow_velocity: Sample, ramp: Ramp) {
        let v = finite_or(bow_velocity, self.bow_velocity.target()).clamp(0.0, 1.0);
        self.bow_velocity.set_target(v, ramp);
    }

    /// Sets a new target output amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the fundamental frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the bow position (`beta`).
    #[inline]
    #[must_use]
    pub fn bow_position(&self) -> Sample {
        self.bow_position
    }

    /// Returns the bow force (pressure) in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn bow_force(&self) -> Sample {
        self.bow_force
    }

    /// Returns the brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the target normalized bow velocity in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn bow_velocity(&self) -> Sample {
        self.bow_velocity.target()
    }

    /// Returns the target output amplitude (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Recomputes the latched loop coefficients from the user-facing parameters.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    fn recompute(&mut self) {
        let sr = self.sample_rate;
        // One-zero bridge-filter coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping_s = s;
        self.slope = BOW_SLOPE_MIN + self.bow_force * (BOW_SLOPE_MAX - BOW_SLOPE_MIN);

        // Total loop delay D = sr / f0 = bridge + nut + S. Reserve S for the
        // bridge filter and split the remainder by the bow position.
        let max_total = (self.bridge_line.buf.len() - 2) as Sample;
        let d_total = (sr / self.frequency_hz - s).clamp(2.0, max_total);
        self.bridge_delay = (d_total * self.bow_position).max(1.0);
        self.nut_delay = (d_total * (1.0 - self.bow_position)).max(1.0);
    }

    /// Applies the one-zero bridge loss filter and advances its memory.
    #[inline]
    fn bridge_filter(&mut self, x: Sample) -> Sample {
        let s = self.damping_s;
        let y = STRING_REFLECTION_GAIN * ((1.0 - s) * x + s * self.bridge_filter_z1);
        self.bridge_filter_z1 = x;
        y
    }

    /// Advances the waveguide by one sample and returns the radiated output.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let bow_v = self.bow_velocity.next_sample() * MAX_BOW_VELOCITY;

        let bridge_in = self.bridge_line.read(self.bridge_delay);
        let nut_in = self.nut_line.read(self.nut_delay);
        // Reflections: both terminations invert velocity; the bridge is lossy.
        let bridge_r = -self.bridge_filter(bridge_in);
        let nut_r = -nut_in;

        let string_vel = bridge_r + nut_r;
        let dv = bow_v - string_vel;
        let bow_out = dv * bow_friction(dv, self.slope);

        self.nut_line.write_sample(bridge_r + bow_out);
        self.bridge_line.write_sample(nut_r + bow_out);

        flush_denormal(bridge_in * self.amplitude.next_sample())
    }
}

impl AudioNode for BowedStringNode {
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
        self.bridge_line.clear();
        self.nut_line.clear();
        self.bridge_filter_z1 = 0.0;
        self.bow_velocity = Smoothed::new(self.bow_velocity.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
        self.recompute();
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

    /// Renders `frames` of mono output into a flat vector.
    fn render(node: &mut BowedStringNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout` and returns per-channel
    /// sample vectors.
    fn render_layout(
        node: &mut BowedStringNode,
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
        (0..channels).map(|ch| outputs[0].channel(ch).to_vec()).collect()
    }

    fn peak(block: &[Sample]) -> Sample {
        block.iter().fold(0.0, |m, &s| m.max(s.abs()))
    }

    fn energy(block: &[Sample]) -> f64 {
        block.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
    }


    #[test]
    fn default_self_oscillates() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        let block = render(&mut node, SR as usize);
        // The bow drives sustained oscillation, not a single decaying pluck.
        assert!(peak(&block) > 0.0);
        assert!(peak(&block[SR as usize / 2..]) > 1e-2);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn long_run_stays_bounded() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        let block = render(&mut node, 4 * SR as usize);
        assert!(block.iter().all(|s| s.is_finite()));
        // The self-limiting friction nonlinearity plus sub-unity loop gain keep
        // the output well bounded; it must never run away.
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = BowedStringNode::new(SR, BowedStringParams::default());
        let mut b = BowedStringNode::new(SR, BowedStringParams::default());
        assert_eq!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        let first = render(&mut node, 8192);
        node.reset();
        let second = render(&mut node, 8192);
        assert_eq!(first, second);
    }

    #[test]
    fn zero_bow_velocity_is_silent() {
        let params = BowedStringParams {
            bow_velocity: 0.0,
            ..BowedStringParams::default()
        };
        let mut node = BowedStringNode::new(SR, params);
        // With no bow motion and no stored energy the string never excites.
        let block = render(&mut node, SR as usize / 2);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn amplitude_scales_output_energy() {
        let loud = BowedStringParams {
            amplitude: 1.0,
            ..BowedStringParams::default()
        };
        let soft = BowedStringParams {
            amplitude: 0.5,
            ..BowedStringParams::default()
        };
        let mut a = BowedStringNode::new(SR, loud);
        let mut b = BowedStringNode::new(SR, soft);
        let ea = energy(&render(&mut a, 8192));
        let eb = energy(&render(&mut b, 8192));
        // Amplitude only scales the radiated signal; halving it quarters energy.
        assert!(eb > 0.0);
        assert!((ea / eb - 4.0).abs() < 1e-3);
    }

    #[test]
    fn mono_core_replicated_to_all_channels() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        let chans = render_layout(&mut node, 4096, ChannelLayout::Stereo);
        assert_eq!(chans.len(), 2);
        assert_eq!(chans[0], chans[1]);
        assert!(peak(&chans[0]) > 0.0);
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = BowedStringParams {
            frequency_hz: 196.0,
            bow_velocity: 0.6,
            bow_force: 0.4,
            bow_position: 0.2,
            brightness: 0.7,
            amplitude: 0.8,
        };
        let node = BowedStringNode::new(SR, params);
        assert!((node.frequency_hz() - 196.0).abs() < 1e-3);
        assert!((node.bow_velocity() - 0.6).abs() < 1e-6);
        assert!((node.bow_force() - 0.4).abs() < 1e-6);
        assert!((node.bow_position() - 0.2).abs() < 1e-6);
        assert!((node.brightness() - 0.7).abs() < 1e-6);
        assert!((node.amplitude() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn set_frequency_clamps_to_range() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        node.set_frequency_hz(1.0e9);
        assert!((node.frequency_hz() - (SR as Sample) * 0.5).abs() < 1e-3);
        node.set_frequency_hz(1.0);
        assert!((node.frequency_hz() - MIN_FREQUENCY_HZ).abs() < 1e-3);
    }

    #[test]
    fn set_bow_position_clamps_to_range() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        node.set_bow_position(10.0);
        assert!((node.bow_position() - MAX_BOW_POSITION).abs() < 1e-6);
        node.set_bow_position(-1.0);
        assert!((node.bow_position() - MIN_BOW_POSITION).abs() < 1e-6);
    }

    #[test]
    fn set_bow_force_and_brightness_clamp() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        node.set_bow_force(5.0);
        assert!((node.bow_force() - 1.0).abs() < 1e-6);
        node.set_bow_force(-2.0);
        assert!((node.bow_force() - 0.0).abs() < 1e-6);
        node.set_brightness(9.0);
        assert!((node.brightness() - 1.0).abs() < 1e-6);
        node.set_brightness(-9.0);
        assert!((node.brightness() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        let f = node.frequency_hz();
        let p = node.bow_position();
        let force = node.bow_force();
        let b = node.brightness();
        let v = node.bow_velocity();
        let a = node.amplitude();
        node.set_frequency_hz(Sample::NAN);
        node.set_bow_position(Sample::INFINITY);
        node.set_bow_force(Sample::NAN);
        node.set_brightness(Sample::NEG_INFINITY);
        node.set_bow_velocity(Sample::NAN, Ramp::Immediate);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), f);
        assert_eq!(node.bow_position(), p);
        assert_eq!(node.bow_force(), force);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.bow_velocity(), v);
        assert_eq!(node.amplitude(), a);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let params = BowedStringParams {
            frequency_hz: Sample::NAN,
            bow_velocity: Sample::INFINITY,
            bow_force: Sample::NAN,
            bow_position: Sample::NEG_INFINITY,
            brightness: Sample::NAN,
            amplitude: Sample::NAN,
        };
        let node = BowedStringNode::new(SR, params);
        assert!(node.frequency_hz().is_finite());
        assert!(node.bow_velocity().is_finite());
        assert!(node.bow_force().is_finite());
        assert!(node.bow_position().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        let low = BowedStringParams {
            frequency_hz: 110.0,
            ..BowedStringParams::default()
        };
        let high = BowedStringParams {
            frequency_hz: 440.0,
            ..BowedStringParams::default()
        };
        let mut a = BowedStringNode::new(SR, low);
        let mut b = BowedStringNode::new(SR, high);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn bow_force_changes_timbre() {
        let light = BowedStringParams {
            bow_force: 0.1,
            ..BowedStringParams::default()
        };
        let heavy = BowedStringParams {
            bow_force: 0.9,
            ..BowedStringParams::default()
        };
        let mut a = BowedStringNode::new(SR, light);
        let mut b = BowedStringNode::new(SR, heavy);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn bow_position_changes_timbre() {
        let near = BowedStringParams {
            bow_position: 0.05,
            ..BowedStringParams::default()
        };
        let mid = BowedStringParams {
            bow_position: 0.4,
            ..BowedStringParams::default()
        };
        let mut a = BowedStringNode::new(SR, near);
        let mut b = BowedStringNode::new(SR, mid);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn brightness_changes_output() {
        let dark = BowedStringParams {
            brightness: 0.05,
            ..BowedStringParams::default()
        };
        let bright = BowedStringParams {
            brightness: 0.95,
            ..BowedStringParams::default()
        };
        let mut a = BowedStringNode::new(SR, dark);
        let mut b = BowedStringNode::new(SR, bright);
        let da = render(&mut a, 8192);
        let db = render(&mut b, 8192);
        // The bridge loss filter coefficient tracks brightness, so the radiated
        // waveform must differ.
        assert_ne!(da, db);
    }

    #[test]
    fn high_and_low_frequencies_both_oscillate() {
        for &f in &[MIN_FREQUENCY_HZ, 2_000.0] {
            let params = BowedStringParams {
                frequency_hz: f,
                ..BowedStringParams::default()
            };
            let mut node = BowedStringNode::new(SR, params);
            let block = render(&mut node, SR as usize);
            assert!(peak(&block) > 0.0, "silent at {f} Hz");
            assert!(block.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn bow_velocity_target_tracks_setter() {
        let mut node = BowedStringNode::new(SR, BowedStringParams::default());
        node.set_bow_velocity(0.8, Ramp::linear_seconds(0.01, SR));
        assert!((node.bow_velocity() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = BowedStringNode::new(SR, BowedStringParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
