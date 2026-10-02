//! Reed-woodwind digital-waveguide physical-modeling source node.
//!
//! [`ReedWoodwindNode`] is a *source* (zero inputs, one output) that synthesizes
//! a sustained, self-oscillating single-reed woodwind tone (clarinet family) by
//! modeling a cylindrical bore driven by a nonlinear reed valve. Like
//! [`super::bowed_string::BowedStringNode`] it is *continuously driven*: steady
//! breath pressure feeds energy into the bore every sample through the reed
//! nonlinearity, so the tone sustains for as long as the player blows.
//!
//! # The bore
//!
//! The air column is modeled as a pair of digital-waveguide delay lines
//! carrying the pressure waves travelling toward the bell and back toward the
//! mouthpiece:
//!
//! ```text
//! at_mouth  = mouth_line.read(delay)   (wave arriving at the mouthpiece)
//! at_bell   = bell_line.read(delay)    (wave arriving at the open bell)
//! bell_refl = -g * ((1-S)*at_bell + S*z^-1)   (bell: loss, low-pass, invert)
//! ```
//!
//! The bell is an *open* end: it reflects the pressure wave with inversion and a
//! gentle one-zero low-pass loss `(1-S) + S z^-1` scaled by a loop gain `g`,
//! modeling radiation loss that grows with frequency. The mouthpiece is a
//! near-*closed* end whose reflection is positive, so each round trip inverts
//! the wave exactly once: the bore therefore resonates only on its *odd*
//! harmonics, the physical origin of the hollow clarinet timbre that sounds an
//! octave below a comparable open-open tube.
//!
//! # The reed (pressure-controlled valve)
//!
//! The reed is driven by the pressure drop `delta = breath - at_mouth` between
//! the player's steady mouth pressure `breath` and the bore wave returning to
//! the mouthpiece. A clipped-linear reed table returns a reflection coefficient
//! that falls as the drop grows (the reed closes against the mouthpiece and
//! finally beats shut, clamped at `1`):
//!
//! ```text
//! h  = clamp(REED_REST - slope*delta, -1, 1)
//! mouth_refl = at_mouth + h*(breath - at_mouth)
//! ```
//!
//! `mouth_refl` is the pressure wave launched back down the bore. `slope` is set
//! from `reed_stiffness`: a stiffer reed closes over a narrower pressure range,
//! sharpening the stick-beat motion so the tone brightens and gains upper odd
//! partials. Because `h` is clamped to `[-1, 1]` the reed cannot inject
//! unbounded energy, so the nonlinearity is self-limiting and (with the
//! sub-unity bell loop gain) the loop stays stable without any hard clamp.
//!
//! ```text
//! bell_line.write(mouth_refl)   (send toward the bell)
//! mouth_line.write(bell_refl)   (send toward the mouthpiece)
//! output = at_mouth * amplitude (radiated mouthpiece pressure)
//! ```
//!
//! # Pitch and timbre
//!
//! Because one inversion per round trip selects the odd harmonics, the sounding
//! fundamental is `f0 = sample_rate / (4 * delay)` where `delay` is each line's
//! length. Given `delay = sample_rate / (4 * f0)` (minus the bell filter's small
//! phase delay, folded in for tuning accuracy) the lines read at a fractional
//! length by linear interpolation, so the instrument tunes continuously.
//! `brightness` sets the bell low-pass coefficient `S = 0.5 * (1 - brightness)`
//! exactly as in the Karplus-Strong loop; `breath` sets loudness and, through
//! the reed, the richness of the beating regime.
//!
//! # Determinism
//!
//! The excitation is the deterministic breath pressure, not noise, so the node
//! holds no random state: two [`ReedWoodwindNode`]s built with the same sample
//! rate and parameters emit bit-identical streams on every platform via
//! [`bevy_math::ops`], and [`AudioNode::reset`] restarts the identical attack.
//!
//! # Relationship
//!
//! This is the *reed-driven* sibling of the *bow-driven*
//! [`super::bowed_string::BowedStringNode`] and the *plucked*
//! [`super::karplus_strong::KarplusStrongNode`]: all three recirculate energy in
//! a tuned, loss-filtered delay loop with fractional-delay tuning, but the
//! woodwind is sustained by a reed valve rather than bow friction or a single
//! pluck, and its single per-round-trip inversion gives the odd-only harmonic
//! series the others do not share. Unlike the geometric
//! [`super::oscillator::OscillatorNode`] (a fixed waveform) it has no prescribed
//! waveform: the square-ish bore pressure emerges from the loop physics. It
//! reuses only this crate's own [`Sample`] type, [`Smoothed`] parameter
//! smoother, and denormal-flushing primitive.
//!
//! # Real-time contract
//!
//! Both delay lines are pre-allocated in [`ReedWoodwindNode::new`] for the lowest
//! supported pitch, so [`process`](crate::graph::AudioNode::process) performs no
//! allocation, takes no locks, and cannot panic: every recirculated sample is
//! denormal-flushed, the bell loop gain is below unity, the reed coefficient is
//! clamped to `[-1, 1]`, and non-finite parameters are rejected at the setters.
//! The breath pressure and output amplitude are driven through [`Smoothed`]
//! values so performance gestures never zipper.
//!
//! # Provenance
//!
//! The reed valve and bore model follow the standard public digital-waveguide
//! treatment of single-reed instruments in J. O. Smith's *Physical Audio Signal
//! Processing* (public online text), itself built on the reed physics of
//! `McIntyre`, Schumacher and Woodhouse ("On the oscillations of musical
//! instruments", *Journal of the Acoustical Society of America*, 1983). The
//! one-zero bell loss filter and `brightness` mapping are shared with the
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

/// Lowest tunable fundamental in hertz. Bounds the pre-allocated delay lines.
pub const MIN_FREQUENCY_HZ: Sample = 40.0;

/// Default fundamental (pitch) frequency in hertz (roughly a clarinet A3).
pub const DEFAULT_FREQUENCY_HZ: Sample = 220.0;

/// Default normalized breath pressure in `[0, 1]`.
pub const DEFAULT_BREATH_PRESSURE: Sample = 0.5;

/// Default normalized reed stiffness in `[0, 1]`.
pub const DEFAULT_REED_STIFFNESS: Sample = 0.5;

/// Default brightness in `[0, 1]`.
pub const DEFAULT_BRIGHTNESS: Sample = 0.5;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.5;

/// Internal scale mapping normalized breath pressure `[0, 1]` to bore units.
const MAX_BREATH_PRESSURE: Sample = 0.55;

/// Reed-table value at zero pressure drop (reed slightly open at rest).
const REED_REST: Sample = 0.7;

/// Reed-table slope at the softest reed.
const REED_SLOPE_MIN: Sample = 1.5;

/// Reed-table slope at the stiffest reed.
const REED_SLOPE_MAX: Sample = 6.0;

/// Bell reflection loop gain, just below unity so the passive loop is stable
/// while the breath supplies the sustaining energy.
const BELL_REFLECTION_GAIN: Sample = 0.97;

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

/// The nonlinear reed reflection characteristic: a clipped-linear coefficient in
/// `[-1, 1]` that falls as the pressure drop closes the reed.
#[inline]
fn reed_reflection(pressure_drop: Sample, slope: Sample) -> Sample {
    (REED_REST - slope * pressure_drop).clamp(-1.0, 1.0)
}

/// A linear-interpolating waveguide delay line (a travelling bore segment).
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

/// Construction parameters for a [`ReedWoodwindNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ReedWoodwindParams {
    /// Fundamental (pitch) frequency in hertz.
    pub frequency_hz: Sample,
    /// Normalized breath pressure in `[0, 1]` (loudness / drive).
    pub breath_pressure: Sample,
    /// Normalized reed stiffness in `[0, 1]` (timbre / beating sharpness).
    pub reed_stiffness: Sample,
    /// Brightness in `[0, 1]` (bell filter high-frequency loss).
    pub brightness: Sample,
    /// Linear output amplitude.
    pub amplitude: Sample,
}

impl Default for ReedWoodwindParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_FREQUENCY_HZ,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            reed_stiffness: DEFAULT_REED_STIFFNESS,
            brightness: DEFAULT_BRIGHTNESS,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

/// A reed-woodwind digital-waveguide voice source node (0 inputs, 1 output).
///
/// The mono bore signal is replicated into every output channel.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{ReedWoodwindNode, ReedWoodwindParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = ReedWoodwindNode::new(48_000, ReedWoodwindParams::default());
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 48_000)];
/// outputs[0].set_active_frames(48_000);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 48_000, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The breath drives the bore into sustained self-oscillation.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct ReedWoodwindNode {
    /// Sample rate the delay lines were sized for.
    sample_rate: Sample,
    /// Fundamental frequency in hertz.
    frequency_hz: Sample,
    /// Reed stiffness in `[0, 1]`.
    reed_stiffness: Sample,
    /// Brightness in `[0, 1]`.
    brightness: Sample,
    /// Smoothed normalized breath pressure in `[0, 1]`.
    breath_pressure: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,

    /// Mouthpiece-bound (bell to mouth) waveguide segment.
    mouth_line: WaveguideDelay,
    /// Bell-bound (mouth to bell) waveguide segment.
    bell_line: WaveguideDelay,
    /// Previous bell-filter input (`z^-1`) for the one-zero loss filter.
    bell_filter_z1: Sample,

    /// Latched per-line fractional delay in samples.
    delay: Sample,
    /// Latched one-zero bell-filter coefficient `S`.
    damping_s: Sample,
    /// Latched reed-table slope.
    slope: Sample,
}

impl ReedWoodwindNode {
    /// Builds a reed woodwind for `sample_rate` Hz from a parameter bundle.
    ///
    /// The delay lines are sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, and `breath_pressure`/`reed_stiffness`/`brightness` to `[0, 1]`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "the buffer length is tiny relative to f32's integer precision"
    )]
    pub fn new(sample_rate: u32, params: ReedWoodwindParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest per-line delay (frames at the lowest pitch): delay = sr /
        // (4 f0), with headroom for the bell filter and interpolation taps.
        let max_delay_frames = ops::round(sr / (4.0 * MIN_FREQUENCY_HZ)) as usize;
        let capacity = max_delay_frames + 4;

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let reed_stiffness = finite_or(params.reed_stiffness, DEFAULT_REED_STIFFNESS).clamp(0.0, 1.0);
        let brightness = finite_or(params.brightness, DEFAULT_BRIGHTNESS).clamp(0.0, 1.0);
        let breath_pressure = finite_or(params.breath_pressure, DEFAULT_BREATH_PRESSURE).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, DEFAULT_AMPLITUDE);

        let mut node = Self {
            sample_rate: sr,
            frequency_hz,
            reed_stiffness,
            brightness,
            breath_pressure: Smoothed::new(breath_pressure),
            amplitude: Smoothed::new(amplitude),
            mouth_line: WaveguideDelay::new(capacity),
            bell_line: WaveguideDelay::new(capacity),
            bell_filter_z1: 0.0,
            delay: 1.0,
            damping_s: 0.25,
            slope: REED_SLOPE_MIN,
        };
        node.recompute();
        node
    }

    /// Retunes the bore to `frequency_hz` (clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`).
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        let requested = finite_or(frequency_hz, self.frequency_hz);
        self.frequency_hz = sanitize_frequency(requested, self.sample_rate);
        self.recompute();
    }

    /// Sets the reed stiffness in `[0, 1]`.
    #[inline]
    pub fn set_reed_stiffness(&mut self, reed_stiffness: Sample) {
        self.reed_stiffness = finite_or(reed_stiffness, self.reed_stiffness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the brightness in `[0, 1]`.
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
        self.recompute();
    }

    /// Sets the normalized breath pressure in `[0, 1]`, smoothing over `ramp`.
    #[inline]
    pub fn set_breath_pressure(&mut self, breath_pressure: Sample, ramp: Ramp) {
        let v = finite_or(breath_pressure, self.breath_pressure.target()).clamp(0.0, 1.0);
        self.breath_pressure.set_target(v, ramp);
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

    /// Returns the reed stiffness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn reed_stiffness(&self) -> Sample {
        self.reed_stiffness
    }

    /// Returns the brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the target normalized breath pressure in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn breath_pressure(&self) -> Sample {
        self.breath_pressure.target()
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
        // One-zero bell-filter coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping_s = s;
        self.slope = REED_SLOPE_MIN + self.reed_stiffness * (REED_SLOPE_MAX - REED_SLOPE_MIN);

        // Odd-harmonic loop: one inversion per round trip -> f0 = sr / (4 D).
        // The bell filter contributes an extra phase delay of S samples per
        // round trip (S/2 per one-way segment); fold it into the tuning.
        let max_delay = (self.mouth_line.buf.len() - 2) as Sample;
        let d = (sr / (4.0 * self.frequency_hz) - 0.5 * s).clamp(1.0, max_delay);
        self.delay = d;
    }

    /// Applies the one-zero bell loss filter and advances its memory.
    #[inline]
    fn bell_filter(&mut self, x: Sample) -> Sample {
        let s = self.damping_s;
        let y = BELL_REFLECTION_GAIN * ((1.0 - s) * x + s * self.bell_filter_z1);
        self.bell_filter_z1 = x;
        y
    }

    /// Advances the waveguide by one sample and returns the radiated output.
    #[inline]
    fn render_sample(&mut self) -> Sample {
        let breath = self.breath_pressure.next_sample() * MAX_BREATH_PRESSURE;

        let at_mouth = self.mouth_line.read(self.delay);
        let at_bell = self.bell_line.read(self.delay);
        // Bell (open end): lossy low-pass reflection with inversion.
        let bell_refl = -self.bell_filter(at_bell);

        // Reed (near-closed end): nonlinear pressure-controlled valve.
        let delta = breath - at_mouth;
        let h = reed_reflection(delta, self.slope);
        let mouth_refl = at_mouth + h * delta;

        self.bell_line.write_sample(mouth_refl);
        self.mouth_line.write_sample(bell_refl);

        flush_denormal(at_mouth * self.amplitude.next_sample())
    }
}

impl AudioNode for ReedWoodwindNode {
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
        self.mouth_line.clear();
        self.bell_line.clear();
        self.bell_filter_z1 = 0.0;
        self.breath_pressure = Smoothed::new(self.breath_pressure.target());
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
    fn render(node: &mut ReedWoodwindNode, frames: usize) -> Vec<Sample> {
        render_layout(node, frames, ChannelLayout::Mono).remove(0)
    }

    /// Renders `frames` into every channel of `layout` and returns per-channel
    /// sample vectors.
    fn render_layout(
        node: &mut ReedWoodwindNode,
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

    /// Magnitude of the Goertzel estimate at frequency `f` over `block`.
    fn goertzel(block: &[Sample], f: Sample, sr: Sample) -> f64 {
        let w = core::f32::consts::TAU * f / sr;
        let c = 2.0 * ops::cos(w);
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for &x in block {
            let s0 = f64::from(x) + f64::from(c) * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - f64::from(c) * s1 * s2).sqrt()
    }

    #[test]
    fn default_self_oscillates() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let block = render(&mut node, SR as usize);
        // Steady breath drives sustained self-oscillation, not a decaying pluck.
        assert!(peak(&block) > 0.0);
        assert!(peak(&block[SR as usize / 2..]) > 1e-2);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn long_run_stays_bounded() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let block = render(&mut node, 4 * SR as usize);
        assert!(block.iter().all(|s| s.is_finite()));
        // The clamped reed valve plus sub-unity bell loop gain bound the loop; it
        // must never run away.
        assert!(peak(&block) < 1.0);
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let mut b = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        assert_eq!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reset_restarts_identical_attack() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let first = render(&mut node, 8192);
        node.reset();
        let second = render(&mut node, 8192);
        assert_eq!(first, second);
    }

    #[test]
    fn zero_breath_pressure_is_silent() {
        let params = ReedWoodwindParams {
            breath_pressure: 0.0,
            ..ReedWoodwindParams::default()
        };
        let mut node = ReedWoodwindNode::new(SR, params);
        // With no breath and no stored energy the reed never opens: delta = 0,
        // h = REED_REST, mouth_refl = 0, so the bore stays silent forever.
        let block = render(&mut node, SR as usize / 2);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn amplitude_scales_output_energy() {
        let loud = ReedWoodwindParams {
            amplitude: 1.0,
            ..ReedWoodwindParams::default()
        };
        let soft = ReedWoodwindParams {
            amplitude: 0.5,
            ..ReedWoodwindParams::default()
        };
        let mut a = ReedWoodwindNode::new(SR, loud);
        let mut b = ReedWoodwindNode::new(SR, soft);
        let ea = energy(&render(&mut a, 8192));
        let eb = energy(&render(&mut b, 8192));
        // Amplitude only scales the radiated signal; halving it quarters energy.
        assert!(eb > 0.0);
        assert!((ea / eb - 4.0).abs() < 1e-3);
    }

    #[test]
    fn odd_harmonics_dominate() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let block = render(&mut node, 2 * SR as usize);
        // Analyze the settled second half so the attack transient is excluded.
        let tail = &block[SR as usize..];
        let f0 = DEFAULT_FREQUENCY_HZ;
        let h1 = goertzel(tail, f0, SR as Sample);
        let h2 = goertzel(tail, 2.0 * f0, SR as Sample);
        let h3 = goertzel(tail, 3.0 * f0, SR as Sample);
        // One inversion per round trip selects the odd series: the first and
        // third harmonics must tower over the (suppressed) even second.
        assert!(h1 > 0.0);
        assert!(h1 > 5.0 * h2, "fundamental not dominant: h1={h1} h2={h2}");
        assert!(h3 > 5.0 * h2, "third not dominant: h3={h3} h2={h2}");
    }

    #[test]
    fn mono_core_replicated_to_all_channels() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let chans = render_layout(&mut node, 4096, ChannelLayout::Stereo);
        assert_eq!(chans.len(), 2);
        assert_eq!(chans[0], chans[1]);
        assert!(peak(&chans[0]) > 0.0);
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let out = render(&mut node, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn getters_report_constructed_values() {
        let params = ReedWoodwindParams {
            frequency_hz: 196.0,
            breath_pressure: 0.6,
            reed_stiffness: 0.4,
            brightness: 0.7,
            amplitude: 0.8,
        };
        let node = ReedWoodwindNode::new(SR, params);
        assert!((node.frequency_hz() - 196.0).abs() < 1e-3);
        assert!((node.breath_pressure() - 0.6).abs() < 1e-6);
        assert!((node.reed_stiffness() - 0.4).abs() < 1e-6);
        assert!((node.brightness() - 0.7).abs() < 1e-6);
        assert!((node.amplitude() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn set_frequency_clamps_to_range() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        node.set_frequency_hz(1.0e9);
        assert!((node.frequency_hz() - (SR as Sample) * 0.5).abs() < 1e-3);
        node.set_frequency_hz(1.0);
        assert!((node.frequency_hz() - MIN_FREQUENCY_HZ).abs() < 1e-3);
    }

    #[test]
    fn set_reed_stiffness_and_brightness_clamp() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        node.set_reed_stiffness(5.0);
        assert!((node.reed_stiffness() - 1.0).abs() < 1e-6);
        node.set_reed_stiffness(-2.0);
        assert!((node.reed_stiffness() - 0.0).abs() < 1e-6);
        node.set_brightness(9.0);
        assert!((node.brightness() - 1.0).abs() < 1e-6);
        node.set_brightness(-9.0);
        assert!((node.brightness() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn set_breath_pressure_clamps_to_range() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        node.set_breath_pressure(5.0, Ramp::Immediate);
        assert!((node.breath_pressure() - 1.0).abs() < 1e-6);
        node.set_breath_pressure(-5.0, Ramp::Immediate);
        assert!((node.breath_pressure() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn setters_reject_non_finite() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        let f = node.frequency_hz();
        let st = node.reed_stiffness();
        let b = node.brightness();
        let bp = node.breath_pressure();
        let a = node.amplitude();
        node.set_frequency_hz(Sample::NAN);
        node.set_reed_stiffness(Sample::INFINITY);
        node.set_brightness(Sample::NEG_INFINITY);
        node.set_breath_pressure(Sample::NAN, Ramp::Immediate);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.frequency_hz(), f);
        assert_eq!(node.reed_stiffness(), st);
        assert_eq!(node.brightness(), b);
        assert_eq!(node.breath_pressure(), bp);
        assert_eq!(node.amplitude(), a);
    }

    #[test]
    fn constructor_sanitizes_non_finite_params() {
        let params = ReedWoodwindParams {
            frequency_hz: Sample::NAN,
            breath_pressure: Sample::INFINITY,
            reed_stiffness: Sample::NAN,
            brightness: Sample::NEG_INFINITY,
            amplitude: Sample::NAN,
        };
        let node = ReedWoodwindNode::new(SR, params);
        assert!(node.frequency_hz().is_finite());
        assert!(node.breath_pressure().is_finite());
        assert!(node.reed_stiffness().is_finite());
        assert!(node.brightness().is_finite());
        assert!(node.amplitude().is_finite());
    }

    #[test]
    fn frequency_changes_output() {
        let low = ReedWoodwindParams {
            frequency_hz: 110.0,
            ..ReedWoodwindParams::default()
        };
        let high = ReedWoodwindParams {
            frequency_hz: 440.0,
            ..ReedWoodwindParams::default()
        };
        let mut a = ReedWoodwindNode::new(SR, low);
        let mut b = ReedWoodwindNode::new(SR, high);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn reed_stiffness_changes_timbre() {
        let soft = ReedWoodwindParams {
            reed_stiffness: 0.1,
            ..ReedWoodwindParams::default()
        };
        let stiff = ReedWoodwindParams {
            reed_stiffness: 0.9,
            ..ReedWoodwindParams::default()
        };
        let mut a = ReedWoodwindNode::new(SR, soft);
        let mut b = ReedWoodwindNode::new(SR, stiff);
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn brightness_changes_output() {
        let dark = ReedWoodwindParams {
            brightness: 0.05,
            ..ReedWoodwindParams::default()
        };
        let bright = ReedWoodwindParams {
            brightness: 0.95,
            ..ReedWoodwindParams::default()
        };
        let mut a = ReedWoodwindNode::new(SR, dark);
        let mut b = ReedWoodwindNode::new(SR, bright);
        // The bell loss coefficient S tracks brightness, so the radiated
        // waveform must differ.
        assert_ne!(render(&mut a, 8192), render(&mut b, 8192));
    }

    #[test]
    fn high_and_low_frequencies_both_oscillate() {
        for &f in &[MIN_FREQUENCY_HZ, 1_500.0] {
            let params = ReedWoodwindParams {
                frequency_hz: f,
                ..ReedWoodwindParams::default()
            };
            let mut node = ReedWoodwindNode::new(SR, params);
            let block = render(&mut node, SR as usize);
            assert!(peak(&block) > 0.0, "silent at {f} Hz");
            assert!(block.iter().all(|s| s.is_finite()));
        }
    }

    #[test]
    fn breath_pressure_target_tracks_setter() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        node.set_breath_pressure(0.8, Ramp::linear_seconds(0.01, SR));
        assert!((node.breath_pressure() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn amplitude_target_tracks_setter() {
        let mut node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        node.set_amplitude(0.25, Ramp::linear_seconds(0.01, SR));
        assert!((node.amplitude() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn latency_is_zero() {
        let node = ReedWoodwindNode::new(SR, ReedWoodwindParams::default());
        assert_eq!(node.latency_frames(), 0);
    }
}
