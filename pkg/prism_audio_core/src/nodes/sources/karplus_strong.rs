//! Extended Karplus-Strong plucked-string physical-modeling source node.
//!
//! [`KarplusStrongNode`] is a *source* (zero inputs, one output) that synthesizes
//! a plucked-string tone from a short noise burst recirculating through a tuned,
//! damped feedback delay line. Unlike a geometric oscillator (which traces a
//! fixed waveform) or a PCM player (which replays recorded audio), a
//! Karplus-Strong string *is* its own physical model: the only energy injected
//! is the initial pluck, and the timbre emerges entirely from how the loop
//! filters and recirculates that energy.
//!
//! # The loop
//!
//! The recirculating loop is a pure integer delay line of `N` samples followed,
//! in the feedback path, by a one-zero damping filter, a first-order allpass
//! tuning filter, and a loop gain:
//!
//! ```text
//! delayed[n] = line[n - N]                              (integer delay line)
//! lp[n]      = (1 - S) * delayed[n] + S * delayed[n-1]  (one-zero loop damping)
//! ap[n]      = C * lp[n] + lp[n-1] - C * ap[n-1]        (first-order allpass)
//! line[n]    = g * ap[n]                                (loop gain, written back)
//! ```
//!
//! The audible fundamental is `f0 = sample_rate / D`, where `D` is the *total*
//! loop delay in samples. Three stages contribute to `D`:
//!
//! - the integer delay line contributes `N` samples;
//! - the one-zero damping filter `(1 - S) + S z^-1` has a phase delay of `S`
//!   samples at low frequency;
//! - the allpass `(C + z^-1) / (1 + C z^-1)` contributes a fractional phase
//!   delay of `eps` samples at low frequency, where `eps = (1 - C) / (1 + C)`.
//!
//! So `D = N + S + eps`. Given a desired `D = sample_rate / f0`, the node sets
//! `delta = D - S`, `N = floor(delta)`, `eps = delta - N` in `[0, 1)`, and
//! solves the allpass coefficient `C = (1 - eps) / (1 + eps)`. This splits the
//! required delay into an integer part (the delay line) and a sub-sample part
//! (the allpass), so the string tunes *continuously* rather than snapping to
//! integer-sample pitches.
//!
//! # Damping (brightness) and decay
//!
//! `brightness` in `[0, 1]` maps to the one-zero damping coefficient
//! `S = 0.5 * (1 - brightness)` in `[0, 0.5]`. At `brightness = 1` the damping
//! filter is the identity (`S = 0`): nothing but the loop gain attenuates the
//! partials, so the tone stays bright and metallic. At `brightness = 0` the
//! filter is the classic two-tap average `0.5 + 0.5 z^-1` (`S = 0.5`), whose
//! magnitude falls to zero at Nyquist, so upper partials decay much faster than
//! the fundamental and the tone darkens quickly, exactly as a real string loses
//! its highs first.
//!
//! Because the damping filter has unity gain at DC, the fundamental decays at
//! the pure loop gain `g` per round trip. To hit a `60 dB` decay time of
//! `decay_seconds`, the loop makes `decay_seconds * f0` round trips in that
//! window, so `g = 10^(-3 / (decay_seconds * f0))` (since `-60 dB` is a factor
//! of `10^-3`). The gain is clamped just below unity so the loop always decays.
//!
//! # Determinism
//!
//! The pluck excitation is a white-noise burst drawn from a self-contained
//! xorshift64 PRNG seeded via `SplitMix64`, so two nodes constructed with the
//! same `seed` and plucked identically emit bit-identical streams on every
//! platform via [`bevy_math::ops`]. This matters for reproducible mixes,
//! regression tests, and networked lockstep.
//!
//! # Relationship
//!
//! This is the *source* counterpart to the insert-style
//! [`CombResonatorNode`](crate::nodes::effects::CombResonatorNode): the comb
//! resonator rings an *external* input through a linearly interpolated,
//! lowpass-damped feedback comb, whereas this node is self-excited (an internal
//! pluck), extends the loop with an allpass tuning filter for sub-sample pitch
//! accuracy, exposes an explicit `60 dB` decay time rather than a raw feedback
//! coefficient, and shapes brightness with the Jaffe-Smith one-zero loop
//! filter. It reuses only this crate's own [`Sample`] type, [`Smoothed`]
//! parameter smoother, and denormal-flushing primitive.
//!
//! # Real-time contract
//!
//! The delay line is pre-allocated in [`KarplusStrongNode::new`] for the lowest
//! supported pitch, so [`process`](crate::graph::AudioNode::process) performs no
//! allocation, takes no locks, and cannot panic: a pluck refills a bounded
//! prefix of that buffer, every recirculated sample is denormal-flushed, the
//! loop gain is clamped below unity, and non-finite parameters are rejected at
//! the setters. The output amplitude is driven through a [`Smoothed`] value so
//! level automation never introduces zipper noise.
//!
//! # Provenance
//!
//! The plucked-string algorithm is that of K. Karplus and A. Strong ("Digital
//! Synthesis of Plucked-String and Drum Timbres", *Computer Music Journal*,
//! 1983); the tuning allpass, brightness loop filter, and decay-stretching
//! extensions are those of D. Jaffe and J. O. Smith ("Extensions of the
//! Karplus-Strong Plucked-String Algorithm", *Computer Music Journal*, 1983).
//! The first-order allpass fractional-delay interpolation follows the standard
//! treatment in J. O. Smith's *Physical Audio Signal Processing* (public online
//! text). The PRNG is Marsaglia's xorshift64 ("Xorshift RNGs", *Journal of
//! Statistical Software*, 2003) seeded via `SplitMix64` (S. Vigna,
//! public-domain reference). This module contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**; it is implemented purely from that publicly documented theory.

use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Lowest tunable fundamental in hertz. This bounds the pre-allocated delay
/// line length (`sample_rate / MIN_FREQUENCY_HZ` frames of maximum delay).
pub const MIN_FREQUENCY_HZ: Sample = 20.0;

/// Largest stable loop gain. Kept just below unity so a plucked string always
/// decays to silence instead of ringing forever.
pub const MAX_LOOP_GAIN: Sample = 0.9999;

/// Largest magnitude allowed for the allpass tuning coefficient. Clamping just
/// below unity keeps the allpass pole off the unit circle (the raw formula
/// yields `C = 1` only for a zero fractional delay, where the detuning from the
/// clamp is a few ten-thousandths of a sample, i.e. inaudible).
const MAX_ALLPASS_COEFF: Sample = 0.9995;

/// Smallest `60 dB` decay time in seconds. Guards the loop-gain formula against
/// a divide-by-zero and keeps even the shortest pluck audible for a frame.
const MIN_DECAY_SECONDS: Sample = 1.0e-3;

/// Returns `value` when finite, otherwise `fallback`. Guards the public setters
/// against `NaN`/infinity leaking into the loop state (a `NaN` would otherwise
/// survive `clamp`).
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Self-contained deterministic PRNG (Marsaglia xorshift64) seeded via
/// `SplitMix64`, used to synthesize the pluck noise burst.
#[derive(Debug, Clone, Copy)]
struct PluckRng {
    /// Current 64-bit generator state; kept non-zero by the seeding routine.
    state: u64,
}

impl PluckRng {
    /// Builds a generator whose state is diffused from `seed` via `SplitMix64`.
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: seed_to_state(seed),
        }
    }

    /// Advances the generator one step and returns the next 32-bit word.
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // Marsaglia's xorshift64 (shift triple 13/7/17), full period 2^64 - 1.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // The high 32 bits carry the best statistical quality for xorshift.
        (x >> 32) as u32
    }

    /// Returns the next white sample uniformly distributed in `[-1.0, 1.0)`.
    #[inline]
    fn next_bipolar(&mut self) -> Sample {
        let bits = self.next_u32();
        // Use the top 24 bits to form a float in [0, 1), then affine-map it to
        // [-1, 1). This keeps every excitation sample bounded by 1 in magnitude.
        let unit = (bits >> 8) as Sample * (1.0 / 16_777_216.0);
        unit * 2.0 - 1.0
    }
}

/// Diffuses a user seed into a non-zero `xorshift64` state via `SplitMix64`.
///
/// `SplitMix64` is a bijection, so distinct seeds map to distinct states (except
/// the single seed that would map to zero, which is remapped to a fixed golden
/// constant to keep the xorshift generator valid).
#[inline]
fn seed_to_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        z
    }
}

/// Configuration for a [`KarplusStrongNode`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct KarplusStrongParams {
    /// Plucked fundamental in hertz, clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`. The loop delay is
    /// `sample_rate / frequency_hz` frames.
    pub frequency_hz: Sample,
    /// `60 dB` decay time in seconds for the fundamental, clamped to at least
    /// `MIN_DECAY_SECONDS`. Longer values ring longer.
    pub decay_seconds: Sample,
    /// Brightness in `[0, 1]`. `1` leaves the loop undamped (bright, metallic,
    /// slow high-frequency decay); `0` applies the classic two-tap averaging
    /// filter (dark, fast high-frequency decay).
    pub brightness: Sample,
    /// Pluck strength in `[0, 1]`: the peak amplitude of the noise burst loaded
    /// into the delay line on each trigger.
    pub excitation: Sample,
    /// Settled linear output amplitude (a plain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for KarplusStrongParams {
    fn default() -> Self {
        Self {
            frequency_hz: 220.0,
            decay_seconds: 2.0,
            brightness: 0.5,
            excitation: 1.0,
            amplitude: 1.0,
        }
    }
}

/// An extended Karplus-Strong plucked-string source node.
///
/// Zero inputs, one output. The string is monophonic; every output channel
/// receives the same signal so stereo/surround downstream nodes see a coherent
/// voice. Call [`KarplusStrongNode::trigger`] to pluck; the string then rings
/// and decays on its own.
///
/// # Example
///
/// ```
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{KarplusStrongNode, KarplusStrongParams};
///
/// let mut node = KarplusStrongNode::new(48_000, 0x1234, KarplusStrongParams::default());
/// node.trigger();
///
/// let inputs: [AudioBuffer; 0] = [];
/// let mut out = AudioBuffer::new(ChannelLayout::Mono, 256);
/// out.set_active_frames(256);
/// let mut outputs = [out];
/// let ctx = RenderContext { sample_rate: 48_000, frames: 256, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // The pluck injects energy, so the block is no longer silent.
/// let energy: f32 = outputs[0].channel(0).iter().map(|s| s * s).sum();
/// assert!(energy > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct KarplusStrongNode {
    /// Sample rate the loop coefficients are derived against (set at
    /// construction, like the sibling resonator).
    sample_rate: Sample,
    /// Pre-allocated delay line; only the first `n` samples circulate.
    line: Vec<Sample>,
    /// Current integer loop-delay length in samples (`<= line.len()`).
    n: usize,
    /// Circulating read/write cursor into `line[0..n]`.
    pos: usize,
    /// Previous input to the one-zero damping filter (`delayed[n-1]`).
    damp_prev: Sample,
    /// Previous input to the allpass (`lp[n-1]`).
    ap_x_prev: Sample,
    /// Previous output of the allpass (`ap[n-1]`).
    ap_y_prev: Sample,
    /// Latched one-zero damping coefficient `S` in `[0, 0.5]`.
    damping: Sample,
    /// Latched allpass tuning coefficient `C` in `[0, MAX_ALLPASS_COEFF]`.
    allpass_coeff: Sample,
    /// Latched per-round-trip loop gain `g` in `[0, MAX_LOOP_GAIN]`.
    loop_gain: Sample,
    /// Deterministic excitation generator.
    rng: PluckRng,
    /// User-facing fundamental in hertz (latched into `n`/coefficients on the
    /// next trigger).
    frequency_hz: Sample,
    /// User-facing `60 dB` decay time in seconds.
    decay_seconds: Sample,
    /// User-facing brightness in `[0, 1]`.
    brightness: Sample,
    /// User-facing pluck strength in `[0, 1]`.
    excitation: Sample,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// `true` once a pluck has been requested but not yet applied in `process`.
    pending_trigger: bool,
    /// `true` while the string holds energy (set on trigger, cleared on reset).
    ringing: bool,
}

impl KarplusStrongNode {
    /// Builds a plucked string for `sample_rate` Hz with excitation `seed`.
    ///
    /// The delay line is sized so a fundamental as low as [`MIN_FREQUENCY_HZ`]
    /// fits. All parameters are sanitized: non-finite values fall back to safe
    /// defaults, `frequency_hz` is clamped to `[MIN_FREQUENCY_HZ, sample_rate /
    /// 2]`, `decay_seconds` to at least [`MIN_DECAY_SECONDS`], and
    /// `brightness`/`excitation` to `[0, 1]`. The string starts silent; call
    /// [`trigger`](Self::trigger) to pluck it.
    #[must_use]
    pub fn new(sample_rate: u32, seed: u64, params: KarplusStrongParams) -> Self {
        let sr = (sample_rate.max(1)) as Sample;
        // Longest delay we ever need (frames at the lowest supported pitch),
        // plus headroom for the one-sample damping and allpass memory.
        let max_delay_frames = ops::round(sr / MIN_FREQUENCY_HZ) as usize;
        let line_len = max_delay_frames + 2;
        let mut line = Vec::with_capacity(line_len);
        line.resize(line_len, 0.0);

        let frequency_hz = sanitize_frequency(params.frequency_hz, sr);
        let decay_seconds = finite_or(params.decay_seconds, 1.0).max(MIN_DECAY_SECONDS);
        let brightness = finite_or(params.brightness, 0.5).clamp(0.0, 1.0);
        let excitation = finite_or(params.excitation, 1.0).clamp(0.0, 1.0);
        let amplitude = finite_or(params.amplitude, 1.0);

        let mut node = Self {
            sample_rate: sr,
            line,
            n: 2,
            pos: 0,
            damp_prev: 0.0,
            ap_x_prev: 0.0,
            ap_y_prev: 0.0,
            damping: 0.25,
            allpass_coeff: 0.0,
            loop_gain: 0.0,
            rng: PluckRng::new(seed),
            frequency_hz,
            decay_seconds,
            brightness,
            excitation,
            amplitude: Smoothed::new(amplitude),
            pending_trigger: false,
            ringing: false,
        };
        node.recompute_coefficients();
        node
    }

    /// Requests a pluck. The noise burst and freshly latched loop coefficients
    /// are applied at the start of the next [`process`](AudioNode::process)
    /// call, so a trigger is sample-accurate to the block boundary and never
    /// touches the delay line from the control thread.
    #[inline]
    pub fn trigger(&mut self) {
        self.pending_trigger = true;
    }

    /// Retunes the string to `frequency_hz` (clamped to
    /// `[MIN_FREQUENCY_HZ, sample_rate / 2]`). The new pitch is latched on the
    /// next [`trigger`](Self::trigger); a sounding string is left undisturbed so
    /// retuning never glitches the current note.
    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: Sample) {
        self.frequency_hz = sanitize_frequency(frequency_hz, self.sample_rate);
    }

    /// Sets the `60 dB` decay time in seconds (clamped to at least
    /// [`MIN_DECAY_SECONDS`]). Latched on the next [`trigger`](Self::trigger).
    #[inline]
    pub fn set_decay_seconds(&mut self, decay_seconds: Sample) {
        self.decay_seconds = finite_or(decay_seconds, self.decay_seconds).max(MIN_DECAY_SECONDS);
    }

    /// Sets the brightness in `[0, 1]`. Latched on the next
    /// [`trigger`](Self::trigger).
    #[inline]
    pub fn set_brightness(&mut self, brightness: Sample) {
        self.brightness = finite_or(brightness, self.brightness).clamp(0.0, 1.0);
    }

    /// Sets the pluck strength in `[0, 1]`. Applies to the next
    /// [`trigger`](Self::trigger).
    #[inline]
    pub fn set_excitation(&mut self, excitation: Sample) {
        self.excitation = finite_or(excitation, self.excitation).clamp(0.0, 1.0);
    }

    /// Sets a new target output amplitude (linear), gliding toward it with
    /// `ramp` to avoid zipper noise.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(linear, ramp);
    }

    /// Returns the plucked fundamental in hertz.
    #[inline]
    #[must_use]
    pub fn frequency_hz(&self) -> Sample {
        self.frequency_hz
    }

    /// Returns the `60 dB` decay time in seconds.
    #[inline]
    #[must_use]
    pub fn decay_seconds(&self) -> Sample {
        self.decay_seconds
    }

    /// Returns the brightness in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn brightness(&self) -> Sample {
        self.brightness
    }

    /// Returns the pluck strength in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn excitation(&self) -> Sample {
        self.excitation
    }

    /// Returns the latched per-round-trip loop gain `g`.
    #[inline]
    #[must_use]
    pub fn loop_gain(&self) -> Sample {
        self.loop_gain
    }

    /// Returns the latched integer loop-delay length in samples.
    #[inline]
    #[must_use]
    pub fn delay_len(&self) -> usize {
        self.n
    }

    /// Returns `true` while the string holds energy (plucked and not reset).
    #[inline]
    #[must_use]
    pub fn is_ringing(&self) -> bool {
        self.ringing
    }

    /// Recomputes the latched loop coefficients (`n`, `damping`,
    /// `allpass_coeff`, `loop_gain`) from the current user-facing parameters.
    fn recompute_coefficients(&mut self) {
        let sr = self.sample_rate;
        let f0 = self.frequency_hz;
        // One-zero damping coefficient S in [0, 0.5]: brightness 1 -> S 0.
        let s = 0.5 * (1.0 - self.brightness);
        self.damping = s;

        // Total loop delay D = sr / f0 split as D = N + S + eps.
        let d = sr / f0;
        let max_delta = (self.line.len() - 1) as Sample;
        let delta = (d - s).clamp(2.0, max_delta);
        let n = ops::floor(delta);
        let eps = delta - n;
        self.n = n as usize;

        // First-order allpass phase delay eps -> coefficient C = (1-eps)/(1+eps).
        let c = ((1.0 - eps) / (1.0 + eps)).clamp(0.0, MAX_ALLPASS_COEFF);
        self.allpass_coeff = c;

        // Loop gain for a 60 dB (factor 10^-3) decay over decay_seconds, during
        // which the loop makes decay_seconds * f0 round trips.
        let trips = self.decay_seconds * f0;
        let g = ops::powf(10.0, -3.0 / trips).clamp(0.0, MAX_LOOP_GAIN);
        self.loop_gain = g;
    }

    /// Loads a fresh noise burst into `line[0..n]` and clears the filter memory.
    fn pluck(&mut self) {
        self.recompute_coefficients();
        let amp = self.excitation;
        for slot in self.line.iter_mut().take(self.n) {
            *slot = self.rng.next_bipolar() * amp;
        }
        self.pos = 0;
        self.damp_prev = 0.0;
        self.ap_x_prev = 0.0;
        self.ap_y_prev = 0.0;
        self.ringing = true;
    }

    /// Advances the loop one sample and returns the value read this step (the
    /// pre-overwrite delay-line output, i.e. the string's radiated sample).
    #[inline]
    fn next_sample(&mut self) -> Sample {
        let pos = self.pos;
        let delayed = self.line[pos];

        // One-zero loop damping filter: lp = (1-S)*delayed + S*delayed[n-1].
        let s = self.damping;
        let lp = (1.0 - s) * delayed + s * self.damp_prev;
        self.damp_prev = delayed;

        // First-order allpass tuning: ap = C*lp + lp[n-1] - C*ap[n-1].
        let c = self.allpass_coeff;
        let ap = c * lp + self.ap_x_prev - c * self.ap_y_prev;
        self.ap_x_prev = lp;
        self.ap_y_prev = ap;

        // Apply loop gain and write back; flush denormals to dodge CPU stalls.
        self.line[pos] = flush_denormal(self.loop_gain * ap);
        self.pos = if pos + 1 == self.n { 0 } else { pos + 1 };
        delayed
    }
}

/// Clamps `frequency_hz` to the node's playable range `[MIN_FREQUENCY_HZ,
/// sample_rate / 2]`, substituting [`MIN_FREQUENCY_HZ`] for non-finite input.
#[inline]
fn sanitize_frequency(frequency_hz: Sample, sample_rate: Sample) -> Sample {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQUENCY_HZ);
    finite_or(frequency_hz, MIN_FREQUENCY_HZ).clamp(MIN_FREQUENCY_HZ, nyquist)
}

impl AudioNode for KarplusStrongNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let channels = out.channels();
        if channels == 0 {
            return;
        }

        // Apply a pending pluck exactly at the block boundary.
        if self.pending_trigger {
            self.pending_trigger = false;
            self.pluck();
        }

        // Synthesize the mono voice into channel 0, advancing the amplitude
        // smoother once per frame.
        {
            let buf = out.channel_mut(0);
            for sample in buf.iter_mut() {
                let amp = self.amplitude.next_sample();
                *sample = self.next_sample() * amp;
            }
        }

        // Replicate the mono voice into every remaining channel.
        for ch in 1..channels {
            let (src, dst) = out.channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        for slot in &mut self.line {
            *slot = 0.0;
        }
        self.pos = 0;
        self.damp_prev = 0.0;
        self.ap_x_prev = 0.0;
        self.ap_y_prev = 0.0;
        self.pending_trigger = false;
        self.ringing = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 44_100;

    /// Renders `frames` of a freshly plucked mono string into a flat vector.
    fn render(node: &mut KarplusStrongNode, frames: usize) -> Vec<Sample> {
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames.max(1));
        out.set_active_frames(frames);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        outputs[0].channel(0).to_vec()
    }

    fn energy(block: &[Sample]) -> f64 {
        block.iter().map(|&s| (s as f64) * (s as f64)).sum()
    }

    /// Crude high-frequency energy proxy: energy of the first difference.
    fn hf_energy(block: &[Sample]) -> f64 {
        block
            .windows(2)
            .map(|w| {
                let d = (w[1] - w[0]) as f64;
                d * d
            })
            .sum()
    }

    #[test]
    fn silent_before_trigger() {
        let mut node = KarplusStrongNode::new(SR, 1, KarplusStrongParams::default());
        let block = render(&mut node, 512);
        assert!(block.iter().all(|&s| s == 0.0));
        assert!(!node.is_ringing());
    }

    #[test]
    fn pluck_injects_energy() {
        let mut node = KarplusStrongNode::new(SR, 7, KarplusStrongParams::default());
        node.trigger();
        let block = render(&mut node, 512);
        assert!(energy(&block) > 0.0);
        assert!(node.is_ringing());
    }

    #[test]
    fn pitch_matches_fundamental() {
        // f0 = 441 Hz at 44100 -> loop delay exactly 100 samples.
        let params = KarplusStrongParams {
            frequency_hz: 441.0,
            decay_seconds: 5.0,
            brightness: 1.0,
            ..KarplusStrongParams::default()
        };
        let mut node = KarplusStrongNode::new(SR, 42, params);
        assert_eq!(node.delay_len(), 100);
        node.trigger();
        let signal = render(&mut node, 8192);

        // Autocorrelation over an early, energetic window; the peak lag in a
        // neighborhood of the expected period marks the fundamental.
        let win = 4096;
        let mut best_lag = 0usize;
        let mut best = f64::NEG_INFINITY;
        for lag in 60..=160 {
            let mut acc = 0.0f64;
            for i in 0..win {
                acc += (signal[i] as f64) * (signal[i + lag] as f64);
            }
            if acc > best {
                best = acc;
                best_lag = lag;
            }
        }
        assert!(
            (best_lag as isize - 100).abs() <= 2,
            "expected period ~100, got {best_lag}"
        );
    }

    #[test]
    fn energy_decays_block_over_block() {
        let params = KarplusStrongParams {
            frequency_hz: 220.0,
            decay_seconds: 3.0,
            brightness: 1.0,
            ..KarplusStrongParams::default()
        };
        let mut node = KarplusStrongNode::new(SR, 3, params);
        node.trigger();
        let mut prev = f64::INFINITY;
        for _ in 0..8 {
            let e = energy(&render(&mut node, 2048));
            assert!(e <= prev + 1e-9, "block energy rose: {e} > {prev}");
            prev = e;
        }
    }

    #[test]
    fn brighter_strings_retain_highs_longer() {
        let base = KarplusStrongParams {
            frequency_hz: 196.0,
            decay_seconds: 6.0,
            ..KarplusStrongParams::default()
        };
        let bright_params = KarplusStrongParams {
            brightness: 1.0,
            ..base
        };
        let dark_params = KarplusStrongParams {
            brightness: 0.0,
            ..base
        };

        let mut bright = KarplusStrongNode::new(SR, 11, bright_params);
        let mut dark = KarplusStrongNode::new(SR, 11, dark_params);
        bright.trigger();
        dark.trigger();
        let bright_sig = render(&mut bright, 44_100);
        let dark_sig = render(&mut dark, 44_100);

        let early = 1000..3000;
        let late = 20_000..22_000;
        let bright_ratio = hf_energy(&bright_sig[late.clone()]) / hf_energy(&bright_sig[early.clone()]);
        let dark_ratio = hf_energy(&dark_sig[late]) / hf_energy(&dark_sig[early]);
        assert!(
            bright_ratio > dark_ratio,
            "bright HF ratio {bright_ratio} should exceed dark {dark_ratio}"
        );
    }

    #[test]
    fn deterministic_for_equal_seed() {
        let params = KarplusStrongParams::default();
        let mut a = KarplusStrongNode::new(SR, 0xABCD, params);
        let mut b = KarplusStrongNode::new(SR, 0xABCD, params);
        a.trigger();
        b.trigger();
        assert_eq!(render(&mut a, 4096), render(&mut b, 4096));
    }

    #[test]
    fn retrigger_refills_energy() {
        let params = KarplusStrongParams {
            frequency_hz: 220.0,
            decay_seconds: 0.3,
            brightness: 0.6,
            ..KarplusStrongParams::default()
        };
        let mut node = KarplusStrongNode::new(SR, 5, params);
        node.trigger();
        let _ = render(&mut node, 20_000);
        let tail = energy(&render(&mut node, 1024));
        node.trigger();
        let fresh = energy(&render(&mut node, 1024));
        assert!(fresh > tail, "retrigger energy {fresh} should exceed tail {tail}");
    }

    #[test]
    fn longer_decay_rings_longer() {
        let short_params = KarplusStrongParams {
            frequency_hz: 220.0,
            decay_seconds: 0.4,
            brightness: 1.0,
            ..KarplusStrongParams::default()
        };
        let long_params = KarplusStrongParams {
            decay_seconds: 5.0,
            ..short_params
        };
        let mut short = KarplusStrongNode::new(SR, 9, short_params);
        let mut long = KarplusStrongNode::new(SR, 9, long_params);
        short.trigger();
        long.trigger();
        let short_tail = energy(&render(&mut short, 44_100)[40_000..44_100]);
        let long_tail = energy(&render(&mut long, 44_100)[40_000..44_100]);
        assert!(
            long_tail > short_tail,
            "long-decay tail {long_tail} should exceed short {short_tail}"
        );
    }

    #[test]
    fn non_finite_parameters_are_safe() {
        let params = KarplusStrongParams {
            frequency_hz: Sample::NAN,
            decay_seconds: Sample::INFINITY,
            brightness: Sample::NAN,
            excitation: Sample::INFINITY,
            amplitude: Sample::NAN,
        };
        let mut node = KarplusStrongNode::new(SR, 1, params);
        node.set_frequency_hz(Sample::NAN);
        node.set_decay_seconds(Sample::NEG_INFINITY);
        node.set_brightness(Sample::NAN);
        node.set_excitation(Sample::INFINITY);
        node.trigger();
        let block = render(&mut node, 2048);
        assert!(block.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn zero_frames_is_safe() {
        let mut node = KarplusStrongNode::new(SR, 1, KarplusStrongParams::default());
        node.trigger();
        let block = render(&mut node, 0);
        assert!(block.is_empty());
    }

    #[test]
    fn stereo_channels_are_identical() {
        let mut node = KarplusStrongNode::new(SR, 2, KarplusStrongParams::default());
        node.trigger();
        let inputs: [AudioBuffer; 0] = [];
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 256);
        out.set_active_frames(256);
        let mut outputs = [out];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: 256,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        let left = outputs[0].channel(0).to_vec();
        let right = outputs[0].channel(1).to_vec();
        assert_eq!(left, right);
        assert!(energy(&left) > 0.0);
    }

    #[test]
    fn reset_silences_the_string() {
        let mut node = KarplusStrongNode::new(SR, 1, KarplusStrongParams::default());
        node.trigger();
        let _ = render(&mut node, 512);
        node.reset();
        assert!(!node.is_ringing());
        let block = render(&mut node, 512);
        assert!(block.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn frequency_is_clamped_to_range() {
        let mut node = KarplusStrongNode::new(SR, 1, KarplusStrongParams::default());
        node.set_frequency_hz(5.0);
        assert!(node.frequency_hz() >= MIN_FREQUENCY_HZ);
        node.set_frequency_hz(1.0e9);
        assert!(node.frequency_hz() <= (SR as Sample) * 0.5);
    }

    #[test]
    fn loop_gain_is_below_unity() {
        let params = KarplusStrongParams {
            decay_seconds: 1.0e6,
            ..KarplusStrongParams::default()
        };
        let node = KarplusStrongNode::new(SR, 1, params);
        assert!(node.loop_gain() <= MAX_LOOP_GAIN);
        assert!(node.loop_gain() > 0.0);
    }
}
