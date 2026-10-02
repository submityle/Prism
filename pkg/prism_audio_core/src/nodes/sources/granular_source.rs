//! Granular synthesis source node: a cloud of windowed sinusoidal grains.
//!
//! A [`GranularSourceNode`] is a zero-input, one-output source that emits a
//! continuous *grain cloud*. New grains are spawned at a controllable rate and
//! each grain is a short, Hann-windowed sinusoid (a Gabor atom); summed
//! together the overlapping grains form evolving, textural timbres that range
//! from a smooth drone (many overlapping grains) through shimmering clouds to a
//! sparse, pointillistic stream (few, isolated grains).
//!
//! # Model
//!
//! Each grain `g` is the product of a raised-cosine (Hann) window and a sine
//! carrier:
//!
//! ```text
//! grain_g(n) = w(phi_g) * sin(2*pi * theta_g)
//! w(phi)     = 0.5 - 0.5*cos(2*pi * phi)      (Hann window, phi in [0, 1))
//! ```
//!
//! where `theta_g` advances by `f_g / sample_rate` per sample (the grain's
//! carrier phase) and `phi_g` advances by `1 / lifetime_g` per sample across
//! the grain's lifetime (so `w` opens from and returns to zero, giving each
//! grain a click-free attack and release). A grain retires once `phi_g`
//! reaches `1`.
//!
//! A scheduler phase accumulator advances by `density / sample_rate` each
//! sample; every time it wraps past `1` a fresh grain is allocated from a
//! fixed-size pool (sized by [`MAX_GRAINS`]) and seeded with randomized
//! parameters drawn from a deterministic PRNG:
//!
//! - carrier frequency `f_g = frequency * 2^(u * pitch_spread / 12)` where
//!   `u` is a uniform draw in `[-1, 1]`, so the spread is specified in
//!   semitones and is musically symmetric in pitch rather than linear hertz;
//! - lifetime `grain_ms * (1 + u * duration_jitter)` (clamped positive);
//! - stereo pan `u * pan_spread`, mapped through an equal-power law so a wide
//!   `pan_spread` scatters grains across the stereo field.
//!
//! The summed cloud is scaled by `1 / sqrt(max(1, density * grain_seconds))`,
//! a power-preserving normalization: because the (randomized, decorrelated)
//! grains add incoherently their RMS grows like the square root of the mean
//! overlap count, so dividing by that square root keeps the overall level
//! roughly constant as `density` or `grain_ms` change. Momentary peaks can
//! still exceed unity when many grains align, so route the output through a
//! limiter if a hard ceiling is required (consistent with the other `sources`,
//! which do not clamp).
//!
//! # Relationship
//!
//! This node is a *pure-synthesis* grain cloud: its grains are generated from
//! an internal sine carrier, so it needs no sample buffer and is a true source.
//! That distinguishes it from the capture-based granulator in
//! [`crate::nodes::effects::granular`], which slices grains out of a live input
//! signal (and is therefore an effect, not a source). It also differs from the
//! deterministic partials of
//! [`super::additive_oscillator::AdditiveOscillatorNode`] (a fixed harmonic
//! sum) and the detuned unison of [`super::supersaw::SupersawNode`] (seven
//! continuously running saws): here the character comes from the stochastic
//! birth, lifetime, pitch, and placement of many transient atoms. For a single
//! steady tone use [`super::oscillator::OscillatorNode`] instead.
//!
//! # Determinism
//!
//! The grain scheduler draws from a self-contained `xorshift64` PRNG seeded
//! through `SplitMix64`, so two [`GranularSourceNode`]s built with the same
//! `seed` and parameters emit bit-identical streams on every platform. This
//! keeps clouds reproducible for mixes, regression tests, and networked
//! lockstep.
//!
//! # Real-time contract
//!
//! The grain pool is a fixed-size array sized at compile time, so
//! [`GranularSourceNode::process`] performs no allocation, locking, or
//! panicking: spawning a grain reuses a retired slot (or is dropped if the pool
//! is saturated), and every per-sample step is a bounded set of multiply-adds
//! plus two transcendental evaluations per active grain. `amplitude` is driven
//! through a [`Smoothed`] value so level automation never zippers; the
//! frequency, density, grain-length, and spread controls are plain scalars
//! because they only influence future grains or continuous phase slopes and
//! therefore never introduce a discontinuity in the running output.
//!
//! # Provenance
//!
//! Implemented from first principles from the public granular-synthesis
//! literature -- D. Gabor's "acoustical quanta" (1947) and C. Roads,
//! *Microsound* (MIT Press, 2001) for the grain-cloud model, with the standard
//! Hann window and equal-power pan law. The PRNG is Marsaglia's `xorshift64`
//! ("Xorshift RNGs", *Journal of Statistical Software*, 2003) seeded via
//! S. Vigna's public-domain `SplitMix64`. Nothing here is derived from any
//! AI/ML technique, nor from the source code of Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, or Web Audio; only the
//! shared mathematical ideas are referenced.

use bevy_math::ops;
use core::f32::consts::TAU;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, equal_power_pan, flush_denormal};
use crate::param::{Ramp, Smoothed};

/// Maximum number of grains that can sound simultaneously. Spawns beyond this
/// are dropped, bounding the per-sample work.
pub const MAX_GRAINS: usize = 64;

/// Default carrier center frequency in hertz.
pub const DEFAULT_GRANULAR_FREQUENCY_HZ: Sample = 220.0;

/// Default grain spawn rate in grains per second.
pub const DEFAULT_GRANULAR_DENSITY_HZ: Sample = 40.0;

/// Default grain length in milliseconds.
pub const DEFAULT_GRANULAR_GRAIN_MS: Sample = 60.0;

/// Default carrier pitch spread in semitones (uniform, +/- this value).
pub const DEFAULT_GRANULAR_PITCH_SPREAD: Sample = 0.0;

/// Default stereo pan spread in `[0, 1]` (0 = centered, 1 = full width).
pub const DEFAULT_GRANULAR_PAN_SPREAD: Sample = 0.0;

/// Default grain-length jitter in `[0, 1]` (fractional randomization).
pub const DEFAULT_GRANULAR_DURATION_JITTER: Sample = 0.0;

/// Default linear output amplitude.
pub const DEFAULT_GRANULAR_AMPLITUDE: Sample = 1.0;

/// Default PRNG seed.
pub const DEFAULT_GRANULAR_SEED: u64 = 0x5265_736F_6E61_6E63; // "Resonanc"

/// Maximum spawn rate in grains per second (bounds scheduler work).
pub const MAX_GRANULAR_DENSITY_HZ: Sample = 2_000.0;

/// Minimum grain length in milliseconds (keeps the window well-defined).
pub const MIN_GRANULAR_GRAIN_MS: Sample = 1.0;

/// Maximum grain length in milliseconds.
pub const MAX_GRANULAR_GRAIN_MS: Sample = 2_000.0;

/// Maximum carrier pitch spread in semitones.
pub const MAX_GRANULAR_PITCH_SPREAD: Sample = 48.0;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() { value } else { fallback }
}

/// Wraps a normalized phase accumulator back into `[0, 1)` without a transcendental.
#[inline]
fn wrap01(phase: Sample) -> Sample {
    let p = phase - (phase as i64 as Sample);
    if p < 0.0 { p + 1.0 } else { p }
}

/// Self-contained deterministic PRNG (Marsaglia `xorshift64`) seeded via
/// `SplitMix64`, matching the convention used by the other `sources`.
#[derive(Debug, Clone)]
struct GrainRng {
    /// Current 64-bit state; kept non-zero by the seeding routine.
    state: u64,
}

impl GrainRng {
    /// Builds a generator whose state is diffused from `seed` via `SplitMix64`.
    #[inline]
    fn new(seed: u64) -> Self {
        Self { state: seed_to_state(seed) }
    }

    /// Advances the generator and returns the next 32-bit word.
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

    /// Returns the next sample uniformly distributed in `[0, 1)`.
    #[inline]
    fn next_unit(&mut self) -> Sample {
        // 24-bit mantissa worth of entropy mapped to [0, 1).
        (self.next_u32() >> 8) as Sample / (1u32 << 24) as Sample
    }

    /// Returns the next sample uniformly distributed in `[-1, 1)`.
    #[inline]
    fn next_bipolar(&mut self) -> Sample {
        2.0 * self.next_unit() - 1.0
    }
}

/// Diffuses a user seed into a non-zero `xorshift64` state via `SplitMix64`.
#[inline]
fn seed_to_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Guarantee a non-zero state (xorshift cannot leave the all-zero state).
    if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z }
}

/// A single live grain: a Hann-windowed sine carrier with a fixed pan.
#[derive(Debug, Clone, Copy, Default)]
struct Grain {
    /// Whether this slot is currently sounding.
    active: bool,
    /// Carrier phase in `[0, 1)`.
    osc_phase: Sample,
    /// Carrier phase increment per sample (`f_g / sample_rate`).
    osc_inc: Sample,
    /// Window phase in `[0, 1)` across the grain's lifetime.
    win_phase: Sample,
    /// Window phase increment per sample (`1 / lifetime_samples`).
    win_inc: Sample,
    /// Left-channel gain from the equal-power pan law.
    left_gain: Sample,
    /// Right-channel gain from the equal-power pan law.
    right_gain: Sample,
}

/// Construction parameters for a [`GranularSourceNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GranularSourceParams {
    /// Carrier center frequency in hertz. Clamped non-negative.
    pub frequency_hz: Sample,
    /// Grain spawn rate in grains per second. Clamped to `[0, MAX]`.
    pub density_hz: Sample,
    /// Grain length in milliseconds. Clamped to `[MIN, MAX]`.
    pub grain_ms: Sample,
    /// Carrier pitch spread in semitones (uniform, +/- this). Clamped to `[0, MAX]`.
    pub pitch_spread: Sample,
    /// Stereo pan spread in `[0, 1]` (0 = centered, 1 = full width).
    pub pan_spread: Sample,
    /// Grain-length jitter in `[0, 1]` (fractional randomization).
    pub duration_jitter: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
    /// PRNG seed; the same seed replays an identical grain stream.
    pub seed: u64,
}

impl Default for GranularSourceParams {
    fn default() -> Self {
        Self {
            frequency_hz: DEFAULT_GRANULAR_FREQUENCY_HZ,
            density_hz: DEFAULT_GRANULAR_DENSITY_HZ,
            grain_ms: DEFAULT_GRANULAR_GRAIN_MS,
            pitch_spread: DEFAULT_GRANULAR_PITCH_SPREAD,
            pan_spread: DEFAULT_GRANULAR_PAN_SPREAD,
            duration_jitter: DEFAULT_GRANULAR_DURATION_JITTER,
            amplitude: DEFAULT_GRANULAR_AMPLITUDE,
            seed: DEFAULT_GRANULAR_SEED,
        }
    }
}

impl GranularSourceParams {
    /// Returns a copy with every field coerced into its valid range and all
    /// non-finite values replaced by their defaults.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            frequency_hz: finite_or(self.frequency_hz, DEFAULT_GRANULAR_FREQUENCY_HZ).max(0.0),
            density_hz: finite_or(self.density_hz, DEFAULT_GRANULAR_DENSITY_HZ)
                .clamp(0.0, MAX_GRANULAR_DENSITY_HZ),
            grain_ms: finite_or(self.grain_ms, DEFAULT_GRANULAR_GRAIN_MS)
                .clamp(MIN_GRANULAR_GRAIN_MS, MAX_GRANULAR_GRAIN_MS),
            pitch_spread: finite_or(self.pitch_spread, DEFAULT_GRANULAR_PITCH_SPREAD)
                .clamp(0.0, MAX_GRANULAR_PITCH_SPREAD),
            pan_spread: finite_or(self.pan_spread, DEFAULT_GRANULAR_PAN_SPREAD).clamp(0.0, 1.0),
            duration_jitter: finite_or(self.duration_jitter, DEFAULT_GRANULAR_DURATION_JITTER)
                .clamp(0.0, 1.0),
            amplitude: finite_or(self.amplitude, DEFAULT_GRANULAR_AMPLITUDE),
            seed: self.seed,
        }
    }
}

/// A granular synthesis source node (0 inputs, 1 output).
///
/// Channel 0 carries the left mix and channel 1 the right mix; a mono layout
/// receives the energy-preserving `(left + right) * SQRT_HALF` downmix, and any
/// channels beyond the first two receive that same mono downmix.
///
/// # Examples
///
/// ```
/// use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
/// use prism_audio_core::nodes::sources::{GranularSourceNode, GranularSourceParams};
/// use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
///
/// let mut node = GranularSourceNode::from_params(GranularSourceParams {
///     frequency_hz: 220.0,
///     density_hz: 80.0,
///     grain_ms: 50.0,
///     ..GranularSourceParams::default()
/// });
/// let inputs: [AudioBuffer; 0] = [];
/// let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 4_800)];
/// outputs[0].set_active_frames(4_800);
/// let ctx = RenderContext { sample_rate: 48_000, frames: 4_800, playhead: 0 };
/// let mut io = ProcessIo::new(&inputs, &mut outputs);
/// node.process(&ctx, &mut io);
///
/// // A dense cloud is not silent and stays finite.
/// let peak = outputs[0].channel(0).iter().fold(0.0_f32, |m, s| m.max(s.abs()));
/// assert!(peak > 0.0 && peak.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct GranularSourceNode {
    /// Carrier center frequency in hertz. Always non-negative.
    frequency: Sample,
    /// Grain spawn rate in grains per second, in `[0, MAX]`.
    density: Sample,
    /// Grain length in milliseconds, in `[MIN, MAX]`.
    grain_ms: Sample,
    /// Carrier pitch spread in semitones, in `[0, MAX]`.
    pitch_spread: Sample,
    /// Stereo pan spread in `[0, 1]`.
    pan_spread: Sample,
    /// Grain-length jitter in `[0, 1]`.
    duration_jitter: Sample,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Fixed-size grain pool.
    grains: [Grain; MAX_GRAINS],
    /// Scheduler phase accumulator in `[0, 1)`; a wrap spawns a grain.
    spawn_phase: Sample,
    /// Deterministic grain-parameter PRNG.
    rng: GrainRng,
    /// Original seed, retained so [`AudioNode::reset`] replays the same stream.
    seed: u64,
}

/// `sqrt(1/2)`, the energy-preserving mono-downmix coefficient.
const SQRT_HALF: Sample = core::f32::consts::FRAC_1_SQRT_2;

impl GranularSourceNode {
    /// Creates a granular source from a [`GranularSourceParams`] bundle.
    ///
    /// All parameters are sanitized via [`GranularSourceParams::sanitised`].
    #[must_use]
    pub fn from_params(params: GranularSourceParams) -> Self {
        let p = params.sanitised();
        Self {
            frequency: p.frequency_hz,
            density: p.density_hz,
            grain_ms: p.grain_ms,
            pitch_spread: p.pitch_spread,
            pan_spread: p.pan_spread,
            duration_jitter: p.duration_jitter,
            amplitude: Smoothed::new(p.amplitude),
            grains: [Grain::default(); MAX_GRAINS],
            spawn_phase: 0.0,
            rng: GrainRng::new(p.seed),
            seed: p.seed,
        }
    }

    /// Carrier center frequency in hertz.
    #[inline]
    #[must_use]
    pub fn frequency(&self) -> Sample {
        self.frequency
    }

    /// Grain spawn rate in grains per second.
    #[inline]
    #[must_use]
    pub fn density(&self) -> Sample {
        self.density
    }

    /// Grain length in milliseconds.
    #[inline]
    #[must_use]
    pub fn grain_ms(&self) -> Sample {
        self.grain_ms
    }

    /// Carrier pitch spread in semitones.
    #[inline]
    #[must_use]
    pub fn pitch_spread(&self) -> Sample {
        self.pitch_spread
    }

    /// Stereo pan spread in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn pan_spread(&self) -> Sample {
        self.pan_spread
    }

    /// Grain-length jitter in `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn duration_jitter(&self) -> Sample {
        self.duration_jitter
    }

    /// Target output amplitude.
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Sets the carrier center frequency in hertz (clamped non-negative).
    ///
    /// Click-free: it only affects grains spawned after the change.
    #[inline]
    pub fn set_frequency(&mut self, hz: Sample) {
        self.frequency = finite_or(hz, self.frequency).max(0.0);
    }

    /// Sets the grain spawn rate in grains per second (clamped to `[0, MAX]`).
    #[inline]
    pub fn set_density(&mut self, density_hz: Sample) {
        self.density = finite_or(density_hz, self.density).clamp(0.0, MAX_GRANULAR_DENSITY_HZ);
    }

    /// Sets the grain length in milliseconds (clamped to `[MIN, MAX]`).
    #[inline]
    pub fn set_grain_ms(&mut self, grain_ms: Sample) {
        self.grain_ms =
            finite_or(grain_ms, self.grain_ms).clamp(MIN_GRANULAR_GRAIN_MS, MAX_GRANULAR_GRAIN_MS);
    }

    /// Sets the carrier pitch spread in semitones (clamped to `[0, MAX]`).
    #[inline]
    pub fn set_pitch_spread(&mut self, semitones: Sample) {
        self.pitch_spread =
            finite_or(semitones, self.pitch_spread).clamp(0.0, MAX_GRANULAR_PITCH_SPREAD);
    }

    /// Sets the stereo pan spread in `[0, 1]`.
    #[inline]
    pub fn set_pan_spread(&mut self, spread: Sample) {
        self.pan_spread = finite_or(spread, self.pan_spread).clamp(0.0, 1.0);
    }

    /// Sets the grain-length jitter in `[0, 1]`.
    #[inline]
    pub fn set_duration_jitter(&mut self, jitter: Sample) {
        self.duration_jitter = finite_or(jitter, self.duration_jitter).clamp(0.0, 1.0);
    }

    /// Sets the target output amplitude, smoothing over `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, amplitude: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(amplitude, self.amplitude.target()), ramp);
    }

    /// Grain duration in seconds, honoring the millisecond clamp.
    #[inline]
    fn grain_seconds(&self) -> Sample {
        self.grain_ms * 0.001
    }

    /// Power-preserving normalization factor for the current overlap.
    #[inline]
    fn norm_factor(&self) -> Sample {
        let overlap = self.density * self.grain_seconds();
        1.0 / ops::sqrt(overlap.max(1.0))
    }

    /// Allocates a grain into a free pool slot (dropped if the pool is full),
    /// seeding it with randomized carrier/lifetime/pan parameters.
    fn spawn_grain(&mut self, sample_rate: Sample) {
        if self.frequency <= 0.0 {
            return;
        }
        // Draw all randoms first so the stream stays deterministic regardless of
        // whether a free slot exists.
        let pitch_u = self.rng.next_bipolar();
        let dur_u = self.rng.next_bipolar();
        let pan_u = self.rng.next_bipolar();

        let ratio = ops::exp2(pitch_u * self.pitch_spread / 12.0);
        let freq = self.frequency * ratio;
        let life_ms = (self.grain_ms * (1.0 + dur_u * self.duration_jitter))
            .clamp(MIN_GRANULAR_GRAIN_MS, MAX_GRANULAR_GRAIN_MS);
        let lifetime_samples = (life_ms * 0.001 * sample_rate).max(1.0);
        let (left_gain, right_gain) = equal_power_pan(pan_u * self.pan_spread);

        for grain in &mut self.grains {
            if !grain.active {
                *grain = Grain {
                    active: true,
                    osc_phase: 0.0,
                    osc_inc: freq / sample_rate,
                    win_phase: 0.0,
                    win_inc: 1.0 / lifetime_samples,
                    left_gain,
                    right_gain,
                };
                return;
            }
        }
        // Pool saturated: the grain is dropped (its randoms were already drawn).
    }

    /// Renders one stereo sample `(left, right)`, advancing the scheduler, every
    /// active grain, and the smoothed amplitude exactly once.
    #[inline]
    fn render_sample(&mut self, sample_rate: Sample) -> (Sample, Sample) {
        // Advance the scheduler and spawn any grains whose onset falls in this
        // sample. `density / sample_rate` is well below 1 for realistic rates,
        // so this loop almost always runs zero or one iteration; it is bounded
        // by the clamped density regardless.
        self.spawn_phase += self.density / sample_rate;
        while self.spawn_phase >= 1.0 {
            self.spawn_phase -= 1.0;
            self.spawn_grain(sample_rate);
        }

        let mut left = 0.0;
        let mut right = 0.0;
        for grain in &mut self.grains {
            if !grain.active {
                continue;
            }
            let window = 0.5 - 0.5 * ops::cos(TAU * grain.win_phase);
            let carrier = ops::sin(TAU * grain.osc_phase);
            let sample = window * carrier;
            left += sample * grain.left_gain;
            right += sample * grain.right_gain;

            grain.osc_phase = wrap01(grain.osc_phase + grain.osc_inc);
            grain.win_phase += grain.win_inc;
            if grain.win_phase >= 1.0 {
                grain.active = false;
            }
        }

        let gain = self.amplitude.next_sample() * self.norm_factor();
        (flush_denormal(left * gain), flush_denormal(right * gain))
    }
}

impl AudioNode for GranularSourceNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }
        let frames = io.output(0).active_frames();
        if frames == 0 {
            return;
        }
        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sample_rate = ctx.sample_rate.max(1) as Sample;

        if channels == 1 {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                let (l, r) = self.render_sample(sample_rate);
                *s = (l + r) * SQRT_HALF;
            }
            return;
        }

        // Stereo (or wider): render left into channel 0, right into channel 1.
        {
            let (left, right) = io.output(0).channel_pair_mut(0, 1);
            for (ls, rs) in left.iter_mut().zip(right.iter_mut()) {
                let (l, r) = self.render_sample(sample_rate);
                *ls = l;
                *rs = r;
            }
        }
        // Any further channels receive the energy-preserving mono downmix.
        // Two serial borrows keep each `channel_pair_mut` call disjoint.
        for ch in 2..channels {
            {
                let (left, dst) = io.output(0).channel_pair_mut(0, ch);
                for (d, l) in dst.iter_mut().zip(left.iter()) {
                    *d = *l * SQRT_HALF;
                }
            }
            {
                let (right, dst) = io.output(0).channel_pair_mut(1, ch);
                for (d, r) in dst.iter_mut().zip(right.iter()) {
                    *d += *r * SQRT_HALF;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.grains = [Grain::default(); MAX_GRAINS];
        self.spawn_phase = 0.0;
        self.rng = GrainRng::new(self.seed);
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
        RenderContext { sample_rate, frames, playhead: 0 }
    }

    fn render(node: &mut GranularSourceNode, layout: ChannelLayout, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(layout, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, frames), &mut io);
        let [buf] = outputs;
        buf
    }

    fn peak(buf: &AudioBuffer, ch: usize) -> Sample {
        buf.channel(ch).iter().fold(0.0_f32, |m, s| m.max(s.abs()))
    }

    fn rms(buf: &AudioBuffer, ch: usize) -> Sample {
        let data = buf.channel(ch);
        let sum: Sample = data.iter().map(|s| s * s).sum();
        ops::sqrt(sum / data.len() as Sample)
    }

    fn dense() -> GranularSourceParams {
        GranularSourceParams {
            density_hz: 200.0,
            grain_ms: 50.0,
            ..GranularSourceParams::default()
        }
    }

    #[test]
    fn defaults_land_in_range() {
        let node = GranularSourceNode::from_params(GranularSourceParams::default());
        assert_eq!(node.frequency(), DEFAULT_GRANULAR_FREQUENCY_HZ);
        assert_eq!(node.density(), DEFAULT_GRANULAR_DENSITY_HZ);
        assert_eq!(node.grain_ms(), DEFAULT_GRANULAR_GRAIN_MS);
        assert_eq!(node.pitch_spread(), DEFAULT_GRANULAR_PITCH_SPREAD);
        assert_eq!(node.pan_spread(), DEFAULT_GRANULAR_PAN_SPREAD);
        assert_eq!(node.duration_jitter(), DEFAULT_GRANULAR_DURATION_JITTER);
        assert_eq!(node.amplitude(), DEFAULT_GRANULAR_AMPLITUDE);
    }

    #[test]
    fn sanitise_clamps_out_of_range() {
        let p = GranularSourceParams {
            frequency_hz: -10.0,
            density_hz: 1.0e9,
            grain_ms: 0.0,
            pitch_spread: 1_000.0,
            pan_spread: 5.0,
            duration_jitter: -1.0,
            amplitude: 2.0,
            seed: 7,
        }
        .sanitised();
        assert_eq!(p.frequency_hz, 0.0);
        assert_eq!(p.density_hz, MAX_GRANULAR_DENSITY_HZ);
        assert_eq!(p.grain_ms, MIN_GRANULAR_GRAIN_MS);
        assert_eq!(p.pitch_spread, MAX_GRANULAR_PITCH_SPREAD);
        assert_eq!(p.pan_spread, 1.0);
        assert_eq!(p.duration_jitter, 0.0);
        assert_eq!(p.amplitude, 2.0);
        assert_eq!(p.seed, 7);
    }

    #[test]
    fn sanitise_replaces_non_finite_with_defaults() {
        let p = GranularSourceParams {
            frequency_hz: Sample::NAN,
            density_hz: Sample::INFINITY,
            grain_ms: Sample::NEG_INFINITY,
            pitch_spread: Sample::NAN,
            pan_spread: Sample::NAN,
            duration_jitter: Sample::INFINITY,
            amplitude: Sample::NAN,
            seed: 0,
        }
        .sanitised();
        assert_eq!(p.frequency_hz, DEFAULT_GRANULAR_FREQUENCY_HZ);
        // Infinite density is finite-replaced by its default (not clamped to max).
        assert_eq!(p.density_hz, DEFAULT_GRANULAR_DENSITY_HZ);
        assert_eq!(p.grain_ms, DEFAULT_GRANULAR_GRAIN_MS);
        assert_eq!(p.pitch_spread, DEFAULT_GRANULAR_PITCH_SPREAD);
        assert_eq!(p.pan_spread, DEFAULT_GRANULAR_PAN_SPREAD);
        assert_eq!(p.duration_jitter, DEFAULT_GRANULAR_DURATION_JITTER);
        assert_eq!(p.amplitude, DEFAULT_GRANULAR_AMPLITUDE);
    }

    #[test]
    fn dense_cloud_is_not_silent() {
        let mut node = GranularSourceNode::from_params(dense());
        let out = render(&mut node, ChannelLayout::Mono, 48_000);
        assert!(peak(&out, 0) > 0.0);
    }

    #[test]
    fn output_is_finite_and_free_of_denormals() {
        let mut node = GranularSourceNode::from_params(dense());
        let out = render(&mut node, ChannelLayout::Stereo, 24_000);
        for ch in 0..out.channels() {
            for &s in out.channel(ch) {
                assert!(s.is_finite());
                assert!(s == 0.0 || s.abs() >= Sample::MIN_POSITIVE);
            }
        }
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let mut a = GranularSourceNode::from_params(dense());
        let mut b = GranularSourceNode::from_params(dense());
        let oa = render(&mut a, ChannelLayout::Stereo, 8_192);
        let ob = render(&mut b, ChannelLayout::Stereo, 8_192);
        assert_eq!(oa.channel(0), ob.channel(0));
        assert_eq!(oa.channel(1), ob.channel(1));
    }

    #[test]
    fn different_seed_diverges() {
        // Spreads must be non-zero for the per-grain randoms to matter; with
        // all spreads at zero every grain is identical regardless of seed.
        let spread = GranularSourceParams {
            pitch_spread: 7.0,
            pan_spread: 0.5,
            duration_jitter: 0.3,
            ..dense()
        };
        let mut a = GranularSourceNode::from_params(GranularSourceParams { seed: 1, ..spread });
        let mut b = GranularSourceNode::from_params(GranularSourceParams { seed: 2, ..spread });
        let oa = render(&mut a, ChannelLayout::Mono, 8_192);
        let ob = render(&mut b, ChannelLayout::Mono, 8_192);
        assert_ne!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn zero_spread_ignores_seed() {
        // Documents the above: with no spread the seed cannot change the cloud.
        let mut a = GranularSourceNode::from_params(GranularSourceParams { seed: 1, ..dense() });
        let mut b = GranularSourceNode::from_params(GranularSourceParams { seed: 999, ..dense() });
        let oa = render(&mut a, ChannelLayout::Mono, 4_096);
        let ob = render(&mut b, ChannelLayout::Mono, 4_096);
        assert_eq!(oa.channel(0), ob.channel(0));
    }

    #[test]
    fn reset_replays_the_same_stream() {
        let mut node = GranularSourceNode::from_params(dense());
        let first = render(&mut node, ChannelLayout::Mono, 8_192);
        node.reset();
        let second = render(&mut node, ChannelLayout::Mono, 8_192);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn zero_density_is_always_silent() {
        let mut node = GranularSourceNode::from_params(GranularSourceParams {
            density_hz: 0.0,
            ..GranularSourceParams::default()
        });
        let out = render(&mut node, ChannelLayout::Stereo, 4_800);
        assert_eq!(peak(&out, 0), 0.0);
        assert_eq!(peak(&out, 1), 0.0);
    }

    #[test]
    fn zero_frequency_is_always_silent() {
        let mut node = GranularSourceNode::from_params(GranularSourceParams {
            frequency_hz: 0.0,
            ..dense()
        });
        let out = render(&mut node, ChannelLayout::Mono, 4_800);
        assert_eq!(peak(&out, 0), 0.0);
    }

    #[test]
    fn amplitude_scales_linearly() {
        let mut unit = GranularSourceNode::from_params(dense());
        let mut doubled =
            GranularSourceNode::from_params(GranularSourceParams { amplitude: 2.0, ..dense() });
        let a = render(&mut unit, ChannelLayout::Mono, 4_096);
        let b = render(&mut doubled, ChannelLayout::Mono, 4_096);
        for (x, y) in a.channel(0).iter().zip(b.channel(0).iter()) {
            assert!((2.0 * x - y).abs() <= 1e-6 * (1.0 + y.abs()), "x={x} y={y}");
        }
    }

    #[test]
    fn pan_spread_zero_keeps_channels_equal() {
        let mut node = GranularSourceNode::from_params(dense());
        let out = render(&mut node, ChannelLayout::Stereo, 4_096);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn pan_spread_separates_channels() {
        let mut node = GranularSourceNode::from_params(GranularSourceParams {
            pan_spread: 1.0,
            ..dense()
        });
        let out = render(&mut node, ChannelLayout::Stereo, 4_096);
        assert_ne!(out.channel(0), out.channel(1));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = GranularSourceNode::from_params(dense());
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 16)];
        outputs[0].set_active_frames(0);
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        // No active frames: process must early-return without touching samples.
        node.process(&ctx(SR, 0), &mut io);
        assert_eq!(outputs[0].channel(0).len(), 0);
    }

    #[test]
    fn mono_is_energy_preserving_downmix_of_stereo() {
        let mut mono_node = GranularSourceNode::from_params(GranularSourceParams {
            pan_spread: 0.8,
            ..dense()
        });
        let mut stereo_node = GranularSourceNode::from_params(GranularSourceParams {
            pan_spread: 0.8,
            ..dense()
        });
        let mono = render(&mut mono_node, ChannelLayout::Mono, 4_096);
        let stereo = render(&mut stereo_node, ChannelLayout::Stereo, 4_096);
        for (i, &m) in mono.channel(0).iter().enumerate() {
            let expected = (stereo.channel(0)[i] + stereo.channel(1)[i]) * SQRT_HALF;
            assert!((m - expected).abs() <= 1e-6, "m={m} expected={expected}");
        }
    }

    #[test]
    fn surround_extra_channels_receive_mono_downmix() {
        let mut node = GranularSourceNode::from_params(GranularSourceParams {
            pan_spread: 0.6,
            ..dense()
        });
        let out = render(&mut node, ChannelLayout::Surround5_1, 4_096);
        for i in 0..out.active_frames() {
            let expected = (out.channel(0)[i] + out.channel(1)[i]) * SQRT_HALF;
            for ch in 2..out.channels() {
                assert!((out.channel(ch)[i] - expected).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn higher_density_keeps_rms_bounded() {
        let mut low = GranularSourceNode::from_params(GranularSourceParams {
            density_hz: 50.0,
            ..dense()
        });
        let mut high = GranularSourceNode::from_params(GranularSourceParams {
            density_hz: 1_500.0,
            ..dense()
        });
        let lo = render(&mut low, ChannelLayout::Mono, 48_000);
        let hi = render(&mut high, ChannelLayout::Mono, 48_000);
        // Power-preserving normalization keeps RMS from exploding with density.
        assert!(rms(&hi, 0) < 4.0 * rms(&lo, 0).max(1e-4));
    }

    #[test]
    fn long_grains_stay_active_across_the_block() {
        let mut node = GranularSourceNode::from_params(GranularSourceParams {
            density_hz: 20.0,
            grain_ms: 500.0,
            ..GranularSourceParams::default()
        });
        let out = render(&mut node, ChannelLayout::Mono, 24_000);
        // A long sustained grain should keep the tail of the block sounding.
        let tail = &out.channel(0)[20_000..];
        assert!(tail.iter().any(|s| s.abs() > 0.0));
    }

    #[test]
    fn setters_clamp_and_reject_non_finite() {
        let mut node = GranularSourceNode::from_params(GranularSourceParams::default());

        node.set_frequency(-5.0);
        assert_eq!(node.frequency(), 0.0);
        node.set_frequency(Sample::NAN);
        assert_eq!(node.frequency(), 0.0);

        node.set_density(1.0e9);
        assert_eq!(node.density(), MAX_GRANULAR_DENSITY_HZ);
        node.set_density(Sample::INFINITY);
        assert_eq!(node.density(), MAX_GRANULAR_DENSITY_HZ);

        node.set_grain_ms(0.0);
        assert_eq!(node.grain_ms(), MIN_GRANULAR_GRAIN_MS);

        node.set_pitch_spread(1_000.0);
        assert_eq!(node.pitch_spread(), MAX_GRANULAR_PITCH_SPREAD);

        node.set_pan_spread(9.0);
        assert_eq!(node.pan_spread(), 1.0);

        node.set_duration_jitter(-3.0);
        assert_eq!(node.duration_jitter(), 0.0);

        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), DEFAULT_GRANULAR_AMPLITUDE);
    }

    #[test]
    fn latency_is_zero() {
        let node = GranularSourceNode::from_params(GranularSourceParams::default());
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn pool_is_bounded_under_extreme_density() {
        // Even at the maximum spawn rate the fixed pool keeps output finite.
        let mut node = GranularSourceNode::from_params(GranularSourceParams {
            density_hz: MAX_GRANULAR_DENSITY_HZ,
            grain_ms: 500.0,
            ..GranularSourceParams::default()
        });
        let out = render(&mut node, ChannelLayout::Mono, 24_000);
        assert!(peak(&out, 0).is_finite());
    }

    #[test]
    fn seed_to_state_is_never_zero() {
        assert_ne!(seed_to_state(0), 0);
        assert_ne!(seed_to_state(0x9E37_79B9_7F4A_7C15_u64.wrapping_neg()), 0);
    }
}
