//! Deterministic colored-noise source node (white, pink, brown).
//!
//! A [`NoiseNode`] is a zero-input, one-output source that synthesizes a noise
//! stream directly from an internal pseudo-random generator — it never reads
//! its inputs. Three spectral "colors" are supported:
//!
//! - [`NoiseColor::White`] — flat power spectrum, straight from the PRNG.
//! - [`NoiseColor::Pink`] — `-3 dB`/octave, via the Paul Kellet economy filter.
//! - [`NoiseColor::Brown`] — `-6 dB`/octave, via a leaky integrator of white.
//!
//! # Determinism
//!
//! The generator is a fully self-contained xorshift64 PRNG seeded through
//! `SplitMix64`, so two [`NoiseNode`]s constructed with the same `seed` and
//! `color` emit bit-identical sample streams regardless of platform. This
//! matters for reproducible mixes, regression tests, and networked lockstep.
//!
//! # Real-time contract
//!
//! [`NoiseNode::process`] performs no allocation, locking, or panic: it is a
//! pure state machine advancing integer PRNG state and a handful of one-pole
//! accumulators. The `amplitude` parameter is driven through a
//! [`Smoothed`] value so level automation never introduces zipper noise.
//!
//! # Provenance
//!
//! All algorithms here are classic public-domain DSP: xorshift64
//! (G. Marsaglia, "Xorshift RNGs", *Journal of Statistical Software*, 2003),
//! `SplitMix64` (S. Vigna, public-domain reference implementation), the Paul
//! Kellet economy pink-noise filter (posted to the `musicdsp.org` archive,
//! public domain), and the standard leaky-integrator brown-noise formulation.
//! No engine (Unreal, Unity, Godot, Wwise, FMOD) source is used or derived.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Normalization applied to the Paul Kellet pink-filter sum so that pink output
/// for unit-amplitude white input sits roughly within `[-1.0, 1.0]`.
const PINK_NORM: Sample = 0.11;

/// Integration step for the brown-noise leaky integrator.
const BROWN_STEP: Sample = 0.02;

/// Leak factor (`1.0 / 1.02`) that pulls the brown integrator back toward zero,
/// preventing unbounded DC wander of the accumulated random walk.
const BROWN_LEAK: Sample = 0.980_392_16;

/// Makeup gain restoring brown output to a usable level after the leak/clamp.
const BROWN_NORM: Sample = 3.5;

/// Spectral color (power-spectrum slope) produced by a [`NoiseNode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum NoiseColor {
    /// Flat spectrum: the raw PRNG output (`0 dB`/octave).
    White,
    /// `-3 dB`/octave: equal energy per octave, via the Paul Kellet filter.
    Pink,
    /// `-6 dB`/octave (a.k.a. red noise): a leaky integral of white noise.
    Brown,
}

/// Self-contained deterministic PRNG (Marsaglia xorshift64) seeded via
/// `SplitMix64`.
#[derive(Debug, Clone, Copy)]
struct Xorshift64 {
    /// Current 64-bit generator state; kept non-zero by the seeding routine.
    state: u64,
}

impl Xorshift64 {
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
        // [-1, 1). This keeps every white sample bounded by 1 in magnitude.
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

/// Paul Kellet economy pink-noise filter memory: seven one-pole accumulators.
///
/// The classic refined coefficient set gives a spectrum within `+/-0.5 dB` of
/// true pink from ~10 Hz to ~20 kHz while costing only seven multiply-adds.
#[derive(Debug, Clone, Copy, Default)]
struct PinkState {
    /// One-pole accumulator `b0` (slowest pole, `p = 0.99886`).
    b0: Sample,
    /// One-pole accumulator `b1` (`p = 0.99332`).
    b1: Sample,
    /// One-pole accumulator `b2` (`p = 0.96900`).
    b2: Sample,
    /// One-pole accumulator `b3` (`p = 0.86650`).
    b3: Sample,
    /// One-pole accumulator `b4` (`p = 0.55000`).
    b4: Sample,
    /// One-pole accumulator `b5` (fastest pole, `p = -0.7616`).
    b5: Sample,
    /// Direct feed-through term `b6`, refreshed each sample from white input.
    b6: Sample,
}

impl PinkState {
    /// Filters one white sample and returns the (normalized) pink sample.
    #[inline]
    fn process(&mut self, white: Sample) -> Sample {
        // Paul Kellet's refined economy method (public domain, musicdsp.org).
        self.b0 = 0.998_86 * self.b0 + white * 0.055_517_9;
        self.b1 = 0.993_32 * self.b1 + white * 0.075_075_9;
        self.b2 = 0.969_00 * self.b2 + white * 0.153_852;
        self.b3 = 0.866_50 * self.b3 + white * 0.310_485_6;
        self.b4 = 0.550_00 * self.b4 + white * 0.532_952_2;
        self.b5 = -0.761_6 * self.b5 - white * 0.016_898_0;
        let pink =
            self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6 + white * 0.536_2;
        self.b6 = white * 0.115_926;
        pink * PINK_NORM
    }
}

/// A deterministic colored-noise source with a smoothed output amplitude.
///
/// Zero inputs, one output. Every channel of the output receives the *same*
/// noise sample per frame (mono-correlated). This is the common expectation for
/// a single noise source and, more importantly, keeps the stream fully
/// reproducible from just `(color, seed)`: generating an independent stream per
/// channel would make the output depend on the channel count and complicate
/// deterministic replay.
#[derive(Debug, Clone)]
pub struct NoiseNode {
    /// Spectral color currently being generated.
    color: NoiseColor,
    /// Seed the generator was constructed/reseeded with; restored by `reset`.
    seed: u64,
    /// Amplitude the node was constructed with; restored by `reset`.
    initial_amplitude: Sample,
    /// The deterministic pseudo-random white-noise generator.
    rng: Xorshift64,
    /// Pink-noise filter memory (only advanced when `color == Pink`).
    pink: PinkState,
    /// Brown-noise leaky-integrator state (only advanced when `color == Brown`).
    brown: Sample,
    /// Click-free linear output amplitude.
    amplitude: Smoothed,
}

impl NoiseNode {
    /// Creates a noise source of the given `color`, PRNG `seed`, and settled
    /// linear `amplitude` (a plain multiplier, not decibels).
    #[must_use]
    pub fn new(color: NoiseColor, seed: u64, amplitude: Sample) -> Self {
        Self {
            color,
            seed,
            initial_amplitude: amplitude,
            rng: Xorshift64::new(seed),
            pink: PinkState::default(),
            brown: 0.0,
            amplitude: Smoothed::new(amplitude),
        }
    }

    /// Returns the spectral color currently being generated.
    #[inline]
    #[must_use]
    pub fn color(&self) -> NoiseColor {
        self.color
    }

    /// Returns the seed the generator is currently running from.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the instantaneous (current) linear output amplitude.
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.current()
    }

    /// Switches the spectral color. Filter memory is left intact so a live
    /// color change does not click; the stale color's state simply resumes if
    /// switched back.
    #[inline]
    pub fn set_color(&mut self, color: NoiseColor) {
        self.color = color;
    }

    /// Sets a new target amplitude, gliding toward it with `ramp` to avoid
    /// zipper noise.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude.set_target(linear, ramp);
    }

    /// Reseeds the PRNG and clears all filter state, restarting the (now
    /// different) deterministic stream from the beginning.
    #[inline]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Xorshift64::new(seed);
        self.pink = PinkState::default();
        self.brown = 0.0;
    }

    /// Generates the next colored noise sample (pre-amplitude), advancing the
    /// PRNG and — depending on `color` — the relevant filter state.
    #[inline]
    fn next_colored(&mut self) -> Sample {
        let white = self.rng.next_bipolar();
        match self.color {
            NoiseColor::White => white,
            NoiseColor::Pink => self.pink.process(white),
            NoiseColor::Brown => {
                // Leaky integrator: accumulate white, leak toward zero to kill
                // DC drift, and clamp the state for absolute divergence safety.
                self.brown = ((self.brown + white * BROWN_STEP) * BROWN_LEAK).clamp(-1.0, 1.0);
                self.brown * BROWN_NORM
            }
        }
    }
}

impl AudioNode for NoiseNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let out = io.output(0);
        let channels = out.channels();

        // Every channel must carry the identical noise sequence, so snapshot
        // the generator, filter, and amplitude state and replay it from the
        // same point for each channel. Because each channel advances the state
        // by the same frame count, the committed state after the last channel
        // is exactly the per-block advance the next block should continue from.
        let start_rng = self.rng;
        let start_pink = self.pink;
        let start_brown = self.brown;
        let start_amp = self.amplitude;
        for ch in 0..channels {
            self.rng = start_rng;
            self.pink = start_pink;
            self.brown = start_brown;
            self.amplitude = start_amp;
            for s in out.channel_mut(ch).iter_mut() {
                *s = self.next_colored() * self.amplitude.next_sample();
            }
        }
    }

    fn reset(&mut self) {
        self.rng = Xorshift64::new(self.seed);
        self.pink = PinkState::default();
        self.brown = 0.0;
        self.amplitude = Smoothed::new(self.initial_amplitude);
    }

    // `latency_frames` uses the trait default of 0: a noise source is
    // memoryless with respect to any input, so it adds no latency.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;
    use bevy_math::ops;

    /// Builds a minimal render context for the given block size.
    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        }
    }

    /// Renders `frames` samples of channel 0 from `node` into a fresh `Vec`.
    fn render(node: &mut NoiseNode, frames: usize) -> Vec<Sample> {
        let mut outs = [AudioBuffer::new(ChannelLayout::Mono, frames)];
        let mut io = ProcessIo::new(&[], &mut outs);
        node.process(&ctx(frames), &mut io);
        let mut v = Vec::with_capacity(frames);
        v.extend_from_slice(outs[0].channel(0));
        v
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let mut a = NoiseNode::new(NoiseColor::White, 0xDEAD_BEEF, 1.0);
        let mut b = NoiseNode::new(NoiseColor::White, 0xDEAD_BEEF, 1.0);
        let va = render(&mut a, 1024);
        let vb = render(&mut b, 1024);
        assert_eq!(va, vb, "identical seeds must produce identical streams");
    }

    #[test]
    fn different_seed_differs() {
        let mut a = NoiseNode::new(NoiseColor::White, 42, 1.0);
        let mut b = NoiseNode::new(NoiseColor::White, 1234, 1.0);
        let va = render(&mut a, 1024);
        let vb = render(&mut b, 1024);
        assert_ne!(va, vb, "distinct seeds must produce distinct streams");
    }

    #[test]
    fn white_is_bounded_and_zero_mean() {
        let amp = 0.5;
        let mut node = NoiseNode::new(NoiseColor::White, 7, amp);
        let n = 16_384;
        let v = render(&mut node, n);
        let mut sum = 0.0;
        for &x in &v {
            assert!(x.is_finite());
            assert!(x.abs() <= amp + 1e-6, "white sample {x} exceeds amplitude");
            sum += x;
        }
        let mean = sum / n as Sample;
        assert!(mean.abs() < 0.02, "white mean {mean} should be near zero");
    }

    #[test]
    fn pink_is_finite_and_bounded() {
        let mut node = NoiseNode::new(NoiseColor::Pink, 99, 1.0);
        let n = 8_192;
        let v = render(&mut node, n);
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f32;
        for &x in &v {
            assert!(x.is_finite(), "pink sample must stay finite");
            max_abs = max_abs.max(x.abs());
            sum_sq += x * x;
        }
        // The Kellet filter poles are all < 1, so the stream cannot diverge.
        assert!(max_abs < 4.0, "pink peak {max_abs} unexpectedly large");
        let rms = ops::sqrt(sum_sq / n as Sample);
        assert!(rms.is_finite() && rms > 0.0, "pink rms {rms} invalid");
    }

    #[test]
    fn brown_is_finite_and_bounded() {
        let mut node = NoiseNode::new(NoiseColor::Brown, 5, 1.0);
        let n = 8_192;
        let v = render(&mut node, n);
        let mut max_abs = 0.0f32;
        for &x in &v {
            assert!(x.is_finite(), "brown sample must stay finite");
            max_abs = max_abs.max(x.abs());
        }
        // Internal state is clamped to [-1, 1] and scaled by BROWN_NORM.
        assert!(
            max_abs <= BROWN_NORM + 1e-6,
            "brown peak {max_abs} out of range"
        );
    }

    #[test]
    fn reset_reproduces_stream() {
        let mut node = NoiseNode::new(NoiseColor::Pink, 0x1234_5678, 0.8);
        let first = render(&mut node, 2_048);
        node.reset();
        let second = render(&mut node, 2_048);
        assert_eq!(first, second, "reset must restart the identical stream");
    }

    #[test]
    fn all_channels_receive_same_noise() {
        let mut node = NoiseNode::new(NoiseColor::White, 314, 1.0);
        let frames = 256;
        let mut outs = [AudioBuffer::new(ChannelLayout::Stereo, frames)];
        let mut io = ProcessIo::new(&[], &mut outs);
        node.process(&ctx(frames), &mut io);
        let left = outs[0].channel(0);
        let right = outs[0].channel(1);
        assert_eq!(left, right, "both channels must carry identical noise");
    }
}
