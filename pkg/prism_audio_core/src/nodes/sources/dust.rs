//! Random (Poisson) impulse source node.
//!
//! [`DustNode`] emits isolated single-sample impulses at random times whose
//! average rate is set by a `density` control in impulses per second. Each
//! active sample is drawn independently, so the inter-impulse gaps follow a
//! geometric (discrete Poisson) distribution: the output is a sparse stream of
//! random-amplitude clicks, the classic "dust" texture used to seed granular
//! clouds, stochastic triggers, and noise-shaped percussion.
//!
//! # Model
//!
//! Let `dt = 1 / sample_rate` and `thresh = clamp(density * dt, 0, 1)`. Every
//! sample draws one uniform `z` in `[0, 1)` from a deterministic generator:
//!
//! ```text
//!   z < thresh  ->  a = z / thresh          (a uniform in [0, 1))
//!                   unipolar: out = a
//!                   bipolar:  out = 2 * a - 1
//!   z >= thresh ->  out = 0
//! ```
//!
//! A sample fires with probability `thresh`, giving a mean rate of
//! `thresh * sample_rate = density` impulses per second. Reusing `z` for the
//! amplitude yields a uniformly distributed impulse height at no extra cost:
//! `[0, 1)` in unipolar mode (a drop-in random trigger) or `[-1, 1)` in
//! bipolar mode (zero-mean clicks). Both modes stay within `[-1, 1]`, so only
//! `amplitude` scales the stream and no normalization is required. The
//! generator is advanced every sample regardless of whether an impulse fires,
//! so the random stream depends only on the seed and sample count, never on
//! the density automation.
//!
//! # Relationship
//!
//! Unlike [`super::impulse_train::ImpulseTrainNode`], which emits a periodic,
//! band-limited impulse train at a fixed pitch, this node emits aperiodic
//! impulses at random times with no harmonic structure. Where
//! [`super::noise::NoiseNode`] fills every sample with a colored random value,
//! `DustNode` leaves most samples silent and fires only sparse clicks, so it
//! reads as a rate rather than a timbre.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so [`DustNode::process`] performs
//! no allocation, no locking, and no panicking: it is a pure per-sample state
//! machine advancing an integer generator state. `density` and `amplitude` are
//! driven through [`Smoothed`] values so automation never produces zipper
//! clicks on the mean rate or gain. Two nodes built with the same `seed`
//! produce bit-identical output, and [`DustNode::reset`] restarts the exact
//! same stream.
//!
//! # Provenance
//!
//! Implemented from first principles from the classic public-domain
//! random-impulse ("dust") technique: fire a sample with probability
//! proportional to density, reusing the draw as the impulse height. The
//! deterministic generator is a self-contained Marsaglia `xorshift64` seeded
//! through `SplitMix64`, both long-standing public-domain integer algorithms.
//! It contains no code, data, or derivative of Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web Audio API, the
//! Synthesis Toolkit, or any other audio engine or toolkit; only the shared
//! mathematical ideas are used. There is no AI or machine learning of any kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::param::{Ramp, Smoothed};

/// Minimum impulse density in impulses per second (zero means silence).
pub const MIN_DENSITY_HZ: Sample = 0.0;

/// Default impulse density in impulses per second.
pub const DEFAULT_DENSITY_HZ: Sample = 200.0;

/// Maximum impulse density in impulses per second.
pub const MAX_DENSITY_HZ: Sample = 20_000.0;

/// Default linear output amplitude.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Default generator seed.
pub const DEFAULT_SEED: u64 = 0x7265_736F_6E61_6E63;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Output polarity of the random impulses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Polarity {
    /// Impulse heights are uniform in `[0, 1)` (a drop-in random trigger).
    #[default]
    Unipolar,
    /// Impulse heights are uniform in `[-1, 1)` (zero-mean clicks).
    Bipolar,
}

/// Self-contained deterministic PRNG (Marsaglia `xorshift64`) seeded via
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

    /// Returns the next uniform sample in `[0, 1)`.
    #[inline]
    fn next_unit(&mut self) -> Sample {
        let bits = self.next_u32();
        // Use the top 24 bits to form a float in [0, 1) with no rounding bias.
        (bits >> 8) as Sample * (1.0 / 16_777_216.0)
    }
}

/// Diffuses a user seed into a non-zero `xorshift64` state via `SplitMix64`.
///
/// `SplitMix64` is a bijection, so distinct seeds map to distinct states
/// (except the single seed that would map to zero, which is remapped to a
/// fixed golden constant to keep the xorshift generator valid).
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

/// Construction parameters for a [`DustNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DustParams {
    /// Impulse density in impulses per second. Clamped to `[MIN, MAX]`.
    pub density_hz: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
    /// Output polarity.
    pub polarity: Polarity,
    /// Deterministic generator seed.
    pub seed: u64,
}

impl Default for DustParams {
    fn default() -> Self {
        Self {
            density_hz: DEFAULT_DENSITY_HZ,
            amplitude: DEFAULT_AMPLITUDE,
            polarity: Polarity::Unipolar,
            seed: DEFAULT_SEED,
        }
    }
}

impl DustParams {
    /// Returns a copy with every field finite and inside its documented domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            density_hz: finite_or(self.density_hz, DEFAULT_DENSITY_HZ)
                .clamp(MIN_DENSITY_HZ, MAX_DENSITY_HZ),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
            polarity: self.polarity,
            seed: self.seed,
        }
    }
}

/// A random (Poisson) impulse source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::{DustNode, Polarity};
///
/// let mut node = DustNode::new(200.0, 0.8, Polarity::Unipolar, 1);
/// assert_eq!(node.density_hz(), 200.0);
/// assert_eq!(node.polarity(), Polarity::Unipolar);
/// ```
#[derive(Debug, Clone)]
pub struct DustNode {
    /// Smoothed impulse density in impulses per second.
    density_hz: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Output polarity.
    polarity: Polarity,
    /// Seed the generator was constructed/reseeded with; restored by `reset`.
    seed: u64,
    /// Deterministic uniform generator.
    rng: Xorshift64,
}

impl DustNode {
    /// Creates a dust source at `density_hz` with the given linear `amplitude`,
    /// `polarity`, and generator `seed`.
    ///
    /// Non-finite inputs fall back to defaults; density is clamped to
    /// `[MIN_DENSITY_HZ, MAX_DENSITY_HZ]`.
    #[must_use]
    pub fn new(density_hz: Sample, amplitude: Sample, polarity: Polarity, seed: u64) -> Self {
        Self {
            density_hz: Smoothed::new(
                finite_or(density_hz, DEFAULT_DENSITY_HZ).clamp(MIN_DENSITY_HZ, MAX_DENSITY_HZ),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            polarity,
            seed,
            rng: Xorshift64::new(seed),
        }
    }

    /// Builds a dust source from a [`DustParams`] bundle.
    #[must_use]
    pub fn from_params(params: DustParams) -> Self {
        let p = params.sanitised();
        Self::new(p.density_hz, p.amplitude, p.polarity, p.seed)
    }

    /// Sets a new target impulse density in hertz, gliding with `ramp`.
    #[inline]
    pub fn set_density_hz(&mut self, hz: Sample, ramp: Ramp) {
        self.density_hz.set_target(
            finite_or(hz, self.density_hz.target()).clamp(MIN_DENSITY_HZ, MAX_DENSITY_HZ),
            ramp,
        );
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Sets the output polarity (applied immediately).
    #[inline]
    pub fn set_polarity(&mut self, polarity: Polarity) {
        self.polarity = polarity;
    }

    /// Reseeds the generator and restarts the (now different) impulse stream.
    #[inline]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Xorshift64::new(seed);
    }

    /// Returns the target impulse density the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn density_hz(&self) -> Sample {
        self.density_hz.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Returns the current output polarity.
    #[inline]
    #[must_use]
    pub fn polarity(&self) -> Polarity {
        self.polarity
    }

    /// Returns the seed the generator is currently running from.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Produces one output sample, advancing the generator and the smoothed
    /// controls. `inv_sr` is the reciprocal of the sample rate.
    #[inline]
    fn render_sample(&mut self, inv_sr: Sample) -> Sample {
        let density = self.density_hz.next_sample();
        let amp = self.amplitude.next_sample();
        let thresh = (density * inv_sr).clamp(0.0, 1.0);

        // The generator advances every sample so the stream depends only on
        // seed and sample count, never on the density automation.
        let z = self.rng.next_unit();
        let raw = if thresh > 0.0 && z < thresh {
            let a = z / thresh;
            match self.polarity {
                Polarity::Unipolar => a,
                Polarity::Bipolar => a * 2.0 - 1.0,
            }
        } else {
            0.0
        };

        raw * amp
    }
}

impl AudioNode for DustNode {
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
        self.rng = Xorshift64::new(self.seed);
        self.density_hz = Smoothed::new(self.density_hz.target());
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
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut DustNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        render_layout(node, sample_rate, frames, ChannelLayout::Mono)
    }

    fn render_layout(
        node: &mut DustNode,
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

    fn nonzero_count(buf: &AudioBuffer) -> usize {
        buf.channel(0).iter().filter(|&&s| s != 0.0).count()
    }

    #[test]
    fn renders_bounded_finite() {
        for &density in &[0.0, 50.0, 1_000.0, 20_000.0] {
            for &polarity in &[Polarity::Unipolar, Polarity::Bipolar] {
                let mut node = DustNode::new(density, 0.8, polarity, 1);
                let out = render(&mut node, SR, 8_192);
                for &s in out.channel(0) {
                    assert!(
                        s.is_finite() && s.abs() <= 1.0 + 1e-3,
                        "density={density} polarity={polarity:?} s={s}"
                    );
                }
            }
        }
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = DustNode::new(2_000.0, 0.0, Polarity::Bipolar, 7);
        let out = render(&mut node, SR, 4_096);
        assert_eq!(energy(&out), 0.0);
    }

    #[test]
    fn not_silent() {
        let mut node = DustNode::new(2_000.0, 0.8, Polarity::Unipolar, 7);
        let out = render(&mut node, SR, 4_096);
        assert!(energy(&out) > 0.0);
    }

    #[test]
    fn unipolar_is_non_negative() {
        let mut node = DustNode::new(5_000.0, 0.8, Polarity::Unipolar, 3);
        let out = render(&mut node, SR, 8_192);
        for &s in out.channel(0) {
            assert!(s >= 0.0, "unipolar sample should be non-negative: {s}");
        }
    }

    #[test]
    fn bipolar_has_both_signs() {
        let mut node = DustNode::new(5_000.0, 0.8, Polarity::Bipolar, 3);
        let out = render(&mut node, SR, 16_384);
        let has_pos = out.channel(0).iter().any(|&s| s > 0.0);
        let has_neg = out.channel(0).iter().any(|&s| s < 0.0);
        assert!(has_pos && has_neg, "bipolar stream should have both signs");
    }

    #[test]
    fn density_controls_impulse_rate() {
        let mut low = DustNode::new(200.0, 0.8, Polarity::Unipolar, 11);
        let mut high = DustNode::new(5_000.0, 0.8, Polarity::Unipolar, 11);
        let low_out = render(&mut low, SR, 48_000);
        let high_out = render(&mut high, SR, 48_000);
        assert!(
            nonzero_count(&high_out) > nonzero_count(&low_out),
            "higher density should fire more impulses"
        );
    }

    #[test]
    fn zero_density_is_silent() {
        let mut node = DustNode::new(0.0, 0.8, Polarity::Bipolar, 5);
        let out = render(&mut node, SR, 4_096);
        assert_eq!(energy(&out), 0.0);
    }

    #[test]
    fn mean_rate_approximates_density() {
        // One second at 48 kHz, density 1000 Hz: expected 1000 impulses with a
        // binomial standard deviation of about 31, so a +/-20% window is a
        // ~6-sigma guard that stays robust across seeds.
        let mut node = DustNode::new(1_000.0, 0.8, Polarity::Unipolar, 1);
        let out = render(&mut node, SR, 48_000);
        let fired = nonzero_count(&out);
        assert!(
            (800..=1_200).contains(&fired),
            "mean rate out of tolerance: fired={fired}"
        );
    }

    #[test]
    fn deterministic_across_instances() {
        let mut a = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 42);
        let mut b = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 42);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_eq!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 1);
        let mut b = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 2);
        let out_a = render(&mut a, SR, 4_096);
        let out_b = render(&mut b, SR, 4_096);
        assert_ne!(out_a.channel(0), out_b.channel(0));
    }

    #[test]
    fn reset_replays_identically() {
        let mut node = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 99);
        let first = render(&mut node, SR, 4_096);
        node.reset();
        let second = render(&mut node, SR, 4_096);
        assert_eq!(first.channel(0), second.channel(0));
    }

    #[test]
    fn set_seed_changes_stream() {
        let mut node = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 1);
        let before = render(&mut node, SR, 4_096);
        node.set_seed(2);
        let after = render(&mut node, SR, 4_096);
        assert_ne!(before.channel(0), after.channel(0));
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        // Same seed and density reuse the identical raw stream, so the energy
        // ratio is exactly the square of the amplitude ratio.
        let mut quiet = DustNode::new(4_000.0, 0.25, Polarity::Bipolar, 7);
        let mut loud = DustNode::new(4_000.0, 0.5, Polarity::Bipolar, 7);
        let e_quiet = energy(&render(&mut quiet, SR, 8_192));
        let e_loud = energy(&render(&mut loud, SR, 8_192));
        assert!(e_quiet > 0.0);
        let ratio = e_loud / e_quiet;
        assert!(
            (ratio - 4.0).abs() < 1e-2,
            "energy should scale quadratically: ratio={ratio}"
        );
    }

    #[test]
    fn identical_across_stereo_and_quad() {
        let mut mono = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 8);
        let mono_out = render(&mut mono, SR, 2_048);
        let mut stereo = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 8);
        let stereo_out = render_layout(&mut stereo, SR, 2_048, ChannelLayout::Stereo);
        let mut quad = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 8);
        let quad_out = render_layout(&mut quad, SR, 2_048, ChannelLayout::Quad);
        assert_eq!(mono_out.channel(0), stereo_out.channel(0));
        assert_eq!(stereo_out.channel(0), stereo_out.channel(1));
        assert_eq!(mono_out.channel(0), quad_out.channel(0));
        assert_eq!(quad_out.channel(0), quad_out.channel(3));
    }

    #[test]
    fn zero_frames_is_noop() {
        // Processing a zero-frame block must not advance the generator, so a
        // node that renders 0 frames then N frames matches a fresh node.
        let mut idle = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 13);
        let mut empty = AudioBuffer::new(ChannelLayout::Mono, 64);
        empty.set_active_frames(0);
        {
            let inputs: [AudioBuffer; 0] = [];
            let mut outputs = [empty];
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            idle.process(&ctx(SR, 0), &mut io);
        }
        let after_idle = render(&mut idle, SR, 2_048);
        let mut fresh = DustNode::new(3_000.0, 0.8, Polarity::Bipolar, 13);
        let fresh_out = render(&mut fresh, SR, 2_048);
        assert_eq!(after_idle.channel(0), fresh_out.channel(0));
    }

    #[test]
    fn latency_is_zero() {
        let node = DustNode::new(1_000.0, 0.8, Polarity::Unipolar, 1);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_state() {
        let mut node = DustNode::new(1_500.0, 0.6, Polarity::Bipolar, 77);
        assert_eq!(node.density_hz(), 1_500.0);
        assert_eq!(node.amplitude(), 0.6);
        assert_eq!(node.polarity(), Polarity::Bipolar);
        assert_eq!(node.seed(), 77);
        node.set_polarity(Polarity::Unipolar);
        assert_eq!(node.polarity(), Polarity::Unipolar);
    }

    #[test]
    fn default_params_in_domain() {
        let p = DustParams::default();
        assert_eq!(p.density_hz, DEFAULT_DENSITY_HZ);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
        assert_eq!(p.polarity, Polarity::Unipolar);
        assert_eq!(p.seed, DEFAULT_SEED);
        let s = p.sanitised();
        assert!(s.density_hz >= MIN_DENSITY_HZ && s.density_hz <= MAX_DENSITY_HZ);
        assert!(s.amplitude.is_finite());
    }

    #[test]
    fn from_params_matches_new() {
        let params = DustParams {
            density_hz: 2_500.0,
            amplitude: 0.7,
            polarity: Polarity::Bipolar,
            seed: 55,
        };
        let mut from = DustNode::from_params(params);
        let mut direct = DustNode::new(2_500.0, 0.7, Polarity::Bipolar, 55);
        let from_out = render(&mut from, SR, 2_048);
        let direct_out = render(&mut direct, SR, 2_048);
        assert_eq!(from_out.channel(0), direct_out.channel(0));
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let node = DustNode::new(1.0e9, 0.8, Polarity::Unipolar, 1);
        assert_eq!(node.density_hz(), MAX_DENSITY_HZ);
        let p = DustParams {
            density_hz: -100.0,
            amplitude: 0.8,
            polarity: Polarity::Unipolar,
            seed: 1,
        }
        .sanitised();
        assert_eq!(p.density_hz, MIN_DENSITY_HZ);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node = DustNode::new(Sample::NAN, Sample::INFINITY, Polarity::Unipolar, 1);
        assert_eq!(node.density_hz(), DEFAULT_DENSITY_HZ);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = DustNode::new(1_000.0, 0.8, Polarity::Unipolar, 1);
        node.set_density_hz(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.density_hz(), 1_000.0);
        node.set_density_hz(1.0e9, Ramp::Immediate);
        assert_eq!(node.density_hz(), MAX_DENSITY_HZ);
        node.set_amplitude(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
        node.set_amplitude(0.3, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.3);
    }

    #[test]
    fn density_change_takes_effect() {
        let mut node = DustNode::new(200.0, 0.8, Polarity::Unipolar, 21);
        let low = render(&mut node, SR, 48_000);
        node.set_density_hz(8_000.0, Ramp::Immediate);
        let high = render(&mut node, SR, 48_000);
        assert!(
            nonzero_count(&high) > nonzero_count(&low),
            "raising density should fire more impulses"
        );
    }
}
