//! Dither and noise-shaping requantizer for mastering-grade bit-depth
//! reduction.
//!
//! When a high-resolution mix (here `32`-bit float) is delivered at a lower
//! fixed-point depth, every sample must be rounded onto a coarse amplitude
//! grid. Rounding alone turns the rounding error into a signal-correlated
//! distortion that is audible as gritty harmonics and as a "fade to grain" on
//! quiet tails. The classic fix is two cooperating steps:
//!
//! - **Dither** adds a tiny, carefully-shaped random signal (on the order of
//!   one least-significant bit, `LSB`) *before* rounding. This decorrelates the
//!   rounding error from the input so the residual is a steady, program-
//!   independent noise floor instead of distortion. Triangular-probability-
//!   density (`TPDF`) dither, the sum of two independent uniform draws spanning
//!   two `LSB`, additionally fixes the noise *modulation* that rectangular
//!   (`RPDF`) dither still leaves behind.
//! - **Noise shaping** feeds the past quantization error back through a small
//!   high-pass filter so the added noise is pushed up out of the ear's most
//!   sensitive midband into the high treble, lowering *perceived* noise without
//!   changing its total power.
//!
//! This module is the final output stage of a mastering chain: it is applied
//! once, last, after any loudness and peak processing, to render the float mix
//! to its delivery bit depth.
//!
//! # The model
//!
//! With `bits` bits the amplitude grid has `2^bits` codes spanning `[-1, 1)`
//! with a step (one `LSB`) of `q = 2 / 2^bits`; a value maps to
//! `round(value / q) * q` with the integer code clamped to the signed
//! two's-complement range `[-2^(bits-1), 2^(bits-1) - 1]`, exactly as a real
//! converter behaves.
//!
//! Dither is drawn in `LSB` units and scaled by `q`: `RPDF` is one uniform draw
//! on `[-0.5, 0.5)` `LSB`; `TPDF` is the sum of two independent draws, a
//! triangular density on `(-1, 1)` `LSB`.
//!
//! Noise shaping uses the standard error-feedback topology. Let `x[n]` be the
//! input, `e[n]` the realised error `y[n] - u[n]`, and `h` a short feedback
//! `FIR` with no direct term. Then
//! `u[n] = x[n] - sum_k h[k] * e[n-k]`, `y[n] = Q(u[n] + dither[n])`, and the
//! output noise spectrum is multiplied by the noise transfer function
//! `1 - H(z)`. A first-order shaper uses `h = [1]` (so the noise transfer
//! function is `1 - z^-1`, a gentle high-pass); a second-order shaper uses
//! `h = [2, -1]` (noise transfer function `(1 - z^-1)^2`, a steeper high-pass).
//! Because the feedback filter is `FIR`, the loop is unconditionally stable.
//!
//! # Real-time contract
//!
//! All per-channel error-feedback memory and the pseudo-random generator are
//! allocated once in [`Dither::new`] / [`DitherNode::new`].
//! [`Dither::process_sample`] and [`DitherNode::process`] perform no
//! allocation, take no locks, and cannot panic: non-finite inputs are treated
//! as silence, mismatched channel counts and zero-length blocks degrade
//! gracefully. Two instances built with the same [`DitherParams`] produce
//! bit-identical output, so the stage is golden-testable. All transcendental
//! math routes through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! Dither (`RPDF` / `TPDF`) and error-feedback noise shaping are elementary,
//! long-published quantization techniques described in the standard literature
//! (e.g. Lipshitz, Wannamaker and Vanderkooy, "Quantization and Dither: A
//! Theoretical Survey", JAES 1992; Zoelzer, "DAFX"). The feedback coefficients
//! used here are the generic textbook first- and second-order high-pass error
//! feedbacks, not any proprietary psychoacoustically-weighted coefficient
//! table. The pseudo-random generator is this crate's own self-contained
//! `xorshift64` seeded through `SplitMix64`. This module reuses only this
//! crate's own [`Sample`] and [`flush_denormal`] primitives. It is pure classic
//! DSP with no AI or ML and contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented theory.
//!
//! # Relationship
//!
//! This stage complements the read-only
//! [`analysis`](crate::nodes::analysis) meters: those verify loudness and true
//! peak, while this stage renders the verified float mix down to its delivery
//! bit depth. It performs transparent, mastering-grade requantization and is
//! deliberately distinct from the lo-fi [`BitcrusherNode`](crate::nodes::effects)
//! effect, which degrades a signal for character and does not dither or
//! noise-shape. Both share this crate's [`Sample`] scalar and quantization
//! idea; neither reuses the other's code.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Smallest supported delivery bit depth (a two-level, one-bit grid).
pub const MIN_DITHER_BITS: u32 = 1;

/// Largest supported delivery bit depth (finer than `24`-bit is transparent).
pub const MAX_DITHER_BITS: u32 = 24;

/// Default delivery bit depth (`16`-bit, the compact-disc standard).
pub const DEFAULT_DITHER_BITS: u32 = 16;

/// Probability density of the dither added before quantization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DitherType {
    /// No dither: plain rounding (signal-correlated distortion). Use only for
    /// measurement or when a downstream stage dithers.
    None,
    /// Rectangular probability density (`RPDF`): one uniform draw spanning one
    /// `LSB`. Removes the mean error but leaves audible noise modulation.
    Rectangular,
    /// Triangular probability density (`TPDF`): the sum of two independent
    /// uniform draws spanning two `LSB`. The standard transparent choice; it
    /// removes both the mean error and the noise modulation.
    Triangular,
}

/// High-pass error-feedback curve applied to the quantization noise.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum NoiseShaping {
    /// No shaping: the noise floor is spectrally flat (white).
    None,
    /// First-order high-pass shaping, noise transfer function `1 - z^-1`.
    FirstOrder,
    /// Second-order high-pass shaping, noise transfer function `(1 - z^-1)^2`;
    /// pushes more noise into the treble than [`NoiseShaping::FirstOrder`].
    SecondOrder,
}

impl NoiseShaping {
    /// Returns the feedback `FIR` taps `[h[1], h[2]]`. Unused taps are zero, so
    /// every variant can run through the same two-tap evaluation.
    #[must_use]
    fn feedback_taps(self) -> [Sample; 2] {
        match self {
            Self::None => [0.0, 0.0],
            Self::FirstOrder => [1.0, 0.0],
            Self::SecondOrder => [2.0, -1.0],
        }
    }
}

/// Configuration for a [`Dither`] engine or a [`DitherNode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DitherParams {
    /// Delivery bit depth, clamped to `[MIN_DITHER_BITS, MAX_DITHER_BITS]`.
    pub bits: u32,
    /// Dither probability density added before rounding.
    pub dither: DitherType,
    /// High-pass error-feedback curve applied to the quantization noise.
    pub shaping: NoiseShaping,
    /// Seed for the deterministic dither generator; equal seeds give
    /// bit-identical output.
    pub seed: u64,
}

impl Default for DitherParams {
    /// `16`-bit `TPDF` dither, no noise shaping, with a fixed non-zero seed:
    /// the transparent, broadly-safe mastering default.
    fn default() -> Self {
        Self {
            bits: DEFAULT_DITHER_BITS,
            dither: DitherType::Triangular,
            shaping: NoiseShaping::None,
            seed: 0x5EED_1770_D17E_B171,
        }
    }
}

/// Self-contained deterministic pseudo-random generator (Marsaglia
/// `xorshift64`) seeded through `SplitMix64`, producing uniform draws on
/// `[0, 1)`.
#[derive(Clone, Debug)]
struct DitherRng {
    /// Current `64`-bit state; kept non-zero by the seeding routine.
    state: u64,
}

impl DitherRng {
    /// Builds a generator whose state is diffused from `seed` via `SplitMix64`.
    #[must_use]
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Avoid the single all-zero state, which xorshift cannot leave.
        let state = if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z };
        Self { state }
    }

    /// Advances the state and returns a uniform draw on `[0, 1)` built from the
    /// top `24` bits (one float mantissa's worth of entropy).
    #[must_use]
    fn next_unit(&mut self) -> Sample {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let bits24 = x >> 40;
        (bits24 as Sample) / ((1u64 << 24) as Sample)
    }
}

/// Deterministic dither and noise-shaping requantizer engine.
///
/// Holds the per-channel error-feedback memory and the pseudo-random dither
/// generator. Build one with [`Dither::new`], then call
/// [`Dither::process_sample`] once per sample per channel. The companion
/// [`DitherNode`] wraps this engine as an [`AudioNode`].
#[derive(Clone, Debug)]
pub struct Dither {
    /// Delivery bit depth, clamped to the supported range.
    bits: u32,
    /// Quantization step (one `LSB`): `2 / 2^bits`.
    step: Sample,
    /// Largest code magnitude `2^(bits-1)`; codes span `[-half, half - 1]`.
    half: Sample,
    /// Dither probability density.
    dither: DitherType,
    /// Error-feedback curve.
    shaping: NoiseShaping,
    /// Feedback `FIR` taps `[h[1], h[2]]` derived from `shaping`.
    taps: [Sample; 2],
    /// Per-channel realised-error history `[e[n-1], e[n-2]]`.
    errors: Vec<[Sample; 2]>,
    /// Seed the generator was constructed/reseeded with; restored by `reset`.
    seed: u64,
    /// Deterministic dither generator.
    rng: DitherRng,
}

/// Clamps a requested bit depth into the supported range.
#[must_use]
fn clamp_bits(bits: u32) -> u32 {
    bits.clamp(MIN_DITHER_BITS, MAX_DITHER_BITS)
}

/// Returns `(step, half)` for a (already clamped) bit depth.
#[must_use]
fn grid_for(bits: u32) -> (Sample, Sample) {
    let levels = (1u32 << bits) as Sample;
    let step = 2.0 / levels;
    let half = levels * 0.5;
    (step, half)
}

impl Dither {
    /// Creates a requantizer for `channels` channels from `params`.
    ///
    /// All state is allocated here; the processing methods never allocate.
    #[must_use]
    pub fn new(params: DitherParams, channels: usize) -> Self {
        let bits = clamp_bits(params.bits);
        let (step, half) = grid_for(bits);
        Self {
            bits,
            step,
            half,
            dither: params.dither,
            shaping: params.shaping,
            taps: params.shaping.feedback_taps(),
            errors: vec![[0.0; 2]; channels],
            seed: params.seed,
            rng: DitherRng::new(params.seed),
        }
    }

    /// The delivery bit depth in use (always within the supported range).
    #[must_use]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The quantization step (one `LSB`) of the current grid.
    #[must_use]
    pub fn quantization_step(&self) -> Sample {
        self.step
    }

    /// The dither probability density in use.
    #[must_use]
    pub fn dither_type(&self) -> DitherType {
        self.dither
    }

    /// The noise-shaping curve in use.
    #[must_use]
    pub fn noise_shaping(&self) -> NoiseShaping {
        self.shaping
    }

    /// The number of channels this engine was built for.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.errors.len()
    }

    /// Sets the delivery bit depth (clamped to the supported range),
    /// recomputing the quantization grid.
    pub fn set_bits(&mut self, bits: u32) {
        self.bits = clamp_bits(bits);
        let (step, half) = grid_for(self.bits);
        self.step = step;
        self.half = half;
    }

    /// Sets the dither probability density.
    pub fn set_dither_type(&mut self, dither: DitherType) {
        self.dither = dither;
    }

    /// Sets the noise-shaping curve, updating the feedback taps.
    pub fn set_noise_shaping(&mut self, shaping: NoiseShaping) {
        self.shaping = shaping;
        self.taps = shaping.feedback_taps();
    }

    /// Clears the error-feedback memory and restarts the dither generator from
    /// the original seed, restoring bit-identical behaviour.
    pub fn reset(&mut self) {
        for e in &mut self.errors {
            *e = [0.0; 2];
        }
        self.rng = DitherRng::new(self.seed);
    }

    /// Draws a dither value in sample units (already scaled by one `LSB`).
    #[must_use]
    fn draw_dither(&mut self) -> Sample {
        match self.dither {
            DitherType::None => 0.0,
            // One uniform draw on [-0.5, 0.5): rectangular, one LSB wide.
            DitherType::Rectangular => (self.rng.next_unit() - 0.5) * self.step,
            // Sum of two independent draws: triangular, two LSB wide.
            DitherType::Triangular => {
                let a = self.rng.next_unit();
                let b = self.rng.next_unit();
                (a + b - 1.0) * self.step
            }
        }
    }

    /// Rounds a value onto the current amplitude grid.
    #[must_use]
    fn quantize(&self, value: Sample) -> Sample {
        let code = ops::round(value / self.step).clamp(-self.half, self.half - 1.0);
        code * self.step
    }

    /// Requantizes one sample on channel `channel`, applying dither and noise
    /// shaping. Out-of-range channels and non-finite inputs return `0`.
    #[must_use]
    pub fn process_sample(&mut self, channel: usize, input: Sample) -> Sample {
        if channel >= self.errors.len() {
            return 0.0;
        }
        // Non-finite inputs are treated as clean silence: they bypass the
        // dither and shaper so a stray NaN/inf cannot poison the error-feedback
        // state or leak a one-LSB click into an otherwise silent stream.
        if !input.is_finite() {
            return 0.0;
        }
        let x = input;
        let prev = self.errors[channel];
        let feedback = self.taps[0] * prev[0] + self.taps[1] * prev[1];
        let u = x - feedback;
        let dithered = u + self.draw_dither();
        let y = self.quantize(dithered);
        // Realised error relative to the (pre-dither) shaper input.
        let error = flush_denormal(y - u);
        self.errors[channel] = [error, prev[0]];
        y
    }
}

/// An [`AudioNode`] that renders its input to a fixed delivery bit depth with
/// dither and optional noise shaping.
///
/// This is the final output stage of a mastering chain. It has one input and
/// one output; the signal passes through unchanged in channel count and frame
/// count, requantized sample by sample.
#[derive(Clone, Debug)]
pub struct DitherNode {
    /// The requantizer engine.
    engine: Dither,
}

impl DitherNode {
    /// Creates a dither node for `channels` channels from `params`.
    #[must_use]
    pub fn new(params: DitherParams, channels: usize) -> Self {
        Self {
            engine: Dither::new(params, channels),
        }
    }

    /// Borrows the underlying requantizer engine.
    #[must_use]
    pub fn engine(&self) -> &Dither {
        &self.engine
    }

    /// Mutably borrows the underlying requantizer engine (to retune bit depth,
    /// dither, or shaping at control rate).
    pub fn engine_mut(&mut self) -> &mut Dither {
        &mut self.engine
    }
}

impl AudioNode for DitherNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let channels = output
            .channels()
            .min(input.channels())
            .min(self.engine.channels());
        let frames = output.active_frames();
        if frames == 0 || channels == 0 {
            return;
        }
        for ch in 0..channels {
            for f in 0..frames {
                let x = input.channel(ch)[f];
                output.channel_mut(ch)[f] = self.engine.process_sample(ch, x);
            }
        }
    }

    fn reset(&mut self) {
        self.engine.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn run(node: &mut DitherNode, input: &AudioBuffer) -> AudioBuffer {
        let frames = input.active_frames();
        let mut out = AudioBuffer::new(input.layout(), input.capacity_frames());
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        o
    }

    fn stereo_dc(frames: usize, left: Sample, right: Sample) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Stereo, frames);
        buf.set_active_frames(frames);
        for s in buf.channel_mut(0) {
            *s = left;
        }
        for s in buf.channel_mut(1) {
            *s = right;
        }
        buf
    }

    #[test]
    fn grid_step_matches_bit_depth() {
        let d = Dither::new(
            DitherParams {
                bits: 16,
                dither: DitherType::None,
                shaping: NoiseShaping::None,
                seed: 1,
            },
            1,
        );
        // 16-bit: 65536 codes over [-1, 1), step 2 / 65536.
        assert!((d.quantization_step() - 2.0 / 65_536.0).abs() < 1e-12);
        assert_eq!(d.bits(), 16);
    }

    #[test]
    fn bits_are_clamped_to_supported_range() {
        let low = Dither::new(
            DitherParams {
                bits: 0,
                ..DitherParams::default()
            },
            1,
        );
        assert_eq!(low.bits(), MIN_DITHER_BITS);
        let high = Dither::new(
            DitherParams {
                bits: 999,
                ..DitherParams::default()
            },
            1,
        );
        assert_eq!(high.bits(), MAX_DITHER_BITS);
    }

    #[test]
    fn no_dither_is_plain_rounding_onto_grid() {
        let mut d = Dither::new(
            DitherParams {
                bits: 8,
                dither: DitherType::None,
                shaping: NoiseShaping::None,
                seed: 1,
            },
            1,
        );
        let step = d.quantization_step();
        // Output always lands on an integer multiple of the step.
        for i in 0..200 {
            let x = -0.9 + 0.009 * (i as Sample);
            let y = d.process_sample(0, x);
            let codes = y / step;
            assert!((codes - ops::round(codes)).abs() < 1e-4, "y {y} off-grid");
            assert!(y.is_finite());
        }
    }

    #[test]
    fn output_lands_on_grid_even_with_dither() {
        let mut d = Dither::new(
            DitherParams {
                bits: 12,
                dither: DitherType::Triangular,
                shaping: NoiseShaping::None,
                seed: 7,
            },
            1,
        );
        let step = d.quantization_step();
        for i in 0..500 {
            let x = ops::sin(0.01 * i as Sample) * 0.8;
            let y = d.process_sample(0, x);
            let codes = y / step;
            assert!((codes - ops::round(codes)).abs() < 1e-3, "off grid: {y}");
        }
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let params = DitherParams {
            bits: 16,
            dither: DitherType::Triangular,
            shaping: NoiseShaping::SecondOrder,
            seed: 0xABCD_1234,
        };
        let mut a = Dither::new(params, 1);
        let mut b = Dither::new(params, 1);
        for i in 0..1000 {
            let x = ops::sin(0.013 * i as Sample) * 0.5;
            assert_eq!(a.process_sample(0, x), b.process_sample(0, x));
        }
    }

    #[test]
    fn different_seed_diverges() {
        let base = DitherParams {
            bits: 16,
            dither: DitherType::Triangular,
            shaping: NoiseShaping::None,
            seed: 1,
        };
        let mut a = Dither::new(base, 1);
        let mut b = Dither::new(
            DitherParams { seed: 2, ..base },
            1,
        );
        let mut differed = false;
        for i in 0..1000 {
            let x = ops::sin(0.01 * i as Sample) * 0.3;
            if a.process_sample(0, x) != b.process_sample(0, x) {
                differed = true;
            }
        }
        assert!(differed, "distinct seeds produced identical streams");
    }

    #[test]
    fn reset_restores_deterministic_stream() {
        let params = DitherParams {
            bits: 16,
            dither: DitherType::Triangular,
            shaping: NoiseShaping::FirstOrder,
            seed: 99,
        };
        let mut d = Dither::new(params, 1);
        let mut first = Vec::new();
        for i in 0..500 {
            let x = ops::sin(0.02 * i as Sample) * 0.6;
            first.push(d.process_sample(0, x));
        }
        d.reset();
        for (i, want) in first.iter().enumerate() {
            let x = ops::sin(0.02 * i as Sample) * 0.6;
            assert_eq!(d.process_sample(0, x), *want);
        }
    }

    #[test]
    fn dither_decorrelates_quiet_tone_error() {
        // A tone far below the LSB produces constant (correlated) error with no
        // dither, but a varying error once TPDF dither is added.
        let bits = 8;
        let amp = (2.0 / (1u32 << bits) as Sample) * 0.1; // 0.1 LSB tone.
        let mut flat = Dither::new(
            DitherParams {
                bits,
                dither: DitherType::None,
                shaping: NoiseShaping::None,
                seed: 1,
            },
            1,
        );
        let mut dithered = Dither::new(
            DitherParams {
                bits,
                dither: DitherType::Triangular,
                shaping: NoiseShaping::None,
                seed: 1,
            },
            1,
        );
        let mut flat_unique = 0;
        let mut dith_unique = 0;
        let mut last_flat = Sample::NAN;
        let mut last_dith = Sample::NAN;
        for i in 0..400 {
            let x = ops::sin(0.05 * i as Sample) * amp;
            let yf = flat.process_sample(0, x);
            let yd = dithered.process_sample(0, x);
            if yf != last_flat {
                flat_unique += 1;
                last_flat = yf;
            }
            if yd != last_dith {
                dith_unique += 1;
                last_dith = yd;
            }
        }
        // The undithered quiet tone collapses to a near-constant code; the
        // dithered one keeps toggling.
        assert!(dith_unique > flat_unique, "dith {dith_unique} flat {flat_unique}");
    }

    #[test]
    fn noise_shaping_preserves_grid_and_stays_bounded() {
        let mut d = Dither::new(
            DitherParams {
                bits: 10,
                dither: DitherType::Triangular,
                shaping: NoiseShaping::SecondOrder,
                seed: 5,
            },
            1,
        );
        let step = d.quantization_step();
        for i in 0..2000 {
            let x = ops::sin(0.03 * i as Sample) * 0.7;
            let y = d.process_sample(0, x);
            assert!(y.is_finite());
            assert!(y.abs() <= 1.0 + step, "out of range: {y}");
            let codes = y / step;
            assert!((codes - ops::round(codes)).abs() < 1e-3);
        }
    }

    #[test]
    fn non_finite_input_is_safe() {
        let mut d = Dither::new(DitherParams::default(), 2);
        assert_eq!(d.process_sample(0, Sample::NAN), 0.0);
        assert_eq!(d.process_sample(1, Sample::INFINITY), 0.0);
        assert_eq!(d.process_sample(0, Sample::NEG_INFINITY), 0.0);
    }

    #[test]
    fn out_of_range_channel_returns_zero() {
        let mut d = Dither::new(DitherParams::default(), 1);
        assert_eq!(d.process_sample(5, 0.5), 0.0);
        assert_eq!(d.channels(), 1);
    }

    #[test]
    fn full_scale_input_does_not_overflow_grid() {
        let mut d = Dither::new(
            DitherParams {
                bits: 16,
                dither: DitherType::Triangular,
                shaping: NoiseShaping::None,
                seed: 3,
            },
            1,
        );
        let step = d.quantization_step();
        for &x in &[-1.0, -0.999, 0.999, 1.0, 2.0, -2.0] {
            let y = d.process_sample(0, x);
            assert!(y.is_finite());
            assert!(y >= -1.0 - step && y < 1.0, "y {y} out of signed range");
        }
    }

    #[test]
    fn setters_update_grid_and_shaping() {
        let mut d = Dither::new(DitherParams::default(), 1);
        d.set_bits(24);
        assert_eq!(d.bits(), 24);
        assert!((d.quantization_step() - 2.0 / (1u32 << 24) as Sample).abs() < 1e-12);
        d.set_dither_type(DitherType::Rectangular);
        assert_eq!(d.dither_type(), DitherType::Rectangular);
        d.set_noise_shaping(NoiseShaping::SecondOrder);
        assert_eq!(d.noise_shaping(), NoiseShaping::SecondOrder);
    }

    #[test]
    fn node_passes_through_shape_and_quantizes() {
        let mut node = DitherNode::new(
            DitherParams {
                bits: 12,
                dither: DitherType::Triangular,
                shaping: NoiseShaping::FirstOrder,
                seed: 11,
            },
            2,
        );
        let input = stereo_dc(64, 0.5, -0.5);
        let out = run(&mut node, &input);
        assert_eq!(out.active_frames(), 64);
        assert_eq!(out.channels(), 2);
        let step = node.engine().quantization_step();
        for ch in 0..2 {
            for &s in out.channel(ch) {
                assert!(s.is_finite());
                let codes = s / step;
                assert!((codes - ops::round(codes)).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn node_zero_frames_is_safe() {
        let mut node = DitherNode::new(DitherParams::default(), 2);
        // A live buffer that currently carries no active frames (capacity stays
        // non-zero; AudioBuffer forbids a zero capacity).
        let mut input = stereo_dc(8, 0.0, 0.0);
        input.set_active_frames(0);
        let out = run(&mut node, &input);
        assert_eq!(out.active_frames(), 0);
    }

    #[test]
    fn node_reset_restores_stream() {
        let mut node = DitherNode::new(
            DitherParams {
                bits: 16,
                dither: DitherType::Triangular,
                shaping: NoiseShaping::SecondOrder,
                seed: 42,
            },
            1,
        );
        let input = {
            let mut b = AudioBuffer::new(ChannelLayout::Mono, 128);
            b.set_active_frames(128);
            for (i, s) in b.channel_mut(0).iter_mut().enumerate() {
                *s = ops::sin(0.04 * i as Sample) * 0.6;
            }
            b
        };
        let first = run(&mut node, &input);
        node.reset();
        let second = run(&mut node, &input);
        for (a, b) in first.channel(0).iter().zip(second.channel(0)) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn rng_draws_are_in_unit_interval() {
        let mut rng = DitherRng::new(0xDEAD_BEEF);
        for _ in 0..10_000 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u), "u {u} out of [0,1)");
        }
    }

    #[test]
    fn one_bit_grid_is_well_defined() {
        let mut d = Dither::new(
            DitherParams {
                bits: 1,
                dither: DitherType::None,
                shaping: NoiseShaping::None,
                seed: 1,
            },
            1,
        );
        // 1-bit: step 1.0, codes clamp to {-1, 0}.
        assert!((d.quantization_step() - 1.0).abs() < 1e-9);
        assert_eq!(d.process_sample(0, 0.9), 0.0);
        assert_eq!(d.process_sample(0, -0.9), -1.0);
    }
}
