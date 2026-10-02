//! The path-space sampling driver that feeds low-discrepancy dimensions into
//! the integrator.
//!
//! [`super::film`] already drives the sub-pixel jitter and the thin-lens
//! aperture with Owen-scrambled Sobol (0, 2)-nets. The remaining variance in a
//! path tracer lives in the *on-path* dimensions: the hemisphere and light
//! samples taken at every bounce. [`PathSampler`] closes that gap. It
//! implements [`SampleSource`] so the integrator keeps drawing scalars from a
//! single flat stream, but it transparently routes the first few scalar
//! dimensions through quasi-Monte-Carlo (`QMC`) sequences before padding the
//! tail with a classical `PCG` stream.
//!
//! # Padded `QMC`
//!
//! The [`SampleSource`] contract exposes only [`SampleSource::next_f32`], so the
//! sampler never learns the semantic 2D boundaries the integrator cares about
//! (a hemisphere sample is two consecutive draws, a light sample two more, and
//! so on). We therefore treat the draw stream as a sequence of 2D *pairs*: draw
//! `i` belongs to pair `i / 2`, component `i % 2`. The first
//! [`PATH_QMC_DIMENSION_PAIRS`] pairs are each backed by their own
//! Owen-scrambled Sobol (0, 2)-net, queried at the current sample index; every
//! later pair, and every draw once a path runs past the reserved budget, falls
//! back to the `PCG` stream. This is the classical *padded* `QMC` construction:
//! each reserved scalar dimension is a 1D projection of a (0, 2)-net (an
//! Owen-scrambled van der Corput radical inverse), which is low discrepancy on
//! its own, so any pairing the integrator happens to impose still lowers
//! variance. Dimensions beyond the budget degrade gracefully to independent
//! sampling rather than aliasing.
//!
//! # Determinism
//!
//! A sampler is keyed by `(seed, pixel_index)` and re-pointed at each
//! per-pixel sample through [`PathSampler::reset_for_sample`]. Identical
//! arguments reproduce an identical scalar stream, so the whole render stays
//! bit-identical across runs and order-independent across pixels.
//!
//! [`PathSampler::passthrough`] disables the `QMC` reservation entirely and
//! forwards every draw to the `PCG` stream, which reproduces the pre-existing
//! independent-sampling renderer bit-for-bit.

use super::sampler::{Rng, Sample2, SampleSource};
use super::sobol::OwenScrambledSobolSampler;

/// The number of leading 2D dimension pairs backed by Owen-scrambled Sobol
/// (0, 2)-nets before the sampler pads the tail with the `PCG` stream.
///
/// Eight pairs reserve sixteen scalar dimensions, enough to cover the dominant
/// hemisphere and next-event light samples of the first several bounces, where
/// the integrand carries the most energy and `QMC` stratification pays off the
/// most. Deeper bounces contribute progressively less, so padding them with
/// independent samples trades negligible quality for a bounded sampler state.
pub const PATH_QMC_DIMENSION_PAIRS: usize = 8;

/// Salt that decorrelates the on-path `QMC` streams from the sub-pixel jitter
/// and thin-lens aperture streams.
///
/// It is the multiplier stage of the `SplitMix64` finaliser, an odd 64-bit
/// constant with good avalanche, applied so a shared render seed produces
/// unrelated Owen scramble seeds for the path dimensions.
const PATH_STREAM_SALT: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second mixing multiplier of the `SplitMix64` finaliser used to derive a
/// decorrelated Owen scramble seed per reserved dimension pair.
const PATH_STREAM_MIX: u64 = 0x94D0_49BB_1331_11EB;

/// Derives a decorrelated 64-bit stream salt for the `pair`-th reserved
/// dimension pair.
///
/// The pair index is run through a `SplitMix64` finaliser so each reserved pair
/// owns an Owen scramble seed that is statistically independent of its
/// neighbours and of the sub-pixel and aperture streams. Pure integer mixing
/// keeps the derivation reproducible and free of transcendental functions.
#[must_use]
fn per_pair_salt(pair: usize) -> u64 {
    let mut z = (pair as u64).wrapping_add(PATH_STREAM_SALT);
    z = (z ^ (z >> 30)).wrapping_mul(PATH_STREAM_SALT);
    z = (z ^ (z >> 27)).wrapping_mul(PATH_STREAM_MIX);
    z ^ (z >> 31)
}

/// Drives the integrator's on-path scalar draws, routing the leading
/// dimensions through Owen-scrambled Sobol (0, 2)-nets and padding the tail
/// with a classical `PCG` stream.
///
/// See the [module documentation](self) for the padded-`QMC` construction and
/// the determinism guarantees.
#[derive(Clone, Debug)]
pub struct PathSampler {
    /// One Owen-scrambled Sobol (0, 2)-net per reserved dimension pair, or
    /// [`None`] in pass-through mode where every draw uses the `PCG` stream.
    qmc: Option<[OwenScrambledSobolSampler; PATH_QMC_DIMENSION_PAIRS]>,
    /// The (0, 2)-net sample for the pair currently being read, cached so the
    /// `x` and `y` components of one pair come from a single net query.
    cached_pair: Sample2,
    /// The `PCG` stream that pads dimensions past the reserved budget and backs
    /// the whole stream in pass-through mode.
    rng: Rng,
    /// The index of the per-pixel sample currently being traced; selects which
    /// point of each reserved (0, 2)-net the draws read.
    sample_index: u64,
    /// The number of scalars drawn so far within the current sample, used to
    /// map draws onto dimension pairs and components.
    cursor: u32,
}

impl PathSampler {
    /// Builds a sampler that routes the first [`PATH_QMC_DIMENSION_PAIRS`]
    /// dimension pairs through Owen-scrambled Sobol (0, 2)-nets keyed by
    /// `(seed, pixel_index)` and pads the remaining draws with `rng`.
    ///
    /// `rng` must be the pixel's dedicated `PCG` stream; it advances only on
    /// padded draws, so the reserved `QMC` dimensions never perturb it.
    #[must_use]
    pub fn new(seed: u64, pixel_index: u64, rng: Rng) -> Self {
        let qmc = core::array::from_fn(|pair| {
            OwenScrambledSobolSampler::new(seed ^ per_pair_salt(pair), pixel_index)
        });
        Self {
            qmc: Some(qmc),
            cached_pair: Sample2 { x: 0.0, y: 0.0 },
            rng,
            sample_index: 0,
            cursor: 0,
        }
    }

    /// Builds a sampler that forwards every draw to `rng`, reproducing the
    /// classical independent-sampling renderer bit-for-bit.
    ///
    /// This is the driver the default render path uses so that enabling the
    /// `QMC` path is an explicit, opt-in choice that never silently changes the
    /// reference image.
    #[must_use]
    pub fn passthrough(rng: Rng) -> Self {
        Self {
            qmc: None,
            cached_pair: Sample2 { x: 0.0, y: 0.0 },
            rng,
            sample_index: 0,
            cursor: 0,
        }
    }

    /// Re-points the sampler at the `sample_index`-th per-pixel sample.
    ///
    /// The reserved (0, 2)-nets are read at this index while the draw cursor
    /// rewinds to the first dimension, so every sample of a pixel starts a
    /// fresh low-discrepancy path. The padding `PCG` stream is intentionally
    /// left running: consecutive samples draw from disjoint tail segments, just
    /// as the independent-sampling renderer did.
    pub fn reset_for_sample(&mut self, sample_index: u64) {
        self.sample_index = sample_index;
        self.cursor = 0;
    }
}

impl SampleSource for PathSampler {
    fn next_f32(&mut self) -> f32 {
        let draw = self.cursor;
        self.cursor = self.cursor.wrapping_add(1);
        if let Some(qmc) = &self.qmc {
            let pair = (draw / 2) as usize;
            if pair < qmc.len() {
                if draw.is_multiple_of(2) {
                    self.cached_pair = qmc[pair].sample(self.sample_index);
                    return self.cached_pair.x;
                }
                return self.cached_pair.y;
            }
        }
        self.rng.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pass-through mode must reproduce the raw `PCG` stream exactly, so the
    /// default renderer stays bit-identical to the independent-sampling path.
    #[test]
    fn passthrough_matches_raw_rng_bit_for_bit() {
        let mut reference = Rng::with_stream(7, 11);
        let mut sampler = PathSampler::passthrough(Rng::with_stream(7, 11));
        for _ in 0..64 {
            assert_eq!(sampler.next_f32().to_bits(), reference.next_f32().to_bits());
        }
    }

    /// The reserved pairs must reconstruct the underlying (0, 2)-net exactly:
    /// the even draw returns the net's `x`, the following odd draw its `y`, all
    /// from a single net query at the active sample index.
    #[test]
    fn reserved_pairs_reconstruct_the_net() {
        let seed = 0x1234_5678;
        let pixel = 42;
        let mut sampler = PathSampler::new(seed, pixel, Rng::with_stream(seed, pixel + 1));
        sampler.reset_for_sample(5);
        for pair in 0..PATH_QMC_DIMENSION_PAIRS {
            let direct =
                OwenScrambledSobolSampler::new(seed ^ per_pair_salt(pair), pixel).sample(5);
            let x = sampler.next_f32();
            let y = sampler.next_f32();
            assert_eq!(x.to_bits(), direct.x.to_bits());
            assert_eq!(y.to_bits(), direct.y.to_bits());
        }
    }

    /// Every draw, reserved or padded, must stay inside the half-open unit
    /// interval so the integrator's warps never see an out-of-range coordinate.
    #[test]
    fn all_draws_stay_in_unit_interval() {
        let mut sampler = PathSampler::new(99, 3, Rng::with_stream(99, 4));
        for s in 0..32 {
            sampler.reset_for_sample(s);
            for _ in 0..40 {
                let u = sampler.next_f32();
                assert!(
                    (0.0..1.0).contains(&u),
                    "draw {u} escaped the unit interval"
                );
            }
        }
    }

    /// Draws past the reserved budget must fall back to the `PCG` stream, so a
    /// long path keeps producing well-distributed samples instead of aliasing.
    #[test]
    fn draws_past_the_budget_pad_from_the_rng() {
        let seed = 5;
        let pixel = 1;
        let mut padding = Rng::with_stream(seed, pixel + 1);
        let mut sampler = PathSampler::new(seed, pixel, Rng::with_stream(seed, pixel + 1));
        sampler.reset_for_sample(0);
        // Consume the reserved dimensions first.
        for _ in 0..(2 * PATH_QMC_DIMENSION_PAIRS) {
            let _ = sampler.next_f32();
        }
        // The next draws must match the untouched padding stream.
        for _ in 0..16 {
            assert_eq!(sampler.next_f32().to_bits(), padding.next_f32().to_bits());
        }
    }

    /// A sampler keyed identically must reproduce an identical scalar stream so
    /// the render stays deterministic.
    #[test]
    fn same_key_is_deterministic() {
        let make = || PathSampler::new(13, 2, Rng::with_stream(13, 3));
        let mut a = make();
        let mut b = make();
        for s in 0..8 {
            a.reset_for_sample(s);
            b.reset_for_sample(s);
            for _ in 0..24 {
                assert_eq!(a.next_f32().to_bits(), b.next_f32().to_bits());
            }
        }
    }

    /// The reserved dimension zero is an Owen-scrambled van der Corput radical
    /// inverse, so integrating the identity over a power-of-two sample budget
    /// through it converges far tighter than the independent `PCG` stream.
    #[test]
    fn reserved_dimension_integrates_tighter_than_independent() {
        const SAMPLES: u32 = 4096;
        let seed = 2024;
        let pixel = 77;

        let mut sampler = PathSampler::new(seed, pixel, Rng::with_stream(seed, pixel + 1));
        let mut qmc_sum = 0.0_f64;
        for s in 0..SAMPLES {
            sampler.reset_for_sample(u64::from(s));
            qmc_sum += f64::from(sampler.next_f32());
        }
        let qmc_error = (qmc_sum / f64::from(SAMPLES) - 0.5).abs();

        let mut rng = Rng::with_stream(seed ^ 0xABCD, 9);
        let mut independent_sum = 0.0_f64;
        for _ in 0..SAMPLES {
            independent_sum += f64::from(rng.next_f32());
        }
        let independent_error = (independent_sum / f64::from(SAMPLES) - 0.5).abs();

        // The (0, 2)-net stratifies a power-of-two budget into exact dyadic
        // strata, so the quasi-Monte-Carlo estimate of a linear integrand is
        // tight in an absolute sense and strictly beats independent sampling.
        assert!(qmc_error < 1e-3, "QMC error {qmc_error} is not tight");
        assert!(
            qmc_error < independent_error,
            "QMC error {qmc_error} did not beat independent error {independent_error}",
        );
    }
}
