//! Selectable sub-pixel jitter sequence shared by the fixed-budget and adaptive
//! renderers.
//!
//! Both the scrambled Halton sampler ([`super::halton`]) and the Owen-scrambled
//! Sobol sampler ([`super::sobol`]) expose the same contract: build one
//! sequence per pixel from a `(seed, pixel_index)` pair, then draw the
//! `sample_index`-th sub-pixel offset in `[0, 1)^2`. This module makes that
//! choice a first-class, data-driven option so the renderers can switch the
//! primary-visibility quasi-Monte-Carlo (`QMC`) sequence without duplicating
//! their pixel loops.
//!
//! The Halton base-2/3 sampler with Cranley-Patterson (`CP`) rotation is the
//! default because it keeps existing images bit-identical; the Owen-scrambled
//! Sobol (0, 2)-net is the stronger construction at higher sample counts and is
//! what production path tracers reach for.

use super::halton::HaltonPixelSampler;
use super::sampler::Sample2;
use super::sobol::OwenScrambledSobolSampler;

/// Seed salt mixed in to derive the thin-lens aperture's quasi-random stream
/// from the sub-pixel jitter seed.
///
/// Reusing the same `(seed, pixel_index)` for both the sub-pixel jitter and the
/// lens sample would couple the two dimensions and leave visible structure in
/// the bokeh. Mixing this golden-ratio constant (`floor(2^64 / phi)`, odd, with
/// a well-mixed bit pattern) into the seed yields a decorrelated but still
/// low-discrepancy lens stream from the same sampler.
pub(crate) const LENS_STREAM_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Which low-discrepancy sequence drives the two sub-pixel jitter dimensions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SubpixelSampler {
    /// The scrambled Halton base-2/3 sampler with per-pixel `CP` rotation. The
    /// default, chosen so existing renders stay bit-identical.
    #[default]
    Halton,
    /// The Owen-scrambled Sobol (0, 2)-net sampler. Lower integration error at
    /// higher sample counts, with per-pixel decorrelation from the Owen scramble.
    OwenSobol,
}

/// A per-pixel sub-pixel jitter sequence, built from a [`SubpixelSampler`]
/// choice. Holds whichever concrete sampler was selected so the per-sample draw
/// is a cheap match with no further allocation.
#[derive(Clone, Copy, Debug)]
pub enum PixelSequence {
    /// A scrambled Halton sequence for this pixel.
    Halton(HaltonPixelSampler),
    /// An Owen-scrambled Sobol sequence for this pixel.
    OwenSobol(OwenScrambledSobolSampler),
}

impl SubpixelSampler {
    /// Builds the per-pixel jitter sequence for the pixel at flat `pixel_index`,
    /// seeded by `seed`. Each concrete sampler derives its own decorrelation
    /// state so neighbouring pixels never share a jitter pattern.
    #[must_use]
    pub fn build(self, seed: u64, pixel_index: u64) -> PixelSequence {
        match self {
            Self::Halton => PixelSequence::Halton(HaltonPixelSampler::new(seed, pixel_index)),
            Self::OwenSobol => {
                PixelSequence::OwenSobol(OwenScrambledSobolSampler::new(seed, pixel_index))
            }
        }
    }
}

impl PixelSequence {
    /// The sub-pixel jitter for the `sample_index`-th sample, in `[0, 1)^2`.
    #[must_use]
    pub fn sample(&self, sample_index: u64) -> Sample2 {
        match self {
            Self::Halton(sampler) => sampler.sample(sample_index),
            Self::OwenSobol(sampler) => sampler.sample(sample_index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_halton() {
        assert_eq!(SubpixelSampler::default(), SubpixelSampler::Halton);
    }

    #[test]
    fn halton_sequence_matches_direct_sampler() {
        let direct = HaltonPixelSampler::new(11, 42);
        let seq = SubpixelSampler::Halton.build(11, 42);
        for s in 0..64u64 {
            let a = direct.sample(s);
            let b = seq.sample(s);
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
        }
    }

    #[test]
    fn owen_sobol_sequence_matches_direct_sampler() {
        let direct = OwenScrambledSobolSampler::new(11, 42);
        let seq = SubpixelSampler::OwenSobol.build(11, 42);
        for s in 0..64u64 {
            let a = direct.sample(s);
            let b = seq.sample(s);
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
        }
    }

    #[test]
    fn the_two_sequences_differ() {
        // The selection must be observable: the two constructions produce
        // different sub-pixel offsets for the same pixel and sample index.
        let halton = SubpixelSampler::Halton.build(3, 7);
        let sobol = SubpixelSampler::OwenSobol.build(3, 7);
        let mut any_different = false;
        for s in 0..64u64 {
            let a = halton.sample(s);
            let b = sobol.sample(s);
            if a.x.to_bits() != b.x.to_bits() || a.y.to_bits() != b.y.to_bits() {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "the two samplers must not be identical");
    }
}
