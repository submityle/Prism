//! Deterministic, seedable pseudo-random number generator.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Shared determinism primitive for design section 11 (procedural noise source
//! nodes) and section 12 (sample-and-hold and random modulation sources). All
//! stochastic authoring behaviour routes through this generator so that a given
//! seed reproduces an identical sample stream across platforms.

use prism_audio_core::Sample;

/// A small, fast, fully deterministic 64-bit pseudo-random generator.
///
/// This is a `SplitMix64`-style mixing generator: a monotonically advancing
/// state combined with an avalanche finalizer. It needs no `unsafe`, carries no
/// heap state, and is `Copy`, so it can live inside real-time
/// [`AudioNode`](prism_audio_core::graph::AudioNode)s and modulation sources.
///
/// It is not cryptographically secure; it is intended purely for reproducible
/// audio content generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Golden-ratio odd increment used by the `SplitMix64` advance step.
    const INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Creates a generator from an explicit 64-bit seed.
    ///
    /// Every seed yields a distinct, repeatable stream; seed `0` is remapped so
    /// the first draw is still well distributed.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xDEAD_BEEF_CAFE_F00D,
        }
    }

    /// Advances the state and returns the next raw 64-bit draw.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::INCREMENT);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns the next draw as a 32-bit value.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Returns a uniform [`Sample`] in the half-open range `[0, 1)`.
    #[inline]
    pub fn next_unit(&mut self) -> Sample {
        // Use the high 24 bits for an exact float in [0, 1) (24-bit mantissa).
        let bits = self.next_u64() >> 40; // 24 bits.
        (bits as Sample) * (1.0 / 16_777_216.0)
    }

    /// Returns a uniform [`Sample`] in the bipolar range `[-1, 1)`.
    #[inline]
    pub fn next_bipolar(&mut self) -> Sample {
        self.next_unit() * 2.0 - 1.0
    }
}

impl Default for Rng {
    #[inline]
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-6;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(12_345);
        let mut b = Rng::new(12_345);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let mut equal = 0usize;
        for _ in 0..100 {
            if a.next_u64() == b.next_u64() {
                equal += 1;
            }
        }
        assert!(equal < 5, "streams should rarely collide, got {equal}");
    }

    #[test]
    fn unit_is_in_range() {
        let mut r = Rng::new(99);
        for _ in 0..10_000 {
            let v = r.next_unit();
            assert!((0.0..1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn bipolar_is_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let v = r.next_bipolar();
            assert!((-1.0..1.0).contains(&v), "{v}");
        }
    }

    #[test]
    fn mean_is_roughly_centered() {
        let mut r = Rng::new(2024);
        let mut sum = 0.0f64;
        let n = 100_000;
        for _ in 0..n {
            sum += f64::from(r.next_unit());
        }
        let mean = sum / f64::from(n);
        assert!((mean - 0.5).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn unit_lower_bound_reachable_is_nonnegative() {
        let mut r = Rng::new(0);
        let v = r.next_unit();
        assert!(v >= 0.0 - EPS);
    }
}
