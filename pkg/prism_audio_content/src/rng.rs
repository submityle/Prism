//! A tiny self-contained deterministic pseudo-random generator used by random
//! containers and value randomisation.
//!
//! The content model must be bit-for-bit reproducible across runs and
//! platforms (replays, networked lockstep, golden tests), so it never touches
//! the OS entropy source. Instead each randomising site owns a seeded
//! [`Rng`] whose sequence is a pure function of its seed.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The generator is
//! Marsaglia's public-domain xorshift64 (G. Marsaglia, "Xorshift RNGs",
//! *Journal of Statistical Software*, 2003), the same classic algorithm used
//! by the engine's noise sources. No AI/ML.
//!
//! # Relationship
//!
//! [`Rng`] seeds the shuffle/weighted selection in [`crate::container`] and the
//! per-play value randomisation. It holds no audio state.

use prism_audio_core::math::Sample;

/// Deterministic xorshift64 generator (shift triple 13/7/17, period 2^64 − 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seeds the generator. A zero seed is remapped to a fixed non-zero
    /// constant because xorshift cannot leave the all-zero state.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state }
    }

    /// Advances the generator and returns the next 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniformly distributed value in `[0, 1)`.
    ///
    /// Uses the top 24 bits (the best-quality bits of xorshift) scaled by
    /// `2^-24`, giving an exact dyadic value with no bias toward `1.0`.
    pub fn next_unit(&mut self) -> Sample {
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits
        (bits as Sample) * (1.0 / 16_777_216.0)
    }

    /// Returns a uniformly distributed value in `[min, max]`.
    ///
    /// If `max < min` the bounds are swapped so the range is always valid.
    pub fn next_range(&mut self, min: Sample, max: Sample) -> Sample {
        let (lo, hi) = if max < min { (max, min) } else { (min, max) };
        lo + self.next_unit() * (hi - lo)
    }

    /// Returns a uniform integer index in `[0, len)`.
    ///
    /// Returns `0` when `len == 0` (callers must guard empty collections).
    pub fn next_index(&mut self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        // Multiply-shift mapping of a 32-bit draw: unbiased enough for small
        // container sizes and fully deterministic.
        let draw = self.next_u64() >> 32;
        ((draw * len as u64) >> 32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_seed_is_remapped_and_nonzero() {
        let mut a = Rng::new(0);
        // A zero seed must not get stuck at the all-zero fixed point.
        assert_ne!(a.next_u64(), 0);
    }

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(0xDEAD_BEEF);
        let mut b = Rng::new(0xDEAD_BEEF);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let mut differed = false;
        for _ in 0..64 {
            if a.next_u64() != b.next_u64() {
                differed = true;
                break;
            }
        }
        assert!(differed);
    }

    #[test]
    fn unit_is_in_half_open_range() {
        let mut r = Rng::new(42);
        for _ in 0..100_000 {
            let v = r.next_unit();
            assert!((0.0..1.0).contains(&v), "unit draw out of range: {v}");
        }
    }

    #[test]
    fn range_respects_bounds_and_swaps() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let v = r.next_range(5.0, -5.0); // swapped on purpose
            assert!((-5.0..=5.0).contains(&v));
        }
    }

    #[test]
    fn index_zero_len_is_zero_and_bounded() {
        let mut r = Rng::new(9);
        assert_eq!(r.next_index(0), 0);
        for _ in 0..10_000 {
            let i = r.next_index(7);
            assert!(i < 7);
        }
    }

    #[test]
    fn index_covers_all_buckets() {
        let mut r = Rng::new(123);
        let mut seen = [false; 5];
        for _ in 0..10_000 {
            seen[r.next_index(5)] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }
}
