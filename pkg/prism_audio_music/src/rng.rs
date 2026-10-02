//! A tiny self-contained deterministic pseudo-random generator used by random
//! and shuffle playlists.
//!
//! The music planner must be bit-for-bit reproducible across runs and
//! platforms (replays, networked lockstep, golden tests), so it never touches
//! the OS entropy source. Instead the live [`crate::system::MusicSystem`] owns
//! a seeded [`Rng`] whose sequence is a pure function of its seed.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The generator is
//! Marsaglia's public-domain xorshift64 (G. Marsaglia, "Xorshift RNGs",
//! *Journal of Statistical Software*, 2003), a classic published algorithm.
//! No AI/ML.
//!
//! # Relationship
//!
//! [`Rng`] seeds the random/shuffle selection in [`crate::playlist`]. It holds
//! no audio state and performs no DSP.

/// Deterministic xorshift64 generator (shift triple 13/7/17, period 2^64 - 1).
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

    /// Returns a uniform integer index in `[0, len)`.
    ///
    /// Returns `0` when `len == 0` (callers must guard empty collections).
    pub fn next_index(&mut self, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        // Multiply-shift mapping of a 32-bit draw: unbiased enough for small
        // playlist sizes and fully deterministic.
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
    fn index_zero_len_is_zero_and_bounded() {
        let mut r = Rng::new(9);
        assert_eq!(r.next_index(0), 0);
        for _ in 0..10_000 {
            assert!(r.next_index(7) < 7);
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
