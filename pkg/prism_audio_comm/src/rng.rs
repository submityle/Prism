//! Deterministic seeded pseudo-random generator for the communication crate.
//!
//! Comfort-noise injection in packet-loss concealment and the dithered fade
//! tails need a reproducible noise source: given the same integer seed and the
//! same call sequence, the generator yields a bit-identical stream on every
//! platform. The core is a `xorshift64` word mixed by a `SplitMix64` seeding
//! step; both are standard, publicly documented integer generators and the
//! whole module is pure integer arithmetic with a single floating-point
//! division at the output, so stream order never diverges across targets.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Backs design section 45.3 (packet-loss concealment comfort noise) with the
//! same integer-only RNG discipline used by `prism_audio_procedural::rng`,
//! without sharing code.

use prism_audio_core::math::Sample;

/// A reproducible `xorshift64` generator seeded through `SplitMix64`.
///
/// The generator holds a single 64-bit word and is `Copy`, so a snapshot of
/// its state can be taken and later restored to replay an identical stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CommRng {
    state: u64,
}

impl CommRng {
    /// Creates a generator from the integer `seed`.
    ///
    /// The seed is passed through a `SplitMix64` mixing step so that nearby
    /// seeds (`0`, `1`, `2`, ...) still produce well-separated streams. A
    /// resulting all-zero state is remapped to a fixed non-zero constant
    /// because `xorshift` cannot escape the all-zero state.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self {
            state: if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z },
        }
    }

    /// Returns the next 64-bit pseudo-random word and advances the state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform [`Sample`] in the half-open range `[0, 1)`.
    #[inline]
    pub fn next_unit(&mut self) -> Sample {
        // Use the top 24 bits so every representable mantissa step is reachable.
        let bits = self.next_u64() >> 40;
        (bits as Sample) / ((1u32 << 24) as Sample)
    }

    /// Returns a uniform [`Sample`] in the bipolar range `[-1, 1)`.
    #[inline]
    pub fn next_bipolar(&mut self) -> Sample {
        self.next_unit() * 2.0 - 1.0
    }

    /// Captures the current state for a later [`CommRng::restore`].
    #[inline]
    #[must_use]
    pub fn snapshot(&self) -> u64 {
        self.state
    }

    /// Restores a state previously captured with [`CommRng::snapshot`].
    #[inline]
    pub fn restore(&mut self, state: u64) {
        self.state = if state == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            state
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = CommRng::new(0xDEAD_BEEF);
        let mut b = CommRng::new(0xDEAD_BEEF);
        for _ in 0..256 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_differs() {
        let mut a = CommRng::new(1);
        let mut b = CommRng::new(2);
        let mut diff = false;
        for _ in 0..32 {
            if a.next_u64() != b.next_u64() {
                diff = true;
                break;
            }
        }
        assert!(diff);
    }

    #[test]
    fn unit_is_in_range() {
        let mut rng = CommRng::new(42);
        for _ in 0..10_000 {
            let u = rng.next_unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn snapshot_replays_stream() {
        let mut rng = CommRng::new(99);
        let snap = rng.snapshot();
        let first: [u64; 8] = core::array::from_fn(|_| rng.next_u64());
        rng.restore(snap);
        let second: [u64; 8] = core::array::from_fn(|_| rng.next_u64());
        assert_eq!(first, second);
    }

    #[test]
    fn zero_seed_is_non_degenerate() {
        let mut rng = CommRng::new(0);
        assert_ne!(rng.next_u64(), 0);
    }
}
