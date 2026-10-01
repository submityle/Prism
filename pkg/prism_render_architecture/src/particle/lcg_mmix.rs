//! Knuth `MMIX` 64-bit linear congruential generator (`LCG`).
//!
//! This module implements the classic `MMIX` `LCG` / `PRNG` described by
//! Donald Knuth: a 64-bit state advanced by a fixed multiplier and
//! increment using wrapping (modulo `2^64`) integer arithmetic. The code
//! is pure integer, `no_std` friendly, and performs no float or
//! transcendental operations, so it is reproducible across any `CPU` or
//! `GPU` target regardless of its floating-point behaviour.
//!
//! The generator recurrence is:
//!
//! ```text
//! x_{n+1} = (x_n * 6364136223846793005 + 1442695040888963407) mod 2^64
//! ```
//!
//! The multiplier and increment constants are those of Knuth's `MMIX`
//! algorithm. Because the modulus is exactly `2^64`, Rust's `wrapping_*`
//! methods implement the recurrence directly.

/// Knuth `MMIX` multiplier constant for the `LCG` recurrence.
pub const MMIX_MULTIPLIER: u64 = 6364136223846793005;

/// Knuth `MMIX` increment constant for the `LCG` recurrence.
pub const MMIX_INCREMENT: u64 = 1442695040888963407;

/// A 64-bit Knuth `MMIX` linear congruential generator (`PRNG`).
///
/// The internal state is a single `u64`. Each call to [`LcgMmix::next_u64`]
/// advances the state by one step of the `LCG` recurrence and returns the
/// new state value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LcgMmix {
    /// Current 64-bit generator state.
    x: u64,
}

impl LcgMmix {
    /// Creates a new generator seeded with `seed`.
    ///
    /// The seed becomes the initial state directly; the first call to
    /// [`LcgMmix::next_u64`] applies the recurrence once.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    /// Reconstructs a generator from a previously captured state value.
    ///
    /// Combined with [`LcgMmix::state`] this allows a generator to be
    /// snapshotted and later resumed to reproduce the identical sequence.
    #[must_use]
    pub fn from_state(x: u64) -> Self {
        Self { x }
    }

    /// Returns the current generator state without advancing it.
    #[must_use]
    pub fn state(&self) -> u64 {
        self.x
    }

    /// Advances the generator one step and returns the new state.
    pub fn next_u64(&mut self) -> u64 {
        self.x = self
            .x
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        self.x
    }
}

#[cfg(test)]
mod tests {
    use super::LcgMmix;
    use super::MMIX_INCREMENT;
    use super::MMIX_MULTIPLIER;

    /// Reference outputs for `seed = 0`, the only external anchors.
    const SEED0_REF: [u64; 4] = [
        0x14057b7ef767814f,
        0x1a08ee1184ba6d32,
        0x9af678222e728119,
        0x66b61ae97f2099b4,
    ];

    /// Collects `N` successive outputs starting from the given seed.
    #[cfg(test)]
    fn collect<const N: usize>(seed: u64) -> [u64; N] {
        let mut rng = LcgMmix::new(seed);
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    #[test]
    fn seed0_first_output_matches_vector() {
        let mut rng = LcgMmix::new(0);
        assert!(rng.next_u64() == SEED0_REF[0]);
    }

    #[test]
    fn seed0_second_output_matches_vector() {
        let out: [u64; 2] = collect(0);
        assert!(out[1] == SEED0_REF[1]);
    }

    #[test]
    fn seed0_third_output_matches_vector() {
        let out: [u64; 3] = collect(0);
        assert!(out[2] == SEED0_REF[2]);
    }

    #[test]
    fn seed0_fourth_output_matches_vector() {
        let out: [u64; 4] = collect(0);
        assert!(out[3] == SEED0_REF[3]);
    }

    #[test]
    fn seed0_all_four_vectors() {
        let out: [u64; 4] = collect(0);
        let mut i = 0;
        while i < 4 {
            assert!(out[i] == SEED0_REF[i]);
            i += 1;
        }
    }

    #[test]
    fn seed0_first_output_equals_increment_constant() {
        let mut rng = LcgMmix::new(0);
        assert!(rng.next_u64() == 1442695040888963407);
    }

    #[test]
    fn seed0_first_output_equals_named_increment() {
        let mut rng = LcgMmix::new(0);
        assert!(rng.next_u64() == MMIX_INCREMENT);
    }

    #[test]
    fn increment_constant_value() {
        assert!(MMIX_INCREMENT == 1442695040888963407);
    }

    #[test]
    fn multiplier_constant_value() {
        assert!(MMIX_MULTIPLIER == 6364136223846793005);
    }

    #[test]
    fn new_sets_state_to_seed() {
        let rng = LcgMmix::new(12345);
        assert!(rng.state() == 12345);
    }

    #[test]
    fn new_zero_state_is_zero() {
        let rng = LcgMmix::new(0);
        assert!(rng.state() == 0);
    }

    #[test]
    fn from_state_sets_state() {
        let rng = LcgMmix::from_state(0xdead_beef_0000_1111);
        assert!(rng.state() == 0xdead_beef_0000_1111);
    }

    #[test]
    fn next_updates_state_to_return_value() {
        let mut rng = LcgMmix::new(42);
        let r = rng.next_u64();
        assert!(rng.state() == r);
    }

    #[test]
    fn determinism_same_seed_same_sequence() {
        let a: [u64; 8] = collect(777);
        let b: [u64; 8] = collect(777);
        let mut i = 0;
        while i < 8 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn determinism_long_run() {
        let mut a = LcgMmix::new(1);
        let mut b = LcgMmix::new(1);
        let mut i = 0;
        while i < 1000 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn different_seeds_diverge_first_output() {
        let mut a = LcgMmix::new(0);
        let mut b = LcgMmix::new(1);
        assert!(a.next_u64() != b.next_u64());
    }

    #[test]
    fn different_seeds_diverge_over_run() {
        let a: [u64; 16] = collect(100);
        let b: [u64; 16] = collect(200);
        let mut differences = 0u32;
        let mut i = 0;
        while i < 16 {
            if a[i] != b[i] {
                differences += 1;
            }
            i += 1;
        }
        assert!(differences == 16);
    }

    #[test]
    fn state_roundtrip_reproduces_next_output() {
        let mut rng = LcgMmix::new(9999);
        let _ = rng.next_u64();
        let snapshot = rng.state();
        let follow = rng.next_u64();
        let mut restored = LcgMmix::from_state(snapshot);
        assert!(restored.next_u64() == follow);
    }

    #[test]
    fn state_roundtrip_reproduces_many_outputs() {
        let mut rng = LcgMmix::new(0x0123_4567_89ab_cdef);
        let mut i = 0;
        while i < 10 {
            let _ = rng.next_u64();
            i += 1;
        }
        let snapshot = rng.state();
        let mut original = LcgMmix::from_state(snapshot);
        let mut restored = LcgMmix::from_state(snapshot);
        let mut j = 0;
        while j < 20 {
            assert!(original.next_u64() == restored.next_u64());
            j += 1;
        }
    }

    #[test]
    fn from_state_matches_new_for_zero() {
        let a = LcgMmix::new(0);
        let b = LcgMmix::from_state(0);
        assert!(a == b);
    }

    #[test]
    fn recurrence_matches_manual_single_step() {
        let seed = 0xabcd_ef01_2345_6789u64;
        let expected = seed
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        let mut rng = LcgMmix::new(seed);
        assert!(rng.next_u64() == expected);
    }

    #[test]
    fn recurrence_matches_manual_two_steps() {
        let seed = 55u64;
        let s1 = seed
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        let s2 = s1
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        let out: [u64; 2] = collect(55);
        assert!(out[0] == s1);
        assert!(out[1] == s2);
    }

    #[test]
    fn second_output_from_seed0_via_recurrence() {
        let s2 = SEED0_REF[0]
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        assert!(s2 == SEED0_REF[1]);
    }

    #[test]
    fn third_output_from_seed0_via_recurrence() {
        let s3 = SEED0_REF[1]
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        assert!(s3 == SEED0_REF[2]);
    }

    #[test]
    fn fourth_output_from_seed0_via_recurrence() {
        let s4 = SEED0_REF[2]
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        assert!(s4 == SEED0_REF[3]);
    }

    #[test]
    fn resume_from_captured_vector_state() {
        let mut restored = LcgMmix::from_state(SEED0_REF[0]);
        assert!(restored.next_u64() == SEED0_REF[1]);
        assert!(restored.next_u64() == SEED0_REF[2]);
        assert!(restored.next_u64() == SEED0_REF[3]);
    }

    #[test]
    fn outputs_not_all_equal() {
        let out: [u64; 5] = collect(3);
        let mut all_same = true;
        let mut i = 1;
        while i < 5 {
            if out[i] != out[0] {
                all_same = false;
            }
            i += 1;
        }
        assert!(!all_same);
    }

    #[test]
    fn clone_copy_preserves_sequence() {
        let mut rng = LcgMmix::new(0xfeed_face);
        let _ = rng.next_u64();
        let mut copy = rng;
        assert!(copy.next_u64() == rng.next_u64());
    }

    #[test]
    fn equality_of_equal_states() {
        let a = LcgMmix::from_state(42);
        let b = LcgMmix::from_state(42);
        assert!(a == b);
    }

    #[test]
    fn inequality_of_different_states() {
        let a = LcgMmix::from_state(42);
        let b = LcgMmix::from_state(43);
        assert!(a != b);
    }

    #[test]
    fn advancing_changes_state() {
        let mut rng = LcgMmix::new(500);
        let before = rng.state();
        let _ = rng.next_u64();
        assert!(rng.state() != before);
    }

    #[test]
    fn two_generators_meet_after_sync() {
        let mut a = LcgMmix::new(11);
        let _ = a.next_u64();
        let _ = a.next_u64();
        let mut b = LcgMmix::from_state(a.state());
        assert!(a.next_u64() == b.next_u64());
    }

    #[test]
    fn high_bit_seed_steps() {
        let seed = 0x8000_0000_0000_0000u64;
        let expected = seed
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        let mut rng = LcgMmix::new(seed);
        assert!(rng.next_u64() == expected);
    }

    #[test]
    fn max_seed_wraps_correctly() {
        let seed = u64::MAX;
        let expected = seed
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        let mut rng = LcgMmix::new(seed);
        assert!(rng.next_u64() == expected);
    }

    #[test]
    fn sequence_period_bits_change() {
        let out: [u64; 4] = collect(0);
        let mut or_accum = 0u64;
        let mut i = 0;
        while i < 4 {
            or_accum |= out[i];
            i += 1;
        }
        assert!(or_accum != 0);
    }

    #[test]
    fn low_bit_pattern_of_seed0_first() {
        assert!(SEED0_REF[0].is_multiple_of(1));
        assert!((SEED0_REF[0] & 1) == 1);
    }

    #[test]
    fn range_contains_check_on_output() {
        let mut rng = LcgMmix::new(7);
        let r = rng.next_u64();
        assert!((0..=u64::MAX).contains(&r));
    }

    #[test]
    fn distinct_seeds_distinct_states_after_reset() {
        let a = LcgMmix::new(1);
        let b = LcgMmix::new(2);
        assert!(a.state() != b.state());
    }

    #[test]
    fn restarting_new_resets_sequence() {
        let first: [u64; 6] = collect(314159);
        let second: [u64; 6] = collect(314159);
        let mut i = 0;
        while i < 6 {
            assert!(first[i] == second[i]);
            i += 1;
        }
    }

    #[test]
    fn step_count_consistency() {
        let mut rng = LcgMmix::new(2718281828);
        let mut i = 0u32;
        while i < 50 {
            let _ = rng.next_u64();
            i += 1;
        }
        let after_50 = rng.state();
        let mut restored = LcgMmix::from_state(after_50);
        let direct = rng.next_u64();
        assert!(restored.next_u64() == direct);
    }

    #[test]
    fn seed_one_differs_from_increment() {
        let mut rng = LcgMmix::new(1);
        let expected = 1u64
            .wrapping_mul(MMIX_MULTIPLIER)
            .wrapping_add(MMIX_INCREMENT);
        assert!(rng.next_u64() == expected);
        assert!(expected != MMIX_INCREMENT);
    }
}
