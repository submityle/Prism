//! The Lehmer / `MINSTD` multiplicative linear congruential generator
//! (`Park-Miller`), a minimal-state deterministic integer stream for
//! reproducible particle spawning, jitter, and stochastic effects.
//!
//! A fixed hash in this crate (`fnv1a_hash`, `crc32`, `wang_hash`) maps an
//! input to a single digest. A `PRNG` here is instead a *stateful, advancing
//! stream*: the generator holds one `u32` register that mutates on every draw,
//! so repeated calls walk a long, reproducible cycle. Given the same seed the
//! generator always produces the same sequence -- exactly what a deterministic
//! simulation needs when a frame must replay identically between machines or
//! between the `CPU` reference path and a `GPU` implementation.
//!
//! The engine is the classic multiplicative congruential generator of Lehmer,
//! popularized by Park and Miller as the "minimal standard" (`MINSTD`). It
//! advances a single register `x` by
//!
//! ```text
//! x_{n+1} = (A * x_n) mod M
//! ```
//!
//! with multiplier `A = 16807` and modulus `M = 2147483647 = 2^31 - 1`, the
//! eighth Mersenne prime. Because `M` is prime and `A` is a primitive root
//! modulo `M`, the state cycles through every integer in `1..=M-1` exactly once
//! before repeating, giving a full period of `M - 1 = 2147483646`. The value
//! `0` is an absorbing fixed point and never appears inside the cycle, so a
//! live register is always in the range `1..=M-1`.
//!
//! The intermediate product `A * x` can reach `16807 * 2147483646 ~= 2^45`,
//! which overflows `u32`. The recurrence therefore widens both operands to
//! `u64`, multiplies, reduces modulo `M`, and narrows the result back to `u32`;
//! no value in that chain exceeds `2^45`, so the `u64` arithmetic never
//! overflows. The modulo (`%`) is an ordinary integer remainder.
//!
//! Seeding reduces the caller seed modulo `M` and, if the reduced value is `0`
//! (which happens for `seed == 0`, `seed == M`, and any multiple of `M`
//! representable in `u32`), substitutes `1`. This guarantees the register
//! starts inside the productive cycle instead of at the degenerate all-zero
//! fixed point, from which the stream could never escape.
//!
//! All arithmetic is integer only -- a widening multiply, one remainder, and a
//! narrowing cast. No floating point, no `unsafe`, and only [`core`] is used,
//! so the module is fully self-contained under `no_std`.
//!
//! Scope: this is a fast, non-cryptographic generator. `MINSTD` output is
//! trivially predictable from a few samples and the low-order bits are weak;
//! it must never be used for security, key material, or anywhere an adversary
//! could exploit predictability. It exists purely for reproducible,
//! high-throughput simulation randomness.

/// The generator modulus `M = 2^31 - 1 = 2147483647`, the eighth Mersenne
/// prime. The live register ranges over `1..=M-1`.
pub const MODULUS: u32 = 2_147_483_647;

/// The generator multiplier `A = 16807 = 7^5`, a primitive root modulo
/// [`MODULUS`] that yields the full-period `MINSTD` stream.
pub const MULTIPLIER: u32 = 16_807;

/// The Lehmer / `MINSTD` multiplicative congruential generator
/// (`Park-Miller`).
///
/// Advances a single `u32` register `x` by `x <- (A * x) mod M` with
/// `A =` [`MULTIPLIER`] and `M =` [`MODULUS`]. The register is kept inside the
/// productive cycle `1..=M-1` by construction, so the stream has full period
/// `M - 1` and never collapses to the all-zero fixed point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MinStd {
    state: u32,
}

impl MinStd {
    /// Constructs a generator from `seed`.
    ///
    /// The seed is reduced modulo [`MODULUS`] so that any `u32` maps into the
    /// valid state range. If the reduced value is `0` -- which occurs for
    /// `seed == 0`, `seed == MODULUS`, and any representable multiple of
    /// `MODULUS` -- it is replaced with `1`, because `0` is the degenerate
    /// fixed point of the recurrence and would emit only zeros. Any other seed
    /// is used after reduction.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let reduced = seed % MODULUS;
        let state = if reduced == 0 { 1 } else { reduced };
        Self { state }
    }

    /// Advances the register one step and returns the new state.
    ///
    /// Computes `x <- (A * x) mod M` using a widening `u64` multiply to avoid
    /// overflow, then narrows the reduced result back to `u32`. The returned
    /// value equals the register after the step and always lies in `1..=M-1`.
    pub fn next_u32(&mut self) -> u32 {
        let product = u64::from(MULTIPLIER) * u64::from(self.state);
        self.state = (product % u64::from(MODULUS)) as u32;
        self.state
    }

    /// Returns the current register value without advancing it.
    ///
    /// The value always lies in the productive cycle `1..=MODULUS-1`.
    #[must_use]
    pub const fn state(&self) -> u32 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    /// The first five outputs for `seed == 1`, the canonical `Park-Miller`
    /// reference vector.
    const SEED1_FIRST5: [u32; 5] = [16807, 282475249, 1622650073, 984943658, 1144108930];

    /// The register value after 10000 steps from `seed == 1`, the classic
    /// `Park-Miller` validation constant.
    const SEED1_AFTER_10000: u32 = 1_043_618_065;

    // --- Hard reference vectors ---

    #[test]
    fn first_output_from_seed_one_is_multiplier() {
        let mut rng = MinStd::new(1);
        assert_eq!(rng.next_u32(), 16807);
    }

    #[test]
    fn first_output_equals_multiplier_constant() {
        let mut rng = MinStd::new(1);
        assert_eq!(rng.next_u32(), MULTIPLIER);
    }

    #[test]
    fn second_output_from_seed_one() {
        let mut rng = MinStd::new(1);
        rng.next_u32();
        assert_eq!(rng.next_u32(), 282475249);
    }

    #[test]
    fn third_output_from_seed_one() {
        let mut rng = MinStd::new(1);
        for _ in 0..2 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 1622650073);
    }

    #[test]
    fn fourth_output_from_seed_one() {
        let mut rng = MinStd::new(1);
        for _ in 0..3 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 984943658);
    }

    #[test]
    fn fifth_output_from_seed_one() {
        let mut rng = MinStd::new(1);
        for _ in 0..4 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 1144108930);
    }

    #[test]
    fn first_five_outputs_match_reference_vector() {
        let mut rng = MinStd::new(1);
        for &expected in &SEED1_FIRST5 {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn park_miller_validation_after_10000_steps() {
        let mut rng = MinStd::new(1);
        for _ in 0..10_000 {
            rng.next_u32();
        }
        assert_eq!(rng.state(), SEED1_AFTER_10000);
    }

    #[test]
    fn park_miller_validation_via_last_return_value() {
        let mut rng = MinStd::new(1);
        let mut last = 0;
        for _ in 0..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, SEED1_AFTER_10000);
    }

    // --- Constants ---

    #[test]
    fn modulus_is_mersenne_prime() {
        assert_eq!(MODULUS, (1u32 << 31) - 1);
        assert_eq!(MODULUS, 2_147_483_647);
    }

    #[test]
    fn multiplier_value_is_correct() {
        assert_eq!(MULTIPLIER, 16_807);
    }

    #[test]
    fn multiplier_is_seven_to_the_fifth() {
        assert_eq!(MULTIPLIER, 7u32.pow(5));
    }

    #[test]
    fn intermediate_product_fits_in_u64() {
        let max_product = u64::from(MULTIPLIER) * u64::from(MODULUS - 1);
        assert!(max_product < u64::from(u32::MAX) * u64::from(u32::MAX));
        assert!(max_product < 1u64 << 46);
    }

    // --- Seed regularization ---

    #[test]
    fn seed_zero_is_regularized_to_one() {
        let rng = MinStd::new(0);
        assert_eq!(rng.state(), 1);
    }

    #[test]
    fn seed_zero_produces_same_stream_as_seed_one() {
        let mut a = MinStd::new(0);
        let mut b = MinStd::new(1);
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn seed_modulus_is_regularized_to_one() {
        let rng = MinStd::new(MODULUS);
        assert_eq!(rng.state(), 1);
    }

    #[test]
    fn seed_above_modulus_is_reduced() {
        let rng = MinStd::new(MODULUS + 5);
        assert_eq!(rng.state(), 5);
    }

    #[test]
    fn seed_one_is_used_directly() {
        let rng = MinStd::new(1);
        assert_eq!(rng.state(), 1);
    }

    #[test]
    fn small_seeds_are_used_directly() {
        for seed in 1u32..=1000 {
            assert_eq!(MinStd::new(seed).state(), seed);
        }
    }

    #[test]
    fn u32_max_seed_is_reduced_into_range() {
        let rng = MinStd::new(u32::MAX);
        let expected = u32::MAX % MODULUS;
        assert_eq!(rng.state(), expected);
        assert_ne!(rng.state(), 0);
    }

    #[test]
    fn reduced_seed_matches_remainder_for_large_seeds() {
        for seed in [MODULUS + 1, MODULUS + 100, 3_000_000_000, 4_000_000_000] {
            let reduced = seed % MODULUS;
            let expected = if reduced == 0 { 1 } else { reduced };
            assert_eq!(MinStd::new(seed).state(), expected);
        }
    }

    // --- Determinism ---

    #[test]
    fn same_seed_same_sequence() {
        let mut a = MinStd::new(12345);
        let mut b = MinStd::new(12345);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn clone_continues_identical_stream() {
        let mut a = MinStd::new(999);
        for _ in 0..37 {
            a.next_u32();
        }
        let mut b = a;
        for _ in 0..256 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn restarting_from_seed_reproduces_prefix() {
        let mut a = MinStd::new(2024);
        let first: Vec<u32> = (0..100).map(|_| a.next_u32()).collect();
        let mut b = MinStd::new(2024);
        let second: Vec<u32> = (0..100).map(|_| b.next_u32()).collect();
        assert_eq!(first, second);
    }

    // --- Distinct seeds differ ---

    #[test]
    fn different_seeds_produce_different_sequences() {
        let mut a = MinStd::new(1);
        let mut b = MinStd::new(2);
        let mut any_diff = false;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff);
    }

    #[test]
    fn first_output_is_distinct_across_small_seeds() {
        let mut seen = BTreeSet::new();
        for seed in 1u32..=5000 {
            let mut rng = MinStd::new(seed);
            assert!(seen.insert(rng.next_u32()));
        }
    }

    #[test]
    fn first_output_equals_seed_times_multiplier() {
        for seed in [1u32, 2, 3, 100, 55555] {
            let mut rng = MinStd::new(seed);
            let expected = (u64::from(MULTIPLIER) * u64::from(seed) % u64::from(MODULUS)) as u32;
            assert_eq!(rng.next_u32(), expected);
        }
    }

    // --- Zero never appears in the cycle ---

    #[test]
    fn zero_never_appears_over_many_steps() {
        let mut rng = MinStd::new(1);
        for _ in 0..1_000_000 {
            assert_ne!(rng.next_u32(), 0);
        }
    }

    #[test]
    fn zero_never_appears_for_various_seeds() {
        for seed in [1u32, 42, 7777, 123456, u32::MAX] {
            let mut rng = MinStd::new(seed);
            for _ in 0..10_000 {
                assert_ne!(rng.next_u32(), 0);
            }
        }
    }

    // --- State stays within the valid range ---

    #[test]
    fn state_stays_within_valid_range() {
        let mut rng = MinStd::new(1);
        for _ in 0..200_000 {
            let x = rng.next_u32();
            assert!(x >= 1);
            assert!(x <= MODULUS - 1);
        }
    }

    #[test]
    fn returned_value_equals_state() {
        let mut rng = MinStd::new(31415);
        for _ in 0..500 {
            let returned = rng.next_u32();
            assert_eq!(returned, rng.state());
        }
    }

    #[test]
    fn state_never_reaches_modulus() {
        let mut rng = MinStd::new(1);
        for _ in 0..100_000 {
            assert_ne!(rng.next_u32(), MODULUS);
        }
    }

    // --- Structural / recurrence properties ---

    #[test]
    fn manual_recurrence_matches_next_u32() {
        let mut rng = MinStd::new(7);
        let mut manual: u64 = 7;
        for _ in 0..1000 {
            manual = (u64::from(MULTIPLIER) * manual) % u64::from(MODULUS);
            assert_eq!(rng.next_u32(), manual as u32);
        }
    }

    #[test]
    fn stepping_once_changes_state_for_seed_one() {
        let mut rng = MinStd::new(1);
        let before = rng.state();
        rng.next_u32();
        assert_ne!(rng.state(), before);
    }

    #[test]
    fn two_steps_differ_from_one_step() {
        let mut one = MinStd::new(5);
        one.next_u32();
        let mut two = MinStd::new(5);
        two.next_u32();
        two.next_u32();
        assert_ne!(one.state(), two.state());
    }

    #[test]
    fn outputs_are_well_distributed_in_halves() {
        let mut rng = MinStd::new(2_000_000_000);
        let mut low = 0u32;
        let mut high = 0u32;
        let half = MODULUS / 2;
        for _ in 0..100_000 {
            if rng.next_u32() < half {
                low += 1;
            } else {
                high += 1;
            }
        }
        // Both halves should be well populated; a wildly skewed split would
        // indicate a broken recurrence.
        assert!(low > 30_000);
        assert!(high > 30_000);
    }

    #[test]
    fn large_run_has_no_duplicates_in_short_window() {
        // Over a short window of a full-period generator every value is unique.
        let mut rng = MinStd::new(424242);
        let mut seen = BTreeSet::new();
        for _ in 0..50_000 {
            assert!(seen.insert(rng.next_u32()));
        }
    }

    #[test]
    fn equality_tracks_internal_state() {
        let a = MinStd::new(77);
        let b = MinStd::new(77);
        let c = MinStd::new(78);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn debug_format_is_non_empty() {
        let rng = MinStd::new(1);
        let text = alloc::format!("{rng:?}");
        assert!(!text.is_empty());
    }

    #[test]
    fn independent_generators_do_not_interfere() {
        let mut a = MinStd::new(100);
        let mut b = MinStd::new(200);
        let a_only: Vec<u32> = {
            let mut x = MinStd::new(100);
            (0..50).map(|_| x.next_u32()).collect()
        };
        for i in 0..50 {
            let av = a.next_u32();
            b.next_u32();
            assert_eq!(av, a_only[i]);
        }
    }

    #[test]
    fn advancing_many_steps_returns_to_valid_range() {
        let mut rng = MinStd::new(314159);
        for _ in 0..1_234_567 {
            rng.next_u32();
        }
        let x = rng.state();
        assert!(x >= 1 && x <= MODULUS - 1);
    }

    #[test]
    fn seed_at_modulus_minus_one_stays_in_cycle() {
        let mut rng = MinStd::new(MODULUS - 1);
        assert_eq!(rng.state(), MODULUS - 1);
        for _ in 0..10_000 {
            assert_ne!(rng.next_u32(), 0);
        }
    }
}
