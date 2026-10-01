//! `xoshiro256++` pseudo-random number generator (`PRNG`).
//!
//! This module implements the `xoshiro256++` algorithm by David Blackman and
//! Sebastiano Vigna. It is a fast, 64-bit output generator built on a
//! `[u64; 4]` state word. The implementation is pure integer code: it relies
//! only on wrapping addition, exclusive-or, bit shifts, and bit rotation, so it
//! is suitable for a `no_std` + `alloc` crate and avoids every floating-point
//! or transcendental operation.
//!
//! The generator advances its internal state on each call to [`next_u64`] and
//! returns a scrambled 64-bit value. The state can be inspected via
//! [`state`] and rebuilt via [`from_state`], which makes the sequence fully
//! reproducible across runs.
//!
//! [`next_u64`]: Xoshiro256PlusPlus::next_u64
//! [`state`]: Xoshiro256PlusPlus::state
//! [`from_state`]: Xoshiro256PlusPlus::from_state

/// A `xoshiro256++` pseudo-random number generator (`PRNG`).
///
/// The generator holds a `[u64; 4]` state word and produces 64-bit outputs.
/// Construct one with [`Xoshiro256PlusPlus::new`] or
/// [`Xoshiro256PlusPlus::from_state`], then call
/// [`Xoshiro256PlusPlus::next_u64`] to advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Creates a new generator from the given 256-bit seed state.
    #[must_use]
    pub const fn new(s: [u64; 4]) -> Self {
        Self { s }
    }

    /// Rebuilds a generator from a previously captured state word.
    ///
    /// This is the inverse of [`Xoshiro256PlusPlus::state`]: feeding the output
    /// of `state` back into `from_state` yields a generator that reproduces the
    /// exact same subsequent sequence.
    #[must_use]
    pub const fn from_state(s: [u64; 4]) -> Self {
        Self { s }
    }

    /// Returns a copy of the current 256-bit state word.
    #[must_use]
    pub const fn state(&self) -> [u64; 4] {
        self.s
    }

    /// Advances the state and returns the next 64-bit output value.
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Xoshiro256PlusPlus;

    /// Reference seed used across many tests.
    const SEED: [u64; 4] = [1, 2, 3, 4];

    /// External anchor vectors for seed `[1, 2, 3, 4]`.
    const VECTORS: [u64; 4] = [
        0x0000_0000_0280_0001,
        0x0000_0000_0380_0067,
        0x000c_c000_0380_0067,
        0x000c_c201_9944_00b2,
    ];

    /// Builds a generator seeded with [`SEED`].
    fn seeded() -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::new(SEED)
    }

    /// Collects `N` consecutive outputs into a fixed array.
    fn collect<const N: usize>(rng: &mut Xoshiro256PlusPlus) -> [u64; N] {
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    #[test]
    fn first_output_matches_vector() {
        let mut rng = seeded();
        assert_eq!(rng.next_u64(), VECTORS[0]);
    }

    #[test]
    fn second_output_matches_vector() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        assert_eq!(rng.next_u64(), VECTORS[1]);
    }

    #[test]
    fn third_output_matches_vector() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        assert_eq!(rng.next_u64(), VECTORS[2]);
    }

    #[test]
    fn fourth_output_matches_vector() {
        let mut rng = seeded();
        let mut i = 0;
        while i < 3 {
            let _ = rng.next_u64();
            i += 1;
        }
        assert_eq!(rng.next_u64(), VECTORS[3]);
    }

    #[test]
    fn four_vector_sequence_in_order() {
        let mut rng = seeded();
        let got: [u64; 4] = collect(&mut rng);
        assert_eq!(got, VECTORS);
    }

    #[test]
    fn deterministic_same_seed_eight_outputs() {
        let mut a = seeded();
        let mut b = seeded();
        let x: [u64; 8] = collect(&mut a);
        let y: [u64; 8] = collect(&mut b);
        assert_eq!(x, y);
    }

    #[test]
    fn divergent_seeds_first_output_differs() {
        let mut a = Xoshiro256PlusPlus::new([1, 2, 3, 4]);
        let mut b = Xoshiro256PlusPlus::new([4, 3, 2, 1]);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn state_roundtrip_identity() {
        let rng = seeded();
        let captured = rng.state();
        let rebuilt = Xoshiro256PlusPlus::from_state(captured);
        assert_eq!(rebuilt.state(), captured);
    }

    #[test]
    fn from_state_reproduces_subsequent_outputs() {
        let mut rng = seeded();
        let mut i = 0;
        while i < 5 {
            let _ = rng.next_u64();
            i += 1;
        }
        let snapshot = rng.state();
        let expected: [u64; 6] = collect(&mut rng);
        let mut rebuilt = Xoshiro256PlusPlus::from_state(snapshot);
        let actual: [u64; 6] = collect(&mut rebuilt);
        assert_eq!(expected, actual);
    }

    #[test]
    fn new_stores_full_state() {
        let rng = Xoshiro256PlusPlus::new(SEED);
        assert_eq!(rng.state(), SEED);
    }

    #[test]
    fn state_mutates_after_next() {
        let mut rng = seeded();
        let before = rng.state();
        let _ = rng.next_u64();
        let after = rng.state();
        assert_ne!(before, after);
    }

    #[test]
    fn zero_state_stays_zero() {
        let mut rng = Xoshiro256PlusPlus::new([0, 0, 0, 0]);
        let _ = rng.next_u64();
        assert_eq!(rng.state(), [0u64; 4]);
    }

    #[test]
    fn zero_state_outputs_zero() {
        let mut rng = Xoshiro256PlusPlus::new([0, 0, 0, 0]);
        assert_eq!(rng.next_u64(), 0);
    }

    #[test]
    fn independent_instances_do_not_interfere() {
        let mut a = seeded();
        let mut b = seeded();
        let _ = a.next_u64();
        let _ = a.next_u64();
        assert_eq!(b.next_u64(), VECTORS[0]);
    }

    #[test]
    fn clone_via_state_matches() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let mut forked = Xoshiro256PlusPlus::from_state(rng.state());
        assert_eq!(rng.next_u64(), forked.next_u64());
    }

    #[test]
    fn two_different_seeds_full_divergence() {
        let mut a = Xoshiro256PlusPlus::new([7, 8, 9, 10]);
        let mut b = Xoshiro256PlusPlus::new([10, 9, 8, 7]);
        let x: [u64; 8] = collect(&mut a);
        let y: [u64; 8] = collect(&mut b);
        assert_ne!(x, y);
    }

    #[test]
    fn advance_many_without_panic() {
        let mut rng = seeded();
        let mut acc = 0u64;
        let mut i = 0;
        while i < 1000 {
            acc = acc.wrapping_add(rng.next_u64());
            i += 1;
        }
        assert_ne!(acc, 0);
    }

    #[test]
    fn consecutive_outputs_differ_for_seed() {
        let mut rng = seeded();
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn reproduce_after_ten_steps() {
        let mut rng = seeded();
        let mut i = 0;
        while i < 10 {
            let _ = rng.next_u64();
            i += 1;
        }
        let snapshot = rng.state();
        let next = rng.next_u64();
        let mut rebuilt = Xoshiro256PlusPlus::from_state(snapshot);
        assert_eq!(rebuilt.next_u64(), next);
    }

    #[test]
    fn state_has_four_words() {
        let rng = seeded();
        assert_eq!(rng.state().len(), 4);
    }

    #[test]
    fn from_state_then_next_matches_direct() {
        let direct = {
            let mut rng = seeded();
            rng.next_u64()
        };
        let mut rebuilt = Xoshiro256PlusPlus::from_state(SEED);
        assert_eq!(rebuilt.next_u64(), direct);
    }

    #[test]
    fn seed_a_vs_b_specific_divergence() {
        let mut a = Xoshiro256PlusPlus::new([100, 200, 300, 400]);
        let mut b = Xoshiro256PlusPlus::new([400, 300, 200, 100]);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn shifted_seed_divergence() {
        let mut a = Xoshiro256PlusPlus::new([1, 2, 3, 4]);
        let mut b = Xoshiro256PlusPlus::new([2, 3, 4, 5]);
        let x: [u64; 4] = collect(&mut a);
        let y: [u64; 4] = collect(&mut b);
        assert_ne!(x, y);
    }

    #[test]
    fn max_seed_values_no_panic() {
        let mut rng = Xoshiro256PlusPlus::new([u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
        let out: [u64; 16] = collect(&mut rng);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn determinism_long_run() {
        let mut a = seeded();
        let mut b = seeded();
        let x: [u64; 100] = collect(&mut a);
        let y: [u64; 100] = collect(&mut b);
        assert_eq!(x, y);
    }

    #[test]
    fn snapshot_midstream_resumes() {
        let mut rng = seeded();
        let mut i = 0;
        while i < 42 {
            let _ = rng.next_u64();
            i += 1;
        }
        let snapshot = rng.state();
        let tail: [u64; 10] = collect(&mut rng);
        let mut resumed = Xoshiro256PlusPlus::from_state(snapshot);
        let resumed_tail: [u64; 10] = collect(&mut resumed);
        assert_eq!(tail, resumed_tail);
    }

    #[test]
    fn different_states_after_same_step_count() {
        let mut a = Xoshiro256PlusPlus::new([1, 2, 3, 4]);
        let mut b = Xoshiro256PlusPlus::new([5, 6, 7, 8]);
        let mut i = 0;
        while i < 20 {
            let _ = a.next_u64();
            let _ = b.next_u64();
            i += 1;
        }
        assert_ne!(a.state(), b.state());
    }

    #[test]
    fn state_after_one_step_is_known() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let expected: [u64; 4] = [7, 0, 0x0004_0002, 0x0000_c000_0000_0000];
        assert_eq!(rng.state(), expected);
    }

    #[test]
    fn first_outputs_are_pairwise_distinct() {
        let mut rng = seeded();
        let out: [u64; 10] = collect(&mut rng);
        let mut i = 0;
        while i < 10 {
            let mut j = i + 1;
            while j < 10 {
                assert_ne!(out[i], out[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn from_state_zero_outputs_zero() {
        let mut rng = Xoshiro256PlusPlus::from_state([0, 0, 0, 0]);
        assert_eq!(rng.next_u64(), 0);
    }

    #[test]
    fn single_nonzero_word_seed_runs() {
        let mut rng = Xoshiro256PlusPlus::new([1, 0, 0, 0]);
        let out: [u64; 8] = collect(&mut rng);
        assert_ne!(out[0], out[3]);
    }

    #[test]
    fn high_bit_seed_runs() {
        let mut rng = Xoshiro256PlusPlus::new([1u64 << 63, 1u64 << 62, 1u64 << 61, 1u64 << 60]);
        let out: [u64; 8] = collect(&mut rng);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn reproduce_from_midstream_state() {
        let mut rng = seeded();
        let mut i = 0;
        while i < 7 {
            let _ = rng.next_u64();
            i += 1;
        }
        let mid = rng.state();
        let a = rng.next_u64();
        let mut copy = Xoshiro256PlusPlus::from_state(mid);
        let b = copy.next_u64();
        assert_eq!(a, b);
    }

    #[test]
    fn fork_then_advance_independently() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let mut fork = Xoshiro256PlusPlus::from_state(rng.state());
        let _ = fork.next_u64();
        let _ = fork.next_u64();
        assert_ne!(rng.state(), fork.state());
    }

    #[test]
    fn equal_states_produce_equal_outputs() {
        let a = Xoshiro256PlusPlus::new([11, 22, 33, 44]);
        let b = Xoshiro256PlusPlus::new([11, 22, 33, 44]);
        let mut ca = a;
        let mut cb = b;
        assert_eq!(ca.next_u64(), cb.next_u64());
    }

    #[test]
    fn unequal_states_detected() {
        let a = Xoshiro256PlusPlus::new([11, 22, 33, 44]);
        let b = Xoshiro256PlusPlus::new([44, 33, 22, 11]);
        assert_ne!(a.state(), b.state());
    }

    #[test]
    fn vectors_three_and_four_are_distinct() {
        assert_ne!(VECTORS[2], VECTORS[3]);
    }

    #[test]
    fn full_small_cycle_runs() {
        let mut rng = Xoshiro256PlusPlus::new([0xdead, 0xbeef, 0xcafe, 0xf00d]);
        let out: [u64; 32] = collect(&mut rng);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn copy_preserves_sequence() {
        let mut original = seeded();
        let _ = original.next_u64();
        let snapshot = original;
        let mut a = original;
        let mut b = snapshot;
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn long_sequence_contains_variation() {
        let mut rng = seeded();
        let out: [u64; 64] = collect(&mut rng);
        assert_ne!(out[0], out[63]);
    }

    #[test]
    fn two_step_reproducibility() {
        let mut rng = seeded();
        let a: [u64; 2] = collect(&mut rng);
        let mut again = seeded();
        let b: [u64; 2] = collect(&mut again);
        assert_eq!(a, b);
    }

    #[test]
    fn divergence_grows_over_steps() {
        let mut a = Xoshiro256PlusPlus::new([1, 1, 1, 1]);
        let mut b = Xoshiro256PlusPlus::new([1, 1, 1, 2]);
        let mut matched = 0u32;
        let mut i = 0;
        while i < 16 {
            if a.next_u64() == b.next_u64() {
                matched += 1;
            }
            i += 1;
        }
        assert_ne!(matched, 16);
    }

    #[test]
    fn state_roundtrip_after_many_steps() {
        let mut rng = seeded();
        let mut i = 0;
        while i < 123 {
            let _ = rng.next_u64();
            i += 1;
        }
        let captured = rng.state();
        let rebuilt = Xoshiro256PlusPlus::from_state(captured);
        assert_eq!(rebuilt.state(), captured);
    }
}
