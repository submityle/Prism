//! `xoroshiro64**` pseudo-random number generator (`PRNG`).
//!
//! This module implements the `xoroshiro64**` `algorithm` by Blackman and
//! Vigna. It is a 32-bit generator built on a 64-bit state held as two `u32`
//! words (`s0`, `s1`). Each step emits one 32-bit result and advances the
//! state using rotations and `XOR`/shift mixing only, so the implementation is
//! pure integer code with no floating-point or transcendental operations.
//!
//! The all-zero state (`s0 == 0 && s1 == 0`) is degenerate: the generator is
//! stuck at zero forever. Callers should seed with at least one nonzero word.
//! This `RNG` is deterministic and reproducible given a known seed, which is
//! useful for `CPU`/`GPU` particle simulations that must replay identically.

/// A `xoroshiro64**` pseudo-random number generator.
///
/// The generator state is two `u32` words. Use [`Xoroshiro64StarStar::new`]
/// to seed it, [`Xoroshiro64StarStar::next_u32`] to draw values, and
/// [`Xoroshiro64StarStar::state`] / [`Xoroshiro64StarStar::from_state`] to
/// capture and restore an exact position in the output stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoroshiro64StarStar {
    s0: u32,
    s1: u32,
}

impl Xoroshiro64StarStar {
    /// Creates a new generator from the two seed words.
    ///
    /// At least one of `s0` / `s1` should be nonzero; an all-zero seed yields
    /// the degenerate stream of all zeros.
    #[must_use]
    pub fn new(s0: u32, s1: u32) -> Self {
        Self { s0, s1 }
    }

    /// Restores a generator from a previously captured state tuple.
    #[must_use]
    pub fn from_state(s0: u32, s1: u32) -> Self {
        Self { s0, s1 }
    }

    /// Returns the current internal state as `(s0, s1)`.
    #[must_use]
    pub fn state(&self) -> (u32, u32) {
        (self.s0, self.s1)
    }

    /// Advances the generator and returns the next 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        let s0 = self.s0;
        let s1 = self.s1;
        let result = s0.wrapping_mul(0x9E37_79BB).rotate_left(5).wrapping_mul(5);
        let t = s1 ^ s0;
        self.s0 = s0.rotate_left(26) ^ t ^ (t << 9);
        self.s1 = t.rotate_left(13);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Xoroshiro64StarStar;

    fn collect<const N: usize>(rng: &mut Xoroshiro64StarStar) -> [u32; N] {
        let mut out = [0u32; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u32();
            i += 1;
        }
        out
    }

    fn count_distinct<const N: usize>(values: &[u32; N]) -> usize {
        let mut distinct = 0;
        let mut i = 0;
        while i < N {
            let mut seen = false;
            let mut j = 0;
            while j < i {
                if values[j] == values[i] {
                    seen = true;
                }
                j += 1;
            }
            if !seen {
                distinct += 1;
            }
            i += 1;
        }
        distinct
    }

    #[test]
    fn ref_vector_zero() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        assert_eq!(rng.next_u32(), 0xE2AC_153F);
    }

    #[test]
    fn ref_vector_one() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        rng.next_u32();
        assert_eq!(rng.next_u32(), 0x3081_7EAA);
    }

    #[test]
    fn ref_vector_two() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        rng.next_u32();
        rng.next_u32();
        assert_eq!(rng.next_u32(), 0x607A_3436);
    }

    #[test]
    fn ref_vector_three() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        rng.next_u32();
        rng.next_u32();
        rng.next_u32();
        assert_eq!(rng.next_u32(), 0xB030_543B);
    }

    #[test]
    fn ref_vector_sequence() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let out: [u32; 4] = collect(&mut rng);
        assert_eq!(out[0], 0xE2AC_153F);
        assert_eq!(out[1], 0x3081_7EAA);
        assert_eq!(out[2], 0x607A_3436);
        assert_eq!(out[3], 0xB030_543B);
    }

    #[test]
    fn determinism_same_seed() {
        let mut a = Xoroshiro64StarStar::new(1, 2);
        let mut b = Xoroshiro64StarStar::new(1, 2);
        let va: [u32; 16] = collect(&mut a);
        let vb: [u32; 16] = collect(&mut b);
        let mut i = 0;
        while i < 16 {
            assert_eq!(va[i], vb[i]);
            i += 1;
        }
    }

    #[test]
    fn divergence_different_s0() {
        let mut a = Xoroshiro64StarStar::new(1, 2);
        let mut b = Xoroshiro64StarStar::new(3, 2);
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn divergence_different_s1() {
        let mut a = Xoroshiro64StarStar::new(1, 2);
        let mut b = Xoroshiro64StarStar::new(1, 7);
        let va: [u32; 4] = collect(&mut a);
        let vb: [u32; 4] = collect(&mut b);
        let mut differs = false;
        let mut i = 0;
        while i < 4 {
            if va[i] != vb[i] {
                differs = true;
            }
            i += 1;
        }
        assert!(differs);
    }

    #[test]
    fn state_initial() {
        let rng = Xoroshiro64StarStar::new(11, 22);
        assert_eq!(rng.state(), (11, 22));
    }

    #[test]
    fn state_changes_after_step() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let before = rng.state();
        rng.next_u32();
        let after = rng.state();
        assert!(before != after);
    }

    #[test]
    fn from_state_roundtrip() {
        let rng = Xoroshiro64StarStar::from_state(123, 456);
        assert_eq!(rng.state(), (123, 456));
    }

    #[test]
    fn from_state_matches_new() {
        let a = Xoroshiro64StarStar::new(99, 100);
        let b = Xoroshiro64StarStar::from_state(99, 100);
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn from_state_reproduces_sequence() {
        let mut source = Xoroshiro64StarStar::new(42, 24);
        let _ = source.next_u32();
        let _ = source.next_u32();
        let (s0, s1) = source.state();
        let mut resumed = Xoroshiro64StarStar::from_state(s0, s1);
        let a: [u32; 8] = collect(&mut source);
        let b: [u32; 8] = collect(&mut resumed);
        let mut i = 0;
        while i < 8 {
            assert_eq!(a[i], b[i]);
            i += 1;
        }
    }

    #[test]
    fn seed_zero_s0_nonzero_s1() {
        let mut rng = Xoroshiro64StarStar::new(0, 1);
        let out: [u32; 4] = collect(&mut rng);
        assert!(count_distinct(&out) >= 3);
    }

    #[test]
    fn seed_nonzero_s0_zero_s1() {
        let mut rng = Xoroshiro64StarStar::new(1, 0);
        let out: [u32; 4] = collect(&mut rng);
        assert!(count_distinct(&out) >= 3);
    }

    #[test]
    fn degenerate_all_zero() {
        let mut rng = Xoroshiro64StarStar::new(0, 0);
        let out: [u32; 5] = collect(&mut rng);
        let mut i = 0;
        while i < 5 {
            assert_eq!(out[i], 0);
            i += 1;
        }
        assert_eq!(rng.state(), (0, 0));
    }

    #[test]
    fn outputs_vary() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let out: [u32; 32] = collect(&mut rng);
        assert!(count_distinct(&out) >= 30);
    }

    #[test]
    fn two_instances_parallel() {
        let mut a = Xoroshiro64StarStar::new(7, 13);
        let mut b = Xoroshiro64StarStar::new(7, 13);
        let mut i = 0;
        while i < 20 {
            assert_eq!(a.next_u32(), b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn resume_from_captured_state() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let first: [u32; 4] = collect(&mut rng);
        let (s0, s1) = rng.state();
        let continued: [u32; 4] = collect(&mut rng);
        let mut replay = Xoroshiro64StarStar::from_state(s0, s1);
        let replayed: [u32; 4] = collect(&mut replay);
        assert_eq!(first[0], 0xE2AC_153F);
        let mut i = 0;
        while i < 4 {
            assert_eq!(continued[i], replayed[i]);
            i += 1;
        }
    }

    #[test]
    fn state_tuple_order() {
        let rng = Xoroshiro64StarStar::new(5, 9);
        let (s0, s1) = rng.state();
        assert_eq!(s0, 5);
        assert_eq!(s1, 9);
    }

    #[test]
    fn copy_semantics_independent() {
        let mut original = Xoroshiro64StarStar::new(1, 2);
        let mut copy = original;
        let a = original.next_u32();
        let b = copy.next_u32();
        assert_eq!(a, b);
        let _ = original.next_u32();
        let _ = original.next_u32();
        assert!(original.state() != copy.state());
    }

    #[test]
    fn clone_preserves_stream() {
        let mut original = Xoroshiro64StarStar::new(321, 654);
        let _ = original.next_u32();
        let cloned = original;
        let mut a = original;
        let mut b = cloned;
        let va: [u32; 5] = collect(&mut a);
        let vb: [u32; 5] = collect(&mut b);
        let mut i = 0;
        while i < 5 {
            assert_eq!(va[i], vb[i]);
            i += 1;
        }
    }

    #[test]
    fn many_steps_no_panic() {
        let mut rng = Xoroshiro64StarStar::new(0xDEAD_BEEF, 0x1234_5678);
        let mut i = 0;
        while i < 10_000 {
            let _ = rng.next_u32();
            i += 1;
        }
        assert!(rng.state() != (0xDEAD_BEEF, 0x1234_5678));
    }

    #[test]
    fn step_count_matches_from_state_continuation() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let mut i = 0;
        while i < 100 {
            let _ = rng.next_u32();
            i += 1;
        }
        let (s0, s1) = rng.state();
        let mut a = rng;
        let mut b = Xoroshiro64StarStar::from_state(s0, s1);
        let va: [u32; 10] = collect(&mut a);
        let vb: [u32; 10] = collect(&mut b);
        let mut k = 0;
        while k < 10 {
            assert_eq!(va[k], vb[k]);
            k += 1;
        }
    }

    #[test]
    fn different_seeds_diverge_quickly() {
        let mut a = Xoroshiro64StarStar::new(1, 1);
        let mut b = Xoroshiro64StarStar::new(2, 2);
        let va: [u32; 8] = collect(&mut a);
        let vb: [u32; 8] = collect(&mut b);
        let mut equal = 0;
        let mut i = 0;
        while i < 8 {
            if va[i] == vb[i] {
                equal += 1;
            }
            i += 1;
        }
        assert!(equal < 4);
    }

    #[test]
    fn high_bit_seeds() {
        let mut rng = Xoroshiro64StarStar::new(0x8000_0000, 0x8000_0001);
        let out: [u32; 6] = collect(&mut rng);
        assert!(count_distinct(&out) >= 5);
    }

    #[test]
    fn max_seed() {
        let mut rng = Xoroshiro64StarStar::new(u32::MAX, u32::MAX);
        let out: [u32; 6] = collect(&mut rng);
        assert!(count_distinct(&out) >= 5);
    }

    #[test]
    fn min_nonzero_seed() {
        let mut rng = Xoroshiro64StarStar::new(0, 1);
        let before = rng.state();
        let _ = rng.next_u32();
        assert!(rng.state() != before);
    }

    #[test]
    fn state_not_immediately_periodic() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let initial = rng.state();
        let _ = rng.next_u32();
        assert!(rng.state() != initial);
    }

    #[test]
    fn interleaved_independent_streams() {
        let mut a = Xoroshiro64StarStar::new(10, 20);
        let mut b = Xoroshiro64StarStar::new(10, 20);
        let a0 = a.next_u32();
        let b0 = b.next_u32();
        let a1 = a.next_u32();
        let b1 = b.next_u32();
        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
    }

    #[test]
    fn second_seed_pair_vector() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let out: [u32; 4] = collect(&mut rng);
        let mut mirror = Xoroshiro64StarStar::from_state(1, 2);
        let mout: [u32; 4] = collect(&mut mirror);
        let mut i = 0;
        while i < 4 {
            assert_eq!(out[i], mout[i]);
            i += 1;
        }
    }

    #[test]
    fn capture_restore_midstream() {
        let mut rng = Xoroshiro64StarStar::new(55, 66);
        let mut i = 0;
        while i < 37 {
            let _ = rng.next_u32();
            i += 1;
        }
        let snapshot = rng.state();
        let next_a = rng.next_u32();
        let mut restored = Xoroshiro64StarStar::from_state(snapshot.0, snapshot.1);
        let next_b = restored.next_u32();
        assert_eq!(next_a, next_b);
    }

    #[test]
    fn long_run_distinctness() {
        let mut rng = Xoroshiro64StarStar::new(0xABCD, 0xEF01);
        let out: [u32; 64] = collect(&mut rng);
        assert!(count_distinct(&out) >= 60);
    }

    #[test]
    fn swapped_seed_differs() {
        let mut a = Xoroshiro64StarStar::new(3, 5);
        let mut b = Xoroshiro64StarStar::new(5, 3);
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn repeated_new_is_stable() {
        let mut a = Xoroshiro64StarStar::new(1, 2);
        let first_a: [u32; 4] = collect(&mut a);
        let mut b = Xoroshiro64StarStar::new(1, 2);
        let first_b: [u32; 4] = collect(&mut b);
        let mut i = 0;
        while i < 4 {
            assert_eq!(first_a[i], first_b[i]);
            i += 1;
        }
    }

    #[test]
    fn first_output_nonzero_for_small_seed() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        assert!(rng.next_u32() != 0);
    }

    #[test]
    fn state_after_four_steps_matches_vectors_resume() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let _: [u32; 4] = collect(&mut rng);
        let (s0, s1) = rng.state();
        let mut resume = Xoroshiro64StarStar::from_state(s0, s1);
        let a = rng.next_u32();
        let b = resume.next_u32();
        assert_eq!(a, b);
    }

    #[test]
    fn adjacent_outputs_usually_differ() {
        let mut rng = Xoroshiro64StarStar::new(1, 2);
        let out: [u32; 16] = collect(&mut rng);
        let mut adjacent_equal = 0;
        let mut i = 1;
        while i < 16 {
            if out[i] == out[i - 1] {
                adjacent_equal += 1;
            }
            i += 1;
        }
        assert!(adjacent_equal <= 1);
    }

    #[test]
    fn from_state_zero_is_degenerate() {
        let mut rng = Xoroshiro64StarStar::from_state(0, 0);
        assert_eq!(rng.next_u32(), 0);
        assert_eq!(rng.state(), (0, 0));
    }

    #[test]
    fn large_offset_reproducible() {
        let mut rng = Xoroshiro64StarStar::new(777, 888);
        let mut i = 0;
        while i < 500 {
            let _ = rng.next_u32();
            i += 1;
        }
        let (s0, s1) = rng.state();
        let mut again = Xoroshiro64StarStar::new(777, 888);
        let mut j = 0;
        while j < 500 {
            let _ = again.next_u32();
            j += 1;
        }
        assert_eq!(again.state(), (s0, s1));
    }

    #[test]
    fn distinct_seeds_distinct_output() {
        // The first output depends only on `s0`, so differing `s0` must change
        // output[0], while differing only in `s1` diverges by the second draw.
        let mut a = Xoroshiro64StarStar::new(100, 200);
        let mut b = Xoroshiro64StarStar::new(101, 200);
        let mut c = Xoroshiro64StarStar::new(100, 201);
        let va = a.next_u32();
        let vb = b.next_u32();
        let vc0 = c.next_u32();
        let va1 = a.next_u32();
        let vc1 = c.next_u32();
        assert!(va != vb);
        assert_eq!(va, vc0);
        assert!(va1 != vc1);
    }

    #[test]
    fn eq_derive_reflects_state() {
        let a = Xoroshiro64StarStar::new(1, 2);
        let b = Xoroshiro64StarStar::new(1, 2);
        let c = Xoroshiro64StarStar::new(1, 3);
        assert!(a == b);
        assert!(a != c);
    }
}
