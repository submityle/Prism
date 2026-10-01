//! `wyrand`: the final-form (`wyhash` v4.2) pseudo-random number generator by
//! Wang Yi, a minimal single-register engine built from one wrapping add, one
//! exclusive-or, and one widening `64`x`64`->`128` multiply whose two halves
//! are folded together to produce each output word.
//!
//! A `PRNG` in this crate is a *stateful, advancing stream*, not a one-shot
//! hash. The [`WyRand`] engine holds a single `u64` register. On every draw the
//! register is advanced by the fixed odd increment `0x2d358dccaa6c78a5`, the
//! advanced register is xored with the constant `0x8bb84b93962eacc9`, the two
//! `u64` operands are promoted to `u128` and multiplied, and the high and low
//! `64`-bit halves of the `128`-bit product are folded with exclusive-or. The
//! stream is therefore a pure deterministic function of the seed: the same seed
//! always replays the same sequence, exactly what a reproducible particle
//! simulation needs when a frame must match bit for bit between a `CPU`
//! reference path and a future `GPU` implementation.
//!
//! Every step is pure integer arithmetic: a wrapping add, exclusive-ors, and a
//! wrapping widening multiply performed through `u128` intermediates. No
//! floating-point value, transcendental function, rounding, or floating-point
//! comparison appears anywhere in this module, so results are bit-identical
//! across every target that honours two's-complement wrapping semantics.
//!
//! The register advance is a constant stride over the additive group modulo
//! `2^64`; because the increment is odd it is a unit, so the raw register walks
//! through a maximal additive cycle of period `2^64` before the fold scrambles
//! each value. [`WyRand::state`] exposes the current register and
//! [`WyRand::from_state`] rebuilds an engine from a captured register, so a
//! stream can be snapshotted and resumed to reproduce every subsequent draw.
//!
//! Scope: `wyrand` is a fast, non-cryptographic generator. Its register is
//! trivially recoverable from outputs, so it must never be used for security,
//! key material, or anywhere an adversary could exploit predictability. It
//! exists purely for reproducible, high-throughput simulation randomness.

/// The odd additive stride added to the register before each draw.
///
/// Being odd it is a unit modulo `2^64`, so repeated addition walks the raw
/// register through a maximal additive cycle of period `2^64`.
pub const WYRAND_INCREMENT: u64 = 0x2d35_8dcc_aa6c_78a5;

/// The exclusive-or constant mixed into the advanced register to form the
/// second operand of the widening multiply.
pub const WYRAND_XOR_CONSTANT: u64 = 0x8bb8_4b93_962e_acc9;

/// The `wyrand` (`wyhash` v4.2) pseudo-random number generator.
///
/// Holds a single `u64` register advanced by [`WYRAND_INCREMENT`] on each draw.
/// Construct one with [`WyRand::new`], draw words with [`WyRand::next_u64`],
/// read the register with [`WyRand::state`], and rebuild from a captured
/// register with [`WyRand::from_state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WyRand {
    /// The single `u64` register that fully determines the stream.
    seed: u64,
}

impl WyRand {
    /// Creates a `wyrand` engine seeded with `seed`.
    ///
    /// The seed is stored verbatim as the initial register; the first
    /// [`WyRand::next_u64`] advances it by [`WYRAND_INCREMENT`] before folding.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Rebuilds an engine from a register previously read via [`WyRand::state`].
    ///
    /// `WyRand::from_state(e.state())` reproduces an engine that replays exactly
    /// the same subsequent sequence as `e`.
    #[must_use]
    pub const fn from_state(seed: u64) -> Self {
        Self { seed }
    }

    /// Returns the current `u64` register.
    ///
    /// Combined with [`WyRand::from_state`] this snapshots and resumes a stream.
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.seed
    }

    /// Advances the register and returns the next `u64` draw.
    ///
    /// Adds [`WYRAND_INCREMENT`], xors with [`WYRAND_XOR_CONSTANT`], forms the
    /// widening `64`x`64`->`128` product through `u128` intermediates, and folds
    /// the high and low halves with exclusive-or.
    pub const fn next_u64(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(WYRAND_INCREMENT);
        let t = self.seed ^ WYRAND_XOR_CONSTANT;
        let product = (self.seed as u128).wrapping_mul(t as u128);
        let lo = product as u64;
        let hi = (product >> 64) as u64;
        lo ^ hi
    }
}

#[cfg(test)]
mod tests {
    use super::{WyRand, WYRAND_INCREMENT, WYRAND_XOR_CONSTANT};

    /// The seed used by the externally anchored reference vectors.
    const ANCHOR_SEED: u64 = 0x0123_4567_89ab_cdef;

    /// Draws `N` consecutive `u64` words from an engine seeded with `seed`.
    fn draw<const N: usize>(seed: u64) -> [u64; N] {
        let mut rng = WyRand::new(seed);
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    #[test]
    fn reference_vector_0() {
        let got = draw::<4>(ANCHOR_SEED);
        assert!(got[0] == 0x368d_5c95_2174_cc4d);
    }

    #[test]
    fn reference_vector_1() {
        let got = draw::<4>(ANCHOR_SEED);
        assert!(got[1] == 0x0901_4ced_49dd_0226);
    }

    #[test]
    fn reference_vector_2() {
        let got = draw::<4>(ANCHOR_SEED);
        assert!(got[2] == 0x385a_54d9_be57_5d3f);
    }

    #[test]
    fn reference_vector_3() {
        let got = draw::<4>(ANCHOR_SEED);
        assert!(got[3] == 0x96d9_7603_a281_87a2);
    }

    #[test]
    fn all_four_reference_vectors_at_once() {
        let got = draw::<4>(ANCHOR_SEED);
        let want: [u64; 4] = [
            0x368d_5c95_2174_cc4d,
            0x0901_4ced_49dd_0226,
            0x385a_54d9_be57_5d3f,
            0x96d9_7603_a281_87a2,
        ];
        let mut i = 0;
        while i < 4 {
            assert!(got[i] == want[i]);
            i += 1;
        }
    }

    #[test]
    fn reference_vectors_are_pairwise_distinct() {
        let got = draw::<4>(ANCHOR_SEED);
        let mut i = 0;
        while i < 4 {
            let mut j = i + 1;
            while j < 4 {
                assert!(got[i] != got[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn determinism_same_seed_same_stream() {
        let a = draw::<16>(ANCHOR_SEED);
        let b = draw::<16>(ANCHOR_SEED);
        let mut i = 0;
        while i < 16 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn determinism_zero_seed() {
        let a = draw::<8>(0);
        let b = draw::<8>(0);
        let mut i = 0;
        while i < 8 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn determinism_max_seed() {
        let a = draw::<8>(u64::MAX);
        let b = draw::<8>(u64::MAX);
        let mut i = 0;
        while i < 8 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn divergence_adjacent_seeds() {
        let a = draw::<8>(ANCHOR_SEED);
        let b = draw::<8>(ANCHOR_SEED.wrapping_add(1));
        let mut equal = 0;
        let mut i = 0;
        while i < 8 {
            if a[i] == b[i] {
                equal += 1;
            }
            i += 1;
        }
        assert!(equal < 8);
    }

    #[test]
    fn divergence_zero_vs_one() {
        let a = draw::<8>(0);
        let b = draw::<8>(1);
        assert!(a[0] != b[0]);
    }

    #[test]
    fn divergence_distant_seeds() {
        let a = draw::<4>(0x1111_1111_1111_1111);
        let b = draw::<4>(0xeeee_eeee_eeee_eeee);
        assert!(a[0] != b[0]);
        assert!(a[3] != b[3]);
    }

    #[test]
    fn divergence_single_high_bit_flip() {
        let a = draw::<4>(0);
        let b = draw::<4>(1u64 << 63);
        assert!(a[0] != b[0]);
    }

    #[test]
    fn state_starts_equal_to_seed() {
        let rng = WyRand::new(ANCHOR_SEED);
        assert!(rng.state() == ANCHOR_SEED);
    }

    #[test]
    fn state_after_one_call_is_seed_plus_increment() {
        let mut rng = WyRand::new(ANCHOR_SEED);
        let _ = rng.next_u64();
        assert!(rng.state() == ANCHOR_SEED.wrapping_add(WYRAND_INCREMENT));
    }

    #[test]
    fn state_after_one_call_zero_seed() {
        let mut rng = WyRand::new(0);
        let _ = rng.next_u64();
        assert!(rng.state() == WYRAND_INCREMENT);
    }

    #[test]
    fn state_advances_by_increment_each_call() {
        let mut rng = WyRand::new(ANCHOR_SEED);
        let mut expected = ANCHOR_SEED;
        let mut i = 0;
        while i < 32 {
            let _ = rng.next_u64();
            expected = expected.wrapping_add(WYRAND_INCREMENT);
            assert!(rng.state() == expected);
            i += 1;
        }
    }

    #[test]
    fn state_wraps_around_near_overflow() {
        let seed = u64::MAX.wrapping_sub(1);
        let mut rng = WyRand::new(seed);
        let _ = rng.next_u64();
        assert!(rng.state() == seed.wrapping_add(WYRAND_INCREMENT));
    }

    #[test]
    fn from_state_round_trip_reproduces_next() {
        let mut rng = WyRand::new(ANCHOR_SEED);
        let mut i = 0;
        while i < 5 {
            let _ = rng.next_u64();
            i += 1;
        }
        let snapshot = rng.state();
        let mut resumed = WyRand::from_state(snapshot);
        let mut j = 0;
        while j < 8 {
            assert!(rng.next_u64() == resumed.next_u64());
            j += 1;
        }
    }

    #[test]
    fn from_state_at_start_matches_new() {
        let a = WyRand::new(ANCHOR_SEED);
        let b = WyRand::from_state(ANCHOR_SEED);
        assert!(a.state() == b.state());
    }

    #[test]
    fn from_state_reproduces_full_sequence() {
        let original = WyRand::new(0xdead_beef_cafe_f00d);
        let captured = original.state();
        let rebuilt = WyRand::from_state(captured);
        let mut a = original;
        let mut b = rebuilt;
        let mut i = 0;
        while i < 20 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn snapshot_mid_stream_resumes_exactly() {
        let mut rng = WyRand::new(42);
        let mut i = 0;
        while i < 10 {
            let _ = rng.next_u64();
            i += 1;
        }
        let mut clone = WyRand::from_state(rng.state());
        let direct = rng.next_u64();
        let resumed = clone.next_u64();
        assert!(direct == resumed);
    }

    #[test]
    fn clone_is_independent_but_equal_stream() {
        let mut a = WyRand::new(777);
        let _ = a.next_u64();
        let mut b = a;
        assert!(a.next_u64() == b.next_u64());
        assert!(a.next_u64() == b.next_u64());
    }

    #[test]
    fn copy_semantics_preserve_state() {
        let a = WyRand::new(12345);
        let b = a;
        assert!(a.state() == b.state());
    }

    #[test]
    fn equality_reflects_state() {
        let a = WyRand::new(99);
        let b = WyRand::new(99);
        let c = WyRand::new(100);
        assert!(a == b);
        assert!(a != c);
    }

    #[test]
    fn equality_diverges_after_draws() {
        let mut a = WyRand::new(5);
        let b = WyRand::new(5);
        let _ = a.next_u64();
        assert!(a != b);
    }

    #[test]
    fn increment_constant_is_odd() {
        assert!((WYRAND_INCREMENT & 1) == 1);
    }

    #[test]
    fn increment_constant_value() {
        assert!(WYRAND_INCREMENT == 0x2d35_8dcc_aa6c_78a5);
    }

    #[test]
    fn xor_constant_value() {
        assert!(WYRAND_XOR_CONSTANT == 0x8bb8_4b93_962e_acc9);
    }

    #[test]
    fn fold_matches_manual_computation() {
        let seed = 0x0f0f_0f0f_0f0f_0f0f;
        let mut rng = WyRand::new(seed);
        let got = rng.next_u64();
        let advanced = seed.wrapping_add(WYRAND_INCREMENT);
        let t = advanced ^ WYRAND_XOR_CONSTANT;
        let product = (advanced as u128).wrapping_mul(t as u128);
        let lo = product as u64;
        let hi = (product >> 64) as u64;
        assert!(got == (lo ^ hi));
    }

    #[test]
    fn fold_matches_manual_zero_seed() {
        let mut rng = WyRand::new(0);
        let got = rng.next_u64();
        let advanced = WYRAND_INCREMENT;
        let t = advanced ^ WYRAND_XOR_CONSTANT;
        let product = (advanced as u128).wrapping_mul(t as u128);
        let folded = (product as u64) ^ ((product >> 64) as u64);
        assert!(got == folded);
    }

    #[test]
    fn widening_multiply_uses_full_128_bits() {
        // With these operands the high half is non-zero, proving the multiply
        // genuinely widens rather than truncating to `64` bits.
        let advanced = 0xffff_ffff_ffff_ffffu64;
        let t = advanced ^ WYRAND_XOR_CONSTANT;
        let product = (advanced as u128).wrapping_mul(t as u128);
        let hi = (product >> 64) as u64;
        assert!(hi != 0);
    }

    #[test]
    fn outputs_not_all_identical() {
        let got = draw::<8>(ANCHOR_SEED);
        let mut all_same = true;
        let mut i = 1;
        while i < 8 {
            if got[i] != got[0] {
                all_same = false;
            }
            i += 1;
        }
        assert!(!all_same);
    }

    #[test]
    fn outputs_not_all_zero() {
        let got = draw::<8>(ANCHOR_SEED);
        let mut any_nonzero = false;
        let mut i = 0;
        while i < 8 {
            if got[i] != 0 {
                any_nonzero = true;
            }
            i += 1;
        }
        assert!(any_nonzero);
    }

    #[test]
    fn high_bits_vary_across_outputs() {
        let got = draw::<8>(ANCHOR_SEED);
        let mut seen_set = false;
        let mut seen_clear = false;
        let mut i = 0;
        while i < 8 {
            if ((got[i] >> 63) & 1) == 1 {
                seen_set = true;
            } else {
                seen_clear = true;
            }
            i += 1;
        }
        assert!(seen_set);
        assert!(seen_clear);
    }

    #[test]
    fn low_bits_vary_across_outputs() {
        let got = draw::<16>(ANCHOR_SEED);
        let mut seen_set = false;
        let mut seen_clear = false;
        let mut i = 0;
        while i < 16 {
            if (got[i] & 1) == 1 {
                seen_set = true;
            } else {
                seen_clear = true;
            }
            i += 1;
        }
        assert!(seen_set);
        assert!(seen_clear);
    }

    #[test]
    fn no_duplicates_in_short_run() {
        let got = draw::<32>(ANCHOR_SEED);
        let mut i = 0;
        while i < 32 {
            let mut j = i + 1;
            while j < 32 {
                assert!(got[i] != got[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn many_seeds_first_output_distinct() {
        let mut outs = [0u64; 16];
        let mut s = 0;
        while s < 16 {
            let mut rng = WyRand::new(s as u64);
            outs[s] = rng.next_u64();
            s += 1;
        }
        let mut i = 0;
        while i < 16 {
            let mut j = i + 1;
            while j < 16 {
                assert!(outs[i] != outs[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn range_contains_bounded_reduction() {
        // Lemire-style bounded reduction via the widening multiply stays inside
        // the requested half-open range.
        let bound: u64 = 100;
        let mut rng = WyRand::new(0xabcd_ef01_2345_6789);
        let mut i = 0;
        while i < 64 {
            let x = rng.next_u64();
            let reduced = (((x as u128) * (bound as u128)) >> 64) as u64;
            assert!((0..bound).contains(&reduced));
            i += 1;
        }
    }

    #[test]
    fn output_count_is_multiple_of_block() {
        let count: u64 = 32;
        assert!(count.is_multiple_of(8));
    }

    #[test]
    fn block_div_ceil_is_exact_for_aligned_count() {
        let count: u64 = 24;
        assert!(count.div_ceil(8) == 3);
    }

    #[test]
    fn rotate_of_output_is_invertible() {
        let mut rng = WyRand::new(0x5555_aaaa_5555_aaaa);
        let x = rng.next_u64();
        assert!(x.rotate_left(17).rotate_right(17) == x);
    }

    #[test]
    fn parenthesized_bitops_match_fold() {
        let mut rng = WyRand::new(0x0102_0304_0506_0708);
        let x = rng.next_u64();
        let advanced = 0x0102_0304_0506_0708u64.wrapping_add(WYRAND_INCREMENT);
        let t = advanced ^ WYRAND_XOR_CONSTANT;
        let product = (advanced as u128).wrapping_mul(t as u128);
        let combined = (product as u64) ^ ((product >> 64) as u64);
        assert!(x == combined);
    }

    #[test]
    fn two_streams_cross_check_four_draws() {
        let mut a = WyRand::new(0xcafe_babe_dead_c0de);
        let mut b = WyRand::from_state(0xcafe_babe_dead_c0de);
        let four_a = [a.next_u64(), a.next_u64(), a.next_u64(), a.next_u64()];
        let four_b = [b.next_u64(), b.next_u64(), b.next_u64(), b.next_u64()];
        let mut i = 0;
        while i < 4 {
            assert!(four_a[i] == four_b[i]);
            i += 1;
        }
    }

    #[test]
    fn long_run_remains_deterministic() {
        let a = draw::<64>(0x1234_5678_9abc_def0);
        let b = draw::<64>(0x1234_5678_9abc_def0);
        let mut i = 0;
        while i < 64 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn state_round_trip_after_long_run() {
        let mut rng = WyRand::new(0xfeed_face_0000_1111);
        let mut i = 0;
        while i < 100 {
            let _ = rng.next_u64();
            i += 1;
        }
        let snap = rng.state();
        let mut resumed = WyRand::from_state(snap);
        assert!(rng.next_u64() == resumed.next_u64());
    }
}
