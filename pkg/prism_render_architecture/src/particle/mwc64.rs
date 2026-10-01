//! `mwc64` Marsaglia multiply-with-carry `PRNG`: 64-bit state, 32-bit output, pure integers.
//!
//! This module provides a `no_std` + `alloc` friendly implementation of the
//! `MWC` (`multiply-with-carry`) `RNG`. The recurrence multiplies the low 32
//! bits of the state by a fixed `type`-level multiplier `A` and folds in the
//! carry stored in the high 32 bits. All arithmetic uses wrapping semantics on
//! `u64`; no floating point or transcendental operations are involved, so the
//! `algorithm` is fully deterministic and reproducible across backends.

/// `mwc64` multiply-with-carry `PRNG`.
///
/// Holds a 64-bit state whose low 32 bits form the carry-folded output word and
/// whose high 32 bits hold the running carry. Each call to [`Mwc64::next_u32`]
/// advances the state and yields one `u32`. Identical seeds always reproduce
/// the same output sequence.
pub struct Mwc64 {
    x: u64,
}

impl Mwc64 {
    /// Multiply-with-carry multiplier constant `A` for the 64-bit `MWC` recurrence.
    pub const A: u64 = 4_294_957_665;

    /// Create a new `mwc64` generator from the given `seed`.
    ///
    /// The `seed` becomes the initial 64-bit state verbatim; its low 32 bits are
    /// the first working value and its high 32 bits are the initial carry.
    pub const fn new(seed: u64) -> Self {
        Self { x: seed }
    }

    /// Reconstruct a generator from a previously captured raw 64-bit state.
    ///
    /// Pairing this with [`Mwc64::state`] lets callers checkpoint and resume a
    /// stream without replaying earlier outputs.
    pub const fn from_state(x: u64) -> Self {
        Self { x }
    }

    /// Return the current raw 64-bit state.
    pub const fn state(&self) -> u64 {
        self.x
    }

    /// Advance the internal state and return the next 32-bit output.
    ///
    /// The recurrence is `x = A * (x & 0xffff_ffff) + (x >> 32)` using wrapping
    /// multiplication and addition; the output is the low 32 bits of the new
    /// state.
    pub fn next_u32(&mut self) -> u32 {
        self.x = Self::A
            .wrapping_mul(self.x & 0xffff_ffff)
            .wrapping_add(self.x >> 32);
        (self.x & 0xffff_ffff) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::Mwc64;

    /// Collect the first `N` outputs of a generator seeded with `seed`.
    #[cfg(test)]
    fn first_n<const N: usize>(seed: u64) -> [u32; N] {
        let mut generator = Mwc64::new(seed);
        core::array::from_fn(|_| generator.next_u32())
    }

    /// Independent re-implementation of a single `MWC` step for cross-checking.
    #[cfg(test)]
    fn manual_step(x: u64) -> u64 {
        let lo = x & 0xffff_ffff;
        let carry = x >> 32;
        Mwc64::A.wrapping_mul(lo).wrapping_add(carry)
    }

    /// External anchor: first four outputs with `seed == 1`.
    const SEED1_VEC: [u32; 4] = [0xffff_da61, 0x0587_58c1, 0x011b_afe3, 0x54e9_4af5];

    #[test]
    fn multiplier_constant_value() {
        assert_eq!(Mwc64::A, 4_294_957_665);
    }

    #[test]
    fn new_sets_state_verbatim() {
        let generator = Mwc64::new(12_345);
        assert_eq!(generator.state(), 12_345);
    }

    #[test]
    fn from_state_sets_state_verbatim() {
        let generator = Mwc64::from_state(0xdead_beef_0000_0001);
        assert_eq!(generator.state(), 0xdead_beef_0000_0001);
    }

    #[test]
    fn new_and_from_state_agree() {
        let a = Mwc64::new(777);
        let b = Mwc64::from_state(777);
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn vector_seed1_index0() {
        let got: [u32; 1] = first_n(1);
        assert_eq!(got[0], SEED1_VEC[0]);
    }

    #[test]
    fn vector_seed1_index1() {
        let got: [u32; 2] = first_n(1);
        assert_eq!(got[1], SEED1_VEC[1]);
    }

    #[test]
    fn vector_seed1_index2() {
        let got: [u32; 3] = first_n(1);
        assert_eq!(got[2], SEED1_VEC[2]);
    }

    #[test]
    fn vector_seed1_index3() {
        let got: [u32; 4] = first_n(1);
        assert_eq!(got[3], SEED1_VEC[3]);
    }

    #[test]
    fn vector_seed1_full_array() {
        let got: [u32; 4] = first_n(1);
        assert_eq!(got, SEED1_VEC);
    }

    #[test]
    fn first_step_state_seed1_is_multiplier() {
        let mut generator = Mwc64::new(1);
        let _ = generator.next_u32();
        assert_eq!(generator.state(), Mwc64::A);
    }

    #[test]
    fn output_is_low_half_of_state() {
        let mut generator = Mwc64::new(0x0123_4567_89ab_cdef);
        let mut i = 0;
        while i < 64 {
            let out = generator.next_u32();
            let expected = (generator.state() & 0xffff_ffff) as u32;
            assert_eq!(out, expected);
            i += 1;
        }
    }

    #[test]
    fn determinism_two_instances() {
        let a: [u32; 16] = first_n(42);
        let b: [u32; 16] = first_n(42);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_long_run() {
        let mut g1 = Mwc64::new(0xabcd_ef01);
        let mut g2 = Mwc64::new(0xabcd_ef01);
        let mut i = 0;
        while i < 512 {
            assert_eq!(g1.next_u32(), g2.next_u32());
            i += 1;
        }
    }

    #[test]
    fn divergence_seed1_vs_seed2_first_output() {
        let a: [u32; 1] = first_n(1);
        let b: [u32; 1] = first_n(2);
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn divergence_many_seed_pairs() {
        let mut seed = 1u64;
        while seed < 24 {
            let a: [u32; 4] = first_n(seed);
            let b: [u32; 4] = first_n(seed + 1);
            assert_ne!(a, b);
            seed += 1;
        }
    }

    #[test]
    fn state_roundtrip_continues_stream() {
        let mut source = Mwc64::new(0x1111_2222_3333_4444);
        let mut i = 0;
        while i < 10 {
            let _ = source.next_u32();
            i += 1;
        }
        let saved = source.state();
        let mut resumed = Mwc64::from_state(saved);
        let mut j = 0;
        while j < 20 {
            assert_eq!(source.next_u32(), resumed.next_u32());
            j += 1;
        }
    }

    #[test]
    fn from_state_midstream_reproduces() {
        let mut source = Mwc64::new(9_000_000);
        let mut skip = 0;
        while skip < 5 {
            let _ = source.next_u32();
            skip += 1;
        }
        let checkpoint = source.state();
        let expected: [u32; 8] = {
            let mut tmp = Mwc64::from_state(checkpoint);
            core::array::from_fn(|_| tmp.next_u32())
        };
        let got: [u32; 8] = core::array::from_fn(|_| source.next_u32());
        assert_eq!(got, expected);
    }

    #[test]
    fn seed_zero_outputs_stay_zero() {
        let outputs: [u32; 32] = first_n(0);
        let mut i = 0;
        while i < outputs.len() {
            assert_eq!(outputs[i], 0);
            i += 1;
        }
    }

    #[test]
    fn seed_zero_state_stays_zero() {
        let mut generator = Mwc64::new(0);
        let mut i = 0;
        while i < 16 {
            let _ = generator.next_u32();
            assert_eq!(generator.state(), 0);
            i += 1;
        }
    }

    #[test]
    fn next_advances_state() {
        let mut generator = Mwc64::new(123_456_789);
        let before = generator.state();
        let _ = generator.next_u32();
        let after = generator.state();
        assert_ne!(before, after);
    }

    #[test]
    fn cross_check_single_step() {
        let seed = 0x00ab_cdef_1234_5678;
        let mut generator = Mwc64::new(seed);
        let _ = generator.next_u32();
        assert_eq!(generator.state(), manual_step(seed));
    }

    #[test]
    fn cross_check_many_steps() {
        let mut generator = Mwc64::new(0x55aa_55aa_55aa_55aa);
        let mut mirror = 0x55aa_55aa_55aa_55aau64;
        let mut i = 0;
        while i < 256 {
            let out = generator.next_u32();
            mirror = manual_step(mirror);
            assert_eq!(generator.state(), mirror);
            assert_eq!(out, (mirror & 0xffff_ffff) as u32);
            i += 1;
        }
    }

    #[test]
    fn high_word_after_first_step_seed1_is_zero() {
        let mut generator = Mwc64::new(1);
        let _ = generator.next_u32();
        assert_eq!(generator.state() >> 32, 0);
    }

    #[test]
    fn output_within_full_u32_range() {
        let mut generator = Mwc64::new(0xfeed_face);
        let mut i = 0;
        while i < 48 {
            let out = generator.next_u32();
            assert!((0..=u32::MAX).contains(&out));
            i += 1;
        }
    }

    #[test]
    fn doubled_output_is_multiple_of_two() {
        let mut generator = Mwc64::new(0x0bad_c0de);
        let mut i = 0;
        while i < 24 {
            let doubled = generator.next_u32().wrapping_mul(2);
            assert!(doubled.is_multiple_of(2));
            i += 1;
        }
    }

    #[test]
    fn div_ceil_by_one_is_identity() {
        let mut generator = Mwc64::new(0x1357_9bdf);
        let mut i = 0;
        while i < 24 {
            let out = generator.next_u32();
            assert_eq!(out.div_ceil(1), out);
            i += 1;
        }
    }

    #[test]
    fn rotate_full_width_is_identity() {
        let mut generator = Mwc64::new(0x2468_ace0);
        let mut i = 0;
        while i < 24 {
            let out = generator.next_u32();
            assert_eq!(out.rotate_left(32), out);
            i += 1;
        }
    }

    #[test]
    fn rotate_left_right_round_trip() {
        let mut generator = Mwc64::new(0x9e37_79b9);
        let mut i = 0;
        while i < 24 {
            let out = generator.next_u32();
            assert_eq!(out.rotate_left(11).rotate_right(11), out);
            i += 1;
        }
    }

    #[test]
    fn different_seeds_differ_after_one_step() {
        let mut a = Mwc64::new(1);
        let mut b = Mwc64::new(2);
        let _ = a.next_u32();
        let _ = b.next_u32();
        assert_ne!(a.state(), b.state());
    }

    #[test]
    fn reproduce_from_saved_state_array() {
        let mut source = Mwc64::new(0x00c0_ffee_00c0_ffee);
        let mut i = 0;
        while i < 7 {
            let _ = source.next_u32();
            i += 1;
        }
        let saved = source.state();
        let a: [u32; 5] = {
            let mut g = Mwc64::from_state(saved);
            core::array::from_fn(|_| g.next_u32())
        };
        let b: [u32; 5] = {
            let mut g = Mwc64::from_state(saved);
            core::array::from_fn(|_| g.next_u32())
        };
        assert_eq!(a, b);
    }

    #[test]
    fn seed_max_is_deterministic() {
        let a: [u32; 12] = first_n(u64::MAX);
        let b: [u32; 12] = first_n(u64::MAX);
        assert_eq!(a, b);
    }

    #[test]
    fn consecutive_seed1_outputs_differ() {
        let got: [u32; 2] = first_n(1);
        assert_ne!(got[0], got[1]);
    }

    #[test]
    fn state_low_half_equals_last_output() {
        let mut generator = Mwc64::new(0x4242_4242);
        let mut last = 0u32;
        let mut i = 0;
        while i < 20 {
            last = generator.next_u32();
            i += 1;
        }
        assert_eq!((generator.state() & 0xffff_ffff) as u32, last);
    }

    #[test]
    fn clone_via_state_matches() {
        let mut source = Mwc64::new(0x7777_8888_9999_aaaa);
        let mut i = 0;
        while i < 13 {
            let _ = source.next_u32();
            i += 1;
        }
        let mut copy = Mwc64::from_state(source.state());
        assert_eq!(source.next_u32(), copy.next_u32());
    }

    #[test]
    fn xor_accumulation_is_deterministic() {
        let fold = |seed: u64| -> u32 {
            let mut generator = Mwc64::new(seed);
            let mut acc = 0u32;
            let mut i = 0;
            while i < 64 {
                acc ^= generator.next_u32();
                i += 1;
            }
            acc
        };
        assert_eq!(fold(0x1234_5678), fold(0x1234_5678));
    }

    #[test]
    fn wrapping_sum_is_deterministic() {
        let total = |seed: u64| -> u32 {
            let mut generator = Mwc64::new(seed);
            let mut acc = 0u32;
            let mut i = 0;
            while i < 64 {
                acc = acc.wrapping_add(generator.next_u32());
                i += 1;
            }
            acc
        };
        assert_eq!(total(321), total(321));
    }

    #[test]
    fn shift_mask_matches_low_word() {
        let mut generator = Mwc64::new(0x0f0f_0f0f_f0f0_f0f0);
        let mut i = 0;
        while i < 32 {
            let out = generator.next_u32();
            let via_shift = (generator.state() << 32 >> 32) as u32;
            assert_eq!(out, via_shift);
            i += 1;
        }
    }

    #[test]
    fn two_checkpoints_cover_full_stream() {
        let mut reference = Mwc64::new(0x0102_0304_0506_0708);
        let first_half: [u32; 8] = core::array::from_fn(|_| reference.next_u32());
        let mid = reference.state();
        let second_half: [u32; 8] = core::array::from_fn(|_| reference.next_u32());

        let mut head = Mwc64::new(0x0102_0304_0506_0708);
        let head_out: [u32; 8] = core::array::from_fn(|_| head.next_u32());
        let mut tail = Mwc64::from_state(mid);
        let tail_out: [u32; 8] = core::array::from_fn(|_| tail.next_u32());

        assert_eq!(head_out, first_half);
        assert_eq!(tail_out, second_half);
    }

    #[test]
    fn seed1_second_state_low_matches_vector() {
        let mut generator = Mwc64::new(1);
        let _ = generator.next_u32();
        let second = generator.next_u32();
        assert_eq!(second, SEED1_VEC[1]);
        assert_eq!((generator.state() & 0xffff_ffff) as u32, SEED1_VEC[1]);
    }

    #[test]
    fn distinct_low_seeds_distinct_first_outputs() {
        let mut seed = 1u64;
        while seed < 10 {
            let here: [u32; 1] = first_n(seed);
            let next: [u32; 1] = first_n(seed + 1);
            assert_ne!(here[0], next[0]);
            seed += 1;
        }
    }

    #[test]
    fn carry_folds_into_next_step() {
        // Choose a state with a non-zero carry (high word) and confirm the
        // manual recurrence folds it back in.
        let seed = (3u64 << 32) | 0x0000_00ab;
        let mut generator = Mwc64::new(seed);
        let _ = generator.next_u32();
        let expected = Mwc64::A.wrapping_mul(0x0000_00ab).wrapping_add(3);
        assert_eq!(generator.state(), expected);
    }

    #[test]
    fn restart_from_initial_seed_matches() {
        let seed = 0x00ff_00ff_00ff_00ff;
        let a: [u32; 10] = first_n(seed);
        let b: [u32; 10] = {
            let mut g = Mwc64::new(seed);
            core::array::from_fn(|_| g.next_u32())
        };
        assert_eq!(a, b);
    }

    #[test]
    fn state_type_is_sixtyfour_bit_width() {
        // The raw state occupies the full 64-bit `type`; a high-bit seed survives
        // intact until the first step consumes it.
        let seed = 1u64 << 63;
        let generator = Mwc64::from_state(seed);
        assert_eq!(generator.state() >> 63, 1);
    }
}
