//! `xoroshiro128+` pseudo-random number generator (`PRNG`).
//!
//! This module provides a pure-integer, `CPU` gold-standard implementation of
//! the Blackman/Vigna `xoroshiro128+` generator (128-bit state, `u64` output).
//! It is the `+` sibling of the `xoroshiro128++` generator that also lives in
//! this subsystem: both keep a 128-bit state and advance with a single
//! `rotl`-based step, but the `+` variant uses the plain sum `s0 + s1` as its
//! output scrambler instead of the stronger `++` tempering.
//!
//! The state transition, in `u64` wrapping arithmetic, is:
//!
//! ```text
//! result = s0 + s1
//! t      = s1 ^ s0
//! s0'    = rotl(s0, 24) ^ t ^ (t << 16)
//! s1'    = rotl(t, 37)
//! ```
//!
//! where `rotl(x, k)` is a left bit-rotation implemented via
//! [`u64::rotate_left`]. The jump constants are `a = 24`, `b = 16`, `c = 37`.
//! Seeding is performed with `splitmix64`, mixing a single `u64` seed into the
//! two state words.
//!
//! The implementation is `no_std` + `alloc` friendly: it uses only `u64`
//! bitwise `XOR`, fixed-width shifts, [`u64::rotate_left`], [`u64::wrapping_add`]
//! and [`u64::wrapping_mul`]. There are no floating-point, division, or
//! transcendental operations, which keeps the `RNG` bit-exact across targets.

/// A `xoroshiro128+` pseudo-random number generator (`PRNG`).
///
/// Holds a 128-bit state split across two `u64` words. Construct it from a
/// single seed via [`Xoroshiro128Plus::from_seed`] (which uses `splitmix64`)
/// or directly from raw state words via [`Xoroshiro128Plus::from_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xoroshiro128Plus {
    s0: u64,
    s1: u64,
}

/// `splitmix64` mixing step: advances `state` and returns the next `u64` value.
///
/// This is the standard `splitmix64` finalizer used to seed `xoroshiro`/`xoshiro`
/// family generators from a single `u64` seed. It is exposed to make seeding
/// deterministic and independently testable.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// Compile-time enshrinement of the first `xoroshiro128+` output for the raw
// state (s0 = 1, s1 = 2): the `+` variant emits s0 + s1 directly.
const _FIRST_OUTPUT_FROM_STATE_1_2: () = const {
    let s0: u64 = 1;
    let s1: u64 = 2;
    let result = s0.wrapping_add(s1);
    assert!(result == 0x0000_0000_0000_0003);
};

// Compile-time enshrinement that `rotl` is a true bit-rotation (via
// `rotate_left`) and not a plain shift.
const _ROTL_IS_ROTATION: () = const {
    assert!(1u64.rotate_left(24) == (1u64 << 24));
    assert!(0x8000_0000_0000_0000u64.rotate_left(1) == 1u64);
};

impl Xoroshiro128Plus {
    /// Builds a generator from a single `u64` seed using `splitmix64` to derive
    /// the two 128-bit state words.
    pub fn from_seed(seed: u64) -> Self {
        let mut st = seed;
        let s0 = splitmix64(&mut st);
        let s1 = splitmix64(&mut st);
        Self { s0, s1 }
    }

    /// Builds a generator directly from the two raw `u64` state words.
    ///
    /// Note that the all-zero state is a fixed point of the generator and will
    /// only ever produce zero; prefer [`Xoroshiro128Plus::from_seed`] for
    /// general use.
    pub fn from_state(s0: u64, s1: u64) -> Self {
        Self { s0, s1 }
    }

    /// Returns the current raw state words `(s0, s1)`.
    ///
    /// Useful for snapshotting a generator so its sequence can be reproduced
    /// later via [`Xoroshiro128Plus::from_state`].
    pub fn state(&self) -> (u64, u64) {
        (self.s0, self.s1)
    }

    /// Advances the generator and returns the next `u64` output.
    pub fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let s1 = self.s1;
        let result = s0.wrapping_add(s1);
        let t = s1 ^ s0;
        self.s0 = s0.rotate_left(24) ^ t ^ (t << 16);
        self.s1 = t.rotate_left(37);
        result
    }

    /// Advances the generator `N` times, returning the outputs as an array.
    pub fn next_array<const N: usize>(&mut self) -> [u64; N] {
        core::array::from_fn(|_| self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known `splitmix64`-derived initial state for seed = 1.
    const SEED1_S0: u64 = 0x910a_2dec_8902_5cc1;
    const SEED1_S1: u64 = 0xbeeb_8da1_658e_ec67;

    // First twelve outputs from the state derived by seeding with seed = 1.
    const SEED1_OUTPUTS: [u64; 12] = [
        0x4ff5_bb8d_ee91_4928,
        0xf4bb_6363_99ef_c448,
        0x676c_e74b_b045_e184,
        0x85a5_e215_3b0d_8255,
        0xce7f_0a02_de29_ba2d,
        0xd3e6_2e54_36b1_1a09,
        0x7435_598e_291c_8bd3,
        0x663d_71e2_e2ff_214f,
        0xb3e9_aee0_a309_e59b,
        0x6f4d_568c_ff88_8816,
        0x24ce_1ab2_f878_9f89,
        0xbf83_e8b6_241d_4e6d,
    ];

    // First twelve outputs from the raw state (s0 = 1, s1 = 2).
    const STATE_1_2_OUTPUTS: [u64; 12] = [
        0x0000_0000_0000_0003,
        0x0000_0060_0103_0003,
        0x20c1_02c3_0200_0c03,
        0x8101_8067_0d23_ad61,
        0x26d1_3a49_4133_3a42,
        0x538a_501c_02f5_8b2e,
        0x2ab2_076d_ee38_2f7e,
        0x30df_cfb7_22fe_cd9c,
        0x67fc_b951_936c_f290,
        0x0cc7_57f0_3a0a_e0cf,
        0x6ec3_9d07_b4ce_755d,
        0x6c97_332b_2f02_464f,
    ];

    #[test]
    fn seed1_initial_state_s0() {
        let rng = Xoroshiro128Plus::from_seed(1);
        assert!(rng.state().0 == SEED1_S0);
    }

    #[test]
    fn seed1_initial_state_s1() {
        let rng = Xoroshiro128Plus::from_seed(1);
        assert!(rng.state().1 == SEED1_S1);
    }

    #[test]
    fn seed1_initial_state_pair() {
        let rng = Xoroshiro128Plus::from_seed(1);
        assert!(rng.state() == (SEED1_S0, SEED1_S1));
    }

    #[test]
    fn seed1_output_0() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        assert!(rng.next_u64() == SEED1_OUTPUTS[0]);
    }

    #[test]
    fn seed1_output_1() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _ = rng.next_u64();
        assert!(rng.next_u64() == SEED1_OUTPUTS[1]);
    }

    #[test]
    fn seed1_output_2() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _: [u64; 2] = rng.next_array();
        assert!(rng.next_u64() == SEED1_OUTPUTS[2]);
    }

    #[test]
    fn seed1_output_3() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _: [u64; 3] = rng.next_array();
        assert!(rng.next_u64() == SEED1_OUTPUTS[3]);
    }

    #[test]
    fn seed1_output_4() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _: [u64; 4] = rng.next_array();
        assert!(rng.next_u64() == SEED1_OUTPUTS[4]);
    }

    #[test]
    fn seed1_output_5() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _: [u64; 5] = rng.next_array();
        assert!(rng.next_u64() == SEED1_OUTPUTS[5]);
    }

    #[test]
    fn seed1_output_6() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _: [u64; 6] = rng.next_array();
        assert!(rng.next_u64() == SEED1_OUTPUTS[6]);
    }

    #[test]
    fn seed1_output_7() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let _: [u64; 7] = rng.next_array();
        assert!(rng.next_u64() == SEED1_OUTPUTS[7]);
    }

    #[test]
    fn seed1_sequence_array_first_five() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let got: [u64; 5] = rng.next_array();
        let expected: [u64; 5] = [
            SEED1_OUTPUTS[0],
            SEED1_OUTPUTS[1],
            SEED1_OUTPUTS[2],
            SEED1_OUTPUTS[3],
            SEED1_OUTPUTS[4],
        ];
        assert!(got == expected);
    }

    #[test]
    fn seed1_sequence_array_twelve() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let got: [u64; 12] = rng.next_array();
        assert!(got == SEED1_OUTPUTS);
    }

    #[test]
    fn seed1_sequence_matches_elementwise() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        let got: [u64; 12] = rng.next_array();
        for (g, e) in got.iter().zip(SEED1_OUTPUTS.iter()) {
            assert!(g == e);
        }
    }

    #[test]
    fn state_1_2_output_0() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[0]);
    }

    #[test]
    fn state_1_2_output_1() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _ = rng.next_u64();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[1]);
    }

    #[test]
    fn state_1_2_output_2() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _: [u64; 2] = rng.next_array();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[2]);
    }

    #[test]
    fn state_1_2_output_3() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _: [u64; 3] = rng.next_array();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[3]);
    }

    #[test]
    fn state_1_2_output_4() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _: [u64; 4] = rng.next_array();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[4]);
    }

    #[test]
    fn state_1_2_output_5() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _: [u64; 5] = rng.next_array();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[5]);
    }

    #[test]
    fn state_1_2_output_6() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _: [u64; 6] = rng.next_array();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[6]);
    }

    #[test]
    fn state_1_2_output_7() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _: [u64; 7] = rng.next_array();
        assert!(rng.next_u64() == STATE_1_2_OUTPUTS[7]);
    }

    #[test]
    fn state_1_2_first_output_is_three() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        assert!(rng.next_u64() == 0x0000_0000_0000_0003);
    }

    #[test]
    fn state_1_2_sequence_array_first_five() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let got: [u64; 5] = rng.next_array();
        let expected: [u64; 5] = [
            STATE_1_2_OUTPUTS[0],
            STATE_1_2_OUTPUTS[1],
            STATE_1_2_OUTPUTS[2],
            STATE_1_2_OUTPUTS[3],
            STATE_1_2_OUTPUTS[4],
        ];
        assert!(got == expected);
    }

    #[test]
    fn state_1_2_sequence_array_twelve() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let got: [u64; 12] = rng.next_array();
        assert!(got == STATE_1_2_OUTPUTS);
    }

    #[test]
    fn splitmix64_seed1_first() {
        let mut st = 1u64;
        assert!(splitmix64(&mut st) == SEED1_S0);
    }

    #[test]
    fn splitmix64_seed1_second() {
        let mut st = 1u64;
        let _ = splitmix64(&mut st);
        assert!(splitmix64(&mut st) == SEED1_S1);
    }

    #[test]
    fn splitmix64_seed1_pair() {
        let mut st = 1u64;
        let pair: [u64; 2] = core::array::from_fn(|_| splitmix64(&mut st));
        assert!(pair == [SEED1_S0, SEED1_S1]);
    }

    #[test]
    fn splitmix64_advances_state() {
        let mut st = 0u64;
        let _ = splitmix64(&mut st);
        assert!(st != 0);
        assert!(st == 0x9E37_79B9_7F4A_7C15);
    }

    #[test]
    fn splitmix64_sequence_distinct() {
        let mut st = 123u64;
        let seq: [u64; 4] = core::array::from_fn(|_| splitmix64(&mut st));
        assert!(seq[0] != seq[1]);
        assert!(seq[1] != seq[2]);
        assert!(seq[2] != seq[3]);
    }

    #[test]
    fn splitmix64_different_initial_states_differ() {
        let mut a = 10u64;
        let mut b = 11u64;
        assert!(splitmix64(&mut a) != splitmix64(&mut b));
    }

    #[test]
    fn seed_determinism_same_seed_same_sequence() {
        let mut a = Xoroshiro128Plus::from_seed(42);
        let mut b = Xoroshiro128Plus::from_seed(42);
        let xa: [u64; 8] = a.next_array();
        let xb: [u64; 8] = b.next_array();
        assert!(xa == xb);
    }

    #[test]
    fn seed_determinism_repeated_construction() {
        let first: [u64; 6] = Xoroshiro128Plus::from_seed(777).next_array();
        let second: [u64; 6] = Xoroshiro128Plus::from_seed(777).next_array();
        assert!(first == second);
    }

    #[test]
    fn different_seed_different_sequence() {
        let a: [u64; 5] = Xoroshiro128Plus::from_seed(1).next_array();
        let b: [u64; 5] = Xoroshiro128Plus::from_seed(2).next_array();
        assert!(a != b);
    }

    #[test]
    fn seed1_differs_from_seed3() {
        let a: [u64; 4] = Xoroshiro128Plus::from_seed(1).next_array();
        let b: [u64; 4] = Xoroshiro128Plus::from_seed(3).next_array();
        assert!(a != b);
    }

    #[test]
    fn from_state_from_seed_consistency() {
        let mut a = Xoroshiro128Plus::from_seed(1);
        let mut b = Xoroshiro128Plus::from_state(SEED1_S0, SEED1_S1);
        let xa: [u64; 10] = a.next_array();
        let xb: [u64; 10] = b.next_array();
        assert!(xa == xb);
    }

    #[test]
    fn from_state_from_seed_same_initial_state() {
        let a = Xoroshiro128Plus::from_seed(1);
        let b = Xoroshiro128Plus::from_state(SEED1_S0, SEED1_S1);
        assert!(a == b);
    }

    #[test]
    fn reproducible_advance_via_state_snapshot() {
        let mut a = Xoroshiro128Plus::from_seed(99);
        let _: [u64; 10] = a.next_array();
        let (s0, s1) = a.state();
        let mut b = Xoroshiro128Plus::from_state(s0, s1);
        let xa: [u64; 5] = a.next_array();
        let xb: [u64; 5] = b.next_array();
        assert!(xa == xb);
    }

    #[test]
    fn reproducible_skip_matches_tail() {
        let mut full = Xoroshiro128Plus::from_seed(7);
        let all: [u64; 20] = full.next_array();
        let mut skipped = Xoroshiro128Plus::from_seed(7);
        let _: [u64; 10] = skipped.next_array();
        let tail: [u64; 10] = skipped.next_array();
        for (t, a) in tail.iter().zip(all.iter().skip(10)) {
            assert!(t == a);
        }
    }

    #[test]
    fn next_array_matches_repeated_next_u64() {
        let mut a = Xoroshiro128Plus::from_seed(555);
        let mut b = Xoroshiro128Plus::from_seed(555);
        let arr: [u64; 6] = a.next_array();
        let manual: [u64; 6] = core::array::from_fn(|_| b.next_u64());
        assert!(arr == manual);
    }

    #[test]
    fn state_changes_after_advance() {
        let mut rng = Xoroshiro128Plus::from_seed(12345);
        let before = rng.state();
        let _ = rng.next_u64();
        assert!(before != rng.state());
    }

    #[test]
    fn two_instances_are_independent() {
        let mut a = Xoroshiro128Plus::from_seed(1);
        let b = Xoroshiro128Plus::from_seed(1);
        let _ = a.next_u64();
        assert!(a.state() != b.state());
    }

    #[test]
    fn seed_zero_produces_nonzero_output() {
        let mut rng = Xoroshiro128Plus::from_seed(0);
        let out: [u64; 4] = rng.next_array();
        assert!(out.iter().any(|&v| v != 0));
    }

    #[test]
    fn seed_max_produces_nonzero_output() {
        let mut rng = Xoroshiro128Plus::from_seed(u64::MAX);
        let out: [u64; 4] = rng.next_array();
        assert!(out.iter().any(|&v| v != 0));
    }

    #[test]
    fn all_zero_state_is_fixed_point() {
        let mut rng = Xoroshiro128Plus::from_state(0, 0);
        let out: [u64; 8] = rng.next_array();
        assert!(out.iter().all(|&v| v == 0));
        assert!(rng.state() == (0, 0));
    }

    #[test]
    fn long_run_does_not_panic() {
        let mut rng = Xoroshiro128Plus::from_seed(0xDEAD_BEEF);
        let mut acc: u64 = 0;
        let out: [u64; 1000] = rng.next_array();
        for &v in out.iter() {
            acc ^= v;
        }
        // The accumulator is overwhelmingly unlikely to be exactly zero.
        assert!(acc != 0);
    }

    #[test]
    fn sequence_is_not_constant() {
        let mut rng = Xoroshiro128Plus::from_seed(0xABCD);
        let out: [u64; 5] = rng.next_array();
        assert!(out[0] != out[1]);
        assert!(out[1] != out[2]);
    }

    #[test]
    fn consecutive_outputs_differ_from_state() {
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let out: [u64; 5] = rng.next_array();
        assert!(out[0] != out[1]);
        assert!(out[2] != out[3]);
    }

    #[test]
    fn some_output_has_high_bit_set() {
        let mut rng = Xoroshiro128Plus::from_seed(2024);
        let out: [u64; 16] = rng.next_array();
        assert!(out.iter().any(|&v| (v >> 63) == 1));
    }

    #[test]
    fn clone_copy_reproduces_same_sequence() {
        let mut original = Xoroshiro128Plus::from_seed(314);
        let _ = original.next_u64();
        let snapshot = original;
        let mut a = snapshot;
        let mut b = snapshot;
        let xa: [u64; 4] = a.next_array();
        let xb: [u64; 4] = b.next_array();
        assert!(xa == xb);
    }

    #[test]
    fn next_u64_single_step_transition() {
        // Mirror the documented transition for (s0 = 1, s1 = 2) and verify the
        // resulting state words match an independent computation.
        let mut rng = Xoroshiro128Plus::from_state(1, 2);
        let _ = rng.next_u64();
        let t: u64 = 2 ^ 1;
        let expected_s0 = 1u64.rotate_left(24) ^ t ^ (t << 16);
        let expected_s1 = t.rotate_left(37);
        assert!(rng.state() == (expected_s0, expected_s1));
    }

    #[test]
    fn seed_sequences_are_well_distributed() {
        let mut rng = Xoroshiro128Plus::from_seed(0x1234_5678_9abc_def0);
        let out: [u64; 8] = rng.next_array();
        // All eight outputs from a good seed should be pairwise distinct.
        for i in 0..out.len() {
            for j in (i + 1)..out.len() {
                assert!(out[i] != out[j]);
            }
        }
    }

    #[test]
    fn from_state_roundtrip_through_state_getter() {
        let rng = Xoroshiro128Plus::from_state(0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555);
        assert!(rng.state() == (0xAAAA_AAAA_AAAA_AAAA, 0x5555_5555_5555_5555));
    }

    #[test]
    fn different_initial_state_different_sequence() {
        let a: [u64; 6] = Xoroshiro128Plus::from_state(1, 2).next_array();
        let b: [u64; 6] = Xoroshiro128Plus::from_state(2, 1).next_array();
        assert!(a != b);
    }

    #[test]
    fn seed1_full_sequence_hard_vector() {
        let mut rng = Xoroshiro128Plus::from_seed(1);
        assert!(rng.state() == (SEED1_S0, SEED1_S1));
        let got: [u64; 12] = rng.next_array();
        assert!(got == SEED1_OUTPUTS);
    }
}
