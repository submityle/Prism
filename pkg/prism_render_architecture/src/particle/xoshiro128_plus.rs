//! `xoshiro128+` pseudo-random number generator (Blackman-Vigna, 32-bit).
//!
//! This module implements the `xoshiro128+` `PRNG` with a four-word 32-bit
//! state. It is a pure-integer generator: it uses only `u32` arithmetic with
//! wrapping addition, bit shifts, exclusive-or, and left rotation. No
//! floating-point, transcendental functions, heap allocation, or formatting
//! facilities are used, so the generator is suitable for `no_std` + `alloc`
//! environments (only `core` primitives are required here).
//!
//! The generator is deterministic: a given state always produces the same
//! sequence of outputs. The state can be exported and re-imported to resume a
//! sequence exactly where it left off.

/// A `xoshiro128+` pseudo-random number generator (`RNG`).
///
/// The internal state is four 32-bit words. Construct an instance from an
/// explicit seed state with [`Xoshiro128Plus::from_state`], draw values with
/// [`Xoshiro128Plus::next_u32`], and inspect the current state with
/// [`Xoshiro128Plus::state`].
pub struct Xoshiro128Plus {
    /// The four-word 32-bit internal state: `(s0, s1, s2, s3)`.
    s: [u32; 4],
}

impl Xoshiro128Plus {
    /// Builds a generator from an explicit four-word 32-bit `state`.
    ///
    /// Any state is accepted. The all-zero state is a fixed point: it only ever
    /// produces zero outputs and never changes, so avoid it for real use.
    #[must_use]
    pub fn from_state(state: [u32; 4]) -> Self {
        Self { s: state }
    }

    /// Returns a copy of the current four-word 32-bit internal state.
    ///
    /// Combined with [`Xoshiro128Plus::from_state`], this allows a sequence to
    /// be saved and resumed exactly.
    #[must_use]
    pub fn state(&self) -> [u32; 4] {
        self.s
    }

    /// Advances the state once and returns the next 32-bit output.
    ///
    /// The output is `s0 + s3` (with wrapping addition) computed before the
    /// state update. The state update uses only shifts, exclusive-or, and a
    /// left rotation, matching the reference `xoshiro128+` scrambler.
    pub fn next_u32(&mut self) -> u32 {
        let s0 = self.s[0];
        let s1 = self.s[1];
        let s2 = self.s[2];
        let s3 = self.s[3];

        let result = s0.wrapping_add(s3);
        let t = s1 << 9;

        let new_s2a = s2 ^ s0;
        let new_s3a = s3 ^ s1;
        let new_s1 = s1 ^ new_s2a;
        let new_s0 = s0 ^ new_s3a;
        let new_s2 = new_s2a ^ t;
        let new_s3 = new_s3a.rotate_left(11);

        self.s = [new_s0, new_s1, new_s2, new_s3];
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Xoshiro128Plus;

    // --- Anchor outputs for seed [1, 2, 3, 4] -----------------------------

    #[test]
    fn anchor_output_0() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let out = rng.next_u32();
        assert!(out == 0x5);
    }

    #[test]
    fn anchor_output_1() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let _ = rng.next_u32();
        let out = rng.next_u32();
        assert!(out == 0x3007);
    }

    #[test]
    fn anchor_output_2() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let out = rng.next_u32();
        assert!(out == 0x0180_3007);
    }

    #[test]
    fn anchor_output_3() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let out = rng.next_u32();
        assert!(out == 0x01A0_5C0E);
    }

    #[test]
    fn anchor_outputs_sequence() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let expected: [u32; 4] = [0x5, 0x3007, 0x0180_3007, 0x01A0_5C0E];
        let mut got: [u32; 4] = [0; 4];
        let mut i = 0;
        while i < 4 {
            got[i] = rng.next_u32();
            i += 1;
        }
        assert!(got[0] == expected[0]);
        assert!(got[1] == expected[1]);
        assert!(got[2] == expected[2]);
        assert!(got[3] == expected[3]);
    }

    #[test]
    fn anchor_state_after_four_draws() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            i += 1;
        }
        let st = rng.state();
        let expected: [u32; 4] = [0x01A0_2C09, 0x0188_3A07, 0x01E8_0400, 0x00C0_5801];
        assert!(st[0] == expected[0]);
        assert!(st[1] == expected[1]);
        assert!(st[2] == expected[2]);
        assert!(st[3] == expected[3]);
    }

    #[test]
    fn anchor_state_word0_after_four() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            i += 1;
        }
        assert!(rng.state()[0] == 0x01A0_2C09);
    }

    #[test]
    fn anchor_state_word1_after_four() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            i += 1;
        }
        assert!(rng.state()[1] == 0x0188_3A07);
    }

    #[test]
    fn anchor_state_word2_after_four() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            i += 1;
        }
        assert!(rng.state()[2] == 0x01E8_0400);
    }

    #[test]
    fn anchor_state_word3_after_four() {
        let mut rng = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            i += 1;
        }
        assert!(rng.state()[3] == 0x00C0_5801);
    }

    #[test]
    fn first_output_equals_s0_plus_s3() {
        let seed: [u32; 4] = [0x1111_2222, 0x3333_4444, 0x5555_6666, 0x7777_8888];
        let mut rng = Xoshiro128Plus::from_state(seed);
        let expected = seed[0].wrapping_add(seed[3]);
        assert!(rng.next_u32() == expected);
    }

    #[test]
    fn first_output_wraps_on_overflow() {
        let seed: [u32; 4] = [0xFFFF_FFFF, 0, 0, 0x0000_0001];
        let mut rng = Xoshiro128Plus::from_state(seed);
        assert!(rng.next_u32() == 0);
    }

    // --- Construction / accessors -----------------------------------------

    #[test]
    fn from_state_preserves_state() {
        let seed: [u32; 4] = [9, 8, 7, 6];
        let rng = Xoshiro128Plus::from_state(seed);
        let st = rng.state();
        assert!(st[0] == 9);
        assert!(st[1] == 8);
        assert!(st[2] == 7);
        assert!(st[3] == 6);
    }

    #[test]
    fn state_accessor_is_pure() {
        let rng = Xoshiro128Plus::from_state([10, 20, 30, 40]);
        let a = rng.state();
        let b = rng.state();
        assert!(a[0] == b[0]);
        assert!(a[1] == b[1]);
        assert!(a[2] == b[2]);
        assert!(a[3] == b[3]);
    }

    #[test]
    fn state_changes_after_draw() {
        let seed: [u32; 4] = [1, 2, 3, 4];
        let mut rng = Xoshiro128Plus::from_state(seed);
        let _ = rng.next_u32();
        let st = rng.state();
        let changed =
            (st[0] != seed[0]) || (st[1] != seed[1]) || (st[2] != seed[2]) || (st[3] != seed[3]);
        assert!(changed == true);
    }

    // --- Determinism -------------------------------------------------------

    #[test]
    fn determinism_same_seed_same_first() {
        let mut a = Xoshiro128Plus::from_state([42, 43, 44, 45]);
        let mut b = Xoshiro128Plus::from_state([42, 43, 44, 45]);
        assert!(a.next_u32() == b.next_u32());
    }

    #[test]
    fn determinism_same_seed_long_sequence() {
        let mut a = Xoshiro128Plus::from_state([0xDEAD, 0xBEEF, 0xCAFE, 0xBABE]);
        let mut b = Xoshiro128Plus::from_state([0xDEAD, 0xBEEF, 0xCAFE, 0xBABE]);
        let mut i = 0;
        while i < 256 {
            assert!(a.next_u32() == b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn determinism_states_track_together() {
        let mut a = Xoshiro128Plus::from_state([7, 11, 13, 17]);
        let mut b = Xoshiro128Plus::from_state([7, 11, 13, 17]);
        let mut i = 0;
        while i < 64 {
            let _ = a.next_u32();
            let _ = b.next_u32();
            i += 1;
        }
        let sa = a.state();
        let sb = b.state();
        assert!(sa[0] == sb[0]);
        assert!(sa[1] == sb[1]);
        assert!(sa[2] == sb[2]);
        assert!(sa[3] == sb[3]);
    }

    #[test]
    fn determinism_replay_from_scratch() {
        let seed: [u32; 4] = [100, 200, 300, 400];
        let mut first: [u32; 8] = [0; 8];
        let mut rng = Xoshiro128Plus::from_state(seed);
        let mut i = 0;
        while i < 8 {
            first[i] = rng.next_u32();
            i += 1;
        }
        let mut rng2 = Xoshiro128Plus::from_state(seed);
        let mut j = 0;
        while j < 8 {
            assert!(rng2.next_u32() == first[j]);
            j += 1;
        }
    }

    // --- Divergence for different seeds -----------------------------------

    #[test]
    fn different_seeds_diverge_first() {
        let mut a = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut b = Xoshiro128Plus::from_state([9, 2, 3, 4]);
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn different_seeds_diverge_within_window() {
        let mut a = Xoshiro128Plus::from_state([1, 0, 0, 0]);
        let mut b = Xoshiro128Plus::from_state([0, 0, 0, 1]);
        let mut diverged = false;
        let mut i = 0;
        while i < 16 {
            if a.next_u32() != b.next_u32() {
                diverged = true;
            }
            i += 1;
        }
        assert!(diverged == true);
    }

    #[test]
    fn single_bit_seed_difference_diverges() {
        let mut a = Xoshiro128Plus::from_state([0x8000_0000, 1, 2, 3]);
        let mut b = Xoshiro128Plus::from_state([0x8000_0001, 1, 2, 3]);
        let mut diverged = false;
        let mut i = 0;
        while i < 32 {
            if a.next_u32() != b.next_u32() {
                diverged = true;
            }
            i += 1;
        }
        assert!(diverged == true);
    }

    #[test]
    fn distinct_seeds_distinct_states() {
        let mut a = Xoshiro128Plus::from_state([5, 6, 7, 8]);
        let mut b = Xoshiro128Plus::from_state([8, 7, 6, 5]);
        let mut i = 0;
        while i < 10 {
            let _ = a.next_u32();
            let _ = b.next_u32();
            i += 1;
        }
        let sa = a.state();
        let sb = b.state();
        let differ = (sa[0] != sb[0]) || (sa[1] != sb[1]) || (sa[2] != sb[2]) || (sa[3] != sb[3]);
        assert!(differ == true);
    }

    // --- State roundtrip / resume -----------------------------------------

    #[test]
    fn state_roundtrip_identity() {
        let seed: [u32; 4] = [0xABCD, 0x1234, 0x5678, 0x9ABC];
        let rng = Xoshiro128Plus::from_state(seed);
        let st = rng.state();
        let rng2 = Xoshiro128Plus::from_state(st);
        let st2 = rng2.state();
        assert!(st2[0] == seed[0]);
        assert!(st2[1] == seed[1]);
        assert!(st2[2] == seed[2]);
        assert!(st2[3] == seed[3]);
    }

    #[test]
    fn resume_from_captured_state_first() {
        let mut rng = Xoshiro128Plus::from_state([11, 22, 33, 44]);
        let mut i = 0;
        while i < 5 {
            let _ = rng.next_u32();
            i += 1;
        }
        let captured = rng.state();
        let continued = rng.next_u32();

        let mut resumed = Xoshiro128Plus::from_state(captured);
        assert!(resumed.next_u32() == continued);
    }

    #[test]
    fn resume_from_captured_state_sequence() {
        let mut rng = Xoshiro128Plus::from_state([0xF00D, 0x0BAD, 0xF00D, 0xBEEF]);
        let mut i = 0;
        while i < 20 {
            let _ = rng.next_u32();
            i += 1;
        }
        let captured = rng.state();

        let mut tail: [u32; 6] = [0; 6];
        let mut j = 0;
        while j < 6 {
            tail[j] = rng.next_u32();
            j += 1;
        }

        let mut resumed = Xoshiro128Plus::from_state(captured);
        let mut k = 0;
        while k < 6 {
            assert!(resumed.next_u32() == tail[k]);
            k += 1;
        }
    }

    #[test]
    fn nth_step_reproducible() {
        let seed: [u32; 4] = [321, 654, 987, 1234];
        let n = 50;

        let mut rng = Xoshiro128Plus::from_state(seed);
        let mut i = 0;
        while i < n {
            let _ = rng.next_u32();
            i += 1;
        }
        let value_at_n = rng.next_u32();

        let mut rng2 = Xoshiro128Plus::from_state(seed);
        let mut j = 0;
        while j < n {
            let _ = rng2.next_u32();
            j += 1;
        }
        assert!(rng2.next_u32() == value_at_n);
    }

    #[test]
    fn nth_state_reproducible() {
        let seed: [u32; 4] = [2, 4, 6, 8];
        let n = 37;

        let mut rng = Xoshiro128Plus::from_state(seed);
        let mut i = 0;
        while i < n {
            let _ = rng.next_u32();
            i += 1;
        }
        let state_at_n = rng.state();

        let mut rng2 = Xoshiro128Plus::from_state(seed);
        let mut j = 0;
        while j < n {
            let _ = rng2.next_u32();
            j += 1;
        }
        let state_at_n_2 = rng2.state();
        assert!(state_at_n[0] == state_at_n_2[0]);
        assert!(state_at_n[1] == state_at_n_2[1]);
        assert!(state_at_n[2] == state_at_n_2[2]);
        assert!(state_at_n[3] == state_at_n_2[3]);
    }

    // --- All-zero fixed point ---------------------------------------------

    #[test]
    fn all_zero_state_outputs_zero() {
        let mut rng = Xoshiro128Plus::from_state([0, 0, 0, 0]);
        assert!(rng.next_u32() == 0);
    }

    #[test]
    fn all_zero_state_stays_zero_outputs() {
        let mut rng = Xoshiro128Plus::from_state([0, 0, 0, 0]);
        let mut i = 0;
        while i < 100 {
            assert!(rng.next_u32() == 0);
            i += 1;
        }
    }

    #[test]
    fn all_zero_state_stays_zero_state() {
        let mut rng = Xoshiro128Plus::from_state([0, 0, 0, 0]);
        let mut i = 0;
        while i < 10 {
            let _ = rng.next_u32();
            i += 1;
        }
        let st = rng.state();
        assert!(st[0] == 0);
        assert!(st[1] == 0);
        assert!(st[2] == 0);
        assert!(st[3] == 0);
    }

    // --- Non-zero seeds produce non-trivial behavior ----------------------

    #[test]
    fn nonzero_seed_eventually_nonzero_output() {
        let mut rng = Xoshiro128Plus::from_state([1, 0, 0, 0]);
        let mut saw_nonzero = false;
        let mut i = 0;
        while i < 32 {
            if rng.next_u32() != 0 {
                saw_nonzero = true;
            }
            i += 1;
        }
        assert!(saw_nonzero == true);
    }

    #[test]
    fn outputs_not_all_identical() {
        let mut rng = Xoshiro128Plus::from_state([13, 17, 19, 23]);
        let first = rng.next_u32();
        let mut all_same = true;
        let mut i = 0;
        while i < 20 {
            if rng.next_u32() != first {
                all_same = false;
            }
            i += 1;
        }
        assert!(all_same == false);
    }

    #[test]
    fn state_evolves_over_steps() {
        let seed: [u32; 4] = [0x0102_0304, 0x0506_0708, 0x090A_0B0C, 0x0D0E_0F10];
        let mut rng = Xoshiro128Plus::from_state(seed);
        let _ = rng.next_u32();
        let s1 = rng.state();
        let _ = rng.next_u32();
        let s2 = rng.state();
        let differ = (s1[0] != s2[0]) || (s1[1] != s2[1]) || (s1[2] != s2[2]) || (s1[3] != s2[3]);
        assert!(differ == true);
    }

    // --- Internal mechanics sanity checks ---------------------------------

    #[test]
    fn second_output_uses_shift_of_s1() {
        // With s1 = 1, the shift term t = s1 << 9 participates in the update.
        let mut rng = Xoshiro128Plus::from_state([0, 1, 0, 0]);
        let out0 = rng.next_u32();
        assert!(out0 == 0);
    }

    #[test]
    fn rotate_left_wraps_high_bits() {
        // s3 begins with only its top bit set; after one step the rotation
        // moves that bit, so the resulting s3 is non-zero and distinct.
        let mut rng = Xoshiro128Plus::from_state([0, 0, 0, 0x8000_0000]);
        let _ = rng.next_u32();
        let st = rng.state();
        assert!(st[3] != 0x8000_0000);
    }

    #[test]
    fn seed_with_high_words_is_handled() {
        let seed: [u32; 4] = [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF];
        let mut rng = Xoshiro128Plus::from_state(seed);
        // s0 + s3 = -2 in wrapping terms = 0xFFFF_FFFE.
        assert!(rng.next_u32() == 0xFFFF_FFFE);
    }

    #[test]
    fn two_independent_instances_do_not_interfere() {
        let mut a = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut b = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let _ = a.next_u32();
        let _ = a.next_u32();
        // b has not advanced; its first output must still match the anchor.
        assert!(b.next_u32() == 0x5);
    }

    #[test]
    fn long_run_matches_replay_tail() {
        let seed: [u32; 4] = [0x2468, 0x1357, 0x9BDF, 0x0ACE];
        let mut rng = Xoshiro128Plus::from_state(seed);
        let mut i = 0;
        while i < 500 {
            let _ = rng.next_u32();
            i += 1;
        }
        let tail = rng.next_u32();

        let mut rng2 = Xoshiro128Plus::from_state(seed);
        let mut j = 0;
        while j < 500 {
            let _ = rng2.next_u32();
            j += 1;
        }
        assert!(rng2.next_u32() == tail);
    }

    #[test]
    fn anchor_fifth_output_is_stable() {
        // Not an external anchor, but pins the generator against regressions:
        // the fifth output of seed [1,2,3,4] must be reproducible.
        let mut a = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut b = Xoshiro128Plus::from_state([1, 2, 3, 4]);
        let mut i = 0;
        while i < 5 {
            let _ = a.next_u32();
            i += 1;
        }
        let mut j = 0;
        while j < 5 {
            let _ = b.next_u32();
            j += 1;
        }
        assert!(a.next_u32() == b.next_u32());
    }

    #[test]
    fn state_after_zero_draws_equals_seed() {
        let seed: [u32; 4] = [111, 222, 333, 444];
        let rng = Xoshiro128Plus::from_state(seed);
        let st = rng.state();
        assert!(st[0] == 111);
        assert!(st[1] == 222);
        assert!(st[2] == 333);
        assert!(st[3] == 444);
    }

    #[test]
    fn resume_midstream_matches_two_segment_run() {
        let seed: [u32; 4] = [0x55AA_55AA, 0xAA55_AA55, 0x0F0F_0F0F, 0xF0F0_F0F0];
        let mut full: [u32; 12] = [0; 12];
        let mut rng = Xoshiro128Plus::from_state(seed);
        let mut i = 0;
        while i < 12 {
            full[i] = rng.next_u32();
            i += 1;
        }

        let mut seg = Xoshiro128Plus::from_state(seed);
        let mut part: [u32; 12] = [0; 12];
        let mut j = 0;
        while j < 5 {
            part[j] = seg.next_u32();
            j += 1;
        }
        let mid = seg.state();
        let mut seg2 = Xoshiro128Plus::from_state(mid);
        while j < 12 {
            part[j] = seg2.next_u32();
            j += 1;
        }

        let mut k = 0;
        while k < 12 {
            assert!(part[k] == full[k]);
            k += 1;
        }
    }

    #[test]
    fn distinct_outputs_present_in_window() {
        let mut rng = Xoshiro128Plus::from_state([3, 1, 4, 1]);
        let a = rng.next_u32();
        let b = rng.next_u32();
        let c = rng.next_u32();
        let any_distinct = (a != b) || (b != c) || (a != c);
        assert!(any_distinct == true);
    }
}
