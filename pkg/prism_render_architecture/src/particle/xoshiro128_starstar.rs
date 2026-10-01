//! `xoshiro128**` pseudo-random number generator (Blackman-Vigna, 32-bit).
//!
//! This module implements the `xoshiro128**` `PRNG` with a four-word 32-bit
//! state. It is a pure-integer generator: it uses only `u32` arithmetic with
//! wrapping multiplication, bit shifts, exclusive-or, and left rotation. No
//! floating-point, transcendental functions, heap allocation, or formatting
//! facilities are used, so the generator is suitable for `no_std` + `alloc`
//! environments (only `core` primitives are required here).
//!
//! The state transition is identical to `xoshiro128+`; only the output
//! scrambler differs. `xoshiro128**` reads `s1` before the state advances and
//! scrambles it with `(s1 * 5) <<< 7 * 9`, which gives it better statistical
//! quality in the low bits than the additive `xoshiro128+` variant.
//!
//! The generator is deterministic: a given state always produces the same
//! sequence of outputs. The state can be exported and re-imported to resume a
//! sequence exactly where it left off.

/// A `xoshiro128**` pseudo-random number generator (`RNG`).
///
/// The internal state is four 32-bit words. Construct an instance from an
/// explicit seed state with [`Xoshiro128StarStar::from_state`], draw values
/// with [`Xoshiro128StarStar::next_u32`], and inspect the current state with
/// [`Xoshiro128StarStar::state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro128StarStar {
    /// The four-word 32-bit internal state: `(s0, s1, s2, s3)`.
    s: [u32; 4],
}

impl Xoshiro128StarStar {
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
    /// Combined with [`Xoshiro128StarStar::from_state`], this allows a sequence
    /// to be saved and resumed exactly.
    #[must_use]
    pub fn state(&self) -> [u32; 4] {
        self.s
    }

    /// Advances the state once and returns the next 32-bit output.
    ///
    /// The output is the `**` scrambler applied to `s1` as it stands before the
    /// state update: `((s1 * 5) <<< 7) * 9`, where `*` is wrapping
    /// multiplication and `<<<` is a left rotation. The state update itself uses
    /// only shifts, exclusive-or, and a left rotation, matching the reference
    /// `xoshiro128**` state transition (identical to `xoshiro128+`).
    pub fn next_u32(&mut self) -> u32 {
        let s0 = self.s[0];
        let s1 = self.s[1];
        let s2 = self.s[2];
        let s3 = self.s[3];

        let result = (s1.wrapping_mul(5)).rotate_left(7).wrapping_mul(9);

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
    use super::Xoshiro128StarStar;

    /// The canonical anchor seed used across the hard reference vectors.
    const SEED: [u32; 4] = [1, 2, 3, 4];

    // --- Anchor outputs for seed [1, 2, 3, 4] -----------------------------

    #[test]
    fn anchor_output_0() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let out = rng.next_u32();
        assert!(out == 0x2d00);
    }

    #[test]
    fn anchor_output_1() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let _ = rng.next_u32();
        let out = rng.next_u32();
        assert!(out == 0x0);
    }

    #[test]
    fn anchor_output_2() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let out = rng.next_u32();
        assert!(out == 0x5a7080);
    }

    #[test]
    fn anchor_output_3() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let out = rng.next_u32();
        assert!(out == 0x0438_9d80);
    }

    #[test]
    fn anchor_full_output_sequence() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let expected: [u32; 4] = [0x2d00, 0x0, 0x5a_7080, 0x0438_9d80];
        let mut i = 0;
        while i < 4 {
            let out = rng.next_u32();
            assert!(out == expected[i]);
            i += 1;
        }
    }

    #[test]
    fn anchor_state_after_four() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            i += 1;
        }
        let expected: [u32; 4] = [0x01a0_2c09, 0x0188_3a07, 0x01e8_0400, 0x00c0_5801];
        assert!(rng.state() == expected);
    }

    // --- Intermediate state vectors ---------------------------------------

    #[test]
    fn state_after_one() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let _ = rng.next_u32();
        let expected: [u32; 4] = [0x0000_0007, 0x0000_0000, 0x0000_0402, 0x0000_3000];
        assert!(rng.state() == expected);
    }

    #[test]
    fn state_after_two() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let expected: [u32; 4] = [0x0000_3007, 0x0000_0405, 0x0000_0405, 0x0180_0000];
        assert!(rng.state() == expected);
    }

    #[test]
    fn state_after_three() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let _ = rng.next_u32();
        let expected: [u32; 4] = [0x0180_3402, 0x0000_3007, 0x0008_3e02, 0x0020_280c];
        assert!(rng.state() == expected);
    }

    #[test]
    fn state_progression_matches_known_vectors() {
        let expected: [[u32; 4]; 4] = [
            [0x0000_0007, 0x0000_0000, 0x0000_0402, 0x0000_3000],
            [0x0000_3007, 0x0000_0405, 0x0000_0405, 0x0180_0000],
            [0x0180_3402, 0x0000_3007, 0x0008_3e02, 0x0020_280c],
            [0x01a0_2c09, 0x0188_3a07, 0x01e8_0400, 0x00c0_5801],
        ];
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let mut i = 0;
        while i < 4 {
            let _ = rng.next_u32();
            assert!(rng.state() == expected[i]);
            i += 1;
        }
    }

    // --- Scrambler behaviour (output depends only on s1) ------------------

    #[test]
    fn scrambler_s1_zero() {
        let mut rng = Xoshiro128StarStar::from_state([7, 0, 11, 13]);
        let out = rng.next_u32();
        assert!(out == 0x0);
    }

    #[test]
    fn scrambler_s1_one() {
        let mut rng = Xoshiro128StarStar::from_state([0, 1, 0, 0]);
        let out = rng.next_u32();
        assert!(out == 0x1680);
    }

    #[test]
    fn scrambler_s1_two() {
        let mut rng = Xoshiro128StarStar::from_state([99, 2, 50, 7]);
        let out = rng.next_u32();
        assert!(out == 0x2d00);
    }

    #[test]
    fn scrambler_s1_six() {
        let mut rng = Xoshiro128StarStar::from_state([0, 6, 0, 0]);
        let out = rng.next_u32();
        assert!(out == 0x8700);
    }

    #[test]
    fn scrambler_s1_1029() {
        let mut rng = Xoshiro128StarStar::from_state([0, 1029, 0, 0]);
        let out = rng.next_u32();
        assert!(out == 0x5a_7080);
    }

    #[test]
    fn scrambler_s1_12295() {
        let mut rng = Xoshiro128StarStar::from_state([0, 12295, 0, 0]);
        let out = rng.next_u32();
        assert!(out == 0x0438_9d80);
    }

    #[test]
    fn scrambler_depends_only_on_s1() {
        let mut a = Xoshiro128StarStar::from_state([1, 42, 3, 4]);
        let mut b = Xoshiro128StarStar::from_state([99, 42, 77, 88]);
        assert!(a.next_u32() == b.next_u32());
    }

    #[test]
    fn scrambler_wraps_with_large_s1() {
        // Pure smoke test: a large s1 must not panic on the shift/multiply.
        let mut rng = Xoshiro128StarStar::from_state([0, 0xffff_ffff, 0, 0]);
        let out = rng.next_u32();
        // ((0xffffffff * 5) rotl 7) * 9, all wrapping, matches reference below.
        assert!(out == 0xffff_edf7);
    }

    // --- Determinism ------------------------------------------------------

    #[test]
    fn determinism_same_seed() {
        let mut a = Xoshiro128StarStar::from_state([9, 8, 7, 6]);
        let mut b = Xoshiro128StarStar::from_state([9, 8, 7, 6]);
        let mut i = 0;
        while i < 64 {
            assert!(a.next_u32() == b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn determinism_anchor_seed() {
        let mut a = Xoshiro128StarStar::from_state(SEED);
        let mut b = Xoshiro128StarStar::from_state(SEED);
        let mut i = 0;
        while i < 128 {
            assert!(a.next_u32() == b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn determinism_state_tracks_identically() {
        let mut a = Xoshiro128StarStar::from_state([13, 17, 19, 23]);
        let mut b = Xoshiro128StarStar::from_state([13, 17, 19, 23]);
        let mut i = 0;
        while i < 50 {
            let _ = a.next_u32();
            let _ = b.next_u32();
            assert!(a.state() == b.state());
            i += 1;
        }
    }

    // --- Divergence -------------------------------------------------------

    #[test]
    fn diverge_first_output() {
        let mut a = Xoshiro128StarStar::from_state([1, 2, 3, 4]);
        let mut b = Xoshiro128StarStar::from_state([4, 3, 2, 1]);
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn diverge_sequences() {
        let mut a = Xoshiro128StarStar::from_state([1, 2, 3, 4]);
        let mut b = Xoshiro128StarStar::from_state([10, 20, 30, 40]);
        let mut diff = false;
        let mut i = 0;
        while i < 16 {
            let x = a.next_u32();
            let y = b.next_u32();
            if x != y {
                diff = true;
            }
            i += 1;
        }
        assert!(diff);
    }

    #[test]
    fn diverge_single_bit_seed() {
        let mut a = Xoshiro128StarStar::from_state([0, 1, 0, 0]);
        let mut b = Xoshiro128StarStar::from_state([0, 3, 0, 0]);
        assert!(a.next_u32() != b.next_u32());
    }

    #[test]
    fn diverge_states_after_steps() {
        let mut a = Xoshiro128StarStar::from_state([5, 6, 7, 8]);
        let mut b = Xoshiro128StarStar::from_state([8, 7, 6, 5]);
        let mut i = 0;
        while i < 8 {
            let _ = a.next_u32();
            let _ = b.next_u32();
            i += 1;
        }
        assert!(a.state() != b.state());
    }

    // --- All-zero fixed point --------------------------------------------

    #[test]
    fn all_zero_output_is_zero() {
        let mut rng = Xoshiro128StarStar::from_state([0, 0, 0, 0]);
        let mut i = 0;
        while i < 32 {
            assert!(rng.next_u32() == 0);
            i += 1;
        }
    }

    #[test]
    fn all_zero_state_stays_zero() {
        let mut rng = Xoshiro128StarStar::from_state([0, 0, 0, 0]);
        let mut i = 0;
        while i < 32 {
            let _ = rng.next_u32();
            assert!(rng.state() == [0, 0, 0, 0]);
            i += 1;
        }
    }

    // --- state / from_state round trips -----------------------------------

    #[test]
    fn from_state_roundtrip_seed() {
        let rng = Xoshiro128StarStar::from_state(SEED);
        assert!(rng.state() == SEED);
    }

    #[test]
    fn from_state_roundtrip_zero() {
        let rng = Xoshiro128StarStar::from_state([0, 0, 0, 0]);
        assert!(rng.state() == [0, 0, 0, 0]);
    }

    #[test]
    fn from_state_roundtrip_arbitrary() {
        let seed: [u32; 4] = [0xdead_beef, 0x0bad_f00d, 0xfeed_face, 0x8bad_f00d];
        let rng = Xoshiro128StarStar::from_state(seed);
        assert!(rng.state() == seed);
    }

    #[test]
    fn reading_state_does_not_mutate() {
        let rng = Xoshiro128StarStar::from_state([111, 222, 333, 444]);
        let a = rng.state();
        let b = rng.state();
        assert!(a == b);
    }

    // --- Resume from captured state ---------------------------------------

    #[test]
    fn reproduce_from_captured_state_short() {
        let mut a = Xoshiro128StarStar::from_state([11, 22, 33, 44]);
        let mut i = 0;
        while i < 20 {
            let _ = a.next_u32();
            i += 1;
        }
        let captured = a.state();
        let mut b = Xoshiro128StarStar::from_state(captured);
        let mut j = 0;
        while j < 20 {
            assert!(a.next_u32() == b.next_u32());
            j += 1;
        }
    }

    #[test]
    fn reproduce_from_captured_state_long() {
        let mut a = Xoshiro128StarStar::from_state([101, 202, 303, 404]);
        let mut i = 0;
        while i < 100 {
            let _ = a.next_u32();
            i += 1;
        }
        let captured = a.state();
        let mut b = Xoshiro128StarStar::from_state(captured);
        let mut j = 0;
        while j < 50 {
            assert!(a.next_u32() == b.next_u32());
            j += 1;
        }
    }

    #[test]
    fn nth_step_reproduce_from_anchor() {
        let mut a = Xoshiro128StarStar::from_state(SEED);
        let mut i = 0;
        while i < 10 {
            let _ = a.next_u32();
            i += 1;
        }
        let captured = a.state();
        let mut b = Xoshiro128StarStar::from_state(captured);
        let mut j = 0;
        while j < 10 {
            assert!(a.next_u32() == b.next_u32());
            j += 1;
        }
    }

    #[test]
    fn rerun_same_prefix_matches() {
        let mut a = Xoshiro128StarStar::from_state([7, 7, 7, 7]);
        let mut b = Xoshiro128StarStar::from_state([7, 7, 7, 7]);
        let mut i = 0;
        while i < 50 {
            let _ = a.next_u32();
            let _ = b.next_u32();
            i += 1;
        }
        assert!(a.state() == b.state());
    }

    // --- Mutation / independence ------------------------------------------

    #[test]
    fn advancing_changes_state() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let before = rng.state();
        let _ = rng.next_u32();
        let after = rng.state();
        assert!(before != after);
    }

    #[test]
    fn two_instances_are_independent() {
        let mut a = Xoshiro128StarStar::from_state([2, 4, 6, 8]);
        let b = Xoshiro128StarStar::from_state([2, 4, 6, 8]);
        let _ = a.next_u32();
        // Advancing `a` must not disturb the untouched instance `b`.
        assert!(b.state() == [2, 4, 6, 8]);
    }

    #[test]
    fn copy_produces_same_sequence() {
        let mut a = Xoshiro128StarStar::from_state([3, 1, 4, 1]);
        let mut b = a;
        let mut i = 0;
        while i < 20 {
            assert!(a.next_u32() == b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn copy_then_advance_one_diverges_state() {
        let a = Xoshiro128StarStar::from_state([5, 9, 2, 6]);
        let mut b = a;
        let _ = b.next_u32();
        assert!(a.state() != b.state());
    }

    // --- Output variety ---------------------------------------------------

    #[test]
    fn anchor_outputs_are_not_all_equal() {
        let mut rng = Xoshiro128StarStar::from_state(SEED);
        let o0 = rng.next_u32();
        let o1 = rng.next_u32();
        assert!(o0 != o1);
    }

    #[test]
    fn sequence_contains_distinct_values() {
        let mut rng = Xoshiro128StarStar::from_state([12, 34, 56, 78]);
        let first = rng.next_u32();
        let mut found_distinct = false;
        let mut i = 0;
        while i < 32 {
            if rng.next_u32() != first {
                found_distinct = true;
            }
            i += 1;
        }
        assert!(found_distinct);
    }

    #[test]
    fn nonzero_seed_eventually_produces_nonzero() {
        let mut rng = Xoshiro128StarStar::from_state([1, 2, 3, 4]);
        let mut any_nonzero = false;
        let mut i = 0;
        while i < 8 {
            if rng.next_u32() != 0 {
                any_nonzero = true;
            }
            i += 1;
        }
        assert!(any_nonzero);
    }

    #[test]
    fn state_words_change_over_time() {
        let mut rng = Xoshiro128StarStar::from_state([1, 1, 1, 1]);
        let start = rng.state();
        let mut i = 0;
        while i < 5 {
            let _ = rng.next_u32();
            i += 1;
        }
        assert!(rng.state() != start);
    }

    #[test]
    fn long_run_is_reproducible_end_to_end() {
        let mut a =
            Xoshiro128StarStar::from_state([0x1234_5678, 0x9abc_def0, 0x0f0f_0f0f, 0xf0f0_f0f0]);
        let mut b =
            Xoshiro128StarStar::from_state([0x1234_5678, 0x9abc_def0, 0x0f0f_0f0f, 0xf0f0_f0f0]);
        let mut i = 0;
        while i < 256 {
            assert!(a.next_u32() == b.next_u32());
            i += 1;
        }
        assert!(a.state() == b.state());
    }
}
