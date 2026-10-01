//! `RomuDuo` pseudo-random number generator (`PRNG`).
//!
//! This module implements the official two-register 64-bit `RomuDuo`
//! generator. It is written for a `no_std` + `alloc` environment and uses
//! only integer arithmetic: no floating point, no transcendental functions.
//! Every add, subtract, and multiply uses the wrapping variants so the
//! behaviour is identical on every target `CPU` and `GPU` host.
//!
//! The generator keeps two 64-bit registers, `x` and `y`. Each call to
//! [`RomuDuo::next_u64`] returns the previous value of `x` and then mixes the
//! state forward. The multiplier constant is
//! `C == 0xD383_3E80_4F4C_574B` (decimal `15241094284759029579`).

/// Multiplier constant used by the `RomuDuo` `PRNG`.
const C: u64 = 0xD383_3E80_4F4C_574B;

/// Two-register 64-bit `RomuDuo` `PRNG`.
///
/// The fields are public so the state is observable without triggering
/// dead-code lints, and the same state is also available through
/// [`RomuDuo::state`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RomuDuo {
    /// First state register (also the next value returned).
    pub x: u64,
    /// Second state register.
    pub y: u64,
}

impl RomuDuo {
    /// Create a new generator from two seed words.
    pub fn new(x: u64, y: u64) -> Self {
        Self { x, y }
    }

    /// Rebuild a generator from a previously captured `(x, y)` tuple.
    pub fn from_state(state: (u64, u64)) -> Self {
        Self {
            x: state.0,
            y: state.1,
        }
    }

    /// Return the current `(x, y)` state tuple.
    pub fn state(&self) -> (u64, u64) {
        (self.x, self.y)
    }

    /// Advance the generator and return the next 64-bit output.
    ///
    /// The returned value is the old `x`. The new `x` is derived from the old
    /// `y`, and the new `y` is derived from two rotations of the old `y` and
    /// the old `x`.
    pub fn next_u64(&mut self) -> u64 {
        let xp = self.x;
        self.x = C.wrapping_mul(self.y);
        self.y = self
            .y
            .rotate_left(36)
            .wrapping_add(self.y.rotate_left(15))
            .wrapping_sub(xp);
        xp
    }
}

#[cfg(test)]
mod tests {
    use super::{RomuDuo, C};

    // --- Constant correctness -------------------------------------------

    #[test]
    fn constant_hex_value() {
        assert!(C == 0xD383_3E80_4F4C_574B);
    }

    #[test]
    fn constant_decimal_value() {
        assert!(C == 15241094284759029579_u64);
    }

    #[test]
    fn constant_times_two() {
        assert!(C.wrapping_mul(2) == 0xA706_7D00_9E98_AE96);
    }

    // --- Hard reference vectors (anchor) --------------------------------

    #[test]
    fn out0_is_one() {
        let mut rng = RomuDuo::new(1, 2);
        assert!(rng.next_u64() == 0x1);
    }

    #[test]
    fn out1_matches() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        assert!(rng.next_u64() == 0xA706_7D00_9E98_AE96);
    }

    #[test]
    fn out2_matches() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        rng.next_u64();
        assert!(rng.next_u64() == 0x5487_FA2C_07FE_A8B5);
    }

    #[test]
    fn out3_matches() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        rng.next_u64();
        rng.next_u64();
        assert!(rng.next_u64() == 0xD399_E8A5_7470_F60E);
    }

    #[test]
    fn first_four_outputs_array() {
        let mut rng = RomuDuo::new(1, 2);
        let expected: [u64; 4] = [
            0x1,
            0xA706_7D00_9E98_AE96,
            0x5487_FA2C_07FE_A8B5,
            0xD399_E8A5_7470_F60E,
        ];
        let mut got = [0_u64; 4];
        for i in 0..4 {
            got[i] = rng.next_u64();
        }
        assert!(got == expected);
    }

    #[test]
    fn state_after_four_steps() {
        let mut rng = RomuDuo::new(1, 2);
        for _ in 0..4 {
            rng.next_u64();
        }
        assert!(rng.state() == (0x37BC_07E7_D910_C767, 0xF7E7_BE5A_1ACC_9E6F));
    }

    #[test]
    fn state_x_after_four_steps() {
        let mut rng = RomuDuo::new(1, 2);
        for _ in 0..4 {
            rng.next_u64();
        }
        assert!(rng.x == 0x37BC_07E7_D910_C767);
    }

    #[test]
    fn state_y_after_four_steps() {
        let mut rng = RomuDuo::new(1, 2);
        for _ in 0..4 {
            rng.next_u64();
        }
        assert!(rng.y == 0xF7E7_BE5A_1ACC_9E6F);
    }

    #[test]
    fn out0_equals_initial_x() {
        let mut rng = RomuDuo::new(1, 2);
        let seed_x = rng.x;
        assert!(rng.next_u64() == seed_x);
    }

    // --- Single-step state transition -----------------------------------

    #[test]
    fn x_after_first_step() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        assert!(rng.x == 0xA706_7D00_9E98_AE96);
    }

    #[test]
    fn y_after_first_step() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        assert!(rng.y == 0x20_0000_FFFF);
    }

    #[test]
    fn state_after_first_step() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        assert!(rng.state() == (0xA706_7D00_9E98_AE96, 0x20_0000_FFFF));
    }

    // --- Determinism -----------------------------------------------------

    #[test]
    fn deterministic_same_seed() {
        let mut a = RomuDuo::new(42, 99);
        let mut b = RomuDuo::new(42, 99);
        for _ in 0..32 {
            assert!(a.next_u64() == b.next_u64());
        }
    }

    #[test]
    fn deterministic_long_run() {
        let mut a = RomuDuo::new(0xDEAD_BEEF, 0xCAFE_BABE);
        let mut b = RomuDuo::new(0xDEAD_BEEF, 0xCAFE_BABE);
        for _ in 0..1000 {
            assert!(a.next_u64() == b.next_u64());
        }
        assert!(a.state() == b.state());
    }

    #[test]
    fn large_seed_deterministic() {
        let mut a = RomuDuo::new(u64::MAX, u64::MAX - 1);
        let mut b = RomuDuo::new(u64::MAX, u64::MAX - 1);
        for _ in 0..64 {
            assert!(a.next_u64() == b.next_u64());
        }
    }

    // --- Divergence ------------------------------------------------------

    #[test]
    fn different_seeds_diverge() {
        let mut a = RomuDuo::new(1, 2);
        let mut b = RomuDuo::new(3, 4);
        let mut differ = false;
        for _ in 0..16 {
            if a.next_u64() != b.next_u64() {
                differ = true;
            }
        }
        assert!(differ);
    }

    #[test]
    fn different_x_seed_diverges() {
        let mut a = RomuDuo::new(10, 7);
        let mut b = RomuDuo::new(11, 7);
        let mut differ = false;
        for _ in 0..16 {
            if a.next_u64() != b.next_u64() {
                differ = true;
            }
        }
        assert!(differ);
    }

    #[test]
    fn different_y_seed_diverges() {
        let mut a = RomuDuo::new(5, 20);
        let mut b = RomuDuo::new(5, 21);
        let mut differ = false;
        for _ in 0..16 {
            if a.next_u64() != b.next_u64() {
                differ = true;
            }
        }
        assert!(differ);
    }

    // --- state / from_state round trips ---------------------------------

    #[test]
    fn state_roundtrip() {
        let mut rng = RomuDuo::new(123, 456);
        for _ in 0..9 {
            rng.next_u64();
        }
        let snapshot = rng.state();
        let rebuilt = RomuDuo::from_state(snapshot);
        assert!(rebuilt.state() == snapshot);
    }

    #[test]
    fn from_state_equals_new() {
        let built = RomuDuo::from_state((77, 88));
        let made = RomuDuo::new(77, 88);
        assert!(built == made);
    }

    #[test]
    fn new_sets_fields() {
        let rng = RomuDuo::new(0xABCD, 0x1234);
        assert!(rng.x == 0xABCD);
        assert!(rng.y == 0x1234);
    }

    #[test]
    fn state_returns_fields() {
        let rng = RomuDuo::new(555, 666);
        assert!(rng.state() == (555, 666));
    }

    #[test]
    fn reset_via_from_state_reproduces_sequence() {
        let mut rng = RomuDuo::new(9, 10);
        let saved = rng.state();
        let mut first = [0_u64; 5];
        for i in 0..5 {
            first[i] = rng.next_u64();
        }
        let mut restored = RomuDuo::from_state(saved);
        let mut second = [0_u64; 5];
        for i in 0..5 {
            second[i] = restored.next_u64();
        }
        assert!(first == second);
    }

    #[test]
    fn reproduce_from_mid_sequence_state() {
        let mut rng = RomuDuo::new(2024, 1);
        for _ in 0..13 {
            rng.next_u64();
        }
        let mid = rng.state();
        let next = rng.next_u64();
        let mut clone = RomuDuo::from_state(mid);
        assert!(clone.next_u64() == next);
    }

    // --- Nth step reproduction ------------------------------------------

    #[test]
    fn nth_step_reproduce() {
        let mut a = RomuDuo::new(314, 159);
        let mut b = RomuDuo::new(314, 159);
        for _ in 0..100 {
            a.next_u64();
            b.next_u64();
        }
        assert!(a.next_u64() == b.next_u64());
    }

    #[test]
    fn two_rngs_match_after_n_steps() {
        let mut a = RomuDuo::new(7, 11);
        let mut b = RomuDuo::new(7, 11);
        for _ in 0..250 {
            a.next_u64();
        }
        for _ in 0..250 {
            b.next_u64();
        }
        assert!(a.state() == b.state());
    }

    // --- State progression ----------------------------------------------

    #[test]
    fn state_changes_each_step() {
        let mut rng = RomuDuo::new(1, 2);
        let before = rng.state();
        rng.next_u64();
        assert!(rng.state() != before);
    }

    #[test]
    fn consecutive_states_differ() {
        let mut rng = RomuDuo::new(321, 123);
        let s0 = rng.state();
        rng.next_u64();
        let s1 = rng.state();
        rng.next_u64();
        let s2 = rng.state();
        assert!(s0 != s1);
        assert!(s1 != s2);
    }

    #[test]
    fn sequence_not_all_equal() {
        let mut rng = RomuDuo::new(2, 3);
        let mut outs = [0_u64; 6];
        for i in 0..6 {
            outs[i] = rng.next_u64();
        }
        let mut all_same = true;
        for i in 1..6 {
            if outs[i] != outs[0] {
                all_same = false;
            }
        }
        assert!(!all_same);
    }

    #[test]
    fn output_changes_between_calls() {
        let mut rng = RomuDuo::new(99, 100);
        let first = rng.next_u64();
        let second = rng.next_u64();
        assert!(first != second);
    }

    // --- Copy / Clone semantics -----------------------------------------

    #[test]
    fn clone_is_independent() {
        let mut rng = RomuDuo::new(8, 16);
        let mut cloned = rng.clone();
        rng.next_u64();
        rng.next_u64();
        assert!(cloned.next_u64() == 8);
    }

    #[test]
    fn copy_is_independent() {
        let rng = RomuDuo::new(64, 128);
        let mut copied = rng;
        copied.next_u64();
        assert!(rng.x == 64);
    }

    // --- Misc edge cases -------------------------------------------------

    #[test]
    fn equal_seed_registers_equal() {
        let rng = RomuDuo::new(7, 7);
        let x = rng.x;
        let y = rng.y;
        assert!(x == y);
    }

    #[test]
    fn zero_seed_is_fixed_point() {
        // `(0, 0)` is a degenerate fixed point of `RomuDuo`: every output is
        // zero and the state never changes.
        let mut rng = RomuDuo::new(0, 0);
        for _ in 0..5 {
            assert!(rng.next_u64() == 0);
        }
        assert!(rng.state() == (0, 0));
    }

    #[test]
    fn nonzero_seed_progresses() {
        let mut rng = RomuDuo::new(0, 1);
        let before = rng.state();
        for _ in 0..5 {
            rng.next_u64();
        }
        assert!(rng.state() != before);
    }

    #[test]
    fn advance_many_steps_no_panic() {
        let mut rng = RomuDuo::new(0x1234_5678, 0x9ABC_DEF0);
        let mut acc = 0_u64;
        for _ in 0..10_000 {
            acc = acc.wrapping_add(rng.next_u64());
        }
        assert!(acc == acc);
    }

    #[test]
    fn state_matches_after_reconstruct() {
        let mut rng = RomuDuo::new(0xFEED, 0xFACE);
        for _ in 0..21 {
            rng.next_u64();
        }
        let snap = rng.state();
        let rebuilt = RomuDuo::from_state(snap);
        assert!(rebuilt.x == snap.0);
        assert!(rebuilt.y == snap.1);
    }

    #[test]
    fn first_output_one_only_for_this_seed() {
        let mut rng = RomuDuo::new(1, 2);
        assert!(rng.next_u64() == 1);
        let mut other = RomuDuo::new(5, 6);
        assert!(other.next_u64() == 5);
    }

    #[test]
    fn second_instance_fresh_sequence() {
        let mut a = RomuDuo::new(1, 2);
        for _ in 0..3 {
            a.next_u64();
        }
        let mut b = RomuDuo::new(1, 2);
        assert!(b.next_u64() == 1);
    }

    #[test]
    fn full_four_step_state_path() {
        let mut rng = RomuDuo::new(1, 2);
        rng.next_u64();
        assert!(rng.state() == (0xA706_7D00_9E98_AE96, 0x20_0000_FFFF));
        for _ in 0..3 {
            rng.next_u64();
        }
        assert!(rng.state() == (0x37BC_07E7_D910_C767, 0xF7E7_BE5A_1ACC_9E6F));
    }
}
