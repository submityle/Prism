//! `RomuTrio` pseudo-random number generator (`PRNG`): Mark Overton's official
//! 64-bit variant, implemented with pure integer `u64` arithmetic.
//!
//! The generator keeps three 64-bit words of state (`x`, `y`, `z`) and emits a
//! single `u64` per step. Every update uses wrapping multiplication/subtraction
//! and bit rotation only. No floating point, transcendental functions, or
//! heap-allocated containers appear anywhere in this module.

/// `RomuTrio` generator state: three 64-bit words.
pub struct RomuTrio {
    x: u64,
    y: u64,
    z: u64,
}

impl RomuTrio {
    /// Builds a `RomuTrio` generator from an explicit `(x, y, z)` state.
    pub fn from_state(x: u64, y: u64, z: u64) -> Self {
        Self { x, y, z }
    }

    /// Advances the generator by one step and returns the previous `x` word.
    pub fn next_u64(&mut self) -> u64 {
        let xp = self.x;
        let yp = self.y;
        let zp = self.z;
        self.x = 15241094284759029579u64.wrapping_mul(zp);
        self.y = (yp.wrapping_sub(xp)).rotate_left(12);
        self.z = (zp.wrapping_sub(yp)).rotate_left(44);
        xp
    }
}

#[cfg(test)]
mod tests {
    use super::RomuTrio;

    /// Draws `N` consecutive outputs into a fixed-size array (no heap use).
    fn draw<const N: usize>(rng: &mut RomuTrio) -> [u64; N] {
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    const SET1: [u64; 6] = [
        0x0000000000000001,
        0x7a89bb80ede505e1,
        0xc574b00000000000,
        0x61cc0dd6fbb3a8b5,
        0x995c06dc2702cb77,
        0xd865c9526c9df272,
    ];

    const SET2: [u64; 6] = [
        0xdeadbeefcafef00d,
        0x7c447f53146e1ab0,
        0xdd90d4fb4953fb2e,
        0x98b618e6a394013d,
        0x8906feadb5a63fb4,
        0xc671c3a3d50a7a3c,
    ];

    fn set1_gen() -> RomuTrio {
        RomuTrio::from_state(1, 2, 3)
    }

    fn set2_gen() -> RomuTrio {
        RomuTrio::from_state(0xdeadbeefcafef00d, 0x0123456789abcdef, 0xfedcba9876543210)
    }

    #[test]
    fn hard_vector_set1_first() {
        let mut g = set1_gen();
        assert_eq!(g.next_u64(), SET1[0]);
    }

    #[test]
    fn hard_vector_set1_second() {
        let mut g = set1_gen();
        g.next_u64();
        assert_eq!(g.next_u64(), SET1[1]);
    }

    #[test]
    fn hard_vector_set1_third() {
        let mut g = set1_gen();
        g.next_u64();
        g.next_u64();
        assert_eq!(g.next_u64(), SET1[2]);
    }

    #[test]
    fn hard_vector_set1_fourth() {
        let mut g = set1_gen();
        let _ = draw::<3>(&mut g);
        assert_eq!(g.next_u64(), SET1[3]);
    }

    #[test]
    fn hard_vector_set1_fifth() {
        let mut g = set1_gen();
        let _ = draw::<4>(&mut g);
        assert_eq!(g.next_u64(), SET1[4]);
    }

    #[test]
    fn hard_vector_set1_sixth() {
        let mut g = set1_gen();
        let _ = draw::<5>(&mut g);
        assert_eq!(g.next_u64(), SET1[5]);
    }

    #[test]
    fn hard_vector_set1_sequence() {
        let mut g = set1_gen();
        let got: [u64; 6] = draw(&mut g);
        assert_eq!(got, SET1);
    }

    #[test]
    fn hard_vector_set2_first() {
        let mut g = set2_gen();
        assert_eq!(g.next_u64(), SET2[0]);
    }

    #[test]
    fn hard_vector_set2_second() {
        let mut g = set2_gen();
        g.next_u64();
        assert_eq!(g.next_u64(), SET2[1]);
    }

    #[test]
    fn hard_vector_set2_third() {
        let mut g = set2_gen();
        g.next_u64();
        g.next_u64();
        assert_eq!(g.next_u64(), SET2[2]);
    }

    #[test]
    fn hard_vector_set2_fourth() {
        let mut g = set2_gen();
        let _ = draw::<3>(&mut g);
        assert_eq!(g.next_u64(), SET2[3]);
    }

    #[test]
    fn hard_vector_set2_fifth() {
        let mut g = set2_gen();
        let _ = draw::<4>(&mut g);
        assert_eq!(g.next_u64(), SET2[4]);
    }

    #[test]
    fn hard_vector_set2_sixth() {
        let mut g = set2_gen();
        let _ = draw::<5>(&mut g);
        assert_eq!(g.next_u64(), SET2[5]);
    }

    #[test]
    fn hard_vector_set2_sequence() {
        let mut g = set2_gen();
        let got: [u64; 6] = draw(&mut g);
        assert_eq!(got, SET2);
    }

    #[test]
    fn first_output_equals_initial_x_set1() {
        let mut g = RomuTrio::from_state(1, 2, 3);
        assert_eq!(g.next_u64(), 1);
    }

    #[test]
    fn first_output_equals_initial_x_set2() {
        let mut g = RomuTrio::from_state(0xdeadbeefcafef00d, 7, 11);
        assert_eq!(g.next_u64(), 0xdeadbeefcafef00d);
    }

    #[test]
    fn first_output_equals_initial_x_arbitrary() {
        let mut g =
            RomuTrio::from_state(0x1122334455667788, 0x99aabbccddeeff00, 0x5555555555555555);
        assert_eq!(g.next_u64(), 0x1122334455667788);
    }

    #[test]
    fn determinism_same_seed_set1() {
        let mut a = set1_gen();
        let mut b = set1_gen();
        let va: [u64; 8] = draw(&mut a);
        let vb: [u64; 8] = draw(&mut b);
        assert_eq!(va, vb);
    }

    #[test]
    fn determinism_same_seed_set2() {
        let mut a = set2_gen();
        let mut b = set2_gen();
        let va: [u64; 8] = draw(&mut a);
        let vb: [u64; 8] = draw(&mut b);
        assert_eq!(va, vb);
    }

    #[test]
    fn determinism_same_seed_arbitrary() {
        let mut a = RomuTrio::from_state(42, 43, 44);
        let mut b = RomuTrio::from_state(42, 43, 44);
        let va: [u64; 16] = draw(&mut a);
        let vb: [u64; 16] = draw(&mut b);
        assert_eq!(va, vb);
    }

    #[test]
    fn determinism_long_run() {
        let mut a = RomuTrio::from_state(0xabc, 0xdef, 0x123);
        let mut b = RomuTrio::from_state(0xabc, 0xdef, 0x123);
        let mut i = 0;
        while i < 10_000 {
            assert_eq!(a.next_u64(), b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn different_x_differs() {
        let mut a = RomuTrio::from_state(1, 2, 3);
        let mut b = RomuTrio::from_state(2, 2, 3);
        let va: [u64; 6] = draw(&mut a);
        let vb: [u64; 6] = draw(&mut b);
        assert!(va != vb);
    }

    #[test]
    fn different_y_differs() {
        let mut a = RomuTrio::from_state(1, 2, 3);
        let mut b = RomuTrio::from_state(1, 9, 3);
        let va: [u64; 6] = draw(&mut a);
        let vb: [u64; 6] = draw(&mut b);
        assert!(va != vb);
    }

    #[test]
    fn different_z_differs() {
        let mut a = RomuTrio::from_state(1, 2, 3);
        let mut b = RomuTrio::from_state(1, 2, 99);
        let va: [u64; 6] = draw(&mut a);
        let vb: [u64; 6] = draw(&mut b);
        assert!(va != vb);
    }

    #[test]
    fn distinct_states_distinct_sequences() {
        let mut a = RomuTrio::from_state(0, 0, 1);
        let mut b = RomuTrio::from_state(0, 1, 0);
        let va: [u64; 12] = draw(&mut a);
        let vb: [u64; 12] = draw(&mut b);
        assert!(va != vb);
    }

    #[test]
    fn two_hard_sets_differ() {
        let mut a = set1_gen();
        let mut b = set2_gen();
        let va: [u64; 6] = draw(&mut a);
        let vb: [u64; 6] = draw(&mut b);
        assert!(va != vb);
    }

    #[test]
    fn from_state_stores_x() {
        let g = RomuTrio::from_state(0xaaaa, 0xbbbb, 0xcccc);
        assert_eq!(g.x, 0xaaaa);
    }

    #[test]
    fn from_state_stores_y() {
        let g = RomuTrio::from_state(0xaaaa, 0xbbbb, 0xcccc);
        assert_eq!(g.y, 0xbbbb);
    }

    #[test]
    fn from_state_stores_z() {
        let g = RomuTrio::from_state(0xaaaa, 0xbbbb, 0xcccc);
        assert_eq!(g.z, 0xcccc);
    }

    #[test]
    fn from_state_consistency_two_instances() {
        let mut a = RomuTrio::from_state(5, 6, 7);
        let mut b = RomuTrio::from_state(5, 6, 7);
        let _ = draw::<3>(&mut a);
        let _ = draw::<3>(&mut b);
        assert_eq!(a.x, b.x);
        assert_eq!(a.y, b.y);
        assert_eq!(a.z, b.z);
    }

    #[test]
    fn snapshot_reproduction_mid_stream() {
        let mut g = set1_gen();
        let _ = draw::<10>(&mut g);
        let mut clone = RomuTrio::from_state(g.x, g.y, g.z);
        let tail_g: [u64; 8] = draw(&mut g);
        let tail_c: [u64; 8] = draw(&mut clone);
        assert_eq!(tail_g, tail_c);
    }

    #[test]
    fn snapshot_reproduction_from_captured_state() {
        let mut g = set2_gen();
        let _ = draw::<25>(&mut g);
        let captured = (g.x, g.y, g.z);
        let continued: [u64; 5] = draw(&mut g);
        let mut replay = RomuTrio::from_state(captured.0, captured.1, captured.2);
        let replayed: [u64; 5] = draw(&mut replay);
        assert_eq!(continued, replayed);
    }

    #[test]
    fn state_after_first_step_set1() {
        let mut g = set1_gen();
        g.next_u64();
        assert_eq!(g.x, 0x7a89bb80ede505e1);
        assert_eq!(g.y, (1u64).rotate_left(12));
        assert_eq!(g.z, (1u64).rotate_left(44));
    }

    #[test]
    fn x_update_formula_check() {
        let mut g = RomuTrio::from_state(10, 20, 30);
        let expected_x = 15241094284759029579u64.wrapping_mul(30);
        g.next_u64();
        assert_eq!(g.x, expected_x);
    }

    #[test]
    fn y_update_formula_check() {
        let mut g = RomuTrio::from_state(10, 20, 30);
        let expected_y = (20u64.wrapping_sub(10)).rotate_left(12);
        g.next_u64();
        assert_eq!(g.y, expected_y);
    }

    #[test]
    fn z_update_formula_check() {
        let mut g = RomuTrio::from_state(10, 20, 30);
        let expected_z = (30u64.wrapping_sub(20)).rotate_left(44);
        g.next_u64();
        assert_eq!(g.z, expected_z);
    }

    #[test]
    fn wrapping_subtraction_underflow() {
        let mut g = RomuTrio::from_state(100, 0, 0);
        let expected_y = (0u64.wrapping_sub(100)).rotate_left(12);
        g.next_u64();
        assert_eq!(g.y, expected_y);
    }

    #[test]
    fn zero_state_first_output() {
        let mut g = RomuTrio::from_state(0, 0, 0);
        assert_eq!(g.next_u64(), 0);
    }

    #[test]
    fn zero_state_stays_zero() {
        let mut g = RomuTrio::from_state(0, 0, 0);
        let out: [u64; 10] = draw(&mut g);
        let zeros = [0u64; 10];
        assert_eq!(out, zeros);
    }

    #[test]
    fn all_ones_state_is_deterministic() {
        let mut a = RomuTrio::from_state(u64::MAX, u64::MAX, u64::MAX);
        let mut b = RomuTrio::from_state(u64::MAX, u64::MAX, u64::MAX);
        let va: [u64; 8] = draw(&mut a);
        let vb: [u64; 8] = draw(&mut b);
        assert_eq!(va, vb);
    }

    #[test]
    fn sequence_is_not_constant() {
        let mut g = set2_gen();
        let out: [u64; 6] = draw(&mut g);
        assert!(out[0] != out[1]);
        assert!(out[1] != out[2]);
        assert!(out[2] != out[3]);
    }

    #[test]
    fn advance_count_matches_sequence() {
        let mut streamed = set1_gen();
        let bulk: [u64; 6] = draw(&mut set1_gen());
        let mut i = 0;
        while i < 6 {
            assert_eq!(streamed.next_u64(), bulk[i]);
            i += 1;
        }
    }

    #[test]
    fn reseed_resets_sequence() {
        let mut g = set1_gen();
        let _ = draw::<50>(&mut g);
        let mut fresh = set1_gen();
        let a: [u64; 6] = draw(&mut fresh);
        assert_eq!(a, SET1);
    }

    #[test]
    fn independent_instances_do_not_interfere() {
        let mut a = RomuTrio::from_state(1, 2, 3);
        let mut b = RomuTrio::from_state(4, 5, 6);
        let _ = draw::<7>(&mut b);
        let va: [u64; 6] = draw(&mut a);
        assert_eq!(va, SET1);
    }

    #[test]
    fn second_output_matches_prior_state_x() {
        let mut g = set1_gen();
        g.next_u64();
        let state_x = g.x;
        assert_eq!(g.next_u64(), state_x);
    }

    #[test]
    fn long_stream_matches_restart() {
        let mut a = RomuTrio::from_state(0xf00d, 0xbaad, 0xc0de);
        let first: [u64; 32] = draw(&mut a);
        let mut b = RomuTrio::from_state(0xf00d, 0xbaad, 0xc0de);
        let second: [u64; 32] = draw(&mut b);
        assert_eq!(first, second);
    }

    #[test]
    fn snapshot_at_zero_equals_fresh() {
        let g = set1_gen();
        let mut clone = RomuTrio::from_state(g.x, g.y, g.z);
        let out: [u64; 6] = draw(&mut clone);
        assert_eq!(out, SET1);
    }
}
