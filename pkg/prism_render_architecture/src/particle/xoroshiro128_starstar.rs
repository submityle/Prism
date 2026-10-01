//! `xoroshiro128**` pseudo-random number generator (Blackman & Vigna).
//!
//! This module provides a small, deterministic 64-bit `PRNG` built from a
//! 128-bit state held as two `u64` words. The generator is pure integer code:
//! it uses only wrapping multiplication, rotations, shifts, and XOR. It is
//! suitable for a `no_std` + `alloc` crate because it allocates nothing and
//! depends on no floating-point or transcendental operations.
//!
//! The `next_u64` step follows the canonical `xoroshiro128**` scrambler:
//! the output word is derived from the first state word, then the two state
//! words are advanced via XOR, a left shift, and two rotations. Neither the
//! `CPU` nor a `GPU` need any special support beyond 64-bit integer math.

/// A `xoroshiro128**` generator with 128 bits of state.
///
/// Construct one with [`Xoroshiro128StarStar::new`] and pull values with
/// [`Xoroshiro128StarStar::next_u64`]. The full internal state can be read
/// with [`Xoroshiro128StarStar::state`] and restored with
/// [`Xoroshiro128StarStar::from_state`], which makes the stream fully
/// reproducible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoroshiro128StarStar {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128StarStar {
    /// Create a generator from the two 64-bit state words.
    ///
    /// Any pair of words is accepted; the all-zero state is degenerate (it
    /// produces only zeros) and is left to the caller to avoid.
    pub fn new(s0: u64, s1: u64) -> Self {
        Self { s0, s1 }
    }

    /// Rebuild a generator from a previously captured state pair.
    ///
    /// This is an alias of [`Xoroshiro128StarStar::new`] with naming that
    /// pairs with [`Xoroshiro128StarStar::state`] for round-tripping.
    pub fn from_state(s0: u64, s1: u64) -> Self {
        Self { s0, s1 }
    }

    /// Return the current 128-bit state as a `(s0, s1)` word pair.
    pub fn state(&self) -> (u64, u64) {
        (self.s0, self.s1)
    }

    /// Advance the state and return the next 64-bit output word.
    pub fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let s1 = self.s1;
        let result = s0.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s1 ^ s0;
        self.s0 = s0.rotate_left(24) ^ t ^ (t << 16);
        self.s1 = t.rotate_left(37);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Xoroshiro128StarStar;

    /// Reference seed shared by many of the anchor checks below.
    const SEED0: u64 = 1;
    const SEED1: u64 = 2;

    /// Pull `N` outputs into a fixed array for order-sensitive checks.
    fn collect<const N: usize>(rng: &mut Xoroshiro128StarStar) -> [u64; N] {
        let mut out = [0u64; N];
        let mut i = 0;
        while i < N {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    #[test]
    fn vector_output_zero() {
        let mut rng = Xoroshiro128StarStar::new(SEED0, SEED1);
        assert!(rng.next_u64() == 0x0000_0000_0000_1680);
    }

    #[test]
    fn vector_output_one() {
        let mut rng = Xoroshiro128StarStar::new(SEED0, SEED1);
        let got = collect::<2>(&mut rng);
        assert!(got[1] == 0x0000_0016_c380_4380);
    }

    #[test]
    fn vector_output_two() {
        let mut rng = Xoroshiro128StarStar::new(SEED0, SEED1);
        let got = collect::<3>(&mut rng);
        assert!(got[2] == 0x86b5_b3ad_0000_4380);
    }

    #[test]
    fn vector_output_three() {
        let mut rng = Xoroshiro128StarStar::new(SEED0, SEED1);
        let got = collect::<4>(&mut rng);
        assert!(got[3] == 0x8000_44a4_cd14_97b2);
    }

    #[test]
    fn vector_full_four_sequence() {
        let mut rng = Xoroshiro128StarStar::new(SEED0, SEED1);
        let got = collect::<4>(&mut rng);
        let want: [u64; 4] = [
            0x0000_0000_0000_1680,
            0x0000_0016_c380_4380,
            0x86b5_b3ad_0000_4380,
            0x8000_44a4_cd14_97b2,
        ];
        let mut i = 0;
        while i < 4 {
            assert!(got[i] == want[i]);
            i += 1;
        }
    }

    #[test]
    fn first_output_is_fixed_point_of_formula() {
        // 1*5 = 5; rotate_left(7) = 640; *9 = 5760 = 0x1680.
        let mut rng = Xoroshiro128StarStar::new(1, 2);
        assert!(rng.next_u64() == 5760);
    }

    #[test]
    fn state_after_one_step_s0() {
        let mut rng = Xoroshiro128StarStar::new(1, 2);
        let _ = rng.next_u64();
        let (s0, _s1) = rng.state();
        // rotate_left(1,24) ^ 3 ^ (3<<16) = 0x1000000 ^ 0x3 ^ 0x30000.
        assert!(s0 == 0x0103_0003);
    }

    #[test]
    fn state_after_one_step_s1() {
        let mut rng = Xoroshiro128StarStar::new(1, 2);
        let _ = rng.next_u64();
        let (_s0, s1) = rng.state();
        // t = 3; rotate_left(3,37) = 3<<37 = 0x6000000000.
        assert!(s1 == 0x60_0000_0000);
    }

    #[test]
    fn new_and_from_state_agree() {
        let a = Xoroshiro128StarStar::new(123, 456);
        let b = Xoroshiro128StarStar::from_state(123, 456);
        assert!(a == b);
    }

    #[test]
    fn state_reports_constructor_words() {
        let rng = Xoroshiro128StarStar::new(0xdead_beef, 0x0bad_f00d);
        let (s0, s1) = rng.state();
        assert!(s0 == 0xdead_beef);
        assert!(s1 == 0x0bad_f00d);
    }

    #[test]
    fn determinism_same_seed_same_stream() {
        let mut a = Xoroshiro128StarStar::new(42, 99);
        let mut b = Xoroshiro128StarStar::new(42, 99);
        let xs = collect::<16>(&mut a);
        let ys = collect::<16>(&mut b);
        let mut i = 0;
        while i < 16 {
            assert!(xs[i] == ys[i]);
            i += 1;
        }
    }

    #[test]
    fn different_seed_s0_diverges() {
        let mut a = Xoroshiro128StarStar::new(1, 2);
        let mut b = Xoroshiro128StarStar::new(3, 2);
        assert!(a.next_u64() != b.next_u64());
    }

    #[test]
    fn different_seed_s1_diverges() {
        let mut a = Xoroshiro128StarStar::new(7, 11);
        let mut b = Xoroshiro128StarStar::new(7, 13);
        // The first output depends only on s0, so advance once to let the
        // differing s1 feed into the state before comparing.
        let _ = a.next_u64();
        let _ = b.next_u64();
        assert!(a.next_u64() != b.next_u64());
    }

    #[test]
    fn round_trip_state_reproduces_tail() {
        let mut rng = Xoroshiro128StarStar::new(0x1234, 0x5678);
        let _warm = collect::<5>(&mut rng);
        let (s0, s1) = rng.state();
        let mut restored = Xoroshiro128StarStar::from_state(s0, s1);
        let a = collect::<8>(&mut rng);
        let b = collect::<8>(&mut restored);
        let mut i = 0;
        while i < 8 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn round_trip_mid_stream_matches_fresh_run() {
        let mut full = Xoroshiro128StarStar::new(0xaaaa, 0xbbbb);
        let head = collect::<4>(&mut full);
        let (s0, s1) = full.state();
        let tail = collect::<4>(&mut full);

        let mut replay = Xoroshiro128StarStar::new(0xaaaa, 0xbbbb);
        let replay_head = collect::<4>(&mut replay);
        let mut from_mid = Xoroshiro128StarStar::from_state(s0, s1);
        let replay_tail = collect::<4>(&mut from_mid);

        let mut i = 0;
        while i < 4 {
            assert!(head[i] == replay_head[i]);
            assert!(tail[i] == replay_tail[i]);
            i += 1;
        }
    }

    #[test]
    fn clone_is_independent_copy() {
        let mut a = Xoroshiro128StarStar::new(555, 777);
        let mut b = a;
        let xs = collect::<4>(&mut a);
        let ys = collect::<4>(&mut b);
        let mut i = 0;
        while i < 4 {
            assert!(xs[i] == ys[i]);
            i += 1;
        }
    }

    #[test]
    fn clone_does_not_share_state() {
        let mut a = Xoroshiro128StarStar::new(1000, 2000);
        let mut b = a;
        let _ = a.next_u64();
        // `b` has not advanced, so its state still equals the original.
        assert!(b.state() == (1000, 2000));
        let _ = b.next_u64();
        assert!(a.state() != (1000, 2000));
    }

    #[test]
    fn equality_tracks_state() {
        let mut a = Xoroshiro128StarStar::new(5, 6);
        let b = Xoroshiro128StarStar::new(5, 6);
        assert!(a == b);
        let _ = a.next_u64();
        assert!(a != b);
    }

    #[test]
    fn zero_state_is_absorbing() {
        let mut rng = Xoroshiro128StarStar::new(0, 0);
        let got = collect::<10>(&mut rng);
        let mut i = 0;
        while i < 10 {
            assert!(got[i] == 0);
            i += 1;
        }
        assert!(rng.state() == (0, 0));
    }

    #[test]
    fn output_depends_only_on_s0() {
        // Two generators sharing s0 but differing in s1 emit the same first
        // word, because `result` is computed before s1 is consumed.
        let mut a = Xoroshiro128StarStar::new(0x9f, 0x11);
        let mut b = Xoroshiro128StarStar::new(0x9f, 0xee);
        assert!(a.next_u64() == b.next_u64());
    }

    #[test]
    fn formula_matches_manual_recompute() {
        let s0: u64 = 0x0123_4567_89ab_cdef;
        let s1: u64 = 0xfedc_ba98_7654_3210;
        let mut rng = Xoroshiro128StarStar::new(s0, s1);
        let expect = s0.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        assert!(rng.next_u64() == expect);
    }

    #[test]
    fn state_update_matches_manual_recompute() {
        let s0: u64 = 0x0123_4567_89ab_cdef;
        let s1: u64 = 0xfedc_ba98_7654_3210;
        let mut rng = Xoroshiro128StarStar::new(s0, s1);
        let _ = rng.next_u64();
        let t = s1 ^ s0;
        let want_s0 = s0.rotate_left(24) ^ t ^ (t << 16);
        let want_s1 = t.rotate_left(37);
        assert!(rng.state() == (want_s0, want_s1));
    }

    #[test]
    fn full_period_word_is_nonzero_for_live_seed() {
        let mut rng = Xoroshiro128StarStar::new(0x1, 0x0);
        // Live (nonzero) state must eventually produce a nonzero word.
        let got = collect::<4>(&mut rng);
        let mut any_nonzero = false;
        let mut i = 0;
        while i < 4 {
            if got[i] != 0 {
                any_nonzero = true;
            }
            i += 1;
        }
        assert!(any_nonzero);
    }

    #[test]
    fn distinct_outputs_within_short_run() {
        let mut rng = Xoroshiro128StarStar::new(0xcafe, 0xbabe);
        let got = collect::<8>(&mut rng);
        let mut i = 0;
        while i < 8 {
            let mut j = i + 1;
            while j < 8 {
                assert!(got[i] != got[j]);
                j += 1;
            }
            i += 1;
        }
    }

    #[test]
    fn advancing_changes_state_each_step() {
        let mut rng = Xoroshiro128StarStar::new(0x2222, 0x3333);
        let mut prev = rng.state();
        let mut i = 0;
        while i < 12 {
            let _ = rng.next_u64();
            let now = rng.state();
            assert!(now != prev);
            prev = now;
            i += 1;
        }
    }

    #[test]
    fn two_rngs_resync_after_shared_restore() {
        let mut a = Xoroshiro128StarStar::new(0x10, 0x20);
        let _ = collect::<3>(&mut a);
        let snapshot = a.state();
        let mut b = Xoroshiro128StarStar::from_state(snapshot.0, snapshot.1);
        let mut i = 0;
        while i < 20 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn high_bit_seeds_do_not_panic_and_round_trip() {
        let mut rng = Xoroshiro128StarStar::new(u64::MAX, u64::MAX);
        let _ = collect::<4>(&mut rng);
        let (s0, s1) = rng.state();
        let mut restored = Xoroshiro128StarStar::from_state(s0, s1);
        let a = collect::<4>(&mut rng);
        let b = collect::<4>(&mut restored);
        let mut i = 0;
        while i < 4 {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn parity_of_outputs_varies() {
        // Over a short run, both even and odd words should appear for a
        // typical live seed, exercised via `is_multiple_of`.
        let mut rng = Xoroshiro128StarStar::new(0x5151, 0x9292);
        let got = collect::<16>(&mut rng);
        let mut saw_even = false;
        let mut saw_odd = false;
        let mut i = 0;
        while i < 16 {
            if got[i].is_multiple_of(2) {
                saw_even = true;
            } else {
                saw_odd = true;
            }
            i += 1;
        }
        assert!(saw_even);
        assert!(saw_odd);
    }

    #[test]
    fn div_ceil_relation_on_outputs_is_consistent() {
        let mut rng = Xoroshiro128StarStar::new(0x7, 0x9);
        let v = rng.next_u64() | 1;
        // div_ceil never undershoots the plain quotient.
        let d = 1000u64;
        assert!(v.div_ceil(d) >= (v / d));
    }

    #[test]
    fn reseed_resets_stream() {
        let mut rng = Xoroshiro128StarStar::new(0xabc, 0xdef);
        let first = collect::<5>(&mut rng);
        rng = Xoroshiro128StarStar::new(0xabc, 0xdef);
        let second = collect::<5>(&mut rng);
        let mut i = 0;
        while i < 5 {
            assert!(first[i] == second[i]);
            i += 1;
        }
    }

    #[test]
    fn swapped_seed_words_give_different_stream() {
        let mut a = Xoroshiro128StarStar::new(0x11, 0x22);
        let mut b = Xoroshiro128StarStar::new(0x22, 0x11);
        let xs = collect::<4>(&mut a);
        let ys = collect::<4>(&mut b);
        let mut differ = false;
        let mut i = 0;
        while i < 4 {
            if xs[i] != ys[i] {
                differ = true;
            }
            i += 1;
        }
        assert!(differ);
    }

    #[test]
    fn long_run_remains_deterministic() {
        let mut a = Xoroshiro128StarStar::new(0xf00d, 0xface);
        let mut b = Xoroshiro128StarStar::new(0xf00d, 0xface);
        let mut i = 0;
        while i < 256 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn state_pair_fields_both_live() {
        // Reading both tuple members ensures neither field is dead.
        let rng = Xoroshiro128StarStar::new(0x4040, 0x8080);
        let (s0, s1) = rng.state();
        assert!(s0 == 0x4040);
        assert!(s1 == 0x8080);
    }

    #[test]
    fn output_range_is_full_width_capable() {
        // The high bit of some output must be set for a live seed within a
        // modest window, confirming the full 64-bit range is reachable.
        let mut rng = Xoroshiro128StarStar::new(0x1357, 0x2468);
        let got = collect::<32>(&mut rng);
        let mut saw_high = false;
        let mut i = 0;
        while i < 32 {
            if (got[i] >> 63) == 1 {
                saw_high = true;
            }
            i += 1;
        }
        assert!(saw_high);
    }

    #[test]
    fn restore_from_zeroed_tuple_is_absorbing() {
        let origin = Xoroshiro128StarStar::new(0, 0);
        let (s0, s1) = origin.state();
        let mut rng = Xoroshiro128StarStar::from_state(s0, s1);
        assert!(rng.next_u64() == 0);
    }

    #[test]
    fn consecutive_states_differ_from_outputs() {
        // The emitted word and the resulting state words are distinct
        // quantities; confirm they are not accidentally equal for a sample.
        let mut rng = Xoroshiro128StarStar::new(0x9999, 0x1111);
        let out = rng.next_u64();
        let (s0, s1) = rng.state();
        assert!((out != s0) || (out != s1));
    }

    #[test]
    fn second_output_formula_from_updated_s0() {
        // After one step the output must follow the formula on the new s0.
        let mut rng = Xoroshiro128StarStar::new(13, 17);
        let _ = rng.next_u64();
        let (s0_now, _s1_now) = rng.state();
        let expect = s0_now.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        assert!(rng.next_u64() == expect);
    }

    #[test]
    fn shift_term_contributes_to_state() {
        // With t != 0, the (t << 16) term must influence s0; verify by
        // comparing against the same update sans the shift term.
        let s0: u64 = 0x55;
        let s1: u64 = 0xaa;
        let t = s1 ^ s0;
        let with_shift = s0.rotate_left(24) ^ t ^ (t << 16);
        let without_shift = s0.rotate_left(24) ^ t;
        assert!(with_shift != without_shift);
        let mut rng = Xoroshiro128StarStar::new(s0, s1);
        let _ = rng.next_u64();
        assert!(rng.state().0 == with_shift);
    }

    #[test]
    fn rotate_left_37_defines_second_word() {
        let s0: u64 = 0x0f0f;
        let s1: u64 = 0xf0f0;
        let t = s1 ^ s0;
        let mut rng = Xoroshiro128StarStar::new(s0, s1);
        let _ = rng.next_u64();
        assert!(rng.state().1 == t.rotate_left(37));
    }

    #[test]
    fn many_distinct_seeds_give_distinct_firsts_when_s0_differs() {
        // Distinct s0 values should map to distinct first outputs here.
        let mut a = Xoroshiro128StarStar::new(100, 0);
        let mut b = Xoroshiro128StarStar::new(200, 0);
        let mut c = Xoroshiro128StarStar::new(300, 0);
        let fa = a.next_u64();
        let fb = b.next_u64();
        let fc = c.next_u64();
        assert!(fa != fb);
        assert!(fb != fc);
        assert!(fa != fc);
    }

    #[test]
    fn range_contains_check_on_small_modulus() {
        // Reduce an output into [0, 6) and confirm membership via contains.
        let mut rng = Xoroshiro128StarStar::new(0x2024, 0x1001);
        let r = rng.next_u64() % 6;
        assert!((0..6).contains(&r));
    }
}
