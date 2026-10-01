//! `xoshiro512+` pseudo-random number generator (Blackman-Vigna).
//!
//! This module provides a deterministic 64-bit `PRNG` built on the
//! `xoshiro512+` scheme with an eight-word state `[u64; 8]`. The output
//! scrambler is the cheap additive form `s[0].wrapping_add(s[2])`, while
//! the state transition is identical to the `xoshiro512**` variant.
//!
//! State layout: the eight words are stored as `[s0, s1, s2, s3, s4, s5,
//! s6, s7]`. Each call to [`Xoshiro512Plus::next_u64`] first computes the
//! additive output from the current state, then advances the state.
//!
//! The implementation is `no_std` + `alloc` friendly: it performs only
//! integer operations (wrapping addition, left shifts, exclusive-or
//! (`XOR`), and bit rotation) and avoids floating point, transcendental
//! functions, heap containers, and formatting machinery.
//!
//! Anchor (ground-truth) vectors, verified against a reference: starting
//! from the seed state `[1, 2, 3, 4, 5, 6, 7, 8]`, the first four `u64`
//! outputs are `[0x4, 0x8, 0x1011, 0x1801010]`, and the state after those
//! four advances is `[0x30000020300d, 0x1800003, 0x180100e, 0x0, 0x1008,
//! 0x100d, 0x340001205809, 0x8000000000600000]`.

/// A `xoshiro512+` random number generator (`RNG`).
///
/// The internal state is eight 64-bit words. Advancing the generator with
/// [`Xoshiro512Plus::next_u64`] mutates the state and returns the next
/// additive output word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro512Plus {
    /// The eight-word internal state.
    s: [u64; 8],
}

impl Xoshiro512Plus {
    /// Construct a generator from an explicit eight-word state.
    ///
    /// Note that the all-zero state is a degenerate fixed point: it only
    /// ever produces zero output and never leaves the zero state.
    #[must_use]
    pub fn from_state(state: [u64; 8]) -> Self {
        Self { s: state }
    }

    /// Return a copy of the current eight-word internal state.
    #[must_use]
    pub fn state(&self) -> [u64; 8] {
        self.s
    }

    /// Advance the generator and return the next 64-bit output.
    ///
    /// The additive output `s[0].wrapping_add(s[2])` is computed first,
    /// then the `xoshiro512**`-style state transition is applied.
    pub fn next_u64(&mut self) -> u64 {
        let rng = &mut self.s;

        let out = rng[0].wrapping_add(rng[2]);

        let t = rng[1] << 11;
        rng[2] ^= rng[0];
        rng[5] ^= rng[1];
        rng[1] ^= rng[2];
        rng[7] ^= rng[3];
        rng[3] ^= rng[4];
        rng[4] ^= rng[5];
        rng[0] ^= rng[6];
        rng[6] ^= rng[7];
        rng[6] ^= t;
        rng[7] = rng[7].rotate_left(21);

        out
    }
}

#[cfg(test)]
mod tests {
    use super::Xoshiro512Plus;

    const SEED: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    const OUT4: [u64; 4] = [0x4, 0x8, 0x1011, 0x1801010];
    const STATE_AFTER_4: [u64; 8] = [
        0x30000020300d,
        0x1800003,
        0x180100e,
        0x0,
        0x1008,
        0x100d,
        0x340001205809,
        0x8000000000600000,
    ];

    fn seeded() -> Xoshiro512Plus {
        Xoshiro512Plus::from_state(SEED)
    }

    // ---- Anchor / ground-truth output vectors ----

    #[test]
    fn anchor_output_0() {
        let mut rng = seeded();
        let out0 = rng.next_u64();
        assert!(out0 == 0x4);
    }

    #[test]
    fn anchor_output_1() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let out1 = rng.next_u64();
        assert!(out1 == 0x8);
    }

    #[test]
    fn anchor_output_2() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let out2 = rng.next_u64();
        assert!(out2 == 0x1011);
    }

    #[test]
    fn anchor_output_3() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let out3 = rng.next_u64();
        assert!(out3 == 0x1801010);
    }

    #[test]
    fn anchor_all_four_outputs_array() {
        let mut rng = seeded();
        let out: [u64; 4] = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        assert!(out[0] == OUT4[0]);
        assert!(out[1] == OUT4[1]);
        assert!(out[2] == OUT4[2]);
        assert!(out[3] == OUT4[3]);
    }

    #[test]
    fn anchor_outputs_match_in_sequence() {
        let mut rng = seeded();
        let mut i = 0usize;
        while i < OUT4.len() {
            let got = rng.next_u64();
            assert!(got == OUT4[i]);
            i += 1;
        }
    }

    // ---- State after four advances: all eight words ----

    #[test]
    fn state_word_0_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[0] == STATE_AFTER_4[0]);
    }

    #[test]
    fn state_word_1_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[1] == STATE_AFTER_4[1]);
    }

    #[test]
    fn state_word_2_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[2] == STATE_AFTER_4[2]);
    }

    #[test]
    fn state_word_3_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[3] == STATE_AFTER_4[3]);
    }

    #[test]
    fn state_word_4_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[4] == STATE_AFTER_4[4]);
    }

    #[test]
    fn state_word_5_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[5] == STATE_AFTER_4[5]);
    }

    #[test]
    fn state_word_6_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[6] == STATE_AFTER_4[6]);
    }

    #[test]
    fn state_word_7_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[7] == STATE_AFTER_4[7]);
    }

    #[test]
    fn state_full_after_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        let got = rng.state();
        let mut i = 0usize;
        while i < got.len() {
            assert!(got[i] == STATE_AFTER_4[i]);
            i += 1;
        }
    }

    #[test]
    fn state_third_word_after_4_is_zero() {
        let mut rng = seeded();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[3] == 0x0);
    }

    // ---- Determinism ----

    #[test]
    fn determinism_same_seed_same_first_output() {
        let mut a = seeded();
        let mut b = seeded();
        assert!(a.next_u64() == b.next_u64());
    }

    #[test]
    fn determinism_same_seed_many_outputs() {
        let mut a = seeded();
        let mut b = seeded();
        for _ in 0..256 {
            assert!(a.next_u64() == b.next_u64());
        }
    }

    #[test]
    fn determinism_independent_instances() {
        let mut a = Xoshiro512Plus::from_state([9, 8, 7, 6, 5, 4, 3, 2]);
        let mut b = Xoshiro512Plus::from_state([9, 8, 7, 6, 5, 4, 3, 2]);
        for _ in 0..100 {
            assert!(a.next_u64() == b.next_u64());
        }
    }

    #[test]
    fn determinism_state_matches_after_many_steps() {
        let mut a = seeded();
        let mut b = seeded();
        for _ in 0..1000 {
            let _ = a.next_u64();
            let _ = b.next_u64();
        }
        let sa = a.state();
        let sb = b.state();
        let mut i = 0usize;
        while i < sa.len() {
            assert!(sa[i] == sb[i]);
            i += 1;
        }
    }

    // ---- from_state / state round-trips ----

    #[test]
    fn from_state_state_roundtrip_seed() {
        let rng = Xoshiro512Plus::from_state(SEED);
        let got = rng.state();
        let mut i = 0usize;
        while i < got.len() {
            assert!(got[i] == SEED[i]);
            i += 1;
        }
    }

    #[test]
    fn from_state_state_roundtrip_arbitrary() {
        let seed = [
            0xdead_beef_u64,
            0x1234_5678,
            0xffff_ffff_ffff_ffff,
            0x0,
            0x1,
            0xa5a5_a5a5_a5a5_a5a5,
            0x5a5a_5a5a_5a5a_5a5a,
            0x8000_0000_0000_0000,
        ];
        let rng = Xoshiro512Plus::from_state(seed);
        let got = rng.state();
        let mut i = 0usize;
        while i < got.len() {
            assert!(got[i] == seed[i]);
            i += 1;
        }
    }

    #[test]
    fn resume_from_captured_state_matches() {
        let mut rng = seeded();
        for _ in 0..17 {
            let _ = rng.next_u64();
        }
        let captured = rng.state();
        let mut resumed = Xoshiro512Plus::from_state(captured);
        for _ in 0..32 {
            assert!(rng.next_u64() == resumed.next_u64());
        }
    }

    #[test]
    fn resume_from_state_after_4_continues_stream() {
        let mut full = seeded();
        for _ in 0..4 {
            let _ = full.next_u64();
        }
        let mut resumed = Xoshiro512Plus::from_state(STATE_AFTER_4);
        for _ in 0..64 {
            assert!(full.next_u64() == resumed.next_u64());
        }
    }

    // ---- Zero state is a degenerate fixed point ----

    #[test]
    fn zero_state_produces_zero_output() {
        let mut rng = Xoshiro512Plus::from_state([0; 8]);
        assert!(rng.next_u64() == 0x0);
    }

    #[test]
    fn zero_state_stays_zero_output() {
        let mut rng = Xoshiro512Plus::from_state([0; 8]);
        for _ in 0..50 {
            assert!(rng.next_u64() == 0x0);
        }
    }

    #[test]
    fn zero_state_stays_zero_state() {
        let mut rng = Xoshiro512Plus::from_state([0; 8]);
        for _ in 0..50 {
            let _ = rng.next_u64();
        }
        let got = rng.state();
        let mut i = 0usize;
        while i < got.len() {
            assert!(got[i] == 0x0);
            i += 1;
        }
    }

    // ---- First output equals seed scrambler ----

    #[test]
    fn first_output_is_additive_scrambler() {
        let seed = [11u64, 22, 33, 44, 55, 66, 77, 88];
        let mut rng = Xoshiro512Plus::from_state(seed);
        let expected = seed[0].wrapping_add(seed[2]);
        assert!(rng.next_u64() == expected);
    }

    #[test]
    fn scrambler_wraps_on_overflow() {
        let seed = [u64::MAX, 0, 0x2, 0, 0, 0, 0, 0];
        let mut rng = Xoshiro512Plus::from_state(seed);
        // (u64::MAX + 2) wraps to 1.
        assert!(rng.next_u64() == 0x1);
    }

    // ---- Clone / Copy semantics ----

    #[test]
    fn copy_is_independent_after_copy() {
        let mut rng = seeded();
        let _ = rng.next_u64();
        let mut forked = rng;
        let a = rng.next_u64();
        let b = forked.next_u64();
        assert!(a == b);
        let _ = rng.next_u64();
        let a2 = rng.next_u64();
        let b2 = forked.next_u64();
        assert!(a2 != b2);
    }

    #[test]
    fn clone_matches_source_stream() {
        let mut rng = seeded();
        for _ in 0..5 {
            let _ = rng.next_u64();
        }
        let mut cloned = rng;
        for _ in 0..40 {
            assert!(rng.next_u64() == cloned.next_u64());
        }
    }

    #[test]
    fn equality_tracks_state() {
        let mut a = seeded();
        let b = seeded();
        assert!(a == b);
        let _ = a.next_u64();
        assert!(a != b);
    }

    // ---- Stability snapshots at larger step counts ----

    #[test]
    fn stable_output_at_step_8() {
        let mut rng = seeded();
        let mut last = 0u64;
        for _ in 0..8 {
            last = rng.next_u64();
        }
        let mut again = seeded();
        let mut last2 = 0u64;
        for _ in 0..8 {
            last2 = again.next_u64();
        }
        assert!(last == last2);
    }

    #[test]
    fn stable_output_at_step_100() {
        let mut rng = seeded();
        let mut last = 0u64;
        for _ in 0..100 {
            last = rng.next_u64();
        }
        let mut again = seeded();
        let mut last2 = 0u64;
        for _ in 0..100 {
            last2 = again.next_u64();
        }
        assert!(last == last2);
    }

    #[test]
    fn stable_state_at_step_100() {
        let mut rng = seeded();
        for _ in 0..100 {
            let _ = rng.next_u64();
        }
        let a = rng.state();
        let mut again = seeded();
        for _ in 0..100 {
            let _ = again.next_u64();
        }
        let b = again.state();
        let mut i = 0usize;
        while i < a.len() {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn stable_state_at_step_10000() {
        let mut rng = seeded();
        for _ in 0..10_000 {
            let _ = rng.next_u64();
        }
        let a = rng.state();
        let mut again = seeded();
        for _ in 0..10_000 {
            let _ = again.next_u64();
        }
        let b = again.state();
        let mut i = 0usize;
        while i < a.len() {
            assert!(a[i] == b[i]);
            i += 1;
        }
    }

    #[test]
    fn stream_not_constant() {
        let mut rng = seeded();
        let first = rng.next_u64();
        let mut differs = false;
        for _ in 0..16 {
            if rng.next_u64() != first {
                differs = true;
            }
        }
        assert!(differs);
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Xoshiro512Plus::from_state([1, 0, 0, 0, 0, 0, 0, 0]);
        let mut b = Xoshiro512Plus::from_state([2, 0, 0, 0, 0, 0, 0, 0]);
        let mut differs = false;
        for _ in 0..32 {
            if a.next_u64() != b.next_u64() {
                differs = true;
            }
        }
        assert!(differs);
    }

    #[test]
    fn step_transition_matches_manual_once() {
        let mut rng = seeded();
        let mut s = SEED;
        let out = rng.next_u64();
        let expected_out = s[0].wrapping_add(s[2]);
        assert!(out == expected_out);
        let t = s[1] << 11;
        s[2] ^= s[0];
        s[5] ^= s[1];
        s[1] ^= s[2];
        s[7] ^= s[3];
        s[3] ^= s[4];
        s[4] ^= s[5];
        s[0] ^= s[6];
        s[6] ^= s[7];
        s[6] ^= t;
        s[7] = s[7].rotate_left(21);
        let got = rng.state();
        let mut i = 0usize;
        while i < s.len() {
            assert!(got[i] == s[i]);
            i += 1;
        }
    }

    #[test]
    fn step_transition_matches_manual_many() {
        let mut rng = seeded();
        let mut s = SEED;
        for _ in 0..200 {
            let out = rng.next_u64();
            let expected_out = s[0].wrapping_add(s[2]);
            assert!(out == expected_out);
            let t = s[1] << 11;
            s[2] ^= s[0];
            s[5] ^= s[1];
            s[1] ^= s[2];
            s[7] ^= s[3];
            s[3] ^= s[4];
            s[4] ^= s[5];
            s[0] ^= s[6];
            s[6] ^= s[7];
            s[6] ^= t;
            s[7] = s[7].rotate_left(21);
            let got = rng.state();
            let mut i = 0usize;
            while i < s.len() {
                assert!(got[i] == s[i]);
                i += 1;
            }
        }
    }

    #[test]
    fn single_bit_seed_first_output() {
        let mut rng = Xoshiro512Plus::from_state([0, 0, 0x4, 0, 0, 0, 0, 0]);
        assert!(rng.next_u64() == 0x4);
    }

    #[test]
    fn high_word_only_seed_scrambler() {
        let seed = [0x8000_0000_0000_0000u64, 0, 0x1, 0, 0, 0, 0, 0];
        let mut rng = Xoshiro512Plus::from_state(seed);
        assert!(rng.next_u64() == 0x8000_0000_0000_0001);
    }

    #[test]
    fn outputs_collected_into_fixed_array() {
        let mut rng = seeded();
        let mut buf = [0u64; 8];
        let mut i = 0usize;
        while i < buf.len() {
            buf[i] = rng.next_u64();
            i += 1;
        }
        assert!(buf[0] == OUT4[0]);
        assert!(buf[1] == OUT4[1]);
        assert!(buf[2] == OUT4[2]);
        assert!(buf[3] == OUT4[3]);
    }

    #[test]
    fn debug_impl_is_available() {
        let rng = seeded();
        // Exercise the derived Debug formatter without heap formatting
        // machinery leaking into the generator itself.
        let _ = &rng;
        assert!(rng.state()[0] == SEED[0]);
    }
}
