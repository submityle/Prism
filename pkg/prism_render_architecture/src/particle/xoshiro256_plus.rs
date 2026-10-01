//! `xoshiro256+` pseudo-random number generator (Blackman-Vigna).
//!
//! This module provides a deterministic 64-bit `PRNG` built on the
//! `xoshiro256+` scheme with a four-word state. It is written to be
//! `no_std` + `alloc` friendly: it performs only integer operations
//! (wrapping addition, shifts, exclusive-or, and bit rotation) and does
//! not use floating point, transcendental functions, heap containers,
//! or formatting machinery.
//!
//! The generator is suitable as a fast, high-quality source of random
//! bits on both `CPU` and `GPU`-adjacent pipelines where reproducible,
//! platform-independent streams are required.

/// A `xoshiro256+` random number generator (`RNG`).
///
/// The internal state is four 64-bit words `(s0, s1, s2, s3)`. Advancing
/// the generator with [`Xoshiro256Plus::next_u64`] mutates the state and
/// returns the next output word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro256Plus {
    /// The four-word internal state.
    s: [u64; 4],
}

impl Xoshiro256Plus {
    /// Construct a generator from an explicit four-word state.
    ///
    /// Note that the all-zero state is a degenerate fixed point: it only
    /// ever produces zero output and never leaves the zero state.
    #[must_use]
    pub fn from_state(state: [u64; 4]) -> Self {
        Self { s: state }
    }

    /// Return a copy of the current four-word internal state.
    #[must_use]
    pub fn state(&self) -> [u64; 4] {
        self.s
    }

    /// Advance the generator and return the next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let s0 = self.s[0];
        let s1 = self.s[1];
        let s2 = self.s[2];
        let s3 = self.s[3];

        let result = s0.wrapping_add(s3);

        let t = s1 << 17;

        let mut n2 = s2 ^ s0;
        let n3 = s3 ^ s1;
        let n1 = s1 ^ n2;
        let n0 = s0 ^ n3;
        n2 ^= t;
        let n3 = n3.rotate_left(45);

        self.s[0] = n0;
        self.s[1] = n1;
        self.s[2] = n2;
        self.s[3] = n3;

        result
    }
}

#[cfg(test)]
mod tests {
    use super::Xoshiro256Plus;

    // ---- Anchor / ground-truth vectors ----

    #[test]
    fn anchor_output_0() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let out0 = rng.next_u64();
        assert!(out0 == 0x5);
    }

    #[test]
    fn anchor_output_1() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let _ = rng.next_u64();
        let out1 = rng.next_u64();
        assert!(out1 == 0xc00000000007);
    }

    #[test]
    fn anchor_output_2() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let out2 = rng.next_u64();
        assert!(out2 == 0xc00018000007);
    }

    #[test]
    fn anchor_output_3() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        let out3 = rng.next_u64();
        assert!(out3 == 0x8001600018040302);
    }

    #[test]
    fn anchor_all_four_outputs() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let out: [u64; 4] = [
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];
        let expected: [u64; 4] = [0x5, 0xc00000000007, 0xc00018000007, 0x8001600018040302];
        assert!(out[0] == expected[0]);
        assert!(out[1] == expected[1]);
        assert!(out[2] == expected[2]);
        assert!(out[3] == expected[3]);
    }

    #[test]
    fn anchor_state_after_four_outputs() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        let st = rng.state();
        let expected: [u64; 4] = [
            0x8000a00018040305,
            0x0000c008180a0007,
            0x8000000818040000,
            0x0060f0000c000000,
        ];
        assert!(st[0] == expected[0]);
        assert!(st[1] == expected[1]);
        assert!(st[2] == expected[2]);
        assert!(st[3] == expected[3]);
    }

    #[test]
    fn anchor_state_word0() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[0] == 0x8000a00018040305);
    }

    #[test]
    fn anchor_state_word1() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[1] == 0x0000c008180a0007);
    }

    #[test]
    fn anchor_state_word2() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[2] == 0x8000000818040000);
    }

    #[test]
    fn anchor_state_word3() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert!(rng.state()[3] == 0x0060f0000c000000);
    }

    // ---- First-step algebra checks ----

    #[test]
    fn first_output_is_s0_plus_s3() {
        let mut rng = Xoshiro256Plus::from_state([10, 20, 30, 40]);
        let out = rng.next_u64();
        assert!(out == 50);
    }

    #[test]
    fn first_output_wraps_on_overflow() {
        let mut rng = Xoshiro256Plus::from_state([u64::MAX, 0, 0, 1]);
        let out = rng.next_u64();
        assert!(out == 0);
    }

    #[test]
    fn first_output_wraps_near_max() {
        let mut rng = Xoshiro256Plus::from_state([u64::MAX, 0, 0, 2]);
        let out = rng.next_u64();
        assert!(out == 1);
    }

    // ---- Determinism ----

    #[test]
    fn deterministic_same_seed_same_stream() {
        let mut a = Xoshiro256Plus::from_state([111, 222, 333, 444]);
        let mut b = Xoshiro256Plus::from_state([111, 222, 333, 444]);
        for _ in 0..64 {
            let x = a.next_u64();
            let y = b.next_u64();
            assert!(x == y);
        }
    }

    #[test]
    fn deterministic_state_matches() {
        let mut a = Xoshiro256Plus::from_state([7, 8, 9, 10]);
        let mut b = Xoshiro256Plus::from_state([7, 8, 9, 10]);
        for _ in 0..100 {
            let _ = a.next_u64();
            let _ = b.next_u64();
        }
        let sa = a.state();
        let sb = b.state();
        assert!(sa[0] == sb[0]);
        assert!(sa[1] == sb[1]);
        assert!(sa[2] == sb[2]);
        assert!(sa[3] == sb[3]);
    }

    #[test]
    fn deterministic_restart_reproduces() {
        let mut a = Xoshiro256Plus::from_state([42, 43, 44, 45]);
        let mut first: [u64; 16] = [0; 16];
        for slot in first.iter_mut() {
            *slot = a.next_u64();
        }
        let mut b = Xoshiro256Plus::from_state([42, 43, 44, 45]);
        for slot in first.iter() {
            let v = b.next_u64();
            assert!(v == *slot);
        }
    }

    // ---- Different seeds diverge ----

    #[test]
    fn different_seed_diverges() {
        let mut a = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let mut b = Xoshiro256Plus::from_state([4, 3, 2, 1]);
        let mut differ = false;
        for _ in 0..16 {
            let x = a.next_u64();
            let y = b.next_u64();
            if x != y {
                differ = true;
            }
        }
        assert!(differ);
    }

    #[test]
    fn different_seed_single_bit_diverges() {
        let mut a = Xoshiro256Plus::from_state([0, 0, 0, 1]);
        let mut b = Xoshiro256Plus::from_state([0, 0, 0, 2]);
        let mut differ = false;
        for _ in 0..16 {
            if a.next_u64() != b.next_u64() {
                differ = true;
            }
        }
        assert!(differ);
    }

    #[test]
    fn different_seed_state_diverges() {
        let mut a = Xoshiro256Plus::from_state([5, 6, 7, 8]);
        let mut b = Xoshiro256Plus::from_state([5, 6, 7, 9]);
        for _ in 0..32 {
            let _ = a.next_u64();
            let _ = b.next_u64();
        }
        let sa = a.state();
        let sb = b.state();
        let same = (sa[0] == sb[0]) && (sa[1] == sb[1]) && (sa[2] == sb[2]) && (sa[3] == sb[3]);
        assert!(!same);
    }

    // ---- state / from_state round-trip ----

    #[test]
    fn state_roundtrip_preserves() {
        let seed: [u64; 4] = [0xdead, 0xbeef, 0xcafe, 0xf00d];
        let rng = Xoshiro256Plus::from_state(seed);
        let st = rng.state();
        assert!(st[0] == seed[0]);
        assert!(st[1] == seed[1]);
        assert!(st[2] == seed[2]);
        assert!(st[3] == seed[3]);
    }

    #[test]
    fn state_roundtrip_after_steps() {
        let mut a = Xoshiro256Plus::from_state([123, 456, 789, 101112]);
        for _ in 0..50 {
            let _ = a.next_u64();
        }
        let captured = a.state();
        let mut b = Xoshiro256Plus::from_state(captured);
        let x = a.next_u64();
        let y = b.next_u64();
        assert!(x == y);
    }

    #[test]
    fn state_roundtrip_continues_stream() {
        let mut a = Xoshiro256Plus::from_state([9, 9, 9, 9]);
        for _ in 0..10 {
            let _ = a.next_u64();
        }
        let snapshot = a.state();
        let mut tail_a: [u64; 8] = [0; 8];
        for slot in tail_a.iter_mut() {
            *slot = a.next_u64();
        }
        let mut b = Xoshiro256Plus::from_state(snapshot);
        for slot in tail_a.iter() {
            let v = b.next_u64();
            assert!(v == *slot);
        }
    }

    #[test]
    fn clone_is_independent() {
        let mut a = Xoshiro256Plus::from_state([2, 4, 6, 8]);
        let mut b = a;
        let x = a.next_u64();
        let y = b.next_u64();
        assert!(x == y);
        let _ = a.next_u64();
        // b has not advanced the second time; its next must match a's
        // original second output when driven from a fresh clone.
        let mut c = Xoshiro256Plus::from_state([2, 4, 6, 8]);
        let _ = c.next_u64();
        let c2 = c.next_u64();
        let b2 = b.next_u64();
        assert!(b2 == c2);
    }

    // ---- N-th step reproduction ----

    #[test]
    fn nth_step_reproducible_step_7() {
        let seed: [u64; 4] = [17, 19, 23, 29];
        let mut a = Xoshiro256Plus::from_state(seed);
        let mut target = 0u64;
        for i in 0..8 {
            let v = a.next_u64();
            if i == 7 {
                target = v;
            }
        }
        let mut b = Xoshiro256Plus::from_state(seed);
        let mut got = 0u64;
        for i in 0..8 {
            let v = b.next_u64();
            if i == 7 {
                got = v;
            }
        }
        assert!(got == target);
    }

    #[test]
    fn nth_step_reproducible_step_31() {
        let seed: [u64; 4] = [100, 200, 300, 400];
        let mut a = Xoshiro256Plus::from_state(seed);
        let mut first = 0u64;
        for i in 0..32 {
            let v = a.next_u64();
            if i == 31 {
                first = v;
            }
        }
        let mut b = Xoshiro256Plus::from_state(seed);
        let mut second = 0u64;
        for i in 0..32 {
            let v = b.next_u64();
            if i == 31 {
                second = v;
            }
        }
        assert!(first == second);
    }

    #[test]
    fn nth_step_reproducible_step_63() {
        let seed: [u64; 4] = [0xaa, 0xbb, 0xcc, 0xdd];
        let mut a = Xoshiro256Plus::from_state(seed);
        let mut first = 0u64;
        for i in 0..64 {
            let v = a.next_u64();
            if i == 63 {
                first = v;
            }
        }
        let mut b = Xoshiro256Plus::from_state(seed);
        let mut second = 0u64;
        for i in 0..64 {
            let v = b.next_u64();
            if i == 63 {
                second = v;
            }
        }
        assert!(first == second);
    }

    // ---- All-zero degenerate fixed point ----

    #[test]
    fn zero_state_outputs_zero() {
        let mut rng = Xoshiro256Plus::from_state([0, 0, 0, 0]);
        let out = rng.next_u64();
        assert!(out == 0);
    }

    #[test]
    fn zero_state_outputs_zero_repeatedly() {
        let mut rng = Xoshiro256Plus::from_state([0, 0, 0, 0]);
        for _ in 0..128 {
            let out = rng.next_u64();
            assert!(out == 0);
        }
    }

    #[test]
    fn zero_state_remains_zero() {
        let mut rng = Xoshiro256Plus::from_state([0, 0, 0, 0]);
        for _ in 0..64 {
            let _ = rng.next_u64();
        }
        let st = rng.state();
        assert!(st[0] == 0);
        assert!(st[1] == 0);
        assert!(st[2] == 0);
        assert!(st[3] == 0);
    }

    // ---- Non-zero seeds eventually produce non-zero output ----

    #[test]
    fn nonzero_seed_produces_nonzero() {
        let mut rng = Xoshiro256Plus::from_state([1, 0, 0, 0]);
        let mut any_nonzero = false;
        for _ in 0..32 {
            if rng.next_u64() != 0 {
                any_nonzero = true;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn nonzero_seed_single_high_bit() {
        let mut rng = Xoshiro256Plus::from_state([0, 0, 0, 0x8000000000000000]);
        let out = rng.next_u64();
        assert!(out == 0x8000000000000000);
    }

    // ---- Stream distinctness (not stuck on a constant) ----

    #[test]
    fn stream_is_not_constant() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let a = rng.next_u64();
        let mut varied = false;
        for _ in 0..16 {
            if rng.next_u64() != a {
                varied = true;
            }
        }
        assert!(varied);
    }

    #[test]
    fn consecutive_outputs_differ() {
        let mut rng = Xoshiro256Plus::from_state([99, 98, 97, 96]);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert!(a != b);
    }

    // ---- State evolution ----

    #[test]
    fn state_changes_after_step() {
        let seed: [u64; 4] = [11, 22, 33, 44];
        let mut rng = Xoshiro256Plus::from_state(seed);
        let _ = rng.next_u64();
        let st = rng.state();
        let unchanged =
            (st[0] == seed[0]) && (st[1] == seed[1]) && (st[2] == seed[2]) && (st[3] == seed[3]);
        assert!(!unchanged);
    }

    #[test]
    fn equal_rngs_remain_equal() {
        let a = Xoshiro256Plus::from_state([3, 1, 4, 1]);
        let b = Xoshiro256Plus::from_state([3, 1, 4, 1]);
        assert!(a == b);
    }

    #[test]
    fn distinct_rngs_are_unequal() {
        let a = Xoshiro256Plus::from_state([3, 1, 4, 1]);
        let b = Xoshiro256Plus::from_state([5, 9, 2, 6]);
        assert!(a != b);
    }

    // ---- Longer anchored stream snapshot ----

    #[test]
    fn tenth_output_from_canonical_seed() {
        let mut rng = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let mut tenth = 0u64;
        for i in 0..10 {
            let v = rng.next_u64();
            if i == 9 {
                tenth = v;
            }
        }
        // Reproduce independently to confirm stability of step 10.
        let mut other = Xoshiro256Plus::from_state([1, 2, 3, 4]);
        let mut tenth2 = 0u64;
        for i in 0..10 {
            let v = other.next_u64();
            if i == 9 {
                tenth2 = v;
            }
        }
        assert!(tenth == tenth2);
    }

    #[test]
    fn second_output_canonical() {
        let mut rng = Xoshiro256Plus::from_state([10, 20, 30, 40]);
        let _ = rng.next_u64();
        let out = rng.next_u64();
        // Compute expected via an independent fresh run.
        let mut check = Xoshiro256Plus::from_state([10, 20, 30, 40]);
        let _ = check.next_u64();
        let expected = check.next_u64();
        assert!(out == expected);
    }

    #[test]
    fn long_run_matches_fresh_run() {
        let seed: [u64; 4] = [0x1234, 0x5678, 0x9abc, 0xdef0];
        let mut a = Xoshiro256Plus::from_state(seed);
        for _ in 0..256 {
            let _ = a.next_u64();
        }
        let st_a = a.state();
        let mut b = Xoshiro256Plus::from_state(seed);
        for _ in 0..256 {
            let _ = b.next_u64();
        }
        let st_b = b.state();
        assert!(st_a[0] == st_b[0]);
        assert!(st_a[1] == st_b[1]);
        assert!(st_a[2] == st_b[2]);
        assert!(st_a[3] == st_b[3]);
    }
}
