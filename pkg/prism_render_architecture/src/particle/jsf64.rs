//! `jsf64`: a 64-bit variant of Bob `Jenkins`' Small Fast `PRNG` (the
//! three-rotate `7`/`13`/`37` design).
//!
//! This is a tiny, deterministic, integer-only pseudo-random number
//! generator (`RNG`) intended for reproducible particle work on both the
//! `CPU` and the `GPU` side of the pipeline. It is `no_std` friendly: it
//! depends only on `core` primitives and never touches floating point,
//! transcendental functions, heap allocation, or formatting.
//!
//! The algorithm keeps a four-word state `(a, b, c, d)`. Seeding sets
//! `a = 0xf1ea5eed` and `b = c = d = seed`, then discards `20` outputs so
//! the state is well mixed before the first value is observed. Each step
//! performs three `.rotate_left(..)` operations and returns `d`.
//!
//! All additions, subtractions, and multiplications use the `wrapping_*`
//! family so behavior is identical regardless of overflow checks.

/// The 64-bit `jsf64` generator state.
///
/// Fields are public so callers can inspect or serialize the raw state;
/// [`Jsf64::state`] and [`Jsf64::from_state`] offer the same access in a
/// tuple form that is convenient for round-tripping.
#[derive(Clone, Copy, Debug)]
pub struct Jsf64 {
    /// First state word.
    pub a: u64,
    /// Second state word.
    pub b: u64,
    /// Third state word.
    pub c: u64,
    /// Fourth state word; also the value most recently returned.
    pub d: u64,
}

impl Jsf64 {
    /// Number of outputs discarded during seeding to warm up the state.
    const WARMUP_ROUNDS: u32 = 20;

    /// Creates a new generator from `seed`, applying the standard warmup.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            a: 0xf1ea5eed,
            b: seed,
            c: seed,
            d: seed,
        };
        let mut round = 0u32;
        while round < Self::WARMUP_ROUNDS {
            let _ = rng.next_u64();
            round = round.wrapping_add(1);
        }
        rng
    }

    /// Advances the generator one step and returns the next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let e = self.a.wrapping_sub(self.b.rotate_left(7));
        self.a = self.b ^ (self.c.rotate_left(13));
        self.b = self.c.wrapping_add(self.d.rotate_left(37));
        self.c = self.d.wrapping_add(e);
        self.d = e.wrapping_add(self.a);
        self.d
    }

    /// Returns the raw state words as `(a, b, c, d)`.
    #[must_use]
    pub fn state(&self) -> (u64, u64, u64, u64) {
        (self.a, self.b, self.c, self.d)
    }

    /// Rebuilds a generator from previously captured raw state words.
    ///
    /// No warmup is applied: the state is used exactly as given, so this
    /// is the inverse of [`Jsf64::state`].
    #[must_use]
    pub fn from_state(a: u64, b: u64, c: u64, d: u64) -> Self {
        Self { a, b, c, d }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seed used by the hard reference vectors.
    const ANCHOR_SEED: u64 = 0x0123456789abcdef;

    /// Collects the first `N` outputs for `seed` into a fixed-size array.
    fn first_n<const N: usize>(seed: u64) -> [u64; N] {
        let mut rng = Jsf64::new(seed);
        let mut out = [0u64; N];
        let mut i = 0usize;
        while i < N {
            out[i] = rng.next_u64();
            i = i.wrapping_add(1);
        }
        out
    }

    /// Advances `rng` by `steps` discarded outputs.
    fn skip(rng: &mut Jsf64, steps: u32) {
        let mut i = 0u32;
        while i < steps {
            let _ = rng.next_u64();
            i = i.wrapping_add(1);
        }
    }

    #[test]
    fn anchor_out0() {
        let out = first_n::<4>(ANCHOR_SEED);
        assert!(out[0] == 0x43526f6e3ac54b42);
    }

    #[test]
    fn anchor_out1() {
        let out = first_n::<4>(ANCHOR_SEED);
        assert!(out[1] == 0xbff8927dcd72b566);
    }

    #[test]
    fn anchor_out2() {
        let out = first_n::<4>(ANCHOR_SEED);
        assert!(out[2] == 0x59eb2a58286546bc);
    }

    #[test]
    fn anchor_out3() {
        let out = first_n::<4>(ANCHOR_SEED);
        assert!(out[3] == 0x0b10af5193921ac7);
    }

    #[test]
    fn anchor_all_four() {
        let out = first_n::<4>(ANCHOR_SEED);
        let want: [u64; 4] = [
            0x43526f6e3ac54b42,
            0xbff8927dcd72b566,
            0x59eb2a58286546bc,
            0x0b10af5193921ac7,
        ];
        assert!(out == want);
    }

    #[test]
    fn anchor_state_tuple() {
        let mut rng = Jsf64::new(ANCHOR_SEED);
        skip(&mut rng, 4);
        let want = (
            0x2102631a9c3bc72d,
            0x4716bed09da6c33f,
            0x43f9768f1fbb9a56,
            0x0b10af5193921ac7,
        );
        assert!(rng.state() == want);
    }

    #[test]
    fn anchor_state_a() {
        let mut rng = Jsf64::new(ANCHOR_SEED);
        skip(&mut rng, 4);
        assert!(rng.a == 0x2102631a9c3bc72d);
    }

    #[test]
    fn anchor_state_b() {
        let mut rng = Jsf64::new(ANCHOR_SEED);
        skip(&mut rng, 4);
        assert!(rng.b == 0x4716bed09da6c33f);
    }

    #[test]
    fn anchor_state_c() {
        let mut rng = Jsf64::new(ANCHOR_SEED);
        skip(&mut rng, 4);
        assert!(rng.c == 0x43f9768f1fbb9a56);
    }

    #[test]
    fn anchor_state_d() {
        let mut rng = Jsf64::new(ANCHOR_SEED);
        skip(&mut rng, 4);
        assert!(rng.d == 0x0b10af5193921ac7);
    }

    #[test]
    fn anchor_state_d_matches_last_output() {
        let out = first_n::<4>(ANCHOR_SEED);
        let mut rng = Jsf64::new(ANCHOR_SEED);
        skip(&mut rng, 4);
        assert!(rng.d == out[3]);
    }

    #[test]
    fn determinism_two_instances() {
        let a = first_n::<16>(0xdead_beef_cafe_f00d);
        let b = first_n::<16>(0xdead_beef_cafe_f00d);
        assert!(a == b);
    }

    #[test]
    fn determinism_long_sequence() {
        let mut x = Jsf64::new(777);
        let mut y = Jsf64::new(777);
        let mut i = 0u32;
        let mut equal = true;
        while i < 256 {
            if x.next_u64() != y.next_u64() {
                equal = false;
            }
            i = i.wrapping_add(1);
        }
        assert!(equal);
    }

    #[test]
    fn different_seed_diverges() {
        let a = first_n::<8>(1);
        let b = first_n::<8>(2);
        assert!(a != b);
    }

    #[test]
    fn adjacent_seeds_diverge() {
        let a = first_n::<8>(0x1000);
        let b = first_n::<8>(0x1001);
        assert!(a != b);
    }

    #[test]
    fn state_roundtrip_continues() {
        let mut original = Jsf64::new(0x5555_aaaa_5555_aaaa);
        skip(&mut original, 10);
        let (a, b, c, d) = original.state();
        let mut restored = Jsf64::from_state(a, b, c, d);
        let mut i = 0u32;
        let mut equal = true;
        while i < 32 {
            if original.next_u64() != restored.next_u64() {
                equal = false;
            }
            i = i.wrapping_add(1);
        }
        assert!(equal);
    }

    #[test]
    fn from_state_tuple_roundtrip() {
        let mut rng = Jsf64::new(42);
        skip(&mut rng, 7);
        let s = rng.state();
        let rebuilt = Jsf64::from_state(s.0, s.1, s.2, s.3);
        assert!(rebuilt.state() == s);
    }

    #[test]
    fn from_state_sets_all_fields() {
        let rng = Jsf64::from_state(1, 2, 3, 4);
        assert!(rng.a == 1 && rng.b == 2 && rng.c == 3 && rng.d == 4);
    }

    #[test]
    fn warmup_changes_first_output() {
        let warmed = Jsf64::new(ANCHOR_SEED).next_u64();
        let mut raw = Jsf64::from_state(0xf1ea5eed, ANCHOR_SEED, ANCHOR_SEED, ANCHOR_SEED);
        let raw_first = raw.next_u64();
        assert!(warmed != raw_first);
    }

    #[test]
    fn warmup_changes_state() {
        let warmed = Jsf64::new(ANCHOR_SEED).state();
        let raw = Jsf64::from_state(0xf1ea5eed, ANCHOR_SEED, ANCHOR_SEED, ANCHOR_SEED).state();
        assert!(warmed != raw);
    }

    #[test]
    fn nth_step_reproducible() {
        let mut a = Jsf64::new(0xabc);
        skip(&mut a, 100);
        let va = a.next_u64();
        let mut b = Jsf64::new(0xabc);
        skip(&mut b, 100);
        let vb = b.next_u64();
        assert!(va == vb);
    }

    #[test]
    fn captured_state_reproduces_future() {
        let mut single = Jsf64::new(0x9999);
        skip(&mut single, 20);
        let tail = single.next_u64();

        let mut split = Jsf64::new(0x9999);
        skip(&mut split, 10);
        let (a, b, c, d) = split.state();
        let mut resumed = Jsf64::from_state(a, b, c, d);
        skip(&mut resumed, 10);
        let resumed_tail = resumed.next_u64();

        assert!(tail == resumed_tail);
    }

    #[test]
    fn seed_zero_deterministic() {
        let a = first_n::<8>(0);
        let b = first_n::<8>(0);
        assert!(a == b);
    }

    #[test]
    fn seed_max_deterministic() {
        let a = first_n::<8>(u64::MAX);
        let b = first_n::<8>(u64::MAX);
        assert!(a == b);
    }

    #[test]
    fn seed_one_deterministic() {
        let a = first_n::<8>(1);
        let b = first_n::<8>(1);
        assert!(a == b);
    }

    #[test]
    fn seed_zero_and_max_diverge() {
        let a = first_n::<8>(0);
        let b = first_n::<8>(u64::MAX);
        assert!(a != b);
    }

    #[test]
    fn next_u64_advances_state() {
        let mut rng = Jsf64::new(0x1234);
        let before = rng.state();
        let _ = rng.next_u64();
        let after = rng.state();
        assert!(before != after);
    }

    #[test]
    fn next_u64_returns_d() {
        let mut rng = Jsf64::new(0x4321);
        let value = rng.next_u64();
        assert!(value == rng.d);
    }

    #[test]
    fn state_tuple_matches_fields() {
        let mut rng = Jsf64::new(0x0f0f);
        skip(&mut rng, 3);
        let (a, b, c, d) = rng.state();
        assert!(a == rng.a && b == rng.b && c == rng.c && d == rng.d);
    }

    #[test]
    fn copy_is_independent() {
        let mut primary = Jsf64::new(0x2468);
        let mut snapshot = primary;
        let _ = primary.next_u64();
        let from_snapshot = snapshot.next_u64();
        let fresh = Jsf64::new(0x2468).next_u64();
        assert!(from_snapshot == fresh);
    }

    #[test]
    fn clone_is_independent() {
        let primary = Jsf64::new(0x1357);
        let mut copy_a = primary;
        let mut copy_b = primary.clone();
        let mut i = 0u32;
        let mut equal = true;
        while i < 32 {
            if copy_a.next_u64() != copy_b.next_u64() {
                equal = false;
            }
            i = i.wrapping_add(1);
        }
        assert!(equal);
    }

    #[test]
    fn same_seed_equal_state_after_new() {
        let x = Jsf64::new(0xfeed_face);
        let y = Jsf64::new(0xfeed_face);
        assert!(x.state() == y.state());
    }

    #[test]
    fn different_seed_state_differs_after_new() {
        let x = Jsf64::new(0xfeed_face);
        let y = Jsf64::new(0xdead_beef);
        assert!(x.state() != y.state());
    }

    #[test]
    fn prefix_is_stable_across_lengths() {
        let short = first_n::<4>(0x55aa);
        let long = first_n::<64>(0x55aa);
        let mut i = 0usize;
        let mut equal = true;
        while i < 4 {
            if short[i] != long[i] {
                equal = false;
            }
            i = i.wrapping_add(1);
        }
        assert!(equal);
    }

    #[test]
    fn consecutive_outputs_differ_for_anchor() {
        let out = first_n::<4>(ANCHOR_SEED);
        assert!(out[0] != out[1] && out[1] != out[2] && out[2] != out[3]);
    }

    #[test]
    fn large_skip_reproducible() {
        let mut a = Jsf64::new(0xc0ffee);
        skip(&mut a, 1000);
        let va = a.next_u64();
        let mut b = Jsf64::new(0xc0ffee);
        skip(&mut b, 1000);
        let vb = b.next_u64();
        assert!(va == vb);
    }

    #[test]
    fn alternating_bit_seed_deterministic() {
        let a = first_n::<8>(0xaaaa_aaaa_aaaa_aaaa);
        let b = first_n::<8>(0xaaaa_aaaa_aaaa_aaaa);
        assert!(a == b);
    }

    #[test]
    fn high_bit_seed_deterministic() {
        let a = first_n::<8>(0x8000_0000_0000_0000);
        let b = first_n::<8>(0x8000_0000_0000_0000);
        assert!(a == b);
    }

    #[test]
    fn sequence_has_distinct_values() {
        let out = first_n::<32>(0x13579bdf);
        let mut i = 0usize;
        let mut collisions = 0u32;
        while i < out.len() {
            let mut j = i.wrapping_add(1);
            while j < out.len() {
                if out[i] == out[j] {
                    collisions = collisions.wrapping_add(1);
                }
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(1);
        }
        assert!(collisions == 0);
    }

    #[test]
    fn from_state_matches_running_instance() {
        let mut live = Jsf64::new(0x2020);
        skip(&mut live, 5);
        let (a, b, c, d) = live.state();
        let mut rebuilt = Jsf64::from_state(a, b, c, d);
        assert!(live.next_u64() == rebuilt.next_u64());
    }

    #[test]
    fn independent_instances_do_not_interfere() {
        let mut x = Jsf64::new(11);
        let mut y = Jsf64::new(22);
        let _ = x.next_u64();
        let _ = x.next_u64();
        let y_val = y.next_u64();
        let fresh_y = Jsf64::new(22).next_u64();
        assert!(y_val == fresh_y);
    }

    #[test]
    fn fifth_output_follows_anchor_prefix() {
        let five = first_n::<5>(ANCHOR_SEED);
        let four = first_n::<4>(ANCHOR_SEED);
        assert!(five[0] == four[0] && five[3] == four[3]);
    }

    #[test]
    fn seed_two_versus_three_diverge() {
        let a = first_n::<8>(2);
        let b = first_n::<8>(3);
        assert!(a != b);
    }

    #[test]
    fn warmup_round_count_is_twenty() {
        let via_new = Jsf64::new(0x7777).state();
        let mut manual = Jsf64::from_state(0xf1ea5eed, 0x7777, 0x7777, 0x7777);
        skip(&mut manual, 20);
        assert!(via_new == manual.state());
    }
}
