//! The `xoshiro512++` deterministic pseudo-random number generator (`PRNG`) of
//! David Blackman and Sebastiano Vigna, used for reproducible particle
//! spawning, jitter, and stochastic effects where a very long period and a
//! wide state matter.
//!
//! State layout: this `RNG` keeps a `512`-bit state of eight `u64` words
//! `s[0..8]` and advances it with a linear `XOR`/shift/rotate recurrence, then
//! scrambles its output through a *`plusplus`* (`++`) step. The result of a
//! draw is `s[0].wrapping_add(s[2]).rotate_left(17).wrapping_add(s[2])`; the
//! add-rotate-add scrambler is what distinguishes the `++` variant from the
//! `**` variant, while the underlying linear engine (shared verbatim with
//! `xoshiro512**`) supplies a period of `2^512 - 1`.
//!
//! The state transition (`step`) applied after reading each output word is:
//!
//! ```text
//! t     = s[1] << 11;
//! s[2] ^= s[0];
//! s[5] ^= s[1];
//! s[1] ^= s[2];
//! s[7] ^= s[3];
//! s[3] ^= s[4];
//! s[4] ^= s[5];
//! s[0] ^= s[6];
//! s[6] ^= s[7];
//! s[6] ^= t;
//! s[7]  = s[7].rotate_left(21);
//! ```
//!
//! Each [`Xoshiro512PlusPlus::next_u64`] call first computes the scrambled
//! output from the *current* `s[0]` and `s[2]`, then performs the `step`
//! above, then returns the previously computed output word. Because the `step`
//! is identical to the one in `xoshiro512**`, both variants reach the same
//! state after the same number of draws; only the per-draw scrambler differs.
//!
//! Anchors (reference vectors, verified): seeding the state with
//! `[1, 2, 3, 4, 5, 6, 7, 8]` (so `s[0] = 1 ..= s[7] = 8`) yields four
//! consecutive outputs `[0x80003, 0x100002, 0x20220004, 0x30020201009]`, after
//! which the state is
//! `[0x30000020300d, 0x1800003, 0x180100e, 0x0, 0x1008, 0x100d,
//! 0x340001205809, 0x8000000000600000]`. That state equals the one reached by
//! `xoshiro512**` after four draws, which cross-validates the shared `step`.
//!
//! Every transition uses only integer exclusive-or (`XOR`), fixed shifts,
//! [`u64::rotate_left`], and [`u64::wrapping_add`]. No floating point,
//! division, or transcendental function is involved anywhere in generation.
//!
//! This generator is fast and non-cryptographic. A `xoshiro512++` stream is
//! predictable from a handful of outputs and must never be used for security,
//! key material, or anywhere an adversary could exploit predictability. It
//! exists purely for reproducible, high-throughput simulation randomness. The
//! all-zero state is the degenerate fixed point of the linear recurrence and
//! only ever produces zero, so callers must seed with at least one non-zero
//! word.

/// A `xoshiro512++` generator: eight `u64` state words advanced by the
/// Blackman-Vigna recurrence with an add-rotate-add output scrambler.
///
/// Construct one from an explicit `512`-bit state with
/// [`Xoshiro512PlusPlus::from_state`] and read the current state back with
/// [`Xoshiro512PlusPlus::state`]. Draw successive `u64` words with
/// [`Xoshiro512PlusPlus::next_u64`], or a fixed-size batch with
/// [`Xoshiro512PlusPlus::next_array`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro512PlusPlus {
    /// The eight `u64` state words `s[0..8]` of the `512`-bit linear engine.
    state: [u64; 8],
}

impl Xoshiro512PlusPlus {
    /// Builds a generator directly from an explicit eight-word state.
    ///
    /// The caller is responsible for providing at least one non-zero word; an
    /// all-zero state is the fixed point of the recurrence and yields only
    /// zero outputs forever.
    #[must_use]
    pub const fn from_state(state: [u64; 8]) -> Self {
        Self { state }
    }

    /// Returns a copy of the current eight-word state.
    #[must_use]
    pub const fn state(&self) -> [u64; 8] {
        self.state
    }

    /// Computes the `++` scrambled output word from the current `s[0]` and
    /// `s[2]` without mutating the state.
    #[must_use]
    const fn scramble(&self) -> u64 {
        self.state[0]
            .wrapping_add(self.state[2])
            .rotate_left(17)
            .wrapping_add(self.state[2])
    }

    /// Advances the eight-word linear state by one `step` of the recurrence.
    fn step(&mut self) {
        let s = &mut self.state;
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
    }

    /// Draws the next `u64` from the stream.
    ///
    /// The output is scrambled from the current state, then the linear state
    /// is advanced one `step`, and finally the scrambled word is returned.
    pub fn next_u64(&mut self) -> u64 {
        let out = self.scramble();
        self.step();
        out
    }

    /// Draws `N` successive `u64` words into a fixed-size array.
    ///
    /// This is exactly equivalent to calling [`Xoshiro512PlusPlus::next_u64`]
    /// `N` times and collecting the results in order.
    pub fn next_array<const N: usize>(&mut self) -> [u64; N] {
        core::array::from_fn(|_| self.next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::Xoshiro512PlusPlus;

    /// The canonical seed state used throughout the anchor tests.
    const SEED: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    /// The four reference outputs produced from [`SEED`].
    const FIRST_FOUR: [u64; 4] = [0x80003, 0x100002, 0x20220004, 0x30020201009];

    /// The sixteen reference outputs produced from [`SEED`].
    const FIRST_SIXTEEN: [u64; 16] = [
        0x80003,
        0x100002,
        0x20220004,
        0x30020201009,
        0x6000034081b6100e,
        0x6800354111ae2003,
        0xc81835c0e0c94807,
        0x981a05edb10d630a,
        0xfdae14ed31011b46,
        0x8dae44e7938d9ec5,
        0xb48383b7838fa8ce,
        0x86b173be4055b270,
        0xe6f50c2690e8c7a0,
        0xc9fba1334394d028,
        0xc628b61aa8e79af8,
        0x14a052849017d992,
    ];

    /// The reference state after one draw from [`SEED`].
    const STATE_AFTER_ONE: [u64; 8] = [0x6, 0x0, 0x2, 0x1, 0x1, 0x4, 0x100b, 0x1800000];

    /// The reference state after four draws from [`SEED`].
    const STATE_AFTER_FOUR: [u64; 8] = [
        0x30000020300d,
        0x1800003,
        0x180100e,
        0x0,
        0x1008,
        0x100d,
        0x340001205809,
        0x8000000000600000,
    ];

    /// The reference state after eight draws from [`SEED`].
    const STATE_AFTER_EIGHT: [u64; 8] = [
        0x20320a03800002,
        0x1800c010090100b,
        0x81800c0d00f01806,
        0x340000a04807,
        0x8000300c00403805,
        0x8000340c01405004,
        0x41e07a0a03803008,
        0x400a0400c80804,
    ];

    /// The reference state after one thousand draws from [`SEED`].
    const STATE_AFTER_THOUSAND: [u64; 8] = [
        0x88140c0a0065586b,
        0xdc8de7b4ff76299b,
        0xf629542f402cc05d,
        0xcf30c1500c3d309a,
        0xb93d5a9ae703103b,
        0x76cd53dcfe570d7f,
        0x626c40472afa661f,
        0xcc59a4039d5dec24,
    ];

    /// Builds a generator seeded with the canonical [`SEED`].
    fn seeded() -> Xoshiro512PlusPlus {
        Xoshiro512PlusPlus::from_state(SEED)
    }

    /// Advances `rng` by `count` draws, discarding the outputs.
    fn drain(rng: &mut Xoshiro512PlusPlus, count: usize) {
        let mut i = 0;
        while i < count {
            rng.next_u64();
            i += 1;
        }
    }

    /// Collects ten consecutive outputs into a fixed-size array.
    fn collect10(rng: &mut Xoshiro512PlusPlus) -> [u64; 10] {
        let mut out = [0u64; 10];
        let mut i = 0;
        while i < 10 {
            out[i] = rng.next_u64();
            i += 1;
        }
        out
    }

    #[test]
    fn output_word_0_matches_anchor() {
        let mut rng = seeded();
        assert!(rng.next_u64() == FIRST_FOUR[0]);
    }

    #[test]
    fn output_word_1_matches_anchor() {
        let mut rng = seeded();
        rng.next_u64();
        assert!(rng.next_u64() == FIRST_FOUR[1]);
    }

    #[test]
    fn output_word_2_matches_anchor() {
        let mut rng = seeded();
        rng.next_u64();
        rng.next_u64();
        assert!(rng.next_u64() == FIRST_FOUR[2]);
    }

    #[test]
    fn output_word_3_matches_anchor() {
        let mut rng = seeded();
        rng.next_u64();
        rng.next_u64();
        rng.next_u64();
        assert!(rng.next_u64() == FIRST_FOUR[3]);
    }

    #[test]
    fn first_four_outputs_match_anchor() {
        let mut rng = seeded();
        let got: [u64; 4] = rng.next_array::<4>();
        assert!(got == FIRST_FOUR);
    }

    #[test]
    fn first_four_via_loop_match_anchor() {
        let mut rng = seeded();
        let mut got = [0u64; 4];
        let mut i = 0;
        while i < 4 {
            got[i] = rng.next_u64();
            i += 1;
        }
        assert!(got == FIRST_FOUR);
    }

    #[test]
    fn first_sixteen_outputs_match_anchor() {
        let mut rng = seeded();
        let got: [u64; 16] = rng.next_array::<16>();
        assert!(got == FIRST_SIXTEEN);
    }

    #[test]
    fn fifth_output_matches_anchor() {
        let mut rng = seeded();
        let mut last = 0u64;
        let mut i = 0;
        while i < 5 {
            last = rng.next_u64();
            i += 1;
        }
        assert!(last == FIRST_SIXTEEN[4]);
    }

    #[test]
    fn eighth_output_matches_anchor() {
        let mut rng = seeded();
        let mut last = 0u64;
        let mut i = 0;
        while i < 8 {
            last = rng.next_u64();
            i += 1;
        }
        assert!(last == FIRST_SIXTEEN[7]);
    }

    #[test]
    fn sixteenth_output_matches_anchor() {
        let mut rng = seeded();
        let mut last = 0u64;
        let mut i = 0;
        while i < 16 {
            last = rng.next_u64();
            i += 1;
        }
        assert!(last == FIRST_SIXTEEN[15]);
    }

    #[test]
    fn state_word_0_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[0] == STATE_AFTER_FOUR[0]);
    }

    #[test]
    fn state_word_1_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[1] == STATE_AFTER_FOUR[1]);
    }

    #[test]
    fn state_word_2_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[2] == STATE_AFTER_FOUR[2]);
    }

    #[test]
    fn state_word_3_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[3] == STATE_AFTER_FOUR[3]);
    }

    #[test]
    fn state_word_4_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[4] == STATE_AFTER_FOUR[4]);
    }

    #[test]
    fn state_word_5_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[5] == STATE_AFTER_FOUR[5]);
    }

    #[test]
    fn state_word_6_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[6] == STATE_AFTER_FOUR[6]);
    }

    #[test]
    fn state_word_7_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state()[7] == STATE_AFTER_FOUR[7]);
    }

    #[test]
    fn full_state_after_one_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 1);
        assert!(rng.state() == STATE_AFTER_ONE);
    }

    #[test]
    fn full_state_after_four_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 4);
        assert!(rng.state() == STATE_AFTER_FOUR);
    }

    #[test]
    fn full_state_after_eight_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 8);
        assert!(rng.state() == STATE_AFTER_EIGHT);
    }

    #[test]
    fn full_state_after_thousand_matches_anchor() {
        let mut rng = seeded();
        drain(&mut rng, 1000);
        assert!(rng.state() == STATE_AFTER_THOUSAND);
    }

    #[test]
    fn hundredth_output_is_stable() {
        let mut rng = seeded();
        let mut last = 0u64;
        let mut i = 0;
        while i < 100 {
            last = rng.next_u64();
            i += 1;
        }
        assert!(last == 0xe8fd21a39b9db536);
    }

    #[test]
    fn xor_accumulator_over_1000_draws_is_stable() {
        let mut rng = seeded();
        let mut acc = 0u64;
        let mut i = 0;
        while i < 1000 {
            acc ^= rng.next_u64();
            i += 1;
        }
        assert!(acc == 0x58260a681b5074bf);
    }

    #[test]
    fn wrapping_sum_over_1000_draws_is_stable() {
        let mut rng = seeded();
        let mut acc = 0u64;
        let mut i = 0;
        while i < 1000 {
            acc = acc.wrapping_add(rng.next_u64());
            i += 1;
        }
        assert!(acc == 0xc543f01a9ece4939);
    }

    #[test]
    fn next_array_matches_sequential_draws() {
        let mut batched = seeded();
        let got: [u64; 32] = batched.next_array::<32>();
        let mut sequential = seeded();
        let mut i = 0;
        while i < 32 {
            assert!(got[i] == sequential.next_u64());
            i += 1;
        }
    }

    #[test]
    fn next_array_leaves_state_like_sequential() {
        let mut batched = seeded();
        let _got: [u64; 20] = batched.next_array::<20>();
        let mut sequential = seeded();
        drain(&mut sequential, 20);
        assert!(batched.state() == sequential.state());
    }

    #[test]
    fn next_array_zero_length_is_noop() {
        let mut rng = seeded();
        let got: [u64; 0] = rng.next_array::<0>();
        assert!(got == []);
        assert!(rng.state() == SEED);
    }

    #[test]
    fn determinism_same_seed_four_draws() {
        let mut a = seeded();
        let mut b = seeded();
        let mut i = 0;
        while i < 4 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn determinism_same_seed_long_run() {
        let mut a = seeded();
        let mut b = seeded();
        let mut i = 0;
        while i < 512 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
    }

    #[test]
    fn long_run_determinism_states_and_outputs() {
        let mut a = seeded();
        let mut b = Xoshiro512PlusPlus::from_state(SEED);
        let mut i = 0;
        while i < 2048 {
            assert!(a.next_u64() == b.next_u64());
            i += 1;
        }
        assert!(a.state() == b.state());
    }

    #[test]
    fn from_state_roundtrips_initial_seed() {
        let rng = Xoshiro512PlusPlus::from_state(SEED);
        assert!(rng.state() == SEED);
    }

    #[test]
    fn from_state_roundtrips_arbitrary_state() {
        let bits = [
            0xdead_beef_0000_0001,
            0x0102_0304_0506_0708,
            0xffff_ffff_ffff_ffff,
            0x0,
            0x8000_0000_0000_0000,
            0x1234_5678_9abc_def0,
            0x5555_5555_5555_5555,
            0xaaaa_aaaa_aaaa_aaaa,
        ];
        let rng = Xoshiro512PlusPlus::from_state(bits);
        assert!(rng.state() == bits);
    }

    #[test]
    fn roundtrip_after_draw_preserves_snapshot() {
        let mut rng = seeded();
        drain(&mut rng, 3);
        let snapshot = rng.state();
        let rebuilt = Xoshiro512PlusPlus::from_state(snapshot);
        assert!(rebuilt.state() == snapshot);
    }

    #[test]
    fn rebuilt_from_state_continues_identically() {
        let mut rng = seeded();
        drain(&mut rng, 5);
        let mut rebuilt = Xoshiro512PlusPlus::from_state(rng.state());
        let mut i = 0;
        while i < 16 {
            assert!(rng.next_u64() == rebuilt.next_u64());
            i += 1;
        }
    }

    #[test]
    fn replay_from_snapshot_mid_stream() {
        let mut rng = seeded();
        drain(&mut rng, 37);
        let snapshot = rng.state();
        let tail: [u64; 10] = collect10(&mut rng);
        let mut replay = Xoshiro512PlusPlus::from_state(snapshot);
        let replay_tail: [u64; 10] = collect10(&mut replay);
        assert!(tail == replay_tail);
    }

    #[test]
    fn state_accessor_reports_initial_seed() {
        let rng = seeded();
        assert!(rng.state() == SEED);
    }

    #[test]
    fn next_u64_advances_state() {
        let mut rng = seeded();
        let before = rng.state();
        rng.next_u64();
        assert!(rng.state() != before);
    }

    #[test]
    fn two_instances_are_independent() {
        let mut a = seeded();
        let b = seeded();
        a.next_u64();
        assert!(a.state() != b.state());
    }

    #[test]
    fn copy_semantics_snapshot_is_independent() {
        let mut rng = seeded();
        let snapshot = rng;
        rng.next_u64();
        assert!(snapshot.state() == SEED);
        assert!(rng.state() != SEED);
    }

    #[test]
    fn equality_of_equal_states() {
        let a = seeded();
        let b = seeded();
        assert!(a == b);
    }

    #[test]
    fn inequality_after_single_draw() {
        let mut a = seeded();
        let b = seeded();
        a.next_u64();
        assert!(a != b);
    }

    #[test]
    fn distinct_seed_produces_distinct_stream() {
        let mut a = seeded();
        let mut b = Xoshiro512PlusPlus::from_state([8, 7, 6, 5, 4, 3, 2, 1]);
        assert!(a.next_u64() != b.next_u64());
    }

    #[test]
    fn seed_variation_in_low_word_changes_first_output() {
        let mut a = Xoshiro512PlusPlus::from_state([1, 1, 1, 1, 1, 1, 1, 1]);
        let mut b = Xoshiro512PlusPlus::from_state([2, 1, 1, 1, 1, 1, 1, 1]);
        assert!(a.next_u64() != b.next_u64());
    }

    #[test]
    fn output_word_0_is_nonzero() {
        let mut rng = seeded();
        assert!(rng.next_u64() != 0);
    }

    #[test]
    fn all_zero_state_outputs_only_zero() {
        let mut rng = Xoshiro512PlusPlus::from_state([0; 8]);
        let mut i = 0;
        while i < 16 {
            assert!(rng.next_u64() == 0);
            i += 1;
        }
    }

    #[test]
    fn all_zero_state_stays_all_zero() {
        let mut rng = Xoshiro512PlusPlus::from_state([0; 8]);
        drain(&mut rng, 16);
        assert!(rng.state() == [0; 8]);
    }

    #[test]
    fn single_nonzero_word_seed_is_nonzero_eventually() {
        let mut rng = Xoshiro512PlusPlus::from_state([0, 0, 0, 0, 0, 0, 0, 1]);
        let mut any = false;
        let mut i = 0;
        while i < 32 {
            if rng.next_u64() != 0 {
                any = true;
            }
            i += 1;
        }
        assert!(any);
    }

    #[test]
    fn single_nonzero_word_first_nonzero_at_index_two() {
        let mut rng = Xoshiro512PlusPlus::from_state([0, 0, 0, 0, 0, 0, 0, 1]);
        let mut first = -1i32;
        let mut i = 0;
        while i < 32 {
            if rng.next_u64() != 0 {
                first = i;
                break;
            }
            i += 1;
        }
        assert!(first == 2);
    }

    #[test]
    fn outputs_populate_both_half_spaces() {
        let mut rng = seeded();
        let mut saw_low = false;
        let mut saw_high = false;
        let mut i = 0;
        while i < 64 {
            let v = rng.next_u64();
            if v & (1 << 63) == 0 {
                saw_low = true;
            } else {
                saw_high = true;
            }
            i += 1;
        }
        assert!(saw_low);
        assert!(saw_high);
    }

    #[test]
    fn aggregate_popcount_is_broadly_balanced() {
        let mut rng = seeded();
        let mut ones = 0u32;
        let mut i = 0;
        while i < 256 {
            ones += rng.next_u64().count_ones();
            i += 1;
        }
        // 256 draws * 64 bits = 16384 bits; a healthy stream is near half ones.
        assert!(ones > 7000);
        assert!(ones < 9400);
    }

    #[test]
    fn every_bit_position_toggles_across_the_stream() {
        let mut rng = seeded();
        let mut seen_set = 0u64;
        let mut seen_clear = 0u64;
        let mut i = 0;
        while i < 128 {
            let v = rng.next_u64();
            seen_set |= v;
            seen_clear |= !v;
            i += 1;
        }
        assert!(seen_set == u64::MAX);
        assert!(seen_clear == u64::MAX);
    }
}
