//! The `xoshiro128++` deterministic pseudo-random number generator of
//! David Blackman and Sebastiano Vigna, specialized here as the `32`-bit
//! engine used for reproducible particle spawning, jitter, and stochastic
//! effects on platforms where `32`-bit integer throughput is preferred.
//!
//! `xoshiro128++` keeps a `128`-bit state of four `u32` words and advances
//! it with a linear `xor`/`shift`/`rotate` recurrence, then scrambles the
//! output through a *plus-plus* (`++`) step: the result of a draw is
//! `rotate_left(s0.wrapping_add(s3), 7).wrapping_add(s0)`. The
//! add-rotate-add scrambler is what distinguishes the `++` variant and gives
//! it excellent statistical quality while the underlying linear engine supplies
//! a period of `2^128 - 1`.
//!
//! Two constant-time skip-ahead operations are provided.
//! [`Xoshiro128PlusPlus::jump`] advances the stream by `2^64` draws and
//! [`Xoshiro128PlusPlus::long_jump`] by `2^96` draws. These let a
//! simulation partition one seed into many non-overlapping sub-streams -- one
//! per worker, tile, or emitter -- without any coordination, which is exactly
//! what a deterministic `CPU`/`GPU` particle system needs when the same frame
//! must replay identically.
//!
//! Seeding via [`Xoshiro128PlusPlus::from_seeds`] accepts four raw `u32`
//! words directly and only guards against the degenerate all-zero fixed point,
//! from which the linear recurrence can never escape, by forcing `s[0]` to `1`
//! when every seed word is zero. A convenience
//! [`Xoshiro128PlusPlus::from_u64_seed`] expands a single `u64` seed through a
//! `splitmix`-style diffuser so that a low-entropy seed still fills the state
//! with well-mixed words.
//!
//! Scope and boundaries: this module is deliberately narrow and completely
//! self-contained. It is the `32`-bit `++` scrambler over a `[u32; 4]`
//! state and shares no code with the `64`-bit `xoshiro256**` engine, whose
//! wider `[u64; 4]` state and multiply-based `**` scrambler are an entirely
//! separate implementation. Do not conflate the two.
//!
//! Every transition uses only integer exclusive-or, fixed shifts,
//! [`u32::rotate_left`], and wrapping addition. No floating point, division,
//! or transcendental function is involved anywhere in generation.
//!
//! These generators are fast and non-cryptographic. A `xoshiro128++` stream
//! is predictable from a handful of outputs and must never be used for
//! security, key material, or anywhere an adversary could exploit
//! predictability. It exists purely for reproducible, high-throughput
//! simulation randomness.

/// The `jump` polynomial coefficients. Applying [`Xoshiro128PlusPlus::jump`]
/// is equivalent to advancing the stream by `2^64` draws, so two generators
/// that differ by one jump produce non-overlapping sub-sequences.
const JUMP: [u32; 4] = [0x8764_000b, 0xf542_d2d3, 0x6fa0_35c3, 0x77f2_db5b];

/// The `long_jump` polynomial coefficients. Applying
/// [`Xoshiro128PlusPlus::long_jump`] is equivalent to advancing the stream by
/// `2^96` draws, partitioning the period into far fewer, far larger sub-streams
/// than [`Xoshiro128PlusPlus::jump`].
const LONG_JUMP: [u32; 4] = [0xb523_952e, 0x0b6f_099f, 0xccf5_a0ef, 0x1c58_0662];

/// The golden-ratio increment used by the `from_u64_seed` `splitmix` diffuser.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First multiplicative mixing constant of the `from_u64_seed` finalizer.
const SPLITMIX64_MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second multiplicative mixing constant of the `from_u64_seed` finalizer.
const SPLITMIX64_MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// A `xoshiro128++` `32`-bit deterministic pseudo-random number generator.
///
/// The generator holds a `128`-bit state as four `u32` words and produces a
/// reproducible `u32` stream with a period of `2^128 - 1`. Clone the value
/// to branch an identical stream, or use [`Xoshiro128PlusPlus::jump`] and
/// [`Xoshiro128PlusPlus::long_jump`] to derive non-overlapping sub-streams.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xoshiro128PlusPlus {
    /// The four `u32` state words. Never left all-zero after construction.
    s: [u32; 4],
}

impl Xoshiro128PlusPlus {
    /// Builds a generator from four raw `u32` seed words.
    ///
    /// The all-zero state is the single fixed point of the linear recurrence
    /// and would make the generator emit a degenerate stream, so when every
    /// seed word is zero the first state word is forced to `1`.
    #[must_use]
    pub fn from_seeds(a: u32, b: u32, c: u32, d: u32) -> Self {
        let mut s = [a, b, c, d];
        if s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 {
            s[0] = 1;
        }
        Self { s }
    }

    /// Builds a generator by expanding a single `u64` seed through a
    /// `splitmix`-style diffuser into four well-mixed `u32` words.
    ///
    /// This is a convenience for low-entropy seeds; it is fully deterministic
    /// and, like [`Xoshiro128PlusPlus::from_seeds`], never leaves the engine at
    /// the all-zero fixed point.
    #[must_use]
    pub fn from_u64_seed(seed: u64) -> Self {
        let mut state = seed;
        let mut draw = || {
            state = state.wrapping_add(SPLITMIX64_GAMMA);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX64_MIX_1);
            z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX64_MIX_2);
            z ^= z >> 31;
            z
        };
        let first = draw();
        let second = draw();
        let a = first as u32;
        let b = (first >> 32) as u32;
        let c = second as u32;
        let d = (second >> 32) as u32;
        Self::from_seeds(a, b, c, d)
    }

    /// Advances the state once and returns the next `u32` of the stream.
    ///
    /// The returned value is the `++` scrambler applied to the current state;
    /// the linear recurrence then updates all four words in place.
    pub fn next_u32(&mut self) -> u32 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(7).wrapping_add(s[0]);
        let t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(11);
        result
    }

    /// Advances the stream by `2^64` draws in constant time.
    ///
    /// Two generators that start from the same state and differ by one call to
    /// this method produce non-overlapping sub-sequences.
    pub fn jump(&mut self) {
        self.apply_jump(&JUMP);
    }

    /// Advances the stream by `2^96` draws in constant time.
    ///
    /// Use this to carve the period into a small number of enormous
    /// sub-streams, each itself safely subdivisible with
    /// [`Xoshiro128PlusPlus::jump`].
    pub fn long_jump(&mut self) {
        self.apply_jump(&LONG_JUMP);
    }

    /// Shared skip-ahead routine driven by a jump polynomial.
    ///
    /// For every set bit of each polynomial word the current state is folded
    /// into an accumulator; the stream is advanced once per bit regardless.
    /// The accumulator becomes the new state.
    fn apply_jump(&mut self, poly: &[u32; 4]) {
        let mut s0 = 0u32;
        let mut s1 = 0u32;
        let mut s2 = 0u32;
        let mut s3 = 0u32;
        for &jw in poly {
            for b in 0..32 {
                if jw & (1 << b) != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next_u32();
            }
        }
        self.s = [s0, s1, s2, s3];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The authoritative first six outputs for the seed `(1, 2, 3, 4)`.
    const REF_FIRST_SIX: [u32; 6] = [
        641,
        1_573_767,
        3_222_811_527,
        3_517_856_514,
        836_907_274,
        4_247_214_768,
    ];

    /// The authoritative three outputs after one `jump` from seed `(1, 2, 3, 4)`.
    const REF_AFTER_JUMP: [u32; 3] = [3_129_740_764, 111_290_574, 1_158_071_106];

    fn seeded() -> Xoshiro128PlusPlus {
        Xoshiro128PlusPlus::from_seeds(1, 2, 3, 4)
    }

    #[test]
    fn first_output_matches_reference() {
        assert_eq!(seeded().next_u32(), REF_FIRST_SIX[0]);
    }

    #[test]
    fn first_output_is_641() {
        assert_eq!(seeded().next_u32(), 641);
    }

    #[test]
    fn second_output_matches_reference() {
        let mut rng = seeded();
        rng.next_u32();
        assert_eq!(rng.next_u32(), REF_FIRST_SIX[1]);
    }

    #[test]
    fn third_output_matches_reference() {
        let mut rng = seeded();
        rng.next_u32();
        rng.next_u32();
        assert_eq!(rng.next_u32(), REF_FIRST_SIX[2]);
    }

    #[test]
    fn fourth_output_matches_reference() {
        let mut rng = seeded();
        for _ in 0..3 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REF_FIRST_SIX[3]);
    }

    #[test]
    fn fifth_output_matches_reference() {
        let mut rng = seeded();
        for _ in 0..4 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REF_FIRST_SIX[4]);
    }

    #[test]
    fn sixth_output_matches_reference() {
        let mut rng = seeded();
        for _ in 0..5 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REF_FIRST_SIX[5]);
    }

    #[test]
    fn first_six_outputs_match_reference_in_order() {
        let mut rng = seeded();
        for &expected in &REF_FIRST_SIX {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn first_six_outputs_collected_match_reference() {
        let mut rng = seeded();
        let got: [u32; 6] = core::array::from_fn(|_| rng.next_u32());
        assert_eq!(got, REF_FIRST_SIX);
    }

    #[test]
    fn jump_then_three_outputs_match_reference() {
        let mut rng = seeded();
        rng.jump();
        for &expected in &REF_AFTER_JUMP {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn jump_first_output_matches_reference() {
        let mut rng = seeded();
        rng.jump();
        assert_eq!(rng.next_u32(), REF_AFTER_JUMP[0]);
    }

    #[test]
    fn jump_second_output_matches_reference() {
        let mut rng = seeded();
        rng.jump();
        rng.next_u32();
        assert_eq!(rng.next_u32(), REF_AFTER_JUMP[1]);
    }

    #[test]
    fn jump_third_output_matches_reference() {
        let mut rng = seeded();
        rng.jump();
        rng.next_u32();
        rng.next_u32();
        assert_eq!(rng.next_u32(), REF_AFTER_JUMP[2]);
    }

    #[test]
    fn determinism_identical_seeds_same_sequence() {
        let mut a = seeded();
        let mut b = seeded();
        for _ in 0..256 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn determinism_from_u64_seed_same_sequence() {
        let mut a = Xoshiro128PlusPlus::from_u64_seed(0xDEAD_BEEF_CAFE_1234);
        let mut b = Xoshiro128PlusPlus::from_u64_seed(0xDEAD_BEEF_CAFE_1234);
        for _ in 0..128 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn determinism_clone_tracks_original() {
        let mut a = seeded();
        for _ in 0..17 {
            a.next_u32();
        }
        let mut b = a.clone();
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_differ_quickly() {
        let mut a = Xoshiro128PlusPlus::from_seeds(1, 2, 3, 4);
        let mut b = Xoshiro128PlusPlus::from_seeds(4, 3, 2, 1);
        let mut any_diff = false;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                any_diff = true;
            }
        }
        assert!(any_diff);
    }

    #[test]
    fn all_zero_seed_is_promoted() {
        let rng = Xoshiro128PlusPlus::from_seeds(0, 0, 0, 0);
        assert_eq!(rng.s, [1, 0, 0, 0]);
    }

    #[test]
    fn all_zero_seed_does_not_emit_constant_zero() {
        let mut rng = Xoshiro128PlusPlus::from_seeds(0, 0, 0, 0);
        let mut saw_nonzero = false;
        for _ in 0..64 {
            if rng.next_u32() != 0 {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero);
    }

    #[test]
    fn all_zero_seed_sequence_is_not_constant() {
        let mut rng = Xoshiro128PlusPlus::from_seeds(0, 0, 0, 0);
        let first = rng.next_u32();
        let mut all_equal = true;
        for _ in 0..64 {
            if rng.next_u32() != first {
                all_equal = false;
            }
        }
        assert!(!all_equal);
    }

    #[test]
    fn nonzero_seed_is_left_untouched() {
        let rng = Xoshiro128PlusPlus::from_seeds(9, 8, 7, 6);
        assert_eq!(rng.s, [9, 8, 7, 6]);
    }

    #[test]
    fn single_nonzero_word_is_not_promoted() {
        let rng = Xoshiro128PlusPlus::from_seeds(0, 0, 0, 5);
        assert_eq!(rng.s, [0, 0, 0, 5]);
    }

    #[test]
    fn jump_changes_state() {
        let mut jumped = seeded();
        jumped.jump();
        let plain = seeded();
        assert_ne!(jumped.s, plain.s);
    }

    #[test]
    fn jump_is_deterministic() {
        let mut a = seeded();
        let mut b = seeded();
        a.jump();
        b.jump();
        assert_eq!(a.s, b.s);
        for _ in 0..32 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn long_jump_runs_and_changes_state() {
        let mut jumped = seeded();
        jumped.long_jump();
        let plain = seeded();
        assert_ne!(jumped.s, plain.s);
    }

    #[test]
    fn long_jump_is_deterministic() {
        let mut a = seeded();
        let mut b = seeded();
        a.long_jump();
        b.long_jump();
        assert_eq!(a.s, b.s);
        for _ in 0..32 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn jump_and_long_jump_land_on_different_states() {
        let mut jumped = seeded();
        let mut long_jumped = seeded();
        jumped.jump();
        long_jumped.long_jump();
        assert_ne!(jumped.s, long_jumped.s);
    }

    #[test]
    fn jump_produces_non_overlapping_prefix() {
        let base = seeded();
        let mut jumped = base.clone();
        jumped.jump();
        let mut plain = base;
        let plain_prefix: [u32; 8] = core::array::from_fn(|_| plain.next_u32());
        let jumped_prefix: [u32; 8] = core::array::from_fn(|_| jumped.next_u32());
        assert_ne!(plain_prefix, jumped_prefix);
    }

    #[test]
    fn outputs_not_all_equal() {
        let mut rng = seeded();
        let first = rng.next_u32();
        let mut all_equal = true;
        for _ in 0..128 {
            if rng.next_u32() != first {
                all_equal = false;
            }
        }
        assert!(!all_equal);
    }

    #[test]
    fn outputs_not_all_zero() {
        let mut rng = seeded();
        let mut saw_nonzero = false;
        for _ in 0..128 {
            if rng.next_u32() != 0 {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero);
    }

    #[test]
    fn outputs_span_both_halves_of_range() {
        let mut rng = seeded();
        let mut saw_low = false;
        let mut saw_high = false;
        for _ in 0..256 {
            let value = rng.next_u32();
            if value < 0x8000_0000 {
                saw_low = true;
            } else {
                saw_high = true;
            }
        }
        assert!(saw_low && saw_high);
    }

    #[test]
    fn outputs_are_mostly_distinct() {
        let mut rng = seeded();
        let values: [u32; 64] = core::array::from_fn(|_| rng.next_u32());
        let mut duplicates = 0usize;
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                if values[i] == values[j] {
                    duplicates += 1;
                }
            }
        }
        assert!(duplicates < 4);
    }

    #[test]
    fn low_bits_are_not_constant() {
        let mut rng = seeded();
        let mut seen_even = false;
        let mut seen_odd = false;
        for _ in 0..64 {
            if rng.next_u32() & 1 == 0 {
                seen_even = true;
            } else {
                seen_odd = true;
            }
        }
        assert!(seen_even && seen_odd);
    }

    #[test]
    fn high_bits_are_not_constant() {
        let mut rng = seeded();
        let mut seen_set = false;
        let mut seen_clear = false;
        for _ in 0..64 {
            if rng.next_u32() & 0x8000_0000 != 0 {
                seen_set = true;
            } else {
                seen_clear = true;
            }
        }
        assert!(seen_set && seen_clear);
    }

    #[test]
    fn from_u64_seed_zero_is_not_degenerate() {
        let mut rng = Xoshiro128PlusPlus::from_u64_seed(0);
        let mut saw_nonzero = false;
        for _ in 0..64 {
            if rng.next_u32() != 0 {
                saw_nonzero = true;
            }
        }
        assert!(saw_nonzero);
    }

    #[test]
    fn from_u64_seed_distinct_seeds_differ() {
        let mut a = Xoshiro128PlusPlus::from_u64_seed(1);
        let mut b = Xoshiro128PlusPlus::from_u64_seed(2);
        let mut any_diff = false;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                any_diff = true;
            }
        }
        assert!(any_diff);
    }

    #[test]
    fn jump_twice_differs_from_jump_once() {
        let mut once = seeded();
        once.jump();
        let mut twice = seeded();
        twice.jump();
        twice.jump();
        assert_ne!(once.s, twice.s);
    }

    #[test]
    fn long_jump_then_jump_is_deterministic() {
        let mut a = seeded();
        let mut b = seeded();
        a.long_jump();
        a.jump();
        b.long_jump();
        b.jump();
        for _ in 0..32 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn advancing_changes_internal_state() {
        let mut rng = seeded();
        let before = rng.s;
        rng.next_u32();
        assert_ne!(rng.s, before);
    }

    #[test]
    fn sequence_restarts_identically_from_fresh_seed() {
        let mut a = seeded();
        let first_run: [u32; 16] = core::array::from_fn(|_| a.next_u32());
        let mut b = seeded();
        let second_run: [u32; 16] = core::array::from_fn(|_| b.next_u32());
        assert_eq!(first_run, second_run);
    }
}
