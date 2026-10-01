//! The `xoshiro256**` deterministic pseudo-random number generator of David
//! Blackman and Sebastiano Vigna, used for reproducible particle spawning,
//! jitter, and stochastic effects where a long period and splittable parallel
//! streams matter.
//!
//! `xoshiro256**` keeps a `256`-bit state of four `u64` words and advances it
//! with a linear `xor`/`shift`/`rotate` recurrence, then scrambles the output
//! through a *`starstar`* (`**`) non-linear step: the result of a draw is
//! `rotate_left(s1.wrapping_mul(5), 7).wrapping_mul(9)`. The multiply-rotate-
//! multiply scrambler is what distinguishes the `**` variant and gives it
//! excellent statistical quality while the underlying linear engine supplies a
//! period of `2^256 - 1`.
//!
//! Two constant-time skip-ahead operations are provided. [`Xoshiro256StarStar::jump`]
//! advances the stream by `2^128` draws and [`Xoshiro256StarStar::long_jump`]
//! by `2^192` draws. These let a simulation partition one seed into many
//! non-overlapping sub-streams -- one per worker, tile, or emitter -- without
//! any coordination, which is exactly what a deterministic `CPU`/`GPU`
//! particle system needs when the same frame must replay identically.
//!
//! Seeding runs the caller's `u64` seed through an internal `splitmix64`
//! diffuser (see [`Xoshiro256StarStar::from_seed_splitmix64`]) so that a
//! low-entropy seed such as `0` or `1` still expands into four well-mixed
//! words and never leaves the engine at its degenerate all-zero fixed point,
//! from which the linear recurrence can never escape.
//!
//! Scope and boundaries: this module is deliberately narrow. Unlike
//! `xorshift_rng`, whose engines use a plain `xorshift` recurrence with no
//! rotate-and-multiply output scrambler, `xoshiro256**` adds the non-linear
//! `**` scrambler on top of a wider `256`-bit linear state. Unlike
//! `splitmix64`, which is a single-variable `64`-bit mixer used only to expand
//! a seed, this engine is a full stateful stream with jump functions. And
//! unlike `pcg_hash` or `wang_hash`, which are stateless *hashes* mapping one
//! input to one digest, `xoshiro256**` is a *stream*: each call mutates the
//! state and returns the next element of a long reproducible sequence.
//!
//! Every transition uses only integer exclusive-or, fixed shifts,
//! [`u64::rotate_left`], and wrapping multiply. No floating point, division,
//! or transcendental function is involved anywhere in generation.
//!
//! These generators are fast and non-cryptographic. A `xoshiro256**` stream is
//! predictable from a handful of outputs and must never be used for security,
//! key material, or anywhere an adversary could exploit predictability. It
//! exists purely for reproducible, high-throughput simulation randomness.

/// The golden-ratio increment of the internal `splitmix64` seed diffuser.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First multiplicative mixing constant of the internal `splitmix64` finalizer.
const SPLITMIX64_MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second multiplicative mixing constant of the internal `splitmix64` finalizer.
const SPLITMIX64_MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// The `jump` polynomial coefficients. Applying [`Xoshiro256StarStar::jump`] is
/// equivalent to advancing the stream by `2^128` draws, so two generators that
/// differ by one jump produce non-overlapping sub-sequences.
const JUMP: [u64; 4] = [
    0x180E_C6D3_3CFD_0ABA,
    0xD5A6_1266_F0C9_392C,
    0xA958_2618_E03F_C9AA,
    0x39AB_DC45_29B1_661C,
];

/// The `long_jump` polynomial coefficients. Applying
/// [`Xoshiro256StarStar::long_jump`] is equivalent to advancing the stream by
/// `2^192` draws, giving `2^64` well-separated starting points for parallel
/// streams.
const LONG_JUMP: [u64; 4] = [
    0x76E1_5D3E_FEFD_CBBF,
    0xC500_4E44_1C52_2FB3,
    0x7771_0069_854E_E241,
    0x3910_9BB0_2ACB_E635,
];

/// A `xoshiro256**` generator: four `u64` state words advanced by the
/// Blackman-Vigna recurrence with a rotate-and-multiply output scrambler.
///
/// Construct one from a single seed with
/// [`Xoshiro256StarStar::from_seed_splitmix64`], or from an explicit
/// `256`-bit state with [`Xoshiro256StarStar::from_state`]. Draw values with
/// [`Xoshiro256StarStar::next_u64`] and split streams with
/// [`Xoshiro256StarStar::jump`] / [`Xoshiro256StarStar::long_jump`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Xoshiro256StarStar {
    /// The four `u64` state words. The all-zero state is a fixed point and is
    /// never produced by [`Xoshiro256StarStar::from_seed_splitmix64`].
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Builds a generator from an explicit `256`-bit state.
    ///
    /// The caller is responsible for not passing the all-zero state, which is
    /// the degenerate fixed point of the recurrence; prefer
    /// [`Xoshiro256StarStar::from_seed_splitmix64`] when expanding a plain
    /// seed.
    #[must_use]
    pub const fn from_state(state: [u64; 4]) -> Self {
        Self { s: state }
    }

    /// Builds a generator by expanding a single `u64` `seed` into four
    /// well-mixed state words with an internal `splitmix64` diffuser.
    ///
    /// The diffuser advances a running register by the golden-ratio increment
    /// `SPLITMIX64_GAMMA` and applies the standard `splitmix64` finalizer four
    /// times. Because the finalizer is a bijection and the increment is odd,
    /// the four words differ even for a `seed` of `0`, so the resulting state
    /// is never all-zero.
    #[must_use]
    pub const fn from_seed_splitmix64(seed: u64) -> Self {
        let mut register = seed;
        let mut state = [0u64; 4];
        let mut index = 0;
        while index < 4 {
            register = register.wrapping_add(SPLITMIX64_GAMMA);
            let mut z = register;
            z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX64_MIX_1);
            z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX64_MIX_2);
            z ^= z >> 31;
            state[index] = z;
            index += 1;
        }
        Self { s: state }
    }

    /// Returns a copy of the current `256`-bit state as four `u64` words.
    #[must_use]
    pub const fn state(&self) -> [u64; 4] {
        self.s
    }

    /// Advances the stream and returns the next `u64`.
    ///
    /// The returned value is the `starstar` scramble of the current state,
    /// `rotate_left(s1.wrapping_mul(5), 7).wrapping_mul(9)`, computed before
    /// the linear state transition is applied. Every operation is wrapping
    /// integer arithmetic.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Returns a `bool` drawn from the high bit of the next `u64`.
    ///
    /// The most-significant bit is used because it has the longest period in
    /// the underlying linear engine.
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 63) & 1 == 1
    }

    /// Returns a uniformly distributed value in the half-open range
    /// `[0, bound)` with no modulo bias, or `0` when `bound` is `0`.
    ///
    /// Low raw outputs that would skew the distribution are rejected using the
    /// standard Lemire threshold `(2^64 mod bound)`; the expected number of
    /// draws is just above one for all but the largest bounds.
    pub fn next_bounded(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let raw = self.next_u64();
            if raw >= threshold {
                return raw % bound;
            }
        }
    }

    /// Fills `out` with pseudo-random bytes, drawing one `u64` per eight bytes
    /// and emitting its little-endian representation.
    ///
    /// A trailing chunk shorter than eight bytes consumes one full draw and
    /// keeps only its leading bytes, so the number of draws is
    /// `out.len().div_ceil(8)`.
    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }

    /// Advances the stream by `2^128` draws in constant time.
    ///
    /// Useful for partitioning one seed into non-overlapping sub-streams: give
    /// each worker a clone and apply a distinct number of jumps.
    pub fn jump(&mut self) {
        self.apply_jump(&JUMP);
    }

    /// Advances the stream by `2^192` draws in constant time, yielding `2^64`
    /// widely separated starting points.
    pub fn long_jump(&mut self) {
        self.apply_jump(&LONG_JUMP);
    }

    /// Applies a jump polynomial: for every set bit of every coefficient the
    /// current state is folded into an accumulator, and the stream is advanced
    /// one step per bit. The accumulated state replaces the current state.
    fn apply_jump(&mut self, table: &[u64; 4]) {
        let mut accumulator = [0u64; 4];
        for &word in table.iter() {
            for bit in 0..64u32 {
                if (word >> bit) & 1 == 1 {
                    for (slot, &current) in accumulator.iter_mut().zip(self.s.iter()) {
                        *slot ^= current;
                    }
                }
                let _ = self.next_u64();
            }
        }
        self.s = accumulator;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed low-entropy state used as the primary regression anchor.
    const ANCHOR_STATE: [u64; 4] = [1, 2, 3, 4];

    /// Builds the anchor generator from [`ANCHOR_STATE`].
    #[cfg(test)]
    fn anchor() -> Xoshiro256StarStar {
        Xoshiro256StarStar::from_state(ANCHOR_STATE)
    }

    /// Draws `count` values into a fixed-size array helper.
    #[cfg(test)]
    fn draw_many<const N: usize>(rng: &mut Xoshiro256StarStar) -> [u64; N] {
        let mut out = [0u64; N];
        for slot in out.iter_mut() {
            *slot = rng.next_u64();
        }
        out
    }

    #[test]
    fn first_output_anchor() {
        let mut rng = anchor();
        assert_eq!(rng.next_u64(), 11_520);
    }

    #[test]
    fn second_output_is_zero() {
        let mut rng = anchor();
        let _ = rng.next_u64();
        assert_eq!(rng.next_u64(), 0);
    }

    #[test]
    fn third_output_anchor() {
        let mut rng = anchor();
        let _ = rng.next_u64();
        let _ = rng.next_u64();
        assert_eq!(rng.next_u64(), 1_509_978_240);
    }

    #[test]
    fn fourth_output_anchor() {
        let mut rng = anchor();
        for _ in 0..3 {
            let _ = rng.next_u64();
        }
        assert_eq!(rng.next_u64(), 1_215_971_899_390_074_240);
    }

    #[test]
    fn fifth_output_anchor() {
        let mut rng = anchor();
        for _ in 0..4 {
            let _ = rng.next_u64();
        }
        assert_eq!(rng.next_u64(), 1_216_172_134_540_287_360);
    }

    #[test]
    fn first_five_outputs_sequence() {
        let mut rng = anchor();
        let seq: [u64; 5] = draw_many(&mut rng);
        assert_eq!(
            seq,
            [
                11_520,
                0,
                1_509_978_240,
                1_215_971_899_390_074_240,
                1_216_172_134_540_287_360,
            ]
        );
    }

    #[test]
    fn state_after_five_outputs() {
        let mut rng = anchor();
        for _ in 0..5 {
            let _ = rng.next_u64();
        }
        assert_eq!(
            rng.state(),
            [
                9_250_551_998_855_381_762,
                105_553_519_575_810,
                9_228_034_086_416_548_613,
                4_611_932_360_672_346_753,
            ]
        );
    }

    #[test]
    fn splitmix64_fill_zero_seed() {
        let rng = Xoshiro256StarStar::from_seed_splitmix64(0);
        assert_eq!(
            rng.state(),
            [
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
            ]
        );
    }

    #[test]
    fn splitmix64_fill_seed_42() {
        let rng = Xoshiro256StarStar::from_seed_splitmix64(42);
        assert_eq!(
            rng.state(),
            [
                13_679_457_532_755_275_413,
                2_949_826_092_126_892_291,
                5_139_283_748_462_763_858,
                6_349_198_060_258_255_764,
            ]
        );
    }

    #[test]
    fn seeded_zero_first_five() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(0);
        let seq: [u64; 5] = draw_many(&mut rng);
        assert_eq!(
            seq,
            [
                11_091_344_671_253_066_420,
                13_793_997_310_169_335_082,
                1_900_383_378_846_508_768,
                7_684_712_102_626_143_532,
                13_521_403_990_117_723_737,
            ]
        );
    }

    #[test]
    fn seeded_42_first_three() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(42);
        let seq: [u64; 3] = draw_many(&mut rng);
        assert_eq!(
            seq,
            [
                1_546_998_764_402_558_742,
                6_990_951_692_964_543_102,
                12_544_586_762_248_559_009,
            ]
        );
    }

    #[test]
    fn jump_state_from_anchor() {
        let mut rng = anchor();
        rng.jump();
        assert_eq!(
            rng.state(),
            [
                10_122_426_448_480_695_249,
                8_079_205_330_032_121_950,
                7_289_065_458_748_526_725,
                9_477_464_255_293_849_680,
            ]
        );
    }

    #[test]
    fn jump_then_next_three() {
        let mut rng = anchor();
        rng.jump();
        let seq: [u64; 3] = draw_many(&mut rng);
        assert_eq!(
            seq,
            [
                13_534_147_089_533_256_664,
                7_126_240_192_422_241_655,
                3_805_973_808_039_778_091,
            ]
        );
    }

    #[test]
    fn long_jump_state_from_anchor() {
        let mut rng = anchor();
        rng.long_jump();
        assert_eq!(
            rng.state(),
            [
                678_511_610_814_637_056,
                15_850_499_779_492_529_430,
                6_002_989_639_035_333_134,
                3_559_352_929_785_830_385,
            ]
        );
    }

    #[test]
    fn jump_from_splitmix_zero() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(0);
        rng.jump();
        assert_eq!(
            rng.state(),
            [
                18_367_075_165_535_767_938,
                16_958_246_640_038_696_355,
                535_696_466_813_022_735,
                12_729_174_445_987_518_331,
            ]
        );
    }

    #[test]
    fn jump_differs_from_long_jump() {
        let mut jumped = anchor();
        jumped.jump();
        let mut long_jumped = anchor();
        long_jumped.long_jump();
        assert_ne!(jumped.state(), long_jumped.state());
    }

    #[test]
    fn jump_changes_state() {
        let before = anchor();
        let mut after = anchor();
        after.jump();
        assert_ne!(before.state(), after.state());
    }

    #[test]
    fn long_jump_changes_state() {
        let before = anchor();
        let mut after = anchor();
        after.long_jump();
        assert_ne!(before.state(), after.state());
    }

    #[test]
    fn jump_is_deterministic() {
        let mut a = anchor();
        let mut b = anchor();
        a.jump();
        b.jump();
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn long_jump_is_deterministic() {
        let mut a = anchor();
        let mut b = anchor();
        a.long_jump();
        b.long_jump();
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn fill_bytes_first_sixteen() {
        let mut rng = anchor();
        let mut buffer = [0u8; 16];
        rng.fill_bytes(&mut buffer);
        assert_eq!(buffer, [0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn fill_bytes_matches_next_u64_le() {
        let mut via_bytes = anchor();
        let mut buffer = [0u8; 8];
        via_bytes.fill_bytes(&mut buffer);

        let mut via_draw = anchor();
        let expected = via_draw.next_u64().to_le_bytes();
        assert_eq!(buffer, expected);
    }

    #[test]
    fn fill_bytes_partial_length_consumes_one_draw() {
        let mut rng = anchor();
        let mut three = [0u8; 3];
        rng.fill_bytes(&mut three);
        // The first draw is 11_520 = 0x2D00, so the leading three LE bytes are
        // 0x00, 0x2D, 0x00.
        assert_eq!(three, [0, 45, 0]);
        // Only one draw was consumed, so the next draw is the second output.
        assert_eq!(rng.next_u64(), 0);
    }

    #[test]
    fn fill_bytes_zero_length_is_noop() {
        let mut rng = anchor();
        let mut empty: [u8; 0] = [];
        rng.fill_bytes(&mut empty);
        // No draw was consumed, so the stream still starts at the first output.
        assert_eq!(rng.next_u64(), 11_520);
    }

    #[test]
    fn fill_bytes_length_is_multiple_of_eight() {
        let full = 32usize;
        assert!(full.is_multiple_of(8));
        let mut rng = anchor();
        let mut buffer = [0u8; 32];
        rng.fill_bytes(&mut buffer);
        // Reconstruct the four draws from little-endian chunks.
        let mut reference = anchor();
        for chunk in buffer.chunks_exact(8) {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            assert_eq!(u64::from_le_bytes(word), reference.next_u64());
        }
    }

    #[test]
    fn next_bounded_ten_sequence() {
        let mut rng = anchor();
        let seq: [u64; 10] = draw_bounded(&mut rng, 10);
        assert_eq!(seq, [0, 0, 0, 0, 0, 5, 2, 7, 6, 6]);
    }

    #[test]
    fn next_bounded_six_sequence() {
        let mut rng = anchor();
        let seq: [u64; 12] = draw_bounded(&mut rng, 6);
        assert_eq!(seq, [0, 0, 0, 0, 0, 3, 4, 5, 4, 2, 0, 4]);
    }

    #[test]
    fn next_bounded_one_always_zero() {
        let mut rng = anchor();
        for _ in 0..32 {
            assert_eq!(rng.next_bounded(1), 0);
        }
    }

    #[test]
    fn next_bounded_zero_returns_zero() {
        let mut rng = anchor();
        assert_eq!(rng.next_bounded(0), 0);
    }

    #[test]
    fn next_bounded_stays_within_range() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(7);
        let bound = 97u64;
        let range = 0..bound;
        for _ in 0..2_000 {
            let value = rng.next_bounded(bound);
            assert!(range.contains(&value));
        }
    }

    #[test]
    fn next_bounded_power_of_two_masks_cleanly() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(9);
        let bound = 256u64;
        for _ in 0..1_000 {
            assert!((0..bound).contains(&rng.next_bounded(bound)));
        }
    }

    #[test]
    fn next_bool_sequence_from_zero_seed() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(0);
        let expected = [true, true, false, false, true, true, false, true];
        for &want in expected.iter() {
            assert_eq!(rng.next_bool(), want);
        }
    }

    #[test]
    fn next_bool_mixes_both_values() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(123);
        let mut ones: u64 = 0;
        for _ in 0..1_000 {
            ones += u64::from(rng.next_bool());
        }
        // A fair bit should land well away from both extremes.
        assert!((300..700).contains(&ones));
    }

    #[test]
    fn determinism_same_seed_same_stream() {
        let mut a = Xoshiro256StarStar::from_seed_splitmix64(2024);
        let mut b = Xoshiro256StarStar::from_seed_splitmix64(2024);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Xoshiro256StarStar::from_seed_splitmix64(1);
        let mut b = Xoshiro256StarStar::from_seed_splitmix64(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn adjacent_seeds_diverge_quickly() {
        let mut a = Xoshiro256StarStar::from_seed_splitmix64(1_000);
        let mut b = Xoshiro256StarStar::from_seed_splitmix64(1_001);
        let mut differences = 0u64;
        for _ in 0..32 {
            if a.next_u64() != b.next_u64() {
                differences += 1;
            }
        }
        assert!(differences >= 30);
    }

    #[test]
    fn state_roundtrip_from_state() {
        let original = Xoshiro256StarStar::from_seed_splitmix64(555);
        let rebuilt = Xoshiro256StarStar::from_state(original.state());
        assert_eq!(original, rebuilt);
    }

    #[test]
    fn seeding_zero_is_not_all_zero_state() {
        let rng = Xoshiro256StarStar::from_seed_splitmix64(0);
        assert_ne!(rng.state(), [0, 0, 0, 0]);
    }

    #[test]
    fn seeding_one_is_not_all_zero_state() {
        let rng = Xoshiro256StarStar::from_seed_splitmix64(1);
        assert_ne!(rng.state(), [0, 0, 0, 0]);
    }

    #[test]
    fn nonzero_state_stays_nonzero_after_draws() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(0);
        for _ in 0..256 {
            let _ = rng.next_u64();
            assert_ne!(rng.state(), [0, 0, 0, 0]);
        }
    }

    #[test]
    fn copy_semantics_are_independent() {
        let mut rng = anchor();
        let mut snapshot = rng;
        let _ = rng.next_u64();
        // The copy is unaffected by advancing the original.
        assert_eq!(snapshot.next_u64(), 11_520);
    }

    #[test]
    fn stream_not_immediately_repeating() {
        let mut rng = Xoshiro256StarStar::from_seed_splitmix64(314);
        let first = rng.next_u64();
        let mut repeats = 0u64;
        for _ in 0..64 {
            if rng.next_u64() == first {
                repeats += 1;
            }
        }
        assert_eq!(repeats, 0);
    }

    #[test]
    fn splitmix64_fill_words_are_distinct() {
        let state = Xoshiro256StarStar::from_seed_splitmix64(0).state();
        for (offset, &earlier) in state.iter().enumerate() {
            for &later in state.iter().skip(offset + 1) {
                assert_ne!(earlier, later);
            }
        }
    }

    #[test]
    fn jump_matches_two_half_sub_streams_are_disjoint() {
        // A plain generator and its jumped twin should not immediately collide.
        let mut base = anchor();
        let mut jumped = anchor();
        jumped.jump();
        let base_first: [u64; 8] = draw_many(&mut base);
        let jumped_first: [u64; 8] = draw_many(&mut jumped);
        assert_ne!(base_first, jumped_first);
    }

    #[test]
    fn long_jump_twin_is_disjoint() {
        let mut base = anchor();
        let mut long_jumped = anchor();
        long_jumped.long_jump();
        assert_ne!(base.next_u64(), long_jumped.next_u64());
    }

    #[test]
    fn next_u64_advances_state_each_call() {
        let mut rng = anchor();
        let before = rng.state();
        let _ = rng.next_u64();
        assert_ne!(before, rng.state());
    }

    /// Draws `N` bounded values into a fixed-size array helper.
    #[cfg(test)]
    fn draw_bounded<const N: usize>(rng: &mut Xoshiro256StarStar, bound: u64) -> [u64; N] {
        let mut out = [0u64; N];
        for slot in out.iter_mut() {
            *slot = rng.next_bounded(bound);
        }
        out
    }
}
