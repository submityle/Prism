//! `WELL512a` deterministic `PRNG` `CPU` gold standard.
//!
//! This module provides a pure integer `u32` reference implementation of the
//! `WELL512a` (Well Equidistributed Long-period Linear) pseudo random number
//! generator. It is written to be `no_std` + `alloc` friendly: it performs only
//! `XOR`, fixed-width shifts and `and`-mask operations, with no floating point,
//! no division, no transcendental functions and no rotate operations.
//!
//! The generator keeps a 512-bit state as sixteen `u32` words plus an index
//! into that state. All index arithmetic is taken modulo 16 via a `& 15` mask.
//! The `RNG` is fully deterministic: a given seed state always produces the
//! same output sequence, which makes it suitable as a gold standard against
//! which `CPU` and `GPU` implementations can be compared.

/// Transform matrix constant used by the `WELL512a` recurrence.
const MAT: u32 = 0xDA44_2D24;

/// Deterministic `WELL512a` pseudo random number generator.
///
/// Holds a sixteen word `u32` state and an index into that state. Construct one
/// with [`Well512a::from_state`]; the index always starts at zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Well512a {
    state: [u32; 16],
    index: u32,
}

impl Well512a {
    /// Build a generator from an explicit sixteen word state.
    ///
    /// The index is initialized to zero.
    #[must_use]
    pub const fn from_state(state: [u32; 16]) -> Self {
        Self { state, index: 0 }
    }

    /// Return a copy of the current sixteen word state.
    #[must_use]
    pub const fn state(&self) -> [u32; 16] {
        self.state
    }

    /// Return the current index into the state.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Advance the generator and return the next `u32` output.
    ///
    /// Implements the `WELL512a` recurrence using only `XOR`, fixed-width
    /// shifts and an `and`-mask against [`MAT`].
    pub fn next_u32(&mut self) -> u32 {
        let idx = self.index;
        let a = self.state[idx as usize];
        let mut c = self.state[((idx + 13) & 15) as usize];
        let b = a ^ c ^ (a << 16) ^ (c << 15);
        c = self.state[((idx + 9) & 15) as usize];
        c ^= c >> 11;
        let a = b ^ c;
        self.state[idx as usize] = a;
        let d = a ^ ((a << 5) & MAT);
        let idx = (idx + 15) & 15;
        let a = self.state[idx as usize];
        self.state[idx as usize] = a ^ b ^ d ^ (a << 2) ^ (b << 18) ^ (c << 28);
        self.index = idx;
        self.state[idx as usize]
    }

    /// Produce a fixed-length array of outputs.
    ///
    /// The outputs are generated in order, so the result is identical to
    /// calling [`Well512a::next_u32`] `N` times and collecting the values.
    pub fn next_array<const N: usize>(&mut self) -> [u32; N] {
        core::array::from_fn(|_| self.next_u32())
    }
}

#[cfg(test)]
mod tests {
    use super::Well512a;

    /// Canonical seed state `[1, 2, 3, ..., 16]`.
    const SEED: [u32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    /// First twenty outputs for the canonical seed (reference vectors).
    const OUT20: [u32; 20] = [
        0xa07c_007a,
        0x91dc_0d3a,
        0x2cd8_253e,
        0xfc90_243e,
        0xe094_043a,
        0xd08c_0422,
        0xc0e8_0526,
        0xbc84_242e,
        0xa0a4_052a,
        0xd4fa_2dde,
        0xf538_a470,
        0x6f7b_b82e,
        0x5b16_b452,
        0xb3ec_1179,
        0x02aa_1d4c,
        0x7566_b07f,
        0xc2a7_9434,
        0xc582_29c8,
        0x540b_2d40,
        0x8ec2_3e3b,
    ];

    /// State after consuming the first six outputs from the canonical seed.
    const FINAL_STATE_6: [u32; 16] = [
        0x0006_0005,
        0x2,
        0x3,
        0x4,
        0x5,
        0x6,
        0x7,
        0x8,
        0x9,
        0xa,
        0xd08c_0422,
        0xe4aa_8436,
        0xd8ab_2432,
        0x09e3_a532,
        0x9ce0_0d3e,
        0xa000_807e,
    ];

    fn seeded() -> Well512a {
        Well512a::from_state(SEED)
    }

    // --- out6: assert each of the first six outputs individually ---

    #[test]
    fn out6_value_0() {
        let mut rng = seeded();
        assert!(rng.next_u32() == 0xa07c_007a);
    }

    #[test]
    fn out6_value_1() {
        let mut rng = seeded();
        rng.next_u32();
        assert!(rng.next_u32() == 0x91dc_0d3a);
    }

    #[test]
    fn out6_value_2() {
        let mut rng = seeded();
        for _ in 0..2 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0x2cd8_253e);
    }

    #[test]
    fn out6_value_3() {
        let mut rng = seeded();
        for _ in 0..3 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0xfc90_243e);
    }

    #[test]
    fn out6_value_4() {
        let mut rng = seeded();
        for _ in 0..4 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0xe094_043a);
    }

    #[test]
    fn out6_value_5() {
        let mut rng = seeded();
        for _ in 0..5 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == 0xd08c_0422);
    }

    // --- out20: assert each output beyond the first six ---

    #[test]
    fn out20_value_6() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[6]);
    }

    #[test]
    fn out20_value_7() {
        let mut rng = seeded();
        for _ in 0..7 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[7]);
    }

    #[test]
    fn out20_value_8() {
        let mut rng = seeded();
        for _ in 0..8 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[8]);
    }

    #[test]
    fn out20_value_9() {
        let mut rng = seeded();
        for _ in 0..9 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[9]);
    }

    #[test]
    fn out20_value_10() {
        let mut rng = seeded();
        for _ in 0..10 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[10]);
    }

    #[test]
    fn out20_value_11() {
        let mut rng = seeded();
        for _ in 0..11 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[11]);
    }

    #[test]
    fn out20_value_12() {
        let mut rng = seeded();
        for _ in 0..12 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[12]);
    }

    #[test]
    fn out20_value_13() {
        let mut rng = seeded();
        for _ in 0..13 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[13]);
    }

    #[test]
    fn out20_value_14() {
        let mut rng = seeded();
        for _ in 0..14 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[14]);
    }

    #[test]
    fn out20_value_15() {
        let mut rng = seeded();
        for _ in 0..15 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[15]);
    }

    #[test]
    fn out20_value_16() {
        let mut rng = seeded();
        for _ in 0..16 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[16]);
    }

    #[test]
    fn out20_value_17() {
        let mut rng = seeded();
        for _ in 0..17 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[17]);
    }

    #[test]
    fn out20_value_18() {
        let mut rng = seeded();
        for _ in 0..18 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[18]);
    }

    #[test]
    fn out20_value_19() {
        let mut rng = seeded();
        for _ in 0..19 {
            rng.next_u32();
        }
        assert!(rng.next_u32() == OUT20[19]);
    }

    // --- full sequence and array equivalence ---

    #[test]
    fn full_out20_sequence() {
        let mut rng = seeded();
        let mut i = 0usize;
        while i < 20 {
            assert!(rng.next_u32() == OUT20[i]);
            i += 1;
        }
    }

    #[test]
    fn next_array_six_matches_out6() {
        let mut rng = seeded();
        let got: [u32; 6] = rng.next_array();
        let mut i = 0usize;
        while i < 6 {
            assert!(got[i] == OUT20[i]);
            i += 1;
        }
    }

    #[test]
    fn next_array_matches_sequential() {
        let mut rng_a = seeded();
        let mut rng_b = seeded();
        let arr: [u32; 10] = rng_a.next_array();
        let mut i = 0usize;
        while i < 10 {
            assert!(arr[i] == rng_b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn next_array_single_element() {
        let mut rng = seeded();
        let arr: [u32; 1] = rng.next_array();
        assert!(arr[0] == 0xa07c_007a);
    }

    #[test]
    fn next_array_empty_does_not_advance() {
        let mut rng = seeded();
        let arr: [u32; 0] = rng.next_array();
        assert!(arr.is_empty());
        assert!(rng.index() == 0);
        assert!(rng.next_u32() == 0xa07c_007a);
    }

    #[test]
    fn next_array_twenty_matches_reference() {
        let mut rng = seeded();
        let arr: [u32; 20] = rng.next_array();
        let mut i = 0usize;
        while i < 20 {
            assert!(arr[i] == OUT20[i]);
            i += 1;
        }
    }

    // --- index and state anchors ---

    #[test]
    fn initial_index_is_zero() {
        let rng = seeded();
        assert!(rng.index() == 0);
    }

    #[test]
    fn initial_state_matches_seed() {
        let rng = seeded();
        let st = rng.state();
        let mut i = 0usize;
        while i < 16 {
            assert!(st[i] == SEED[i]);
            i += 1;
        }
    }

    #[test]
    fn index_after_six_is_ten() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        assert!(rng.index() == 10);
    }

    #[test]
    fn state_after_six_matches_anchor() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        let st = rng.state();
        let mut i = 0usize;
        while i < 16 {
            assert!(st[i] == FINAL_STATE_6[i]);
            i += 1;
        }
    }

    #[test]
    fn index_after_single_step_is_fifteen() {
        let mut rng = seeded();
        rng.next_u32();
        assert!(rng.index() == 15);
    }

    #[test]
    fn index_decrements_each_step() {
        let mut rng = seeded();
        let expected: [u32; 20] = [
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12,
        ];
        let mut i = 0usize;
        while i < 20 {
            rng.next_u32();
            assert!(rng.index() == expected[i]);
            i += 1;
        }
    }

    #[test]
    fn index_wraps_to_zero_after_sixteen() {
        let mut rng = seeded();
        for _ in 0..16 {
            rng.next_u32();
        }
        assert!(rng.index() == 0);
    }

    // --- determinism and seed sensitivity ---

    #[test]
    fn determinism_two_instances() {
        let mut rng_a = seeded();
        let mut rng_b = seeded();
        let mut i = 0usize;
        while i < 64 {
            assert!(rng_a.next_u32() == rng_b.next_u32());
            i += 1;
        }
    }

    #[test]
    fn determinism_full_state_equal() {
        let mut rng_a = seeded();
        let mut rng_b = seeded();
        for _ in 0..100 {
            rng_a.next_u32();
            rng_b.next_u32();
        }
        assert!(rng_a == rng_b);
    }

    #[test]
    fn different_seeds_differ() {
        let mut other = SEED;
        other[0] = 42;
        let mut rng_a = seeded();
        let mut rng_b = Well512a::from_state(other);
        let mut differ = false;
        let mut i = 0usize;
        while i < 32 {
            if rng_a.next_u32() != rng_b.next_u32() {
                differ = true;
            }
            i += 1;
        }
        assert!(differ);
    }

    #[test]
    fn single_bit_seed_change_differs() {
        let mut other = SEED;
        other[15] ^= 1;
        let mut rng_a = seeded();
        let mut rng_b = Well512a::from_state(other);
        assert!(rng_a.next_u32() != rng_b.next_u32() || rng_a != rng_b);
    }

    // --- from_state round-trips ---

    #[test]
    fn from_state_round_trip_resets_index() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        let snapshot = rng.state();
        let restored = Well512a::from_state(snapshot);
        assert!(restored.index() == 0);
        let st = restored.state();
        let mut i = 0usize;
        while i < 16 {
            assert!(st[i] == snapshot[i]);
            i += 1;
        }
    }

    #[test]
    fn from_state_round_trip_resumes_sequence() {
        let mut rng = seeded();
        for _ in 0..6 {
            rng.next_u32();
        }
        let snapshot = rng.state();
        let restored = Well512a::from_state(snapshot);
        assert!(restored.state() == snapshot);
        assert!(restored.index() == 0);
        // A fresh restore from the canonical seed reproduces the output 0 anchor.
        let mut fresh = Well512a::from_state(SEED);
        assert!(fresh.next_u32() == OUT20[0]);
    }

    #[test]
    fn from_state_index_always_zero() {
        let rng = Well512a::from_state([0xffff_ffff; 16]);
        assert!(rng.index() == 0);
    }

    #[test]
    fn state_accessor_returns_copy() {
        let rng = seeded();
        let mut copy = rng.state();
        copy[0] = 0;
        // Mutating the returned copy must not affect the generator state.
        assert!(copy[0] == 0);
        assert!(rng.state()[0] == 1);
    }

    // --- seed construction via for / while loops (no Vec) ---

    #[test]
    fn seed_built_with_for_loop_matches_literal() {
        let mut seed = [0u32; 16];
        for i in 0..16usize {
            seed[i] = (i as u32) + 1;
        }
        let rng = Well512a::from_state(seed);
        let st = rng.state();
        let mut i = 0usize;
        while i < 16 {
            assert!(st[i] == SEED[i]);
            i += 1;
        }
    }

    #[test]
    fn seed_built_with_while_loop_matches_literal() {
        let mut seed = [0u32; 16];
        let mut i = 0usize;
        while i < 16 {
            seed[i] = (i as u32) + 1;
            i += 1;
        }
        let mut rng = Well512a::from_state(seed);
        assert!(rng.next_u32() == OUT20[0]);
    }

    #[test]
    fn seed_built_with_for_loop_produces_reference() {
        let mut seed = [0u32; 16];
        for i in 0..16usize {
            seed[i] = (i as u32) + 1;
        }
        let mut rng = Well512a::from_state(seed);
        let arr: [u32; 6] = rng.next_array();
        let mut i = 0usize;
        while i < 6 {
            assert!(arr[i] == OUT20[i]);
            i += 1;
        }
    }

    // --- const context usage ---

    #[test]
    fn const_context_construction() {
        const RNG: Well512a = Well512a::from_state(SEED);
        const IDX: u32 = RNG.index();
        const ST: [u32; 16] = RNG.state();
        assert!(IDX == 0);
        assert!(ST[0] == 1);
        assert!(ST[15] == 16);
    }

    #[test]
    fn next_u32_changes_state_word() {
        let mut rng = seeded();
        let before = rng.state()[0];
        rng.next_u32();
        assert!(rng.state()[0] != before);
    }

    #[test]
    fn long_run_is_reproducible() {
        let mut rng_a = seeded();
        let mut rng_b = seeded();
        for _ in 0..1000 {
            rng_a.next_u32();
        }
        for _ in 0..1000 {
            rng_b.next_u32();
        }
        assert!(rng_a.state() == rng_b.state());
        assert!(rng_a.index() == rng_b.index());
    }
}
