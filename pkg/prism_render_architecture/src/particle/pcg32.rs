//! `Pcg32`: Melissa O'Neill's `PCG-XSH-RR` 64/32 permuted-congruential
//! generator, a compact `PRNG` that advances a `64`-bit linear-congruential
//! state and emits a well-distributed `32`-bit word per draw.
//!
//! A `PCG` (permuted congruential generator) layers an output *permutation* on
//! top of a classic `LCG` (linear congruential generator). The underlying
//! `LCG` is statistically weak in its low bits, so `PCG` discards that weakness
//! by applying an xor-shift followed by a data-dependent rotation to the state
//! *before* advancing it. The `XSH-RR` variant used here computes an
//! xor-shift-high word and then performs a `rotate-right` whose amount is drawn
//! from the top bits of the same state, which decorrelates successive outputs
//! far better than the raw `LCG` could.
//!
//! Because the increment is a fixed odd constant per stream and the state
//! update is a single wrapping multiply-add, the whole sequence is a
//! deterministic function of the seed: the same seed always replays the same
//! `u32` stream. That bit-for-bit reproducibility is exactly what a particle
//! simulation needs when a frame must match between the `CPU` reference path
//! and a future `GPU` implementation.
//!
//! Two seed words are supplied. `init_state` chooses the starting point within
//! a stream, while `init_seq` selects *which* stream: the sequence constant is
//! `(init_seq << 1) | 1`, forced odd so the `LCG` has full period. Distinct
//! `init_seq` values therefore produce independent streams even from the same
//! `init_state`.
//!
//! Every generation step is pure integer arithmetic: `wrapping_mul`,
//! `wrapping_add`, exclusive-or, right shift, and a `u32::rotate_right`. No
//! floating point, transcendental function, or rounding appears anywhere.
//!
//! Scope: `PCG-XSH-RR` is a fast, non-cryptographic generator. Its state is
//! recoverable from a handful of outputs, so it must never be used for security
//! or key material; it exists purely for reproducible, high-throughput
//! simulation randomness.

/// The `LCG` multiplier used by the 64-bit `PCG` state advance, Melissa
/// O'Neill's canonical constant `6364136223846793005`.
///
/// This odd multiplier gives the underlying linear-congruential recurrence its
/// full `2^64` period; the output permutation then repairs the low-bit
/// weakness that the bare multiplier would otherwise leave behind.
const MUL: u64 = 6_364_136_223_846_793_005;

/// A `PCG-XSH-RR` 64/32 generator: a `64`-bit `LCG` register paired with the
/// odd per-stream increment that selects which sequence it walks.
///
/// Advancing the register and permuting the previous value yields one `u32`
/// per call to [`Pcg32::next_u32`]. The generator is fully deterministic: equal
/// `(init_state, init_seq)` pairs replay identical streams.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pcg32 {
    /// The `64`-bit `LCG` state register, advanced by one multiply-add per draw.
    state: u64,
    /// The odd per-stream increment; its low bit is forced to `1` so the `LCG`
    /// attains full period and distinct values yield independent streams.
    inc: u64,
}

impl Pcg32 {
    /// Seeds a generator from a starting `init_state` and a stream selector
    /// `init_seq`, following O'Neill's canonical initialization.
    ///
    /// The increment is derived as `(init_seq << 1) | 1`, guaranteeing an odd
    /// value and thus full `LCG` period. The register is then zeroed, stepped
    /// once, offset by `init_state`, and stepped once more so that both seed
    /// words are fully mixed into the state before the first output.
    #[must_use]
    pub fn seed_with(init_state: u64, init_seq: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (init_seq << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(init_state);
        rng.step();
        rng
    }

    /// Advances the internal `LCG` register by one multiply-add step.
    ///
    /// This is the raw state transition `state = state * MUL + inc` under
    /// wrapping arithmetic; it is shared by seeding and by every draw.
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MUL).wrapping_add(self.inc);
    }

    /// Draws the next `32`-bit output and advances the generator.
    ///
    /// The previous register value is permuted with the `XSH-RR` scheme: an
    /// xor-shift forms the high output word, and the top bits of the old state
    /// drive a `rotate-right`. The register is advanced with the same
    /// multiply-add used during seeding, so the stream is deterministic.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MUL).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The six canonical reference outputs of `seed_with(42, 54)`.
    const REFERENCE: [u32; 6] = [
        0xA15C_02B7,
        0x7B47_F409,
        0xBA1D_3330,
        0x83D2_F293,
        0xBFA4_784B,
        0xCBED_606E,
    ];

    #[test]
    fn mul_constant_matches_oneill() {
        assert_eq!(MUL, 6_364_136_223_846_793_005);
    }

    #[test]
    fn mul_constant_is_odd() {
        assert_eq!(MUL & 1, 1);
    }

    #[test]
    fn seed_forces_odd_increment() {
        let rng = Pcg32::seed_with(42, 54);
        assert_eq!(rng.inc & 1, 1);
    }

    #[test]
    fn seed_increment_from_init_seq() {
        let rng = Pcg32::seed_with(0, 54);
        assert_eq!(rng.inc, (54u64 << 1) | 1);
    }

    #[test]
    fn seed_increment_zero_seq() {
        let rng = Pcg32::seed_with(123, 0);
        assert_eq!(rng.inc, 1);
    }

    #[test]
    fn reference_vector_0() {
        let mut rng = Pcg32::seed_with(42, 54);
        assert_eq!(rng.next_u32(), REFERENCE[0]);
    }

    #[test]
    fn reference_vector_1() {
        let mut rng = Pcg32::seed_with(42, 54);
        rng.next_u32();
        assert_eq!(rng.next_u32(), REFERENCE[1]);
    }

    #[test]
    fn reference_vector_2() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..2 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REFERENCE[2]);
    }

    #[test]
    fn reference_vector_3() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..3 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REFERENCE[3]);
    }

    #[test]
    fn reference_vector_4() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..4 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REFERENCE[4]);
    }

    #[test]
    fn reference_vector_5() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..5 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), REFERENCE[5]);
    }

    #[test]
    fn reference_vector_full_sequence() {
        let mut rng = Pcg32::seed_with(42, 54);
        for &expected in &REFERENCE {
            assert_eq!(rng.next_u32(), expected);
        }
    }

    #[test]
    fn reference_vector_decimal_0() {
        let mut rng = Pcg32::seed_with(42, 54);
        assert_eq!(rng.next_u32(), 2_707_161_783);
    }

    #[test]
    fn reference_vector_decimal_1() {
        let mut rng = Pcg32::seed_with(42, 54);
        rng.next_u32();
        assert_eq!(rng.next_u32(), 2_068_313_097);
    }

    #[test]
    fn reference_vector_decimal_2() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..2 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 3_122_475_824);
    }

    #[test]
    fn reference_vector_decimal_3() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..3 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 2_211_639_955);
    }

    #[test]
    fn reference_vector_decimal_4() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..4 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 3_215_226_955);
    }

    #[test]
    fn reference_vector_decimal_5() {
        let mut rng = Pcg32::seed_with(42, 54);
        for _ in 0..5 {
            rng.next_u32();
        }
        assert_eq!(rng.next_u32(), 3_421_331_566);
    }

    #[test]
    fn decimal_matches_hex() {
        assert_eq!(REFERENCE[0], 2_707_161_783);
        assert_eq!(REFERENCE[1], 2_068_313_097);
        assert_eq!(REFERENCE[2], 3_122_475_824);
        assert_eq!(REFERENCE[3], 2_211_639_955);
        assert_eq!(REFERENCE[4], 3_215_226_955);
        assert_eq!(REFERENCE[5], 3_421_331_566);
    }

    #[test]
    fn deterministic_same_seed() {
        let mut a = Pcg32::seed_with(42, 54);
        let mut b = Pcg32::seed_with(42, 54);
        for _ in 0..256 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn deterministic_other_seed() {
        let mut a = Pcg32::seed_with(0xDEAD_BEEF, 7);
        let mut b = Pcg32::seed_with(0xDEAD_BEEF, 7);
        for _ in 0..256 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn clone_replays_identically() {
        let mut a = Pcg32::seed_with(99, 1);
        for _ in 0..10 {
            a.next_u32();
        }
        let mut b = a.clone();
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seq_different_stream() {
        let mut a = Pcg32::seed_with(42, 54);
        let mut b = Pcg32::seed_with(42, 55);
        let mut differ = false;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                differ = true;
                break;
            }
        }
        assert!(differ);
    }

    #[test]
    fn different_seq_many_streams_differ() {
        for seq in 0..16u64 {
            let mut a = Pcg32::seed_with(1, seq);
            let mut b = Pcg32::seed_with(1, seq + 1);
            let mut differ = false;
            for _ in 0..64 {
                if a.next_u32() != b.next_u32() {
                    differ = true;
                    break;
                }
            }
            assert!(differ, "streams {seq} and {} should differ", seq + 1);
        }
    }

    #[test]
    fn different_state_different_output() {
        let mut a = Pcg32::seed_with(10, 54);
        let mut b = Pcg32::seed_with(11, 54);
        let mut differ = false;
        for _ in 0..32 {
            if a.next_u32() != b.next_u32() {
                differ = true;
                break;
            }
        }
        assert!(differ);
    }

    #[test]
    fn not_all_zero() {
        let mut rng = Pcg32::seed_with(42, 54);
        let mut any_nonzero = false;
        for _ in 0..64 {
            if rng.next_u32() != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn not_all_equal() {
        let mut rng = Pcg32::seed_with(42, 54);
        let first = rng.next_u32();
        let mut varied = false;
        for _ in 0..64 {
            if rng.next_u32() != first {
                varied = true;
                break;
            }
        }
        assert!(varied);
    }

    #[test]
    fn seed_zero_state_zero_seq_produces_values() {
        let mut rng = Pcg32::seed_with(0, 0);
        let mut any_nonzero = false;
        for _ in 0..64 {
            if rng.next_u32() != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn advances_state_each_draw() {
        let mut rng = Pcg32::seed_with(7, 7);
        let before = rng.state;
        rng.next_u32();
        assert_ne!(rng.state, before);
    }

    #[test]
    fn step_matches_manual_recurrence() {
        let mut rng = Pcg32::seed_with(5, 9);
        let expected = rng.state.wrapping_mul(MUL).wrapping_add(rng.inc);
        rng.step();
        assert_eq!(rng.state, expected);
    }

    #[test]
    fn next_u32_permutation_formula() {
        let mut rng = Pcg32::seed_with(42, 54);
        let old = rng.state;
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        let expected = xorshifted.rotate_right(rot);
        assert_eq!(rng.next_u32(), expected);
    }

    #[test]
    fn rotation_amount_within_range() {
        // The rotate amount is drawn from the top 5 bits, so it is always < 32.
        let rng = Pcg32::seed_with(42, 54);
        let rot = (rng.state >> 59) as u32;
        assert!(rot < 32);
    }

    #[test]
    fn large_state_seed_mixes() {
        let mut rng = Pcg32::seed_with(u64::MAX, u64::MAX);
        let mut any_nonzero = false;
        for _ in 0..64 {
            if rng.next_u32() != 0 {
                any_nonzero = true;
                break;
            }
        }
        assert!(any_nonzero);
    }

    #[test]
    fn distinct_outputs_in_window() {
        let mut rng = Pcg32::seed_with(42, 54);
        let mut seen = [0u32; 32];
        for slot in &mut seen {
            *slot = rng.next_u32();
        }
        let mut distinct = 0;
        for i in 0..seen.len() {
            if !seen[..i].contains(&seen[i]) {
                distinct += 1;
            }
        }
        // Expect essentially all values unique over such a short window.
        assert!(distinct >= 30);
    }

    #[test]
    fn high_and_low_bits_both_vary() {
        let mut rng = Pcg32::seed_with(42, 54);
        let mut high_set = false;
        let mut high_clear = false;
        let mut low_set = false;
        let mut low_clear = false;
        for _ in 0..64 {
            let v = rng.next_u32();
            if v & 0x8000_0000 != 0 {
                high_set = true;
            } else {
                high_clear = true;
            }
            if v & 1 != 0 {
                low_set = true;
            } else {
                low_clear = true;
            }
        }
        assert!(high_set && high_clear && low_set && low_clear);
    }

    #[test]
    fn equality_reflects_internal_state() {
        let a = Pcg32::seed_with(3, 3);
        let b = Pcg32::seed_with(3, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_for_different_seeds() {
        let a = Pcg32::seed_with(3, 3);
        let b = Pcg32::seed_with(4, 3);
        assert_ne!(a, b);
    }

    #[test]
    fn resume_after_partial_draws() {
        let mut full = Pcg32::seed_with(42, 54);
        let mut tail: [u32; 6] = [0; 6];
        for _ in 0..6 {
            full.next_u32();
        }
        for slot in &mut tail {
            *slot = full.next_u32();
        }
        let mut again = Pcg32::seed_with(42, 54);
        for _ in 0..6 {
            again.next_u32();
        }
        for &expected in &tail {
            assert_eq!(again.next_u32(), expected);
        }
    }

    #[test]
    fn seq_shift_does_not_overflow_low_bit() {
        let rng = Pcg32::seed_with(0, 0x7FFF_FFFF_FFFF_FFFF);
        assert_eq!(rng.inc & 1, 1);
    }

    #[test]
    fn two_independent_streams_rarely_collide() {
        let mut a = Pcg32::seed_with(0, 1);
        let mut b = Pcg32::seed_with(0, 2);
        let mut collisions = 0;
        for _ in 0..256 {
            if a.next_u32() == b.next_u32() {
                collisions += 1;
            }
        }
        // Independent 32-bit streams should almost never collide over 256 draws.
        assert!(collisions <= 2);
    }

    #[test]
    fn long_run_stays_deterministic() {
        let mut a = Pcg32::seed_with(0x1234_5678, 0x9ABC);
        let mut b = Pcg32::seed_with(0x1234_5678, 0x9ABC);
        for _ in 0..4096 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }
}
