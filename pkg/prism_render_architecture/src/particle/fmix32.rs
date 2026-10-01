//! `fmix32`: the 32-bit finalizer from `MurmurHash3`.
//!
//! This module implements the avalanche-mixing finalizer `fmix32` used by the
//! `MurmurHash3` family of hash functions. It takes a single `u32` input and
//! scrambles it so that each input bit influences many output bits (the
//! avalanche property), using only pure integer operations: `XOR`, logical
//! shifts, and wrapping multiplications.
//!
//! # Algorithm
//!
//! The finalizer performs the following sequence on a `u32` value `x`:
//! 1. `x ^= x >> 16` (`XOR`-shift)
//! 2. `x *= 0x85eb_ca6b` (wrapping multiply)
//! 3. `x ^= x >> 13` (`XOR`-shift)
//! 4. `x *= 0xc2b2_ae35` (wrapping multiply)
//! 5. `x ^= x >> 16` (`XOR`-shift)
//!
//! The two magic constants are `0x85eb_ca6b` and `0xc2b2_ae35`.
//!
//! # Anchors
//!
//! Reference vectors (verified against `MurmurHash3`):
//! - `0x0000_0000` -> `0x0000_0000`
//! - `0x0000_0001` -> `0x514e_28b7`
//! - `0x0000_0002` -> `0x30f4_c306`
//! - `0xdead_beef` -> `0x0de5_c6a9`
//! - `0xffff_ffff` -> `0x81f1_6f39`
//!
//! Note that `0` is a fixed point: `fmix32(0) == 0`.

/// First mixing constant of the `MurmurHash3` 32-bit finalizer.
const C1: u32 = 0x85eb_ca6b;

/// Second mixing constant of the `MurmurHash3` 32-bit finalizer.
const C2: u32 = 0xc2b2_ae35;

/// Applies the `MurmurHash3` 32-bit finalizer (`fmix32`) to `x`.
///
/// This performs an avalanche mix of the input `u32` using `XOR`-shifts and
/// wrapping multiplications by the constants `0x85eb_ca6b` and `0xc2b2_ae35`.
///
/// # Examples
///
/// ```
/// # fn fmix32(mut x: u32) -> u32 {
/// #     x ^= x >> 16;
/// #     x = x.wrapping_mul(0x85eb_ca6b);
/// #     x ^= x >> 13;
/// #     x = x.wrapping_mul(0xc2b2_ae35);
/// #     x ^= x >> 16;
/// #     x
/// # }
/// assert!(fmix32(0) == 0);
/// assert!(fmix32(1) == 0x514e_28b7);
/// ```
pub fn fmix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(C1);
    x ^= x >> 13;
    x = x.wrapping_mul(C2);
    x ^= x >> 16;
    x
}

#[cfg(test)]
mod tests {
    use super::fmix32;

    // --- Five hard anchors ------------------------------------------------

    #[test]
    fn anchor_zero() {
        assert!(fmix32(0x0000_0000) == 0x0000_0000);
    }

    #[test]
    fn anchor_one() {
        assert!(fmix32(0x0000_0001) == 0x514e_28b7);
    }

    #[test]
    fn anchor_two() {
        assert!(fmix32(0x0000_0002) == 0x30f4_c306);
    }

    #[test]
    fn anchor_deadbeef() {
        assert!(fmix32(0xdead_beef) == 0x0de5_c6a9);
    }

    #[test]
    fn anchor_all_ones() {
        assert!(fmix32(0xffff_ffff) == 0x81f1_6f39);
    }

    // --- Zero is a fixed point -------------------------------------------

    #[test]
    fn zero_is_fixed_point() {
        assert!(fmix32(0) == 0);
    }

    #[test]
    fn zero_fixed_point_twice() {
        let once = fmix32(0);
        assert!(fmix32(once) == 0);
    }

    // --- Additional hardcoded input/output vectors -----------------------

    #[test]
    fn vector_three() {
        assert!(fmix32(0x0000_0003) == 0x85f0_b427);
    }

    #[test]
    fn vector_four() {
        assert!(fmix32(0x0000_0004) == 0x249c_b285);
    }

    #[test]
    fn vector_five() {
        assert!(fmix32(0x0000_0005) == 0xcc0d_53cd);
    }

    #[test]
    fn vector_ten() {
        assert!(fmix32(0x0000_000a) == 0xe925_0490);
    }

    #[test]
    fn vector_hundred() {
        assert!(fmix32(0x0000_0064) == 0xfdce_5cea);
    }

    #[test]
    fn vector_high_bit() {
        assert!(fmix32(0x8000_0000) == 0x6d3c_65a0);
    }

    #[test]
    fn vector_cafebabe() {
        assert!(fmix32(0xcafe_babe) == 0x79ff_04e8);
    }

    #[test]
    fn vector_12345678() {
        assert!(fmix32(0x1234_5678) == 0xe37c_d1bc);
    }

    #[test]
    fn vector_ffff0000() {
        assert!(fmix32(0xffff_0000) == 0x4ae3_8127);
    }

    #[test]
    fn vector_0000ffff() {
        assert!(fmix32(0x0000_ffff) == 0xa23b_ae67);
    }

    // --- Determinism: same input twice yields same output ----------------

    #[test]
    fn deterministic_zero() {
        assert!(fmix32(0) == fmix32(0));
    }

    #[test]
    fn deterministic_one() {
        assert!(fmix32(1) == fmix32(1));
    }

    #[test]
    fn deterministic_deadbeef() {
        assert!(fmix32(0xdead_beef) == fmix32(0xdead_beef));
    }

    #[test]
    fn deterministic_all_ones() {
        assert!(fmix32(0xffff_ffff) == fmix32(0xffff_ffff));
    }

    #[test]
    fn deterministic_sample_loop() {
        let mut i: u32 = 0;
        while i < 64 {
            let scaled = i.wrapping_mul(0x0100_1001);
            assert!(fmix32(scaled) == fmix32(scaled));
            i += 1;
        }
    }

    // --- Distinctness / separability samples -----------------------------

    #[test]
    fn distinct_zero_one() {
        assert!(fmix32(0) != fmix32(1));
    }

    #[test]
    fn distinct_one_two() {
        assert!(fmix32(1) != fmix32(2));
    }

    #[test]
    fn distinct_two_three() {
        assert!(fmix32(2) != fmix32(3));
    }

    #[test]
    fn distinct_adjacent_small() {
        let mut i: u32 = 0;
        while i < 128 {
            assert!(fmix32(i) != fmix32(i + 1));
            i += 1;
        }
    }

    #[test]
    fn distinct_deadbeef_cafebabe() {
        assert!(fmix32(0xdead_beef) != fmix32(0xcafe_babe));
    }

    #[test]
    fn distinct_high_low() {
        assert!(fmix32(0x8000_0000) != fmix32(0x0000_0001));
    }

    #[test]
    fn distinct_all_ones_zero() {
        assert!(fmix32(0xffff_ffff) != fmix32(0x0000_0000));
    }

    // --- Output range sanity ---------------------------------------------

    #[test]
    fn output_in_u32_range_zero() {
        let out = fmix32(0);
        assert!((u32::MIN..=u32::MAX).contains(&out));
    }

    #[test]
    fn output_in_u32_range_sample() {
        let out = fmix32(0xdead_beef);
        assert!((u32::MIN..=u32::MAX).contains(&out));
    }

    // --- Non-zero inputs produce non-zero outputs (sampled) --------------

    #[test]
    fn nonzero_input_nonzero_output_one() {
        assert!(fmix32(1) != 0);
    }

    #[test]
    fn nonzero_input_nonzero_output_deadbeef() {
        assert!(fmix32(0xdead_beef) != 0);
    }

    #[test]
    fn nonzero_small_inputs_nonzero_outputs() {
        let mut i: u32 = 1;
        while i < 100 {
            assert!(fmix32(i) != 0);
            i += 1;
        }
    }

    // --- Avalanche: single-bit flip changes output -----------------------

    #[test]
    fn single_bit_flip_changes_output() {
        let mut bit: u32 = 0;
        while bit < 32 {
            let base: u32 = 0;
            let flipped = base ^ (1u32 << bit);
            assert!(fmix32(base) != fmix32(flipped));
            bit += 1;
        }
    }

    #[test]
    fn single_bit_flip_from_deadbeef() {
        let base: u32 = 0xdead_beef;
        let mut bit: u32 = 0;
        while bit < 32 {
            let flipped = base ^ (1u32 << bit);
            assert!(fmix32(base) != fmix32(flipped));
            bit += 1;
        }
    }

    // --- Cross-check against inline reference implementation --------------

    #[test]
    fn matches_reference_impl_sample() {
        fn reference(mut x: u32) -> u32 {
            x ^= x >> 16;
            x = x.wrapping_mul(0x85eb_ca6b);
            x ^= x >> 13;
            x = x.wrapping_mul(0xc2b2_ae35);
            x ^= x >> 16;
            x
        }
        let mut i: u32 = 0;
        while i < 256 {
            let v = i.wrapping_mul(0x9e37_79b1);
            assert!(fmix32(v) == reference(v));
            i += 1;
        }
    }
}
