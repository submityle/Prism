//! Population count, `Hamming` weight/distance, and parity utilities.
//!
//! This module provides pure-integer, CPU golden-reference implementations of
//! bit-population counting and related distance/parity primitives. The primary
//! `popcount` implementation uses the classic `SWAR` (SIMD Within A Register)
//! parallel-bit-count algorithm with the standard masks `0x5555...`,
//! `0x3333...`, `0x0F0F...`, followed by a multiply by `0x0101...` to sum the
//! per-byte counts into the high byte.
//!
//! This is distinct from `de_bruijn_log2`, which finds the integer base-2
//! logarithm (the highest set bit position) of a value. In contrast, this
//! module *counts* how many bits are set (`popcount` / `Hamming` weight) and
//! how many bits differ between two values (`Hamming` distance via `XOR` +
//! `popcount`). It also exposes bit `parity` (the `XOR` of all bits).
//!
//! No floating point, no transcendental functions, and no `unsafe` are used.

/// `SWAR` population count for a `u32`.
///
/// Implements the classic parallel-bit-count algorithm using the masks
/// `0x55555555`, `0x33333333`, `0x0F0F0F0F`, and a final `wrapping_mul` by
/// `0x01010101` to accumulate the byte sums into the top byte. This is the
/// golden reference and intentionally does not call `count_ones`.
pub fn popcount_u32(x: u32) -> u32 {
    let mut v = x;
    v = v - ((v >> 1) & 0x5555_5555);
    v = (v & 0x3333_3333) + ((v >> 2) & 0x3333_3333);
    v = (v + (v >> 4)) & 0x0F0F_0F0F;
    (v.wrapping_mul(0x0101_0101) >> 24) & 0x3F
}

/// `SWAR` population count for a `u64`.
///
/// The `u64` analogue of [`popcount_u32`], using the 64-bit masks
/// `0x5555555555555555`, `0x3333333333333333`, `0x0F0F0F0F0F0F0F0F`, and a
/// final `wrapping_mul` by `0x0101010101010101`.
pub fn popcount_u64(x: u64) -> u32 {
    let mut v = x;
    v = v - ((v >> 1) & 0x5555_5555_5555_5555);
    v = (v & 0x3333_3333_3333_3333) + ((v >> 2) & 0x3333_3333_3333_3333);
    v = (v + (v >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    ((v.wrapping_mul(0x0101_0101_0101_0101) >> 56) & 0x7F) as u32
}

/// Kernighan-style sparse population count for a `u64`.
///
/// Repeatedly clears the lowest set bit (`v & (v - 1)`) and counts the number
/// of iterations. This runs in time proportional to the number of set bits and
/// serves as an independent cross-check against the `SWAR` `popcount`.
pub fn popcount_sparse(x: u64) -> u32 {
    let mut v = x;
    let mut count = 0u32;
    while v != 0 {
        v &= v - 1;
        count += 1;
    }
    count
}

/// `Hamming` distance between two `u32` values.
///
/// Defined as the `popcount` of their `XOR`: the number of bit positions in
/// which the two values differ.
pub fn hamming_distance_u32(a: u32, b: u32) -> u32 {
    popcount_u32(a ^ b)
}

/// `Hamming` distance between two `u64` values.
///
/// Defined as the `popcount` of their `XOR`: the number of bit positions in
/// which the two values differ.
pub fn hamming_distance_u64(a: u64, b: u64) -> u32 {
    popcount_u64(a ^ b)
}

/// `Hamming` weight of a byte slice.
///
/// Returns the total number of set bits across all bytes in `bytes`.
pub fn hamming_weight_slice(bytes: &[u8]) -> u64 {
    let mut total = 0u64;
    for b in bytes.iter() {
        total += popcount_u32(*b as u32) as u64;
    }
    total
}

/// `Hamming` distance between two byte slices.
///
/// Returns `None` if the slices have different lengths; otherwise returns the
/// total number of differing bits across all byte positions.
pub fn hamming_distance_slice(a: &[u8], b: &[u8]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let mut total = 0u64;
    let mut i = 0usize;
    while i < a.len() {
        total += popcount_u32((a[i] ^ b[i]) as u32) as u64;
        i += 1;
    }
    Some(total)
}

/// Bit `parity` of a `u32`.
///
/// Returns `true` when the number of set bits is odd (the `XOR` of all bits is
/// `1`), and `false` when it is even.
pub fn parity_u32(x: u32) -> bool {
    let mut v = x;
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    (v & 1) == 1
}

/// Bit `parity` of a `u64`.
///
/// Returns `true` when the number of set bits is odd (the `XOR` of all bits is
/// `1`), and `false` when it is even.
pub fn parity_u64(x: u64) -> bool {
    let mut v = x;
    v ^= v >> 32;
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    (v & 1) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny inline linear congruential generator for pseudo-random sweeps.
    ///
    /// Uses the Numerical Recipes constants; test-only, no external deps.
    #[cfg(test)]
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn popcount_u32_zero() {
        assert_eq!(popcount_u32(0), 0);
    }

    #[test]
    fn popcount_u64_zero() {
        assert_eq!(popcount_u64(0), 0);
    }

    #[test]
    fn popcount_u32_max_is_32() {
        assert_eq!(popcount_u32(u32::MAX), 32);
    }

    #[test]
    fn popcount_u64_max_is_64() {
        assert_eq!(popcount_u64(u64::MAX), 64);
    }

    #[test]
    fn popcount_sparse_zero() {
        assert_eq!(popcount_sparse(0), 0);
    }

    #[test]
    fn popcount_sparse_max_is_64() {
        assert_eq!(popcount_sparse(u64::MAX), 64);
    }

    #[test]
    fn popcount_u32_single_bits() {
        let mut i = 0u32;
        while i < 32 {
            let v = 1u32 << i;
            assert_eq!(popcount_u32(v), 1);
            i += 1;
        }
    }

    #[test]
    fn popcount_u64_single_bits() {
        let mut i = 0u32;
        while i < 64 {
            let v = 1u64 << i;
            assert_eq!(popcount_u64(v), 1);
            i += 1;
        }
    }

    #[test]
    fn popcount_sparse_single_bits() {
        let mut i = 0u32;
        while i < 64 {
            let v = 1u64 << i;
            assert_eq!(popcount_sparse(v), 1);
            i += 1;
        }
    }

    #[test]
    fn popcount_u32_alternating_5555() {
        assert_eq!(popcount_u32(0x5555_5555), 16);
    }

    #[test]
    fn popcount_u32_alternating_aaaa() {
        assert_eq!(popcount_u32(0xAAAA_AAAA), 16);
    }

    #[test]
    fn popcount_u64_alternating_5555() {
        assert_eq!(popcount_u64(0x5555_5555_5555_5555), 32);
    }

    #[test]
    fn popcount_u64_alternating_aaaa() {
        assert_eq!(popcount_u64(0xAAAA_AAAA_AAAA_AAAA), 32);
    }

    #[test]
    fn popcount_u32_matches_count_ones_sweep() {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut n = 0;
        while n < 2000 {
            let v = lcg_next(&mut state) as u32;
            assert_eq!(popcount_u32(v), v.count_ones());
            n += 1;
        }
    }

    #[test]
    fn popcount_u64_matches_count_ones_sweep() {
        let mut state = 0xDEAD_BEEF_CAFE_1234u64;
        let mut n = 0;
        while n < 2000 {
            let v = lcg_next(&mut state);
            assert_eq!(popcount_u64(v), v.count_ones());
            n += 1;
        }
    }

    #[test]
    fn swar_sparse_count_ones_agreement_sweep() {
        let mut state = 0x0BAD_F00D_1337_5EEDu64;
        let mut n = 0;
        while n < 2000 {
            let v = lcg_next(&mut state);
            let swar = popcount_u64(v);
            let sparse = popcount_sparse(v);
            let builtin = v.count_ones();
            assert_eq!(swar, sparse);
            assert_eq!(swar, builtin);
            n += 1;
        }
    }

    #[test]
    fn swar_u32_sparse_agreement_sweep() {
        let mut state = 0x00C0_FFEE_BEEF_0001u64;
        let mut n = 0;
        while n < 2000 {
            let v = lcg_next(&mut state) as u32;
            assert_eq!(popcount_u32(v), popcount_sparse(v as u64));
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_u32_self_zero() {
        let mut state = 0x5EED_1234_5678_9ABCu64;
        let mut n = 0;
        while n < 500 {
            let v = lcg_next(&mut state) as u32;
            assert_eq!(hamming_distance_u32(v, v), 0);
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_u64_self_zero() {
        let mut state = 0xABCD_1234_5678_9EEDu64;
        let mut n = 0;
        while n < 500 {
            let v = lcg_next(&mut state);
            assert_eq!(hamming_distance_u64(v, v), 0);
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_u32_symmetry() {
        let mut state = 0x1111_2222_3333_4444u64;
        let mut n = 0;
        while n < 1000 {
            let a = lcg_next(&mut state) as u32;
            let b = lcg_next(&mut state) as u32;
            assert_eq!(hamming_distance_u32(a, b), hamming_distance_u32(b, a));
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_u64_symmetry() {
        let mut state = 0x9999_8888_7777_6666u64;
        let mut n = 0;
        while n < 1000 {
            let a = lcg_next(&mut state);
            let b = lcg_next(&mut state);
            assert_eq!(hamming_distance_u64(a, b), hamming_distance_u64(b, a));
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_u32_known() {
        assert_eq!(hamming_distance_u32(0b0000, 0b1111), 4);
        assert_eq!(hamming_distance_u32(0x0000_0000, 0xFFFF_FFFF), 32);
        assert_eq!(hamming_distance_u32(0b1010, 0b0101), 4);
    }

    #[test]
    fn hamming_distance_u64_known() {
        assert_eq!(hamming_distance_u64(0, u64::MAX), 64);
        assert_eq!(
            hamming_distance_u64(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA),
            64
        );
    }

    #[test]
    fn hamming_distance_u32_matches_xor_popcount_sweep() {
        let mut state = 0x4242_4242_2424_2424u64;
        let mut n = 0;
        while n < 1000 {
            let a = lcg_next(&mut state) as u32;
            let b = lcg_next(&mut state) as u32;
            assert_eq!(hamming_distance_u32(a, b), (a ^ b).count_ones());
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_u64_matches_xor_popcount_sweep() {
        let mut state = 0x7A7A_7A7A_5B5B_5B5Bu64;
        let mut n = 0;
        while n < 1000 {
            let a = lcg_next(&mut state);
            let b = lcg_next(&mut state);
            assert_eq!(hamming_distance_u64(a, b), (a ^ b).count_ones());
            n += 1;
        }
    }

    #[test]
    fn parity_u32_known() {
        assert!(!parity_u32(0));
        assert!(parity_u32(1));
        assert!(!parity_u32(0b11));
        assert!(parity_u32(0b111));
        assert!(!parity_u32(u32::MAX));
    }

    #[test]
    fn parity_u64_known() {
        assert!(!parity_u64(0));
        assert!(parity_u64(1));
        assert!(!parity_u64(0b11));
        assert!(parity_u64(0b111));
        assert!(!parity_u64(u64::MAX));
    }

    #[test]
    fn parity_u32_matches_popcount_lsb_sweep() {
        let mut state = 0x3C3C_3C3C_C3C3_C3C3u64;
        let mut n = 0;
        while n < 2000 {
            let v = lcg_next(&mut state) as u32;
            let expected = (popcount_u32(v) & 1) == 1;
            assert_eq!(parity_u32(v), expected);
            n += 1;
        }
    }

    #[test]
    fn parity_u64_matches_popcount_lsb_sweep() {
        let mut state = 0x1E1E_1E1E_E1E1_E1E1u64;
        let mut n = 0;
        while n < 2000 {
            let v = lcg_next(&mut state);
            let expected = (popcount_u64(v) & 1) == 1;
            assert_eq!(parity_u64(v), expected);
            n += 1;
        }
    }

    #[test]
    fn parity_u32_single_bits_all_odd() {
        let mut i = 0u32;
        while i < 32 {
            assert!(parity_u32(1u32 << i));
            i += 1;
        }
    }

    #[test]
    fn parity_u64_single_bits_all_odd() {
        let mut i = 0u32;
        while i < 64 {
            assert!(parity_u64(1u64 << i));
            i += 1;
        }
    }

    #[test]
    fn hamming_weight_slice_empty() {
        let empty: &[u8] = &[];
        assert_eq!(hamming_weight_slice(empty), 0);
    }

    #[test]
    fn hamming_weight_slice_known() {
        assert_eq!(hamming_weight_slice(&[0x00, 0x00]), 0);
        assert_eq!(hamming_weight_slice(&[0xFF]), 8);
        assert_eq!(hamming_weight_slice(&[0xFF, 0xFF, 0xFF, 0xFF]), 32);
        assert_eq!(hamming_weight_slice(&[0x0F, 0xF0]), 8);
    }

    #[test]
    fn hamming_weight_slice_matches_elementwise_sweep() {
        let mut state = 0x6060_6060_0606_0606u64;
        let mut n = 0;
        while n < 500 {
            let b0 = lcg_next(&mut state) as u8;
            let b1 = lcg_next(&mut state) as u8;
            let b2 = lcg_next(&mut state) as u8;
            let bytes = [b0, b1, b2];
            let expected = (b0.count_ones() + b1.count_ones() + b2.count_ones()) as u64;
            assert_eq!(hamming_weight_slice(&bytes), expected);
            n += 1;
        }
    }

    #[test]
    fn hamming_distance_slice_length_mismatch_is_none() {
        let a: &[u8] = &[0x00, 0x01];
        let b: &[u8] = &[0x00, 0x01, 0x02];
        assert_eq!(hamming_distance_slice(a, b), None);
    }

    #[test]
    fn hamming_distance_slice_empty_is_zero() {
        let a: &[u8] = &[];
        let b: &[u8] = &[];
        assert_eq!(hamming_distance_slice(a, b), Some(0));
    }

    #[test]
    fn hamming_distance_slice_self_zero() {
        let a: &[u8] = &[0x12, 0x34, 0x56, 0x78];
        assert_eq!(hamming_distance_slice(a, a), Some(0));
    }

    #[test]
    fn hamming_distance_slice_known() {
        let a: &[u8] = &[0x00, 0x00];
        let b: &[u8] = &[0xFF, 0xFF];
        assert_eq!(hamming_distance_slice(a, b), Some(16));
        let c: &[u8] = &[0b1010_1010];
        let d: &[u8] = &[0b0101_0101];
        assert_eq!(hamming_distance_slice(c, d), Some(8));
    }

    #[test]
    fn hamming_distance_slice_symmetry_sweep() {
        let mut state = 0x0F0F_F0F0_0F0F_F0F0u64;
        let mut n = 0;
        while n < 500 {
            let a0 = lcg_next(&mut state) as u8;
            let a1 = lcg_next(&mut state) as u8;
            let b0 = lcg_next(&mut state) as u8;
            let b1 = lcg_next(&mut state) as u8;
            let a = [a0, a1];
            let b = [b0, b1];
            assert_eq!(
                hamming_distance_slice(&a, &b),
                hamming_distance_slice(&b, &a)
            );
            n += 1;
        }
    }
}
