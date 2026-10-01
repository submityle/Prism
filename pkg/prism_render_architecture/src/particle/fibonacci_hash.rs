//! Fibonacci (Knuth multiplicative) hashing for hash-table bucket indexing.
//!
//! This module implements Knuth's multiplicative hashing (TAOCP vol. 3,
//! section 6.4): multiply the `key` by a fixed-point approximation of the
//! golden ratio and keep the **high** bits of the product as a bucket index
//! into a `2^bits` table. The magic constants are the golden-ratio fractions
//! scaled to word width: `0x9E3779B9` for `u32` (decimal `2654435769`) and
//! `0x9E3779B97F4A7C15` for `u64`.
//!
//! # Scope and boundaries (read before using)
//!
//! - This is a *bucket-index* hash for hash-table slot location, **not** an
//!   avalanche bit-mixer. Only the high bits of the product are usable; the
//!   low bits have poor statistical quality because multiplication only
//!   propagates influence upward. Always keep the high `bits` via
//!   `fib_hash_u32_to_bits` / `fib_hash_u64_to_bits`, never mask the low bits.
//! - For strong avalanche / finalization (every output bit depends on every
//!   input bit) use `splitmix64`, `wang_hash`, `murmur3`, or `fnv1a` instead.
//!   Those are distinct modules with distinct contracts; do not substitute
//!   this `fib_hash` for them, and do not substitute them here.
//! - All functions in this module are pure integer arithmetic on `u32` / `u64`.
//!   There is no `f32`, no transcendental math, and no allocation.
//!
//! # Why the high bits
//!
//! The product `key * C` wraps modulo `2^w`. Knuth's analysis shows that the
//! most-significant `bits` of that product are well distributed across
//! `0..2^bits` for consecutive keys because `C / 2^w` approximates the golden
//! ratio, the irrational that is hardest to approximate by rationals. Shifting
//! right by `w - bits` keeps exactly those high lanes.

/// Golden-ratio multiplier for 32-bit Knuth multiplicative hashing.
///
/// `0x9E3779B9` is `round(2^32 / phi)` where `phi` is the golden ratio; its
/// decimal value is `2654435769`.
pub const FIB_HASH_U32_CONST: u32 = 0x9E37_79B9;

/// Golden-ratio multiplier for 64-bit Knuth multiplicative hashing.
///
/// `0x9E3779B97F4A7C15` is the 64-bit golden-ratio fraction (the same odd
/// gamma used by `splitmix64`'s increment); its decimal value is
/// `11400714819323198485`.
pub const FIB_HASH_U64_CONST: u64 = 0x9E37_79B9_7F4A_7C15;

/// Multiplies a `u32` `key` by the golden-ratio constant and returns the full
/// 32-bit low product (`wrapping_mul`).
///
/// This is the raw multiplicative hash. Only its **high** bits are well
/// distributed; use `fib_hash_u32_to_bits` to obtain a bucket index.
#[must_use]
pub const fn fib_hash_u32(key: u32) -> u32 {
    key.wrapping_mul(FIB_HASH_U32_CONST)
}

/// Hashes a `u32` `key` into a `2^bits` bucket index by keeping the high `bits`
/// of the multiplicative product: `(key * C) >> (32 - bits)`.
///
/// `bits` must be in `1..=32`. When `bits == 32` there is no shift and the full
/// product is returned. The result is always in `0..2^bits` (for `bits < 32`).
#[must_use]
pub fn fib_hash_u32_to_bits(key: u32, bits: u32) -> u32 {
    debug_assert!(
        (1..=32).contains(&bits),
        "fib_hash_u32_to_bits: bits must be in 1..=32"
    );
    fib_hash_u32(key) >> (32 - bits)
}

/// Multiplies a `u64` `key` by the golden-ratio constant and returns the full
/// 64-bit low product (`wrapping_mul`).
///
/// As with the `u32` variant, only the **high** bits are well distributed; use
/// `fib_hash_u64_to_bits` to obtain a bucket index.
#[must_use]
pub const fn fib_hash_u64(key: u64) -> u64 {
    key.wrapping_mul(FIB_HASH_U64_CONST)
}

/// Hashes a `u64` `key` into a `2^bits` bucket index by keeping the high `bits`
/// of the multiplicative product: `(key * C) >> (64 - bits)`.
///
/// `bits` must be in `1..=64`. When `bits == 64` there is no shift and the full
/// product is returned. The result is always in `0..2^bits` (for `bits < 64`).
#[must_use]
pub fn fib_hash_u64_to_bits(key: u64, bits: u32) -> u64 {
    debug_assert!(
        (1..=64).contains(&bits),
        "fib_hash_u64_to_bits: bits must be in 1..=64"
    );
    fib_hash_u64(key) >> (64 - bits)
}

/// Folds a new `u32` `key` into an existing `hash` to build a composite-key
/// hash, in the spirit of `boost::hash_combine` but using the Knuth golden
/// constant.
///
/// This mixes the multiplicative hash of `key` with shifted copies of the
/// accumulator so the combination is order-dependent (`combine(a, b)` differs
/// from `combine(b, a)` in general). It is a *combiner*, not an avalanche
/// finalizer: feed its result through a dedicated mixer if strong avalanche is
/// required downstream.
#[must_use]
pub const fn fib_hash_combine(hash: u32, key: u32) -> u32 {
    let mixed = fib_hash_u32(key);
    let folded = mixed
        .wrapping_add(FIB_HASH_U32_CONST)
        .wrapping_add(hash << 6)
        .wrapping_add(hash >> 2);
    hash ^ folded
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- constants / reference values -----------------------------------

    #[test]
    fn u32_const_value() {
        assert_eq!(FIB_HASH_U32_CONST, 0x9E37_79B9);
        assert_eq!(FIB_HASH_U32_CONST, 2_654_435_769);
    }

    #[test]
    fn u64_const_value() {
        assert_eq!(FIB_HASH_U64_CONST, 0x9E37_79B9_7F4A_7C15);
        assert_eq!(FIB_HASH_U64_CONST, 11_400_714_819_323_198_485);
    }

    // ---- fib_hash_u32 raw product ---------------------------------------

    #[test]
    fn u32_key_zero() {
        // 0 * C == 0.
        assert_eq!(fib_hash_u32(0), 0);
    }

    #[test]
    fn u32_key_one() {
        // 1 * C == C.
        assert_eq!(fib_hash_u32(1), 0x9E37_79B9);
    }

    #[test]
    fn u32_key_two() {
        // 2 * 0x9E3779B9 = 0x13C6EF372, low 32 bits = 0x3C6EF372.
        assert_eq!(fib_hash_u32(2), 0x3C6E_F372);
        assert_eq!(fib_hash_u32(2), 1_013_904_242);
    }

    #[test]
    fn u32_key_max() {
        // (2^32 - 1) * C == -C mod 2^32 == 2^32 - C == 0x61C88647.
        assert_eq!(fib_hash_u32(u32::MAX), 0x61C8_8647);
        assert_eq!(fib_hash_u32(u32::MAX), 1_640_531_527);
    }

    #[test]
    fn u32_consecutive_differ_by_const() {
        // Linearity: (k + 1) * C - k * C == C (wrapping).
        for key in [0u32, 1, 2, 1000, 0xFFFF_FFFE, u32::MAX] {
            let diff = fib_hash_u32(key.wrapping_add(1)).wrapping_sub(fib_hash_u32(key));
            assert_eq!(diff, FIB_HASH_U32_CONST);
        }
    }

    #[test]
    fn u32_determinism() {
        for key in [0u32, 1, 42, 12_345, 0xDEAD_BEEF, u32::MAX] {
            assert_eq!(fib_hash_u32(key), fib_hash_u32(key));
        }
    }

    // ---- fib_hash_u32_to_bits -------------------------------------------

    #[test]
    fn u32_to_bits_bits32_identity() {
        // bits == 32 means no shift: identical to the raw product.
        for key in [0u32, 1, 2, 7, 255, 65_535, u32::MAX] {
            assert_eq!(fib_hash_u32_to_bits(key, 32), fib_hash_u32(key));
        }
    }

    #[test]
    fn u32_to_bits_key_one_16() {
        // 0x9E3779B9 >> 16 == 0x9E37.
        assert_eq!(fib_hash_u32_to_bits(1, 16), 0x9E37);
        assert_eq!(fib_hash_u32_to_bits(1, 16), 40_503);
    }

    #[test]
    fn u32_to_bits_key_one_8() {
        // 0x9E3779B9 >> 24 == 0x9E.
        assert_eq!(fib_hash_u32_to_bits(1, 8), 0x9E);
        assert_eq!(fib_hash_u32_to_bits(1, 8), 158);
    }

    #[test]
    fn u32_to_bits_key_one_4() {
        // 0x9E3779B9 >> 28 == 0x9.
        assert_eq!(fib_hash_u32_to_bits(1, 4), 0x9);
    }

    #[test]
    fn u32_to_bits_key_one_1() {
        // Top bit of 0x9E3779B9 is set.
        assert_eq!(fib_hash_u32_to_bits(1, 1), 1);
    }

    #[test]
    fn u32_to_bits_key_two_16() {
        // 0x3C6EF372 >> 16 == 0x3C6E.
        assert_eq!(fib_hash_u32_to_bits(2, 16), 0x3C6E);
        assert_eq!(fib_hash_u32_to_bits(2, 16), 15_470);
    }

    #[test]
    fn u32_to_bits_key_two_1() {
        // Top bit of 0x3C6EF372 is clear.
        assert_eq!(fib_hash_u32_to_bits(2, 1), 0);
    }

    #[test]
    fn u32_to_bits_key_max_16() {
        // 0x61C88647 >> 16 == 0x61C8.
        assert_eq!(fib_hash_u32_to_bits(u32::MAX, 16), 0x61C8);
        assert_eq!(fib_hash_u32_to_bits(u32::MAX, 16), 25_032);
    }

    #[test]
    fn u32_to_bits_key_max_1() {
        // Top bit of 0x61C88647 is clear.
        assert_eq!(fib_hash_u32_to_bits(u32::MAX, 1), 0);
    }

    #[test]
    fn u32_to_bits_zero_key_all_zero() {
        // key == 0 maps to bucket 0 for every width.
        for bits in 1u32..=32 {
            assert_eq!(fib_hash_u32_to_bits(0, bits), 0);
        }
    }

    #[test]
    fn u32_to_bits_range_bound() {
        // For bits < 32 the index must stay inside 0..2^bits.
        for key in [0u32, 1, 2, 12_345, 0xABCD_1234, u32::MAX] {
            for bits in 1u32..=31 {
                let idx = fib_hash_u32_to_bits(key, bits);
                assert!(idx < (1u32 << bits));
            }
        }
    }

    #[test]
    fn u32_to_bits_matches_manual_shift() {
        for key in [0u32, 1, 2, 7, 255, 1024, 0xDEAD_BEEF, u32::MAX] {
            for bits in 1u32..=32 {
                let expected = fib_hash_u32(key) >> (32 - bits);
                assert_eq!(fib_hash_u32_to_bits(key, bits), expected);
            }
        }
    }

    #[test]
    fn u32_to_bits_determinism() {
        for key in [0u32, 1, 999, 0x1234_5678, u32::MAX] {
            for bits in 1u32..=32 {
                assert_eq!(
                    fib_hash_u32_to_bits(key, bits),
                    fib_hash_u32_to_bits(key, bits)
                );
            }
        }
    }

    #[test]
    fn u32_bits1_equals_top_bit() {
        for key in [0u32, 1, 2, 3, 100, 0xFFFF_0000, u32::MAX] {
            let top = fib_hash_u32(key) >> 31;
            assert_eq!(fib_hash_u32_to_bits(key, 1), top);
        }
    }

    // ---- fib_hash_u64 raw product ---------------------------------------

    #[test]
    fn u64_key_zero() {
        assert_eq!(fib_hash_u64(0), 0);
    }

    #[test]
    fn u64_key_one() {
        assert_eq!(fib_hash_u64(1), 0x9E37_79B9_7F4A_7C15);
    }

    #[test]
    fn u64_key_two() {
        // 2 * 0x9E3779B97F4A7C15 low 64 bits = 0x3C6EF372FE94F82A.
        assert_eq!(fib_hash_u64(2), 0x3C6E_F372_FE94_F82A);
    }

    #[test]
    fn u64_key_max() {
        // (2^64 - 1) * C == 2^64 - C == 0x61C8864680B583EB.
        assert_eq!(fib_hash_u64(u64::MAX), 0x61C8_8646_80B5_83EB);
    }

    #[test]
    fn u64_consecutive_differ_by_const() {
        for key in [0u64, 1, 2, 1000, u64::MAX - 1, u64::MAX] {
            let diff = fib_hash_u64(key.wrapping_add(1)).wrapping_sub(fib_hash_u64(key));
            assert_eq!(diff, FIB_HASH_U64_CONST);
        }
    }

    #[test]
    fn u64_determinism() {
        for key in [0u64, 1, 42, 0x0123_4567_89AB_CDEF, u64::MAX] {
            assert_eq!(fib_hash_u64(key), fib_hash_u64(key));
        }
    }

    // ---- fib_hash_u64_to_bits -------------------------------------------

    #[test]
    fn u64_to_bits_bits64_identity() {
        for key in [0u64, 1, 2, 255, 0xDEAD_BEEF_CAFE_F00D, u64::MAX] {
            assert_eq!(fib_hash_u64_to_bits(key, 64), fib_hash_u64(key));
        }
    }

    #[test]
    fn u64_to_bits_key_one_32() {
        // 0x9E3779B97F4A7C15 >> 32 == 0x9E3779B9.
        assert_eq!(fib_hash_u64_to_bits(1, 32), 0x9E37_79B9);
        assert_eq!(fib_hash_u64_to_bits(1, 32), 2_654_435_769);
    }

    #[test]
    fn u64_to_bits_key_one_16() {
        // 0x9E3779B97F4A7C15 >> 48 == 0x9E37.
        assert_eq!(fib_hash_u64_to_bits(1, 16), 0x9E37);
    }

    #[test]
    fn u64_to_bits_key_one_1() {
        // Top bit of 0x9E3779B9... is set.
        assert_eq!(fib_hash_u64_to_bits(1, 1), 1);
    }

    #[test]
    fn u64_to_bits_zero_key_all_zero() {
        for bits in 1u32..=64 {
            assert_eq!(fib_hash_u64_to_bits(0, bits), 0);
        }
    }

    #[test]
    fn u64_to_bits_range_bound() {
        for key in [0u64, 1, 2, 12_345, 0xABCD_1234_5678_9ABC, u64::MAX] {
            for bits in 1u32..=63 {
                let idx = fib_hash_u64_to_bits(key, bits);
                assert!(idx < (1u64 << bits));
            }
        }
    }

    #[test]
    fn u64_to_bits_matches_manual_shift() {
        for key in [0u64, 1, 2, 7, 255, 0xDEAD_BEEF_0000_0001, u64::MAX] {
            for bits in 1u32..=64 {
                let expected = fib_hash_u64(key) >> (64 - bits);
                assert_eq!(fib_hash_u64_to_bits(key, bits), expected);
            }
        }
    }

    #[test]
    fn u64_bits1_equals_top_bit() {
        for key in [0u64, 1, 2, 3, 100, u64::MAX] {
            let top = fib_hash_u64(key) >> 63;
            assert_eq!(fib_hash_u64_to_bits(key, 1), top);
        }
    }

    // ---- fib_hash_combine -----------------------------------------------

    #[test]
    fn combine_known_0_1() {
        // hash == 0: folded == C + C == 0x3C6EF372, XOR 0 == itself.
        assert_eq!(fib_hash_combine(0, 1), 0x3C6E_F372);
    }

    #[test]
    fn combine_known_1_0() {
        // mixed == 0, folded == 0 + C + (1<<6) + 0 == 0x9E3779F9, XOR 1 == 0x9E3779F8.
        assert_eq!(fib_hash_combine(1, 0), 0x9E37_79F8);
    }

    #[test]
    fn combine_determinism() {
        for &(h, k) in &[(0u32, 0u32), (1, 2), (0xDEAD, 0xBEEF), (u32::MAX, 7)] {
            assert_eq!(fib_hash_combine(h, k), fib_hash_combine(h, k));
        }
    }

    #[test]
    fn combine_order_matters() {
        // The combiner is intentionally order-dependent.
        assert_ne!(fib_hash_combine(1, 2), fib_hash_combine(2, 1));
    }

    // ---- distribution sanity --------------------------------------------

    #[test]
    fn distribution_u32_bits4_full_coverage() {
        // 16 buckets; consecutive keys walk a golden-ratio sequence that must
        // cover every bucket.
        let mut seen = [false; 16];
        for key in 0u32..256 {
            let idx = fib_hash_u32_to_bits(key, 4) as usize;
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn distribution_u32_bits8_full_coverage() {
        // 256 buckets covered by a modest run of consecutive keys.
        let mut seen = [false; 256];
        for key in 0u32..8192 {
            let idx = fib_hash_u32_to_bits(key, 8) as usize;
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn distribution_u64_bits4_full_coverage() {
        let mut seen = [false; 16];
        for key in 0u64..256 {
            let idx = fib_hash_u64_to_bits(key, 4) as usize;
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
