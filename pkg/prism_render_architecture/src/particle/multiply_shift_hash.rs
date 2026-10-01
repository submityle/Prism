//! Dietzfelbinger multiply-shift hashing: the canonical member of the
//! universal (and, in its two-parameter form, strongly 2-independent) family
//! of integer hash functions used to map a `64`-bit key into a short `m`-bit
//! bucket index for open-addressing hash tables, spatial-cell lookups, and
//! stateless `GPU`-friendly content addressing.
//!
//! The scheme. For a `64`-bit input `x`, an odd multiplier `a: u64`, and an
//! output width `m` in `1..=63`, the hash is
//!
//! ```text
//! h_a(x) = (a * x) mod 2^64, then take the top m bits
//!        = (a.wrapping_mul(x)) >> (64 - m)
//! ```
//!
//! Intuitively the wrapping multiply spreads the entropy of every input bit
//! across the full `64`-bit product, and the right shift keeps only the most
//! significant `m` bits — the bits that received contributions from the widest
//! span of input bits. Keeping the *high* bits (rather than masking the low
//! bits) is exactly what makes the family universal: Dietzfelbinger, Hagerup,
//! Katajainen, and Penttonen proved that if `a` is drawn uniformly at random
//! from the odd residues modulo `2^64`, then for any two distinct keys the
//! collision probability is at most `2^-(m-1)`. The multiplier must be odd for
//! this guarantee; an even multiplier would zero out low input bits and bias
//! the distribution.
//!
//! Two-parameter variant. Adding a second random word `b` and computing
//! `(a * x + b) >> (64 - m)` upgrades the family from (1-)universal to strongly
//! 2-independent (pairwise independent): any single key hashes uniformly, and
//! any two distinct keys hash independently. The additive term `b` is free to
//! be any `64`-bit value; only `a` is constrained to be odd.
//!
//! Scope and boundary. This module is a *single-word, parameterized* hash: it
//! multiplies exactly one `u64` key by a caller-supplied constant and shifts.
//! It is deliberately disjoint from the sibling hashing modules in this crate:
//!
//! * `wang_hash` and the Jenkins integer finalizers are *fixed, unparameterized*
//!   avalanche mixers; they take no multiplier argument and target bit-level
//!   avalanche rather than a provable universal-family collision bound.
//! * `fnv1a_hash` and `murmur3_hash` are *byte-stream* hashes over slices of
//!   arbitrary length; this module never consumes a byte slice.
//! * `xorshift_rng` is a *stateful sequence* generator; everything here is
//!   stateless and referentially transparent.
//!
//! This is a non-cryptographic hash. The universal / 2-independent guarantees
//! are *average-case over the random choice of `a` (and `b`)*; they do not make
//! the function collision resistant against an adversary who can see the chosen
//! parameters, and it must never be used to authenticate data. All arithmetic
//! is wrapping (modulo `2^64`); there are no floating-point steps and no lookup
//! tables, so the result is bit-for-bit reproducible across the `CPU` and the
//! `GPU`.
//!
//! Shift-overflow safety. The expression `>> (64 - m)` is only well defined for
//! `m` in `1..=63`; `m = 0` would shift by `64` (undefined in the hardware
//! shift and a panic in debug Rust) and `m >= 64` would underflow the
//! `64 - m` subtraction. Every public entry point documents and debug-asserts
//! the `1..=63` precondition.

/// Golden-ratio multiplier (`0x9E37_79B9_7F4A_7C15`): the odd `64`-bit integer
/// nearest to `2^64 / phi`, where `phi` is the golden ratio.
///
/// It is odd (its low bit is set), so it is a valid universal-family
/// multiplier, and its bit pattern is well mixed, which makes it a good default
/// when the caller does not sample a random `a`. This is the same golden-ratio
/// constant used by Fibonacci hashing and by many `hash`-combine recipes.
pub const DEFAULT_A: u64 = 0x9E37_79B9_7F4A_7C15;

/// Smallest supported output width, in bits.
pub const MIN_BITS: u32 = 1;

/// Largest supported output width, in bits.
///
/// The shift `64 - m` must stay in `1..=63`, so `m` is capped at `63`.
pub const MAX_BITS: u32 = 63;

/// Dietzfelbinger multiply-shift hash of `x` with multiplier `a` and output
/// width `m` bits.
///
/// Computes `(a.wrapping_mul(x)) >> (64 - m)`, i.e. the top `m` bits of the
/// `64`-bit wrapping product. The result is always in `0..(1 << m)`.
///
/// For the universal-family collision bound to hold, `a` should be odd (a
/// uniformly random odd residue modulo `2^64` is ideal); callers that do not
/// have a random multiplier can use [`multiply_shift_default`]. The function
/// does not itself require `a` to be odd — any `a` yields a deterministic
/// result — but an even `a` discards low input entropy and weakens the
/// distribution.
///
/// # Panics
///
/// In debug builds, panics unless `m` is in `1..=63` (`MIN_BITS..=MAX_BITS`).
/// In release builds the precondition is assumed; an out-of-range `m` produces
/// an unspecified-but-safe shift result or wraps the `64 - m` subtraction.
#[must_use]
pub const fn multiply_shift(x: u64, a: u64, m: u32) -> u64 {
    debug_assert!(
        m >= MIN_BITS && m <= MAX_BITS,
        "multiply_shift: output width m must be in 1..=63"
    );
    a.wrapping_mul(x) >> (64 - m)
}

/// Multiply-shift hash of `x` with the recommended [`DEFAULT_A`] multiplier and
/// output width `m` bits.
///
/// Equivalent to `multiply_shift(x, DEFAULT_A, m)`. Convenient when the caller
/// does not need to vary the multiplier; because `DEFAULT_A` is a fixed public
/// constant, this variant is *not* randomized and therefore does not enjoy the
/// average-case universal guarantee — use it for well-mixed deterministic
/// bucketing rather than adversarial resistance.
///
/// # Panics
///
/// In debug builds, panics unless `m` is in `1..=63`.
#[must_use]
pub const fn multiply_shift_default(x: u64, m: u32) -> u64 {
    multiply_shift(x, DEFAULT_A, m)
}

/// Strongly 2-independent (pairwise-independent) multiply-shift hash.
///
/// Computes `(a.wrapping_mul(x).wrapping_add(b)) >> (64 - m)`, the top `m` bits
/// of the affine map `a * x + b` taken modulo `2^64`. With `a` an odd random
/// multiplier and `b` an arbitrary random addend, this upgrades the family from
/// universal to strongly 2-independent. The result is always in `0..(1 << m)`.
///
/// # Panics
///
/// In debug builds, panics unless `m` is in `1..=63`.
#[must_use]
pub const fn multiply_shift_strong(x: u64, a: u64, b: u64, m: u32) -> u64 {
    debug_assert!(
        m >= MIN_BITS && m <= MAX_BITS,
        "multiply_shift_strong: output width m must be in 1..=63"
    );
    a.wrapping_mul(x).wrapping_add(b) >> (64 - m)
}

/// Returns `true` if `a` is a valid universal-family multiplier (i.e. odd).
///
/// The universal collision bound requires an odd multiplier; this helper lets
/// callers that sample their own `a` cheaply check the precondition.
#[must_use]
pub const fn is_valid_multiplier(a: u64) -> bool {
    a & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u64 = DEFAULT_A;

    // ---- Hard reference vectors (independently verified) ----

    #[test]
    fn hard_vector_zero_is_zero() {
        assert_eq!(multiply_shift(0, A, 16), 0);
    }

    #[test]
    fn hard_vector_one_is_top16_bits() {
        // a * 1 = a; (a) >> 48 == 0x9E37 == 40503.
        assert_eq!(multiply_shift(1, A, 16), 40503);
    }

    #[test]
    fn hard_vector_one_equals_a_shr_48() {
        assert_eq!(multiply_shift(1, A, 16), A >> 48);
    }

    #[test]
    fn hard_vector_one_hex_matches() {
        assert_eq!(multiply_shift(1, A, 16), 0x9E37);
    }

    #[test]
    fn default_variant_matches_explicit_a() {
        assert_eq!(multiply_shift_default(1, 16), 40503);
        assert_eq!(multiply_shift_default(0, 16), 0);
    }

    // ---- Determinism ----

    #[test]
    fn deterministic_repeated_calls() {
        for x in 0u64..256 {
            let h1 = multiply_shift(x, A, 20);
            let h2 = multiply_shift(x, A, 20);
            assert_eq!(h1, h2);
        }
    }

    #[test]
    fn deterministic_strong_variant() {
        let b = 0x1234_5678_9ABC_DEF0;
        for x in 0u64..128 {
            assert_eq!(
                multiply_shift_strong(x, A, b, 24),
                multiply_shift_strong(x, A, b, 24)
            );
        }
    }

    // ---- Output range: always in 0..(1<<m) ----

    #[test]
    fn range_m16_sampled() {
        let bound = 1u64 << 16;
        for x in 0u64..10_000 {
            assert!(multiply_shift(x, A, 16) < bound);
        }
    }

    #[test]
    fn range_m8_sampled() {
        let bound = 1u64 << 8;
        for x in 0u64..5_000 {
            assert!(multiply_shift(x.wrapping_mul(2_654_435_761), A, 8) < bound);
        }
    }

    #[test]
    fn range_m1_is_single_bit() {
        for x in 0u64..1_000 {
            let h = multiply_shift(x, A, 1);
            assert!(h < 2);
        }
    }

    #[test]
    fn range_m32_sampled() {
        let bound = 1u64 << 32;
        for x in 0u64..5_000 {
            assert!(multiply_shift(x.wrapping_mul(0x9E37_79B9), A, 32) < bound);
        }
    }

    #[test]
    fn range_m63_sampled() {
        let bound = 1u64 << 63;
        for x in (0u64..2_000).map(|k| k.wrapping_mul(0xDEAD_BEEF)) {
            assert!(multiply_shift(x, A, 63) < bound);
        }
    }

    #[test]
    fn range_strong_sampled() {
        let bound = 1u64 << 12;
        let b = 0xFEDC_BA98_7654_3210;
        for x in 0u64..8_000 {
            assert!(multiply_shift_strong(x, A, b, 12) < bound);
        }
    }

    #[test]
    fn range_all_widths_hold() {
        for m in MIN_BITS..=MAX_BITS {
            let bound_minus_one = (1u128 << m) - 1;
            for x in [0u64, 1, 42, u64::MAX, 0x8000_0000_0000_0000] {
                let h = multiply_shift(x, A, m) as u128;
                assert!(h <= bound_minus_one);
            }
        }
    }

    // ---- Boundary widths m=1 and m=32 ----

    #[test]
    fn boundary_m1_top_bit() {
        // The single output bit is bit 63 of the product.
        assert_eq!(multiply_shift(1, A, 1), A >> 63);
    }

    #[test]
    fn boundary_m1_produces_both_values() {
        let mut saw0 = false;
        let mut saw1 = false;
        for x in 0u64..1_000 {
            match multiply_shift(x, A, 1) {
                0 => saw0 = true,
                1 => saw1 = true,
                other => panic!("m=1 produced {other}"),
            }
        }
        assert!(saw0 && saw1);
    }

    #[test]
    fn boundary_m32_top_half() {
        assert_eq!(multiply_shift(1, A, 32), A >> 32);
    }

    #[test]
    fn boundary_m32_value() {
        assert_eq!(multiply_shift(1, A, 32), 0x9E37_79B9);
    }

    // ---- Distribution: different x do not all collapse to one value ----

    #[test]
    fn distribution_not_all_equal_m8() {
        let first = multiply_shift(0, A, 8);
        let mut differs = false;
        for x in 1u64..2_000 {
            if multiply_shift(x, A, 8) != first {
                differs = true;
                break;
            }
        }
        assert!(differs);
    }

    #[test]
    fn distribution_many_distinct_m16() {
        // Sequential keys should spread across many buckets.
        let mut seen = alloc_bitset();
        let mut distinct = 0usize;
        for x in 0u64..4_096 {
            let h = multiply_shift(x, A, 16) as usize;
            if !bitset_get(&seen, h) {
                bitset_set(&mut seen, h);
                distinct += 1;
            }
        }
        assert!(distinct > 1_000, "only {distinct} distinct buckets");
    }

    #[test]
    fn distribution_sequential_keys_change_bucket() {
        // Consecutive keys differ by a >> (64-m) step; many are distinct.
        let mut changes = 0usize;
        let mut prev = multiply_shift(0, A, 20);
        for x in 1u64..1_000 {
            let h = multiply_shift(x, A, 20);
            if h != prev {
                changes += 1;
            }
            prev = h;
        }
        assert!(changes > 500);
    }

    #[test]
    fn distribution_strong_differs_from_universal() {
        // Nonzero additive term shifts at least some buckets.
        let b = 0x0F0F_0F0F_0F0F_0F0F;
        let mut differ = false;
        for x in 0u64..1_000 {
            if multiply_shift_strong(x, A, b, 16) != multiply_shift(x, A, 16) {
                differ = true;
                break;
            }
        }
        assert!(differ);
    }

    // ---- Odd multiplier property ----

    #[test]
    fn default_a_is_odd() {
        assert!(is_valid_multiplier(DEFAULT_A));
        assert_eq!(DEFAULT_A & 1, 1);
    }

    #[test]
    fn is_valid_multiplier_detects_even() {
        assert!(is_valid_multiplier(1));
        assert!(is_valid_multiplier(3));
        assert!(is_valid_multiplier(u64::MAX));
        assert!(!is_valid_multiplier(0));
        assert!(!is_valid_multiplier(2));
        assert!(!is_valid_multiplier(u64::MAX - 1));
    }

    #[test]
    fn even_multiplier_zeroes_lsb_contribution() {
        // With a = 2 (even), the low input bit never reaches the top bits for
        // small keys: a*x = 2x, so x and the shift just scale linearly. This
        // documents why odd multipliers are required for universality.
        let h0 = multiply_shift(0, 2, 8);
        assert_eq!(h0, 0);
    }

    // ---- No panic / no overflow on extreme inputs ----

    #[test]
    fn u64_max_input_does_not_panic() {
        for m in MIN_BITS..=MAX_BITS {
            let _ = multiply_shift(u64::MAX, A, m);
            let _ = multiply_shift_default(u64::MAX, m);
            let _ = multiply_shift_strong(u64::MAX, A, u64::MAX, m);
        }
    }

    #[test]
    fn u64_max_multiplier_does_not_panic() {
        let _ = multiply_shift(u64::MAX, u64::MAX, 32);
        let _ = multiply_shift(0xDEAD_BEEF_CAFE_F00D, u64::MAX, 17);
    }

    #[test]
    fn wrapping_multiply_never_overflows() {
        // Products that exceed 2^64 must wrap, not panic.
        let x = 0xFFFF_FFFF_FFFF_FFFF;
        let a = 0x8000_0000_0000_0001;
        let _ = multiply_shift(x, a, 40);
        let _ = multiply_shift_strong(x, a, x, 40);
    }

    // ---- Relationship / algebraic identities ----

    #[test]
    fn strong_with_zero_b_equals_universal() {
        for x in [0u64, 1, 99, u64::MAX, 0x1234] {
            for m in [1u32, 8, 16, 32, 63] {
                assert_eq!(multiply_shift_strong(x, A, 0, m), multiply_shift(x, A, m));
            }
        }
    }

    #[test]
    fn narrower_width_is_prefix_of_wider() {
        // m bits are the top m bits; m-1 bits are those shifted right by one.
        for x in 0u64..500 {
            let wide = multiply_shift(x, A, 20);
            let narrow = multiply_shift(x, A, 19);
            assert_eq!(wide >> 1, narrow);
        }
    }

    #[test]
    fn width_relationship_across_all_pairs() {
        for x in [0u64, 7, 123, 0xABCD_EF01, u64::MAX] {
            for m in (MIN_BITS + 1)..=MAX_BITS {
                let wide = multiply_shift(x, A, m);
                let narrow = multiply_shift(x, A, m - 1);
                assert_eq!(wide >> 1, narrow);
            }
        }
    }

    #[test]
    fn multiplier_zero_hashes_everything_to_zero() {
        for x in 0u64..1_000 {
            assert_eq!(multiply_shift(x, 0, 24), 0);
        }
    }

    #[test]
    fn additive_only_when_x_zero() {
        // x = 0 => a*0 + b = b; top m bits are b >> (64-m).
        let b = 0xABCD_1234_5678_9ABC;
        for m in [1u32, 8, 16, 32, 63] {
            assert_eq!(multiply_shift_strong(0, A, b, m), b >> (64 - m));
        }
    }

    // ---- Const-context usability ----

    #[test]
    fn usable_in_const_context() {
        const H: u64 = multiply_shift(1, DEFAULT_A, 16);
        const HD: u64 = multiply_shift_default(1, 16);
        const HS: u64 = multiply_shift_strong(1, DEFAULT_A, 0, 16);
        assert_eq!(H, 40503);
        assert_eq!(HD, 40503);
        assert_eq!(HS, 40503);
    }

    // ---- Varied multipliers stay in range ----

    #[test]
    fn arbitrary_odd_multipliers_in_range() {
        let bound = 1u64 << 14;
        for a in [1u64, 3, 0xDEAD_BEEF_DEAD_BEEF, 0x1357_9BDF_1357_9BDF] {
            for x in 0u64..1_000 {
                assert!(multiply_shift(x, a, 14) < bound);
            }
        }
    }

    #[test]
    fn high_bit_only_key_is_handled() {
        let x = 0x8000_0000_0000_0000;
        let h = multiply_shift(x, A, 16);
        assert!(h < (1 << 16));
    }

    #[test]
    fn two_distinct_keys_can_differ() {
        assert_ne!(multiply_shift(1, A, 16), multiply_shift(2, A, 16));
    }

    #[test]
    fn min_and_max_bits_constants() {
        assert_eq!(MIN_BITS, 1);
        assert_eq!(MAX_BITS, 63);
    }

    #[test]
    fn default_a_value_is_golden_ratio_constant() {
        assert_eq!(DEFAULT_A, 0x9E37_79B9_7F4A_7C15);
    }

    #[test]
    fn strong_variant_in_const_context_full_width() {
        const H: u64 = multiply_shift_strong(u64::MAX, DEFAULT_A, 7, 63);
        const { assert!(H < (1u64 << 63)) };
    }

    #[test]
    fn alternating_bit_keys_stay_in_range() {
        let keys = [0x5555_5555_5555_5555u64, 0xAAAA_AAAA_AAAA_AAAA];
        for &x in &keys {
            for m in MIN_BITS..=MAX_BITS {
                assert!((multiply_shift(x, A, m) as u128) < (1u128 << m));
            }
        }
    }

    // ---- tiny test-only bitset helpers (no external deps) ----

    fn alloc_bitset() -> [u64; 1024] {
        [0u64; 1024]
    }

    fn bitset_get(set: &[u64; 1024], idx: usize) -> bool {
        (set[idx >> 6] >> (idx & 63)) & 1 == 1
    }

    fn bitset_set(set: &mut [u64; 1024], idx: usize) {
        set[idx >> 6] |= 1u64 << (idx & 63);
    }
}
