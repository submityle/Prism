//! Thomas Wang integer hashes and the Bob Jenkins integer bit-mix finalizer:
//! stateless, pure-integer `hash(u32) -> u32` and `hash(u64) -> u64` bit
//! scramblers of the kind used on the `GPU` and in open-addressing hash tables
//! to turn a counter, particle index, or packed cell coordinate into a
//! well-distributed pseudo-random word (design § content-addressing helpers).
//!
//! Scope and boundary. This module implements *single-integer finalizers* only:
//! each function takes one fixed-width integer and scrambles its bits through a
//! sequence of shifts, exclusive-ors, wrapping adds, and (for the `32`-bit Wang
//! hash) one wrapping multiply. It is deliberately disjoint from the sibling
//! hashing modules in this crate:
//!
//! * `fnv1a_hash` and `murmur3_hash` are *byte-stream* hashes: they fold a slice
//!   of bytes of arbitrary length into a digest. This module never consumes a
//!   byte slice; it mixes exactly one integer word.
//! * `pcg_hash` is the permuted-congruential-generator family (a multiply/shift
//!   permutation driven by a `64`-bit `LCG` state constant). This module uses
//!   the classic Wang and Jenkins integer-mix sequences, not the `PCG` output
//!   permutation.
//! * `xorshift_rng` is a *stateful sequence* generator: it carries a mutable
//!   state word and advances it on each draw. Everything here is stateless and
//!   referentially transparent: the same input always yields the same output.
//! * `spatial_hash` buckets positions into a grid; it is a spatial data
//!   structure, not an integer finalizer.
//!
//! Avalanche is the design goal: flipping a single input bit should flip, on
//! average, about half of the output bits, so that structured inputs (dense
//! runs such as `0, 1, 2, 3, …`) scatter uniformly across the full word and
//! both the `LSB` end and the `MSB` end carry entropy. These are
//! non-cryptographic mixers: they are not collision resistant against an
//! adversary and must never be used to authenticate data.
//!
//! Sources. The `32`- and `64`-bit integer hashes are Thomas Wang's classic
//! integer hash sequences; the `6`-shift `32`-bit mixer is Bob Jenkins' integer
//! hash. All arithmetic is wrapping (modulo `2^width`); there are no
//! floating-point steps in the integer paths, no transcendental functions, and
//! no lookup tables.

/// Multiplier constant used in the `32`-bit Thomas Wang integer hash.
pub const WANG_32_MULTIPLIER: u32 = 0x27d4_eb2d;

/// Golden-ratio mixing constant (`0x9e3779b9`) used by [`wang_hash_combine`].
///
/// This is the same `32`-bit fractional-golden-ratio constant popularized by
/// the `Boost` hash-combine recipe.
pub const WANG_COMBINE_GOLDEN: u32 = 0x9e37_79b9;

/// Thomas Wang's `32`-bit integer hash.
///
/// Scrambles a `32`-bit `key` into a well-distributed `32`-bit word through the
/// classic shift / exclusive-or / wrapping-add / wrapping-multiply sequence.
/// Stateless and deterministic: equal inputs always produce equal outputs.
#[must_use]
pub const fn wang_hash_u32(mut key: u32) -> u32 {
    key = (key ^ 61) ^ (key >> 16);
    key = key.wrapping_add(key << 3);
    key ^= key >> 4;
    key = key.wrapping_mul(WANG_32_MULTIPLIER);
    key ^= key >> 15;
    key
}

/// Thomas Wang's `64`-bit integer hash.
///
/// Scrambles a `64`-bit `key` into a well-distributed `64`-bit word through the
/// classic complement / shift / exclusive-or / wrapping-add sequence. Stateless
/// and deterministic.
#[must_use]
pub const fn wang_hash_u64(mut key: u64) -> u64 {
    key = (!key).wrapping_add(key << 21);
    key ^= key >> 24;
    key = key.wrapping_add(key << 3).wrapping_add(key << 8);
    key ^= key >> 14;
    key = key.wrapping_add(key << 2).wrapping_add(key << 4);
    key ^= key >> 28;
    key = key.wrapping_add(key << 31);
    key
}

/// Bob Jenkins' `32`-bit integer hash (the `6`-shift variant).
///
/// Mixes a `32`-bit `key` with three shift / wrapping-add rounds interleaved
/// with shift / exclusive-or folds. The `wrapping_add(!0 - (a << n))` steps add
/// the bitwise complement of a shifted copy (`!0 - x` equals `!x` for unsigned
/// words). Stateless and deterministic.
#[must_use]
pub const fn jenkins_hash_u32(mut a: u32) -> u32 {
    a = a.wrapping_add(!0 - (a << 15));
    a ^= a >> 10;
    a = a.wrapping_add(a << 3);
    a ^= a >> 6;
    a = a.wrapping_add(!0 - (a << 11));
    a ^= a >> 16;
    a
}

/// Combines two `32`-bit hashes into one, order-sensitively.
///
/// Folds `b` into `a` with the golden-ratio recipe
/// `a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2)` (all wrapping). Because the fold
/// mixes `a` into itself, the operation is not commutative: swapping the two
/// arguments generally yields a different result.
#[must_use]
pub const fn wang_hash_combine(a: u32, b: u32) -> u32 {
    a ^ b
        .wrapping_add(WANG_COMBINE_GOLDEN)
        .wrapping_add(a << 6)
        .wrapping_add(a >> 2)
}

/// Maps a `32`-bit `key` to a `f32` in the half-open unit interval `[0, 1)`.
///
/// Runs `key` through [`wang_hash_u32`], keeps the top `24` bits (the full
/// mantissa width of a `f32`), and scales by the exact power-of-two reciprocal
/// `2^-24`. The product is always representable exactly and strictly less than
/// `1.0`, so the result lies in `[0, 1)` using multiplication only (no `f32`
/// equality tests and no transcendental functions).
#[must_use]
pub fn wang_f32_unit(key: u32) -> f32 {
    // 2^-24 is exact in `f32`; (hashed >> 8) is at most 2^24 - 1.
    const INV_2_POW_24: f32 = 1.0 / 16_777_216.0;
    let hashed = wang_hash_u32(key);
    (hashed >> 8) as f32 * INV_2_POW_24
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    // --- Determinism: equal inputs produce equal outputs ---

    #[test]
    fn wang32_is_deterministic() {
        let keys = [0u32, 1, 2, 7, 0x1234_5678, 0xdead_beef, u32::MAX];
        for &k in &keys {
            assert_eq!(wang_hash_u32(k), wang_hash_u32(k));
        }
    }

    #[test]
    fn wang64_is_deterministic() {
        let keys = [0u64, 1, 2, 7, 0x0123_4567_89ab_cdef, u64::MAX];
        for &k in &keys {
            assert_eq!(wang_hash_u64(k), wang_hash_u64(k));
        }
    }

    #[test]
    fn jenkins32_is_deterministic() {
        let keys = [0u32, 1, 2, 7, 0x1234_5678, 0xdead_beef, u32::MAX];
        for &k in &keys {
            assert_eq!(jenkins_hash_u32(k), jenkins_hash_u32(k));
        }
    }

    #[test]
    fn combine_is_deterministic() {
        assert_eq!(wang_hash_combine(3, 9), wang_hash_combine(3, 9));
        assert_eq!(wang_hash_combine(0, 0), wang_hash_combine(0, 0));
    }

    #[test]
    fn f32_unit_is_deterministic() {
        for k in 0u32..256 {
            let a = wang_f32_unit(k);
            let b = wang_f32_unit(k);
            // Compare the exact bit patterns instead of using `f32` equality.
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // --- Known reference values (independently reproduced in Python) ---

    #[test]
    fn wang32_known_points() {
        assert_eq!(wang_hash_u32(0), 0xc0a9_496a);
        assert_eq!(wang_hash_u32(1), 0x2792_2c9d);
        assert_eq!(wang_hash_u32(2), 0xc679_3575);
        assert_eq!(wang_hash_u32(3), 0x87d0_6fbe);
        assert_eq!(wang_hash_u32(0xdead_beef), 0x572e_7c2d);
        assert_eq!(wang_hash_u32(u32::MAX), 0x70f4_99d3);
    }

    #[test]
    fn wang64_known_points() {
        assert_eq!(wang_hash_u64(0), 0x77cf_a1ee_f01b_ca90);
        assert_eq!(wang_hash_u64(1), 0x5bca_7c69_b794_f8ce);
        assert_eq!(wang_hash_u64(2), 0xb795_033f_6f2a_0674);
        assert_eq!(wang_hash_u64(0xdead_beef_cafe_f00d), 0x0013_507e_2211_31a3);
    }

    #[test]
    fn wang64_all_ones_known_point() {
        assert_eq!(wang_hash_u64(u64::MAX), 0x1f89_206e_3f8e_c794);
    }

    #[test]
    fn jenkins32_known_points() {
        assert_eq!(jenkins_hash_u32(0), 0x4636_b9c9);
        assert_eq!(jenkins_hash_u32(1), 0x62ba_f5a0);
        assert_eq!(jenkins_hash_u32(2), 0xff4d_1170);
        assert_eq!(jenkins_hash_u32(3), 0x2bf0_62cf);
        assert_eq!(jenkins_hash_u32(0xdead_beef), 0xcd42_a50d);
    }

    #[test]
    fn combine_known_points() {
        // Reproduced independently: see module tests notes.
        assert_eq!(wang_hash_combine(0, 0), 0x9e37_79b9);
        assert_eq!(wang_hash_combine(1, 2), 0x9e37_79fa);
        assert_eq!(wang_hash_combine(2, 1), 0x9e37_7a38);
    }

    // --- Avalanche: flipping one input bit changes many output bits ---

    #[test]
    fn wang32_single_bit_flip_avalanches() {
        let base = wang_hash_u32(0x1357_9bdf);
        let mut bit = 0u32;
        while bit < 32 {
            let flipped = wang_hash_u32(0x1357_9bdf ^ (1 << bit));
            let changed = (base ^ flipped).count_ones();
            assert!(changed > 6, "bit {bit} changed only {changed} output bits");
            bit += 1;
        }
    }

    #[test]
    fn wang32_average_avalanche_near_half() {
        let mut total = 0u32;
        let base = wang_hash_u32(0x9e37_79b9);
        let mut bit = 0u32;
        while bit < 32 {
            let flipped = wang_hash_u32(0x9e37_79b9 ^ (1 << bit));
            total += (base ^ flipped).count_ones();
            bit += 1;
        }
        let average = total / 32;
        assert!(average > 11, "average avalanche {average} too low");
    }

    #[test]
    fn wang64_single_bit_flip_avalanches() {
        let base = wang_hash_u64(0x0123_4567_89ab_cdef);
        let mut bit = 0u32;
        while bit < 64 {
            let flipped = wang_hash_u64(0x0123_4567_89ab_cdef ^ (1u64 << bit));
            let changed = (base ^ flipped).count_ones();
            assert!(changed > 10, "bit {bit} changed only {changed} output bits");
            bit += 1;
        }
    }

    #[test]
    fn jenkins32_single_bit_flip_avalanches() {
        let base = jenkins_hash_u32(0x2468_ace0);
        let mut bit = 0u32;
        while bit < 32 {
            let flipped = jenkins_hash_u32(0x2468_ace0 ^ (1 << bit));
            let changed = (base ^ flipped).count_ones();
            assert!(changed > 4, "bit {bit} changed only {changed} output bits");
            bit += 1;
        }
    }

    // --- Collision behaviour over dense key ranges ---

    #[test]
    fn wang32_no_collisions_on_first_4096_keys() {
        let mut seen = BTreeSet::new();
        for k in 0u32..=4095 {
            assert!(seen.insert(wang_hash_u32(k)), "collision at key {k}");
        }
        assert_eq!(seen.len(), 4096);
    }

    #[test]
    fn jenkins32_no_collisions_on_first_4096_keys() {
        let mut seen = BTreeSet::new();
        for k in 0u32..=4095 {
            assert!(seen.insert(jenkins_hash_u32(k)), "collision at key {k}");
        }
        assert_eq!(seen.len(), 4096);
    }

    #[test]
    fn wang64_no_collisions_on_sampled_keys() {
        let mut seen = BTreeSet::new();
        let mut k = 0u64;
        while k < 4096 {
            // Spread the samples across the whole `64`-bit range.
            let key = k.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            assert!(seen.insert(wang_hash_u64(key)), "collision at sample {k}");
            k += 1;
        }
        assert_eq!(seen.len(), 4096);
    }

    #[test]
    fn combine_distinct_pairs_mostly_differ() {
        let mut seen = BTreeSet::new();
        let mut a = 0u32;
        while a < 64 {
            let mut b = 0u32;
            while b < 64 {
                seen.insert(wang_hash_combine(a, b));
                b += 1;
            }
            a += 1;
        }
        // 4096 pairs should map to (almost) 4096 distinct combined hashes.
        assert!(seen.len() >= 4000, "only {} distinct", seen.len());
    }

    // --- Order sensitivity of the combiner ---

    #[test]
    fn combine_is_order_sensitive() {
        assert_ne!(wang_hash_combine(1, 2), wang_hash_combine(2, 1));
        assert_ne!(wang_hash_combine(7, 99), wang_hash_combine(99, 7));
        assert_ne!(
            wang_hash_combine(0xdead, 0xbeef),
            wang_hash_combine(0xbeef, 0xdead)
        );
    }

    #[test]
    fn combine_absorbs_second_argument() {
        // Changing only `b` must change the result for a fixed `a`.
        assert_ne!(wang_hash_combine(5, 0), wang_hash_combine(5, 1));
    }

    // --- Distribution: both halves of the word carry entropy ---

    #[test]
    fn wang32_high_and_low_bits_both_vary() {
        let mut high_or = 0u32;
        let mut high_and = u32::MAX;
        let mut low_or = 0u32;
        let mut low_and = u32::MAX;
        let mut k = 0u32;
        while k < 1024 {
            let h = wang_hash_u32(k);
            let high = h >> 16;
            let low = h & 0xffff;
            high_or |= high;
            high_and &= high;
            low_or |= low;
            low_and &= low;
            k += 1;
        }
        // Across the sample every `MSB`-half and `LSB`-half bit is seen both set
        // and clear.
        assert_eq!(high_or, 0xffff);
        assert_eq!(high_and, 0);
        assert_eq!(low_or, 0xffff);
        assert_eq!(low_and, 0);
    }

    #[test]
    fn wang64_high_and_low_bits_both_vary() {
        let mut high_or = 0u32;
        let mut low_or = 0u32;
        let mut k = 0u64;
        while k < 2048 {
            let h = wang_hash_u64(k);
            high_or |= (h >> 32) as u32;
            low_or |= h as u32;
            k += 1;
        }
        assert_eq!(high_or, 0xffff_ffff);
        assert_eq!(low_or, 0xffff_ffff);
    }

    #[test]
    fn jenkins32_high_and_low_bits_both_vary() {
        let mut high_or = 0u32;
        let mut low_or = 0u32;
        let mut k = 0u32;
        while k < 1024 {
            let h = jenkins_hash_u32(k);
            high_or |= h >> 16;
            low_or |= h & 0xffff;
            k += 1;
        }
        assert_eq!(high_or, 0xffff);
        assert_eq!(low_or, 0xffff);
    }

    #[test]
    fn wang32_quadrant_distribution_is_balanced() {
        // Count which quarter of the output range each hash lands in.
        let mut buckets = [0u32; 4];
        let mut k = 0u32;
        while k < 4096 {
            let q = (wang_hash_u32(k) >> 30) as usize;
            buckets[q] += 1;
            k += 1;
        }
        for (q, &count) in buckets.iter().enumerate() {
            // A fair split is 1024 per quadrant; allow generous slack.
            assert!(
                (768..=1280).contains(&count),
                "quadrant {q} had {count} hits"
            );
        }
    }

    // --- Hashes are not the identity and differ across algorithms ---

    #[test]
    fn wang32_is_not_identity() {
        let mut differ = 0u32;
        let mut k = 0u32;
        while k < 256 {
            if wang_hash_u32(k) != k {
                differ += 1;
            }
            k += 1;
        }
        assert_eq!(differ, 256);
    }

    #[test]
    fn wang_and_jenkins_differ() {
        let keys = [1u32, 2, 3, 0x1234_5678, 0xdead_beef];
        for &k in &keys {
            assert_ne!(wang_hash_u32(k), jenkins_hash_u32(k));
        }
    }

    // --- Pure-function (idempotence) over many calls ---

    #[test]
    fn wang32_repeated_calls_are_pure() {
        let first = wang_hash_u32(0xcafe_f00d);
        let mut i = 0u32;
        while i < 1000 {
            assert_eq!(wang_hash_u32(0xcafe_f00d), first);
            i += 1;
        }
    }

    // --- `wang_f32_unit`: half-open unit interval and spread ---

    #[test]
    fn f32_unit_stays_in_unit_interval() {
        let mut k = 0u32;
        while k < 10_000 {
            let v = wang_f32_unit(k);
            assert!((0.0..1.0).contains(&v), "value {v} out of range at {k}");
            k += 1;
        }
    }

    #[test]
    fn f32_unit_sampled_large_keys_in_range() {
        let samples = [0u32, 1, 0x7fff_ffff, 0x8000_0000, 0xdead_beef, u32::MAX];
        for &k in &samples {
            let v = wang_f32_unit(k);
            assert!((0.0..1.0).contains(&v), "value {v} out of range at {k:#x}");
        }
    }

    #[test]
    fn f32_unit_takes_distinct_values() {
        let mut bits = BTreeSet::new();
        let mut k = 0u32;
        while k < 1000 {
            bits.insert(wang_f32_unit(k).to_bits());
            k += 1;
        }
        // The mapping should spread 1000 keys over many distinct fractions.
        assert!(bits.len() > 900, "only {} distinct fractions", bits.len());
    }

    #[test]
    fn f32_unit_mean_is_near_one_half() {
        let mut sum = 0.0f32;
        let mut k = 0u32;
        while k < 8192 {
            sum += wang_f32_unit(k);
            k += 1;
        }
        let mean = sum * (1.0 / 8192.0);
        assert!((0.45..0.55).contains(&mean), "mean {mean} off centre");
    }
}
