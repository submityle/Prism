//! `Barrett` reduction: division-free modular reduction against a fixed
//! modulus using a precomputed reciprocal.
//!
//! For a fixed modulus `m` (`u64`, `m >= 1`) the classical `Barrett` method
//! precomputes `mu = floor(2^k / m)` once, then replaces each later `x mod m`
//! with a multiply and a shift instead of a hardware division. This module
//! fixes `k = 64` and performs the reciprocal arithmetic in `u128`, so
//! `mu = floor(2^64 / m)`. A single-word reduction of `x` (`0 <= x < 2^64`)
//! computes `q = floor(x * mu / 2^64)` and `r = x - q * m`, then subtracts `m`
//! at most once more to land in `[0, m)`.
//!
//! ## Why at most one correction
//!
//! Write `mu = (2^64 - e) / m` with `e = 2^64 mod m` and `0 <= e < m`. The
//! real-valued gap between the exact quotient and the estimate is
//! `x / m - x * mu / 2^64 = x * e / (m * 2^64)`. For `x < 2^64` and `e < m`
//! this is strictly below `1`, so the floored estimate `q` equals either the
//! true quotient or one less than it. Hence `r` starts in `[0, 2m)` and a
//! single conditional subtraction suffices; the loop form `while r >= m` is
//! retained for defensive clarity and never iterates more than twice.
//!
//! ## Wide inputs
//!
//! Products of two residues below `m` reach up to `m^2 < 2^128`, which the
//! single-word `mu` shortcut cannot reduce directly (the quotient estimate
//! error grows past one for wide inputs). [`Barrett::reduce_u128`] therefore
//! delegates to the fast `Barrett` path when the value fits in `u64` and
//! otherwise falls back to a carry-safe binary long division that only uses
//! shifts, comparisons, and subtractions. [`Barrett::mul_mod`] builds on this
//! to multiply two residues exactly.
//!
//! ## Numerics
//!
//! Everything is integer-exact: inputs are `u64`, intermediates widen to
//! `u128`, and no floating-point arithmetic appears anywhere. The reference
//! for correctness is equivalence with the naive `x % m` and the naive
//! `((a as u128 * b as u128) % m as u128)` modular product.

/// Precomputed `Barrett` reducer for a fixed modulus.
///
/// Holds the modulus `m` and the reciprocal `mu = floor(2^64 / m)`. Construct
/// one with [`Barrett::new`] and reuse it across many reductions.
pub struct Barrett {
    m: u64,
    mu: u128,
}

impl Barrett {
    /// Builds a reducer for modulus `m`.
    ///
    /// Requires `m >= 1`. When `m == 1` every reduction is `0`, which the
    /// query methods handle directly; the stored `mu` is still well defined.
    ///
    /// # Panics
    ///
    /// Panics when `m == 0`, since reduction modulo zero is undefined.
    pub fn new(m: u64) -> Self {
        assert!(m >= 1, "Barrett modulus must be at least 1");
        let mu = (1u128 << 64) / (m as u128);
        Self { m, mu }
    }

    /// Returns the modulus this reducer was built for.
    pub fn modulus(&self) -> u64 {
        self.m
    }

    /// Reduces a single-word value, returning `x mod m`.
    ///
    /// Uses the precomputed reciprocal `mu`: one `u128` multiply, one shift,
    /// one multiply-subtract, and at most one corrective subtraction.
    pub fn reduce(&self, x: u64) -> u64 {
        if self.m == 1 {
            return 0;
        }
        let x = x as u128;
        let m = self.m as u128;
        let q = (x * self.mu) >> 64;
        let mut r = x - q * m;
        while r >= m {
            r -= m;
        }
        r as u64
    }

    /// Reduces a double-word value, returning `x mod m`.
    ///
    /// Delegates to [`Barrett::reduce`] when `x` fits in `u64`. For wider
    /// inputs it performs a carry-safe binary reduction (shift in one bit at a
    /// time, subtract `m` when the running remainder reaches it), which keeps
    /// the remainder below `m < 2^64` throughout and never overflows `u128`.
    pub fn reduce_u128(&self, x: u128) -> u64 {
        if self.m == 1 {
            return 0;
        }
        if x < (1u128 << 64) {
            return self.reduce(x as u64);
        }
        let m = self.m as u128;
        let bits = 128 - x.leading_zeros();
        let mut r: u128 = 0;
        for i in (0..bits).rev() {
            r = (r << 1) | ((x >> i) & 1);
            if r >= m {
                r -= m;
            }
        }
        r as u64
    }

    /// Returns `(a * b) mod m` for arbitrary `u64` inputs.
    ///
    /// Both operands are reduced first, so the exact product stays below
    /// `m^2 < 2^128`; it is then reduced with [`Barrett::reduce_u128`]. The
    /// result matches the naive `((a as u128 * b as u128) % m as u128)` for all
    /// inputs, including `a` or `b` that are not already below `m`.
    pub fn mul_mod(&self, a: u64, b: u64) -> u64 {
        if self.m == 1 {
            return 0;
        }
        let a = self.reduce(a) as u128;
        let b = self.reduce(b) as u128;
        self.reduce_u128(a * b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Minimal `splitmix64` generator for deterministic random test vectors.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Returns a value in `[lo, hi]` (inclusive) with `lo <= hi`.
        fn range(&mut self, lo: u64, hi: u64) -> u64 {
            let span = hi - lo + 1;
            lo + (self.next_u64() % span)
        }
    }

    /// Naive modular product used as the golden reference.
    fn naive_mulmod(a: u64, b: u64, m: u64) -> u64 {
        ((a as u128 * b as u128) % (m as u128)) as u64
    }

    #[test]
    fn hard_vector_reduce_small() {
        assert_eq!(Barrett::new(1000).reduce(123456789), 789);
    }

    #[test]
    fn hard_vector_mul_mod_prime() {
        let b = Barrett::new(1_000_000_007);
        assert_eq!(b.mul_mod(123456789, 987654321), 259106859);
    }

    #[test]
    fn hard_vector_mod_one_arbitrary() {
        let b = Barrett::new(1);
        assert_eq!(b.reduce(0), 0);
        assert_eq!(b.reduce(1), 0);
        assert_eq!(b.reduce(42), 0);
        assert_eq!(b.reduce(u64::MAX), 0);
    }

    #[test]
    fn modulus_accessor_roundtrips() {
        assert_eq!(Barrett::new(1).modulus(), 1);
        assert_eq!(Barrett::new(97).modulus(), 97);
        assert_eq!(Barrett::new(u64::MAX).modulus(), u64::MAX);
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn new_zero_panics() {
        let _ = Barrett::new(0);
    }

    #[test]
    fn reduce_zero_is_zero() {
        for &m in &[1u64, 2, 3, 7, 1000, 1_000_000_007, u64::MAX] {
            assert_eq!(Barrett::new(m).reduce(0), 0);
        }
    }

    #[test]
    fn reduce_value_below_modulus_is_identity() {
        let b = Barrett::new(1000);
        for x in 0..1000u64 {
            assert_eq!(b.reduce(x), x);
        }
    }

    #[test]
    fn reduce_at_modulus_boundaries() {
        let m = 97u64;
        let b = Barrett::new(m);
        assert_eq!(b.reduce(m - 1), m - 1);
        assert_eq!(b.reduce(m), 0);
        assert_eq!(b.reduce(m + 1), 1);
        assert_eq!(b.reduce(2 * m), 0);
        assert_eq!(b.reduce(2 * m + 3), 3);
    }

    #[test]
    fn reduce_power_of_two_modulus() {
        let m = 1u64 << 20;
        let b = Barrett::new(m);
        for &x in &[0u64, 1, m - 1, m, m + 1, 123456789, u64::MAX] {
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn reduce_modulus_two() {
        let b = Barrett::new(2);
        for x in 0..64u64 {
            assert_eq!(b.reduce(x), x % 2);
        }
        assert_eq!(b.reduce(u64::MAX), 1);
    }

    #[test]
    fn reduce_prime_modulus_small_range() {
        let m = 7919u64;
        let b = Barrett::new(m);
        for x in 0..40000u64 {
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn reduce_composite_modulus_small_range() {
        let m = 7920u64;
        let b = Barrett::new(m);
        for x in 0..40000u64 {
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn reduce_near_u64_max() {
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        for delta in 0..256u64 {
            let x = u64::MAX - delta;
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn reduce_near_u64_max_power_of_two() {
        let m = 1u64 << 40;
        let b = Barrett::new(m);
        for delta in 0..256u64 {
            let x = u64::MAX - delta;
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn reduce_large_prime_modulus() {
        let m = 18_446_744_073_709_551_557u64; // largest prime below 2^64
        let b = Barrett::new(m);
        assert_eq!(b.reduce(0), 0);
        assert_eq!(b.reduce(m - 1), m - 1);
        assert_eq!(b.reduce(u64::MAX), u64::MAX % m);
        for delta in 0..64u64 {
            let x = u64::MAX - delta;
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn reduce_u128_matches_naive_small() {
        let m = 1009u64;
        let b = Barrett::new(m);
        for x in 0u128..200_000 {
            assert_eq!(b.reduce_u128(x), (x % (m as u128)) as u64);
        }
    }

    #[test]
    fn reduce_u128_handles_values_above_u64() {
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        let samples = [
            1u128 << 64,
            (1u128 << 64) + 1,
            (1u128 << 100) + 12345,
            u128::MAX,
            (m as u128) * (m as u128) - 1,
        ];
        for &x in &samples {
            assert_eq!(b.reduce_u128(x), (x % (m as u128)) as u64);
        }
    }

    #[test]
    fn reduce_u128_mod_one_is_zero() {
        let b = Barrett::new(1);
        assert_eq!(b.reduce_u128(0), 0);
        assert_eq!(b.reduce_u128(u128::MAX), 0);
    }

    #[test]
    fn mul_mod_zero_operand() {
        let b = Barrett::new(1_000_000_007);
        assert_eq!(b.mul_mod(0, 123456), 0);
        assert_eq!(b.mul_mod(123456, 0), 0);
    }

    #[test]
    fn mul_mod_identity_operand() {
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        for x in 0..1000u64 {
            assert_eq!(b.mul_mod(x, 1), x % m);
            assert_eq!(b.mul_mod(1, x), x % m);
        }
    }

    #[test]
    fn mul_mod_reduces_large_operands() {
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        // Operands not already below the modulus must still be correct.
        let a = u64::MAX;
        let c = u64::MAX - 12345;
        assert_eq!(b.mul_mod(a, c), naive_mulmod(a, c, m));
    }

    #[test]
    fn mul_mod_max_residues_large_prime() {
        let m = 18_446_744_073_709_551_557u64;
        let b = Barrett::new(m);
        let a = m - 1;
        let c = m - 2;
        assert_eq!(b.mul_mod(a, c), naive_mulmod(a, c, m));
    }

    #[test]
    fn mul_mod_power_of_two_modulus() {
        let m = 1u64 << 32;
        let b = Barrett::new(m);
        for a in 0..100u64 {
            for c in 0..100u64 {
                let x = a.wrapping_mul(1_000_003);
                let y = c.wrapping_mul(7_654_321);
                assert_eq!(b.mul_mod(x, y), naive_mulmod(x, y, m));
            }
        }
    }

    #[test]
    fn mul_mod_mod_one_is_zero() {
        let b = Barrett::new(1);
        assert_eq!(b.mul_mod(12345, 67890), 0);
        assert_eq!(b.mul_mod(u64::MAX, u64::MAX), 0);
    }

    #[test]
    fn reduce_matches_naive_exhaustive_small_moduli() {
        for m in 1..=64u64 {
            let b = Barrett::new(m);
            for x in 0..512u64 {
                assert_eq!(b.reduce(x), x % m, "m={m} x={x}");
            }
        }
    }

    #[test]
    fn mul_mod_matches_naive_exhaustive_small_moduli() {
        for m in 1..=32u64 {
            let b = Barrett::new(m);
            for a in 0..m.min(32) {
                for c in 0..m.min(32) {
                    assert_eq!(b.mul_mod(a, c), naive_mulmod(a, c, m), "m={m}");
                }
            }
        }
    }

    #[test]
    fn random_reduce_cross_check_small_modulus() {
        let mut rng = SplitMix64::new(0x1234_5678);
        let m = 1_000_003u64;
        let b = Barrett::new(m);
        let mut count = 0;
        for _ in 0..2000 {
            let x = rng.next_u64();
            assert_eq!(b.reduce(x), x % m);
            count += 1;
        }
        assert!(count >= 100);
    }

    #[test]
    fn random_reduce_cross_check_random_moduli() {
        let mut rng = SplitMix64::new(0x9ABC_DEF0);
        let mut count = 0;
        for _ in 0..500 {
            let m = rng.range(1, u64::MAX);
            let b = Barrett::new(m);
            for _ in 0..4 {
                let x = rng.next_u64();
                assert_eq!(b.reduce(x), x % m, "m={m} x={x}");
                count += 1;
            }
        }
        assert!(count >= 100);
    }

    #[test]
    fn random_reduce_near_max_random_moduli() {
        let mut rng = SplitMix64::new(0x5151_2727);
        let mut count = 0;
        for _ in 0..300 {
            let m = rng.range(2, u64::MAX);
            let b = Barrett::new(m);
            for delta in 0..4u64 {
                let x = u64::MAX - delta;
                assert_eq!(b.reduce(x), x % m, "m={m} x={x}");
                count += 1;
            }
        }
        assert!(count >= 100);
    }

    #[test]
    fn random_mul_mod_cross_check_prime() {
        let mut rng = SplitMix64::new(0xDEAD_BEEF);
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        let mut count = 0;
        for _ in 0..2000 {
            let a = rng.next_u64() % m;
            let c = rng.next_u64() % m;
            assert_eq!(b.mul_mod(a, c), naive_mulmod(a, c, m));
            count += 1;
        }
        assert!(count >= 100);
    }

    #[test]
    fn random_mul_mod_cross_check_random_moduli() {
        let mut rng = SplitMix64::new(0x0F0F_0F0F);
        let mut count = 0;
        for _ in 0..500 {
            let m = rng.range(1, u64::MAX);
            let b = Barrett::new(m);
            for _ in 0..4 {
                let a = rng.next_u64();
                let c = rng.next_u64();
                assert_eq!(b.mul_mod(a, c), naive_mulmod(a, c, m), "m={m}");
                count += 1;
            }
        }
        assert!(count >= 100);
    }

    #[test]
    fn random_mul_mod_large_prime_modulus() {
        let mut rng = SplitMix64::new(0xABCD_1234);
        let m = 18_446_744_073_709_551_557u64;
        let b = Barrett::new(m);
        let mut count = 0;
        for _ in 0..1000 {
            let a = rng.next_u64() % m;
            let c = rng.next_u64() % m;
            assert_eq!(b.mul_mod(a, c), naive_mulmod(a, c, m));
            count += 1;
        }
        assert!(count >= 100);
    }

    #[test]
    fn random_reduce_u128_cross_check() {
        let mut rng = SplitMix64::new(0x7777_1111);
        let mut count = 0;
        for _ in 0..300 {
            let m = rng.range(2, u64::MAX);
            let b = Barrett::new(m);
            for _ in 0..4 {
                let hi = rng.next_u64() as u128;
                let lo = rng.next_u64() as u128;
                let x = (hi << 64) | lo;
                assert_eq!(b.reduce_u128(x), (x % (m as u128)) as u64, "m={m}");
                count += 1;
            }
        }
        assert!(count >= 100);
    }

    #[test]
    fn determinism_reduce_is_stable() {
        let m = 1_000_000_007u64;
        let first: Vec<u64> = {
            let b = Barrett::new(m);
            (0..256u64)
                .map(|x| b.reduce(x.wrapping_mul(2654435761)))
                .collect()
        };
        let second: Vec<u64> = {
            let b = Barrett::new(m);
            (0..256u64)
                .map(|x| b.reduce(x.wrapping_mul(2654435761)))
                .collect()
        };
        assert_eq!(first, second);
    }

    #[test]
    fn determinism_mul_mod_is_stable() {
        let m = 998_244_353u64;
        let b = Barrett::new(m);
        let mut rng = SplitMix64::new(0xCAFE_F00D);
        let first: Vec<u64> = (0..256)
            .map(|_| b.mul_mod(rng.next_u64(), rng.next_u64()))
            .collect();
        let mut rng = SplitMix64::new(0xCAFE_F00D);
        let second: Vec<u64> = (0..256)
            .map(|_| b.mul_mod(rng.next_u64(), rng.next_u64()))
            .collect();
        assert_eq!(first, second);
    }

    #[test]
    fn reduce_correction_runs_at_most_twice() {
        // Values where the quotient estimate is one short still land correctly.
        let m = 3u64;
        let b = Barrett::new(m);
        for x in 0..1000u64 {
            assert_eq!(b.reduce(x), x % m);
        }
    }

    #[test]
    fn mul_mod_commutative() {
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        let mut rng = SplitMix64::new(0x1357_9BDF);
        for _ in 0..500 {
            let a = rng.next_u64() % m;
            let c = rng.next_u64() % m;
            assert_eq!(b.mul_mod(a, c), b.mul_mod(c, a));
        }
    }

    #[test]
    fn mul_mod_square_matches_naive() {
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        for x in 0..2000u64 {
            assert_eq!(b.mul_mod(x, x), naive_mulmod(x, x, m));
        }
    }

    #[test]
    fn reduce_u128_fast_path_consistency() {
        // Inputs below 2^64 must match the single-word reduce exactly.
        let m = 1_000_000_007u64;
        let b = Barrett::new(m);
        let mut rng = SplitMix64::new(0x2468_ACE0);
        for _ in 0..1000 {
            let x = rng.next_u64();
            assert_eq!(b.reduce_u128(x as u128), b.reduce(x));
        }
    }
}
