//! Exact modular exponentiation, modular multiplication, and modular inverse.
//!
//! Deterministic particle hashing, period rotation, and small ring arithmetic
//! all need modular math that is bit-for-bit reproducible on a `CPU` core or a
//! `GPU` integer pipeline alike, so this module works purely with integers.
//! The 64-bit operands are widened into the `u128` domain for every
//! intermediate product, which guarantees that squares and multiplies never
//! wrap even when the modulus approaches `u64::MAX`. No floating point, no
//! transcendental function, and no lossy detour is ever touched.
//!
//! [`mod_pow`] uses the classic square-and-multiply ladder, [`mod_mul`] is the
//! widened multiply-then-reduce primitive, [`ext_gcd`] runs the iterative
//! extended Euclidean recurrence returning Bezout coefficients `(x, y)` with
//! `a * x + b * y == g` and `g >= 0`, and [`mod_inverse`] inverts a residue
//! when it is coprime to the modulus. The degenerate `modulus <= 1` case is
//! defined to return `0` from the reduction primitives because every residue
//! collapses to the single class `0` in a ring of size `0` or `1`.

/// Reduces the product `a * b` modulo `modulus` without overflow.
///
/// Both operands are widened into `u128` before the multiply, so the exact
/// product is formed before the remainder is taken; the reduced value always
/// fits back into `u64`. By convention a `modulus <= 1` returns `0`, since the
/// only residue class in such a ring is `0`.
#[must_use]
pub fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }
    ((a as u128 * b as u128) % modulus as u128) as u64
}

/// Raises `base` to the `exp` power modulo `modulus` via square-and-multiply.
///
/// The running `result` and `base` are reduced with [`mod_mul`] on every step,
/// and the `u128` widening inside that primitive keeps the intermediate squares
/// and multiplies from wrapping even near `u64::MAX`. By convention `0^0 == 1`,
/// and any exponent against `modulus <= 1` returns `0` because the only residue
/// class is `0` (so `x^e mod 1 == 0`).
#[must_use]
pub fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }
    let mut result: u64 = 1;
    let mut b: u64 = base % modulus;
    let mut e: u64 = exp;
    while e > 0 {
        if (e & 1) == 1 {
            result = mod_mul(result, b, modulus);
        }
        b = mod_mul(b, b, modulus);
        e >>= 1;
    }
    result
}

/// Runs the iterative extended Euclidean algorithm.
///
/// Returns a triple `(g, x, y)` such that `a * x + b * y == g`, where `g` is
/// the non-negative greatest common divisor of `a` and `b`. The coefficients
/// are accumulated in the `i128` domain so intermediate products never wrap.
#[must_use]
pub fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    let mut old_r = a;
    let mut r = b;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;
    let mut old_t: i128 = 0;
    let mut t: i128 = 1;
    while r != 0 {
        let q = old_r / r;
        let new_r = old_r - q * r;
        old_r = r;
        r = new_r;
        let new_s = old_s - q * s;
        old_s = s;
        s = new_s;
        let new_t = old_t - q * t;
        old_t = t;
        t = new_t;
    }
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// Returns the modular inverse of `a` modulo `modulus`, if it exists.
///
/// Uses [`ext_gcd`]; when `gcd(a, modulus) != 1` the inverse is undefined and
/// `None` is returned. Otherwise the Bezout coefficient is normalized into the
/// half-open range `[0, modulus)` and returned as `Some(inv)`. A
/// `modulus <= 1` has no multiplicative units, so `None` is returned.
#[must_use]
pub fn mod_inverse(a: u64, modulus: u64) -> Option<u64> {
    if modulus <= 1 {
        return None;
    }
    let m = modulus as i128;
    let ar = (a % modulus) as i128;
    let (g, x, _y) = ext_gcd(ar, m);
    if g != 1 {
        return None;
    }
    // Normalize x into [0, modulus).
    let inv = ((x % m) + m) % m;
    Some(inv as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- mod_pow reference vectors ----

    #[test]
    fn mod_pow_two_pow_ten_mod_thousand() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
    }

    #[test]
    fn mod_pow_fermat_pseudoprime_645() {
        assert_eq!(mod_pow(3, 644, 645), 36);
    }

    #[test]
    fn mod_pow_exp_zero_is_one_mod_m() {
        assert_eq!(mod_pow(5, 0, 7), 1 % 7);
        assert_eq!(mod_pow(123, 0, 2), 1 % 2);
        assert_eq!(mod_pow(0, 0, 9), 1 % 9);
    }

    #[test]
    fn mod_pow_zero_zero_is_one() {
        assert_eq!(mod_pow(0, 0, 13), 1);
    }

    #[test]
    fn mod_pow_base_zero_positive_exp_is_zero() {
        assert_eq!(mod_pow(0, 5, 13), 0);
        assert_eq!(mod_pow(0, 1, 97), 0);
    }

    #[test]
    fn mod_pow_modulus_one_is_zero() {
        assert_eq!(mod_pow(7, 3, 1), 0);
        assert_eq!(mod_pow(0, 0, 1), 0);
    }

    #[test]
    fn mod_pow_modulus_zero_is_zero() {
        assert_eq!(mod_pow(7, 3, 0), 0);
    }

    #[test]
    fn mod_pow_identity_base_one() {
        for e in 0..50u64 {
            assert_eq!(mod_pow(1, e, 1000), 1);
        }
    }

    #[test]
    fn mod_pow_three_pow_small() {
        assert_eq!(mod_pow(3, 4, 1000), 81);
        assert_eq!(mod_pow(3, 5, 1000), 243);
    }

    #[test]
    fn mod_pow_large_modulus_no_overflow() {
        let m = u64::MAX - 58; // a value near the top of the range
                               // Any power of 1 is 1, confirms the ladder with a huge modulus.
        assert_eq!(mod_pow(1, 12345, m), 1 % m);
        // base^1 reduces to base.
        assert_eq!(mod_pow(m - 2, 1, m), m - 2);
    }

    #[test]
    fn mod_pow_squares_near_max() {
        let m = (1u64 << 62) + 7;
        let base = m - 3;
        let expected = mod_mul(base, base, m);
        assert_eq!(mod_pow(base, 2, m), expected);
    }

    // ---- mod_pow agrees with naive repeated multiplication ----

    fn naive_mod_pow(base: u64, exp: u64, modulus: u64) -> u64 {
        if modulus <= 1 {
            return 0;
        }
        let mut acc: u64 = 1;
        let b = base % modulus;
        for _ in 0..exp {
            acc = ((acc as u128 * b as u128) % modulus as u128) as u64;
        }
        acc
    }

    #[test]
    fn mod_pow_matches_naive_small() {
        for base in 0..20u64 {
            for exp in 0..20u64 {
                for m in 2..25u64 {
                    assert_eq!(mod_pow(base, exp, m), naive_mod_pow(base, exp, m));
                }
            }
        }
    }

    #[test]
    fn mod_pow_matches_naive_mixed() {
        let cases = [
            (7u64, 13u64, 19u64),
            (10, 17, 23),
            (123, 11, 1000),
            (255, 8, 257),
        ];
        for &(b, e, m) in &cases {
            assert_eq!(mod_pow(b, e, m), naive_mod_pow(b, e, m));
        }
    }

    // ---- mod_mul reference and overflow checks ----

    #[test]
    fn mod_mul_small() {
        assert_eq!(mod_mul(6, 7, 10), 2);
        assert_eq!(mod_mul(0, 99, 7), 0);
        assert_eq!(mod_mul(5, 5, 13), 12);
    }

    #[test]
    fn mod_mul_modulus_one_is_zero() {
        assert_eq!(mod_mul(5, 9, 1), 0);
    }

    #[test]
    fn mod_mul_modulus_zero_is_zero() {
        assert_eq!(mod_mul(5, 9, 0), 0);
    }

    #[test]
    fn mod_mul_no_overflow_near_max() {
        let a = u64::MAX - 1;
        let b = u64::MAX - 3;
        let m = u64::MAX - 7;
        let expected = ((a as u128 * b as u128) % m as u128) as u64;
        assert_eq!(mod_mul(a, b, m), expected);
    }

    #[test]
    fn mod_mul_no_overflow_many() {
        let samples = [
            (u64::MAX, u64::MAX, u64::MAX - 1),
            (u64::MAX - 10, u64::MAX - 20, 1_000_000_007),
            (1u64 << 63, (1u64 << 63) + 5, (1u64 << 61) + 3),
            (9_999_999_937, 9_999_999_967, 9_999_999_999),
        ];
        for &(a, b, m) in &samples {
            let expected = ((a as u128 * b as u128) % m as u128) as u64;
            assert_eq!(mod_mul(a, b, m), expected);
        }
    }

    #[test]
    fn mod_mul_commutative() {
        for a in 0..30u64 {
            for b in 0..30u64 {
                assert_eq!(mod_mul(a, b, 31), mod_mul(b, a, 31));
            }
        }
    }

    // ---- ext_gcd reference and Bezout identity ----

    #[test]
    fn ext_gcd_240_46() {
        let (g, x, y) = ext_gcd(240, 46);
        assert_eq!(g, 2);
        assert_eq!(240 * x + 46 * y, 2);
    }

    #[test]
    fn ext_gcd_coprime() {
        let (g, x, y) = ext_gcd(17, 31);
        assert_eq!(g, 1);
        assert_eq!(17 * x + 31 * y, 1);
    }

    #[test]
    fn ext_gcd_zero_second() {
        let (a, b) = (42i128, 0i128);
        let (g, x, y) = ext_gcd(a, b);
        assert_eq!(g, 42);
        assert_eq!(a * x + b * y, 42);
    }

    #[test]
    fn ext_gcd_zero_first() {
        let (a, b) = (0i128, 42i128);
        let (g, x, y) = ext_gcd(a, b);
        assert_eq!(g, 42);
        assert_eq!(a * x + b * y, 42);
    }

    #[test]
    fn ext_gcd_both_zero() {
        let (a, b) = (0i128, 0i128);
        let (g, x, y) = ext_gcd(a, b);
        assert_eq!(g, 0);
        assert_eq!(a * x + b * y, 0);
    }

    #[test]
    fn ext_gcd_negative_inputs() {
        let (g, x, y) = ext_gcd(-240, 46);
        assert_eq!(g, 2);
        assert_eq!(-240 * x + 46 * y, 2);
    }

    #[test]
    fn ext_gcd_both_negative() {
        let (g, x, y) = ext_gcd(-240, -46);
        assert_eq!(g, 2);
        assert_eq!(-240 * x + -46 * y, 2);
    }

    #[test]
    fn ext_gcd_identity_many() {
        let pairs = [
            (240i128, 46i128),
            (99, 78),
            (1234, 5678),
            (1_000_000, 999),
            (97, 1),
            (13, 13),
            (1_000_003, 1_000_033),
        ];
        for &(a, b) in &pairs {
            let (g, x, y) = ext_gcd(a, b);
            assert!(g >= 0);
            assert_eq!(a * x + b * y, g);
        }
    }

    #[test]
    fn ext_gcd_g_is_nonnegative() {
        let pairs = [(-5i128, -10i128), (-7, 3), (8, -12), (-1, -1)];
        for &(a, b) in &pairs {
            let (g, _x, _y) = ext_gcd(a, b);
            assert!(g >= 0);
        }
    }

    // ---- mod_inverse reference vectors ----

    #[test]
    fn mod_inverse_three_mod_eleven() {
        assert_eq!(mod_inverse(3, 11), Some(4));
    }

    #[test]
    fn mod_inverse_two_mod_four_none() {
        assert_eq!(mod_inverse(2, 4), None);
    }

    #[test]
    fn mod_inverse_one_is_one() {
        assert_eq!(mod_inverse(1, 7), Some(1));
        assert_eq!(mod_inverse(1, 2), Some(1));
    }

    #[test]
    fn mod_inverse_modulus_one_none() {
        assert_eq!(mod_inverse(3, 1), None);
    }

    #[test]
    fn mod_inverse_modulus_zero_none() {
        assert_eq!(mod_inverse(3, 0), None);
    }

    #[test]
    fn mod_inverse_zero_none() {
        // gcd(0, m) == m != 1 for m > 1.
        assert_eq!(mod_inverse(0, 7), None);
    }

    #[test]
    fn mod_inverse_shares_factor_none() {
        assert_eq!(mod_inverse(6, 9), None);
        assert_eq!(mod_inverse(10, 15), None);
        assert_eq!(mod_inverse(4, 8), None);
    }

    #[test]
    fn mod_inverse_prime_modulus_examples() {
        assert_eq!(mod_inverse(2, 11), Some(6)); // 2*6=12≡1
        assert_eq!(mod_inverse(5, 13), Some(8)); // 5*8=40≡1
        assert_eq!(mod_inverse(7, 97), Some(14)); // 7*14=98≡1
    }

    #[test]
    fn mod_inverse_result_in_range() {
        for m in 2..60u64 {
            for a in 0..m {
                if let Some(inv) = mod_inverse(a, m) {
                    assert!(inv < m);
                }
            }
        }
    }

    // ---- mod_inverse round-trip property (many pairs) ----

    #[test]
    fn mod_inverse_roundtrip_prime_11() {
        let m = 11u64;
        for a in 1..m {
            if let Some(inv) = mod_inverse(a, m) {
                assert_eq!(mod_mul(a, inv, m), 1);
            }
        }
    }

    #[test]
    fn mod_inverse_roundtrip_prime_97() {
        let m = 97u64;
        for a in 1..m {
            let inv = mod_inverse(a, m).expect("prime modulus has full inverse set");
            assert_eq!(mod_mul(a, inv, m), 1);
        }
    }

    #[test]
    fn mod_inverse_roundtrip_composite() {
        let m = 100u64;
        for a in 0..m {
            if let Some(inv) = mod_inverse(a, m) {
                assert_eq!(mod_mul(a, inv, m), 1);
            }
        }
    }

    #[test]
    fn mod_inverse_roundtrip_many_moduli() {
        let moduli = [7u64, 13, 26, 45, 101, 1000, 1_000_000_007];
        for &m in &moduli {
            let mut checked = 0u32;
            let mut a = 1u64;
            while a < m && checked < 50 {
                if let Some(inv) = mod_inverse(a, m) {
                    assert_eq!(mod_mul(a, inv, m), 1);
                    checked += 1;
                }
                a += (m / 97).max(1);
            }
        }
    }

    #[test]
    fn mod_inverse_large_prime() {
        let m = 1_000_000_007u64;
        let a = 123_456_789u64;
        let inv = mod_inverse(a, m).expect("prime modulus");
        assert_eq!(mod_mul(a, inv, m), 1);
    }

    // ---- combined / cross-checks ----

    #[test]
    fn mod_pow_fermat_little_theorem() {
        // For prime p and a not divisible by p: a^(p-1) ≡ 1 (mod p).
        let p = 97u64;
        for a in 1..p {
            assert_eq!(mod_pow(a, p - 1, p), 1);
        }
    }

    #[test]
    fn mod_pow_inverse_via_fermat() {
        // a^(p-2) is the modular inverse of a for prime p.
        let p = 101u64;
        for a in 1..p {
            let inv = mod_pow(a, p - 2, p);
            assert_eq!(mod_mul(a, inv, p), 1);
            assert_eq!(mod_inverse(a, p), Some(inv));
        }
    }

    #[test]
    fn mod_pow_chaining_additive_exponent() {
        // base^(i+j) == base^i * base^j (mod m).
        let m = 1009u64;
        let base = 7u64;
        for i in 0..15u64 {
            for j in 0..15u64 {
                let lhs = mod_pow(base, i + j, m);
                let rhs = mod_mul(mod_pow(base, i, m), mod_pow(base, j, m), m);
                assert_eq!(lhs, rhs);
            }
        }
    }
}
