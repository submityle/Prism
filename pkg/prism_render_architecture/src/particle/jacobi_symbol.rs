//! Exact `Jacobi` symbol `(a / n)` for a positive odd modulus `n`.
//!
//! Deterministic particle hashing, quadratic-residue probes, and small-ring
//! primality screens occasionally need the `Jacobi` symbol, the multiplicative
//! generalization of the `Legendre` symbol to any positive odd modulus. This
//! module computes it purely with integers so the answer is bit-for-bit
//! reproducible on a `CPU` core or a `GPU` integer pipeline alike: no floating
//! point, no transcendental function, and no lossy detour is ever touched.
//!
//! The algorithm is the classic reciprocity-driven reduction. The numerator is
//! first normalized into `0..n` with `rem_euclid`, then the loop repeatedly
//! strips factors of two (flipping the sign according to `n mod 8`) and applies
//! quadratic reciprocity by swapping `(a, n)` (flipping the sign when both are
//! `3 mod 4`). The intermediate values are carried in the `i128` domain so the
//! normalization and swaps never wrap and the sign bookkeeping stays exact.
//!
//! The result is always one of `-1`, `0`, or `1`: it is `0` exactly when `a`
//! and `n` share a common factor, it collapses to `1` for every `a` when
//! `n == 1`, and it equals the `Legendre` symbol whenever `n` is an odd prime.

/// Returns the `Jacobi` symbol `(a / n)` as a value in `{-1, 0, 1}`.
///
/// `n` must be a positive odd integer; this is asserted. The numerator `a` may
/// be any `i128` and is normalized modulo `n` with `rem_euclid`, so negative
/// inputs and inputs larger than `n` are handled, and `(a / n)` depends only on
/// `a mod n` (the symbol is periodic in its numerator with period `n`).
///
/// The symbol is `0` exactly when `gcd(a, n) != 1`, is `1` for every `a` when
/// `n == 1`, and coincides with the `Legendre` symbol when `n` is an odd prime.
#[must_use]
pub fn jacobi_symbol(mut a: i128, mut n: i128) -> i32 {
    // Precondition: `n` must be a positive odd modulus.
    assert!(n > 0, "jacobi_symbol requires a positive modulus");
    assert!(n % 2 == 1, "jacobi_symbol requires an odd modulus");
    // Normalize the numerator into the canonical residue range `0..n`.
    a = a.rem_euclid(n);
    let mut result: i32 = 1;
    while a != 0 {
        // Strip factors of two, flipping the sign when `n` is `3` or `5`
        // modulo `8` (the two residues where `(2 / n) == -1`).
        while a % 2 == 0 {
            a /= 2;
            let r = n % 8;
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        // Quadratic reciprocity: swap the pair, flipping the sign when both are
        // `3 mod 4`.
        core::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    // `n` is reduced to `gcd(original_a, original_n)`; a shared factor forces
    // the symbol to `0`.
    if n == 1 {
        result
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Computes the `Legendre` symbol `(a / p)` for an odd prime `p` by Euler's
    /// criterion `a^((p - 1) / 2) mod p`, mapping the residue `p - 1` to `-1`.
    /// Pure integer fast exponentiation keeps the reference bit-exact.
    fn legendre_via_pow(a: i128, p: i128) -> i32 {
        let base = a.rem_euclid(p);
        if base == 0 {
            return 0;
        }
        let mut result: i128 = 1;
        let mut b = base % p;
        let mut e = (p - 1) / 2;
        while e > 0 {
            if e % 2 == 1 {
                result = result * b % p;
            }
            b = b * b % p;
            e /= 2;
        }
        if result == 1 {
            1
        } else if result == p - 1 {
            -1
        } else {
            // Unreachable for a genuine odd prime; surfaces a bad test prime.
            0
        }
    }

    #[test]
    fn hard_vector_one_one() {
        assert_eq!(jacobi_symbol(1, 1), 1);
    }

    #[test]
    fn hard_vector_two_fifteen() {
        assert_eq!(jacobi_symbol(2, 15), 1);
    }

    #[test]
    fn hard_vector_five_twentyone() {
        assert_eq!(jacobi_symbol(5, 21), 1);
    }

    #[test]
    fn hard_vector_1001_9907() {
        assert_eq!(jacobi_symbol(1001, 9907), -1);
    }

    #[test]
    fn hard_vector_nineteen_fortyfive() {
        assert_eq!(jacobi_symbol(19, 45), 1);
    }

    #[test]
    fn hard_vector_eight_twentyone() {
        assert_eq!(jacobi_symbol(8, 21), -1);
    }

    #[test]
    fn hard_vector_five_nine() {
        assert_eq!(jacobi_symbol(5, 9), 1);
    }

    #[test]
    fn hard_vector_1236_20003() {
        assert_eq!(jacobi_symbol(1236, 20003), 1);
    }

    #[test]
    fn hard_vector_three_fifteen_is_zero() {
        // gcd(3, 15) == 3, so the symbol must vanish.
        assert_eq!(jacobi_symbol(3, 15), 0);
    }

    #[test]
    fn hard_vector_zero_one_is_one() {
        // `n == 1` is the trivial ring; every numerator maps to `1`.
        assert_eq!(jacobi_symbol(0, 1), 1);
    }

    #[test]
    fn hard_vector_zero_three_is_zero() {
        assert_eq!(jacobi_symbol(0, 3), 0);
    }

    #[test]
    fn hard_vector_seven_fifteen() {
        assert_eq!(jacobi_symbol(7, 15), -1);
    }

    #[test]
    fn hard_vector_one_three() {
        assert_eq!(jacobi_symbol(1, 3), 1);
    }

    #[test]
    fn hard_vector_two_three() {
        assert_eq!(jacobi_symbol(2, 3), -1);
    }

    #[test]
    fn hard_vector_two_seven() {
        assert_eq!(jacobi_symbol(2, 7), 1);
    }

    #[test]
    fn hard_vector_three_seven() {
        assert_eq!(jacobi_symbol(3, 7), -1);
    }

    #[test]
    fn modulus_one_is_always_one() {
        // For `n == 1` the symbol is identically `1`, including negatives.
        for a in -50_i128..=50 {
            assert_eq!(jacobi_symbol(a, 1), 1);
        }
    }

    #[test]
    fn result_is_always_in_range() {
        for n in (1_i128..=199).step_by(2) {
            for a in -20_i128..=220 {
                let s = jacobi_symbol(a, n);
                assert!(s == -1 || s == 0 || s == 1);
            }
        }
    }

    #[test]
    fn periodic_in_numerator() {
        // `(a / n)` depends only on `a mod n`, so `a` and `a + n` agree.
        for n in (1_i128..=99).step_by(2) {
            for a in -30_i128..=130 {
                assert_eq!(jacobi_symbol(a, n), jacobi_symbol(a + n, n));
            }
        }
    }

    #[test]
    fn periodic_multiple_shifts() {
        for n in (3_i128..=61).step_by(2) {
            for a in 0_i128..=61 {
                let base = jacobi_symbol(a, n);
                assert_eq!(jacobi_symbol(a + 3 * n, n), base);
                assert_eq!(jacobi_symbol(a - 2 * n, n), base);
            }
        }
    }

    #[test]
    fn zero_numerator_rules() {
        // `(0 / 1) == 1`; `(0 / n) == 0` for every odd `n > 1`.
        assert_eq!(jacobi_symbol(0, 1), 1);
        for n in (3_i128..=99).step_by(2) {
            assert_eq!(jacobi_symbol(0, n), 0);
        }
    }

    #[test]
    fn zero_exactly_when_not_coprime() {
        // The symbol vanishes precisely when `gcd(a, n) != 1`.
        for n in (1_i128..=121).step_by(2) {
            for a in 0_i128..=121 {
                let g = gcd(a, n);
                let s = jacobi_symbol(a, n);
                if g == 1 {
                    assert!(s == -1 || s == 1);
                } else {
                    assert_eq!(s, 0);
                }
            }
        }
    }

    #[test]
    fn matches_legendre_for_prime_seven() {
        for a in -10_i128..=40 {
            assert_eq!(jacobi_symbol(a, 7), legendre_via_pow(a, 7));
        }
    }

    #[test]
    fn matches_legendre_for_prime_eleven() {
        for a in -10_i128..=60 {
            assert_eq!(jacobi_symbol(a, 11), legendre_via_pow(a, 11));
        }
    }

    #[test]
    fn matches_legendre_for_prime_thirteen() {
        for a in -10_i128..=70 {
            assert_eq!(jacobi_symbol(a, 13), legendre_via_pow(a, 13));
        }
    }

    #[test]
    fn matches_legendre_for_several_primes() {
        let primes: Vec<i128> = [3_i128, 5, 7, 11, 13, 17, 19, 23, 29, 31].into();
        for &p in &primes {
            for a in 0_i128..=(p * 3) {
                assert_eq!(jacobi_symbol(a, p), legendre_via_pow(a, p));
            }
        }
    }

    #[test]
    fn completely_multiplicative_in_numerator() {
        // `(a*b / n) == (a / n) * (b / n)` for any odd `n`.
        for n in (1_i128..=81).step_by(2) {
            for a in 0_i128..=25 {
                for b in 0_i128..=25 {
                    let lhs = jacobi_symbol(a * b, n);
                    let rhs = jacobi_symbol(a, n) * jacobi_symbol(b, n);
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }

    #[test]
    fn multiplicative_in_denominator() {
        // `(a / m*n) == (a / m) * (a / n)` for odd `m`, `n`.
        for m in (1_i128..=41).step_by(2) {
            for n in (1_i128..=41).step_by(2) {
                for a in 0_i128..=30 {
                    let lhs = jacobi_symbol(a, m * n);
                    let rhs = jacobi_symbol(a, m) * jacobi_symbol(a, n);
                    assert_eq!(lhs, rhs);
                }
            }
        }
    }

    #[test]
    fn one_numerator_is_always_one() {
        // `(1 / n) == 1` for every odd `n`.
        for n in (1_i128..=199).step_by(2) {
            assert_eq!(jacobi_symbol(1, n), 1);
        }
    }

    #[test]
    fn negative_one_follows_sign_rule() {
        // `(-1 / n) == 1` iff `n == 1 mod 4`, else `-1`.
        for n in (1_i128..=199).step_by(2) {
            let expected = if n % 4 == 1 { 1 } else { -1 };
            assert_eq!(jacobi_symbol(-1, n), expected);
        }
    }

    #[test]
    fn two_follows_eight_rule() {
        // `(2 / n) == 1` iff `n == 1 or 7 mod 8`, else `-1`.
        for n in (1_i128..=199).step_by(2) {
            let r = n % 8;
            let expected = if r == 1 || r == 7 { 1 } else { -1 };
            assert_eq!(jacobi_symbol(2, n), expected);
        }
    }

    #[test]
    fn perfect_squares_never_negative() {
        // A square numerator coprime to `n` yields `1`; otherwise `0`.
        for n in (1_i128..=121).step_by(2) {
            for k in 0_i128..=20 {
                let sq = k * k;
                let s = jacobi_symbol(sq, n);
                if gcd(sq, n) == 1 {
                    assert_eq!(s, 1);
                } else {
                    assert_eq!(s, 0);
                }
            }
        }
    }

    #[test]
    fn negative_numerator_normalizes() {
        // Negative inputs reduce via `rem_euclid`, matching the positive class.
        for n in (1_i128..=99).step_by(2) {
            for a in 1_i128..=99 {
                assert_eq!(jacobi_symbol(-a, n), jacobi_symbol(n - a % n, n));
            }
        }
    }

    #[test]
    fn large_prime_vector() {
        // 9907 is prime, so the symbol equals the `Legendre` symbol.
        assert_eq!(jacobi_symbol(1001, 9907), legendre_via_pow(1001, 9907));
        assert_eq!(jacobi_symbol(1001, 9907), -1);
    }

    #[test]
    fn composite_vector_20003() {
        // 20003 == 83 * 241 (both odd primes), so the hard vector factors
        // through the `Legendre` symbols of its prime divisors.
        let via_factors = legendre_via_pow(1236, 83) * legendre_via_pow(1236, 241);
        assert_eq!(jacobi_symbol(1236, 20003), via_factors);
        assert_eq!(jacobi_symbol(1236, 20003), 1);
    }

    #[test]
    fn big_numerator_does_not_overflow() {
        // Enormous numerators reduce cleanly in the `i128` domain.
        let big = 1_000_000_000_000_000_i128;
        for n in (1_i128..=51).step_by(2) {
            let s = jacobi_symbol(big, n);
            assert_eq!(s, jacobi_symbol(big % n, n));
            assert!(s == -1 || s == 0 || s == 1);
        }
    }

    #[test]
    fn negative_big_numerator_matches_mod() {
        let big = -987_654_321_123_i128;
        for n in (3_i128..=49).step_by(2) {
            assert_eq!(jacobi_symbol(big, n), jacobi_symbol(big.rem_euclid(n), n));
        }
    }

    #[test]
    fn additive_shift_by_full_modulus() {
        // Adding `n` cannot change the symbol for a batch of odd moduli.
        let moduli: Vec<i128> = [15_i128, 21, 35, 45, 9907, 20003].into();
        for &n in &moduli {
            for a in 0_i128..=40 {
                assert_eq!(jacobi_symbol(a, n), jacobi_symbol(a + n, n));
            }
        }
    }

    #[test]
    fn product_of_two_odd_primes_matches_factor_product() {
        // For `n = p*q`, cross-check the multiplicative split explicitly.
        let p = 13_i128;
        let q = 17_i128;
        let n = p * q;
        for a in 0_i128..=(n + 5) {
            let lhs = jacobi_symbol(a, n);
            let rhs = jacobi_symbol(a, p) * jacobi_symbol(a, q);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn symbol_matches_bruteforce_residue_for_primes() {
        // For an odd prime, a nonzero residue is `1` iff it is a quadratic
        // residue (some `x*x == a mod p`). Brute force confirms the sign.
        let primes = [3_i128, 5, 7, 11, 13, 17, 19];
        for &p in &primes {
            for a in 1_i128..p {
                let mut is_qr = false;
                for x in 1_i128..p {
                    if x * x % p == a {
                        is_qr = true;
                        break;
                    }
                }
                let expected = if is_qr { 1 } else { -1 };
                assert_eq!(jacobi_symbol(a, p), expected);
            }
        }
    }

    #[test]
    fn consistency_with_odd_squared_modulus() {
        // `(a / p^2)` is `1` whenever coprime (since `p^2` is a perfect square
        // modulus and the symbol is `(a/p)^2`), and `0` otherwise.
        for p in [3_i128, 5, 7, 11, 13] {
            let n = p * p;
            for a in 0_i128..=(n + 3) {
                let s = jacobi_symbol(a, n);
                if gcd(a, n) == 1 {
                    assert_eq!(s, 1);
                } else {
                    assert_eq!(s, 0);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn zero_modulus_panics() {
        let _ = jacobi_symbol(1, 0);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn negative_modulus_panics() {
        let _ = jacobi_symbol(1, -3);
    }

    #[test]
    #[should_panic(expected = "odd")]
    fn even_modulus_panics() {
        let _ = jacobi_symbol(1, 4);
    }

    /// Euclidean `gcd` on `i128` magnitudes; a tiny local test helper.
    fn gcd(a: i128, b: i128) -> i128 {
        let mut x = a.abs();
        let mut y = b.abs();
        while y != 0 {
            let r = x % y;
            x = y;
            y = r;
        }
        x
    }
}
