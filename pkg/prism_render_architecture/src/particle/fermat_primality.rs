//! Fermat probabilistic primality testing over plain integers.
//!
//! This module is a CPU gold-standard, allocation-free implementation of the
//! `Fermat` primality test. It operates purely on `u32` inputs with `u64`
//! intermediate arithmetic and avoids any floating point or transcendental
//! operations: modular exponentiation is performed with an integer
//! square-and-multiply routine (`modpow`).
//!
//! The `Fermat` test checks whether `base^(n-1) == 1 (mod n)` for a chosen
//! `base` coprime to `n`. A composite that still satisfies this congruence is
//! a *Fermat pseudoprime* to that `base` (classic example: `341 = 11 * 31`,
//! which passes `base` `2`). Worse, a `Carmichael` number passes the test for
//! *every* `base` coprime to it (smallest example: `561 = 3 * 11 * 17`),
//! revealing the fundamental weakness of the `Fermat` test. These weaknesses
//! are explicitly characterized in the test suite.
//!
//! Convention: a `base` is reduced modulo `n` before testing. If the reduced
//! `base` is `0` (i.e. `n` divides the original `base`, so `gcd(base, n) != 1`)
//! the `base` is not a valid witness and the test reports `false`.

/// Modular exponentiation `base^exp (mod modulus)` using `u64` intermediates.
///
/// Implements square-and-multiply (binary exponentiation). All products are
/// taken in `u64`, which is sufficient because every factor stays strictly
/// below `modulus <= u32::MAX`, so each product stays below `2^64`.
/// By convention `modulus == 1` yields `0`.
pub const fn modpow_u32(base: u32, exp: u32, modulus: u32) -> u32 {
    const { assert!((u32::MAX as u64) * (u32::MAX as u64) < u64::MAX) };

    if modulus == 1 {
        return 0;
    }

    let m: u64 = modulus as u64;
    let mut result: u64 = 1;
    let mut b: u64 = (base as u64) % m;
    let mut e: u32 = exp;

    while e > 0 {
        if (e & 1) == 1 {
            result = (result * b) % m;
        }
        b = (b * b) % m;
        e >>= 1;
    }

    result as u32
}

/// Returns whether `n` passes the `Fermat` test for the given `base`.
///
/// Returns `false` for `n < 2`. The `base` is reduced via `base % n`; a reduced
/// `base` of `0` is not coprime to `n` and yields `false`. Otherwise returns
/// `modpow(base, n - 1, n) == 1`.
///
/// A `true` result does not prove primality: composites may be `Fermat`
/// pseudoprimes to this `base`.
pub const fn fermat_test(n: u32, base: u32) -> bool {
    if n < 2 {
        return false;
    }

    let reduced = base % n;
    if reduced == 0 {
        return false;
    }

    modpow_u32(reduced, n - 1, n) == 1
}

/// Returns `true` only if `n` passes the `Fermat` test for every `base` in
/// `bases`; any failing `base` proves `n` composite and yields `false`.
///
/// This is probabilistic: `Carmichael` numbers pass for all coprime bases and
/// are therefore misclassified as prime. An empty `bases` slice is vacuously
/// `true`.
pub fn is_probable_prime(n: u32, bases: &[u32]) -> bool {
    // Compile-time enshrined reference vectors (verified facts).
    const { assert!(modpow_u32(2, 10, 1000) == 24) };
    const { assert!(fermat_test(97, 2)) };
    const { assert!(!fermat_test(15, 2)) };
    const { assert!(fermat_test(341, 2)) };
    const { assert!(!fermat_test(341, 3)) };
    const { assert!(fermat_test(561, 2)) };
    const { assert!(fermat_test(561, 5)) };

    bases.iter().all(|&base| fermat_test(n, base))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only greatest common divisor used to document coprimality of bases.
    #[cfg(test)]
    fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }

    // --- Hard reference vectors (fermat_test) ---

    #[test]
    fn hard_97_base2_true() {
        assert!(fermat_test(97, 2));
    }

    #[test]
    fn hard_15_base2_false() {
        assert!(!fermat_test(15, 2));
    }

    #[test]
    fn hard_341_base2_pseudoprime_true() {
        // 341 = 11 * 31 is the classic base-2 Fermat pseudoprime.
        assert!(fermat_test(341, 2));
    }

    #[test]
    fn hard_341_base3_detects_composite() {
        assert!(!fermat_test(341, 3));
    }

    #[test]
    fn hard_561_base2_carmichael_true() {
        assert!(fermat_test(561, 2));
    }

    #[test]
    fn hard_561_base5_carmichael_true() {
        assert!(fermat_test(561, 5));
    }

    // --- Carmichael weakness: ALL coprime bases pass (561 = 3 * 11 * 17) ---

    #[test]
    fn carmichael_coprime_bases_all_pass() {
        for base in [2u32, 5, 7, 13] {
            assert_eq!(gcd_u32(base, 561), 1, "base must be coprime to 561");
            assert!(fermat_test(561, base), "coprime base must be fooled");
        }
    }

    #[test]
    fn carmichael_factor_base_three_is_not_coprime() {
        // base 3 shares factor 3 with 561, so it is NOT a valid witness.
        assert_eq!(gcd_u32(3, 561), 3);
        assert!(!fermat_test(561, 3));
    }

    #[test]
    fn carmichael_factor_base_seventeen_is_not_coprime() {
        // 17 divides 561, so it is caught (not a coprime base).
        assert_eq!(gcd_u32(17, 561), 17);
        assert!(!fermat_test(561, 17));
    }

    #[test]
    fn carmichael_coprime_witnesses_do_not_divide() {
        for base in [2u32, 5, 7, 13] {
            assert!(!561u32.is_multiple_of(base));
        }
    }

    // --- is_probable_prime hard vectors + weakness characterization ---

    #[test]
    fn is_probable_prime_97_multi_base_true() {
        assert!(is_probable_prime(97, &[2, 3, 5]));
    }

    #[test]
    fn is_probable_prime_341_base23_detected_false() {
        assert!(!is_probable_prime(341, &[2, 3]));
    }

    #[test]
    fn is_probable_prime_341_base2_only_is_fooled() {
        // Documenting the pseudoprime weakness: base 2 alone misclassifies 341.
        assert!(is_probable_prime(341, &[2]));
    }

    #[test]
    fn is_probable_prime_carmichael_fooled_by_coprime_bases() {
        // Explicit record of Fermat failing on Carmichael 561: with only coprime
        // bases every witness passes, so 561 is (wrongly) reported "prime".
        // NOTE: the originally suggested set [2,5,13,17] is invalid because 17
        // divides 561; coprime bases [2,5,7,13] are used instead.
        assert!(is_probable_prime(561, &[2, 5, 7, 13]));
    }

    #[test]
    fn is_probable_prime_carmichael_caught_with_factor_base() {
        // Including a factor base (17) does expose 561 as composite.
        assert!(!is_probable_prime(561, &[2, 17]));
    }

    #[test]
    fn is_probable_prime_carmichael_caught_with_base_three() {
        assert!(!is_probable_prime(561, &[3]));
    }

    // --- True primes pass base 2 ---

    #[test]
    fn prime_3_base2() {
        assert!(fermat_test(3, 2));
    }

    #[test]
    fn prime_5_base2() {
        assert!(fermat_test(5, 2));
    }

    #[test]
    fn prime_7_base2() {
        assert!(fermat_test(7, 2));
    }

    #[test]
    fn prime_13_base2() {
        assert!(fermat_test(13, 2));
    }

    #[test]
    fn prime_7919_base2() {
        assert!(fermat_test(7919, 2));
    }

    #[test]
    fn prime_65537_base2() {
        assert!(fermat_test(65537, 2));
    }

    #[test]
    fn prime_mersenne_2147483647_base2() {
        assert!(fermat_test(2147483647, 2));
    }

    #[test]
    fn primes_pass_multiple_bases() {
        for base in [2u32, 3, 5, 7, 11] {
            assert!(fermat_test(97, base));
        }
    }

    // --- n < 2 edge cases ---

    #[test]
    fn zero_is_not_prime() {
        assert!(!fermat_test(0, 2));
    }

    #[test]
    fn one_is_not_prime() {
        assert!(!fermat_test(1, 2));
    }

    #[test]
    fn zero_base_zero_is_not_prime() {
        assert!(!fermat_test(0, 0));
    }

    #[test]
    fn one_with_other_base_is_not_prime() {
        assert!(!fermat_test(1, 5));
    }

    // --- base reduction (a % n) and base%n == 0 convention ---

    #[test]
    fn base_reduction_matches_small_base_prime() {
        // 99 % 97 == 2, so results must match.
        assert_eq!(fermat_test(97, 99), fermat_test(97, 2));
        assert!(fermat_test(97, 99));
    }

    #[test]
    fn base_reduction_matches_small_base_composite() {
        // 17 % 15 == 2.
        assert_eq!(fermat_test(15, 17), fermat_test(15, 2));
        assert!(!fermat_test(15, 17));
    }

    #[test]
    fn base_reduction_matches_detecting_base() {
        // 344 % 341 == 3, which detects 341.
        assert_eq!(fermat_test(341, 344), fermat_test(341, 3));
        assert!(!fermat_test(341, 344));
    }

    #[test]
    fn base_multiple_of_n_reduces_to_zero_is_false() {
        // n == 2 with base 2: 2 % 2 == 0 -> not coprime -> false.
        assert!(!fermat_test(2, 2));
        // n == 2 with base 4: 4 % 2 == 0 -> false.
        assert!(!fermat_test(2, 4));
    }

    #[test]
    fn two_passes_odd_coprime_bases() {
        assert!(fermat_test(2, 3));
        assert!(fermat_test(2, 5));
    }

    #[test]
    fn is_probable_prime_two_with_odd_bases_true() {
        assert!(is_probable_prime(2, &[3, 5]));
    }

    // --- Even composites (other than 2) are detected ---

    #[test]
    fn even_composite_four_detected() {
        assert!(!fermat_test(4, 3));
    }

    #[test]
    fn even_composite_six_detected() {
        assert!(!fermat_test(6, 5));
    }

    #[test]
    fn even_composite_eight_detected() {
        assert!(!fermat_test(8, 3));
    }

    #[test]
    fn even_composite_ten_detected() {
        assert!(!fermat_test(10, 3));
    }

    #[test]
    fn even_composites_are_multiples_of_two() {
        for n in [4u32, 6, 8, 10, 100] {
            assert!(n.is_multiple_of(2));
        }
        assert!(!15u32.is_multiple_of(2));
    }

    #[test]
    fn odd_composite_nine_detected() {
        assert!(!fermat_test(9, 2));
    }

    // --- modpow known values ---

    #[test]
    fn modpow_two_pow_ten_mod_thousand() {
        assert_eq!(modpow_u32(2, 10, 1000), 24);
    }

    #[test]
    fn modpow_three_pow_four_mod_five() {
        assert_eq!(modpow_u32(3, 4, 5), 1);
    }

    #[test]
    fn modpow_exponent_zero_is_one() {
        assert_eq!(modpow_u32(2, 0, 7), 1);
    }

    #[test]
    fn modpow_base_zero_is_zero() {
        assert_eq!(modpow_u32(0, 5, 7), 0);
    }

    #[test]
    fn modpow_five_pow_three_mod_thirteen() {
        assert_eq!(modpow_u32(5, 3, 13), 8);
    }

    #[test]
    fn modpow_seven_pow_two_mod_ten() {
        assert_eq!(modpow_u32(7, 2, 10), 9);
    }

    #[test]
    fn modpow_modulus_one_is_zero() {
        assert_eq!(modpow_u32(2, 10, 1), 0);
        assert_eq!(modpow_u32(10, 0, 1), 0);
    }

    #[test]
    fn modpow_pseudoprime_core_341_base2() {
        assert_eq!(modpow_u32(2, 340, 341), 1);
    }

    #[test]
    fn modpow_pseudoprime_core_341_base3() {
        assert!(modpow_u32(3, 340, 341) != 1);
    }

    #[test]
    fn modpow_carmichael_core_561_base2() {
        assert_eq!(modpow_u32(2, 560, 561), 1);
    }

    // --- is_probable_prime on true primes with many bases ---

    #[test]
    fn is_probable_prime_97_many_bases() {
        assert!(is_probable_prime(97, &[2, 3, 5, 7, 11]));
    }

    #[test]
    fn is_probable_prime_65537_bases() {
        assert!(is_probable_prime(65537, &[2, 3, 5]));
    }

    #[test]
    fn is_probable_prime_mersenne_bases() {
        assert!(is_probable_prime(2147483647, &[2, 3, 5, 7]));
    }

    #[test]
    fn is_probable_prime_one_is_false() {
        assert!(!is_probable_prime(1, &[2, 3]));
    }

    #[test]
    fn is_probable_prime_empty_bases_is_vacuously_true() {
        assert!(is_probable_prime(4, &[]));
    }
}
