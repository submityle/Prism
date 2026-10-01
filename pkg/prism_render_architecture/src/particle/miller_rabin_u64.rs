//! Deterministic `Miller-Rabin` primality test covering every `u64`.
//!
//! Unlike the companion `miller_rabin_u32` module, this routine must classify
//! candidates all the way up to `u64::MAX` (just below `2^64`), where squaring
//! a residue would overflow a `u64`. To stay exact and overflow free the
//! modular multiply promotes both factors to `u128`, multiplies there, and
//! reduces back to `u64`. The entire implementation is pure integer
//! arithmetic: no floating point, no transcendental functions, and a hand
//! written `modpow` built from the `u128`-backed `mulmod`.
//!
//! The witness set `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` (the first
//! twelve primes) makes the `Miller-Rabin` test deterministic for every
//! `n < 3.317e24`. Since the largest `u64` value is roughly `1.8e19`, far below
//! that bound, these witnesses classify the full `u64` range correctly.
//!
//! Small prime factors are screened by trial division first. This accelerates
//! the common case and guarantees that any candidate reaching the witness loop
//! is odd and larger than every witness, which makes the `a % n == 0`
//! witness-skip rule safe.

/// Deterministic witness set for the full `u64` range (first twelve primes).
pub const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Small prime factors removed by trial division before the witness loop.
///
/// These are the same twelve primes as `WITNESSES`. Removing them up front
/// ensures any composite below `41 * 41 = 1_681` is caught here, so no
/// composite that is `<= 37` ever reaches the strong-probable-prime stage.
/// That keeps the `a % n == 0` skip rule correct for every witness.
const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Modular multiplication using a `u128` intermediate.
///
/// Computes `(a * b) mod m` exactly for all `u64` inputs. Promoting both
/// factors to `u128` keeps the product below `2^128`, so the multiplication
/// never overflows before the reduction back to `u64`.
pub fn mulmod_u64(a: u64, b: u64, m: u64) -> u64 {
    (((a as u128) * (b as u128)) % (m as u128)) as u64
}

/// Modular exponentiation using `u128`-backed `mulmod`.
///
/// Computes `base^exp mod modulus`. Each squaring and multiply routes through
/// `mulmod_u64`, so no intermediate overflows a `u64`. When `modulus == 1`
/// the result is defined as `0`.
pub fn modpow_u64(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    let mut b: u64 = base % modulus;
    let mut e = exp;
    while e > 0 {
        if (e & 1) == 1 {
            result = mulmod_u64(result, b, modulus);
        }
        b = mulmod_u64(b, b, modulus);
        e >>= 1;
    }
    result
}

/// Deterministic primality test for every `u64`.
///
/// Returns `true` when `n` is prime and `false` otherwise. The result is exact
/// for all `u64` values thanks to the enshrined twelve-prime witness set.
pub fn is_prime(n: u64) -> bool {
    const { assert!(WITNESSES.len() == 12) };
    const { assert!(WITNESSES[0] == 2) };
    const { assert!(WITNESSES[11] == 37) };

    if n < 2 {
        return false;
    }

    // Trial division by small primes: an exact match is prime, a proper
    // multiple is composite.
    let mut si: usize = 0;
    while si < SMALL_PRIMES.len() {
        let p = SMALL_PRIMES[si];
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
        si += 1;
    }

    // Here `n` is odd and larger than every small prime (hence every witness).

    // Write `n - 1 = d * 2^r` with `d` odd and `r >= 1`.
    let mut d = n - 1;
    let mut r: u32 = 0;
    while d.is_multiple_of(2) {
        d >>= 1;
        r += 1;
    }

    for &a in WITNESSES.iter() {
        // When `a % n == 0` the witness reduces to `0` mod `n`; such candidates
        // are already decided by trial division, so skipping is safe.
        if a.is_multiple_of(n) {
            continue;
        }

        let mut x = modpow_u64(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }

        let mut composite = true;
        let mut j: u32 = 1;
        while j < r {
            x = mulmod_u64(x, x, n);
            if x == n - 1 {
                composite = false;
                break;
            }
            j += 1;
        }

        if composite {
            return false;
        }
    }

    true
}

#[cfg(test)]
fn slice_contains(items: &[u64], value: u64) -> bool {
    let mut i: usize = 0;
    while i < items.len() {
        if items[i] == value {
            return true;
        }
        i += 1;
    }
    false
}

#[cfg(test)]
fn naive_is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut i: u64 = 2;
    while i * i <= n {
        if n.is_multiple_of(i) {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hard prime reference vectors (true). ---

    #[test]
    fn prime_largest_u64_prime() {
        assert!(is_prime(18_446_744_073_709_551_557));
    }

    #[test]
    fn prime_9223372036854775783() {
        assert!(is_prime(9_223_372_036_854_775_783));
    }

    #[test]
    fn prime_1000000007() {
        assert!(is_prime(1_000_000_007));
    }

    #[test]
    fn prime_4294967311() {
        assert!(is_prime(4_294_967_311));
    }

    #[test]
    fn prime_mersenne_2_pow_61_minus_1() {
        assert!(is_prime(2_305_843_009_213_693_951));
    }

    // --- Hard composite reference vectors (false). ---

    #[test]
    fn composite_u64_max() {
        assert!(!is_prime(18_446_744_073_709_551_615));
    }

    #[test]
    fn composite_2_pow_63_minus_1() {
        assert!(!is_prime(9_223_372_036_854_775_807));
    }

    #[test]
    fn composite_1000000005() {
        assert!(!is_prime(1_000_000_005));
    }

    #[test]
    fn composite_classic_strong_pseudoprime() {
        assert!(!is_prime(3_825_123_056_546_413_051));
    }

    #[test]
    fn composite_3215031751() {
        assert!(!is_prime(3_215_031_751));
    }

    // --- Small deterministic primes and composites. ---

    #[test]
    fn prime_2() {
        assert!(is_prime(2));
    }

    #[test]
    fn prime_3() {
        assert!(is_prime(3));
    }

    #[test]
    fn prime_37() {
        assert!(is_prime(37));
    }

    #[test]
    fn prime_97() {
        assert!(is_prime(97));
    }

    #[test]
    fn prime_7919() {
        assert!(is_prime(7919));
    }

    #[test]
    fn prime_65537() {
        assert!(is_prime(65_537));
    }

    #[test]
    fn composite_0() {
        assert!(!is_prime(0));
    }

    #[test]
    fn composite_1() {
        assert!(!is_prime(1));
    }

    #[test]
    fn composite_4() {
        assert!(!is_prime(4));
    }

    #[test]
    fn composite_561_carmichael() {
        assert!(!is_prime(561));
    }

    #[test]
    fn composite_1105_carmichael() {
        assert!(!is_prime(1105));
    }

    // --- mulmod known values. ---

    #[test]
    fn mulmod_small_known() {
        assert_eq!(mulmod_u64(7, 8, 10), 6);
    }

    #[test]
    fn mulmod_identity() {
        assert_eq!(mulmod_u64(123_456, 1, 1_000_000_007), 123_456);
    }

    #[test]
    fn mulmod_large_no_overflow() {
        // Both factors near u64::MAX would overflow a u64 product; the u128
        // intermediate keeps it exact.
        let a: u64 = 18_446_744_073_709_551_557;
        let b: u64 = 18_446_744_073_709_551_533;
        let m: u64 = 1_000_000_007;
        let expected = (((a as u128) * (b as u128)) % (m as u128)) as u64;
        assert_eq!(mulmod_u64(a, b, m), expected);
    }

    #[test]
    fn mulmod_zero_factor() {
        assert_eq!(mulmod_u64(0, 999, 7), 0);
    }

    // --- modpow known values and Fermat's little theorem. ---

    #[test]
    fn modpow_small_known() {
        // 2^10 = 1024, mod 1000 = 24.
        assert_eq!(modpow_u64(2, 10, 1000), 24);
    }

    #[test]
    fn modpow_modulus_one_is_zero() {
        assert_eq!(modpow_u64(5, 3, 1), 0);
    }

    #[test]
    fn modpow_exponent_zero_is_one() {
        assert_eq!(modpow_u64(123_456_789, 0, 1_000_000_007), 1);
    }

    #[test]
    fn fermat_little_theorem_p_1000000007() {
        let p: u64 = 1_000_000_007;
        assert_eq!(modpow_u64(2, p - 1, p), 1);
        assert_eq!(modpow_u64(3, p - 1, p), 1);
        assert_eq!(modpow_u64(123_456_789, p - 1, p), 1);
    }

    #[test]
    fn fermat_little_theorem_large_prime() {
        let p: u64 = 2_305_843_009_213_693_951;
        assert_eq!(modpow_u64(2, p - 1, p), 1);
        assert_eq!(modpow_u64(999_999_999_999, p - 1, p), 1);
    }

    // --- Structural scans and consistency checks. ---

    #[test]
    fn scan_0_to_100_matches_known_set() {
        let known: [u64; 25] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        let mut n: u64 = 0;
        while n <= 100 {
            assert_eq!(is_prime(n), slice_contains(&known, n));
            n += 1;
        }
    }

    #[test]
    fn even_numbers_above_two_are_composite() {
        let mut n: u64 = 4;
        while n <= 10_000 {
            assert!(!is_prime(n));
            n += 2;
        }
    }

    #[test]
    fn perfect_squares_above_one_are_composite() {
        let mut k: u64 = 2;
        while k <= 300 {
            assert!(!is_prime(k * k));
            k += 1;
        }
    }

    #[test]
    fn agrees_with_naive_up_to_5000() {
        let mut n: u64 = 2;
        while n < 5000 {
            assert_eq!(is_prime(n), naive_is_prime(n));
            n += 1;
        }
    }

    #[test]
    fn witnesses_are_prime_and_sorted() {
        let mut i: usize = 0;
        while i < WITNESSES.len() {
            assert!(is_prime(WITNESSES[i]));
            if i > 0 {
                assert!(WITNESSES[i - 1] < WITNESSES[i]);
            }
            i += 1;
        }
    }

    #[test]
    fn primes_around_power_of_two_boundary() {
        // 2^32 = 4_294_967_296; nearby prime 4_294_967_311 is prime, and the
        // boundary itself is composite.
        assert!(!is_prime(4_294_967_296));
        assert!(is_prime(4_294_967_311));
    }

    #[test]
    fn product_of_two_large_primes_is_composite() {
        let a: u64 = 1_000_000_007;
        let b: u64 = 1_000_000_009;
        assert!(!is_prime(a * b));
    }
}
