//! Deterministic `Miller-Rabin` primality test specialized for every `u32`.
//!
//! For any `n` below `4_759_123_141` the witness set `{2, 7, 61}` makes the
//! `Miller-Rabin` test fully deterministic. Since the largest `u32` value is
//! `4_294_967_295`, which is strictly below that bound, the witnesses
//! `{2, 7, 61}` classify every `u32` correctly. This module is pure integer
//! arithmetic: no floating point, no transcendental functions, and a hand
//! written `modpow` based on `u64` intermediates so that squaring values below
//! `2^32` never overflows `u64`.
//!
//! The implementation also screens small prime factors up front. This both
//! accelerates the common case and guarantees that any residual candidate that
//! reaches the witness loop is odd and larger than the small primes, which in
//! turn makes the `a >= n` witness-skip rule safe (small primes such as the
//! `Carmichael` numbers `561` and `1105` are rejected by trial division long
//! before the strong-probable-prime stage).

/// Deterministic witness set for the full `u32` range.
pub const WITNESSES: [u32; 3] = [2, 7, 61];

/// Small prime factors removed by trial division before the witness loop.
///
/// Covering primes up to `37` means any composite below `41 * 41 = 1_681`
/// is caught here, so no composite that is `<= 61` ever reaches the strong
/// probable prime stage. That keeps the `a >= n` skip rule for the `61`
/// witness correct.
const SMALL_PRIMES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Modular exponentiation using `u64` intermediates.
///
/// Computes `base^exp mod modulus`. Every intermediate product keeps both
/// factors below `2^32`, so each `u64` multiplication stays below `2^64` and
/// cannot overflow. When `modulus == 1` the result is defined as `0`.
pub fn modpow_u32(base: u32, exp: u32, modulus: u32) -> u32 {
    let m = modulus as u64;
    if m == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    let mut b: u64 = (base as u64) % m;
    let mut e = exp;
    while e > 0 {
        if (e & 1) == 1 {
            result = (result * b) % m;
        }
        b = (b * b) % m;
        e >>= 1;
    }
    result as u32
}

/// Deterministic primality test for every `u32`.
///
/// Returns `true` when `n` is prime and `false` otherwise. The result is exact
/// for all `u32` values thanks to the enshrined witness set `{2, 7, 61}`.
pub fn is_prime(n: u32) -> bool {
    const { assert!(WITNESSES.len() == 3) };
    const { assert!(WITNESSES[0] == 2) };
    const { assert!(WITNESSES[1] == 7) };
    const { assert!(WITNESSES[2] == 61) };

    if n < 2 {
        return false;
    }

    // Trial division by small primes: exact match is prime, a proper multiple
    // is composite.
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

    // Here `n` is odd and larger than every small prime.
    let n64 = n as u64;

    // Write `n - 1 = d * 2^r` with `d` odd and `r >= 1`.
    let mut d = n - 1;
    let mut r: u32 = 0;
    while d.is_multiple_of(2) {
        d >>= 1;
        r += 1;
    }

    let mut wi: usize = 0;
    while wi < WITNESSES.len() {
        let a = WITNESSES[wi];
        wi += 1;

        // When `a >= n` the witness reduces to `0` mod `n`; such candidates are
        // already decided by trial division, so skipping is safe.
        if a >= n {
            continue;
        }

        let mut x = modpow_u32(a, d, n) as u64;
        if x == 1 || x == n64 - 1 {
            continue;
        }

        let mut composite = true;
        let mut j: u32 = 1;
        while j < r {
            x = (x * x) % n64;
            if x == n64 - 1 {
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
fn slice_contains(items: &[u32], value: u32) -> bool {
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
mod tests {
    use super::*;

    #[test]
    fn prime_2() {
        assert!(is_prime(2));
    }

    #[test]
    fn prime_3() {
        assert!(is_prime(3));
    }

    #[test]
    fn prime_5() {
        assert!(is_prime(5));
    }

    #[test]
    fn prime_7() {
        assert!(is_prime(7));
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
    fn prime_104729() {
        assert!(is_prime(104_729));
    }

    #[test]
    fn prime_mersenne_2147483647() {
        assert!(is_prime(2_147_483_647));
    }

    #[test]
    fn prime_largest_u32_4294967291() {
        assert!(is_prime(4_294_967_291));
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
    fn composite_9() {
        assert!(!is_prime(9));
    }

    #[test]
    fn composite_15() {
        assert!(!is_prime(15));
    }

    #[test]
    fn composite_25() {
        assert!(!is_prime(25));
    }

    #[test]
    fn composite_carmichael_561() {
        assert!(!is_prime(561));
    }

    #[test]
    fn composite_carmichael_1105() {
        assert!(!is_prime(1105));
    }

    #[test]
    fn composite_strong_pseudoprime_base2_2047() {
        // 2047 = 23 * 89 is a base-2 strong pseudoprime.
        assert!(!is_prime(2047));
    }

    #[test]
    fn composite_strong_pseudoprime_3215031751() {
        // Strong pseudoprime to bases {2, 3, 5, 7}; the witness set {2, 7, 61}
        // still classifies it correctly as composite.
        assert!(!is_prime(3_215_031_751));
    }

    #[test]
    fn composite_max_u32_4294967295() {
        // 4_294_967_295 = 3 * 5 * 17 * 257 * 65537.
        assert!(!is_prime(4_294_967_295));
    }

    #[test]
    fn modpow_2_10_1000() {
        assert_eq!(modpow_u32(2, 10, 1000), 24);
    }

    #[test]
    fn modpow_3_0_7() {
        assert_eq!(modpow_u32(3, 0, 7), 1);
    }

    #[test]
    fn modpow_7_61_61() {
        // By Fermat's little theorem 7^61 == 7 (mod 61).
        assert_eq!(modpow_u32(7, 61, 61), 7);
    }

    #[test]
    fn modpow_2_5_7() {
        assert_eq!(modpow_u32(2, 5, 7), 4);
    }

    #[test]
    fn modpow_5_3_13() {
        assert_eq!(modpow_u32(5, 3, 13), 8);
    }

    #[test]
    fn modpow_3_4_5() {
        assert_eq!(modpow_u32(3, 4, 5), 1);
    }

    #[test]
    fn modpow_0_5_7() {
        assert_eq!(modpow_u32(0, 5, 7), 0);
    }

    #[test]
    fn modpow_1_100_7() {
        assert_eq!(modpow_u32(1, 100, 7), 1);
    }

    #[test]
    fn modpow_modulus_one_is_zero() {
        assert_eq!(modpow_u32(5, 3, 1), 0);
        assert_eq!(modpow_u32(2, 0, 1), 0);
    }

    #[test]
    fn modpow_exponent_one_is_base_mod() {
        assert_eq!(modpow_u32(123, 1, 1000), 123);
        assert_eq!(modpow_u32(1500, 1, 1000), 500);
    }

    #[test]
    fn modpow_fermat_little_theorem() {
        // For prime p, a^p == a (mod p) for all a.
        assert_eq!(modpow_u32(3, 7, 7), 3);
        assert_eq!(modpow_u32(10, 7, 7), 3);
        assert_eq!(modpow_u32(2, 13, 13), 2);
    }

    #[test]
    fn modpow_large_base_no_overflow() {
        // Exercises u64 intermediates with near-maximal factors.
        assert_eq!(modpow_u32(4_294_967_290, 2, 4_294_967_291), 1);
    }

    #[test]
    fn primes_under_100_match_reference_set() {
        const PRIMES: [u32; 25] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ];
        let mut n: u32 = 0;
        while n < 100 {
            let expected = slice_contains(&PRIMES, n);
            assert_eq!(is_prime(n), expected, "mismatch at {n}");
            n += 1;
        }
    }

    #[test]
    fn even_numbers_above_two_are_composite() {
        assert!(is_prime(2));
        let mut n: u32 = 4;
        while n <= 200 {
            assert!(!is_prime(n), "even {n} should be composite");
            n += 2;
        }
    }

    #[test]
    fn perfect_squares_are_not_prime() {
        let mut k: u32 = 2;
        while k < 100 {
            assert!(!is_prime(k * k), "square of {k} should be composite");
            k += 1;
        }
    }

    #[test]
    fn witnesses_constant_is_enshrined() {
        assert_eq!(WITNESSES, [2, 7, 61]);
    }

    #[test]
    fn no_value_below_two_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
    }
}
