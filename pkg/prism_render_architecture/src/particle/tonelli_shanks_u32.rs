//! `Tonelli–Shanks` modular square root over an odd prime modulus.
//!
//! This module solves `r*r ≡ n (mod p)` for an odd prime `p`, using only
//! integer arithmetic. All modular multiplications widen into `u128` to avoid
//! overflow, then narrow back to `u64`. No floating-point or transcendental
//! functions are used: the square root is computed purely via the integer
//! `Tonelli–Shanks` algorithm, selecting the least non-residue starting from
//! `2` so results are deterministic.
//!
//! The `Legendre` symbol `legendre(a, p)` returns `0`, `1`, or `p - 1`,
//! indicating `a ≡ 0`, `a` is a quadratic residue, or `a` is a non-residue.

/// Modular multiplication `a*b mod p`, widening into `u128` to avoid overflow.
pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % (p as u128)) as u64
}

/// Modular exponentiation `base^exp mod p` via square-and-multiply.
pub fn modpow(base: u64, exp: u64, p: u64) -> u64 {
    let mut result = 1u64 % p;
    let mut b = base % p;
    let mut e = exp;
    while e > 0 {
        if !e.is_multiple_of(2) {
            result = mulmod(result, b, p);
        }
        b = mulmod(b, b, p);
        e /= 2;
    }
    result
}

/// The `Legendre` symbol `a^((p-1)/2) mod p`, yielding `0`, `1`, or `p - 1`.
pub fn legendre(a: u64, p: u64) -> u64 {
    modpow(a, (p - 1) / 2, p)
}

/// Returns some `r` with `r*r ≡ n (mod p)` for an odd prime `p`, or `None`
/// when `n` is a quadratic non-residue.
pub fn mod_sqrt(n: u64, p: u64) -> Option<u64> {
    let n = n % p;
    if n == 0 {
        return Some(0);
    }
    if legendre(n, p) != 1 {
        return None;
    }
    if (p % 4) == 3 {
        return Some(modpow(n, (p + 1) / 4, p));
    }

    // General case: p ≡ 1 (mod 4). Write p - 1 = q * 2^s with q odd.
    let mut q = p - 1;
    let mut s = 0u32;
    while q.is_multiple_of(2) {
        q /= 2;
        s += 1;
    }

    // z: the least integer from 2 upward that is a quadratic non-residue.
    let mut z = 2u64;
    while legendre(z, p) != p - 1 {
        z += 1;
    }

    let mut c = modpow(z, q, p);
    let mut r = modpow(n, q.div_ceil(2), p);
    let mut t = modpow(n, q, p);
    let mut m = s;

    while t != 1 {
        // Least i with 0 < i < m such that t^(2^i) ≡ 1.
        let mut i = 0u32;
        let mut temp = t;
        while temp != 1 {
            temp = mulmod(temp, temp, p);
            i += 1;
        }

        // b = c squared (m - i - 1) times.
        let mut b = c;
        let mut j = 0u32;
        while j < m - i - 1 {
            b = mulmod(b, b, p);
            j += 1;
        }

        r = mulmod(r, b, p);
        c = mulmod(b, b, p);
        t = mulmod(t, c, p);
        m = i;
    }

    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIMES: [u64; 14] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 101, 113];

    // --- Hard reference vectors (KAT) ---

    #[test]
    fn kat_mod_sqrt_10_13() {
        assert_eq!(mod_sqrt(10, 13), Some(7));
        assert_eq!(mulmod(7, 7, 13), 10 % 13);
    }

    #[test]
    fn kat_mod_sqrt_2_7() {
        assert_eq!(mod_sqrt(2, 7), Some(4));
        assert_eq!(mulmod(4, 4, 7), 2 % 7);
    }

    #[test]
    fn kat_mod_sqrt_5_41() {
        assert_eq!(mod_sqrt(5, 41), Some(28));
        assert_eq!(mulmod(28, 28, 41), 5 % 41);
    }

    #[test]
    fn kat_mod_sqrt_2_113() {
        assert_eq!(mod_sqrt(2, 113), Some(62));
        assert_eq!(mulmod(62, 62, 113), 2 % 113);
    }

    #[test]
    fn kat_mod_sqrt_1_101() {
        assert_eq!(mod_sqrt(1, 101), Some(1));
        assert_eq!(mulmod(1, 1, 101), 1 % 101);
    }

    #[test]
    fn kat_mod_sqrt_0_17() {
        assert_eq!(mod_sqrt(0, 17), Some(0));
        assert_eq!(mulmod(0, 0, 17), 0 % 17);
    }

    #[test]
    fn kat_mod_sqrt_3_11() {
        assert_eq!(mod_sqrt(3, 11), Some(5));
        assert_eq!(mulmod(5, 5, 11), 3 % 11);
    }

    #[test]
    fn kat_mod_sqrt_1000_100003() {
        assert_eq!(mod_sqrt(1000, 100003), Some(31425));
        assert_eq!(mulmod(31425, 31425, 100003), 1000 % 100003);
    }

    // --- Non-residue reference vectors ---

    #[test]
    fn kat_none_3_7() {
        assert_eq!(mod_sqrt(3, 7), None);
    }

    #[test]
    fn kat_none_5_7() {
        assert_eq!(mod_sqrt(5, 7), None);
    }

    #[test]
    fn kat_none_6_7() {
        assert_eq!(mod_sqrt(6, 7), None);
    }

    #[test]
    fn kat_none_2_5() {
        assert_eq!(mod_sqrt(2, 5), None);
    }

    // --- Legendre reference vectors ---

    #[test]
    fn kat_legendre_3_7() {
        assert_eq!(legendre(3, 7), 6);
    }

    #[test]
    fn kat_legendre_2_5() {
        assert_eq!(legendre(2, 5), 4);
    }

    #[test]
    fn kat_legendre_3_13() {
        assert_eq!(legendre(3, 13), 1);
    }

    // --- modpow / mulmod unit tests ---

    #[test]
    fn modpow_basic() {
        assert_eq!(modpow(2, 10, 1000), 24);
        assert_eq!(modpow(3, 3, 11), 5);
    }

    #[test]
    fn modpow_zero_exponent() {
        assert_eq!(modpow(5, 0, 13), 1);
        assert_eq!(modpow(0, 0, 7), 1);
    }

    #[test]
    fn modpow_zero_base() {
        assert_eq!(modpow(0, 5, 13), 0);
    }

    #[test]
    fn modpow_fermat_little() {
        for p in PRIMES {
            for a in 1..p {
                assert_eq!(modpow(a, p - 1, p), 1);
            }
        }
    }

    #[test]
    fn mulmod_basic() {
        assert_eq!(mulmod(7, 8, 13), 56 % 13);
        assert_eq!(mulmod(0, 123, 13), 0);
    }

    #[test]
    fn mulmod_large_no_overflow() {
        let p = 18_446_744_073_709_551_557u64; // large prime below 2^64
        let a = p - 1;
        let b = p - 2;
        assert_eq!(mulmod(a, b, p), mulmod(1, 2, p));
    }

    #[test]
    fn mulmod_identity() {
        for p in PRIMES {
            for a in 0..p {
                assert_eq!(mulmod(a, 1, p), a % p);
            }
        }
    }

    // --- Legendre properties ---

    #[test]
    fn legendre_zero_is_zero() {
        for p in PRIMES {
            assert_eq!(legendre(0, p), 0);
        }
    }

    #[test]
    fn legendre_residue_returns_one() {
        for p in PRIMES {
            for a in 1..p {
                let sq = mulmod(a, a, p);
                assert_eq!(legendre(sq, p), 1);
            }
        }
    }

    #[test]
    fn legendre_only_three_values() {
        for p in PRIMES {
            for a in 0..p {
                let l = legendre(a, p);
                assert!(l == 0 || l == 1 || l == p - 1);
            }
        }
    }

    // --- mod_sqrt structural properties ---

    #[test]
    fn mod_sqrt_zero_various() {
        for p in PRIMES {
            assert_eq!(mod_sqrt(0, p), Some(0));
        }
    }

    #[test]
    fn mod_sqrt_one_various() {
        for p in PRIMES {
            let r = mod_sqrt(1, p).unwrap();
            assert_eq!(mulmod(r, r, p), 1);
        }
    }

    #[test]
    fn mod_sqrt_reduces_input() {
        // n larger than p must reduce consistently.
        assert_eq!(mod_sqrt(10 + 13, 13), mod_sqrt(10, 13));
        assert_eq!(mod_sqrt(2 + 7, 7), mod_sqrt(2, 7));
    }

    #[test]
    fn mod_sqrt_both_roots_valid() {
        for p in PRIMES {
            for n in 0..p {
                if let Some(r) = mod_sqrt(n, p) {
                    let other = (p - r) % p;
                    assert_eq!(mulmod(r, r, p), n % p);
                    assert_eq!(mulmod(other, other, p), n % p);
                }
            }
        }
    }

    #[test]
    fn roundtrip_p13() {
        for n in 0..13u64 {
            match mod_sqrt(n, 13) {
                Some(r) => assert_eq!(mulmod(r, r, 13), n),
                None => assert_ne!(legendre(n, 13), 1),
            }
        }
    }

    #[test]
    fn roundtrip_p17() {
        for n in 0..17u64 {
            match mod_sqrt(n, 17) {
                Some(r) => assert_eq!(mulmod(r, r, 17), n),
                None => assert_ne!(legendre(n, 17), 1),
            }
        }
    }

    #[test]
    fn roundtrip_p101() {
        for n in 0..101u64 {
            match mod_sqrt(n, 101) {
                Some(r) => assert_eq!(mulmod(r, r, 101), n),
                None => assert_ne!(legendre(n, 101), 1),
            }
        }
    }

    #[test]
    fn roundtrip_p113() {
        for n in 0..113u64 {
            match mod_sqrt(n, 113) {
                Some(r) => assert_eq!(mulmod(r, r, 113), n),
                None => assert_ne!(legendre(n, 113), 1),
            }
        }
    }

    #[test]
    fn property_many_primes_roundtrip() {
        for p in PRIMES {
            for n in 0..p {
                match mod_sqrt(n, p) {
                    Some(r) => assert_eq!(mulmod(r, r, p), n % p),
                    None => assert_ne!(legendre(n, p), 1),
                }
            }
        }
    }

    #[test]
    fn property_none_implies_non_residue() {
        for p in PRIMES {
            for n in 1..p {
                if mod_sqrt(n, p).is_none() {
                    assert_eq!(legendre(n, p), p - 1);
                }
            }
        }
    }

    #[test]
    fn property_some_implies_residue_or_zero() {
        for p in PRIMES {
            for n in 0..p {
                if let Some(r) = mod_sqrt(n, p) {
                    assert!(r < p);
                    assert_eq!(mulmod(r, r, p), n);
                }
            }
        }
    }

    #[test]
    fn p_mod_4_eq_3_fast_path() {
        // Primes congruent to 3 mod 4 use the direct exponent formula.
        for p in [7u64, 11, 19, 23, 31] {
            assert_eq!(p % 4, 3);
            for n in 0..p {
                if let Some(r) = mod_sqrt(n, p) {
                    assert_eq!(mulmod(r, r, p), n);
                }
            }
        }
    }

    #[test]
    fn p_mod_4_eq_1_general_path() {
        // Primes congruent to 1 mod 4 exercise the general loop.
        for p in [5u64, 13, 17, 29, 37, 41, 101, 113] {
            assert_eq!(p % 4, 1);
            for n in 0..p {
                if let Some(r) = mod_sqrt(n, p) {
                    assert_eq!(mulmod(r, r, p), n);
                }
            }
        }
    }

    #[test]
    fn exactly_half_are_residues() {
        // For an odd prime, among 1..p there are (p-1)/2 quadratic residues.
        for p in PRIMES {
            let mut residues = 0u64;
            for n in 1..p {
                if mod_sqrt(n, p).is_some() {
                    residues += 1;
                }
            }
            assert_eq!(residues, (p - 1) / 2);
        }
    }

    #[test]
    fn mod_sqrt_large_prime_roundtrip() {
        let p = 100003u64;
        for n in [1u64, 2, 3, 1000, 50000, 99999] {
            if let Some(r) = mod_sqrt(n, p) {
                assert_eq!(mulmod(r, r, p), n);
            } else {
                assert_eq!(legendre(n, p), p - 1);
            }
        }
    }

    #[test]
    fn mod_sqrt_specific_small_roots() {
        assert_eq!(mod_sqrt(4, 7), Some(2));
        assert_eq!(mulmod(2, 2, 7), 4);
        assert_eq!(mod_sqrt(9, 23), Some(3));
        assert_eq!(mulmod(3, 3, 23), 9);
    }

    #[test]
    fn legendre_matches_mod_sqrt_existence() {
        for p in PRIMES {
            for n in 1..p {
                let is_residue = legendre(n, p) == 1;
                assert_eq!(is_residue, mod_sqrt(n, p).is_some());
            }
        }
    }

    #[test]
    fn modpow_consistency_with_mulmod_chain() {
        for p in PRIMES {
            for a in 0..p {
                let mut acc = 1u64 % p;
                for _ in 0..5u32 {
                    acc = mulmod(acc, a, p);
                }
                assert_eq!(modpow(a, 5, p), acc);
            }
        }
    }
}
