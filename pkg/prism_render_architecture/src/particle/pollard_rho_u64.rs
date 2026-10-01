//! `Pollard`'s `rho` integer factorization (`Brent`'s cycle-finding variant)
//! plus a complete prime factorization, as a `CPU` gold-standard that is pure
//! integer arithmetic and free of heap allocation.
//!
//! This module is the particle subsystem's reference answer to "break a `u64`
//! into its prime factors". It is deliberately self-contained: it ships its own
//! deterministic `Miller-Rabin` primality test (witness set is the first twelve
//! primes `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}`, which is exact across
//! the entire `u64` range) rather than reaching into a sibling module, so the
//! factorizer and its primality oracle always agree.
//!
//! The core routines are:
//!
//! * [`is_prime`] — deterministic `Miller-Rabin` over all of `u64`.
//! * [`find_factor`] — returns one non-trivial factor of a composite, or the
//!   input itself when the input is `<= 1` or prime.
//! * [`factorize`] — the full factorization: it writes the ascending list of
//!   prime factors (with multiplicity) into a caller-supplied fixed array and
//!   returns how many were written.
//!
//! Because the contract crate forbids heap allocation in non-test code, there
//! is no `Vec`: the result is written into a fixed `[u64; 64]` the caller owns.
//! A `u64` has at most sixty-three prime factors (since `2^63` already uses up
//! sixty-three factors of two and `2^64` overflows), so sixty-four slots are
//! always sufficient. The internal depth-first splitting also runs on a
//! fixed-size stack of the same bound.
//!
//! All modular multiplication is done through [`mulmod`], which widens to
//! `u128` for the product so the reduction never overflows; modular
//! exponentiation ([`modpow`]) is square-and-multiply built on top of it. There
//! are no floating-point or transcendental operations anywhere. Divisibility is
//! expressed with `.is_multiple_of(k)`, and every shift is parenthesized.
//!
//! Conventions: `factorize(0)` and `factorize(1)` both write nothing and return
//! `0` (neither has a prime factorization). `find_factor(0)` and
//! `find_factor(1)` return the input unchanged.

/// Deterministic `Miller-Rabin` witness set covering the full `u64` range.
///
/// The first twelve primes are a known sufficient witness set for every
/// `64`-bit integer, so the test below is exact (not probabilistic) for all
/// `u64` inputs.
const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Modular multiplication `(a * b) mod m` using a `u128` intermediate.
///
/// Widening both factors to `u128` means the product is computed exactly before
/// the reduction, so there is no overflow for any `a`, `b`, `m` in `u64`.
pub fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    const { assert!((u64::MAX as u128) * (u64::MAX as u128) < u128::MAX) };
    (((a as u128) * (b as u128)) % (m as u128)) as u64
}

/// Modular exponentiation `base^exp mod m` via square-and-multiply.
///
/// Every multiply goes through [`mulmod`], so there is no overflow. By
/// convention `m == 1` yields `0`.
pub fn modpow(base: u64, exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result: u64 = 1 % m;
    let mut b: u64 = base % m;
    let mut e: u64 = exp;
    while e > 0 {
        if (e & 1) == 1 {
            result = mulmod(result, b, m);
        }
        b = mulmod(b, b, m);
        e >>= 1;
    }
    result
}

/// Greatest common divisor via the `Euclidean` algorithm.
///
/// `gcd(a, 0) == a` and `gcd(0, 0) == 0`.
pub fn gcd(a: u64, b: u64) -> u64 {
    let mut x = a;
    let mut y = b;
    while y != 0 {
        let t = x % y;
        x = y;
        y = t;
    }
    x
}

/// Deterministic `Miller-Rabin` primality test for every `u64`.
///
/// Returns `true` when `n` is prime and `false` otherwise. The twelve-witness
/// set [`WITNESSES`] makes the result exact across the whole `u64` range.
///
/// The small-prime screen at the top both accelerates the common case and
/// guarantees that any candidate reaching the strong-probable-prime loop is odd
/// and strictly larger than every witness, so no witness is ever `>= n`.
pub fn is_prime(n: u64) -> bool {
    const { assert!(WITNESSES.len() == 12) };
    const { assert!(WITNESSES[0] == 2) };
    const { assert!(WITNESSES[11] == 37) };

    if n < 2 {
        return false;
    }

    // Screen the witnesses as small prime factors. If `n` equals a witness it
    // is prime; if it is a larger multiple of one it is composite.
    let mut wi = 0;
    while wi < WITNESSES.len() {
        let p = WITNESSES[wi];
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
        wi += 1;
    }

    // Here `n` is odd and larger than 37. Write `n - 1 = d * 2^s`.
    let mut d = n - 1;
    let mut s: u32 = 0;
    while d.is_multiple_of(2) {
        d >>= 1;
        s += 1;
    }

    let mut ai = 0;
    while ai < WITNESSES.len() {
        let a = WITNESSES[ai];
        let mut x = modpow(a, d, n);
        if x != 1 && x != n - 1 {
            let mut composite = true;
            let mut i: u32 = 1;
            while i < s {
                x = mulmod(x, x, n);
                if x == n - 1 {
                    composite = false;
                    break;
                }
                i += 1;
            }
            if composite {
                return false;
            }
        }
        ai += 1;
    }

    true
}

/// One step of the `Pollard` `rho` iteration `f(x) = (x*x + c) mod n`.
///
/// The add is widened to `u128` so `mulmod(x, x, n) + c` cannot overflow even
/// when `n` is close to `u64::MAX`.
fn rho_step(x: u64, c: u64, n: u64) -> u64 {
    let sq = mulmod(x, x, n);
    (((sq as u128) + (c as u128)) % (n as u128)) as u64
}

/// `Brent`'s variant of `Pollard` `rho` for a single polynomial constant `c`.
///
/// Returns a divisor of `n` (possibly `n` itself, signalling failure for this
/// `c`). Uses `Brent`'s batched `gcd` with a backtracking phase when the batch
/// collapses to the whole modulus.
fn brent_try(n: u64, c: u64) -> u64 {
    let mut y: u64 = 2;
    let mut r: u64 = 1;
    let mut q: u64 = 1;
    let mut g: u64 = 1;
    let mut x: u64 = 0;
    let mut ys: u64 = 0;

    while g == 1 {
        x = y;
        let mut i: u64 = 0;
        while i < r {
            y = rho_step(y, c, n);
            i += 1;
        }

        let mut k: u64 = 0;
        while k < r && g == 1 {
            ys = y;
            let remaining = r - k;
            let batch = if remaining < 128 { remaining } else { 128 };
            let mut j: u64 = 0;
            while j < batch {
                y = rho_step(y, c, n);
                q = mulmod(q, x.abs_diff(y), n);
                j += 1;
            }
            g = gcd(q, n);
            k += batch;
        }

        r <<= 1;
    }

    if g == n {
        loop {
            ys = rho_step(ys, c, n);
            g = gcd(x.abs_diff(ys), n);
            if g > 1 {
                break;
            }
        }
    }

    g
}

/// `Brent`'s `Pollard` `rho`: returns a non-trivial factor of `n`.
///
/// For even `n` it returns `2` immediately. For an odd composite it runs
/// [`brent_try`] with `c = 1` and increments `c` on failure, which is
/// guaranteed to eventually surface a non-trivial divisor.
fn brent(n: u64) -> u64 {
    if n.is_multiple_of(2) {
        return 2;
    }
    let mut c: u64 = 1;
    loop {
        let d = brent_try(n, c);
        if d != 1 && d != n {
            return d;
        }
        c += 1;
    }
}

/// Returns one non-trivial factor of `n`, or `n` itself when `n <= 1` or prime.
///
/// For a composite `n` the returned value `d` satisfies `1 < d < n` and
/// `n.is_multiple_of(d)`.
pub fn find_factor(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    if is_prime(n) {
        return n;
    }
    brent(n)
}

/// Full prime factorization of `n` into the caller-supplied array.
///
/// Writes the ascending prime factors of `n`, repeated by multiplicity, into
/// `out` and returns how many were written. `factorize(0)` and `factorize(1)`
/// write nothing and return `0`.
///
/// The work list is a fixed-size stack: a composite is split by [`find_factor`]
/// into two smaller pieces that are pushed back, while primes are collected.
/// Since a `u64` has at most sixty-three prime factors the sixty-four-slot
/// stack and `out` array can never overflow. The collected factors are sorted
/// in place with the slice sort, which needs no allocation.
pub fn factorize(n: u64, out: &mut [u64; 64]) -> usize {
    if n <= 1 {
        return 0;
    }

    let mut stack = [0u64; 64];
    let mut sp: usize = 0;
    stack[sp] = n;
    sp += 1;

    let mut count: usize = 0;

    while sp > 0 {
        sp -= 1;
        let m = stack[sp];
        if m <= 1 {
            continue;
        }
        if is_prime(m) {
            out[count] = m;
            count += 1;
        } else {
            let d = find_factor(m);
            stack[sp] = d;
            sp += 1;
            stack[sp] = m / d;
            sp += 1;
        }
    }

    out[..count].sort_unstable();
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Factorizes `n` and asserts the result equals `expected` exactly.
    #[cfg(test)]
    fn check(n: u64, expected: &[u64]) {
        let mut out = [0u64; 64];
        let count = factorize(n, &mut out);
        assert_eq!(count, expected.len(), "factor count mismatch for {n}");
        assert_eq!(&out[..count], expected, "factor list mismatch for {n}");
    }

    /// Verifies the structural invariants of `factorize` for a single `n`:
    /// ascending order, every factor prime, and the product recovering `n`.
    #[cfg(test)]
    fn verify(n: u64) {
        let mut out = [0u64; 64];
        let count = factorize(n, &mut out);

        let mut i = 1;
        while i < count {
            assert!(out[i - 1] <= out[i], "not ascending for {n}");
            i += 1;
        }

        let mut j = 0;
        while j < count {
            assert!(is_prime(out[j]), "non-prime factor {} for {n}", out[j]);
            j += 1;
        }

        let mut product: u128 = 1;
        let mut k = 0;
        while k < count {
            product *= out[k] as u128;
            k += 1;
        }

        if n >= 2 {
            assert_eq!(product, n as u128, "product mismatch for {n}");
        } else {
            assert_eq!(count, 0, "expected empty factorization for {n}");
        }
    }

    /// Naive trial-division primality test for cross-checking small `n`.
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

    /// Naive trial-division factorization writing ascending factors into `out`.
    #[cfg(test)]
    fn naive_factorize(n: u64, out: &mut [u64; 64]) -> usize {
        let mut m = n;
        let mut count = 0;
        let mut d: u64 = 2;
        while d * d <= m {
            while m.is_multiple_of(d) {
                out[count] = d;
                count += 1;
                m /= d;
            }
            d += 1;
        }
        if m > 1 {
            out[count] = m;
            count += 1;
        }
        count
    }

    // ---- Hard reference vectors -----------------------------------------

    #[test]
    fn vector_one_is_empty() {
        check(1, &[]);
    }

    #[test]
    fn vector_two() {
        check(2, &[2]);
    }

    #[test]
    fn vector_ninety_seven() {
        check(97, &[97]);
    }

    #[test]
    fn vector_seven_twenty() {
        check(720, &[2, 2, 2, 2, 3, 3, 5]);
    }

    #[test]
    fn vector_eight_oh_five_one() {
        check(8051, &[83, 97]);
    }

    #[test]
    fn vector_ten_four_oh_three() {
        check(10403, &[101, 103]);
    }

    #[test]
    fn vector_project_euler() {
        check(600851475143, &[71, 839, 1471, 6857]);
    }

    #[test]
    fn vector_large_semiprime() {
        check(1000000016000000063, &[1000000007, 1000000009]);
    }

    #[test]
    fn vector_zero_is_empty() {
        check(0, &[]);
    }

    // ---- is_prime -------------------------------------------------------

    #[test]
    fn is_prime_small_truths() {
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(is_prime(37));
        assert!(is_prime(41));
    }

    #[test]
    fn is_prime_small_falsehoods() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(!is_prime(4));
        assert!(!is_prime(6));
        assert!(!is_prime(8));
        assert!(!is_prime(9));
        assert!(!is_prime(25));
        assert!(!is_prime(49));
    }

    #[test]
    fn is_prime_billion_primes() {
        assert!(is_prime(1000000007));
        assert!(is_prime(1000000009));
    }

    #[test]
    fn is_prime_composite_classifications() {
        assert!(!is_prime(97 * 83));
        assert!(is_prime(97));
        assert!(!is_prime(8051));
        assert!(!is_prime(10403));
    }

    #[test]
    fn is_prime_carmichael_numbers() {
        assert!(!is_prime(561));
        assert!(!is_prime(1105));
        assert!(!is_prime(1729));
        assert!(!is_prime(2465));
        assert!(!is_prime(6601));
    }

    #[test]
    fn is_prime_mersenne() {
        // 2^61 - 1 is a Mersenne prime.
        assert!(is_prime(2305843009213693951));
    }

    #[test]
    fn is_prime_large_composite() {
        // Product of two large primes must read as composite.
        assert!(!is_prime(1000000016000000063));
        assert!(!is_prime(600851475143));
    }

    #[test]
    fn is_prime_matches_naive_small_range() {
        let mut n: u64 = 0;
        while n < 2000 {
            assert_eq!(is_prime(n), naive_is_prime(n), "mismatch at {n}");
            n += 1;
        }
    }

    // ---- find_factor ----------------------------------------------------

    #[test]
    fn find_factor_prime_returns_self() {
        assert_eq!(find_factor(97), 97);
        assert_eq!(find_factor(1000000007), 1000000007);
        assert_eq!(find_factor(2), 2);
        assert_eq!(find_factor(2305843009213693951), 2305843009213693951);
    }

    #[test]
    fn find_factor_trivial_inputs() {
        assert_eq!(find_factor(0), 0);
        assert_eq!(find_factor(1), 1);
    }

    #[test]
    fn find_factor_8051_is_nontrivial_divisor() {
        let d = find_factor(8051);
        assert!(d > 1 && d < 8051, "d={d} not strictly between 1 and 8051");
        assert!(8051u64.is_multiple_of(d), "d={d} does not divide 8051");
    }

    #[test]
    fn find_factor_10403_is_nontrivial_divisor() {
        let d = find_factor(10403);
        assert!(d > 1 && d < 10403);
        assert!(10403u64.is_multiple_of(d));
    }

    #[test]
    fn find_factor_large_semiprime_is_nontrivial() {
        let n = 1000000016000000063u64;
        let d = find_factor(n);
        assert!(d > 1 && d < n);
        assert!(n.is_multiple_of(d));
    }

    #[test]
    fn find_factor_even_returns_two() {
        assert_eq!(find_factor(720), 2);
        assert_eq!(find_factor(1000000), 2);
    }

    // ---- Prime powers ---------------------------------------------------

    #[test]
    fn power_of_two_cube() {
        check(8, &[2, 2, 2]);
    }

    #[test]
    fn powers_of_two_general() {
        let mut k: u32 = 1;
        while k <= 20 {
            let n = 1u64 << k;
            let mut expected = [0u64; 64];
            let mut i = 0;
            while i < k as usize {
                expected[i] = 2;
                i += 1;
            }
            check(n, &expected[..k as usize]);
            k += 1;
        }
    }

    #[test]
    fn power_of_two_sixty_three() {
        let n = 1u64 << 63;
        let mut out = [0u64; 64];
        let count = factorize(n, &mut out);
        assert_eq!(count, 63);
        let mut i = 0;
        while i < count {
            assert_eq!(out[i], 2);
            i += 1;
        }
    }

    #[test]
    fn power_of_three() {
        check(243, &[3, 3, 3, 3, 3]);
    }

    #[test]
    fn prime_power_mixed() {
        // 2^3 * 3^2 * 5 = 360.
        check(360, &[2, 2, 2, 3, 3, 5]);
        // 100 = 2^2 * 5^2.
        check(100, &[2, 2, 5, 5]);
    }

    // ---- Semiprimes -----------------------------------------------------

    #[test]
    fn semiprime_small() {
        check(15, &[3, 5]);
        check(35, &[5, 7]);
        check(77, &[7, 11]);
        check(221, &[13, 17]);
    }

    #[test]
    fn semiprime_equal_primes() {
        check(9, &[3, 3]);
        check(49, &[7, 7]);
        check(1000000014000000049, &[1000000007, 1000000007]);
    }

    #[test]
    fn semiprime_large_distinct() {
        check(1000000016000000063, &[1000000007, 1000000009]);
    }

    // ---- Primes factorize to themselves ---------------------------------

    #[test]
    fn prime_factorizes_to_singleton() {
        check(1000000007, &[1000000007]);
        check(2305843009213693951, &[2305843009213693951]);
        check(104729, &[104729]);
    }

    // ---- Structural property checks -------------------------------------

    #[test]
    fn verify_product_and_primality_batch() {
        let samples = [
            2u64,
            3,
            720,
            8051,
            10403,
            600851475143,
            1000000016000000063,
            999983,
            1024,
            123456789,
            987654321,
            4294967291,
            18446744073709551557,
        ];
        let mut i = 0;
        while i < samples.len() {
            verify(samples[i]);
            i += 1;
        }
    }

    #[test]
    fn verify_edges() {
        verify(0);
        verify(1);
        verify(2);
    }

    #[test]
    fn factorization_is_ascending_for_range() {
        let mut n: u64 = 2;
        while n < 500 {
            let mut out = [0u64; 64];
            let count = factorize(n, &mut out);
            let mut i = 1;
            while i < count {
                assert!(out[i - 1] <= out[i], "not ascending for {n}");
                i += 1;
            }
            n += 1;
        }
    }

    #[test]
    fn matches_naive_factorize_small_range() {
        let mut n: u64 = 2;
        while n < 2000 {
            let mut got = [0u64; 64];
            let mut want = [0u64; 64];
            let gc = factorize(n, &mut got);
            let wc = naive_factorize(n, &mut want);
            assert_eq!(gc, wc, "count mismatch at {n}");
            assert_eq!(&got[..gc], &want[..wc], "factors mismatch at {n}");
            n += 1;
        }
    }

    #[test]
    fn matches_naive_factorize_pseudo_random() {
        // A simple LCG walk over mid-sized composites, cross-checked by
        // trial division (kept small enough that naive stays fast).
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut iter = 0;
        while iter < 200 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let n = 2 + (state % 4_000_000);
            let mut got = [0u64; 64];
            let mut want = [0u64; 64];
            let gc = factorize(n, &mut got);
            let wc = naive_factorize(n, &mut want);
            assert_eq!(gc, wc, "count mismatch at {n}");
            assert_eq!(&got[..gc], &want[..wc], "factors mismatch at {n}");
            iter += 1;
        }
    }

    #[test]
    fn factor_count_never_exceeds_bound() {
        let samples = [1u64 << 63, 720, 1000000016000000063, 18446744073709551615];
        let mut i = 0;
        while i < samples.len() {
            let mut out = [0u64; 64];
            let count = factorize(samples[i], &mut out);
            assert!(count <= 63, "count {count} exceeds 63 for {}", samples[i]);
            i += 1;
        }
    }

    // ---- Low-level helpers ----------------------------------------------

    #[test]
    fn mulmod_basic() {
        assert_eq!(mulmod(0, 123, 7), 0);
        assert_eq!(mulmod(6, 6, 7), 1);
        assert_eq!(mulmod(1000000006, 1000000006, 1000000007), 1);
    }

    #[test]
    fn mulmod_no_overflow_near_max() {
        let m = u64::MAX;
        // (m-1)^2 mod m == 1 for any modulus m > 1.
        assert_eq!(mulmod(m - 1, m - 1, m), 1);
    }

    #[test]
    fn modpow_basic() {
        assert_eq!(modpow(2, 10, 1000), 24);
        assert_eq!(modpow(3, 0, 7), 1);
        assert_eq!(modpow(7, 1, 7), 0);
        assert_eq!(modpow(5, 3, 13), 8);
    }

    #[test]
    fn modpow_fermat_little_theorem() {
        // a^(p-1) == 1 (mod p) for prime p and gcd(a, p) == 1.
        let p = 1000000007u64;
        assert_eq!(modpow(2, p - 1, p), 1);
        assert_eq!(modpow(123456, p - 1, p), 1);
    }

    #[test]
    fn modpow_modulus_one() {
        assert_eq!(modpow(5, 3, 1), 0);
    }

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(12, 0), 12);
        assert_eq!(gcd(0, 9), 9);
        assert_eq!(gcd(54, 24), 6);
        assert_eq!(gcd(1000000007, 1000000009), 1);
        assert_eq!(gcd(720, 48), 48);
    }

    #[test]
    fn u64_max_factorization() {
        // u64::MAX = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417.
        check(18446744073709551615, &[3, 5, 17, 257, 641, 65537, 6700417]);
    }

    #[test]
    fn consecutive_integers_round_trip() {
        let mut n: u64 = 2;
        while n < 300 {
            let mut out = [0u64; 64];
            let count = factorize(n, &mut out);
            let mut product: u128 = 1;
            let mut i = 0;
            while i < count {
                product *= out[i] as u128;
                i += 1;
            }
            assert_eq!(product, n as u128, "round trip failed for {n}");
            n += 1;
        }
    }
}
