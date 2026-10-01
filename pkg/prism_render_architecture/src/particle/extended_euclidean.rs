//! Extended Euclidean algorithm over the integers: greatest common divisor
//! (`gcd`) together with the `Bezout` coefficients, plus a modular inverse
//! built on top of it. CPU gold standard, pure integer arithmetic.
//!
//! For inputs `a` and `b` ([`i64`]), [`egcd`] returns a triple `(g, x, y)`
//! satisfying the `Bezout` identity `a*x + b*y == g`, where
//! `g == gcd(|a|, |b|)` and `g >= 0` by convention. Unlike a plain
//! `gcd`/`lcm` routine or a modular-exponentiation inverse, the defining
//! product of this module is the pair of `Bezout` coefficients `x` and `y`.
//!
//! ## Iterative formulation
//!
//! The algorithm keeps the running remainders `(old_r, r)` alongside the two
//! coefficient sequences `(old_s, s)` and `(old_t, t)`. Each step takes the
//! quotient `q = old_r / r` and applies the identical linear update to all
//! three pairs. When `r` reaches zero, `old_r` holds the `gcd` and
//! `(old_s, old_t)` are the matching `Bezout` coefficients. If `old_r` came
//! out negative (possible when exactly one input is negative), the whole
//! triple is negated so that the reported `g` is non-negative while the
//! identity is preserved.
//!
//! ## Overflow safety
//!
//! All intermediate arithmetic runs in [`i128`]. The products `q*r`, `q*s`,
//! and `q*t` can exceed the range of [`i64`] during the sweep even though the
//! final `Bezout` coefficients for [`i64`] inputs are bounded and fit back
//! into [`i64`]. Signed overflow panics in debug builds, so the wider
//! accumulator type is what keeps the routine total. No transcendental or
//! floating-point operations are used; only integer division and remainder
//! appear.
//!
//! ## Modular inverse
//!
//! [`mod_inverse`] computes the inverse of `a` modulo `m`. It reduces `a`
//! into `[0, m)`, runs [`egcd`] against `m`, and succeeds only when the `gcd`
//! is `1` (i.e. `a` and `m` are coprime); otherwise there is no inverse and
//! the result is `None`. The returned residue is normalized into `[0, m)`
//! using [`i128`] arithmetic so no negative intermediate escapes.

/// Extended Euclidean algorithm.
///
/// Returns `(g, x, y)` with `a*x + b*y == g` and `g == gcd(|a|, |b|) >= 0`.
/// All intermediate arithmetic is performed in [`i128`] to avoid signed
/// overflow; the final triple fits back into [`i64`].
pub fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut old_r, mut r): (i128, i128) = (a as i128, b as i128);
    let (mut old_s, mut s): (i128, i128) = (1, 0);
    let (mut old_t, mut t): (i128, i128) = (0, 1);

    while r != 0 {
        let q = old_r / r;

        let next_r = old_r - q * r;
        old_r = r;
        r = next_r;

        let next_s = old_s - q * s;
        old_s = s;
        s = next_s;

        let next_t = old_t - q * t;
        old_t = t;
        t = next_t;
    }

    if old_r < 0 {
        old_r = -old_r;
        old_s = -old_s;
        old_t = -old_t;
    }

    (old_r as i64, old_s as i64, old_t as i64)
}

/// Greatest common divisor of two [`i64`] values, always non-negative.
///
/// Convenience wrapper around [`egcd`] that discards the `Bezout`
/// coefficients.
pub fn gcd_i64(a: i64, b: i64) -> i64 {
    egcd(a, b).0
}

/// Modular inverse of `a` modulo `m`.
///
/// Returns `Some(inv)` with `inv` in `[0, m)` and `(inv * a) mod m == 1` when
/// `a` and `m` are coprime; returns `None` when `gcd(a, m) != 1` (including the
/// degenerate modulus `m == 0`).
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }

    let a_mod = (a % m) as i64;
    let (g, x, _y) = egcd(a_mod, m as i64);
    if g != 1 {
        return None;
    }

    let m_i = m as i128;
    let inv = ((x as i128 % m_i) + m_i) % m_i;
    Some(inv as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies the `Bezout` identity `a*x + b*y == g` using [`i128`] so the
    /// check itself cannot overflow.
    #[cfg(test)]
    fn identity_holds(a: i64, b: i64, g: i64, x: i64, y: i64) -> bool {
        (a as i128) * (x as i128) + (b as i128) * (y as i128) == g as i128
    }

    // -- Hard reference vectors (checked against an independent implementation).

    #[test]
    fn egcd_240_46() {
        assert_eq!(egcd(240, 46), (2, -9, 47));
    }

    #[test]
    fn egcd_46_240() {
        assert_eq!(egcd(46, 240), (2, 47, -9));
    }

    #[test]
    fn egcd_17_5() {
        assert_eq!(egcd(17, 5), (1, -2, 7));
    }

    #[test]
    fn egcd_0_5() {
        assert_eq!(egcd(0, 5), (5, 0, 1));
    }

    #[test]
    fn egcd_5_0() {
        assert_eq!(egcd(5, 0), (5, 1, 0));
    }

    #[test]
    fn egcd_35_15() {
        assert_eq!(egcd(35, 15), (5, 1, -2));
    }

    #[test]
    fn egcd_1234_56() {
        assert_eq!(egcd(1234, 56), (2, 1, -22));
    }

    #[test]
    fn mod_inverse_3_11() {
        assert_eq!(mod_inverse(3, 11), Some(4));
    }

    #[test]
    fn mod_inverse_7_26() {
        assert_eq!(mod_inverse(7, 26), Some(15));
    }

    #[test]
    fn mod_inverse_2_4_none() {
        assert_eq!(mod_inverse(2, 4), None);
    }

    // -- Boundary cases.

    #[test]
    fn egcd_0_0() {
        let (g, x, y) = egcd(0, 0);
        assert_eq!(g, 0);
        assert!(identity_holds(0, 0, g, x, y));
    }

    #[test]
    fn egcd_1_1() {
        let (g, x, y) = egcd(1, 1);
        assert_eq!(g, 1);
        assert!(identity_holds(1, 1, g, x, y));
    }

    #[test]
    fn egcd_1_0() {
        assert_eq!(egcd(1, 0), (1, 1, 0));
    }

    #[test]
    fn egcd_0_1() {
        assert_eq!(egcd(0, 1), (1, 0, 1));
    }

    #[test]
    fn egcd_7_1() {
        let (g, x, y) = egcd(7, 1);
        assert_eq!(g, 1);
        assert!(identity_holds(7, 1, g, x, y));
    }

    #[test]
    fn egcd_coprime_pair() {
        let (g, x, y) = egcd(13, 17);
        assert_eq!(g, 1);
        assert!(identity_holds(13, 17, g, x, y));
    }

    // -- Negative inputs: identity and non-negative `g` must still hold.

    #[test]
    fn egcd_neg_240_46() {
        let (g, x, y) = egcd(-240, 46);
        assert_eq!(g, 2);
        assert!(g >= 0);
        assert!(identity_holds(-240, 46, g, x, y));
    }

    #[test]
    fn egcd_240_neg_46() {
        let (g, x, y) = egcd(240, -46);
        assert_eq!(g, 2);
        assert!(g >= 0);
        assert!(identity_holds(240, -46, g, x, y));
    }

    #[test]
    fn egcd_neg_240_neg_46() {
        let (g, x, y) = egcd(-240, -46);
        assert_eq!(g, 2);
        assert!(g >= 0);
        assert!(identity_holds(-240, -46, g, x, y));
    }

    #[test]
    fn egcd_neg_17_5() {
        let (g, x, y) = egcd(-17, 5);
        assert_eq!(g, 1);
        assert!(g >= 0);
        assert!(identity_holds(-17, 5, g, x, y));
    }

    // -- Property sweeps.

    #[test]
    fn bezout_identity_positive_sweep() {
        let pairs: [(i64, i64); 8] = [
            (240, 46),
            (46, 240),
            (17, 5),
            (35, 15),
            (1234, 56),
            (99, 78),
            (1000, 1),
            (123456, 7890),
        ];
        for &(a, b) in pairs.iter() {
            let (g, x, y) = egcd(a, b);
            assert!(identity_holds(a, b, g, x, y));
        }
    }

    #[test]
    fn bezout_identity_negative_sweep() {
        let pairs: [(i64, i64); 6] = [
            (-240, -46),
            (-17, -5),
            (-35, -15),
            (-1234, -56),
            (-99, -78),
            (-1000, -7),
        ];
        for &(a, b) in pairs.iter() {
            let (g, x, y) = egcd(a, b);
            assert!(g >= 0);
            assert!(identity_holds(a, b, g, x, y));
        }
    }

    #[test]
    fn bezout_identity_mixed_sweep() {
        let pairs: [(i64, i64); 6] = [
            (-240, 46),
            (240, -46),
            (-17, 5),
            (17, -5),
            (-123456, 7890),
            (123456, -7890),
        ];
        for &(a, b) in pairs.iter() {
            let (g, x, y) = egcd(a, b);
            assert!(g >= 0);
            assert!(identity_holds(a, b, g, x, y));
        }
    }

    #[test]
    fn g_is_nonnegative_sweep() {
        let pairs: [(i64, i64); 9] = [
            (0, 0),
            (0, 5),
            (5, 0),
            (-5, 0),
            (0, -5),
            (-12, -18),
            (-12, 18),
            (12, -18),
            (-1, -1),
        ];
        for &(a, b) in pairs.iter() {
            let (g, _x, _y) = egcd(a, b);
            assert!(g >= 0);
        }
    }

    #[test]
    fn gcd_symmetry_sweep() {
        let pairs: [(i64, i64); 7] = [
            (240, 46),
            (35, 15),
            (1234, 56),
            (99, 78),
            (-12, 18),
            (0, 7),
            (13, 17),
        ];
        for &(a, b) in pairs.iter() {
            assert_eq!(gcd_i64(a, b), gcd_i64(b, a));
        }
    }

    #[test]
    fn gcd_matches_known_values() {
        let cases: [(i64, i64, i64); 7] = [
            (240, 46, 2),
            (35, 15, 5),
            (1234, 56, 2),
            (17, 5, 1),
            (0, 5, 5),
            (5, 0, 5),
            (0, 0, 0),
        ];
        for &(a, b, g) in cases.iter() {
            assert_eq!(gcd_i64(a, b), g);
        }
    }

    #[test]
    fn gcd_divides_inputs_is_multiple_of() {
        let pairs: [(u64, u64); 5] = [(240, 46), (35, 15), (1234, 56), (99, 78), (1000, 24)];
        for &(a, b) in pairs.iter() {
            let g = gcd_i64(a as i64, b as i64) as u64;
            assert!(g >= 1);
            assert!(a.is_multiple_of(g));
            assert!(b.is_multiple_of(g));
        }
    }

    // -- Modular inverse.

    #[test]
    fn mod_inverse_product_is_one_sweep() {
        let cases: [(u64, u64); 7] = [
            (3, 11),
            (7, 26),
            (5, 7),
            (10, 17),
            (123, 4567),
            (9, 100),
            (2, 15),
        ];
        for &(a, m) in cases.iter() {
            let inv = mod_inverse(a, m).expect("inverse exists for coprime pair");
            let prod = ((a as u128) * (inv as u128)) % (m as u128);
            assert_eq!(prod, 1);
        }
    }

    #[test]
    fn mod_inverse_none_when_not_coprime_sweep() {
        let cases: [(u64, u64); 6] = [(2, 4), (6, 9), (4, 8), (10, 15), (14, 21), (0, 5)];
        for &(a, m) in cases.iter() {
            assert_eq!(mod_inverse(a, m), None);
        }
    }

    #[test]
    fn mod_inverse_5_7() {
        assert_eq!(mod_inverse(5, 7), Some(3));
    }

    #[test]
    fn mod_inverse_mod_one() {
        // Everything is congruent to 0 modulo 1, so the inverse residue is 0.
        assert_eq!(mod_inverse(5, 1), Some(0));
        assert_eq!(mod_inverse(0, 1), Some(0));
    }

    #[test]
    fn mod_inverse_a_larger_than_m() {
        // 100 mod 7 == 2, and 2 * 4 == 8 == 1 (mod 7).
        assert_eq!(mod_inverse(100, 7), Some(4));
    }

    #[test]
    fn mod_inverse_1_m() {
        let mods: [u64; 4] = [2, 7, 26, 1000];
        for &m in mods.iter() {
            assert_eq!(mod_inverse(1, m), Some(1));
        }
    }

    #[test]
    fn mod_inverse_large_prime_modulus() {
        let m: u64 = 1_000_000_007;
        let a: u64 = 123_456_789;
        let inv = mod_inverse(a, m).expect("inverse exists modulo a prime");
        let prod = ((a as u128) * (inv as u128)) % (m as u128);
        assert_eq!(prod, 1);
    }

    #[test]
    fn mod_inverse_even_modulus_none() {
        // gcd(8, 12) == 4 != 1, so no inverse exists.
        assert_eq!(mod_inverse(8, 12), None);
    }

    #[test]
    fn mod_inverse_zero_modulus_none() {
        assert_eq!(mod_inverse(5, 0), None);
    }

    #[test]
    fn mod_inverse_residue_in_range_sweep() {
        let cases: [(u64, u64); 5] = [
            (3, 11),
            (7, 26),
            (123, 4567),
            (10, 17),
            (123_456_789, 1_000_000_007),
        ];
        for &(a, m) in cases.iter() {
            let inv = mod_inverse(a, m).expect("coprime inverse exists");
            assert!(inv < m);
        }
    }

    // -- Larger magnitude `egcd` behavior.

    #[test]
    fn egcd_large_values() {
        let a: i64 = 1_000_000_007;
        let b: i64 = 998_244_353;
        let (g, x, y) = egcd(a, b);
        assert_eq!(g, 1);
        assert!(identity_holds(a, b, g, x, y));
    }

    #[test]
    fn egcd_large_common_factor() {
        let a: i64 = 1_000_000_000;
        let b: i64 = 600_000_000;
        let (g, x, y) = egcd(a, b);
        assert_eq!(g, 200_000_000);
        assert!(identity_holds(a, b, g, x, y));
    }

    #[test]
    fn egcd_identity_large_sweep() {
        let pairs: [(i64, i64); 6] = [
            (9_999_999_967, 7),
            (123_456_789, 987_654_321),
            (2_000_000_000, 1_999_999_999),
            (-123_456_789, 987_654_321),
            (123_456_789, -987_654_321),
            (-2_000_000_000, -1_999_999_999),
        ];
        for &(a, b) in pairs.iter() {
            let (g, x, y) = egcd(a, b);
            assert!(g >= 0);
            assert!(identity_holds(a, b, g, x, y));
        }
    }

    #[test]
    fn gcd_i64_helper_basic() {
        assert_eq!(gcd_i64(240, 46), 2);
        assert_eq!(gcd_i64(-240, 46), 2);
        assert_eq!(gcd_i64(0, 0), 0);
    }

    #[test]
    fn gcd_zero_with_value() {
        assert_eq!(gcd_i64(0, 42), 42);
        assert_eq!(gcd_i64(42, 0), 42);
        assert_eq!(gcd_i64(0, -42), 42);
    }

    #[test]
    fn egcd_coefficients_fit_i64() {
        // Final `Bezout` coefficients must round-trip through `i64` without
        // loss; re-checking the identity confirms no truncation occurred.
        let pairs: [(i64, i64); 4] = [
            (1_000_000_007, 998_244_353),
            (9_999_999_967, 7),
            (123_456_789, 987_654_321),
            (2_000_000_000, 1_999_999_999),
        ];
        for &(a, b) in pairs.iter() {
            let (g, x, y) = egcd(a, b);
            assert!(identity_holds(a, b, g, x, y));
        }
    }
}
