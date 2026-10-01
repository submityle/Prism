//! Exact integer greatest-common-divisor / least-common-multiple math.
//!
//! Particle scheduling, ratio quantization, and deterministic period math all
//! need exact divisibility answers, so this module works purely with integers:
//! Euclidean remainders, `Stein`'s binary reduction with `trailing_zeros`
//! shifts, and the extended Euclidean recurrence. No floating point, no
//! transcendental function, and no lossy `f64` detour is ever touched, so every
//! result is bit-for-bit reproducible on a `CPU` core or a `GPU` integer
//! pipeline alike.
//!
//! The classic `gcd`/`lcm` identity `lcm(a, b) == a / gcd(a, b) * b` is
//! implemented divide-first so the product cannot overflow when the true `lcm`
//! still fits in `u64`; a `checked_mul` variant reports the overflow case
//! explicitly. The extended routine returns Bezout coefficients `(x, y)` with
//! `a * x + b * y == g`; those coefficients are accumulated in the `i128`
//! domain so the intermediate products never wrap even for extreme `i64`
//! inputs, and the final coefficients always fit back into `i64`.

/// Returns the greatest common divisor of two unsigned values.
///
/// This is the iterative Euclidean algorithm: repeatedly replace `(a, b)` with
/// `(b, a % b)` until the second operand is zero. By convention
/// `gcd(0, 0) == 0`, and `gcd(n, 0) == gcd(0, n) == n`.
#[must_use]
pub fn gcd_u64(a: u64, b: u64) -> u64 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Returns the greatest common divisor of two signed values as an unsigned
/// magnitude.
///
/// Each operand is reduced to its magnitude with `unsigned_abs`, which maps
/// `i64::MIN` to `1 << 63` without the overflow that plain negation would
/// cause, and the `gcd` is then computed in the `u64` domain.
#[must_use]
pub fn gcd_i64(a: i64, b: i64) -> u64 {
    gcd_u64(a.unsigned_abs(), b.unsigned_abs())
}

/// Returns the greatest common divisor using `Stein`'s binary algorithm.
///
/// `Stein`'s method replaces the division of the Euclidean algorithm with the
/// cheaper primitives `trailing_zeros`, subtraction, and shifts. It factors out
/// the common power of two once, then repeatedly strips odd/even factors and
/// subtracts the smaller odd value from the larger. The result agrees with
/// [`gcd_u64`] on every input, including the `gcd(0, 0) == 0` convention.
#[must_use]
pub fn binary_gcd_u64(a: u64, b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    // Extract the common factor of two shared by both operands.
    let shift = (a | b).trailing_zeros();
    // Reduce `a` to odd once; it stays odd for the rest of the loop.
    let mut a = a >> a.trailing_zeros();
    let mut b = b;
    while b != 0 {
        // Reduce `b` to odd; factors of two never divide the final `gcd`
        // beyond the common `shift` removed above.
        b >>= b.trailing_zeros();
        // Both `a` and `b` are odd here; order them so the subtraction stays
        // non-negative and yields an even (hence further reducible) value.
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
    }
    a << shift
}

/// Returns the least common multiple of two unsigned values.
///
/// Uses the identity `lcm(a, b) == a / gcd(a, b) * b`, dividing before
/// multiplying so the intermediate stays as small as possible. Returns `0` when
/// either operand is `0`, matching the degenerate convention that the multiples
/// of zero contain only zero. This can still overflow silently when the true
/// `lcm` exceeds `u64::MAX`; use [`lcm_checked_u64`] to detect that case.
#[must_use]
pub fn lcm_u64(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    // `a / gcd` is exact because `gcd` divides `a`; dividing first keeps the
    // product below the true `lcm` value.
    a / gcd_u64(a, b) * b
}

/// Returns the least common multiple, or `None` when it overflows `u64`.
///
/// Identical to [`lcm_u64`] except the final multiply uses `checked_mul`, so an
/// `lcm` that does not fit in `u64` reports `None` instead of wrapping. Returns
/// `Some(0)` when either operand is `0`.
#[must_use]
pub fn lcm_checked_u64(a: u64, b: u64) -> Option<u64> {
    if a == 0 || b == 0 {
        return Some(0);
    }
    // The division is exact, then `checked_mul` guards the final product.
    (a / gcd_u64(a, b)).checked_mul(b)
}

/// Returns `(g, x, y)` such that `a * x + b * y == g`, where `g` is the
/// non-negative greatest common divisor of `a` and `b`.
///
/// This is the iterative extended Euclidean algorithm. The Bezout coefficients
/// are carried in the `i128` domain so the running products never overflow even
/// for the widest `i64` inputs; the returned coefficients and `g` always fit
/// back into `i64`. The identity `a * x + b * y == g` holds for negative
/// operands and for zero operands (`ext_gcd_i64(0, 0)` returns
/// `(0, 0, 0)`).
#[must_use]
pub fn ext_gcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    // Work entirely in `i128` to keep the coefficient recurrence overflow-free.
    let mut old_r: i128 = i128::from(a);
    let mut r: i128 = i128::from(b);
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
    // Normalize so the returned `gcd` is non-negative, flipping the Bezout
    // coefficients in tandem to preserve the identity.
    if old_r < 0 {
        old_r = -old_r;
        old_s = -old_s;
        old_t = -old_t;
    }
    (old_r as i64, old_s as i64, old_t as i64)
}

/// Returns `true` when `a` and `b` are coprime, i.e. their `gcd` is `1`.
#[must_use]
pub fn coprime_u64(a: u64, b: u64) -> bool {
    gcd_u64(a, b) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_basic_pairs() {
        assert_eq!(gcd_u64(12, 18), 6);
        assert_eq!(gcd_u64(17, 5), 1);
        assert_eq!(gcd_u64(48, 36), 12);
        assert_eq!(gcd_u64(1071, 462), 21);
    }

    #[test]
    fn gcd_with_zero() {
        assert_eq!(gcd_u64(0, 5), 5);
        assert_eq!(gcd_u64(5, 0), 5);
        assert_eq!(gcd_u64(0, 0), 0);
    }

    #[test]
    fn gcd_equal_operands() {
        assert_eq!(gcd_u64(7, 7), 7);
        assert_eq!(gcd_u64(1, 1), 1);
    }

    #[test]
    fn gcd_one_is_coprime() {
        assert_eq!(gcd_u64(1, 999_983), 1);
        assert_eq!(gcd_u64(999_983, 1), 1);
    }

    #[test]
    fn gcd_commutative_property() {
        let pairs = [(12u64, 18u64), (1071, 462), (100, 75), (0, 9), (13, 0)];
        for &(a, b) in &pairs {
            assert_eq!(gcd_u64(a, b), gcd_u64(b, a));
        }
    }

    #[test]
    fn gcd_large_values() {
        assert_eq!(gcd_u64(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(gcd_u64(u64::MAX, 0), u64::MAX);
        // `u64::MAX == 3 * 5 * 17 * 257 * 641 * 65537 * 6700417`.
        assert_eq!(gcd_u64(u64::MAX, 3), 3);
    }

    #[test]
    fn binary_gcd_matches_reference_pairs() {
        let pairs = [
            (12u64, 18u64),
            (17, 5),
            (0, 5),
            (5, 0),
            (0, 0),
            (48, 36),
            (1071, 462),
        ];
        for &(a, b) in &pairs {
            assert_eq!(binary_gcd_u64(a, b), gcd_u64(a, b));
        }
    }

    #[test]
    fn binary_gcd_zero_conventions() {
        assert_eq!(binary_gcd_u64(0, 0), 0);
        assert_eq!(binary_gcd_u64(0, 42), 42);
        assert_eq!(binary_gcd_u64(42, 0), 42);
    }

    #[test]
    fn binary_gcd_matches_euclid_on_lcg() {
        // Deterministic 64-bit LCG (Knuth MMIX constants).
        let mut state: u64 = 0x0BAD_C0FF_EE12_3456;
        for _ in 0..50_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let a = state;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let b = state;
            assert_eq!(binary_gcd_u64(a, b), gcd_u64(a, b));
        }
    }

    #[test]
    fn binary_gcd_powers_of_two() {
        for exp in 0u32..64 {
            let n = 1u64 << exp;
            assert_eq!(binary_gcd_u64(n, n), n);
        }
        assert_eq!(binary_gcd_u64(1 << 20, 1 << 10), 1 << 10);
    }

    #[test]
    fn gcd_i64_absolute_value() {
        assert_eq!(gcd_i64(-12, 18), 6);
        assert_eq!(gcd_i64(12, -18), 6);
        assert_eq!(gcd_i64(-12, -18), 6);
        assert_eq!(gcd_i64(-17, 5), 1);
    }

    #[test]
    fn gcd_i64_min_does_not_panic() {
        // `i64::MIN.unsigned_abs() == 1 << 63`; this must not overflow.
        assert_eq!(gcd_i64(i64::MIN, 0), 1u64 << 63);
        assert_eq!(gcd_i64(0, i64::MIN), 1u64 << 63);
        assert_eq!(gcd_i64(i64::MIN, i64::MIN), 1u64 << 63);
    }

    #[test]
    fn gcd_i64_zero_pair() {
        assert_eq!(gcd_i64(0, 0), 0);
    }

    #[test]
    fn lcm_basic_pairs() {
        assert_eq!(lcm_u64(4, 6), 12);
        assert_eq!(lcm_u64(21, 6), 42);
        assert_eq!(lcm_u64(3, 4), 12);
        assert_eq!(lcm_u64(6, 8), 24);
    }

    #[test]
    fn lcm_with_zero() {
        assert_eq!(lcm_u64(0, 5), 0);
        assert_eq!(lcm_u64(5, 0), 0);
        assert_eq!(lcm_u64(0, 0), 0);
    }

    #[test]
    fn lcm_equal_and_coprime() {
        assert_eq!(lcm_u64(7, 7), 7);
        assert_eq!(lcm_u64(5, 7), 35);
    }

    #[test]
    fn lcm_divide_before_multiply_no_overflow() {
        // A large pair whose `lcm` still fits: divide-first avoids overflow.
        let a = 1_000_000_000u64;
        let b = 999_999_999u64;
        // These are coprime, so the `lcm` is their product.
        assert_eq!(lcm_u64(a, b), a * b);
    }

    #[test]
    fn lcm_checked_normal_cases() {
        assert_eq!(lcm_checked_u64(4, 6), Some(12));
        assert_eq!(lcm_checked_u64(21, 6), Some(42));
        assert_eq!(lcm_checked_u64(0, 5), Some(0));
        assert_eq!(lcm_checked_u64(5, 0), Some(0));
    }

    #[test]
    fn lcm_checked_detects_overflow() {
        // Two large coprime values whose product exceeds `u64::MAX`.
        let a = (1u64 << 62) + 1;
        let b = (1u64 << 62) + 3;
        assert_eq!(gcd_u64(a, b), 1);
        assert_eq!(lcm_checked_u64(a, b), None);
    }

    #[test]
    fn lcm_checked_at_boundary() {
        // `lcm` exactly equal to `u64::MAX` must still be `Some`.
        assert_eq!(lcm_checked_u64(u64::MAX, u64::MAX), Some(u64::MAX));
        assert_eq!(lcm_checked_u64(u64::MAX, 1), Some(u64::MAX));
    }

    #[test]
    fn ext_gcd_identity_positive() {
        let pairs = [(12i64, 18i64), (1071, 462), (48, 36), (17, 5)];
        for &(a, b) in &pairs {
            let (g, x, y) = ext_gcd_i64(a, b);
            assert_eq!(g as u64, gcd_i64(a, b));
            assert!(g >= 0);
            assert_eq!(a * x + b * y, g);
        }
    }

    #[test]
    fn ext_gcd_identity_with_negatives() {
        let pairs = [(-12i64, 18i64), (12, -18), (-12, -18), (-1071, 462)];
        for &(a, b) in &pairs {
            let (g, x, y) = ext_gcd_i64(a, b);
            assert_eq!(g as u64, gcd_i64(a, b));
            assert!(g >= 0);
            assert_eq!(a * x + b * y, g);
        }
    }

    #[test]
    fn ext_gcd_with_zero_operands() {
        // Use bound variables for the operands so the Bezout identity check
        // `a * x + b * y == g` exercises the returned coefficients instead of
        // multiplying by a literal zero (which would be a trivial constant).
        let (a, b) = (0_i64, 5_i64);
        let (g, x, y) = ext_gcd_i64(a, b);
        assert_eq!(g, 5);
        assert_eq!(a * x + b * y, g);

        let (a, b) = (7_i64, 0_i64);
        let (g, x, y) = ext_gcd_i64(a, b);
        assert_eq!(g, 7);
        assert_eq!(a * x + b * y, g);

        let (a, b) = (0_i64, 0_i64);
        let (g, x, y) = ext_gcd_i64(a, b);
        assert_eq!(g, 0);
        // Any coefficients satisfy `0 * x + 0 * y == 0`; only `g` is defined.
        assert_eq!(a * x + b * y, g);
    }

    #[test]
    fn ext_gcd_coprime_pair() {
        let (g, x, y) = ext_gcd_i64(17, 5);
        assert_eq!(g, 1);
        assert_eq!(17 * x + 5 * y, 1);
    }

    #[test]
    fn ext_gcd_g_matches_gcd_on_lcg() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_0001;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Keep magnitudes moderate so `a * x + b * y` stays inside `i64`.
            let a = (state >> 40) as i64 - (1 << 23);
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let b = (state >> 40) as i64 - (1 << 23);
            let (g, x, y) = ext_gcd_i64(a, b);
            assert_eq!(g as u64, gcd_i64(a, b));
            assert!(g >= 0);
            assert_eq!(a * x + b * y, g);
        }
    }

    #[test]
    fn ext_gcd_extreme_inputs() {
        let (g, x, y) = ext_gcd_i64(i64::MAX, i64::MIN);
        assert!(g >= 0);
        // Verify the identity in the `i128` domain to avoid `i64` overflow.
        let lhs = i128::from(i64::MAX) * i128::from(x) + i128::from(i64::MIN) * i128::from(y);
        assert_eq!(lhs, i128::from(g));
    }

    #[test]
    fn coprime_true_and_false() {
        assert!(coprime_u64(17, 5));
        assert!(coprime_u64(9, 28));
        assert!(!coprime_u64(12, 18));
        assert!(!coprime_u64(100, 75));
    }

    #[test]
    fn coprime_edge_cases() {
        assert!(coprime_u64(1, 1));
        assert!(coprime_u64(1, 0));
        assert!(!coprime_u64(0, 0));
    }

    #[test]
    fn lcm_gcd_product_identity_on_lcg() {
        // For non-zero `a`, `b`: `gcd(a, b) * lcm(a, b) == a * b` whenever the
        // product fits. Use small operands so the product stays inside `u64`.
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let a = (state & 0xFFFF) + 1;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let b = (state & 0xFFFF) + 1;
            let g = gcd_u64(a, b);
            let l = lcm_u64(a, b);
            assert_eq!(g * l, a * b);
        }
    }

    #[test]
    fn gcd_divides_both_operands() {
        let pairs = [(48u64, 36u64), (1071, 462), (1000, 250), (13, 91)];
        for &(a, b) in &pairs {
            let g = gcd_u64(a, b);
            assert_eq!(a % g, 0);
            assert_eq!(b % g, 0);
        }
    }
}
