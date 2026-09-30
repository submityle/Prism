//! Exact integer square/cube roots for deterministic particle math.
//!
//! This module computes *floor* integer roots with pure integer arithmetic:
//! shifts, comparisons, `wrapping` add/sub, and exclusive-or only. No floating
//! point, no `sqrt`/`cbrt`, and no transcendental function is ever touched.
//! That is the whole selling point — the result is bit-for-bit reproducible on
//! any platform, including a `GPU` integer pipeline or a `CPU` core with no
//! `FPU`, and it carries zero rounding error.
//!
//! Contrast this with the common shortcut of `(n as f64).sqrt() as u64`: that
//! path first rounds `n` into a 53-bit mantissa, rounds again after the square
//! root, and truncates a third time. Near a perfect square, and especially for
//! large 64-bit inputs, those unit-in-the-last-place errors flip the floor by
//! one, so `is_perfect_square` misclassifies boundary values. The routines here
//! are exact by construction: for every input the post-condition
//! `r*r <= n < (r+1)*(r+1)` holds (checked in the `u64` domain without
//! overflowing `(r+1)^2` via the equivalent `n - r*r <= 2*r` test).
//!
//! The square-root routines use the classic bit-by-bit algorithm. It seeds a
//! probe bit at the highest even bit position not exceeding `n`, then walks
//! down two bits at a time, tentatively setting each bit of the result and
//! subtracting when the trial square still fits. Because the trial square is
//! built incrementally from the high bits, no intermediate value overflows the
//! integer width, so `isqrt_u64` is safe across the entire `u64` range.

/// Returns the floor integer square root of `n`: the largest `r` with
/// `r * r <= n`.
///
/// Pure integer bit-by-bit algorithm, computed in the `u32` domain. No
/// floating point and no overflow: the running remainder and trial value stay
/// within `u32`.
#[must_use]
pub fn isqrt_u32(n: u32) -> u32 {
    let mut remainder = n;
    let mut result: u32 = 0;
    // Highest even-positioned probe bit for a 32-bit value is `1 << 30`.
    let mut bit: u32 = 1 << 30;
    while bit > remainder {
        bit >>= 2;
    }
    while bit != 0 {
        let trial = result + bit;
        if remainder >= trial {
            remainder -= trial;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result
}

/// Returns the floor integer square root of `n`: the largest `r` with
/// `r * r <= n`.
///
/// Pure integer bit-by-bit algorithm over the full `u64` range. The trial
/// square is assembled from the high bits downward, so no intermediate value
/// overflows `u64` even for `n == u64::MAX`.
#[must_use]
pub fn isqrt_u64(n: u64) -> u64 {
    let mut remainder = n;
    let mut result: u64 = 0;
    // Highest even-positioned probe bit for a 64-bit value is `1 << 62`.
    let mut bit: u64 = 1 << 62;
    while bit > remainder {
        bit >>= 2;
    }
    while bit != 0 {
        let trial = result + bit;
        if remainder >= trial {
            remainder -= trial;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    result
}

/// Returns `true` when `n` is a perfect square, using the exact integer root.
#[must_use]
pub fn is_perfect_square_u64(n: u64) -> bool {
    let r = isqrt_u64(n);
    r * r == n
}

/// Returns the floor integer cube root of `n`: the largest `r` with
/// `r * r * r <= n`.
///
/// Pure integer bit-by-bit algorithm. The result is built one bit at a time
/// from the most significant candidate bit down; each tentative bit is kept
/// only when the candidate's cube still fits inside `n`. The cube is formed
/// with checked multiplication so an over-large trial is simply rejected
/// rather than wrapping, keeping every step inside the `u64` domain. The cube
/// root of `u64::MAX` is below `1 << 22`, so the probe starts at bit 21.
#[must_use]
pub fn icbrt_u64(n: u64) -> u64 {
    let mut result: u64 = 0;
    let mut bit: u64 = 1 << 21;
    while bit != 0 {
        let candidate = result | bit;
        let cube = candidate
            .checked_mul(candidate)
            .and_then(|square| square.checked_mul(candidate));
        if cube.is_some_and(|c| c <= n) {
            result = candidate;
        }
        bit >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_u32_small_values() {
        assert_eq!(isqrt_u32(0), 0);
        assert_eq!(isqrt_u32(1), 1);
        assert_eq!(isqrt_u32(2), 1);
        assert_eq!(isqrt_u32(3), 1);
        assert_eq!(isqrt_u32(4), 2);
    }

    #[test]
    fn isqrt_u32_perfect_squares() {
        for r in 0u32..=1000 {
            assert_eq!(isqrt_u32(r * r), r);
        }
    }

    #[test]
    fn isqrt_u32_square_boundaries() {
        for r in 1u32..=1000 {
            let sq = r * r;
            assert_eq!(isqrt_u32(sq - 1), r - 1);
            assert_eq!(isqrt_u32(sq), r);
            assert_eq!(isqrt_u32(sq + 1), r);
        }
    }

    #[test]
    fn isqrt_u32_max() {
        let r = isqrt_u32(u32::MAX);
        assert_eq!(r, 65535);
        assert!(u64::from(r) * u64::from(r) <= u64::from(u32::MAX));
        let next = u64::from(r) + 1;
        assert!(next * next > u64::from(u32::MAX));
    }

    #[test]
    fn isqrt_u32_matches_u64_path() {
        for n in 0u32..=5000 {
            assert_eq!(u64::from(isqrt_u32(n)), isqrt_u64(u64::from(n)));
        }
    }

    #[test]
    fn isqrt_u64_small_values() {
        assert_eq!(isqrt_u64(0), 0);
        assert_eq!(isqrt_u64(1), 1);
        assert_eq!(isqrt_u64(2), 1);
        assert_eq!(isqrt_u64(3), 1);
        assert_eq!(isqrt_u64(4), 2);
        assert_eq!(isqrt_u64(8), 2);
        assert_eq!(isqrt_u64(9), 3);
    }

    #[test]
    fn isqrt_u64_perfect_squares() {
        for r in 0u64..=2000 {
            assert_eq!(isqrt_u64(r * r), r);
        }
    }

    #[test]
    fn isqrt_u64_large_perfect_squares() {
        let roots = [
            1_000_000u64,
            2_147_483_647,
            3_000_000_000,
            4_294_967_295,
            1u64 << 31,
        ];
        for &r in &roots {
            assert_eq!(isqrt_u64(r * r), r);
        }
    }

    #[test]
    fn isqrt_u64_square_boundaries() {
        for r in 1u64..=2000 {
            let sq = r * r;
            assert_eq!(isqrt_u64(sq - 1), r - 1);
            assert_eq!(isqrt_u64(sq), r);
            assert_eq!(isqrt_u64(sq + 1), r);
        }
    }

    #[test]
    fn isqrt_u64_max() {
        let r = isqrt_u64(u64::MAX);
        assert_eq!(r, 4_294_967_295);
        assert!(r * r <= u64::MAX);
        // (r+1)^2 would overflow, so use the equivalent remainder test.
        assert!(u64::MAX - r * r <= 2 * r);
    }

    #[test]
    fn isqrt_u64_near_2_pow_62() {
        let base = 1u64 << 62;
        for delta in 0u64..8 {
            let n = base + delta;
            let r = isqrt_u64(n);
            assert!(r * r <= n);
            assert!(n - r * r <= 2 * r);
        }
        // `2^62` is `(2^31)^2`, an exact perfect square.
        assert_eq!(isqrt_u64(base), 1u64 << 31);
    }

    #[test]
    fn isqrt_u64_near_2_pow_63() {
        let base = 1u64 << 63;
        for delta in 0u64..8 {
            let n = base + delta;
            let r = isqrt_u64(n);
            assert!(r * r <= n);
            assert!(n - r * r <= 2 * r);
        }
    }

    #[test]
    fn is_perfect_square_true_cases() {
        for r in 0u64..=500 {
            assert!(is_perfect_square_u64(r * r));
        }
        assert!(is_perfect_square_u64(1u64 << 62));
        assert!(is_perfect_square_u64(4_294_967_295 * 4_294_967_295));
    }

    #[test]
    fn is_perfect_square_false_cases() {
        assert!(!is_perfect_square_u64(2));
        assert!(!is_perfect_square_u64(3));
        assert!(!is_perfect_square_u64(5));
        assert!(!is_perfect_square_u64(8));
        assert!(!is_perfect_square_u64(u64::MAX));
        for r in 2u64..=500 {
            let sq = r * r;
            assert!(!is_perfect_square_u64(sq - 1));
            assert!(!is_perfect_square_u64(sq + 1));
        }
    }

    #[test]
    fn is_perfect_square_zero_and_one() {
        assert!(is_perfect_square_u64(0));
        assert!(is_perfect_square_u64(1));
    }

    #[test]
    fn icbrt_u64_small_values() {
        assert_eq!(icbrt_u64(0), 0);
        assert_eq!(icbrt_u64(1), 1);
        assert_eq!(icbrt_u64(7), 1);
        assert_eq!(icbrt_u64(8), 2);
        assert_eq!(icbrt_u64(26), 2);
        assert_eq!(icbrt_u64(27), 3);
        assert_eq!(icbrt_u64(63), 3);
        assert_eq!(icbrt_u64(64), 4);
    }

    #[test]
    fn icbrt_u64_perfect_cubes() {
        for r in 0u64..=2_000 {
            assert_eq!(icbrt_u64(r * r * r), r);
        }
    }

    #[test]
    fn icbrt_u64_cube_boundaries() {
        for r in 1u64..=2_000 {
            let cube = r * r * r;
            assert_eq!(icbrt_u64(cube - 1), r - 1);
            assert_eq!(icbrt_u64(cube), r);
            assert_eq!(icbrt_u64(cube + 1), r);
        }
    }

    #[test]
    fn icbrt_u64_max() {
        let r = icbrt_u64(u64::MAX);
        assert_eq!(r, 2_642_245);
        assert!(r * r * r <= u64::MAX);
        let next = r + 1;
        // Guard against overflow before comparing the next cube.
        assert!(
            next.checked_mul(next)
                .and_then(|v| v.checked_mul(next))
                .is_none()
                || next * next * next > u64::MAX
        );
    }

    #[test]
    fn icbrt_u64_large_cubes() {
        let roots = [1_000u64, 100_000, 1_000_000, 2_000_000, 2_642_245];
        for &r in &roots {
            let cube = r * r * r;
            assert_eq!(icbrt_u64(cube), r);
        }
    }

    #[test]
    fn isqrt_u32_postcondition_lcg() {
        // Deterministic linear congruential generator (Numerical Recipes).
        let mut state: u32 = 0x1234_5678;
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = state;
            let r = isqrt_u32(n);
            let rr = u64::from(r) * u64::from(r);
            assert!(rr <= u64::from(n));
            let next = u64::from(r) + 1;
            assert!(next * next > u64::from(n));
        }
    }

    #[test]
    fn isqrt_u64_postcondition_lcg() {
        // Deterministic 64-bit LCG (Knuth MMIX constants).
        let mut state: u64 = 0x0BAD_C0FF_EE12_3456;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let n = state;
            let r = isqrt_u64(n);
            let rr = r * r;
            assert!(rr <= n);
            // Equivalent to `n < (r+1)^2` without overflowing `(r+1)^2`.
            assert!(n - rr <= 2 * r);
        }
    }

    #[test]
    fn icbrt_u64_postcondition_lcg() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_0001;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            let n = state;
            let r = icbrt_u64(n);
            assert!(r * r * r <= n);
            let next = r + 1;
            // `next^3` can overflow near the top of the range; guard first.
            let overflows = next
                .checked_mul(next)
                .and_then(|v| v.checked_mul(next))
                .is_none();
            assert!(overflows || next * next * next > n);
        }
    }

    #[test]
    fn isqrt_matches_between_widths_on_lcg() {
        let mut state: u32 = 0x9E37_79B9;
        for _ in 0..10_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = state;
            assert_eq!(u64::from(isqrt_u32(n)), isqrt_u64(u64::from(n)));
        }
    }

    #[test]
    fn isqrt_u64_powers_of_two() {
        for exp in 0u32..64 {
            let n = 1u64 << exp;
            let r = isqrt_u64(n);
            assert!(r * r <= n);
            assert!(n - r * r <= 2 * r);
        }
    }

    #[test]
    fn is_perfect_square_powers_of_four() {
        // `4^k = (2^k)^2` are perfect squares; odd powers of two are not.
        for exp in 0u32..32 {
            let even = 1u64 << (2 * exp);
            assert!(is_perfect_square_u64(even));
        }
        for exp in 0u32..31 {
            let odd = 1u64 << (2 * exp + 1);
            assert!(!is_perfect_square_u64(odd));
        }
    }
}
