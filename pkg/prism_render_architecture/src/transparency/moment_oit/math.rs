//! Deterministic transcendental and small-matrix primitives for moment `OIT`.
//!
//! The workspace determinism policy forbids the `f32` transcendental intrinsics
//! (`f32::ln`, `f32::exp`, …) because their results are not reproducible across
//! platforms. Moment-based `OIT` needs a logarithm (to turn coverage into
//! optical absorbance) and an exponential (to turn reconstructed optical depth
//! back into transmittance), plus a tiny symmetric solver for the power-moment
//! reconstruction. This module provides reproducible, add/multiply/compare-only
//! implementations of each, accurate to well within the precision the resolve
//! needs.

/// Natural logarithm for strictly-positive `x`, reproducibly.
///
/// Uses the exact `IEEE-754` decomposition `x = m * 2^e` with `m` in `[1, 2)`
/// (read from the float's bit pattern, which is deterministic) and the rapidly
/// converging `atanh` series `ln(m) = 2*(s + s^3/3 + s^5/5 + …)` where
/// `s = (m - 1) / (m + 1)` lies in `[0, 1/3]`. Non-positive inputs return a
/// large negative sentinel so callers that forgot to clamp fail toward "fully
/// transparent" rather than producing a `NaN`.
#[must_use]
pub fn ln(x: f32) -> f32 {
    if x <= 0.0 {
        return -87.0; // exp(-87) underflows to ~0; a safe "minus infinity".
    }
    let bits = x.to_bits();
    let exponent = (((bits >> 23) & 0xff) as i32) - 127;
    let mantissa_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
    let m = f32::from_bits(mantissa_bits); // in [1, 2)
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    // Horner evaluation of (1 + s^2/3 + s^4/5 + s^6/7 + s^8/9).
    let mut poly = 1.0 / 9.0;
    poly = poly * s2 + 1.0 / 7.0;
    poly = poly * s2 + 1.0 / 5.0;
    poly = poly * s2 + 1.0 / 3.0;
    poly = poly * s2 + 1.0;
    let ln_m = 2.0 * s * poly;
    (exponent as f32) * core::f32::consts::LN_2 + ln_m
}

/// Exponential of `x`, reproducibly.
///
/// Range-reduces `x = k*ln2 + r` with `|r| <= ln2/2`, evaluates `exp(r)` with a
/// sixth-order Taylor polynomial (accurate because `|r|` is small), and scales
/// by `2^k` built directly from the exponent bits. Saturates to `0` / `+inf`
/// outside the representable exponent range instead of producing a `NaN`.
#[must_use]
pub fn exp(x: f32) -> f32 {
    let kf = (x * core::f32::consts::LOG2_E).round();
    let r = x - kf * core::f32::consts::LN_2;
    // exp(r), Horner form, terms through r^6/720.
    let er = 1.0
        + r * (1.0
            + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    let k = kf as i32;
    if k > 127 {
        return f32::INFINITY;
    }
    if k < -126 {
        return 0.0;
    }
    let two_k = f32::from_bits((((k + 127) as u32) & 0xff) << 23);
    er * two_k
}

/// Solves the symmetric positive-definite 3x3 system `A x = b` by Cholesky.
///
/// `a` is the lower triangle packed as `[a00, a10, a11, a20, a21, a22]`. Returns
/// `None` when `A` is not positive definite (a non-positive pivot), which the
/// moment reconstruction treats as a degenerate measure.
#[must_use]
pub fn solve_spd3(a: [f32; 6], b: [f32; 3]) -> Option<[f32; 3]> {
    let (a00, a10, a11, a20, a21, a22) = (a[0], a[1], a[2], a[3], a[4], a[5]);
    // Cholesky: A = L L^T.
    if a00 <= 0.0 {
        return None;
    }
    let l00 = a00.sqrt();
    let l10 = a10 / l00;
    let l20 = a20 / l00;
    let d11 = a11 - l10 * l10;
    if d11 <= 0.0 {
        return None;
    }
    let l11 = d11.sqrt();
    let l21 = (a21 - l20 * l10) / l11;
    let d22 = a22 - l20 * l20 - l21 * l21;
    if d22 <= 0.0 {
        return None;
    }
    let l22 = d22.sqrt();
    // Forward solve L y = b.
    let y0 = b[0] / l00;
    let y1 = (b[1] - l10 * y0) / l11;
    let y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    // Back solve L^T x = y.
    let x2 = y2 / l22;
    let x1 = (y1 - l21 * x2) / l11;
    let x0 = (y0 - l10 * x1 - l20 * x2) / l00;
    Some([x0, x1, x2])
}

/// Real roots of `c2 x^2 + c1 x + c0 = 0`, ascending, when both are real.
///
/// Returns `None` for a (near-)degenerate leading coefficient or a negative
/// discriminant. Uses the numerically stable form that avoids catastrophic
/// cancellation.
#[must_use]
pub fn quadratic_roots(c2: f32, c1: f32, c0: f32) -> Option<(f32, f32)> {
    if c2.abs() <= f32::EPSILON {
        return None;
    }
    let disc = c1 * c1 - 4.0 * c2 * c0;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    // Stable quadratic: q = -(c1 + sign(c1)*sqrt_disc)/2.
    let q = -0.5 * (c1 + if c1 >= 0.0 { sqrt_disc } else { -sqrt_disc });
    let r0 = q / c2;
    let r1 = if q.abs() <= f32::EPSILON {
        -c1 / c2 - r0
    } else {
        c0 / q
    };
    if r0 <= r1 {
        Some((r0, r1))
    } else {
        Some((r1, r0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_matches_reference_values() {
        // Reference values; tolerance comfortably covers the series truncation.
        let cases = [
            (1.0_f32, 0.0_f32),
            (2.0, core::f32::consts::LN_2),
            (0.5, -core::f32::consts::LN_2),
            (10.0, core::f32::consts::LN_10),
            (0.1, -core::f32::consts::LN_10),
            (0.001, -3.0 * core::f32::consts::LN_10),
        ];
        for (x, want) in cases {
            let got = ln(x);
            assert!((got - want).abs() < 1e-4, "ln({x}) = {got}, want {want}");
        }
        assert!(ln(0.0) < -50.0);
        assert!(ln(-1.0) < -50.0);
    }

    #[test]
    fn exp_matches_reference_values() {
        let cases = [
            (0.0_f32, 1.0_f32),
            (1.0, core::f32::consts::E),
            (-1.0, 0.367_879_44),
            (2.0, 7.389_056),
            (-5.0, 0.006_737_947),
            (5.0, 148.413_16),
        ];
        for (x, want) in cases {
            let got = exp(x);
            assert!(
                (got - want).abs() <= want.abs() * 1e-4 + 1e-6,
                "exp({x}) = {got}, want {want}"
            );
        }
        assert_eq!(exp(1000.0), f32::INFINITY);
        assert_eq!(exp(-1000.0), 0.0);
    }

    #[test]
    fn exp_inverts_ln() {
        for &x in &[0.05_f32, 0.3, 0.7, 0.95, 1.0, 3.3, 12.0] {
            let round = exp(ln(x));
            assert!((round - x).abs() <= x * 1e-3 + 1e-5, "exp(ln({x})) = {round}");
        }
    }

    #[test]
    fn solve_spd3_recovers_known_solution() {
        // A = L L^T with L lower-triangular, positive diagonal -> SPD.
        // Pick A = [[4,2,2],[2,5,3],[2,3,6]], x = [1,2,3], compute b = A x.
        let a = [4.0, 2.0, 5.0, 2.0, 3.0, 6.0];
        let x = [1.0, 2.0, 3.0];
        let b = [
            4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2],
            2.0 * x[0] + 5.0 * x[1] + 3.0 * x[2],
            2.0 * x[0] + 3.0 * x[1] + 6.0 * x[2],
        ];
        let got = solve_spd3(a, b).expect("SPD");
        for i in 0..3 {
            assert!((got[i] - x[i]).abs() < 1e-4, "x[{i}] = {}", got[i]);
        }
    }

    #[test]
    fn solve_spd3_rejects_indefinite() {
        // Negative pivot -> not positive definite.
        assert!(solve_spd3([-1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]).is_none());
    }

    #[test]
    fn quadratic_roots_are_correct_and_ordered() {
        // (x-2)(x-5) = x^2 -7x +10
        let (r0, r1) = quadratic_roots(1.0, -7.0, 10.0).expect("real");
        assert!((r0 - 2.0).abs() < 1e-4 && (r1 - 5.0).abs() < 1e-4, "{r0},{r1}");
        // No real roots.
        assert!(quadratic_roots(1.0, 0.0, 1.0).is_none());
        // Degenerate leading term.
        assert!(quadratic_roots(0.0, 1.0, 1.0).is_none());
    }
}
