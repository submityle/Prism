//! A deterministic exponential built only from allowed scalar operations.
//!
//! Snow hardening multiplies the Lamé parameters by `exp(ξ(1 − Jp))`. The
//! engine forbids `f32::exp` (for cross-platform determinism), so this module
//! implements the exponential with range reduction to base two followed by a
//! short Taylor series, using only multiplication, `round`, and `sqrt`-free
//! integer powering.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Range
//! reduction `exp(x) = 2ⁿ · exp(r)` with a truncated Taylor series is a
//! standard, publicly documented technique for evaluating the exponential.

use crate::math::scalar::Real;

/// `log2(e)`, used to reduce `exp(x)` to a power of two.
const LOG2_E: Real = core::f32::consts::LOG2_E;
/// `ln(2)`, used to rescale the reduced argument back for the Taylor series.
const LN_2: Real = core::f32::consts::LN_2;

/// Returns `2ⁿ` for a (possibly negative) integer exponent by repeated
/// multiplication — deterministic and free of the disallowed `powi`.
#[must_use]
fn pow2_int(n: i32) -> Real {
    let mut result: Real = 1.0;
    let mut k = n.abs();
    while k > 0 {
        result *= 2.0;
        k -= 1;
    }
    if n < 0 {
        1.0 / result
    } else {
        result
    }
}

/// Evaluates `exp(x)` deterministically using only allowed scalar operations.
///
/// The relative error is below `1e-6` across the argument range used by snow
/// hardening. Very large magnitudes saturate to `0`/`+∞` gracefully via the
/// integer power.
#[must_use]
pub fn exp_stable(x: Real) -> Real {
    // exp(x) = 2^(x * log2 e); split the base-two exponent into integer +
    // fractional parts, keeping the fraction in [-0.5, 0.5].
    let y = x * LOG2_E;
    let n = y.round();
    let r = (y - n) * LN_2; // remaining argument in ~[-0.3466, 0.3466]
                            // Taylor series of exp(r) (7 terms; |r| <= 0.347 gives < 1e-7 error).
    let series = 1.0
        + r * (1.0
            + r * (0.5
                + r * ((1.0 / 6.0)
                    + r * ((1.0 / 24.0) + r * ((1.0 / 120.0) + r * (1.0 / 720.0))))));
    pow2_int(n as i32) * series
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_at_zero_and_one() {
        assert!((exp_stable(0.0) - 1.0).abs() < 1.0e-6);
        // Compare against the standard library constant e.
        assert!((exp_stable(1.0) - core::f32::consts::E).abs() < 1.0e-4);
    }

    #[test]
    fn matches_reference_over_range() {
        let samples = [-3.0, -1.5, -0.25, 0.5, 2.0, 4.0];
        for &x in &samples {
            let reference = reference_exp(x as f64) as Real;
            let got = exp_stable(x);
            let rel = (got - reference).abs() / reference.abs();
            assert!(rel < 1.0e-5, "x={x} got={got} ref={reference}");
        }
    }

    /// A high-order Taylor reference (test-only, not used by the solver).
    fn reference_exp(x: f64) -> f64 {
        let mut term = 1.0f64;
        let mut sum = 1.0f64;
        for n in 1..40 {
            term *= x / n as f64;
            sum += term;
        }
        sum
    }
}
