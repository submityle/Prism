//! Internal, deterministic math helpers.
//!
//! These wrap the handful of transcendental functions used by the animation
//! primitives. They are implemented with range-reduced polynomial
//! approximations so results are identical with and without the `std` feature
//! and across platforms (the surrounding engine bans the platform `f32`
//! transcendental intrinsics for determinism reasons).
//!
//! All helpers are crate-private.

use core::f32::consts::{FRAC_PI_2, LN_2, PI, TAU};

/// Absolute value of an `f32` without relying on `std`.
#[inline]
pub(crate) fn absf(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Minimum of two `f32` values (favouring the first on ties).
#[inline]
pub(crate) fn minf(a: f32, b: f32) -> f32 {
    if b < a {
        b
    } else {
        a
    }
}

/// Maximum of two `f32` values (favouring the first on ties).
#[inline]
pub(crate) fn maxf(a: f32, b: f32) -> f32 {
    if b > a {
        b
    } else {
        a
    }
}

/// Clamp `x` into the inclusive range `[lo, hi]`.
#[inline]
pub(crate) fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
    maxf(lo, minf(hi, x))
}

/// Floor of `x` for the finite range exercised by this crate.
#[inline]
pub(crate) fn floorf(x: f32) -> f32 {
    // Truncation towards zero via an integer cast, corrected for negatives.
    let truncated = (x as i64) as f32;
    if truncated > x {
        truncated - 1.0
    } else {
        truncated
    }
}

/// Square root of `x` (returns `0.0` for non-positive inputs).
///
/// Uses Newton–Raphson, which converges to `f32` precision in a few
/// iterations over the range used by this crate.
#[inline]
pub(crate) fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = if x > 1.0 { x } else { 1.0 };
    let mut i = 0;
    while i < 30 {
        guess = 0.5 * (guess + x / guess);
        i += 1;
    }
    guess
}

/// Sine of `x` (radians).
#[inline]
pub(crate) fn sinf(x: f32) -> f32 {
    // Reduce to [-pi, pi].
    let k = floorf(x / TAU + 0.5);
    let mut r = x - k * TAU;
    // Fold into [-pi/2, pi/2] using sin(pi - r) == sin(r).
    if r > FRAC_PI_2 {
        r = PI - r;
    } else if r < -FRAC_PI_2 {
        r = -PI - r;
    }
    // Taylor series about 0; accurate to ~1e-6 on [-pi/2, pi/2].
    let r2 = r * r;
    r * (1.0
        + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362_880.0)))))
}

/// Cosine of `x` (radians).
#[inline]
pub(crate) fn cosf(x: f32) -> f32 {
    sinf(x + FRAC_PI_2)
}

/// `2.0_f32` raised to an integer power, computed exactly where representable.
#[inline]
fn exp2i(mut k: i32) -> f32 {
    if k > 127 {
        return f32::INFINITY;
    }
    if k < -149 {
        return 0.0;
    }
    let mut result = 1.0_f32;
    if k >= 0 {
        while k > 0 {
            result *= 2.0;
            k -= 1;
        }
    } else {
        while k < 0 {
            result *= 0.5;
            k += 1;
        }
    }
    result
}

/// Natural exponential of `x`.
#[inline]
pub(crate) fn expf(x: f32) -> f32 {
    // x = k*ln2 + r, with r in [-ln2/2, ln2/2]; exp(x) = 2^k * exp(r).
    let k = floorf(x / LN_2 + 0.5);
    let r = x - k * LN_2;
    // Taylor series for exp(r); accurate to ~1e-7 on the reduced range.
    let er = 1.0
        + r * (1.0
            + r * (1.0 / 2.0
                + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    er * exp2i(k as i32)
}

/// `2.0_f32` raised to an arbitrary real power.
#[inline]
pub(crate) fn exp2f(x: f32) -> f32 {
    expf(x * LN_2)
}
