//! Deterministic scalar floating-point operations for no_std callers.
//!
//! The facade mirrors the operation names commonly used by engine math
//! crates, while routing every function through `prism_math`'s libm-backed
//! reference implementation in both `std` and `no_std` builds.

pub use crate::float::f32::{
    abs, acos, asin, atan, atan2, atanh, cbrt, ceil, cos, cosh, exp, exp2, exp_m1, floor, hypot,
    ln, ln_1p, log10, log2, powf, round, sin, sin_cos, sinh, sqrt, tan, tanh,
};

/// Return the fractional part of `x`.
#[inline]
pub fn fract(x: f32) -> f32 {
    x - floor(x)
}

/// Return the least non-negative remainder of `x` divided by `y`.
#[inline]
pub fn rem_euclid(x: f32, y: f32) -> f32 {
    let r = x % y;
    if r < 0.0 {
        r + y.abs()
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facade_covers_audio_math_operations() {
        let (sin_value, cos_value) = sin_cos(0.25);
        assert!((sin_value - sin(0.25)).abs() < 1.0e-6);
        assert!((cos_value - cos(0.25)).abs() < 1.0e-6);
        assert_eq!(floor(1.75), 1.0);
        assert_eq!(ceil(1.25), 2.0);
        assert!((hypot(3.0, 4.0) - 5.0).abs() < 1.0e-6);
        assert!((log10(100.0) - 2.0).abs() < 1.0e-6);
        assert!((log2(8.0) - 3.0).abs() < 1.0e-6);
        assert!((exp2(3.0) - 8.0).abs() < 1.0e-6);
        assert!((ln_1p(0.5) - ln(1.5)).abs() < 1.0e-6);
        assert!((tanh(0.5) - 0.46211717).abs() < 1.0e-6);
    }
}
