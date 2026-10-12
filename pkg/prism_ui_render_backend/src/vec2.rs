//! Shared 2D point/vector primitives for the geometry modules.
//!
//! Every geometry module in this crate works on bare `(f32, f32)` points — no
//! wrapper type — so the same handful of operations (difference, dot/cross
//! products, length, distance, interpolation) were being re-declared privately
//! in each one. This module is the single home for those primitives; the other
//! modules `use` them rather than rolling their own, so the reduction order and
//! rounding are identical everywhere.
//!
//! Everything is pure `+ - * /` and `sqrt` on `f32`, so the helpers are
//! `no_std`-clean and bit-stable.

/// `a - b`.
#[inline]
pub(crate) fn sub(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 - b.0, a.1 - b.1)
}

/// Dot product `a · b`.
#[inline]
pub(crate) fn dot(a: (f32, f32), b: (f32, f32)) -> f32 {
    a.0 * b.0 + a.1 * b.1
}

/// 2D cross product `a × b` (the `z` of the 3D cross).
#[inline]
pub(crate) fn cross(a: (f32, f32), b: (f32, f32)) -> f32 {
    a.0 * b.1 - a.1 * b.0
}

/// Turn cross product `(a - o) × (b - o)`.
///
/// Its sign classifies the turn `o -> a -> b`: positive for a left
/// (counter-clockwise) turn, negative for a right turn, zero when collinear.
#[inline]
pub(crate) fn cross3(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Euclidean length of vector `v`.
#[inline]
pub(crate) fn length(v: (f32, f32)) -> f32 {
    length_xy(v.0, v.1)
}

/// Euclidean length of the vector `(x, y)`.
#[inline]
pub(crate) fn length_xy(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

/// Euclidean distance between points `a` and `b`.
#[inline]
pub(crate) fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    length(sub(a, b))
}

/// Linear interpolation: `a` at `t = 0`, `b` at `t = 1` (extrapolates outside
/// `[0, 1]`).
#[inline]
pub(crate) fn lerp(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
    )
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;

    #[test]
    fn basic_operations() {
        assert_eq!(sub((3.0, 5.0), (1.0, 2.0)), (2.0, 3.0));
        assert_eq!(dot((1.0, 2.0), (3.0, 4.0)), 11.0);
        assert_eq!(cross((1.0, 0.0), (0.0, 1.0)), 1.0);
        // A left turn about the origin has positive `cross3`.
        assert!(cross3((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)) > 0.0);
        // Collinear points have zero turn.
        assert_eq!(cross3((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)), 0.0);
    }

    #[test]
    fn lengths_and_distance() {
        assert!((length((3.0, 4.0)) - 5.0).abs() < 1e-6);
        assert!((length_xy(3.0, 4.0) - 5.0).abs() < 1e-6);
        assert!((distance((0.0, 0.0), (3.0, 4.0)) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        assert_eq!(lerp((0.0, 0.0), (10.0, 20.0), 0.0), (0.0, 0.0));
        assert_eq!(lerp((0.0, 0.0), (10.0, 20.0), 1.0), (10.0, 20.0));
        assert_eq!(lerp((0.0, 0.0), (10.0, 20.0), 0.5), (5.0, 10.0));
    }
}
