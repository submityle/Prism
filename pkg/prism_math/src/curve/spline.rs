//! Cubic spline segments: Hermite, Catmull-Rom, and Bézier.
//!
//! All functions are generic over [`Interpolatable`] values and share the
//! `t in [0, 1]` segment convention. Each positional evaluator has a matching
//! `*_tangent` function returning the derivative with respect to `t`, used for
//! continuity and velocity queries.

use crate::curve::Interpolatable;

/// Evaluate a cubic Hermite segment.
///
/// `p0`/`p1` are the segment endpoints at `t = 0`/`t = 1`, and `m0`/`m1` are
/// the tangents (derivatives w.r.t. `t`) at those endpoints.
#[inline]
pub fn hermite<T: Interpolatable>(p0: T, m0: T, p1: T, m1: T, t: f32) -> T {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Derivative (tangent) of [`hermite`] with respect to `t`.
#[inline]
pub fn hermite_tangent<T: Interpolatable>(p0: T, m0: T, p1: T, m1: T, t: f32) -> T {
    let t2 = t * t;
    let h00 = 6.0 * t2 - 6.0 * t;
    let h10 = 3.0 * t2 - 4.0 * t + 1.0;
    let h01 = -6.0 * t2 + 6.0 * t;
    let h11 = 3.0 * t2 - 2.0 * t;
    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Evaluate the centripetal-free (uniform) Catmull-Rom segment running from
/// `p1` (at `t = 0`) to `p2` (at `t = 1`), using neighbors `p0` and `p3` to
/// derive the tangents.
#[inline]
pub fn catmull_rom<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let m1 = (p2 - p0) * 0.5;
    let m2 = (p3 - p1) * 0.5;
    hermite(p1, m1, p2, m2, t)
}

/// Derivative (tangent) of [`catmull_rom`] with respect to `t`.
#[inline]
pub fn catmull_rom_tangent<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let m1 = (p2 - p0) * 0.5;
    let m2 = (p3 - p1) * 0.5;
    hermite_tangent(p1, m1, p2, m2, t)
}

/// Evaluate a cubic Bézier segment with control points `p0..p3` using the
/// Bernstein form. The curve passes through `p0` at `t = 0` and `p3` at
/// `t = 1`.
#[inline]
pub fn bezier_cubic<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    let b0 = uu * u;
    let b1 = 3.0 * uu * t;
    let b2 = 3.0 * u * tt;
    let b3 = tt * t;
    p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3
}

/// Derivative (tangent) of [`bezier_cubic`] with respect to `t`.
///
/// At the endpoints this equals `3 * (p1 - p0)` at `t = 0` and
/// `3 * (p3 - p2)` at `t = 1`.
#[inline]
pub fn bezier_cubic_tangent<T: Interpolatable>(p0: T, p1: T, p2: T, p3: T, t: f32) -> T {
    let u = 1.0 - t;
    let c0 = 3.0 * u * u;
    let c1 = 6.0 * u * t;
    let c2 = 3.0 * t * t;
    (p1 - p0) * c0 + (p2 - p1) * c1 + (p3 - p2) * c2
}
