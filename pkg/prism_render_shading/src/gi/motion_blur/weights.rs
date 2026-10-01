//! Core weighting primitives for the McGuire motion-blur reconstruction filter.
//!
//! This module is the backend-neutral CPU golden reference for the scalar
//! weight functions used by McGuire et al. (2012), *"A Reconstruction Filter for
//! Plausible Motion Blur"*.  The reconstruction filter decides, for a center
//! pixel `X` and a neighbouring sample `Y` taken along the dominant tile
//! velocity, how much `Y` should contribute to `X`.  That decision is a product
//! of three ingredients:
//!
//! * a **cone** falloff that models a point smearing into a line segment of
//!   length equal to its velocity magnitude (`cone`),
//! * a **cylinder** falloff that models a region that is uniformly blurred along
//!   its whole extent (`cylinder`), and
//! * a **soft depth comparison** that classifies whether `X` or `Y` is the
//!   nearer (foreground) surface without a hard z-test, so silhouettes blend
//!   smoothly instead of popping (`soft_depth_compare` / `soft_z_compare`).
//!
//! A handful of velocity helpers (`velocity_magnitude`, `half_velocity`,
//! `is_moving`) round out the primitives the sampling and reconstruction stages
//! consume.
//!
//! # Conventions
//! * All distances and velocities are in **pixels** (the units the tile
//!   NeighborMax stage from [`crate::gi::motion`] produces for motion blur).
//! * Every function is a deterministic pure function: no RNG, IO, GPU, or
//!   `unsafe`, and no allocation.
//! * Transcendental functions, when needed, go through [`bevy_math::ops`]; the
//!   polynomial `smoothstep` here needs none.
//! * Defensive clamping is pervasive: radii/extents are forced non-negative,
//!   divide-by-zero is guarded with explicit degenerate fallbacks, and every
//!   return value is finite and (where it represents a weight) in `[0, 1]`.
//!   No `NaN`/`inf` can escape even for `NaN`/`inf` inputs.

use bevy_math::Vec2;

/// Lower bound used to detect a "degenerate" (effectively zero) radius or
/// extent before dividing by it.
///
/// Chosen well below a pixel so sub-pixel velocities still behave sensibly while
/// a genuinely zero radius is treated as a hard point.
const DEGENERATE_EPSILON: f32 = 1.0e-6;

/// Replaces a non-finite scalar with `0.0`, otherwise returns it unchanged.
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Sanitizes a radius/extent: non-finite becomes `0.0`, negatives are clamped
/// up to `0.0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    finite_or_zero(x).max(0.0)
}

/// Smooth Hermite interpolation matching GPU `smoothstep` semantics.
///
/// Returns `0.0` for `x <= edge0`, `1.0` for `x >= edge1`, and the cubic
/// `t * t * (3 - 2t)` in between, where `t = (x - edge0) / (edge1 - edge0)`.
///
/// Degenerate (`edge1 <= edge0`) or non-finite edges collapse to a hard step at
/// `edge0`, so the result is always finite and in `[0, 1]`.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let edge0 = finite_or_zero(edge0);
    let edge1 = finite_or_zero(edge1);
    let x = finite_or_zero(x);
    let span = edge1 - edge0;
    if span <= DEGENERATE_EPSILON {
        // Degenerate interval: behave as a hard step at `edge0`.
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// McGuire **cone** weight: `clamp(1 - dist / radius, 0, 1)`.
///
/// Models a moving point as a line segment of length `radius` (its velocity
/// magnitude): a sample at distance `dist` from the center receives full weight
/// at `dist == 0`, falls off linearly, and reaches zero at `dist == radius`.
///
/// A `radius` of (near) zero means the point does not move, so it only covers
/// `dist == 0`; the result is clamped to `[0, 1]` and is always finite.
#[inline]
pub fn cone(dist: f32, radius: f32) -> f32 {
    let dist = sanitize_nonneg(dist);
    let radius = sanitize_nonneg(radius);
    if radius <= DEGENERATE_EPSILON {
        return if dist <= DEGENERATE_EPSILON { 1.0 } else { 0.0 };
    }
    (1.0 - dist / radius).clamp(0.0, 1.0)
}

/// McGuire **cylinder** weight with a smooth shoulder:
/// `1 - smoothstep(0.95 * r_lo, 1.05 * r_hi, dist)`.
///
/// Models a region that is uniformly blurred across its whole extent: weight is
/// (near) `1.0` while `dist` is inside the blur radius and smoothly falls to
/// `0.0` just past it.  Two radii are accepted so the caller can pass the two
/// velocity magnitudes being compared; `r_lo`/`r_hi` are ordered internally so
/// the shoulder is well-formed regardless of argument order.
///
/// The result is finite and in `[0, 1]`; a degenerate (zero) radius pair yields
/// a hard cutoff at `dist == 0`.
#[inline]
pub fn cylinder(dist: f32, r1: f32, r2: f32) -> f32 {
    let dist = sanitize_nonneg(dist);
    let r1 = sanitize_nonneg(r1);
    let r2 = sanitize_nonneg(r2);
    let r_lo = r1.min(r2);
    let r_hi = r1.max(r2);
    (1.0 - smoothstep(0.95 * r_lo, 1.05 * r_hi, dist)).clamp(0.0, 1.0)
}

/// McGuire **soft depth comparison**: `clamp(1 - (za - zb) / extent, 0, 1)`.
///
/// Interpreted with *smaller depth = nearer* (the common convention): the result
/// is `1.0` when `za` is clearly in front of `zb` (`za` much smaller), `0.0`
/// when `za` is clearly behind, and a smooth ramp of width `extent` across the
/// transition.  `extent` controls how many depth units count as "the same
/// surface".
///
/// `extent` is forced to a small positive value to avoid divide-by-zero, and
/// non-finite depths are sanitized to `0.0`, so the output is finite and in
/// `[0, 1]`.
#[inline]
pub fn soft_depth_compare(za: f32, zb: f32, extent: f32) -> f32 {
    let za = finite_or_zero(za);
    let zb = finite_or_zero(zb);
    let extent = sanitize_nonneg(extent).max(DEGENERATE_EPSILON);
    (1.0 - (za - zb) / extent).clamp(0.0, 1.0)
}

/// Alias spelling of [`soft_depth_compare`] used throughout the McGuire
/// literature (`softZCompare`).
///
/// Returns how strongly `za` is classified as being in *front* of `zb` given a
/// transition width of `extent`.
#[inline]
pub fn soft_z_compare(za: f32, zb: f32, extent: f32) -> f32 {
    soft_depth_compare(za, zb, extent)
}

/// Euclidean magnitude (length, in pixels) of a screen-space velocity.
///
/// Non-finite components are sanitized to `0.0`, so the result is always a
/// finite, non-negative length.
#[inline]
pub fn velocity_magnitude(velocity: Vec2) -> f32 {
    let v = Vec2::new(finite_or_zero(velocity.x), finite_or_zero(velocity.y));
    let len = v.length();
    if len.is_finite() { len.max(0.0) } else { 0.0 }
}

/// Half of a velocity vector — the signed extent the blur spans on each side of
/// the pixel center.
///
/// Non-finite components are sanitized to `0.0`.
#[inline]
pub fn half_velocity(velocity: Vec2) -> Vec2 {
    Vec2::new(finite_or_zero(velocity.x), finite_or_zero(velocity.y)) * 0.5
}

/// Returns `true` when a velocity is large enough to produce visible blur, i.e.
/// its magnitude is at least `threshold_px` pixels.
///
/// The classic McGuire threshold is half a pixel: motion below it is rounded
/// away as "not moving".  `threshold_px` is clamped non-negative.
#[inline]
pub fn is_moving(velocity: Vec2, threshold_px: f32) -> bool {
    velocity_magnitude(velocity) >= sanitize_nonneg(threshold_px)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn cone_boundaries() {
        // Full weight at the center, zero at/after the radius, clamped outside.
        assert!((cone(0.0, 10.0) - 1.0).abs() < EPS);
        assert!(cone(10.0, 10.0).abs() < EPS);
        assert!(cone(20.0, 10.0).abs() < EPS);
        // Linear midpoint.
        assert!((cone(5.0, 10.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn cone_is_monotonic_nonincreasing() {
        let radius = 8.0;
        let mut prev = cone(0.0, radius);
        let mut d = 0.0;
        while d <= 2.0 * radius {
            let w = cone(d, radius);
            assert!(w <= prev + EPS, "cone increased at d={d}");
            assert!((0.0..=1.0).contains(&w));
            prev = w;
            d += 0.25;
        }
    }

    #[test]
    fn cone_degenerate_radius_is_a_point() {
        assert!((cone(0.0, 0.0) - 1.0).abs() < EPS);
        assert!(cone(0.001, 0.0).abs() < EPS);
    }

    #[test]
    fn cylinder_plateau_then_falloff() {
        // Inside the shorter radius it is (near) full weight.
        assert!(cylinder(0.0, 10.0, 20.0) > 0.99);
        assert!(cylinder(5.0, 10.0, 20.0) > 0.9);
        // Well past the longer radius it is zero.
        assert!(cylinder(40.0, 10.0, 20.0).abs() < EPS);
    }

    #[test]
    fn cylinder_is_order_independent_and_monotonic() {
        for d in [0.0_f32, 3.0, 7.0, 12.0, 18.0, 25.0] {
            let a = cylinder(d, 6.0, 15.0);
            let b = cylinder(d, 15.0, 6.0);
            assert!((a - b).abs() < EPS, "asymmetric at d={d}");
            assert!((0.0..=1.0).contains(&a));
        }
        let mut prev = cylinder(0.0, 6.0, 15.0);
        let mut d = 0.0;
        while d <= 30.0 {
            let w = cylinder(d, 6.0, 15.0);
            assert!(w <= prev + 1.0e-4, "cylinder increased at d={d}");
            prev = w;
            d += 0.5;
        }
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!(smoothstep(0.0, 1.0, -0.5).abs() < EPS);
        assert!((smoothstep(0.0, 1.0, 1.5) - 1.0).abs() < EPS);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < EPS);
        // Degenerate interval collapses to a hard step.
        assert!(smoothstep(2.0, 2.0, 1.0).abs() < EPS);
        assert!((smoothstep(2.0, 2.0, 2.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn soft_depth_compare_front_back_and_ramp() {
        // za far in front of zb -> 1; far behind -> 0.
        assert!((soft_depth_compare(0.0, 10.0, 1.0) - 1.0).abs() < EPS);
        assert!(soft_depth_compare(10.0, 0.0, 1.0).abs() < EPS);
        // Equal depth -> 1 (on the "same or in front" side of the ramp).
        assert!((soft_depth_compare(5.0, 5.0, 2.0) - 1.0).abs() < EPS);
        // Mid ramp.
        assert!((soft_depth_compare(1.0, 0.0, 2.0) - 0.5).abs() < EPS);
        // Alias agrees.
        assert!((soft_z_compare(1.0, 0.0, 2.0) - soft_depth_compare(1.0, 0.0, 2.0)).abs() < EPS);
    }

    #[test]
    fn soft_depth_compare_is_monotonic_in_za() {
        let extent = 3.0;
        let mut prev = soft_depth_compare(-10.0, 0.0, extent);
        let mut za = -10.0;
        while za <= 10.0 {
            let w = soft_depth_compare(za, 0.0, extent);
            assert!(w <= prev + EPS, "non-monotonic at za={za}");
            assert!((0.0..=1.0).contains(&w));
            prev = w;
            za += 0.5;
        }
    }

    #[test]
    fn velocity_helpers() {
        let v = Vec2::new(3.0, 4.0);
        assert!((velocity_magnitude(v) - 5.0).abs() < EPS);
        assert_eq!(half_velocity(v), Vec2::new(1.5, 2.0));
        assert!(is_moving(v, 0.5));
        assert!(!is_moving(Vec2::new(0.1, 0.0), 0.5));
    }

    #[test]
    fn non_finite_inputs_are_sanitized() {
        assert!(cone(f32::NAN, 10.0).is_finite());
        assert!(cone(5.0, f32::INFINITY).is_finite());
        assert!(cylinder(f32::NAN, f32::NAN, f32::NAN).is_finite());
        assert!(soft_depth_compare(f32::NAN, f32::INFINITY, f32::NAN).is_finite());
        assert_eq!(velocity_magnitude(Vec2::new(f32::NAN, 4.0)), 4.0);
        assert_eq!(half_velocity(Vec2::new(f32::INFINITY, 2.0)), Vec2::new(0.0, 1.0));
    }
}
