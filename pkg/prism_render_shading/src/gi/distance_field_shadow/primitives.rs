//! Analytic signed-distance primitives and CSG operators — CPU golden.
//!
//! The distance-field soft-shadow marcher
//! ([`super::march::soft_shadow`]) needs a scene it can sample as a single
//! `Fn(Vec3) -> f32` closure returning the signed distance to the nearest
//! surface.  This module supplies the building blocks for such a scene: the
//! handful of closed-form shapes from Inigo Quilez's public distance-function
//! collection whose exact signed distance is cheap to evaluate, plus the
//! constructive-solid-geometry operators that compose them.
//!
//! Every primitive is expressed in its own frame; placement is left to the
//! caller (subtract a centre, rotate the query point).  A *signed* distance is
//! negative inside the solid, zero on the surface, and positive outside, and
//! for these exact forms it equals the true Euclidean distance to the surface,
//! which is exactly the Lipschitz-1 property the sphere/soft-shadow marcher
//! relies on to take safe steps.
//!
//! The combination operators are the standard min/max CSG set:
//!
//! ```text
//! union(a, b)     = min(a, b)      // A ∪ B
//! subtract(a, b)  = max(-a, b)     // B \ A
//! intersect(a, b) = max(a, b)      // A ∩ B
//! ```
//!
//! together with the polynomial smooth-minimum blend
//! ([`op_smooth_union`]) that rounds the union seam over a finite radius `k`.
//!
//! # Conventions
//! * **Pure functions.** Everything here is a deterministic pure function —
//!   no RNG, I/O, GPU, allocation, or `unsafe`.  Primitives take the query
//!   `point` first so they read like `shape(point, params…)`.
//! * **Defensive clamps.** Radii and extents are clamped non-negative so a
//!   degenerate shape collapses to its skeleton rather than inverting its
//!   sign; the smooth-union radius is clamped to a tiny positive value before
//!   it is used as a divisor so no path divides by zero or yields `NaN`.
//! * **Transcendental math.** This file needs none; where a square root is
//!   required the inherent `f32::sqrt` method is used (Quilez's box form), and
//!   all `Vec3` component algebra routes through `bevy_math`.  No `f32::tan`
//!   /`f32::powf`-style free functions appear anywhere.
//! * **No qualified `Vec`/`alloc`.** This module stores nothing in a heap
//!   collection, so it imports none.
//!
//! # References
//! * Inigo Quilez, "distance functions" (2008–),
//!   <https://iquilezles.org/articles/distfunctions/>.
//! * Inigo Quilez, "smooth minimum" (2013),
//!   <https://iquilezles.org/articles/smin/>.

use bevy_math::{Vec2, Vec3};

/// Exact signed distance from `point` to a sphere of `radius` centred at the
/// local origin.
///
/// `|point| - radius`, negative inside.  `radius` is clamped non-negative so a
/// negative request collapses to a point rather than inverting the sign.
#[inline]
#[must_use]
pub fn sphere(point: Vec3, radius: f32) -> f32 {
    point.length() - radius.max(0.0)
}

/// Exact signed distance from `point` to an axis-aligned box of
/// `half_extents` centred at the local origin.
///
/// Quilez's closed form: with `q = |point| - half_extents` the distance is
/// `|max(q, 0)| + min(max(q.x, q.y, q.z), 0)`, correct both outside
/// (positive) and inside (negative).  Negative half extents are clamped to
/// zero.
#[inline]
#[must_use]
pub fn box_exact(point: Vec3, half_extents: Vec3) -> f32 {
    let q = point.abs() - half_extents.max(Vec3::ZERO);
    let outside = q.max(Vec3::ZERO).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

/// Signed distance to a *rounded* box: the box field of `half_extents` offset
/// inward by `radius`, giving rounded edges and corners of that radius.
///
/// `box_exact(point, half_extents) - radius`.  Both the extents and the round
/// radius are clamped non-negative.
#[inline]
#[must_use]
pub fn rounded_box(point: Vec3, half_extents: Vec3, radius: f32) -> f32 {
    box_exact(point, half_extents) - radius.max(0.0)
}

/// Signed distance to a capsule (a line segment of radius `radius`) with
/// endpoints `a` and `b`.
///
/// Quilez's segment form: project `point - a` onto `b - a`, clamp the
/// parameter to `[0, 1]`, and take the distance to the clamped point minus
/// `radius`.  A degenerate (zero-length) segment reduces to a sphere about
/// `a`; `radius` is clamped non-negative.
#[inline]
#[must_use]
pub fn capsule(point: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let pa = point - a;
    let ba = b - a;
    let denom = ba.dot(ba);
    // Guard the projection divisor: a zero-length segment is a sphere at `a`.
    let h = if denom > f32::EPSILON {
        (pa.dot(ba) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (pa - ba * h).length() - radius.max(0.0)
}

/// Signed distance to an infinite plane with unit `normal` and signed offset
/// `height` from the origin along that normal.
///
/// `dot(point, n) + height` where `n` is `normal` renormalised.  A degenerate
/// (near-zero) normal falls back to `+Y` so the result stays finite.
#[inline]
#[must_use]
pub fn plane(point: Vec3, normal: Vec3, height: f32) -> f32 {
    let n = normal.normalize_or_zero();
    let n = if n.length_squared() > 0.5 { n } else { Vec3::Y };
    point.dot(n) + height
}

/// Exact signed distance to a torus lying in the XZ plane, with major radius
/// `major` (ring centre circle) and minor radius `minor` (tube), centred at
/// the local origin.
///
/// Quilez's form: `q = (len(point.xz) - major, point.y)`, distance
/// `len(q) - minor`.  Both radii are clamped non-negative.
#[inline]
#[must_use]
pub fn torus(point: Vec3, major: f32, minor: f32) -> f32 {
    let radial = Vec2::new(point.x, point.z).length() - major.max(0.0);
    Vec2::new(radial, point.y).length() - minor.max(0.0)
}

/// Constructive-solid-geometry union: the solid occupied by *either* field.
///
/// `min(a, b)`.  Exact for distance fields outside both shapes and a safe
/// (conservative) under-estimate inside the overlap, which is all the marcher
/// needs.
#[inline]
#[must_use]
pub fn op_union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Constructive-solid-geometry subtraction: `b` with the solid `a` carved out.
///
/// `max(-a, b)`.
#[inline]
#[must_use]
pub fn op_subtract(a: f32, b: f32) -> f32 {
    (-a).max(b)
}

/// Constructive-solid-geometry intersection: the solid occupied by *both*
/// fields.
///
/// `max(a, b)`.
#[inline]
#[must_use]
pub fn op_intersect(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Polynomial smooth-minimum union of two distance fields over blend radius
/// `k`.
///
/// Quilez's quadratic `smin`: with `h = clamp(0.5 + 0.5·(b − a)/k, 0, 1)` the
/// blended distance is `mix(b, a, h) − k·h·(1 − h)`, which rounds the union
/// seam over a band of width `k`.  As `k → 0` this collapses to the hard
/// [`op_union`].  `k` is clamped to a tiny positive value before it is used as
/// a divisor so no path divides by zero.
#[inline]
#[must_use]
pub fn op_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(f32::EPSILON);
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    // mix(b, a, h) == b + (a - b) * h
    (b + (a - b) * h) - k * h * (1.0 - h)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn sphere_sign_and_magnitude() {
        assert!((sphere(Vec3::new(2.0, 0.0, 0.0), 1.0) - 1.0).abs() < EPS);
        assert!(sphere(Vec3::ZERO, 1.0) < 0.0);
        assert!((sphere(Vec3::new(0.0, 1.0, 0.0), 1.0)).abs() < EPS);
    }

    #[test]
    fn sphere_negative_radius_collapses() {
        // Negative radius is clamped to a point: distance == |point|.
        let p = Vec3::new(3.0, 4.0, 0.0);
        assert!((sphere(p, -5.0) - 5.0).abs() < EPS);
    }

    #[test]
    fn box_outside_inside_and_surface() {
        let he = Vec3::splat(1.0);
        assert!((box_exact(Vec3::new(2.0, 0.0, 0.0), he) - 1.0).abs() < EPS);
        assert!(box_exact(Vec3::ZERO, he) < 0.0);
        assert!(box_exact(Vec3::new(1.0, 0.0, 0.0), he).abs() < EPS);
    }

    #[test]
    fn rounded_box_shrinks_surface_inward() {
        let he = Vec3::splat(1.0);
        // On the +X face a round of 0.25 pulls the surface to x = 1.25.
        assert!(rounded_box(Vec3::new(1.25, 0.0, 0.0), he, 0.25).abs() < 1.0e-4);
    }

    #[test]
    fn capsule_matches_endpoint_sphere_and_midline() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        // Beyond endpoint `b`: behaves like a sphere about `b`.
        assert!((capsule(Vec3::new(3.0, 0.0, 0.0), a, b, 0.5) - 1.5).abs() < EPS);
        // Perpendicular from the midline.
        assert!((capsule(Vec3::new(0.0, 2.0, 0.0), a, b, 0.5) - 1.5).abs() < EPS);
    }

    #[test]
    fn capsule_degenerate_segment_is_sphere() {
        let a = Vec3::ZERO;
        let d = capsule(Vec3::new(0.0, 0.0, 2.0), a, a, 0.5);
        assert!((d - 1.5).abs() < EPS);
    }

    #[test]
    fn plane_signed_offset() {
        assert!((plane(Vec3::new(0.0, 3.0, 0.0), Vec3::Y, 0.0) - 3.0).abs() < EPS);
        assert!((plane(Vec3::new(0.0, -2.0, 0.0), Vec3::Y, 0.0) + 2.0).abs() < EPS);
    }

    #[test]
    fn plane_degenerate_normal_falls_back() {
        let d = plane(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, 0.0);
        assert!(d.is_finite());
        assert!((d - 5.0).abs() < EPS);
    }

    #[test]
    fn torus_surface_point() {
        // Major 2, minor 0.5: the outer equator sits at x = 2.5.
        assert!(torus(Vec3::new(2.5, 0.0, 0.0), 2.0, 0.5).abs() < EPS);
        // The hole centre is `minor` below the ring: distance major - minor.
        assert!((torus(Vec3::ZERO, 2.0, 0.5) - 1.5).abs() < EPS);
    }

    #[test]
    fn csg_operators() {
        assert!((op_union(0.3, -0.2) + 0.2).abs() < EPS);
        assert!((op_intersect(0.3, -0.2) - 0.3).abs() < EPS);
        // Subtract A (=0.3 means outside A -> -A = -0.3) from B (=0.1).
        assert!((op_subtract(0.3, 0.1) - 0.1).abs() < EPS);
    }

    #[test]
    fn smooth_union_bounds_and_limit() {
        let a = 0.4_f32;
        let b = 0.1_f32;
        let s = op_smooth_union(a, b, 0.5);
        // Smooth union never exceeds the hard union and rounds it down a bit.
        assert!(s <= op_union(a, b) + EPS);
        assert!(s < op_union(a, b));
        // k -> 0 recovers the hard minimum.
        assert!((op_smooth_union(a, b, 0.0) - a.min(b)).abs() < 1.0e-3);
    }

    #[test]
    fn smooth_union_is_finite_for_zero_k() {
        assert!(op_smooth_union(1.0, 2.0, 0.0).is_finite());
    }
}
