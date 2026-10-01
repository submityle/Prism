//! Closest-point queries against segments, boxes, and triangles.
//!
//! These are exact analytic closest-feature computations following the
//! standard Voronoi-region derivations (Ericson, *Real-Time Collision
//! Detection*). They are engine-agnostic implementations of publicly
//! documented algorithms and contain no Unreal Engine source or derived code.
//!
//! All routines avoid transcendental functions: they rely only on dot/cross
//! products, component clamps, and ratios, so they are exact up to floating
//! point rounding and cheap enough for narrow-phase refinement.

use glam::Vec3;

use crate::bounding::Aabb;

/// Returns the point on the segment `[a, b]` closest to `p`.
///
/// A degenerate segment (`a == b`) returns `a`.
pub fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let ab = b - a;
    let denom = ab.dot(ab);
    if denom <= f32::EPSILON {
        return a;
    }
    let t = ((p - a).dot(ab) / denom).clamp(0.0, 1.0);
    a + ab * t
}

/// Returns the point on or inside `aabb` closest to `p`.
///
/// When `p` is already inside the box the point itself is returned.
pub fn closest_point_on_aabb(p: Vec3, aabb: &Aabb) -> Vec3 {
    p.clamp(aabb.min, aabb.max)
}

/// Returns the point on the triangle `(a, b, c)` closest to `p`.
///
/// The result lies in one of the triangle's seven Voronoi regions: a vertex,
/// an edge, or the interior face.
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    // Vertex region outside a.
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    // Vertex region outside b.
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    // Edge region ab.
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    // Vertex region outside c.
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    // Edge region ac.
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }

    // Edge region bc.
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    // Interior face region: project via barycentric coordinates.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// The closest pair of points between two segments.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SegmentClosest {
    /// Parameter in `[0, 1]` of the closest point on the first segment.
    pub s: f32,
    /// Parameter in `[0, 1]` of the closest point on the second segment.
    pub t: f32,
    /// Closest point on the first segment.
    pub c1: Vec3,
    /// Closest point on the second segment.
    pub c2: Vec3,
    /// Squared distance between `c1` and `c2`.
    pub distance_squared: f32,
}

/// Computes the closest pair of points between the segments `[p1, q1]` and
/// `[p2, q2]`.
///
/// Handles both degenerate (point) segments and the parallel case, always
/// returning parameters clamped to `[0, 1]`.
pub fn closest_points_segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> SegmentClosest {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    let (s, t);
    if a <= f32::EPSILON && e <= f32::EPSILON {
        // Both segments degenerate to points.
        s = 0.0;
        t = 0.0;
    } else if a <= f32::EPSILON {
        // First segment degenerates to a point.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= f32::EPSILON {
            // Second segment degenerates to a point.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let s0 = if denom > f32::EPSILON {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                // Parallel segments: pick an arbitrary point on the first.
                0.0
            };
            let t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t0;
                s = s0;
            }
        }
    }

    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    SegmentClosest {
        s,
        t,
        c1,
        c2,
        distance_squared: (c1 - c2).length_squared(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        closest_point_on_aabb, closest_point_on_segment, closest_point_on_triangle,
        closest_points_segment_segment,
    };
    use crate::bounding::Aabb;
    use approx::assert_relative_eq;
    use glam::Vec3;

    #[test]
    fn segment_projection_interior_and_clamped() {
        let a = Vec3::new(-1.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        // Projects to the midpoint.
        let mid = closest_point_on_segment(Vec3::new(0.0, 2.0, 0.0), a, b);
        assert!(mid.abs_diff_eq(Vec3::ZERO, 1e-6));
        // Clamps past the b endpoint.
        let clamped = closest_point_on_segment(Vec3::new(5.0, 2.0, 0.0), a, b);
        assert!(clamped.abs_diff_eq(b, 1e-6));
    }

    #[test]
    fn segment_degenerate_returns_start() {
        let a = Vec3::new(3.0, 1.0, -2.0);
        assert!(closest_point_on_segment(Vec3::ZERO, a, a).abs_diff_eq(a, 1e-6));
    }

    #[test]
    fn aabb_clamps_outside_and_keeps_inside() {
        let box_ = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let outside = closest_point_on_aabb(Vec3::new(5.0, -3.0, 0.5), &box_);
        assert!(outside.abs_diff_eq(Vec3::new(1.0, -1.0, 0.5), 1e-6));
        let inside = Vec3::new(0.25, -0.5, 0.75);
        assert!(closest_point_on_aabb(inside, &box_).abs_diff_eq(inside, 1e-6));
    }

    #[test]
    fn triangle_interior_edge_and_vertex() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);
        // Point above the face projects straight down into the interior.
        let interior = closest_point_on_triangle(Vec3::new(0.5, 0.5, 3.0), a, b, c);
        assert!(interior.abs_diff_eq(Vec3::new(0.5, 0.5, 0.0), 1e-6));
        // Point beyond vertex a snaps to a.
        let vertex = closest_point_on_triangle(Vec3::new(-1.0, -1.0, 0.0), a, b, c);
        assert!(vertex.abs_diff_eq(a, 1e-6));
        // Point outside edge ab snaps onto the edge.
        let edge = closest_point_on_triangle(Vec3::new(1.0, -1.0, 0.0), a, b, c);
        assert!(edge.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn segment_segment_skew_lines() {
        // X-axis segment and a parallel-to-Y segment offset in z.
        let r = closest_points_segment_segment(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        );
        assert!(r.c1.abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(r.c2.abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-6));
        assert_relative_eq!(r.distance_squared, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn segment_segment_parallel() {
        let r = closest_points_segment_segment(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
        );
        assert_relative_eq!(r.distance_squared, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn segment_segment_both_points() {
        let r = closest_points_segment_segment(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 6.0, 3.0),
            Vec3::new(4.0, 6.0, 3.0),
        );
        assert_relative_eq!(r.distance_squared, 25.0, epsilon = 1e-6);
    }
}
