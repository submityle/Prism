//! Unsigned distance primitives for the `CPU` golden path.
//!
//! The analytic solids in [`super::sdf_primitives`] are closed surfaces whose
//! sign tells inside from outside. Some geometry has no well-defined interior:
//! a line segment, or a single triangle in space. For those the natural query
//! is the *unsigned* Euclidean distance to the closest point of the feature,
//! which is exactly what point-to-mesh proximity, capsule-vs-triangle contact,
//! and brush-stroke distance fields need before any sign is assigned.
//!
//! [`segment_distance`] returns the distance to a finite 3-D line segment, and
//! [`triangle_distance`] returns the distance to a 3-D triangle, following
//! Inigo Quilez's `udTriangle`: it first classifies the orthogonal projection
//! against the three edge planes; if the projection falls inside the triangle
//! the distance is the perpendicular drop onto the plane, otherwise it is the
//! smallest distance to the three bounding edge segments.
//!
//! Both are built from cross and dot products, `clamp`, `sign`, and the `sqrt`
//! inside a length — all permitted — so the module stays transcendental-free.

/// Dot product of two 3-vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Squared length of a 3-vector (`dot(v, v)`), the kernel of every distance
/// comparison below so the costly `sqrt` is taken exactly once at the end.
fn dot2(v: [f32; 3]) -> f32 {
    dot(v, v)
}

/// Cross product of two 3-vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Difference `a - b` of two 3-vectors.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Squared distance from `point` to the closest point on the segment `a`-`b`.
///
/// Projects the point onto the infinite line, clamps the parameter to the
/// finite segment, and returns the squared distance to that closest point.
/// A degenerate segment (`a == b`) reduces to the squared distance to `a`.
fn segment_distance_sq(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let pa = sub(point, a);
    let ba = sub(b, a);
    let ba_len_sq = dot2(ba);
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        (dot(pa, ba) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dot2([pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h])
}

/// Unsigned distance from `point` to the finite line segment from `a` to `b`.
///
/// This is the distance to the closest point of the segment, clamped to the
/// endpoints; a degenerate segment (`a == b`) returns the distance to `a`.
pub fn segment_distance(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    segment_distance_sq(point, a, b).sqrt()
}

/// Unsigned distance from `point` to the triangle with vertices `a`, `b`, `c`.
///
/// Follows Inigo Quilez's `udTriangle`: the three edge-plane sign tests decide
/// whether the orthogonal projection lands inside the triangle. When it does,
/// the distance is the perpendicular drop onto the triangle's plane; when it
/// does not, it is the smallest distance to the three edge segments. A
/// degenerate triangle (collinear or coincident vertices) has a near-zero
/// face normal, so the edge-segment branch is taken and the result is the
/// distance to the closest bounding edge.
pub fn triangle_distance(point: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ba = sub(b, a);
    let pa = sub(point, a);
    let cb = sub(c, b);
    let pb = sub(point, b);
    let ac = sub(a, c);
    let pc = sub(point, c);
    let normal = cross(ba, ac);

    // Each term is +1 when the point is on the outer side of an edge plane.
    let edge_votes = dot(cross(ba, normal), pa).signum()
        + dot(cross(cb, normal), pb).signum()
        + dot(cross(ac, normal), pc).signum();

    if edge_votes < 2.0 {
        // Projection falls outside the triangle: nearest of the three edges.
        let e0 = segment_distance_sq(point, a, b);
        let e1 = segment_distance_sq(point, b, c);
        let e2 = segment_distance_sq(point, c, a);
        e0.min(e1).min(e2).sqrt()
    } else {
        // Projection falls inside: perpendicular drop onto the face plane.
        let num = dot(normal, pa);
        (num * num / dot2(normal)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::{segment_distance, triangle_distance};

    #[test]
    fn segment_distance_beside_the_midpoint() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        // One unit to the side of the midpoint.
        assert!((segment_distance([1.0, 0.5, 0.0], a, b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn segment_distance_clamps_past_the_endpoint() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        // Beyond `b`: distance is to the endpoint, not the infinite line.
        assert!((segment_distance([0.0, 2.0, 0.0], a, b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn segment_distance_degenerate_is_point_distance() {
        let a = [1.0, 1.0, 1.0];
        assert!((segment_distance([4.0, 1.0, 1.0], a, a) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn triangle_distance_perpendicular_above_the_face() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // Above a point whose projection lands inside: perpendicular drop.
        assert!((triangle_distance([0.25, 0.25, 1.0], a, b, c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn triangle_distance_outside_falls_back_to_an_edge() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        // Beyond vertex `a`, in-plane: closest feature is the vertex itself.
        let d = triangle_distance([-1.0, -1.0, 0.0], a, b, c);
        assert!((d - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn triangle_distance_on_the_face_is_zero() {
        let a = [0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0];
        let c = [0.0, 2.0, 0.0];
        // A point inside the triangle lies on the surface.
        assert!(triangle_distance([0.5, 0.5, 0.0], a, b, c) < 1e-6);
    }
}
