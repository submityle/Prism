//! 2D segment/line analytic intersection for the particle spatial contracts
//! (design §8.2, §12-§13).
//!
//! Several particle stages reduce a query to "do these two 2D edges cross, and
//! where": a collision-broadphase splits a swept footprint against a barrier
//! edge, a trail renderer clips a ribbon segment against a screen-space guide,
//! and an authoring gizmo snaps a drag handle onto the nearest guide line. This
//! module owns the small, `CPU`-verifiable contract those stages share: the
//! `orientation` predicate, the on-segment containment test, the two-segment
//! intersection classifier, its parametric form, and the infinite-line
//! intersection point.
//!
//! Segment intersection follows the classic `orientation`-triple method: the
//! signed area of the triangle formed by the two segment endpoints and a query
//! point classifies the turn as counter-clockwise, clockwise, or `collinear`.
//! Two segments cross in their interiors when each segment straddles the line
//! of the other; boundary touches (an endpoint lying on the other segment) and
//! `collinear` overlaps are handled as explicit special cases so the classifier
//! can distinguish a single crossing point from a `collinear` overlap range.
//!
//! # Strict scope
//! This module only performs *analytic 2D segment and line intersection*. It
//! deliberately does not test point containment in a polygon
//! (`point_in_polygon`), build or measure a hull
//! ([`super::convex_hull_2d`]), measure general polygon area
//! (`polygon_area_2d`), or intersect a 3D ray against geometry (`ray` /
//! `ray_triangle`); it neither imports nor reconstructs those contracts and
//! keeps its own private math helpers rather than sharing them.
//!
//! # No transcendental math
//! Every predicate here is pure `+`, `-`, `*` cross-product and comparison
//! arithmetic. There is no `sin`, `cos`, `atan`, `exp`, `ln`, `powf`, or any
//! other transcendental call, and no `f32` equality: near-zero magnitudes are
//! compared against [`CMP_EPS`].

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which a cross product, a coordinate difference, or a
/// parametric denominator is treated as zero. This is the comparison rule used
/// throughout instead of `==` on `f32`: two scalars are "equal" when their
/// absolute difference does not exceed this bound.
pub const CMP_EPS: f32 = 1.0e-6;

/// The classification of a two-segment intersection query.
///
/// `f32` payloads mean this type carries no `Eq`/`Hash`: it derives only the
/// arithmetic-free `PartialEq` needed to compare a computed crossing against an
/// expected point in tests and callers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SegIntersect {
    /// The segments do not meet.
    None,
    /// The segments meet at exactly one point (a proper crossing, a boundary
    /// touch, or a `collinear` endpoint contact).
    Point([f32; 2]),
    /// The segments are `collinear` and overlap along a shared sub-segment.
    Collinear,
}

/// The 2D cross product `(b - a) x (c - a)`, i.e. twice the signed area of the
/// triangle `a, b, c`.
fn cross_diff(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// The 2D cross product of two free vectors `u x v`.
fn cross_vec(u: [f32; 2], v: [f32; 2]) -> f32 {
    u[0] * v[1] - u[1] * v[0]
}

/// The dot product of two free vectors `u . v`.
fn dot_vec(u: [f32; 2], v: [f32; 2]) -> f32 {
    u[0] * v[0] + u[1] * v[1]
}

/// Returns `true` when two points coincide within [`CMP_EPS`] on both axes.
fn points_equal(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() <= CMP_EPS && (a[1] - b[1]).abs() <= CMP_EPS
}

/// The point `a + t * (b - a)` along the line through `a` and `b`.
fn point_at(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// Classifies the turn `a -> b -> c` by the sign of the triangle's signed area.
///
/// Returns `1` for a counter-clockwise (left) turn, `-1` for a clockwise
/// (right) turn, and `0` when the three points are `collinear` (the cross
/// product's magnitude does not exceed [`CMP_EPS`]).
#[must_use]
pub fn orientation(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> i32 {
    let v = cross_diff(a, b, c);
    if v > CMP_EPS {
        1
    } else if v < -CMP_EPS {
        -1
    } else {
        0
    }
}

/// Returns `true` when point `p` lies on segment `ab`.
///
/// The test assumes `p` is (near-)`collinear` with `a` and `b` and combines two
/// checks: `p` must fall inside the axis-aligned bounding box of `ab` (widened
/// by [`CMP_EPS`]), and its projection onto the segment direction must land
/// between the endpoints. A degenerate segment (`a` equal to `b`) is on-segment
/// only for a `p` coincident with that shared point.
#[must_use]
pub fn on_segment(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> bool {
    let min_x = a[0].min(b[0]);
    let max_x = a[0].max(b[0]);
    let min_y = a[1].min(b[1]);
    let max_y = a[1].max(b[1]);
    let in_box = p[0] >= min_x - CMP_EPS
        && p[0] <= max_x + CMP_EPS
        && p[1] >= min_y - CMP_EPS
        && p[1] <= max_y + CMP_EPS;
    if !in_box {
        return false;
    }

    let dir = [b[0] - a[0], b[1] - a[1]];
    let len2 = dot_vec(dir, dir);
    if len2 <= CMP_EPS {
        // Degenerate segment: `a` and `b` coincide, so only a coincident `p`
        // lies on it.
        return points_equal(a, p);
    }

    let proj = dot_vec([p[0] - a[0], p[1] - a[1]], dir);
    proj >= -CMP_EPS && proj <= len2 + CMP_EPS
}

/// Picks a non-degenerate direction from the two segments, preferring the first.
///
/// Returns `None` when both segments are degenerate points.
fn overlap_direction(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> Option<[f32; 2]> {
    let d1 = [p2[0] - p1[0], p2[1] - p1[1]];
    if dot_vec(d1, d1) > CMP_EPS {
        return Some(d1);
    }
    let d2 = [p4[0] - p3[0], p4[1] - p3[1]];
    if dot_vec(d2, d2) > CMP_EPS {
        return Some(d2);
    }
    None
}

/// Classifies the overlap of two `collinear` segments along their shared line.
///
/// Projects every endpoint onto the segment direction, intersects the two
/// 1D intervals, and reports [`SegIntersect::None`] for a gap, a single
/// [`SegIntersect::Point`] for an endpoint touch, or [`SegIntersect::Collinear`]
/// for a shared sub-segment.
fn collinear_overlap(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> SegIntersect {
    let Some(dir) = overlap_direction(p1, p2, p3, p4) else {
        // Both segments are single points: they meet only if coincident.
        return if points_equal(p1, p3) {
            SegIntersect::Point(p1)
        } else {
            SegIntersect::None
        };
    };

    let s1 = dot_vec(p1, dir);
    let s2 = dot_vec(p2, dir);
    let s3 = dot_vec(p3, dir);
    let s4 = dot_vec(p4, dir);
    let lo = s1.min(s2).max(s3.min(s4));
    let hi = s1.max(s2).min(s3.max(s4));

    if lo > hi + CMP_EPS {
        return SegIntersect::None;
    }
    if (hi - lo).abs() <= CMP_EPS {
        // A single shared point: return the endpoint whose projection matches.
        for &p in &[p1, p2, p3, p4] {
            if (dot_vec(p, dir) - lo).abs() <= CMP_EPS {
                return SegIntersect::Point(p);
            }
        }
        return SegIntersect::None;
    }
    SegIntersect::Collinear
}

/// Returns the parametric coordinates `(t, u)` of the intersection of the two
/// infinite lines through `p1 -> p2` and `p3 -> p4`.
///
/// `t` runs along `p1 -> p2` and `u` along `p3 -> p4`, so the crossing point is
/// `p1 + t * (p2 - p1) = p3 + u * (p4 - p3)`. The two *segments* intersect
/// exactly when both parameters lie in the closed unit interval, i.e.
/// `(0.0..=1.0).contains(&t)` and `(0.0..=1.0).contains(&u)`.
///
/// Returns `None` when the segment directions are parallel (or one is
/// degenerate): the guarded denominator's magnitude does not exceed
/// [`CMP_EPS`], so there is no unique solution.
#[must_use]
pub fn intersect_params(
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    p4: [f32; 2],
) -> Option<(f32, f32)> {
    let d1 = [p2[0] - p1[0], p2[1] - p1[1]];
    let d2 = [p4[0] - p3[0], p4[1] - p3[1]];
    let denom = cross_vec(d1, d2);
    if denom.abs() <= CMP_EPS {
        return None;
    }
    let diff = [p3[0] - p1[0], p3[1] - p1[1]];
    let t = cross_vec(diff, d2) / denom;
    let u = cross_vec(diff, d1) / denom;
    Some((t, u))
}

/// Classifies the intersection of segment `p1 p2` with segment `p3 p4`.
///
/// A proper interior crossing yields a [`SegIntersect::Point`] computed from the
/// parametric solution; a boundary touch (an endpoint lying on the other
/// segment) yields the touched endpoint; two `collinear` segments yield either a
/// single [`SegIntersect::Point`] endpoint contact or a
/// [`SegIntersect::Collinear`] overlap; disjoint segments yield
/// [`SegIntersect::None`].
#[must_use]
pub fn intersect(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], p4: [f32; 2]) -> SegIntersect {
    let o1 = orientation(p1, p2, p3);
    let o2 = orientation(p1, p2, p4);
    let o3 = orientation(p3, p4, p1);
    let o4 = orientation(p3, p4, p2);

    // Proper interior crossing: each segment strictly straddles the other's
    // line, so all four turns are non-zero with opposite signs per segment.
    if o1 != 0
        && o2 != 0
        && o3 != 0
        && o4 != 0
        && (o1 > 0) != (o2 > 0)
        && (o3 > 0) != (o4 > 0)
        && let Some((t, _u)) = intersect_params(p1, p2, p3, p4)
    {
        return SegIntersect::Point(point_at(p1, p2, t));
    }

    // Fully `collinear`: classify the 1D overlap along the shared line.
    if o1 == 0 && o2 == 0 && o3 == 0 && o4 == 0 {
        return collinear_overlap(p1, p2, p3, p4);
    }

    // Boundary touches: an endpoint of one segment lies on the other segment.
    if o1 == 0 && on_segment(p1, p2, p3) {
        return SegIntersect::Point(p3);
    }
    if o2 == 0 && on_segment(p1, p2, p4) {
        return SegIntersect::Point(p4);
    }
    if o3 == 0 && on_segment(p3, p4, p1) {
        return SegIntersect::Point(p1);
    }
    if o4 == 0 && on_segment(p3, p4, p2) {
        return SegIntersect::Point(p2);
    }

    SegIntersect::None
}

/// Returns the intersection point of the two *infinite* lines through
/// `p1 -> p2` and `p3 -> p4`.
///
/// Unlike [`intersect`], the parameters are not clamped to the segments, so the
/// crossing may lie on the extensions of either input. Returns `None` when the
/// lines are parallel or coincident (the denominator's magnitude does not
/// exceed [`CMP_EPS`]).
#[must_use]
pub fn line_intersection(
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    p4: [f32; 2],
) -> Option<[f32; 2]> {
    let (t, _u) = intersect_params(p1, p2, p3, p4)?;
    Some(point_at(p1, p2, t))
}

/// Total `std430` storage bytes for a `GPU` buffer of `seg_count` 2D segments.
///
/// Each segment occupies two `vec2<f32>` endpoints, so the element count is
/// `2 * seg_count`. The size is computed through [`storage_bytes`], which
/// clamps an empty buffer up to a single element so a `WebGPU` storage binding
/// is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(seg_count: usize) -> usize {
    storage_bytes(VEC2_STRIDE, seg_count.saturating_mul(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_classifies_turns() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        assert_eq!(orientation(a, b, [0.5, 1.0]), 1);
        assert_eq!(orientation(a, b, [0.5, -1.0]), -1);
        assert_eq!(orientation(a, b, [2.0, 0.0]), 0);
    }

    #[test]
    fn orientation_treats_tiny_area_as_collinear() {
        // A deviation below CMP_EPS must still read as `collinear`.
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        assert_eq!(orientation(a, b, [0.5, 1.0e-7]), 0);
    }

    #[test]
    fn on_segment_accepts_endpoints_and_midpoint() {
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        assert!(on_segment(a, b, a));
        assert!(on_segment(a, b, b));
        assert!(on_segment(a, b, [2.0, 0.0]));
    }

    #[test]
    fn on_segment_rejects_points_beyond_endpoints() {
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        assert!(!on_segment(a, b, [-0.5, 0.0]));
        assert!(!on_segment(a, b, [4.5, 0.0]));
    }

    #[test]
    fn on_segment_degenerate_only_matches_shared_point() {
        let a = [1.0, 1.0];
        assert!(on_segment(a, a, [1.0, 1.0]));
        assert!(!on_segment(a, a, [1.0, 2.0]));
    }

    #[test]
    fn intersect_general_crossing_returns_point() {
        let hit = intersect([-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]);
        assert_eq!(hit, SegIntersect::Point([0.0, 0.0]));
    }

    #[test]
    fn intersect_general_crossing_offset() {
        // Two diagonals of the unit square meet at its center.
        let hit = intersect([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]);
        assert_eq!(hit, SegIntersect::Point([1.0, 1.0]));
    }

    #[test]
    fn intersect_params_midpoint_values() {
        let params = intersect_params([-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]);
        let (t, u) = params.expect("non-parallel segments have parameters");
        assert!((t - 0.5).abs() <= CMP_EPS);
        assert!((u - 0.5).abs() <= CMP_EPS);
    }

    #[test]
    fn intersect_params_range_gates_the_crossing() {
        // The lines meet, but the crossing lies outside segment `p3 p4`.
        let params = intersect_params([-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 3.0]);
        let (t, u) = params.expect("non-parallel");
        assert!((0.0..=1.0).contains(&t));
        assert!(!(0.0..=1.0).contains(&u));
    }

    #[test]
    fn intersect_params_parallel_is_none() {
        assert!(intersect_params([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]).is_none());
    }

    #[test]
    fn intersect_disjoint_returns_none() {
        let hit = intersect([0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]);
        assert_eq!(hit, SegIntersect::None);
    }

    #[test]
    fn intersect_parallel_offset_returns_none() {
        let hit = intersect([0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]);
        assert_eq!(hit, SegIntersect::None);
    }

    #[test]
    fn intersect_crossing_outside_segments_is_none() {
        // Their infinite lines meet, but neither segment reaches the crossing.
        let hit = intersect([0.0, 0.0], [1.0, 0.0], [2.0, -1.0], [2.0, 1.0]);
        assert_eq!(hit, SegIntersect::None);
    }

    #[test]
    fn intersect_shared_endpoint_returns_point() {
        let hit = intersect([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]);
        assert_eq!(hit, SegIntersect::Point([1.0, 1.0]));
    }

    #[test]
    fn intersect_t_junction_touch_returns_point() {
        // The endpoint `p3` lies on the interior of segment `p1 p2`.
        let hit = intersect([0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.0, 3.0]);
        assert_eq!(hit, SegIntersect::Point([2.0, 0.0]));
    }

    #[test]
    fn intersect_collinear_overlap_range() {
        let hit = intersect([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]);
        assert_eq!(hit, SegIntersect::Collinear);
    }

    #[test]
    fn intersect_collinear_fully_contained() {
        let hit = intersect([0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [2.0, 0.0]);
        assert_eq!(hit, SegIntersect::Collinear);
    }

    #[test]
    fn intersect_collinear_endpoint_touch_returns_point() {
        let hit = intersect([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]);
        assert_eq!(hit, SegIntersect::Point([1.0, 0.0]));
    }

    #[test]
    fn intersect_collinear_disjoint_returns_none() {
        let hit = intersect([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]);
        assert_eq!(hit, SegIntersect::None);
    }

    #[test]
    fn intersect_collinear_diagonal_overlap() {
        // Same predicate holds off the axes: a shared diagonal sub-segment.
        let hit = intersect([0.0, 0.0], [2.0, 2.0], [1.0, 1.0], [3.0, 3.0]);
        assert_eq!(hit, SegIntersect::Collinear);
    }

    #[test]
    fn intersect_degenerate_point_on_segment() {
        // A zero-length segment sitting on the other segment is a point contact.
        let hit = intersect([2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [4.0, 0.0]);
        assert_eq!(hit, SegIntersect::Point([2.0, 0.0]));
    }

    #[test]
    fn intersect_degenerate_point_off_segment_is_none() {
        let hit = intersect([2.0, 5.0], [2.0, 5.0], [0.0, 0.0], [4.0, 0.0]);
        assert_eq!(hit, SegIntersect::None);
    }

    #[test]
    fn line_intersection_meets_at_crossing() {
        let pt = line_intersection([-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]);
        assert_eq!(pt, Some([0.0, 0.0]));
    }

    #[test]
    fn line_intersection_extends_beyond_segments() {
        // The segments never touch, but their infinite lines cross at (2, 0).
        let pt = line_intersection([0.0, 0.0], [1.0, 0.0], [2.0, -1.0], [2.0, 1.0]);
        let hit = pt.expect("non-parallel lines cross");
        assert!((hit[0] - 2.0).abs() <= CMP_EPS);
        assert!(hit[1].abs() <= CMP_EPS);
    }

    #[test]
    fn line_intersection_parallel_is_none() {
        assert!(line_intersection([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]).is_none());
    }

    #[test]
    fn line_intersection_coincident_is_none() {
        // Coincident lines share every point, so there is no unique crossing.
        assert!(line_intersection([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]).is_none());
    }

    #[test]
    fn seg_intersect_equality_discriminates_variants() {
        assert_eq!(
            SegIntersect::Point([1.0, 2.0]),
            SegIntersect::Point([1.0, 2.0])
        );
        assert_ne!(
            SegIntersect::Point([1.0, 2.0]),
            SegIntersect::Point([2.0, 1.0])
        );
        assert_ne!(SegIntersect::None, SegIntersect::Collinear);
    }

    #[test]
    fn gpu_storage_bytes_counts_two_endpoints_per_segment() {
        assert_eq!(gpu_storage_bytes(4), storage_bytes(VEC2_STRIDE, 8));
        assert_eq!(gpu_storage_bytes(4), VEC2_STRIDE * 8);
    }

    #[test]
    fn gpu_storage_bytes_empty_is_one_element() {
        assert_eq!(gpu_storage_bytes(0), VEC2_STRIDE);
    }
}
