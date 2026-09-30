//! Point-in-polygon containment tests and signed boundary distance for the 2D
//! particle spatial contracts (design §8.2, §12-§13).
//!
//! Several particle stages need to know whether a 2D sample lies inside an
//! arbitrary polygon and, if so, how deep: a spawn-shape module masks emission
//! to an authored region; a collision proxy rejects particles that leave a
//! keep-in area; and a screen-space kill-volume fades particles by their signed
//! distance to a boundary. This module owns the small, `CPU`-verifiable
//! contract those stages share: the two classic containment rules (even-odd ray
//! crossing and non-zero winding), the raw winding number they build on, the
//! point-to-segment distance primitive, and the signed distance to a polygon
//! boundary (negative inside, positive outside).
//!
//! Two containment rules are provided because they disagree on
//! self-intersecting rings. The even-odd (crossing) rule counts how many times a
//! ray from the point crosses the boundary and calls the point inside when that
//! count is odd. The non-zero winding rule sums the signed turns the boundary
//! makes around the point and calls the point inside when that sum is not zero.
//! For a simple (non-self-intersecting) polygon the two agree; for a star or a
//! figure-eight they differ, and callers pick the rule that matches their
//! authoring intent.
//!
//! # Strict scope
//! This module only *tests containment* and *measures signed boundary
//! distance*. It deliberately does not build a hull ([`super::convex_hull_2d`]),
//! compute polygon area / centroid / boundary orientation (the `polygon_area_2d`
//! sibling), triangulate ([`super::ear_clip_triangulate`]), or intersect
//! segments (the `segment_intersect_2d` sibling). It neither imports nor
//! reconstructs those contracts, and it keeps its own small math helpers rather
//! than sharing them.
//!
//! # No transcendental math
//! Every routine here is pure `+`, `-`, `*`, `/`, comparison and one
//! `f32::sqrt` (the segment distance). There is no `sin`, `cos`, `atan`, `exp`,
//! `ln`, `powf` or any other transcendental call, and no `f32` equality:
//! near-zero magnitudes are compared against [`CMP_EPS`].

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which a squared edge length is treated as degenerate. This
/// is the comparison rule used instead of `==` on `f32`: a segment whose
/// squared length does not exceed this bound is treated as a single point when
/// projecting onto it.
pub const CMP_EPS: f32 = 1.0e-6;

/// Twice the signed area of triangle `a, b, p`, i.e. the 2D cross product
/// `(b - a) x (p - a)`.
///
/// The sign tells which side of the directed edge `a -> b` the point `p` lies
/// on: strictly positive when `p` is to the left (a counter-clockwise turn),
/// strictly negative when to the right, and zero when the three points are
/// collinear.
fn is_left(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])
}

/// Tests containment with the even-odd ray-crossing rule.
///
/// A horizontal ray is cast from `point` and the number of polygon edges it
/// crosses is counted; the point is inside when that count is odd. Vertices on
/// the ray are handled by the half-open `>` comparison so a shared vertex is
/// counted exactly once. A polygon with fewer than three vertices encloses no
/// area and always returns `false`.
///
/// For a self-intersecting ring this differs from
/// [`point_in_polygon_winding`]: the overlapping core of a star is *outside*
/// under the even-odd rule.
#[must_use]
pub fn point_in_polygon_crossing(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let px = point[0];
    let py = point[1];
    let mut inside = false;
    for (i, &vi) in polygon.iter().enumerate() {
        let vj = polygon[(i + n - 1) % n];
        let (xi, yi) = (vi[0], vi[1]);
        let (xj, yj) = (vj[0], vj[1]);
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
    }
    inside
}

/// Computes the winding number of `polygon` around `point`.
///
/// The winding number is the net number of counter-clockwise turns the closed
/// boundary makes around the point: `0` when the point is outside a simple
/// polygon, `+1` for a point enclosed once by a counter-clockwise (`CCW`) ring,
/// `-1` for a clockwise (`CW`) ring, and larger magnitudes for a boundary that
/// wraps the point more than once. A polygon with fewer than three vertices
/// returns `0`.
///
/// This uses the crossing-number-with-side-test formulation: only edges that
/// cross the point's horizontal level contribute, and the [`is_left`] side test
/// decides the sign of each crossing, so no `atan2` accumulation is needed.
#[must_use]
pub fn winding_number(point: [f32; 2], polygon: &[[f32; 2]]) -> i32 {
    let n = polygon.len();
    if n < 3 {
        return 0;
    }
    let py = point[1];
    let mut wn: i32 = 0;
    for (i, &a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % n];
        if a[1] <= py {
            if b[1] > py && is_left(a, b, point) > 0.0 {
                wn += 1;
            }
        } else if b[1] <= py && is_left(a, b, point) < 0.0 {
            wn -= 1;
        }
    }
    wn
}

/// Tests containment with the non-zero winding rule.
///
/// The point is inside when [`winding_number`] is not zero. For a simple
/// polygon this agrees with [`point_in_polygon_crossing`]; for a
/// self-intersecting ring (a star, a figure-eight) the overlapping regions that
/// the boundary wraps twice are still inside under this rule even though the
/// even-odd rule reports them as outside.
#[must_use]
pub fn point_in_polygon_winding(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    winding_number(point, polygon) != 0
}

/// Euclidean distance from `point` to the line segment `a -> b`.
///
/// The point is projected onto the infinite line through `a` and `b`; the
/// projection parameter is clamped to `[0, 1]` with [`f32::clamp`] so the
/// nearest point never leaves the segment. A degenerate segment (endpoints
/// closer than [`CMP_EPS`] in squared length) collapses to its start point
/// `a`, avoiding a divide-by-zero.
#[must_use]
pub fn distance_to_edge(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let apx = point[0] - a[0];
    let apy = point[1] - a[1];
    let denom = abx * abx + aby * aby;
    let t = if denom <= CMP_EPS {
        0.0
    } else {
        ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    };
    let dx = point[0] - (a[0] + abx * t);
    let dy = point[1] - (a[1] + aby * t);
    (dx * dx + dy * dy).sqrt()
}

/// Signed distance from `point` to the boundary of `polygon`.
///
/// The magnitude is the distance to the nearest polygon edge; the sign is
/// negative when the point is inside (by the non-zero winding rule) and
/// positive when it is outside, so the boundary itself sits near zero. A
/// polygon with fewer than two vertices has no edge and returns
/// [`f32::INFINITY`].
///
/// Containment uses [`point_in_polygon_winding`], so a self-intersecting ring's
/// double-wrapped interior is treated as inside (negative).
#[must_use]
pub fn signed_distance_to_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> f32 {
    let n = polygon.len();
    if n < 2 {
        return f32::INFINITY;
    }
    let mut min_d = f32::INFINITY;
    for (i, &a) in polygon.iter().enumerate() {
        let b = polygon[(i + 1) % n];
        min_d = min_d.min(distance_to_edge(point, a, b));
    }
    if point_in_polygon_winding(point, polygon) {
        -min_d
    } else {
        min_d
    }
}

/// Byte size of the `std430` `GPU` storage buffer that holds `vertex_count`
/// polygon vertices as `vec2<f32>` elements.
///
/// Delegates to [`storage_bytes`] with the shared [`VEC2_STRIDE`], so an empty
/// polygon still reserves a single non-empty element (a `WebGPU` storage
/// binding may not be zero-sized) and the multiplication saturates rather than
/// wrapping.
#[must_use]
pub fn gpu_storage_bytes(vertex_count: usize) -> usize {
    storage_bytes(VEC2_STRIDE, vertex_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPS: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    fn unit_square() -> [[f32; 2]; 4] {
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    }

    fn triangle() -> [[f32; 2]; 3] {
        [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]
    }

    // A five-pointed star whose self-intersecting core distinguishes the
    // even-odd rule from the non-zero winding rule.
    fn star() -> [[f32; 2]; 5] {
        [
            [0.0, 3.0],
            [2.0, -3.0],
            [-3.0, 1.0],
            [3.0, 1.0],
            [-2.0, -3.0],
        ]
    }

    #[test]
    fn crossing_inside_square_is_true() {
        assert!(point_in_polygon_crossing([1.0, 1.0], &unit_square()));
    }

    #[test]
    fn crossing_outside_square_is_false() {
        assert!(!point_in_polygon_crossing([3.0, 1.0], &unit_square()));
        assert!(!point_in_polygon_crossing([1.0, -1.0], &unit_square()));
    }

    #[test]
    fn crossing_inside_triangle_is_true() {
        assert!(point_in_polygon_crossing([1.0, 1.0], &triangle()));
    }

    #[test]
    fn crossing_outside_triangle_is_false() {
        // Beyond the hypotenuse x + y = 4.
        assert!(!point_in_polygon_crossing([3.0, 3.0], &triangle()));
    }

    #[test]
    fn crossing_empty_polygon_is_false() {
        assert!(!point_in_polygon_crossing([0.0, 0.0], &[]));
    }

    #[test]
    fn crossing_degenerate_two_vertices_is_false() {
        assert!(!point_in_polygon_crossing(
            [0.5, 0.0],
            &[[0.0, 0.0], [1.0, 0.0]]
        ));
    }

    #[test]
    fn crossing_concave_l_shape_notch_is_outside() {
        // An L-shape: the notch in the upper-right quadrant is outside.
        let l_shape = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [2.0, 2.0],
            [2.0, 4.0],
            [0.0, 4.0],
        ];
        assert!(point_in_polygon_crossing([1.0, 1.0], &l_shape));
        assert!(!point_in_polygon_crossing([3.0, 3.0], &l_shape));
    }

    #[test]
    fn winding_ccw_square_encloses_once() {
        assert_eq!(winding_number([1.0, 1.0], &unit_square()), 1);
    }

    #[test]
    fn winding_outside_square_is_zero() {
        assert_eq!(winding_number([5.0, 5.0], &unit_square()), 0);
    }

    #[test]
    fn winding_cw_square_is_negative_one() {
        let cw = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]];
        assert_eq!(winding_number([1.0, 1.0], &cw), -1);
    }

    #[test]
    fn winding_degenerate_polygon_is_zero() {
        assert_eq!(winding_number([0.0, 0.0], &[[0.0, 0.0], [1.0, 0.0]]), 0);
    }

    #[test]
    fn winding_rule_matches_crossing_for_simple_polygon() {
        let probes = [[1.0, 1.0], [0.5, 1.5], [3.0, 1.0], [1.0, -1.0], [1.9, 0.1]];
        for p in probes {
            assert_eq!(
                point_in_polygon_winding(p, &unit_square()),
                point_in_polygon_crossing(p, &unit_square()),
                "disagreement at {p:?}",
            );
        }
    }

    #[test]
    fn winding_inside_true_outside_false() {
        assert!(point_in_polygon_winding([1.0, 1.0], &unit_square()));
        assert!(!point_in_polygon_winding([-1.0, 1.0], &unit_square()));
    }

    #[test]
    fn star_core_differs_between_rules() {
        // The central overlap of a five-pointed star is wrapped twice: the
        // non-zero winding rule calls it inside, the even-odd rule outside.
        let center = [0.0, 0.0];
        assert!(point_in_polygon_winding(center, &star()));
        assert!(!point_in_polygon_crossing(center, &star()));
    }

    #[test]
    fn distance_to_edge_interior_projection() {
        // Nearest point is the perpendicular foot on the segment.
        let d = distance_to_edge([1.0, 2.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(approx(d, 2.0));
    }

    #[test]
    fn distance_to_edge_beyond_start_clamps_to_a() {
        let d = distance_to_edge([-3.0, 4.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(approx(d, 5.0));
    }

    #[test]
    fn distance_to_edge_beyond_end_clamps_to_b() {
        let d = distance_to_edge([5.0, 0.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(approx(d, 3.0));
    }

    #[test]
    fn distance_to_edge_degenerate_segment_uses_start() {
        let d = distance_to_edge([3.0, 4.0], [0.0, 0.0], [0.0, 0.0]);
        assert!(approx(d, 5.0));
    }

    #[test]
    fn distance_to_edge_on_segment_is_zero() {
        let d = distance_to_edge([1.0, 0.0], [0.0, 0.0], [2.0, 0.0]);
        assert!(approx(d, 0.0));
    }

    #[test]
    fn signed_distance_inside_is_negative() {
        // Centre of the square is 1 unit from every edge.
        let d = signed_distance_to_polygon([1.0, 1.0], &unit_square());
        assert!(d < 0.0);
        assert!(approx(d, -1.0));
    }

    #[test]
    fn signed_distance_outside_is_positive() {
        let d = signed_distance_to_polygon([4.0, 1.0], &unit_square());
        assert!(d > 0.0);
        assert!(approx(d, 2.0));
    }

    #[test]
    fn signed_distance_magnitude_is_nearest_edge() {
        // Point just left of the left edge at x = 0.
        let d = signed_distance_to_polygon([-0.5, 1.0], &unit_square());
        assert!(approx(d, 0.5));
    }

    #[test]
    fn signed_distance_on_boundary_is_near_zero() {
        let d = signed_distance_to_polygon([1.0, 0.0], &unit_square());
        assert!(approx(d, 0.0));
    }

    #[test]
    fn signed_distance_empty_polygon_is_infinite() {
        assert!(signed_distance_to_polygon([0.0, 0.0], &[]).is_infinite());
    }

    #[test]
    fn gpu_storage_bytes_scales_with_vertices() {
        assert_eq!(gpu_storage_bytes(4), storage_bytes(VEC2_STRIDE, 4));
        assert_eq!(gpu_storage_bytes(4), 32);
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one_element() {
        assert_eq!(gpu_storage_bytes(0), VEC2_STRIDE);
    }

    #[test]
    fn is_left_sign_classifies_turn() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        assert!(is_left(a, b, [0.5, 1.0]) > 0.0);
        assert!(is_left(a, b, [0.5, -1.0]) < 0.0);
        assert!(approx(is_left(a, b, [0.5, 0.0]), 0.0));
    }
}
