//! 2D convex-hull construction and hull metrics for the particle spatial
//! contracts (design §8.2, §12-§13).
//!
//! Several particle stages need the tight convex boundary of a point set: a
//! bounds-reduction pass wants the smallest enclosing polygon of a splat's
//! footprint; a collision-broadphase wants a cheap separating-axis proxy; and a
//! screen-space culling reference wants the silhouette of a projected cluster.
//! This module owns the small, `CPU`-verifiable contract those stages share:
//! turning an unordered slice of 2D points into their convex hull, expressed as
//! a `CCW` (counter-clockwise) vertex ring, plus the handful of scalar metrics
//! (area, perimeter, squared diameter, convexity) that describe that hull.
//!
//! The hull is built with Andrew's monotone chain: sort the points by
//! (`x` ascending, then `y` ascending), sweep once to build the lower chain and
//! once (in reverse) to build the upper chain, popping any vertex that does not
//! make a strict left turn. Collinear interior points are discarded, so the
//! result is a minimal `CCW` ring with no redundant vertices.
//!
//! # Strict scope
//! This module only *constructs* the convex hull and measures the hull it
//! built (area, perimeter, squared diameter, convexity). It deliberately does
//! not compute general polygon area/centroid/winding metrics ([`super`]'s
//! `polygon_area_2d` sibling), test point containment (`point_in_polygon`),
//! triangulate (`ear_clip_triangulate`), or solve barycentric coordinates
//! ([`super::barycentric_coord`]); it neither imports nor reconstructs those
//! contracts, and it keeps its own math helpers rather than sharing them.
//!
//! # No transcendental math
//! Hull construction and its metrics are pure `+`, `-`, `*` cross-product and
//! comparison arithmetic; the only irrational operation is the edge-length
//! `f32::sqrt` used by [`hull_perimeter`]. There is no `sin`, `cos`, `atan`,
//! `exp`, `ln`, `powf` or any other transcendental call, and no `f32`
//! equality: near-equal magnitudes are compared against [`CMP_EPS`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which a cross product, a coordinate difference, or a turn is
/// treated as zero. This is the comparison rule used throughout instead of `==`
/// on `f32`: two scalars are "equal" when their absolute difference does not
/// exceed this bound, and a turn is "left" only when its cross product exceeds
/// it.
pub const CMP_EPS: f32 = 1.0e-6;

/// Twice the signed area of triangle `o, a, b`, i.e. the 2D cross product
/// `(a - o) x (b - o)`.
///
/// The sign classifies the turn `o -> a -> b`: strictly positive for a
/// counter-clockwise (left) turn, strictly negative for a clockwise (right)
/// turn, and zero (within [`CMP_EPS`]) when the three points are collinear.
#[must_use]
pub fn cross2(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Returns `true` when two points coincide within [`CMP_EPS`] on both axes.
fn points_equal(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() <= CMP_EPS && (a[1] - b[1]).abs() <= CMP_EPS
}

/// Euclidean distance between two 2D points.
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Builds the convex hull of `points` as a `CCW` (counter-clockwise) vertex
/// ring using Andrew's monotone chain.
///
/// The points are first sorted by (`x` ascending, then `y` ascending) using
/// [`f32::total_cmp`] so `NaN`/`-0.0` never break the ordering, and exact
/// duplicates are collapsed. The lower and upper chains are then swept, popping
/// any vertex whose turn is not a strict left turn (cross product not exceeding
/// [`CMP_EPS`]); this discards collinear interior points, so the ring is
/// minimal.
///
/// Degenerate inputs are handled explicitly:
/// * an empty slice returns an empty `Vec`;
/// * a single distinct point returns that point;
/// * two distinct points return both (a segment);
/// * three-or-more collinear points return only the two extreme endpoints.
///
/// The returned ring is *open*: the first vertex is not repeated at the end.
#[must_use]
pub fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|a, b| points_equal(*a, *b));

    // After deduplication a segment or a single surviving point is already the
    // hull; the chain sweep below needs at least three distinct points.
    if pts.len() <= 2 {
        return pts;
    }

    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2
            && cross2(lower[lower.len() - 2], lower[lower.len() - 1], p) <= CMP_EPS
        {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2
            && cross2(upper[upper.len() - 2], upper[upper.len() - 1], p) <= CMP_EPS
        {
            upper.pop();
        }
        upper.push(p);
    }

    // Drop each chain's last vertex: it is the shared endpoint duplicated by the
    // other chain's first vertex.
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Area enclosed by a hull ring via the shoelace formula.
///
/// The ring is treated as closed (the last vertex connects back to the first).
/// A degenerate ring of fewer than three vertices encloses no area and returns
/// `0.0`. The result is the absolute value, so it is winding-agnostic.
#[must_use]
pub fn hull_area(hull: &[[f32; 2]]) -> f32 {
    let n = hull.len();
    if n < 3 {
        return 0.0;
    }
    let mut twice_signed = 0.0f32;
    for (i, &cur) in hull.iter().enumerate() {
        let next = hull[(i + 1) % n];
        twice_signed += cur[0] * next[1] - next[0] * cur[1];
    }
    (twice_signed * 0.5).abs()
}

/// Perimeter of a hull ring: the sum of its closed edge lengths.
///
/// A single point (or empty ring) has zero perimeter; a two-vertex ring is a
/// segment and its perimeter is the single edge length (not the closed
/// there-and-back traversal). Rings of three or more vertices are summed as a
/// closed loop.
#[must_use]
pub fn hull_perimeter(hull: &[[f32; 2]]) -> f32 {
    let n = hull.len();
    if n < 2 {
        return 0.0;
    }
    if n == 2 {
        return distance(hull[0], hull[1]);
    }
    let mut sum = 0.0f32;
    for (i, &cur) in hull.iter().enumerate() {
        let next = hull[(i + 1) % n];
        sum += distance(cur, next);
    }
    sum
}

/// Returns `true` when `poly` is a convex polygon wound counter-clockwise.
///
/// Every consecutive turn must be a left turn or collinear: the cross product
/// at each vertex must be at least `-`[`CMP_EPS`]. A polygon needs at least
/// three vertices, so a shorter ring is never convex and returns `false`.
#[must_use]
pub fn is_convex_ccw(poly: &[[f32; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    for (i, &o) in poly.iter().enumerate() {
        let a = poly[(i + 1) % n];
        let b = poly[(i + 2) % n];
        if cross2(o, a, b) < -CMP_EPS {
            return false;
        }
    }
    true
}

/// Squared diameter of a hull: the largest squared distance between any two of
/// its vertices.
///
/// This is the simple O(n^2) all-pairs scan, which is exact and adequate for
/// the small vertex counts a particle-cluster hull produces. Working in squared
/// distance avoids an unnecessary `sqrt`. Fewer than two vertices have zero
/// diameter.
#[must_use]
pub fn diameter_squared(hull: &[[f32; 2]]) -> f32 {
    let mut best = 0.0f32;
    for (i, &p) in hull.iter().enumerate() {
        for &q in &hull[i + 1..] {
            let dx = p[0] - q[0];
            let dy = p[1] - q[1];
            best = best.max(dx * dx + dy * dy);
        }
    }
    best
}

/// Byte size of the `std430` `GPU` storage buffer holding `point_count` hull
/// vertices, one `vec2<f32>` each.
///
/// Reuses [`storage_bytes`], so an empty hull still reserves one element and
/// yields a valid (non-zero-sized) `WebGPU` binding.
#[must_use]
pub fn gpu_storage_bytes(point_count: usize) -> usize {
    storage_bytes(VEC2_STRIDE, point_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    fn pt_approx(a: [f32; 2], b: [f32; 2]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1])
    }

    fn hull_matches(got: &[[f32; 2]], expected: &[[f32; 2]]) -> bool {
        got.len() == expected.len()
            && got
                .iter()
                .zip(expected.iter())
                .all(|(&g, &e)| pt_approx(g, e))
    }

    #[test]
    fn cross2_is_positive_for_left_turn() {
        let c = cross2([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
        assert!(c > CMP_EPS);
    }

    #[test]
    fn cross2_is_negative_for_right_turn() {
        let c = cross2([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]);
        assert!(c < -CMP_EPS);
    }

    #[test]
    fn cross2_is_zero_for_collinear() {
        let c = cross2([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]);
        assert!(c.abs() <= CMP_EPS);
    }

    #[test]
    fn empty_input_yields_empty_hull() {
        let hull = convex_hull(&[]);
        assert!(hull.is_empty());
    }

    #[test]
    fn single_point_yields_single_point() {
        let hull = convex_hull(&[[3.0, 4.0]]);
        assert!(hull_matches(&hull, &[[3.0, 4.0]]));
    }

    #[test]
    fn two_distinct_points_yield_segment() {
        let hull = convex_hull(&[[1.0, 1.0], [4.0, 5.0]]);
        assert_eq!(hull.len(), 2);
        assert!(hull.iter().any(|&p| pt_approx(p, [1.0, 1.0])));
        assert!(hull.iter().any(|&p| pt_approx(p, [4.0, 5.0])));
    }

    #[test]
    fn duplicate_pair_collapses_to_single_point() {
        let hull = convex_hull(&[[2.0, 2.0], [2.0, 2.0]]);
        assert!(hull_matches(&hull, &[[2.0, 2.0]]));
    }

    #[test]
    fn unit_square_returns_four_ccw_corners() {
        let hull = convex_hull(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(hull_matches(
            &hull,
            &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ));
    }

    #[test]
    fn interior_points_are_excluded() {
        let hull = convex_hull(&[
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ]);
        assert!(hull_matches(
            &hull,
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        ));
    }

    #[test]
    fn boundary_collinear_points_are_excluded() {
        // Midpoints of each square edge lie on the hull boundary but are not
        // vertices; the minimal ring must drop them.
        let hull = convex_hull(&[
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [4.0, 4.0],
            [2.0, 4.0],
            [0.0, 4.0],
            [0.0, 2.0],
        ]);
        assert!(hull_matches(
            &hull,
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        ));
    }

    #[test]
    fn triangle_returns_three_vertices() {
        let hull = convex_hull(&[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]);
        assert!(hull_matches(&hull, &[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]));
    }

    #[test]
    fn all_collinear_points_return_endpoint_pair() {
        let hull = convex_hull(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
        assert_eq!(hull.len(), 2);
        assert!(hull.iter().any(|&p| pt_approx(p, [0.0, 0.0])));
        assert!(hull.iter().any(|&p| pt_approx(p, [3.0, 3.0])));
    }

    #[test]
    fn collinear_with_duplicates_returns_endpoint_pair() {
        let hull = convex_hull(&[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0]]);
        assert_eq!(hull.len(), 2);
        assert!(hull.iter().any(|&p| pt_approx(p, [0.0, 0.0])));
        assert!(hull.iter().any(|&p| pt_approx(p, [2.0, 0.0])));
    }

    #[test]
    fn concave_point_is_removed_from_hull() {
        // A square with a fifth point pushed inward; the dent must not appear.
        let hull = convex_hull(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 1.0]]);
        assert!(hull_matches(
            &hull,
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
        ));
    }

    #[test]
    fn duplicate_square_corners_still_give_four_vertices() {
        let hull = convex_hull(&[
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]);
        assert!(hull_matches(
            &hull,
            &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ));
    }

    #[test]
    fn hull_output_is_convex_ccw() {
        let hull = convex_hull(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 2.0]]);
        assert!(is_convex_ccw(&hull));
    }

    #[test]
    fn unordered_cloud_produces_ccw_convex_ring() {
        let hull = convex_hull(&[
            [2.0, 5.0],
            [5.0, 1.0],
            [1.0, 1.0],
            [4.0, 4.0],
            [3.0, 2.0],
            [0.0, 3.0],
        ]);
        assert!(is_convex_ccw(&hull));
        // Every input point must be inside or on the reported hull area.
        assert!(hull_area(&hull) > 0.0);
    }

    #[test]
    fn hull_area_of_unit_square_is_one() {
        let hull = convex_hull(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(approx(hull_area(&hull), 1.0));
    }

    #[test]
    fn hull_area_of_right_triangle_is_half_base_times_height() {
        let hull = convex_hull(&[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]);
        assert!(approx(hull_area(&hull), 6.0));
    }

    #[test]
    fn hull_area_of_degenerate_ring_is_zero() {
        assert!(approx(hull_area(&[]), 0.0));
        assert!(approx(hull_area(&[[1.0, 1.0]]), 0.0));
        assert!(approx(hull_area(&[[0.0, 0.0], [1.0, 0.0]]), 0.0));
    }

    #[test]
    fn hull_perimeter_of_unit_square_is_four() {
        let hull = convex_hull(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(approx(hull_perimeter(&hull), 4.0));
    }

    #[test]
    fn hull_perimeter_of_3_4_5_triangle_is_twelve() {
        let hull = convex_hull(&[[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]);
        assert!(approx(hull_perimeter(&hull), 12.0));
    }

    #[test]
    fn hull_perimeter_of_segment_is_single_edge_length() {
        assert!(approx(hull_perimeter(&[[0.0, 0.0], [3.0, 4.0]]), 5.0));
    }

    #[test]
    fn hull_perimeter_of_point_or_empty_is_zero() {
        assert!(approx(hull_perimeter(&[]), 0.0));
        assert!(approx(hull_perimeter(&[[7.0, 7.0]]), 0.0));
    }

    #[test]
    fn is_convex_ccw_rejects_clockwise_ring() {
        // The unit square wound clockwise.
        let cw = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        assert!(!is_convex_ccw(&cw));
    }

    #[test]
    fn is_convex_ccw_rejects_concave_ring() {
        // An arrowhead: the third vertex dents inward.
        let concave = [[0.0, 0.0], [4.0, 0.0], [2.0, 1.0], [4.0, 4.0], [0.0, 4.0]];
        assert!(!is_convex_ccw(&concave));
    }

    #[test]
    fn is_convex_ccw_rejects_short_rings() {
        assert!(!is_convex_ccw(&[]));
        assert!(!is_convex_ccw(&[[0.0, 0.0]]));
        assert!(!is_convex_ccw(&[[0.0, 0.0], [1.0, 0.0]]));
    }

    #[test]
    fn diameter_squared_of_unit_square_is_two() {
        let hull = convex_hull(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(approx(diameter_squared(&hull), 2.0));
    }

    #[test]
    fn diameter_squared_of_collinear_pair_is_length_squared() {
        let hull = convex_hull(&[[0.0, 0.0], [3.0, 4.0]]);
        assert!(approx(diameter_squared(&hull), 25.0));
    }

    #[test]
    fn diameter_squared_of_single_or_empty_is_zero() {
        assert!(approx(diameter_squared(&[]), 0.0));
        assert!(approx(diameter_squared(&[[9.0, 9.0]]), 0.0));
    }

    #[test]
    fn gpu_storage_bytes_clamps_empty_and_scales_linearly() {
        assert_eq!(gpu_storage_bytes(0), VEC2_STRIDE);
        assert_eq!(gpu_storage_bytes(1), VEC2_STRIDE);
        assert_eq!(gpu_storage_bytes(8), VEC2_STRIDE * 8);
    }
}
