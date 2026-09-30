//! 2D simple-polygon *metrics* for the particle spatial contracts
//! (design §8.2, §12-§13).
//!
//! Several particle stages describe a shape as an ordered ring of 2D vertices
//! and then need scalar facts about that ring: a bounds-reduction pass wants
//! the enclosed `area` and `centroid` of an emitter footprint; a collision
//! proxy wants the outward edge normals of a convex obstacle; a culling
//! reference wants the ring's `perimeter` and axis-aligned extent. This module
//! owns the small, `CPU`-verifiable contract those stages share: the signed
//! shoelace `area`, the derived absolute area and winding sense, the
//! area-weighted centroid, the closed perimeter, a convexity predicate, the
//! axis-aligned bounding box, and the per-edge outward unit normals.
//!
//! A polygon is passed as an *open* ring `&[[f32; 2]]`: the closing edge from
//! the last vertex back to the first is implied and never repeated in the
//! slice. Winding is a first-class output — [`signed_area`] is positive for a
//! `CCW` (counter-clockwise) ring and negative for a `CW` (clockwise) ring —
//! and the outward-normal contract keys off that sense so the same code serves
//! either authoring convention.
//!
//! # Strict scope
//! This module only *measures* a ring the caller already has. It deliberately
//! does not *construct* a hull ([`super::convex_hull_2d`]), test point
//! containment (`point_in_polygon`), or triangulate
//! ([`super::ear_clip_triangulate`]); it neither imports nor reconstructs those
//! contracts, and its private helpers are not shared outward.
//!
//! # No transcendental math
//! Every metric is pure `+`, `-`, `*` shoelace and cross-product arithmetic;
//! the only irrational operation is the edge-length `f32::sqrt` used by
//! [`perimeter`] and [`edge_normals_outward`]. There is no `sin`, `cos`,
//! `atan`, `exp`, `ln`, `powf` or any other transcendental call, and no `f32`
//! equality: near-zero magnitudes are compared against [`CMP_EPS`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE};

/// Magnitude below which an area, an edge length, or a coordinate difference is
/// treated as zero.
///
/// This is the comparison rule used throughout instead of `==` on `f32`: a
/// scalar is "zero" when its absolute value does not exceed this bound, which
/// is how degenerate (collinear / zero-length) inputs are detected without ever
/// comparing two floats for exact equality.
pub const CMP_EPS: f32 = 1.0e-6;

/// Signed area of a simple polygon via the shoelace formula, positive for a
/// `CCW` (counter-clockwise) ring and negative for a `CW` (clockwise) ring.
///
/// The ring is treated as closed: the edge from the last vertex back to the
/// first is included automatically. Fewer than three vertices enclose no area,
/// so the result is `0.0`.
#[must_use]
pub fn signed_area(polygon: &[[f32; 2]]) -> f32 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % n];
        acc += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * acc
}

/// Absolute (unsigned) area enclosed by the polygon ring.
///
/// This is `signed_area(polygon).abs()`, so it is winding-independent and never
/// negative.
#[must_use]
pub fn area(polygon: &[[f32; 2]]) -> f32 {
    signed_area(polygon).abs()
}

/// Returns `true` when the ring winds `CCW` (counter-clockwise), i.e. its
/// [`signed_area`] is strictly positive.
///
/// A degenerate ring (fewer than three vertices, or a near-zero signed area
/// within [`CMP_EPS`]) has no well-defined orientation and reports `false`.
#[must_use]
pub fn winding_is_ccw(polygon: &[[f32; 2]]) -> bool {
    signed_area(polygon) > CMP_EPS
}

/// Area-weighted centroid (the polygon's center of mass) of the ring.
///
/// The centroid is the shoelace-weighted average of the edge midpoints divided
/// by six times the signed area. When that area is degenerate (its magnitude
/// does not exceed [`CMP_EPS`], as for a collinear or zero-area ring) the
/// area-weighted formula would divide by zero, so the result falls back to the
/// plain arithmetic mean of the vertices. An empty slice yields the origin and
/// a single vertex yields itself.
#[must_use]
pub fn centroid(polygon: &[[f32; 2]]) -> [f32; 2] {
    let n = polygon.len();
    if n == 0 {
        return [0.0, 0.0];
    }
    if n == 1 {
        return polygon[0];
    }

    let signed = signed_area(polygon);
    if signed.abs() <= CMP_EPS {
        return vertex_mean(polygon);
    }

    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % n];
        let w = p[0] * q[1] - q[0] * p[1];
        cx += (p[0] + q[0]) * w;
        cy += (p[1] + q[1]) * w;
    }
    let denom = 6.0 * signed;
    [cx / denom, cy / denom]
}

/// Plain arithmetic mean of the vertices, used as the centroid fallback for a
/// degenerate (zero-area) ring.
fn vertex_mean(polygon: &[[f32; 2]]) -> [f32; 2] {
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    for &p in polygon {
        sx += p[0];
        sy += p[1];
    }
    let inv = 1.0 / (polygon.len() as f32);
    [sx * inv, sy * inv]
}

/// Closed perimeter: the summed Euclidean length of every edge, including the
/// implied closing edge from the last vertex back to the first.
///
/// Fewer than two vertices bound no edge, so the result is `0.0`.
#[must_use]
pub fn perimeter(polygon: &[[f32; 2]]) -> f32 {
    let n = polygon.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % n];
        let dx = q[0] - p[0];
        let dy = q[1] - p[1];
        total += (dx * dx + dy * dy).sqrt();
    }
    total
}

/// Returns `true` when the ring is convex: every turn between consecutive edges
/// has the same orientation.
///
/// Each vertex contributes the cross product of its incoming and outgoing edge;
/// collinear vertices (cross product within [`CMP_EPS`]) are skipped, and the
/// ring is convex when all remaining turns share one sign. Fewer than three
/// vertices do not form a polygon and report `false`.
#[must_use]
pub fn is_convex(polygon: &[[f32; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut sign: i32 = 0;
    for (i, &cur) in polygon.iter().enumerate() {
        let prev = polygon[(i + n - 1) % n];
        let next = polygon[(i + 1) % n];
        let ax = cur[0] - prev[0];
        let ay = cur[1] - prev[1];
        let bx = next[0] - cur[0];
        let by = next[1] - cur[1];
        let turn = ax * by - ay * bx;
        if turn > CMP_EPS {
            if sign < 0 {
                return false;
            }
            sign = 1;
        } else if turn < -CMP_EPS {
            if sign > 0 {
                return false;
            }
            sign = -1;
        }
    }
    true
}

/// Axis-aligned bounding box of the ring as `(min_corner, max_corner)`.
///
/// The minimum corner holds the smallest `x` and `y` found across all vertices
/// and the maximum corner the largest. An empty slice has no extent and yields
/// two origin corners.
#[must_use]
pub fn bounding_box(polygon: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let Some((&first, rest)) = polygon.split_first() else {
        return ([0.0, 0.0], [0.0, 0.0]);
    };
    let mut min = first;
    let mut max = first;
    for &p in rest {
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    (min, max)
}

/// Outward unit normal of every edge, one per vertex, ordered to match the
/// ring's edges (edge `i` runs from vertex `i` to vertex `i + 1`, wrapping).
///
/// The outward direction is chosen from the ring's winding: for a `CCW`
/// (counter-clockwise) ring the outward normal is the right-hand perpendicular
/// of the edge direction, and for a `CW` (clockwise) ring it is the left-hand
/// perpendicular; a degenerate winding defaults to the `CCW` convention. Each
/// normal is normalized to unit length, and a zero-length edge (within
/// [`CMP_EPS`]) has no direction, so its normal is the zero vector rather than
/// a division by zero. Fewer than two vertices bound no edge and yield an empty
/// `Vec`.
#[must_use]
pub fn edge_normals_outward(polygon: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = polygon.len();
    if n < 2 {
        return Vec::new();
    }
    let ccw = winding_is_ccw(polygon) || signed_area(polygon).abs() <= CMP_EPS;

    let mut normals = Vec::with_capacity(n);
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % n];
        let dx = q[0] - p[0];
        let dy = q[1] - p[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len <= CMP_EPS {
            normals.push([0.0, 0.0]);
            continue;
        }
        // Right-hand perpendicular (dy, -dx) points outward for a CCW ring; the
        // left-hand perpendicular (-dy, dx) does so for a CW ring.
        let (nx, ny) = if ccw { (dy, -dx) } else { (-dy, dx) };
        normals.push([nx / len, ny / len]);
    }
    normals
}

/// Total `std430` storage byte size of a `GPU` vertex buffer holding
/// `vertex_count` `vec2<f32>` polygon vertices.
///
/// This defers to the shared [`storage_bytes`] rule, so an empty ring still
/// reserves one non-empty `WebGPU` binding and the multiplication saturates
/// instead of overflowing.
#[must_use]
pub fn gpu_storage_bytes(vertex_count: usize) -> usize {
    storage_bytes(VEC2_STRIDE, vertex_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_EPS: f32 = 1.0e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    fn close2(a: [f32; 2], b: [f32; 2]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1])
    }

    const CCW_SQUARE: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    const CW_SQUARE: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];

    #[test]
    fn signed_area_ccw_square_is_positive_unit() {
        assert!(close(signed_area(&CCW_SQUARE), 1.0));
    }

    #[test]
    fn signed_area_cw_square_is_negative_unit() {
        assert!(close(signed_area(&CW_SQUARE), -1.0));
    }

    #[test]
    fn signed_area_triangle() {
        let tri = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        assert!(close(signed_area(&tri), 6.0));
    }

    #[test]
    fn signed_area_too_few_vertices_is_zero() {
        assert!(close(signed_area(&[]), 0.0));
        assert!(close(signed_area(&[[1.0, 2.0]]), 0.0));
        assert!(close(signed_area(&[[0.0, 0.0], [1.0, 1.0]]), 0.0));
    }

    #[test]
    fn area_is_absolute_regardless_of_winding() {
        assert!(close(area(&CCW_SQUARE), 1.0));
        assert!(close(area(&CW_SQUARE), 1.0));
    }

    #[test]
    fn area_of_scaled_square() {
        let sq = [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]];
        assert!(close(area(&sq), 9.0));
    }

    #[test]
    fn winding_ccw_is_true_for_ccw_ring() {
        assert!(winding_is_ccw(&CCW_SQUARE));
    }

    #[test]
    fn winding_ccw_is_false_for_cw_ring() {
        assert!(!winding_is_ccw(&CW_SQUARE));
    }

    #[test]
    fn winding_ccw_is_false_for_degenerate_ring() {
        let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        assert!(!winding_is_ccw(&line));
        assert!(!winding_is_ccw(&[]));
    }

    #[test]
    fn centroid_of_unit_square_is_center() {
        assert!(close2(centroid(&CCW_SQUARE), [0.5, 0.5]));
    }

    #[test]
    fn centroid_of_cw_square_is_still_center() {
        assert!(close2(centroid(&CW_SQUARE), [0.5, 0.5]));
    }

    #[test]
    fn centroid_of_triangle() {
        let tri = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
        assert!(close2(centroid(&tri), [4.0 / 3.0, 1.0]));
    }

    #[test]
    fn centroid_degenerate_falls_back_to_vertex_mean() {
        let line = [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]];
        assert!(close2(centroid(&line), [2.0, 0.0]));
    }

    #[test]
    fn centroid_empty_is_origin() {
        assert!(close2(centroid(&[]), [0.0, 0.0]));
    }

    #[test]
    fn centroid_single_vertex_is_itself() {
        assert!(close2(centroid(&[[3.0, -7.0]]), [3.0, -7.0]));
    }

    #[test]
    fn perimeter_of_unit_square_is_four() {
        assert!(close(perimeter(&CCW_SQUARE), 4.0));
    }

    #[test]
    fn perimeter_of_345_triangle_is_twelve() {
        let tri = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]];
        assert!(close(perimeter(&tri), 12.0));
    }

    #[test]
    fn perimeter_too_few_vertices_is_zero() {
        assert!(close(perimeter(&[]), 0.0));
        assert!(close(perimeter(&[[1.0, 1.0]]), 0.0));
    }

    #[test]
    fn perimeter_of_segment_counts_both_directions() {
        // Two vertices form a degenerate ring: out and back, so 2 x length.
        assert!(close(perimeter(&[[0.0, 0.0], [3.0, 4.0]]), 10.0));
    }

    #[test]
    fn is_convex_true_for_square() {
        assert!(is_convex(&CCW_SQUARE));
        assert!(is_convex(&CW_SQUARE));
    }

    #[test]
    fn is_convex_true_with_collinear_edge_vertex() {
        // A midpoint on the bottom edge is collinear, not a reflex turn.
        let sq = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(is_convex(&sq));
    }

    #[test]
    fn is_convex_false_for_concave_ring() {
        // An arrowhead / dart with one reflex vertex.
        let dart = [[0.0, 0.0], [2.0, 1.0], [0.0, 2.0], [0.5, 1.0]];
        assert!(!is_convex(&dart));
    }

    #[test]
    fn is_convex_false_for_too_few_vertices() {
        assert!(!is_convex(&[[0.0, 0.0], [1.0, 0.0]]));
        assert!(!is_convex(&[]));
    }

    #[test]
    fn bounding_box_covers_all_vertices() {
        let poly = [[-1.0, 2.0], [3.0, -4.0], [0.0, 5.0]];
        let (min, max) = bounding_box(&poly);
        assert!(close2(min, [-1.0, -4.0]));
        assert!(close2(max, [3.0, 5.0]));
    }

    #[test]
    fn bounding_box_empty_is_origin_pair() {
        let (min, max) = bounding_box(&[]);
        assert!(close2(min, [0.0, 0.0]));
        assert!(close2(max, [0.0, 0.0]));
    }

    #[test]
    fn bounding_box_single_vertex_is_degenerate_point() {
        let (min, max) = bounding_box(&[[2.0, 3.0]]);
        assert!(close2(min, [2.0, 3.0]));
        assert!(close2(max, [2.0, 3.0]));
    }

    #[test]
    fn edge_normals_point_outward_for_ccw_square() {
        let normals = edge_normals_outward(&CCW_SQUARE);
        assert_eq!(normals.len(), 4);
        assert!(close2(normals[0], [0.0, -1.0]));
        assert!(close2(normals[1], [1.0, 0.0]));
        assert!(close2(normals[2], [0.0, 1.0]));
        assert!(close2(normals[3], [-1.0, 0.0]));
    }

    #[test]
    fn edge_normals_point_outward_for_cw_square() {
        let normals = edge_normals_outward(&CW_SQUARE);
        assert_eq!(normals.len(), 4);
        assert!(close2(normals[0], [-1.0, 0.0]));
        assert!(close2(normals[1], [0.0, 1.0]));
        assert!(close2(normals[2], [1.0, 0.0]));
        assert!(close2(normals[3], [0.0, -1.0]));
    }

    #[test]
    fn edge_normals_are_unit_length() {
        let poly = [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]];
        for nrm in edge_normals_outward(&poly) {
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1]).sqrt();
            assert!(close(len, 1.0));
        }
    }

    #[test]
    fn edge_normals_zero_length_edge_is_zero_vector() {
        // A repeated vertex creates a zero-length edge with no direction.
        let poly = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let normals = edge_normals_outward(&poly);
        assert_eq!(normals.len(), 5);
        assert!(close2(normals[0], [0.0, 0.0]));
    }

    #[test]
    fn edge_normals_too_few_vertices_is_empty() {
        assert!(edge_normals_outward(&[]).is_empty());
        assert!(edge_normals_outward(&[[1.0, 1.0]]).is_empty());
    }

    #[test]
    fn gpu_storage_bytes_reuses_shared_rule() {
        assert_eq!(gpu_storage_bytes(0), VEC2_STRIDE);
        assert_eq!(gpu_storage_bytes(4), 4 * VEC2_STRIDE);
        assert_eq!(gpu_storage_bytes(3), 24);
    }
}
