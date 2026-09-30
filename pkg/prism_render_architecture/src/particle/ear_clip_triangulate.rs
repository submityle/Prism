//! Ear-clipping triangulation of 2D simple polygons (design §14, §24).
//!
//! A great many particle-authoring tasks reduce to *fan-free* triangulation of
//! a flat outline: turning a ribbon cross-section, a decal footprint, a
//! text-glyph contour, or a hand-drawn spawn region into the index-buffer
//! triangle soup a `GPU` draw call consumes. This module implements the classic
//! ear-clipping algorithm for a single, non-self-intersecting (simple) polygon:
//! it repeatedly locates a convex "ear" vertex whose triangle contains no other
//! vertex, emits that triangle, and removes the ear until three vertices
//! remain. A simple polygon of `n` vertices always yields exactly `n - 2`
//! triangles, so the output index count is fixed and predictable for buffer
//! sizing.
//!
//! # Strict scope
//! This module owns *only* triangulation. Sibling contracts own the neighboring
//! geometry primitives and this module never reaches into them: rasterization
//! coverage lives in `conservative_raster`, barycentric interpolation in
//! `barycentric_coord`, the convex-hull construction in `convex_hull_2d`, and
//! generic point-in-polygon testing in `point_in_polygon`. The
//! [`point_in_triangle`] routine here is a private helper of the ear test and
//! is intentionally *not* shared, so the contracts stay independently
//! verifiable.
//!
//! # Winding
//! Input may be wound clockwise (`CW`) or counter-clockwise (`CCW`); the
//! triangulator detects the signed area and walks the index ring in whichever
//! direction makes the interior consistently `CCW`, so emitted triangles are
//! `CCW`-wound regardless of input orientation. Output triples index into the
//! *original* polygon vertex array.
//!
//! # No transcendental math
//! Every routine is polynomial cross-product arithmetic; there is no
//! trigonometry anywhere. Float comparisons use an explicit epsilon rather than
//! exact equality so nearly-collinear vertices are handled as degenerate ears
//! instead of producing a `NaN` or stalling the clip loop.

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE};
use alloc::vec::Vec;

/// Epsilon used for tolerant float comparisons instead of exact `==`/`!=`.
///
/// A cross product whose magnitude falls at or below this threshold is treated
/// as zero (collinear), and a point that lies within this tolerance of a
/// triangle edge is treated as on the boundary.
const CMP_EPS: f32 = 1e-6;

/// Twice the signed area of triangle `(a, b, c)` (the `z` component of the edge
/// cross product).
///
/// Positive for a `CCW` triple, negative for a `CW` triple, and near zero when
/// the three points are collinear.
#[must_use]
fn tri_cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    let ex = b[0] - a[0];
    let ey = b[1] - a[1];
    let fx = c[0] - b[0];
    let fy = c[1] - b[1];
    ex * fy - ey * fx
}

/// The side of directed edge `a -> b` that point `p` falls on.
///
/// Positive to the left, negative to the right, near zero when `p` is on the
/// line through `a` and `b`.
#[must_use]
fn edge_side(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// The signed area of the polygon via the shoelace formula.
///
/// Positive when the vertices are wound `CCW`, negative when `CW`. Returns zero
/// for a degenerate polygon of fewer than three vertices.
#[must_use]
pub fn signed_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (i, p) in poly.iter().enumerate() {
        let q = poly[(i + 1) % n];
        sum += p[0] * q[1] - q[0] * p[1];
    }
    sum * 0.5
}

/// Returns `true` when the polygon is wound counter-clockwise (`CCW`).
///
/// A degenerate polygon (fewer than three vertices, or zero area) reports
/// `false`, matching the "not positively oriented" convention.
#[must_use]
pub fn is_ccw(poly: &[[f32; 2]]) -> bool {
    signed_area(poly) > CMP_EPS
}

/// Returns `true` when point `p` lies inside triangle `(a, b, c)` or on its
/// boundary, within [`CMP_EPS`].
///
/// The test is orientation-agnostic: `p` is inside when it is on a consistent
/// side of all three edges, allowing a small epsilon slack so exact-boundary
/// points count as contained. This is a private helper of the ear test and is
/// deliberately not shared with the point-in-polygon contract.
#[must_use]
pub fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = edge_side(p, a, b);
    let d2 = edge_side(p, b, c);
    let d3 = edge_side(p, c, a);
    let has_neg = d1 < -CMP_EPS || d2 < -CMP_EPS || d3 < -CMP_EPS;
    let has_pos = d1 > CMP_EPS || d2 > CMP_EPS || d3 > CMP_EPS;
    !(has_neg && has_pos)
}

/// Returns `true` when the vertex `cur`, with polygon neighbors `prev` and
/// `next`, is strictly convex for the given winding.
///
/// For a `CCW` polygon (`ccw == true`) a convex vertex turns left (positive
/// cross product); for a `CW` polygon it turns right. Collinear vertices
/// (cross magnitude at or below [`CMP_EPS`]) are neither convex nor reflex and
/// report `false`.
#[must_use]
pub fn is_convex_vertex(prev: [f32; 2], cur: [f32; 2], next: [f32; 2], ccw: bool) -> bool {
    let cross = tri_cross(prev, cur, next);
    if ccw {
        cross > CMP_EPS
    } else {
        cross < -CMP_EPS
    }
}

/// Returns `true` when the vertex at ring position `i` is an ear: it is a
/// convex vertex and the triangle it forms with its immediate neighbors
/// contains no other polygon vertex.
///
/// `indices` is the current index ring into `poly`; `i` is a position within
/// that ring; `ccw` states the ring's winding. A ring of fewer than three
/// entries can have no ear and reports `false`.
#[must_use]
pub fn is_ear(poly: &[[f32; 2]], indices: &[usize], i: usize, ccw: bool) -> bool {
    let m = indices.len();
    if m < 3 {
        return false;
    }
    let ip = (i + m - 1) % m;
    let inx = (i + 1) % m;
    let a = poly[indices[ip]];
    let b = poly[indices[i]];
    let c = poly[indices[inx]];
    if !is_convex_vertex(a, b, c, ccw) {
        return false;
    }
    for (k, &vi) in indices.iter().enumerate() {
        if k == ip || k == i || k == inx {
            continue;
        }
        if point_in_triangle(poly[vi], a, b, c) {
            return false;
        }
    }
    true
}

/// Triangulates a 2D simple polygon by ear clipping.
///
/// Returns triangles as triples of indices into the *original* `polygon`
/// vertex array, `CCW`-wound regardless of input orientation. A simple polygon
/// of `n >= 3` vertices produces exactly `n - 2` triangles; degenerate input
/// (`n < 3`) produces an empty result.
///
/// The clip loop is guaranteed to terminate: each iteration removes exactly one
/// vertex from the working ring. Collinear (degenerate) vertices are clipped as
/// zero-area ears so they do not stall the loop, and if no strict ear is found
/// in a pass, the loop falls back to clipping the first remaining vertex so a
/// mildly non-simple input can never spin forever.
#[must_use]
pub fn triangulate(polygon: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    if n < 3 {
        return triangles;
    }

    let mut ring: Vec<usize> = (0..n).collect();
    // Walk the ring so the interior is consistently CCW; reversing the index
    // order flips a CW input without touching the vertex coordinates.
    if !is_ccw(polygon) {
        ring.reverse();
    }

    while ring.len() > 3 {
        let m = ring.len();
        let mut ear_at: Option<usize> = None;
        let mut collinear_at: Option<usize> = None;

        for i in 0..m {
            let ip = (i + m - 1) % m;
            let inx = (i + 1) % m;
            let prev = polygon[ring[ip]];
            let cur = polygon[ring[i]];
            let next = polygon[ring[inx]];
            let cross = tri_cross(prev, cur, next);
            if cross.abs() <= CMP_EPS {
                if collinear_at.is_none() {
                    collinear_at = Some(i);
                }
                continue;
            }
            if is_ear(polygon, &ring, i, true) {
                ear_at = Some(i);
                break;
            }
        }

        // Prefer a real ear; otherwise clip a collinear vertex as a zero-area
        // ear; failing both, clip vertex 0 to guarantee forward progress.
        let clip = ear_at.or(collinear_at).unwrap_or(0);
        let m = ring.len();
        let ip = (clip + m - 1) % m;
        let inx = (clip + 1) % m;
        triangles.push([ring[ip], ring[clip], ring[inx]]);
        ring.remove(clip);
    }

    if ring.len() == 3 {
        triangles.push([ring[0], ring[1], ring[2]]);
    }

    triangles
}

/// Byte size of a `u32` (`std430`) index buffer holding `triangle_count`
/// triangles (three indices per triangle, four bytes each).
///
/// The multiplications saturate so a degenerate `triangle_count` can never wrap
/// to a small allocation.
#[must_use]
pub fn index_buffer_bytes(triangle_count: usize) -> usize {
    triangle_count.saturating_mul(3).saturating_mul(U32_STRIDE)
}

/// Byte size of the `GPU` storage binding that backs the `u32` index buffer for
/// `triangle_count` triangles.
///
/// Delegates to [`crate::particle::gpu_layout::storage_bytes`], which clamps an
/// empty buffer up to a single element because a `WebGPU` storage binding may
/// not be zero-sized.
#[must_use]
pub fn gpu_storage_bytes(triangle_count: usize) -> usize {
    storage_bytes(U32_STRIDE, triangle_count.saturating_mul(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    const APPROX_EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= APPROX_EPS
    }

    /// Sum of the areas of the emitted triangles, using the original polygon
    /// vertices referenced by each index triple.
    fn triangulated_area(polygon: &[[f32; 2]], tris: &[[usize; 3]]) -> f32 {
        let mut sum = 0.0f32;
        for t in tris {
            let a = polygon[t[0]];
            let b = polygon[t[1]];
            let c = polygon[t[2]];
            sum += (tri_cross(a, b, c) * 0.5).abs();
        }
        sum
    }

    fn square_ccw() -> [[f32; 2]; 4] {
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    }

    fn square_cw() -> [[f32; 2]; 4] {
        [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]
    }

    fn pentagon_ccw() -> [[f32; 2]; 5] {
        [[0.0, 0.0], [2.0, 0.0], [3.0, 1.5], [1.0, 3.0], [-1.0, 1.5]]
    }

    fn hexagon_ccw() -> [[f32; 2]; 6] {
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [-1.0, 1.0],
        ]
    }

    /// An `L`-shaped concave hexagon (one reflex vertex).
    fn l_shape() -> [[f32; 2]; 6] {
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 3.0],
            [0.0, 3.0],
        ]
    }

    #[test]
    fn signed_area_ccw_is_positive() {
        assert!(close(signed_area(&square_ccw()), 4.0));
    }

    #[test]
    fn signed_area_cw_is_negative() {
        assert!(close(signed_area(&square_cw()), -4.0));
    }

    #[test]
    fn signed_area_degenerate_is_zero() {
        assert!(close(signed_area(&[]), 0.0));
        assert!(close(signed_area(&[[0.0, 0.0]]), 0.0));
        assert!(close(signed_area(&[[0.0, 0.0], [1.0, 1.0]]), 0.0));
    }

    #[test]
    fn is_ccw_matches_winding() {
        assert!(is_ccw(&square_ccw()));
        assert!(!is_ccw(&square_cw()));
    }

    #[test]
    fn point_in_triangle_interior_is_inside() {
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        let c = [0.0, 4.0];
        assert!(point_in_triangle([1.0, 1.0], a, b, c));
    }

    #[test]
    fn point_in_triangle_exterior_is_outside() {
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        let c = [0.0, 4.0];
        assert!(!point_in_triangle([3.0, 3.0], a, b, c));
        assert!(!point_in_triangle([-1.0, -1.0], a, b, c));
    }

    #[test]
    fn point_in_triangle_boundary_is_inside() {
        let a = [0.0, 0.0];
        let b = [4.0, 0.0];
        let c = [0.0, 4.0];
        assert!(point_in_triangle([2.0, 0.0], a, b, c));
        assert!(point_in_triangle([0.0, 0.0], a, b, c));
    }

    #[test]
    fn convex_vertex_detected_ccw() {
        // Right-angle corner of a CCW square is convex.
        let prev = [0.0, 0.0];
        let cur = [2.0, 0.0];
        let next = [2.0, 2.0];
        assert!(is_convex_vertex(prev, cur, next, true));
    }

    #[test]
    fn reflex_vertex_rejected_ccw() {
        // A right-turning corner in a CCW ring is reflex, not convex.
        let prev = [0.0, 0.0];
        let cur = [2.0, 0.0];
        let next = [2.0, -2.0];
        assert!(!is_convex_vertex(prev, cur, next, true));
    }

    #[test]
    fn collinear_vertex_is_not_convex() {
        let prev = [0.0, 0.0];
        let cur = [1.0, 0.0];
        let next = [2.0, 0.0];
        assert!(!is_convex_vertex(prev, cur, next, true));
        assert!(!is_convex_vertex(prev, cur, next, false));
    }

    #[test]
    fn convex_vertex_detected_cw() {
        let prev = [0.0, 0.0];
        let cur = [0.0, 2.0];
        let next = [2.0, 2.0];
        assert!(is_convex_vertex(prev, cur, next, false));
    }

    #[test]
    fn square_yields_two_triangles() {
        let poly = square_ccw();
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
    }

    #[test]
    fn pentagon_yields_three_triangles() {
        let poly = pentagon_ccw();
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
    }

    #[test]
    fn hexagon_yields_four_triangles() {
        let poly = hexagon_ccw();
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
    }

    #[test]
    fn cw_input_is_auto_flipped() {
        let poly = square_cw();
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
        // Each emitted triangle is CCW-wound (positive cross) despite CW input.
        for t in &tris {
            let cross = tri_cross(poly[t[0]], poly[t[1]], poly[t[2]]);
            assert!(cross > 0.0, "expected CCW winding, got cross {cross}");
        }
    }

    #[test]
    fn concave_l_shape_has_correct_count() {
        let poly = l_shape();
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
    }

    #[test]
    fn collinear_vertex_polygon_triangulates() {
        // A square with an extra collinear vertex on the bottom edge.
        let poly = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), poly.len() - 2);
        assert!(close(
            triangulated_area(&poly, &tris),
            signed_area(&poly).abs()
        ));
    }

    #[test]
    fn degenerate_inputs_return_empty() {
        assert!(triangulate(&[]).is_empty());
        assert!(triangulate(&[[0.0, 0.0]]).is_empty());
        assert!(triangulate(&[[0.0, 0.0], [1.0, 0.0]]).is_empty());
    }

    #[test]
    fn triangle_passthrough_keeps_single_triangle() {
        let poly = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]];
        let tris = triangulate(&poly);
        assert_eq!(tris.len(), 1);
        assert!(close(
            triangulated_area(&poly, &tris),
            signed_area(&poly).abs()
        ));
    }

    #[test]
    fn area_conserved_for_square() {
        let poly = square_ccw();
        let tris = triangulate(&poly);
        assert!(close(
            triangulated_area(&poly, &tris),
            signed_area(&poly).abs()
        ));
    }

    #[test]
    fn area_conserved_for_pentagon() {
        let poly = pentagon_ccw();
        let tris = triangulate(&poly);
        assert!(close(
            triangulated_area(&poly, &tris),
            signed_area(&poly).abs()
        ));
    }

    #[test]
    fn area_conserved_for_l_shape() {
        let poly = l_shape();
        let tris = triangulate(&poly);
        assert!(close(
            triangulated_area(&poly, &tris),
            signed_area(&poly).abs()
        ));
    }

    #[test]
    fn area_conserved_for_cw_square() {
        let poly = square_cw();
        let tris = triangulate(&poly);
        assert!(close(
            triangulated_area(&poly, &tris),
            signed_area(&poly).abs()
        ));
    }

    #[test]
    fn indices_are_in_range_and_distinct() {
        let poly = hexagon_ccw();
        let n = poly.len();
        let tris = triangulate(&poly);
        for t in &tris {
            for &idx in t {
                assert!(idx < n);
            }
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
        }
    }

    #[test]
    fn is_ear_detects_convex_ear_on_square() {
        let poly = square_ccw();
        let ring = [0usize, 1, 2, 3];
        // Every corner of a convex square is an ear.
        assert!(is_ear(&poly, &ring, 0, true));
        assert!(is_ear(&poly, &ring, 1, true));
    }

    #[test]
    fn is_ear_rejects_reflex_vertex() {
        let poly = l_shape();
        let ring: Vec<usize> = (0..poly.len()).collect();
        // Vertex 3 ([1,1]) is the reflex corner of the L; it is never an ear.
        assert!(!is_ear(&poly, &ring, 3, true));
    }

    #[test]
    fn is_ear_false_for_short_ring() {
        let poly = square_ccw();
        let ring = [0usize, 1];
        assert!(!is_ear(&poly, &ring, 0, true));
    }

    #[test]
    fn index_buffer_bytes_matches_layout() {
        assert_eq!(index_buffer_bytes(0), 0);
        assert_eq!(index_buffer_bytes(1), 12);
        assert_eq!(index_buffer_bytes(4), 48);
    }

    #[test]
    fn gpu_storage_bytes_clamps_empty_to_one_element() {
        assert_eq!(gpu_storage_bytes(0), U32_STRIDE);
        assert_eq!(gpu_storage_bytes(2), U32_STRIDE * 6);
    }

    #[test]
    fn every_triangle_area_is_finite_and_nonnegative() {
        let poly = pentagon_ccw();
        let tris = triangulate(&poly);
        for t in &tris {
            let area = (tri_cross(poly[t[0]], poly[t[1]], poly[t[2]]) * 0.5).abs();
            assert!(area.is_finite());
            assert!(area >= 0.0);
        }
    }
}
