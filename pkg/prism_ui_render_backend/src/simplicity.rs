//! Simple-path (self-intersection) tests for polylines and polygon rings.
//!
//! Several modules in this crate *assume* their input is a simple path — no edge
//! crosses a non-adjacent edge — including [`is_convex`](crate::convexity::is_convex),
//! [`triangulate`](crate::triangulate::triangulate),
//! [`offset_convex`](crate::offset::offset_convex) and
//! [`sd_polygon`](crate::polygon::sd_polygon). This module is the guard that
//! establishes that precondition: it reports the first pair of edges that
//! intersect anywhere other than at a shared vertex of adjacent edges.
//!
//! Intersection is the full predicate — proper crossings, endpoint touches, and
//! collinear overlap all count. It is a closed test (an edge touching a
//! non-adjacent edge's interior, or two collinear edges overlapping, is *not*
//! simple), matching the strict definition the consumers rely on. The
//! arithmetic is exact-orientation `f32` (`+ - *`), consistent with
//! [`is_convex`](crate::convexity::is_convex); zero-length edges are ignored.

use crate::vec2::cross3;

/// Whether `p`, known to be collinear with `a` and `b`, lies within the
/// segment `a..=b` (bounding-box containment).
fn on_segment(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> bool {
    p.0 >= a.0.min(b.0)
        && p.0 <= a.0.max(b.0)
        && p.1 >= a.1.min(b.1)
        && p.1 <= a.1.max(b.1)
}

/// Whether segment `a..b` and segment `c..d` intersect (proper crossing, endpoint
/// touch, or collinear overlap).
fn segments_intersect(a: (f32, f32), b: (f32, f32), c: (f32, f32), d: (f32, f32)) -> bool {
    let d1 = cross3(c, d, a);
    let d2 = cross3(c, d, b);
    let d3 = cross3(a, b, c);
    let d4 = cross3(a, b, d);

    // Proper crossing: each segment straddles the other's line.
    if (d1 > 0.0 && d2 < 0.0 || d1 < 0.0 && d2 > 0.0)
        && (d3 > 0.0 && d4 < 0.0 || d3 < 0.0 && d4 > 0.0)
    {
        return true;
    }

    // Touching / collinear overlap: an endpoint on the other segment.
    (d1 == 0.0 && on_segment(c, d, a))
        || (d2 == 0.0 && on_segment(c, d, b))
        || (d3 == 0.0 && on_segment(a, b, c))
        || (d4 == 0.0 && on_segment(a, b, d))
}

/// Whether the two edge indices are adjacent (share a vertex), so an
/// intersection at that shared vertex is expected and allowed.
fn adjacent(i: usize, j: usize, edges: usize, closed: bool) -> bool {
    j == i + 1 || (closed && i == 0 && j + 1 == edges)
}

/// The first pair `(i, j)` of non-adjacent edges (`i < j`) that intersect, or
/// `None` when the path is simple.
///
/// `closed` treats `points` as a ring (edge `n-1` joins back to vertex `0`);
/// otherwise it is an open polyline. Zero-length edges are ignored. An input
/// with fewer than two edges is trivially simple.
#[must_use]
pub fn first_self_intersection(points: &[(f32, f32)], closed: bool) -> Option<(usize, usize)> {
    let n = points.len();
    let edges = if closed { n } else { n.saturating_sub(1) };
    if edges < 2 {
        return None;
    }
    let edge = |i: usize| (points[i], points[(i + 1) % n]);

    for i in 0..edges {
        let (a, b) = edge(i);
        if a == b {
            continue;
        }
        for j in (i + 1)..edges {
            if adjacent(i, j, edges, closed) {
                continue;
            }
            let (c, d) = edge(j);
            if c == d {
                continue;
            }
            if segments_intersect(a, b, c, d) {
                return Some((i, j));
            }
        }
    }
    None
}

/// Whether the closed ring `verts` is a simple (non-self-intersecting) polygon.
///
/// Returns `false` for fewer than three vertices (a ring that cannot bound an
/// area), and `false` for any self-intersection — a crossing, a vertex lying on a
/// non-adjacent edge, or collinear overlap. Adjacent edges meeting at their
/// shared vertex are allowed.
#[must_use]
pub fn is_simple_polygon(verts: &[(f32, f32)]) -> bool {
    verts.len() >= 3 && first_self_intersection(verts, true).is_none()
}

/// Whether the open polyline `points` crosses itself.
///
/// Adjacent segments meeting at their shared vertex are allowed; any other
/// contact (crossing, touch, or collinear overlap) makes the polyline
/// self-intersecting.
#[must_use]
pub fn polyline_self_intersects(points: &[(f32, f32)]) -> bool {
    first_self_intersection(points, false).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_and_concave_rings_are_simple() {
        assert!(is_simple_polygon(&[
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0)
        ]));
        // Concave L-shape.
        assert!(is_simple_polygon(&[
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0)
        ]));
    }

    #[test]
    fn bowtie_ring_is_not_simple() {
        let bowtie = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(!is_simple_polygon(&bowtie));
        assert_eq!(first_self_intersection(&bowtie, true), Some((0, 2)));
    }

    #[test]
    fn too_few_vertices_is_not_a_simple_polygon() {
        assert!(!is_simple_polygon(&[]));
        assert!(!is_simple_polygon(&[(0.0, 0.0), (1.0, 1.0)]));
    }

    #[test]
    fn vertex_touching_a_non_adjacent_edge_is_not_simple() {
        // Vertex 3 at (2,0) lies on edge 0 = (0,0)->(4,0).
        let ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)];
        assert_eq!(first_self_intersection(&ring, true), Some((0, 2)));
    }

    #[test]
    fn collinear_overlap_is_not_simple() {
        // Edge 3 = (3,0)->(1,0) overlaps edge 0 = (0,0)->(4,0).
        let path = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (3.0, 0.0), (1.0, 0.0)];
        assert!(polyline_self_intersects(&path));
    }

    #[test]
    fn open_polylines() {
        // Non-crossing zigzag.
        assert!(!polyline_self_intersects(&[
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 0.0),
            (3.0, 1.0)
        ]));
        // Figure-eight: edge 0 crosses edge 2.
        assert!(polyline_self_intersects(&[
            (0.0, 0.0),
            (2.0, 2.0),
            (2.0, 0.0),
            (0.0, 2.0)
        ]));
        // Fewer than two edges is never self-intersecting.
        assert!(!polyline_self_intersects(&[(0.0, 0.0)]));
        assert!(!polyline_self_intersects(&[(0.0, 0.0), (1.0, 1.0)]));
    }

    #[test]
    fn zero_length_edges_are_ignored() {
        // The repeated vertex adds a degenerate edge but no real crossing.
        assert!(!polyline_self_intersects(&[
            (0.0, 0.0),
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0)
        ]));
    }
}
