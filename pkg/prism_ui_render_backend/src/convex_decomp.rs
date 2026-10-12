//! Convex decomposition of a simple polygon (Hertel–Mehlhorn).
//!
//! Rasterisers, physics proxies and CSG all prefer a polygon cut into **convex**
//! pieces: each piece can be filled by a convex fast path. [`convex_decompose`]
//! triangulates the polygon ([`triangulate`](crate::triangulate::triangulate)),
//! then greedily merges any two pieces across a shared edge while the union stays
//! convex — the classic Hertel–Mehlhorn reduction, which gives far fewer pieces
//! than a raw triangulation (a convex input collapses back to a single piece).
//!
//! The input must be a simple polygon; a self-intersecting or degenerate ring has
//! no well-defined interior, so it yields an empty decomposition rather than a
//! wrong one. The pieces partition the input exactly: their areas sum to the
//! input's area and each piece lies inside it.

use alloc::vec::Vec;

use crate::convexity::is_convex;
use crate::measure::area;
use crate::simplicity::is_simple_polygon;
use crate::triangulate::triangulate;

/// The convex decomposition of the simple polygon `polygon`.
///
/// Returns one convex sub-polygon (as a vertex ring, in the input's winding) per
/// piece, or an empty vector when `polygon` is not a simple non-degenerate
/// polygon. Pieces share edges but never overlap, and their areas sum to the
/// input's.
#[must_use]
pub fn convex_decompose(polygon: &[(f32, f32)]) -> Vec<Vec<(f32, f32)>> {
    let n = polygon.len();
    if n < 3 || !is_simple_polygon(polygon) || area(polygon) <= 0.0 {
        return Vec::new();
    }
    let triangles = triangulate(polygon);
    if triangles.len() != n - 2 {
        return Vec::new();
    }

    let mut pieces: Vec<Vec<usize>> = triangles.iter().map(|t| t.to_vec()).collect();

    // Greedy Hertel-Mehlhorn: repeatedly merge the first edge-adjacent pair
    // whose union is convex, until no more merges are possible.
    loop {
        let mut merged = false;
        'search: for p in 0..pieces.len() {
            for q in (p + 1)..pieces.len() {
                let Some((i, j)) = shared_edge(&pieces[p], &pieces[q]) else {
                    continue;
                };
                let candidate = merge(&pieces[p], &pieces[q], i, j);
                let coords: Vec<(f32, f32)> =
                    candidate.iter().map(|&k| polygon[k]).collect();
                if is_convex(&coords) && is_simple_polygon(&coords) {
                    pieces[p] = candidate;
                    pieces.remove(q);
                    merged = true;
                    break 'search;
                }
            }
        }
        if !merged {
            break;
        }
    }

    pieces
        .iter()
        .map(|ring| ring.iter().map(|&k| polygon[k]).collect())
        .collect()
}

/// The `(index in a, index in b)` of the edge `a` and `b` share, traversed in
/// opposite directions (as two rings on either side of the edge do), or `None`.
fn shared_edge(a: &[usize], b: &[usize]) -> Option<(usize, usize)> {
    for i in 0..a.len() {
        let a0 = a[i];
        let a1 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            if b[j] == a1 && b[(j + 1) % b.len()] == a0 {
                return Some((i, j));
            }
        }
    }
    None
}

/// Merge `b` into `a` across the shared edge `a[i] -> a[i+1]` (which is
/// `b[j] -> b[j+1]` reversed), replacing that edge with `b`'s other path.
fn merge(a: &[usize], b: &[usize], i: usize, j: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len() - 2);
    out.extend_from_slice(&a[..=i]);
    // Ship `b`'s intermediate vertices: from just after `a[i]` (= `b[j+1]`) up to
    // just before `a[i+1]` (= `b[j]`).
    let lb = b.len();
    for step in 1..lb - 1 {
        out.push(b[(j + 1 + step) % lb]);
    }
    out.extend_from_slice(&a[i + 1..]);
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::inside::point_in_polygon;
    use crate::measure::centroid;

    fn l_shape() -> Vec<(f32, f32)> {
        alloc::vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ]
    }

    #[test]
    fn convex_input_collapses_to_a_single_piece() {
        let square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        let pieces = convex_decompose(&square);
        assert_eq!(pieces.len(), 1, "a convex polygon merges back to one piece");
        assert!((area(&pieces[0]) - 16.0).abs() < 1e-3);
    }

    #[test]
    fn concave_pieces_are_convex_and_cover_the_input() {
        let poly = l_shape();
        let pieces = convex_decompose(&poly);
        assert!(pieces.len() >= 2, "a concave polygon needs more than one piece");

        let mut total = 0.0_f32;
        for piece in &pieces {
            assert!(is_convex(piece), "piece is not convex: {piece:?}");
            let c = centroid(piece).expect("non-degenerate piece");
            assert!(point_in_polygon(&poly, c), "piece centroid escapes the input");
            total += area(piece);
        }
        assert!((total - area(&poly)).abs() < 1e-3, "areas sum to {total}");
    }

    #[test]
    fn invalid_inputs_yield_no_pieces() {
        // Self-intersecting bowtie.
        let bowtie = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(convex_decompose(&bowtie).is_empty());
        // Collinear (zero area).
        assert!(convex_decompose(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]).is_empty());
        // Too few vertices.
        assert!(convex_decompose(&[(0.0, 0.0), (1.0, 1.0)]).is_empty());
    }
}
