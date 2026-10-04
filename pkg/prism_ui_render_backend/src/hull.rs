//! Convex hull of a 2D point set via Andrew's monotone chain.
//!
//! The geometry primitives in [`crate::polygon`] assume a caller already has a
//! simple polygon in hand; many callers instead start from a loose *cloud* of
//! points — glyph outline extrema, the flattened vertices of several
//! [`CubicBezier`](crate::CubicBezier) paths, a scattered set of hit-test
//! anchors — and need the tightest convex boundary that encloses them. That
//! boundary is the natural input for broad-phase culling, fitting a bounding
//! polygon, or feeding [`sd_polygon`](crate::polygon::sd_polygon).
//!
//! [`convex_hull`] computes it with Andrew's monotone chain: sort the points,
//! then sweep once for the lower boundary and once for the upper. The result is
//! the *strict* hull in counter-clockwise order — collinear points on an edge
//! are dropped, and the start/end vertex is not repeated.
//!
//! Everything is pure `f32` arithmetic (`+ - *` and comparisons); there are no
//! transcendental calls and no `sqrt`, so the routine is `no_std`-clean and
//! bit-stable.

use alloc::vec::Vec;

/// Twice the signed area of triangle `o -> a -> b`.
///
/// Positive when `o -> a -> b` turns counter-clockwise, negative for a
/// clockwise turn, and exactly zero when the three points are collinear.
#[inline]
fn cross(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Computes the convex hull of `points`, returned as its vertices in
/// counter-clockwise order starting from the lexicographically smallest point
/// (smallest `x`, ties broken by smallest `y`).
///
/// The hull is *strict*: points lying on the interior of a hull edge are
/// omitted, so no three consecutive output vertices are collinear whenever the
/// hull has three or more vertices. Every returned vertex is one of the input
/// points (the routine only copies points, never synthesises new ones).
///
/// Degenerate inputs are handled directly: an empty slice yields an empty
/// vector, a slice whose points are all equal yields that single point, and a
/// set of two or more distinct but collinear points yields exactly the two
/// extreme endpoints.
#[must_use]
pub fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    // Sort lexicographically and drop exact duplicates so the sweep sees each
    // distinct location once.
    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup();

    if pts.len() <= 2 {
        return pts;
    }

    // Lower hull: left-to-right, keeping only counter-clockwise turns. A
    // non-positive cross product means the previous vertex is a right turn or
    // collinear, so it is popped — this is what strips interior-edge points.
    let mut lower: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    // Upper hull: the same sweep right-to-left completes the boundary.
    let mut upper: Vec<(f32, f32)> = Vec::with_capacity(pts.len());
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    // Drop each chain's last point (it is the first point of the other chain)
    // and concatenate into one counter-clockwise ring.
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use alloc::vec;

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let bits = (next_rand(state) >> 40) as u32;
        let unit = (bits as f32) / 16_777_216.0;
        lo + (hi - lo) * unit
    }

    fn rand_point(state: &mut u64) -> (f32, f32) {
        (rand_in(state, -50.0, 50.0), rand_in(state, -50.0, 50.0))
    }

    /// Signed area times two of a (closed) polygon ring, via the shoelace sum.
    /// Positive for counter-clockwise orientation.
    fn signed_area2(poly: &[(f32, f32)]) -> f32 {
        let mut acc = 0.0_f32;
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            acc += a.0 * b.1 - b.0 * a.1;
        }
        acc
    }

    /// Independent oracle: is `p` inside or on the boundary of a convex ccw
    /// polygon? True iff every edge keeps `p` on its left (cross >= -eps).
    fn inside_convex_ccw(p: (f32, f32), poly: &[(f32, f32)]) -> bool {
        let eps = 1e-2_f32;
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            if cross(a, b, p) < -eps {
                return false;
            }
        }
        true
    }

    fn contains_point(poly: &[(f32, f32)], p: (f32, f32)) -> bool {
        poly.contains(&p)
    }

    #[test]
    fn empty_and_singletons() {
        assert_eq!(convex_hull(&[]), vec![]);
        assert_eq!(convex_hull(&[(1.0, 2.0)]), vec![(1.0, 2.0)]);
        // All identical collapses to a single point.
        assert_eq!(
            convex_hull(&[(3.0, 3.0), (3.0, 3.0), (3.0, 3.0)]),
            vec![(3.0, 3.0)]
        );
    }

    #[test]
    fn collinear_reduces_to_endpoints() {
        let pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        assert_eq!(convex_hull(&pts), vec![(0.0, 0.0), (3.0, 3.0)]);
    }

    #[test]
    fn square_with_interior_points() {
        // Four corners plus noise strictly inside must yield exactly the corners
        // (counter-clockwise from the lexicographically smallest corner).
        let pts = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (2.0, 2.0),
            (1.0, 3.0),
            (3.0, 1.0),
            (2.0, 1.0),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull, vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]);
        assert!(signed_area2(&hull) > 0.0, "hull must be ccw");
    }

    #[test]
    fn edge_midpoints_are_dropped() {
        // Collinear points sitting on the square's edges must not appear.
        let pts = [
            (0.0, 0.0),
            (2.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (4.0, 4.0),
            (2.0, 4.0),
            (0.0, 4.0),
            (0.0, 2.0),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        for &c in &[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)] {
            assert!(contains_point(&hull, c));
        }
    }

    #[test]
    fn random_hull_contains_all_points_and_is_convex() {
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 40) as usize;
            let pts: Vec<(f32, f32)> = (0..n).map(|_| rand_point(&mut state)).collect();
            let hull = convex_hull(&pts);
            assert!(hull.len() >= 2 && hull.len() <= pts.len());

            // Every input point lies inside or on the hull.
            for &p in &pts {
                assert!(inside_convex_ccw(p, &hull), "point {p:?} outside hull");
            }
            // Every hull vertex is an original input point.
            for &h in &hull {
                assert!(contains_point(&pts, h));
            }
            if hull.len() >= 3 {
                // Strictly convex and counter-clockwise: every turn is a left
                // turn (cross strictly positive).
                for i in 0..hull.len() {
                    let a = hull[i];
                    let b = hull[(i + 1) % hull.len()];
                    let c = hull[(i + 2) % hull.len()];
                    assert!(cross(a, b, c) > 0.0, "non-left turn at {i}");
                }
                assert!(signed_area2(&hull) > 0.0);
            }
        }
    }

    #[test]
    fn hull_is_idempotent() {
        // Running the hull on its own output reproduces it exactly, since the
        // output is already a canonically ordered strict hull.
        let mut state = 0xDEAD_BEEF_0BAD_F00D_u64;
        for _ in 0..100 {
            let n = 3 + (next_rand(&mut state) % 30) as usize;
            let pts: Vec<(f32, f32)> = (0..n).map(|_| rand_point(&mut state)).collect();
            let hull = convex_hull(&pts);
            let twice = convex_hull(&hull);
            assert_eq!(hull, twice);
        }
    }

    #[test]
    fn interior_points_never_change_hull() {
        // Adding points strictly inside the current hull leaves it unchanged.
        let mut state = 0x00C0_FFEE_1234_5678_u64;
        for _ in 0..60 {
            let base = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)];
            let mut pts: Vec<(f32, f32)> = base.to_vec();
            let before = convex_hull(&pts);
            for _ in 0..20 {
                pts.push((rand_in(&mut state, -9.0, 9.0), rand_in(&mut state, -9.0, 9.0)));
            }
            let after = convex_hull(&pts);
            assert_eq!(before, after);
        }
    }
}
