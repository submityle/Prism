//! Logarithmic point-in-convex-polygon test.
//!
//! The general even-odd test in [`crate::inside::point_in_polygon`] scans every
//! edge, costing time linear in the vertex count. Hit-testing against a *convex*
//! region — a scissor rectangle, a rotated bounding box, a convex widget mask —
//! can instead run in logarithmic time, which matters when a convex clip is
//! queried against many points per frame (hover, picking, culling).
//!
//! [`point_in_convex`] fans the polygon from its first vertex and binary-searches
//! for the angular wedge that contains the query point, then performs a single
//! edge-side test. Only `+ - *` and comparisons are used, so it is exact and
//! `no_std`-clean. The winding (clockwise or counter-clockwise) does not matter:
//! the signed area selects the orientation, matching the rest of this crate.

use alloc::vec::Vec;

use crate::measure::signed_area;

/// Twice the signed area of triangle `o`, `a`, `b`; its sign tells which side of
/// the directed line `o->a` the point `b` lies on (positive for the left side).
fn cross(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Tests `p` against a counter-clockwise convex `poly` (at least three vertices)
/// by fanning from `poly[0]` and binary-searching the containing wedge.
fn in_ccw(poly: &[(f32, f32)], p: (f32, f32)) -> bool {
    let n = poly.len();
    let a0 = poly[0];

    // Reject points outside the fan's angular span: right of the first fan ray
    // or left of the last one.
    if cross(a0, poly[1], p) < 0.0 {
        return false;
    }
    if cross(a0, poly[n - 1], p) > 0.0 {
        return false;
    }

    // Binary-search the last index `lo` whose fan ray keeps `p` on the interior
    // side; `p` then lies in the wedge spanned by `poly[lo]` and `poly[lo + 1]`.
    let mut lo = 1usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if cross(a0, poly[mid], p) >= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Inside the triangle `poly[0], poly[lo], poly[hi]` iff `p` is on the
    // interior side of the far edge `poly[lo]->poly[hi]`.
    cross(poly[lo], poly[hi], p) >= 0.0
}

/// Returns `true` when point `p` lies inside or on the boundary of the convex
/// polygon `poly`, in time logarithmic in the vertex count.
///
/// `poly` must be a convex ring given in either winding order with no repeated
/// closing vertex (for example the output of [`crate::hull::convex_hull`]). A
/// polygon with fewer than three vertices has no interior and always returns
/// `false`. Results are undefined for a non-convex `poly`.
#[must_use]
pub fn point_in_convex(poly: &[(f32, f32)], p: (f32, f32)) -> bool {
    if poly.len() < 3 {
        return false;
    }

    // Normalise to counter-clockwise so the fan sweep is monotone.
    if signed_area(poly) >= 0.0 {
        in_ccw(poly, p)
    } else {
        let rev: Vec<(f32, f32)> = poly.iter().rev().copied().collect();
        in_ccw(&rev, p)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::hull::convex_hull;
    use crate::inside::point_in_polygon;
    use crate::polygon::sd_polygon;
    use alloc::vec::Vec;

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let b = (next_rand(state) >> 40) as u32;
        lo + (hi - lo) * (b as f32 / 16_777_216.0)
    }

    /// A random convex polygon (hull of random points), at least a triangle.
    fn random_convex(state: &mut u64, spread: f32) -> Vec<(f32, f32)> {
        loop {
            let n = 6 + (next_rand(state) % 12) as usize;
            let mut pts: Vec<(f32, f32)> = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push((rand_in(state, -spread, spread), rand_in(state, -spread, spread)));
            }
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// Strong independent oracle: away from the boundary, agree with the exact
    /// signed-distance field (negative inside, positive outside).
    #[test]
    fn agrees_with_signed_distance() {
        let mut state = 0x0_B0B1_u64;
        for _ in 0..200 {
            let hull = random_convex(&mut state, 10.0);
            for _ in 0..40 {
                let p = (rand_in(&mut state, -14.0, 14.0), rand_in(&mut state, -14.0, 14.0));
                let sd = sd_polygon(p.0, p.1, &hull);
                let got = point_in_convex(&hull, p);
                if sd < -1e-3 {
                    assert!(got, "sd {sd} inside but reported outside at {p:?}");
                } else if sd > 1e-3 {
                    assert!(!got, "sd {sd} outside but reported inside at {p:?}");
                }
            }
        }
    }

    /// Away from the boundary, agree with the linear even-odd test.
    #[test]
    fn agrees_with_even_odd() {
        let mut state = 0x1234_ABCD_u64;
        for _ in 0..200 {
            let hull = random_convex(&mut state, 8.0);
            for _ in 0..40 {
                let p = (rand_in(&mut state, -12.0, 12.0), rand_in(&mut state, -12.0, 12.0));
                let sd = sd_polygon(p.0, p.1, &hull);
                if sd.abs() <= 1e-3 {
                    continue;
                }
                assert_eq!(
                    point_in_convex(&hull, p),
                    point_in_polygon(&hull, p),
                    "disagreement at {p:?} (sd {sd})"
                );
            }
        }
    }

    /// Winding order must not change the answer.
    #[test]
    fn winding_independent() {
        let mut state = 0xDEAD_BEEF_u64;
        for _ in 0..200 {
            let hull = random_convex(&mut state, 9.0);
            let mut rev = hull.clone();
            rev.reverse();
            for _ in 0..30 {
                let p = (rand_in(&mut state, -13.0, 13.0), rand_in(&mut state, -13.0, 13.0));
                assert_eq!(point_in_convex(&hull, p), point_in_convex(&rev, p), "at {p:?}");
            }
        }
    }

    /// Every vertex is reported on the boundary (inside).
    #[test]
    fn vertices_are_inside() {
        let mut state = 0xFEED_FACE_u64;
        for _ in 0..200 {
            let hull = random_convex(&mut state, 7.0);
            for &v in &hull {
                assert!(point_in_convex(&hull, v), "vertex {v:?} reported outside");
            }
        }
    }

    /// A point just inside each edge midpoint is reported inside. The exact
    /// midpoint sits on the boundary, a measure-zero case whose float rounding
    /// is inherently fragile; nudging a hair toward the (strictly interior)
    /// vertex average tests that edges bound the interior without relying on
    /// exact boundary classification.
    #[test]
    fn edge_midpoints_inside() {
        let mut state = 0x7777_3333_u64;
        for _ in 0..200 {
            let hull = random_convex(&mut state, 6.0);
            let n = hull.len();
            // The average of a convex polygon's vertices is strictly interior.
            let inv = 1.0 / n as f32;
            let mut cx = 0.0;
            let mut cy = 0.0;
            for &(x, y) in &hull {
                cx += x;
                cy += y;
            }
            let c = (cx * inv, cy * inv);
            for i in 0..n {
                let a = hull[i];
                let b = hull[(i + 1) % n];
                let mid = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
                let p = (mid.0 + (c.0 - mid.0) * 1e-3, mid.1 + (c.1 - mid.1) * 1e-3);
                assert!(point_in_convex(&hull, p), "near-edge interior point {p:?} outside");
            }
        }
    }

    /// Fixed square and triangle, including interior, exterior, and boundary.
    #[test]
    fn fixed_cases() {
        let square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(point_in_convex(&square, (2.0, 2.0)));
        assert!(point_in_convex(&square, (0.0, 0.0)));
        assert!(point_in_convex(&square, (4.0, 2.0)));
        assert!(!point_in_convex(&square, (5.0, 2.0)));
        assert!(!point_in_convex(&square, (-0.1, 2.0)));
        assert!(!point_in_convex(&square, (2.0, 4.5)));

        let tri = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)];
        assert!(point_in_convex(&tri, (1.0, 1.0)));
        assert!(point_in_convex(&tri, (3.0, 3.0))); // on the hypotenuse
        assert!(!point_in_convex(&tri, (4.0, 4.0)));
        assert!(!point_in_convex(&tri, (-1.0, 1.0)));
    }

    /// Degenerate inputs have no interior.
    #[test]
    fn degenerate_inputs() {
        assert!(!point_in_convex(&[], (0.0, 0.0)));
        assert!(!point_in_convex(&[(0.0, 0.0)], (0.0, 0.0)));
        assert!(!point_in_convex(&[(0.0, 0.0), (1.0, 0.0)], (0.5, 0.0)));
    }
}
