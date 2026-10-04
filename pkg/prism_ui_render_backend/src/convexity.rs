//! Convexity test for a simple polygon ring.
//!
//! Convex regions enable fast paths across this crate: logarithmic hit-testing
//! ([`crate::point_convex::point_in_convex`]), single-pass Sutherland-Hodgman
//! clipping ([`crate::clip::clip_polygon`]), and separating-axis queries all
//! assume their input is convex. [`is_convex`] is the cheap guard that decides
//! whether those paths are legal before taking them.
//!
//! A simple (non-self-intersecting) polygon is convex exactly when every turn
//! around the ring goes the same way: the cross product of each pair of
//! consecutive edges never changes sign. Straight (collinear) vertices do not
//! break convexity, so a zero cross product is skipped rather than rejected.
//! Only `+ - *` and comparisons are used, so the test is exact and
//! `no_std`-clean. The winding order (clockwise or counter-clockwise) is
//! irrelevant; both are convex.
//!
//! The simple-polygon assumption is load-bearing: a self-intersecting ring that
//! happens to turn the same way at every vertex (for example a pentagram traced
//! as a star) is reported convex. Callers that may hold non-simple rings must
//! establish simplicity separately; the fast paths above already require it.

/// Cross product of the directed edges `prev->cur` and `cur->next`. Its sign is
/// the turn direction at `cur` (positive left, negative right, zero straight).
fn turn(prev: (f32, f32), cur: (f32, f32), next: (f32, f32)) -> f32 {
    (cur.0 - prev.0) * (next.1 - cur.1) - (cur.1 - prev.1) * (next.0 - cur.0)
}

/// Returns `true` when the simple polygon ring `poly` is convex.
///
/// `poly` is a closed ring listed once (no repeated closing vertex), in either
/// winding order, and is assumed simple (non-self-intersecting). A ring with
/// fewer than three vertices, or one whose vertices are all collinear, encloses
/// no area and is reported non-convex. Collinear vertices on an otherwise
/// convex ring are permitted and do not change the result.
#[must_use]
pub fn is_convex(poly: &[(f32, f32)]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }

    // Track the sign of the turn seen so far. `saw_turn` guards the all-collinear
    // (degenerate) ring, which has no interior and is not convex.
    let mut positive = false;
    let mut negative = false;
    for i in 0..n {
        let prev = poly[(i + n - 1) % n];
        let cur = poly[i];
        let next = poly[(i + 1) % n];
        let t = turn(prev, cur, next);
        if t > 0.0 {
            positive = true;
        } else if t < 0.0 {
            negative = true;
        }
        if positive && negative {
            return false;
        }
    }

    positive || negative
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::hull::convex_hull;
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

    /// A random strictly convex polygon (hull of random points), at least a
    /// quadrilateral so a vertex can be dented into a reflex corner.
    fn random_hull(state: &mut u64, spread: f32) -> Vec<(f32, f32)> {
        loop {
            let n = 6 + (next_rand(state) % 12) as usize;
            let mut pts: Vec<(f32, f32)> = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push((rand_in(state, -spread, spread), rand_in(state, -spread, spread)));
            }
            let hull = convex_hull(&pts);
            if hull.len() >= 4 {
                return hull;
            }
        }
    }

    /// Strong independent oracle: `convex_hull` output is strictly convex, so
    /// `is_convex` must accept it in either winding order.
    #[test]
    fn accepts_hulls_both_windings() {
        let mut state = 0x0_C0C0_u64;
        for _ in 0..400 {
            let hull = random_hull(&mut state, 10.0);
            assert!(is_convex(&hull), "hull reported non-convex: {hull:?}");
            let mut rev = hull.clone();
            rev.reverse();
            assert!(is_convex(&rev), "reversed hull reported non-convex: {rev:?}");
        }
    }

    /// Strong independent oracle for rejection: denting one hull vertex toward
    /// the chord of its neighbors makes it interior to the hull of the rest, so
    /// `convex_hull` drops it (fewer vertices) — an independent proof the dented
    /// ring is non-convex. `is_convex` must then reject it.
    #[test]
    fn rejects_dented_rings() {
        let mut state = 0x5151_2727_u64;
        for _ in 0..400 {
            let hull = random_hull(&mut state, 9.0);
            let n = hull.len();
            let i = (next_rand(&mut state) as usize) % n;
            // The average of the OTHER n-1 vertices is strictly interior to
            // their convex hull, so moving vertex i there both dents it into a
            // reflex corner and leaves it inside the hull of the rest.
            let inv = 1.0 / (n - 1) as f32;
            let mut cx = 0.0;
            let mut cy = 0.0;
            for (j, &(x, y)) in hull.iter().enumerate() {
                if j != i {
                    cx += x;
                    cy += y;
                }
            }
            let dented_vertex = (cx * inv, cy * inv);
            let mut poly = hull.clone();
            poly[i] = dented_vertex;

            // Independent proof of non-convexity: the hull of the dented ring
            // drops the interior vertex.
            let rehull = convex_hull(&poly);
            assert!(
                rehull.len() < n,
                "dent did not create a reflex vertex (hull kept all {n})"
            );
            assert!(!is_convex(&poly), "dented ring reported convex: {poly:?}");
        }
    }

    /// Fixed convex shapes, both windings, including a straight (collinear)
    /// vertex that must not break convexity.
    #[test]
    fn fixed_convex() {
        let square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(is_convex(&square));
        let cw = [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)];
        assert!(is_convex(&cw));
        let tri = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)];
        assert!(is_convex(&tri));
        // A collinear midpoint on the bottom edge keeps the pentagon convex.
        let with_straight = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(is_convex(&with_straight));
    }

    /// Fixed non-convex shapes.
    #[test]
    fn fixed_non_convex() {
        // Classic L / arrow with one reflex vertex at (2, 2).
        let arrow = [(0.0, 0.0), (4.0, 0.0), (2.0, 2.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(!is_convex(&arrow));
        // Chevron, reflex at (2, 1).
        let chevron = [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0), (2.0, 4.0)];
        assert!(!is_convex(&chevron));
    }

    /// Degenerate rings enclose no area and are not convex.
    #[test]
    fn degenerate_inputs() {
        assert!(!is_convex(&[]));
        assert!(!is_convex(&[(0.0, 0.0)]));
        assert!(!is_convex(&[(0.0, 0.0), (1.0, 1.0)]));
        // Three collinear points: no interior.
        assert!(!is_convex(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]));
    }
}
