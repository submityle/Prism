//! Diameter of a convex polygon via rotating calipers.
//!
//! The *diameter* of a point set is the greatest distance between any two of
//! its points; for a convex polygon that extreme pair is always a pair of
//! vertices. It answers "how wide is this shape at its widest?" — sizing a
//! focus ring around an arbitrary selection, choosing a camera pull-back that
//! frames a convex region, or bounding the longest span of a dragged lasso.
//!
//! A naive scan compares every vertex pair in quadratic time. The rotating
//! calipers walk instead sweeps a single antipodal vertex forward in lockstep
//! with each hull edge, visiting only the antipodal pairs and so running in
//! time linear in the vertex count. The farthest pair is always antipodal, so
//! the sweep is exhaustive.
//!
//! The input must be a convex polygon in counter-clockwise order with no
//! repeated start vertex, exactly as produced by
//! [`convex_hull`](crate::hull::convex_hull): strictly convex, lowest-then-left
//! start, collinear points already dropped. Only `+ - *`, comparisons and a
//! single `sqrt` are used, so the routine is `no_std`-clean and bit-stable
//! across targets.

/// Twice the signed area of triangle `a b c`; equivalently the cross product
/// `(b - a) × (c - a)`. For a counter-clockwise edge `a -> b` it grows with the
/// perpendicular distance of `c` to the left of that edge.
#[inline]
fn tri_area2(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Squared Euclidean distance between two points (no `sqrt`, so exact ordering).
#[inline]
fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// The two farthest-apart vertices of a convex polygon (its diameter pair).
///
/// `hull` must be a convex polygon in counter-clockwise order with no repeated
/// start vertex, as returned by [`convex_hull`](crate::hull::convex_hull).
/// Returns [`None`] for fewer than two vertices; for exactly two it returns
/// that pair. When several pairs tie for the maximum, one of them is returned.
pub fn convex_diameter_pair(hull: &[(f32, f32)]) -> Option<((f32, f32), (f32, f32))> {
    let n = hull.len();
    if n < 2 {
        return None;
    }
    if n == 2 {
        return Some((hull[0], hull[1]));
    }

    let mut best_sq = -1.0f32;
    let mut best = (hull[0], hull[1]);
    // `j` is the vertex antipodal to edge `i -> i + 1`; it only ever advances,
    // so the total work across every edge is linear. `budget` caps the sweep at
    // one full revolution as a defensive guard against malformed (non-convex)
    // input, and is never reached for a valid strictly-convex hull.
    let mut j = 1;
    let mut budget = 2 * n;
    for i in 0..n {
        let ni = (i + 1) % n;
        while budget > 0 {
            let nj = (j + 1) % n;
            if tri_area2(hull[i], hull[ni], hull[nj]) > tri_area2(hull[i], hull[ni], hull[j]) {
                j = nj;
                budget -= 1;
            } else {
                break;
            }
        }
        // Both endpoints of the edge can be the diameter partner of `j`.
        let d1 = dist_sq(hull[i], hull[j]);
        if d1 > best_sq {
            best_sq = d1;
            best = (hull[i], hull[j]);
        }
        let d2 = dist_sq(hull[ni], hull[j]);
        if d2 > best_sq {
            best_sq = d2;
            best = (hull[ni], hull[j]);
        }
    }
    Some(best)
}

/// The diameter of a convex polygon: the greatest distance between any two of
/// its vertices.
///
/// `hull` must satisfy the same convexity precondition as
/// [`convex_diameter_pair`]. Fewer than two vertices yields `0.0`.
pub fn convex_diameter(hull: &[(f32, f32)]) -> f32 {
    match convex_diameter_pair(hull) {
        Some((a, b)) => dist_sq(a, b).sqrt(),
        None => 0.0,
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
    use alloc::vec::Vec;

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(s: &mut u64, lo: f32, hi: f32) -> f32 {
        let b = (next_rand(s) >> 40) as u32;
        lo + (hi - lo) * (b as f32 / 16_777_216.0)
    }

    /// A convex polygon of at least three vertices via the crate's hull.
    fn random_hull(state: &mut u64, spread: f32) -> Vec<(f32, f32)> {
        loop {
            let n = 6 + (next_rand(state) % 12) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(state, -spread, spread), rand_in(state, -spread, spread)))
                .collect();
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// Independent gold oracle: the max distance over every unordered pair.
    fn brute_diameter(pts: &[(f32, f32)]) -> f32 {
        let mut best = 0.0f32;
        for (i, &p) in pts.iter().enumerate() {
            for &q in &pts[i + 1..] {
                best = best.max(dist_sq(p, q).sqrt());
            }
        }
        best
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * (a.abs() + b.abs()) + 1e-3
    }

    /// The calipers diameter must equal the quadratic brute-force maximum.
    #[test]
    fn matches_brute_force() {
        let mut state = 0x0C01D_u64;
        for _ in 0..400 {
            let hull = random_hull(&mut state, 25.0);
            let got = convex_diameter(&hull);
            let gold = brute_diameter(&hull);
            assert!(approx(got, gold), "calipers {got} vs brute {gold}");
        }
    }

    /// The returned pair's own distance must equal the reported diameter, and
    /// both members must be hull vertices.
    #[test]
    fn pair_realises_diameter() {
        let mut state = 0x1A1B2_u64;
        for _ in 0..300 {
            let hull = random_hull(&mut state, 20.0);
            let d = convex_diameter(&hull);
            let (a, b) = convex_diameter_pair(&hull).expect("hull has >= 2 vertices");
            assert!(approx(dist_sq(a, b).sqrt(), d), "pair distance != diameter");
            assert!(hull.contains(&a), "first endpoint not a vertex");
            assert!(hull.contains(&b), "second endpoint not a vertex");
        }
    }

    /// Translating the whole polygon leaves the diameter unchanged.
    #[test]
    fn translation_invariant() {
        let mut state = 0x2B3C4_u64;
        for _ in 0..300 {
            let hull = random_hull(&mut state, 18.0);
            let (tx, ty) = (rand_in(&mut state, -40.0, 40.0), rand_in(&mut state, -40.0, 40.0));
            let moved: Vec<(f32, f32)> = hull.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            assert!(approx(convex_diameter(&hull), convex_diameter(&moved)));
        }
    }

    /// Scaling the polygon by `k` scales the diameter by `k`.
    #[test]
    fn scale_covariant() {
        let mut state = 0x3C4D5_u64;
        for _ in 0..300 {
            let hull = random_hull(&mut state, 16.0);
            let k = rand_in(&mut state, 0.2, 5.0);
            let scaled: Vec<(f32, f32)> = hull.iter().map(|&(x, y)| (x * k, y * k)).collect();
            // `convex_hull` keeps counter-clockwise winding under positive scale.
            let scaled_hull = convex_hull(&scaled);
            assert!(approx(convex_diameter(&scaled_hull), k * convex_diameter(&hull)));
        }
    }

    /// Rotating the polygon by an arbitrary angle leaves the diameter unchanged
    /// (distances are rotation-invariant). `f64` trigonometry is allowed.
    #[test]
    fn rotation_invariant() {
        let mut state = 0x4D5E6_u64;
        for _ in 0..300 {
            let hull = random_hull(&mut state, 22.0);
            let theta = f64::from(rand_in(&mut state, -3.0, 3.0));
            let (s, c) = (theta.sin(), theta.cos());
            let rotated: Vec<(f32, f32)> = hull
                .iter()
                .map(|&(x, y)| {
                    let (xd, yd) = (f64::from(x), f64::from(y));
                    ((xd * c - yd * s) as f32, (xd * s + yd * c) as f32)
                })
                .collect();
            let rotated_hull = convex_hull(&rotated);
            assert!(approx(convex_diameter(&rotated_hull), convex_diameter(&hull)));
        }
    }

    #[test]
    fn fixed_cases() {
        // Empty and single-vertex inputs have no diameter.
        assert_eq!(convex_diameter(&[]), 0.0);
        assert_eq!(convex_diameter_pair(&[]), None);
        assert_eq!(convex_diameter(&[(3.0, 4.0)]), 0.0);
        assert_eq!(convex_diameter_pair(&[(3.0, 4.0)]), None);

        // A two-point "polygon" is just the segment length.
        assert!(approx(convex_diameter(&[(0.0, 0.0), (3.0, 4.0)]), 5.0));
        assert_eq!(
            convex_diameter_pair(&[(0.0, 0.0), (3.0, 4.0)]),
            Some(((0.0, 0.0), (3.0, 4.0)))
        );

        // Unit square: the diagonal is the diameter.
        let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(approx(convex_diameter(&unit), 2.0f32.sqrt()));

        // A wide, thin rectangle: the long diagonal dominates.
        let rect = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)];
        assert!(approx(convex_diameter(&rect), 40.0f32.sqrt()));
    }
}
