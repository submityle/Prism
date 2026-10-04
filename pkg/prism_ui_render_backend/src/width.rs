//! Minimum width of a convex polygon via rotating calipers.
//!
//! The *width* of a convex polygon is the thinnest parallel-sided slab that
//! still contains it: the smallest distance you could squeeze two parallel
//! lines to while trapping the whole shape between them. It is the natural
//! complement to the [diameter](crate::calipers::convex_diameter) — where the
//! diameter is the longest span, the width is the shortest. It drives tasks
//! like checking whether a convex selection fits through a gutter, picking the
//! short dimension of an oriented bounding slab, or rejecting degenerate
//! near-sliver hulls.
//!
//! For a convex polygon the minimum-width slab always has one side flush with a
//! polygon edge, so the width is the minimum, over every edge, of the greatest
//! perpendicular distance from that edge's supporting line to any vertex. A
//! naive evaluation is quadratic. The rotating calipers sweep instead advances
//! a single antipodal vertex forward in lockstep with each edge, visiting only
//! the supporting vertex per edge and so running in time linear in the vertex
//! count.
//!
//! The input must be a convex polygon in counter-clockwise order with no
//! repeated start vertex, exactly as produced by
//! [`convex_hull`](crate::hull::convex_hull). Only `+ - *`, comparisons and
//! `sqrt` are used, so the routine is `no_std`-clean and bit-stable across
//! targets.

/// Twice the signed area of triangle `a b c`; equivalently the cross product
/// `(b - a) × (c - a)`. Its magnitude divided by the length of edge `a -> b` is
/// the perpendicular distance of `c` from the line through `a` and `b`.
#[inline]
fn tri_area2(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Euclidean distance between two points.
#[inline]
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// The minimum width of a convex polygon: the thinnest slab of parallel lines
/// that contains it.
///
/// `hull` must be a convex polygon in counter-clockwise order with no repeated
/// start vertex, as returned by [`convex_hull`](crate::hull::convex_hull).
/// Degenerate inputs with fewer than three vertices (a point or a segment) fit
/// in a zero-width slab and yield `0.0`.
pub fn convex_width(hull: &[(f32, f32)]) -> f32 {
    let n = hull.len();
    if n < 3 {
        return 0.0;
    }

    let mut best = f32::INFINITY;
    // `j` is the vertex farthest from edge `i -> i + 1`; it only ever advances,
    // so the total work across every edge is linear. `budget` caps the sweep at
    // one full revolution as a defensive guard against malformed (non-convex)
    // input, and is never reached for a valid strictly-convex hull.
    let mut j = 1;
    let mut budget = 2 * n;
    for i in 0..n {
        let ni = (i + 1) % n;
        while budget > 0 {
            let nj = (j + 1) % n;
            if tri_area2(hull[i], hull[ni], hull[nj]).abs()
                > tri_area2(hull[i], hull[ni], hull[j]).abs()
            {
                j = nj;
                budget -= 1;
            } else {
                break;
            }
        }
        let edge_len = dist(hull[i], hull[ni]);
        if edge_len > 0.0 {
            let w = tri_area2(hull[i], hull[ni], hull[j]).abs() / edge_len;
            best = best.min(w);
        }
    }

    if best.is_finite() {
        best
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::calipers::convex_diameter;
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

    /// Independent gold oracle: for every edge take the greatest perpendicular
    /// distance to any vertex, then take the minimum over all edges.
    fn brute_width(hull: &[(f32, f32)]) -> f32 {
        let n = hull.len();
        if n < 3 {
            return 0.0;
        }
        let mut best = f32::INFINITY;
        for i in 0..n {
            let a = hull[i];
            let b = hull[(i + 1) % n];
            let edge_len = dist(a, b);
            if edge_len <= 0.0 {
                continue;
            }
            let mut far = 0.0f32;
            for &c in hull {
                far = far.max(tri_area2(a, b, c).abs() / edge_len);
            }
            best = best.min(far);
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 2e-3 * (a.abs() + b.abs()) + 2e-3
    }

    /// The calipers width must equal the quadratic brute-force minimum.
    #[test]
    fn matches_brute_force() {
        let mut state = 0x5_1D7A_u64;
        for _ in 0..400 {
            let hull = random_hull(&mut state, 25.0);
            let got = convex_width(&hull);
            let gold = brute_width(&hull);
            assert!(approx(got, gold), "calipers {got} vs brute {gold}");
        }
    }

    /// The width never exceeds the diameter (the thin span is at most the long
    /// span): a cross-check against an independent calipers routine.
    #[test]
    fn width_not_greater_than_diameter() {
        let mut state = 0x6_2E8B_u64;
        for _ in 0..400 {
            let hull = random_hull(&mut state, 20.0);
            let w = convex_width(&hull);
            let d = convex_diameter(&hull);
            assert!(w <= d + 1e-3, "width {w} exceeds diameter {d}");
        }
    }

    /// Translating the whole polygon leaves the width unchanged.
    #[test]
    fn translation_invariant() {
        let mut state = 0x7_3F9C_u64;
        for _ in 0..300 {
            let hull = random_hull(&mut state, 18.0);
            let (tx, ty) = (rand_in(&mut state, -40.0, 40.0), rand_in(&mut state, -40.0, 40.0));
            let moved: Vec<(f32, f32)> = hull.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            assert!(approx(convex_width(&hull), convex_width(&moved)));
        }
    }

    /// Scaling the polygon by `k` scales the width by `k`.
    #[test]
    fn scale_covariant() {
        let mut state = 0x8_40AD_u64;
        for _ in 0..300 {
            let hull = random_hull(&mut state, 16.0);
            let k = rand_in(&mut state, 0.2, 5.0);
            let scaled: Vec<(f32, f32)> = hull.iter().map(|&(x, y)| (x * k, y * k)).collect();
            let scaled_hull = convex_hull(&scaled);
            assert!(approx(convex_width(&scaled_hull), k * convex_width(&hull)));
        }
    }

    /// Rotating the polygon leaves the width unchanged. `f64` trigonometry is
    /// allowed.
    #[test]
    fn rotation_invariant() {
        let mut state = 0x9_51BE_u64;
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
            assert!(approx(convex_width(&rotated_hull), convex_width(&hull)));
        }
    }

    #[test]
    fn fixed_cases() {
        // Degenerate inputs fit in a zero-width slab.
        assert_eq!(convex_width(&[]), 0.0);
        assert_eq!(convex_width(&[(3.0, 4.0)]), 0.0);
        assert_eq!(convex_width(&[(0.0, 0.0), (3.0, 4.0)]), 0.0);

        // Unit square: both slab orientations give width 1.
        let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(approx(convex_width(&unit), 1.0));

        // Wide thin rectangle: the short side is the width.
        let rect = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)];
        assert!(approx(convex_width(&rect), 2.0));

        // 3-4-5 right triangle: the thinnest slab is flush with the hypotenuse,
        // whose supporting height is 2*area / 5 = 12 / 5.
        let tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)];
        assert!(approx(convex_width(&tri), 12.0 / 5.0));
    }
}
