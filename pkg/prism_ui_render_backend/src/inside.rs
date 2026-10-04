//! Point-in-polygon testing by crossing number (ray casting).
//!
//! [`sd_polygon`](crate::polygon::sd_polygon) answers *how far* a point is from
//! a polygon, which is what anti-aliased coverage needs. Hit-testing — "did the
//! click land inside this polygonal control?" — only needs the boolean, and a
//! dedicated crossing-number test is both cheaper (no distance accumulation or
//! `sqrt`) and clearer at the call site than inspecting the sign of a distance.
//!
//! [`point_in_polygon`] casts a conceptual ray from the sample point and counts
//! how many polygon edges it crosses: an odd count means inside (the even-odd
//! fill rule). The winding order of the vertices does not matter, and the test
//! works for concave as well as convex simple (non-self-intersecting) polygons.
//!
//! Points exactly on an edge are a boundary case whose result depends on the
//! half-open crossing convention and floating-point rounding; callers that care
//! about the boundary should use [`sd_polygon`](crate::polygon::sd_polygon) and
//! compare against a tolerance instead.
//!
//! Only `+ - * /` and comparisons are used — no transcendental functions — so
//! the test is `no_std`-clean and bit-stable across targets.

/// Returns `true` when `p` lies inside the simple polygon `verts` under the
/// even-odd fill rule.
///
/// The polygon is treated as implicitly closed (the last vertex joins back to
/// the first) and may be concave; the vertex winding order is irrelevant.
/// Fewer than three vertices cannot bound an area, so the result is `false`.
/// Points lying exactly on an edge are an unspecified boundary case.
#[must_use]
pub fn point_in_polygon(verts: &[(f32, f32)], p: (f32, f32)) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let (x, y) = p;
    let mut inside = false;
    let mut j = n - 1;
    for (i, &(xi, yi)) in verts.iter().enumerate() {
        let (xj, yj) = verts[j];
        // The edge straddles the horizontal line `y` iff exactly one endpoint is
        // above it (half-open: strictly above). When it does, intersect the edge
        // with that line and flip the parity if the crossing is to the right of
        // `x`. The straddle guard guarantees `yj != yi`, so the divide is safe.
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::{area, convex_hull, polygon::sd_polygon};
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

    fn random_convex(state: &mut u64) -> Vec<(f32, f32)> {
        loop {
            let n = 8 + (next_rand(state) % 24) as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push((rand_in(state, -20.0, 20.0), rand_in(state, -20.0, 20.0)));
            }
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// Signed winding number of `verts` around `p` (Sunday's algorithm). This is
    /// direction-aware and algebraically distinct from even-odd parity, yet for
    /// a simple polygon `wn != 0` agrees with "inside" — an independent oracle.
    fn winding_number(verts: &[(f32, f32)], p: (f32, f32)) -> i32 {
        let n = verts.len();
        let (x, y) = p;
        let mut wn = 0_i32;
        for (i, &(ax, ay)) in verts.iter().enumerate() {
            let (bx, by) = verts[(i + 1) % n];
            // Cross product of edge a->b with a->p; >0 means p is left of a->b.
            let is_left = (bx - ax) * (y - ay) - (x - ax) * (by - ay);
            if ay <= y {
                if by > y && is_left > 0.0 {
                    wn += 1;
                }
            } else if by <= y && is_left < 0.0 {
                wn -= 1;
            }
        }
        wn
    }

    /// Inside test for a CCW convex polygon: the point must be left of (or on)
    /// every directed edge. Fully independent of ray casting.
    fn inside_convex_ccw(verts: &[(f32, f32)], p: (f32, f32)) -> bool {
        let n = verts.len();
        for (i, &(ax, ay)) in verts.iter().enumerate() {
            let (bx, by) = verts[(i + 1) % n];
            let cross = (bx - ax) * (p.1 - ay) - (by - ay) * (p.0 - ax);
            if cross < 0.0 {
                return false;
            }
        }
        true
    }

    #[test]
    fn too_few_vertices_is_outside() {
        assert!(!point_in_polygon(&[], (0.0, 0.0)));
        assert!(!point_in_polygon(&[(0.0, 0.0)], (0.0, 0.0)));
        assert!(!point_in_polygon(&[(0.0, 0.0), (1.0, 1.0)], (0.5, 0.5)));
    }

    #[test]
    fn unit_square_obvious_points() {
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(point_in_polygon(&sq, (0.5, 0.5)));
        assert!(point_in_polygon(&sq, (0.01, 0.99)));
        assert!(!point_in_polygon(&sq, (1.5, 0.5)));
        assert!(!point_in_polygon(&sq, (-0.1, 0.5)));
        assert!(!point_in_polygon(&sq, (0.5, 2.0)));
    }

    #[test]
    fn concave_shape_rejects_the_notch() {
        // A C-shape: the hollow on the right must read as outside even though it
        // sits within the bounding box.
        let c = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (4.0, 3.0),
            (4.0, 4.0),
            (0.0, 4.0),
        ];
        assert!(point_in_polygon(&c, (0.5, 2.0))); // in the spine
        assert!(point_in_polygon(&c, (2.0, 0.5))); // in the bottom arm
        assert!(!point_in_polygon(&c, (2.5, 2.0))); // in the notch
        assert!(!point_in_polygon(&c, (5.0, 2.0))); // right of everything
    }

    #[test]
    fn agrees_with_convex_sign_test() {
        // Strong exact oracle on convex polygons: even-odd vs all-edges-left.
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            for _ in 0..40 {
                let p = (rand_in(&mut state, -25.0, 25.0), rand_in(&mut state, -25.0, 25.0));
                if sd_polygon(p.0, p.1, &poly).abs() < 0.1 {
                    continue; // skip boundary-ambiguous samples
                }
                assert_eq!(
                    point_in_polygon(&poly, p),
                    inside_convex_ccw(&poly, p),
                    "disagreement at {p:?}"
                );
            }
        }
    }

    #[test]
    fn agrees_with_winding_number_on_concave_shapes() {
        // Even-odd vs the direction-aware winding number on fixed simple concave
        // polygons, sampled on a dense grid away from the boundary.
        let shapes: [&[(f32, f32)]; 3] = [
            // L-shape
            &[(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)],
            // Plus / cross
            &[
                (2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (6.0, 2.0), (6.0, 4.0), (4.0, 4.0),
                (4.0, 6.0), (2.0, 6.0), (2.0, 4.0), (0.0, 4.0), (0.0, 2.0), (2.0, 2.0),
            ],
            // Arrowhead (concave at the tail)
            &[(0.0, 0.0), (6.0, 3.0), (0.0, 6.0), (2.0, 3.0)],
        ];
        for shape in shapes {
            for gy in 0..60 {
                for gx in 0..60 {
                    let p = (gx as f32 * 0.1 + 0.03, gy as f32 * 0.1 + 0.07);
                    if sd_polygon(p.0, p.1, shape).abs() < 0.05 {
                        continue;
                    }
                    assert_eq!(
                        point_in_polygon(shape, p),
                        winding_number(shape, p) != 0,
                        "disagreement at {p:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn monte_carlo_area_matches_shoelace() {
        // Integrating the indicator over the bounding box must recover the
        // shoelace area, tying the predicate to measure::area independently.
        let mut state = 0xFACE_B00C_1234_5678_u64;
        for _ in 0..12 {
            let poly = random_convex(&mut state);
            let (mut minx, mut miny) = (f32::INFINITY, f32::INFINITY);
            let (mut maxx, mut maxy) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &(x, y) in &poly {
                minx = minx.min(x);
                miny = miny.min(y);
                maxx = maxx.max(x);
                maxy = maxy.max(y);
            }
            let bbox = (maxx - minx) * (maxy - miny);
            let samples = 40_000;
            let mut hits = 0_u32;
            for _ in 0..samples {
                let p = (rand_in(&mut state, minx, maxx), rand_in(&mut state, miny, maxy));
                if point_in_polygon(&poly, p) {
                    hits += 1;
                }
            }
            let estimate = bbox * (hits as f32 / samples as f32);
            let exact = area(&poly);
            assert!(
                (estimate - exact).abs() <= 0.05 * exact + 0.5,
                "monte carlo {estimate} vs shoelace {exact}"
            );
        }
    }

    #[test]
    fn convex_centroid_is_inside() {
        // Cross-module invariant: a convex polygon's centroid lies inside it.
        let mut state = 0x0BAD_C0DE_F00D_BEEF_u64;
        for _ in 0..200 {
            let poly = random_convex(&mut state);
            let c = crate::centroid(&poly).unwrap();
            assert!(point_in_polygon(&poly, c), "centroid {c:?} not inside");
        }
    }
}
