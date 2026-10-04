//! Closest point on a segment or polygon boundary to a query point.
//!
//! Snapping a cursor to the nearest edge, drag handles that cling to a shape's
//! outline, and leader lines that terminate on a badge all need the actual
//! *point* on the boundary, not just its distance. [`crate::polygon::sd_polygon`]
//! already yields the signed distance magnitude, but discards the witness
//! point; these routines recover it.
//!
//! [`closest_point_on_segment`] clamps the perpendicular projection to the
//! segment's endpoints. [`closest_point_on_polygon`] takes the nearest such
//! projection across every edge of a closed loop. Both use only `+ - * /` and
//! [`f32::clamp`], so they are `no_std`-clean and bit-stable across targets.

/// The point on segment `a..b` closest to `p`.
///
/// The perpendicular projection of `p` onto the infinite line through `a` and
/// `b`, clamped to the segment. A degenerate segment (`a == b`) returns `a`.
///
/// ```
/// use prism_ui_render_backend::nearest::closest_point_on_segment;
/// // Projecting onto a horizontal segment drops the y and clamps x.
/// let q = closest_point_on_segment((3.0, 5.0), (0.0, 0.0), (10.0, 0.0));
/// assert_eq!(q, (3.0, 0.0));
/// // A point past the far end clamps to that endpoint.
/// let q = closest_point_on_segment((20.0, 5.0), (0.0, 0.0), (10.0, 0.0));
/// assert_eq!(q, (10.0, 0.0));
/// ```
#[must_use]
pub fn closest_point_on_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let ex = b.0 - a.0;
    let ey = b.1 - a.1;
    let denom = ex * ex + ey * ey;
    if denom <= 0.0 {
        return a;
    }
    let t = (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / denom).clamp(0.0, 1.0);
    (a.0 + ex * t, a.1 + ey * t)
}

#[inline]
fn dist2(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// The point on the boundary of `verts` closest to `p`, or `None` for an empty
/// slice.
///
/// `verts` is a closed loop: the final vertex is implicitly joined back to the
/// first. The result is the nearest projection across all edges, so it always
/// lies exactly on the boundary (which equals the zero set of
/// [`crate::polygon::sd_polygon`]). The orientation of `verts` does not matter.
///
/// A single vertex returns that vertex; two vertices are treated as one
/// segment (not a degenerate loop).
///
/// ```
/// use prism_ui_render_backend::nearest::closest_point_on_polygon;
/// let square = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
/// // A point to the right of the square snaps onto the right edge.
/// let q = closest_point_on_polygon(&square, (12.0, 1.0)).unwrap();
/// assert_eq!(q, (5.0, 1.0));
/// ```
#[must_use]
pub fn closest_point_on_polygon(verts: &[(f32, f32)], p: (f32, f32)) -> Option<(f32, f32)> {
    let n = verts.len();
    match n {
        0 => return None,
        1 => return Some(verts[0]),
        _ => {}
    }

    // Two vertices describe a single open segment; three or more a closed loop.
    let edge_count = if n == 2 { 1 } else { n };

    let mut best = closest_point_on_segment(p, verts[0], verts[1 % n]);
    let mut best_d2 = dist2(p, best);
    for i in 1..edge_count {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        let q = closest_point_on_segment(p, a, b);
        let d2 = dist2(p, q);
        if d2 < best_d2 {
            best_d2 = d2;
            best = q;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::hull::convex_hull;
    use crate::polygon::sd_polygon;
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

    fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
        dist2(a, b).sqrt()
    }

    /// A convex polygon (via the crate's hull) of at least three vertices.
    fn random_polygon(state: &mut u64) -> Vec<(f32, f32)> {
        loop {
            let n = 8 + (next_rand(state) % 12) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(state, -30.0, 30.0), rand_in(state, -30.0, 30.0)))
                .collect();
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// Independent gold oracle: densely sample every edge and keep the closest
    /// sampled point. Shares no projection code with the implementation.
    fn brute_force(verts: &[(f32, f32)], p: (f32, f32)) -> (f32, f32) {
        let n = verts.len();
        let edge_count = if n == 2 { 1 } else { n };
        let mut best = verts[0];
        let mut best_d2 = dist2(p, best);
        for i in 0..edge_count {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            for step in 0..=2000u32 {
                let t = step as f32 / 2000.0;
                let q = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                let d2 = dist2(p, q);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = q;
                }
            }
        }
        best
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 2e-3 * (a.abs() + b.abs()) + 2e-3
    }

    /// Distance to the returned point must equal the dense brute-force minimum.
    #[test]
    fn matches_brute_force() {
        let mut state = 0x0C70_5E57_1234_0001_u64;
        for _ in 0..300 {
            let poly = random_polygon(&mut state);
            let p = (rand_in(&mut state, -45.0, 45.0), rand_in(&mut state, -45.0, 45.0));
            let got = closest_point_on_polygon(&poly, p).unwrap();
            let want = brute_force(&poly, p);
            assert!(
                approx(dist(p, got), dist(p, want)),
                "dist {} vs brute {}",
                dist(p, got),
                dist(p, want)
            );
        }
    }

    /// The returned point lies on the boundary (`|sd_polygon|` ~ 0) and its
    /// distance equals `|sd_polygon(p)|` — cross-checking the independent SDF.
    #[test]
    fn on_boundary_and_matches_sdf() {
        let mut state = 0x5DF0_1111_2222_3333_u64;
        for _ in 0..300 {
            let poly = random_polygon(&mut state);
            let p = (rand_in(&mut state, -45.0, 45.0), rand_in(&mut state, -45.0, 45.0));
            let q = closest_point_on_polygon(&poly, p).unwrap();
            // On the boundary.
            assert!(sd_polygon(q.0, q.1, &poly).abs() <= 5e-2, "off boundary");
            // Distance agrees with the signed-distance magnitude.
            let sdf = sd_polygon(p.0, p.1, &poly).abs();
            assert!(approx(dist(p, q), sdf), "dist {} vs sdf {}", dist(p, q), sdf);
        }
    }

    /// Projecting a point already on the boundary returns (essentially) itself.
    #[test]
    fn idempotent_on_boundary() {
        let mut state = 0xDEAD_BEEF_0F0F_7777_u64;
        for _ in 0..200 {
            let poly = random_polygon(&mut state);
            let p = (rand_in(&mut state, -45.0, 45.0), rand_in(&mut state, -45.0, 45.0));
            let q = closest_point_on_polygon(&poly, p).unwrap();
            let q2 = closest_point_on_polygon(&poly, q).unwrap();
            assert!(approx(q.0, q2.0) && approx(q.1, q2.1));
        }
    }

    /// Translating the polygon and query translates the witness point.
    #[test]
    fn translation_invariant() {
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        for _ in 0..200 {
            let poly = random_polygon(&mut state);
            let p = (rand_in(&mut state, -45.0, 45.0), rand_in(&mut state, -45.0, 45.0));
            let (tx, ty) = (rand_in(&mut state, -50.0, 50.0), rand_in(&mut state, -50.0, 50.0));
            let shifted: Vec<(f32, f32)> = poly.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            let q = closest_point_on_polygon(&poly, p).unwrap();
            let qs = closest_point_on_polygon(&shifted, (p.0 + tx, p.1 + ty)).unwrap();
            assert!(approx(q.0 + tx, qs.0) && approx(q.1 + ty, qs.1));
        }
    }

    #[test]
    fn segment_cases() {
        let a = (0.0, 0.0);
        let b = (10.0, 0.0);
        // Interior projection.
        assert_eq!(closest_point_on_segment((4.0, 7.0), a, b), (4.0, 0.0));
        // Clamp before the start.
        assert_eq!(closest_point_on_segment((-3.0, 2.0), a, b), (0.0, 0.0));
        // Clamp past the end.
        assert_eq!(closest_point_on_segment((99.0, 2.0), a, b), (10.0, 0.0));
        // Degenerate segment returns the shared endpoint.
        assert_eq!(closest_point_on_segment((5.0, 5.0), (2.0, 2.0), (2.0, 2.0)), (2.0, 2.0));
    }

    #[test]
    fn trivial_cases() {
        assert_eq!(closest_point_on_polygon(&[], (1.0, 1.0)), None);
        assert_eq!(closest_point_on_polygon(&[(3.0, -4.0)], (0.0, 0.0)), Some((3.0, -4.0)));

        // Two vertices: nearest point on the single segment.
        let seg = [(0.0, 0.0), (0.0, 10.0)];
        assert_eq!(closest_point_on_polygon(&seg, (7.0, 4.0)), Some((0.0, 4.0)));

        // Fixed concave L: a point just right of the lower arm snaps onto the
        // right edge at the same height (unambiguous nearest).
        let l = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 2.0),
            (2.0, 2.0),
            (2.0, 4.0),
            (0.0, 4.0),
        ];
        let q = closest_point_on_polygon(&l, (5.0, 1.0)).unwrap();
        assert!(approx(q.0, 4.0) && approx(q.1, 1.0), "got {q:?}");
    }
}
