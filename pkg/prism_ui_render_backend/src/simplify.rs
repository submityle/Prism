//! Ramer-Douglas-Peucker polyline simplification.
//!
//! [`CubicBezier::flatten`](crate::CubicBezier::flatten) turns a curve into a
//! dense polyline sized for the *tightest* tolerance a frame might need. Once
//! several such paths are concatenated — or when a stroke is re-tessellated at
//! a coarser zoom — that density is wasteful: long near-straight runs carry far
//! more vertices than the raster or [`sd_polygon`](crate::polygon::sd_polygon)
//! stage can perceive.
//!
//! [`simplify`] removes those redundant vertices with the Ramer-Douglas-Peucker
//! algorithm: keep the two endpoints, find the interior vertex farthest from
//! the chord between them, and recurse into the two halves only while that
//! distance exceeds `epsilon`. Every discarded vertex is therefore guaranteed
//! to lie within `epsilon` of the retained polyline, which is exactly the
//! fidelity contract a renderer needs.
//!
//! The sweep uses an explicit index stack rather than call recursion, so even a
//! pathological input that keeps every vertex cannot overflow the stack. All
//! arithmetic is pure `f32` (`+ - * /` and [`f32::sqrt`]); there are no
//! transcendental calls, so the routine is `no_std`-clean and bit-stable.

use alloc::vec;
use alloc::vec::Vec;

/// Distance from `p` to the line segment `a..=b`, clamping the projection to
/// the segment so overshoot past an endpoint is measured honestly. Falls back
/// to the point-to-`a` distance for a degenerate zero-length segment.
fn point_segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ex = b.0 - a.0;
    let ey = b.1 - a.1;
    let len2 = ex * ex + ey * ey;
    let t = if len2 > 0.0 {
        ((p.0 - a.0) * ex + (p.1 - a.1) * ey) / len2
    } else {
        0.0
    };
    let t = t.clamp(0.0, 1.0);
    let dx = p.0 - a.0 - ex * t;
    let dy = p.1 - a.1 - ey * t;
    (dx * dx + dy * dy).sqrt()
}

/// Simplifies a polyline, dropping every vertex that lies within `epsilon` of
/// the retained path (Ramer-Douglas-Peucker).
///
/// The first and last vertices are always kept, the output is an ordered
/// subsequence of the input (vertices are copied, never moved or synthesised),
/// and the returned polyline always has at least two vertices when the input
/// does. A larger `epsilon` keeps fewer vertices; a non-positive `epsilon` is
/// treated as `0`, which drops only vertices that are exactly collinear with
/// (or duplicated on) their retained neighbours.
///
/// Inputs of two or fewer vertices are returned unchanged.
#[must_use]
pub fn simplify(points: &[(f32, f32)], epsilon: f32) -> Vec<(f32, f32)> {
    let n = points.len();
    if n <= 2 {
        return points.to_vec();
    }
    let eps = if epsilon > 0.0 { epsilon } else { 0.0 };

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // Each stack entry is an inclusive index span whose interior is still to be
    // examined. Popping until empty visits every span the recursion would.
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];
    while let Some((lo, hi)) = stack.pop() {
        if hi <= lo + 1 {
            continue;
        }
        let mut farthest = lo;
        let mut max_dist = -1.0_f32;
        for (offset, &p) in points[lo + 1..hi].iter().enumerate() {
            let d = point_segment_distance(p, points[lo], points[hi]);
            if d > max_dist {
                max_dist = d;
                farthest = lo + 1 + offset;
            }
        }
        if max_dist > eps {
            keep[farthest] = true;
            stack.push((lo, farthest));
            stack.push((farthest, hi));
        }
    }

    points
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| keep[i].then_some(p))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;

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

    /// Independent oracle: shortest distance from `p` to a whole polyline.
    fn point_polyline_distance(p: (f32, f32), poly: &[(f32, f32)]) -> f32 {
        let mut best = f32::INFINITY;
        for w in poly.windows(2) {
            best = best.min(point_segment_distance(p, w[0], w[1]));
        }
        best
    }

    /// A moderately wiggly random walk — the realistic shape RDP operates on.
    fn random_walk(state: &mut u64, n: usize) -> Vec<(f32, f32)> {
        let mut out = Vec::with_capacity(n);
        let mut x = 0.0_f32;
        let mut y = 0.0_f32;
        for _ in 0..n {
            x += rand_in(state, -1.0, 3.0);
            y += rand_in(state, -2.0, 2.0);
            out.push((x, y));
        }
        out
    }

    fn is_ordered_subsequence(sub: &[(f32, f32)], full: &[(f32, f32)]) -> bool {
        let mut it = full.iter();
        sub.iter().all(|s| it.any(|f| f == s))
    }

    #[test]
    fn short_inputs_pass_through() {
        assert_eq!(simplify(&[], 1.0), vec![]);
        assert_eq!(simplify(&[(1.0, 2.0)], 1.0), vec![(1.0, 2.0)]);
        assert_eq!(
            simplify(&[(1.0, 2.0), (3.0, 4.0)], 1.0),
            vec![(1.0, 2.0), (3.0, 4.0)]
        );
    }

    #[test]
    fn collinear_run_collapses_to_endpoints() {
        let pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        assert_eq!(simplify(&pts, 0.0), vec![(0.0, 0.0), (4.0, 0.0)]);
    }

    #[test]
    fn sharp_corner_is_preserved() {
        // A clear spike well above epsilon must survive.
        let pts = [(0.0, 0.0), (5.0, 10.0), (10.0, 0.0)];
        assert_eq!(simplify(&pts, 0.5), vec![(0.0, 0.0), (5.0, 10.0), (10.0, 0.0)]);
    }

    #[test]
    fn endpoints_always_survive() {
        let mut state = 0x1357_9BDF_2468_ACE0_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 60) as usize;
            let pts = random_walk(&mut state, n);
            let simp = simplify(&pts, rand_in(&mut state, 0.1, 5.0));
            assert_eq!(*simp.first().unwrap(), *pts.first().unwrap());
            assert_eq!(*simp.last().unwrap(), *pts.last().unwrap());
            assert!(simp.len() >= 2);
        }
    }

    #[test]
    fn output_is_ordered_subsequence() {
        let mut state = 0x2468_ACE0_1357_9BDF_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 60) as usize;
            let pts = random_walk(&mut state, n);
            let simp = simplify(&pts, rand_in(&mut state, 0.1, 5.0));
            assert!(is_ordered_subsequence(&simp, &pts));
            assert!(simp.len() <= pts.len());
        }
    }

    #[test]
    fn every_original_point_within_epsilon() {
        // The RDP contract: no original vertex is farther than epsilon from the
        // simplified polyline. Independent recompute via point-to-polyline dist.
        let mut state = 0xF00D_1234_5678_9ABC_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 80) as usize;
            let pts = random_walk(&mut state, n);
            let eps = rand_in(&mut state, 0.2, 4.0);
            let simp = simplify(&pts, eps);
            for &p in &pts {
                let d = point_polyline_distance(p, &simp);
                assert!(d <= eps + 1e-3, "point {p:?} is {d} from simplified (eps {eps})");
            }
        }
    }

    #[test]
    fn larger_epsilon_never_adds_points() {
        let mut state = 0x9BDF_1357_ACE0_2468_u64;
        for _ in 0..200 {
            let n = 4 + (next_rand(&mut state) % 80) as usize;
            let pts = random_walk(&mut state, n);
            let coarse = simplify(&pts, 4.0).len();
            let fine = simplify(&pts, 0.2).len();
            assert!(coarse <= fine, "coarse {coarse} > fine {fine}");
        }
    }

    #[test]
    fn simplification_is_idempotent() {
        // Re-running at the same epsilon is a fixed point.
        let mut state = 0xABCD_1234_DCBA_4321_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 80) as usize;
            let pts = random_walk(&mut state, n);
            let eps = rand_in(&mut state, 0.2, 4.0);
            let once = simplify(&pts, eps);
            let twice = simplify(&once, eps);
            assert_eq!(once, twice);
        }
    }
}
