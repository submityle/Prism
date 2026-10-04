//! Minimum enclosing circle (smallest bounding circle) of a point set.
//!
//! Culling, focus rings, and broad-phase bounds all want the *tightest* circle
//! that contains a set of points — smaller than the bounding box's circumcircle
//! and rotation-invariant. [`min_enclosing_circle`] computes it exactly (up to
//! `f32` rounding) using Welzl's incremental algorithm.
//!
//! The minimal circle is always pinned by at most three of the input points on
//! its boundary. The algorithm grows a candidate circle one point at a time:
//! whenever a point falls outside the current circle it must lie on the new
//! circle's boundary, which fixes one, two, or three boundary points and
//! rebuilds the circle from them. Processing points in order is correct for any
//! input (the randomized variant only improves the expected running time).
//!
//! Arithmetic uses `+ - * /` plus `sqrt` (which is permitted; only the
//! transcendental `f32` functions are banned), so the routine is `no_std`-clean.

/// A circle defined by its `center` and `radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    /// Centre point.
    pub center: (f32, f32),
    /// Radius; `0.0` for a single-point set.
    pub radius: f32,
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// Whether `p` lies within circle `c`, with a small relative tolerance so that
/// the points defining the circle reliably test as enclosed under `f32`
/// rounding.
fn in_circle(c: &Circle, p: (f32, f32)) -> bool {
    dist(c.center, p) <= c.radius + c.radius * 1e-4 + 1e-4
}

/// Smallest circle through two points: the circle whose diameter is the segment
/// `a..b`.
fn from_two(a: (f32, f32), b: (f32, f32)) -> Circle {
    let center = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
    Circle { center, radius: dist(a, b) * 0.5 }
}

/// Circumcircle of three points. Falls back to the diameter circle of the two
/// farthest-apart points when the triple is (near-)collinear.
fn from_three(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> Circle {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d == 0.0 {
        // Collinear: the enclosing circle is pinned by the extreme pair.
        let ab = dist(a, b);
        let bc = dist(b, c);
        let ca = dist(c, a);
        if ab >= bc && ab >= ca {
            return from_two(a, b);
        } else if bc >= ca {
            return from_two(b, c);
        }
        return from_two(c, a);
    }
    let a2 = a.0 * a.0 + a.1 * a.1;
    let b2 = b.0 * b.0 + b.1 * b.1;
    let c2 = c.0 * c.0 + c.1 * c.1;
    let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
    let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
    let center = (ux, uy);
    Circle { center, radius: dist(center, a) }
}

/// Returns the smallest circle enclosing every point in `points`, or `None` for
/// an empty input.
///
/// A single point yields a radius-`0.0` circle centred on it. The result is
/// exact up to `f32` rounding and is independent of translation and rotation of
/// the inputs. Worst-case running time is cubic in the point count, but typical
/// inputs settle in roughly linear time; UI point sets are small either way.
#[must_use]
pub fn min_enclosing_circle(points: &[(f32, f32)]) -> Option<Circle> {
    let first = *points.first()?;
    let mut c = Circle { center: first, radius: 0.0 };

    for (i, &pi) in points.iter().enumerate() {
        if in_circle(&c, pi) {
            continue;
        }
        // pi must lie on the boundary of the updated circle.
        c = Circle { center: pi, radius: 0.0 };
        for (j, &pj) in points[..i].iter().enumerate() {
            if in_circle(&c, pj) {
                continue;
            }
            // pi and pj are both on the boundary.
            c = from_two(pi, pj);
            for &pk in &points[..j] {
                if in_circle(&c, pk) {
                    continue;
                }
                // pi, pj, pk are all on the boundary.
                c = from_three(pi, pj, pk);
            }
        }
    }

    Some(c)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
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

    fn encloses_all(c: &Circle, pts: &[(f32, f32)]) -> bool {
        pts.iter().all(|&p| in_circle(c, p))
    }

    /// Independent gold oracle: the minimal enclosing circle is pinned by two or
    /// three boundary points, so brute-forcing all pairs and triples (keeping the
    /// smallest circle that still encloses everything) gives the exact radius.
    fn brute_force(pts: &[(f32, f32)]) -> Option<Circle> {
        match pts.len() {
            0 => return None,
            1 => return Some(Circle { center: pts[0], radius: 0.0 }),
            _ => {}
        }
        let mut best: Option<Circle> = None;
        let consider = |c: Circle, best: &mut Option<Circle>| {
            if encloses_all(&c, pts) && best.is_none_or(|b| c.radius < b.radius) {
                *best = Some(c);
            }
        };
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                consider(from_two(pts[i], pts[j]), &mut best);
                for k in (j + 1)..pts.len() {
                    consider(from_three(pts[i], pts[j], pts[k]), &mut best);
                }
            }
        }
        best
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 2e-3 * (a.abs() + b.abs()) + 2e-3
    }

    /// Welzl's radius must match the brute-force minimal radius.
    #[test]
    fn matches_brute_force() {
        let mut state = 0x5EED_1234_ABCD_0001_u64;
        for _ in 0..500 {
            let n = 1 + (next_rand(&mut state) % 11) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(&mut state, -30.0, 30.0), rand_in(&mut state, -30.0, 30.0)))
                .collect();
            let got = min_enclosing_circle(&pts).unwrap();
            let want = brute_force(&pts).unwrap();
            assert!(
                approx(got.radius, want.radius),
                "radius {} vs brute {}",
                got.radius,
                want.radius
            );
            // Welzl's circle must still enclose everything.
            assert!(encloses_all(&got, &pts), "welzl circle misses a point");
        }
    }

    /// The circle encloses every point and at least two points sit on (near) its
    /// boundary for a non-trivial set.
    #[test]
    fn tight_and_enclosing() {
        let mut state = 0xBEEF_0F0F_7777_2222_u64;
        for _ in 0..300 {
            let n = 2 + (next_rand(&mut state) % 20) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(&mut state, -25.0, 25.0), rand_in(&mut state, -25.0, 25.0)))
                .collect();
            let c = min_enclosing_circle(&pts).unwrap();
            assert!(encloses_all(&c, &pts));
            let on_boundary = pts
                .iter()
                .filter(|&&p| (dist(c.center, p) - c.radius).abs() <= 1e-2 * c.radius + 1e-2)
                .count();
            assert!(on_boundary >= 2, "only {on_boundary} boundary points");
        }
    }

    /// Translating the inputs translates the centre and leaves the radius fixed.
    #[test]
    fn translation_invariant() {
        let mut state = 0x1111_2222_3333_4444_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 15) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(&mut state, -20.0, 20.0), rand_in(&mut state, -20.0, 20.0)))
                .collect();
            let (tx, ty) = (rand_in(&mut state, -40.0, 40.0), rand_in(&mut state, -40.0, 40.0));
            let shifted: Vec<(f32, f32)> = pts.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            let a = min_enclosing_circle(&pts).unwrap();
            let b = min_enclosing_circle(&shifted).unwrap();
            assert!(approx(a.radius, b.radius));
            assert!(approx(a.center.0 + tx, b.center.0) && approx(a.center.1 + ty, b.center.1));
        }
    }

    /// Scaling the inputs about the origin scales the radius by the same factor.
    #[test]
    fn scale_covariant() {
        let mut state = 0xABCD_EF01_2345_6789_u64;
        for _ in 0..200 {
            let n = 3 + (next_rand(&mut state) % 15) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(&mut state, -15.0, 15.0), rand_in(&mut state, -15.0, 15.0)))
                .collect();
            let k = rand_in(&mut state, 0.25, 4.0);
            let scaled: Vec<(f32, f32)> = pts.iter().map(|&(x, y)| (x * k, y * k)).collect();
            let a = min_enclosing_circle(&pts).unwrap();
            let b = min_enclosing_circle(&scaled).unwrap();
            assert!(approx(a.radius * k, b.radius), "{} vs {}", a.radius * k, b.radius);
        }
    }

    #[test]
    fn trivial_cases() {
        assert_eq!(min_enclosing_circle(&[]), None);

        let single = min_enclosing_circle(&[(3.0, -4.0)]).unwrap();
        assert_eq!(single.center, (3.0, -4.0));
        assert_eq!(single.radius, 0.0);

        let pair = min_enclosing_circle(&[(0.0, 0.0), (4.0, 0.0)]).unwrap();
        assert!(approx(pair.center.0, 2.0) && approx(pair.center.1, 0.0));
        assert!(approx(pair.radius, 2.0));

        // Unit square: centre at (0,0), radius = half-diagonal = sqrt(2).
        let square = min_enclosing_circle(&[(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
            .unwrap();
        assert!(approx(square.center.0, 0.0) && approx(square.center.1, 0.0));
        assert!(approx(square.radius, core::f32::consts::SQRT_2));
    }
}
