//! Minimum-area oriented bounding box of a convex polygon.
//!
//! An axis-aligned box is cheap but wasteful for a tilted shape; the tightest
//! rectangle is usually rotated to hug the hull. The minimum-area *oriented*
//! bounding box (OBB) is that tightest enclosing rectangle at any orientation.
//! It is the workhorse of broad-phase culling, drag-handle framing, texture
//! atlas packing of rotated glyph runs, and snug hit-testing of tilted chrome.
//!
//! By the Freeman–Shapira theorem the minimum-area enclosing rectangle of a
//! convex polygon always has one side collinear with a polygon edge. So the
//! search space is finite: align a candidate rectangle to each hull edge,
//! project every vertex onto that edge direction and its perpendicular to get
//! the tight extents, and keep the smallest-area candidate. Each candidate is a
//! linear projection pass, giving an exact quadratic search over the hull — no
//! sampling, no approximation.
//!
//! The input must be a convex polygon in counter-clockwise order with no
//! repeated start vertex, exactly as produced by
//! [`convex_hull`](crate::hull::convex_hull). Only `+ - *`, comparisons and
//! `sqrt` are used, so the routine is `no_std`-clean and bit-stable across
//! targets.

/// An oriented bounding box: a rectangle at an arbitrary orientation.
///
/// `axis_u` and `axis_v` are orthonormal direction vectors (the box's local
/// axes); `half_u` and `half_v` are the half-lengths along them, so the box
/// spans `center ± half_u·axis_u ± half_v·axis_v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obb {
    /// Center of the rectangle.
    pub center: (f32, f32),
    /// Unit direction of the first (flush) axis.
    pub axis_u: (f32, f32),
    /// Unit direction of the second axis, perpendicular to `axis_u`.
    pub axis_v: (f32, f32),
    /// Half-length along `axis_u`.
    pub half_u: f32,
    /// Half-length along `axis_v`.
    pub half_v: f32,
}

impl Obb {
    /// Area of the rectangle: the product of its two full side lengths.
    #[inline]
    pub fn area(&self) -> f32 {
        4.0 * self.half_u * self.half_v
    }

    /// The four corner points, counter-clockwise in the box's local frame.
    pub fn corners(&self) -> [(f32, f32); 4] {
        let (cx, cy) = self.center;
        let (ux, uy) = self.axis_u;
        let (vx, vy) = self.axis_v;
        let hu = self.half_u;
        let hv = self.half_v;
        [
            (cx - ux * hu - vx * hv, cy - uy * hu - vy * hv),
            (cx + ux * hu - vx * hv, cy + uy * hu - vy * hv),
            (cx + ux * hu + vx * hv, cy + uy * hu + vy * hv),
            (cx - ux * hu + vx * hv, cy - uy * hu + vy * hv),
        ]
    }
}

/// The minimum-area oriented bounding box of a convex polygon.
///
/// `hull` must be a convex polygon in counter-clockwise order with no repeated
/// start vertex, as returned by [`convex_hull`](crate::hull::convex_hull).
/// Returns [`None`] for an empty input. A single point yields a zero-area box
/// at that point with the default axes; two points yield a zero-area box flush
/// with the segment.
pub fn min_area_obb(hull: &[(f32, f32)]) -> Option<Obb> {
    let n = hull.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(Obb {
            center: hull[0],
            axis_u: (1.0, 0.0),
            axis_v: (0.0, 1.0),
            half_u: 0.0,
            half_v: 0.0,
        });
    }

    let mut best: Option<Obb> = None;
    for i in 0..n {
        let ni = (i + 1) % n;
        let ex = hull[ni].0 - hull[i].0;
        let ey = hull[ni].1 - hull[i].1;
        let len = (ex * ex + ey * ey).sqrt();
        if len <= 0.0 {
            continue;
        }
        // Orthonormal frame flush with this edge.
        let u = (ex / len, ey / len);
        let v = (-u.1, u.0);

        let mut umin = f32::INFINITY;
        let mut umax = f32::NEG_INFINITY;
        let mut vmin = f32::INFINITY;
        let mut vmax = f32::NEG_INFINITY;
        for &p in hull {
            let du = p.0 * u.0 + p.1 * u.1;
            let dv = p.0 * v.0 + p.1 * v.1;
            umin = umin.min(du);
            umax = umax.max(du);
            vmin = vmin.min(dv);
            vmax = vmax.max(dv);
        }

        let half_u = (umax - umin) * 0.5;
        let half_v = (vmax - vmin) * 0.5;
        let area = (umax - umin) * (vmax - vmin);
        let replace = match &best {
            Some(b) => area < b.area(),
            None => true,
        };
        if replace {
            // Rebuild the center from its (u, v) coordinates.
            let uc = (umin + umax) * 0.5;
            let vc = (vmin + vmax) * 0.5;
            best = Some(Obb {
                center: (u.0 * uc + v.0 * vc, u.1 * uc + v.1 * vc),
                axis_u: u,
                axis_v: v,
                half_u,
                half_v,
            });
        }
    }

    best
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::hull::convex_hull;
    use crate::measure::area as poly_area;
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

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 3e-3 * (a.abs() + b.abs()) + 3e-3
    }

    /// Every hull vertex lies inside (or on) the returned box.
    fn contains_all(obb: &Obb, hull: &[(f32, f32)]) -> bool {
        hull.iter().all(|&p| {
            let dx = p.0 - obb.center.0;
            let dy = p.1 - obb.center.1;
            let du = dx * obb.axis_u.0 + dy * obb.axis_u.1;
            let dv = dx * obb.axis_v.0 + dy * obb.axis_v.1;
            du.abs() <= obb.half_u + 1e-2 && dv.abs() <= obb.half_v + 1e-2
        })
    }

    /// Independent oracle: sweep a dense grid of orientations, and for each
    /// compute the axis-projected enclosing-rectangle area. Returns the minimum
    /// found. `f64` trigonometry is allowed in tests.
    fn sampled_min_area(hull: &[(f32, f32)], steps: u32) -> f32 {
        let mut best = f32::INFINITY;
        for k in 0..steps {
            let theta = core::f64::consts::PI * f64::from(k) / f64::from(steps);
            let (s, c) = (theta.sin(), theta.cos());
            let (ux, uy) = (c as f32, s as f32);
            let (vx, vy) = (-uy, ux);
            let mut umin = f32::INFINITY;
            let mut umax = f32::NEG_INFINITY;
            let mut vmin = f32::INFINITY;
            let mut vmax = f32::NEG_INFINITY;
            for &p in hull {
                let du = p.0 * ux + p.1 * uy;
                let dv = p.0 * vx + p.1 * vy;
                umin = umin.min(du);
                umax = umax.max(du);
                vmin = vmin.min(dv);
                vmax = vmax.max(dv);
            }
            best = best.min((umax - umin) * (vmax - vmin));
        }
        best
    }

    /// The exact box must contain the hull and be no larger than any sampled
    /// orientation (and the fine grid cannot beat the exact optimum by more
    /// than its angular resolution).
    #[test]
    fn matches_sampled_minimum() {
        let mut state = 0x0_B0B1_u64;
        for _ in 0..60 {
            let hull = random_hull(&mut state, 25.0);
            let obb = min_area_obb(&hull).expect("non-empty hull");
            assert!(contains_all(&obb, &hull), "box does not contain hull");
            let sampled = sampled_min_area(&hull, 2048);
            // Exact optimum is a lower bound on any sampled orientation.
            assert!(obb.area() <= sampled + 1e-2, "exact {} > sampled {sampled}", obb.area());
            // The dense grid lands near an edge direction, so it cannot be far
            // above the exact optimum.
            assert!(sampled <= obb.area() * 1.02 + 1e-1, "sampled {sampled} >> exact {}", obb.area());
        }
    }

    /// The box must enclose the polygon, so its area is at least the polygon's.
    #[test]
    fn area_bounds_polygon() {
        let mut state = 0x1_C1C2_u64;
        for _ in 0..200 {
            let hull = random_hull(&mut state, 20.0);
            let obb = min_area_obb(&hull).expect("non-empty hull");
            assert!(obb.area() >= poly_area(&hull) - 1e-2, "box smaller than polygon");
        }
    }

    /// The minimum-area box is at most the axis-aligned box (one orientation).
    #[test]
    fn not_worse_than_aabb() {
        let mut state = 0x2_D2D3_u64;
        for _ in 0..200 {
            let hull = random_hull(&mut state, 18.0);
            let obb = min_area_obb(&hull).expect("non-empty hull");
            let mut xmin = f32::INFINITY;
            let mut xmax = f32::NEG_INFINITY;
            let mut ymin = f32::INFINITY;
            let mut ymax = f32::NEG_INFINITY;
            for &(x, y) in &hull {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
            let aabb = (xmax - xmin) * (ymax - ymin);
            assert!(obb.area() <= aabb + 1e-2, "obb {} > aabb {aabb}", obb.area());
        }
    }

    /// Rotating the polygon rotates the box but preserves its area and shape.
    #[test]
    fn rotation_preserves_area() {
        let mut state = 0x3_E3E4_u64;
        for _ in 0..200 {
            let hull = random_hull(&mut state, 22.0);
            let base = min_area_obb(&hull).expect("non-empty hull");
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
            let after = min_area_obb(&rotated_hull).expect("non-empty hull");
            assert!(approx(base.area(), after.area()), "area {} vs {}", base.area(), after.area());
        }
    }

    #[test]
    fn fixed_cases() {
        assert_eq!(min_area_obb(&[]), None);

        let point = min_area_obb(&[(2.0, 5.0)]).expect("point");
        assert_eq!(point.area(), 0.0);
        assert_eq!(point.center, (2.0, 5.0));

        // Axis-aligned rectangle: the box is the rectangle itself.
        let rect = [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (0.0, 2.0)];
        let obb = min_area_obb(&rect).expect("rect");
        assert!(approx(obb.area(), 12.0), "rect area {}", obb.area());
        assert!(contains_all(&obb, &rect));

        // A 45-degree-tilted square (diagonals of length 2, side sqrt(2)): the
        // tight box is the square itself (area 2), half its axis-aligned box
        // (area 4).
        let diamond = [(1.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0)];
        let obb = min_area_obb(&diamond).expect("diamond");
        assert!(approx(obb.area(), 2.0), "diamond obb area {}", obb.area());
        assert!(contains_all(&obb, &diamond));
    }
}
