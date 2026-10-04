//! Minkowski sum of two convex polygons.
//!
//! The Minkowski sum `A ⊕ B = { a + b : a ∈ A, b ∈ B }` is the workhorse of
//! collision inflation and motion planning: growing a UI drop target by the
//! shape of the dragged chrome, computing the swept region of a moving badge,
//! or building the configuration-space obstacle a cursor must avoid. For two
//! convex polygons the sum is again convex and has at most `n + m` edges.
//!
//! [`minkowski_sum`] merges the two edge rings by polar angle in linear time
//! (the classic convex-convex algorithm). Because both operands are convex and
//! walked counter-clockwise from their lowest vertex, their edge directions are
//! each monotonically increasing in angle, so a single simultaneous sweep emits
//! the summed boundary directly.
//!
//! Only `+ - *` and comparisons are used (polar order is decided by a
//! cross-product sign, never a transcendental), so the routine is
//! `no_std`-clean and bit-stable across targets.

use crate::hull::convex_hull;
use crate::measure::signed_area;
use alloc::vec::Vec;

/// Cross product of 2D vectors `u × v`; its sign orders the two by polar angle.
#[inline]
fn cross2(u: (f32, f32), v: (f32, f32)) -> f32 {
    u.0 * v.1 - u.1 * v.0
}

/// Edge vector leaving vertex `k` of the closed ring `poly`.
#[inline]
fn edge(poly: &[(f32, f32)], k: usize) -> (f32, f32) {
    let next = poly[(k + 1) % poly.len()];
    (next.0 - poly[k].0, next.1 - poly[k].1)
}

/// Reorders a convex polygon into counter-clockwise order beginning at its
/// lowest vertex (smallest `y`, ties broken by smallest `x`), so its edge
/// directions sweep monotonically through increasing polar angle.
fn to_ccw_from_lowest(poly: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut v = poly.to_vec();
    if signed_area(&v) < 0.0 {
        v.reverse();
    }
    let mut start = 0;
    for (k, &p) in v.iter().enumerate().skip(1) {
        let s = v[start];
        if p.1 < s.1 || (p.1 == s.1 && p.0 < s.0) {
            start = k;
        }
    }
    v.rotate_left(start);
    v
}

/// All pairwise vertex sums, used as the degenerate fallback when an operand is
/// a point or segment (fewer than three vertices bound no area).
fn pairwise_hull(a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut sums = Vec::with_capacity(a.len() * b.len());
    for &pa in a {
        for &pb in b {
            sums.push((pa.0 + pb.0, pa.1 + pb.1));
        }
    }
    convex_hull(&sums)
}

/// Returns the Minkowski sum of convex polygons `a` and `b` as a
/// counter-clockwise ring, or an empty vector if either input is empty.
///
/// Both operands are treated as convex; their winding is normalised
/// internally, so either orientation is accepted. When an operand has fewer
/// than three vertices (a single point or a segment) the sum is produced from
/// the convex hull of all pairwise vertex sums; otherwise the linear-time
/// edge-merge is used. The returned ring may carry collinear seam vertices but
/// every vertex lies exactly on the sum's boundary and is the sum of one
/// vertex from each operand.
///
/// ```
/// use prism_ui_render_backend::minkowski::minkowski_sum;
/// let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
/// // A unit square summed with itself is the 2x2 square (area 4).
/// let sum = minkowski_sum(&unit, &unit);
/// assert_eq!(sum.len(), 4);
/// ```
#[must_use]
pub fn minkowski_sum(a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    if a.len() < 3 || b.len() < 3 {
        return pairwise_hull(a, b);
    }

    let a = to_ccw_from_lowest(a);
    let b = to_ccw_from_lowest(b);
    let (n, m) = (a.len(), b.len());

    let mut res = Vec::with_capacity(n + m);
    let (mut i, mut j) = (0usize, 0usize);
    while i < n || j < m {
        res.push((a[i % n].0 + b[j % m].0, a[i % n].1 + b[j % m].1));
        if j >= m {
            i += 1;
        } else if i >= n {
            j += 1;
        } else {
            let c = cross2(edge(&a, i % n), edge(&b, j % m));
            if c > 0.0 {
                i += 1;
            } else if c < 0.0 {
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::measure::area;
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

    /// A convex polygon of at least three vertices via the crate's hull.
    fn random_polygon(state: &mut u64, spread: f32) -> Vec<(f32, f32)> {
        loop {
            let n = 6 + (next_rand(state) % 10) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(state, -spread, spread), rand_in(state, -spread, spread)))
                .collect();
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// A point inside (or on) a convex polygon: a random convex combination of
    /// its vertices.
    fn random_interior(state: &mut u64, poly: &[(f32, f32)]) -> (f32, f32) {
        let mut w: Vec<f32> = poly.iter().map(|_| rand_in(state, 0.01, 1.0)).collect();
        let total: f32 = w.iter().sum();
        for wi in &mut w {
            *wi /= total;
        }
        let mut p = (0.0, 0.0);
        for (&wi, &v) in w.iter().zip(poly.iter()) {
            p.0 += wi * v.0;
            p.1 += wi * v.1;
        }
        p
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 3e-3 * (a.abs() + b.abs()) + 3e-3
    }

    /// Defining property: every `a + b` for interior points of the operands
    /// lies inside (or on) the sum.
    #[test]
    fn contains_pointwise_sums() {
        let mut state = 0x11115_u64;
        for _ in 0..200 {
            let a = random_polygon(&mut state, 20.0);
            let b = random_polygon(&mut state, 12.0);
            let sum = minkowski_sum(&a, &b);
            for _ in 0..8 {
                let pa = random_interior(&mut state, &a);
                let pb = random_interior(&mut state, &b);
                let p = (pa.0 + pb.0, pa.1 + pb.1);
                let sd = sd_polygon(p.0, p.1, &sum);
                assert!(sd <= 1e-1, "pointwise sum outside, sd = {sd}");
            }
        }
    }

    /// The merged ring must describe the same region as the independent hull of
    /// all pairwise vertex sums: equal area and mutual boundary containment.
    #[test]
    fn matches_hull_of_pairwise() {
        let mut state = 0x22226_u64;
        for _ in 0..200 {
            let a = random_polygon(&mut state, 18.0);
            let b = random_polygon(&mut state, 15.0);
            let sum = minkowski_sum(&a, &b);
            let oracle = pairwise_hull(&a, &b);
            assert!(approx(area(&sum), area(&oracle)), "area {} vs {}", area(&sum), area(&oracle));
            for &v in &sum {
                assert!(sd_polygon(v.0, v.1, &oracle).abs() <= 1e-1, "sum vertex off oracle");
            }
            for &v in &oracle {
                assert!(sd_polygon(v.0, v.1, &sum).abs() <= 1e-1, "oracle vertex off sum");
            }
        }
    }

    /// Each output vertex is the sum of one vertex from each operand.
    #[test]
    fn vertices_are_vertex_sums() {
        let mut state = 0x33337_u64;
        for _ in 0..150 {
            let a = random_polygon(&mut state, 16.0);
            let b = random_polygon(&mut state, 11.0);
            let sum = minkowski_sum(&a, &b);
            for &v in &sum {
                let found = a.iter().any(|&pa| {
                    b.iter()
                        .any(|&pb| approx(pa.0 + pb.0, v.0) && approx(pa.1 + pb.1, v.1))
                });
                assert!(found, "vertex {v:?} is not a vertex sum");
            }
        }
    }

    /// Translating an operand translates the whole sum by the same vector.
    #[test]
    fn translation_additive() {
        let mut state = 0x44448_u64;
        for _ in 0..150 {
            let a = random_polygon(&mut state, 17.0);
            let b = random_polygon(&mut state, 13.0);
            let (tx, ty) = (rand_in(&mut state, -30.0, 30.0), rand_in(&mut state, -30.0, 30.0));
            let a_shift: Vec<(f32, f32)> = a.iter().map(|&(x, y)| (x + tx, y + ty)).collect();
            let base = minkowski_sum(&a, &b);
            let shifted = minkowski_sum(&a_shift, &b);
            assert!(approx(area(&base), area(&shifted)));
            // Every shifted vertex is a base vertex plus (tx, ty).
            for &v in &shifted {
                let found = base
                    .iter()
                    .any(|&bv| approx(bv.0 + tx, v.0) && approx(bv.1 + ty, v.1));
                assert!(found, "shift not additive at {v:?}");
            }
        }
    }

    /// The sum is commutative: `A ⊕ B` and `B ⊕ A` cover the same region.
    #[test]
    fn commutative() {
        let mut state = 0x55559_u64;
        for _ in 0..150 {
            let a = random_polygon(&mut state, 14.0);
            let b = random_polygon(&mut state, 10.0);
            let ab = minkowski_sum(&a, &b);
            let ba = minkowski_sum(&b, &a);
            assert!(approx(area(&ab), area(&ba)));
            for &v in &ab {
                assert!(sd_polygon(v.0, v.1, &ba).abs() <= 1e-1);
            }
        }
    }

    #[test]
    fn fixed_cases() {
        let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let sum = minkowski_sum(&unit, &unit);
        assert!(approx(area(&sum), 4.0), "unit+unit area = {}", area(&sum));
        // Every corner of the expected 2x2 square sits on the boundary.
        for &c in &[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)] {
            assert!(sd_polygon(c.0, c.1, &sum).abs() <= 1e-3);
        }

        // Square + single point = translated square (degenerate operand path).
        let point = [(5.0, -3.0)];
        let translated = minkowski_sum(&unit, &point);
        assert!(approx(area(&translated), 1.0));
        for &c in &[(5.0, -3.0), (6.0, -3.0), (6.0, -2.0), (5.0, -2.0)] {
            assert!(sd_polygon(c.0, c.1, &translated).abs() <= 1e-3, "corner {c:?} off");
        }

        // Empty operand yields an empty sum.
        assert!(minkowski_sum(&unit, &[]).is_empty());
    }
}
