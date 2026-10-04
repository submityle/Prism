//! Adaptive flattening of cubic Bézier curves into polylines.
//!
//! The SDF primitives in [`crate::sdf`] and the polygon field in
//! [`crate::polygon`] cover filled shapes, but vector chrome — rounded stroke
//! paths, custom icon outlines, chart splines — is authored as cubic Bézier
//! segments. Before such a path can be rasterised or fed to the polygon field
//! it has to be *flattened*: approximated by a polyline whose every point lies
//! within a caller-chosen `tolerance` of the true curve.
//!
//! [`CubicBezier::flatten`] does this with recursive de Casteljau subdivision,
//! stopping as soon as a sub-curve's control points sit within `tolerance` of
//! its chord. [`CubicBezier::eval`] evaluates the curve directly from the
//! Bernstein basis, and [`CubicBezier::split`] subdivides via de Casteljau;
//! the two use independent math, which is exactly what lets the subdivision be
//! cross-checked.
//!
//! Everything is pure `f32` arithmetic (`+ - * /` and [`f32::sqrt`]) with no
//! transcendental functions, so results are bit-stable and `no_std`-clean.

use alloc::vec;
use alloc::vec::Vec;

/// Hard cap on subdivision recursion depth. `2^24` leaf segments is far beyond
/// any sane `tolerance`, and the cap guarantees termination for pathological
/// (near-cusp) input without ever consulting floating-point edge cases.
const MAX_DEPTH: u32 = 24;

/// A cubic Bézier curve defined by its two anchors (`p0`, `p3`) and two control
/// points (`p1`, `p2`), each an `(x, y)` pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier {
    /// Start anchor, reached at `t == 0`.
    pub p0: (f32, f32),
    /// First control point.
    pub p1: (f32, f32),
    /// Second control point.
    pub p2: (f32, f32),
    /// End anchor, reached at `t == 1`.
    pub p3: (f32, f32),
}

impl CubicBezier {
    /// Creates a curve from its four control points.
    #[must_use]
    pub fn new(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluates the curve at parameter `t` using the Bernstein basis.
    ///
    /// `t` is clamped to `[0, 1]`. This is deliberately independent of
    /// [`split`](Self::split) so the two can validate each other.
    #[must_use]
    pub fn eval(&self, t: f32) -> (f32, f32) {
        let t = clamp01(t);
        let u = 1.0 - t;
        let b0 = u * u * u;
        let b1 = 3.0 * u * u * t;
        let b2 = 3.0 * u * t * t;
        let b3 = t * t * t;
        (
            b0 * self.p0.0 + b1 * self.p1.0 + b2 * self.p2.0 + b3 * self.p3.0,
            b0 * self.p0.1 + b1 * self.p1.1 + b2 * self.p2.1 + b3 * self.p3.1,
        )
    }

    /// Splits the curve at parameter `t` into the two sub-curves that exactly
    /// reproduce the left (`0..=t`) and right (`t..=1`) portions, via de
    /// Casteljau's construction.
    ///
    /// `t` is clamped to `[0, 1]`.
    #[must_use]
    pub fn split(&self, t: f32) -> (Self, Self) {
        let t = clamp01(t);
        let p01 = lerp(self.p0, self.p1, t);
        let p12 = lerp(self.p1, self.p2, t);
        let p23 = lerp(self.p2, self.p3, t);
        let p012 = lerp(p01, p12, t);
        let p123 = lerp(p12, p23, t);
        let p0123 = lerp(p012, p123, t);
        (
            Self { p0: self.p0, p1: p01, p2: p012, p3: p0123 },
            Self { p0: p0123, p1: p123, p2: p23, p3: self.p3 },
        )
    }

    /// Returns `true` when the curve is within `tolerance` of its chord and can
    /// be replaced by the single segment `p0 -> p3`.
    ///
    /// Flatness is measured as the larger of the two control points' distances
    /// to the chord *segment* `p0..=p3` (not the infinite line, so tangential
    /// overshoot past an endpoint also forces subdivision). Because a cubic
    /// point is a convex combination of its control points and distance to a
    /// convex set is itself convex, this bounds the curve's distance to the
    /// chord by `0.75 * max(d1, d2)` (Jensen), so passing this test keeps the
    /// whole sub-curve within `tolerance` of the emitted segment.
    #[must_use]
    pub fn is_flat(&self, tolerance: f32) -> bool {
        let d1 = point_segment_distance(self.p1, self.p0, self.p3);
        let d2 = point_segment_distance(self.p2, self.p0, self.p3);
        max(d1, d2) <= tolerance
    }

    /// Flattens the curve into a polyline whose every point lies within
    /// `tolerance` of the true curve.
    ///
    /// The returned vector always starts at `p0` and ends at `p3`, and contains
    /// at least those two points. A larger `tolerance` yields fewer points; a
    /// non-positive `tolerance` is treated as a tiny positive value so the
    /// recursion still terminates at [`MAX_DEPTH`].
    #[must_use]
    pub fn flatten(&self, tolerance: f32) -> Vec<(f32, f32)> {
        let tol = if tolerance > 0.0 { tolerance } else { f32::EPSILON };
        let mut out = vec![self.p0];
        self.flatten_rec(tol, MAX_DEPTH, &mut out);
        out
    }

    fn flatten_rec(&self, tol: f32, depth: u32, out: &mut Vec<(f32, f32)>) {
        if depth == 0 || self.is_flat(tol) {
            out.push(self.p3);
            return;
        }
        let (left, right) = self.split(0.5);
        left.flatten_rec(tol, depth - 1, out);
        right.flatten_rec(tol, depth - 1, out);
    }
}

#[inline]
fn lerp(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[inline]
fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Distance from `p` to the line *segment* `a..=b`, falling back to the
/// point-to-`a` distance when `a == b`. Clamping the projection to the segment
/// (rather than using the infinite line) counts tangential overshoot past an
/// endpoint, which is what makes [`CubicBezier::is_flat`] a sound bound.
fn point_segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let ex = b.0 - a.0;
    let ey = b.1 - a.1;
    let len2 = ex * ex + ey * ey;
    let t = if len2 > 0.0 {
        let raw = ((p.0 - a.0) * ex + (p.1 - a.1) * ey) / len2;
        clamp01(raw)
    } else {
        0.0
    };
    let dx = p.0 - a.0 - ex * t;
    let dy = p.1 - a.1 - ey * t;
    (dx * dx + dy * dy).sqrt()
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

    fn rand_point(state: &mut u64) -> (f32, f32) {
        (rand_in(state, -50.0, 50.0), rand_in(state, -50.0, 50.0))
    }

    /// Exact distance from a point to a line segment — the independent oracle
    /// for the flattening tolerance guarantee.
    fn point_segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
        let ex = b.0 - a.0;
        let ey = b.1 - a.1;
        let wx = p.0 - a.0;
        let wy = p.1 - a.1;
        let denom = ex * ex + ey * ey;
        let t = if denom > 0.0 {
            let raw = (wx * ex + wy * ey) / denom;
            raw.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dx = wx - ex * t;
        let dy = wy - ey * t;
        (dx * dx + dy * dy).sqrt()
    }

    fn point_polyline_distance(p: (f32, f32), poly: &[(f32, f32)]) -> f32 {
        let mut best = f32::INFINITY;
        for w in poly.windows(2) {
            best = best.min(point_segment_distance(p, w[0], w[1]));
        }
        best
    }

    #[test]
    fn eval_hits_both_anchors() {
        let c = CubicBezier::new((0.0, 0.0), (1.0, 4.0), (5.0, 4.0), (6.0, 0.0));
        assert_eq!(c.eval(0.0), (0.0, 0.0));
        assert_eq!(c.eval(1.0), (6.0, 0.0));
    }

    #[test]
    fn split_reproduces_parent_eval() {
        // de Casteljau split vs Bernstein eval: two independent evaluators must
        // agree on the whole curve. left(s) == parent(t*s), right(s) ==
        // parent(t + (1-t)*s).
        let mut state = 0x1111_2222_3333_4444_u64;
        for _ in 0..200 {
            let c = CubicBezier::new(
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
            );
            let t = rand_in(&mut state, 0.05, 0.95);
            let (l, r) = c.split(t);
            for k in 0..=10 {
                let s = (k as f32) / 10.0;
                let (lx, ly) = l.eval(s);
                let (px, py) = c.eval(t * s);
                assert!((lx - px).abs() < 1e-3 && (ly - py).abs() < 1e-3);
                let (rx, ry) = r.eval(s);
                let (qx, qy) = c.eval(t + (1.0 - t) * s);
                assert!((rx - qx).abs() < 1e-3 && (ry - qy).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn flatten_stays_within_tolerance() {
        // The core guarantee: densely sampled curve points lie within tolerance
        // of the output polyline. Independent recompute (eval vs segment dist).
        let tol = 0.5_f32;
        let mut state = 0xCAFE_F00D_1234_5678_u64;
        for _ in 0..150 {
            let c = CubicBezier::new(
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
            );
            let poly = c.flatten(tol);
            assert!(poly.len() >= 2);
            for k in 0..=400 {
                let t = (k as f32) / 400.0;
                let pt = c.eval(t);
                let d = point_polyline_distance(pt, &poly);
                assert!(d <= tol + 1e-2, "curve point {d} off polyline (tol {tol})");
            }
        }
    }

    #[test]
    fn flatten_preserves_endpoints() {
        let c = CubicBezier::new((2.0, 3.0), (10.0, -4.0), (-5.0, 8.0), (7.0, 7.0));
        let poly = c.flatten(0.25);
        assert_eq!(*poly.first().unwrap(), (2.0, 3.0));
        assert_eq!(*poly.last().unwrap(), (7.0, 7.0));
    }

    #[test]
    fn collinear_curve_is_a_single_segment() {
        // Control points on the chord => already flat => just the two anchors.
        let c = CubicBezier::new((0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (9.0, 0.0));
        let poly = c.flatten(0.01);
        assert_eq!(poly, vec![(0.0, 0.0), (9.0, 0.0)]);
    }

    #[test]
    fn tighter_tolerance_never_reduces_detail() {
        let mut state = 0x9ABC_DEF0_1234_5678_u64;
        for _ in 0..100 {
            let c = CubicBezier::new(
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
            );
            let coarse = c.flatten(2.0).len();
            let fine = c.flatten(0.1).len();
            assert!(fine >= coarse, "fine {fine} < coarse {coarse}");
        }
    }

    #[test]
    fn flattened_points_lie_in_control_bbox() {
        // A cubic is contained in the convex hull of its control points, so the
        // polyline must stay inside their axis-aligned bounding box.
        let mut state = 0x0F0F_0F0F_0F0F_0F0F_u64;
        for _ in 0..100 {
            let c = CubicBezier::new(
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
                rand_point(&mut state),
            );
            let min_x = c.p0.0.min(c.p1.0).min(c.p2.0).min(c.p3.0);
            let max_x = c.p0.0.max(c.p1.0).max(c.p2.0).max(c.p3.0);
            let min_y = c.p0.1.min(c.p1.1).min(c.p2.1).min(c.p3.1);
            let max_y = c.p0.1.max(c.p1.1).max(c.p2.1).max(c.p3.1);
            for &(x, y) in &c.flatten(0.5) {
                assert!(x >= min_x - 1e-3 && x <= max_x + 1e-3);
                assert!(y >= min_y - 1e-3 && y <= max_y + 1e-3);
            }
        }
    }

    #[test]
    fn degenerate_loop_is_still_flattened() {
        // Zero-length chord but a real bulge must not collapse to one segment.
        let c = CubicBezier::new((0.0, 0.0), (10.0, 10.0), (-10.0, 10.0), (0.0, 0.0));
        let poly = c.flatten(0.5);
        assert!(poly.len() > 2, "loop collapsed to {} points", poly.len());
        for k in 0..=200 {
            let t = (k as f32) / 200.0;
            let d = point_polyline_distance(c.eval(t), &poly);
            assert!(d <= 0.5 + 1e-2);
        }
    }
}
