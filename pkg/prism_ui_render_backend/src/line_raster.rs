//! Integer line rasterisation via Bresenham's algorithm.
//!
//! The SDF rasteriser in [`crate::raster`] shades area-covering primitives, but
//! a GPU-driven backend also needs a *thin* integer-pixel path for debug
//! overlays, wireframes, selection outlines, and gizmo rendering, where exactly
//! one pixel per major-axis step is wanted rather than anti-aliased coverage.
//!
//! [`raster_line`] walks the integer grid with the classic all-octant Bresenham
//! error term, choosing at each major-axis step the pixel whose minor-axis
//! coordinate is closest to the true line. Only `+ - *` on integers and sign
//! comparisons are used, so the routine is exact, `no_std`-clean, and identical
//! on every target. [`raster_polyline`] chains segments while dropping the
//! duplicated joint pixel so a closed or open path lights each junction once.
//!
//! Both functions return pixels in traversal order, start pixel first and end
//! pixel last, inclusive at both ends.

use alloc::vec::Vec;

/// Rasterises the line segment from `(x0, y0)` to `(x1, y1)` into the integer
/// pixels Bresenham's algorithm would light, in order from start to end.
///
/// The result always begins at `(x0, y0)` and ends at `(x1, y1)` (both ends are
/// lit) and contains exactly `max(|x1 - x0|, |y1 - y0|) + 1` pixels: one per
/// step along the dominant axis. A zero-length segment yields a single pixel.
/// Consecutive pixels are 8-connected (each coordinate changes by at most one),
/// and every pixel lies within half a pixel of the ideal line.
#[must_use]
pub fn raster_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    // Work the error term in i64 so wide coordinate spans cannot overflow.
    let dx = (i64::from(x1) - i64::from(x0)).abs();
    let dy = -(i64::from(y1) - i64::from(y0)).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };

    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let mut out: Vec<(i32, i32)> = Vec::new();

    loop {
        out.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        // `e2` is twice the error; comparing against the axis deltas decides
        // whether this step advances the major axis, the minor axis, or both
        // (a diagonal step). At least one branch fires for a non-degenerate
        // segment, guaranteeing progress toward the endpoint.
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }

    out
}

/// Rasterises an open polyline through `points`, chaining [`raster_line`] over
/// each consecutive pair and dropping the shared joint pixel so every junction
/// is lit exactly once.
///
/// Returns an empty vector for no points and a single pixel for one point. The
/// result starts at the first point and ends at the last, passes through every
/// input vertex, and stays 8-connected across segment boundaries.
#[must_use]
pub fn raster_polyline(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let mut out: Vec<(i32, i32)> = Vec::new();
    if points.is_empty() {
        return out;
    }
    if points.len() == 1 {
        out.push(points[0]);
        return out;
    }

    for pair in points.windows(2) {
        let seg = raster_line(pair[0].0, pair[0].1, pair[1].0, pair[1].1);
        if out.is_empty() {
            out.extend(seg);
        } else {
            // `seg[0]` repeats the previous segment's final pixel; skip it.
            out.extend(seg.into_iter().skip(1));
        }
    }

    out
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

    /// A pseudo-random integer coordinate in `[-span, span]`.
    fn rand_coord(state: &mut u64, span: i32) -> i32 {
        let r = (next_rand(state) >> 33) as i64;
        let width = i64::from(span) * 2 + 1;
        (r % width - i64::from(span)) as i32
    }

    /// Perpendicular distance from pixel `p` to the infinite line through `a`
    /// and `b`. The caller guarantees `a != b`.
    fn perp_distance(a: (i32, i32), b: (i32, i32), p: (i32, i32)) -> f64 {
        let abx = f64::from(b.0 - a.0);
        let aby = f64::from(b.1 - a.1);
        let apx = f64::from(p.0 - a.0);
        let apy = f64::from(p.1 - a.1);
        let cross = (abx * apy - aby * apx).abs();
        let len = (abx * abx + aby * aby).sqrt();
        cross / len
    }

    /// Expected pixel count: one per step along the dominant axis, plus the
    /// start pixel.
    fn expected_len(a: (i32, i32), b: (i32, i32)) -> usize {
        let dx = (b.0 - a.0).unsigned_abs();
        let dy = (b.1 - a.1).unsigned_abs();
        dx.max(dy) as usize + 1
    }

    /// The line always starts at its first endpoint and ends at its second.
    #[test]
    fn endpoints_lit() {
        let mut state = 0x0_B0B1_u64;
        for _ in 0..400 {
            let a = (rand_coord(&mut state, 40), rand_coord(&mut state, 40));
            let b = (rand_coord(&mut state, 40), rand_coord(&mut state, 40));
            let line = raster_line(a.0, a.1, b.0, b.1);
            assert_eq!(line.first().copied(), Some(a));
            assert_eq!(line.last().copied(), Some(b));
        }
    }

    /// The pixel count equals the dominant-axis span plus one.
    #[test]
    fn count_is_major_span() {
        let mut state = 0x1234_ABCD_u64;
        for _ in 0..400 {
            let a = (rand_coord(&mut state, 50), rand_coord(&mut state, 50));
            let b = (rand_coord(&mut state, 50), rand_coord(&mut state, 50));
            let line = raster_line(a.0, a.1, b.0, b.1);
            assert_eq!(line.len(), expected_len(a, b));
        }
    }

    /// Independent geometric oracle: every lit pixel lies within half a pixel of
    /// the ideal line (perpendicular distance), confirming Bresenham chose the
    /// closest pixel at each step.
    #[test]
    fn within_half_pixel() {
        let mut state = 0xDEAD_BEEF_u64;
        for _ in 0..400 {
            let a = (rand_coord(&mut state, 60), rand_coord(&mut state, 60));
            let b = (rand_coord(&mut state, 60), rand_coord(&mut state, 60));
            if a == b {
                continue;
            }
            for &p in &raster_line(a.0, a.1, b.0, b.1) {
                let d = perp_distance(a, b, p);
                assert!(d <= 0.5 + 1e-9, "pixel {p:?} is {d} from line {a:?}->{b:?}");
            }
        }
    }

    /// Consecutive pixels are 8-connected: each coordinate changes by at most
    /// one and never both stay fixed.
    #[test]
    fn eight_connected() {
        let mut state = 0xFEED_FACE_u64;
        for _ in 0..400 {
            let a = (rand_coord(&mut state, 45), rand_coord(&mut state, 45));
            let b = (rand_coord(&mut state, 45), rand_coord(&mut state, 45));
            let line = raster_line(a.0, a.1, b.0, b.1);
            for w in line.windows(2) {
                let ddx = (w[1].0 - w[0].0).abs();
                let ddy = (w[1].1 - w[0].1).abs();
                assert!(ddx <= 1 && ddy <= 1, "jump {:?}->{:?}", w[0], w[1]);
                assert!(ddx + ddy >= 1, "no progress at {:?}", w[0]);
            }
        }
    }

    /// Reversing the endpoints keeps the pixel count and swaps the ends; the
    /// reversed line still satisfies the half-pixel oracle.
    #[test]
    fn reverse_symmetry() {
        let mut state = 0xCAFE_D00D_u64;
        for _ in 0..300 {
            let a = (rand_coord(&mut state, 55), rand_coord(&mut state, 55));
            let b = (rand_coord(&mut state, 55), rand_coord(&mut state, 55));
            if a == b {
                continue;
            }
            let fwd = raster_line(a.0, a.1, b.0, b.1);
            let rev = raster_line(b.0, b.1, a.0, a.1);
            assert_eq!(fwd.len(), rev.len());
            assert_eq!(rev.first().copied(), Some(b));
            assert_eq!(rev.last().copied(), Some(a));
            for &p in &rev {
                assert!(perp_distance(b, a, p) <= 0.5 + 1e-9);
            }
        }
    }

    /// Fixed horizontal, vertical, and 45-degree diagonals.
    #[test]
    fn axis_and_diagonal() {
        assert_eq!(
            raster_line(0, 3, 4, 3),
            [(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
        );
        assert_eq!(
            raster_line(2, 0, 2, -3),
            [(2, 0), (2, -1), (2, -2), (2, -3)]
        );
        assert_eq!(
            raster_line(0, 0, 3, 3),
            [(0, 0), (1, 1), (2, 2), (3, 3)]
        );
        assert_eq!(
            raster_line(0, 0, -3, 3),
            [(0, 0), (-1, 1), (-2, 2), (-3, 3)]
        );
    }

    /// A gentle slope visits each column exactly once with the expected rounded
    /// rows.
    #[test]
    fn shallow_slope() {
        // Slope 1/4: ideal rows 0, .25, .5, .75, 1 -> rounded 0,0,(0 or 1),1,1.
        let line = raster_line(0, 0, 4, 1);
        assert_eq!(line.len(), 5);
        assert_eq!(line.first().copied(), Some((0, 0)));
        assert_eq!(line.last().copied(), Some((4, 1)));
        for (i, p) in line.iter().enumerate() {
            assert_eq!(p.0, i as i32, "column skipped at {p:?}");
        }
    }

    /// A zero-length segment lights a single pixel.
    #[test]
    fn single_point() {
        assert_eq!(raster_line(7, -2, 7, -2), [(7, -2)]);
    }

    /// Degenerate polylines: empty input and a lone point.
    #[test]
    fn polyline_degenerate() {
        assert!(raster_polyline(&[]).is_empty());
        assert_eq!(raster_polyline(&[(3, 4)]), [(3, 4)]);
    }

    /// A polyline length equals the sum of its segment lengths minus the shared
    /// joints, starts and ends at the extreme points, and visits every vertex.
    #[test]
    fn polyline_concat() {
        let mut state = 0x9999_1111_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut state) % 5) as usize;
            let mut pts: Vec<(i32, i32)> = Vec::new();
            for _ in 0..n {
                pts.push((rand_coord(&mut state, 30), rand_coord(&mut state, 30)));
            }
            let poly = raster_polyline(&pts);

            let mut expected = 0usize;
            for (k, w) in pts.windows(2).enumerate() {
                let seg = expected_len(w[0], w[1]);
                expected += if k == 0 { seg } else { seg - 1 };
            }
            assert_eq!(poly.len(), expected);
            assert_eq!(poly.first().copied(), Some(pts[0]));
            assert_eq!(poly.last().copied(), Some(pts[n - 1]));
            for v in &pts {
                assert!(poly.contains(v), "vertex {v:?} not on polyline");
            }
            for w in poly.windows(2) {
                let ddx = (w[1].0 - w[0].0).abs();
                let ddy = (w[1].1 - w[0].1).abs();
                assert!(ddx <= 1 && ddy <= 1, "polyline jump {:?}->{:?}", w[0], w[1]);
            }
        }
    }
}
