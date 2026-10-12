//! Signed-distance field for an **open** polyline.
//!
//! [`sd_polygon`](crate::polygon::sd_polygon) covers closed fills; a stroked path
//! needs the field of an *open* run of segments, where the draw coverage is
//! `sd_polyline <= half_width` (fed through
//! [`coverage`](crate::sdf::coverage) for the same anti-aliased edge every other
//! primitive gets). [`sd_polyline`] is that field: the minimum
//! [`sd_segment`](crate::sdf::sd_segment) over the polyline's edges.
//!
//! The result is unsigned (a segment has no interior): zero on the path, growing
//! away from it in every direction. Only `+ - * /`, [`f32::min`] and
//! [`f32::sqrt`] are used, so the field composes with the rest of this crate and
//! stays `no_std`-clean and bit-stable.

use crate::sdf::sd_segment;

/// Distance from `(px, py)` to the open polyline `points`.
///
/// The polyline is open — the last vertex is **not** joined back to the first —
/// so this is the field a stroked path is drawn from (coverage where the
/// distance is at most the stroke half-width). Returns the minimum distance over
/// the polyline's segments. An empty polyline has no geometry and returns
/// [`f32::INFINITY`]; a single vertex is a degenerate segment, returning the
/// distance to that point.
#[must_use]
pub fn sd_polyline(px: f32, py: f32, points: &[(f32, f32)]) -> f32 {
    match points {
        [] => f32::INFINITY,
        [p] => sd_segment(px, py, p.0, p.1, p.0, p.1),
        _ => {
            let mut best = f32::INFINITY;
            for w in points.windows(2) {
                let d = sd_segment(px, py, w[0].0, w[0].1, w[1].0, w[1].1);
                best = best.min(d);
            }
            best
        }
    }
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

    /// Independent `f64` point-to-segment distance (the oracle).
    fn point_segment_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f64 {
        let (px, py) = (f64::from(p.0), f64::from(p.1));
        let (ax, ay) = (f64::from(a.0), f64::from(a.1));
        let (bx, by) = (f64::from(b.0), f64::from(b.1));
        let (abx, aby) = (bx - ax, by - ay);
        let (apx, apy) = (px - ax, py - ay);
        let len2 = abx * abx + aby * aby;
        let t = if len2 <= 0.0 {
            0.0
        } else {
            ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0)
        };
        let (dx, dy) = (px - (ax + abx * t), py - (ay + aby * t));
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn straight_polyline_is_the_perpendicular_distance() {
        let line = [(0.0, 0.0), (10.0, 0.0)];
        assert!((sd_polyline(5.0, 3.0, &line) - 3.0).abs() < 1e-5);
        assert!(sd_polyline(5.0, 0.0, &line).abs() < 1e-5);
        // Past the end, the nearest point is the endpoint.
        assert!((sd_polyline(14.0, 0.0, &line) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn bent_polyline_picks_the_nearest_segment() {
        let bend = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        assert!((sd_polyline(13.0, 5.0, &bend) - 3.0).abs() < 1e-5);
        assert!((sd_polyline(5.0, -2.0, &bend) - 2.0).abs() < 1e-5);
        // The shared corner is on the path.
        assert!(sd_polyline(10.0, 0.0, &bend).abs() < 1e-5);
    }

    #[test]
    fn empty_and_single_point() {
        assert_eq!(sd_polyline(1.0, 1.0, &[]), f32::INFINITY);
        assert!((sd_polyline(3.0, 4.0, &[(0.0, 0.0)]) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn matches_bruteforce_min_over_segments() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut s) % 12) as usize;
            let pts: Vec<(f32, f32)> = (0..n)
                .map(|_| (rand_in(&mut s, -30.0, 30.0), rand_in(&mut s, -30.0, 30.0)))
                .collect();
            let p = (rand_in(&mut s, -40.0, 40.0), rand_in(&mut s, -40.0, 40.0));
            let got = sd_polyline(p.0, p.1, &pts);
            let want = pts.windows(2).fold(f64::INFINITY, |acc, w| {
                acc.min(point_segment_dist(p, w[0], w[1]))
            }) as f32;
            assert!(
                (got - want).abs() <= 1e-3 * want.max(1.0),
                "sd {got} vs oracle {want}"
            );
        }
    }
}
