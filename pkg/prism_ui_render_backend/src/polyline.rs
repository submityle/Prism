//! Polyline arc length, point sampling, and uniform resampling.
//!
//! Flattening a curve ([`CubicBezier::flatten`](crate::curve::CubicBezier::flatten)),
//! an arc ([`flatten_arc`](crate::arc::flatten_arc)), or an ellipse
//! ([`flatten_ellipse`](crate::ellipse::flatten_ellipse)) yields an ordered run
//! of vertices — an *open polyline*. Downstream work keeps asking the same three
//! questions about that run:
//!
//! - **How long is it?** [`polyline_length`] sums the segment lengths.
//! - **Where is the point a given distance along it?** [`point_at_distance`]
//!   walks the segments and interpolates within the one containing the query,
//!   clamping to the endpoints.
//! - **Give me a copy with evenly spaced vertices.** [`resample_uniform`]
//!   redistributes the vertices so consecutive samples are at most `spacing`
//!   apart, preserving both endpoints. Every sample is placed *on* an original
//!   edge, but consecutive samples are joined by straight chords, so the
//!   resampled polyline is a *chord approximation*: at a corner a chord cuts
//!   inside the original path, making the resampled length at most the input's
//!   (equal when samples land on the original vertices).
//!
//! All arithmetic is pure `f32` (`+ - * /` and `sqrt`); there are no
//! transcendental calls, so the routines are `no_std`-clean and bit-stable.

use alloc::vec::Vec;

/// Euclidean distance between two points.
fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

/// Point `t` of the way from `a` to `b`, with `t` normally in `[0, 1]`.
fn lerp(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

/// Total arc length of the open polyline `points`.
///
/// Returns `0.0` for an empty input or a single vertex. The polyline is treated
/// as *open* — the first and last vertices are **not** connected — so a closed
/// shape must repeat its first vertex at the end to include the closing edge.
#[must_use]
pub fn polyline_length(points: &[(f32, f32)]) -> f32 {
    let mut total = 0.0_f32;
    for w in points.windows(2) {
        total += dist(w[0], w[1]);
    }
    total
}

/// The point at arc length `distance` from the start of the open polyline.
///
/// Returns `None` for an empty input. For a single vertex the lone point is
/// returned for any `distance`. `distance` is clamped to `[0, len]`, so negative
/// values yield the first vertex and values past the total length yield the last
/// vertex; a `NaN` distance yields the first vertex. This keeps the walk
/// branch-free at the ends and avoids panics on out-of-range queries.
#[must_use]
pub fn point_at_distance(points: &[(f32, f32)], distance: f32) -> Option<(f32, f32)> {
    let first = *points.first()?;
    // `first` exists, so `last` does too; returning the stored vertices directly
    // keeps the ends bit-exact instead of reconstructing them through `lerp`.
    let last = *points.last()?;
    if distance <= 0.0 || distance.is_nan() {
        return Some(first);
    }
    if distance >= polyline_length(points) {
        return Some(last);
    }
    let mut travelled = 0.0_f32;
    for w in points.windows(2) {
        let seg = dist(w[0], w[1]);
        if seg <= 0.0 {
            continue;
        }
        let next = travelled + seg;
        if distance <= next {
            return Some(lerp(w[0], w[1], (distance - travelled) / seg));
        }
        travelled = next;
    }
    // Only reachable if rounding left the query just short of the total; the
    // last vertex is the correct clamp either way.
    Some(last)
}

/// Unit tangent direction of the open polyline at arc length `distance`.
///
/// Returns `None` for an empty input or a polyline with no non-degenerate
/// segment. `distance` is clamped like [`point_at_distance`]: negatives give the
/// first segment's direction, values past the total give the last segment's, and
/// a `NaN` gives the first. At an interior vertex the *incoming* segment's
/// direction is used, matching the segment that [`point_at_distance`] (which
/// closes each interval at its end) would interpolate within.
#[must_use]
pub fn tangent_at_distance(points: &[(f32, f32)], distance: f32) -> Option<(f32, f32)> {
    let query = if distance.is_nan() { 0.0 } else { distance };
    let mut travelled = 0.0_f32;
    let mut last_dir = None;
    for w in points.windows(2) {
        let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
        let seg = (dx * dx + dy * dy).sqrt();
        if seg <= 0.0 {
            continue;
        }
        let dir = (dx / seg, dy / seg);
        let next = travelled + seg;
        if query <= next {
            return Some(dir);
        }
        last_dir = Some(dir);
        travelled = next;
    }
    last_dir
}

/// Upper bound on the interval count [`resample_uniform`] produces in one call.
///
/// A `spacing` so small that it would demand more intervals than this is clamped
/// to exactly this many, capping the output at roughly one million vertices and
/// keeping the allocation bounded for pathological requests.
const MAX_RESAMPLE_INTERVALS: u32 = 1 << 20;

/// Resample the open polyline into evenly spaced vertices.
///
/// Consecutive samples are separated by at most `spacing` along the arc, the
/// first and last vertices are preserved exactly, and every returned vertex lies
/// on an original edge. The interval count is `ceil(len / spacing)`, so the
/// achieved spacing is `len / intervals <= spacing`: the requested value is an
/// upper bound, which avoids leaving a short final segment.
///
/// Degenerate inputs are returned unchanged: an empty or single-vertex polyline,
/// a zero-length polyline (all vertices coincident), or a `spacing` that is not
/// positive and finite. Output is clamped to
/// [`MAX_RESAMPLE_INTERVALS`] intervals. Non-finite (`NaN`/infinite) vertex
/// coordinates are the caller's responsibility and propagate through the walk.
#[must_use]
pub fn resample_uniform(points: &[(f32, f32)], spacing: f32) -> Vec<(f32, f32)> {
    let len = polyline_length(points);
    if points.len() < 2 || len <= 0.0 || !spacing.is_finite() || spacing <= 0.0 {
        return points.to_vec();
    }
    let intervals_f = (len / spacing).ceil();
    let intervals = if intervals_f >= MAX_RESAMPLE_INTERVALS as f32 {
        MAX_RESAMPLE_INTERVALS
    } else {
        // `ceil()` of a positive value is at least 1, so this is never zero.
        intervals_f as u32
    };
    // Endpoints are copied verbatim (never reconstructed through the walk, which
    // could round a hair off), so the first and last vertices are bit-exact.
    let mut out = Vec::with_capacity(intervals as usize + 1);
    out.push(points[0]);
    for i in 1..intervals {
        let d = len * (i as f32) / (intervals as f32);
        // `d` lies in `[0, len]` and the polyline is non-empty, so the walk
        // always finds a point; the fallback only satisfies the type.
        out.push(point_at_distance(points, d).unwrap_or(points[0]));
    }
    out.push(points[points.len() - 1]);
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

    fn rand_in(s: &mut u64, lo: f32, hi: f32) -> f32 {
        let b = (next_rand(s) >> 40) as u32;
        lo + (hi - lo) * (b as f32 / 16_777_216.0)
    }

    fn random_polyline(state: &mut u64, n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|_| (rand_in(state, -50.0, 50.0), rand_in(state, -50.0, 50.0)))
            .collect()
    }

    /// Strictly increasing x (spacing 4), so segment x-ranges are disjoint and a
    /// point strictly inside one segment has that segment as its unique nearest
    /// feature.
    fn monotone_polyline(state: &mut u64, n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| (i as f32 * 4.0, rand_in(state, -40.0, 40.0)))
            .collect()
    }

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    }

    /// Normalized direction of the unique nearest non-degenerate segment, or
    /// `None` when two segments tie (the corner-ambiguity case).
    fn nearest_segment_dir(p: (f32, f32), pts: &[(f32, f32)]) -> Option<(f32, f32)> {
        let mut best = f64::INFINITY;
        let mut second = f64::INFINITY;
        let mut dir = None;
        for w in pts.windows(2) {
            let dx = f64::from(w[1].0) - f64::from(w[0].0);
            let dy = f64::from(w[1].1) - f64::from(w[0].1);
            let seg = (dx * dx + dy * dy).sqrt();
            if seg <= 0.0 {
                continue;
            }
            let d = point_segment_dist(p, w[0], w[1]);
            if d < best {
                second = best;
                best = d;
                dir = Some((dx / seg, dy / seg));
            } else if d < second {
                second = d;
            }
        }
        match dir {
            Some(v) if second - best > 1e-3 => Some((v.0 as f32, v.1 as f32)),
            _ => None,
        }
    }

    /// Independent `f64` oracle for the total arc length.
    fn brute_length(pts: &[(f32, f32)]) -> f64 {
        let mut total = 0.0_f64;
        for w in pts.windows(2) {
            let dx = f64::from(w[1].0) - f64::from(w[0].0);
            let dy = f64::from(w[1].1) - f64::from(w[0].1);
            total += (dx * dx + dy * dy).sqrt();
        }
        total
    }

    /// Independent point-to-segment distance (`f64`).
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

    fn min_dist_to_polyline(p: (f32, f32), pts: &[(f32, f32)]) -> f64 {
        pts.windows(2)
            .map(|w| point_segment_dist(p, w[0], w[1]))
            .fold(f64::INFINITY, f64::min)
    }

    /// Arc position of a point known to lie on `pts`, found by locating the
    /// nearest segment — an oracle independent of the module's own walk.
    fn arc_position(p: (f32, f32), pts: &[(f32, f32)]) -> f64 {
        let (px, py) = (f64::from(p.0), f64::from(p.1));
        let mut best = f64::INFINITY;
        let mut best_arc = 0.0_f64;
        let mut travelled = 0.0_f64;
        for w in pts.windows(2) {
            let (ax, ay) = (f64::from(w[0].0), f64::from(w[0].1));
            let (bx, by) = (f64::from(w[1].0), f64::from(w[1].1));
            let (abx, aby) = (bx - ax, by - ay);
            let seg = (abx * abx + aby * aby).sqrt();
            let (apx, apy) = (px - ax, py - ay);
            let t = if seg <= 0.0 {
                0.0
            } else {
                ((apx * abx + apy * aby) / (seg * seg)).clamp(0.0, 1.0)
            };
            let (dx, dy) = (px - (ax + abx * t), py - (ay + aby * t));
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
                best_arc = travelled + t * seg;
            }
            travelled += seg;
        }
        best_arc
    }

    #[test]
    fn length_of_open_and_closed_shapes() {
        // Open corner: 3 + 4 = 7.
        let corner = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        assert!((polyline_length(&corner) - 7.0).abs() < 1e-5);
        // Closed unit square (first vertex repeated) has perimeter 4.
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        assert!((polyline_length(&square) - 4.0).abs() < 1e-5);
        // Degenerate inputs.
        assert!(polyline_length(&[]) == 0.0);
        assert!(polyline_length(&[(1.0, 2.0)]) == 0.0);
    }

    #[test]
    fn length_matches_bruteforce_oracle() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut s) % 30) as usize;
            let pts = random_polyline(&mut s, n);
            let len = polyline_length(&pts);
            let oracle = brute_length(&pts) as f32;
            let tol = 1e-3 * oracle.abs().max(1.0);
            assert!((len - oracle).abs() <= tol, "len {len} vs {oracle}");
        }
    }

    #[test]
    fn point_at_known_arc_lengths() {
        let pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]; // length 20
        assert!(close(point_at_distance(&pts, 5.0).unwrap(), (5.0, 0.0)));
        assert!(close(point_at_distance(&pts, 10.0).unwrap(), (10.0, 0.0)));
        assert!(close(point_at_distance(&pts, 15.0).unwrap(), (10.0, 5.0)));
        assert!(close(point_at_distance(&pts, 0.0).unwrap(), (0.0, 0.0)));
        assert!(close(point_at_distance(&pts, 20.0).unwrap(), (10.0, 10.0)));
        // Clamping at both ends.
        assert!(close(point_at_distance(&pts, -3.0).unwrap(), (0.0, 0.0)));
        assert!(close(point_at_distance(&pts, 99.0).unwrap(), (10.0, 10.0)));
        // Degenerate inputs.
        assert!(point_at_distance(&[], 1.0).is_none());
        assert!(close(
            point_at_distance(&[(4.0, 5.0)], 7.0).unwrap(),
            (4.0, 5.0)
        ));
    }

    #[test]
    fn sampled_point_lies_on_polyline() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut s) % 20) as usize;
            let pts = random_polyline(&mut s, n);
            let len = polyline_length(&pts);
            for _ in 0..8 {
                let d = rand_in(&mut s, 0.0, len);
                let p = point_at_distance(&pts, d).unwrap();
                let off = min_dist_to_polyline(p, &pts);
                assert!(off < 1e-3, "point {p:?} off by {off} at d={d}");
            }
        }
    }

    #[test]
    fn sampled_point_is_monotone_along_x_for_x_monotone_polyline() {
        // Strictly increasing x, so arc length and x advance together.
        let pts = [(0.0, 0.0), (2.0, 5.0), (6.0, -1.0), (9.0, 4.0)];
        let len = polyline_length(&pts);
        let mut prev_x = f32::NEG_INFINITY;
        let mut d = 0.0_f32;
        while d <= len {
            let p = point_at_distance(&pts, d).unwrap();
            assert!(p.0 + 1e-4 >= prev_x, "x regressed at d={d}");
            prev_x = p.0;
            d += len / 64.0;
        }
    }

    #[test]
    fn resample_straight_line_is_uniform_grid() {
        // ceil(10 / 3) = 4 intervals, so x = 0, 2.5, 5, 7.5, 10.
        let out = resample_uniform(&[(0.0, 0.0), (10.0, 0.0)], 3.0);
        assert_eq!(out.len(), 5);
        for (i, p) in out.iter().enumerate() {
            let expect = 2.5 * i as f32;
            assert!((p.0 - expect).abs() < 1e-4 && p.1.abs() < 1e-6);
        }
    }

    #[test]
    fn resample_preserves_length_and_arc_spacing() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut s) % 25) as usize;
            let pts = random_polyline(&mut s, n);
            let len = polyline_length(&pts);
            if len <= 1e-2 {
                continue;
            }
            let spacing = (len * rand_in(&mut s, 0.05, 0.5)).max(1e-2);
            let out = resample_uniform(&pts, spacing);

            // Sample count is exactly ceil(len / spacing) + 1.
            let intervals = (len / spacing).ceil() as usize;
            assert_eq!(out.len(), intervals + 1);
            // Chords never exceed the arc, so the resampled path is at most as
            // long as the input (it cuts corners, but only shrinks).
            let out_len = polyline_length(&out);
            assert!(
                out_len <= len * (1.0 + 1e-4),
                "resampled longer than arc {len} -> {out_len}"
            );
            // Endpoints are exact.
            assert!(close(out[0], pts[0]));
            assert!(close(out[out.len() - 1], pts[n - 1]));

            // Every sample lies on the original polyline and is arc-uniform.
            let step = f64::from(len) / intervals as f64;
            for (i, &p) in out.iter().enumerate() {
                assert!(min_dist_to_polyline(p, &pts) < 1e-3);
                let arc = arc_position(p, &pts);
                let expect = step * i as f64;
                let tol = 1e-3 * (1.0 + f64::from(len));
                assert!((arc - expect).abs() <= tol, "arc {arc} vs {expect}");
            }
        }
    }

    #[test]
    fn resample_aligned_to_vertices_is_exact_subdivision() {
        // Length 20, spacing 5: samples land on every vertex, so no corner is
        // cut and the chords reproduce the original path exactly.
        let pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        let out = resample_uniform(&pts, 5.0);
        let expect = [
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (10.0, 5.0),
            (10.0, 10.0),
        ];
        assert_eq!(out.len(), expect.len());
        for (got, want) in out.iter().zip(expect) {
            assert!(close(*got, want), "got {got:?} want {want:?}");
        }
        assert!((polyline_length(&out) - 20.0).abs() < 1e-4);
    }

    #[test]
    fn resample_gap_never_exceeds_spacing() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut s) % 20) as usize;
            let pts = random_polyline(&mut s, n);
            let len = polyline_length(&pts);
            if len <= 1e-2 {
                continue;
            }
            let spacing = (len * rand_in(&mut s, 0.02, 0.6)).max(1e-2);
            let out = resample_uniform(&pts, spacing);
            let mut max_gap = 0.0_f32;
            for w in out.windows(2) {
                max_gap = max_gap.max(dist(w[0], w[1]));
            }
            // Euclidean gap <= arc gap <= spacing.
            assert!(max_gap <= spacing + 1e-3, "gap {max_gap} > {spacing}");
        }
    }

    #[test]
    fn resample_degenerate_inputs() {
        assert!(resample_uniform(&[], 1.0).is_empty());
        assert_eq!(resample_uniform(&[(2.0, 3.0)], 1.0).len(), 1);
        // Zero-length polyline (all coincident) returns the input unchanged.
        let same = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)];
        assert_eq!(resample_uniform(&same, 1.0).len(), same.len());
        // Non-positive / non-finite spacing returns the input unchanged.
        let pts = [(0.0, 0.0), (3.0, 4.0)];
        assert_eq!(resample_uniform(&pts, 0.0).len(), 2);
        assert_eq!(resample_uniform(&pts, -1.0).len(), 2);
        assert_eq!(resample_uniform(&pts, f32::NAN).len(), 2);
        assert_eq!(resample_uniform(&pts, f32::INFINITY).len(), 2);
    }

    #[test]
    fn tangent_of_straight_line_and_corners() {
        let line = [(0.0, 0.0), (10.0, 0.0)];
        assert!(close(tangent_at_distance(&line, 5.0).unwrap(), (1.0, 0.0)));
        let bend = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        assert!(close(tangent_at_distance(&bend, 5.0).unwrap(), (1.0, 0.0)));
        assert!(close(tangent_at_distance(&bend, 15.0).unwrap(), (0.0, 1.0)));
        // Clamping, matching point_at_distance.
        assert!(close(tangent_at_distance(&bend, -4.0).unwrap(), (1.0, 0.0)));
        assert!(close(tangent_at_distance(&bend, 99.0).unwrap(), (0.0, 1.0)));
        // Degenerate inputs: no non-zero segment.
        assert!(tangent_at_distance(&[], 1.0).is_none());
        assert!(tangent_at_distance(&[(1.0, 1.0)], 0.0).is_none());
        assert!(tangent_at_distance(&[(2.0, 2.0), (2.0, 2.0)], 0.5).is_none());
    }

    #[test]
    fn tangent_is_unit_length_over_random_polylines() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..200 {
            let n = 2 + (next_rand(&mut s) % 20) as usize;
            let pts = random_polyline(&mut s, n);
            let len = polyline_length(&pts);
            for _ in 0..8 {
                let d = rand_in(&mut s, -5.0, len + 5.0);
                let t = tangent_at_distance(&pts, d).unwrap();
                let mag = (t.0 * t.0 + t.1 * t.1).sqrt();
                assert!((mag - 1.0).abs() < 1e-4, "|t| = {mag}");
            }
        }
    }

    #[test]
    fn tangent_matches_containing_segment_on_monotone_polyline() {
        let mut s = 0x5_C11A_u64;
        let mut checked = 0;
        for _ in 0..300 {
            let n = 2 + (next_rand(&mut s) % 15) as usize;
            let pts = monotone_polyline(&mut s, n);
            let len = polyline_length(&pts);
            for _ in 0..6 {
                let d = rand_in(&mut s, 0.0, len);
                let p = point_at_distance(&pts, d).unwrap();
                // Skip samples sitting on a vertex (corner ambiguity).
                if pts.iter().any(|v| (v.0 - p.0).abs() < 1e-3) {
                    continue;
                }
                let Some(dir) = nearest_segment_dir(p, &pts) else {
                    continue;
                };
                let t = tangent_at_distance(&pts, d).unwrap();
                assert!(
                    (t.0 - dir.0).abs() < 2e-3 && (t.1 - dir.1).abs() < 2e-3,
                    "t {t:?} vs dir {dir:?} at d={d}"
                );
                checked += 1;
            }
        }
        assert!(checked > 100, "too few unambiguous samples: {checked}");
    }
}
