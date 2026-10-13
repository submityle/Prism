//! Polyline dashing: split a path into the drawn runs of an on/off pattern.
//!
//! Dashed borders, marching-ants selection marquees, gridlines and trail effects
//! all need the same thing — walk a path by arc length, draw the "on" spans and
//! skip the "off" ones. [`dash_polyline`] does exactly that, reusing
//! [`sub_polyline`](crate::polyline::sub_polyline) to cut out each run so no
//! corner inside a dash is dropped, and it returns one sub-polyline per dash.
//!
//! The `pattern` alternates on/off lengths and always starts with an "on"
//! length: `[4.0, 2.0]` means 4 on, 2 off, repeated. `phase` is the arc distance
//! into the pattern where the path starts — advance it over time for a marching
//! -ants animation. A pattern with no positive entries (or an empty one) leaves
//! the whole path solid.
//!
//! Everything is `f32` arithmetic plus `sqrt`; there are no transcendentals, so
//! the routine is `no_std`-clean and byte-for-byte repeatable.

use alloc::vec::Vec;

use crate::polyline::{polyline_length, sub_polyline};

/// Upper bound on the number of dash runs produced in one call. A period tiny
/// relative to the path length would otherwise demand an unbounded number of
/// dashes; past this cap the remaining spans are dropped, keeping the output
/// (and its allocation) bounded.
const MAX_DASHES: usize = 1 << 20;

/// A non-negative, finite pattern entry, or `0.0` for anything else.
fn sanitize_len(x: f32) -> f32 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Split `points` into the drawn ("on") runs of a dash pattern.
///
/// `pattern` is a sequence of on/off lengths starting with an "on" length;
/// entries that are negative, zero, `NaN` or infinite are treated as `0.0`.
/// `phase` is the arc distance into the pattern at which the polyline begins (a
/// non-finite value is treated as `0.0`). Returns the on-runs in path order, each
/// a sub-polyline with at least two points; a first or last dash truncated by the
/// pattern phase or the path ends is returned at its truncated length.
///
/// An empty pattern, or one whose entries all sanitize to zero, leaves the whole
/// path as a single run. A path with fewer than two points, or zero length,
/// yields no runs.
#[must_use]
pub fn dash_polyline(points: &[(f32, f32)], pattern: &[f32], phase: f32) -> Vec<Vec<(f32, f32)>> {
    let total = polyline_length(points);
    if points.len() < 2 || total <= 0.0 {
        return Vec::new();
    }

    let period: f32 = pattern.iter().map(|&x| sanitize_len(x)).sum();
    if pattern.is_empty() || period <= 0.0 {
        let whole = sub_polyline(points, 0.0, total);
        if whole.len() < 2 {
            return Vec::new();
        }
        return alloc::vec![whole];
    }
    let phase = if phase.is_finite() {
        phase.rem_euclid(period)
    } else {
        0.0
    };

    // "On" spans in pattern space, i.e. positions in `[0, period)`.
    let mut on = Vec::new();
    let mut pos = 0.0_f32;
    let mut is_on = true;
    for &raw in pattern {
        let seg = sanitize_len(raw);
        if is_on && seg > 0.0 {
            on.push((pos, pos + seg));
        }
        pos += seg;
        is_on = !is_on;
    }
    if on.is_empty() {
        return Vec::new();
    }

    // Map each pattern-space span to arc space at every period offset that can
    // touch `[0, total]`, then clip.
    let span_f = ((total + phase) / period).ceil();
    let span = if span_f.is_finite() && span_f > 0.0 {
        span_f as i64
    } else {
        0
    };
    let k_max = (span + 2).min(MAX_DASHES as i64);

    let mut intervals: Vec<(f32, f32)> = Vec::new();
    'outer: for k in -1..=k_max {
        let shift = k as f32 * period;
        for &(a, b) in &on {
            let lo = (a - phase + shift).max(0.0);
            let hi = (b - phase + shift).min(total);
            if hi > lo {
                intervals.push((lo, hi));
                if intervals.len() >= MAX_DASHES {
                    break 'outer;
                }
            }
        }
    }
    intervals.sort_by(|x, y| x.0.total_cmp(&y.0));

    let mut runs = Vec::with_capacity(intervals.len());
    for (lo, hi) in intervals {
        let run = sub_polyline(points, lo, hi);
        if run.len() >= 2 {
            runs.push(run);
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::polyline::polyline_length;
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

    #[test]
    fn straight_line_dashes() {
        let line = [(0.0, 0.0), (10.0, 0.0)];
        let runs = dash_polyline(&line, &[1.0, 1.0], 0.0);
        assert_eq!(runs.len(), 5);
        for (i, run) in runs.iter().enumerate() {
            assert_eq!(run.len(), 2);
            let s = 2.0 * i as f32;
            assert!((run[0].0 - s).abs() < 1e-4 && run[0].1.abs() < 1e-6);
            assert!((run[1].0 - (s + 1.0)).abs() < 1e-4 && run[1].1.abs() < 1e-6);
        }
    }

    #[test]
    fn phase_offsets_and_truncates_dashes() {
        let line = [(0.0, 0.0), (10.0, 0.0)];
        let runs = dash_polyline(&line, &[1.0, 1.0], 0.5);
        // A leading half dash, five more, and a trailing one clipped at the end.
        assert_eq!(runs.len(), 6);
        assert!((runs[0][0].0 - 0.0).abs() < 1e-4);
        assert!((runs[0][1].0 - 0.5).abs() < 1e-4);
        assert!((runs[1][0].0 - 1.5).abs() < 1e-4);
        assert!((runs[1][1].0 - 2.5).abs() < 1e-4);
        // The final dash is cut short by the path end.
        assert!((runs[5][0].0 - 9.5).abs() < 1e-4);
        assert!((runs[5][1].0 - 10.0).abs() < 1e-4);
    }

    #[test]
    fn dash_lengths_and_total_follow_the_pattern() {
        let runs = dash_polyline(&[(0.0, 0.0), (20.0, 0.0)], &[2.0, 3.0], 0.0);
        assert_eq!(runs.len(), 4);
        for (i, run) in runs.iter().enumerate() {
            let s = 5.0 * i as f32;
            assert!((run[0].0 - s).abs() < 1e-3);
            assert!((run[1].0 - (s + 2.0)).abs() < 1e-3);
        }
        let total_on: f32 = runs.iter().map(|r| polyline_length(r)).sum();
        assert!((total_on - 8.0).abs() < 1e-3);
    }

    #[test]
    fn dashes_are_ordered_and_disjoint() {
        let runs = dash_polyline(&[(0.0, 0.0), (30.0, 0.0)], &[1.5, 1.0], 0.0);
        for w in runs.windows(2) {
            let end = w[0][w[0].len() - 1];
            assert!(end.0 <= w[1][0].0 + 1e-4, "overlapping dashes");
        }
    }

    #[test]
    fn every_dash_lies_on_the_polyline() {
        let mut s = 0x5_C11A_u64;
        for _ in 0..100 {
            let n = 2 + (next_rand(&mut s) % 12) as usize;
            let pts = random_polyline(&mut s, n);
            if polyline_length(&pts) <= 1e-2 {
                continue;
            }
            let pattern = [rand_in(&mut s, 0.2, 2.0), rand_in(&mut s, 0.2, 2.0)];
            let phase = rand_in(&mut s, 0.0, 10.0);
            for run in dash_polyline(&pts, &pattern, phase) {
                assert!(run.len() >= 2);
                for &p in &run {
                    assert!(min_dist_to_polyline(p, &pts) < 1e-3);
                }
            }
        }
    }

    #[test]
    fn degenerate_inputs() {
        assert!(dash_polyline(&[], &[1.0, 1.0], 0.0).is_empty());
        assert!(dash_polyline(&[(1.0, 1.0)], &[1.0, 1.0], 0.0).is_empty());
        let line = [(0.0, 0.0), (3.0, 0.0)];
        // Empty and all-zero patterns leave the path solid (one full run).
        assert_eq!(dash_polyline(&line, &[], 0.0).len(), 1);
        assert_eq!(dash_polyline(&line, &[0.0, 0.0], 0.0).len(), 1);
        // A lone "on" entry is also a single solid run.
        let runs = dash_polyline(&line, &[5.0], 0.0);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len(), 2);
    }
}
