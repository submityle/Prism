//! Circular-arc flattening with a guaranteed chord (sagitta) error bound.
//!
//! Rounded rectangles, pie slices, radial gauges, and stroked circles all need
//! to turn a circular arc into a short polyline before it can be filled or fed
//! to [`crate::raster`]. The number of segments must adapt to the radius and the
//! requested tolerance: a large circle needs more segments than a small one to
//! keep the visible flat-spot error below a pixel.
//!
//! For a sub-arc spanning angle `delta` on a circle of radius `r`, the maximum
//! deviation of its chord from the true arc (the *sagitta*) is
//! `r * (1 - cos(delta / 2))`, attained at the arc midpoint. [`arc_segment_count`]
//! inverts that relation to find the fewest equal sub-arcs whose sagitta stays
//! within `tolerance`, and [`flatten_arc`] samples that many points.
//!
//! Angles are in radians; `sweep` may be positive (counter-clockwise) or
//! negative (clockwise), and its magnitude may exceed a full turn. Trigonometry
//! runs in `f64` for accuracy and the samples are returned as `f32`.

use alloc::vec;
use alloc::vec::Vec;

/// Upper bound on the segment count so a tiny tolerance (or a non-positive one)
/// cannot request an unbounded allocation.
const MAX_SEGMENTS: u32 = 4096;

/// Returns the minimal number of equal-angle chords (at least one, at most
/// [`MAX_SEGMENTS`]) whose sagitta stays within `tolerance` for an arc of the
/// given `radius` and angular `sweep`.
///
/// A zero or negative radius, or a zero sweep, needs a single segment. A
/// tolerance of at least twice the radius is always met by one chord. A
/// non-positive tolerance is treated as "as fine as allowed" and returns
/// [`MAX_SEGMENTS`].
#[must_use]
pub fn arc_segment_count(radius: f32, sweep: f32, tolerance: f32) -> u32 {
    let r = f64::from(radius.abs());
    let span = f64::from(sweep.abs());
    if r <= 0.0 || span <= 0.0 {
        return 1;
    }
    let tol = f64::from(tolerance);
    if tol <= 0.0 {
        return MAX_SEGMENTS;
    }

    // Largest half-chord angle whose sagitta equals the tolerance:
    //   r * (1 - cos(a)) = tol  =>  a = acos(1 - tol / r).
    // Clamp the argument so an over-large tolerance maps to a = pi (one chord).
    let a = (1.0 - tol / r).clamp(-1.0, 1.0).acos();
    if a <= 0.0 {
        return MAX_SEGMENTS;
    }

    let n = (span / (2.0 * a)).ceil();
    if !n.is_finite() || n >= f64::from(MAX_SEGMENTS) {
        return MAX_SEGMENTS;
    }
    let n = n as u32;
    if n < 1 {
        1
    } else {
        n
    }
}

/// Flattens the circular arc centred at `center` with the given `radius`,
/// starting at `start_angle` and turning by `sweep` radians, into a polyline
/// whose chords stay within `tolerance` of the true arc.
///
/// The result starts exactly at the arc's start point and ends exactly at its
/// end point, with `arc_segment_count(..) + 1` samples equally spaced in angle.
/// A zero radius collapses to a single point at `center`; a zero sweep collapses
/// to the single start point.
#[must_use]
pub fn flatten_arc(
    center: (f32, f32),
    radius: f32,
    start_angle: f32,
    sweep: f32,
    tolerance: f32,
) -> Vec<(f32, f32)> {
    let cx = f64::from(center.0);
    let cy = f64::from(center.1);
    let r = f64::from(radius);
    let a0 = f64::from(start_angle);

    if radius.abs() <= 0.0 {
        return vec![center];
    }

    let sample = |ang: f64| -> (f32, f32) {
        ((cx + r * ang.cos()) as f32, (cy + r * ang.sin()) as f32)
    };

    if sweep == 0.0 {
        return vec![sample(a0)];
    }

    let n = arc_segment_count(radius, sweep, tolerance);
    let dsw = f64::from(sweep);
    let mut out: Vec<(f32, f32)> = Vec::with_capacity(n as usize + 1);
    for i in 0..=n {
        let t = f64::from(i) / f64::from(n);
        out.push(sample(a0 + dsw * t));
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
    use core::f64::consts::PI;

    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_in(state: &mut u64, lo: f64, hi: f64) -> f64 {
        let b = (next_rand(state) >> 40) as u32;
        lo + (hi - lo) * (f64::from(b) / 16_777_216.0)
    }

    /// True sagitta of a chord spanning `delta` radians on radius `r`.
    fn sagitta(r: f64, delta: f64) -> f64 {
        r * (1.0 - (delta / 2.0).cos())
    }

    /// Distance from point `p` to the segment `a`-`b`, all in f64.
    fn dist_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        let abx = b.0 - a.0;
        let aby = b.1 - a.1;
        let len2 = abx * abx + aby * aby;
        let t = if len2 <= 0.0 {
            0.0
        } else {
            (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len2).clamp(0.0, 1.0)
        };
        let qx = a.0 + t * abx;
        let qy = a.1 + t * aby;
        let dx = p.0 - qx;
        let dy = p.1 - qy;
        (dx * dx + dy * dy).sqrt()
    }

    /// Every sampled vertex lies on the circle of the requested radius.
    #[test]
    fn vertices_on_circle() {
        let mut state = 0x0_B0B1_u64;
        for _ in 0..300 {
            let center = (rand_in(&mut state, -20.0, 20.0) as f32, rand_in(&mut state, -20.0, 20.0) as f32);
            let radius = rand_in(&mut state, 0.5, 50.0) as f32;
            let start = rand_in(&mut state, -PI, PI) as f32;
            let sweep = rand_in(&mut state, -2.0 * PI, 2.0 * PI) as f32;
            let tol = rand_in(&mut state, 0.01, 1.0) as f32;
            let pts = flatten_arc(center, radius, start, sweep, tol);
            let r = f64::from(radius);
            let eps = 1e-3 * r + 1e-4;
            for &(x, y) in &pts {
                let dx = f64::from(x) - f64::from(center.0);
                let dy = f64::from(y) - f64::from(center.1);
                let d = (dx * dx + dy * dy).sqrt();
                assert!((d - r).abs() <= eps, "vertex off circle: {d} vs {r}");
            }
        }
    }

    /// The polyline starts and ends exactly on the arc endpoints.
    #[test]
    fn endpoints_exact() {
        let mut state = 0x1234_ABCD_u64;
        for _ in 0..300 {
            let center = (rand_in(&mut state, -10.0, 10.0) as f32, rand_in(&mut state, -10.0, 10.0) as f32);
            let radius = rand_in(&mut state, 0.5, 30.0) as f32;
            let start = rand_in(&mut state, -PI, PI) as f32;
            let sweep = rand_in(&mut state, -2.0 * PI, 2.0 * PI) as f32;
            let tol = rand_in(&mut state, 0.02, 0.5) as f32;
            let pts = flatten_arc(center, radius, start, sweep, tol);
            let r = f64::from(radius);
            let (cx, cy) = (f64::from(center.0), f64::from(center.1));
            let a0 = f64::from(start);
            let a1 = a0 + f64::from(sweep);
            let want0 = ((cx + r * a0.cos()) as f32, (cy + r * a0.sin()) as f32);
            let want1 = ((cx + r * a1.cos()) as f32, (cy + r * a1.sin()) as f32);
            assert_eq!(pts.first().copied(), Some(want0));
            assert_eq!(pts.last().copied(), Some(want1));
        }
    }

    /// Independent oracle: the true arc midpoint of every chord is within the
    /// tolerance of that chord.
    #[test]
    fn chord_error_within_tolerance() {
        let mut state = 0xDEAD_BEEF_u64;
        for _ in 0..300 {
            let center = (rand_in(&mut state, -15.0, 15.0) as f32, rand_in(&mut state, -15.0, 15.0) as f32);
            let radius = rand_in(&mut state, 1.0, 40.0) as f32;
            let start = rand_in(&mut state, -PI, PI) as f32;
            let sweep = rand_in(&mut state, -2.0 * PI, 2.0 * PI) as f32;
            let tol = rand_in(&mut state, 0.02, 0.8) as f32;
            let pts = flatten_arc(center, radius, start, sweep, tol);
            if pts.len() < 2 {
                continue;
            }
            let n = (pts.len() - 1) as u32;
            let r = f64::from(radius);
            let (cx, cy) = (f64::from(center.0), f64::from(center.1));
            let a0 = f64::from(start);
            let dsw = f64::from(sweep);
            // f32 sampling can nudge chord endpoints; allow a small slack.
            let eps = 2e-3 * r + 1e-3;
            for i in 0..n {
                let a = pts[i as usize];
                let b = pts[(i + 1) as usize];
                let mid = a0 + dsw * (f64::from(i) + 0.5) / f64::from(n);
                let truept = (cx + r * mid.cos(), cy + r * mid.sin());
                let d = dist_to_segment(
                    truept,
                    (f64::from(a.0), f64::from(a.1)),
                    (f64::from(b.0), f64::from(b.1)),
                );
                assert!(
                    d <= f64::from(tol) + eps,
                    "chord error {d} exceeds tol {tol} (eps {eps})"
                );
            }
        }
    }

    /// The segment count is minimal: `n` chords meet the sagitta bound and
    /// `n - 1` chords would not (whenever more than one chord is used).
    #[test]
    fn segment_count_minimal() {
        let mut state = 0xFEED_FACE_u64;
        for _ in 0..400 {
            let radius = rand_in(&mut state, 1.0, 60.0) as f32;
            let sweep = rand_in(&mut state, -2.0 * PI, 2.0 * PI) as f32;
            let tol = rand_in(&mut state, 0.02, 1.0) as f32;
            let n = arc_segment_count(radius, sweep, tol);
            let r = f64::from(radius);
            let span = f64::from(sweep.abs());
            let t = f64::from(tol);
            if span <= 0.0 {
                assert_eq!(n, 1);
                continue;
            }
            // n chords satisfy the bound (allow a hair of f64 slack).
            assert!(sagitta(r, span / f64::from(n)) <= t + 1e-9, "n={n} violates bound");
            // n - 1 chords (if any) would violate it: proves minimality.
            if n > 1 {
                assert!(
                    sagitta(r, span / f64::from(n - 1)) > t - 1e-9,
                    "n={n} is not minimal"
                );
            }
        }
    }

    /// A tighter tolerance never needs fewer segments.
    #[test]
    fn finer_tolerance_monotone() {
        let mut state = 0xCAFE_D00D_u64;
        for _ in 0..300 {
            let radius = rand_in(&mut state, 1.0, 50.0) as f32;
            let sweep = rand_in(&mut state, 0.1, 2.0 * PI) as f32;
            let coarse = rand_in(&mut state, 0.3, 1.0) as f32;
            let fine = rand_in(&mut state, 0.01, 0.2) as f32;
            let nc = arc_segment_count(radius, sweep, coarse);
            let nf = arc_segment_count(radius, sweep, fine);
            assert!(nf >= nc, "finer tol gave fewer segments: {nf} < {nc}");
        }
    }

    /// A full circle closes: first and last samples coincide.
    #[test]
    fn full_circle_closes() {
        let pts = flatten_arc((0.0, 0.0), 10.0, 0.0, (2.0 * PI) as f32, 0.05);
        let first = pts.first().copied().unwrap();
        let last = pts.last().copied().unwrap();
        assert!((first.0 - last.0).abs() <= 1e-3);
        assert!((first.1 - last.1).abs() <= 1e-3);
    }

    /// Fixed quarter circle: endpoints land on the axes.
    #[test]
    fn fixed_quarter_circle() {
        let pts = flatten_arc((0.0, 0.0), 1.0, 0.0, (PI / 2.0) as f32, 0.1);
        let first = pts.first().copied().unwrap();
        let last = pts.last().copied().unwrap();
        assert!((first.0 - 1.0).abs() <= 1e-6 && first.1.abs() <= 1e-6, "start {first:?}");
        assert!(last.0.abs() <= 1e-6 && (last.1 - 1.0).abs() <= 1e-6, "end {last:?}");
    }

    /// Degenerate inputs: zero radius -> centre point; zero sweep -> start point.
    #[test]
    fn degenerate_inputs() {
        assert_eq!(flatten_arc((3.0, 4.0), 0.0, 1.0, 2.0, 0.1), [(3.0, 4.0)]);
        let zero_sweep = flatten_arc((0.0, 0.0), 5.0, 0.0, 0.0, 0.1);
        assert_eq!(zero_sweep.len(), 1);
        assert!((zero_sweep[0].0 - 5.0).abs() <= 1e-6 && zero_sweep[0].1.abs() <= 1e-6);
    }
}
