//! Adaptive flattening of elliptical arcs into polylines.
//!
//! [`crate::arc`] flattens *circular* arcs, where curvature is constant and a
//! uniform angular step suffices. An ellipse bends unevenly — tightly at the
//! ends of its major axis, gently at the ends of its minor axis — so a uniform
//! step either oversamples the flat parts or violates the tolerance at the
//! sharp ones. [`flatten_ellipse`] instead subdivides *adaptively*: it splits a
//! parameter span only where the chord strays from the true curve by more than
//! `tolerance`, spending samples where curvature demands them.
//!
//! The ellipse is axis-aligned and parametrised by the eccentric angle `t`:
//! `P(t) = center + (rx*cos t, ry*sin t)`. Transcendental evaluation is done in
//! `f64` (f32 trig is banned workspace-wide) and cast to `f32`. The output
//! starts exactly at the start point and ends exactly at the end point.

use alloc::vec;
use alloc::vec::Vec;

/// Hard recursion-depth cap so pathological inputs still terminate.
const MAX_DEPTH: u32 = 20;
/// Hard output cap so a tiny tolerance cannot explode the vertex count.
const MAX_POINTS: usize = 8192;

/// Interior parameter fractions at which a candidate chord is checked against
/// the true curve. Denser than a lone midpoint so an off-centre bulge is caught.
const CHECK_FRACTIONS: [f64; 7] = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];

/// Distance from point `p` to segment `a`-`b`, all in `f64`. A degenerate
/// (zero-length) segment reduces to the distance to its endpoint.
fn dist_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let abx = b.0 - a.0;
    let aby = b.1 - a.1;
    let len2 = abx * abx + aby * aby;
    let (dx, dy) = if len2 <= 0.0 {
        (p.0 - a.0, p.1 - a.1)
    } else {
        let t = (((p.0 - a.0) * abx + (p.1 - a.1) * aby) / len2).clamp(0.0, 1.0);
        (p.0 - (a.0 + t * abx), p.1 - (a.1 + t * aby))
    };
    (dx * dx + dy * dy).sqrt()
}

/// Recursively flattens the parameter span `[t0, t1]`, appending the end point
/// of each accepted chord to `out`. `p0`/`p1` are the already-sampled curve
/// points at `t0`/`t1`.
#[expect(clippy::too_many_arguments, reason = "internal recursion carries its working state")]
fn subdivide(
    sample: &impl Fn(f64) -> (f64, f64),
    t0: f64,
    t1: f64,
    p0: (f64, f64),
    p1: (f64, f64),
    tol: f64,
    depth: u32,
    out: &mut Vec<(f64, f64)>,
) {
    if depth == 0 || out.len() >= MAX_POINTS {
        out.push(p1);
        return;
    }

    let mut within = true;
    for &f in &CHECK_FRACTIONS {
        let pm = sample(t0 + (t1 - t0) * f);
        if dist_to_segment(pm, p0, p1) > tol {
            within = false;
            break;
        }
    }

    if within {
        out.push(p1);
    } else {
        let tm = (t0 + t1) * 0.5;
        let pm = sample(tm);
        subdivide(sample, t0, tm, p0, pm, tol, depth - 1, out);
        subdivide(sample, tm, t1, pm, p1, tol, depth - 1, out);
    }
}

/// Flattens the axis-aligned elliptical arc centred at `center` with radii
/// `(rx, ry)`, from eccentric angle `start_angle`, sweeping `sweep` radians,
/// into a polyline that stays within `tolerance` of the true ellipse.
///
/// The result starts exactly at `P(start_angle)` and ends exactly at
/// `P(start_angle + sweep)`, with interior vertices placed by adaptive
/// subdivision. A non-positive `tolerance` is treated as a very small positive
/// value; the vertex count is still bounded by an internal cap. A zero sweep
/// collapses to the single start point; radii that are both zero collapse to a
/// single point at `center`.
#[must_use]
pub fn flatten_ellipse(
    center: (f32, f32),
    radii: (f32, f32),
    start_angle: f32,
    sweep: f32,
    tolerance: f32,
) -> Vec<(f32, f32)> {
    let cx = f64::from(center.0);
    let cy = f64::from(center.1);
    let rx = f64::from(radii.0);
    let ry = f64::from(radii.1);
    let a0 = f64::from(start_angle);

    let sample = |t: f64| -> (f64, f64) { (cx + rx * t.cos(), cy + ry * t.sin()) };
    let to_f32 = |p: (f64, f64)| -> (f32, f32) { (p.0 as f32, p.1 as f32) };

    if radii.0.abs() <= 0.0 && radii.1.abs() <= 0.0 {
        return vec![center];
    }
    if sweep == 0.0 {
        return vec![to_f32(sample(a0))];
    }

    let mut tol = f64::from(tolerance);
    if tol <= 0.0 || tol.is_nan() {
        tol = 1e-6;
    }

    let a1 = a0 + f64::from(sweep);
    let p0 = sample(a0);
    let p1 = sample(a1);

    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(64);
    pts.push(p0);
    subdivide(&sample, a0, a1, p0, p1, tol, MAX_DEPTH, &mut pts);

    pts.into_iter().map(to_f32).collect()
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

    fn dist_to_segment_f64(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        dist_to_segment(p, a, b)
    }

    /// Minimum distance from `p` to the polyline `poly` (treated as open).
    fn dist_to_polyline(p: (f64, f64), poly: &[(f32, f32)]) -> f64 {
        let mut best = f64::INFINITY;
        for w in poly.windows(2) {
            let a = (f64::from(w[0].0), f64::from(w[0].1));
            let b = (f64::from(w[1].0), f64::from(w[1].1));
            let d = dist_to_segment_f64(p, a, b);
            if d < best {
                best = d;
            }
        }
        best
    }

    /// Every emitted vertex lies on the ellipse: the implicit form evaluates to 1.
    #[test]
    fn vertices_on_ellipse() {
        let mut state = 0x0_E111_u64;
        for _ in 0..300 {
            let cx = rand_in(&mut state, -5.0, 5.0);
            let cy = rand_in(&mut state, -5.0, 5.0);
            let rx = rand_in(&mut state, 1.0, 15.0);
            let ry = rand_in(&mut state, 1.0, 15.0);
            let start = rand_in(&mut state, -PI, PI);
            let sweep = rand_in(&mut state, -2.0 * PI, 2.0 * PI);
            let poly = flatten_ellipse(
                (cx as f32, cy as f32),
                (rx as f32, ry as f32),
                start as f32,
                sweep as f32,
                0.05,
            );
            for &(x, y) in &poly {
                let nx = (f64::from(x) - cx) / rx;
                let ny = (f64::from(y) - cy) / ry;
                let implicit = nx * nx + ny * ny;
                assert!(
                    (implicit - 1.0).abs() <= 1e-3,
                    "vertex off ellipse: implicit {implicit}"
                );
            }
        }
    }

    /// Endpoints are reproduced exactly (same f64->f32 cast as the sampler).
    #[test]
    fn endpoints_exact() {
        let mut state = 0x1_E222_u64;
        for _ in 0..200 {
            let cx = rand_in(&mut state, -4.0, 4.0);
            let cy = rand_in(&mut state, -4.0, 4.0);
            let rx = rand_in(&mut state, 1.0, 12.0);
            let ry = rand_in(&mut state, 1.0, 12.0);
            let start = rand_in(&mut state, -PI, PI);
            let sweep = rand_in(&mut state, 0.2, 2.0 * PI);
            let poly = flatten_ellipse(
                (cx as f32, cy as f32),
                (rx as f32, ry as f32),
                start as f32,
                sweep as f32,
                0.08,
            );
            // Mirror the sampler exactly: inputs reach the function as f32 and
            // are promoted with `f64::from` before the trig, so the oracle must
            // use the same f32-rounded inputs rather than the wider f64 draws.
            let cxf = f64::from(cx as f32);
            let cyf = f64::from(cy as f32);
            let rxf = f64::from(rx as f32);
            let ryf = f64::from(ry as f32);
            let a0 = f64::from(start as f32);
            let a1 = a0 + f64::from(sweep as f32);
            let want_start = ((cxf + rxf * a0.cos()) as f32, (cyf + ryf * a0.sin()) as f32);
            let want_end = ((cxf + rxf * a1.cos()) as f32, (cyf + ryf * a1.sin()) as f32);
            assert_eq!(poly[0], want_start, "start endpoint mismatch");
            assert_eq!(*poly.last().unwrap(), want_end, "end endpoint mismatch");
        }
    }

    /// Independent oracle: densely sampled true-curve points stay within the
    /// tolerance of the flattened polyline.
    #[test]
    fn within_tolerance() {
        let mut state = 0x2_E333_u64;
        for _ in 0..150 {
            let cx = rand_in(&mut state, -3.0, 3.0);
            let cy = rand_in(&mut state, -3.0, 3.0);
            let rx = rand_in(&mut state, 1.0, 14.0);
            let ry = rand_in(&mut state, 1.0, 14.0);
            let start = rand_in(&mut state, -PI, PI);
            let sweep = rand_in(&mut state, 0.3, 2.0 * PI);
            let tol = rand_in(&mut state, 0.05, 0.3);
            let poly = flatten_ellipse(
                (cx as f32, cy as f32),
                (rx as f32, ry as f32),
                start as f32,
                sweep as f32,
                tol as f32,
            );
            assert!(poly.len() >= 2);
            let steps = 2000;
            for k in 0..=steps {
                let t = start + sweep * (f64::from(k) / f64::from(steps));
                let p = (cx + rx * t.cos(), cy + ry * t.sin());
                let d = dist_to_polyline(p, &poly);
                assert!(
                    d <= tol + 2e-3,
                    "curve point {d} from polyline exceeds tol {tol}"
                );
            }
        }
    }

    /// A finer tolerance never produces fewer vertices.
    #[test]
    fn finer_tolerance_more_points() {
        let mut state = 0x3_E444_u64;
        for _ in 0..150 {
            let rx = rand_in(&mut state, 2.0, 14.0);
            let ry = rand_in(&mut state, 2.0, 14.0);
            let start = rand_in(&mut state, -PI, PI);
            let sweep = rand_in(&mut state, 0.5, 2.0 * PI);
            let coarse = flatten_ellipse((0.0, 0.0), (rx as f32, ry as f32), start as f32, sweep as f32, 0.4);
            let fine = flatten_ellipse((0.0, 0.0), (rx as f32, ry as f32), start as f32, sweep as f32, 0.02);
            assert!(
                fine.len() >= coarse.len(),
                "finer tolerance gave fewer points: {} < {}",
                fine.len(),
                coarse.len()
            );
        }
    }

    /// A full sweep returns to the start point.
    #[test]
    fn full_ellipse_closes() {
        let poly = flatten_ellipse((1.0, 2.0), (5.0, 3.0), 0.0, (2.0 * PI) as f32, 0.05);
        let first = poly[0];
        let last = *poly.last().unwrap();
        assert!((first.0 - last.0).abs() <= 1e-4);
        assert!((first.1 - last.1).abs() <= 1e-4);
    }

    /// A circle (rx == ry) flattened this way agrees with the dedicated circular
    /// arc flattener to within both tolerances.
    #[test]
    fn circle_matches_arc() {
        use crate::arc::flatten_arc;
        let r = 7.0_f32;
        let tol = 0.1_f32;
        let ell = flatten_ellipse((0.0, 0.0), (r, r), 0.0, (1.5 * PI) as f32, tol);
        let arc = flatten_arc((0.0, 0.0), r, 0.0, (1.5 * PI) as f32, tol);
        // Each arc sample is on the same circle; verify the ellipse polyline
        // tracks it within tolerance.
        for &(x, y) in &arc {
            let d = dist_to_polyline((f64::from(x), f64::from(y)), &ell);
            assert!(d <= f64::from(tol) + 2e-3, "arc point {d} from ellipse polyline");
        }
    }

    /// Degenerate inputs collapse cleanly.
    #[test]
    fn degenerate_inputs() {
        assert_eq!(flatten_ellipse((1.0, 2.0), (0.0, 0.0), 0.0, 1.0, 0.1), vec![(1.0, 2.0)]);
        let zero_sweep = flatten_ellipse((0.0, 0.0), (3.0, 4.0), 0.5, 0.0, 0.1);
        assert_eq!(zero_sweep.len(), 1);
    }
}
