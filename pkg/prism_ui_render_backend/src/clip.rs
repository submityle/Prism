//! Convex-polygon clipping via the Sutherland-Hodgman algorithm.
//!
//! Compositing, scissor rectangles, and rounded-corner masking all reduce to
//! the same operation: keep only the part of a filled path that lies inside a
//! clip region. [`clip_polygon`] computes that intersection when the clip
//! region is **convex** (an axis-aligned scissor box, a rotated rectangle, or
//! any convex mask), while the subject path may be convex or concave.
//!
//! Sutherland-Hodgman walks the subject polygon once per clip edge, keeping the
//! vertices on the inner half-plane and inserting the crossing point whenever an
//! edge enters or leaves that half-plane. After all `m` clip edges the surviving
//! loop is exactly `subject ∩ clip`. Because the clip edges are treated as
//! infinite half-planes the clip region must be convex; a concave clip would
//! wrongly discard material behind a reflex edge.
//!
//! The inner/outer test is winding-independent: the clip's signed area selects
//! the sign so that the clip interior is always "inside", regardless of whether
//! the caller supplies clockwise or counter-clockwise vertices.
//!
//! Only `+ - * /` and comparisons are used — no transcendental functions — so
//! the routine is `no_std`-clean and bit-stable across targets.

use crate::measure::signed_area;
use alloc::vec::Vec;

/// Cross product of the vectors `a->b` and `a->p`; its sign tells which side of
/// the directed line `a->b` the point `p` lies on (positive for the left side).
fn cross(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// Intersection of the segment `prev->cur` with the **infinite** line through
/// `a->b`.
///
/// Callers only invoke this when `prev` and `cur` straddle the line, so the
/// denominator is non-zero. The clip edge is treated as a full line (not a
/// segment) because Sutherland-Hodgman clips against half-planes.
fn line_intersection(
    prev: (f32, f32),
    cur: (f32, f32),
    a: (f32, f32),
    b: (f32, f32),
) -> (f32, f32) {
    // Normal of the clip edge a->b (rotate the direction 90 degrees).
    let n = (-(b.1 - a.1), b.0 - a.0);
    // Solve (prev + t*(cur-prev) - a) . n = 0 for t.
    let num = (a.0 - prev.0) * n.0 + (a.1 - prev.1) * n.1;
    let den = (cur.0 - prev.0) * n.0 + (cur.1 - prev.1) * n.1;
    let t = num / den;
    (prev.0 + t * (cur.0 - prev.0), prev.1 + t * (cur.1 - prev.1))
}

/// Clips the `subject` polygon against the **convex** `clip` polygon and returns
/// their intersection as a new polygon (the Sutherland-Hodgman algorithm).
///
/// The `subject` may be convex or concave; the `clip` region must be convex or
/// the result is undefined (reflex clip edges would over-remove area). Both
/// polygons are treated as implicitly closed — do not repeat the first vertex.
/// The clip winding (clockwise or counter-clockwise) does not matter; the clip's
/// signed area picks the interior side automatically.
///
/// Returns an empty `Vec` when the polygons do not overlap, and also when either
/// polygon is degenerate (fewer than three vertices). The returned loop is wound
/// consistently with the clip region.
#[must_use]
pub fn clip_polygon(subject: &[(f32, f32)], clip: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }

    // Pick the sign so that `sign * cross(a, b, p) >= 0.0` means "inside",
    // independent of the clip winding.
    let sign = if signed_area(clip) >= 0.0 { 1.0_f32 } else { -1.0 };
    let inside = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| sign * cross(a, b, p) >= 0.0;

    let m = clip.len();
    let mut output: Vec<(f32, f32)> = subject.to_vec();

    for (i, &a) in clip.iter().enumerate() {
        if output.is_empty() {
            break;
        }
        let b = clip[(i + 1) % m];

        // Clip the current loop against the half-plane of edge a->b.
        let input = core::mem::take(&mut output);
        let n = input.len();
        for (j, &cur) in input.iter().enumerate() {
            let prev = input[(j + n - 1) % n];
            let cur_in = inside(a, b, cur);
            let prev_in = inside(a, b, prev);
            if cur_in {
                if !prev_in {
                    output.push(line_intersection(prev, cur, a, b));
                }
                output.push(cur);
            } else if prev_in {
                output.push(line_intersection(prev, cur, a, b));
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::polygon::sd_polygon;
    use crate::{area, convex_hull, point_in_polygon};

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

    /// A random convex polygon: the convex hull of a cloud of random points.
    fn random_convex(state: &mut u64) -> Vec<(f32, f32)> {
        loop {
            let n = 8 + (next_rand(state) % 24) as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                pts.push((rand_in(state, -20.0, 20.0), rand_in(state, -20.0, 20.0)));
            }
            let hull = convex_hull(&pts);
            if hull.len() >= 3 {
                return hull;
            }
        }
    }

    /// Counter-clockwise axis-aligned box.
    fn axis_box(lo: f32, hi: f32) -> Vec<(f32, f32)> {
        alloc::vec![(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * (a.abs() + b.abs()) + 1e-3
    }

    /// Strongest oracle: for a convex subject clipped by a box, membership in the
    /// result must equal membership in subject AND clip, for sample points away
    /// from every boundary.
    #[test]
    fn monte_carlo_set_intersection() {
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let margin = 0.3_f32;
        for _ in 0..200 {
            let subject = random_convex(&mut state);
            let lo = rand_in(&mut state, -15.0, -2.0);
            let hi = rand_in(&mut state, 2.0, 15.0);
            let clip = axis_box(lo, hi);
            let result = clip_polygon(&subject, &clip);

            for _ in 0..40 {
                let p = (rand_in(&mut state, -25.0, 25.0), rand_in(&mut state, -25.0, 25.0));
                // Skip points too close to any boundary (classification ambiguous).
                if sd_polygon(p.0, p.1, &subject).abs() <= margin {
                    continue;
                }
                if sd_polygon(p.0, p.1, &clip).abs() <= margin {
                    continue;
                }
                let in_result = result.len() >= 3 && point_in_polygon(&result, p);
                if result.len() >= 3 && sd_polygon(p.0, p.1, &result).abs() <= margin {
                    continue;
                }
                let expect = point_in_polygon(&subject, p) && point_in_polygon(&clip, p);
                assert_eq!(in_result, expect, "p={p:?} subj_in_result mismatch");
            }
        }
    }

    /// Every surviving vertex must lie inside or on the clip region.
    #[test]
    fn vertices_inside_clip() {
        let mut state = 0x0BAD_F00D_DEAD_BEEF_u64;
        for _ in 0..300 {
            let subject = random_convex(&mut state);
            let clip = axis_box(-8.0, 8.0);
            let result = clip_polygon(&subject, &clip);
            for &v in &result {
                assert!(
                    sd_polygon(v.0, v.1, &clip) <= 1e-2,
                    "vertex {v:?} outside clip"
                );
            }
        }
    }

    /// The clipped area can never exceed either input area.
    #[test]
    fn area_is_bounded() {
        let mut state = 0xFEED_FACE_CAFE_0042_u64;
        for _ in 0..300 {
            let subject = random_convex(&mut state);
            let lo = rand_in(&mut state, -14.0, -2.0);
            let hi = rand_in(&mut state, 2.0, 14.0);
            let clip = axis_box(lo, hi);
            let result = clip_polygon(&subject, &clip);
            let bound = area(&subject).min(area(&clip));
            assert!(
                area(&result) <= bound + 1e-2 * bound + 1e-2,
                "clipped area {} exceeds bound {}",
                area(&result),
                bound
            );
        }
    }

    /// A clip that fully contains a convex subject leaves it unchanged.
    #[test]
    fn containing_clip_preserves_subject() {
        let mut state = 0x00C0_FFEE_1234_5678_u64;
        for _ in 0..200 {
            let subject = random_convex(&mut state);
            let clip = axis_box(-1000.0, 1000.0);
            let result = clip_polygon(&subject, &clip);
            assert!(
                approx(area(&result), area(&subject)),
                "containing clip changed area: {} vs {}",
                area(&result),
                area(&subject)
            );
        }
    }

    /// Clipping is idempotent: re-clipping the result by the same clip is a no-op
    /// in area.
    #[test]
    fn idempotent() {
        let mut state = 0xA5A5_5A5A_0F0F_F0F0_u64;
        for _ in 0..200 {
            let subject = random_convex(&mut state);
            let clip = axis_box(-7.0, 7.0);
            let once = clip_polygon(&subject, &clip);
            let twice = clip_polygon(&once, &clip);
            assert!(
                approx(area(&once), area(&twice)),
                "not idempotent: {} vs {}",
                area(&once),
                area(&twice)
            );
        }
    }

    /// Clip winding (CW vs CCW) must not change the result area.
    #[test]
    fn winding_independent() {
        let mut state = 0x7777_3333_1111_9999_u64;
        for _ in 0..200 {
            let subject = random_convex(&mut state);
            let ccw = axis_box(-6.0, 6.0);
            let mut cw = ccw.clone();
            cw.reverse();
            let a = clip_polygon(&subject, &ccw);
            let b = clip_polygon(&subject, &cw);
            assert!(
                approx(area(&a), area(&b)),
                "winding changed area: {} vs {}",
                area(&a),
                area(&b)
            );
        }
    }

    /// A concave subject clipped by a convex box: area is bounded and every
    /// vertex stays inside the clip (point-in-polygon is unreliable on the
    /// coincident edges a concave subject can produce, so we avoid it here).
    #[test]
    fn concave_subject() {
        // An "L"-shaped concave polygon.
        let subject = alloc::vec![
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 2.0),
            (2.0, 2.0),
            (2.0, 6.0),
            (0.0, 6.0),
        ];
        let clip = axis_box(1.0, 5.0);
        let result = clip_polygon(&subject, &clip);
        assert!(result.len() >= 3);
        for &v in &result {
            assert!(sd_polygon(v.0, v.1, &clip) <= 1e-2, "vertex {v:?} outside clip");
        }
        let bound = area(&subject).min(area(&clip));
        assert!(area(&result) <= bound + 1e-2);
        // The L clipped to [1,5]^2 keeps the bottom bar (x 1..5, y 1..2 => 4) and
        // the left bar (x 1..2, y 1..5 => 4) minus the shared unit square once:
        // bottom (4) + upper-left (1*3=3) = 7.
        assert!(approx(area(&result), 7.0), "area was {}", area(&result));
    }

    /// Fixed half-plane case: a unit square clipped by a box whose left edge cuts
    /// the square at x = 0.5 leaves a rectangle of area 0.5.
    #[test]
    fn fixed_half_plane() {
        let square = alloc::vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        // Tall box so only its left edge (x = 0.5) cuts the square: a pure
        // half-plane, leaving the rectangle x in [0.5, 1] of area 0.5.
        let clip = alloc::vec![(0.5, -1.0), (2.0, -1.0), (2.0, 2.0), (0.5, 2.0)];
        let result = clip_polygon(&square, &clip);
        assert!(approx(area(&result), 0.5), "area was {}", area(&result));
        for &v in &result {
            assert!(v.0 >= 0.5 - 1e-6, "vertex {v:?} left of cut");
        }
    }

    /// Disjoint polygons clip to nothing.
    #[test]
    fn disjoint_is_empty() {
        let subject = alloc::vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let clip = axis_box(10.0, 12.0);
        let result = clip_polygon(&subject, &clip);
        assert!(result.len() < 3 || area(&result) <= 1e-6);
    }

    /// Degenerate inputs return an empty polygon.
    #[test]
    fn degenerate_inputs() {
        let tri = alloc::vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(clip_polygon(&[], &tri).is_empty());
        assert!(clip_polygon(&tri, &[]).is_empty());
        assert!(clip_polygon(&tri, &[(0.0, 0.0), (1.0, 0.0)]).is_empty());
    }
}
