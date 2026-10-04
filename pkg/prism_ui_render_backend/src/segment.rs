//! Line-segment intersection.
//!
//! Clipping, hit-testing against stroked paths, and the half-plane step of
//! convex polygon clipping all reduce to the same question: do two line
//! segments cross, and if so, where? [`segment_intersection`] answers both at
//! once, returning the single crossing point when the closed segments meet
//! transversally.
//!
//! The point is found parametrically. Writing the segments as `a0 + t·(a1-a0)`
//! and `b0 + u·(b1-b0)`, the two parameters come from a 2x2 solve whose
//! determinant is the cross product of the direction vectors. A zero
//! determinant means the segments are parallel (or collinear); collinear
//! overlap is a shared *sub-segment* rather than a single point, so it is
//! reported as no intersection — callers that need the overlap span should test
//! for collinearity separately.
//!
//! Only `+ - * /` and comparisons are used — no transcendental functions — so
//! the routine is `no_std`-clean and bit-stable across targets.

/// Returns the point where the closed segments `a0..=a1` and `b0..=b1` cross,
/// or `None` when they do not meet at a single point.
///
/// Endpoints count: a segment that touches the other segment's interior or
/// endpoint yields that contact point. Parallel segments — including collinear
/// ones that overlap along a shared span — return `None`, because the result is
/// not a single point. A zero-length segment (`a0 == a1`) is parallel to
/// everything and likewise returns `None`.
#[must_use]
pub fn segment_intersection(
    a0: (f32, f32),
    a1: (f32, f32),
    b0: (f32, f32),
    b1: (f32, f32),
) -> Option<(f32, f32)> {
    let r = (a1.0 - a0.0, a1.1 - a0.1);
    let s = (b1.0 - b0.0, b1.1 - b0.1);
    let denom = r.0 * s.1 - r.1 * s.0;
    if denom == 0.0 {
        return None; // parallel or collinear
    }
    let qp = (b0.0 - a0.0, b0.1 - a0.1);
    // t along segment a, u along segment b.
    let t = (qp.0 * s.1 - qp.1 * s.0) / denom;
    let u = (qp.0 * r.1 - qp.1 * r.0) / denom;
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some((a0.0 + t * r.0, a0.1 + t * r.1))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;

    /// Cross product of the vectors `o->a` and `o->b`; its sign gives the turn
    /// direction (positive for a left turn, negative for a right turn).
    fn cross(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

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

    fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Distance from `p` to the closed segment `a..=b`.
    fn point_segment_distance(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
        let ex = b.0 - a.0;
        let ey = b.1 - a.1;
        let len2 = ex * ex + ey * ey;
        let t = if len2 > 0.0 {
            (((p.0 - a.0) * ex + (p.1 - a.1) * ey) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        dist(p, (a.0 + ex * t, a.1 + ey * t))
    }

    /// Classic orientation-based predicate (CLRS): the open segments cross iff
    /// each segment's endpoints straddle the other's supporting line. Distinct
    /// in form from the parametric solve, so it is an independent oracle for the
    /// transversal (non-touching) case.
    fn straddles(a0: (f32, f32), a1: (f32, f32), b0: (f32, f32), b1: (f32, f32)) -> bool {
        let d1 = cross(a0, a1, b0);
        let d2 = cross(a0, a1, b1);
        let d3 = cross(b0, b1, a0);
        let d4 = cross(b0, b1, a1);
        ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
    }

    #[test]
    fn simple_crossing() {
        let p = segment_intersection((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)).unwrap();
        assert!((p.0 - 1.0).abs() < 1e-6 && (p.1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parallel_and_collinear_return_none() {
        // Parallel, disjoint.
        assert_eq!(
            segment_intersection((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)),
            None
        );
        // Collinear, overlapping — a span, not a point.
        assert_eq!(
            segment_intersection((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
            None
        );
        // Zero-length segment.
        assert_eq!(
            segment_intersection((1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (2.0, 2.0)),
            None
        );
    }

    #[test]
    fn disjoint_non_parallel_misses() {
        // Lines cross, but outside both segment spans.
        assert_eq!(
            segment_intersection((0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (2.0, 1.0)),
            None
        );
    }

    #[test]
    fn touching_endpoint_is_reported() {
        // b starts exactly on a's interior (a T-junction).
        let p = segment_intersection((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (2.0, 3.0)).unwrap();
        assert!((p.0 - 2.0).abs() < 1e-6 && p.1.abs() < 1e-6);
    }

    #[test]
    fn intersection_lies_on_both_segments() {
        // Whenever a point is returned, it must sit on both segments.
        let mut state = 0x1357_2468_ACE0_9BDF_u64;
        for _ in 0..2000 {
            let a0 = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            let a1 = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            let b0 = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            let b1 = (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0));
            if let Some(p) = segment_intersection(a0, a1, b0, b1) {
                assert!(point_segment_distance(p, a0, a1) < 1e-2, "off segment a: {p:?}");
                assert!(point_segment_distance(p, b0, b1) < 1e-2, "off segment b: {p:?}");
            }
        }
    }

    #[test]
    fn agrees_with_orientation_predicate() {
        // Away from the degenerate/boundary cases, Some(..) must match the
        // independent straddle predicate.
        let mut state = 0xACE0_9BDF_1357_2468_u64;
        for _ in 0..4000 {
            let a0 = (rand_in(&mut state, -8.0, 8.0), rand_in(&mut state, -8.0, 8.0));
            let a1 = (rand_in(&mut state, -8.0, 8.0), rand_in(&mut state, -8.0, 8.0));
            let b0 = (rand_in(&mut state, -8.0, 8.0), rand_in(&mut state, -8.0, 8.0));
            let b1 = (rand_in(&mut state, -8.0, 8.0), rand_in(&mut state, -8.0, 8.0));
            // Skip near-parallel and near-touching configs where the open-segment
            // predicate and the closed-segment solver legitimately differ.
            let r = (a1.0 - a0.0, a1.1 - a0.1);
            let s = (b1.0 - b0.0, b1.1 - b0.1);
            let denom = r.0 * s.1 - r.1 * s.0;
            if denom.abs() < 1.0 {
                continue;
            }
            let near_boundary = [
                cross(a0, a1, b0),
                cross(a0, a1, b1),
                cross(b0, b1, a0),
                cross(b0, b1, a1),
            ]
            .iter()
            .any(|c| c.abs() < 0.5);
            if near_boundary {
                continue;
            }
            assert_eq!(
                segment_intersection(a0, a1, b0, b1).is_some(),
                straddles(a0, a1, b0, b1),
                "a {a0:?}->{a1:?}, b {b0:?}->{b1:?}"
            );
        }
    }

    #[test]
    fn argument_order_is_symmetric() {
        // Swapping the two segments yields the same crossing point.
        let mut state = 0x9BDF_1357_2468_ACE0_u64;
        for _ in 0..2000 {
            let a0 = (rand_in(&mut state, -6.0, 6.0), rand_in(&mut state, -6.0, 6.0));
            let a1 = (rand_in(&mut state, -6.0, 6.0), rand_in(&mut state, -6.0, 6.0));
            let b0 = (rand_in(&mut state, -6.0, 6.0), rand_in(&mut state, -6.0, 6.0));
            let b1 = (rand_in(&mut state, -6.0, 6.0), rand_in(&mut state, -6.0, 6.0));
            let ab = segment_intersection(a0, a1, b0, b1);
            let ba = segment_intersection(b0, b1, a0, a1);
            match (ab, ba) {
                (Some(p), Some(q)) => assert!(dist(p, q) < 1e-2, "{p:?} vs {q:?}"),
                (None, None) => {}
                _ => panic!("symmetry broken: {ab:?} vs {ba:?}"),
            }
        }
    }
}
