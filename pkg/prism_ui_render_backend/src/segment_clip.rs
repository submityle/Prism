//! Liang-Barsky clipping of a line segment against an axis-aligned rectangle.
//!
//! Stroked paths, debug lines, and vector primitives all have to be trimmed to
//! the active scissor box before rasterisation. [`crate::clip`] trims a *filled
//! polygon* against a convex region; this module trims a single *open segment*
//! against an axis-aligned rectangle, returning the portion that survives.
//!
//! The Liang-Barsky method parametrises the segment as `P(t) = a + t·(b - a)`
//! for `t` in `[0, 1]` and intersects that parameter interval with the four
//! half-planes of the rectangle. Each boundary contributes an inequality
//! `p·t <= q`; an entering boundary (`p < 0`) raises the lower bound `t0`, a
//! leaving boundary (`p > 0`) lowers the upper bound `t1`, and a boundary the
//! segment runs parallel to (`p == 0`) rejects the whole segment when it starts
//! outside that edge (`q < 0`). If `t0 > t1` after processing all four edges the
//! segment misses the rectangle entirely.
//!
//! Only `+ - * /` and comparisons are used — no transcendental functions — so
//! the routine is `no_std`-clean and bit-stable across targets. The returned
//! endpoints are evaluated from the clamped parameters, so they lie exactly on
//! the original segment.

/// Clips the closed segment `a..=b` to the closed, axis-aligned rectangle whose
/// opposite corners are `corner0` and `corner1` (given in any order), returning
/// the surviving sub-segment or `None` when the segment lies wholly outside.
///
/// The rectangle is normalised internally, so the two corners may be supplied in
/// any order. A degenerate segment (`a == b`) is treated as a point: it returns
/// `Some((a, a))` when the point lies in the rectangle and `None` otherwise. A
/// degenerate rectangle (zero width and/or height) is a valid closed region — a
/// line or a point — and clips normally. The returned endpoints are the exact
/// values of `P(t) = a + t·(b - a)` at the clamped parameters and therefore lie
/// on the input segment.
#[must_use]
pub fn clip_segment_rect(
    a: (f32, f32),
    b: (f32, f32),
    corner0: (f32, f32),
    corner1: (f32, f32),
) -> Option<((f32, f32), (f32, f32))> {
    let xmin = corner0.0.min(corner1.0);
    let xmax = corner0.0.max(corner1.0);
    let ymin = corner0.1.min(corner1.1);
    let ymax = corner0.1.max(corner1.1);

    let dx = b.0 - a.0;
    let dy = b.1 - a.1;

    let mut t0 = 0.0_f32;
    let mut t1 = 1.0_f32;

    // Each (p, q) encodes a boundary inequality `p·t <= q`. Order: left, right,
    // bottom, top. `p < 0` is an entering edge, `p > 0` a leaving edge.
    let edges = [
        (-dx, a.0 - xmin), // left:   x >= xmin
        (dx, xmax - a.0),  // right:  x <= xmax
        (-dy, a.1 - ymin), // bottom: y >= ymin
        (dy, ymax - a.1),  // top:    y <= ymax
    ];

    for (p, q) in edges {
        if p == 0.0 {
            // Parallel to this edge: reject only if it starts outside it.
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                // Entering: raise the lower bound.
                if r > t1 {
                    return None;
                }
                if r > t0 {
                    t0 = r;
                }
            } else {
                // Leaving: lower the upper bound.
                if r < t0 {
                    return None;
                }
                if r < t1 {
                    t1 = r;
                }
            }
        }
    }

    if t0 > t1 {
        return None;
    }

    let p0 = (a.0 + t0 * dx, a.1 + t0 * dy);
    let p1 = (a.0 + t1 * dx, a.1 + t1 * dy);
    Some((p0, p1))
}

#[cfg(test)]
mod tests {
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
        lo + (hi - lo) * (bits as f32 / 16_777_216.0)
    }

    /// Strictly-inside test with a margin so boundary-touching samples near the
    /// floating-point edge do not make the oracle flaky.
    fn strictly_inside(p: (f32, f32), lo: (f32, f32), hi: (f32, f32), margin: f32) -> bool {
        p.0 > lo.0 + margin
            && p.0 < hi.0 - margin
            && p.1 > lo.1 + margin
            && p.1 < hi.1 - margin
    }

    fn in_closed(p: (f32, f32), lo: (f32, f32), hi: (f32, f32), eps: f32) -> bool {
        p.0 >= lo.0 - eps && p.0 <= hi.0 + eps && p.1 >= lo.1 - eps && p.1 <= hi.1 + eps
    }

    /// Independent oracle: for a dense sweep of `t`, a point `P(t)` that is
    /// strictly inside the rectangle must fall inside the returned parameter
    /// span; and when the function reports no overlap, no sampled point may be
    /// strictly inside. This pins the result to the true inside-interval without
    /// reusing the Liang-Barsky arithmetic.
    #[test]
    fn matches_bruteforce_inside_interval() {
        let mut state = 0x5_C11A_u64;
        for _ in 0..400 {
            let a = (rand_in(&mut state, -6.0, 6.0), rand_in(&mut state, -6.0, 6.0));
            let b = (rand_in(&mut state, -6.0, 6.0), rand_in(&mut state, -6.0, 6.0));
            let c0 = (rand_in(&mut state, -4.0, 4.0), rand_in(&mut state, -4.0, 4.0));
            let c1 = (rand_in(&mut state, -4.0, 4.0), rand_in(&mut state, -4.0, 4.0));
            let lo = (c0.0.min(c1.0), c0.1.min(c1.1));
            let hi = (c0.0.max(c1.0), c0.1.max(c1.1));

            let clipped = clip_segment_rect(a, b, c0, c1);

            // Recover the inside-interval of t by dense sampling.
            let n = 2000;
            let mut inside_lo = f32::INFINITY;
            let mut inside_hi = f32::NEG_INFINITY;
            let mut any_inside = false;
            for i in 0..=n {
                let t = i as f32 / n as f32;
                let p = (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1));
                if strictly_inside(p, lo, hi, 1e-3) {
                    any_inside = true;
                    if t < inside_lo {
                        inside_lo = t;
                    }
                    if t > inside_hi {
                        inside_hi = t;
                    }
                }
            }

            match clipped {
                Some((p0, p1)) => {
                    // Returned endpoints lie on the closed rectangle.
                    assert!(in_closed(p0, lo, hi, 2e-3), "p0 outside rect: {p0:?}");
                    assert!(in_closed(p1, lo, hi, 2e-3), "p1 outside rect: {p1:?}");
                    // Returned endpoints lie on the original segment.
                    assert!(on_segment(a, b, p0), "p0 off segment: {p0:?}");
                    assert!(on_segment(a, b, p1), "p1 off segment: {p1:?}");
                    if any_inside {
                        // The strictly-inside samples must be covered by [t0, t1].
                        let t_p0 = recover_t(a, b, p0);
                        let t_p1 = recover_t(a, b, p1);
                        let (tlo, thi) = (t_p0.min(t_p1), t_p0.max(t_p1));
                        assert!(
                            tlo <= inside_lo + 2e-3 && thi >= inside_hi - 2e-3,
                            "interval [{tlo},{thi}] misses inside span [{inside_lo},{inside_hi}]"
                        );
                    }
                }
                None => {
                    assert!(
                        !any_inside,
                        "reported no overlap but a sample is strictly inside"
                    );
                }
            }
        }
    }

    /// Parameter `t` of the point `p` along segment `a..=b`, using the axis with
    /// the larger extent for numerical stability.
    fn recover_t(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        if dx.abs() >= dy.abs() {
            if dx == 0.0 {
                0.0
            } else {
                (p.0 - a.0) / dx
            }
        } else {
            (p.1 - a.1) / dy
        }
    }

    /// Whether `p` lies on the closed segment `a..=b` within a small tolerance.
    fn on_segment(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> bool {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let len2 = dx * dx + dy * dy;
        if len2 == 0.0 {
            let ex = p.0 - a.0;
            let ey = p.1 - a.1;
            return ex * ex + ey * ey <= 1e-5;
        }
        let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
        let cx = a.0 + t * dx;
        let cy = a.1 + t * dy;
        let ex = p.0 - cx;
        let ey = p.1 - cy;
        ex * ex + ey * ey <= 1e-4
    }

    /// A segment fully inside the rectangle is returned unchanged.
    #[test]
    fn fully_inside_unchanged() {
        let lo = (-2.0, -2.0);
        let hi = (3.0, 4.0);
        let a = (0.0, 0.0);
        let b = (1.5, 2.0);
        let (p0, p1) = clip_segment_rect(a, b, lo, hi).expect("segment is inside");
        assert_eq!(p0, a);
        assert_eq!(p1, b);
    }

    /// A segment crossing the rectangle is trimmed to the two boundary crossings.
    #[test]
    fn crossing_trimmed_to_boundaries() {
        // Horizontal line y = 0 crossing the box x in [-1, 1].
        let (p0, p1) =
            clip_segment_rect((-5.0, 0.0), (5.0, 0.0), (-1.0, -1.0), (1.0, 1.0)).expect("crosses");
        let (left, right) = if p0.0 <= p1.0 { (p0, p1) } else { (p1, p0) };
        assert!((left.0 - (-1.0)).abs() < 1e-6);
        assert!((right.0 - 1.0).abs() < 1e-6);
        assert!(left.1.abs() < 1e-6 && right.1.abs() < 1e-6);
    }

    /// A segment with an endpoint inside keeps that endpoint and trims the other.
    #[test]
    fn one_endpoint_inside() {
        let lo = (0.0, 0.0);
        let hi = (10.0, 10.0);
        let a = (5.0, 5.0); // inside
        let b = (20.0, 5.0); // outside to the right
        let (p0, p1) = clip_segment_rect(a, b, lo, hi).expect("partly inside");
        assert_eq!(p0, a);
        assert!((p1.0 - 10.0).abs() < 1e-6 && (p1.1 - 5.0).abs() < 1e-6);
    }

    /// A segment that misses the rectangle returns `None`.
    #[test]
    fn disjoint_returns_none() {
        assert!(clip_segment_rect((5.0, 5.0), (9.0, 9.0), (0.0, 0.0), (4.0, 4.0)).is_none());
        // Parallel to and left of the box.
        assert!(clip_segment_rect((-1.0, -5.0), (-1.0, 5.0), (0.0, 0.0), (4.0, 4.0)).is_none());
    }

    /// Degenerate point segment: inside returns itself, outside returns `None`.
    #[test]
    fn degenerate_point_segment() {
        let lo = (0.0, 0.0);
        let hi = (4.0, 4.0);
        assert_eq!(
            clip_segment_rect((2.0, 2.0), (2.0, 2.0), lo, hi),
            Some(((2.0, 2.0), (2.0, 2.0)))
        );
        assert!(clip_segment_rect((9.0, 9.0), (9.0, 9.0), lo, hi).is_none());
    }

    /// Corners may be supplied in any order.
    #[test]
    fn corner_order_irrelevant() {
        let a = (-5.0, 1.0);
        let b = (5.0, 1.0);
        let r1 = clip_segment_rect(a, b, (-2.0, -2.0), (2.0, 2.0));
        let r2 = clip_segment_rect(a, b, (2.0, 2.0), (-2.0, -2.0));
        let r3 = clip_segment_rect(a, b, (-2.0, 2.0), (2.0, -2.0));
        assert_eq!(r1, r2);
        assert_eq!(r1, r3);
    }
}
