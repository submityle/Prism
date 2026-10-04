//! Signed-distance field for arbitrary simple polygons.
//!
//! The box, circle and segment primitives in [`crate::sdf`] cover the common
//! rounded-rectangle UI surface, but chrome such as dropdown carets, play
//! triangles, chevrons and polygonal badges needs a general polygon field.
//! [`sd_polygon`] evaluates the exact signed distance to any simple
//! (non-self-intersecting) polygon, and [`sd_triangle`] is a thin convenience
//! for the overwhelmingly common three-vertex case.
//!
//! The result is negative inside, positive outside and zero on the boundary,
//! independent of the winding order. Only `+ - * /`, [`f32::sqrt`],
//! [`f32::min`] and [`f32::max`] are used — no transcendental functions — so
//! the field composes with [`crate::sdf::coverage`] for the same anti-aliased
//! edges every other primitive gets, and stays bit-stable across targets.

/// Signed distance from `(px, py)` to the simple polygon described by `verts`
/// (a closed loop; the final vertex is implicitly joined back to the first).
///
/// Negative inside, positive outside, zero on the boundary. The sign is derived
/// from a crossing-number winding test, so the orientation of `verts` does not
/// matter. Fewer than three vertices cannot bound an area, so the unsigned
/// distance to the degenerate point or segment is returned instead; an empty
/// slice yields [`f32::INFINITY`].
///
/// ```
/// use prism_ui_render_backend::polygon::sd_polygon;
/// let square = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
/// // Dead centre is 5px inside every edge.
/// assert!((sd_polygon(0.0, 0.0, &square) + 5.0).abs() < 1e-5);
/// // A point on the right edge is on the boundary.
/// assert!(sd_polygon(5.0, 0.0, &square).abs() < 1e-5);
/// ```
#[must_use]
pub fn sd_polygon(px: f32, py: f32, verts: &[(f32, f32)]) -> f32 {
    match verts.len() {
        0 => return f32::INFINITY,
        1 => return length(px - verts[0].0, py - verts[0].1),
        2 => return point_segment_distance(px, py, verts[0], verts[1]),
        _ => {}
    }

    let n = verts.len();
    // Seed the running squared distance with the first vertex so the loop only
    // ever needs `min`.
    let mut d2 = {
        let wx = px - verts[0].0;
        let wy = py - verts[0].1;
        wx * wx + wy * wy
    };
    // Crossing-number parity: flip the sign once per edge the downward ray from
    // the point crosses. An odd number of crossings means the point is inside.
    let mut sign = 1.0_f32;
    let mut j = n - 1;
    for (i, &(vix, viy)) in verts.iter().enumerate() {
        let (vjx, vjy) = verts[j];
        // Edge from `v[i]` towards the previous vertex `v[j]`, and the vector
        // from `v[i]` to the sample point.
        let ex = vjx - vix;
        let ey = vjy - viy;
        let wx = px - vix;
        let wy = py - viy;
        // Closest point on the (clamped) edge segment to the sample.
        let denom = ex * ex + ey * ey;
        let t = if denom > 0.0 {
            clamp((wx * ex + wy * ey) / denom, 0.0, 1.0)
        } else {
            0.0
        };
        let bx = wx - ex * t;
        let by = wy - ey * t;
        d2 = min(d2, bx * bx + by * by);

        // Winding test for this edge (half-open in `y` to avoid double-counting
        // shared vertices).
        let c0 = py >= viy;
        let c1 = py < vjy;
        let c2 = ex * wy > ey * wx;
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            sign = -sign;
        }
        j = i;
    }

    sign * d2.sqrt()
}

/// Signed distance from `(px, py)` to the triangle `a`, `b`, `c`.
///
/// A convenience wrapper over [`sd_polygon`] for the common three-vertex case
/// (carets, play buttons, chevrons). Negative inside, positive outside, zero on
/// the boundary, independent of vertex winding.
///
/// ```
/// use prism_ui_render_backend::polygon::sd_triangle;
/// // Right triangle with legs on the axes; a point near a leg is inside.
/// let d = sd_triangle(1.0, 1.0, (0.0, 0.0), (6.0, 0.0), (0.0, 6.0));
/// assert!(d < 0.0);
/// ```
#[must_use]
pub fn sd_triangle(px: f32, py: f32, a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    sd_polygon(px, py, &[a, b, c])
}

#[inline]
fn point_segment_distance(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let ex = b.0 - a.0;
    let ey = b.1 - a.1;
    let wx = px - a.0;
    let wy = py - a.1;
    let denom = ex * ex + ey * ey;
    let t = if denom > 0.0 {
        clamp((wx * ex + wy * ey) / denom, 0.0, 1.0)
    } else {
        0.0
    };
    length(wx - ex * t, wy - ey * t)
}

#[inline]
fn length(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

#[inline]
fn min(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
fn max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    max(lo, min(hi, v))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use crate::sdf::sd_rounded_box;

    /// `SplitMix64` — a tiny deterministic PRNG for property sampling.
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

    /// Brute-force unsigned distance to the polygon boundary: the minimum over
    /// every edge of the exact point-to-segment distance. Never consults
    /// `sd_polygon`, so it is a fully independent oracle for the magnitude.
    fn brute_force_edge_distance(px: f32, py: f32, verts: &[(f32, f32)]) -> f32 {
        let n = verts.len();
        let mut best = f32::INFINITY;
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            best = best.min(point_segment_distance(px, py, a, b));
        }
        best
    }

    /// Independent inside test by the even-odd crossing number of a ray cast in
    /// `+x`. Shares no code with `sd_polygon`'s own sign logic.
    fn inside_crossing(px: f32, py: f32, verts: &[(f32, f32)]) -> bool {
        let n = verts.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = verts[i];
            let (xj, yj) = verts[j];
            let straddles = (yi > py) != (yj > py);
            if straddles {
                let x_cross = (xj - xi) * (py - yi) / (yj - yi) + xi;
                if px < x_cross {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }

    fn grad_mag(f: impl Fn(f32, f32) -> f32, x: f32, y: f32) -> f32 {
        let h = 0.05_f32;
        let dx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
        let dy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn square_matches_rounded_box() {
        // A square polygon's field must equal the dedicated rounded-box field
        // with zero corner radius — two completely independent implementations.
        let square = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
        let mut state = 0xDEAD_BEEF_CAFE_1234_u64;
        for _ in 0..1000 {
            let x = rand_in(&mut state, -12.0, 12.0);
            let y = rand_in(&mut state, -12.0, 12.0);
            let poly = sd_polygon(x, y, &square);
            let boxed = sd_rounded_box(x, y, 5.0, 5.0, 0.0);
            assert!(
                (poly - boxed).abs() < 1e-4,
                "square {poly} vs box {boxed} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn magnitude_matches_brute_force_edges() {
        // |sd_polygon| equals the brute-force nearest-edge distance everywhere.
        let mut state = 0x0123_4567_89AB_CDEF_u64;
        for _ in 0..600 {
            let tri = [
                (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0)),
                (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0)),
                (rand_in(&mut state, -10.0, 10.0), rand_in(&mut state, -10.0, 10.0)),
            ];
            for _ in 0..5 {
                let x = rand_in(&mut state, -14.0, 14.0);
                let y = rand_in(&mut state, -14.0, 14.0);
                let got = sd_polygon(x, y, &tri).abs();
                let want = brute_force_edge_distance(x, y, &tri);
                assert!(
                    (got - want).abs() < 1e-3,
                    "|sd| {got} vs brute {want} at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn sign_matches_crossing_number() {
        // The reported sign must agree with an independent inside test, except
        // for points right on the boundary where the sign is ambiguous.
        let pentagon = [
            (0.0, 8.0),
            (7.6, 2.5),
            (4.7, -6.5),
            (-4.7, -6.5),
            (-7.6, 2.5),
        ];
        let mut state = 0xABCD_1234_5678_9EF0_u64;
        for _ in 0..3000 {
            let x = rand_in(&mut state, -12.0, 12.0);
            let y = rand_in(&mut state, -12.0, 12.0);
            let d = sd_polygon(x, y, &pentagon);
            if d.abs() < 0.1 {
                continue;
            }
            let inside = inside_crossing(x, y, &pentagon);
            assert_eq!(d < 0.0, inside, "sign {d} disagrees at ({x}, {y})");
        }
    }

    #[test]
    fn vertices_and_edge_midpoints_are_zero() {
        let tri = [(0.0, 0.0), (6.0, 0.0), (3.0, 7.0)];
        for i in 0..3 {
            let (vx, vy) = tri[i];
            assert!(sd_polygon(vx, vy, &tri).abs() < 1e-5, "vertex {i}");
            let (nx, ny) = tri[(i + 1) % 3];
            let mx = (vx + nx) * 0.5;
            let my = (vy + ny) * 0.5;
            assert!(sd_polygon(mx, my, &tri).abs() < 1e-5, "midpoint {i}");
        }
    }

    #[test]
    fn winding_order_is_irrelevant() {
        let ccw = [(-4.0, -3.0), (5.0, -3.0), (0.0, 6.0)];
        let cw = [(0.0, 6.0), (5.0, -3.0), (-4.0, -3.0)];
        let mut state = 0x5555_AAAA_3333_CCCC_u64;
        for _ in 0..500 {
            let x = rand_in(&mut state, -10.0, 10.0);
            let y = rand_in(&mut state, -10.0, 10.0);
            assert!((sd_polygon(x, y, &ccw) - sd_polygon(x, y, &cw)).abs() < 1e-6);
        }
    }

    #[test]
    fn eikonal_outside() {
        // Every exterior point has a unique nearest boundary point, so a true
        // signed-distance field satisfies |grad| == 1 there.
        let quad = [(-6.0, -4.0), (7.0, -5.0), (5.0, 6.0), (-4.0, 5.0)];
        let f = |px: f32, py: f32| sd_polygon(px, py, &quad);
        let mut state = 0x9E37_79B1_1111_2222_u64;
        for _ in 0..800 {
            let x = rand_in(&mut state, -20.0, 20.0);
            let y = rand_in(&mut state, -20.0, 20.0);
            if f(x, y) > 1.0 {
                assert!((grad_mag(f, x, y) - 1.0).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn triangle_wrapper_matches_polygon() {
        let a = (0.0, 0.0);
        let b = (6.0, 1.0);
        let c = (2.0, 7.0);
        let mut state = 0x2468_ACE0_1357_9BDF_u64;
        for _ in 0..500 {
            let x = rand_in(&mut state, -10.0, 12.0);
            let y = rand_in(&mut state, -10.0, 12.0);
            let w = sd_triangle(x, y, a, b, c);
            let p = sd_polygon(x, y, &[a, b, c]);
            assert_eq!(w, p);
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert_eq!(sd_polygon(1.0, 1.0, &[]), f32::INFINITY);
        assert!((sd_polygon(3.0, 4.0, &[(0.0, 0.0)]) - 5.0).abs() < 1e-5);
        assert!((sd_polygon(0.0, 5.0, &[(-5.0, 0.0), (5.0, 0.0)]) - 5.0).abs() < 1e-5);
        // Duplicate consecutive vertices must not divide by zero.
        let dup = [(0.0, 0.0), (0.0, 0.0), (6.0, 0.0), (3.0, 7.0)];
        assert!(sd_polygon(3.0, 2.0, &dup).is_finite());
    }
}
