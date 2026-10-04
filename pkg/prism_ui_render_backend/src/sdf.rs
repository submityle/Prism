//! Signed-distance-field primitives for the retained draw stream.
//!
//! The GPU backend rasterises every box through the same closed-form distance
//! functions that the headless reference backend ([`crate::raster`]) evaluates
//! on the CPU. Keeping the math in one place is what lets the two backends be
//! validated against each other pixel-for-pixel.
//!
//! All functions are pure, deterministic and use only `+ - * /`, [`f32::abs`],
//! [`f32::sqrt`], [`f32::min`] and [`f32::max`] — no transcendental functions —
//! so results are bit-stable across targets.

/// Signed distance from point `(px, py)` to an axis-aligned box centred at the
/// origin with the given half-extents and a uniform corner `radius`.
///
/// Negative inside, positive outside, zero on the boundary. The `radius` is
/// clamped to the largest value the box can hold so degenerate input never
/// produces NaN.
///
/// ```
/// use prism_ui_render_backend::sdf::sd_rounded_box;
/// // Dead centre of a 10x10 box is 5px inside each edge.
/// assert!((sd_rounded_box(0.0, 0.0, 5.0, 5.0, 0.0) + 5.0).abs() < 1e-6);
/// // A point exactly on the right edge is on the boundary.
/// assert!(sd_rounded_box(5.0, 0.0, 5.0, 5.0, 0.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn sd_rounded_box(px: f32, py: f32, half_w: f32, half_h: f32, radius: f32) -> f32 {
    let r = clamp(radius, 0.0, min(half_w, half_h));
    // Distance to the inner rectangle shrunk by the corner radius.
    let qx = abs(px) - (half_w - r);
    let qy = abs(py) - (half_h - r);
    let outside = length(max(qx, 0.0), max(qy, 0.0));
    let inside = min(max(qx, qy), 0.0);
    outside + inside - r
}

/// Signed distance from point `(px, py)` to a circle of the given `radius`
/// centred at the origin.
///
/// Negative inside, positive outside, zero on the boundary. This is the
/// canonical closed-form circle field, `length(px, py) - radius`, and is the
/// building block for dots, radio marks and circular hit halos.
///
/// ```
/// use prism_ui_render_backend::sdf::sd_circle;
/// assert!((sd_circle(0.0, 0.0, 4.0) + 4.0).abs() < 1e-6);
/// assert!(sd_circle(4.0, 0.0, 4.0).abs() < 1e-6);
/// assert!(sd_circle(6.0, 0.0, 4.0) > 0.0);
/// ```
#[must_use]
pub fn sd_circle(px: f32, py: f32, radius: f32) -> f32 {
    length(px, py) - radius
}

/// Unsigned distance from point `(px, py)` to the line segment running from
/// `(ax, ay)` to `(bx, by)`.
///
/// A segment encloses no area, so the result is always non-negative: the
/// perpendicular distance to the segment body, falling back to the nearest
/// endpoint once the projection passes an end. Subtract a half-width to turn
/// this into a signed capsule field for stroked lines, underlines and focus
/// rings. A degenerate zero-length segment collapses to the distance to
/// `(ax, ay)`.
///
/// ```
/// use prism_ui_render_backend::sdf::sd_segment;
/// // A point on the segment has zero distance.
/// assert!(sd_segment(1.0, 0.0, 0.0, 0.0, 2.0, 0.0).abs() < 1e-6);
/// // One unit above the midpoint is one unit away.
/// assert!((sd_segment(1.0, 1.0, 0.0, 0.0, 2.0, 0.0) - 1.0).abs() < 1e-6);
/// // Three units past an endpoint falls back to the endpoint distance.
/// assert!((sd_segment(5.0, 0.0, 0.0, 0.0, 2.0, 0.0) - 3.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn sd_segment(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let pax = px - ax;
    let pay = py - ay;
    let bax = bx - ax;
    let bay = by - ay;
    let denom = bax * bax + bay * bay;
    // Projection parameter of P onto the segment, clamped to the body `[0, 1]`.
    // A zero-length segment degenerates to the distance to endpoint `a`.
    let h = if denom > 0.0 {
        clamp((pax * bax + pay * bay) / denom, 0.0, 1.0)
    } else {
        0.0
    };
    length(pax - bax * h, pay - bay * h)
}

/// Signed distance from point `(px, py)` to an axis-aligned box centred at the
/// origin with the given half-extents and four independent corner radii.
///
/// This is the per-corner analogue of [`sd_rounded_box`], matching the CSS
/// `border-radius` shorthand where each corner may round by a different amount.
/// Radii are supplied in the order `r_tr`, `r_br`, `r_tl`, `r_bl` (top-right,
/// bottom-right, top-left, bottom-left in a y-up frame) and each is clamped to
/// the largest value the box can hold so degenerate input never produces NaN.
///
/// Negative inside, positive outside, zero on the boundary. When all four radii
/// are equal this reduces exactly to [`sd_rounded_box`].
///
/// ```
/// use prism_ui_render_backend::sdf::sd_rounded_box_per_corner;
/// // Dead centre of a sharp 10x10 box is 5px inside each edge.
/// let d = sd_rounded_box_per_corner(0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0);
/// assert!((d + 5.0).abs() < 1e-6);
/// // Rounding only the top-right corner leaves the bottom-left corner sharp.
/// let round_tr = sd_rounded_box_per_corner(5.0, 5.0, 5.0, 5.0, 4.0, 0.0, 0.0, 0.0);
/// let sharp_bl = sd_rounded_box_per_corner(-5.0, -5.0, 5.0, 5.0, 4.0, 0.0, 0.0, 0.0);
/// assert!(round_tr > 0.0 && sharp_bl.abs() < 1e-6);
/// ```
#[must_use]
pub fn sd_rounded_box_per_corner(
    px: f32,
    py: f32,
    half_w: f32,
    half_h: f32,
    r_tr: f32,
    r_br: f32,
    r_tl: f32,
    r_bl: f32,
) -> f32 {
    let limit = min(half_w, half_h);
    // Select the corner radius for the quadrant the point lives in.
    let r = match (px > 0.0, py > 0.0) {
        (true, true) => r_tr,
        (true, false) => r_br,
        (false, true) => r_tl,
        (false, false) => r_bl,
    };
    let r = clamp(r, 0.0, limit);
    let qx = abs(px) - (half_w - r);
    let qy = abs(py) - (half_h - r);
    let outside = length(max(qx, 0.0), max(qy, 0.0));
    let inside = min(max(qx, qy), 0.0);
    outside + inside - r
}

/// Antialiased coverage in `0.0..=1.0` for a signed distance, assuming a
/// one-pixel transition band centred on the boundary.
///
/// `dist <= -0.5` is fully covered, `dist >= 0.5` is fully outside, and the
/// half-pixel band in between ramps linearly. This matches a 1px-wide
/// `smoothstep`-free antialiasing kernel and is identical on CPU and GPU.
#[must_use]
pub fn coverage(dist: f32) -> f32 {
    clamp(0.5 - dist, 0.0, 1.0)
}

/// Coverage of a stroked border of `width` logical pixels sitting *inside* the
/// box edge, for a point at signed distance `dist` from that edge.
///
/// Returns the fraction of the pixel that lands on the ring
/// `-width <= dist <= 0`.
#[must_use]
pub fn border_coverage(dist: f32, width: f32) -> f32 {
    if width <= 0.0 {
        return 0.0;
    }
    // Outer edge coverage minus inner (fill) edge coverage.
    let outer = coverage(dist);
    let inner = coverage(dist + width);
    clamp(outer - inner, 0.0, 1.0)
}

/// Soft-shadow falloff for a point at signed distance `dist` outside a box,
/// spread over `blur` logical pixels.
///
/// Uses a cubic `smoothstep` ramp (no `exp`), so it is cheap, monotonic and
/// deterministic. `dist <= 0` is fully opaque shadow, `dist >= blur` is clear.
#[must_use]
pub fn shadow_alpha(dist: f32, blur: f32) -> f32 {
    if blur <= 0.0 {
        return if dist <= 0.0 { 1.0 } else { 0.0 };
    }
    let t = clamp(dist / blur, 0.0, 1.0);
    // 1 - smoothstep(0,1,t) == smoothstep is 3t^2 - 2t^3.
    let s = t * t * (3.0 - 2.0 * t);
    1.0 - s
}

#[inline]
fn length(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

#[inline]
fn abs(v: f32) -> f32 {
    if v < 0.0 {
        -v
    } else {
        v
    }
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
    use super::*;

    #[test]
    fn centre_is_most_negative() {
        let d = sd_rounded_box(0.0, 0.0, 10.0, 6.0, 0.0);
        assert!(
            (d + 6.0).abs() < 1e-5,
            "centre distance should be -min(hw,hh)"
        );
    }

    #[test]
    fn boundary_is_zero() {
        assert!(sd_rounded_box(10.0, 0.0, 10.0, 6.0, 0.0).abs() < 1e-5);
        assert!(sd_rounded_box(0.0, 6.0, 10.0, 6.0, 0.0).abs() < 1e-5);
    }

    #[test]
    fn outside_is_positive() {
        assert!(sd_rounded_box(12.0, 0.0, 10.0, 6.0, 0.0) > 0.0);
    }

    #[test]
    fn radius_rounds_corner_inward() {
        // The corner of the bounding box is now outside the rounded shape.
        let sharp = sd_rounded_box(10.0, 6.0, 10.0, 6.0, 0.0);
        let round = sd_rounded_box(10.0, 6.0, 10.0, 6.0, 4.0);
        assert!(round > sharp, "rounding pushes the corner outside");
    }

    #[test]
    fn radius_is_clamped() {
        // Over-large radius must not NaN; a disc of radius 5 in a 10x10 box.
        let d = sd_rounded_box(0.0, 0.0, 5.0, 5.0, 100.0);
        assert!(d.is_finite());
        assert!((d + 5.0).abs() < 1e-5);
    }

    #[test]
    fn coverage_ramps() {
        assert_eq!(coverage(-1.0), 1.0);
        assert_eq!(coverage(1.0), 0.0);
        assert!((coverage(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn border_only_on_ring() {
        // 2px border: a point 1px inside the edge is on the ring.
        assert!(border_coverage(-1.0, 2.0) > 0.5);
        // 10px inside is in the fill, not the ring.
        assert!(border_coverage(-10.0, 2.0) < 1e-6);
        assert_eq!(border_coverage(-1.0, 0.0), 0.0);
    }

    #[test]
    fn shadow_monotonic_decay() {
        assert_eq!(shadow_alpha(-1.0, 8.0), 1.0);
        assert_eq!(shadow_alpha(8.0, 8.0), 0.0);
        let a = shadow_alpha(2.0, 8.0);
        let b = shadow_alpha(4.0, 8.0);
        assert!(a > b, "shadow must fade with distance");
        assert_eq!(shadow_alpha(0.0, 0.0), 1.0);
        assert_eq!(shadow_alpha(0.5, 0.0), 0.0);
    }

    // --- Oracle helpers for the primitive distance fields -----------------

    /// `SplitMix64` — a tiny deterministic PRNG for property sampling.
    fn next_rand(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A deterministic sample in `lo..hi` drawn from `state`.
    fn rand_in(state: &mut u64, lo: f32, hi: f32) -> f32 {
        // Top 24 bits give an exact-in-`f32` mantissa fraction in `[0, 1)`.
        let bits = (next_rand(state) >> 40) as u32;
        let unit = (bits as f32) / 16_777_216.0;
        lo + (hi - lo) * unit
    }

    /// Numeric gradient magnitude of a 2-D field by central differences.
    ///
    /// A true signed-distance field satisfies the eikonal equation
    /// `|grad| == 1` everywhere its nearest-boundary point is unique, which is
    /// the whole exterior of a convex shape. This is a strong, independent
    /// oracle: it never consults the formula under test, only resamples it.
    fn grad_mag(f: impl Fn(f32, f32) -> f32, x: f32, y: f32) -> f32 {
        let h = 0.05_f32;
        let dx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
        let dy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn circle_matches_closed_form() {
        // The circle field is exactly `length - r`; check a spread of points.
        assert!((sd_circle(3.0, 4.0, 2.0) - 3.0).abs() < 1e-5);
        assert!((sd_circle(0.0, 0.0, 7.0) + 7.0).abs() < 1e-5);
        assert!(sd_circle(0.0, 7.0, 7.0).abs() < 1e-5);
    }

    #[test]
    fn circle_is_eikonal_outside() {
        // Every exterior point of a disc has |grad| == 1 (unique nearest point).
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        for _ in 0..400 {
            let x = rand_in(&mut state, -40.0, 40.0);
            let y = rand_in(&mut state, -40.0, 40.0);
            let f = |px: f32, py: f32| sd_circle(px, py, 6.0);
            if f(x, y) > 1.0 {
                assert!((grad_mag(f, x, y) - 1.0).abs() < 1e-2);
            }
        }
    }

    #[test]
    fn segment_distance_matches_brute_force() {
        // Independently recompute the distance by sampling 2000 points along
        // the segment and taking the closest; the closed form must match.
        let (ax, ay, bx, by) = (-3.0_f32, 2.0, 5.0, -4.0);
        let mut state = 0x0FED_CBA9_8765_4321_u64;
        for _ in 0..300 {
            let px = rand_in(&mut state, -12.0, 12.0);
            let py = rand_in(&mut state, -12.0, 12.0);
            let mut best = f32::MAX;
            let mut i = 0;
            while i <= 2000 {
                let t = (i as f32) / 2000.0;
                let sx = ax + (bx - ax) * t;
                let sy = ay + (by - ay) * t;
                let d = ((px - sx) * (px - sx) + (py - sy) * (py - sy)).sqrt();
                best = best.min(d);
                i += 1;
            }
            let got = sd_segment(px, py, ax, ay, bx, by);
            assert!(
                (got - best).abs() < 5e-3,
                "segment sdf {got} vs brute force {best}"
            );
        }
    }

    #[test]
    fn segment_endpoints_and_degenerate() {
        assert!(sd_segment(1.0, 2.0, 1.0, 2.0, 9.0, 9.0).abs() < 1e-6);
        assert!(sd_segment(9.0, 9.0, 1.0, 2.0, 9.0, 9.0).abs() < 1e-6);
        // Zero-length segment collapses to the point distance to `a`.
        assert!((sd_segment(3.0, 4.0, 0.0, 0.0, 0.0, 0.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn per_corner_reduces_to_uniform() {
        // With four equal radii the per-corner field must agree with the
        // trusted uniform `sd_rounded_box` at every sampled point.
        let mut state = 0xDEAD_BEEF_CAFE_F00D_u64;
        for _ in 0..600 {
            let px = rand_in(&mut state, -16.0, 16.0);
            let py = rand_in(&mut state, -16.0, 16.0);
            let r = rand_in(&mut state, 0.0, 6.0);
            let uniform = sd_rounded_box(px, py, 8.0, 6.0, r);
            let per = sd_rounded_box_per_corner(px, py, 8.0, 6.0, r, r, r, r);
            assert!(
                (uniform - per).abs() < 1e-5,
                "uniform {uniform} vs per-corner {per}"
            );
        }
    }

    #[test]
    fn per_corner_reduces_to_circle() {
        // A square box whose half-extent equals the radius is a disc.
        let mut state = 0x5151_A2A2_B3B3_C4C4_u64;
        for _ in 0..200 {
            let px = rand_in(&mut state, -10.0, 10.0);
            let py = rand_in(&mut state, -10.0, 10.0);
            let per = sd_rounded_box_per_corner(px, py, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0);
            let circ = sd_circle(px, py, 5.0);
            // Interior of a square-vs-disc differs; only compare outside.
            if circ > 0.5 {
                assert!((per - circ).abs() < 1e-4, "box {per} vs circle {circ}");
            }
        }
    }

    #[test]
    fn per_corner_is_eikonal_outside() {
        // The box is convex, so every exterior point has |grad| == 1 regardless
        // of which quadrant's radius applies.
        let mut state = 0xA5A5_1234_9999_0001_u64;
        for _ in 0..600 {
            let x = rand_in(&mut state, -24.0, 24.0);
            let y = rand_in(&mut state, -24.0, 24.0);
            let f = |px: f32, py: f32| {
                sd_rounded_box_per_corner(px, py, 8.0, 6.0, 4.0, 1.0, 3.0, 2.0)
            };
            if f(x, y) > 1.0 {
                assert!(
                    (grad_mag(f, x, y) - 1.0).abs() < 2e-2,
                    "eikonal violated at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn per_corner_is_quadrant_local() {
        // Changing only the top-right radius must not move a bottom-left point.
        let base = sd_rounded_box_per_corner(-5.0, -4.0, 8.0, 6.0, 0.0, 0.0, 0.0, 3.0);
        let changed = sd_rounded_box_per_corner(-5.0, -4.0, 8.0, 6.0, 5.0, 0.0, 0.0, 3.0);
        assert!((base - changed).abs() < 1e-6, "top-right radius leaked");
        // And rounding the owning (bottom-left) corner does change it.
        let rounded = sd_rounded_box_per_corner(-8.0, -6.0, 8.0, 6.0, 0.0, 0.0, 0.0, 4.0);
        let sharp = sd_rounded_box_per_corner(-8.0, -6.0, 8.0, 6.0, 0.0, 0.0, 0.0, 0.0);
        assert!(rounded > sharp, "bottom-left rounding must push the corner out");
    }
}
