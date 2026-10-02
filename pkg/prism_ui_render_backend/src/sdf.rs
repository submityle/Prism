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
}
