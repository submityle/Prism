//! History reprojection, bicubic resampling, and disocclusion detection.
//!
//! Between two frames the camera and objects move, so the surface visible at an
//! output pixel this frame was at a *different* screen location last frame. A
//! temporal reconstructor must follow that motion to fetch the right history
//! sample, resample it without softening the image, and detect the cases where
//! no valid history exists (a surface that was hidden last frame, or motion
//! that leaves the screen) so it can fall back to the current color instead of
//! dragging a smear behind the moving edge.
//!
//! This module owns those three jobs:
//!
//! 1. **Reprojection** — applying a per-pixel motion vector to find the
//!    previous-frame location, and testing whether it still lies on screen.
//! 2. **Resampling** — a Catmull-Rom bicubic fetch of the history color, which
//!    preserves sharpness far better than bilinear (the standard `FSR` / `TAAU`
//!    history filter). Edge taps use clamp addressing.
//! 3. **Disocclusion** — a relative depth test that rejects history whose
//!    stored depth disagrees with the current surface, i.e. the current pixel
//!    was occluded last frame.
//!
//! The motion-vector convention is **current-to-previous**: a vector points
//! from the current pixel to where that surface was last frame, so the previous
//! location is `current + motion`. All arithmetic is `+`, `-`, `*`, `/`,
//! `min`/`max`, so a `GPU` kernel reproduces it exactly.

/// Applies a current-to-previous motion vector to a display-space pixel.
///
/// Returns the fractional previous-frame location of the surface currently at
/// `(x, y)`. The caller resamples the history buffer at this location (see
/// [`sample_catmull_rom`]) after confirming it is [`on_screen`].
#[must_use]
pub fn reproject_pixel(x: f32, y: f32, motion: [f32; 2]) -> [f32; 2] {
    [x + motion[0], y + motion[1]]
}

/// Tests whether a fractional pixel location lies within `[0, width) x
/// [0, height)`.
///
/// A reprojected location outside the frame has no history to sample, which is
/// one of the disocclusion cases; callers reset accumulation when this returns
/// `false`. Zero-size frames are never on screen.
#[must_use]
pub fn on_screen(pos: [f32; 2], width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    pos[0] >= 0.0 && pos[1] >= 0.0 && pos[0] < width as f32 && pos[1] < height as f32
}

/// The four separable Catmull-Rom weights for a fractional offset `t` in
/// `[0, 1]`.
///
/// The taps sit at integer offsets `-1, 0, +1, +2` relative to the floor of the
/// sample position. These are the `a = -0.5` cubic weights in their compact
/// polynomial form; they sum to exactly `1` for every `t`, so the filter
/// preserves a constant signal, and are interpolating at the endpoints
/// (`t = 0` returns tap `0`, `t = 1` returns tap `+1`). The outer weights are
/// negative, which is what sharpens the result relative to bilinear.
#[must_use]
pub fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Resamples a history color at a fractional location with a `4 x 4`
/// Catmull-Rom bicubic kernel.
///
/// `fetch(x, y)` must return the stored history color at integer pixel
/// `(x, y)`; the caller is responsible for whatever buffer it reads. Tap
/// coordinates are clamped to `[0, width - 1] x [0, height - 1]` (clamp
/// addressing) so edge pixels stay well-defined. The separable kernel evaluates
/// four horizontal taps per row weighted by [`catmull_rom_weights`], then
/// combines the four rows with the vertical weights.
///
/// Catmull-Rom's negative lobes can overshoot into small negative values on a
/// high-contrast edge (ringing); each output channel is clamped to `>= 0` so
/// the resampled `HDR` color stays physically valid. A zero-size frame returns
/// black.
#[must_use]
pub fn sample_catmull_rom<F>(fetch: F, pos: [f32; 2], width: u32, height: u32) -> [f32; 3]
where
    F: Fn(i32, i32) -> [f32; 3],
{
    if width == 0 || height == 0 {
        return [0.0; 3];
    }
    let max_x = width as i32 - 1;
    let max_y = height as i32 - 1;
    // Floor via truncation toward negative infinity: `floor` is allowed (not a
    // transcendental) and gives the integer cell the fractional offset is
    // measured from.
    let fx = pos[0].floor();
    let fy = pos[1].floor();
    let tx = pos[0] - fx;
    let ty = pos[1] - fy;
    let wx = catmull_rom_weights(tx);
    let wy = catmull_rom_weights(ty);
    let base_x = fx as i32 - 1;
    let base_y = fy as i32 - 1;

    let mut acc = [0.0f32; 3];
    for (j, &wyj) in wy.iter().enumerate() {
        let sy = (base_y + j as i32).clamp(0, max_y);
        let mut row = [0.0f32; 3];
        for (i, &wxi) in wx.iter().enumerate() {
            let sx = (base_x + i as i32).clamp(0, max_x);
            let c = fetch(sx, sy);
            row[0] += c[0] * wxi;
            row[1] += c[1] * wxi;
            row[2] += c[2] * wxi;
        }
        acc[0] += row[0] * wyj;
        acc[1] += row[1] * wyj;
        acc[2] += row[2] * wyj;
    }
    [acc[0].max(0.0), acc[1].max(0.0), acc[2].max(0.0)]
}

/// Relative-depth disocclusion test between the current surface and the sampled
/// history depth.
///
/// Both depths are assumed linear (view-space distance or linear `0..1`); the
/// test rejects history when the relative difference
/// `|current - history| / max(current, history, eps)` exceeds `tolerance`
/// (a typical value is a few percent). This catches the case where the current
/// pixel shows a near surface but the history at the reprojected location
/// stored a far one — i.e. the near surface was occluded last frame, so its
/// history is invalid. A non-positive or non-finite `tolerance` disables the
/// test (never disoccludes on depth).
#[must_use]
pub fn depth_disoccluded(current_depth: f32, history_depth: f32, tolerance: f32) -> bool {
    if tolerance.is_nan() || tolerance <= 0.0 {
        return false;
    }
    let denom = current_depth.abs().max(history_depth.abs()).max(1e-6);
    let relative = (current_depth - history_depth).abs() / denom;
    relative > tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error tolerance for the resampling and weight checks.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    #[test]
    fn reprojection_follows_motion() {
        let p = reproject_pixel(100.0, 50.0, [-4.0, 2.5]);
        assert!(approx(p[0], 96.0) && approx(p[1], 52.5));
    }

    #[test]
    fn on_screen_bounds_are_half_open() {
        assert!(on_screen([0.0, 0.0], 10, 10));
        assert!(on_screen([9.99, 9.99], 10, 10));
        assert!(!on_screen([-0.01, 5.0], 10, 10));
        assert!(!on_screen([10.0, 5.0], 10, 10));
        assert!(!on_screen([5.0, 5.0], 0, 10));
    }

    #[test]
    fn catmull_rom_weights_sum_to_one() {
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let w = catmull_rom_weights(t);
            assert!(approx(w[0] + w[1] + w[2] + w[3], 1.0), "t={t}");
        }
    }

    #[test]
    fn catmull_rom_weights_interpolate_endpoints() {
        let w0 = catmull_rom_weights(0.0);
        assert!(
            approx(w0[0], 0.0) && approx(w0[1], 1.0) && approx(w0[2], 0.0) && approx(w0[3], 0.0)
        );
        let w1 = catmull_rom_weights(1.0);
        assert!(
            approx(w1[0], 0.0) && approx(w1[1], 0.0) && approx(w1[2], 1.0) && approx(w1[3], 0.0)
        );
    }

    #[test]
    fn resample_is_exact_at_integer_centers() {
        // A ramp in x: fetch returns the x coordinate as the red channel.
        let fetch = |x: i32, _y: i32| [x as f32, 0.0, 0.0];
        let c = sample_catmull_rom(fetch, [5.0, 3.0], 16, 16);
        assert!(approx(c[0], 5.0), "{}", c[0]);
    }

    #[test]
    fn resample_reproduces_a_linear_ramp() {
        // Catmull-Rom reconstructs linear signals exactly (it is a cubic that
        // contains the linear polynomials), so a midpoint sample of a ramp is
        // the true linear value.
        let fetch = |x: i32, _y: i32| [2.0 * x as f32 + 1.0, 0.0, 0.0];
        let c = sample_catmull_rom(fetch, [5.5, 8.0], 32, 32);
        assert!(approx(c[0], 2.0 * 5.5 + 1.0), "{}", c[0]);
    }

    #[test]
    fn resample_preserves_a_constant() {
        let fetch = |_x: i32, _y: i32| [0.7, 0.3, 0.9];
        let c = sample_catmull_rom(fetch, [4.25, 9.75], 16, 16);
        assert!(approx(c[0], 0.7) && approx(c[1], 0.3) && approx(c[2], 0.9));
    }

    #[test]
    fn resample_clamps_at_edges() {
        // Near the corner the clamped taps all read the same edge value.
        let fetch = |_x: i32, _y: i32| [1.0, 1.0, 1.0];
        let c = sample_catmull_rom(fetch, [0.0, 0.0], 8, 8);
        assert!(approx(c[0], 1.0));
    }

    #[test]
    fn resample_output_is_non_negative() {
        // A sharp 0/large edge can ring negative; the clamp keeps it valid.
        let fetch = |x: i32, _y: i32| {
            if x >= 4 {
                [100.0, 0.0, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        };
        let c = sample_catmull_rom(fetch, [3.5, 2.0], 16, 16);
        assert!(c[0] >= 0.0, "ringing went negative: {}", c[0]);
    }

    #[test]
    fn depth_test_rejects_large_relative_jump() {
        // Near surface now (0.1) vs far history (0.9): clearly disoccluded.
        assert!(depth_disoccluded(0.1, 0.9, 0.05));
        // Matching depths within tolerance: keep history.
        assert!(!depth_disoccluded(0.5, 0.505, 0.05));
    }

    #[test]
    fn depth_test_disabled_by_non_positive_tolerance() {
        assert!(!depth_disoccluded(0.1, 0.9, 0.0));
        assert!(!depth_disoccluded(0.1, 0.9, f32::NAN));
    }
}
