//! Screen-space edge detection for particle stylization (design §16-§21).
//!
//! A stylized particle pass often wants to know *where the silhouettes are* on
//! screen: the outlines of a smoke plume, the crease between overlapping
//! sprites, the boundary where a particle's depth jumps away from the scene
//! behind it. Every `NPR`/outline stack (Unreal's post-process outline,
//! `Frostbite`'s edge pass, comic-shaded VFX) derives that from the same three
//! screen-space signals — a `luminance`/color break, a depth discontinuity, or
//! a normal-direction break — and folds them into a single edge strength that
//! drives an outline stroke or edge-directed anti-aliasing (`AA`) blend.
//!
//! This module owns the `CPU`-verifiable maths of that contract. It supplies
//! the two canonical `Sobel` 3x3 gradient kernels, an alternative `Roberts`
//! cross for the cheap 2x2 case, per-channel edge responses for the
//! `luminance`, depth, and normal buffers, and a [`EdgeParams`] policy whose
//! [`EdgeParams::edge_mask`] thresholds a raw magnitude into a soft `0..=1`
//! edge mask through a `smoothstep` knee.
//!
//! # Strict scope
//!
//! This file is *only* edge detection: `Sobel`/`Roberts` gradient magnitude,
//! depth and normal difference responses, and the `smoothstep` mask. It does
//! **not** blur, downsample, or resolve anything (those are separate passes),
//! and it does **not** reconstruct a normal from depth — [`super::normal_reconstruct`]
//! already owns the depth->normal derivation, and this module only *consumes*
//! the depth, normal, and `luminance` samples handed to it as function
//! arguments.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions
//! (`sin`/`cos`/`exp`/`ln`/`powf`). Gradient magnitude needs a single `sqrt`,
//! the `smoothstep` is the classic cubic `t * t * (3 - 2 t)`, and everything
//! else is `+ - * /` guarded against a zero denominator, so a future `GPU`
//! kernel reproduces the `CPU` result bit for bit.

/// `Rec. 709` `luminance` weight of the red channel.
const LUMA_R: f32 = 0.2126;

/// `Rec. 709` `luminance` weight of the green channel.
const LUMA_G: f32 = 0.7152;

/// `Rec. 709` `luminance` weight of the blue channel.
const LUMA_B: f32 = 0.0722;

/// `smoothstep` intervals (and generic denominators) narrower than this are
/// treated as a hard step so evaluation never divides by zero or emits `NaN`.
const MIN_SPAN: f32 = 1e-6;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Horizontal `Sobel` kernel `Gx` in row-major order. It differences the left
/// column against the right column with a `[1, 2, 1]` smoothing along the
/// vertical axis, so it responds to intensity changes *across* the horizontal
/// direction.
pub const SOBEL_GX: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

/// Vertical `Sobel` kernel `Gy` in row-major order. It is the transpose of
/// [`SOBEL_GX`]: it differences the top row against the bottom row with a
/// `[1, 2, 1]` smoothing along the horizontal axis, so it responds to intensity
/// changes *across* the vertical direction.
pub const SOBEL_GY: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

/// Clamps `x` into the closed unit interval `0..=1`.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0` at or below `edge0`, `1` at or above `edge1`, and the cubic
/// `t * t * (3 - 2 t)` in between, where `t` is the clamped normalized position.
/// A degenerate interval narrower than [`MIN_SPAN`] collapses to a hard step at
/// `edge0` so the division stays defined.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span <= MIN_SPAN {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = clamp01((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// The dot product of two `RGB` triples treated as 3-vectors, written as an
/// explicit hand-rolled sum.
#[must_use]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Convolves a row-major 3x3 kernel against a row-major 3x3 window, returning
/// the scalar gradient response `sum(kernel[i][j] * window[i][j])`.
#[must_use]
fn convolve3x3(kernel: &[[f32; 3]; 3], window: &[f32; 9]) -> f32 {
    kernel
        .iter()
        .flatten()
        .zip(window.iter())
        .map(|(k, v)| k * v)
        .sum()
}

/// The perceptual `luminance` of a linear `RGB` triple,
/// `dot(rgb, [0.2126, 0.7152, 0.0722])`, written as an explicit hand-rolled dot
/// product of the `Rec. 709` weights. This is the scalar channel the
/// `luminance`-based edge response differences.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA_R + rgb[1] * LUMA_G + rgb[2] * LUMA_B
}

/// The `Sobel` gradient magnitude of a row-major 3x3 scalar window.
///
/// The window is convolved with both [`SOBEL_GX`] and [`SOBEL_GY`], and the
/// result is the Euclidean magnitude `sqrt(gx^2 + gy^2)` of the gradient
/// vector. A uniform window returns exactly `0`, and the magnitude is always
/// non-negative.
#[must_use]
pub fn sobel_magnitude(win3x3: &[f32; 9]) -> f32 {
    let gx = convolve3x3(&SOBEL_GX, win3x3);
    let gy = convolve3x3(&SOBEL_GY, win3x3);
    (gx * gx + gy * gy).sqrt()
}

/// The `Roberts` cross gradient magnitude of a 2x2 scalar window whose samples
/// are laid out as
///
/// ```text
/// a b
/// c d
/// ```
///
/// The `Roberts` operator differences the two diagonals: `gx = a - d` and
/// `gy = b - c`, and the response is the Euclidean magnitude
/// `sqrt(gx^2 + gy^2)`. It is the cheap 2x2 alternative to the 3x3 `Sobel`
/// magnitude and is likewise non-negative and zero on a uniform window.
#[must_use]
pub fn roberts_magnitude(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let gx = a - d;
    let gy = b - c;
    (gx * gx + gy * gy).sqrt()
}

/// The depth-buffer edge response of a row-major 3x3 depth window: the `Sobel`
/// gradient magnitude of the depth samples.
///
/// A depth discontinuity (a silhouette against the scene behind it) produces a
/// large response, while a flat or smoothly sloped depth region produces a
/// small one. The depth samples are consumed as given; this module never
/// reconstructs them.
#[must_use]
pub fn depth_edge(depth3x3: &[f32; 9]) -> f32 {
    sobel_magnitude(depth3x3)
}

/// The normal-buffer edge response between a center normal and its neighbors,
/// `1 - dot(n_center, n_neighbor)` maximized over the neighbors.
///
/// For unit normals the per-neighbor term lives in `0..=2`: it is `0` when the
/// neighbor points the same way (a flat surface), `1` at a right-angle crease,
/// and `2` for opposed normals. Taking the maximum picks the sharpest crease in
/// the neighborhood. The result is clamped to be non-negative to absorb any
/// floating-point overshoot past a dot of `1`, and an empty neighbor slice
/// yields `0`.
#[must_use]
pub fn normal_edge(n_center: [f32; 3], n_neighbors: &[[f32; 3]]) -> f32 {
    n_neighbors
        .iter()
        .map(|n| (1.0 - dot3(n_center, *n)).max(0.0))
        .fold(0.0, f32::max)
}

/// Thresholding policy that turns a raw edge magnitude into a soft `0..=1` edge
/// mask (design §16-§21).
///
/// `scale` is a gain applied to the incoming magnitude before thresholding (it
/// normalizes the differing magnitude ranges of the `luminance`, depth, and
/// normal responses onto a common footing); `threshold` is the magnitude at the
/// center of the transition; and `knee` is the half-width of the `smoothstep`
/// band around the threshold (a zero `knee` is a hard cutoff). The mask is `0`
/// on flat interiors and `1` on strong edges, which drives an outline stroke or
/// an edge-directed anti-aliasing (`AA`) blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeParams {
    /// Magnitude at the center of the `smoothstep` transition band.
    pub threshold: f32,
    /// Half-width of the soft `knee` band around the threshold.
    pub knee: f32,
    /// Gain applied to the raw magnitude before thresholding.
    pub scale: f32,
}

impl EdgeParams {
    /// Builds a parameter set from its raw fields, clamping `knee` and `scale`
    /// to be non-negative so the `smoothstep` band and the gain stay well
    /// defined.
    #[must_use]
    pub fn new(threshold: f32, knee: f32, scale: f32) -> Self {
        Self {
            threshold,
            knee: knee.max(0.0),
            scale: scale.max(0.0),
        }
    }

    /// Maps a raw edge magnitude to a soft `0..=1` edge mask.
    ///
    /// The magnitude is first scaled by `scale`, then run through a
    /// `smoothstep` over the band `[threshold - knee, threshold + knee]`. The
    /// result is `0` below the band, `1` above it, exactly `0.5` at the band
    /// center, monotonically non-decreasing in `mag`, and always within
    /// `0..=1`.
    #[must_use]
    pub fn edge_mask(&self, mag: f32) -> f32 {
        let scaled = mag * self.scale;
        let lo = self.threshold - self.knee;
        let hi = self.threshold + self.knee;
        smoothstep(lo, hi, scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn luminance_white_is_one() {
        assert!(approx(luminance([1.0, 1.0, 1.0]), 1.0));
    }

    #[test]
    fn luminance_black_is_zero() {
        assert!(approx(luminance([0.0, 0.0, 0.0]), 0.0));
    }

    #[test]
    fn luminance_weights_favor_green() {
        let r = luminance([1.0, 0.0, 0.0]);
        let g = luminance([0.0, 1.0, 0.0]);
        let b = luminance([0.0, 0.0, 1.0]);
        assert!(g > r);
        assert!(r > b);
        assert!(approx(g, LUMA_G));
    }

    #[test]
    fn sobel_kernels_are_transposes() {
        let gx = SOBEL_GX;
        let gy = SOBEL_GY;
        for (i, row) in gx.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(approx(v, gy[j][i]));
            }
        }
    }

    #[test]
    fn sobel_uniform_window_has_no_edge() {
        let win = [0.5_f32; 9];
        assert!(approx(sobel_magnitude(&win), 0.0));
    }

    #[test]
    fn sobel_constant_offset_has_no_edge() {
        let win = [7.25_f32; 9];
        assert!(approx(sobel_magnitude(&win), 0.0));
    }

    #[test]
    fn sobel_horizontal_step_is_gx() {
        // Left half 0, right column 1: a purely horizontal intensity break.
        let win = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert!(approx(sobel_magnitude(&win), 4.0));
    }

    #[test]
    fn sobel_vertical_step_is_gy() {
        // Bottom row 1: a purely vertical intensity break.
        let win = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert!(approx(sobel_magnitude(&win), 4.0));
    }

    #[test]
    fn sobel_magnitude_is_non_negative() {
        let win = [-3.0, 2.0, -1.0, 5.0, 0.0, -4.0, 1.0, -2.0, 6.0];
        assert!(sobel_magnitude(&win) >= 0.0);
    }

    #[test]
    fn roberts_uniform_window_has_no_edge() {
        assert!(approx(roberts_magnitude(0.3, 0.3, 0.3, 0.3), 0.0));
    }

    #[test]
    fn roberts_main_diagonal_step() {
        // a - d dominates: gx = 1, gy = 0.
        assert!(approx(roberts_magnitude(1.0, 0.0, 0.0, 0.0), 1.0));
    }

    #[test]
    fn roberts_anti_diagonal_step() {
        // b - c dominates: gx = 0, gy = 1.
        assert!(approx(roberts_magnitude(0.0, 1.0, 0.0, 0.0), 1.0));
    }

    #[test]
    fn roberts_both_diagonals() {
        // gx = 1, gy = 1 -> sqrt(2).
        let expected = 2.0_f32.sqrt();
        assert!(approx(roberts_magnitude(1.0, 1.0, 0.0, 0.0), expected));
    }

    #[test]
    fn roberts_magnitude_is_non_negative() {
        assert!(roberts_magnitude(-2.0, 3.0, 5.0, -1.0) >= 0.0);
    }

    #[test]
    fn depth_edge_matches_sobel_magnitude() {
        let win = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9];
        assert!(approx(depth_edge(&win), sobel_magnitude(&win)));
    }

    #[test]
    fn depth_edge_flat_depth_is_zero() {
        let win = [0.42_f32; 9];
        assert!(approx(depth_edge(&win), 0.0));
    }

    #[test]
    fn depth_edge_discontinuity_is_positive() {
        let win = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        assert!(depth_edge(&win) > 0.0);
    }

    #[test]
    fn normal_edge_identical_normals_is_zero() {
        let up = [0.0, 1.0, 0.0];
        assert!(approx(normal_edge(up, &[up, up, up]), 0.0));
    }

    #[test]
    fn normal_edge_opposed_normals_is_two() {
        let up = [0.0, 1.0, 0.0];
        let down = [0.0, -1.0, 0.0];
        assert!(approx(normal_edge(up, &[down]), 2.0));
    }

    #[test]
    fn normal_edge_perpendicular_is_one() {
        let up = [0.0, 1.0, 0.0];
        let right = [1.0, 0.0, 0.0];
        assert!(approx(normal_edge(up, &[right]), 1.0));
    }

    #[test]
    fn normal_edge_takes_sharpest_neighbor() {
        let up = [0.0, 1.0, 0.0];
        let right = [1.0, 0.0, 0.0];
        let down = [0.0, -1.0, 0.0];
        // Neighbors of increasing divergence; the max (opposed) wins.
        assert!(approx(normal_edge(up, &[up, right, down]), 2.0));
    }

    #[test]
    fn normal_edge_empty_neighbors_is_zero() {
        let up = [0.0, 1.0, 0.0];
        assert!(approx(normal_edge(up, &[]), 0.0));
    }

    #[test]
    fn edge_mask_below_band_is_zero() {
        let params = EdgeParams::new(0.5, 0.1, 1.0);
        assert!(approx(params.edge_mask(0.0), 0.0));
    }

    #[test]
    fn edge_mask_above_band_is_one() {
        let params = EdgeParams::new(0.5, 0.1, 1.0);
        assert!(approx(params.edge_mask(10.0), 1.0));
    }

    #[test]
    fn edge_mask_center_is_half() {
        let params = EdgeParams::new(0.5, 0.5, 1.0);
        assert!(approx(params.edge_mask(0.5), 0.5));
    }

    #[test]
    fn edge_mask_is_monotonic_non_decreasing() {
        let params = EdgeParams::new(0.5, 0.4, 1.0);
        let mut prev = -1.0_f32;
        for step in 0u8..=20 {
            let mag = f32::from(step) * 0.05;
            let m = params.edge_mask(mag);
            assert!(m >= prev - CMP_EPS);
            prev = m;
        }
    }

    #[test]
    fn edge_mask_is_clamped_to_unit_interval() {
        let params = EdgeParams::new(0.3, 0.2, 1.0);
        for step in 0u8..=20 {
            let mag = f32::from(step) * 0.1;
            let m = params.edge_mask(mag);
            assert!(m >= 0.0);
            assert!(m <= 1.0);
        }
    }

    #[test]
    fn edge_mask_scale_amplifies_magnitude() {
        // scale = 2 pushes mag = 0.25 to the band center (threshold = 0.5).
        let params = EdgeParams::new(0.5, 0.5, 2.0);
        assert!(approx(params.edge_mask(0.25), 0.5));
    }

    #[test]
    fn edge_mask_zero_knee_is_hard_step() {
        let params = EdgeParams::new(0.5, 0.0, 1.0);
        assert!(approx(params.edge_mask(0.49), 0.0));
        assert!(approx(params.edge_mask(0.51), 1.0));
    }

    #[test]
    fn edge_params_new_clamps_negatives() {
        let params = EdgeParams::new(0.5, -1.0, -2.0);
        assert!(approx(params.knee, 0.0));
        assert!(approx(params.scale, 0.0));
    }
}
