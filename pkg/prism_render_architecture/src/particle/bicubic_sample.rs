//! Bicubic 2D image resampling for the particle texture-filtering contract
//! (design §8.4): the separable four-tap `Catmull-Rom` and `Mitchell-Netravali`
//! cubic reconstruction kernels, a small `CPU`-verifiable `RGBA` image sampler
//! with `clamp-to-edge` addressing, and the `Sigg-Hadwiger`-style five-tap
//! optimization that expresses a `Catmull-Rom` fetch as a handful of hardware
//! `bilinear` lookups so a `GPU` shader can reproduce this reference path.
//!
//! # Why particles need this
//! Flipbook frames, decal footprints, streak trails and heat-shimmer offsets are
//! all sampled at fractional texel coordinates. Nearest-neighbour aliases and a
//! plain `bilinear` fetch smears high-frequency sprite detail; a cubic
//! reconstruction filter keeps sprite edges crisp while staying free of the
//! ringing that a naive sinc window introduces. This module owns the tiny,
//! deterministic contract that both the `CPU` reference and the `GPU` sampler
//! agree on.
//!
//! # Strict scope
//! This module only performs 2D image reconstruction. It is deliberately
//! separate from the neighbourhood blur in `kawase_dual_blur`, from the 3D path
//! interpolation in `spline`, and from the keyframe evaluation in `curves`; it
//! neither imports nor rebuilds any of their types. A cubic here weights four
//! *texels along an axis*, not four control points of a motion curve.
//!
//! # No transcendental math
//! Every kernel is a piecewise cubic polynomial, so evaluation is nothing but
//! `+`, `-`, `*` and a division by the constant `6`. The only library calls are
//! `f32::floor` (to locate the base texel) and `f32::clamp`/`i32::clamp` (for
//! `clamp-to-edge` addressing). There is no `sin`, `exp`, `pow` or other
//! transcendental anywhere in the reconstruction.

use alloc::vec::Vec;

/// Evaluates the generalized `Mitchell-Netravali` cubic reconstruction kernel
/// at a non-negative distance `x` (in texels) for filter parameters `b` and
/// `c`.
///
/// The kernel is the standard two-piece cubic: a support of `[0, 2)` split at
/// `x = 1`. `Catmull-Rom` is the special case `b = 0, c = 0.5`; the cubic
/// `B-spline` is `b = 1, c = 0`; the balanced `Mitchell-Netravali` filter is
/// `b = 1/3, c = 1/3`. The whole family is a partition of unity, so the four
/// taps produced by [`cubic_weights`] always sum to one.
#[must_use]
fn cubic_kernel(x: f32, b: f32, c: f32) -> f32 {
    // `x` is passed as a non-negative distance by every caller in this module.
    if x < 1.0 {
        let x2 = x * x;
        let x3 = x2 * x;
        ((12.0 - 9.0 * b - 6.0 * c) * x3 + (-18.0 + 12.0 * b + 6.0 * c) * x2 + (6.0 - 2.0 * b))
            / 6.0
    } else if x < 2.0 {
        let x2 = x * x;
        let x3 = x2 * x;
        ((-b - 6.0 * c) * x3
            + (6.0 * b + 30.0 * c) * x2
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Returns the four separable cubic tap weights for a fractional position `t`
/// (the offset of the sample from the second of four consecutive texels, in
/// `[0, 1]`), using the generalized `Mitchell-Netravali` parameters `b` and
/// `c`.
///
/// The returned array is ordered from the far-left tap to the far-right tap:
/// index `0` weights the texel one step before the base texel, index `1` the
/// base texel, index `2` the next texel, and index `3` the texel two steps
/// after. Inputs outside `[0, 1]` are clamped so the weights stay on the
/// kernel's support. The four weights always sum to one.
#[must_use]
pub fn cubic_weights(t: f32, b: f32, c: f32) -> [f32; 4] {
    let tc = t.clamp(0.0, 1.0);
    [
        cubic_kernel(1.0 + tc, b, c),
        cubic_kernel(tc, b, c),
        cubic_kernel(1.0 - tc, b, c),
        cubic_kernel(2.0 - tc, b, c),
    ]
}

/// Returns the four `Catmull-Rom` tap weights for a fractional position `t`.
///
/// This is the convenience specialization of [`cubic_weights`] at the
/// interpolating `Catmull-Rom` parameters `b = 0, c = 0.5`. Because the filter
/// interpolates, `t = 0` yields `[0, 1, 0, 0]` (the sample lands exactly on the
/// base texel) and `t = 1` yields `[0, 0, 1, 0]` (exactly on the next texel).
#[must_use]
pub fn catmull_rom_weights(t: f32) -> [f32; 4] {
    cubic_weights(t, 0.0, 0.5)
}

/// A small linear `RGBA` image sampled by this module's reference path.
///
/// Pixels are stored row-major: the texel at column `x`, row `y` lives at
/// `data[y * width + x]`. Each texel is a four-channel `[r, g, b, a]` value in
/// linear space. Addressing outside the image uses the `clamp-to-edge` rule so
/// a bicubic footprint never reads out of bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleImage {
    /// Image width in texels.
    pub width: u32,
    /// Image height in texels.
    pub height: u32,
    /// Row-major `width * height` linear `RGBA` texels.
    pub data: Vec<[f32; 4]>,
}

impl SampleImage {
    /// Builds an image from explicit dimensions and a row-major `RGBA` buffer.
    ///
    /// The buffer length must equal `width * height`; a mismatch is a
    /// programming error in the caller and panics rather than silently
    /// truncating a footprint.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<[f32; 4]>) -> Self {
        let expected = usize::try_from(width)
            .expect("width fits in usize")
            .saturating_mul(usize::try_from(height).expect("height fits in usize"));
        assert_eq!(
            data.len(),
            expected,
            "texel count must equal width * height"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Builds a solid image whose every texel is `color`.
    #[must_use]
    pub fn solid(width: u32, height: u32, color: [f32; 4]) -> Self {
        let count = usize::try_from(width)
            .expect("width fits in usize")
            .saturating_mul(usize::try_from(height).expect("height fits in usize"));
        Self {
            width,
            height,
            data: alloc::vec![color; count],
        }
    }

    /// Returns `true` when the image has no texels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Fetches the texel at integer coordinates `(x, y)` using `clamp-to-edge`
    /// addressing: coordinates below zero or beyond the last texel snap to the
    /// nearest edge texel. An empty image reads as transparent black.
    #[must_use]
    pub fn fetch(&self, x: i32, y: i32) -> [f32; 4] {
        if self.is_empty() {
            return [0.0; 4];
        }
        let w = i32::try_from(self.width).expect("width fits in i32");
        let h = i32::try_from(self.height).expect("height fits in i32");
        let cx = x.clamp(0, w - 1);
        let cy = y.clamp(0, h - 1);
        let idx = usize::try_from(cy * w + cx).expect("clamped index is non-negative");
        self.data[idx]
    }

    /// Samples the image at fractional texel coordinates `(u, v)` with the
    /// generalized `Mitchell-Netravali` cubic filter of parameters `b` and `c`.
    ///
    /// Coordinates are in texel space: `u = 2.0, v = 3.0` addresses the centre
    /// of the texel at column `2`, row `3`. The `4x4` tap footprint is gathered
    /// with `clamp-to-edge` addressing, so sampling near a border re-reads the
    /// edge texels instead of reading out of bounds. An empty image returns
    /// transparent black.
    #[must_use]
    pub fn sample_bicubic(&self, u: f32, v: f32, b: f32, c: f32) -> [f32; 4] {
        if self.is_empty() {
            return [0.0; 4];
        }
        let ix0 = u.floor() as i32;
        let iy0 = v.floor() as i32;
        let tx = u - u.floor();
        let ty = v - v.floor();
        let wx = cubic_weights(tx, b, c);
        let wy = cubic_weights(ty, b, c);
        // Tap offsets relative to the base texel, matching `cubic_weights`.
        let offsets = [-1_i32, 0, 1, 2];
        let mut acc = [0.0f32; 4];
        for (&dy, &wyj) in offsets.iter().zip(wy.iter()) {
            let sy = iy0 + dy;
            for (&dx, &wxi) in offsets.iter().zip(wx.iter()) {
                let texel = self.fetch(ix0 + dx, sy);
                let weight = wxi * wyj;
                acc[0] += texel[0] * weight;
                acc[1] += texel[1] * weight;
                acc[2] += texel[2] * weight;
                acc[3] += texel[3] * weight;
            }
        }
        acc
    }

    /// Samples the image at fractional texel coordinates `(u, v)` with the
    /// interpolating `Catmull-Rom` filter.
    ///
    /// This is the convenience specialization of [`SampleImage::sample_bicubic`]
    /// at `b = 0, c = 0.5`. Because the filter interpolates, an integer
    /// coordinate returns the underlying texel exactly, and a linear ramp is
    /// reconstructed with no error away from the border.
    #[must_use]
    pub fn sample_catmull_rom(&self, u: f32, v: f32) -> [f32; 4] {
        self.sample_bicubic(u, v, 0.0, 0.5)
    }
}

/// Builds the `Sigg-Hadwiger`-style five-tap `Catmull-Rom` sampling plan for a
/// normalized texture coordinate `(u, v)` on a texture of `tex_size` texels.
///
/// A direct `Catmull-Rom` fetch needs a `4x4` point footprint. On the `GPU` the
/// two central taps of each axis can be folded into one hardware `bilinear`
/// lookup placed between them, which turns the `3x3` interior plan into a
/// five-tap "cross": the folded centre plus the four axis-aligned outer taps.
/// The four corners are dropped and the remaining five weights are renormalized
/// to sum to one, so a shader can reproduce this module's reference filter with
/// five `bilinear` samples instead of sixteen point reads.
///
/// The returned pair is `(uvs, weights)`: `uvs[i]` is the normalized `UV`
/// coordinate to feed a hardware `bilinear` sampler for tap `i`, and
/// `weights[i]` is the (renormalized) blend weight for that tap. The five
/// weights sum to one. When the sample lands exactly on a texel centre the plan
/// degenerates to a single centre tap of weight one.
#[must_use]
pub fn catmull_rom_5tap(u: f32, v: f32, tex_size: [f32; 2]) -> ([[f32; 2]; 5], [f32; 5]) {
    let sx = u * tex_size[0];
    let sy = v * tex_size[1];
    // Centre of the base texel (texel centres sit on the half-integers).
    let base_x = (sx - 0.5).floor() + 0.5;
    let base_y = (sy - 0.5).floor() + 0.5;
    let fx = sx - base_x;
    let fy = sy - base_y;
    let wx = catmull_rom_weights(fx);
    let wy = catmull_rom_weights(fy);
    // Fold the two central taps into one `bilinear` lookup. For `Catmull-Rom`
    // the central pair sums to at least one, so this divide is always safe.
    let w12x = wx[1] + wx[2];
    let w12y = wy[1] + wy[2];
    let off_x = wx[2] / w12x;
    let off_y = wy[2] / w12y;
    let inv_w = 1.0 / tex_size[0].max(1.0);
    let inv_h = 1.0 / tex_size[1].max(1.0);
    let px_lo = (base_x - 1.0) * inv_w;
    let px_hi = (base_x + 2.0) * inv_w;
    let px_mid = (base_x + off_x) * inv_w;
    let py_lo = (base_y - 1.0) * inv_h;
    let py_hi = (base_y + 2.0) * inv_h;
    let py_mid = (base_y + off_y) * inv_h;
    let uvs = [
        [px_mid, py_mid],
        [px_lo, py_mid],
        [px_hi, py_mid],
        [px_mid, py_lo],
        [px_mid, py_hi],
    ];
    let raw = [
        w12x * w12y,
        wx[0] * w12y,
        wx[3] * w12y,
        w12x * wy[0],
        w12x * wy[3],
    ];
    // Renormalize the cross so the surviving five weights sum to one; the
    // dropped corners are the small products of the two outer taps.
    let total = raw[0] + raw[1] + raw[2] + raw[3] + raw[4];
    let inv = 1.0 / total;
    let weights = [
        raw[0] * inv,
        raw[1] * inv,
        raw[2] * inv,
        raw[3] * inv,
        raw[4] * inv,
    ];
    (uvs, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Magnitude below which two floats are treated as equal in these tests.
    const CMP_EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx_tol(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn weight_sum(w: [f32; 4]) -> f32 {
        w[0] + w[1] + w[2] + w[3]
    }

    #[test]
    fn catmull_rom_weights_sum_to_one_over_many_t() {
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            assert!(approx(weight_sum(catmull_rom_weights(t)), 1.0));
        }
    }

    #[test]
    fn mitchell_weights_sum_to_one_over_many_t() {
        let b = 1.0 / 3.0;
        let c = 1.0 / 3.0;
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            assert!(approx(weight_sum(cubic_weights(t, b, c)), 1.0));
        }
    }

    #[test]
    fn bspline_weights_sum_to_one_over_many_t() {
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            assert!(approx(weight_sum(cubic_weights(t, 1.0, 0.0)), 1.0));
        }
    }

    #[test]
    fn arbitrary_bc_weights_are_partition_of_unity() {
        let params = [
            [0.0, 0.5],
            [1.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [0.25, 0.75],
            [0.5, 0.25],
        ];
        for &[b, c] in &params {
            for step in 0..=10 {
                let t = step as f32 / 10.0;
                assert!(approx(weight_sum(cubic_weights(t, b, c)), 1.0));
            }
        }
    }

    #[test]
    fn catmull_rom_weights_at_endpoints_interpolate() {
        let w0 = catmull_rom_weights(0.0);
        assert!(approx(w0[0], 0.0));
        assert!(approx(w0[1], 1.0));
        assert!(approx(w0[2], 0.0));
        assert!(approx(w0[3], 0.0));
        let w1 = catmull_rom_weights(1.0);
        assert!(approx(w1[0], 0.0));
        assert!(approx(w1[1], 0.0));
        assert!(approx(w1[2], 1.0));
        assert!(approx(w1[3], 0.0));
    }

    #[test]
    fn catmull_rom_matches_generic_cubic_b0_c05() {
        for step in 0..=16 {
            let t = step as f32 / 16.0;
            let a = catmull_rom_weights(t);
            let b = cubic_weights(t, 0.0, 0.5);
            for k in 0..4 {
                assert!(approx(a[k], b[k]));
            }
        }
    }

    #[test]
    fn cubic_weights_are_symmetric_about_half() {
        // w(t) reversed equals w(1 - t): w(t)[i] == w(1 - t)[3 - i].
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let a = catmull_rom_weights(t);
            let b = catmull_rom_weights(1.0 - t);
            for k in 0..4 {
                assert!(approx(a[k], b[3 - k]));
            }
        }
    }

    #[test]
    fn mitchell_weights_are_symmetric_about_half() {
        let bp = 1.0 / 3.0;
        let cp = 1.0 / 3.0;
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let a = cubic_weights(t, bp, cp);
            let b = cubic_weights(1.0 - t, bp, cp);
            for k in 0..4 {
                assert!(approx(a[k], b[3 - k]));
            }
        }
    }

    #[test]
    fn cubic_weights_clamp_out_of_range_input() {
        assert_eq!(cubic_weights(-0.5, 0.0, 0.5), catmull_rom_weights(0.0));
        assert_eq!(cubic_weights(1.7, 0.0, 0.5), catmull_rom_weights(1.0));
    }

    #[test]
    fn bspline_weights_have_no_negative_lobe() {
        // The cubic `B-spline` (b = 1, c = 0) is purely smoothing: no ringing.
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            let w = cubic_weights(t, 1.0, 0.0);
            for &wi in &w {
                assert!(wi >= -CMP_EPS);
            }
        }
    }

    #[test]
    fn catmull_rom_has_a_negative_outer_lobe() {
        // The interpolating filter must have a negative lobe to stay sharp.
        let w = catmull_rom_weights(0.5);
        assert!(w[0] < 0.0);
        assert!(w[3] < 0.0);
    }

    #[test]
    fn mitchell_lobe_is_smaller_than_catmull_rom() {
        // The balanced `Mitchell-Netravali` filter rings less: its outer lobe
        // is smaller in magnitude than `Catmull-Rom`'s at the midpoint.
        let cat = catmull_rom_weights(0.5);
        let mit = cubic_weights(0.5, 1.0 / 3.0, 1.0 / 3.0);
        assert!(mit[0].abs() < cat[0].abs());
        assert!(mit[3].abs() < cat[3].abs());
    }

    fn ramp_image_x(width: u32, height: u32, base: f32, slope: f32) -> SampleImage {
        let mut data = Vec::new();
        for _y in 0..height {
            for x in 0..width {
                let value = base + slope * x as f32;
                data.push([value, 0.0, 0.0, 1.0]);
            }
        }
        SampleImage::new(width, height, data)
    }

    fn ramp_image_y(width: u32, height: u32, base: f32, slope: f32) -> SampleImage {
        let mut data = Vec::new();
        for y in 0..height {
            for _x in 0..width {
                let value = base + slope * y as f32;
                data.push([0.0, value, 0.0, 1.0]);
            }
        }
        SampleImage::new(width, height, data)
    }

    #[test]
    fn fetch_clamps_to_edge() {
        let img = ramp_image_x(4, 3, 0.0, 1.0);
        // Left / top clamp.
        assert_eq!(img.fetch(-5, -3), img.fetch(0, 0));
        // Right / bottom clamp.
        assert_eq!(img.fetch(99, 99), img.fetch(3, 2));
        // Interior read is exact.
        assert_eq!(img.fetch(2, 1)[0], 2.0);
    }

    #[test]
    fn fetch_on_empty_image_is_transparent_black() {
        let img = SampleImage::new(0, 0, Vec::new());
        assert_eq!(img.fetch(0, 0), [0.0; 4]);
        assert!(img.is_empty());
    }

    #[test]
    fn sample_bicubic_preserves_a_constant_image() {
        let img = SampleImage::solid(6, 5, [0.2, 0.4, 0.6, 0.8]);
        for &(u, v) in &[(2.0, 2.0), (2.3, 1.7), (0.5, 4.2), (5.0, 0.0)] {
            let s = img.sample_bicubic(u, v, 1.0 / 3.0, 1.0 / 3.0);
            assert!(approx(s[0], 0.2));
            assert!(approx(s[1], 0.4));
            assert!(approx(s[2], 0.6));
            assert!(approx(s[3], 0.8));
        }
    }

    #[test]
    fn sample_catmull_rom_preserves_a_constant_image() {
        let img = SampleImage::solid(6, 5, [0.9, 0.1, 0.5, 1.0]);
        let s = img.sample_catmull_rom(3.25, 2.75);
        assert!(approx(s[0], 0.9));
        assert!(approx(s[1], 0.1));
        assert!(approx(s[2], 0.5));
        assert!(approx(s[3], 1.0));
    }

    #[test]
    fn sample_catmull_rom_passes_through_sample_points() {
        let img = ramp_image_x(6, 4, 1.0, 2.0);
        for x in 1..5 {
            let s = img.sample_catmull_rom(x as f32, 2.0);
            assert!(approx(s[0], 1.0 + 2.0 * x as f32));
        }
    }

    #[test]
    fn sample_catmull_rom_passes_through_sample_points_in_v() {
        let img = ramp_image_y(4, 6, 0.5, 1.5);
        for y in 1..5 {
            let s = img.sample_catmull_rom(2.0, y as f32);
            assert!(approx(s[1], 0.5 + 1.5 * y as f32));
        }
    }

    #[test]
    fn sample_catmull_rom_reconstructs_linear_ramp_x() {
        let img = ramp_image_x(8, 4, 0.25, 0.5);
        // Interior fractional positions avoid the clamped border.
        for &u in &[2.3_f32, 3.5, 4.75, 5.1] {
            let s = img.sample_catmull_rom(u, 2.0);
            assert!(approx_tol(s[0], 0.25 + 0.5 * u, 1.0e-4));
        }
    }

    #[test]
    fn sample_catmull_rom_reconstructs_linear_ramp_y() {
        let img = ramp_image_y(4, 8, -1.0, 0.75);
        for &v in &[2.2_f32, 3.6, 4.9, 5.4] {
            let s = img.sample_catmull_rom(2.0, v);
            assert!(approx_tol(s[1], -1.0 + 0.75 * v, 1.0e-4));
        }
    }

    #[test]
    fn sample_catmull_rom_reconstructs_bilinear_ramp_2d() {
        // A separable ramp a*x + b*y + c must be reconstructed exactly.
        let (aa, bb, cc) = (0.3_f32, -0.2_f32, 1.0_f32);
        let mut data = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let value = aa * x as f32 + bb * y as f32 + cc;
                data.push([value, 0.0, 0.0, 1.0]);
            }
        }
        let img = SampleImage::new(8, 8, data);
        for &(u, v) in &[(3.3_f32, 4.2_f32), (4.5, 3.5), (5.1, 5.9)] {
            let s = img.sample_catmull_rom(u, v);
            let expect = aa * u + bb * v + cc;
            assert!(approx_tol(s[0], expect, 1.0e-4));
        }
    }

    #[test]
    fn sample_bicubic_clamps_at_the_border() {
        let img = ramp_image_x(4, 4, 0.0, 1.0);
        // Sampling well past the right edge must return the edge value.
        let s = img.sample_catmull_rom(10.0, 1.0);
        assert!(approx(s[0], 3.0));
        // Sampling before the left edge must return the first column value.
        let s2 = img.sample_catmull_rom(-4.0, 1.0);
        assert!(approx(s2[0], 0.0));
    }

    #[test]
    fn mitchell_does_not_overshoot_a_step_edge() {
        // Build a horizontal step: 0 on the left half, 1 on the right half.
        let mut data = Vec::new();
        for _y in 0..4 {
            for x in 0..8 {
                let value = if x < 4 { 0.0 } else { 1.0 };
                data.push([value, 0.0, 0.0, 1.0]);
            }
        }
        let img = SampleImage::new(8, 4, data);
        let mut cat_max = 0.0f32;
        let mut mit_max = 0.0f32;
        for step in 0..=40 {
            let u = 2.0 + step as f32 / 10.0;
            cat_max = cat_max.max(img.sample_catmull_rom(u, 2.0)[0]);
            mit_max = mit_max.max(img.sample_bicubic(u, 2.0, 1.0 / 3.0, 1.0 / 3.0)[0]);
        }
        // Catmull-Rom overshoots above 1; balanced Mitchell rings much less.
        assert!(cat_max > 1.0);
        assert!(mit_max < cat_max);
    }

    fn five_tap_sum(w: [f32; 5]) -> f32 {
        w[0] + w[1] + w[2] + w[3] + w[4]
    }

    #[test]
    fn five_tap_weights_sum_to_one() {
        let tex = [16.0, 12.0];
        for &(u, v) in &[(0.31, 0.62), (0.5, 0.5), (0.07, 0.93), (0.845, 0.123)] {
            let (_uvs, w) = catmull_rom_5tap(u, v, tex);
            assert!(approx(five_tap_sum(w), 1.0));
        }
    }

    #[test]
    fn five_tap_degenerates_at_a_texel_center() {
        // A sample exactly on a texel centre folds to a single centre tap.
        let tex = [8.0, 8.0];
        let u = (2.0 + 0.5) / tex[0];
        let v = (3.0 + 0.5) / tex[1];
        let (uvs, w) = catmull_rom_5tap(u, v, tex);
        assert!(approx(w[0], 1.0));
        assert!(approx(w[1], 0.0));
        assert!(approx(w[2], 0.0));
        assert!(approx(w[3], 0.0));
        assert!(approx(w[4], 0.0));
        // The centre tap must land on the texel-centre UV itself.
        assert!(approx(uvs[0][0], u));
        assert!(approx(uvs[0][1], v));
    }

    #[test]
    fn five_tap_center_uv_stays_near_the_query() {
        let tex = [32.0, 24.0];
        let (u, v) = (0.4123_f32, 0.7311_f32);
        let (uvs, _w) = catmull_rom_5tap(u, v, tex);
        // The folded centre tap stays within one texel of the query point.
        assert!(approx_tol(uvs[0][0], u, 1.0 / tex[0]));
        assert!(approx_tol(uvs[0][1], v, 1.0 / tex[1]));
    }

    #[test]
    fn five_tap_outer_taps_straddle_the_center() {
        // The left outer tap sits below the centre in U; the right sits above,
        // and likewise the vertical outer taps straddle in V.
        let tex = [16.0, 16.0];
        let (uvs, _w) = catmull_rom_5tap(0.53, 0.53, tex);
        assert!(uvs[1][0] < uvs[0][0]);
        assert!(uvs[2][0] > uvs[0][0]);
        assert!(uvs[3][1] < uvs[0][1]);
        assert!(uvs[4][1] > uvs[0][1]);
    }

    #[test]
    fn five_tap_reconstructs_a_constant_via_its_plan() {
        // Feeding a constant field through the five weighted taps reproduces
        // the constant, because the renormalized weights sum to one.
        let tex = [16.0, 16.0];
        let (_uvs, w) = catmull_rom_5tap(0.37, 0.81, tex);
        let field = 0.625f32;
        let reconstructed = five_tap_sum(w) * field;
        assert!(approx(reconstructed, field));
    }

    #[test]
    fn sample_image_new_accepts_matching_length() {
        let img = SampleImage::new(2, 3, alloc::vec![[0.0; 4]; 6]);
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 3);
        assert_eq!(img.data.len(), 6);
    }

    #[test]
    #[should_panic(expected = "texel count must equal width * height")]
    fn sample_image_new_rejects_mismatched_length() {
        let _ = SampleImage::new(2, 3, alloc::vec![[0.0; 4]; 5]);
    }
}
