//! Half-resolution shaft-mask bilateral upsample + full-resolution recombine
//! (CPU golden reference).
//!
//! Screen-space god rays are expensive, so the radial march in
//! [`super::radial`] usually runs at half resolution.  This module lifts that
//! low-resolution shaft back to full resolution with a **depth-aware (bilateral)
//! upsample** and composites it over the shaded scene with an additive screen
//! blend.
//!
//! A naive bilinear upsample bleeds the shaft across silhouettes, haloing the
//! edges of foreground geometry.  The bilateral upsample fixes this by weighting
//! each of the four low-resolution taps by both its bilinear spatial weight and
//! a depth-similarity weight:
//!
//! ```text
//! w_i      = w_spatial_i * exp(-(z_full - z_low_i)^2 / sigma_z^2)
//! shaft    = sum_i (w_i * mask_low_i) / sum_i w_i
//! ```
//!
//! Taps whose depth disagrees with the full-resolution texel are suppressed, so
//! the shaft stops cleanly at depth discontinuities instead of leaking.  When
//! every tap disagrees (a thin feature between two depth planes), the upsample
//! falls back to the nearest tap so a valid value is always produced.
//!
//! The recombine is the classic additive **screen** blend
//! `out = scene + shaft - scene*shaft` applied per channel, which brightens the
//! scene toward white without the harsh clipping of a plain add.
//!
//! # Conventions
//! * Low- and full-resolution grids are row-major `width * height`; the shaft
//!   mask and both depth buffers share their respective resolutions.
//! * `uv` is normalized `[0, 1]` screen space; depth orientation is irrelevant
//!   here because only depth *differences* are used.
//! * Upsample weights are normalized (sum to `1`) whenever any tap is valid;
//!   results are finite and non-negative.
//! * `sigma_z` is clamped to a small positive floor so the Gaussian never
//!   divides by zero; a non-finite depth tap is dropped from the blend.
//! * Transcendental maths goes through [`bevy_math::ops`] ([`ops::exp`]); every
//!   function is deterministic: no RNG, I/O, GPU, or `unsafe`.

use alloc::vec::Vec;

use bevy_math::{ops, Vec2, Vec3};

/// Smallest depth sigma used by the bilateral weight; guards divide-by-zero and
/// keeps the Gaussian well-conditioned.
const MIN_SIGMA_Z: f32 = 1e-5;

/// Largest exponent fed to [`ops::exp`] for the depth weight; beyond this the
/// weight is numerically zero.
const MAX_EXPONENT: f32 = 50.0;

/// A low-resolution shaft mask paired with the depth buffer it was rendered
/// against (row-major, `width * height`).
///
/// Both buffers must share the half-resolution grid dimensions.  The struct is
/// a thin, validated view used by [`upsample_bilateral`] and
/// [`upsample_buffer`].
#[derive(Clone, Debug, PartialEq)]
pub struct HalfResShaft {
    width: usize,
    height: usize,
    mask: Vec<f32>,
    depth: Vec<f32>,
}

impl HalfResShaft {
    /// Wraps parallel `mask`/`depth` buffers, validating their lengths.
    ///
    /// Returns `None` when either buffer is shorter than `width * height` or a
    /// dimension is zero.
    #[inline]
    pub fn new(width: usize, height: usize, mask: Vec<f32>, depth: Vec<f32>) -> Option<Self> {
        let len = width.checked_mul(height)?;
        if width == 0 || height == 0 || mask.len() < len || depth.len() < len {
            return None;
        }
        Some(Self { width, height, mask, depth })
    }

    /// Low-resolution width in texels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Low-resolution height in texels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Fetches a `(mask, depth)` pair, repairing non-finite storage.
    ///
    /// `mask` is sanitized to a finite, non-negative value; `depth` is returned
    /// as-is when finite, else `f32::NAN` so the caller can drop the tap.
    #[inline]
    fn tap(&self, x: usize, y: usize) -> (f32, f32) {
        let i = y * self.width + x;
        let m = self.mask[i];
        let d = self.depth[i];
        let m = if m.is_finite() { m.max(0.0) } else { 0.0 };
        let d = if d.is_finite() { d } else { f32::NAN };
        (m, d)
    }
}

/// Computes normalized bilateral upsample weights for the four low-resolution
/// taps surrounding a full-resolution texel.
///
/// `frac` is the fractional position of the full-resolution texel within the
/// low-resolution cell (each component in `[0, 1]`); the taps are ordered
/// `(0,0), (1,0), (0,1), (1,1)`.  `low_depths` are the four tap depths and
/// `full_depth` the full-resolution depth.  `sigma_z` controls how quickly a
/// depth mismatch suppresses a tap.
///
/// The returned weights sum to `1` when at least one tap is valid.  A tap with a
/// non-finite depth contributes `0`.  When every tap is invalid or the combined
/// weight underflows, the function falls back to the nearest tap (weight `1`).
pub fn bilateral_weights(
    frac: Vec2,
    full_depth: f32,
    low_depths: [f32; 4],
    sigma_z: f32,
) -> [f32; 4] {
    let fx = clamp_unit(frac.x);
    let fy = clamp_unit(frac.y);
    let spatial = [
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ];

    let sigma = sigma_z.max(MIN_SIGMA_Z);
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let fd = if full_depth.is_finite() { full_depth } else { f32::NAN };

    let mut w = [0.0_f32; 4];
    let mut sum = 0.0_f32;
    for k in 0..4 {
        let zd = low_depths[k];
        if !zd.is_finite() || !fd.is_finite() {
            continue;
        }
        let diff = fd - zd;
        let exponent = (diff * diff * inv_sigma2).min(MAX_EXPONENT);
        let depth_w = ops::exp(-exponent);
        let wk = spatial[k] * depth_w;
        if wk > 0.0 {
            w[k] = wk;
            sum += wk;
        }
    }

    if sum > f32::EPSILON {
        let inv = 1.0 / sum;
        for wk in &mut w {
            *wk *= inv;
        }
        return w;
    }

    // Fallback: pick the spatially nearest tap so a valid value is always
    // produced even across a hard depth discontinuity.
    let nearest = nearest_tap(fx, fy);
    let mut fb = [0.0_f32; 4];
    fb[nearest] = 1.0;
    fb
}

/// Bilateral-upsamples the half-resolution shaft at a full-resolution `uv`.
///
/// `uv` is normalized screen space; it is mapped into the low-resolution grid,
/// the four surrounding taps are gathered, weighted by [`bilateral_weights`],
/// and combined.  `full_depth` is the full-resolution depth at `uv`.  The result
/// is finite and non-negative.
pub fn upsample_bilateral(shaft: &HalfResShaft, uv: Vec2, full_depth: f32, sigma_z: f32) -> f32 {
    let w = shaft.width;
    let h = shaft.height;
    let u = if uv.x.is_finite() { uv.x } else { 0.0 };
    let v = if uv.y.is_finite() { uv.y } else { 0.0 };

    // Low-resolution texel-centre coordinates.
    let fx = (u * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (v * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let frac = Vec2::new(fx - x0 as f32, fy - y0 as f32);

    let (m00, d00) = shaft.tap(x0, y0);
    let (m10, d10) = shaft.tap(x1, y0);
    let (m01, d01) = shaft.tap(x0, y1);
    let (m11, d11) = shaft.tap(x1, y1);

    let weights = bilateral_weights(frac, full_depth, [d00, d10, d01, d11], sigma_z);
    let masks = [m00, m10, m01, m11];
    let mut acc = 0.0_f32;
    for k in 0..4 {
        acc += weights[k] * masks[k];
    }
    if acc.is_finite() { acc.max(0.0) } else { 0.0 }
}

/// Additive **screen** blend of a scalar shaft contribution over a scene colour.
///
/// Applies `out = scene + shaft*tint - scene*(shaft*tint)` per channel, where
/// `tint` is the shaft colour scaled by `intensity`.  Screen blending brightens
/// toward white and saturates at `1` per channel, avoiding the hard clipping of
/// a plain add.  All inputs are sanitized; the result is finite and in
/// `[0, 1]` per channel when the scene is.
#[inline]
pub fn composite_screen(scene: Vec3, shaft: f32, shaft_color: Vec3, intensity: f32) -> Vec3 {
    let s = sanitize_color(scene);
    let add = sanitize_color(shaft_color) * sanitize_scalar(shaft).max(0.0) * sanitize_scalar(intensity).max(0.0);
    let add = sanitize_color(add);
    screen_blend(s, add)
}

/// Plain additive composite `out = scene + shaft*tint` per channel.
///
/// A simpler alternative to [`composite_screen`] when the shaft energy is
/// already exposure-controlled.  Inputs are sanitized; the result is finite and
/// non-negative.
#[inline]
pub fn composite_additive(scene: Vec3, shaft: f32, shaft_color: Vec3, intensity: f32) -> Vec3 {
    let s = sanitize_color(scene);
    let add = sanitize_color(shaft_color) * sanitize_scalar(shaft).max(0.0) * sanitize_scalar(intensity).max(0.0);
    sanitize_color(s + sanitize_color(add))
}

/// Upsamples the whole half-resolution shaft against a full-resolution depth
/// buffer, returning a full-resolution shaft buffer (row-major).
///
/// `full_depth` is `full_width * full_height` row-major.  Mismatched or
/// degenerate sizes yield an empty result.  Every entry is finite and
/// non-negative.
pub fn upsample_buffer(
    shaft: &HalfResShaft,
    full_depth: &[f32],
    full_width: usize,
    full_height: usize,
    sigma_z: f32,
) -> Vec<f32> {
    let len = full_width.saturating_mul(full_height);
    if full_width == 0 || full_height == 0 || full_depth.len() < len {
        return Vec::new();
    }
    let mut out = alloc::vec![0.0_f32; len];
    let inv_w = 1.0 / full_width as f32;
    let inv_h = 1.0 / full_height as f32;
    for y in 0..full_height {
        for x in 0..full_width {
            let i = y * full_width + x;
            let uv = Vec2::new((x as f32 + 0.5) * inv_w, (y as f32 + 0.5) * inv_h);
            out[i] = upsample_bilateral(shaft, uv, full_depth[i], sigma_z);
        }
    }
    out
}

/// Per-channel screen blend `a + b - a*b`, assuming sanitized, non-negative
/// inputs.
#[inline]
fn screen_blend(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.x + b.x - a.x * b.x,
        a.y + b.y - a.y * b.y,
        a.z + b.z - a.z * b.z,
    )
}

/// Index of the spatially nearest of the four taps given fractional offsets.
#[inline]
fn nearest_tap(fx: f32, fy: f32) -> usize {
    let ix = if fx >= 0.5 { 1 } else { 0 };
    let iy = if fy >= 0.5 { 1 } else { 0 };
    iy * 2 + ix
}

/// Clamps a value to `[0, 1]`, mapping non-finite to `0`.
#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

/// Returns `v` when finite, else `0`.
#[inline]
fn sanitize_scalar(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Sanitizes a colour to be finite and non-negative per channel.
#[inline]
fn sanitize_color(c: Vec3) -> Vec3 {
    Vec3::new(
        sanitize_scalar(c.x).max(0.0),
        sanitize_scalar(c.y).max(0.0),
        sanitize_scalar(c.z).max(0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_on_flat_depth() {
        // Flat depth: bilateral reduces to plain bilinear, weights sum to 1.
        let w = bilateral_weights(Vec2::new(0.3, 0.7), 1.0, [1.0; 4], 0.1);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        // Corners recover the exact bilinear coefficients.
        let c = bilateral_weights(Vec2::ZERO, 1.0, [1.0; 4], 0.1);
        assert!((c[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flat_depth_matches_bilinear() {
        // With equal depths the bilateral blend equals a bilinear interpolation
        // of the mask values.
        let frac = Vec2::new(0.25, 0.75);
        let masks = [0.0_f32, 1.0, 2.0, 3.0];
        let w = bilateral_weights(frac, 0.5, [0.5; 4], 1.0);
        let blended: f32 = (0..4).map(|k| w[k] * masks[k]).sum();
        let fx = frac.x;
        let fy = frac.y;
        let bilinear = (1.0 - fx) * (1.0 - fy) * masks[0]
            + fx * (1.0 - fy) * masks[1]
            + (1.0 - fx) * fy * masks[2]
            + fx * fy * masks[3];
        assert!((blended - bilinear).abs() < 1e-5, "blended={blended} bilinear={bilinear}");
    }

    #[test]
    fn depth_discontinuity_does_not_leak() {
        // Three taps share the foreground depth (0.2) and one is a far background
        // tap (5.0).  With a tight sigma the far tap must be suppressed, so the
        // blended value ignores its (very different) mask.
        let frac = Vec2::new(0.5, 0.5);
        let full_depth = 0.2;
        let low_depths = [0.2, 0.2, 0.2, 5.0];
        let w = bilateral_weights(frac, full_depth, low_depths, 0.05);
        assert!(w[3] < 1e-3, "far tap not suppressed: {}", w[3]);
        // The surviving weights still normalize to 1.
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        // Blending a mask where only the far tap is bright yields ~0.
        let masks = [0.0_f32, 0.0, 0.0, 1.0];
        let blended: f32 = (0..4).map(|k| w[k] * masks[k]).sum();
        assert!(blended < 1e-3, "leaked across discontinuity: {blended}");
    }

    #[test]
    fn all_mismatched_depths_fall_back_to_nearest() {
        // Non-finite full depth drops every tap -> nearest-tap fallback.
        let w = bilateral_weights(Vec2::new(0.9, 0.9), f32::NAN, [1.0; 4], 0.1);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // (0.9, 0.9) is nearest the (1,1) tap -> index 3.
        assert_eq!(w[3], 1.0);
    }

    #[test]
    fn non_finite_tap_depth_is_dropped() {
        let w = bilateral_weights(Vec2::new(0.5, 0.5), 1.0, [1.0, f32::NAN, 1.0, 1.0], 0.2);
        assert_eq!(w[1], 0.0);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn upsample_bilateral_on_uniform_mask_is_constant() {
        let mask = alloc::vec![0.5_f32; 4]; // 2x2
        let depth = alloc::vec![1.0_f32; 4];
        let shaft = HalfResShaft::new(2, 2, mask, depth).unwrap();
        let v = upsample_bilateral(&shaft, Vec2::new(0.37, 0.62), 1.0, 0.1);
        assert!((v - 0.5).abs() < 1e-5, "v={v}");
    }

    #[test]
    fn upsample_buffer_dimensions_and_safety() {
        let mask = alloc::vec![1.0_f32; 4];
        let depth = alloc::vec![0.0_f32; 4];
        let shaft = HalfResShaft::new(2, 2, mask, depth).unwrap();
        let full_depth = alloc::vec![0.0_f32; 16];
        let out = upsample_buffer(&shaft, &full_depth, 4, 4, 0.1);
        assert_eq!(out.len(), 16);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5, "v={v}");
        }
        assert!(upsample_buffer(&shaft, &full_depth, 0, 0, 0.1).is_empty());
    }

    #[test]
    fn half_res_shaft_rejects_bad_sizes() {
        assert!(HalfResShaft::new(2, 2, alloc::vec![0.0; 2], alloc::vec![0.0; 4]).is_none());
        assert!(HalfResShaft::new(0, 2, alloc::vec![0.0; 0], alloc::vec![0.0; 0]).is_none());
        assert!(HalfResShaft::new(2, 2, alloc::vec![0.0; 4], alloc::vec![0.0; 4]).is_some());
    }

    #[test]
    fn screen_blend_brightens_and_saturates() {
        let scene = Vec3::splat(0.5);
        let out = composite_screen(scene, 1.0, Vec3::splat(0.5), 1.0);
        // screen(0.5, 0.5) = 0.75 per channel.
        assert!((out.x - 0.75).abs() < 1e-6, "out={out:?}");
        // A zero shaft leaves the scene unchanged.
        let noop = composite_screen(scene, 0.0, Vec3::ONE, 1.0);
        assert!((noop - scene).abs().max_element() < 1e-6);
        // Full white scene stays white (saturates at 1).
        let white = composite_screen(Vec3::ONE, 1.0, Vec3::ONE, 1.0);
        assert!((white - Vec3::ONE).abs().max_element() < 1e-6);
    }

    #[test]
    fn additive_composite_adds_energy() {
        let out = composite_additive(Vec3::splat(0.2), 0.5, Vec3::splat(0.4), 1.0);
        assert!((out.x - (0.2 + 0.2)).abs() < 1e-6, "out={out:?}");
    }

    #[test]
    fn composites_sanitize_non_finite_inputs() {
        let out = composite_screen(Vec3::new(f32::NAN, 0.3, 0.3), f32::INFINITY, Vec3::splat(0.5), 1.0);
        assert!(out.is_finite());
        let out2 = composite_additive(Vec3::splat(0.1), f32::NAN, Vec3::splat(0.5), 1.0);
        assert!(out2.is_finite());
    }
}
