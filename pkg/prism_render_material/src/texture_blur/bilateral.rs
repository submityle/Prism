//! Native-resolution **edge-preserving bilateral** blur for decoded `RGBA8`
//! images.
//!
//! Where [`gaussian`](super::gaussian) and [`box_blur`](super::box_blur) are
//! *linear*, space-invariant filters that smooth across every edge equally,
//! the bilateral filter (Tomasi & Manduchi 1998) multiplies the usual spatial
//! Gaussian by a second Gaussian on the **intensity difference** between the
//! centre and each neighbour. Taps that straddle a strong edge get a tiny
//! range weight, so the kernel collapses to the side of the edge the centre
//! sits on -- the filter smooths flat regions yet leaves edges crisp. That is
//! the fixed-function primitive behind edge-aware denoise (SSAO / shadow /
//! RT-GI), detail-preserving tone and skin softening, and the base layer of a
//! detail-transfer / local-tone-mapping decomposition.
//!
//! Because the range weight depends on the data, the filter is **not
//! separable**: each output texel is a full 2-D gather over the
//! `|dx|,|dy| <= ceil(3 sigma_spatial)` window. The per-pixel weights are
//!
//! ```text
//! w(dx, dy) = exp(-(dx^2 + dy^2) / (2 sigma_spatial^2))
//!           * exp(-(I_center - I_tap)^2 / (2 sigma_range^2))
//! ```
//!
//! and are renormalised per output texel so the result is a **convex
//! combination** of the input: non-negative weights that sum to `1`, hence a
//! constant image is preserved exactly and the output can never over- or
//! under-shoot the input range (no ringing).
//!
//! Properties the oracles below pin down:
//! * as `sigma_range -> infinity` the range weight -> `1` and the filter
//!   **converges to the plain separable Gaussian** [`blur_plane`](super::blur_plane):
//!   a product kernel's 2-D normalisation equals the separable per-axis
//!   normalisation, `sum_ij gx[i] gy[j] = (sum gx)(sum gy)`, so the two agree
//!   to floating-point tolerance -- this ties the new code directly to the
//!   already-verified Gaussian;
//! * on a step edge a small `sigma_range` keeps the output **strictly closer
//!   to the input** than the same-`sigma_spatial` Gaussian would -- the
//!   edge-preservation guarantee;
//! * a non-positive `sigma_spatial` or `sigma_range` is the **identity** copy.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a
//! CPU golden matches a GPU compute bilateral to floating-point tolerance.
//!
//! # References
//! * Tomasi & Manduchi, "Bilateral Filtering for Gray and Color Images" (1998).
//! * Paris, Kornprobst, Tumblin & Durand, "Bilateral Filtering: Theory and
//!   Applications" (2009).

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Half-support of the spatial Gaussian, in texels: taps beyond `3 sigma` have
/// decayed below ~1% and are dropped. Matches `gaussian::radius_for`.
#[inline]
#[must_use]
fn radius_for(sigma: f32) -> usize {
    (sigma * 3.0).ceil().max(0.0) as usize
}

/// Wrap an integer tap index into `[0, n)`.
///
/// `Repeat` and `MirroredRepeat` keep tiling / mirrored sources seamless; every
/// other mode collapses to edge clamp. (Local copy of the shared helper so this
/// module stays self-contained.)
#[inline]
#[must_use]
fn wrap_index(i: i64, n: i64, mode: WrapMode) -> usize {
    match mode {
        WrapMode::Repeat => i.rem_euclid(n) as usize,
        WrapMode::MirroredRepeat => {
            let p = i.rem_euclid(2 * n);
            let m = if p >= n { 2 * n - 1 - p } else { p };
            m as usize
        }
        _ => i.clamp(0, n - 1) as usize,
    }
}

/// Edge-preserving bilateral blur of a single-channel `width * height` row-major
/// plane: spatial standard deviation `sigma_spatial` (texels) combined with a
/// range standard deviation `sigma_range` (same units as the samples),
/// resolving borders through `wrap`.
///
/// Returns a copy when either sigma is non-positive / non-finite, either
/// dimension is `0`, or the length does not match `width * height` (the
/// identity, so the filter degrades gracefully).
#[must_use]
pub fn bilateral_blur_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    sigma_spatial: f32,
    sigma_range: f32,
    wrap: WrapMode,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h {
        return plane.to_vec();
    }
    if !sigma_spatial.is_finite() || sigma_spatial <= 0.0 {
        return plane.to_vec();
    }
    if !sigma_range.is_finite() || sigma_range <= 0.0 {
        return plane.to_vec();
    }

    let r = radius_for(sigma_spatial) as i64;
    if r == 0 {
        return plane.to_vec();
    }
    let wi = width as i64;
    let hi = height as i64;
    let inv2s2 = 1.0 / (2.0 * sigma_spatial * sigma_spatial);
    let inv2r2 = 1.0 / (2.0 * sigma_range * sigma_range);

    // Precompute the separable spatial weights (the range weight is per-tap and
    // data-dependent, so only the spatial factor can be cached).
    let spatial: Vec<f32> = (-r..=r)
        .map(|d| {
            let df = d as f32;
            ops::exp(-df * df * inv2s2)
        })
        .collect();

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let center = plane[y * w + x];
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for (dyi, &sy_w) in spatial.iter().enumerate() {
                let dy = dyi as i64 - r;
                let sy = wrap_index(y as i64 + dy, hi, wrap);
                let srow = sy * w;
                for (dxi, &sx_w) in spatial.iter().enumerate() {
                    let dx = dxi as i64 - r;
                    let sx = wrap_index(x as i64 + dx, wi, wrap);
                    let sample = plane[srow + sx];
                    let diff = sample - center;
                    let rw = ops::exp(-diff * diff * inv2r2);
                    let weight = sy_w * sx_w * rw;
                    acc += weight * sample;
                    wsum += weight;
                }
            }
            // `wsum` includes the centre tap (weight 1), so it is always >= 1.
            out[y * w + x] = acc / wsum;
        }
    }
    out
}

/// Edge-preserving bilateral blur of an `RGBA8` image at its own resolution:
/// spatial standard deviation `sigma_spatial` (texels) and range standard
/// deviation `sigma_range` (in scene-linear `[0, 1]` units), gamma-correctly
/// under `space`, resolving borders through `wrap`.
///
/// Colour channels are filtered in scene-linear light with an independent
/// per-channel range term; alpha is filtered linearly (`/255`). Returns an
/// exact copy when either sigma is non-positive.
#[must_use]
pub fn bilateral_blur(
    src: &Rgba8Image,
    sigma_spatial: f32,
    sigma_range: f32,
    wrap: WrapMode,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return None;
    }
    if !sigma_spatial.is_finite()
        || sigma_spatial <= 0.0
        || !sigma_range.is_finite()
        || sigma_range <= 0.0
    {
        return Rgba8Image::new(w, h, src.as_slice().to_vec());
    }

    let n = w as usize * h as usize;
    let mut r = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut a = vec![0.0f32; n];
    for (i, t) in src.as_slice().iter().enumerate() {
        match space {
            ColorSpace::Linear => {
                r[i] = f32::from(t[0]) / 255.0;
                g[i] = f32::from(t[1]) / 255.0;
                b[i] = f32::from(t[2]) / 255.0;
            }
            ColorSpace::Srgb => {
                r[i] = srgb_to_linear(t[0]);
                g[i] = srgb_to_linear(t[1]);
                b[i] = srgb_to_linear(t[2]);
            }
        }
        a[i] = f32::from(t[3]) / 255.0;
    }

    let r = bilateral_blur_plane(&r, w, h, sigma_spatial, sigma_range, wrap);
    let g = bilateral_blur_plane(&g, w, h, sigma_spatial, sigma_range, wrap);
    let b = bilateral_blur_plane(&b, w, h, sigma_spatial, sigma_range, wrap);
    let a = bilateral_blur_plane(&a, w, h, sigma_spatial, sigma_range, wrap);

    let round_u8 = |v: f32| -> u8 {
        let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        (c * 255.0 + 0.5).floor() as u8
    };
    let texels: Vec<[u8; 4]> = (0..n)
        .map(|i| {
            let au = round_u8(a[i]);
            match space {
                ColorSpace::Linear => [round_u8(r[i]), round_u8(g[i]), round_u8(b[i]), au],
                ColorSpace::Srgb => [
                    linear_to_srgb(r[i]),
                    linear_to_srgb(g[i]),
                    linear_to_srgb(b[i]),
                    au,
                ],
            }
        })
        .collect();
    Rgba8Image::new(w, h, texels)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WRAPS: [WrapMode; 3] = [
        WrapMode::ClampToEdge,
        WrapMode::Repeat,
        WrapMode::MirroredRepeat,
    ];

    #[test]
    fn constant_plane_is_preserved() {
        let (w, h) = (7u32, 5u32);
        let plane = vec![0.42f32; (w * h) as usize];
        for &wrap in &WRAPS {
            let out = bilateral_blur_plane(&plane, w, h, 2.0, 0.1, wrap);
            for v in out {
                assert!((v - 0.42).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn nonpositive_sigma_is_identity() {
        let (w, h) = (6u32, 4u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| (i as f32) * 0.017).collect();
        // Non-positive spatial sigma.
        assert_eq!(
            bilateral_blur_plane(&plane, w, h, 0.0, 0.5, WrapMode::Repeat),
            plane
        );
        assert_eq!(
            bilateral_blur_plane(&plane, w, h, -1.0, 0.5, WrapMode::Repeat),
            plane
        );
        // Non-positive range sigma.
        assert_eq!(
            bilateral_blur_plane(&plane, w, h, 2.0, 0.0, WrapMode::Repeat),
            plane
        );
        assert_eq!(
            bilateral_blur_plane(&plane, w, h, 2.0, -0.3, WrapMode::Repeat),
            plane
        );
    }

    #[test]
    fn output_stays_within_input_range() {
        // Non-negative weights that sum to 1 => convex combination => in range.
        let (w, h) = (9u32, 9u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = ((x * 5 + y * 3) % 7) as f32 * 0.1 + 0.1;
            }
        }
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &wrap in &WRAPS {
            for &sr in &[0.05f32, 0.3, 2.0] {
                let out = bilateral_blur_plane(&plane, w, h, 2.0, sr, wrap);
                for v in out {
                    assert!(
                        v >= lo - 1.0e-6 && v <= hi + 1.0e-6,
                        "v={v} range [{lo},{hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn converges_to_separable_gaussian_as_range_sigma_grows() {
        // As sigma_range -> infinity the range weight -> 1 and the bilateral
        // filter becomes the plain separable Gaussian. A product kernel's 2-D
        // normalisation equals the separable per-axis normalisation, so the two
        // agree to floating-point tolerance (add-order aside). Repeat wrap with
        // the radius smaller than both dimensions avoids multi-fold mismatch.
        let (w, h) = (12u32, 10u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] =
                    0.5 + 0.3 * ops::cos(core::f32::consts::TAU * (x as f32) / w as f32);
            }
        }
        let sigma_spatial = 1.5f32;
        let bil = bilateral_blur_plane(&plane, w, h, sigma_spatial, 1.0e6, WrapMode::Repeat);
        let gauss = super::super::blur_plane(&plane, w, h, sigma_spatial, WrapMode::Repeat);
        for (a, b) in bil.iter().zip(gauss.iter()) {
            assert!((a - b).abs() < 1.0e-4, "bilateral {a} vs gaussian {b}");
        }
    }

    #[test]
    fn preserves_a_step_edge_better_than_a_gaussian() {
        // On a sharp step edge a small range sigma keeps the output strictly
        // closer to the input than a same-spatial-sigma Gaussian, which bleeds
        // the two sides into each other.
        let (w, h) = (16u32, 1u32);
        let plane: Vec<f32> = (0..w).map(|x| if x < w / 2 { 0.0 } else { 1.0 }).collect();
        let sigma_spatial = 2.0f32;
        let bil = bilateral_blur_plane(&plane, w, h, sigma_spatial, 0.05, WrapMode::ClampToEdge);
        let gauss = super::super::blur_plane(&plane, w, h, sigma_spatial, WrapMode::ClampToEdge);

        let bil_err: f32 = plane
            .iter()
            .zip(bil.iter())
            .map(|(p, q)| (p - q).abs())
            .sum();
        let gauss_err: f32 = plane
            .iter()
            .zip(gauss.iter())
            .map(|(p, q)| (p - q).abs())
            .sum();
        assert!(
            bil_err < gauss_err,
            "bilateral should hug the edge: bil {bil_err} vs gauss {gauss_err}"
        );
        // The flat interior far from the edge is left untouched.
        assert!((bil[0] - 0.0).abs() < 1.0e-4);
        assert!((bil[(w - 1) as usize] - 1.0).abs() < 1.0e-4);
    }

    #[test]
    fn image_constant_is_preserved() {
        let texels = vec![[90u8, 140, 200, 255]; 6 * 4];
        let img = Rgba8Image::new(6, 4, texels).unwrap();
        for &space in &[ColorSpace::Linear, ColorSpace::Srgb] {
            let out = bilateral_blur(&img, 2.0, 0.2, WrapMode::Repeat, space).unwrap();
            for (a, b) in out.as_slice().iter().zip(img.as_slice().iter()) {
                assert_eq!(a, b, "constant image preserved");
            }
        }
    }

    #[test]
    fn image_nonpositive_sigma_is_identity_copy() {
        let texels: Vec<[u8; 4]> = (0..6 * 4)
            .map(|i| [i as u8 * 3, i as u8 * 5, i as u8 * 7, 255])
            .collect();
        let img = Rgba8Image::new(6, 4, texels).unwrap();
        let a = bilateral_blur(&img, 0.0, 0.2, WrapMode::Repeat, ColorSpace::Linear).unwrap();
        let b = bilateral_blur(&img, 2.0, 0.0, WrapMode::Repeat, ColorSpace::Linear).unwrap();
        assert_eq!(a.as_slice(), img.as_slice());
        assert_eq!(b.as_slice(), img.as_slice());
    }
}
