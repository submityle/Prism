//! Native-resolution **unsharp-mask** sharpening for decoded `RGBA8` images.
//!
//! The oldest and most widely deployed sharpening operator (named for the
//! darkroom technique of compositing a photo over its own blurred, "unsharp"
//! negative): subtract a low-pass copy to isolate the high-frequency detail,
//! then add a scaled multiple of that detail back onto the original.
//!
//! ```text
//! out = src + amount * (src - blur(src, sigma))
//!     = (1 + amount) * src - amount * blur(src, sigma)
//! ```
//!
//! The low-pass is the already-verified separable, gamma-correct Gaussian
//! [`blur_plane`](super::blur_plane); `sigma` sets the detail *scale* that is
//! boosted and `amount >= 0` the boost strength (`0` is a no-op). This is the
//! fixed-function primitive behind texture-import / mip sharpening, post
//! sharpen passes, and detail re-injection after an edge-preserving denoise.
//!
//! Unlike a blur, unsharp masking is **not** a convex combination -- the
//! negative `-amount * blur` lobe is exactly what overshoots at an edge to
//! create the perceived "crispness", so the output can exceed the input range
//! (and is clamped on the way back to `RGBA8`).
//!
//! Properties the oracles below pin down:
//! * the plane sharpener equals the **closed form**
//!   `(1 + amount) * src - amount * blur_plane(src, sigma)` recomputed from the
//!   verified Gaussian -- tying this unit directly to already-tested code;
//! * `amount = 0` (or a non-positive / non-finite `sigma`) is the **identity**;
//! * the Gaussian reproduces constant and affine (linear-ramp) signals exactly,
//!   so `src - blur = 0` there and **both are preserved** for any `amount` --
//!   sharpening only acts where curvature (detail) exists;
//! * at a step edge a positive `amount` **overshoots** past the input range
//!   (the halo), the defining behaviour of the operator.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a
//! CPU golden matches a GPU compute sharpen to floating-point tolerance.
//!
//! # References
//! * Gonzalez & Woods, *Digital Image Processing* -- unsharp masking and
//!   high-boost filtering.
//! * `DirectXTex` / NVIDIA Texture Tools post-mip sharpen.

use alloc::vec;
use alloc::vec::Vec;

use super::blur_plane;
use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Unsharp-mask a single-channel `width * height` row-major plane: boost detail
/// at scale `sigma` (texels) by `amount` (`out = src + amount * (src - blur)`),
/// resolving the blur's borders through `wrap`.
///
/// Returns a copy when `amount <= 0`, `sigma` is non-positive / non-finite,
/// either dimension is `0`, or the length does not match `width * height`.
#[must_use]
pub fn unsharp_mask_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
    amount: f32,
    wrap: WrapMode,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h {
        return plane.to_vec();
    }
    if !amount.is_finite() || amount <= 0.0 || !sigma.is_finite() || sigma <= 0.0 {
        return plane.to_vec();
    }
    let blurred = blur_plane(plane, width, height, sigma, wrap);
    plane
        .iter()
        .zip(blurred.iter())
        .map(|(&p, &b)| p + amount * (p - b))
        .collect()
}

/// Unsharp-mask an `RGBA8` image at its own resolution: boost detail at scale
/// `sigma` (texels) by `amount`, gamma-correctly under `space`, resolving the
/// blur's borders through `wrap`.
///
/// Colour channels are sharpened in scene-linear light; alpha is sharpened
/// linearly (`/255`). The high-boost result is clamped back to `[0, 1]` before
/// re-encoding. Returns an exact copy when `amount <= 0` or `sigma <= 0`.
#[must_use]
pub fn unsharp_mask(
    src: &Rgba8Image,
    sigma: f32,
    amount: f32,
    wrap: WrapMode,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return None;
    }
    if !amount.is_finite() || amount <= 0.0 || !sigma.is_finite() || sigma <= 0.0 {
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

    let r = unsharp_mask_plane(&r, w, h, sigma, amount, wrap);
    let g = unsharp_mask_plane(&g, w, h, sigma, amount, wrap);
    let b = unsharp_mask_plane(&b, w, h, sigma, amount, wrap);
    let a = unsharp_mask_plane(&a, w, h, sigma, amount, wrap);

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
    use bevy_math::ops;

    const WRAPS: [WrapMode; 3] = [
        WrapMode::ClampToEdge,
        WrapMode::Repeat,
        WrapMode::MirroredRepeat,
    ];

    #[test]
    fn matches_closed_form_high_boost() {
        // out == (1 + amount) * src - amount * blur, recomputed independently
        // from the verified Gaussian blur.
        let (w, h) = (11u32, 9u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] =
                    0.5 + 0.3 * ops::cos(core::f32::consts::TAU * (x as f32) / w as f32);
            }
        }
        let sigma = 1.5f32;
        for &wrap in &WRAPS {
            for &amount in &[0.5f32, 1.0, 2.5] {
                let got = unsharp_mask_plane(&plane, w, h, sigma, amount, wrap);
                let blur = blur_plane(&plane, w, h, sigma, wrap);
                for i in 0..plane.len() {
                    let want = (1.0 + amount) * plane[i] - amount * blur[i];
                    assert!(
                        (got[i] - want).abs() < 1.0e-6,
                        "i={i} {} vs {}",
                        got[i],
                        want
                    );
                }
            }
        }
    }

    #[test]
    fn zero_amount_or_sigma_is_identity() {
        let (w, h) = (6u32, 4u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| (i as f32) * 0.011).collect();
        assert_eq!(
            unsharp_mask_plane(&plane, w, h, 2.0, 0.0, WrapMode::Repeat),
            plane
        );
        assert_eq!(
            unsharp_mask_plane(&plane, w, h, 2.0, -1.0, WrapMode::Repeat),
            plane
        );
        assert_eq!(
            unsharp_mask_plane(&plane, w, h, 0.0, 1.0, WrapMode::Repeat),
            plane
        );
    }

    #[test]
    fn constant_plane_is_preserved() {
        let (w, h) = (7u32, 5u32);
        let plane = vec![0.37f32; (w * h) as usize];
        for &wrap in &WRAPS {
            for &amount in &[0.5f32, 2.0] {
                let out = unsharp_mask_plane(&plane, w, h, 2.0, amount, wrap);
                for v in out {
                    assert!((v - 0.37).abs() < 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn affine_ramp_is_preserved_in_interior() {
        // A symmetric unit-sum kernel reproduces an affine signal exactly, so
        // src - blur = 0 and the sharpener is the identity away from the
        // border (where wrap breaks linearity). Use a 1-D ramp and a radius
        // that stays inside the interior sample set.
        let w = 40u32;
        let plane: Vec<f32> = (0..w).map(|x| 0.02 * x as f32).collect();
        let sigma = 2.0f32;
        let out = unsharp_mask_plane(&plane, w, 1, sigma, 1.5, WrapMode::ClampToEdge);
        let r = (sigma * 3.0).ceil() as usize;
        for x in r..(w as usize - r) {
            assert!(
                (out[x] - plane[x]).abs() < 1.0e-5,
                "x={x} {} vs {}",
                out[x],
                plane[x]
            );
        }
    }

    #[test]
    fn step_edge_overshoots_the_input_range() {
        // The negative blur lobe makes the output exceed the input range at an
        // edge -- the sharpening halo. (Plane math is unclamped.)
        let w = 24u32;
        let plane: Vec<f32> = (0..w).map(|x| if x < w / 2 { 0.2 } else { 0.8 }).collect();
        let lo = 0.2f32;
        let hi = 0.8f32;
        let out = unsharp_mask_plane(&plane, w, 1, 2.0, 1.5, WrapMode::ClampToEdge);
        let omin = out.iter().copied().fold(f32::INFINITY, f32::min);
        let omax = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(omin < lo - 1.0e-3, "undershoot halo: {omin}");
        assert!(omax > hi + 1.0e-3, "overshoot halo: {omax}");
    }

    #[test]
    fn image_constant_is_preserved() {
        let texels = vec![[90u8, 140, 200, 255]; 6 * 4];
        let img = Rgba8Image::new(6, 4, texels).unwrap();
        for &space in &[ColorSpace::Linear, ColorSpace::Srgb] {
            let out = unsharp_mask(&img, 2.0, 1.5, WrapMode::Repeat, space).unwrap();
            for (a, b) in out.as_slice().iter().zip(img.as_slice().iter()) {
                assert_eq!(a, b, "constant image preserved");
            }
        }
    }

    #[test]
    fn image_zero_amount_is_identity_copy() {
        let texels: Vec<[u8; 4]> = (0..6 * 4)
            .map(|i| [i as u8 * 3, i as u8 * 5, i as u8 * 7, 255])
            .collect();
        let img = Rgba8Image::new(6, 4, texels).unwrap();
        let a = unsharp_mask(&img, 2.0, 0.0, WrapMode::Repeat, ColorSpace::Linear).unwrap();
        let b = unsharp_mask(&img, 0.0, 1.5, WrapMode::Repeat, ColorSpace::Linear).unwrap();
        assert_eq!(a.as_slice(), img.as_slice());
        assert_eq!(b.as_slice(), img.as_slice());
    }
}
