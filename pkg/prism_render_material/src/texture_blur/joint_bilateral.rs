//! Native-resolution **joint / cross bilateral** filter for decoded `RGBA8`
//! images.
//!
//! A [bilateral blur](super::bilateral) derives its edge-stopping range weight
//! from the image it is smoothing. The *joint* (a.k.a. *cross*) bilateral
//! filter decouples the two: the spatial-and-range kernel is built from a
//! separate **guide** signal, and that kernel is applied to the data plane.
//! This is the fixed-function primitive behind guided edge-aware upsampling
//! (Petschnigg et al. 2004; the flash / no-flash pair), joint denoise of a
//! noisy channel steered by a clean albedo / normal / depth G-buffer, and
//! detail transfer where the structure to preserve lives in a different buffer
//! than the values being filtered.
//!
//! For every output texel the weight of neighbour `t` is
//!
//! ```text
//! w(t) = exp(-(dx^2 + dy^2) / (2 sigma_spatial^2))
//!      * exp(-(guide_center - guide_t)^2 / (2 sigma_range^2))
//! ```
//!
//! and the data plane is accumulated with those weights and renormalised, so
//! the result is a **convex combination** of the data: non-negative weights
//! summing to `1`, hence the data range is never exceeded and a constant data
//! plane is preserved for *any* guide.
//!
//! Properties the oracles below pin down:
//! * when the guide **equals** the data plane the filter is exactly the plain
//!   [`bilateral_blur_plane`](super::bilateral_blur_plane) -- a direct tie to
//!   already-verified code;
//! * as `sigma_range -> infinity` the range weight -> `1` and the filter
//!   converges to the separable Gaussian [`blur_plane`](super::blur_plane),
//!   independent of the guide;
//! * a constant data plane is preserved exactly for any guide, and the output
//!   never leaves the data range;
//! * a non-positive / non-finite sigma, a zero dimension, or a guide whose
//!   length does not match the data plane is the **identity** copy.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a
//! CPU golden matches a GPU compute joint bilateral to floating-point
//! tolerance.
//!
//! # References
//! * Petschnigg, Szeliski, Agrawala, Cohen, Hoppe & Toyama, "Digital
//!   Photography with Flash and No-Flash Image Pairs" (2004).
//! * Eisemann & Durand, "Flash Photography Enhancement via Intrinsic
//!   Relighting" (2004).
//! * Kopf, Cohen, Lischinski & Uyttendaele, "Joint Bilateral Upsampling"
//!   (2007).

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Half-support of the spatial Gaussian, in texels. Matches
/// `gaussian::radius_for` / `bilateral::radius_for`.
#[inline]
#[must_use]
fn radius_for(sigma: f32) -> usize {
    (sigma * 3.0).ceil().max(0.0) as usize
}

/// Wrap an integer tap index into `[0, n)` (local copy of the shared helper so
/// this module stays self-contained).
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

/// Joint / cross bilateral filter of a single-channel `width * height` row-major
/// data `plane`, with the edge-stopping range weight derived from a separate
/// `guide` plane of the same dimensions: spatial standard deviation
/// `sigma_spatial` (texels) and range standard deviation `sigma_range` (same
/// units as `guide`), resolving borders through `wrap`.
///
/// Returns a copy of `plane` when either sigma is non-positive / non-finite,
/// either dimension is `0`, or either slice length does not match
/// `width * height` (the identity, so the filter degrades gracefully).
#[must_use]
pub fn joint_bilateral_blur_plane(
    plane: &[f32],
    guide: &[f32],
    width: u32,
    height: u32,
    sigma_spatial: f32,
    sigma_range: f32,
    wrap: WrapMode,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h || guide.len() != w * h {
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

    let spatial: Vec<f32> = (-r..=r)
        .map(|d| {
            let df = d as f32;
            ops::exp(-df * df * inv2s2)
        })
        .collect();

    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let guide_center = guide[y * w + x];
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for (dyi, &sy_w) in spatial.iter().enumerate() {
                let dy = dyi as i64 - r;
                let sy = wrap_index(y as i64 + dy, hi, wrap);
                let srow = sy * w;
                for (dxi, &sx_w) in spatial.iter().enumerate() {
                    let dx = dxi as i64 - r;
                    let sx = wrap_index(x as i64 + dx, wi, wrap);
                    let diff = guide[srow + sx] - guide_center;
                    let rw = ops::exp(-diff * diff * inv2r2);
                    let weight = sy_w * sx_w * rw;
                    acc += weight * plane[srow + sx];
                    wsum += weight;
                }
            }
            // `wsum` includes the centre tap (weight 1), so it is always >= 1.
            out[y * w + x] = acc / wsum;
        }
    }
    out
}

/// Joint / cross bilateral filter of an `RGBA8` `src` image steered by a
/// separate `guide` image of the same dimensions: spatial standard deviation
/// `sigma_spatial` (texels) and range standard deviation `sigma_range` (in
/// scene-linear `[0, 1]` units), gamma-correctly under `space`, resolving
/// borders through `wrap`.
///
/// Each colour channel is filtered in scene-linear light steered by the
/// matching scene-linear guide channel; alpha is filtered linearly (`/255`)
/// steered by the guide alpha. Returns `None` on a zero dimension or a guide
/// whose dimensions do not match `src`, and an exact copy of `src` when either
/// sigma is non-positive.
#[must_use]
pub fn joint_bilateral_blur(
    src: &Rgba8Image,
    guide: &Rgba8Image,
    sigma_spatial: f32,
    sigma_range: f32,
    wrap: WrapMode,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 || guide.width() != w || guide.height() != h {
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
    let lift = |texels: &[[u8; 4]]| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = vec![0.0f32; n];
        let mut g = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        let mut a = vec![0.0f32; n];
        for (i, t) in texels.iter().enumerate() {
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
        (r, g, b, a)
    };

    let (sr, sg, sb, sa) = lift(src.as_slice());
    let (gr, gg, gb, ga) = lift(guide.as_slice());

    let r = joint_bilateral_blur_plane(&sr, &gr, w, h, sigma_spatial, sigma_range, wrap);
    let g = joint_bilateral_blur_plane(&sg, &gg, w, h, sigma_spatial, sigma_range, wrap);
    let b = joint_bilateral_blur_plane(&sb, &gb, w, h, sigma_spatial, sigma_range, wrap);
    let a = joint_bilateral_blur_plane(&sa, &ga, w, h, sigma_spatial, sigma_range, wrap);

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

    fn ramp_plane(w: u32, h: u32) -> Vec<f32> {
        (0..(w * h)).map(|i| (i as f32) * 0.013 + 0.05).collect()
    }

    #[test]
    fn equals_plain_bilateral_when_guide_is_the_data() {
        // With guide == plane the joint filter is exactly the plain bilateral.
        let (w, h) = (10u32, 8u32);
        let plane = ramp_plane(w, h);
        for &wrap in &WRAPS {
            for &sr in &[0.1f32, 0.5, 2.0] {
                let joint = joint_bilateral_blur_plane(&plane, &plane, w, h, 1.5, sr, wrap);
                let plain = super::super::bilateral_blur_plane(&plane, w, h, 1.5, sr, wrap);
                for (a, b) in joint.iter().zip(plain.iter()) {
                    assert!((a - b).abs() < 1.0e-6, "{a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn converges_to_separable_gaussian_as_range_sigma_grows() {
        // As sigma_range -> infinity the guide no longer matters and the filter
        // becomes the plain separable Gaussian.
        let (w, h) = (12u32, 10u32);
        let plane = ramp_plane(w, h);
        // An arbitrary, structured guide to prove it is ignored at huge sigma_r.
        let guide: Vec<f32> = (0..(w * h))
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let bil = joint_bilateral_blur_plane(&plane, &guide, w, h, 1.5, 1.0e6, WrapMode::Repeat);
        let gauss = super::super::blur_plane(&plane, w, h, 1.5, WrapMode::Repeat);
        for (a, b) in bil.iter().zip(gauss.iter()) {
            assert!((a - b).abs() < 1.0e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn constant_data_is_preserved_for_any_guide() {
        let (w, h) = (7u32, 5u32);
        let plane = vec![0.37f32; (w * h) as usize];
        let guide: Vec<f32> = (0..(w * h)).map(|i| ((i * 7) % 11) as f32 * 0.1).collect();
        for &wrap in &WRAPS {
            let out = joint_bilateral_blur_plane(&plane, &guide, w, h, 2.0, 0.1, wrap);
            for v in out {
                assert!((v - 0.37).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn output_stays_within_data_range() {
        let (w, h) = (9u32, 9u32);
        let plane = ramp_plane(w, h);
        let guide: Vec<f32> = (0..(w * h)).map(|i| ((i * 7) % 5) as f32 * 0.2).collect();
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &wrap in &WRAPS {
            for &sr in &[0.05f32, 0.3, 2.0] {
                let out = joint_bilateral_blur_plane(&plane, &guide, w, h, 2.0, sr, wrap);
                for v in out {
                    assert!(v >= lo - 1.0e-6 && v <= hi + 1.0e-6, "v={v} [{lo},{hi}]");
                }
            }
        }
    }

    #[test]
    fn nonpositive_sigma_or_mismatched_guide_is_identity() {
        let (w, h) = (6u32, 4u32);
        let plane = ramp_plane(w, h);
        let guide = vec![0.5f32; (w * h) as usize];
        assert_eq!(
            joint_bilateral_blur_plane(&plane, &guide, w, h, 0.0, 0.5, WrapMode::Repeat),
            plane
        );
        assert_eq!(
            joint_bilateral_blur_plane(&plane, &guide, w, h, 2.0, 0.0, WrapMode::Repeat),
            plane
        );
        // Guide length mismatch -> identity copy.
        let short_guide = vec![0.5f32; (w * h) as usize - 1];
        assert_eq!(
            joint_bilateral_blur_plane(&plane, &short_guide, w, h, 2.0, 0.5, WrapMode::Repeat),
            plane
        );
    }

    #[test]
    fn guide_edge_steers_the_filter_away_from_a_plain_gaussian() {
        // A smooth data ramp with a step guide: a small range sigma suppresses
        // taps that cross the guide edge, so the result differs from the plain
        // Gaussian that ignores the guide.
        let w = 24u32;
        let plane: Vec<f32> = (0..w).map(|x| 0.02 * x as f32).collect();
        let guide: Vec<f32> = (0..w).map(|x| if x < w / 2 { 0.0 } else { 1.0 }).collect();
        let joint =
            joint_bilateral_blur_plane(&plane, &guide, w, 1, 2.0, 0.05, WrapMode::ClampToEdge);
        let gauss = super::super::blur_plane(&plane, w, 1, 2.0, WrapMode::ClampToEdge);
        let max_diff = joint
            .iter()
            .zip(gauss.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1.0e-3,
            "guide should steer the kernel: {max_diff}"
        );
    }

    #[test]
    fn image_equals_plain_bilateral_when_guide_is_the_source() {
        let texels: Vec<[u8; 4]> = (0..6 * 4)
            .map(|i| [i as u8 * 3, i as u8 * 5, 200 - i as u8 * 2, 255])
            .collect();
        let img = Rgba8Image::new(6, 4, texels).unwrap();
        for &space in &[ColorSpace::Linear, ColorSpace::Srgb] {
            let joint =
                joint_bilateral_blur(&img, &img, 1.5, 0.2, WrapMode::Repeat, space).unwrap();
            let plain =
                super::super::bilateral_blur(&img, 1.5, 0.2, WrapMode::Repeat, space).unwrap();
            assert_eq!(joint.as_slice(), plain.as_slice());
        }
    }

    #[test]
    fn image_rejects_mismatched_guide_dimensions() {
        let a = Rgba8Image::new(6, 4, vec![[10u8, 20, 30, 255]; 6 * 4]).unwrap();
        let b = Rgba8Image::new(5, 4, vec![[10u8, 20, 30, 255]; 5 * 4]).unwrap();
        assert!(
            joint_bilateral_blur(&a, &b, 1.5, 0.2, WrapMode::Repeat, ColorSpace::Linear).is_none()
        );
    }
}
