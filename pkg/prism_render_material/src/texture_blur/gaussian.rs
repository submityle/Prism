//! Native-resolution separable Gaussian blur for decoded `RGBA8` images.
//!
//! Unlike the Gaussian *mip reducer*
//! ([`gaussian_downsample`](crate::gaussian_downsample)), which always halves a
//! dimension, this module blurs an image **in place at its own resolution** for
//! an arbitrary standard deviation. That is the fixed-function primitive behind
//! a long list of AAA post / authoring passes: bloom / glare pyramids, SSAO and
//! soft-shadow denoise, screen-space subsurface diffusion, coverage / mask
//! softening, and roughness / height prefiltering where a resolution change is
//! not wanted.
//!
//! The blur is a classic **separable, gamma-correct, normalised** Gaussian
//! (Heckbert 1986): colour is lifted once to scene-linear `f32`, convolved
//! along X then along Y with the same 1-D kernel, and re-encoded once. The
//! kernel is the sampled Gaussian `g(k) = exp(-k^2 / (2 sigma^2))` on
//! `|k| <= ceil(3 sigma)`, normalised so the taps sum to `1`.
//!
//! Properties this guarantees (all covered by the oracles below):
//! * the weights are **strictly non-negative and sum to 1**, so the output is a
//!   convex combination of the input -- a constant image is preserved exactly
//!   and the result can never over- or under-shoot the input range (no
//!   ringing), unlike a windowed-sinc blur;
//! * the kernel is **symmetric**, so a periodic cosine is attenuated by the
//!   kernel's real DFT magnitude with **zero phase shift** -- the direct link
//!   between the sampled kernel and its frequency response;
//! * border taps are resolved through a [`WrapMode`] so a tiling source stays
//!   seamless ([`WrapMode::Repeat`]) while a clamped image does not bleed its
//!   edge.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a CPU
//! golden matches a GPU compute blur to floating-point tolerance.
//!
//! # References
//! * Heckbert, "Filtering by Repeated Integration" (1986) and
//!   *Fundamentals of Texture Mapping and Image Warping* (1989).
//! * `DirectXTex` / `NVIDIA` Texture Tools separable Gaussian blur.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Half-support of the sampled Gaussian, in texels: taps beyond `3 sigma` have
/// decayed below ~1% and are dropped.
#[inline]
#[must_use]
fn radius_for(sigma: f32) -> usize {
    (sigma * 3.0).ceil().max(0.0) as usize
}

/// Normalised, symmetric 1-D Gaussian kernel of length `2*radius + 1` (index 0
/// is the centre tap, index `k` and `2*radius - k` are the `+/-` pair).
///
/// Returns a single unit tap for a non-positive `sigma` (the identity kernel),
/// so the blur degrades gracefully to a copy.
#[must_use]
pub fn gaussian_weights_1d(sigma: f32) -> Vec<f32> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return vec![1.0];
    }
    let r = radius_for(sigma) as i64;
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut w: Vec<f32> = (-r..=r)
        .map(|k| {
            let kf = k as f32;
            ops::exp(-kf * kf * inv2s2)
        })
        .collect();
    let sum: f32 = w.iter().sum();
    let inv = 1.0 / sum;
    for x in &mut w {
        *x *= inv;
    }
    w
}

/// Wrap an integer tap index into `[0, n)`.
///
/// `Repeat` and `MirroredRepeat` keep tiling / mirrored sources seamless; every
/// other mode collapses to edge clamp.
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

/// Convolve a single-channel `width * height` row-major plane with a separable
/// Gaussian of standard deviation `sigma`, resolving borders through `wrap`.
///
/// Returns a copy when `sigma <= 0` (identity kernel) or either dimension is
/// `0`.
#[must_use]
pub fn blur_plane(plane: &[f32], width: u32, height: u32, sigma: f32, wrap: WrapMode) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h {
        return plane.to_vec();
    }
    let weights = gaussian_weights_1d(sigma);
    if weights.len() == 1 {
        return plane.to_vec();
    }
    let r = (weights.len() / 2) as i64;
    let wi = width as i64;
    let hi = height as i64;

    // X pass: plane -> tmp.
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &wt) in weights.iter().enumerate() {
                let sx = wrap_index(x as i64 + (ki as i64 - r), wi, wrap);
                acc += wt * plane[row + sx];
            }
            tmp[row + x] = acc;
        }
    }

    // Y pass: tmp -> out.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &wt) in weights.iter().enumerate() {
                let sy = wrap_index(y as i64 + (ki as i64 - r), hi, wrap);
                acc += wt * tmp[sy * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Blur an `RGBA8` image in place at its own resolution with a separable
/// Gaussian of standard deviation `sigma` (in texels), gamma-correctly under
/// `space`, resolving borders through `wrap`.
///
/// Colour channels are filtered in scene-linear light; alpha is filtered
/// linearly (`/255`). Returns an exact copy when `sigma <= 0`.
#[must_use]
pub fn gaussian_blur(
    src: &Rgba8Image,
    sigma: f32,
    wrap: WrapMode,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return None;
    }
    if !sigma.is_finite() || sigma <= 0.0 {
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

    let r = blur_plane(&r, w, h, sigma, wrap);
    let g = blur_plane(&g, w, h, sigma, wrap);
    let b = blur_plane(&b, w, h, sigma, wrap);
    let a = blur_plane(&a, w, h, sigma, wrap);

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
    fn weights_are_normalised_nonneg_and_symmetric() {
        for &sigma in &[0.5f32, 1.0, 2.5, 5.0] {
            let w = gaussian_weights_1d(sigma);
            assert_eq!(w.len() % 2, 1, "odd length");
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-6, "sum {sum}");
            let n = w.len();
            for (k, &wk) in w.iter().enumerate() {
                assert!(wk >= 0.0, "nonneg");
                assert!((wk - w[n - 1 - k]).abs() < 1.0e-7, "symmetric at {k}");
            }
            // Centre tap is the maximum of a Gaussian.
            let mid = n / 2;
            for &wk in &w {
                assert!(wk <= w[mid] + 1.0e-7);
            }
        }
    }

    #[test]
    fn zero_or_negative_sigma_is_single_tap() {
        for &s in &[0.0f32, -1.0] {
            assert_eq!(gaussian_weights_1d(s), vec![1.0]);
        }
    }

    #[test]
    fn constant_plane_is_preserved() {
        let (w, h) = (7u32, 5u32);
        let plane = vec![0.37f32; (w * h) as usize];
        for &wrap in &WRAPS {
            let out = blur_plane(&plane, w, h, 2.0, wrap);
            for v in out {
                assert!((v - 0.37).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn zero_sigma_is_identity() {
        let (w, h) = (6u32, 4u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| (i as f32) * 0.013).collect();
        let out = blur_plane(&plane, w, h, 0.0, WrapMode::Repeat);
        assert_eq!(out, plane);
    }

    #[test]
    fn blur_never_overshoots_the_input_range() {
        // Convex combination of non-negative, unit-sum weights stays in range.
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
            for &sigma in &[0.8f32, 2.0, 4.0] {
                let out = blur_plane(&plane, w, h, sigma, wrap);
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
    fn frequency_response_matches_kernel_dft_on_a_periodic_cosine() {
        // Under Repeat wrap a symmetric kernel attenuates an integer-frequency
        // cosine by its real DFT magnitude with NO phase shift:
        //   blur(0.5 + A cos(2pi f x / N)) = 0.5 + A*H(f) cos(2pi f x / N),
        // H(f) = sum_k w[k] cos(2pi f k / N).  DC (0.5) is preserved (sum=1).
        let n = 16i64;
        let f = 2.0f32;
        let amp = 0.4f32;
        let sigma = 1.5f32;
        let plane: Vec<f32> = (0..n)
            .map(|x| 0.5 + amp * ops::cos(core::f32::consts::TAU * f * x as f32 / n as f32))
            .collect();
        let out = blur_plane(&plane, n as u32, 1, sigma, WrapMode::Repeat);

        let w = gaussian_weights_1d(sigma);
        let r = (w.len() / 2) as i64;
        let hf: f32 = w
            .iter()
            .enumerate()
            .map(|(ki, &wk)| {
                let k = ki as i64 - r;
                wk * ops::cos(core::f32::consts::TAU * f * k as f32 / n as f32)
            })
            .sum();
        for x in 0..n {
            let expect =
                0.5 + amp * hf * ops::cos(core::f32::consts::TAU * f * x as f32 / n as f32);
            assert!(
                (out[x as usize] - expect).abs() < 1.0e-5,
                "x={x} {} vs {expect}",
                out[x as usize]
            );
        }
        // A real low-pass attenuates (does not amplify) a non-DC frequency.
        assert!(hf < 1.0 && hf > 0.0, "hf={hf}");
    }

    #[test]
    fn horizontal_mirror_commutes_under_clamp() {
        let (w, h) = (8u32, 5u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = (x * x + 2 * y) as f32 * 0.02;
            }
        }
        let mut mirrored = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                mirrored[(y * w + x) as usize] = plane[(y * w + (w - 1 - x)) as usize];
            }
        }
        let a = blur_plane(&plane, w, h, 1.7, WrapMode::ClampToEdge);
        let bm = blur_plane(&mirrored, w, h, 1.7, WrapMode::ClampToEdge);
        for y in 0..h {
            for x in 0..w {
                let lhs = a[(y * w + (w - 1 - x)) as usize];
                let rhs = bm[(y * w + x) as usize];
                assert!((lhs - rhs).abs() < 1.0e-6, "x={x} y={y} {lhs} vs {rhs}");
            }
        }
    }

    #[test]
    fn separable_blur_factorises_a_separable_signal() {
        // For a rank-1 field p(x,y)=a(x)*b(y), the separable blur equals the
        // 1-D blur of each factor multiplied together: (Ba)(x) * (Bb)(y).
        let (w, h) = (12u32, 9u32);
        let ax: Vec<f32> = (0..w).map(|x| 0.2 + 0.5 * (x as f32 / w as f32)).collect();
        let by: Vec<f32> = (0..h).map(|y| 0.1 + ((y * 3 % 5) as f32) * 0.1).collect();
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = ax[x as usize] * by[y as usize];
            }
        }
        let sigma = 1.3f32;
        let wrap = WrapMode::Repeat;
        let blurred = blur_plane(&plane, w, h, sigma, wrap);
        let ba = blur_plane(&ax, w, 1, sigma, wrap);
        let bb = blur_plane(&by, 1, h, sigma, wrap);
        for y in 0..h {
            for x in 0..w {
                let expect = ba[x as usize] * bb[y as usize];
                let got = blurred[(y * w + x) as usize];
                assert!(
                    (got - expect).abs() < 1.0e-5,
                    "x={x} y={y} {got} vs {expect}"
                );
            }
        }
    }

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Rgba8Image {
        Rgba8Image::new(w, h, vec![c; (w * h) as usize]).unwrap()
    }

    #[test]
    fn constant_image_is_preserved_in_both_spaces() {
        for space in [ColorSpace::Linear, ColorSpace::Srgb] {
            let img = solid(6, 6, [40, 160, 220, 255]);
            let out = gaussian_blur(&img, 2.5, WrapMode::Repeat, space).unwrap();
            assert_eq!(out.as_slice(), img.as_slice(), "{space:?}");
        }
    }

    #[test]
    fn zero_sigma_image_is_identity() {
        let mut t = Vec::new();
        for i in 0..(5 * 4) {
            t.push([(i * 7) as u8, (i * 3) as u8, (i * 11) as u8, 255]);
        }
        let img = Rgba8Image::new(5, 4, t).unwrap();
        let out = gaussian_blur(&img, 0.0, WrapMode::ClampToEdge, ColorSpace::Srgb).unwrap();
        assert_eq!(out.as_slice(), img.as_slice());
    }
}
