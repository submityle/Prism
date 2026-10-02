//! Native-resolution separable box blur for decoded `RGBA8` images.
//!
//! The box blur is the cheapest low-pass filter: an unweighted average over a
//! `(2*radius + 1)` window per axis. It is the workhorse behind large-radius
//! real-time blurs -- summed-area / variance shadow map prefiltering, cheap
//! depth-of-field and bloom, and the classic "three box passes approximate a
//! Gaussian" trick (central-limit theorem) -- because a running sum makes it
//! `O(width)` per row regardless of radius.
//!
//! This module implements that running-sum (sliding window) path on the shared
//! gamma-correct working-buffer policy. Because every tap weight is
//! `1 / (2*radius + 1) >= 0` and the weights sum to `1`, the output is a convex
//! combination of the input: a constant image is preserved exactly and the
//! result can never over- or under-shoot the input range. Border taps resolve
//! through a [`WrapMode`] exactly as the Gaussian blur
//! ([`gaussian_blur`](super::gaussian_blur)) does.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a CPU
//! golden matches a GPU compute blur to floating-point tolerance.
//!
//! # References
//! * Crow, "Summed-Area Tables for Texture Mapping" (1984).
//! * Kraus & Strengert, "Pyramid Filters Based on Bilinear Interpolation"
//!   (repeated box blur as a Gaussian approximation).

use alloc::vec;
use alloc::vec::Vec;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Wrap an integer tap index into `[0, n)` (same convention as the Gaussian
/// blur): `Repeat` / `MirroredRepeat` tile, every other mode clamps to the
/// edge.
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

/// Blur one length-`n` line (stride `step`, starting at `base`) of `plane` in
/// place-free form into `out` with a `(2*radius + 1)`-tap box via an
/// incremental running sum.
fn box_blur_line(
    plane: &[f32],
    out: &mut [f32],
    base: usize,
    step: usize,
    n: i64,
    radius: i64,
    wrap: WrapMode,
) {
    let m = (2 * radius + 1) as f32;
    let inv = 1.0 / m;
    let fetch = |x: i64| -> f32 { plane[base + wrap_index(x, n, wrap) * step] };
    // Seed the window for output index 0: taps [-radius, radius].
    let mut sum = 0.0f32;
    for j in -radius..=radius {
        sum += fetch(j);
    }
    out[base] = sum * inv;
    for x in 1..n {
        // Slide: drop the tap that left the window, add the one that entered.
        sum += fetch(x + radius) - fetch(x - 1 - radius);
        out[base + x as usize * step] = sum * inv;
    }
}

/// Convolve a single-channel `width * height` row-major plane with a separable
/// box blur of the given per-axis `radius`, resolving borders through `wrap`.
///
/// Returns a copy when `radius == 0` or either dimension is `0`.
#[must_use]
pub fn box_blur_plane(
    plane: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    wrap: WrapMode,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || plane.len() != w * h || radius == 0 {
        return plane.to_vec();
    }
    let r = radius as i64;

    // X pass: plane -> tmp (each row is contiguous, step 1).
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        box_blur_line(plane, &mut tmp, y * w, 1, width as i64, r, wrap);
    }
    // Y pass: tmp -> out (each column has stride w).
    let mut out = vec![0.0f32; w * h];
    for x in 0..w {
        box_blur_line(&tmp, &mut out, x, w, height as i64, r, wrap);
    }
    out
}

/// Blur an `RGBA8` image in place at its own resolution with a separable box
/// blur of the given `radius` (in texels), gamma-correctly under `space`,
/// resolving borders through `wrap`.
///
/// Colour channels are filtered in scene-linear light; alpha is filtered
/// linearly (`/255`). Returns an exact copy when `radius == 0`.
#[must_use]
pub fn box_blur(
    src: &Rgba8Image,
    radius: u32,
    wrap: WrapMode,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 0 || h == 0 {
        return None;
    }
    if radius == 0 {
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

    let r = box_blur_plane(&r, w, h, radius, wrap);
    let g = box_blur_plane(&g, w, h, radius, wrap);
    let b = box_blur_plane(&b, w, h, radius, wrap);
    let a = box_blur_plane(&a, w, h, radius, wrap);

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

    /// Independent oracle: a direct, non-incremental windowed average. The
    /// production path slides a running sum (subtract-left / add-right); this
    /// recomputes the whole window from scratch each pixel, so matching it
    /// cross-checks the incremental arithmetic and the wrap indexing.
    fn direct_box(plane: &[f32], w: u32, h: u32, radius: i64, wrap: WrapMode) -> Vec<f32> {
        let (wi, hi) = (w as i64, h as i64);
        let m = (2 * radius + 1) as f32;
        // Separable: X pass then Y pass, each a from-scratch window sum.
        let mut tmp = vec![0.0f32; (w * h) as usize];
        for y in 0..hi {
            for x in 0..wi {
                let mut s = 0.0f32;
                for j in -radius..=radius {
                    s += plane[(y * wi + wrap_index(x + j, wi, wrap) as i64) as usize];
                }
                tmp[(y * wi + x) as usize] = s / m;
            }
        }
        let mut out = vec![0.0f32; (w * h) as usize];
        for y in 0..hi {
            for x in 0..wi {
                let mut s = 0.0f32;
                for j in -radius..=radius {
                    s += tmp[(wrap_index(y + j, hi, wrap) as i64 * wi + x) as usize];
                }
                out[(y * wi + x) as usize] = s / m;
            }
        }
        out
    }

    #[test]
    fn running_sum_matches_direct_window_oracle() {
        let (w, h) = (11u32, 9u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = ((x * 7 + y * 5) % 13) as f32 * 0.07 + 0.05;
            }
        }
        for &wrap in &WRAPS {
            for radius in 1u32..=4 {
                let fast = box_blur_plane(&plane, w, h, radius, wrap);
                let slow = direct_box(&plane, w, h, radius as i64, wrap);
                for (a, b) in fast.iter().zip(slow.iter()) {
                    assert!((a - b).abs() < 1.0e-5, "r={radius} {wrap:?} {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn constant_plane_is_preserved() {
        let (w, h) = (6u32, 7u32);
        let plane = vec![0.42f32; (w * h) as usize];
        for &wrap in &WRAPS {
            for radius in 1u32..=3 {
                for v in box_blur_plane(&plane, w, h, radius, wrap) {
                    assert!((v - 0.42).abs() < 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn zero_radius_is_identity() {
        let (w, h) = (5u32, 5u32);
        let plane: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 0.01).collect();
        assert_eq!(box_blur_plane(&plane, w, h, 0, WrapMode::Repeat), plane);
    }

    #[test]
    fn box_blur_never_overshoots_range() {
        let (w, h) = (10u32, 8u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = ((x + 2 * y) % 5) as f32 * 0.15 + 0.1;
            }
        }
        let lo = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &wrap in &WRAPS {
            for radius in 1u32..=3 {
                for v in box_blur_plane(&plane, w, h, radius, wrap) {
                    assert!(v >= lo - 1.0e-6 && v <= hi + 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn frequency_response_matches_closed_form_dirichlet() {
        // Under Repeat wrap, a box of M = 2r+1 taps attenuates an integer
        // frequency cosine by the (normalised) Dirichlet kernel
        //   H(f) = sin(pi f M / N) / (M sin(pi f / N))      (f not a multiple of N),
        // phase-free because the kernel is symmetric. Verify both that the
        // output is 0.5 + A H cos(...) and that H equals the closed form, which
        // is an analytic identity independent of the running-sum code.
        let n = 20i64;
        let f = 3.0f32;
        let amp = 0.4f32;
        let radius = 2u32;
        let m = (2 * radius + 1) as f32;
        let plane: Vec<f32> = (0..n)
            .map(|x| 0.5 + amp * ops::cos(core::f32::consts::TAU * f * x as f32 / n as f32))
            .collect();
        let out = box_blur_plane(&plane, n as u32, 1, radius, WrapMode::Repeat);
        let pi = core::f32::consts::PI;
        let hf = ops::sin(pi * f * m / n as f32) / (m * ops::sin(pi * f / n as f32));
        for x in 0..n {
            let expect =
                0.5 + amp * hf * ops::cos(core::f32::consts::TAU * f * x as f32 / n as f32);
            assert!(
                (out[x as usize] - expect).abs() < 1.0e-5,
                "x={x} {} vs {expect}",
                out[x as usize]
            );
        }
    }

    #[test]
    fn horizontal_mirror_commutes_under_clamp() {
        let (w, h) = (9u32, 6u32);
        let mut plane = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                plane[(y * w + x) as usize] = (x * 3 + y * y) as f32 * 0.02;
            }
        }
        let mut mirrored = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                mirrored[(y * w + x) as usize] = plane[(y * w + (w - 1 - x)) as usize];
            }
        }
        let a = box_blur_plane(&plane, w, h, 2, WrapMode::ClampToEdge);
        let bm = box_blur_plane(&mirrored, w, h, 2, WrapMode::ClampToEdge);
        for y in 0..h {
            for x in 0..w {
                let lhs = a[(y * w + (w - 1 - x)) as usize];
                let rhs = bm[(y * w + x) as usize];
                assert!((lhs - rhs).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn constant_image_is_preserved_and_zero_radius_is_identity() {
        let img = Rgba8Image::new(5, 5, vec![[33, 120, 200, 255]; 25]).unwrap();
        let out = box_blur(&img, 2, WrapMode::Repeat, ColorSpace::Srgb).unwrap();
        assert_eq!(out.as_slice(), img.as_slice());

        let mut t = Vec::new();
        for i in 0..20 {
            t.push([(i * 11) as u8, (i * 5) as u8, (i * 3) as u8, 255]);
        }
        let img2 = Rgba8Image::new(5, 4, t).unwrap();
        let id = box_blur(&img2, 0, WrapMode::ClampToEdge, ColorSpace::Linear).unwrap();
        assert_eq!(id.as_slice(), img2.as_slice());
    }
}
