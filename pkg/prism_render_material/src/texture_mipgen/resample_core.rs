//! Shared scene-linear working-buffer core for separable 2x mip reduction.
//!
//! Both the Lanczos ([`windowed`](super::windowed)) and Kaiser
//! ([`kaiser`](super::kaiser)) reducers down-sample by exactly 2 per axis in
//! scene-linear light. The colour lift/encode and the separable clamp-to-edge
//! convolution are identical between them, so this module factors them out and
//! each filter file only declares its own 1-D kernel. This keeps the kernels
//! independently testable and avoids duplicating the gamma-correct resample
//! loop. Everything here is deterministic analytic arithmetic -- no AI/ML.
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule).
//! * Filtering happens on an `f32` scene-linear working buffer so colour is
//!   reconstructed once and re-encoded once; alpha is always linear coverage.
//! * Kernel taps are evaluated at the *true* (unclamped) source position so the
//!   kernel stays symmetric; only the fetch index is clamped to the edge, which
//!   preserves constants (edge-clamp of a constant is the constant).
//! * The kernel closure takes the normalised distance `(s - center) / 2` and
//!   must return `0` outside its support `radius`.

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::srgb::{linear_to_srgb, srgb_to_linear};
use bevy_math::ops;

/// Normalised sinc, `sin(pi x) / (pi x)` with the removable singularity at 0.
#[must_use]
pub(crate) fn sinc(x: f32) -> f32 {
    if x.abs() < 1.0e-8 {
        1.0
    } else {
        let p = PI * x;
        ops::sin(p) / p
    }
}

/// A dimension can be halved only when it is `1` (carried through) or even.
#[inline]
#[must_use]
pub(crate) fn reducible_dim(dim: u32) -> bool {
    dim == 1 || dim.is_multiple_of(2)
}

/// Lift an image into a scene-linear `f32` working buffer (RGB per `space`,
/// alpha always `/255`).
fn to_working(img: &Rgba8Image, space: ColorSpace) -> Vec<[f32; 4]> {
    img.as_slice()
        .iter()
        .map(|t| {
            let a = f32::from(t[3]) / 255.0;
            match space {
                ColorSpace::Linear => [
                    f32::from(t[0]) / 255.0,
                    f32::from(t[1]) / 255.0,
                    f32::from(t[2]) / 255.0,
                    a,
                ],
                ColorSpace::Srgb => {
                    [srgb_to_linear(t[0]), srgb_to_linear(t[1]), srgb_to_linear(t[2]), a]
                }
            }
        })
        .collect()
}

/// Encode a working buffer back to `RGBA8` under `space`.
fn from_working(buf: &[[f32; 4]], w: u32, h: u32, space: ColorSpace) -> Option<Rgba8Image> {
    let round_u8 = |v: f32| -> u8 {
        let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        (c * 255.0 + 0.5).floor() as u8
    };
    let texels: Vec<[u8; 4]> = buf
        .iter()
        .map(|p| {
            let a = round_u8(p[3]);
            match space {
                ColorSpace::Linear => [round_u8(p[0]), round_u8(p[1]), round_u8(p[2]), a],
                ColorSpace::Srgb => {
                    [linear_to_srgb(p[0]), linear_to_srgb(p[1]), linear_to_srgb(p[2]), a]
                }
            }
        })
        .collect();
    Rgba8Image::new(w, h, texels)
}

/// Resample the X axis of `buf` (`w x h`) to `w/2` with `kernel` of `radius`.
fn downsample_x<K: Fn(f32) -> f32>(
    buf: &[[f32; 4]],
    w: u32,
    h: u32,
    radius: f32,
    kernel: &K,
) -> (Vec<[f32; 4]>, u32) {
    let out_w = w / 2;
    let scale = 2.0f32;
    let support = radius * scale;
    let mut out = vec![[0.0f32; 4]; (out_w as usize) * (h as usize)];
    for y in 0..h {
        let row = (y as usize) * (w as usize);
        for ox in 0..out_w {
            let center = ox as f32 * 2.0 + 0.5;
            let lo = (center - support).floor() as i64;
            let hi = (center + support).ceil() as i64;
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for s in lo..=hi {
                let wt = kernel((s as f32 - center) / scale);
                if wt == 0.0 {
                    continue;
                }
                let cs = s.clamp(0, w as i64 - 1) as usize;
                let t = buf[row + cs];
                for c in 0..4 {
                    acc[c] += wt * t[c];
                }
                wsum += wt;
            }
            let inv = if wsum.abs() > 1.0e-12 { 1.0 / wsum } else { 0.0 };
            let o = (y as usize) * (out_w as usize) + ox as usize;
            out[o] = [acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv];
        }
    }
    (out, out_w)
}

/// Resample the Y axis of `buf` (`w x h`) to `h/2` with `kernel` of `radius`.
fn downsample_y<K: Fn(f32) -> f32>(
    buf: &[[f32; 4]],
    w: u32,
    h: u32,
    radius: f32,
    kernel: &K,
) -> (Vec<[f32; 4]>, u32) {
    let out_h = h / 2;
    let scale = 2.0f32;
    let support = radius * scale;
    let mut out = vec![[0.0f32; 4]; (w as usize) * (out_h as usize)];
    for oy in 0..out_h {
        let center = oy as f32 * 2.0 + 0.5;
        let lo = (center - support).floor() as i64;
        let hi = (center + support).ceil() as i64;
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for s in lo..=hi {
                let wt = kernel((s as f32 - center) / scale);
                if wt == 0.0 {
                    continue;
                }
                let cs = s.clamp(0, h as i64 - 1) as usize;
                let t = buf[cs * (w as usize) + x as usize];
                for c in 0..4 {
                    acc[c] += wt * t[c];
                }
                wsum += wt;
            }
            let inv = if wsum.abs() > 1.0e-12 { 1.0 / wsum } else { 0.0 };
            let o = (oy as usize) * (w as usize) + x as usize;
            out[o] = [acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv];
        }
    }
    (out, out_h)
}

/// Produce the next mip level by a separable 2x reduction under `kernel`
/// (defined on the normalised distance domain with the given `radius`), or
/// `None` when `src` is already `1x1` or has a non-reducible (odd, >1)
/// dimension.
#[must_use]
pub(crate) fn downsample_2x<K: Fn(f32) -> f32>(
    src: &Rgba8Image,
    space: ColorSpace,
    radius: f32,
    kernel: K,
) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 1 && h == 1 {
        return None;
    }
    if !reducible_dim(w) || !reducible_dim(h) {
        return None;
    }

    let mut buf = to_working(src, space);
    let (mut cw, mut ch) = (w, h);
    if cw > 1 {
        let (nb, nw) = downsample_x(&buf, cw, ch, radius, &kernel);
        buf = nb;
        cw = nw;
    }
    if ch > 1 {
        let (nb, nh) = downsample_y(&buf, cw, ch, radius, &kernel);
        buf = nb;
        ch = nh;
    }
    from_working(&buf, cw, ch, space)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_identities() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-6);
        assert!(sinc(1.0).abs() < 1e-6);
        assert!(sinc(2.0).abs() < 1e-6);
        assert!((sinc(0.3) - sinc(-0.3)).abs() < 1e-6);
    }

    #[test]
    fn reducible_dim_rule() {
        assert!(reducible_dim(1));
        assert!(reducible_dim(2));
        assert!(reducible_dim(8));
        assert!(!reducible_dim(3));
        assert!(!reducible_dim(5));
    }
}
