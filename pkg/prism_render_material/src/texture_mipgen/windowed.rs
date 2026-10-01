//! Windowed-sinc (Lanczos) separable mip reduction for decoded `RGBA8`.
//!
//! The box filter in [`box_filter`](super::box_filter) is cheap but its boxcar
//! frequency response passes significant energy above the output Nyquist, so
//! AAA texture compressors (`NVIDIA` Texture Tools, `DirectXTex`) down-sample with
//! a windowed-sinc instead to suppress mip shimmering. This module provides a
//! separable Lanczos-2/3 2x reduction over the same [`Rgba8Image`] /
//! [`ColorSpace`] model, keeping the colour path gamma-correct. The kernel is
//! pure analytic trigonometry -- a deterministic CPU golden, no AI/ML.
//!
//! # Conventions
//! * Reduction factor is exactly 2 per axis; a dimension of `1` is carried
//!   through unchanged (GL `max(1, dim >> 1)` rule), matching the box path.
//! * Each pass runs on a scene-linear `f32` working buffer so colour is
//!   filtered in linear light exactly once, then re-encoded (sRGB) at the end;
//!   alpha is always linear coverage. This avoids the double-encode error of
//!   re-quantising between the two separable passes.
//! * Kernel taps are evaluated at the *true* (unclamped) source position so the
//!   kernel stays symmetric; only the fetch index is clamped to the edge, which
//!   preserves constants (edge-clamp of a constant is the constant).
//!
//! # References
//! * Lanczos resampling: windowed sinc `sinc(x) * sinc(x / a)`, support `|x|<a`.
//! * Turkowski, "Filters for Common Resampling Tasks" (1990).

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::srgb::{linear_to_srgb, srgb_to_linear};
use bevy_math::ops;

/// Windowed-sinc kernel selection for the mip reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowedKernel {
    /// Lanczos with `a = 2` (4-tap at unit scale): sharper, slightly more ring.
    Lanczos2,
    /// Lanczos with `a = 3` (6-tap at unit scale): the common texture-tool default.
    Lanczos3,
}

impl WindowedKernel {
    #[must_use]
    fn radius(self) -> f32 {
        match self {
            WindowedKernel::Lanczos2 => 2.0,
            WindowedKernel::Lanczos3 => 3.0,
        }
    }
}

/// Normalised sinc, `sin(pi x) / (pi x)` with the removable singularity at 0.
#[must_use]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1.0e-8 {
        1.0
    } else {
        let p = PI * x;
        ops::sin(p) / p
    }
}

/// Lanczos window `sinc(x) * sinc(x / a)` on `|x| < a`, else 0.
#[must_use]
fn lanczos(x: f32, a: f32) -> f32 {
    if x.abs() < a {
        sinc(x) * sinc(x / a)
    } else {
        0.0
    }
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
                ColorSpace::Srgb => [
                    linear_to_srgb(p[0]),
                    linear_to_srgb(p[1]),
                    linear_to_srgb(p[2]),
                    a,
                ],
            }
        })
        .collect();
    Rgba8Image::new(w, h, texels)
}

/// Resample the X axis of `buf` (`w x h`) to `w/2` via the windowed kernel.
fn downsample_x(buf: &[[f32; 4]], w: u32, h: u32, kernel: WindowedKernel) -> (Vec<[f32; 4]>, u32) {
    let out_w = w / 2;
    let a = kernel.radius();
    let scale = 2.0f32;
    let support = a * scale;
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
                let wt = lanczos((s as f32 - center) / scale, a);
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

/// Resample the Y axis of `buf` (`w x h`) to `h/2` via the windowed kernel.
fn downsample_y(buf: &[[f32; 4]], w: u32, h: u32, kernel: WindowedKernel) -> (Vec<[f32; 4]>, u32) {
    let out_h = h / 2;
    let a = kernel.radius();
    let scale = 2.0f32;
    let support = a * scale;
    let mut out = vec![[0.0f32; 4]; (w as usize) * (out_h as usize)];
    for oy in 0..out_h {
        let center = oy as f32 * 2.0 + 0.5;
        let lo = (center - support).floor() as i64;
        let hi = (center + support).ceil() as i64;
        for x in 0..w {
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for s in lo..=hi {
                let wt = lanczos((s as f32 - center) / scale, a);
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

#[inline]
fn reducible_dim(dim: u32) -> bool {
    dim == 1 || dim.is_multiple_of(2)
}

/// Produce the next mip level with a separable Lanczos reduction, or `None`
/// when `src` is already `1x1` or has a non-reducible (odd, >1) dimension.
#[must_use]
pub fn windowed_downsample(
    src: &Rgba8Image,
    space: ColorSpace,
    kernel: WindowedKernel,
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
        let (nb, nw) = downsample_x(&buf, cw, ch, kernel);
        buf = nb;
        cw = nw;
    }
    if ch > 1 {
        let (nb, nh) = downsample_y(&buf, cw, ch, kernel);
        buf = nb;
        ch = nh;
    }
    from_working(&buf, cw, ch, space)
}

/// Build the full Lanczos mip chain from `base` to `1x1` (inclusive of `base`).
#[must_use]
pub fn generate_mip_chain_windowed(
    base: Rgba8Image,
    space: ColorSpace,
    kernel: WindowedKernel,
) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = windowed_downsample(&current, space, kernel);
        chain.push(current);
        match next {
            Some(level) => current = level,
            None => break,
        }
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Rgba8Image {
        Rgba8Image::new(w, h, vec![c; (w * h) as usize]).unwrap()
    }

    #[test]
    fn sinc_and_lanczos_identities() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-6);
        // sinc is zero at nonzero integers.
        assert!(sinc(1.0).abs() < 1e-6);
        assert!(sinc(2.0).abs() < 1e-6);
        // Lanczos peak and symmetry.
        assert!((lanczos(0.0, 3.0) - 1.0).abs() < 1e-6);
        assert!((lanczos(0.7, 3.0) - lanczos(-0.7, 3.0)).abs() < 1e-6);
        // Compact support.
        assert_eq!(lanczos(3.0, 3.0), 0.0);
        assert_eq!(lanczos(2.0, 2.0), 0.0);
    }

    #[test]
    fn constant_image_is_preserved_linear() {
        let img = solid(8, 8, [10, 20, 30, 200]);
        let mip = windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos3).unwrap();
        assert_eq!((mip.width(), mip.height()), (4, 4));
        // Normalised weights -> constant preserved within +/-1 LSB rounding.
        for t in mip.as_slice() {
            for c in 0..4 {
                let d = i16::from(t[c]) - i16::from(img.as_slice()[0][c]);
                assert!(d.abs() <= 1, "channel {c} drifted by {d}");
            }
        }
    }

    #[test]
    fn constant_image_is_preserved_srgb() {
        let img = solid(8, 8, [128, 64, 200, 255]);
        let mip = windowed_downsample(&img, ColorSpace::Srgb, WindowedKernel::Lanczos2).unwrap();
        for t in mip.as_slice() {
            for c in 0..3 {
                let d = i16::from(t[c]) - i16::from(img.as_slice()[0][c]);
                assert!(d.abs() <= 1, "srgb channel {c} drifted by {d}");
            }
            assert_eq!(t[3], 255);
        }
    }

    #[test]
    fn output_stays_in_gamut() {
        // A high-contrast checker can ring; output must still clamp to [0,255].
        let mut texels = Vec::new();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = if (x + y) % 2 == 0 { 0 } else { 255 };
                texels.push([v, v, v, 255]);
            }
        }
        let img = Rgba8Image::new(8, 8, texels).unwrap();
        let mip = windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos3).unwrap();
        // No assertion needed beyond "did not panic / produced valid u8"; the
        // round_u8 clamp guarantees gamut. Spot-check dimensions.
        assert_eq!((mip.width(), mip.height()), (4, 4));
    }

    #[test]
    fn non_square_pot_carries_dimension_of_one() {
        let img = solid(4, 1, [50, 60, 70, 80]);
        let m1 = windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos2).unwrap();
        assert_eq!((m1.width(), m1.height()), (2, 1));
        let m2 = windowed_downsample(&m1, ColorSpace::Linear, WindowedKernel::Lanczos2).unwrap();
        assert_eq!((m2.width(), m2.height()), (1, 1));
        assert!(windowed_downsample(&m2, ColorSpace::Linear, WindowedKernel::Lanczos2).is_none());
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        let img = solid(3, 2, [1, 2, 3, 4]);
        assert!(windowed_downsample(&img, ColorSpace::Linear, WindowedKernel::Lanczos3).is_none());
    }

    #[test]
    fn mip_chain_has_log2_plus_one_levels() {
        let chain =
            generate_mip_chain_windowed(solid(8, 8, [1, 1, 1, 255]), ColorSpace::Linear, WindowedKernel::Lanczos3);
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }
}
