//! General separable image resampling to an arbitrary target resolution.
//!
//! Mip generation ([`texture_mipgen`](crate::texture_mipgen)) only ever halves
//! a dimension. AAA import/cook pipelines also need to resize a decoded
//! `RGBA8` image to an *arbitrary* resolution: fitting source art to a
//! power-of-two atlas slot, building UI/thumbnail levels, or conforming
//! mismatched texture inputs. This module is that resampler.
//!
//! It is a classic separable, gamma-correct, normalised-weight resampler
//! (Turkowski 1990): colour is lifted once to scene-linear `f32`, the image is
//! filtered along X then Y with a selectable [`ResizeFilter`], and the result
//! is re-encoded once. On **minification** the kernel footprint widens by the
//! inverse scale (acting as a low-pass to avoid aliasing); on **magnification**
//! the kernel keeps its native width (reconstruction). Tap weights are always
//! normalised, so a constant image is preserved exactly at any target size, and
//! only the fetch index is clamped to the edge (edge-clamp of a constant is the
//! constant). Everything is deterministic analytic arithmetic -- no AI/ML -- so
//! a CPU golden matches a GPU compute resampler to floating-point tolerance.
//!
//! # Verified reductions
//! * same-size [`ResizeFilter::Box`] is the identity;
//! * exact 2x [`ResizeFilter::Box`] minification equals the box mip reducer
//!   ([`box_downsample`](crate::box_downsample)) 2x2 average;
//! * integer-factor [`ResizeFilter::Box`] magnification is nearest replication.
//!
//! # References
//! * Turkowski, "Filters for Common Resampling Tasks" (1990).
//! * Heckbert, "Fundamentals of Texture Mapping and Image Warping" (1989),
//!   Chapter 5 (separable resampling, filter scaling for minification).

mod kernels;

use alloc::vec;
use alloc::vec::Vec;

use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image};

pub use kernels::ResizeFilter;

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
                ColorSpace::Srgb => [
                    srgb_to_linear(t[0]),
                    srgb_to_linear(t[1]),
                    srgb_to_linear(t[2]),
                    a,
                ],
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
        .map(|pxl| {
            let a = round_u8(pxl[3]);
            match space {
                ColorSpace::Linear => [round_u8(pxl[0]), round_u8(pxl[1]), round_u8(pxl[2]), a],
                ColorSpace::Srgb => [
                    linear_to_srgb(pxl[0]),
                    linear_to_srgb(pxl[1]),
                    linear_to_srgb(pxl[2]),
                    a,
                ],
            }
        })
        .collect();
    Rgba8Image::new(w, h, texels)
}

/// Per-axis contribution of one output sample: the clamped source taps and
/// their already-normalised weights.
struct Contrib {
    taps: Vec<(usize, f32)>,
}

/// Precompute the clamped-and-normalised taps for every output sample on one
/// axis of length `in_len -> out_len`.
fn axis_contributions(in_len: u32, out_len: u32, filter: ResizeFilter) -> Vec<Contrib> {
    let ratio = out_len as f32 / in_len as f32;
    let fscale = if ratio < 1.0 { 1.0 / ratio } else { 1.0 };
    let inv_fscale = 1.0 / fscale;
    let support = filter.radius() * fscale;
    let last = in_len as i64 - 1;
    (0..out_len)
        .map(|o| {
            let center = (o as f32 + 0.5) / ratio - 0.5;
            let lo = (center - support).floor() as i64;
            let hi = (center + support).ceil() as i64;
            let mut taps: Vec<(usize, f32)> = Vec::new();
            let mut wsum = 0.0f32;
            for s in lo..=hi {
                let wt = filter.eval((s as f32 - center) * inv_fscale);
                if wt == 0.0 {
                    continue;
                }
                let cs = s.clamp(0, last) as usize;
                // Fold duplicate clamped indices (edge taps) together.
                if let Some(entry) = taps.iter_mut().find(|(i, _)| *i == cs) {
                    entry.1 += wt;
                } else {
                    taps.push((cs, wt));
                }
                wsum += wt;
            }
            let inv = if wsum.abs() > 1.0e-12 {
                1.0 / wsum
            } else {
                0.0
            };
            for t in &mut taps {
                t.1 *= inv;
            }
            Contrib { taps }
        })
        .collect()
}

/// Resize `src` to `dst_w x dst_h` with `filter`, gamma-correctly under
/// `space`. Returns `None` when either target dimension is `0` or the
/// re-encoded image fails to construct.
#[must_use]
pub fn resize(
    src: &Rgba8Image,
    dst_w: u32,
    dst_h: u32,
    filter: ResizeFilter,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    if dst_w == 0 || dst_h == 0 {
        return None;
    }
    let (sw, sh) = (src.width(), src.height());
    let lin = to_working(src, space);

    // X pass: sw -> dst_w, keeping sh rows.
    let xc = axis_contributions(sw, dst_w, filter);
    let mut tmp = vec![[0.0f32; 4]; dst_w as usize * sh as usize];
    for y in 0..sh as usize {
        let row = y * sw as usize;
        for (ox, contrib) in xc.iter().enumerate() {
            let mut acc = [0.0f32; 4];
            for &(cs, wt) in &contrib.taps {
                let t = lin[row + cs];
                for c in 0..4 {
                    acc[c] += wt * t[c];
                }
            }
            tmp[y * dst_w as usize + ox] = acc;
        }
    }

    // Y pass: sh -> dst_h, keeping dst_w columns.
    let yc = axis_contributions(sh, dst_h, filter);
    let mut out = vec![[0.0f32; 4]; dst_w as usize * dst_h as usize];
    for (oy, contrib) in yc.iter().enumerate() {
        for x in 0..dst_w as usize {
            let mut acc = [0.0f32; 4];
            for &(cs, wt) in &contrib.taps {
                let t = tmp[cs * dst_w as usize + x];
                for c in 0..4 {
                    acc[c] += wt * t[c];
                }
            }
            out[oy * dst_w as usize + x] = acc;
        }
    }

    from_working(&out, dst_w, dst_h, space)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_downsample;

    const ALL: [ResizeFilter; 6] = [
        ResizeFilter::Box,
        ResizeFilter::Triangle,
        ResizeFilter::CatmullRom,
        ResizeFilter::Mitchell,
        ResizeFilter::BSpline,
        ResizeFilter::Lanczos3,
    ];

    fn img(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Rgba8Image {
        let mut t = Vec::new();
        for y in 0..h {
            for x in 0..w {
                t.push(f(x, y));
            }
        }
        Rgba8Image::new(w, h, t).unwrap()
    }

    #[test]
    fn target_dimensions_are_respected() {
        let src = img(8, 4, |x, _| [x as u8 * 8, 0, 0, 255]);
        let out = resize(&src, 3, 7, ResizeFilter::Triangle, ColorSpace::Linear).unwrap();
        assert_eq!((out.width(), out.height()), (3, 7));
    }

    #[test]
    fn zero_target_is_none() {
        let src = img(4, 4, |_, _| [1, 2, 3, 4]);
        assert!(resize(&src, 0, 4, ResizeFilter::Box, ColorSpace::Linear).is_none());
        assert!(resize(&src, 4, 0, ResizeFilter::Box, ColorSpace::Linear).is_none());
    }

    #[test]
    fn constant_image_is_preserved_for_every_filter_and_size() {
        let c = [37u8, 190, 211, 128];
        let src = img(6, 10, |_, _| c);
        for f in ALL {
            for &(w, h) in &[(3u32, 5u32), (12, 20), (6, 10), (1, 1), (7, 3)] {
                let out = resize(&src, w, h, f, ColorSpace::Linear).unwrap();
                for px in out.as_slice() {
                    assert_eq!(*px, c, "{f:?} -> {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn same_size_box_is_identity() {
        let src = img(5, 3, |x, y| {
            [x as u8 * 20, y as u8 * 30, (x + y) as u8 * 10, 255]
        });
        let out = resize(&src, 5, 3, ResizeFilter::Box, ColorSpace::Linear).unwrap();
        assert_eq!(out.as_slice(), src.as_slice());
    }

    #[test]
    fn box_half_matches_box_mip_reducer() {
        // Exact 2x box minification must equal the 2x2 box mip reducer.
        for space in [ColorSpace::Linear, ColorSpace::Srgb] {
            let src = img(8, 6, |x, y| {
                [
                    (x * 17 + 3) as u8,
                    (y * 29 + 7) as u8,
                    ((x + y) * 11) as u8,
                    (x * 5 + y * 9) as u8,
                ]
            });
            let via_resize = resize(&src, 4, 3, ResizeFilter::Box, space).unwrap();
            let via_mip = box_downsample(&src, space).unwrap();
            assert_eq!(via_resize.as_slice(), via_mip.as_slice(), "{space:?}");
        }
    }

    #[test]
    fn box_integer_upsample_is_nearest_replication() {
        let src = img(2, 2, |x, y| [x as u8 * 100, y as u8 * 100, 0, 255]);
        let out = resize(&src, 4, 4, ResizeFilter::Box, ColorSpace::Linear).unwrap();
        for y in 0..4u32 {
            for x in 0..4u32 {
                assert_eq!(out.texel(x, y), src.texel(x / 2, y / 2), "x={x} y={y}");
            }
        }
    }

    #[test]
    fn triangle_upsample_of_ramp_is_monotonic() {
        // A horizontal linear ramp upsampled with the triangle filter stays
        // monotonic non-decreasing and keeps its clamped endpoints.
        let src = img(4, 1, |x, _| [x as u8 * 60, 0, 0, 255]);
        let out = resize(&src, 11, 1, ResizeFilter::Triangle, ColorSpace::Linear).unwrap();
        let r: Vec<u8> = out.as_slice().iter().map(|p| p[0]).collect();
        for w in r.windows(2) {
            assert!(w[1] >= w[0], "not monotonic: {r:?}");
        }
        assert_eq!(*r.first().unwrap(), 0);
        assert_eq!(*r.last().unwrap(), 180);
    }

    #[test]
    fn resize_commutes_with_horizontal_mirror() {
        let src = img(6, 4, |x, y| [(x * 30) as u8, (y * 40) as u8, 0, 255]);
        let mirror = |im: &Rgba8Image| {
            let w = im.width();
            img(w, im.height(), |x, y| im.texel(w - 1 - x, y))
        };
        for f in ALL {
            let a = resize(&mirror(&src), 3, 4, f, ColorSpace::Linear).unwrap();
            let b = mirror(&resize(&src, 3, 4, f, ColorSpace::Linear).unwrap());
            assert_eq!(a.as_slice(), b.as_slice(), "{f:?}");
        }
    }
}
