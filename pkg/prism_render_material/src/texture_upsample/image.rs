//! Edge-aware **joint bilateral upsampling of an `RGBA8` colour image**
//! (Kopf, Cohen, Lischinski & Uyttendaele 2007), the colour follow-on to
//! [`joint_bilateral_upsample_plane`](crate::joint_bilateral_upsample_plane).
//!
//! A half/quarter-resolution colour solve (e.g. a cheap GI/irradiance colour
//! buffer, a downsampled bloom/translucency layer, or a decimated decal atlas)
//! is upsampled to full resolution under a **full-resolution single-channel
//! guide** (depth, luma, or a packed edge signal). As with the plane version,
//! a low-res tap on the far side of a guide edge gets a near-zero range weight,
//! so colour never bleeds across silhouettes.
//!
//! The four `RGBA8` channels are gathered under **one** shared spatial x range
//! weight set per output pixel (the guide is scalar, so the weights do not
//! depend on the channel). This is both the physically correct choice -- a
//! single guide steers all channels identically -- and roughly 4x cheaper than
//! upsampling four planes independently, because the transcendental spatial and
//! range weights are evaluated once and reused. Colour channels are lifted to
//! scene-linear light under [`ColorSpace`] (so the convex combination happens
//! in linear light, matching a GPU upsample); alpha is always treated as linear.
//!
//! Because every weight is non-negative and the result is renormalised, each
//! channel is a convex combination of the low-res samples: a constant colour is
//! preserved and the output never leaves the low-res sample range.
//!
//! The oracle below pins the implementation to the independently verified
//! single-channel kernel: upsampling the image must equal lifting its four
//! channels to scene-linear planes, running [`joint_bilateral_upsample_plane`]
//! on each with the *same* arguments, and re-encoding -- so the shared-weight
//! gather here cannot silently diverge from the reference math.
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML.
//!
//! # References
//! * Kopf, Cohen, Lischinski & Uyttendaele, "Joint Bilateral Upsampling"
//!   (SIGGRAPH 2007).

use alloc::vec;

use bevy_math::ops;

use super::guided::{nearest_upsample, wrap_index};
use crate::{linear_to_srgb, srgb_to_linear, ColorSpace, Rgba8Image, WrapMode};

/// Lift one stored `RGBA8` texel to a scene-linear `[r, g, b, a]` under `space`.
#[inline]
fn lift(texel: [u8; 4], space: ColorSpace) -> [f32; 4] {
    match space {
        ColorSpace::Linear => [
            f32::from(texel[0]) / 255.0,
            f32::from(texel[1]) / 255.0,
            f32::from(texel[2]) / 255.0,
            f32::from(texel[3]) / 255.0,
        ],
        ColorSpace::Srgb => [
            srgb_to_linear(texel[0]),
            srgb_to_linear(texel[1]),
            srgb_to_linear(texel[2]),
            f32::from(texel[3]) / 255.0,
        ],
    }
}

/// Re-encode a scene-linear `[r, g, b, a]` back to a stored `RGBA8` texel.
#[inline]
fn store(c: [f32; 4], space: ColorSpace) -> [u8; 4] {
    let round_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
    match space {
        ColorSpace::Linear => [
            round_u8(c[0]),
            round_u8(c[1]),
            round_u8(c[2]),
            round_u8(c[3]),
        ],
        ColorSpace::Srgb => [
            linear_to_srgb(c[0]),
            linear_to_srgb(c[1]),
            linear_to_srgb(c[2]),
            round_u8(c[3]),
        ],
    }
}

/// Joint bilateral upsample the low-resolution `RGBA8` colour image `low` to
/// `full_width * full_height`, steered by the full-resolution single-channel
/// `guide`. Colour channels are combined in scene-linear light under `space`;
/// alpha is linear.
///
/// `sigma_s` is the spatial support in **low-res texels**; `sigma_r` is the
/// guide range sigma (same units as the guide values). Borders resolve through
/// `wrap`.
///
/// Returns `None` for a size-mismatched guide (`guide.len() != full_width *
/// full_height`) or a zero full-res dimension, and falls back to a
/// nearest-neighbour upsample when either sigma is non-finite or `<= 0`.
#[must_use]
pub fn joint_bilateral_upsample_rgba8(
    low: &Rgba8Image,
    guide: &[f32],
    full_width: u32,
    full_height: u32,
    sigma_s: f32,
    sigma_r: f32,
    wrap: WrapMode,
    space: ColorSpace,
) -> Option<Rgba8Image> {
    let lw = low.width() as usize;
    let lh = low.height() as usize;
    let fw = full_width as usize;
    let fh = full_height as usize;
    if fw == 0 || fh == 0 || guide.len() != fw * fh {
        return None;
    }

    // Scene-linear low-res channels, de-interleaved for cheap gather.
    let src = low.as_slice();
    let mut lr = vec![0.0f32; lw * lh];
    let mut lg = vec![0.0f32; lw * lh];
    let mut lb = vec![0.0f32; lw * lh];
    let mut la = vec![0.0f32; lw * lh];
    for (i, &texel) in src.iter().enumerate() {
        let c = lift(texel, space);
        lr[i] = c[0];
        lg[i] = c[1];
        lb[i] = c[2];
        la[i] = c[3];
    }

    if !sigma_s.is_finite() || sigma_s <= 0.0 || !sigma_r.is_finite() || sigma_r <= 0.0 {
        let nr = nearest_upsample(&lr, lw, lh, fw, fh);
        let ng = nearest_upsample(&lg, lw, lh, fw, fh);
        let nb = nearest_upsample(&lb, lw, lh, fw, fh);
        let na = nearest_upsample(&la, lw, lh, fw, fh);
        let texels = (0..fw * fh)
            .map(|i| store([nr[i], ng[i], nb[i], na[i]], space))
            .collect();
        return Rgba8Image::new(full_width, full_height, texels);
    }

    let sx = lw as f32 / fw as f32;
    let sy = lh as f32 / fh as f32;
    let inv2s2 = 1.0 / (2.0 * sigma_s * sigma_s);
    let inv2r2 = 1.0 / (2.0 * sigma_r * sigma_r);
    let radius = (3.0 * sigma_s) as i64 + 1;

    let up_x = |qx: i64| (qx as f32 + 0.5) / sx - 0.5;
    let up_y = |qy: i64| (qy as f32 + 0.5) / sy - 0.5;
    let wi = lw as i64;
    let hi = lh as i64;
    let fwi = full_width as i64;
    let fhi = full_height as i64;

    let mut texels = vec![[0u8; 4]; fw * fh];
    for y in 0..fh {
        let ly = (y as f32 + 0.5) * sy - 0.5;
        let cy = ops::round(ly) as i64;
        for x in 0..fw {
            let lx = (x as f32 + 0.5) * sx - 0.5;
            let cx = ops::round(lx) as i64;
            let gp = guide[y * fw + x];

            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for qy in (cy - radius)..=(cy + radius) {
                let dy = qy as f32 - ly;
                let sqy = wrap_index(qy, hi, wrap);
                for qx in (cx - radius)..=(cx + radius) {
                    let dx = qx as f32 - lx;
                    let ws = ops::exp(-(dx * dx + dy * dy) * inv2s2);

                    let gx = wrap_index(ops::round(up_x(qx)) as i64, fwi, wrap);
                    let gy = wrap_index(ops::round(up_y(qy)) as i64, fhi, wrap);
                    let gq = guide[gy * fw + gx];
                    let dr = gp - gq;
                    let wr = ops::exp(-dr * dr * inv2r2);

                    let w = ws * wr;
                    let sqx = wrap_index(qx, wi, wrap);
                    let idx = sqy * lw + sqx;
                    acc[0] += w * lr[idx];
                    acc[1] += w * lg[idx];
                    acc[2] += w * lb[idx];
                    acc[3] += w * la[idx];
                    wsum += w;
                }
            }
            let c = if wsum > 0.0 {
                [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum, acc[3] / wsum]
            } else {
                let idx = wrap_index(cy, hi, wrap) * lw + wrap_index(cx, wi, wrap);
                [lr[idx], lg[idx], lb[idx], la[idx]]
            };
            texels[y * fw + x] = store(c, space);
        }
    }
    Rgba8Image::new(full_width, full_height, texels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint_bilateral_upsample_plane;
    use alloc::vec;
    use alloc::vec::Vec;

    const WRAPS: [WrapMode; 2] = [WrapMode::ClampToEdge, WrapMode::Repeat];

    fn noise_u8(w: u32, h: u32, seed: u64) -> Vec<[u8; 4]> {
        let n = (w * h) as usize;
        let mut v = Vec::with_capacity(n);
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 56) as u8
        };
        for _ in 0..n {
            v.push([next(), next(), next(), next()]);
        }
        v
    }

    fn noise_f32(w: u32, h: u32, seed: u64) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut v = Vec::with_capacity(n);
        let mut state = seed | 1;
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            v.push(((state >> 40) as u32 as f32) / (1u64 << 24) as f32);
        }
        v
    }

    /// Anti-fake oracle: the shared-weight image upsample must equal lifting the
    /// four channels to scene-linear planes, running the independently verified
    /// single-channel kernel on each with identical arguments, and re-encoding.
    #[test]
    fn image_matches_per_channel_plane_kernel() {
        let (lw, lh) = (5u32, 4u32);
        let (fw, fh) = (13u32, 9u32);
        let low = Rgba8Image::new(lw, lh, noise_u8(lw, lh, 0xA11CE)).unwrap();
        let guide = noise_f32(fw, fh, 0xBEEF);
        for space in [ColorSpace::Linear, ColorSpace::Srgb] {
            for wrap in WRAPS {
                let img =
                    joint_bilateral_upsample_rgba8(&low, &guide, fw, fh, 1.3, 0.2, wrap, space)
                        .unwrap();

                // Reference: four scene-linear planes through the plane kernel.
                let src = low.as_slice();
                let mut planes = [
                    vec![0.0f32; (lw * lh) as usize],
                    vec![0.0f32; (lw * lh) as usize],
                    vec![0.0f32; (lw * lh) as usize],
                    vec![0.0f32; (lw * lh) as usize],
                ];
                for (i, &texel) in src.iter().enumerate() {
                    let c = lift(texel, space);
                    for ch in 0..4 {
                        planes[ch][i] = c[ch];
                    }
                }
                let up: Vec<Vec<f32>> = planes
                    .iter()
                    .map(|pl| {
                        joint_bilateral_upsample_plane(pl, lw, lh, &guide, fw, fh, 1.3, 0.2, wrap)
                    })
                    .collect();
                for (i, &got) in img.as_slice().iter().enumerate() {
                    let want = store([up[0][i], up[1][i], up[2][i], up[3][i]], space);
                    assert_eq!(got, want, "pixel {i} space {space:?}");
                }
            }
        }
    }

    #[test]
    fn constant_colour_is_preserved() {
        let (lw, lh) = (3u32, 3u32);
        let (fw, fh) = (10u32, 7u32);
        let colour = [40u8, 160u8, 90u8, 200u8];
        let low = Rgba8Image::new(lw, lh, vec![colour; (lw * lh) as usize]).unwrap();
        let guide = noise_f32(fw, fh, 0x5151);
        for space in [ColorSpace::Linear, ColorSpace::Srgb] {
            let img = joint_bilateral_upsample_rgba8(
                &low,
                &guide,
                fw,
                fh,
                2.0,
                0.1,
                WrapMode::Repeat,
                space,
            )
            .unwrap();
            for &t in img.as_slice() {
                // Round-trip of a single constant colour is exact to +/-1 LSB.
                for ch in 0..4 {
                    let d = i32::from(t[ch]) - i32::from(colour[ch]);
                    assert!(d.abs() <= 1, "ch {ch}: {} vs {}", t[ch], colour[ch]);
                }
            }
        }
    }

    #[test]
    fn output_alpha_within_low_res_range() {
        let (lw, lh) = (4u32, 4u32);
        let (fw, fh) = (11u32, 11u32);
        let texels = noise_u8(lw, lh, 0xD00D);
        let amin = texels.iter().map(|t| t[3]).min().unwrap();
        let amax = texels.iter().map(|t| t[3]).max().unwrap();
        let low = Rgba8Image::new(lw, lh, texels).unwrap();
        let guide = noise_f32(fw, fh, 0x3C3C);
        let img = joint_bilateral_upsample_rgba8(
            &low,
            &guide,
            fw,
            fh,
            1.5,
            0.3,
            WrapMode::ClampToEdge,
            ColorSpace::Linear,
        )
        .unwrap();
        for &t in img.as_slice() {
            assert!(t[3] >= amin.saturating_sub(1) && t[3] <= amax.saturating_add(1));
        }
    }

    #[test]
    fn mismatched_guide_is_none() {
        let low = Rgba8Image::new(2, 2, vec![[1u8; 4]; 4]).unwrap();
        assert!(joint_bilateral_upsample_rgba8(
            &low,
            &[0.0; 3],
            4,
            4,
            1.0,
            0.1,
            WrapMode::Repeat,
            ColorSpace::Linear,
        )
        .is_none());
        assert!(joint_bilateral_upsample_rgba8(
            &low,
            &[],
            0,
            4,
            1.0,
            0.1,
            WrapMode::Repeat,
            ColorSpace::Linear,
        )
        .is_none());
    }

    #[test]
    fn nonpositive_sigma_falls_back_to_nearest() {
        let (lw, lh) = (3u32, 2u32);
        let (fw, fh) = (9u32, 6u32);
        let low = Rgba8Image::new(lw, lh, noise_u8(lw, lh, 0x7777)).unwrap();
        let guide = noise_f32(fw, fh, 0x8888);
        let img = joint_bilateral_upsample_rgba8(
            &low,
            &guide,
            fw,
            fh,
            0.0,
            0.1,
            WrapMode::ClampToEdge,
            ColorSpace::Linear,
        )
        .unwrap();
        // Nearest replication: every full-res texel equals its source low-res
        // texel under the pixel-centre mapping.
        let sx = lw as f32 / fw as f32;
        let sy = lh as f32 / fh as f32;
        let src = low.as_slice();
        for y in 0..fh as usize {
            let ly = ((((y as f32 + 0.5) * sy - 0.5).max(0.0)) as usize).min(lh as usize - 1);
            for x in 0..fw as usize {
                let lx = ((((x as f32 + 0.5) * sx - 0.5).max(0.0)) as usize).min(lw as usize - 1);
                assert_eq!(
                    img.as_slice()[y * fw as usize + x],
                    src[ly * lw as usize + lx]
                );
            }
        }
    }
}
