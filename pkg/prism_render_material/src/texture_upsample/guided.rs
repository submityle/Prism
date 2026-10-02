//! Edge-aware **joint bilateral upsampling** (Kopf, Cohen, Lischinski &
//! Uyttendaele 2007).
//!
//! Many expensive screen-space effects -- ambient occlusion, diffuse GI,
//! soft shadows, subsurface scattering -- are solved at half or quarter
//! resolution for speed and then upsampled. A plain bilinear upsample bleeds
//! the low-res solution across depth/normal discontinuities, haloing object
//! silhouettes. Joint bilateral upsampling instead steers the interpolation
//! with a **full-resolution guide** (depth, normal, or luma): low-res taps on
//! the far side of a guide edge get a near-zero range weight, so the upsampled
//! signal keeps the crisp edges of the full-res frame while staying smooth
//! inside each region.
//!
//! For a full-res output pixel `p` with guide value `G_p`, the low-res signal
//! `S` is gathered in low-res space:
//!
//! ```text
//! out(p) = (1/W) * sum_q  S(q) * exp(-||q - p_low||^2 / (2 sigma_s^2))
//!                             * exp(-(G_p - G_up(q))^2 / (2 sigma_r^2))
//! ```
//!
//! where `p_low` is `p` mapped into low-res coordinates (pixel-centre aligned),
//! `q` ranges over low-res texels near `p_low`, and `G_up(q)` is the full-res
//! guide sampled at the full-res position of low-res texel `q`. Every weight is
//! non-negative and the result is renormalised, so the output is a **convex
//! combination** of the low-res samples: a constant low-res input is preserved
//! exactly for any guide, and the output never leaves the low-res sample range.
//!
//! Properties the oracles below pin down:
//! * with a tiny `sigma_s` and matching resolutions the gather collapses to the
//!   coincident texel, so the upsample is the **identity**;
//! * with `sigma_r -> infinity` the range term vanishes and the result equals
//!   an **independent spatial-only** Gaussian resample (ties the low-res mapping
//!   and spatial weights to separate code);
//! * a constant low-res input is preserved for any guide / sigma;
//! * the output stays within the low-res sample range for any guide;
//! * a step guide with a small `sigma_r` keeps the two sides apart (edge-aware).
//!
//! Everything is deterministic analytic `f32` arithmetic -- no AI/ML -- so a CPU
//! golden matches a GPU compute upsample to floating-point tolerance.
//!
//! # References
//! * Kopf, Cohen, Lischinski & Uyttendaele, "Joint Bilateral Upsampling"
//!   (SIGGRAPH 2007).

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

use crate::WrapMode;

/// Wrap an integer tap index into `[0, n)` (same convention as the blurs).
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

/// Nearest-neighbour upsample of `low` into a `full_w * full_h` plane (the
/// well-defined fallback for a degenerate `sigma`).
fn nearest_upsample(
    low: &[f32],
    low_w: usize,
    low_h: usize,
    full_w: usize,
    full_h: usize,
) -> Vec<f32> {
    let sx = low_w as f32 / full_w as f32;
    let sy = low_h as f32 / full_h as f32;
    let mut out = vec![0.0f32; full_w * full_h];
    for y in 0..full_h {
        let ly = (((y as f32 + 0.5) * sy - 0.5).max(0.0)) as usize;
        let ly = ly.min(low_h - 1);
        for x in 0..full_w {
            let lx = (((x as f32 + 0.5) * sx - 0.5).max(0.0)) as usize;
            let lx = lx.min(low_w - 1);
            out[y * full_w + x] = low[ly * low_w + lx];
        }
    }
    out
}

/// Joint bilateral upsample the single-channel low-resolution plane `low`
/// (`low_width * low_height`) to `full_width * full_height`, steered by the
/// full-resolution single-channel `guide`.
///
/// `sigma_s` is the spatial support in **low-res texels**; `sigma_r` is the
/// guide range sigma (same units as the guide values). Borders resolve through
/// `wrap`.
///
/// Returns an empty `Vec` for empty or size-mismatched inputs, and falls back to
/// a nearest-neighbour upsample when either sigma is non-finite or `<= 0`.
#[must_use]
pub fn joint_bilateral_upsample_plane(
    low: &[f32],
    low_width: u32,
    low_height: u32,
    guide: &[f32],
    full_width: u32,
    full_height: u32,
    sigma_s: f32,
    sigma_r: f32,
    wrap: WrapMode,
) -> Vec<f32> {
    let (lw, lh) = (low_width as usize, low_height as usize);
    let (fw, fh) = (full_width as usize, full_height as usize);
    if fw == 0 || fh == 0 || lw == 0 || lh == 0 || low.len() != lw * lh || guide.len() != fw * fh {
        return Vec::new();
    }
    if !sigma_s.is_finite() || sigma_s <= 0.0 || !sigma_r.is_finite() || sigma_r <= 0.0 {
        return nearest_upsample(low, lw, lh, fw, fh);
    }

    let sx = lw as f32 / fw as f32;
    let sy = lh as f32 / fh as f32;
    let inv2s2 = 1.0 / (2.0 * sigma_s * sigma_s);
    let inv2r2 = 1.0 / (2.0 * sigma_r * sigma_r);
    let radius = (3.0 * sigma_s) as i64 + 1;

    // Full-res position (pixel centre) of a low-res texel coordinate.
    let up_x = |qx: i64| (qx as f32 + 0.5) / sx - 0.5;
    let up_y = |qy: i64| (qy as f32 + 0.5) / sy - 0.5;
    let wi = low_width as i64;
    let hi = low_height as i64;
    let fwi = full_width as i64;
    let fhi = full_height as i64;

    let mut out = vec![0.0f32; fw * fh];
    for y in 0..fh {
        let ly = (y as f32 + 0.5) * sy - 0.5;
        let cy = ops::round(ly) as i64;
        for x in 0..fw {
            let lx = (x as f32 + 0.5) * sx - 0.5;
            let cx = ops::round(lx) as i64;
            let gp = guide[y * fw + x];

            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for qy in (cy - radius)..=(cy + radius) {
                let dy = qy as f32 - ly;
                let sqy = wrap_index(qy, hi, wrap);
                for qx in (cx - radius)..=(cx + radius) {
                    let dx = qx as f32 - lx;
                    let ws = ops::exp(-(dx * dx + dy * dy) * inv2s2);

                    // Full-res guide at the upsampled position of low-res q.
                    let gx = wrap_index(ops::round(up_x(qx)) as i64, fwi, wrap);
                    let gy = wrap_index(ops::round(up_y(qy)) as i64, fhi, wrap);
                    let gq = guide[gy * fw + gx];
                    let dr = gp - gq;
                    let wr = ops::exp(-dr * dr * inv2r2);

                    let w = ws * wr;
                    let sqx = wrap_index(qx, wi, wrap);
                    acc += w * low[sqy * lw + sqx];
                    wsum += w;
                }
            }
            out[y * fw + x] = if wsum > 0.0 {
                acc / wsum
            } else {
                low[wrap_index(cy, hi, wrap) * lw + wrap_index(cx, wi, wrap)]
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const WRAPS: [WrapMode; 2] = [WrapMode::ClampToEdge, WrapMode::Repeat];

    fn noise(w: u32, h: u32, seed: u64) -> Vec<f32> {
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

    /// Independent spatial-only resample (range term forced to 1): the oracle
    /// that ties the low-res mapping + spatial weights to separate code.
    fn spatial_only(
        low: &[f32],
        lw: usize,
        lh: usize,
        fw: usize,
        fh: usize,
        sigma_s: f32,
        wrap: WrapMode,
    ) -> Vec<f32> {
        let sx = lw as f32 / fw as f32;
        let sy = lh as f32 / fh as f32;
        let inv2s2 = 1.0 / (2.0 * sigma_s * sigma_s);
        let radius = (3.0 * sigma_s) as i64 + 1;
        let wi = lw as i64;
        let hi = lh as i64;
        let mut out = vec![0.0f32; fw * fh];
        for y in 0..fh {
            let ly = (y as f32 + 0.5) * sy - 0.5;
            let cy = ops::round(ly) as i64;
            for x in 0..fw {
                let lx = (x as f32 + 0.5) * sx - 0.5;
                let cx = ops::round(lx) as i64;
                let mut acc = 0.0f32;
                let mut wsum = 0.0f32;
                for qy in (cy - radius)..=(cy + radius) {
                    let dy = qy as f32 - ly;
                    let sqy = wrap_index(qy, hi, wrap);
                    for qx in (cx - radius)..=(cx + radius) {
                        let dx = qx as f32 - lx;
                        let w = ops::exp(-(dx * dx + dy * dy) * inv2s2);
                        let sqx = wrap_index(qx, wi, wrap);
                        acc += w * low[sqy * lw + sqx];
                        wsum += w;
                    }
                }
                out[y * fw + x] = acc / wsum;
            }
        }
        out
    }

    #[test]
    fn tiny_sigma_same_resolution_is_identity() {
        let (w, h) = (9u32, 7u32);
        let low = noise(w, h, 0x1234);
        let guide = noise(w, h, 0x9999);
        let out = joint_bilateral_upsample_plane(
            &low,
            w,
            h,
            &guide,
            w,
            h,
            0.05,
            0.5,
            WrapMode::ClampToEdge,
        );
        for i in 0..(w * h) as usize {
            assert!(
                (out[i] - low[i]).abs() < 1.0e-5,
                "i={i} {} vs {}",
                out[i],
                low[i]
            );
        }
    }

    #[test]
    fn huge_range_sigma_matches_spatial_only() {
        let (lw, lh) = (5u32, 4u32);
        let (fw, fh) = (10u32, 8u32);
        let low = noise(lw, lh, 0x55);
        let guide = noise(fw, fh, 0xAA);
        for &wrap in &WRAPS {
            let got =
                joint_bilateral_upsample_plane(&low, lw, lh, &guide, fw, fh, 1.0, 1.0e9, wrap);
            let want = spatial_only(
                &low,
                lw as usize,
                lh as usize,
                fw as usize,
                fh as usize,
                1.0,
                wrap,
            );
            for i in 0..(fw * fh) as usize {
                assert!(
                    (got[i] - want[i]).abs() < 1.0e-4,
                    "wrap={wrap:?} i={i} {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn constant_low_res_is_preserved() {
        let (lw, lh) = (4u32, 4u32);
        let (fw, fh) = (12u32, 11u32);
        let low = vec![0.42f32; (lw * lh) as usize];
        let guide = noise(fw, fh, 0x7);
        for &wrap in &WRAPS {
            let out = joint_bilateral_upsample_plane(&low, lw, lh, &guide, fw, fh, 1.5, 0.1, wrap);
            for &o in &out {
                assert!((o - 0.42).abs() < 1.0e-5, "{o}");
            }
        }
    }

    #[test]
    fn output_stays_within_low_res_range() {
        let (lw, lh) = (6u32, 5u32);
        let (fw, fh) = (13u32, 11u32);
        let low = noise(lw, lh, 0x321);
        let guide = noise(fw, fh, 0x654);
        let lo = low.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = low.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &wrap in &WRAPS {
            let out = joint_bilateral_upsample_plane(&low, lw, lh, &guide, fw, fh, 1.2, 0.2, wrap);
            for &o in &out {
                assert!(o >= lo - 1.0e-5 && o <= hi + 1.0e-5, "{o} [{lo},{hi}]");
            }
        }
    }

    #[test]
    fn step_guide_keeps_sides_apart() {
        // Low-res: left column 0.0, right column 1.0. Guide: hard step at the
        // full-res midline aligned with the low-res split. A small sigma_r must
        // keep the left output near 0 and the right near 1.
        let (lw, lh) = (2u32, 1u32);
        let (fw, fh) = (8u32, 1u32);
        let low = vec![0.0f32, 1.0f32];
        let mut guide = vec![0.0f32; (fw * fh) as usize];
        for (x, g) in guide.iter_mut().enumerate() {
            *g = if x < 4 { 0.0 } else { 1.0 };
        }
        let out = joint_bilateral_upsample_plane(
            &low,
            lw,
            lh,
            &guide,
            fw,
            fh,
            2.0,
            0.05,
            WrapMode::ClampToEdge,
        );
        for (x, &v) in out.iter().enumerate().take(4) {
            assert!(v < 0.1, "left x={x} -> {v}");
        }
        for (x, &v) in out.iter().enumerate().skip(4) {
            assert!(v > 0.9, "right x={x} -> {v}");
        }
    }

    #[test]
    fn empty_or_mismatched_is_empty() {
        assert!(
            joint_bilateral_upsample_plane(&[], 0, 0, &[], 4, 4, 1.0, 1.0, WrapMode::Repeat)
                .is_empty()
        );
        let low = vec![0.5f32; 4];
        let guide = vec![0.5f32; 16];
        // low length mismatch
        assert!(joint_bilateral_upsample_plane(
            &low,
            3,
            3,
            &guide,
            4,
            4,
            1.0,
            1.0,
            WrapMode::Repeat
        )
        .is_empty());
        // guide length mismatch
        assert!(joint_bilateral_upsample_plane(
            &low,
            2,
            2,
            &guide,
            5,
            5,
            1.0,
            1.0,
            WrapMode::Repeat
        )
        .is_empty());
    }

    #[test]
    fn nonpositive_sigma_falls_back_to_nearest() {
        let (lw, lh) = (2u32, 2u32);
        let (fw, fh) = (4u32, 4u32);
        let low = vec![0.0f32, 1.0, 2.0, 3.0];
        let guide = vec![0.0f32; (fw * fh) as usize];
        let out = joint_bilateral_upsample_plane(
            &low,
            lw,
            lh,
            &guide,
            fw,
            fh,
            0.0,
            1.0,
            WrapMode::ClampToEdge,
        );
        let want = nearest_upsample(&low, lw as usize, lh as usize, fw as usize, fh as usize);
        assert_eq!(out, want);
    }
}
