//! Premultiplied-alpha mip reduction for colour textures with transparency.
//!
//! Straight-alpha averaging of `RGBA8` colour mixes the RGB of *fully
//! transparent* texels into the result. Authored cutout/decal/particle assets
//! routinely store arbitrary (often black) RGB under `alpha == 0`, so a plain
//! box filter bleeds those invisible colours into the visible edge, producing
//! the classic dark halo around alpha-blended foliage and sprites. The AAA fix
//! (Blinn's premultiplied/associated alpha; the same reduction `UE`, Unity and
//! `DirectXTex` use for alpha textures) is to weight each texel's colour by its
//! coverage, average the premultiplied colour and the coverage separately, then
//! un-premultiply:
//!
//! ```text
//! C_out = sum_i(a_i * C_i) / sum_i(a_i)      (alpha-weighted colour)
//! A_out = mean_i(a_i)                        (box-averaged coverage)
//! ```
//!
//! Invisible texels (`a_i == 0`) contribute no colour, so the visible edge
//! colour is preserved. When a whole footprint is transparent the colour is
//! undefined; we fall back to the unweighted colour mean so the result is still
//! a finite, reasonable texel (and round-trips if the asset later becomes
//! opaque). For colour/albedo the premultiply happens in scene-linear light
//! (see [`srgb`](super::srgb)); data maps use [`ColorSpace::Linear`]. Alpha is
//! always a linear coverage quantity and is box-averaged in its raw domain.
//!
//! Everything is pure analytic arithmetic -- no AI/ML -- so a CPU golden matches
//! a GPU compute down-sampler to within one LSB.
//!
//! # Conventions
//! * Shares [`Rgba8Image`] / [`ColorSpace`] and the reducible-dimension rule
//!   with [`box_filter`](super::box_filter): each dimension is `1` or even, and
//!   a dimension of `1` is carried through unchanged (GL `max(1, dim >> 1)`).
//! * Coverage weight is `alpha / 255` in `[0, 1]`; the un-premultiply divides by
//!   the summed weight only when it is non-zero.
//!
//! # References
//! * J. Blinn, "Fugue for `MMX`" / "Dirty Pixels" (associated/premultiplied
//!   alpha algebra).
//! * T. Porter & T. Duff, "Compositing Digital Images" (1984) -- premultiplied
//!   alpha compositing.
//! * I. Castano, "Computing Alpha Mipmaps" (`NVIDIA`) -- alpha-aware mip builds.

use alloc::vec::Vec;

use super::box_filter::{ColorSpace, Rgba8Image};
use super::srgb::{linear_to_srgb, srgb_to_linear};

#[inline]
fn reducible_dim(dim: u32) -> bool {
    dim == 1 || dim.is_multiple_of(2)
}

/// Encode a scene-linear coverage/intensity in `[0, 1]` to a `u8`, round-half-up.
#[inline]
fn unit_to_u8(v: f32) -> u8 {
    let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    (c * 255.0 + 0.5).floor() as u8
}

/// Reduce one footprint with alpha-weighted (premultiplied) colour.
fn premultiplied_average(texels: &[[u8; 4]], count: u32, space: ColorSpace) -> [u8; 4] {
    let count = count.max(1);
    let inv_count = 1.0 / count as f32;

    // Box-averaged coverage (raw integer domain, round half-up), matching the
    // straight-alpha path so coverage statistics are unchanged.
    let half = count / 2;
    let alpha_sum: u32 = texels.iter().map(|t| u32::from(t[3])).sum();
    let alpha = ((alpha_sum + half) / count) as u8;

    // Alpha-weighted colour sums. `weight_sum` is the associated-alpha
    // denominator; `linear` keeps the premultiplied colour in whichever domain
    // the un-premultiply/encode expects.
    let mut weight_sum = 0.0f32;
    let mut premult = [0.0f32; 3];
    // Unweighted fallback for a fully transparent footprint.
    let mut plain = [0.0f32; 3];

    for t in texels {
        let w = f32::from(t[3]) / 255.0;
        weight_sum += w;
        for c in 0..3 {
            let lin = match space {
                ColorSpace::Linear => f32::from(t[c]) / 255.0,
                ColorSpace::Srgb => srgb_to_linear(t[c]),
            };
            premult[c] += w * lin;
            plain[c] += lin;
        }
    }

    let mut rgb = [0u8; 4];
    for c in 0..3 {
        let lin = if weight_sum > 0.0 {
            premult[c] / weight_sum
        } else {
            plain[c] * inv_count
        };
        rgb[c] = match space {
            ColorSpace::Linear => unit_to_u8(lin),
            ColorSpace::Srgb => linear_to_srgb(lin),
        };
    }
    rgb[3] = alpha;
    rgb
}

/// Produce the next mip level by premultiplied-alpha box reduction, or `None`
/// when `src` is already `1x1` or has a non-reducible (odd, >1) dimension.
///
/// Use this for colour textures with meaningful transparency (cutout foliage,
/// decals, UI, particles). Data maps and fully opaque colour can use the plain
/// [`box_downsample`](super::box_downsample); for an opaque image the two agree.
#[must_use]
pub fn premultiplied_box_downsample(src: &Rgba8Image, space: ColorSpace) -> Option<Rgba8Image> {
    let (w, h) = (src.width(), src.height());
    if w == 1 && h == 1 {
        return None;
    }
    if !reducible_dim(w) || !reducible_dim(h) {
        return None;
    }

    let out_w = if w > 1 { w / 2 } else { 1 };
    let out_h = if h > 1 { h / 2 } else { 1 };
    let fx: u32 = if w > 1 { 2 } else { 1 };
    let fy: u32 = if h > 1 { 2 } else { 1 };
    let count = fx * fy;

    let mut out = Vec::with_capacity((out_w as usize) * (out_h as usize));
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut footprint = [[0u8; 4]; 4];
            let mut n = 0usize;
            for dy in 0..fy {
                for dx in 0..fx {
                    footprint[n] = src.texel(ox * 2 + dx, oy * 2 + dy);
                    n += 1;
                }
            }
            out.push(premultiplied_average(&footprint[..n], count, space));
        }
    }

    Rgba8Image::new(out_w, out_h, out)
}

/// Build the full premultiplied-alpha mip chain from `base` down to `1x1`,
/// including `base`. A non-power-of-two base yields a single-element chain.
#[must_use]
pub fn generate_mip_chain_premultiplied(base: Rgba8Image, space: ColorSpace) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = premultiplied_box_downsample(&current, space);
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
    use super::super::box_downsample;
    use alloc::vec;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Rgba8Image {
        Rgba8Image::new(w, h, vec![c; (w * h) as usize]).unwrap()
    }

    #[test]
    fn opaque_image_matches_plain_box_filter() {
        // With full coverage everywhere the premultiply denominator equals the
        // texel count, so the two reductions must agree bit-for-bit.
        let img = Rgba8Image::new(
            2,
            2,
            vec![
                [10, 40, 200, 255],
                [250, 30, 20, 255],
                [60, 220, 90, 255],
                [15, 15, 15, 255],
            ],
        )
        .unwrap();
        let a = premultiplied_box_downsample(&img, ColorSpace::Linear).unwrap();
        let b = box_downsample(&img, ColorSpace::Linear).unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn transparent_texels_do_not_bleed_colour() {
        // Three fully transparent black texels and one opaque red. Straight
        // alpha would average the blacks in and darken the red; premultiplied
        // keeps the only visible colour (red) intact.
        let img = Rgba8Image::new(
            2,
            2,
            vec![
                [255, 0, 0, 255],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        .unwrap();
        let out = premultiplied_box_downsample(&img, ColorSpace::Linear).unwrap().as_slice()[0];
        assert_eq!([out[0], out[1], out[2]], [255, 0, 0], "colour preserved");
        // Coverage is still the box average: 255/4 -> 64 (round half-up).
        assert_eq!(out[3], 64);
    }

    #[test]
    fn straight_box_would_darken_but_premultiplied_does_not() {
        let img = Rgba8Image::new(
            2,
            2,
            vec![
                [255, 0, 0, 255],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        .unwrap();
        let straight = box_downsample(&img, ColorSpace::Linear).unwrap().as_slice()[0];
        let premul = premultiplied_box_downsample(&img, ColorSpace::Linear).unwrap().as_slice()[0];
        assert!(premul[0] > straight[0], "premul red {} > straight {}", premul[0], straight[0]);
    }

    #[test]
    fn fully_transparent_footprint_is_finite_and_zero_alpha() {
        let img = solid(2, 2, [123, 45, 67, 0]);
        let out = premultiplied_box_downsample(&img, ColorSpace::Linear).unwrap().as_slice()[0];
        assert_eq!(out[3], 0, "coverage stays zero");
        // Colour falls back to the plain mean (no NaN / no divide-by-zero).
        assert_eq!([out[0], out[1], out[2]], [123, 45, 67]);
    }

    #[test]
    fn half_covered_edge_keeps_opaque_colour() {
        // Two opaque white + two transparent black: weighted colour = white,
        // coverage = 1/2.
        let img = Rgba8Image::new(
            2,
            2,
            vec![
                [255, 255, 255, 255],
                [255, 255, 255, 255],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        .unwrap();
        let out = premultiplied_box_downsample(&img, ColorSpace::Srgb).unwrap().as_slice()[0];
        assert_eq!([out[0], out[1], out[2]], [255, 255, 255], "white edge survives");
        assert_eq!(out[3], 128, "half coverage");
    }

    #[test]
    fn partial_coverage_weights_colour_toward_more_opaque_texel() {
        // One texel at a=255 red, one at a=85 (1/3) green; the weighted colour
        // should lean strongly red, unlike a straight 50/50 blend.
        let img = Rgba8Image::new(
            2,
            1,
            vec![[255, 0, 0, 255], [0, 255, 0, 85]],
        )
        .unwrap();
        let out = premultiplied_box_downsample(&img, ColorSpace::Linear).unwrap().as_slice()[0];
        assert!(out[0] > out[1], "red {} dominates green {}", out[0], out[1]);
        // Weight 1.0 vs ~0.333: red fraction = 1/(1+1/3) = 0.75.
        assert!(out[0] >= 185 && out[0] <= 196, "red = {}", out[0]);
    }

    #[test]
    fn solid_image_reduces_to_same_colour() {
        let img = solid(4, 4, [10, 20, 30, 200]);
        let mip = premultiplied_box_downsample(&img, ColorSpace::Linear).unwrap();
        assert!(mip.as_slice().iter().all(|t| *t == [10, 20, 30, 200]));
    }

    #[test]
    fn chain_dimensions_match_box_chain() {
        let chain = generate_mip_chain_premultiplied(solid(8, 8, [1, 2, 3, 128]), ColorSpace::Srgb);
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        assert!(premultiplied_box_downsample(&solid(3, 2, [1, 2, 3, 4]), ColorSpace::Linear).is_none());
    }

    #[test]
    fn one_by_one_does_not_reduce() {
        assert!(premultiplied_box_downsample(&solid(1, 1, [9, 9, 9, 9]), ColorSpace::Linear).is_none());
    }
}
