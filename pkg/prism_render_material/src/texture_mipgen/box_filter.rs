//! Exact box-filter mip generation for decoded `RGBA8` images.
//!
//! Once a block-compressed texture has been decoded (see
//! [`texture_codec`](crate::texture_codec)), AAA pipelines build the mip chain
//! on the host when the authored asset ships only mip 0. This module provides
//! the classic 2x2 box reduction, done either in raw integer space (data maps:
//! normals, roughness, AO, height -- already linear) or gamma-correctly in
//! scene-linear space (colour/albedo), re-encoding through
//! [`srgb`](super::srgb). It is the CPU golden a GPU compute down-sampler is
//! validated against; the arithmetic is pure and deterministic, with no AI/ML.
//!
//! # Conventions
//! * Images are row-major `[u8; 4]` RGBA, `len == width * height`.
//! * Each dimension of a reducible level is either `1` or even; a dimension of
//!   `1` is carried through unchanged (matching the GL `max(1, dim >> 1)` rule
//!   for non-square power-of-two textures). Non-power-of-two reduction needs a
//!   polyphase box and is intentionally out of scope.
//! * [`ColorSpace::Linear`] averages each channel with round-half-up integer
//!   arithmetic; [`ColorSpace::Srgb`] linearises RGB, averages in linear light,
//!   re-encodes, and averages alpha in its raw integer domain.
//!
//! # References
//! * OpenGL 4.6 spec section 8.14.3 (automatic mipmap box reduction rule).
//! * IEC 61966-2-1 (sRGB) for the gamma-correct colour path.

use alloc::vec::Vec;

use super::srgb::{linear_to_srgb, srgb_to_linear};

/// Colour-space policy for the box reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    /// Average every channel directly in its stored integer domain. Use for
    /// data textures that are already linear (normals, roughness, AO, height).
    Linear,
    /// Linearise RGB through the sRGB EOTF, average in scene-linear light, then
    /// re-encode. Alpha stays linear. Use for colour/albedo.
    Srgb,
}

/// An owned, row-major `RGBA8` image used as a mip level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8Image {
    width: u32,
    height: u32,
    texels: Vec<[u8; 4]>,
}

impl Rgba8Image {
    /// Build an image, validating that `texels.len() == width * height` and
    /// that neither dimension is zero.
    #[must_use]
    pub fn new(width: u32, height: u32, texels: Vec<[u8; 4]>) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let expected = u64::from(width) * u64::from(height);
        if expected != texels.len() as u64 {
            return None;
        }
        Some(Self {
            width,
            height,
            texels,
        })
    }

    /// Image width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major texel view.
    #[must_use]
    pub fn as_slice(&self) -> &[[u8; 4]] {
        &self.texels
    }

    /// Fetch one texel; coordinates are clamped into range defensively.
    #[must_use]
    pub fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let cx = x.min(self.width - 1);
        let cy = y.min(self.height - 1);
        let idx = (cy as usize) * (self.width as usize) + (cx as usize);
        self.texels[idx]
    }
}

#[inline]
fn reducible_dim(dim: u32) -> bool {
    dim == 1 || dim % 2 == 0
}

/// Produce the next mip level by box-reducing `src`, or `None` when `src` is
/// already `1x1` or has a non-reducible (odd, >1) dimension.
#[must_use]
pub fn box_downsample(src: &Rgba8Image, space: ColorSpace) -> Option<Rgba8Image> {
    let (w, h) = (src.width, src.height);
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
            out.push(average(&footprint[..n], count, space));
        }
    }

    // Dimensions are validated above, so the invariant always holds.
    Rgba8Image::new(out_w, out_h, out)
}

/// Average a `count`-texel footprint under the chosen colour-space policy.
fn average(texels: &[[u8; 4]], count: u32, space: ColorSpace) -> [u8; 4] {
    debug_assert_eq!(texels.len() as u32, count);
    let count = count.max(1);
    let half = count / 2;

    let avg_u8 = |sum: u32| -> u8 { ((sum + half) / count) as u8 };

    let alpha = avg_u8(texels.iter().map(|t| u32::from(t[3])).sum());

    match space {
        ColorSpace::Linear => {
            let mut rgb = [0u8; 3];
            for (c, slot) in rgb.iter_mut().enumerate() {
                let sum: u32 = texels.iter().map(|t| u32::from(t[c])).sum();
                *slot = avg_u8(sum);
            }
            [rgb[0], rgb[1], rgb[2], alpha]
        }
        ColorSpace::Srgb => {
            let inv = 1.0 / count as f32;
            let mut rgb = [0u8; 3];
            for (c, slot) in rgb.iter_mut().enumerate() {
                let lin: f32 = texels.iter().map(|t| srgb_to_linear(t[c])).sum::<f32>() * inv;
                *slot = linear_to_srgb(lin);
            }
            [rgb[0], rgb[1], rgb[2], alpha]
        }
    }
}

/// Build the full mip chain from `base` down to `1x1`, including `base`.
///
/// Only power-of-two base dimensions reduce all the way to `1x1`; a
/// non-power-of-two base yields a single-element chain (just `base`) because
/// [`box_downsample`] refuses odd dimensions.
#[must_use]
pub fn generate_mip_chain(base: Rgba8Image, space: ColorSpace) -> Vec<Rgba8Image> {
    let mut chain = Vec::new();
    let mut current = base;
    loop {
        let next = box_downsample(&current, space);
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
    use alloc::vec;

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Rgba8Image {
        Rgba8Image::new(w, h, vec![c; (w * h) as usize]).unwrap()
    }

    #[test]
    fn new_rejects_mismatched_length() {
        assert!(Rgba8Image::new(2, 2, vec![[0u8; 4]; 3]).is_none());
        assert!(Rgba8Image::new(0, 4, vec![]).is_none());
    }

    #[test]
    fn solid_image_reduces_to_same_colour() {
        let img = solid(4, 4, [10, 20, 30, 40]);
        let mip = box_downsample(&img, ColorSpace::Linear).unwrap();
        assert_eq!((mip.width(), mip.height()), (2, 2));
        assert!(mip.as_slice().iter().all(|t| *t == [10, 20, 30, 40]));
    }

    #[test]
    fn linear_average_rounds_half_up() {
        // One 2x2 block of {0, 255, 0, 255} per channel -> 510/4 = 127.5 -> 128.
        let img = Rgba8Image::new(
            2,
            2,
            vec![
                [0, 0, 0, 0],
                [255, 255, 255, 255],
                [0, 0, 0, 0],
                [255, 255, 255, 255],
            ],
        )
        .unwrap();
        let mip = box_downsample(&img, ColorSpace::Linear).unwrap();
        assert_eq!(mip.as_slice(), &[[128, 128, 128, 128]]);
    }

    #[test]
    fn srgb_average_is_brighter_than_naive_integer_average() {
        // Two black + two white sRGB texels average to 0.5 in linear light,
        // which re-encodes to ~188 -- far above the naive integer 128.
        let img = Rgba8Image::new(
            2,
            2,
            vec![
                [0, 0, 0, 255],
                [255, 255, 255, 255],
                [0, 0, 0, 255],
                [255, 255, 255, 255],
            ],
        )
        .unwrap();
        let mip = box_downsample(&img, ColorSpace::Srgb).unwrap();
        let out = mip.as_slice()[0];
        assert!(out[0] >= 185 && out[0] <= 190, "sRGB avg = {}", out[0]);
        // Alpha stays in its raw integer domain.
        assert_eq!(out[3], 255);
    }

    #[test]
    fn non_square_pot_carries_dimension_of_one() {
        // 4x1 reduces along width only: 4x1 -> 2x1 -> 1x1.
        let img = solid(4, 1, [50, 60, 70, 80]);
        let m1 = box_downsample(&img, ColorSpace::Linear).unwrap();
        assert_eq!((m1.width(), m1.height()), (2, 1));
        let m2 = box_downsample(&m1, ColorSpace::Linear).unwrap();
        assert_eq!((m2.width(), m2.height()), (1, 1));
        assert!(box_downsample(&m2, ColorSpace::Linear).is_none());
    }

    #[test]
    fn odd_dimension_is_not_reducible() {
        let img = solid(3, 2, [1, 2, 3, 4]);
        assert!(box_downsample(&img, ColorSpace::Linear).is_none());
    }

    #[test]
    fn mip_chain_has_log2_plus_one_levels() {
        // 8x8 -> 8,4,2,1 = 4 levels.
        let chain = generate_mip_chain(solid(8, 8, [1, 1, 1, 255]), ColorSpace::Linear);
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 8), (4, 4), (2, 2), (1, 1)]);
    }

    #[test]
    fn mip_chain_non_square_pot() {
        // 8x2 -> 8x2, 4x1, 2x1, 1x1.
        let chain = generate_mip_chain(solid(8, 2, [9, 9, 9, 9]), ColorSpace::Linear);
        let dims: Vec<(u32, u32)> = chain.iter().map(|i| (i.width(), i.height())).collect();
        assert_eq!(dims, vec![(8, 2), (4, 1), (2, 1), (1, 1)]);
    }

    #[test]
    fn npot_base_yields_single_level() {
        let chain = generate_mip_chain(solid(3, 3, [0, 0, 0, 0]), ColorSpace::Linear);
        assert_eq!(chain.len(), 1);
        assert_eq!((chain[0].width(), chain[0].height()), (3, 3));
    }
}
