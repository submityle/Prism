//! `BT.601` / `BT.709` `RGB` <-> `YCbCr` video colour transforms for the
//! particle capture and video-encode path (design §21).
//!
//! `YCbCr` splits a colour into one *`luma`* channel (`Y`) and two
//! *`chroma`* channels (`Cb`, blue-difference; `Cr`, red-difference) using the
//! standard-defined `luma` weights `Kr`/`Kg`/`Kb`. Broadcast and video codecs
//! (`H.264`, `HEVC`, `AV1`) always encode `YCbCr` rather than raw `RGB`, both
//! because the eye is far more sensitive to `luma` than to `chroma` and because
//! `chroma` can be subsampled (`4:2:0`, `4:2:2`) with little visible loss. This
//! module owns the `CPU`-verifiable maths of that transform, its `8`-bit
//! quantisation, its `chroma` subsampling, and the `std430` block a `GPU`
//! upload pass binds.
//!
//! # Two coefficient sets
//!
//! [`Coefficients::Bt601`] uses the standard-definition weights
//! (`Kr = 0.299`, `Kg = 0.587`, `Kb = 0.114`); [`Coefficients::Bt709`] uses the
//! high-definition weights (`Kr = 0.2126`, `Kg = 0.7152`, `Kb = 0.0722`). The
//! two produce a *different* `Y` for the same `RGB`, so a round trip must use
//! the same coefficient set it encoded with.
//!
//! # Two signal ranges
//!
//! [`Range::Full`] keeps `Y` in `[0, 1]` and `Cb`/`Cr` in `[-0.5, 0.5]`
//! (the `JFIF` / `sRGB`-desktop convention). [`Range::Limited`] applies the
//! `8`-bit studio-swing scale and offset so `Y` lands in `[16/255, 235/255]`
//! and `Cb`/`Cr` land in `[16/255, 240/255]` (the broadcast convention). The
//! `Limited` variant therefore stores its `chroma` centred on `128/255` rather
//! than on zero.
//!
//! # Relationship to sibling modules
//!
//! This module is deliberately *not* [`super::rgb_ycocg`]: that module owns the
//! `YCoCg` / `YCoCg-R` basis used by the temporal resolver, whose lifting
//! variant is bit-exact reversible on integers. `YCbCr` here is a pure linear
//! matrix keyed by a broadcast `luma`-weight standard, targets a video encoder
//! rather than a `TAA` history clamp, and shares *none* of that module's types
//! (it defines its own [`Rgb`] and [`YCbCr`]).
//!
//! # Determinism
//!
//! Every routine touches only `+ - * /`, integer `div_ceil`, [`f32::clamp`],
//! and [`f32::floor`] (for `8`-bit rounding). No transcendental function
//! (`sin` / `cos` / `exp` / `ln` / `powf`) and no `f32::round` / `f32::ceil` is
//! ever called, so a future `GPU` evaluation reproduces the `CPU` result.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Byte size of one `YCbCr` sample in the `std430` layout: a single `vec4`
/// slot (three populated floats plus one zero-padding float).
pub const YCBCR_STD430_SIZE: usize = VEC4_STRIDE;

/// `Y` studio-swing offset (`16/255`) applied in [`Range::Limited`].
const LIMITED_Y_OFFSET: f32 = 16.0 / 255.0;
/// `Y` studio-swing scale (`219/255`) applied in [`Range::Limited`].
const LIMITED_Y_SCALE: f32 = 219.0 / 255.0;
/// `Cb`/`Cr` studio-swing offset (`128/255`) applied in [`Range::Limited`].
const LIMITED_C_OFFSET: f32 = 128.0 / 255.0;
/// `Cb`/`Cr` studio-swing scale (`224/255`) applied in [`Range::Limited`].
const LIMITED_C_SCALE: f32 = 224.0 / 255.0;

/// A linear `RGB` colour triple.
///
/// Values are conventionally normalised to `[0, 1]` but the matrix maths is
/// defined for any `f32`.
#[derive(Clone, Copy, Debug)]
pub struct Rgb {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

/// A `YCbCr` colour triple (`luma` plus blue- and red-difference `chroma`).
///
/// The exact numeric range of each field depends on the [`Range`] it was
/// produced with: [`Range::Full`] centres `Cb`/`Cr` on zero, while
/// [`Range::Limited`] centres them on `128/255`.
#[derive(Clone, Copy, Debug)]
pub struct YCbCr {
    /// `luma` channel.
    pub y: f32,
    /// Blue-difference `chroma` channel.
    pub cb: f32,
    /// Red-difference `chroma` channel.
    pub cr: f32,
}

/// The `luma`-weight standard selecting the transform matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Coefficients {
    /// Standard-definition `BT.601` (`Kr = 0.299`, `Kg = 0.587`,
    /// `Kb = 0.114`).
    Bt601,
    /// High-definition `BT.709` (`Kr = 0.2126`, `Kg = 0.7152`,
    /// `Kb = 0.0722`).
    Bt709,
}

impl Coefficients {
    /// Returns the `(Kr, Kg, Kb)` `luma` weights for this standard; the three
    /// always sum to `1`.
    #[must_use]
    pub const fn weights(self) -> (f32, f32, f32) {
        match self {
            Coefficients::Bt601 => (0.299, 0.587, 0.114),
            Coefficients::Bt709 => (0.2126, 0.7152, 0.0722),
        }
    }
}

/// The signal range (full swing versus broadcast studio swing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Range {
    /// Full swing: `Y` in `[0, 1]`, `Cb`/`Cr` in `[-0.5, 0.5]`.
    Full,
    /// Studio swing: `Y` in `[16/255, 235/255]`, `Cb`/`Cr` in
    /// `[16/255, 240/255]`.
    Limited,
}

impl Rgb {
    /// Builds an `RGB` triple.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

impl YCbCr {
    /// Builds a `YCbCr` triple.
    #[must_use]
    pub const fn new(y: f32, cb: f32, cr: f32) -> Self {
        Self { y, cb, cr }
    }

    /// Packs this sample into its `std430` `vec4` slot ([`YCBCR_STD430_SIZE`]
    /// bytes): `Y`, `Cb`, `Cr` as little-endian `f32`, then a zero-padding
    /// float so the next sample starts on a `vec4` boundary.
    #[must_use]
    pub fn to_std430(&self) -> [u8; YCBCR_STD430_SIZE] {
        let mut bytes = [0u8; YCBCR_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.y.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.cb.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.cr.to_le_bytes());
        // bytes[12..16] stay zero: the `vec4` padding slot.
        bytes
    }
}

/// Forward transform: `RGB` -> `YCbCr` under the given `coeff` and `range`.
///
/// In [`Range::Full`], `Y = Kr*R + Kg*G + Kb*B`,
/// `Cb = (B - Y) / (2*(1 - Kb))`, `Cr = (R - Y) / (2*(1 - Kr))`. In
/// [`Range::Limited`], the full-swing result is then scaled and offset into the
/// `8`-bit studio range.
#[must_use]
pub fn rgb_to_ycbcr(rgb: &Rgb, coeff: Coefficients, range: Range) -> YCbCr {
    let (kr, kg, kb) = coeff.weights();
    let y = kr * rgb.r + kg * rgb.g + kb * rgb.b;
    let cb = (rgb.b - y) / (2.0 * (1.0 - kb));
    let cr = (rgb.r - y) / (2.0 * (1.0 - kr));
    match range {
        Range::Full => YCbCr { y, cb, cr },
        Range::Limited => YCbCr {
            y: LIMITED_Y_OFFSET + LIMITED_Y_SCALE * y,
            cb: LIMITED_C_OFFSET + LIMITED_C_SCALE * cb,
            cr: LIMITED_C_OFFSET + LIMITED_C_SCALE * cr,
        },
    }
}

/// Inverse transform: `YCbCr` -> `RGB` under the given `coeff` and `range`.
///
/// The inverse of [`rgb_to_ycbcr`]: for any `RGB` triple the round trip
/// reproduces the input to within `f32` tolerance. In [`Range::Limited`] the
/// studio scale and offset are removed before the linear matrix is applied.
#[must_use]
pub fn ycbcr_to_rgb(ycbcr: &YCbCr, coeff: Coefficients, range: Range) -> Rgb {
    let (kr, kg, kb) = coeff.weights();
    let (y, cb, cr) = match range {
        Range::Full => (ycbcr.y, ycbcr.cb, ycbcr.cr),
        Range::Limited => (
            (ycbcr.y - LIMITED_Y_OFFSET) / LIMITED_Y_SCALE,
            (ycbcr.cb - LIMITED_C_OFFSET) / LIMITED_C_SCALE,
            (ycbcr.cr - LIMITED_C_OFFSET) / LIMITED_C_SCALE,
        ),
    };
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = y - (2.0 * kr * (1.0 - kr) / kg) * cr - (2.0 * kb * (1.0 - kb) / kg) * cb;
    Rgb { r, g, b }
}

/// Rounds a digital-scale `f32` to the nearest `[0, 255]` byte without any
/// transcendental call: add `0.5`, [`f32::floor`], then [`f32::clamp`].
fn round_clamp_u8(digital: f32) -> u8 {
    let rounded = (digital + 0.5).floor();
    let clamped = rounded.clamp(0.0, 255.0);
    // `clamped` is integer-valued and inside `[0, 255]`, so the narrowing is
    // exact; `try_from` guards the (unreachable) out-of-domain case.
    u8::try_from(clamped as i32).unwrap_or(0)
}

/// Quantises a `YCbCr` sample to three `8`-bit codewords.
///
/// In [`Range::Full`] the `chroma` is biased by `128` before rounding (so `0`
/// `chroma` maps to codeword `128`); in [`Range::Limited`] the channels already
/// carry their studio offset, so each is simply scaled by `255`.
#[must_use]
pub fn quantize_8bit(ycbcr: &YCbCr, range: Range) -> [u8; 3] {
    match range {
        Range::Full => [
            round_clamp_u8(ycbcr.y * 255.0),
            round_clamp_u8(ycbcr.cb * 255.0 + 128.0),
            round_clamp_u8(ycbcr.cr * 255.0 + 128.0),
        ],
        Range::Limited => [
            round_clamp_u8(ycbcr.y * 255.0),
            round_clamp_u8(ycbcr.cb * 255.0),
            round_clamp_u8(ycbcr.cr * 255.0),
        ],
    }
}

/// Dequantises three `8`-bit codewords back into a `YCbCr` sample, inverting
/// [`quantize_8bit`] for the same [`Range`].
#[must_use]
pub fn dequantize_8bit(codewords: [u8; 3], range: Range) -> YCbCr {
    let y = f32::from(codewords[0]) / 255.0;
    match range {
        Range::Full => YCbCr {
            y,
            cb: (f32::from(codewords[1]) - 128.0) / 255.0,
            cr: (f32::from(codewords[2]) - 128.0) / 255.0,
        },
        Range::Limited => YCbCr {
            y,
            cb: f32::from(codewords[1]) / 255.0,
            cr: f32::from(codewords[2]) / 255.0,
        },
    }
}

/// `4:2:0` `chroma` subsampling by simple `2x2` averaging.
///
/// Keeps every `luma` tap at full resolution and replaces each `2x2` block of
/// `chroma` with its average, returning `(Y full-resolution, Cb quarter-res,
/// Cr quarter-res)`. `width` and `height` are expected to be even; the
/// `chroma` dimensions are computed with [`usize::div_ceil`] so a boundary
/// block that is only partially covered still averages the taps it contains.
#[must_use]
pub fn subsample_420(
    pixels: &[YCbCr],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let luma: Vec<f32> = pixels.iter().map(|p| p.y).collect();
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut cb = Vec::with_capacity(cw * ch);
    let mut cr = Vec::with_capacity(cw * ch);
    for by in 0..ch {
        for bx in 0..cw {
            let mut sum_cb = 0.0_f32;
            let mut sum_cr = 0.0_f32;
            let mut count = 0.0_f32;
            for dy in 0..2 {
                let y = by * 2 + dy;
                for dx in 0..2 {
                    let x = bx * 2 + dx;
                    if y < height && x < width {
                        let idx = y * width + x;
                        sum_cb += pixels[idx].cb;
                        sum_cr += pixels[idx].cr;
                        count += 1.0;
                    }
                }
            }
            cb.push(sum_cb / count);
            cr.push(sum_cr / count);
        }
    }
    (luma, cb, cr)
}

/// `4:2:2` `chroma` subsampling by horizontal pair averaging.
///
/// Keeps every `luma` tap and every `chroma` row at full resolution but halves
/// the horizontal `chroma` resolution, returning `(Y full-resolution, Cb
/// half-width, Cr half-width)`. `width` is expected to be even; the `chroma`
/// width is computed with [`usize::div_ceil`] so an odd trailing column still
/// averages the tap it contains.
#[must_use]
pub fn subsample_422(
    pixels: &[YCbCr],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let luma: Vec<f32> = pixels.iter().map(|p| p.y).collect();
    let cw = width.div_ceil(2);
    let mut cb = Vec::with_capacity(cw * height);
    let mut cr = Vec::with_capacity(cw * height);
    for y in 0..height {
        for bx in 0..cw {
            let mut sum_cb = 0.0_f32;
            let mut sum_cr = 0.0_f32;
            let mut count = 0.0_f32;
            for dx in 0..2 {
                let x = bx * 2 + dx;
                if x < width {
                    let idx = y * width + x;
                    sum_cb += pixels[idx].cb;
                    sum_cr += pixels[idx].cr;
                    count += 1.0;
                }
            }
            cb.push(sum_cb / count);
            cr.push(sum_cr / count);
        }
    }
    (luma, cb, cr)
}

/// Total `std430` byte size of a `GPU` storage buffer holding `count` `YCbCr`
/// samples, clamped up to a single element for the empty case.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(YCBCR_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the test-only `f32` comparisons.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn roundtrip_close(c: Rgb, coeff: Coefficients, range: Range) -> bool {
        let back = ycbcr_to_rgb(&rgb_to_ycbcr(&c, coeff, range), coeff, range);
        (back.r - c.r).abs() < 1e-4 && (back.g - c.g).abs() < 1e-4 && (back.b - c.b).abs() < 1e-4
    }

    fn read_le_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(buf)
    }

    const SAMPLES: [Rgb; 6] = [
        Rgb::new(0.0, 0.0, 0.0),
        Rgb::new(1.0, 1.0, 1.0),
        Rgb::new(0.2, 0.4, 0.6),
        Rgb::new(0.9, 0.1, 0.3),
        Rgb::new(0.05, 0.95, 0.5),
        Rgb::new(0.7, 0.7, 0.2),
    ];

    #[test]
    fn bt601_full_roundtrip_assorted() {
        for &c in &SAMPLES {
            assert!(roundtrip_close(c, Coefficients::Bt601, Range::Full));
        }
    }

    #[test]
    fn bt709_full_roundtrip_assorted() {
        for &c in &SAMPLES {
            assert!(roundtrip_close(c, Coefficients::Bt709, Range::Full));
        }
    }

    #[test]
    fn bt601_limited_roundtrip_assorted() {
        for &c in &SAMPLES {
            assert!(roundtrip_close(c, Coefficients::Bt601, Range::Limited));
        }
    }

    #[test]
    fn bt709_limited_roundtrip_assorted() {
        for &c in &SAMPLES {
            assert!(roundtrip_close(c, Coefficients::Bt709, Range::Limited));
        }
    }

    #[test]
    fn bt601_grey_has_zero_chroma_full() {
        for &v in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let yc = rgb_to_ycbcr(&Rgb::new(v, v, v), Coefficients::Bt601, Range::Full);
            assert!(approx(yc.cb, 0.0));
            assert!(approx(yc.cr, 0.0));
            assert!(approx(yc.y, v));
        }
    }

    #[test]
    fn bt709_grey_has_zero_chroma_full() {
        for &v in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let yc = rgb_to_ycbcr(&Rgb::new(v, v, v), Coefficients::Bt709, Range::Full);
            assert!(approx(yc.cb, 0.0));
            assert!(approx(yc.cr, 0.0));
            assert!(approx(yc.y, v));
        }
    }

    #[test]
    fn full_range_white_luma_is_one() {
        let yc = rgb_to_ycbcr(&Rgb::new(1.0, 1.0, 1.0), Coefficients::Bt709, Range::Full);
        assert!(approx(yc.y, 1.0));
    }

    #[test]
    fn full_range_channel_bounds() {
        for &c in &SAMPLES {
            let yc = rgb_to_ycbcr(&c, Coefficients::Bt709, Range::Full);
            assert!((0.0..=1.0).contains(&yc.y));
            assert!((-0.5..=0.5).contains(&yc.cb));
            assert!((-0.5..=0.5).contains(&yc.cr));
        }
    }

    #[test]
    fn limited_range_channel_bounds() {
        let lo_y = 16.0 / 255.0;
        let hi_y = 235.0 / 255.0;
        let lo_c = 16.0 / 255.0;
        let hi_c = 240.0 / 255.0;
        for &c in &SAMPLES {
            let yc = rgb_to_ycbcr(&c, Coefficients::Bt601, Range::Limited);
            assert!((lo_y - CMP_EPS..=hi_y + CMP_EPS).contains(&yc.y));
            assert!((lo_c - CMP_EPS..=hi_c + CMP_EPS).contains(&yc.cb));
            assert!((lo_c - CMP_EPS..=hi_c + CMP_EPS).contains(&yc.cr));
        }
    }

    #[test]
    fn coefficients_differ_between_standards() {
        let c = Rgb::new(0.2, 0.4, 0.6);
        let y601 = rgb_to_ycbcr(&c, Coefficients::Bt601, Range::Full).y;
        let y709 = rgb_to_ycbcr(&c, Coefficients::Bt709, Range::Full).y;
        assert!((y601 - y709).abs() > 1e-3);
    }

    #[test]
    fn coefficient_weights_sum_to_one() {
        for &coeff in &[Coefficients::Bt601, Coefficients::Bt709] {
            let (kr, kg, kb) = coeff.weights();
            assert!(approx(kr + kg + kb, 1.0));
        }
    }

    #[test]
    fn quantize_dequantize_full_roundtrip() {
        for &c in &SAMPLES {
            let yc = rgb_to_ycbcr(&c, Coefficients::Bt709, Range::Full);
            let bytes = quantize_8bit(&yc, Range::Full);
            let back = dequantize_8bit(bytes, Range::Full);
            assert!((back.y - yc.y).abs() < 1.0 / 255.0 + CMP_EPS);
            assert!((back.cb - yc.cb).abs() < 1.0 / 255.0 + CMP_EPS);
            assert!((back.cr - yc.cr).abs() < 1.0 / 255.0 + CMP_EPS);
        }
    }

    #[test]
    fn quantize_dequantize_limited_roundtrip() {
        for &c in &SAMPLES {
            let yc = rgb_to_ycbcr(&c, Coefficients::Bt601, Range::Limited);
            let bytes = quantize_8bit(&yc, Range::Limited);
            let back = dequantize_8bit(bytes, Range::Limited);
            assert!((back.y - yc.y).abs() < 1.0 / 255.0 + CMP_EPS);
            assert!((back.cb - yc.cb).abs() < 1.0 / 255.0 + CMP_EPS);
            assert!((back.cr - yc.cr).abs() < 1.0 / 255.0 + CMP_EPS);
        }
    }

    #[test]
    fn quantize_full_grey_chroma_is_128() {
        let yc = rgb_to_ycbcr(&Rgb::new(0.5, 0.5, 0.5), Coefficients::Bt709, Range::Full);
        let bytes = quantize_8bit(&yc, Range::Full);
        assert_eq!(bytes[1], 128);
        assert_eq!(bytes[2], 128);
    }

    #[test]
    fn quantize_limited_luma_hits_studio_endpoints() {
        let black = rgb_to_ycbcr(
            &Rgb::new(0.0, 0.0, 0.0),
            Coefficients::Bt709,
            Range::Limited,
        );
        let white = rgb_to_ycbcr(
            &Rgb::new(1.0, 1.0, 1.0),
            Coefficients::Bt709,
            Range::Limited,
        );
        assert_eq!(quantize_8bit(&black, Range::Limited)[0], 16);
        assert_eq!(quantize_8bit(&white, Range::Limited)[0], 235);
    }

    #[test]
    fn quantize_saturates_out_of_range() {
        let hot = YCbCr::new(2.0, 5.0, -5.0);
        let bytes = quantize_8bit(&hot, Range::Full);
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[1], 255);
        assert_eq!(bytes[2], 0);
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(YCBCR_STD430_SIZE, VEC4_STRIDE);
        let bytes = YCbCr::new(0.5, 0.1, -0.2).to_std430();
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn std430_roundtrips_the_triple() {
        let yc = YCbCr::new(0.42, -0.3, 0.17);
        let bytes = yc.to_std430();
        assert!(approx(read_le_f32(&bytes, 0), yc.y));
        assert!(approx(read_le_f32(&bytes, 4), yc.cb));
        assert!(approx(read_le_f32(&bytes, 8), yc.cr));
    }

    #[test]
    fn std430_padding_slot_is_zero() {
        let bytes = YCbCr::new(0.9, 0.4, -0.5).to_std430();
        assert!(approx(read_le_f32(&bytes, 12), 0.0));
    }

    #[test]
    fn std430_channel_layout_is_little_endian() {
        let yc = YCbCr::new(1.0, 2.0, 3.0);
        let bytes = yc.to_std430();
        assert!(approx(read_le_f32(&bytes, 0), 1.0));
        assert!(approx(read_le_f32(&bytes, 4), 2.0));
        assert!(approx(read_le_f32(&bytes, 8), 3.0));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), 16);
        assert_eq!(gpu_storage_bytes(1), 16);
        assert_eq!(gpu_storage_bytes(10), 160);
    }

    #[test]
    fn subsample_420_dimensions() {
        let px = [YCbCr::new(0.5, 0.1, -0.1); 16];
        let (luma, cb, cr) = subsample_420(&px, 4, 4);
        assert_eq!(luma.len(), 16);
        assert_eq!(cb.len(), 4);
        assert_eq!(cr.len(), 4);
    }

    #[test]
    fn subsample_420_averages_block() {
        let px = [
            YCbCr::new(0.1, 0.4, -0.2),
            YCbCr::new(0.2, 0.0, 0.2),
            YCbCr::new(0.3, -0.4, 0.6),
            YCbCr::new(0.4, 0.8, -0.6),
        ];
        let (luma, cb, cr) = subsample_420(&px, 2, 2);
        assert_eq!(luma.len(), 4);
        assert_eq!(cb.len(), 1);
        assert!(approx(cb[0], (0.4 + 0.0 - 0.4 + 0.8) * 0.25));
        assert!(approx(cr[0], (-0.2 + 0.2 + 0.6 - 0.6) * 0.25));
    }

    #[test]
    fn subsample_420_preserves_full_luma() {
        let px = [
            YCbCr::new(0.11, 0.0, 0.0),
            YCbCr::new(0.22, 0.0, 0.0),
            YCbCr::new(0.33, 0.0, 0.0),
            YCbCr::new(0.44, 0.0, 0.0),
        ];
        let (luma, _cb, _cr) = subsample_420(&px, 2, 2);
        assert!(approx(luma[0], 0.11));
        assert!(approx(luma[1], 0.22));
        assert!(approx(luma[2], 0.33));
        assert!(approx(luma[3], 0.44));
    }

    #[test]
    fn subsample_422_dimensions() {
        let px = [YCbCr::new(0.5, 0.1, -0.1); 8];
        let (luma, cb, cr) = subsample_422(&px, 4, 2);
        assert_eq!(luma.len(), 8);
        assert_eq!(cb.len(), 4);
        assert_eq!(cr.len(), 4);
    }

    #[test]
    fn subsample_422_averages_horizontal_pairs() {
        let px = [
            YCbCr::new(0.1, 0.2, -0.2),
            YCbCr::new(0.2, 0.6, 0.4),
            YCbCr::new(0.3, -0.4, 0.0),
            YCbCr::new(0.4, 0.8, 0.2),
        ];
        let (_luma, cb, cr) = subsample_422(&px, 4, 1);
        assert_eq!(cb.len(), 2);
        assert!(approx(cb[0], (0.2 + 0.6) * 0.5));
        assert!(approx(cb[1], (-0.4 + 0.8) * 0.5));
        assert!(approx(cr[0], (-0.2 + 0.4) * 0.5));
        assert!(approx(cr[1], (0.0 + 0.2) * 0.5));
    }

    #[test]
    fn subsample_uniform_chroma_is_preserved() {
        let px = [YCbCr::new(0.3, 0.25, -0.15); 16];
        let (_l, cb, cr) = subsample_420(&px, 4, 4);
        for (&b, &r) in cb.iter().zip(cr.iter()) {
            assert!(approx(b, 0.25));
            assert!(approx(r, -0.15));
        }
    }
}
