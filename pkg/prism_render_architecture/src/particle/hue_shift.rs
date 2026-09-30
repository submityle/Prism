//! Hue / saturation / value colour-adjustment contract for particle shading
//! (design §16, §30).
//!
//! Where [`super::color_temperature`] moves the neutral white point,
//! [`super::color_grade_lut`] applies an arbitrary creative cube, and
//! [`super::oklab_color`] works in a perceptual space, this module owns the
//! classic *cylindrical* colour models a VFX artist reaches for when they want
//! to "rotate the hue" of a spark or "desaturate" a smoke puff: `HSV`
//! (hue/saturation/value) and `HSL` (hue/saturation/lightness). It converts
//! `RGB` to and from both, rotates the hue, and scales saturation, value, and
//! lightness. It never reuses the sibling colour types; a [`Hsv`] is not a
//! white-balance gain and not a `LUT` sample.
//!
//! # No trigonometry, no transcendentals
//!
//! The hue of a colour is usually described as an angle on a wheel, but this
//! contract never touches `sin`/`cos`/`atan`. Instead it uses the standard
//! *six-sextant* piecewise-linear parameterisation: the hue is a value in
//! `[0, 6)` derived purely from which of the red/green/blue channels is largest
//! and the linear ratio of the other two against the chroma (max minus min).
//! Red sits at `0`, green at `2`, blue at `4`, and each unit spans one
//! `60`-degree sextant. Converting back is the inverse piecewise-linear fill.
//! The only floating-point intrinsics used are `f32::floor` and `f32::abs`
//! (plus `f32::max`/`f32::min` for the extrema), so the `CPU` reference stays
//! bit-reproducible against a future `GPU` evaluator, matching the determinism
//! contract of [`super::simulation`].
//!
//! The optional [`rotate_hue_yiq`] helper rotates hue in the `YIQ` chroma plane
//! instead; because a plane rotation genuinely needs a cosine and sine, the
//! caller must *supply* the `(cos, sin)` pair. This module multiplies the
//! matrices but never evaluates a trig function itself.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Number of meaningful scalar fields packed into the `std430` block (the three
/// [`Hsv`] channels `h`, `s`, `v`).
const HUE_FIELD_COUNT: usize = 3;

/// Byte size of the [`Hsv`] `std430` packing: three scalars padded up to a
/// single `vec4` slot so the block honours the 16-byte `std430` base alignment
/// a `GPU` kernel expects.
pub const HUE_SHIFT_STD430_SIZE: usize = HUE_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// A linear-space `RGB` colour triple.
///
/// Channels are nominally in `[0, 1]`; the conversions clamp their own outputs
/// but do not force the inputs, so an `HDR` value above `1` round-trips through
/// the value/lightness axis unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

/// A colour in the `HSV` (hue/saturation/value) cylindrical model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hsv {
    /// Hue in the six-sextant parameterisation, `[0, 6)`; red is `0`, green
    /// `2`, blue `4`.
    pub h: f32,
    /// Saturation in `[0, 1]`; `0` is a pure grey.
    pub s: f32,
    /// Value (brightness of the brightest channel) in `[0, 1]`.
    pub v: f32,
}

/// A colour in the `HSL` (hue/saturation/lightness) cylindrical model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hsl {
    /// Hue in the six-sextant parameterisation, `[0, 6)`; red is `0`, green
    /// `2`, blue `4`.
    pub h: f32,
    /// Saturation in `[0, 1]`; `0` is a pure grey.
    pub s: f32,
    /// Lightness (midpoint of the brightest and darkest channel) in `[0, 1]`.
    pub l: f32,
}

impl Rgb {
    /// Builds an `RGB` triple from explicit channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

impl Hsv {
    /// Builds an `HSV` triple from explicit channels.
    #[must_use]
    pub const fn new(h: f32, s: f32, v: f32) -> Self {
        Self { h, s, v }
    }

    /// Packs the three channels little-endian into a single `vec4` `std430`
    /// slot ([`HUE_SHIFT_STD430_SIZE`] bytes); the trailing scalar is padding
    /// the `GPU` kernel ignores.
    #[must_use]
    pub fn to_std430(&self) -> [u8; HUE_SHIFT_STD430_SIZE] {
        let mut bytes = [0u8; HUE_SHIFT_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.h.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.s.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.v.to_le_bytes());
        bytes
    }
}

impl Hsl {
    /// Builds an `HSL` triple from explicit channels.
    #[must_use]
    pub const fn new(h: f32, s: f32, l: f32) -> Self {
        Self { h, s, l }
    }
}

/// Total `std430` byte size of a storage buffer holding `count` packed [`Hsv`]
/// blocks, clamped up to a single element per the shared [`storage_bytes`] rule.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(HUE_SHIFT_STD430_SIZE, count)
}

/// Wraps a raw sextant hue into the canonical `[0, 6)` range using only
/// `f32::floor`, so a negative or overflowing hue folds back onto the wheel.
#[must_use]
fn wrap_sextant(h: f32) -> f32 {
    h - 6.0 * (h / 6.0).floor()
}

/// Derives the six-sextant hue in `[0, 6)` from the channels and chroma
/// (`delta` = max minus min). An achromatic colour (`delta <= 0`) has no defined
/// hue and returns `0`. Pure comparisons and ratios; no angle is ever formed.
#[must_use]
fn sextant_hue(r: f32, g: f32, b: f32, delta: f32) -> f32 {
    if delta <= 0.0 {
        return 0.0;
    }
    let raw = if r >= g && r >= b {
        (g - b) / delta
    } else if g >= b {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    if raw < 0.0 {
        raw + 6.0
    } else {
        raw
    }
}

/// Converts a linear `RGB` colour into `HSV` using the six-sextant scheme.
#[must_use]
pub fn rgb_to_hsv(rgb: Rgb) -> Hsv {
    let max = rgb.r.max(rgb.g).max(rgb.b);
    let min = rgb.r.min(rgb.g).min(rgb.b);
    let delta = max - min;
    let s = if max <= 0.0 { 0.0 } else { delta / max };
    let h = sextant_hue(rgb.r, rgb.g, rgb.b, delta);
    Hsv { h, s, v: max }
}

/// Converts an `HSV` colour back into linear `RGB`. Saturation and value are
/// clamped to `[0, 1]` and the hue is wrapped before the piecewise fill.
#[must_use]
pub fn hsv_to_rgb(hsv: Hsv) -> Rgb {
    let h = wrap_sextant(hsv.h);
    let s = hsv.s.clamp(0.0, 1.0);
    let v = hsv.v.clamp(0.0, 1.0);
    let f = h - h.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = if h < 1.0 {
        (v, t, p)
    } else if h < 2.0 {
        (q, v, p)
    } else if h < 3.0 {
        (p, v, t)
    } else if h < 4.0 {
        (p, q, v)
    } else if h < 5.0 {
        (t, p, v)
    } else {
        (v, p, q)
    };
    Rgb { r, g, b }
}

/// Converts a linear `RGB` colour into `HSL` using the six-sextant scheme.
#[must_use]
pub fn rgb_to_hsl(rgb: Rgb) -> Hsl {
    let max = rgb.r.max(rgb.g).max(rgb.b);
    let min = rgb.r.min(rgb.g).min(rgb.b);
    let delta = max - min;
    let l = (max + min) * 0.5;
    let denom = 1.0 - (2.0 * l - 1.0).abs();
    let s = if denom <= 0.0 { 0.0 } else { delta / denom };
    let h = sextant_hue(rgb.r, rgb.g, rgb.b, delta);
    Hsl { h, s, l }
}

/// Converts an `HSL` colour back into linear `RGB`. Saturation and lightness
/// are clamped to `[0, 1]` and the hue is wrapped before the piecewise fill.
#[must_use]
pub fn hsl_to_rgb(hsl: Hsl) -> Rgb {
    let h = wrap_sextant(hsl.h);
    let s = hsl.s.clamp(0.0, 1.0);
    let l = hsl.l.clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hmod2 = h - 2.0 * (h * 0.5).floor();
    let x = c * (1.0 - (hmod2 - 1.0).abs());
    let m = l - c * 0.5;
    let (r1, g1, b1) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    Rgb {
        r: r1 + m,
        g: g1 + m,
        b: b1 + m,
    }
}

/// Rotates the hue of a colour by `delta_sextants` (one unit is one `60`-degree
/// sextant) through the `HSV` model, wrapping the result back onto `[0, 6)`.
#[must_use]
pub fn shift_hue(rgb: Rgb, delta_sextants: f32) -> Rgb {
    let mut hsv = rgb_to_hsv(rgb);
    hsv.h = wrap_sextant(hsv.h + delta_sextants);
    hsv_to_rgb(hsv)
}

/// Scales `HSV` saturation by `factor` (clamped to `[0, 1]`); `0` yields a pure
/// grey, `1` is unchanged.
#[must_use]
pub fn adjust_saturation(rgb: Rgb, factor: f32) -> Rgb {
    let mut hsv = rgb_to_hsv(rgb);
    hsv.s = (hsv.s * factor).clamp(0.0, 1.0);
    hsv_to_rgb(hsv)
}

/// Scales `HSV` value (brightness) by `factor` (clamped to `[0, 1]`).
#[must_use]
pub fn adjust_value(rgb: Rgb, factor: f32) -> Rgb {
    let mut hsv = rgb_to_hsv(rgb);
    hsv.v = (hsv.v * factor).clamp(0.0, 1.0);
    hsv_to_rgb(hsv)
}

/// Scales `HSL` lightness by `factor` (clamped to `[0, 1]`).
#[must_use]
pub fn adjust_lightness(rgb: Rgb, factor: f32) -> Rgb {
    let mut hsl = rgb_to_hsl(rgb);
    hsl.l = (hsl.l * factor).clamp(0.0, 1.0);
    hsl_to_rgb(hsl)
}

/// Rotates hue in the `YIQ` chroma plane by the caller-supplied `(cos_d, sin_d)`
/// pair (this module never evaluates a trig function).
///
/// The colour is taken into `YIQ` with the classic `NTSC` matrix, the `I`/`Q`
/// chroma pair is rotated by the given cosine and sine, and the result is
/// mapped back to `RGB` and clamped to `[0, 1]`. Passing `(1, 0)` is the
/// identity rotation.
#[must_use]
pub fn rotate_hue_yiq(rgb: Rgb, cos_d: f32, sin_d: f32) -> Rgb {
    let y = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
    let i = 0.596 * rgb.r - 0.274 * rgb.g - 0.322 * rgb.b;
    let q = 0.211 * rgb.r - 0.523 * rgb.g + 0.312 * rgb.b;
    let i_rot = i * cos_d - q * sin_d;
    let q_rot = i * sin_d + q * cos_d;
    let r = y + 0.956 * i_rot + 0.621 * q_rot;
    let g = y - 0.272 * i_rot - 0.647 * q_rot;
    let b = y - 1.106 * i_rot + 1.703 * q_rot;
    Rgb {
        r: r.clamp(0.0, 1.0),
        g: g.clamp(0.0, 1.0),
        b: b.clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` comparisons used by the tests; direct
    /// `==` on floating point is intentionally avoided.
    const CMP_EPS: f32 = 1e-6;

    /// Looser tolerance for the `YIQ` round-trip, whose rounded `NTSC` matrix
    /// pair is not an exact inverse.
    const YIQ_EPS: f32 = 5e-3;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn rgb_close(a: Rgb, b: Rgb, eps: f32) -> bool {
        (a.r - b.r).abs() < eps && (a.g - b.g).abs() < eps && (a.b - b.b).abs() < eps
    }

    fn read_le_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(buf)
    }

    const SAMPLES: [Rgb; 6] = [
        Rgb::new(1.0, 0.0, 0.0),
        Rgb::new(0.0, 1.0, 0.0),
        Rgb::new(0.0, 0.0, 1.0),
        Rgb::new(0.6, 0.3, 0.2),
        Rgb::new(0.25, 0.75, 0.5),
        Rgb::new(0.8, 0.8, 0.2),
    ];

    #[test]
    fn red_maps_to_hue_zero() {
        let hsv = rgb_to_hsv(Rgb::new(1.0, 0.0, 0.0));
        assert!(approx(hsv.h, 0.0));
        assert!(approx(hsv.s, 1.0));
        assert!(approx(hsv.v, 1.0));
    }

    #[test]
    fn green_maps_to_hue_two() {
        let hsv = rgb_to_hsv(Rgb::new(0.0, 1.0, 0.0));
        assert!(approx(hsv.h, 2.0));
    }

    #[test]
    fn blue_maps_to_hue_four() {
        let hsv = rgb_to_hsv(Rgb::new(0.0, 0.0, 1.0));
        assert!(approx(hsv.h, 4.0));
    }

    #[test]
    fn hsv_value_is_channel_max() {
        let hsv = rgb_to_hsv(Rgb::new(0.2, 0.7, 0.4));
        assert!(approx(hsv.v, 0.7));
    }

    #[test]
    fn hsv_roundtrips_every_sample() {
        for &rgb in &SAMPLES {
            let back = hsv_to_rgb(rgb_to_hsv(rgb));
            assert!(rgb_close(back, rgb, 1e-4), "hsv roundtrip drift on {rgb:?}");
        }
    }

    #[test]
    fn hsl_roundtrips_every_sample() {
        for &rgb in &SAMPLES {
            let back = hsl_to_rgb(rgb_to_hsl(rgb));
            assert!(rgb_close(back, rgb, 1e-4), "hsl roundtrip drift on {rgb:?}");
        }
    }

    #[test]
    fn grey_has_zero_saturation_hsv() {
        let hsv = rgb_to_hsv(Rgb::new(0.4, 0.4, 0.4));
        assert!(approx(hsv.s, 0.0));
        assert!(approx(hsv.v, 0.4));
    }

    #[test]
    fn grey_has_zero_saturation_hsl() {
        let hsl = rgb_to_hsl(Rgb::new(0.4, 0.4, 0.4));
        assert!(approx(hsl.s, 0.0));
        assert!(approx(hsl.l, 0.4));
    }

    #[test]
    fn hsv_to_rgb_rebuilds_red() {
        let rgb = hsv_to_rgb(Hsv::new(0.0, 1.0, 1.0));
        assert!(rgb_close(rgb, Rgb::new(1.0, 0.0, 0.0), CMP_EPS));
    }

    #[test]
    fn hsl_to_rgb_rebuilds_blue() {
        let rgb = hsl_to_rgb(Hsl::new(4.0, 1.0, 0.5));
        assert!(rgb_close(rgb, Rgb::new(0.0, 0.0, 1.0), CMP_EPS));
    }

    #[test]
    fn hsl_lightness_of_white_and_black() {
        assert!(approx(rgb_to_hsl(Rgb::new(1.0, 1.0, 1.0)).l, 1.0));
        assert!(approx(rgb_to_hsl(Rgb::new(0.0, 0.0, 0.0)).l, 0.0));
    }

    #[test]
    fn full_turn_shift_is_identity() {
        for &rgb in &SAMPLES {
            let shifted = shift_hue(rgb, 6.0);
            assert!(
                rgb_close(shifted, rgb, 1e-4),
                "6-sextant shift moved {rgb:?}"
            );
        }
    }

    #[test]
    fn multiples_of_six_are_identity() {
        let base = Rgb::new(0.6, 0.3, 0.2);
        for turns in [0.0_f32, 6.0, 12.0, 18.0, 24.0, 30.0] {
            let shifted = shift_hue(base, turns);
            assert!(rgb_close(shifted, base, 1e-4), "turn {turns} drifted");
        }
    }

    #[test]
    fn shift_red_by_two_sextants_is_green() {
        let g = shift_hue(Rgb::new(1.0, 0.0, 0.0), 2.0);
        assert!(rgb_close(g, Rgb::new(0.0, 1.0, 0.0), 1e-4));
    }

    #[test]
    fn shift_stays_in_gamut() {
        let deltas = [0.3_f32, 1.0, 2.5, 3.7, 5.9, -1.4];
        for &rgb in &SAMPLES {
            for &d in &deltas {
                let out = shift_hue(rgb, d);
                assert!((0.0..=1.0).contains(&out.r), "r out of gamut");
                assert!((0.0..=1.0).contains(&out.g), "g out of gamut");
                assert!((0.0..=1.0).contains(&out.b), "b out of gamut");
            }
        }
    }

    #[test]
    fn saturation_zero_makes_grey() {
        let out = adjust_saturation(Rgb::new(0.9, 0.2, 0.1), 0.0);
        assert!(approx(out.r, out.g));
        assert!(approx(out.g, out.b));
    }

    #[test]
    fn saturation_factor_one_is_identity() {
        for &rgb in &SAMPLES {
            let out = adjust_saturation(rgb, 1.0);
            assert!(rgb_close(out, rgb, 1e-4));
        }
    }

    #[test]
    fn value_factor_scales_brightness() {
        let out = adjust_value(Rgb::new(1.0, 0.0, 0.0), 0.5);
        assert!(rgb_close(out, Rgb::new(0.5, 0.0, 0.0), CMP_EPS));
    }

    #[test]
    fn value_factor_one_is_identity() {
        for &rgb in &SAMPLES {
            let out = adjust_value(rgb, 1.0);
            assert!(rgb_close(out, rgb, 1e-4));
        }
    }

    #[test]
    fn lightness_factor_halves_grey() {
        let out = adjust_lightness(Rgb::new(0.8, 0.8, 0.8), 0.5);
        assert!(rgb_close(out, Rgb::new(0.4, 0.4, 0.4), 1e-4));
    }

    #[test]
    fn lightness_factor_one_is_identity() {
        for &rgb in &SAMPLES {
            let out = adjust_lightness(rgb, 1.0);
            assert!(rgb_close(out, rgb, 1e-4));
        }
    }

    #[test]
    fn yiq_identity_rotation_on_grey_is_exact() {
        let grey = Rgb::new(0.5, 0.5, 0.5);
        let out = rotate_hue_yiq(grey, 1.0, 0.0);
        assert!(rgb_close(out, grey, YIQ_EPS));
    }

    #[test]
    fn yiq_identity_rotation_roundtrips_colours() {
        for &rgb in &SAMPLES {
            let out = rotate_hue_yiq(rgb, 1.0, 0.0);
            assert!(
                rgb_close(out, rgb, YIQ_EPS),
                "yiq identity drift on {rgb:?}"
            );
        }
    }

    #[test]
    fn yiq_rotation_preserves_luma_of_grey() {
        let grey = Rgb::new(0.3, 0.3, 0.3);
        let out = rotate_hue_yiq(grey, 0.0, 1.0);
        assert!(rgb_close(out, grey, YIQ_EPS));
    }

    #[test]
    fn std430_size_is_sixteen() {
        assert_eq!(gpu_storage_bytes(1), HUE_SHIFT_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), VEC4_STRIDE);
    }

    #[test]
    fn std430_roundtrips_channels() {
        let hsv = Hsv::new(3.25, 0.6, 0.42);
        let bytes = hsv.to_std430();
        assert!(approx(read_le_f32(&bytes, 0), 3.25));
        assert!(approx(read_le_f32(&bytes, 4), 0.6));
        assert!(approx(read_le_f32(&bytes, 8), 0.42));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), VEC4_STRIDE);
        assert_eq!(gpu_storage_bytes(1), VEC4_STRIDE);
        assert_eq!(gpu_storage_bytes(8), 8 * VEC4_STRIDE);
    }
}
