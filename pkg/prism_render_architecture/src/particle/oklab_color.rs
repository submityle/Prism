//! `OKLab` / `OKLCh` perceptual colour space contracts (Björn Ottosson, 2020).
//!
//! `OKLab` is a perceptually uniform colour space derived from a cone-response
//! (`LMS`) intermediate: equal Euclidean steps in `OKLab` are close to equal
//! perceived colour differences, which makes it the right space for particle
//! colour interpolation, hue rotation, and gradient sampling. `OKLCh` is the
//! cylindrical form of `OKLab`, exposing lightness, chroma, and hue; this
//! contract stores hue as a unit vector `(h_cos, h_sin)` rather than an angle so
//! every operation stays in pure multiply-add arithmetic and never calls a
//! trigonometric function.
//!
//! This module is deliberately walled off from the sibling tone/grade contracts:
//! `tonemap` owns the `sRGB` / `ACES` transfer curves, `color_temperature` owns
//! the kelvin white-balance gains, and `color_grade_lut` owns the 3D grading
//! table. None of their types are reused here — the only shared dependency is
//! [`crate::particle::gpu_layout`] for the `std430` byte-size rule.
//!
//! Everything is hand-rolled: the forward transform needs a cube root, which is
//! computed by [`cbrt_newton`] using Newton's method (no `cbrt` / `powf`), and
//! the inverse transform cubes its terms with plain integer-count multiplies.
//! The only floating-point library calls are `f32::sqrt` (chroma magnitude) and
//! the multiply/add/divide primitives.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Chroma magnitudes at or below this threshold are treated as achromatic, so a
/// near-grey colour maps to a canonical hue instead of an ill-conditioned
/// division.
const CHROMA_FLOOR: f32 = 1e-6;

/// The real cube root of `2`, used to fold a residual `2^1` factor back into a
/// [`cbrt_newton`] result after the exponent is split into thirds.
const CBRT_2: f32 = 1.259_921_1;

/// The real cube root of `4`, used to fold a residual `2^2` factor back into a
/// [`cbrt_newton`] result after the exponent is split into thirds.
const CBRT_4: f32 = 1.587_401_1;

/// A colour in the linear (un-encoded) `sRGB` working space.
///
/// These are the same linear `RGB` values a renderer blends and lights with,
/// *before* the `sRGB` opto-electronic transfer function is applied for display.
/// Feeding gamma-encoded values in would produce a wrong `OKLab` result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearSrgb {
    /// Linear red channel.
    pub r: f32,
    /// Linear green channel.
    pub g: f32,
    /// Linear blue channel.
    pub b: f32,
}

impl LinearSrgb {
    /// Builds a linear `sRGB` triple from its three channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

/// A colour in the `OKLab` perceptual space: lightness plus two opponent axes.
///
/// `l` is perceptual lightness (roughly `0` at black, `1` at reference white);
/// `a` is the green-red axis (negative toward green, positive toward red); `b`
/// is the blue-yellow axis (negative toward blue, positive toward yellow). A
/// neutral grey has `a` and `b` both near zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkLab {
    /// Perceptual lightness.
    pub l: f32,
    /// Green-red opponent axis.
    pub a: f32,
    /// Blue-yellow opponent axis.
    pub b: f32,
}

impl OkLab {
    /// Number of meaningful scalar fields packed into the `std430` block.
    const FIELD_COUNT: usize = 3;

    /// Byte size of the `std430` packing: the three `OKLab` scalars padded up to
    /// a single `vec4` slot so the block honours the 16-byte `std430` base
    /// alignment the `GPU` shading kernel expects.
    pub const STD430_SIZE: usize = Self::FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

    /// Builds an `OKLab` colour from its lightness and opponent axes.
    #[must_use]
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Packs this `OKLab` colour into its `std430` block.
    ///
    /// Laid out little-endian as the lightness, green-red, and blue-yellow
    /// scalars (`f32`) followed by one `f32` of zero padding, so the block spans
    /// a single `vec4` slot ([`OkLab::STD430_SIZE`] bytes) the `GPU` kernel binds.
    #[must_use]
    pub fn to_std430(&self) -> [u8; Self::STD430_SIZE] {
        let mut bytes = [0u8; Self::STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.l.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.a.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.b.to_le_bytes());
        bytes
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// `OKLab` blocks, clamped up to a single element per the shared
    /// [`storage_bytes`] rule.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(Self::STD430_SIZE, count)
    }
}

/// A colour in the `OKLCh` cylindrical form of `OKLab`.
///
/// `l` matches [`OkLab::l`]; `c` is chroma (radial distance from the neutral
/// axis, always non-negative); and the hue angle is stored as the unit vector
/// `(h_cos, h_sin)` so hue rotation and reconstruction stay free of any inverse
/// trigonometry. For an achromatic colour (`c` near zero) the hue is undefined
/// and canonicalised to `(1, 0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkLch {
    /// Perceptual lightness, shared with [`OkLab::l`].
    pub l: f32,
    /// Chroma: the radial distance from the neutral axis.
    pub c: f32,
    /// Cosine of the hue angle (the green-red component of the unit hue vector).
    pub h_cos: f32,
    /// Sine of the hue angle (the blue-yellow component of the unit hue vector).
    pub h_sin: f32,
}

/// The real cube root of `x`, computed by Newton's method to `f32` precision.
///
/// The workspace bans `f32::cbrt` and `f32::powf` (for cross-backend
/// determinism), so this hand-rolled routine reduces the argument by its
/// base-2 exponent and then refines the mantissa's cube root with the update
/// `y <- (2*y + m/(y*y)) / 3`, which is the Newton iteration for `y^3 - m = 0`.
/// Only multiply, add, divide, and `f32::from_bits` / `f32::to_bits` are used.
///
/// The sign is handled explicitly (`cbrt(-x) == -cbrt(x)`) and a zero (or
/// sub-normal) magnitude returns `0` without any division.
#[must_use]
fn cbrt_newton(x: f32) -> f32 {
    let magnitude = x.abs();
    if magnitude < f32::MIN_POSITIVE {
        return 0.0;
    }
    let bits = magnitude.to_bits();
    // Unbiased base-2 exponent of `magnitude` (guaranteed normal here). The
    // masked field is at most 255, so the conversion never fails.
    let exponent = i32::try_from((bits >> 23) & 0xff).unwrap_or(0) - 127;
    // Mantissa reconstructed into the range [1, 2).
    let mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    // Split the exponent so that 2^exponent = 2^(3*q) * 2^r with r in {0,1,2},
    // using Euclidean division so a negative exponent still yields r >= 0.
    let q = exponent.div_euclid(3);
    let r = exponent.rem_euclid(3);
    // Cube root of the mantissa via Newton from y = 1; for m in [1, 2) seven
    // quadratically-converging steps reach full `f32` precision.
    let mut y = 1.0f32;
    let mut step = 0;
    while step < 7 {
        let y2 = y * y;
        y = (2.0 * y + mantissa / y2) / 3.0;
        step += 1;
    }
    // Re-apply the residual 2^r factor.
    let r_scale = match r {
        0 => 1.0,
        1 => CBRT_2,
        _ => CBRT_4,
    };
    // Re-apply the 2^q factor by scaling by two the matching number of times.
    let mut two_pow_q = 1.0f32;
    if q >= 0 {
        let mut n = 0;
        while n < q {
            two_pow_q *= 2.0;
            n += 1;
        }
    } else {
        let mut n = 0;
        while n < -q {
            two_pow_q *= 0.5;
            n += 1;
        }
    }
    let root = y * r_scale * two_pow_q;
    if x < 0.0 {
        -root
    } else {
        root
    }
}

/// Converts a linear `sRGB` colour into `OKLab`.
///
/// This applies Ottosson's first matrix `M1` to reach the cone-response `LMS`
/// space, takes the cube root of each `LMS` component (via [`cbrt_newton`]),
/// then applies the second matrix `M2` to reach `(l, a, b)`.
#[must_use]
pub fn linear_srgb_to_oklab(c: &LinearSrgb) -> OkLab {
    // M1: linear sRGB -> LMS (Ottosson's official coefficients).
    let l = 0.412_221_47 * c.r + 0.536_332_54 * c.g + 0.051_445_995 * c.b;
    let m = 0.211_903_5 * c.r + 0.680_699_5 * c.g + 0.107_396_96 * c.b;
    let s = 0.088_302_46 * c.r + 0.281_718_84 * c.g + 0.629_978_7 * c.b;

    let l_ = cbrt_newton(l);
    let m_ = cbrt_newton(m);
    let s_ = cbrt_newton(s);

    // M2: non-linear LMS -> OKLab.
    OkLab {
        l: 0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        a: 1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        b: 0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    }
}

/// Converts an `OKLab` colour back into linear `sRGB`.
///
/// This inverts [`linear_srgb_to_oklab`]: the inverse of `M2` reaches the
/// non-linear `LMS` terms, each is cubed with a plain `y*y*y` product, and the
/// inverse of `M1` reaches linear `RGB`.
#[must_use]
pub fn oklab_to_linear_srgb(c: &OkLab) -> LinearSrgb {
    // Inverse of M2: OKLab -> non-linear LMS.
    let l_ = c.l + 0.396_337_78 * c.a + 0.215_803_76 * c.b;
    let m_ = c.l - 0.105_561_346 * c.a - 0.063_854_17 * c.b;
    let s_ = c.l - 0.089_484_18 * c.a - 1.291_485_5 * c.b;

    // Cube each term with an explicit integer-count multiply.
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    // Inverse of M1: LMS -> linear sRGB.
    LinearSrgb {
        r: 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        g: -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        b: -0.004_196_086_4 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    }
}

/// Converts an `OKLab` colour into its cylindrical `OKLCh` form.
///
/// Chroma is the Euclidean magnitude `sqrt(a*a + b*b)` and the hue unit vector
/// is `(a/c, b/c)`. When the chroma is at or below [`CHROMA_FLOOR`] the colour
/// is achromatic and the hue is canonicalised to `(1, 0)`.
#[must_use]
pub fn oklab_to_oklch(c: &OkLab) -> OkLch {
    let chroma = (c.a * c.a + c.b * c.b).sqrt();
    if chroma > CHROMA_FLOOR {
        OkLch {
            l: c.l,
            c: chroma,
            h_cos: c.a / chroma,
            h_sin: c.b / chroma,
        }
    } else {
        OkLch {
            l: c.l,
            c: 0.0,
            h_cos: 1.0,
            h_sin: 0.0,
        }
    }
}

/// Converts an `OKLCh` colour back into `OKLab`.
///
/// The opponent axes are the chroma projected onto the stored hue unit vector:
/// `a = c * h_cos`, `b = c * h_sin`. This is the exact inverse of
/// [`oklab_to_oklch`] for any non-degenerate hue vector.
#[must_use]
pub fn oklch_to_oklab(c: &OkLch) -> OkLab {
    OkLab {
        l: c.l,
        a: c.c * c.h_cos,
        b: c.c * c.h_sin,
    }
}

/// Rotates the hue of an `OKLCh` colour by an angle given as its cosine and
/// sine, leaving lightness and chroma untouched.
///
/// The stored hue unit vector is rotated in the chroma plane by the standard
/// rotation `(cos*x - sin*y, sin*x + cos*y)`, so passing `(1, 0)` is the
/// identity and `(0, 1)` is a quarter-turn. No trigonometric call is made.
#[must_use]
pub fn rotate_hue(c: &OkLch, cos_delta: f32, sin_delta: f32) -> OkLch {
    OkLch {
        l: c.l,
        c: c.c,
        h_cos: c.h_cos * cos_delta - c.h_sin * sin_delta,
        h_sin: c.h_sin * cos_delta + c.h_cos * sin_delta,
    }
}

/// Perceptually uniform linear interpolation between two `OKLab` colours.
///
/// Each channel is blended as `a + (b - a) * t`, so `t = 0` returns `a`, `t = 1`
/// returns `b`, and intermediate `t` walks a straight line in `OKLab`, which is
/// close to a perceptually even colour ramp.
#[must_use]
pub fn lerp(a: &OkLab, b: &OkLab, t: f32) -> OkLab {
    OkLab {
        l: a.l + (b.l - a.l) * t,
        a: a.a + (b.a - a.a) * t,
        b: a.b + (b.b - a.b) * t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for `f32` approximate comparisons in the tests.
    const CMP_EPS: f32 = 1e-6;

    fn close(x: f32, y: f32, eps: f32) -> bool {
        (x - y).abs() < eps
    }

    #[test]
    fn cbrt_of_perfect_cubes_is_exact_enough() {
        assert!(close(cbrt_newton(8.0), 2.0, 1e-5));
        assert!(close(cbrt_newton(27.0), 3.0, 1e-5));
        assert!(close(cbrt_newton(64.0), 4.0, 1e-5));
        assert!(close(cbrt_newton(1000.0), 10.0, 1e-4));
    }

    #[test]
    fn cbrt_of_fractions() {
        assert!(close(cbrt_newton(0.125), 0.5, 1e-6));
        assert!(close(cbrt_newton(0.001), 0.1, 1e-6));
    }

    #[test]
    fn cbrt_of_one_is_one() {
        assert!(close(cbrt_newton(1.0), 1.0, CMP_EPS));
    }

    #[test]
    fn cbrt_of_zero_is_zero() {
        assert!(close(cbrt_newton(0.0), 0.0, CMP_EPS));
        assert!(close(cbrt_newton(-0.0), 0.0, CMP_EPS));
    }

    #[test]
    fn cbrt_of_negatives_is_negative_root() {
        assert!(close(cbrt_newton(-8.0), -2.0, 1e-5));
        assert!(close(cbrt_newton(-27.0), -3.0, 1e-5));
        assert!(close(cbrt_newton(-0.125), -0.5, 1e-6));
    }

    #[test]
    fn cbrt_inverts_cubing() {
        let samples = [0.02f32, 0.17, 0.33, 0.5, 0.81, 1.4, 2.75];
        for v in samples {
            let cubed = v * v * v;
            assert!(close(cbrt_newton(cubed), v, 1e-5));
        }
    }

    #[test]
    fn white_maps_to_lightness_one_and_zero_chroma() {
        let lab = linear_srgb_to_oklab(&LinearSrgb::new(1.0, 1.0, 1.0));
        assert!(close(lab.l, 1.0, 1e-4));
        assert!(close(lab.a, 0.0, 1e-4));
        assert!(close(lab.b, 0.0, 1e-4));
    }

    #[test]
    fn black_maps_to_origin() {
        let lab = linear_srgb_to_oklab(&LinearSrgb::new(0.0, 0.0, 0.0));
        assert!(close(lab.l, 0.0, CMP_EPS));
        assert!(close(lab.a, 0.0, CMP_EPS));
        assert!(close(lab.b, 0.0, CMP_EPS));
    }

    #[test]
    fn roundtrip_red_primary() {
        let c = LinearSrgb::new(1.0, 0.0, 0.0);
        let back = oklab_to_linear_srgb(&linear_srgb_to_oklab(&c));
        assert!(close(back.r, c.r, 1e-4));
        assert!(close(back.g, c.g, 1e-4));
        assert!(close(back.b, c.b, 1e-4));
    }

    #[test]
    fn roundtrip_green_primary() {
        let c = LinearSrgb::new(0.0, 1.0, 0.0);
        let back = oklab_to_linear_srgb(&linear_srgb_to_oklab(&c));
        assert!(close(back.r, c.r, 1e-4));
        assert!(close(back.g, c.g, 1e-4));
        assert!(close(back.b, c.b, 1e-4));
    }

    #[test]
    fn roundtrip_blue_primary() {
        let c = LinearSrgb::new(0.0, 0.0, 1.0);
        let back = oklab_to_linear_srgb(&linear_srgb_to_oklab(&c));
        assert!(close(back.r, c.r, 1e-4));
        assert!(close(back.g, c.g, 1e-4));
        assert!(close(back.b, c.b, 1e-4));
    }

    #[test]
    fn roundtrip_arbitrary_colours() {
        let samples = [
            LinearSrgb::new(0.25, 0.5, 0.75),
            LinearSrgb::new(0.9, 0.1, 0.4),
            LinearSrgb::new(0.05, 0.6, 0.2),
            LinearSrgb::new(0.7, 0.7, 0.3),
        ];
        for c in samples {
            let back = oklab_to_linear_srgb(&linear_srgb_to_oklab(&c));
            assert!(close(back.r, c.r, 1e-4));
            assert!(close(back.g, c.g, 1e-4));
            assert!(close(back.b, c.b, 1e-4));
        }
    }

    #[test]
    fn grayscale_has_negligible_chroma() {
        for g in [0.1f32, 0.3, 0.55, 0.8, 0.95] {
            let lab = linear_srgb_to_oklab(&LinearSrgb::new(g, g, g));
            assert!(close(lab.a, 0.0, 1e-4));
            assert!(close(lab.b, 0.0, 1e-4));
        }
    }

    #[test]
    fn lightness_is_monotonic_in_luminance() {
        let dark = linear_srgb_to_oklab(&LinearSrgb::new(0.1, 0.1, 0.1));
        let mid = linear_srgb_to_oklab(&LinearSrgb::new(0.4, 0.4, 0.4));
        let bright = linear_srgb_to_oklab(&LinearSrgb::new(0.85, 0.85, 0.85));
        assert!(dark.l < mid.l);
        assert!(mid.l < bright.l);
    }

    #[test]
    fn oklab_oklch_roundtrip() {
        let samples = [
            OkLab::new(0.6, 0.12, -0.05),
            OkLab::new(0.4, -0.09, 0.11),
            OkLab::new(0.75, 0.03, 0.02),
        ];
        for lab in samples {
            let back = oklch_to_oklab(&oklab_to_oklch(&lab));
            assert!(close(back.l, lab.l, CMP_EPS));
            assert!(close(back.a, lab.a, CMP_EPS));
            assert!(close(back.b, lab.b, CMP_EPS));
        }
    }

    #[test]
    fn oklch_of_gray_is_canonical() {
        let lch = oklab_to_oklch(&OkLab::new(0.5, 0.0, 0.0));
        assert!(close(lch.c, 0.0, CMP_EPS));
        assert!(close(lch.h_cos, 1.0, CMP_EPS));
        assert!(close(lch.h_sin, 0.0, CMP_EPS));
    }

    #[test]
    fn oklch_hue_vector_is_unit_length() {
        let lch = oklab_to_oklch(&OkLab::new(0.6, 0.12, -0.05));
        let mag = (lch.h_cos * lch.h_cos + lch.h_sin * lch.h_sin).sqrt();
        assert!(close(mag, 1.0, CMP_EPS));
    }

    #[test]
    fn oklch_chroma_matches_magnitude() {
        let lab = OkLab::new(0.5, 0.3, 0.4);
        let lch = oklab_to_oklch(&lab);
        assert!(close(lch.c, 0.5, CMP_EPS));
    }

    #[test]
    fn lerp_hits_endpoints() {
        let a = OkLab::new(0.2, -0.1, 0.05);
        let b = OkLab::new(0.8, 0.1, -0.05);
        let at0 = lerp(&a, &b, 0.0);
        let at1 = lerp(&a, &b, 1.0);
        assert!(
            close(at0.l, a.l, CMP_EPS) && close(at0.a, a.a, CMP_EPS) && close(at0.b, a.b, CMP_EPS)
        );
        assert!(
            close(at1.l, b.l, CMP_EPS) && close(at1.a, b.a, CMP_EPS) && close(at1.b, b.b, CMP_EPS)
        );
    }

    #[test]
    fn lerp_midpoint_is_average() {
        let a = OkLab::new(0.2, -0.1, 0.05);
        let b = OkLab::new(0.8, 0.1, -0.05);
        let mid = lerp(&a, &b, 0.5);
        assert!(close(mid.l, 0.5, CMP_EPS));
        assert!(close(mid.a, 0.0, CMP_EPS));
        assert!(close(mid.b, 0.0, CMP_EPS));
    }

    #[test]
    fn rotate_hue_by_zero_is_identity() {
        let lch = oklab_to_oklch(&OkLab::new(0.6, 0.12, -0.05));
        let rotated = rotate_hue(&lch, 1.0, 0.0);
        assert!(close(rotated.h_cos, lch.h_cos, CMP_EPS));
        assert!(close(rotated.h_sin, lch.h_sin, CMP_EPS));
        assert!(close(rotated.c, lch.c, CMP_EPS));
        assert!(close(rotated.l, lch.l, CMP_EPS));
    }

    #[test]
    fn rotate_hue_preserves_chroma_and_lightness() {
        let lch = oklab_to_oklch(&OkLab::new(0.6, 0.12, -0.05));
        // Rotate by an arbitrary angle whose cosine/sine form a unit vector.
        let rotated = rotate_hue(&lch, 0.6, 0.8);
        assert!(close(rotated.c, lch.c, CMP_EPS));
        assert!(close(rotated.l, lch.l, CMP_EPS));
        let mag = (rotated.h_cos * rotated.h_cos + rotated.h_sin * rotated.h_sin).sqrt();
        assert!(close(mag, 1.0, CMP_EPS));
    }

    #[test]
    fn rotate_hue_quarter_turn() {
        // Start from hue vector (1, 0); a quarter-turn (cos=0, sin=1) -> (0, 1).
        let lch = OkLch {
            l: 0.5,
            c: 0.2,
            h_cos: 1.0,
            h_sin: 0.0,
        };
        let rotated = rotate_hue(&lch, 0.0, 1.0);
        assert!(close(rotated.h_cos, 0.0, CMP_EPS));
        assert!(close(rotated.h_sin, 1.0, CMP_EPS));
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(OkLab::STD430_SIZE, VEC4_STRIDE);
    }

    #[test]
    fn to_std430_packs_channels_little_endian() {
        let lab = OkLab::new(0.25, -0.5, 0.75);
        let bytes = lab.to_std430();
        assert_eq!(&bytes[0..4], &0.25f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &(-0.5f32).to_le_bytes());
        assert_eq!(&bytes[8..12], &0.75f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &[0u8; 4]);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(OkLab::gpu_storage_bytes(0), VEC4_STRIDE);
        assert_eq!(OkLab::gpu_storage_bytes(1), VEC4_STRIDE);
        assert_eq!(OkLab::gpu_storage_bytes(8), 8 * VEC4_STRIDE);
    }
}
