//! `sRGB` `gamut` clipping / mapping for particle color output (design §16).
//!
//! Shading, tone mapping, color grading, and temperature all work in extended
//! or wide ranges, but the final swap-chain surface is a low-dynamic-range
//! `sRGB` target whose channels must land inside the unit cube `0..=1`. This
//! module owns the `CPU` reference for that last mile: taking a linear `sRGB`
//! triple whose channels may sit outside `0..=1` and mapping it back into the
//! displayable `gamut`. It is deliberately narrow — it does *not* reuse the
//! [`super::tonemap`], [`super::color_grade_lut`], [`super::color_temperature`],
//! or [`super::oklab_color`] types, and it performs no curve, `LUT`, white-point,
//! or perceptual-space work. Its only job is the clip.
//!
//! Three strategies are provided, from cheapest to most faithful:
//!
//! 1. [`clip_naive`] — the per-channel hard `f32::clamp` to `0..=1`. Fast, but
//!    it shifts hue and luma because it clamps each channel independently.
//! 2. [`clip_preserve_luma`] — desaturates toward the neutral gray that shares
//!    the color's `Rec. 709` luma, walking along that line only as far as needed
//!    to re-enter the cube. Because the luma weights sum to one, every point on
//!    that line shares the input luma, so brightness is preserved exactly while
//!    saturation drops.
//! 3. [`soft_clip`] — a monotone rational roll-off near the `0` and `1` walls
//!    that trades a hard edge for a smooth knee, avoiding banding on gradients.
//!
//! Only `f32::sqrt`-free rational arithmetic and integer `div_ceil` are used —
//! no transcendental functions — so the result is deterministic and portable to
//! the eventual `GPU` kernel. `GPU` packing follows the shared `std430` `vec4`
//! alignment from [`super::gpu_layout`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Comparison epsilon for the in-`gamut` membership test tolerance.
const CMP_EPS: f32 = 1e-6;

/// Smallest knee half-width; below this a soft knee collapses toward a hard
/// clamp, and it also guards the roll-off denominator against division by zero.
const MIN_KNEE: f32 = 1e-6;

/// `Rec. 709` luma weight for the red channel.
const LUMA_R: f32 = 0.2126;
/// `Rec. 709` luma weight for the green channel.
const LUMA_G: f32 = 0.7152;
/// `Rec. 709` luma weight for the blue channel.
const LUMA_B: f32 = 0.0722;

/// Number of scalar fields packed into the `std430` record.
const GAMUT_FIELD_COUNT: usize = 3;

/// Byte size of one [`Rgb`] record in a `std430` storage buffer.
///
/// The three channels fill the first three scalar slots of a single `vec4`; the
/// fourth scalar is zero padding so the record is a whole `vec4` (16 bytes).
pub const GAMUT_STD430_SIZE: usize = GAMUT_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// A linear `sRGB` color triple whose channels may fall outside `0..=1`.
///
/// This is a plain value type local to the `gamut` clip contract; it is not the
/// shading or grading color type and carries no alpha or color-space tag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    /// Red channel (linear `sRGB`).
    pub r: f32,
    /// Green channel (linear `sRGB`).
    pub g: f32,
    /// Blue channel (linear `sRGB`).
    pub b: f32,
}

impl Rgb {
    /// Builds a color from its three linear channels.
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// A neutral gray with all three channels equal to `v`.
    #[must_use]
    pub fn gray(v: f32) -> Self {
        Self { r: v, g: v, b: v }
    }
}

/// Clamps a scalar into `0..=1` without branching on floating equality.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The `Rec. 709` luma (relative luminance) of a linear color.
///
/// The three weights sum to one, so a neutral gray built from this value shares
/// the color's luma exactly — the invariant that makes [`clip_preserve_luma`]
/// brightness preserving.
#[must_use]
pub fn luma_rec709(c: &Rgb) -> f32 {
    LUMA_R * c.r + LUMA_G * c.g + LUMA_B * c.b
}

/// Returns `true` when every channel lies within `0..=1` up to [`CMP_EPS`].
#[must_use]
pub fn is_in_gamut(c: &Rgb) -> bool {
    let lo = -CMP_EPS;
    let hi = 1.0 + CMP_EPS;
    c.r >= lo && c.r <= hi && c.g >= lo && c.g <= hi && c.b >= lo && c.b <= hi
}

/// Per-channel hard clip: clamps each channel independently into `0..=1`.
///
/// This is the cheapest strategy and the fallback the other strategies defer to
/// when they cannot do better. It can shift hue and luma because the channels
/// are clamped without regard for one another.
#[must_use]
pub fn clip_naive(c: Rgb) -> Rgb {
    Rgb {
        r: clamp01(c.r),
        g: clamp01(c.g),
        b: clamp01(c.b),
    }
}

/// Solves for the largest fraction of the excursion `channel - gray` that keeps
/// `gray + t * (channel - gray)` inside `0..=1`, returning `1.0` when the
/// channel is already in range.
#[must_use]
fn channel_limit(channel: f32, gray: f32) -> f32 {
    let d = channel - gray;
    if channel > 1.0 {
        // `d > 0` because `gray <= 1 < channel`; guard the divide regardless.
        if d > MIN_KNEE {
            return (1.0 - gray) / d;
        }
        return 0.0;
    }
    if channel < 0.0 {
        // `d < 0` because `gray >= 0 > channel`; guard the divide regardless.
        if d < -MIN_KNEE {
            return (0.0 - gray) / d;
        }
        return 0.0;
    }
    1.0
}

/// Desaturates toward the equal-luma gray until the color re-enters `0..=1`.
///
/// The color is pulled along the segment from the neutral gray `(L, L, L)` —
/// where `L` is the input's [`luma_rec709`] — toward the original color, and the
/// largest blend fraction `t` that stays inside the cube is chosen per channel
/// by [`channel_limit`]. Because the luma weights sum to one, every point on
/// that segment shares the luma `L`, so brightness is preserved exactly while
/// saturation is reduced only as much as necessary.
///
/// When the input luma itself lies outside `0..=1` the equal-luma gray is not
/// displayable, so no luma-preserving mapping exists and the routine falls back
/// to [`clip_naive`].
#[must_use]
pub fn clip_preserve_luma(c: Rgb) -> Rgb {
    if is_in_gamut(&c) {
        return c;
    }
    let l = luma_rec709(&c);
    if !(-CMP_EPS..=1.0 + CMP_EPS).contains(&l) {
        return clip_naive(c);
    }
    let gray = clamp01(l);
    let t = channel_limit(c.r, gray)
        .min(channel_limit(c.g, gray))
        .min(channel_limit(c.b, gray))
        .clamp(0.0, 1.0);
    // A final hard clamp removes any float rounding overshoot at the walls; the
    // adjustment is below `CMP_EPS`, so luma stays preserved.
    Rgb {
        r: clamp01(gray + t * (c.r - gray)),
        g: clamp01(gray + t * (c.g - gray)),
        b: clamp01(gray + t * (c.b - gray)),
    }
}

/// Monotone rational soft clip of one channel toward `0..=1`.
///
/// Inside `[knee, 1 - knee]` the mapping is the identity. Above `1 - knee` the
/// excess is compressed by `e / (e + knee)`, a saturating rational that rises
/// with unit slope at the knee (so the join is smooth) and approaches `1`
/// asymptotically without ever reaching or exceeding it. The low wall is the
/// mirror image about `0`. The result is monotone across the whole real line and
/// continuous at both knee joins.
#[must_use]
fn soft_clip_channel(x: f32, knee: f32) -> f32 {
    let k = knee.clamp(MIN_KNEE, 0.5);
    let upper = 1.0 - k;
    let lower = k;
    if x > upper {
        let e = x - upper;
        return upper + k * (e / (e + k));
    }
    if x < lower {
        let e = lower - x;
        return lower - k * (e / (e + k));
    }
    x
}

/// Soft-clips all three channels toward `0..=1` with a smooth knee of half-width
/// `knee`.
///
/// Unlike [`clip_naive`], values near the walls roll off smoothly instead of
/// snapping flat, which avoids the hard contour a hard clamp leaves on a
/// gradient. `knee` is clamped into `[MIN_KNEE, 0.5]`; a larger knee starts the
/// roll-off earlier and softens more aggressively.
#[must_use]
pub fn soft_clip(c: Rgb, knee: f32) -> Rgb {
    Rgb {
        r: soft_clip_channel(c.r, knee),
        g: soft_clip_channel(c.g, knee),
        b: soft_clip_channel(c.b, knee),
    }
}

/// Applies [`clip_preserve_luma`] to every color, preserving order and length.
#[must_use]
pub fn clip_preserve_luma_batch(colors: &[Rgb]) -> Vec<Rgb> {
    colors.iter().copied().map(clip_preserve_luma).collect()
}

/// Packs a color into its `std430` `vec4`-aligned byte layout.
///
/// The three channels fill the first three scalar slots; the fourth scalar tail
/// stays zero so the record is exactly one `vec4` ([`GAMUT_STD430_SIZE`] bytes).
#[must_use]
pub fn to_std430(c: &Rgb) -> [u8; GAMUT_STD430_SIZE] {
    let fields = [c.r, c.g, c.b];
    let mut bytes = [0u8; GAMUT_STD430_SIZE];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Total byte size of a `std430` storage buffer holding `count` colors, clamped
/// up to a single element so an empty set still yields a valid `GPU` binding.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(GAMUT_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Comparison epsilon for the test assertions only.
    const TEST_EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn approx_rgb(a: Rgb, b: Rgb) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b)
    }

    fn in_unit(c: Rgb) -> bool {
        let ok = |x: f32| (-TEST_EPS..=1.0 + TEST_EPS).contains(&x);
        ok(c.r) && ok(c.g) && ok(c.b)
    }

    #[test]
    fn luma_matches_rec709_primaries() {
        assert!(approx(luma_rec709(&Rgb::new(1.0, 0.0, 0.0)), LUMA_R));
        assert!(approx(luma_rec709(&Rgb::new(0.0, 1.0, 0.0)), LUMA_G));
        assert!(approx(luma_rec709(&Rgb::new(0.0, 0.0, 1.0)), LUMA_B));
    }

    #[test]
    fn luma_of_white_is_one() {
        assert!(approx(luma_rec709(&Rgb::gray(1.0)), 1.0));
    }

    #[test]
    fn luma_matches_hand_dot_product() {
        let c = Rgb::new(0.3, 0.6, 0.9);
        let expected = 0.3 * LUMA_R + 0.6 * LUMA_G + 0.9 * LUMA_B;
        assert!(approx(luma_rec709(&c), expected));
    }

    #[test]
    fn in_gamut_accepts_interior_and_corners() {
        assert!(is_in_gamut(&Rgb::new(0.0, 0.5, 1.0)));
        assert!(is_in_gamut(&Rgb::gray(0.0)));
        assert!(is_in_gamut(&Rgb::gray(1.0)));
    }

    #[test]
    fn in_gamut_rejects_negative_channel() {
        assert!(!is_in_gamut(&Rgb::new(-0.01, 0.5, 0.5)));
    }

    #[test]
    fn in_gamut_rejects_over_one_channel() {
        assert!(!is_in_gamut(&Rgb::new(0.5, 1.5, 0.5)));
    }

    #[test]
    fn naive_clamps_negative_up_to_zero() {
        let out = clip_naive(Rgb::new(-0.5, 0.2, 0.7));
        assert!(approx(out.r, 0.0));
        assert!(approx(out.g, 0.2));
        assert!(approx(out.b, 0.7));
    }

    #[test]
    fn naive_clamps_over_one_down_to_one() {
        let out = clip_naive(Rgb::new(0.2, 2.0, 0.7));
        assert!(approx(out.g, 1.0));
    }

    #[test]
    fn naive_leaves_interior_unchanged() {
        let c = Rgb::new(0.1, 0.4, 0.9);
        assert!(approx_rgb(clip_naive(c), c));
    }

    #[test]
    fn preserve_luma_leaves_interior_unchanged() {
        let c = Rgb::new(0.2, 0.5, 0.8);
        assert!(approx_rgb(clip_preserve_luma(c), c));
    }

    #[test]
    fn preserve_luma_maps_over_one_into_unit_cube() {
        let out = clip_preserve_luma(Rgb::new(1.6, 0.1, 0.1));
        assert!(in_unit(out));
    }

    #[test]
    fn preserve_luma_maps_negative_into_unit_cube() {
        let out = clip_preserve_luma(Rgb::new(-0.4, 0.3, 0.3));
        assert!(in_unit(out));
    }

    #[test]
    fn preserve_luma_keeps_luma_for_in_range_luma() {
        // Luma of this color is `0.2126 * 1.6 + ... = 0.4776`, inside `0..=1`.
        let c = Rgb::new(1.6, 0.1, 0.1);
        let before = luma_rec709(&c);
        let after = luma_rec709(&clip_preserve_luma(c));
        assert!(approx(before, after));
    }

    #[test]
    fn preserve_luma_keeps_luma_for_negative_channel() {
        let c = Rgb::new(-0.3, 0.5, 0.6);
        let before = luma_rec709(&c);
        assert!((-CMP_EPS..=1.0 + CMP_EPS).contains(&before));
        let after = luma_rec709(&clip_preserve_luma(c));
        assert!(approx(before, after));
    }

    #[test]
    fn preserve_luma_leaves_gray_unchanged() {
        let c = Rgb::gray(0.5);
        assert!(approx_rgb(clip_preserve_luma(c), c));
    }

    #[test]
    fn preserve_luma_reduces_saturation_toward_gray() {
        let c = Rgb::new(1.6, 0.1, 0.1);
        let gray = clamp01(luma_rec709(&c));
        let out = clip_preserve_luma(c);
        // Every channel moves no farther from gray than the original did.
        assert!((out.r - gray).abs() <= (c.r - gray).abs() + TEST_EPS);
        assert!((out.g - gray).abs() <= (c.g - gray).abs() + TEST_EPS);
        assert!((out.b - gray).abs() <= (c.b - gray).abs() + TEST_EPS);
    }

    #[test]
    fn preserve_luma_falls_back_when_luma_exceeds_one() {
        // Luma far above one: no equal-luma gray is displayable.
        let c = Rgb::new(2.0, 2.0, 2.0);
        assert!(approx_rgb(clip_preserve_luma(c), clip_naive(c)));
    }

    #[test]
    fn preserve_luma_falls_back_when_luma_below_zero() {
        let c = Rgb::new(-1.0, -1.0, -1.0);
        assert!(approx_rgb(clip_preserve_luma(c), clip_naive(c)));
    }

    #[test]
    fn soft_clip_is_identity_in_the_interior() {
        let c = Rgb::new(0.3, 0.5, 0.7);
        assert!(approx_rgb(soft_clip(c, 0.1), c));
    }

    #[test]
    fn soft_clip_is_monotone_across_a_sweep() {
        let knee = 0.2;
        let mut prev = soft_clip_channel(-1.0, knee);
        let mut x = -1.0;
        while x <= 2.0 {
            let y = soft_clip_channel(x, knee);
            assert!(y >= prev - TEST_EPS);
            prev = y;
            x += 0.01;
        }
    }

    #[test]
    fn soft_clip_is_continuous_at_the_knee_joins() {
        // Continuity: approaching each join from either side converges to the
        // join value with unit slope, so the deviation shrinks with the offset.
        let knee = 0.2;
        let upper = 1.0 - knee;
        let lower = knee;
        let h = 1e-5;
        assert!((soft_clip_channel(upper + h, knee) - upper).abs() < 2.0 * h);
        assert!((soft_clip_channel(upper - h, knee) - upper).abs() < 2.0 * h);
        assert!((soft_clip_channel(lower + h, knee) - lower).abs() < 2.0 * h);
        assert!((soft_clip_channel(lower - h, knee) - lower).abs() < 2.0 * h);
    }

    #[test]
    fn soft_clip_keeps_large_values_below_one() {
        let out = soft_clip(Rgb::new(5.0, 10.0, 100.0), 0.25);
        assert!(out.r < 1.0 && out.g < 1.0 && out.b < 1.0);
        assert!(in_unit(out));
    }

    #[test]
    fn soft_clip_keeps_negative_values_above_zero() {
        let out = soft_clip(Rgb::new(-5.0, -10.0, -100.0), 0.25);
        assert!(out.r > 0.0 && out.g > 0.0 && out.b > 0.0);
        assert!(in_unit(out));
    }

    #[test]
    fn soft_clip_clamps_a_degenerate_knee() {
        // A zero knee is lifted to `MIN_KNEE`; the result stays finite and in
        // range rather than dividing by zero.
        let out = soft_clip_channel(2.0, 0.0);
        assert!(out.is_finite());
        assert!(out <= 1.0 + TEST_EPS);
    }

    #[test]
    fn batch_matches_single_element_mapping() {
        let colors = [
            Rgb::new(1.6, 0.1, 0.1),
            Rgb::new(-0.4, 0.3, 0.3),
            Rgb::gray(0.5),
        ];
        let batch = clip_preserve_luma_batch(&colors);
        assert_eq!(batch.len(), colors.len());
        for (out, c) in batch.iter().zip(colors.iter()) {
            assert!(approx_rgb(*out, clip_preserve_luma(*c)));
        }
    }

    #[test]
    fn batch_of_empty_slice_is_empty() {
        let out = clip_preserve_luma_batch(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(GAMUT_STD430_SIZE, 16);
        assert_eq!(GAMUT_STD430_SIZE % VEC4_STRIDE, 0);
    }

    #[test]
    fn std430_round_trips_the_channels() {
        let c = Rgb::new(0.25, 1.5, -0.5);
        let bytes = to_std430(&c);
        assert_eq!(bytes.len(), GAMUT_STD430_SIZE);
        let read = |i: usize| {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            f32::from_le_bytes(b)
        };
        assert!(approx(read(0), 0.25));
        assert!(approx(read(1), 1.5));
        assert!(approx(read(2), -0.5));
        // The padding-tail scalar stays zero.
        assert!(approx(read(3), 0.0));
    }

    #[test]
    fn gpu_storage_bytes_matches_shared_helper() {
        assert_eq!(gpu_storage_bytes(0), storage_bytes(GAMUT_STD430_SIZE, 0));
        assert_eq!(gpu_storage_bytes(10), storage_bytes(GAMUT_STD430_SIZE, 10));
        assert_eq!(gpu_storage_bytes(1), GAMUT_STD430_SIZE);
    }
}
