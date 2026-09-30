//! `Bloom` bright-pass threshold extraction with a soft `knee` and the `Karis`
//! anti-firefly downsample weight (design §16-§21).
//!
//! `Bloom` sells the illusion that a surface is emitting more light than the
//! display can show: the brightest regions of the `HDR` framebuffer bleed a
//! soft halo into their neighbors. Every production stack (Unreal's bloom pass,
//! `Frostbite`'s FX post, Unity `HDRP`) begins that effect with the same
//! *bright-pass*: isolate the pixels above a brightness `threshold`, feed only
//! those into the blur pyramid, and scale the result by an `intensity` gain.
//! This module owns the `CPU`-verifiable maths of that bright-pass contract and
//! packs its parameters into the `std430` block a future `GPU` prefilter kernel
//! binds.
//!
//! # Strict scope
//!
//! This file is *only* the bright-pass prefilter. It extracts the thresholded
//! bloom source color and supplies the [`Karis`](BloomThresholdParams::karis_weight)
//! reweighting factor used when averaging the taps of a downsample step. It
//! deliberately does **not** run the multi-level blur pyramid (no up/down
//! convolution), it does not tone-map (that is [`super::tonemap`]'s
//! `Reinhard`/`ACES` curves), and it does not build a luminance histogram or
//! drive exposure (that is [`super::luminance_hist`]). Its whole job is
//! `HDR` color -> thresholded bloom source color plus the `Karis` weight.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions
//! (`sin`/`cos`/`exp`/`ln`/`powf`). The soft `knee` is a classic quadratic
//! rational curve and the `Karis` weight is the rational `1 / (1 + luma)`, so
//! evaluation touches only `+ - * /` guarded against a zero denominator. No
//! `f32::round`/`f32::ceil` and no lookup table, so a future `GPU` kernel
//! reproduces the `CPU` result bit for bit.

use crate::particle::gpu_layout::VEC4_STRIDE;
use alloc::vec::Vec;

/// Number of scalar fields packed into the [`BloomThresholdParams`] `std430`
/// block: `threshold`, `knee`, and `intensity`.
const BLOOM_FIELD_COUNT: usize = 3;

/// Byte size of the `std430` packing of [`BloomThresholdParams`]: the three
/// scalars rounded up to whole `vec4` slots so the block honors the 16-byte
/// `std430` base alignment. Three scalars occupy a single `vec4` slot (16
/// bytes) with a one-scalar padding tail.
pub const BLOOM_STD430_SIZE: usize = BLOOM_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// Denominators with magnitude below this are treated as (near) zero so
/// evaluation falls back to a defined result instead of dividing by zero or
/// propagating `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// `Rec. 709` `luminance` weight of the red channel.
const LUMA_R: f32 = 0.2126;

/// `Rec. 709` `luminance` weight of the green channel.
const LUMA_G: f32 = 0.7152;

/// `Rec. 709` `luminance` weight of the blue channel.
const LUMA_B: f32 = 0.0722;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// The perceptual `luminance` of a linear `RGB` triple, `dot(rgb, [0.2126,
/// 0.7152, 0.0722])`, written as an explicit hand-rolled dot product of the
/// `Rec. 709` weights. This is the brightness the bright-pass thresholds
/// against and the value the [`Karis`](BloomThresholdParams::karis_weight)
/// weight consumes.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA_R + rgb[1] * LUMA_G + rgb[2] * LUMA_B
}

/// Bright-pass prefilter parameters for the `bloom` source extraction
/// (design §16-§21).
///
/// `threshold` is the `luminance` above which a pixel begins to contribute to
/// bloom; `knee` is the half-width of the soft transition band around the
/// threshold (a zero `knee` is a hard cutoff); and `intensity` is the global
/// gain applied to the extracted source color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomThresholdParams {
    /// `Luminance` threshold where a pixel starts contributing to bloom.
    pub threshold: f32,
    /// Half-width of the soft `knee` transition band around the threshold.
    pub knee: f32,
    /// Global gain applied to the extracted bloom source color.
    pub intensity: f32,
}

impl BloomThresholdParams {
    /// Builds a parameter set from its raw fields, clamping `knee` and
    /// `intensity` to be non-negative so the curves stay well defined.
    #[must_use]
    pub fn new(threshold: f32, knee: f32, intensity: f32) -> Self {
        Self {
            threshold,
            knee: knee.max(0.0),
            intensity: intensity.max(0.0),
        }
    }

    /// The soft-`knee` numerator: the thresholded `luminance` response, i.e.
    /// `max(soft, lum - threshold)` where the `soft` branch is the classic
    /// quadratic `knee` `soft^2 / (4 knee)` over the transition band.
    ///
    /// Below `threshold - knee` the response is exactly `0`; across the band
    /// `[threshold - knee, threshold + knee]` it ramps smoothly with a
    /// continuous first derivative; and far above the threshold it converges to
    /// the hard response `lum - threshold`. The function is monotonically
    /// non-decreasing in `lum`.
    #[must_use]
    pub fn knee_response(&self, lum: f32) -> f32 {
        let over = lum - self.threshold;
        let mut soft = (over + self.knee).clamp(0.0, 2.0 * self.knee);
        soft = soft * soft / (4.0 * self.knee + MIN_DENOM);
        soft.max(over)
    }

    /// The per-pixel bloom `contribution` factor, the normalized soft-`knee`
    /// response `knee_response(lum) / max(lum, eps)`.
    ///
    /// Multiplying the source `RGB` by this factor keeps the pixel's hue while
    /// scaling its magnitude: `~0` well below the threshold, and approaching
    /// `1` far above it where `(lum - threshold) / lum -> 1`.
    #[must_use]
    pub fn contribution(&self, lum: f32) -> f32 {
        self.knee_response(lum) / lum.max(MIN_DENOM)
    }

    /// The thresholded bloom source color for one linear `HDR` `RGB` sample:
    /// the original color scaled by its [`contribution`](Self::contribution)
    /// and the global `intensity`. The hue is preserved because all three
    /// channels share the same scalar factor.
    #[must_use]
    pub fn threshold_color(&self, rgb: [f32; 3]) -> [f32; 3] {
        let factor = self.contribution(luminance(rgb)) * self.intensity;
        [rgb[0] * factor, rgb[1] * factor, rgb[2] * factor]
    }

    /// Batches [`threshold_color`](Self::threshold_color) over many source
    /// colors, one thresholded triple per input, preserving order.
    #[must_use]
    pub fn threshold_batch(&self, colors: &[[f32; 3]]) -> Vec<[f32; 3]> {
        colors
            .iter()
            .map(|&rgb| self.threshold_color(rgb))
            .collect()
    }

    /// The `Karis` anti-firefly downsample weight `1 / (1 + luma)` for a tap of
    /// the given `luminance`.
    ///
    /// Averaging downsample taps by this weight suppresses fireflies: a lone
    /// very bright sub-pixel sample (high `luma`) is pulled toward the group so
    /// it cannot flicker through the blur pyramid. The weight is `1` at zero
    /// `luminance` and strictly decreasing as `luma` grows. Negative inputs are
    /// floored to `0` so the denominator stays `>= 1`.
    #[must_use]
    pub fn karis_weight(luma: f32) -> f32 {
        1.0 / (1.0 + luma.max(0.0))
    }

    /// The `Karis`-weighted average of a set of `RGB` downsample taps.
    ///
    /// Each tap is weighted by [`karis_weight`](Self::karis_weight) of its own
    /// `luminance`, so bright outliers contribute less than uniformly bright
    /// neighbors. An empty tap set (or one whose weights sum below `eps`)
    /// returns black rather than dividing by zero.
    #[must_use]
    pub fn karis_average(taps: &[[f32; 3]]) -> [f32; 3] {
        let mut sum = [0.0f32; 3];
        let mut weight_sum = 0.0f32;
        for &tap in taps {
            let w = Self::karis_weight(luminance(tap));
            sum[0] += tap[0] * w;
            sum[1] += tap[1] * w;
            sum[2] += tap[2] * w;
            weight_sum += w;
        }
        let inv = 1.0 / weight_sum.max(MIN_DENOM);
        [sum[0] * inv, sum[1] * inv, sum[2] * inv]
    }

    /// Packs the parameters into their `std430` uniform-block byte layout.
    ///
    /// The three scalars fill the first [`BLOOM_FIELD_COUNT`] scalar slots of a
    /// single `vec4`; the one-scalar padding tail stays zero so the block is a
    /// whole number of `vec4` slots.
    #[must_use]
    pub fn to_std430(&self) -> [u8; BLOOM_STD430_SIZE] {
        let fields = [self.threshold, self.knee, self.intensity];
        let mut bytes = [0u8; BLOOM_STD430_SIZE];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::storage_bytes;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    fn sample_params() -> BloomThresholdParams {
        BloomThresholdParams::new(1.0, 0.5, 1.0)
    }

    #[test]
    fn luminance_is_the_rec709_dot_product() {
        assert!(approx(luminance([1.0, 0.0, 0.0]), 0.2126));
        assert!(approx(luminance([0.0, 1.0, 0.0]), 0.7152));
        assert!(approx(luminance([0.0, 0.0, 1.0]), 0.0722));
        // White sums the three weights to one.
        assert!(approx(luminance([1.0, 1.0, 1.0]), 1.0));
        // A mixed sample equals the explicit hand dot product.
        let rgb = [0.3, 0.6, 0.9];
        let expected = 0.3 * 0.2126 + 0.6 * 0.7152 + 0.9 * 0.0722;
        assert!(approx(luminance(rgb), expected));
    }

    #[test]
    fn below_threshold_extracts_no_bloom() {
        let p = sample_params();
        // A dim grey whose luminance sits well under (threshold - knee).
        let dim = [0.2, 0.2, 0.2];
        assert!(luminance(dim) < p.threshold - p.knee);
        let out = p.threshold_color(dim);
        assert!(approx3(out, [0.0, 0.0, 0.0]));
        assert!(approx(p.contribution(luminance(dim)), 0.0));
    }

    #[test]
    fn far_above_threshold_preserves_original_color() {
        let p = sample_params();
        // A very bright sample: contribution -> (lum - threshold)/lum -> 1, so
        // the extracted color reconverges on the input up to a tiny relative
        // slack that shrinks as the luminance grows.
        let bright = [400.0, 300.0, 200.0];
        let lum = luminance(bright);
        let c = p.contribution(lum);
        assert!(c > 0.99 && c < 1.0);
        let out = p.threshold_color(bright);
        // Each channel lands within 1% of the original.
        for (o, i) in out.iter().zip(bright.iter()) {
            assert!((o - i).abs() < i * 0.01);
        }
    }

    #[test]
    fn knee_response_is_monotonic_across_the_band() {
        let p = sample_params();
        let samples = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 5.0];
        let mut prev = p.knee_response(samples[0]);
        for &lum in &samples[1..] {
            let cur = p.knee_response(lum);
            assert!(cur >= prev - CMP_EPS);
            prev = cur;
        }
    }

    #[test]
    fn knee_gives_a_smooth_nonzero_transition_inside_the_band() {
        let p = sample_params();
        // At the threshold itself the soft knee is already contributing, unlike
        // a hard cutoff which would still read exactly zero there.
        let at_threshold = p.knee_response(p.threshold);
        assert!(at_threshold > 0.0);
        // The knee floor (threshold - knee) is the last exactly-zero point.
        assert!(approx(p.knee_response(p.threshold - p.knee), 0.0));
    }

    #[test]
    fn zero_knee_is_a_hard_cutoff() {
        let hard = BloomThresholdParams::new(1.0, 0.0, 1.0);
        // Exactly at the threshold the hard response is zero and stays zero
        // just below it, with no divide-by-zero from the guarded denominator.
        assert!(approx(hard.knee_response(1.0), 0.0));
        assert!(approx(hard.knee_response(0.9), 0.0));
        assert!(hard.knee_response(1.5) > 0.0);
    }

    #[test]
    fn karis_weight_decreases_with_luminance() {
        let samples = [0.0, 0.5, 1.0, 2.0, 4.0, 16.0];
        let mut prev = BloomThresholdParams::karis_weight(samples[0]);
        assert!(approx(prev, 1.0));
        for &luma in &samples[1..] {
            let cur = BloomThresholdParams::karis_weight(luma);
            assert!(cur < prev);
            assert!((0.0..=1.0).contains(&cur));
            prev = cur;
        }
    }

    #[test]
    fn karis_average_pulls_a_firefly_toward_its_neighbors() {
        // Three dim taps and one very bright firefly.
        let taps = [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [50.0, 50.0, 50.0],
        ];
        let weighted = BloomThresholdParams::karis_average(&taps);
        let naive = {
            let mut s = [0.0f32; 3];
            for t in &taps {
                s[0] += t[0];
                s[1] += t[1];
                s[2] += t[2];
            }
            [s[0] / 4.0, s[1] / 4.0, s[2] / 4.0]
        };
        // The anti-firefly average is far darker than the naive mean.
        assert!(weighted[0] < naive[0]);
    }

    #[test]
    fn karis_average_of_no_taps_is_black() {
        let out = BloomThresholdParams::karis_average(&[]);
        assert!(approx3(out, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn intensity_scales_the_extracted_color_linearly() {
        let base = BloomThresholdParams::new(1.0, 0.5, 1.0);
        let doubled = BloomThresholdParams::new(1.0, 0.5, 2.0);
        let rgb = [3.0, 2.0, 1.0];
        let a = base.threshold_color(rgb);
        let b = doubled.threshold_color(rgb);
        assert!(approx3(b, [a[0] * 2.0, a[1] * 2.0, a[2] * 2.0]));
    }

    #[test]
    fn threshold_color_preserves_hue_direction() {
        let p = sample_params();
        let rgb = [4.0, 2.0, 1.0];
        let out = p.threshold_color(rgb);
        // Output is a non-negative scalar multiple of the input, so the ratios
        // between channels are preserved.
        assert!(approx(out[0] * rgb[1], out[1] * rgb[0]));
        assert!(approx(out[1] * rgb[2], out[2] * rgb[1]));
        assert!(out[0] > 0.0);
    }

    #[test]
    fn threshold_batch_matches_scalar_path() {
        let p = sample_params();
        let colors = [[0.1, 0.1, 0.1], [2.0, 1.0, 0.5], [10.0, 10.0, 10.0]];
        let batch = p.threshold_batch(&colors);
        assert_eq!(batch.len(), colors.len());
        for (out, &rgb) in batch.iter().zip(colors.iter()) {
            assert!(approx3(*out, p.threshold_color(rgb)));
        }
    }

    #[test]
    fn saturating_boundaries_stay_finite_and_defined() {
        let p = sample_params();
        // Zero color: guarded denominator keeps the result at black, not NaN.
        let black = p.threshold_color([0.0, 0.0, 0.0]);
        assert!(approx3(black, [0.0, 0.0, 0.0]));
        assert!(black[0].is_finite());
        // Enormous color: contribution stays within the unit range.
        let huge = 1.0e9;
        let c = p.contribution(huge);
        assert!((0.0..=1.0).contains(&c));
    }

    #[test]
    fn std430_size_is_a_whole_vec4_block() {
        assert_eq!(BLOOM_STD430_SIZE, VEC4_STRIDE);
        assert_eq!(BLOOM_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(BLOOM_STD430_SIZE, storage_bytes(VEC4_STRIDE, 1));
    }

    #[test]
    fn std430_round_trips_the_scalar_fields() {
        let p = BloomThresholdParams::new(1.25, 0.5, 2.5);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), BLOOM_STD430_SIZE);
        let read = |i: usize| {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            f32::from_le_bytes(b)
        };
        assert!(approx(read(0), 1.25));
        assert!(approx(read(1), 0.5));
        assert!(approx(read(2), 2.5));
        // The padding-tail scalar stays zero.
        assert!(approx(read(3), 0.0));
    }
}
