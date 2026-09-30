//! Anamorphic lens-streak highlight extraction and directional smear contract
//! (design §16-§21).
//!
//! Anamorphic lenses render a bright specular highlight as a long, thin streak
//! that runs along one fixed lens axis (the classic horizontal blue-teal flare
//! of cinema anamorphics). This module owns the `CPU`-verifiable maths of that
//! effect: it extracts the highlights above a soft `threshold`, walks a single
//! one-dimensional multi-tap chain whose tap spacing doubles every level, and
//! accumulates the taps with a per-streak chromatic `tint`. It packs its
//! parameters into the `std430` block a future `GPU` streak kernel binds.
//!
//! # Strict scope
//!
//! This file is *only* the directional one-dimensional streak. It smears along
//! a single configurable axis and never touches the isotropic paths owned by
//! its siblings: it does **not** run an isotropic tent-filter downsample
//! pyramid (that is [`super::bloom_upsample`] and [`super::bloom_threshold`]),
//! and it does **not** synthesize lens ghosts or the halo ring band (that is
//! [`super::lens_flare`]). Its whole job is `HDR` color plus a direction ->
//! thresholded, directionally accumulated, `tint`-ed streak color.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions
//! (`sin`/`cos`/`exp`/`ln`/`powf`) and rounding intrinsics. The soft
//! `threshold` uses the classic `smoothstep` polynomial `t*t*(3-2t)`, the tap
//! spacing doubles by repeated multiplication, and unit-normalization uses
//! `sqrt` only, so a future `GPU` kernel reproduces the `CPU` result closely.

use crate::particle::gpu_layout::VEC4_STRIDE;
use alloc::vec::Vec;

/// Byte size of the `std430` packing of [`StreakParams`].
///
/// The block occupies three `vec4` slots (48 bytes): a leading `direction`
/// `vec2` and two scalars fill the first slot, the `tint` `vec3` is `vec4`
/// aligned into the second slot, and the remaining scalars plus the `tap_count`
/// `u32` fill the third slot with a padding tail.
pub const STREAK_STD430_SIZE: usize = 3 * VEC4_STRIDE;

/// Denominators with magnitude below this are treated as (near) zero so the
/// soft-`threshold` band and unit-normalization fall back to a defined result
/// instead of dividing by zero or propagating `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// `Rec. 709` luminance weight of the red channel.
const LUMA_R: f32 = 0.2126;

/// `Rec. 709` luminance weight of the green channel.
const LUMA_G: f32 = 0.7152;

/// `Rec. 709` luminance weight of the blue channel.
const LUMA_B: f32 = 0.0722;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// The perceptual luminance of a linear `RGB` triple, `dot(rgb, [0.2126,
/// 0.7152, 0.0722])`, written as an explicit hand-rolled dot product of the
/// `Rec. 709` weights. This is the brightness the streak thresholds against.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA_R + rgb[1] * LUMA_G + rgb[2] * LUMA_B
}

/// Anamorphic streak parameters (design §16-§21).
///
/// `direction` is the unit lens axis the streak runs along; `tap_count` is the
/// number of samples in the one-dimensional chain; `stretch` is the base tap
/// spacing (the offset of the first tap, doubling each level); `tint` is the
/// per-streak chromatic color multiplier; `threshold` and `knee` define the
/// soft highlight-extraction band; and `intensity` is the global gain applied
/// to the accumulated streak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreakParams {
    /// Unit lens axis the streak smears along.
    pub direction: [f32; 2],
    /// Number of taps in the one-dimensional streak chain.
    pub tap_count: u32,
    /// Base tap spacing; the first tap offset, doubling each level.
    pub stretch: f32,
    /// Per-streak chromatic color multiplier applied to the accumulation.
    pub tint: [f32; 3],
    /// Luminance where the highlight begins contributing to the streak.
    pub threshold: f32,
    /// Half-width of the soft transition band around the `threshold`.
    pub knee: f32,
    /// Global gain applied to the accumulated streak color.
    pub intensity: f32,
}

impl StreakParams {
    /// Builds a parameter set from its raw fields, clamping the non-negative
    /// scalars, forcing at least one tap, and unit-normalizing `direction`
    /// (falling back to the `+x` axis for a degenerate zero direction).
    #[must_use]
    pub fn new(
        direction: [f32; 2],
        tap_count: u32,
        stretch: f32,
        tint: [f32; 3],
        threshold: f32,
        knee: f32,
        intensity: f32,
    ) -> Self {
        let len_sq = direction[0] * direction[0] + direction[1] * direction[1];
        let dir = if len_sq > MIN_DENOM {
            let inv = 1.0 / len_sq.sqrt();
            [direction[0] * inv, direction[1] * inv]
        } else {
            [1.0, 0.0]
        };
        Self {
            direction: dir,
            tap_count: tap_count.max(1),
            stretch: stretch.max(0.0),
            tint,
            threshold: threshold.max(0.0),
            knee: knee.max(0.0),
            intensity: intensity.max(0.0),
        }
    }

    /// The soft-`threshold` highlight weight in `[0, 1]` for a sample of the
    /// given luminance.
    ///
    /// Below `threshold - knee` the weight is `0`; above `threshold + knee` it
    /// is `1`; across the band it ramps with the `smoothstep` polynomial
    /// `t*t*(3-2t)`, so the weight is exactly `0.5` at the `threshold` and the
    /// response is monotonically non-decreasing. A zero `knee` degenerates to a
    /// hard cutoff at the `threshold`.
    #[must_use]
    pub fn threshold_weight(&self, lum: f32) -> f32 {
        let lo = self.threshold - self.knee;
        let span = 2.0 * self.knee;
        if span <= MIN_DENOM {
            if lum >= self.threshold {
                1.0
            } else {
                0.0
            }
        } else {
            let t = ((lum - lo) / span).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// The one-dimensional tap offsets along `direction`, one per tap, whose
    /// spacing doubles every level.
    ///
    /// Tap `i` sits at `direction * stretch * 2^i`, so consecutive taps are
    /// twice as far apart as the previous pair. All offsets are colinear with
    /// `direction`; a zero `stretch` collapses them onto the origin.
    #[must_use]
    pub fn streak_tap_offsets(&self) -> Vec<[f32; 2]> {
        let mut offsets = Vec::new();
        let mut spacing = self.stretch;
        for _ in 0..self.tap_count {
            offsets.push([self.direction[0] * spacing, self.direction[1] * spacing]);
            spacing *= 2.0;
        }
        offsets
    }

    /// Accumulates the sampled `RGB` taps into the final streak color: a
    /// geometric weighted sum (each successive tap contributes half as much as
    /// the previous one) scaled by the per-streak `tint` and the global
    /// `intensity`.
    ///
    /// The weighting keeps the near taps dominant so the streak fades along its
    /// length, and the `tint` gives the streak its characteristic chromatic
    /// cast. An empty tap set returns black.
    #[must_use]
    pub fn accumulate_streak(&self, taps: &[[f32; 3]]) -> [f32; 3] {
        let mut acc = [0.0f32; 3];
        let mut weight = 1.0f32;
        for &tap in taps {
            acc[0] += tap[0] * weight;
            acc[1] += tap[1] * weight;
            acc[2] += tap[2] * weight;
            weight *= 0.5;
        }
        [
            acc[0] * self.tint[0] * self.intensity,
            acc[1] * self.tint[1] * self.intensity,
            acc[2] * self.tint[2] * self.intensity,
        ]
    }

    /// Packs the parameters into their `std430` uniform-block byte layout.
    ///
    /// The `direction` `vec2` and the `stretch`/`threshold` scalars fill the
    /// first `vec4`; the `tint` `vec3` is `vec4` aligned into the second; and
    /// `knee`, `intensity`, and the `tap_count` `u32` fill the third with a
    /// zeroed padding tail.
    #[must_use]
    pub fn to_std430(&self) -> [u8; STREAK_STD430_SIZE] {
        let mut bytes = [0u8; STREAK_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.direction[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.direction[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.stretch.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.threshold.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.tint[0].to_le_bytes());
        bytes[20..24].copy_from_slice(&self.tint[1].to_le_bytes());
        bytes[24..28].copy_from_slice(&self.tint[2].to_le_bytes());
        bytes[28..32].copy_from_slice(&self.knee.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.intensity.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.tap_count.to_le_bytes());
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

    fn sample_params() -> StreakParams {
        StreakParams::new([1.0, 0.0], 4, 2.0, [0.6, 0.8, 1.0], 1.0, 0.5, 2.0)
    }

    #[test]
    fn luminance_is_the_rec709_dot_product() {
        assert!(approx(luminance([1.0, 0.0, 0.0]), 0.2126));
        assert!(approx(luminance([0.0, 1.0, 0.0]), 0.7152));
        assert!(approx(luminance([0.0, 0.0, 1.0]), 0.0722));
        assert!(approx(luminance([1.0, 1.0, 1.0]), 1.0));
    }

    #[test]
    fn luminance_matches_manual_dot() {
        let rgb = [0.3, 0.6, 0.9];
        let expected = 0.3 * 0.2126 + 0.6 * 0.7152 + 0.9 * 0.0722;
        assert!(approx(luminance(rgb), expected));
    }

    #[test]
    fn new_unit_normalizes_direction() {
        let p = StreakParams::new([3.0, 4.0], 4, 1.0, [1.0, 1.0, 1.0], 1.0, 0.5, 1.0);
        let len = (p.direction[0] * p.direction[0] + p.direction[1] * p.direction[1]).sqrt();
        assert!(approx(len, 1.0));
        // Direction keeps its orientation: (3, 4) normalizes to (0.6, 0.8).
        assert!(approx(p.direction[0], 0.6));
        assert!(approx(p.direction[1], 0.8));
    }

    #[test]
    fn new_zero_direction_falls_back_to_x_axis() {
        let p = StreakParams::new([0.0, 0.0], 4, 1.0, [1.0, 1.0, 1.0], 1.0, 0.5, 1.0);
        assert!(approx(p.direction[0], 1.0));
        assert!(approx(p.direction[1], 0.0));
    }

    #[test]
    fn new_clamps_negative_scalars() {
        let p = StreakParams::new([1.0, 0.0], 4, -2.0, [1.0, 1.0, 1.0], -1.0, -0.5, -3.0);
        assert!(approx(p.stretch, 0.0));
        assert!(approx(p.threshold, 0.0));
        assert!(approx(p.knee, 0.0));
        assert!(approx(p.intensity, 0.0));
    }

    #[test]
    fn new_forces_at_least_one_tap() {
        let p = StreakParams::new([1.0, 0.0], 0, 1.0, [1.0, 1.0, 1.0], 1.0, 0.5, 1.0);
        assert_eq!(p.tap_count, 1);
    }

    #[test]
    fn threshold_weight_is_half_at_the_threshold() {
        let p = sample_params();
        assert!(approx(p.threshold_weight(p.threshold), 0.5));
    }

    #[test]
    fn threshold_weight_is_monotonic() {
        let p = sample_params();
        let samples = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 5.0];
        let mut prev = p.threshold_weight(samples[0]);
        for &lum in &samples[1..] {
            let cur = p.threshold_weight(lum);
            assert!(cur >= prev - CMP_EPS);
            prev = cur;
        }
    }

    #[test]
    fn threshold_weight_below_band_is_zero() {
        let p = sample_params();
        assert!(approx(p.threshold_weight(p.threshold - p.knee), 0.0));
        assert!(approx(p.threshold_weight(0.0), 0.0));
    }

    #[test]
    fn threshold_weight_above_band_is_one() {
        let p = sample_params();
        assert!(approx(p.threshold_weight(p.threshold + p.knee), 1.0));
        assert!(approx(p.threshold_weight(10.0), 1.0));
    }

    #[test]
    fn threshold_weight_zero_knee_is_a_hard_cutoff() {
        let hard = StreakParams::new([1.0, 0.0], 4, 1.0, [1.0, 1.0, 1.0], 1.0, 0.0, 1.0);
        assert!(approx(hard.threshold_weight(1.0), 1.0));
        assert!(approx(hard.threshold_weight(1.5), 1.0));
        assert!(approx(hard.threshold_weight(0.999), 0.0));
    }

    #[test]
    fn tap_offsets_count_matches_tap_count() {
        let p = sample_params();
        let offsets = p.streak_tap_offsets();
        assert_eq!(offsets.len(), usize::try_from(p.tap_count).unwrap());
    }

    #[test]
    fn tap_offsets_spacing_doubles_each_level() {
        let p = sample_params();
        let offsets = p.streak_tap_offsets();
        // Magnitude along the unit direction is just stretch * 2^i.
        for i in 1..offsets.len() {
            let prev = offsets[i - 1];
            let cur = offsets[i];
            let prev_len = (prev[0] * prev[0] + prev[1] * prev[1]).sqrt();
            let cur_len = (cur[0] * cur[0] + cur[1] * cur[1]).sqrt();
            assert!(approx(cur_len, prev_len * 2.0));
        }
        // First tap sits at exactly `stretch` along the direction.
        let first = offsets[0];
        let first_len = (first[0] * first[0] + first[1] * first[1]).sqrt();
        assert!(approx(first_len, p.stretch));
    }

    #[test]
    fn tap_offsets_are_colinear_with_direction() {
        let p = StreakParams::new([1.0, 1.0], 3, 2.0, [1.0, 1.0, 1.0], 1.0, 0.5, 1.0);
        for off in p.streak_tap_offsets() {
            // Cross product with the direction is zero for colinear vectors.
            let cross = off[0] * p.direction[1] - off[1] * p.direction[0];
            assert!(approx(cross, 0.0));
        }
    }

    #[test]
    fn tap_offsets_zero_stretch_collapse_to_origin() {
        let p = StreakParams::new([1.0, 0.0], 4, 0.0, [1.0, 1.0, 1.0], 1.0, 0.5, 1.0);
        for off in p.streak_tap_offsets() {
            assert!(approx(off[0], 0.0));
            assert!(approx(off[1], 0.0));
        }
    }

    #[test]
    fn accumulate_weights_halve_per_tap() {
        // Neutral tint and unit intensity isolate the weighting.
        let p = StreakParams::new([1.0, 0.0], 4, 1.0, [1.0, 1.0, 1.0], 0.0, 0.0, 1.0);
        let taps = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let out = p.accumulate_streak(&taps);
        // Weights 1 + 0.5 + 0.25 = 1.75 on the red channel.
        assert!(approx(out[0], 1.75));
        assert!(approx(out[1], 0.0));
        assert!(approx(out[2], 0.0));
    }

    #[test]
    fn accumulate_applies_tint_and_intensity() {
        let p = StreakParams::new([1.0, 0.0], 4, 1.0, [0.5, 0.25, 2.0], 0.0, 0.0, 3.0);
        let taps = [[1.0, 1.0, 1.0]];
        let out = p.accumulate_streak(&taps);
        // Single tap weight 1: channel = tint * intensity.
        assert!(approx(out[0], 0.5 * 3.0));
        assert!(approx(out[1], 0.25 * 3.0));
        assert!(approx(out[2], 2.0 * 3.0));
    }

    #[test]
    fn accumulate_empty_is_black() {
        let p = sample_params();
        let out = p.accumulate_streak(&[]);
        assert!(approx3(out, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn std430_size_is_three_vec4_slots() {
        assert_eq!(STREAK_STD430_SIZE, storage_bytes(VEC4_STRIDE, 3));
        assert_eq!(STREAK_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(STREAK_STD430_SIZE, 48);
    }

    #[test]
    fn std430_round_trips_the_scalar_fields() {
        let p = StreakParams::new([0.6, 0.8], 5, 2.5, [0.3, 0.4, 0.5], 1.25, 0.5, 3.0);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), STREAK_STD430_SIZE);
        let read = |i: usize| {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i..i + 4]);
            f32::from_le_bytes(b)
        };
        assert!(approx(read(0), 0.6));
        assert!(approx(read(4), 0.8));
        assert!(approx(read(8), 2.5));
        assert!(approx(read(12), 1.25));
        assert!(approx(read(16), 0.3));
        assert!(approx(read(20), 0.4));
        assert!(approx(read(24), 0.5));
        assert!(approx(read(28), 0.5));
        assert!(approx(read(32), 3.0));
    }

    #[test]
    fn std430_encodes_tap_count_bits() {
        let p = StreakParams::new([1.0, 0.0], 7, 1.0, [1.0, 1.0, 1.0], 1.0, 0.5, 1.0);
        let bytes = p.to_std430();
        let mut b = [0u8; 4];
        b.copy_from_slice(&bytes[36..40]);
        assert_eq!(u32::from_le_bytes(b), 7);
        // The padding tail past the tap_count stays zero.
        assert_eq!(&bytes[40..48], &[0u8; 8]);
    }
}
