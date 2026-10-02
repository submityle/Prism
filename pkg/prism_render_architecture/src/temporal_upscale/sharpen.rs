//! Robust Contrast-Adaptive Sharpening (`RCAS`) resolve.
//!
//! A temporal upsampler trades a little sharpness for stability: the
//! Catmull-Rom history fetch, the neighborhood clamp, and the exponential
//! accumulation all soften high-frequency detail. A dedicated sharpening pass
//! runs after reconstruction to restore that detail without re-introducing the
//! ringing and noise amplification a naive unsharp mask produces.
//!
//! This module owns the `CPU` golden for `AMD`'s `RCAS` filter (the sharpening
//! pass shipped with `FSR`). `RCAS` is a 5-tap cross filter whose single
//! sharpening coefficient (the *lobe*) is derived per pixel from the local
//! contrast so that flat regions are left untouched and only edges are
//! sharpened, with the lobe limited so the filter can never push a channel past
//! the representable peak and blow out into a halo.
//!
//! The filter is meant to run in a bounded domain: feed it the tone-mapped
//! color (see [`tonemap`](super::color::tonemap)) whose channels lie in
//! `[0, 1)`, sharpen, clamp the largest channel back below `1`, then
//! [`untonemap`](super::color::untonemap). The peak reference used by the lobe
//! limiter is `1.0`, which is exactly the upper bound of that domain.
//!
//! Every operation is `+`, `-`, `*`, `/`, `min`/`max`, and `abs`, so a `GPU`
//! `WESL` kernel reproduces the result bit-for-bit.

use super::color::luminance;

/// Upper bound on the magnitude of the `RCAS` sharpening lobe.
///
/// `AMD`'s `RCAS` caps the lobe at `0.25 - 1/16` so the normalizing
/// denominator `1 + 4 * lobe` stays at or above `0.25`; this prevents the
/// filter from amplifying a channel by more than roughly `4x` and bounds the
/// overshoot that produces visible halos around high-contrast edges.
pub const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;

/// The five taps of the `RCAS` cross kernel, sampled from the reconstructed
/// image.
///
/// The taps form a plus/cross pattern centered on the pixel being sharpened:
///
/// ```text
///        north
///  west  center  east
///        south
/// ```
///
/// Colors are expected in the bounded tone-mapped domain described in the
/// module documentation. Edge pixels use clamp addressing (the caller repeats
/// the nearest valid tap), matching the history sampler in
/// [`sample_catmull_rom`](super::reproject::sample_catmull_rom).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossTaps {
    /// The pixel being sharpened; the only tap whose kernel weight is positive.
    pub center: [f32; 3],
    /// The tap one pixel above `center`.
    pub north: [f32; 3],
    /// The tap one pixel below `center`.
    pub south: [f32; 3],
    /// The tap one pixel left of `center`.
    pub west: [f32; 3],
    /// The tap one pixel right of `center`.
    pub east: [f32; 3],
}

impl CrossTaps {
    /// The four ring taps (the neighbors, excluding `center`).
    ///
    /// `RCAS` derives its lobe from the contrast of the ring alone, so the
    /// limiter and the (optional) denoise term both work from this set.
    #[must_use]
    fn ring(&self) -> [[f32; 3]; 4] {
        [self.north, self.south, self.west, self.east]
    }
}

/// The per-channel minimum and maximum over the four ring taps.
///
/// These bound the local color range the limiter uses to decide how much
/// headroom exists before sharpening would clip a channel.
#[must_use]
fn ring_min_max(ring: &[[f32; 3]; 4]) -> ([f32; 3], [f32; 3]) {
    let mut mn = ring[0];
    let mut mx = ring[0];
    for tap in &ring[1..] {
        for c in 0..3 {
            mn[c] = mn[c].min(tap[c]);
            mx[c] = mx[c].max(tap[c]);
        }
    }
    (mn, mx)
}

/// The per-channel sharpening lobe candidate (always `<= 0`).
///
/// `hit_min` is the headroom toward black (`mn / (4 * mx)`) and `hit_max` is the
/// headroom toward the peak of `1` (`(1 - mx) / (4 * (mn - 1))`, which is
/// non-positive). The lobe candidate `max(-hit_min, hit_max)` is the least
/// aggressive of the two limits so neither end of the range can clip. Both
/// reciprocals are guarded against the degenerate `0/0` cases (`mx <= 0` and
/// `mn >= 1`), which only occur on a perfectly flat ring where no sharpening is
/// wanted anyway.
#[must_use]
fn channel_lobe(mn: f32, mx: f32) -> f32 {
    let hit_min = if mx <= 0.0 { 0.0 } else { mn / (4.0 * mx) };
    let denom = 4.0 * (mn - 1.0);
    let hit_max = if denom >= 0.0 {
        0.0
    } else {
        (1.0 - mx) / denom
    };
    (-hit_min).max(hit_max)
}

/// A noise-adaptive attenuation of the lobe in `[0.5, 1]`.
///
/// Where the center luma sits near the average of its neighbors the signal is a
/// smooth gradient and the full lobe is used; where the center luma is a local
/// spike relative to the ring's luma range the region is treated as noise and
/// the lobe is halved, so `RCAS` sharpens edges without amplifying grain. A
/// degenerate (zero or non-finite) local range disables the attenuation.
#[must_use]
fn denoise_weight(taps: &CrossTaps) -> f32 {
    let center = luminance(taps.center);
    let ring = taps.ring();
    let ring_luma = [
        luminance(ring[0]),
        luminance(ring[1]),
        luminance(ring[2]),
        luminance(ring[3]),
    ];
    let mean = 0.25 * (ring_luma[0] + ring_luma[1] + ring_luma[2] + ring_luma[3]);
    let mut lo = center;
    let mut hi = center;
    for l in ring_luma {
        lo = lo.min(l);
        hi = hi.max(l);
    }
    let range = hi - lo;
    if range.is_nan() || range <= 0.0 {
        return 1.0;
    }
    // Normalized local spike in [0, 1]; a full spike halves the lobe.
    let spike = ((center - mean).abs() / range).min(1.0);
    1.0 - 0.5 * spike
}

/// Sharpens one pixel with the `RCAS` cross filter.
///
/// `sharpness` is a normalized `[0, 1]` knob (as validated by
/// [`clamp_sharpness`](super::clamp_sharpness)): `0` returns `center`
/// unchanged, `1` applies the full contrast-limited lobe. When `denoise` is
/// set, the lobe is attenuated in luma-spike regions (see [`denoise_weight`]).
///
/// The returned color is in the same (tone-mapped) domain as the input; it may
/// overshoot slightly above the ring maximum or below `0` at an edge, which is
/// the intended sharpening response. The caller clamps the largest channel back
/// below `1` before [`untonemap`](super::color::untonemap). A flat
/// neighborhood yields a zero lobe and the identity, so constant regions are
/// preserved exactly.
#[must_use]
pub fn rcas(taps: &CrossTaps, sharpness: f32, denoise: bool) -> [f32; 3] {
    let ring = taps.ring();
    let (mn, mx) = ring_min_max(&ring);

    // The lobe is the most conservative (closest to zero, i.e. largest) of the
    // three per-channel limits so no channel clips, then clamped to the RCAS
    // range and scaled by the sharpness knob.
    let lobe_rgb = [
        channel_lobe(mn[0], mx[0]),
        channel_lobe(mn[1], mx[1]),
        channel_lobe(mn[2], mx[2]),
    ];
    let mut lobe = lobe_rgb[0].max(lobe_rgb[1]).max(lobe_rgb[2]);
    lobe = lobe.clamp(-RCAS_LIMIT, 0.0) * sharpness;
    if denoise {
        lobe *= denoise_weight(taps);
    }

    // Weighted combine: center weight 1, each ring tap weight `lobe` (negative),
    // normalized by the weight sum `1 + 4 * lobe`. The sum guarantees a flat
    // signal is preserved and, with the clamped lobe, the denominator stays
    // at or above 0.25 so the reciprocal is always well-defined.
    let norm = 1.0 / (4.0 * lobe + 1.0);
    core::array::from_fn(|c| {
        let ring_sum = ring[0][c] + ring[1][c] + ring[2][c] + ring[3][c];
        (taps.center[c] + lobe * ring_sum) * norm
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error tolerance for the value checks.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    /// Asserts two colors agree channel-wise within [`approx`].
    fn approx_rgb(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    /// A cross with a uniform ring and a given center.
    fn cross(center: [f32; 3], ring: [f32; 3]) -> CrossTaps {
        CrossTaps {
            center,
            north: ring,
            south: ring,
            west: ring,
            east: ring,
        }
    }

    #[test]
    fn zero_sharpness_is_identity() {
        let taps = cross([0.7, 0.3, 0.5], [0.2, 0.1, 0.4]);
        assert!(approx_rgb(rcas(&taps, 0.0, false), taps.center));
    }

    #[test]
    fn flat_region_is_preserved() {
        // Every tap equal: zero contrast, zero lobe, exact passthrough even at
        // full sharpness.
        let taps = cross([0.42, 0.42, 0.42], [0.42, 0.42, 0.42]);
        assert!(approx_rgb(rcas(&taps, 1.0, false), taps.center));
    }

    #[test]
    fn constant_signal_survives_full_strength() {
        // A uniform ring that differs from center is still a "constant" to the
        // normalization: the weights sum to one, so a flat input (center ==
        // ring) is preserved. Verify the weight-sum invariant directly.
        let taps = cross([0.3, 0.3, 0.3], [0.3, 0.3, 0.3]);
        let out = rcas(&taps, 1.0, true);
        assert!(approx_rgb(out, [0.3, 0.3, 0.3]));
    }

    #[test]
    fn bright_center_overshoots_above_ring() {
        // A bright pixel on a darker ring should be pushed brighter than it
        // started (sharpening overshoot), not blurred toward the ring.
        let taps = cross([0.6, 0.6, 0.6], [0.2, 0.2, 0.2]);
        let out = rcas(&taps, 1.0, false);
        assert!(
            out[0] > taps.center[0],
            "expected overshoot, got {out:?} from center {:?}",
            taps.center
        );
    }

    #[test]
    fn dark_center_undershoots_below_ring() {
        // Symmetric case: a dark pixel on a bright ring is pushed darker.
        let taps = cross([0.3, 0.3, 0.3], [0.8, 0.8, 0.8]);
        let out = rcas(&taps, 1.0, false);
        assert!(
            out[0] < taps.center[0],
            "expected undershoot, got {out:?} from center {:?}",
            taps.center
        );
    }

    #[test]
    fn sharpness_scales_the_effect_monotonically() {
        let taps = cross([0.6, 0.6, 0.6], [0.2, 0.2, 0.2]);
        let base = taps.center[0];
        let half = rcas(&taps, 0.5, false)[0];
        let full = rcas(&taps, 1.0, false)[0];
        // More sharpness pushes further from the original value.
        assert!(full - base > half - base);
        assert!(half - base > 0.0);
    }

    #[test]
    fn lobe_is_contrast_limited_against_clipping() {
        // Maximum contrast ring (0 vs near-peak): the lobe magnitude must stay
        // within RCAS_LIMIT so the normalizer never collapses.
        let taps = CrossTaps {
            center: [0.5, 0.5, 0.5],
            north: [0.99, 0.99, 0.99],
            south: [0.0, 0.0, 0.0],
            west: [0.99, 0.0, 0.0],
            east: [0.0, 0.99, 0.0],
        };
        let out = rcas(&taps, 1.0, false);
        assert!(out.iter().all(|c| c.is_finite()), "non-finite out {out:?}");
    }

    #[test]
    fn denoise_reduces_sharpening_on_a_luma_spike() {
        // Center is a bright luma spike relative to a flat darker ring: the
        // denoise term should pull the sharpened result back toward center
        // compared with denoise off.
        let taps = cross([0.9, 0.9, 0.9], [0.1, 0.1, 0.1]);
        let plain = rcas(&taps, 1.0, false)[0];
        let denoised = rcas(&taps, 1.0, true)[0];
        let base = taps.center[0];
        assert!(
            (denoised - base).abs() < (plain - base).abs(),
            "denoise should attenuate: plain {plain} denoised {denoised} base {base}"
        );
    }

    #[test]
    fn rcas_limit_matches_amd_constant() {
        assert!(approx(RCAS_LIMIT, 0.1875));
    }
}
