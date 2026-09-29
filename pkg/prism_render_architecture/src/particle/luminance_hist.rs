//! `Luminance` histogram and auto-exposure — the `CPU` reference for the
//! tone-mapping front end of the particle `HDR` shading path (design §16-§21).
//!
//! Production renderers drive automatic exposure from a *`luminance`
//! `histogram`*: the framebuffer's per-pixel `luminance` is binned across a
//! `log2`-`EV` axis, the mean of the binned distribution (usually with the
//! darkest and brightest tails trimmed) becomes the scene's average
//! `luminance`, and a target middle-grey *key* divides that average into an
//! exposure multiplier. Unreal's eye-adaptation histogram pass, Frostbite's
//! auto-exposure, and the classic Reinhard "key value" operator all share this
//! structure. This module owns the `CPU`-verifiable maths of that contract so a
//! future `GPU` compute kernel (which fills a `std430` `u32` histogram buffer)
//! has a bit-checkable reference to match.
//!
//! # No transcendental functions
//!
//! The determinism-locked contract layer forbids `sin`/`cos`/`exp`/`ln`/`powf`
//! and friends. The two operations this file conceptually needs — `log2` (to
//! place a `luminance` on the `EV` axis) and its inverse `exp2` (to turn an
//! averaged `EV` back into a `luminance`) — are therefore reconstructed from
//! the `IEEE`-754 field layout of `f32`: the biased exponent supplies the
//! integer part exactly, and a cheap polynomial closes the fractional part to
//! well under `0.01 EV`. Everything else is add / subtract / multiply / scalar
//! divide (guarded against a zero denominator) plus `f32::floor`. No lookup
//! table, no iteration count that depends on the input magnitude, so the `CPU`
//! and `GPU` paths stay reproducible.
//!
//! # Orthogonality
//!
//! This module consumes only already-shaded `luminance` samples and the
//! [`super::gpu_layout`] stride primitives. It does not reach into the shading
//! router, the tone-map curves, or the camera model; a higher layer wires the
//! exposure multiplier this module computes into those stages.

use super::gpu_layout;
use alloc::vec::Vec;

/// Absolute tolerance used to treat a denominator as degenerate.
///
/// Spans (`EV` range width), sample weights, and average `luminance` values at
/// or below this magnitude are clamped or short-circuited so no routine ever
/// divides by (near) zero or emits `NaN`. Relational comparisons against this
/// constant replace bare `==`/`!=` on floating-point values.
pub const EPS: f32 = 1e-6;

/// Rec. 709 / sRGB `luminance` weights applied to a linear `RGB` triple.
///
/// `luminance = dot(rgb, LUMA_WEIGHTS)`; the weights sum to `1.0` so a neutral
/// grey maps to its own channel value.
pub const LUMA_WEIGHTS: [f32; 3] = [0.212_6, 0.715_2, 0.072_2];

/// Quadratic correction coefficient for the `log2` mantissa term.
///
/// `log2(1 + f) ≈ f + LOG2_CORRECTION * f * (1 - f)` for `f` in `[0, 1)`. The
/// value is `ln(2) / 2`, chosen so the parabola matches `log2(1.5)` at the
/// midpoint; the residual error stays below `0.008 EV` across the octave.
const LOG2_CORRECTION: f32 = 0.346_573_6;

/// Linear coefficient of the quadratic `exp2` fractional approximation.
///
/// `2^f ≈ 1 + EXP2_C1 * f + EXP2_C2 * f * f` for `f` in `[0, 1)`, fitted to the
/// octave endpoints and the `sqrt(2)` midpoint (`EXP2_C1 + EXP2_C2 = 1`).
const EXP2_C1: f32 = 0.656_854_2;

/// Quadratic coefficient of the `exp2` fractional approximation (see
/// [`EXP2_C1`]).
const EXP2_C2: f32 = 0.343_145_8;

/// `2^23`, the width of the `f32` mantissa field, as a float divisor.
const MANTISSA_SCALE: f32 = 8_388_608.0;

/// Sentinel `EV` returned by [`approx_log2`] for non-positive / subnormal
/// inputs, far below any physical `HDR` `luminance` so such samples always land
/// in the lowest histogram bin.
const NEG_EV_FLOOR: f32 = -1000.0;

/// Configuration for building a `luminance` `histogram` and reading an average
/// out of it (design §16).
///
/// The `EV` axis runs from [`min_ev`](Self::min_ev) to
/// [`max_ev`](Self::max_ev) split into [`bins`](Self::bins) equal buckets;
/// [`low_percentile`](Self::low_percentile) and
/// [`high_percentile`](Self::high_percentile) trim the dark and bright tails
/// before averaging so a few black or blown-out pixels cannot drag exposure.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct LumHistConfig {
    /// Lowest `EV` (`log2` `luminance`) represented by bin `0`.
    pub min_ev: f32,
    /// Highest `EV` represented by the final bin's upper edge.
    pub max_ev: f32,
    /// Number of histogram buckets; clamped up to `1` everywhere it is used.
    pub bins: u32,
    /// Fraction of the darkest samples to discard before averaging, in
    /// `[0, 1]`.
    pub low_percentile: f32,
    /// Fraction of the brightest samples to discard before averaging, in
    /// `[0, 1]`.
    pub high_percentile: f32,
}

impl LumHistConfig {
    /// Creates a configuration from an `EV` range, bin count, and trim
    /// percentiles.
    pub const fn new(
        min_ev: f32,
        max_ev: f32,
        bins: u32,
        low_percentile: f32,
        high_percentile: f32,
    ) -> Self {
        Self {
            min_ev,
            max_ev,
            bins,
            low_percentile,
            high_percentile,
        }
    }

    /// Returns the effective bin count, clamped up to `1`.
    ///
    /// A `WebGPU` storage binding may not be empty and dividing the `EV` axis
    /// into zero buckets is meaningless, so callers always see at least one.
    #[must_use]
    pub const fn effective_bins(&self) -> u32 {
        if self.bins < 1 {
            1
        } else {
            self.bins
        }
    }
}

/// A conventional 14-stop `HDR` configuration: `EV` `-8..=6`, 64 buckets, with
/// the darkest `%` and brightest `%` trimmed before averaging.
pub const STANDARD_HDR: LumHistConfig = LumHistConfig::new(-8.0, 6.0, 64, 0.5, 0.9);

/// Clamps `x` into the inclusive range `[lo, hi]`.
///
/// A local helper rather than `f32::clamp` (which panics when `lo > hi`); this
/// layer must stay total on live input.
#[must_use]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Rec. 709 relative `luminance` of a linear `RGB` triple.
///
/// `luminance = 0.2126 R + 0.7152 G + 0.0722 B`, matching [`LUMA_WEIGHTS`].
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA_WEIGHTS[0] + rgb[1] * LUMA_WEIGHTS[1] + rgb[2] * LUMA_WEIGHTS[2]
}

/// Integer-exponent `log2` approximation built from the `f32` bit pattern.
///
/// For a positive normal `x = (1 + f) * 2^e`, the biased exponent field gives
/// `e` exactly and the mantissa fraction `f in [0, 1)` is closed with the
/// quadratic `log2(1 + f) ≈ f + LOG2_CORRECTION * f * (1 - f)`. Powers of two
/// (`f == 0`) are therefore reproduced exactly, and the worst-case error inside
/// an octave stays below `0.008`. Non-positive or subnormal inputs return
/// [`NEG_EV_FLOOR`] rather than `-inf`/`NaN`, so they bin as "darkest". No
/// `ln`/`log2` floating-point function is called.
#[must_use]
pub fn approx_log2(x: f32) -> f32 {
    if x <= 0.0 {
        return NEG_EV_FLOOR;
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 23) & 0xff) as i32;
    if exp_field == 0 {
        // Subnormal: below the smallest normal `luminance` we ever bin.
        return NEG_EV_FLOOR;
    }
    let mantissa = bits & 0x007f_ffff;
    #[expect(
        clippy::cast_precision_loss,
        reason = "mantissa < 2^23 is represented exactly in f32"
    )]
    let frac = mantissa as f32 / MANTISSA_SCALE;
    let log_mant = frac + LOG2_CORRECTION * frac * (1.0 - frac);
    #[expect(
        clippy::cast_precision_loss,
        reason = "unbiased f32 exponent is in [-126, 127], exact in f32"
    )]
    let exponent = (exp_field - 127) as f32;
    exponent + log_mant
}

/// Integer-exponent `exp2` (`2^x`) approximation, the inverse of
/// [`approx_log2`].
///
/// The integer part `floor(x)` is materialized straight into the `f32`
/// exponent field, and the fractional part is closed with the quadratic
/// `2^f ≈ 1 + EXP2_C1 * f + EXP2_C2 * f * f` for `f in [0, 1)`. Inputs beyond
/// the representable exponent range saturate to `0.0` / [`f32::MAX`] instead of
/// producing a subnormal or infinity. No `exp`/`powf` function is called.
#[must_use]
fn approx_exp2(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    let mantissa = 1.0 + EXP2_C1 * frac + EXP2_C2 * frac * frac;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "floor(x) is an integral f32; out-of-range magnitudes are handled below"
    )]
    let exponent = floor as i32;
    if exponent > 127 {
        return f32::MAX;
    }
    if exponent < -126 {
        return 0.0;
    }
    #[expect(
        clippy::cast_sign_loss,
        reason = "exponent + 127 lies in [1, 254], always non-negative"
    )]
    let field = (exponent + 127) as u32;
    let scale = f32::from_bits(field << 23);
    mantissa * scale
}

/// Maps a `luminance` onto its histogram bucket in `[0, bins)`.
///
/// The `luminance` is placed on the `EV` axis via [`approx_log2`], normalized
/// against `[min_ev, max_ev]`, and floored into one of `bins` buckets. The
/// result is monotonic in `lum` (brighter samples never map to a lower bucket)
/// and always in range: values below `min_ev` clamp to bucket `0` and values at
/// or above `max_ev` clamp to the last bucket. A degenerate span or zero bin
/// count yields bucket `0`.
#[must_use]
pub fn bin_index(lum: f32, min_ev: f32, max_ev: f32, bins: u32) -> usize {
    let bin_count = bins.max(1);
    let span = max_ev - min_ev;
    if span <= EPS {
        return 0;
    }
    let ev = approx_log2(lum);
    let t = clamp((ev - min_ev) / span, 0.0, 1.0);
    #[expect(
        clippy::cast_precision_loss,
        reason = "bin_count is a small grid dimension, exact in f32"
    )]
    let bins_f = bin_count as f32;
    let scaled = (t * bins_f).floor();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled is floored from a [0, 1] * bins product, so it is a non-negative integer < 2^24"
    )]
    let raw = scaled as usize;
    let last = (bin_count - 1) as usize;
    if raw > last {
        last
    } else {
        raw
    }
}

/// Accumulates a `luminance` `histogram` from a slice of samples.
///
/// Returns a `Vec<u32>` of length [`LumHistConfig::effective_bins`] where each
/// entry counts the samples whose `EV` fell in that bucket (see [`bin_index`]).
/// Counts saturate rather than wrapping, so a pathologically large input can
/// never alias a small count.
#[must_use]
pub fn build_histogram(luminances: &[f32], cfg: &LumHistConfig) -> Vec<u32> {
    let bins = cfg.effective_bins();
    let len = usize::try_from(bins).unwrap_or(usize::MAX);
    let mut hist = alloc::vec![0u32; len];
    for &lum in luminances {
        let idx = bin_index(lum, cfg.min_ev, cfg.max_ev, bins);
        hist[idx] = hist[idx].saturating_add(1);
    }
    hist
}

/// Center `EV` of histogram bucket `index` of `bins` across `[min_ev, max_ev]`.
#[must_use]
fn bucket_center_ev(index: usize, bins: usize, min_ev: f32, max_ev: f32) -> f32 {
    let span = max_ev - min_ev;
    #[expect(
        clippy::cast_precision_loss,
        reason = "bucket index is a small grid coordinate, exact in f32"
    )]
    let index_f = index as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "bin count is a small grid dimension, exact in f32"
    )]
    let bins_f = bins as f32;
    min_ev + (index_f + 0.5) / bins_f * span
}

/// Average scene `luminance` read out of a `luminance` `histogram`.
///
/// The average is computed in the `log2`-`EV` domain: each bucket contributes
/// its center `EV` weighted by the sample count that survives trimming, and the
/// weighted-mean `EV` is mapped back to a linear `luminance` with
/// [`approx_exp2`]. [`LumHistConfig::low_percentile`] and
/// [`LumHistConfig::high_percentile`] discard that fraction of the darkest and
/// brightest samples first, so a handful of black or blown-out pixels cannot
/// bias exposure. An empty (or fully trimmed) histogram falls back to the
/// midpoint `EV`.
#[must_use]
pub fn average_luminance(hist: &[u32], cfg: &LumHistConfig) -> f32 {
    let mid_ev = (cfg.min_ev + cfg.max_ev) * 0.5;
    let total: u64 = hist.iter().map(|&c| u64::from(c)).sum();
    if total == 0 {
        return approx_exp2(mid_ev);
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample totals well within f32's exact-integer range for realistic frames"
    )]
    let total_f = total as f32;
    let low = clamp(cfg.low_percentile, 0.0, 1.0);
    let high = clamp(cfg.high_percentile, 0.0, 1.0);
    let drop_low = (low * total_f).floor();
    let drop_high = (high * total_f).floor();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "drop counts are floored from a [0, 1] * total product, so non-negative and <= total"
    )]
    let mut lo_cut = drop_low as u64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "drop counts are floored from a [0, 1] * total product, so non-negative and <= total"
    )]
    let hi_drop = drop_high as u64;
    let mut hi_cut = total.saturating_sub(hi_drop);
    if lo_cut >= hi_cut {
        // Percentiles overlap and would discard every sample; keep them all.
        lo_cut = 0;
        hi_cut = total;
    }

    let bins = hist.len();
    let mut cursor: u64 = 0;
    let mut weighted_sum = 0.0f32;
    let mut weight = 0.0f32;
    for (index, &count) in hist.iter().enumerate() {
        let start = cursor;
        let end = cursor + u64::from(count);
        cursor = end;
        let overlap_start = start.max(lo_cut);
        let overlap_end = end.min(hi_cut);
        if overlap_end > overlap_start {
            let included = overlap_end - overlap_start;
            #[expect(
                clippy::cast_precision_loss,
                reason = "included <= total, within f32's exact-integer range for realistic frames"
            )]
            let included_f = included as f32;
            let center = bucket_center_ev(index, bins, cfg.min_ev, cfg.max_ev);
            weighted_sum += center * included_f;
            weight += included_f;
        }
    }

    if weight <= EPS {
        return approx_exp2(mid_ev);
    }
    approx_exp2(weighted_sum / weight)
}

/// Exposure multiplier from an average `luminance` and a middle-grey *key*.
///
/// `exposure = key / avg_luminance` (Reinhard's key-value operator): a brighter
/// scene yields a smaller multiplier, a darker scene a larger one, so the two
/// are inversely monotonic. The denominator is clamped up to [`EPS`] so a fully
/// black frame cannot divide by zero. No `exp`/`powf` is involved.
#[must_use]
pub fn exposure_from_average(avg_lum: f32, key: f32) -> f32 {
    let denom = if avg_lum < EPS { EPS } else { avg_lum };
    key / denom
}

/// `std430` byte size of the `u32` `histogram` storage buffer for `cfg`.
///
/// Sized for [`LumHistConfig::effective_bins`] contiguous `u32` words; an empty
/// configuration still reserves one element because a `WebGPU` storage binding
/// may not be zero-sized (see [`gpu_layout::storage_bytes`]).
#[must_use]
pub fn histogram_storage_bytes(cfg: &LumHistConfig) -> usize {
    let bins = usize::try_from(cfg.effective_bins()).unwrap_or(usize::MAX);
    gpu_layout::storage_bytes(gpu_layout::U32_STRIDE, bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for approximate `f32` comparisons in tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        let diff = a - b;
        let mag = if diff < 0.0 { -diff } else { diff };
        mag <= tol
    }

    #[test]
    fn luminance_uses_rec709_weights() {
        assert!(approx_eq(luminance([1.0, 0.0, 0.0]), 0.2126, CMP_EPS));
        assert!(approx_eq(luminance([0.0, 1.0, 0.0]), 0.7152, CMP_EPS));
        assert!(approx_eq(luminance([0.0, 0.0, 1.0]), 0.0722, CMP_EPS));
        // Weights sum to one, so neutral grey maps to its own value.
        assert!(approx_eq(luminance([0.5, 0.5, 0.5]), 0.5, CMP_EPS));
        assert!(approx_eq(luminance([1.0, 1.0, 1.0]), 1.0, CMP_EPS));
    }

    #[test]
    fn approx_log2_matches_known_powers_of_two() {
        assert!(approx_eq(approx_log2(1.0), 0.0, 0.1));
        assert!(approx_eq(approx_log2(2.0), 1.0, 0.1));
        assert!(approx_eq(approx_log2(4.0), 2.0, 0.1));
        assert!(approx_eq(approx_log2(0.5), -1.0, 0.1));
        assert!(approx_eq(approx_log2(8.0), 3.0, 0.1));
        assert!(approx_eq(approx_log2(0.25), -2.0, 0.1));
    }

    #[test]
    fn approx_log2_powers_of_two_are_exact() {
        // f == 0 for exact powers, so the polynomial contributes nothing.
        assert!(approx_eq(approx_log2(1.0), 0.0, CMP_EPS));
        assert!(approx_eq(approx_log2(2.0), 1.0, CMP_EPS));
        assert!(approx_eq(approx_log2(0.5), -1.0, CMP_EPS));
    }

    #[test]
    fn approx_log2_stays_accurate_inside_an_octave() {
        // log2(1.5) = 0.5849625, log2(3) = 1.5849625.
        assert!(approx_eq(approx_log2(1.5), 0.584_962_5, 0.01));
        assert!(approx_eq(approx_log2(3.0), 1.584_962_5, 0.01));
    }

    #[test]
    fn approx_log2_handles_non_positive_and_subnormal() {
        assert!(approx_eq(approx_log2(0.0), NEG_EV_FLOOR, CMP_EPS));
        assert!(approx_eq(approx_log2(-4.0), NEG_EV_FLOOR, CMP_EPS));
        assert!(approx_eq(
            approx_log2(f32::from_bits(1)),
            NEG_EV_FLOOR,
            CMP_EPS
        ));
    }

    #[test]
    fn approx_exp2_inverts_approx_log2_on_powers() {
        assert!(approx_eq(approx_exp2(0.0), 1.0, CMP_EPS));
        assert!(approx_eq(approx_exp2(1.0), 2.0, CMP_EPS));
        assert!(approx_eq(approx_exp2(2.0), 4.0, CMP_EPS));
        assert!(approx_eq(approx_exp2(-1.0), 0.5, CMP_EPS));
        assert!(approx_eq(approx_exp2(-1.5), 0.353_553_4, 0.01));
        assert!(approx_eq(approx_exp2(0.5), 1.41, 0.01));
    }

    #[test]
    fn approx_exp2_saturates_out_of_range() {
        assert!(approx_eq(approx_exp2(200.0), f32::MAX, CMP_EPS));
        assert!(approx_eq(approx_exp2(-200.0), 0.0, CMP_EPS));
    }

    #[test]
    fn bin_index_is_in_range_and_clamps_the_tails() {
        let (min_ev, max_ev, bins) = (-8.0, 8.0, 32u32);
        // Far below min_ev clamps to bucket 0.
        assert_eq!(bin_index(0.0, min_ev, max_ev, bins), 0);
        assert_eq!(bin_index(1e-6, min_ev, max_ev, bins), 0);
        // Far above max_ev clamps to the last bucket.
        let last = (bins - 1) as usize;
        assert_eq!(bin_index(1e9, min_ev, max_ev, bins), last);
        // Every result is strictly inside [0, bins).
        for i in 0..2000u32 {
            let lum = i as f32 * 0.05;
            let b = bin_index(lum, min_ev, max_ev, bins);
            assert!(b < bins as usize);
        }
    }

    #[test]
    fn bin_index_is_monotonic_in_luminance() {
        let (min_ev, max_ev, bins) = (-6.0, 6.0, 48u32);
        let mut prev = 0usize;
        let mut sample = 0.001f32;
        while sample < 40.0 {
            let b = bin_index(sample, min_ev, max_ev, bins);
            assert!(b >= prev, "bin dropped at lum {sample}: {b} < {prev}");
            prev = b;
            sample *= 1.05;
        }
    }

    #[test]
    fn bin_index_degenerate_span_is_zero() {
        assert_eq!(bin_index(2.0, 4.0, 4.0, 16), 0);
        assert_eq!(bin_index(2.0, 4.0, 3.0, 16), 0);
    }

    #[test]
    fn build_histogram_length_follows_effective_bins() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 20, 0.0, 0.0);
        assert_eq!(build_histogram(&[], &cfg).len(), 20);
        let zero_bins = LumHistConfig::new(-8.0, 8.0, 0, 0.0, 0.0);
        assert_eq!(build_histogram(&[1.0], &zero_bins).len(), 1);
    }

    #[test]
    fn build_histogram_all_dark_collapses_to_low_bins() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 32, 0.0, 0.0);
        let samples = [0.0f32; 100];
        let hist = build_histogram(&samples, &cfg);
        assert_eq!(hist[0], 100);
        assert_eq!(hist.iter().sum::<u32>(), 100);
        assert!(hist[1..].iter().all(|&c| c == 0));
    }

    #[test]
    fn build_histogram_all_bright_collapses_to_high_bins() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 32, 0.0, 0.0);
        let samples = [4096.0f32; 100];
        let hist = build_histogram(&samples, &cfg);
        let last = hist.len() - 1;
        assert_eq!(hist[last], 100);
        assert!(hist[..last].iter().all(|&c| c == 0));
    }

    #[test]
    fn build_histogram_totals_are_conserved() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 24, 0.0, 0.0);
        let samples: Vec<f32> = (0..500).map(|i| i as f32 * 0.02 + 0.01).collect();
        let hist = build_histogram(&samples, &cfg);
        assert_eq!(hist.iter().map(|&c| u64::from(c)).sum::<u64>(), 500);
    }

    #[test]
    fn average_luminance_of_uniform_grey_is_near_the_source() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 128, 0.0, 0.0);
        // A whole frame at luminance 1.0 => EV 0 => average luminance ~ 1.0.
        let samples = [1.0f32; 256];
        let hist = build_histogram(&samples, &cfg);
        let avg = average_luminance(&hist, &cfg);
        assert!(approx_eq(avg, 1.0, 0.2), "avg = {avg}");
    }

    #[test]
    fn average_luminance_lands_inside_the_configured_ev_range() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 64, 0.0, 0.0);
        let samples: Vec<f32> = (1..400).map(|i| i as f32 * 0.1).collect();
        let hist = build_histogram(&samples, &cfg);
        let avg = average_luminance(&hist, &cfg);
        let ev = approx_log2(avg);
        assert!(
            ev >= cfg.min_ev - 0.5 && ev <= cfg.max_ev + 0.5,
            "ev = {ev}"
        );
    }

    #[test]
    fn average_luminance_of_empty_histogram_is_the_midpoint() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 32, 0.25, 0.25);
        let hist = build_histogram(&[], &cfg);
        let avg = average_luminance(&hist, &cfg);
        // Midpoint EV is 0 => luminance ~ 1.0.
        assert!(approx_eq(avg, 1.0, 0.2), "avg = {avg}");
    }

    #[test]
    fn percentile_trimming_rejects_the_outlier_tails() {
        // Drop the top 30% so the one-sided bright spike is discarded.
        let cfg_trim = LumHistConfig::new(-8.0, 8.0, 64, 0.0, 0.3);
        let cfg_full = LumHistConfig::new(-8.0, 8.0, 64, 0.0, 0.0);
        // Mostly mid-grey (EV 0) with a large one-sided bright spike (EV ~ +8).
        let mut samples = alloc::vec![1.0f32; 100];
        samples.extend(core::iter::repeat_n(4000.0f32, 30));
        let hist = build_histogram(&samples, &cfg_full);
        let trimmed = average_luminance(&hist, &cfg_trim);
        let full = average_luminance(&hist, &cfg_full);
        // Trimming removes the bright outliers, pulling the average back toward
        // the mid-grey mode; the untrimmed average is dragged well above it.
        assert!(approx_eq(trimmed, 1.0, 0.3), "trimmed = {trimmed}");
        assert!(full > 2.0, "untrimmed average should be dragged up: {full}");
        assert!(
            approx_log2(trimmed).abs() < approx_log2(full).abs(),
            "trimming should shrink the EV magnitude: trimmed={trimmed} full={full}"
        );
    }

    #[test]
    fn overlapping_percentiles_keep_all_samples() {
        // low + high >= 1 would trim everything; the guard keeps the full set.
        let cfg = LumHistConfig::new(-8.0, 8.0, 32, 0.8, 0.8);
        let samples = [1.0f32; 50];
        let hist = build_histogram(&samples, &cfg);
        let avg = average_luminance(&hist, &cfg);
        assert!(approx_eq(avg, 1.0, 0.2), "avg = {avg}");
    }

    #[test]
    fn exposure_is_inversely_monotonic_in_average_luminance() {
        let key = 0.18f32;
        let dark = exposure_from_average(0.1, key);
        let mid = exposure_from_average(1.0, key);
        let bright = exposure_from_average(10.0, key);
        assert!(dark > mid, "darker scene must get more exposure");
        assert!(mid > bright, "brighter scene must get less exposure");
        assert!(approx_eq(mid, 0.18, CMP_EPS));
    }

    #[test]
    fn exposure_black_frame_does_not_divide_by_zero() {
        let e = exposure_from_average(0.0, 0.18);
        assert!(e.is_finite());
        assert!(e > 0.0);
    }

    #[test]
    fn build_histogram_is_deterministic() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 40, 0.05, 0.05);
        let samples: Vec<f32> = (0..300).map(|i| i as f32 * 0.037 + 0.02).collect();
        let a = build_histogram(&samples, &cfg);
        let b = build_histogram(&samples, &cfg);
        assert_eq!(a, b);
        assert!(approx_eq(
            average_luminance(&a, &cfg),
            average_luminance(&b, &cfg),
            CMP_EPS,
        ));
    }

    #[test]
    fn histogram_storage_bytes_match_std430() {
        let cfg = LumHistConfig::new(-8.0, 8.0, 64, 0.0, 0.0);
        assert_eq!(histogram_storage_bytes(&cfg), 64 * gpu_layout::U32_STRIDE);
        // std430 u32 array: total size is a multiple of the 4-byte element.
        assert_eq!(histogram_storage_bytes(&cfg) % gpu_layout::U32_STRIDE, 0);
        // An empty configuration still reserves one non-empty element.
        let empty = LumHistConfig::new(-8.0, 8.0, 0, 0.0, 0.0);
        assert_eq!(histogram_storage_bytes(&empty), gpu_layout::U32_STRIDE);
    }

    #[test]
    fn effective_bins_clamps_up_to_one() {
        assert_eq!(
            LumHistConfig::new(-8.0, 8.0, 0, 0.0, 0.0).effective_bins(),
            1
        );
        assert_eq!(STANDARD_HDR.effective_bins(), 64);
    }
}
